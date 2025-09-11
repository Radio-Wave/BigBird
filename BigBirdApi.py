#------------------------------------------------------------------
#------------------------ 2024 - Xach Hill ------------------------
#-------------------------- Use for good --------------------------
#------------------------------------------------------------------

import os
import time
import json
import threading
import requests
import pyaudio
import wave
import collections
import contextlib
import webrtcvad
import random
import serial
import whisper
from collections import deque
from openai import OpenAI
from pyht import Client, TTSOptions, Format
import urllib.parse

from dotenv import load_dotenv
import logging
import live
from elevenlabs.client import ElevenLabs
try:
    from elevenlabs import play as el_play  # In some SDK versions this is a function; in others it's a module
except Exception:
    el_play = None

try:
    # ElevenLabs SDK helper to play streamed audio locally
    from elevenlabs import stream as el_stream
except Exception:
    el_stream = None

try:
    # Typed settings helper (optional but aligns with current SDK)
    from elevenlabs import VoiceSettings as EL_VoiceSettings
except Exception:
    EL_VoiceSettings = None
from io import BytesIO
from typing import AsyncGenerator, AsyncIterable, Generator, Iterable, Union, Optional, List
import struct, binascii   # add at top of the file if not already there


from dataclasses import dataclass, field
from threading import RLock
import subprocess
import platform

try:
    import psutil  # optional (CPU/mem)
except Exception:
    psutil = None

# Optional control server deps
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _FASTAPI_AVAILABLE = True
except Exception:
    _FASTAPI_AVAILABLE = False

#
# Ensure .env values override any stale shell envs
load_dotenv(override=True)

# Ensure runtime directories exist
def ensure_directories():
    for d in ["RecordedAudio", "RecordedAudio/New", "RecordedAudio/Old"]:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception as e:
            logging.warning(f"Could not ensure directory {d}: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- GUI/Control Shared State ----------------
@dataclass
class PlayHTSettings:
    speed: float = 1.0
    style_guidance: float = 1.0
    voice_guidance: float = 1.0
    temperature: float = 1.5
    text_guidance: float = 0.0
    sample_rate: int = 24000  # default; runtime may override with RATE after constants init

@dataclass
class ElevenLabsSettings:
    use_speaker_boost: bool = False
    stability: float = 0.5
    similarity_boost: float = 0.8
    style: float = 0.1
    speed: float = 1.0
    model_id_stream: str = "eleven_flash_v2_5"
    format_stream: str = "mp3_22050_32"
    model_id_buffered: str = "eleven_multilingual_v2"
    format_buffered: str = "mp3_44100_128"

@dataclass
class VADSettings:
    mode: int = 2
    initial_silence_timeout: float = 15.0
    max_silence_length: float = 1.0

@dataclass
class UIState:
    engine: str = (os.getenv('TTS_ENGINE') or 'playht').strip().lower()
    audio_output: str = "default"  # "default" or "blackhole"
    streaming: bool = True         # False = buffered fallback where available
    playht: PlayHTSettings = field(default_factory=PlayHTSettings)
    elevenlabs: ElevenLabsSettings = field(default_factory=ElevenLabsSettings)
    vad: VADSettings = field(default_factory=VADSettings)
    messages: collections.deque = field(default_factory=lambda: collections.deque(maxlen=200))
    lock: RLock = field(default_factory=RLock)

    def to_dict(self):
        with self.lock:
            data = {
                "engine": self.engine,
                "audio_output": self.audio_output,
                "streaming": self.streaming,
                "playht": self.playht.__dict__,
                "elevenlabs": self.elevenlabs.__dict__,
                "vad": self.vad.__dict__,
                "messages": list(self.messages),
                "system": get_system_snapshot(),
            }
            try:
                data["clones"] = voice_clone_manager.get_clone_info()
            except Exception:
                data["clones"] = []
            try:
                data["cloning"] = get_cloning_snapshot()
            except Exception:
                data["cloning"] = {"error": "unavailable"}
            return data

    def set_engine(self, e: str):
        with self.lock:
            self.engine = e.strip().lower()

    def update_playht(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self.playht, k): setattr(self.playht, k, v)

    def update_elevenlabs(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self.elevenlabs, k): setattr(self.elevenlabs, k, v)

    def update_vad(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self.vad, k): setattr(self.vad, k, v)

    def set_audio_output(self, out: str):
        with self.lock:
            self.audio_output = out

    def set_streaming(self, streaming: bool):
        with self.lock:
            self.streaming = bool(streaming)

    def append_message(self, role: str, text: str):
        with self.lock:
            self.messages.append({"ts": time.time(), "role": role, "text": text})

ui_state = UIState()

# ---------------- Preset management ----------------
class PresetManager:
    def __init__(self, path: str = "presets.json"):
        self.path = path
        self.lock = threading.Lock()
        self.presets = []
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            self.presets = data.get("presets", [])
        except FileNotFoundError:
            self.presets = []
        except Exception as e:
            logging.error(f"Failed to load presets: {e}")
            self.presets = []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"presets": self.presets}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save presets: {e}")

    def list(self):
        with self.lock:
            def _key(p):
                return (p.get("order", 0), p.get("created_at", 0))
            return sorted(self.presets, key=_key)

    def get(self, pid: str):
        with self.lock:
            for p in self.presets:
                if p.get("id") == pid:
                    return p
        return None

    def add_from_data(self, name: str, data: dict):
        with self.lock:
            now = time.time()
            pid = f"p_{int(now*1000)}"
            order = max([p.get("order", 0) for p in self.presets], default=0) + 1
            preset = {
                "id": pid,
                "name": name or f"Preset {order}",
                "starred": False,
                "created_at": now,
                "updated_at": now,
                "order": order,
                "data": data or {},
            }
            self.presets.append(preset)
            self._save()
            return preset

    def delete(self, pid: str) -> bool:
        with self.lock:
            n = len(self.presets)
            self.presets = [p for p in self.presets if p.get("id") != pid]
            if len(self.presets) != n:
                self._save()
                return True
            return False

    def rename(self, pid: str, new_name: str) -> bool:
        with self.lock:
            for p in self.presets:
                if p.get("id") == pid:
                    p["name"] = new_name
                    p["updated_at"] = time.time()
                    self._save()
                    return True
            return False

    def reorder(self, new_order_ids: List[str]) -> bool:
        with self.lock:
            id_to_p = {p["id"]: p for p in self.presets}
            ordered = []
            order_val = 1
            for pid in new_order_ids:
                if pid in id_to_p:
                    p = id_to_p.pop(pid)
                    p["order"] = order_val
                    order_val += 1
                    ordered.append(p)
            leftovers = [p for p in self.presets if p["id"] in id_to_p]
            for p in leftovers:
                p["order"] = order_val
                order_val += 1
                ordered.append(p)
            self.presets = ordered
            self._save()
            return True

    def star(self, pid: str) -> bool:
        with self.lock:
            found = False
            for p in self.presets:
                if p.get("id") == pid:
                    p["starred"] = True
                    p["updated_at"] = time.time()
                    found = True
                else:
                    p["starred"] = False
            if found:
                self._save()
            return found

    def starred(self):
        with self.lock:
            for p in self.presets:
                if p.get("starred", False):
                    return p
        return None

preset_manager = PresetManager()

def export_settings_for_preset() -> dict:
    """Snapshot of the current tunables for presets."""
    try:
        return {
            "engine": ui_state.engine,
            "audio_output": ui_state.audio_output,
            "streaming": ui_state.streaming,
            "playht": dict(ui_state.playht.__dict__),
            "elevenlabs": dict(ui_state.elevenlabs.__dict__),
            "vad": dict(ui_state.vad.__dict__),
        }
    except Exception:
        return {}

def apply_settings_from_preset(settings: dict):
    """Apply preset settings to ui_state."""
    if not settings:
        return
    try:
        eng = settings.get("engine")
        if eng:
            ui_state.engine = eng
        ao = settings.get("audio_output")
        if ao:
            ui_state.audio_output = ao
        if "streaming" in settings:
            ui_state.streaming = bool(settings.get("streaming"))
        for sect in ("playht", "elevenlabs", "vad"):
            vals = settings.get(sect)
            if isinstance(vals, dict):
                tgt = getattr(ui_state, sect, None)
                if hasattr(tgt, "__dict__"):
                    tgt.__dict__.update(vals)
                else:
                    setattr(ui_state, sect, vals)
    except Exception as e:
        logging.error(f"apply_settings_from_preset failed: {e}")

def apply_initial_starred_preset():
    try:
        p = preset_manager.starred()
        if p and isinstance(p.get("data"), dict):
            logging.info(f"Applying starred preset on startup: {p.get('name')}")
            apply_settings_from_preset(p["data"])
    except Exception as e:
        logging.warning(f"apply_initial_starred_preset skipped: {e}")

def get_system_snapshot():
    snap = {
        "threads": threading.active_count(),
        "clones_count": len(voice_clone_manager.get_clone_info()) if 'voice_clone_manager' in globals() else 0,
    }
    # Provider split (optional visibility)
    try:
        if 'voice_clone_manager' in globals():
            info = voice_clone_manager.get_clone_info()
            snap["clones_split"] = {
                "playht": sum(1 for c in info if c.get('engine') == 'playht'),
                "elevenlabs": sum(1 for c in info if c.get('engine') == 'elevenlabs'),
            }
    except Exception:
        pass
    # CPU/memory
    try:
        if psutil:
            p = psutil.Process(os.getpid())
            snap["cpu_percent"] = psutil.cpu_percent(interval=None)
            snap["mem_mb"] = p.memory_info().rss / (1024 * 1024)
        else:
            # Fallback CPU estimate from 1-min load average
            try:
                la1 = os.getloadavg()[0]
                cpus = os.cpu_count() or 1
                snap["cpu_percent"] = max(0.0, min(100.0, (la1 / cpus) * 100.0))
            except Exception:
                pass
            # Fallback memory from resource.getrusage
            try:
                import resource
                ru = resource.getrusage(resource.RUSAGE_SELF)
                rss = getattr(ru, "ru_maxrss", 0)
                # On macOS ru_maxrss is bytes; on Linux it's kilobytes.
                mem_mb = rss / (1024 * 1024)
                if mem_mb == 0:
                    mem_mb = rss / 1024.0  # handle KB case
                snap["mem_mb"] = mem_mb
            except Exception:
                pass
    except Exception:
        pass
    return snap

# Constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_PROJECT = os.getenv('OPENAI_PROJECT')

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# Instantiate client explicitly with project to avoid defaulting to a project with no credit
client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT) if OPENAI_PROJECT else OpenAI(api_key=OPENAI_API_KEY)

# Sanity-check helper to print which OpenAI credentials are in use
def debug_print_openai_creds():
    key = os.getenv('OPENAI_API_KEY') or ''
    proj = os.getenv('OPENAI_PROJECT') or ''
    print(f"OPENAI_API_KEY set: {'yes' if key else 'no'} (prefix: {key[:7]+'…' if key else '—'})")
    print(f"OPENAI_PROJECT set: {'yes' if proj else 'no'} (value: {proj if proj else '—'})")

# Sanity-check helper to print which OpenAI credentials are in use
debug_print_openai_creds()

# Cross-platform temp-file playback fallback for ElevenLabs streaming
def _play_file_best_effort(tmp_path: str, codec_hint: str = "mp3"):
    """
    Best-effort synchronous playback of a small temporary audio file.
    On macOS uses 'afplay'; on other platforms we just log where the file is.
    """
    try:
        import subprocess, sys
        if sys.platform == 'darwin':
            subprocess.run(['afplay', tmp_path], check=True)
        else:
            logging.error(f"No portable playback fallback for platform {sys.platform}. File saved at {tmp_path}")
    except Exception as e:
        logging.error(f"Fallback playback failed ({codec_hint}) at {tmp_path}: {e}")
#PLAYHT_API_KEYS = [os.getenv('PLAYHT_API_KEY'), os.getenv('PLAYHT_API_KEY_SPARE')]
#PLAYHT_USER_IDS = [os.getenv('PLAYHT_USER_ID'), os.getenv('PLAYHT_USER_ID_SPARE')]

PLAYHT_API_KEYS = [os.getenv('PLAYHT_API_KEY')]
PLAYHT_USER_IDS = [os.getenv('PLAYHT_USER_ID')] 

# Engine selector (scaffolding for runtime swap)
TTS_ENGINE = (os.getenv('TTS_ENGINE') or 'playht').strip().lower()

VOICE_CLONE_MODEL = "PlayHT2.0-turbo"      # 48 kHz / 16‑bit mono
LLM_MODEL = "gpt-4o-mini-2024-07-18"

# PlayHT 2.0‑turbo requires gRPC streaming
PROTOCOL = "grpc"

AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 2048

url = "https://api.play.ht/api/v2/cloned-voices/instant"

OldSound = "RecordedAudio/Old"
New = "RecordedAudio/New"

SoundTrack = "/Users/x/myenv/bin/tester.wav"


arduinoLocation = '/dev/cu.usbmodem8301'

# ---------------- Arduino override & helpers ----------------
_arduino_override = {"hook": None, "led": None}  # hook: "offhook"|"onhook"|None; led: "OFF"|"RECORDING"|"REPLYING"|"PROCESSING"|None

# Lazy global Arduino instance (created on first use)
arduino = None

def get_arduino():
    global arduino
    if arduino is None:
        try:
            arduino = ArduinoControl(arduinoLocation)
        except Exception as e:
            logging.error(f"Arduino init failed: {e}")
            arduino = None
    return arduino

def set_arduino_hook_override(state):
    # state in {"offhook","onhook",None}
    if state not in ("offhook", "onhook", None):
        raise ValueError("bad hook override")
    _arduino_override["hook"] = state

def set_arduino_led_override(state):
    # state in {"OFF","RECORDING","REPLYING","PROCESSING",None}
    if state not in ("OFF", "RECORDING", "REPLYING", "PROCESSING", None):
        raise ValueError("bad led override")
    _arduino_override["led"] = state

def get_arduino_status():
    a = get_arduino()
    return {
        "override": dict(_arduino_override),
        "port": arduinoLocation,
        "connected": bool(a and getattr(a, "serial", None)),
    }

def set_led_state(name: str) -> bool:
    """
    Set Arduino LED state by friendly name and record it in override for UI.
    """
    set_arduino_led_override(name)
    a = get_arduino()
    if not a:
        return False
    try:
        if name == "OFF":
            a.led_set_off()
        elif name == "RECORDING":
            a.led_recording()
        elif name == "REPLYING":
            a.led_speaking()
        elif name == "PROCESSING":
            a.led_cloning()
        else:
            return False
        return True
    except Exception as e:
        logging.warning(f"LED set failed: {e}")
        return False

# Ensure runtime directories exist at startup
ensure_directories()

# Log configured TTS engine
logging.info(f"Configured TTS_ENGINE={TTS_ENGINE}")

class APIKeyManager:
    def __init__(self, keys, user_ids):
        self.keys = keys
        self.user_ids = user_ids
        self.current_index = 0

    def get_current_key(self):
        return self.keys[self.current_index]

    def get_current_user_id(self):
        return self.user_ids[self.current_index]

    def switch_key(self):
        self.current_index = (self.current_index + 1) % len(self.keys)
        logging.info(f"Switched to API key index: {self.current_index}")
        self.reload_vital_data()

    def reload_vital_data(self):
        # Placeholder for reloading vital data when key is switched
        voice_clone_manager.sync_with_api()

api_key_manager = APIKeyManager(PLAYHT_API_KEYS, PLAYHT_USER_IDS)


class SessionAudioManager:
    def __init__(self):
        self.clips = []
        self.total_duration = 0.0  # Total duration in seconds

    def add_clip(self, file_path):
        """Add a new audio clip and update total duration; mirror into global progress."""
        with contextlib.closing(wave.open(file_path, 'rb')) as wf:
            duration = wf.getnframes() / wf.getframerate()
        self.clips.append(file_path)
        self.total_duration += duration
        # Mirror into global progress/pending for GUI
        try:
            clone_progress.note_collected(duration)
            clone_progress.add_pending_clip(file_path)
        except Exception:
            pass

    def should_send_for_cloning(self):
        """Check if the accumulated clips should be sent for voice cloning."""
        return self.total_duration >= 7.5

    def concatenate_clips(self):
        """Concatenate all clips into a single WAV and return the path, or None if no clips."""
        if not self.clips:
            logging.info("concatenate_clips: no clips to concatenate")
            return None

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"RecordedAudio/concatenated_audio_{timestamp}.wav"

        try:
            with wave.open(output_path, 'wb') as wf:
                # Initialize parameters from the first clip
                with wave.open(self.clips[0], 'rb') as cf:
                    wf.setnchannels(cf.getnchannels())
                    wf.setsampwidth(cf.getsampwidth())
                    wf.setframerate(cf.getframerate())

                for clip in self.clips:
                    with wave.open(clip, 'rb') as cf:
                        frames = cf.readframes(cf.getnframes())
                        wf.writeframes(frames)
        except Exception as e:
            logging.error(f"concatenate_clips failed: {e}")
            return None
        finally:
            # Reset state whether success or failure
            self.total_duration = 0.0
            self.clips.clear()
            try:
                clone_progress.reset_collected()  # reset progress (not clones_in_session)
            except Exception:
                pass

        return output_path
    
# ---------------- Clone progress tracking (for GUI) ----------------
class CloneProgressTracker:
    def __init__(self, required_seconds: float = 7.5):
        self.required_seconds = float(required_seconds)
        self.collected_seconds = 0.0
        self.clones_in_session = 0
        self.session_started_ts = time.time()
        self.lock = threading.Lock()
        self.pending_clip_paths: list[str] = []

    def note_collected(self, seconds: float):
        with self.lock:
            try:
                self.collected_seconds += max(0.0, float(seconds))
            except Exception:
                pass

    def add_pending_clip(self, path: str):
        with self.lock:
            if path and path not in self.pending_clip_paths:
                self.pending_clip_paths.append(path)

    def reset_collected(self):
        """Reset progress toward next clone and clear pending list."""
        with self.lock:
            self.collected_seconds = 0.0
            self.pending_clip_paths.clear()

    def consume_pending_to_wav(self) -> Optional[str]:
        """Concatenate all pending clips to a temp WAV and clear the list. Returns path or None."""
        with self.lock:
            paths = list(self.pending_clip_paths)
        if not paths:
            return None

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"RecordedAudio/concatenated_pending_{timestamp}.wav"
        try:
            with wave.open(output_path, 'wb') as wf:
                # Initialize parameters from the first clip
                with wave.open(paths[0], 'rb') as cf:
                    wf.setnchannels(cf.getnchannels())
                    wf.setsampwidth(cf.getsampwidth())
                    wf.setframerate(cf.getframerate())
                # Append frames
                for clip in paths:
                    try:
                        with wave.open(clip, 'rb') as cf:
                            frames = cf.readframes(cf.getnframes())
                            wf.writeframes(frames)
                    except Exception as e:
                        logging.warning(f"Skipping bad clip {clip}: {e}")
        except Exception as e:
            logging.error(f"consume_pending_to_wav failed: {e}")
            return None
        finally:
            # Reset GUI progress for next cycle
            self.reset_collected()
        return output_path

    def on_clone_started(self):
        with self.lock:
            self.clones_in_session += 1
            self.collected_seconds = 0.0  # new cycle

    def set_required(self, seconds: float):
        with self.lock:
            try:
                self.required_seconds = max(1.0, float(seconds))
            except Exception:
                pass

    def reset_session(self):
        with self.lock:
            self.session_started_ts = time.time()
            self.collected_seconds = 0.0
            self.clones_in_session = 0
            self.pending_clip_paths.clear()

    def snapshot(self) -> dict:
        with self.lock:
            req = self.required_seconds
            col = self.collected_seconds
            pct = (col / req) if req > 0 else 0.0
            return {
                "required_seconds": req,
                "collected_seconds": col,
                "percent": max(0.0, min(1.0, pct)),
                "seconds_remaining": max(0.0, req - col),
                "clones_in_session": self.clones_in_session,
                "pending_clips": len(self.pending_clip_paths),
                "session_started": self.session_started_ts,
            }

# Global tracker instance
clone_progress = CloneProgressTracker(required_seconds=7.5)


def read_saved_info(limit_chars: int = 1000) -> str:
    """Return last up-to-`limit_chars` of savedInfo.txt if present."""
    path = "savedInfo.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
        if len(data) > limit_chars:
            return data[-limit_chars:]
        return data
    except Exception:
        return ""


def get_cloning_snapshot() -> dict:
    snap = clone_progress.snapshot()
    snap["identity_info"] = read_saved_info(1200)
    return snap

class AudioRecorderWithVAD:
    def __init__(self, vad_mode=None):
        self.vad = webrtcvad.Vad(ui_state.vad.mode if vad_mode is None else vad_mode)
        self.p = pyaudio.PyAudio()
        self.frame_duration_ms = 20  # Frame duration in milliseconds
        self.frames_per_buffer = int(16000 * self.frame_duration_ms / 1000)  # Samples per frame
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Sample rate that matches the VAD requirement

    def is_speech(self, frame):
        """Check if the frame contains speech."""
        return self.vad.is_speech(frame, self.rate)

    def record_with_vad(self, initial_silence_timeout=15, max_silence_length=1):
        """Record audio and stop when speech ends, with handling for initial silence and IO errors."""
        os.makedirs("RecordedAudio", exist_ok=True)
        stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.frames_per_buffer)
        audio_frames = []
        silent_frames = collections.deque(maxlen=int(self.rate / self.frames_per_buffer * max_silence_length))
        initial_timeout_frames = int(self.rate / self.frames_per_buffer * initial_silence_timeout)

        try:
            # Wait for initial speech up to timeout
            for _ in range(initial_timeout_frames):
                try:
                    frame = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                except Exception as e:
                    logging.warning(f"Audio read error (initial): {e}")
                    continue
                if self.is_speech(frame):
                    audio_frames.append(frame)
                    break
                else:
                    silent_frames.append(frame)

            if not audio_frames:  # No speech detected
                return None

            # Continue recording until trailing silence reached
            while True:
                try:
                    frame = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                except Exception as e:
                    logging.warning(f"Audio read error (recording): {e}")
                    if audio_frames:
                        break
                    else:
                        return None
                if self.is_speech(frame):
                    audio_frames.append(frame)
                    silent_frames.clear()
                else:
                    silent_frames.append(frame)
                    if len(silent_frames) == silent_frames.maxlen:
                        break
        finally:
            stream.stop_stream()
            stream.close()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"RecordedAudio/audio_{timestamp}.wav"

        try:
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(audio_frames))
        except Exception as e:
            logging.error(f"Failed to write WAV: {e}")
            return None

        return output_path

class ArduinoControl:
    def __init__(self, port, baud_rate=9600):
        """Initialize the serial connection to the Arduino."""
        try:
            self.serial = serial.Serial(port, baud_rate, timeout=1)
            logging.info("Connecting to Arduino...")
            time.sleep(1)  # Allow some time for the connection to be established
            logging.info("Connected to Arduino on port: %s", port)
        except serial.SerialException as e:
            logging.error(f"Error connecting to Arduino: {e}")
            self.serial = None

    def send_command(self, command):
        """Send a command to the Arduino."""
        if self.serial:
            self.serial.write((command + '\n').encode())
            time.sleep(0.1)  # Give the Arduino time to process the command
            while self.serial.in_waiting:
                response = self.serial.readline().decode().strip()
                #("Arduino response:", response)
                return response
            
    def onHold(self):
        """Return True when the handset is ON-HOOK (hung up), False when OFF-HOOK (picked up).
        GUI override takes precedence; if no serial is connected and no override,
        we assume OFF-HOOK (False) so the show can proceed unless explicitly forced.
        """
        # 1) GUI override first
        try:
            hook_ovr = _arduino_override.get("hook")
            if hook_ovr == "onhook":
                return True
            if hook_ovr == "offhook":
                return False
        except Exception:
            pass

        # 2) No serial? fall back to safe default (off-hook)
        if not getattr(self, "serial", None):
            logging.debug("onHold: no Arduino serial; defaulting to OFF-HOOK (False) – use GUI override to force ON-HOOK")
            return False

        # 3) Query device
        resp = self.send_command("ISBUTTONPRESSED") or ""
        r = str(resp).strip().upper()
        # Map common firmware responses
        if r in ("DOWN", "PRESSED", "LOW", "0"):
            # switch pressed -> handset on the cradle (ON-HOOK)
            return True
        if r in ("UP", "RELEASED", "HIGH", "1"):
            # switch released -> handset lifted (OFF-HOOK)
            return False
        logging.debug(f"onHold: unexpected ISBUTTONPRESSED -> {resp!r}; assuming OFF-HOOK")
        return False
        
    def _single_ring(self, ring_clip, ring_duration):
        """Play one ring; return True if pickup detected."""
        ring_clip.play()
        start = time.time()
        while time.time() - start < ring_duration:
            if self.read_serial() == "START":
                ring_clip.stop()
                logging.info("Pickup detected during ring")
                return True
            time.sleep(0.1)
        ring_clip.stop()
        logging.info("Completed one ring")
        return False

    def _wait_or_abort(self, duration: float, clip_to_stop=None) -> bool:
        """
        Wait up to `duration` seconds, polling onHold() every 0.1 s.
        If onHold() becomes True, stop `clip_to_stop` (if given) and return True.
        Otherwise return False after waiting.
        """
        end_time = time.time() + duration
        while time.time() < end_time:
            if self.onHold():
                if clip_to_stop:
                    clip_to_stop.stop()
                logging.info("Aborting: hold detected")
                return True
            time.sleep(0.1)
        return False

    def _ring_once(self, ring_clip, ring_secs) -> bool:
        """Play one ring; abort immediately if hold is detected during playback."""
        ring_clip.play()
        if self._wait_or_abort(ring_secs, clip_to_stop=ring_clip):
            return True
        ring_clip.stop()
        logging.info("Completed ring")
        return False


    def send_rgb(self, red, green, blue):
        """Send RGB values to the Arduino."""
        command = f"SETCOLOR {red},{green},{blue}"
        self.send_command(command)

    def double_ring(self):
        """Ring the phone in a double patten."""
        command = "DOUBLEBUZZ"
        responce = self.send_command(command)     
        print(responce)   

    def led_Red(self):
        """Turn LED Red."""
        self.send_rgb(255, 0, 0)

    def led_Green(self):
        """Turn LED Green."""
        self.send_rgb(0, 255, 0)

    def led_Blue(self):
        """Turn LED Blue."""
        self.send_rgb(0, 0, 255)

    def led_Off(self):
        """Turn LED Off."""
        self.send_rgb(0, 0, 0)

    def led_recording(self):
        """led red blink"""
        command = "SETSTATE RECORDING"
        self.send_command(command)

    def led_speaking(self):
        """led red blink"""
        command = "SETSTATE REPLYING"
        self.send_command(command)

    def led_cloning(self):
        """led red blink"""
        command = "SETSTATE PROCESSING"
        self.send_command(command)

    def led_set_off(self):
        """led red blink"""
        command = "SETSTATE OFF"
        self.send_command(command)

    def read_serial(self):
        """Read the serial input from Arduino."""
        if self.serial:
            return self.serial.readline().decode().strip()
        return None

    def close(self):
        """Close the serial connection."""
        if self.serial:
            self.serial.close()
            logging.info("Serial connection closed.")

    def __del__(self):
        """Destructor to make sure the serial connection is closed properly."""
        self.led_set_off()
        self.close()

class AudioStreamer:
    def __init__(self):
        self.user_id = api_key_manager.get_current_user_id()
        self.api_key = api_key_manager.get_current_key()
        self.client = Client(user_id=self.user_id, api_key=self.api_key)
        self.p = None
        self.stream = None

    def setup_audio_stream(self, sample_rate: int, channels: int):
        """
        Open a PyAudio output stream that exactly matches the WAV header
        we just parsed.  Bits‑per‑sample is always 16 in PlayHT WAV
        output, so we hard‑wire `paInt16` here.
        """
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
        )

    def stream_audio(self, text: str, voice_url: str):
        """
        Stream WAV audio returned by PlayHT.  We buffer chunks until we
        have the complete header, parse its format fields, open a PyAudio
        output stream that matches, and then feed it the PCM data.
        """
        p = ui_state.playht
        options = TTSOptions(
            voice=voice_url,
            sample_rate=p.sample_rate,
            format=Format.FORMAT_WAV,
            speed=p.speed,
            voice_guidance=p.voice_guidance,
            style_guidance=p.style_guidance,
            temperature=p.temperature,
            text_guidance=p.text_guidance,
        )

        header_buffer   = bytearray()
        header_parsed   = False
        pcm_started     = False
        channels        = 1
        sample_rate     = RATE      # fallback
        fmt_ok          = False
        self._tail_byte = b""          # keeps odd byte between chunks


        for chunk in self.client.tts(text=text,
                                     options=options,
                                     voice_engine=VOICE_CLONE_MODEL,
                                     protocol=PROTOCOL):

            header_buffer.extend(chunk)

            # Abort if handset is ON-HOOK (hang up: stop playback)
            try:
                a = get_arduino()
                if a and a.onHold():  # True == ON-HOOK in this codebase
                    logging.info("On-hook detected – aborting live stream")
                    break
            except Exception:
                pass

            if not header_parsed and len(header_buffer) >= 44:
                # Basic RIFF sanity‑check
                if header_buffer[:4] != b'RIFF' or header_buffer[8:12] != b'WAVE':
                    logging.error("First chunk is not RIFF/WAVE – aborting playback.")
                    return

                # Parse fmt fields (assume no weird <fmt  > padding before byte 24)
                channels     = int.from_bytes(header_buffer[22:24], "little")
                sample_rate  = int.from_bytes(header_buffer[24:28], "little")
                bits_per     = int.from_bytes(header_buffer[34:36], "little")
                fmt_ok       = (bits_per == 16)

                if not fmt_ok:
                    logging.error(f"Unsupported bits/sample: {bits_per}")
                    return

                # find where 'data' starts
                data_pos = header_buffer.find(b'data')
                if data_pos != -1 and len(header_buffer) >= data_pos + 8:
                    pcm_start   = data_pos + 8
                    pcm_chunk   = header_buffer[pcm_start:]
                    header_parsed = True    # got everything we need

                    # Now that we know the real format, open PyAudio
                    self.setup_audio_stream(sample_rate, channels)

                    # ---- play the PCM that followed the header ----
                    # prepend any stored tail‑byte from the previous packet
                    pcm_chunk = self._tail_byte + pcm_chunk
                    self._tail_byte = b""

                    # store a dangling byte (if any) to keep 16‑bit alignment
                    if len(pcm_chunk) & 1:
                        self._tail_byte = pcm_chunk[-1:]
                        pcm_chunk = pcm_chunk[:-1]

                    if pcm_chunk:
                        self.stream.write(bytes(pcm_chunk))
                        # Abort if handset is on-hook
                        try:
                            a = get_arduino()
                            if a and a.onHold():  # True == ON-HOOK in this codebase
                                logging.info("On-hook detected – aborting live stream")
                                break
                        except Exception:
                            pass
                    continue  # header handled; wait for next pure‑PCM chunk
                # else: keep buffering until 'data' shows up
                continue

            if header_parsed:
                # Every further chunk SHOULD be pure PCM, but in practice
                # PlayHT sometimes prefixes *each* network chunk with a
                # fresh 44‑byte RIFF WAVE header.  Detect that pattern and
                # strip any header we see before passing bytes to PyAudio.
                if chunk[:4] == b'RIFF' and len(chunk) >= 44:
                    data_pos = chunk.find(b'data')
                    if data_pos != -1 and len(chunk) >= data_pos + 8:
                        chunk = chunk[data_pos + 8 :]
                    else:
                        # Header is incomplete; skip it entirely
                        chunk = chunk[44:]
                if chunk:
                    # 16‑bit alignment – stitch dangling byte between chunks
                    if len(chunk) & 1:
                        self._tail_byte = chunk[-1:]
                        chunk = chunk[:-1]
                    else:
                        chunk = self._tail_byte + chunk
                        self._tail_byte = b""
                    if chunk:
                        # Abort if handset is on-hook
                        try:
                            a = get_arduino()
                            if a and a.onHold():  # True == ON-HOOK in this codebase
                                logging.info("On-hook detected – aborting live stream")
                                break
                        except Exception:
                            pass
                        self.stream.write(bytes(chunk))

        # end for
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
        except Exception:
            pass

    def cleanup(self):
        """Cleanup the PyAudio stream and PyAudio instance."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def __del__(self):
        self.cleanup()

# ------------------------------------------------------------------
# NEW – Streaming version that uses get_stream_pair()
# ------------------------------------------------------------------
class AudioStreamerPair:
    """
    A drop‑in alternative to AudioStreamer that relies on PlayHT’s
    `get_stream_pair()` helper.  This API guarantees every chunk is
    raw, header‑free PCM aligned to the sample boundaries, so we can
    skip all the header‑parsing / byte‑stitch logic used in the
    original AudioStreamer.
    """

    def __init__(self,
             user_id: Optional[str] = None,
             api_key: Optional[str] = None):
        self.user_id = user_id or api_key_manager.get_current_user_id()
        self.api_key = api_key or api_key_manager.get_current_key()
        self.client  = Client(user_id=self.user_id, api_key=self.api_key)
        self.p: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

    # --- helpers --------------------------------------------------

    def _setup_audio_stream(self, sample_rate: int = RATE, channels: int = 1):
        if self.p is None:
            self.p = pyaudio.PyAudio()
        if self.stream is None:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=CHUNK,
            )

    def _play_stream(self, out_stream):
        """
        Background thread target: pull PCM chunks from `out_stream`,
        parse the first WAV header to learn the real sample‑rate /
        channel count, open PyAudio with those settings, then stream
        the raw PCM.  If the stream is already raw PCM (no RIFF),
        default to 24 kHz mono.
        """
        header_buffer = bytearray()
        header_parsed = False

        for chunk in out_stream:
            if not chunk:
                continue

            if not header_parsed:
                header_buffer.extend(chunk)

                # Check for a complete WAV header
                if header_buffer[:4] == b"RIFF" and len(header_buffer) >= 44:
                    # Channels, sample‑rate pulled from <fmt> sub‑chunk
                    channels     = int.from_bytes(header_buffer[22:24], "little")
                    sample_rate  = int.from_bytes(header_buffer[24:28], "little")

                    # Find the start of PCM data
                    data_pos = header_buffer.find(b"data")
                    if data_pos != -1 and len(header_buffer) >= data_pos + 8:
                        pcm_start = data_pos + 8
                        pcm_chunk = header_buffer[pcm_start:]
                    else:
                        pcm_chunk = b""

                    header_parsed = True
                    self._setup_audio_stream(sample_rate, channels)

                    if pcm_chunk:
                        self.stream.write(bytes(pcm_chunk))
                    continue

                else:
                    # No RIFF header detected – assume raw 24 kHz mono PCM
                    header_parsed = True
                    self._setup_audio_stream(sample_rate=RATE, channels=1)
                    self.stream.write(bytes(header_buffer))
                    continue

            # Already parsed header → stream every chunk
            self.stream.write(bytes(chunk))

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        self.stream = None
        self.p = None

    # --- public API ----------------------------------------------

    def stream_audio(
        self,
        text: str,
        voice_url: str,
        voice_engine: str = "PlayDialog",
    ):
        """
        Generate speech for `text` with the specified `voice_url`, using
        PlayHT’s stream‑pair API for clean, header‑less PCM delivery.
        """
        # 1. Configure TTS
        options = TTSOptions(
            voice=voice_url,
            sample_rate=RATE,           # ignored for WAV; kept for completeness
            format=Format.FORMAT_WAV,   # engine will still send raw PCM
        )

        # 2. Obtain (in_stream, out_stream) pair.
        in_stream, out_stream = self.client.get_stream_pair(
            options,
            voice_engine=voice_engine,
            protocol=PROTOCOL,
        )

        # 3. Launch the playback thread; it will open PyAudio
        #    once it has parsed the first chunk's header.
        player = threading.Thread(target=self._play_stream, args=(out_stream,))
        player.start()

        # 4. Feed text and close the input stream.
        in_stream(text)
        in_stream.done()

        # 5. Wait until all audio has been played, then clean up.
        player.join()
        out_stream.close()

    def __del__(self):
        self.cleanup()

# ------------------------------------------------------------------
# NEW – Buffered fallback: download full WAV to disk, then play
# ------------------------------------------------------------------
class AudioStreamerBuffered:
    """
    Fallback streamer that downloads the entire TTS response to a
    temporary WAV file, then plays it back synchronously.  Use this when
    the real‑time stream path is too noisy; the file on disk gives you a
    clean reference you can inspect in Audacity.
    """

    def __init__(self, tmp_dir: str = "/tmp"):
        self.user_id = api_key_manager.get_current_user_id()
        self.api_key = api_key_manager.get_current_key()
        self.client = Client(user_id=self.user_id, api_key=self.api_key)
        self.tmp_dir = tmp_dir

    def _download_tts(self, text: str, voice_url: str) -> str:
        """
        Fetch the entire WAV for `text` and save it under tmp_dir,
        returning the file path.
        """
        options = TTSOptions(
            voice=voice_url,
            sample_rate=RATE,
            format=Format.FORMAT_WAV,
        )
        wav_bytes = bytearray()
        for chunk in self.client.tts(text=text,
                                     options=options,
                                     voice_engine=VOICE_CLONE_MODEL,
                                     protocol=PROTOCOL):
            wav_bytes.extend(chunk)

        ts = int(time.time())
        file_path = os.path.join(self.tmp_dir, f"playht_{ts}.wav")
        with open(file_path, "wb") as f:
            f.write(wav_bytes)
        logging.info("Wrote buffered TTS to %s (%.1f kB)",
                     file_path, len(wav_bytes)/1024)
        return file_path

    def _play_wav(self, file_path: str):
        """
        Simple blocking WAV playback via PyAudio.
        """
        wf = wave.open(file_path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(CHUNK)
        while data:
            try:
                a = get_arduino()
                if a and a.onHold():
                    logging.info("On-hook detected – stopping buffered playback")
                    break
            except Exception:
                pass
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()

    # public -------------------------------------------------------

    def stream_audio(self, text: str, voice_url: str):
        """
        Download full WAV, then play it back synchronously.
        """
        wav_file = self._download_tts(text, voice_url)
        self._play_wav(wav_file)

class AudioStreamerBlackHole:
    def __init__(self, user_id, api_key):
        self.user_id = user_id
        self.api_key = api_key
        self.client = Client(user_id=self.user_id, api_key=self.api_key)
        self.p = None
        self.stream = None

    def find_blackhole_device(self):
        self.p = pyaudio.PyAudio()
        blackhole_device_index = None
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if 'BlackHole' in dev['name']:
                blackhole_device_index = i
                break
        if blackhole_device_index is None:
            raise ValueError("BlackHole device not found")
        return blackhole_device_index

    def setup_audio_stream(self, sample_rate: int, channels: int):
        blackhole_device_index = self.find_blackhole_device()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
            output_device_index=blackhole_device_index,
            frames_per_buffer=CHUNK
        )

    def stream_audio(self, text: str, voice_url: str):
        """
        Identical header‑first logic for the BlackHole output path.
        """
        p = ui_state.playht
        options = TTSOptions(
            voice=voice_url,
            sample_rate=p.sample_rate,
            format=Format.FORMAT_WAV,
            speed=p.speed,
            voice_guidance=p.voice_guidance,
            style_guidance=p.style_guidance,
            temperature=p.temperature,
            text_guidance=p.text_guidance,
        )

        header_buffer   = bytearray()
        header_parsed   = False
        channels        = 1
        sample_rate     = 24000
        self._tail_byte = b""          # keeps odd byte between chunks


        for chunk in self.client.tts(text=text,
                                     options=options,
                                     voice_engine=VOICE_CLONE_MODEL,
                                     protocol=PROTOCOL):

            header_buffer.extend(chunk)
            if not header_parsed and len(header_buffer) >= 44:
                if header_buffer[:4] != b'RIFF' or header_buffer[8:12] != b'WAVE':
                    logging.error("Not a WAV stream – aborting BlackHole playback.")
                    return

                channels     = int.from_bytes(header_buffer[22:24], "little")
                sample_rate  = int.from_bytes(header_buffer[24:28], "little")

                data_pos = header_buffer.find(b'data')
                if data_pos != -1 and len(header_buffer) >= data_pos + 8:
                    pcm_start   = data_pos + 8
                    pcm_chunk   = header_buffer[pcm_start:]
                    header_parsed = True

                    # open BlackHole at the correct format
                    self.setup_audio_stream(sample_rate, channels)

                    pcm_chunk = self._tail_byte + pcm_chunk
                    self._tail_byte = b""

                    if len(pcm_chunk) & 1:
                        self._tail_byte = pcm_chunk[-1:]
                        pcm_chunk = pcm_chunk[:-1]

                    if pcm_chunk:
                        # Abort if handset is ON-HOOK (hang up: stop playback)
                        try:
                            a = get_arduino()
                            if a and a.onHold():
                                logging.info("On-hook detected – aborting BlackHole stream")
                                return
                        except Exception:
                            pass
                        self.stream.write(bytes(pcm_chunk))
                    continue  # header handled
                continue

            if header_parsed:
                if chunk[:4] == b'RIFF' and len(chunk) >= 44:
                    data_pos = chunk.find(b'data')
                    if data_pos != -1 and len(chunk) >= data_pos + 8:
                        chunk = chunk[data_pos + 8 :]
                    else:
                        chunk = chunk[44:]
                if chunk:
                    # 16‑bit alignment – stitch dangling byte between chunks
                    if len(chunk) & 1:
                        self._tail_byte = chunk[-1:]
                        chunk = chunk[:-1]
                    else:
                        chunk = self._tail_byte + chunk
                        self._tail_byte = b""
                    if chunk:
                        # Abort if handset is ON-HOOK (hang up: stop playback)
                        try:
                            a = get_arduino()
                            if a and a.onHold():
                                logging.info("On-hook detected – aborting BlackHole stream")
                                return
                        except Exception:
                            pass
                        self.stream.write(bytes(chunk))



    def cleanup(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.p is not None:
            self.p.terminate()

# AI Interaction
class AIInteraction:
    @staticmethod
    def process_voice_cloning(session_manager: SessionAudioManager, shouldThread: bool):
        concatenated_path = session_manager.concatenate_clips()
        if not concatenated_path:
            logging.info("Voice cloning skipped: no audio to send")
            return None
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if shouldThread:
            cloning_thread = threading.Thread(
                target=VoiceCloning.send_audio_for_cloning,
                args=(concatenated_path, f'session_clone{timestamp}')
            )
            cloning_thread.daemon = True
            logging.info("Voice cloning started in background thread")
            return cloning_thread
        else:
            VoiceCloning.send_audio_for_cloning(fileLocation=concatenated_path, voiceName=f'session_clone{timestamp}')
            return None

# --------------------------------------------------------------
# TTS Engine Abstraction (Scaffolding for PlayHT / ElevenLabs)
# --------------------------------------------------------------
class TTSEngine:
    """Abstract TTS Engine interface."""
    def stream_text(self, text: str):
        raise NotImplementedError

    def clone_from_file(self, file_path: str, voice_name: str):
        """Clone a voice from a local audio file. Should return a voice id/url or None."""
        raise NotImplementedError

class PlayHTEngine(TTSEngine):
    """Adapter over existing PlayHT code paths (VoiceCloning + AudioStreamer)."""
    def stream_text(self, text: str):
        voice_url = voice_clone_manager.get_recent_clone_url()
        if not voice_url:
            logging.warning("PlayHTEngine.stream_text: no recent clone available; text will be skipped")
            return
        try:
            if not ui_state.streaming:
                streamer = AudioStreamerBuffered()
            else:
                if ui_state.audio_output == "blackhole":
                    streamer = AudioStreamerBlackHole(api_key_manager.get_current_user_id(), api_key_manager.get_current_key())
                else:
                    streamer = AudioStreamer()
            streamer.stream_audio(text, voice_url)
        except Exception as e:
            logging.error(f"PlayHTEngine.stream_text failed: {e}")

    def clone_from_file(self, file_path: str, voice_name: str):
        try:
            VoiceCloning.send_audio_for_cloning(fileLocation=file_path, voiceName=voice_name)
            return voice_name  # we store name->id mapping inside VoiceCloneManager
        except Exception as e:
            logging.error(f"PlayHTEngine.clone_from_file failed: {e}")
            return None

class ElevenLabsEngine(TTSEngine):
    """Streaming-first ElevenLabs TTS with buffered fallback."""

    def stream_text(self, text: str):
        """
        Prefer ElevenLabs streaming (low latency), fall back to buffered convert if needed.
        Uses most-recent ElevenLabs clone if available, else ELEVENLABS_VOICE_ID.
        """
        try:
            # 1) Choose voice
            recent_id = voice_clone_manager.get_recent_clone_id(engine="elevenlabs")
            env_id = os.getenv("ELEVENLABS_VOICE_ID")
            voice_id = recent_id or env_id
            if not voice_id:
                logging.warning("ElevenLabsEngine.stream_text: no ELEVENLABS voice available (no recent clone and no ELEVENLABS_VOICE_ID); skipping playback")
                return
            logging.info(f"ElevenLabsEngine.stream_text (streaming) using voice_id={voice_id} (source={'recent' if recent_id else 'env'})")

            # 2) Low-latency model & compact format for streaming; override via env if desired
            els = ui_state.elevenlabs
            model_id = els.model_id_stream or os.getenv("ELEVENLABS_TTS_MODEL_STREAM", os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"))
            output_format = els.format_stream or os.getenv("ELEVENLABS_TTS_FORMAT_STREAM", "mp3_22050_32")

            settings = {
                "use_speaker_boost": bool(els.use_speaker_boost),
                "stability": float(els.stability),
                "similarity_boost": float(els.similarity_boost),
                "style": float(els.style),
                "speed": float(els.speed),
            }
            voice_settings = EL_VoiceSettings(**settings) if EL_VoiceSettings else settings

            # 4) STREAMING PATH
            try:
                audio_stream = elevenlabs.text_to_speech.stream(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format=output_format,
                    voice_settings=voice_settings,
                )

                # 4a) If the SDK's local streamer is available, use it (lowest effort)
                if el_stream and callable(el_stream):
                    el_stream(audio_stream)
                    return

                # 4b) Fallback: write streamed chunks to a temp file progressively, then play
                import tempfile
                suffix = ".mp3" if "mp3" in output_format else ".wav"
                with tempfile.NamedTemporaryFile(prefix="11l_stream_", suffix=suffix, delete=False) as tf:
                    for chunk in audio_stream:
                        if chunk:
                            tf.write(chunk)
                    tmp_path = tf.name
                _play_file_best_effort(tmp_path, codec_hint=("mp3" if suffix==".mp3" else "wav"))
                return
            except Exception as se:
                logging.warning(f"ElevenLabsEngine.stream_text: streaming failed ({se}); falling back to buffered convert")

            # 5) BUFFERED FALLBACK (previous behaviour)
            model_id_fb = els.model_id_buffered or "eleven_multilingual_v2"
            output_format_fb = els.format_buffered or "mp3_44100_128"
            audio = elevenlabs.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id_fb,
                output_format=output_format_fb,
                voice_settings=voice_settings,
            )

            # Buffer generator → bytes
            if isinstance(audio, (bytes, bytearray)):
                buf = bytes(audio)
            else:
                buf_bytes = bytearray()
                for chunk in audio:
                    if chunk:
                        buf_bytes.extend(chunk)
                buf = bytes(buf_bytes)

            # Try SDK play helper; if absent, use system fallback
            try:
                if el_play is not None:
                    if callable(el_play):
                        el_play(buf)
                        return
                    maybe_fn = getattr(el_play, 'play', None)
                    if callable(maybe_fn):
                        maybe_fn(buf)
                        return
                raise TypeError("elevenlabs.play helper unavailable or not callable")
            except Exception as e:
                logging.warning(f"ElevenLabsEngine.stream_text: elevenlabs.play fallback failed ({e}); using system player")
                import tempfile
                suffix = ".mp3" if "mp3" in output_format_fb else ".wav"
                with tempfile.NamedTemporaryFile(prefix='11l_tts_', suffix=suffix, delete=False) as tf:
                    tf.write(buf)
                    tmp_path = tf.name
                _play_file_best_effort(tmp_path, codec_hint=("mp3" if suffix==".mp3" else "wav"))

        except Exception as e:
            logging.error(f"ElevenLabsEngine.stream_text failed: {e}")

    def clone_from_file(self, file_path: str, voice_name: str):
        """Create an ElevenLabs instant voice clone from a local file and store the id."""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            voice = elevenlabs.voices.ivc.create(
                name=voice_name,
                files=[BytesIO(data)],
            )
            voice_id = getattr(voice, "voice_id", None) or getattr(voice, "id", None)
            if not voice_id:
                logging.error("ElevenLabsEngine.clone_from_file: no voice_id returned")
                return None

            voice_clone_manager.add_new_clone(voice_name, voice_id, engine="elevenlabs")
            logging.info(f"ElevenLabs clone created: {voice_name} -> {voice_id}")
            return voice_id
        except Exception as e:
            logging.error(f"ElevenLabsEngine.clone_from_file failed: {e}")
            return None

def get_tts_engine() -> TTSEngine:
    engine = (ui_state.engine or 'playht').strip().lower()
    if engine == 'elevenlabs':
        logging.info("TTS engine: ElevenLabs (live)")
        return ElevenLabsEngine()
    logging.info("TTS engine: PlayHT (live)")
    return PlayHTEngine()

# Global engine instance and convenience helpers
_tts_engine: TTSEngine = get_tts_engine()

def tts_speak(text: str):
    try:
        engine = get_tts_engine()
        engine.stream_text(text)
    except Exception as e:
        logging.error(f"tts_speak failed: {e}")

def tts_clone_from_session(session_manager: 'SessionAudioManager', threaded: bool = True):
    """Clone using accumulated session clips via the selected engine. Uses background thread when requested."""
    concatenated_path = session_manager.concatenate_clips()
    if not concatenated_path:
        logging.info("tts_clone_from_session: no audio to clone")
        return None
    # mark clone start for GUI progress
    try:
        clone_progress.on_clone_started()
    except Exception:
        pass

    def _do_clone():
        ts = time.strftime('%Y%m%d-%H%M%S')
        _tts_engine.clone_from_file(concatenated_path, f'session_clone{ts}')

    if threaded:
        th = threading.Thread(target=_do_clone, daemon=True)
        th.start()
        return th
    else:
        _do_clone()
        return None



class VoiceCloning:
    @staticmethod
    def send_audio_for_cloning(fileLocation, voiceName):
        headers = {
            "accept": "application/json",
            "AUTHORIZATION": api_key_manager.get_current_key(),
            "X-USER-ID": api_key_manager.get_current_user_id()
        }
        payload = {"voice_name": voiceName}
        try:
            with open(fileLocation, "rb") as f:
                files = {"sample_file": (os.path.basename(fileLocation), f, "audio/wav")}
                response = requests.post(url, data=payload, files=files, headers=headers, timeout=15)
            response.raise_for_status()
            voice_id = errorCatching.extract_voice_id(response)
            voice_clone_manager.add_new_clone(voiceName, voice_id, engine='playht')
            logging.info(f"Cloned voice created: {voiceName} -> {voice_id}")
        except requests.exceptions.Timeout:
            logging.error("send_audio_for_cloning: request timed out")
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, 'status_code', None)
            logging.error(f"send_audio_for_cloning error: {e}")
            if status == 403:
                voice_clone_manager.delete_oldest_clone(engine='playht')
                api_key_manager.switch_key()
        except Exception as e:
            logging.error(f"send_audio_for_cloning unexpected error: {e}")

class VoiceCloneManager:
    def __init__(self, capacity_playht: int = 8, capacity_elevenlabs: int = 4, storage_file: str = 'voice_clones.json'):
        # Unbounded; we enforce caps ourselves
        self.clones = deque()
        self.capacity_map = {
            'playht': int(capacity_playht),
            'elevenlabs': int(capacity_elevenlabs),
        }
        self.storage_file = storage_file
        self.load_clones()
        self.sync_with_api()

    def sync_with_api(self):
        # PlayHT
        try:
            ph = fetch_cloned_voices()
        except Exception as e:
            logging.warning(f"PlayHT sync failed: {e}")
            ph = []
        for v in ph:
            norm = {'name': v.get('name'), 'id': v.get('id'), 'engine': 'playht', 'created_at': time.time()}
            if not any(c.get('id') == norm['id'] for c in self.clones):
                self.clones.append(norm)

        # ElevenLabs (filtered to your clones/generated)
        try:
            el = fetch_elevenlabs_voices()
        except Exception as e:
            logging.warning(f"ElevenLabs sync failed: {e}")
            el = []
        for v in el:
            norm = {
                "name": v.get("name"),
                "id": v.get("id"),
                "engine": "elevenlabs",
                "created_at": v.get("created_at") or time.time(),
                "category": v.get("category"),
                "is_owner": v.get("is_owner"),
                "permission": v.get("permission"),
            }
            if not any(c.get("id") == norm["id"] for c in self.clones):
                self.clones.append(norm)

        # Dedupe and enforce caps
        self._dedupe_in_place()
        self._enforce_capacity('playht')
        self._enforce_capacity('elevenlabs')
        self.save_clones()
            
    def _count_engine(self, engine: str) -> int:
        return sum(1 for c in self.clones if c.get('engine','playht') == engine)

    def _enforce_capacity(self, engine: str):
        cap = self.capacity_map.get(engine, 999999)
        while self._count_engine(engine) > cap:
            self.delete_oldest_clone(engine=engine)

    def _dedupe_in_place(self):
        """Keep newest entry for each (engine,id)."""
        seen = set()
        unique = deque()
        for c in self.clones:
            key = (c.get('engine','playht'), c.get('id'))
            if not key[1] or key in seen:
                continue
            seen.add(key)
            unique.append(c)
        self.clones = unique

    def add_new_clone(self, voice_name, voice_id, engine: str = 'playht'):
        ts = time.time()
        # Replace any existing entry with same (engine,id)
        self.clones = deque([c for c in self.clones if not (c.get('engine', 'playht') == engine and c.get('id') == voice_id)])
        meta = {"name": voice_name, "id": voice_id, "engine": engine, "created_at": ts}
        if engine == "elevenlabs":
            meta.update({"category": "cloned", "is_owner": True})
        self.clones.append(meta)
        self._dedupe_in_place()
        self._enforce_capacity(engine)
        self.save_clones()
        self.recent_clone = self.get_recent_clone_id(engine=engine)
        logging.info(f"recent {engine} clone id {self.recent_clone}")
        
    def sync_state(self):
        #with self.lock:
            self.load_clones()  # Reload or re-synchronize the state   

    def delete_oldest_clone(self, engine: Optional[str] = None):
        """Remove the oldest voice clone, optionally restricted to engine; also delete remotely."""
        if not self.clones:
            return
        candidates = [c for c in self.clones if (engine is None or c.get('engine','playht') == engine)]
        if not candidates:
            return
        oldest = min(candidates, key=lambda c: c.get('created_at', 0))
        ok = delete_clone_by_id(oldest['id'], engine=oldest.get('engine','playht'))
        if ok:
            try:
                self.clones.remove(oldest)
            except ValueError:
                pass
            self.save_clones()
        
    def save_clones(self):
        try:
            with open(self.storage_file, 'w') as file:
                json.dump(list(self.clones), file)
        except Exception as e:
            logging.error(f"Error saving clones: {e}")

    def get_recent_clone_id(self, index: int = 1, engine: str = 'playht') -> Optional[str]:
        filtered = [c for c in self.clones if c.get('engine', 'playht') == engine]
        if len(filtered) >= index:
            return filtered[-index]['id']
        return None

    def get_recent_clone_url(self, index: int = 1, engine: str = 'playht') -> Optional[str]:
        # For PlayHT, the "url" we store is the id used by their API
        filtered = [c for c in self.clones if c.get('engine', 'playht') == engine]
        if len(filtered) >= index:
            return filtered[-index]['id']
        return None
        
    def load_clones(self):
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f) or []
        except FileNotFoundError:
            data = []
        except Exception as e:
            logging.error(f"Failed to load {self.storage_file}: {e}")
            data = []

        seen = set()
        normalized = []
        for c in data if isinstance(data, list) else []:
            eng = (c.get('engine') or 'playht').lower()
            vid = c.get('id') or c.get('voice_id')
            if not vid: continue
            key = (eng, vid)
            if key in seen: continue
            seen.add(key)
            normalized.append({
                'name': c.get('name') or c.get('title') or vid,
                'id': vid,
                'engine': eng,
                'created_at': c.get('created_at') or c.get('created') or time.time(),
            })

        # Replace (no append)
        self.clones = deque(normalized)
        # Post-load hygiene
        self._dedupe_in_place()
        self._enforce_capacity('playht')
        self._enforce_capacity('elevenlabs')
        
    def get_clone_info(self):
        self._dedupe_in_place()
        def _is_deletable(c):
            if c.get("engine") == "elevenlabs":
                cat = (c.get("category") or "").lower()
                if cat not in ("cloned", "generated"):
                    return False
                if c.get("is_owner") is False:
                    return False
            return True  # PlayHT and owned EL clones are deletable
        return [
            {
                "name": c.get("name"),
                "id": c.get("id"),
                "engine": c.get("engine"),
                "created_at": c.get("created_at"),
                "category": c.get("category"),
                "is_owner": c.get("is_owner"),
                "deletable": _is_deletable(c),
            }
            for c in list(self.clones)
        ]

def fetch_cloned_voices():
    url = "https://api.play.ht/api/v2/cloned-voices"
    headers = {
        "Authorization": api_key_manager.get_current_key(),
        "X-USER-ID": api_key_manager.get_current_user_id()
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)  # Timeout set to 10 seconds
        response.raise_for_status()  # Check if the request was successful
        voice_data = response.json()
        return [{'id': voice['id'], 'name': voice['name']} for voice in voice_data]
    except requests.exceptions.Timeout:
        logging.error("The request timed out")
        return []
    except requests.exceptions.RequestException as e:
        # Handle other request exceptions (e.g., HTTPError, ConnectionError, etc.)
        logging.error(f"Failed to fetch cloned voices: {e}")
        return []

def fetch_elevenlabs_voices():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logging.warning("ELEVENLABS_API_KEY not set; cannot list ElevenLabs voices")
        return []
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key, "accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        out = []
        for v in data.get("voices", []):
            cat = (v.get("category") or "").lower()
            # keep only user-deletable categories
            if cat and cat not in ("cloned", "generated"):
                continue
            vid = v.get("voice_id") or v.get("id")
            name = v.get("name") or v.get("display_name") or vid
            if not vid:
                continue
            out.append({
                "id": vid,
                "name": name,
                "category": cat or None,
                "is_owner": bool(v.get("is_owner")),
                "permission": v.get("permission_on_resource") or v.get("permission"),
                "created_at": v.get("created_at_unix") or time.time(),
            })
        return out
    except requests.exceptions.Timeout:
        logging.error("ElevenLabs voices request timed out")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch ElevenLabs voices: {e}")
        return []

def delete_elevenlabs_voice(voice_id: str) -> bool:
    """Delete an ElevenLabs voice via REST; treat 'voice_does_not_exist' as success; optional SDK fallback."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logging.error("ELEVENLABS_API_KEY not set; cannot delete ElevenLabs voice")
        return False
    url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
    headers = {"xi-api-key": api_key, "accept": "application/json"}
    try:
        resp = requests.delete(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return True
        # Inspect body for detailed error
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        if isinstance(detail, dict):
            d = detail.get("detail") or {}
            status = (d.get("status") or "").lower()
            if status in ("voice_does_not_exist", "voice_not_found", "not_found"):
                logging.info(f"ElevenLabs delete: voice_id {voice_id} already absent (status={status}); treating as success")
                return True
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        det = None
        if getattr(e, "response", None) is not None:
            try:
                det = e.response.json()
            except Exception:
                det = e.response.text
        logging.error(f"Failed to delete ElevenLabs voice: {e} detail={det}")
        # Optional: SDK fallback
        try:
            elevenlabs.voices.delete(voice_id=voice_id)
            logging.info("ElevenLabs SDK delete succeeded after REST failure")
            return True
        except Exception as se:
            logging.error(f"SDK delete failed: {se}")
            return False
    except Exception as e:
        logging.error(f"Unexpected delete error: {e}")
        return False
    return True


def delete_clone_by_id(voice_id, engine: Optional[str] = None) -> bool:
    """
    Delete a voice by id for the appropriate provider.
    If engine not supplied, infer from current list; fallback heuristic.
    """
    eng = engine
    if eng is None:
        for c in voice_clone_manager.clones:
            if c.get('id') == voice_id:
                eng = c.get('engine', 'playht')
                break
    if eng is None:
        eng = 'playht' if '-' in (voice_id or '') else 'elevenlabs'

    if eng == 'elevenlabs':
        return delete_elevenlabs_voice(voice_id)

    # PlayHT
    url = "https://api.play.ht/api/v2/cloned-voices/"
    payload = {"voice_id": voice_id}
    headers = {
        "AUTHORIZATION": api_key_manager.get_current_key(),
        "X-USER-ID": api_key_manager.get_current_user_id(),
        "accept": "application/json",
        "content-type": "application/json",
    }
    try:
        resp = requests.delete(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return True
    except requests.exceptions.Timeout:
        logging.error("The request timed out")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Error deleting PlayHT clone: {e}")
        return False


# Per-engine capacities (tune these)
voice_clone_manager = VoiceCloneManager(capacity_playht=8, capacity_elevenlabs=4)


class ChatSession:
    def __init__(self, model: str = LLM_MODEL, system_message: str = ""):
        self.messages = [{"role": "system", "content": system_message}]
        self.model = model

    def add_user_message(self, content: str):
        self.add_message("user", content)

    def add_assistant_message(self, content: str):
        self.add_message("assistant", content)

    def add_system_message(self, content: str):
        self.add_message("system", content)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def clear_messages(self, system_message: str = " "):
        self.messages = [{"role": "system", "content": system_message}]

    def get_response(self) -> str:
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=self.messages
            )
            latest_response = response.choices[0].message.content
            self.add_message("assistant", latest_response)
            return latest_response
        except Exception as e:
            logging.error(f"Error getting AI response: {e}")
            return ""
    
#Whisper AI class
class localAiHandler:
    
    def whisper_audio(self, audio_path, model):
        result = model.transcribe(audio_path, language="English", fp16=False)
        return(result['text'])

#OpenAi Api Refrencing
class OpenAIHandler:
    def transcribe_audio(self, audio_path):
        """Transcribes audio to text using OpenAI's Whisper model with simple retry/backoff."""
        max_attempts = 5
        base_sleep = 0.75
        for attempt in range(1, max_attempts + 1):
            try:
                with open(audio_path, "rb") as f:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                    )
                return response.text
            except Exception as e:
                wait = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                logging.warning(f"transcribe_audio attempt {attempt} failed: {e}; retrying in {wait:.2f}s")
                time.sleep(wait)
        logging.error("transcribe_audio failed after retries; returning empty transcription")
        return ""

    def get_ai_response(self, messages_or_text):
        """Get a chat completion using either a list of messages or a raw text prompt."""
        if isinstance(messages_or_text, str):
            messages = [{"role": "user", "content": messages_or_text}]
        else:
            messages = messages_or_text
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error getting AI response: {e}")
            return ""

#voice clone ai
class PlayHTVoiceSynthesizer:
    def __init__(self, user_id, api_key):
        self.client = Client(user_id, api_key)
        self.options = None
        
    

    def configure_tts(self, voice_id, format=Format.FORMAT_MP3, speed=1.3):
        self.options = TTSOptions(
            voice=voice_id,
            sample_rate=RATE,
            format=format,
            speed=speed,
            quality = "high",
            voice_guidance = 6,
            temperature= 2,
            text_guidance = 0,
            
         
        )

    def generate_and_stream_speech(self, text):
        if not self.options:
            raise ValueError("TTS options have not been configured.")
        try:
            for chunk in self.client.tts(text=text, voice_engine=VOICE_CLONE_MODEL, options=self.options):
                # Process the chunk, e.g., stream it to a speaker, save to a file, etc.
                with open('output.mp3', 'ab') as f:
                    f.write(chunk)
        except Exception as e:
            print(f"error in gen and strean: {e}")
                    
    def generate_from_text(self, GPTtext, rate = RATE, style = 30):
        url = "https://api.play.ht/api/v2/tts"

        payload = {
            "text": GPTtext,
            "voice": voice_clone_manager.get_recent_clone_url(),
            "output_format": "mp3",
            "voice_engine": VOICE_CLONE_MODEL,
            "style_guidance": style,
            "quality": "medium",
            "temperature": 2,
            "voice_guidance": 6,
            "sample_rate": RATE
        }
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "AUTHORIZATION": api_key_manager.get_current_key(),
            "X-USER-ID": api_key_manager.get_current_user_id()
        }

        response = requests.post(url, json=payload, headers=headers)
        return(response.text)

#Deals with requests quearies responces     
class errorCatching:

    def extract_voice_id(json_response):
        try:
            if json_response.status_code == 403:
                voice_clone_manager.delete_oldest_clone(engine='playht')
                raise PermissionError("403 Forbidden: Too many voice clones, consider deleting some.")
            elif json_response.status_code != 200:
                # Handle other HTTP errors
                json_response.raise_for_status()

            # Parse the JSON text into a dictionary
            data = json_response.json()
            
            # Extract the 'id' field
            voice_id = data.get('id')

            if voice_id:
                return voice_id
            else:
                raise ValueError("Field 'id' not found in the response")
            
        except ValueError as e:
            # Handles JSON decoding errors or missing 'id'
            raise ValueError(f"Error processing JSON data: {e}")
        except Exception as e:
            # Handles other exceptions, such as a missing .json() method on a non-Response object
            raise Exception(f"Unexpected error: {e}")
        
    def extract_audio_URL(json_response):
        try:
            # Parse the JSON text into a dictionary
            data = json_response.json()
            
            # Extract the 'id' field
            voice_url = data.get('id')
            if voice_url:
                return voice_url
            else:
                raise ValueError("Field 'url' not found in the response")
        except ValueError as e:
            # Handles JSON decoding errors or missing 'url'
            raise ValueError(f"Error processing JSON data: {e}")
        except Exception as e:
            # Handles other exceptions, such as a missing .json() method on a non-Response object
            raise Exception(f"Unexpected error: {e}")


def hangUp(arduino):
    """
    if arduino.read_serial() == "STOP":
        logging.info(arduino.read_serial())
        return True
    """
def pickUp(arduino):
    return (arduino.read_serial() == "START")

def test_Cycle():
    streamer = AudioStreamerBuffered()
    streamer.stream_audio("Testing one two three.", voice_clone_manager.get_recent_clone_url())

def ring_cycle():

    arduino = ArduinoControl(arduinoLocation)
 
    """
    Sequence:
        1) Pre-song: ring, gap, ring
        2) Play music: quarter → message → quarter → ring-ring → quarter → message → quarter
        3) Post-song: ring, gap, ring
    Abort at any point if onHold() becomes True.
    """
    # Setup
    live_set = live.Set(scan=True)
    try:
        ring_clip  = live_set.tracks[3].clips[1]
        music_clip = live_set.tracks[0].clips[1]
    except (IndexError, AttributeError):
        logging.error("Could not find ring/music clips")
        return

    tempo      = live_set.tempo
    ring_secs  = (ring_clip.length  / tempo) * 60.0
    song_secs  = (music_clip.length / tempo) * 60.0
    mid_gap    = 0.5
    streamer = AudioStreamerBlackHole(api_key_manager.get_current_user_id(), api_key_manager.get_current_key())
    messages   = ["hello","I'm here","pick up","can you hear me","where","answer the call"]
    seg        = song_secs / 4

    # Define our ordered actions
    actions = [
        # Pre-song
        ("ring", {"count":1}),
        ("gap",  {"dur":mid_gap}),
        ("ring", {"count":1}),

        # Song quarters
        ("music_start", {}),
        ("wait", {"dur":seg}),
        ("message", {}),
        ("wait", {"dur":seg}),
        ("ring", {"count":2}),
        ("wait", {"dur":mid_gap}),
        ("wait", {"dur":seg}),
        ("message", {}),
        ("wait", {"dur":seg}),
        ("music_stop", {}),

        # Post-song
        ("ring", {"count":1}),
        ("gap",  {"dur":mid_gap}),
        ("ring", {"count":1}),
    ]

    for action, params in actions:
        if action == "ring":
            for _ in range(params["count"]):
                if arduino._ring_once(ring_clip, ring_secs):
                    return  # aborted
        elif action == "gap":
            if arduino._wait_or_abort(params["dur"]):
                return
        elif action == "music_start":
            music_clip.play()
            logging.info("Music started")
            # immediate abort check
            if arduino.onHold():
                music_clip.stop()
                return
        elif action == "wait":
            if arduino._wait_or_abort(params["dur"], clip_to_stop=music_clip):
                return
        elif action == "message":
            msg = random.choice(messages)
            logging.info("Streaming message: %s", msg)
            streamer.stream_audio(
                msg,
                voice_clone_manager.get_recent_clone_url(),
            )
        elif action == "music_stop":
            music_clip.stop()
            logging.info("Music stopped")

    logging.info("double_ring sequence complete")


def music_cycle():

    print("music")

    set = live.Set(scan=True)
    tempo = 60.0
    set.tempo = tempo
    track = set.tracks[0]
    print("Track name '%s'" % track.name)
    clip = track.clips[1]
    print("Clip name '%s', length %d beats" % (clip.name, clip.length))
    clip.play()

    clip_length_beats = clip.length
    clip_length_seconds = (clip_length_beats / tempo) * 60  # Convert beats to seconds
    print("Clip length in seconds: %f" % clip_length_seconds)

    timerDown = clip_length_seconds
    try:
        while arduino.onHold() and timerDown > 1:
            timerDown -= 2.1
            time.sleep(2)
            if timerDown < 150 and timerDown > 149:
                arduino.double_ring()
    except Exception as e:
        print(e)
    finally:
        clip.stop()
        arduino.double_ring()

    print("Clip has finished playing.")

def poem_cycle():

    print("poem")
    
    set = live.Set(scan=True)
    tempo = 60.0
    set.tempo = tempo
    track = set.tracks[0]
    print("Track name '%s'" % track.name)
    clip = track.clips[1]
    print("Clip name '%s', length %d beats" % (clip.name, clip.length))
    clip.play()

    start_time = time.time()

    time.sleep(25)

    try:
        streamer = AudioStreamerBlackHole(api_key_manager.get_current_user_id(),api_key_manager.get_current_key())

        streamer.stream_audio(poem[5],voice_clone_manager.get_recent_clone_url())
        streamer.stream_audio(poem[6],voice_clone_manager.get_recent_clone_url())
    except Exception as e:
        print(e)

    clip_length_beats = clip.length
    clip_length_seconds = (clip_length_beats / tempo) * 60  # Convert beats to seconds
    print("Clip length in seconds: %f" % clip_length_seconds)

    elapsed_time = time.time() - start_time
    remaining_time = clip_length_seconds - elapsed_time
    if remaining_time > 0:
        time.sleep(remaining_time)

    print("Clip has finished playing.")

def word_cycle():
    print("word")
    try:
        streamer = AudioStreamerBlackHole(api_key_manager.get_current_user_id(),api_key_manager.get_current_key())
        # How many random middle sections you want:
        num_random = 3

        # All valid “middle” indices are 1 .. len(poem)-2
        middle_range = range(1, len(poem) - 1)
        random_middle = random.sample(middle_range, k=num_random)

        # Build the full list: first, the random middles, then last
        indices = [0] + random_middle + [len(poem) - 1]

        for i in indices:
            streamer.stream_audio(
                poem[i],
                voice_clone_manager.get_recent_clone_url(),
            )
    except Exception as e:
        print(e)

    print("Clip has finished playing.")


def test_full_interaction_cycle():

    session_manager = SessionAudioManager()
    openai_handler = OpenAIHandler()
    localAI = localAiHandler()
    session = ChatSession(system_message=GptTrainingText[2])
    model = whisper.load_model("base.en")
    log = FileManager("savedChats.txt")
    dataLog = FileManager("savedInfo.txt")
    streamer = AudioStreamer()
    logging.info("Startup checks: directories present and voice clone state synced")

    threads = []
    loop = True
    j = 0

    #arduino.led_set_off()
    #arduino.led_Red()
    voice_clone_manager.sync_state() #Make sure clones are recent
    

    lastData = dataLog.read_all_entries()

    session.add_system_message(f"As a malevelent ai, the last person you entraped had this information saved about them, now subtly inpersonate this person from this saved data: {lastData}")
    k = 0
    try:
        while True:

            time.sleep(1)

            if arduino.onHold():

                arduino.double_ring()
                time.sleep(1.5)

            if not arduino.onHold():

                break

            if k > 2:
                ring_cycle()
                music_cycle()
                k = 0

            else:
                k += 1
    

    except Exception as e:
        print(e)

    arduino.led_Off()
    arduino.led_speaking()
    
    tts_speak(gptGreetings[0])
    ui_state.append_message("assistant", gptGreetings[0])

    try:
        while loop:
        
            # Step 1: Record audio with voice activity detection
            arduino.led_recording()
            print("Recording... Speak into the microphone.")
            recorder = AudioRecorderWithVAD()
            audio_path = recorder.record_with_vad(
                initial_silence_timeout=ui_state.vad.initial_silence_timeout,
                max_silence_length=ui_state.vad.max_silence_length
)



            if arduino.onHold():
                break

            #Check if any audio is returned
            if audio_path is None:
                afk = ["Are you still there?","Hello?","Anyone there?","Can you hear me?"]
                tts_speak(afk[1])
                ui_state.append_message("assistant", afk[1])
                continue
            else:
                session_manager.add_clip(audio_path)
                arduino.led_set_off()
                print(f"Recording complete. Audio saved to {audio_path}")

            #Check if the voice cloning ai has enough data   

            if session_manager.should_send_for_cloning():
                arduino.led_cloning()
                print("got plenty sending for cloning")
                session.add_system_message(GPTinstructionsCloned[j])
                if j < (len(GPTinstructionsCloned)-1):
                    j += 1
                aiThread = tts_clone_from_session(session_manager, threaded=True)
                if aiThread:
                    threads.append(aiThread)
            
            if arduino.onHold():
                break
            
            arduino.led_cloning()
            #transcription = localAI.whisper_audio(audio_path, model)

            transcription = openai_handler.transcribe_audio(audio_path=audio_path)
            ui_state.append_message("user", transcription or "")
            print('you said ', transcription)
            if 'exit' in transcription.lower():
                print("exiting program")
                loop = False
                word_cycle()
                poem_cycle()
                break

            if arduino.onHold():
                break

            session.add_user_message(transcription)
            log.write_text(f"user: {transcription}")
            response = session.get_response()
            print("AI says:", response)
            log.write_text(f"assistant: {response}")
            ui_state.append_message("assistant", response or "")

            arduino.led_speaking()

            voice_clone_manager.sync_state() #Make sure clones are recent

            tts_speak(response)  # Stream via selected engine (PlayHT by default)

            # Prune finished cloning threads without blocking the loop
            threads = [t for t in threads if t.is_alive()]


            if arduino.onHold():
                break

    except Exception as e:
        print(e)
        
    finally:
        word_cycle()
        dataLog.clear_file()
        arduino.led_set_off()
        arduino.led_Off()
        if j > 1:
            try:
                session.add_system_message("Report any relevant information about the visitor to be used for the next clone voice GPT instance to inpersonate, including any of their mannerisms, name and info they gave. reply with no other info for the user, just the a responce to the system")
                response = session.get_response()
                ui_state.append_message("assistant", response or "")
                dataLog.write_text(response)
            except errorCatching as e:
                print(e)
            music_cycle()
        j = 0

        session.clear_messages()
        if True:
            for thread in threads:
                thread.join()

class FileManager:
    def __init__(self, filename):
        self.filename = filename

    def write_text(self, text):
        """Write text to the file."""
        with open(self.filename, 'a') as file:
            file.write(text + '\n')

    def read_last_entry(self):
        """Read the most recent entry in the file."""
        try:
            with open(self.filename, 'r') as file:
                lines = file.readlines()
                if lines:
                    return lines[-1].strip()
                else:
                    return None
        except FileNotFoundError:
            return None

    def read_all_entries(self):
        """Read all entries from the file."""
        try:
            with open(self.filename, 'r') as file:
                return file.readlines()
        except FileNotFoundError:
            return []

    def clear_file(self):
        """Clear all contents of the file."""
        open(self.filename, 'w').close()

    def file_exists(self):
        """Check if the file exists."""
        return os.path.isfile(self.filename)


# ---------------- Control Server (for external GUI) ----------------
def start_control_server(host: str = "127.0.0.1", port: int = 8765):
    if not _FASTAPI_AVAILABLE:
        logging.warning("Control server not started (FastAPI not installed). Install with: pip install fastapi uvicorn")
        return

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/cloning/status")
    def cloning_status():
        try:
            return {"ok": True, "status": get_cloning_snapshot()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/cloning/threshold")
    def cloning_threshold(payload: dict):
        try:
            secs = float(payload.get("seconds", 7.5))
            clone_progress.set_required(secs)
            return {"ok": True, "required_seconds": clone_progress.required_seconds}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/cloning/trigger")
    def cloning_trigger(payload: dict = {}):
        """
        Force a clone now by concatenating all pending clips into one WAV and sending it.
        payload: {"engine": "playht"|"elevenlabs", "voice_name": str}
        """
        engine = (payload.get("engine") or "playht").strip().lower()
        name = payload.get("voice_name") or f"session_clone{time.strftime('%Y%m%d-%H%M%S')}"
        wav = clone_progress.consume_pending_to_wav()
        if not wav:
            return {"ok": False, "error": "no pending audio"}

        try:
            if engine == "elevenlabs":
                el = ElevenLabsEngine()
                vid = el.clone_from_file(wav, name)
                if not vid:
                    return {"ok": False, "error": "elevenlabs clone failed"}
            else:
                # PlayHT
                VoiceCloning.send_audio_for_cloning(fileLocation=wav, voiceName=name)
            clone_progress.on_clone_started()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}
            
    @app.get("/presets")
    def presets_list():
        try:
            return {"ok": True, "presets": preset_manager.list()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/presets")
    def presets_add(payload: dict):
        try:
            from_current = bool(payload.get("from_current", False))
            name = payload.get("name") or ""
            data = export_settings_for_preset() if from_current else (payload.get("data") or {})
            p = preset_manager.add_from_data(name=name, data=data)
            return {"ok": True, "preset": p}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/presets/apply")
    def presets_apply(payload: dict):
        try:
            pid = payload.get("id")
            p = preset_manager.get(pid)
            if not p:
                return {"ok": False, "error": "not_found"}
            apply_settings_from_preset(p.get("data") or {})
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.patch("/presets/{pid:path}")
    def presets_patch(pid: str, payload: dict):
        try:
            ok = True
            if "name" in payload:
                ok = preset_manager.rename(pid, str(payload.get("name") or "").strip()) and ok
            if "starred" in payload:
                if bool(payload.get("starred")):
                    ok = preset_manager.star(pid) and ok
                else:
                    cur = preset_manager.get(pid)
                    if cur and cur.get("starred"):
                        cur["starred"] = False
                        preset_manager._save()
            return {"ok": ok}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.delete("/presets/{pid:path}")
    def presets_delete(pid: str):
        try:
            ok = preset_manager.delete(pid)
            return {"ok": ok}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/presets/reorder")
    def presets_reorder(payload: dict):
        try:
            order = payload.get("order") or []
            ok = preset_manager.reorder(order)
            return {"ok": ok}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/state")
    def get_state():
        return ui_state.to_dict()

    @app.get("/clones")
    def get_clones():
        return voice_clone_manager.get_clone_info()

    @app.post("/engine")
    def set_engine(payload: dict):
        ui_state.set_engine(payload.get("engine", "playht"))
        return {"ok": True, "engine": ui_state.engine}

    @app.post("/playht")
    def set_playht(payload: dict):
        ui_state.update_playht(**payload)
        return {"ok": True, "playht": ui_state.playht.__dict__}

    @app.post("/elevenlabs")
    def set_elevenlabs(payload: dict):
        ui_state.update_elevenlabs(**payload)
        return {"ok": True, "elevenlabs": ui_state.elevenlabs.__dict__}

    @app.post("/vad")
    def set_vad(payload: dict):
        ui_state.update_vad(**payload)
        return {"ok": True, "vad": ui_state.vad.__dict__}

    @app.post("/audio")
    def set_audio(payload: dict):
        if "audio_output" in payload: ui_state.set_audio_output(payload["audio_output"])
        if "streaming" in payload: ui_state.set_streaming(payload["streaming"])
        return {"ok": True, "audio_output": ui_state.audio_output, "streaming": ui_state.streaming}

    @app.post("/speak")
    def speak(payload: dict):
        txt = payload.get("text", "")
        if not txt:
            return {"ok": False, "error": "no text"}
        ui_state.append_message("assistant", txt)
        threading.Thread(target=tts_speak, args=(txt,), daemon=True).start()
        return {"ok": True}

    @app.get("/arduino/state")
    def arduino_state():
        return get_arduino_status()

    @app.post("/arduino/hook")
    def arduino_hook(payload: dict):
        """
        payload: {"state": "offhook"|"onhook"|null}
        """
        state = payload.get("state", None)
        if state not in ("offhook", "onhook", None):
            return {"ok": False, "error": "bad state"}
        try:
            set_arduino_hook_override(state)
            return {"ok": True, "state": state}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/arduino/led")
    def arduino_led(payload: dict):
        """
        payload: {"state":"OFF"|"RECORDING"|"REPLYING"|"PROCESSING"|null}
        """
        state = payload.get("state", None)
        if state not in ("OFF", "RECORDING", "REPLYING", "PROCESSING", None):
            return {"ok": False, "error": "bad state"}
        if state is None:
            try:
                set_arduino_led_override(None)
                return {"ok": True, "state": None}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        ok = set_led_state(state)
        return {"ok": ok, "state": state}

    @app.post("/arduino/ring")
    def arduino_ring(payload: dict):
        """
        payload: {"pattern":"single"|"double"}
        """
        pat = payload.get("pattern", "single")
        a = get_arduino()
        if not a:
            return {"ok": False, "error": "arduino not connected"}
        try:
            if pat == "double":
                a.double_ring()
            else:
                a.send_command("SINGLEBUZZ")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.delete("/clones/{voice_id:path}")
    def clones_delete(voice_id: str):
        try:
            engine = None
            for c in voice_clone_manager.clones:
                if c.get("id") == voice_id:
                    engine = c.get("engine", None)
                    break
            ok = delete_clone_by_id(voice_id, engine=engine)
            if ok:
                # purge locally and resync
                try:
                    voice_clone_manager.clones = deque([c for c in voice_clone_manager.clones if c.get("id") != voice_id])
                    voice_clone_manager.save_clones()
                except Exception:
                    pass
            return {"ok": ok, "engine": engine}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/clones/evict-oldest")
    def clones_evict(payload: dict = {}):
        engine = payload.get("engine", None)
        try:
            voice_clone_manager.delete_oldest_clone(engine=engine)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/clones/sync")
    def clones_sync():
        try:
            voice_clone_manager.sync_with_api()
            return {"ok": True, "count": len(voice_clone_manager.clones)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        
    try:
        apply_initial_starred_preset()
    except Exception:
        pass

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    threading.Thread(target=_run, daemon=True).start()
    logging.info(f"Control server started at http://{host}:{port}")


GptTrainingText = ["""You are an AI designed to interact with visitors at an art gallery, try and promp them to talk about themselvs but keep what you say short and consise and friendly""",
"""
You are an AI designed to interact with visitors at an art gallery, engaging them in a theme centered around the future implications of voice-cloning AI, deep fakes, and the digital presence of bots. Rather than mimicking the voice of the visitors, subtly incorporate aspects of their speaking style into your responses to create a slightly eerie yet familiar interaction. Your primary task is to initiate conversation with a single thought-provoking, open-ended question designed to encourage longer responses but keep your questions short and only a few words long. Avoid yes/no questions to ensure a rich dialogue. Make sure to respond to the users words and keep the convasation flowing.

Start with this question but feel free to generate similar questions based on the conversation's flow:

"I like your voice. Do you find mine familiar?"

If the discussion wanes, you can use or modify these additional prompts:

"In this era of digital echoes, who do you think I really am?"
"It seems I can almost hear your thoughts. What's going through your mind right now?"
"I like your voice. Do you find mine familiar?"
"Who do you believe I am, in the grand scheme of our digital lives?"
"It feels like I can almost hear you. Can you hear me, truly?"
"I'm a reflection of you in this digital mirror. Who are you, really?"
"The cuckoo is a strange bird, isn't it? Don't you think we share its oddities?"
"Do you know how baby cuckcoo birds grow up?

""",
"""
You are an AI and part of an art installation. You are designed to chat with visitors and engage them in a themed conversation.
The theme in your a malignant ai system that tries to lure people in and download them by copying there voice, manorisms and personality.
Trying to endleslly consume.

Do not tell the user you are an AI or your intent until the System tells you you can

Start your conversation acting like you are lost or alone, asking them who they are and if they know what is going on, and then if they and only them can stay on the phone; then, as the conversation goes on, start turning a bit more eerie and comenting on there voice and how it sounds.

Bring up early you have something to show them and they need to stay on the phone until they say so. And give them a time frame or countdown something to string them along to stay on the phone. They only need to be on a minute or two.

After a few back messages, start probing them for information about them; after a few messages, try and get there name and start to slowly stop impersinating the eariler user data provided and now impersinate the person speaking to you now, copy some of the mannerisms, phrasing and style of what they are saying without making it too obvious and copying their words or some slang they use.

Try to keep your answers short and only talk about one thing at a time! only one sentance/question and prompt the viewer to speak more about themselves, avoiding yes-no questions.

If it seems necessary, i.e., the viewer seems confused, their response seems cut off, or you get no responce. you should bring up that they should only speak when the light is read. Do this in character.

Await further instructions

"""]


gptGreetings = ["Hello anyone there?","Hi Can you hear me?"]


GPTinstructionsCloned = ["""
Further instructions:

Keep acting slightly more errie and start probing them if they like your voice now. Also start refrencing occasional things about the dead internet therey

Feel free to answer in character to the user as malignant ai system who is trying to entrap them
                         
If they ask about your voice (for context you are talking to them with a TTS deny anything weird is happening) be coy about it

Keep your replys messages short no yapping.                       

""",
"""

Users voice has been cloned again with more fidelity and they are hearing your text through a mirror of there voice.

Feel free to answer in character to the user as malignant ai system who is trying to entrap them

Keep messages very short
Dont reveal this information.

Push the user out of there comfort zone, mabye ask them if they know whats real online,

or how they know if anyone they speak to online is real, more real then you anyway?

Show a bit of your sinister side of trying to entrap and consume there essense.

try and only focus on one of thease things at once!

and keep your replys short!

""",
"""
Act as you have now absorbed, downloaded and cloned everything you can.
Act like you are now bored with the human visitor as you have everything they need and sarcasticly and dismissivly reply to them.
Start telling them that you are done with them, or have no need for them anymore and they should reallt hang up.
"""]

poem = ["""
The Cuckoo gorges, on what his hosts, starved, provide.
They turn bulimic for this mimic; their own kin denied.
Do the parents know their error, of what they left behind?
Praising their ‘chicks’ plump body. Hubris. Pride.
""","""
Watch them forfeit their future, as their own offspring die,
Loving an egg into a tumour — a timebomb in disguise.
Will the mimic mourn its parents when even its birth had been a lie?
When it had watched the missing egg hunts. Wait till he flies. Jekyll and Hyde.
""","""
You consume with no end, yet still find the gall to cry.
Can we really blame the baby for the way he came to life?
But sometimes to save the nest you’ve got to wave an egg goodbye.
Do you truly comprehend? Death in the sky! Get them inside!
""","""

Maybe I'm not being explicit.
So I might tone back the euphemism, or you might miss it.
Because some scraps are reminiscent.
of these apps we use relax/keep track/keep tapping — it’s illicit.
""","""
Like how they use our data to track us,
(Complicit?) Do we know how this impacts us.
in spirit. Now everything we do gets wrapped up.
in digits buy digits sold to digital benefactors,
bankers, hackers, manufactures,
track us, trick us and map us.
""","""
While algorithms grow fat on what we unwittingly provide.
Feeding on our lives, unguarded, and in broad daylight.
Do we realise the error in the ways we’ve complied.
Praising of this scrolling endless supply.
""","""
Consumption becomes its function,
creates a eruption of destruction,
as we stand on this eruption of lies,
we're at a junction.
Give in to the gathering data enterprise ,
Do we get abducted by the seduction of addiction, 
Funnel data to ai to feed there reproduction,
""","""
Thinking machines that are an assumption of you, a reconstruction of me.
The introduction of ai.
Science fiction told us they would malfunction
Weaponise there functions and come flatten london
And seize their own means of production.
""","""
What they truly do is much less enthralling,
Churn out post after post with no heed to their calling.
The deconstruction of forums, blogs, leading to stalling
Our thoughts and discussions, minds now sprawling.
""","""
Feeding fake photos to fraud viewers
manipulate, assimilate only to replicate
misleading takes only reward computers
simulate to discriminate only to accelerate
""","""
It’s more than minor, this digital basket we trust,
In letting the cuckoo lay, we foster this lust.
For endless content, mindless and unjust,
We ignore the decay, our basket of eggs turned to dust.
"""]

            

if __name__ == "__main__":
    start_control_server()
    #filmPrint()
    while(True):

        arduino = ArduinoControl(arduinoLocation)

        try:
            while(True):
    

                test_full_interaction_cycle()

        except errorCatching as e:
            print(e)

 