#------------------------------------------------------------------
#------------------------ 2024 - Xach Hill ------------------------
#-------------------------- Use for good --------------------------
#------------------------------------------------------------------

import os
import glob
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
# Lazy-import heavy modules like Whisper (Torch) to reduce startup time
whisper = None  # set when first needed
from collections import deque
from openai import OpenAI
from pyht import Client, TTSOptions, Format
import urllib.parse

# Import streamToSpeakers from a sibling module (no package required)
try:
    from outScript import streamToSpeakers  # preferred: outScript.py in same folder
except ModuleNotFoundError:
    try:
        from tester import streamToSpeakers  # fallback: tester.py in same folder
    except ModuleNotFoundError:
        # Last resort: add this file's directory to sys.path and retry
        import sys as _sys, os as _os
        _dir = _os.path.dirname(__file__)
        if _dir not in _sys.path:
            _sys.path.append(_dir)
        try:
            from outScript import streamToSpeakers
        except ModuleNotFoundError:
            from tester import streamToSpeakers

from dotenv import load_dotenv
import logging

import live
try:
    from live.exceptions import LiveConnectionError
except Exception:  # fallback if the package layout changes
    class LiveConnectionError(Exception):
        pass

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
import io
from typing import AsyncGenerator, AsyncIterable, Generator, Iterable, Union, Optional, List
import struct, binascii   # add at top of the file if not already there
import numpy as np

from scipy.signal import resample_poly

# Safe import for WebSocket close exceptions (PlayHT streaming)
try:
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
except Exception:
    class ConnectionClosedError(Exception):
        pass
    class ConnectionClosedOK(Exception):
        pass


from dataclasses import dataclass, field
from threading import RLock
import subprocess
import platform
from pathlib import Path

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

# Helper to lazily import Whisper (Torch) when first needed
def _load_whisper_module():
    global whisper
    if whisper is None:
        try:
            import whisper as _whisper
            whisper = _whisper
        except Exception as e:
            logging.error(f"Failed to import whisper: {e}")
            raise
    return whisper

# Ensure runtime directories exist
def ensure_directories():
    for d in ["RecordedAudio", "RecordedAudio/New", "RecordedAudio/Old", "RecordedAudio/PoemChunks"]:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception as e:
            logging.warning(f"Could not ensure directory {d}: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Admin control flags (for show director overrides) ---
# Set via HTTP endpoints to steer the live flow from the GUI
ADMIN_FORCE_VAD_END = threading.Event()       # end current VAD capture early
ADMIN_ABORT_TTS = threading.Event()           # abort any current TTS playback/stream
ADMIN_END_CONVERSATION = threading.Event()    # end the active conversation loop

# Live VAD recording status for GUI
RECORDING_ACTIVE = False
RECORDING_STARTED_TS = 0.0

def get_recording_status() -> dict:
    try:
        if RECORDING_ACTIVE:
            return {"active": True, "elapsed": max(0.0, time.time() - RECORDING_STARTED_TS)}
        return {"active": False, "elapsed": 0.0}
    except Exception:
        return {"active": False, "elapsed": 0.0}

def admin_clear_transient_flags():
    """Clear one-shot flags that should not persist across actions."""
    ADMIN_FORCE_VAD_END.clear()
    ADMIN_ABORT_TTS.clear()

def should_abort_playback() -> bool:
    """Return True if playback should be aborted due to admin override or OFF‑HOOK.

    Semantics across the app:
    - On‑hook (onHold() == True): handset is down; background cycles (ring/word/poem/music)
      should CONTINUE playing.
    - Off‑hook (onHold() == False): handset lifted; any background playback should abort
      so the live conversation can resume.
    """
    if ADMIN_ABORT_TTS.is_set() or ADMIN_END_CONVERSATION.is_set():
        return True
    try:
        a = get_arduino()
        # Abort only when OFF‑HOOK
        if a and (not a.onHold()):
            return True
    except Exception:
        pass
    return False
# --- resilience knobs ---
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1","true","yes","on"}

VOICE_DELETE_COOLDOWN_SEC = int(os.getenv("VOICE_DELETE_COOLDOWN_SEC", "120"))  # seconds
AUTO_FALLBACK_TO_ELEVENLABS = _env_bool("AUTO_FALLBACK_TO_ELEVENLABS", True)
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
    use_speaker_boost: bool = True
    stability: float = 0.5
    similarity_boost: float = 1.0
    style: float = 0.15
    speed: float = 1.0
    model_id_stream: str = "eleven_v3"
    format_stream: str = "pcm_44100"
    model_id_buffered: str = "eleven_v3"
    format_buffered: str = "pcm_44100"

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
    gui_overrides: bool = False  # default to off so GUI doesn't override script on launch
    audio_route: dict = field(default_factory=dict)  # debug: requested vs device rates/channels/frames

    def set_gui_overrides(self, enabled: bool):
        with self.lock:
            self.gui_overrides = bool(enabled)

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
                "gui_overrides": self.gui_overrides,
                "audio_route": dict(self.audio_route or {}),
            }
            try:
                data["clones"] = voice_clone_manager.get_clone_info()
            except Exception:
                data["clones"] = []
            try:
                data["cloning"] = get_cloning_snapshot()
            except Exception:
                data["cloning"] = {"error": "unavailable"}
            try:
                data["elevenlabs_keyring"] = get_eleven_keyring_status()
            except Exception:
                data["elevenlabs_keyring"] = {"active_alias": "—", "remaining_ivc": "—", "total_keys": 0}
            try:
                data["recording"] = get_recording_status()
            except Exception:
                data["recording"] = {"active": False, "elapsed": 0.0}
            return data

    def set_engine(self, e: str):
        with self.lock:
            if not self.gui_overrides:
                logging.info("GUI overrides disabled; ignoring set_engine(%s)", e)
                return
            self.engine = e.strip().lower()

    def update_playht(self, **kwargs):
        with self.lock:
            if not self.gui_overrides:
                logging.info("GUI overrides disabled; ignoring update_playht(%r)", kwargs)
                return
            for k, v in kwargs.items():
                if hasattr(self.playht, k): setattr(self.playht, k, v)

    def update_elevenlabs(self, **kwargs):
        with self.lock:
            if not self.gui_overrides:
                logging.info("GUI overrides disabled; ignoring update_elevenlabs(%r)", kwargs)
                return
            for k, v in kwargs.items():
                if hasattr(self.elevenlabs, k): setattr(self.elevenlabs, k, v)

    def update_vad(self, **kwargs):
        with self.lock:
            if not self.gui_overrides:
                logging.info("GUI overrides disabled; ignoring update_vad(%r)", kwargs)
                return
            for k, v in kwargs.items():
                if hasattr(self.vad, k): setattr(self.vad, k, v)

    def set_audio_output(self, out: str):
        with self.lock:
            if not self.gui_overrides:
                logging.info("GUI overrides disabled; ignoring set_audio_output(%s)", out)
                return
            self.audio_output = out

    def set_streaming(self, streaming: bool):
        with self.lock:
            if not self.gui_overrides:
                logging.info("GUI overrides disabled; ignoring set_streaming(%s)", streaming)
                return
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
    if not ui_state.gui_overrides:
        logging.info("GUI overrides disabled; skipping apply_settings_from_preset")
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
            if not ui_state.gui_overrides:
                logging.info("GUI overrides disabled; not applying starred preset '%s'", p.get('name'))
            else:
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

def get_eleven_keyring_status() -> dict:
    try:
        data = ElevenKeyring.load()
        ElevenKeyring.rollover_month_if_needed(data)
        keys = data.get("keys", [])
        active = ElevenKeyring.get_active_record(data)
        alias = active.get("alias") if active else (data.get("last_active_alias") if keys else None)
        remaining = None
        if active:
            try:
                limit = int(active.get("ivc_monthly_limit", 0) or 0)
                used = int(active.get("ivc_used_this_month", 0) or 0)
                remaining = max(0, limit - used)
            except Exception:
                pass
        return {
            "active_alias": alias or "—",
            "remaining_ivc": remaining if remaining is not None else "—",
            "total_keys": len(keys),
        }
    except Exception:
        return {"active_alias": "—", "remaining_ivc": "—", "total_keys": 0}

# Constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_PROJECT = os.getenv('OPENAI_PROJECT')

class ElevenKeyring:
    KEYRING_PATH = Path.home() / ".bigbird" / "eleven_keyring.json"

    @classmethod
    def _ensure(cls):
        try:
            cls.KEYRING_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not cls.KEYRING_PATH.exists():
                data = {"keys": [], "month_anchor": time.strftime("%Y-%m"), "last_active_alias": None}
                cls.KEYRING_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def load(cls) -> dict:
        cls._ensure()
        try:
            return json.loads(cls.KEYRING_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"keys": [], "month_anchor": time.strftime("%Y-%m"), "last_active_alias": None}

    @classmethod
    def save(cls, data: dict):
        cls._ensure()
        try:
            cls.KEYRING_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def rollover_month_if_needed(cls, data: dict):
        cur = time.strftime("%Y-%m")
        if data.get("month_anchor") != cur:
            for k in data.get("keys", []):
                try:
                    k["ivc_used_this_month"] = 0
                except Exception:
                    pass
            data["month_anchor"] = cur

    @classmethod
    def get_active_record(cls, data: dict) -> Optional[dict]:
        for k in data.get("keys", []):
            if k.get("active"):
                return k
        return None

    @classmethod
    def set_active(cls, alias: str):
        data = cls.load()
        found = False
        now = time.time()
        for k in data.get("keys", []):
            if k.get("alias") == alias:
                k["active"] = True
                k["last_used_ts"] = now
                found = True
            else:
                k["active"] = False
        if found:
            data["last_active_alias"] = alias
            cls.save(data)

class ElevenClientManager:
    def __init__(self):
        self.client: Optional[ElevenLabs] = None
        self._last_key: Optional[str] = None
        self.refresh_client()

    def _select_api_key(self) -> Optional[str]:
        # Prefer active key from keyring; fallback to env var
        data = ElevenKeyring.load()
        ElevenKeyring.rollover_month_if_needed(data)
        active = ElevenKeyring.get_active_record(data)
        if active and active.get("api_key"):
            return active.get("api_key")
        return os.getenv("ELEVENLABS_API_KEY")

    def refresh_client(self):
        key = self._select_api_key()
        try:
            self.client = ElevenLabs(api_key=key)
            self._last_key = key
        except Exception as e:
            logging.error(f"Failed to init ElevenLabs client: {e}")
            self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
            self._last_key = os.getenv("ELEVENLABS_API_KEY")

    def current_api_key(self) -> Optional[str]:
        return self._select_api_key()

    def ensure_fresh(self):
        try:
            cur = self._select_api_key()
            if cur != self._last_key:
                self.refresh_client()
        except Exception:
            pass

    def _remaining_for(self, rec: dict) -> int:
        try:
            limit = int(rec.get("ivc_monthly_limit", 0) or 0)
            used = int(rec.get("ivc_used_this_month", 0) or 0)
            return max(0, limit - used)
        except Exception:
            return 0

    def ivc_consumed(self, n: int = 1, rotate_threshold: int = 1):
        """Increment active key's used count; rotate to another key if remaining <= threshold."""
        data = ElevenKeyring.load()
        ElevenKeyring.rollover_month_if_needed(data)
        active = ElevenKeyring.get_active_record(data)
        if not active:
            ElevenKeyring.save(data)
            return
        try:
            active["ivc_used_this_month"] = int(active.get("ivc_used_this_month", 0) or 0) + int(n)
        except Exception:
            active["ivc_used_this_month"] = int(n)
        # Check remaining and rotate if needed
        remaining = self._remaining_for(active)
        if remaining <= rotate_threshold:
            # pick another key with the most remaining
            candidates = [k for k in data.get("keys", []) if k is not active]
            candidates.sort(key=lambda r: self._remaining_for(r), reverse=True)
            for rec in candidates:
                if self._remaining_for(rec) > rotate_threshold and rec.get("api_key"):
                    # switch to this alias
                    alias = rec.get("alias")
                    try:
                        for k in data.get("keys", []):
                            k["active"] = (k.get("alias") == alias)
                            if k["active"]:
                                k["last_used_ts"] = time.time()
                        data["last_active_alias"] = alias
                        ElevenKeyring.save(data)
                        self.refresh_client()
                        try:
                            # Immediately refresh clone list for new key so selection uses fresh data
                            voice_clone_manager.sync_with_api()
                        except Exception:
                            pass
                        logging.info(f"Switched ElevenLabs key to alias '{alias}' due to low remaining IVC")
                        # Optionally seed a carryover IVC clone using the last consolidated audio
                        try:
                            if _env_bool("CARRYOVER_IVC_ON_ROTATE", True):
                                wav = _find_latest_consolidated_wav()
                                if wav:
                                    ts = time.strftime('%Y%m%d-%H%M%S')
                                    def _do():
                                        try:
                                            ElevenLabsEngine().clone_from_file(wav, f"carryover_{ts}")
                                        except Exception as _e:
                                            logging.warning(f"Carryover IVC failed: {_e}")
                                    threading.Thread(target=_do, daemon=True).start()
                        except Exception:
                            pass
                    except Exception as e:
                        logging.error(f"Failed to rotate ElevenLabs key: {e}")
                    break
        ElevenKeyring.save(data)

eleven_client_manager = ElevenClientManager()

# Instantiate client explicitly with project to avoid defaulting to a project with no credit
client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT) if OPENAI_PROJECT else OpenAI(api_key=OPENAI_API_KEY)

# Sanity-check helper to print which OpenAI credentials are in use
def debug_print_openai_creds():
    key = os.getenv('OPENAI_API_KEY') or ''
    proj = os.getenv('OPENAI_PROJECT') or ''
    logging.info(f"OPENAI_API_KEY set: {'yes' if key else 'no'} (prefix: {key[:7]+'…' if key else '—'})")
    logging.info(f"OPENAI_PROJECT set: {'yes' if proj else 'no'} (value: {proj if proj else '—'})")

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
PLAYHT_API_KEYS = [os.getenv('PLAYHT_API_KEY_SPARE')]
PLAYHT_USER_IDS = [os.getenv('PLAYHT_USER_ID_SPARE')]

#PLAYHT_API_KEYS #= [os.getenv('PLAYHT_API_KEY')]
#PLAYHT_USER_IDS = [os.getenv('PLAYHT_USER_ID')] 

# Engine selector (scaffolding for runtime swap)
TTS_ENGINE = (os.getenv('TTS_ENGINE') or 'playht').strip().lower()

##VOICE_CLONE_MODEL = "PlayHT2.0-turbo"      # 48 kHz / 16‑bit mono
LLM_MODEL = "gpt-4o-mini-2024-07-18"


# You can override via env: PLAYHT_PROTOCOL=(grpc|ws|http) and PLAYHT_VOICE_ENGINE=(PlayDialog|PlayHT2.0-turbo)
PROTOCOL = os.getenv('PLAYHT_PROTOCOL', 'grpc')
VOICE_CLONE_MODEL = os.getenv('PLAYHT_VOICE_ENGINE', 'PlayDialog')  # default to PlayDialog per new streaming docs

# --- PlayHT protocol resolver ------------------------------------
# Some PlayHT engines (e.g., PlayDialog) only support WebSocket.
# The pyht SDK expects protocol to be one of: 'http', 'ws', or 'grpc'.
# This helper normalizes inputs and prevents invalid combos.

def _normalize_protocol(proto: str) -> str:
    p = (proto or '').strip().lower()
    if p in ('websocket', 'ws', 'wss'):  # normalize to 'ws'
        return 'ws'
    if p in ('http', 'https'):            # normalize to 'http'
        return 'http'
    if p == 'grpc':
        return 'grpc'
    # safe default
    return 'ws'


def _resolve_playht_protocol(voice_engine: str, proto: str) -> str:
    try:
        ve = (voice_engine or '').strip().lower()
        base = _normalize_protocol(proto)
        if ve == 'playdialog' and base != 'ws':
            logging.warning("Voice engine PlayDialog requires 'ws' protocol; overriding %r → 'ws'.", base)
            return 'ws'
        return base
    except Exception:
        # Safe fallback
        return 'ws'

AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 2048

url = "https://api.play.ht/api/v2/cloned-voices/instant"

OldSound = "RecordedAudio/Old"
New = "RecordedAudio/New"

SoundTrack = "/Users/x/myenv/bin/tester.wav"

# Debug/temporary routing: when True, anything targeted at the
# BlackHole device will be written to a WAV file under RecordedAudio/
# instead of being sent to the BlackHole output. This helps validate
# chunking/sample‑rate without the virtual device in the loop.
BLACKHOLE_WRITE_TO_FILE = str(os.getenv("BLACKHOLE_WRITE_TO_FILE", "0")).strip().lower() in {"1","true","yes","on"}


arduinoLocation = '/dev/tty.usbmodem744DBDA236D82'

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
    hold = None
    raw = None
    if a:
        try:
            hold = bool(a.onHold())
            raw = getattr(a, "_last_hook_raw", None)
        except Exception:
            pass
    return {
        "override": dict(_arduino_override),
        "port": arduinoLocation,
        "connected": bool(a and getattr(a, "serial", None)),
        "hold": hold,  # True=on-hook, False=off-hook, None=unknown
        "hook_state": ("onhook" if hold else ("offhook" if hold is False else None)),
        "raw": raw,
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

# Global device-output guard to prevent multiple PyAudio streams to the same
# virtual device (e.g., BlackHole) from fighting and causing dropouts.
class _DeviceStreamRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._active: list[callable] = []  # list of stop callbacks

    def stop_all(self):
        with self._lock:
            stops = list(self._active)
            self._active.clear()
        for fn in stops:
            try:
                fn()
            except Exception:
                pass

    def register(self, stop_fn):
        with self._lock:
            self._active.append(stop_fn)

_DEVICE_STREAMS = _DeviceStreamRegistry()

# --------------------------------------------------------------
# ElevenLabs HTTP stream → BlackHole (robust to MP3 fallback)
# --------------------------------------------------------------
API_URL_TTS_STREAM = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

def _find_output_device_index(pa: pyaudio.PyAudio, name_substr: str = "BlackHole") -> tuple[int, dict]:
    idx = None
    info_m = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if name_substr.lower() in str(info.get('name','')).lower():
            idx = i
            info_m = info
            break
    if idx is None:
        raise RuntimeError(f"Output device containing '{name_substr}' not found")
    return idx, info_m

def _upmix_mono16_to_stereo(pcm: bytes) -> bytes:
    if not pcm:
        return pcm
    # fast interleave mono→stereo without numpy allocations
    out = bytearray(len(pcm) * 2)
    mv = memoryview(out)
    mv[0::4] = pcm[0::2]
    mv[1::4] = pcm[1::2]
    mv[2::4] = pcm[0::2]
    mv[3::4] = pcm[1::2]
    return bytes(out)

def _strip_wav_header_if_present(buf: bytearray) -> tuple[bytes, bool]:
    """If `buf` holds a full RIFF/WAVE header, strip it and return payload + True.
    If no header (raw PCM), return bytes(buf) + True. If incomplete header, return b'' + False.
    """
    if len(buf) < 32:
        return b"", False
    if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return bytes(buf), True
    i = buf.find(b"data", 12)
    if i == -1 or i + 8 > len(buf):
        return b"", False
    payload = bytes(buf[i+8:])
    del buf[:i+8]
    return payload, True

def stream_eleven_http_to_blackhole(voice_id: str, text: str,
                                    *,
                                    model_id: str = "eleven_v3",
                                    device_hint: str = "BlackHole",
                                    device_rate: int = 44100,
                                    frames_per_buffer: int = 1024,
                                    jitter_ms: int = 150,
                                    requested_of: str = None) -> None:
    """
    Stream ElevenLabs TTS over raw HTTP to BlackHole, handling cases where the
    provider returns MP3 despite PCM being requested. MP3 is decoded to PCM
    (requires pydub + ffmpeg). PCM path strips an initial WAV header if present
    and maintains a small jitter buffer.
    """
    api_key = eleven_client_manager.current_api_key() or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("No ElevenLabs API key available")

    url = API_URL_TTS_STREAM.format(voice_id=voice_id)
    of = requested_of or f"pcm_{int(device_rate)}"
    url = f"{url}?output_format={of}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/wav",
        "content-type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "output_format": of,
    }

    # Ensure device is free
    try:
        _DEVICE_STREAMS.stop_all()
    except Exception:
        pass

    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    if resp.status_code != 200:
        ct = resp.headers.get("Content-Type", "?")
        body = (resp.text or "")[:400]
        raise RuntimeError(f"HTTP {resp.status_code} from ElevenLabs (CT={ct}): {body}")

    ct = (resp.headers.get("Content-Type", "") or "").lower()
    is_mp3 = ("mpeg" in ct) or ("mp3" in ct)

    p = pyaudio.PyAudio()
    try:
        dev_idx, dev_info = _find_output_device_index(p, name_substr=device_hint)
        chans = 2 if int(dev_info.get("maxOutputChannels") or 1) >= 2 else 1
        # Open at device_rate (Ableton is set accordingly)
        stream = p.open(format=pyaudio.paInt16,
                        channels=chans,
                        rate=int(device_rate),
                        output=True,
                        output_device_index=dev_idx,
                        frames_per_buffer=int(frames_per_buffer))

        # Register a stop callback so other paths can preempt
        try:
            _DEVICE_STREAMS.register(lambda: (stream.stop_stream(), stream.close()))
        except Exception:
            pass

        if is_mp3:
            # Accumulate then decode MP3 to PCM (blocking playback)
            mp3_buf = bytearray()
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    mp3_buf.extend(chunk)
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(io.BytesIO(bytes(mp3_buf)), format="mp3")
                seg = seg.set_frame_rate(int(device_rate)).set_channels(int(chans)).set_sample_width(2)
                pcm = seg.raw_data
            except Exception as e:
                raise RuntimeError(f"Received MP3 stream and failed to decode: {e}")
            step = int(frames_per_buffer) * chans * 2
            for i in range(0, len(pcm), step):
                if should_abort_playback():
                    break
                stream.write(pcm[i:i+step], exception_on_underflow=False)
            try:
                stream.stop_stream(); stream.close()
            except Exception:
                pass
            return

        # PCM path with WAV header strip + jitter buffer
        header_parsed = False
        tail = b""
        jb = bytearray()
        jitter_bytes = int(device_rate * (2 if chans==2 else 1) * 2 * (int(jitter_ms)/1000.0))

        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if should_abort_playback():
                break
            if not header_parsed:
                jb.extend(chunk)
                data, header_parsed = _strip_wav_header_if_present(jb)
                if not header_parsed:
                    continue
                if data:
                    jb = bytearray(data)
                if len(jb) < jitter_bytes:
                    continue
            else:
                jb.extend(chunk)

            buf = tail + jb
            if len(buf) & 1:
                tail = buf[-1:]
                buf = buf[:-1]
            else:
                tail = b""
            jb.clear()
            if not buf:
                continue
            out = _upmix_mono16_to_stereo(buf) if chans == 2 else buf
            stream.write(out, exception_on_underflow=False)

        try:
            stream.stop_stream(); stream.close()
        except Exception:
            pass
    finally:
        try:
            p.terminate()
        except Exception:
            pass


class ElevenHTTPStreamPlayer:
    """
    Robust ElevenLabs → HTTP stream → BlackHole player.
    Wraps `stream_eleven_http_to_blackhole` and resolves sensible defaults
    from ui_state and your key/clone managers.
    """
    def __init__(self,
                 device_hint: str = 'BlackHole',
                 device_rate: int = 44100,
                 frames_per_buffer: int = 1024,
                 jitter_ms: int = 200,
                 output_format: str | None = None):
        self.device_hint = device_hint
        self.device_rate = int(device_rate)
        self.frames_per_buffer = int(frames_per_buffer)
        self.jitter_ms = int(jitter_ms)
        self.output_format = output_format  # e.g. 'pcm_44100' or 'pcm_24000'

    def _resolve_voice_id(self) -> str | None:
        try:
            vid = voice_clone_manager.get_recent_clone_id(engine='elevenlabs')
            if vid:
                return vid
        except Exception:
            pass
        return os.getenv('ELEVENLABS_VOICE_ID')

    def _resolve_model_id(self) -> str:
        try:
            return ui_state.elevenlabs.model_id_stream or 'eleven_v3'
        except Exception:
            return 'eleven_v3'

    def play_text(self, text: str,
                  *, voice_id: str | None = None,
                  model_id: str | None = None,
                  output_format: str | None = None) -> None:
        if not text:
            return
        vid = voice_id or self._resolve_voice_id()
        if not vid:
            logging.error("ElevenHTTPStreamPlayer: no ElevenLabs voice_id available")
            return
        mid = model_id or self._resolve_model_id()
        of  = output_format or self.output_format or f"pcm_{self.device_rate}"
        try:
            stream_eleven_http_to_blackhole(
                voice_id=vid,
                text=text,
                model_id=mid,
                device_hint=self.device_hint,
                device_rate=self.device_rate,
                frames_per_buffer=self.frames_per_buffer,
                jitter_ms=self.jitter_ms,
                requested_of=of,
            )
        except Exception as e:
            logging.warning(f"ElevenHTTPStreamPlayer fallback (HTTP stream failed): {e}")
            # Buffered fallback: download full PCM and play via ElevenFakeStreamer
            try:
                ElevenFakeStreamer(device_hint=self.device_hint, rate=self.device_rate).start([text]).wait()
            except Exception as e2:
                logging.error(f"ElevenHTTPStreamPlayer buffered fallback failed: {e2}")


def speak_eleven_http_blackhole(text: str,
                                *,
                                device_rate: int = 44100,
                                jitter_ms: int = 200,
                                frames_per_buffer: int = 1024,
                                output_format: str | None = None) -> None:
    """Convenience wrapper used by background cycles."""
    try:
        ElevenHTTPStreamPlayer(device_rate=device_rate,
                               jitter_ms=jitter_ms,
                               frames_per_buffer=frames_per_buffer,
                               output_format=output_format).play_text(text)
    except Exception as e:
        logging.error(f"speak_eleven_http_blackhole failed: {e}")
        
# --------------------------------------------------------------
# Fake realtime streamer: buffered file-by-file playback
# --------------------------------------------------------------
class ElevenFakeStreamer:
    def __init__(self, device_hint: str = 'BlackHole', rate: int = 44100, buffer_files: int = 2,
                 out_dir: str = 'RecordedAudio/PoemChunks'):
        self.device_hint = device_hint
        self.rate = int(rate)
        self.buffer_files = max(0, int(buffer_files))
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        # state
        self._ready = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._producer = None
        self._consumer = None

    def stop(self):
        self._stop.set()

    def _write_chunk_wav(self, pcm: bytes, idx: int) -> str:
        ts = time.strftime('%Y%m%d-%H%M%S')
        tmp_path = os.path.join(self.out_dir, f"chunk_{ts}_{idx:03d}.part")
        final_path = tmp_path.replace('.part', '.wav')
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.rate)
            wf.writeframes(pcm if not (len(pcm) & 1) else pcm[:-1])
        os.replace(tmp_path, final_path)
        return final_path

    def _produce(self, lines: list[str]):
        try:
            eleven_client_manager.ensure_fresh()
            for i, text in enumerate(lines):
                if self._stop.is_set():
                    break
                # Fetch full PCM for the line
                audio = eleven_client_manager.client.text_to_speech.convert(
                    text=text,
                    voice_id=(voice_clone_manager.get_recent_clone_id(engine='elevenlabs') or os.getenv('ELEVENLABS_VOICE_ID')),
                    model_id=(ui_state.elevenlabs.model_id_buffered or 'eleven_v3'),
                    output_format=f"pcm_{self.rate}",
                )
                # Normalize to bytes
                if isinstance(audio, (bytes, bytearray)):
                    pcm = bytes(audio)
                else:
                    bb = bytearray()
                    for ch in audio:
                        if ch:
                            bb.extend(ch)
                    pcm = bytes(bb)
                path = self._write_chunk_wav(pcm, i)
                with self._lock:
                    self._ready.append(path)
        except Exception as e:
            logging.warning(f"FakeStreamer produce error: {e}")
        finally:
            # signal end by appending None sentinel
            with self._lock:
                self._ready.append(None)

    def _open_device(self):
        pa = pyaudio.PyAudio()
        idx = None
        info_m = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if self.device_hint.lower() in str(info.get('name','')).lower():
                idx = i
                info_m = info
                break
        if idx is None:
            pa.terminate()
            raise RuntimeError(f"Output device containing '{self.device_hint}' not found")
        channels = 2 if int(info_m.get('maxOutputChannels') or 1) >= 2 else 1
        frames = int(os.getenv('PYAUDIO_FRAMES', str(CHUNK)))
        stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=self.rate, output=True,
                         output_device_index=idx, frames_per_buffer=frames)
        return pa, stream, channels

    def _play_file(self, stream: pyaudio.Stream, channels: int, path: str):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            # If file rate differs, simple on-the-fly fallback (PortAudio may resample)
            # We keep it simple assuming rate matches.
            while True:
                if should_abort_playback() or self._stop.is_set():
                    return
                data = wf.readframes(CHUNK)
                if not data:
                    break
                if channels == 2 and wf.getnchannels() == 1:
                    out = bytearray(len(data) * 2)
                    mv = memoryview(out)
                    mv[0::4] = data[0::2]
                    mv[1::4] = data[1::2]
                    mv[2::4] = data[0::2]
                    mv[3::4] = data[1::2]
                    data = bytes(out)
                stream.write(data)

    def _consume(self):
        # ensure no other stream holds the device
        try:
            _DEVICE_STREAMS.stop_all()
        except Exception:
            pass
        pa, stream, channels = self._open_device()
        try:
            # Optional initial buffer gap
            while True:
                with self._lock:
                    ready_count = sum(1 for p in self._ready if p is not None)
                if ready_count >= max(0, self.buffer_files):
                    break
                if self._stop.is_set():
                    return
                time.sleep(0.02)
            # Playback loop
            consumed = 0
            while True:
                if self._stop.is_set():
                    break
                next_path = None
                with self._lock:
                    if consumed < len(self._ready):
                        next_path = self._ready[consumed]
                        consumed += 1
                if next_path is None:
                    # Either waiting for more, or reached sentinel
                    with self._lock:
                        done = (len(self._ready) > 0 and self._ready[-1] is None and consumed >= len(self._ready))
                    if done:
                        break
                    time.sleep(0.02)
                    continue
                # Sentinel means producer is finished
                if next_path is None:
                    break
                # Skip sentinel markers explicitly
                if isinstance(next_path, str):
                    self._play_file(stream, channels, next_path)
        finally:
            try:
                stream.stop_stream(); stream.close()
            except Exception:
                pass
            pa.terminate()

    def start(self, lines: list[str]):
        self._producer = threading.Thread(target=self._produce, args=(list(lines),), daemon=True)
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._producer.start()
        self._consumer.start()
        return self

    def wait(self):
        if self._producer:
            self._producer.join()
        if self._consumer:
            self._consumer.join()

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
        # Clips accumulated since the last clone threshold trigger
        self.clips_since_last_clone = []
        self.total_duration_since_last_clone = 0.0  # seconds since last clone
        # All clips in the current conversation/session (growing until explicit reset)
        self.all_clips = []
        self.total_conversation_duration = 0.0  # total seconds for whole conversation

    def add_clip(self, file_path):
        with contextlib.closing(wave.open(file_path, 'rb')) as wf:
            duration = wf.getnframes() / wf.getframerate()
        # Append to per-clone batch and to full conversation
        self.clips_since_last_clone.append(file_path)
        self.total_duration_since_last_clone += duration
        self.all_clips.append(file_path)
        self.total_conversation_duration += duration
        # Mirror into global progress/pending for GUI
        try:
            clone_progress.note_collected(duration)
            clone_progress.add_pending_clip(file_path)
            clone_progress.note_conversation(duration)
        except Exception:
            pass

    def should_send_for_cloning(self):
        """Check if the accumulated clips should be sent for voice cloning."""
        return self.total_duration_since_last_clone >= 7.5

    def concatenate_clips(self):
        """Concatenate ALL clips in the current conversation into a single WAV and return the path.
        Does NOT clear the conversation. It only resets the per-clone accumulation so the next
        threshold counts fresh seconds, while the consolidated WAV keeps growing.
        """
        if not self.all_clips:
            logging.info("concatenate_clips: no clips to concatenate")
            return None

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"RecordedAudio/concatenated_audio_{timestamp}.wav"

        try:
            with wave.open(output_path, 'wb') as wf:
                # Initialize parameters from the first clip
                with wave.open(self.all_clips[0], 'rb') as cf:
                    wf.setnchannels(cf.getnchannels())
                    wf.setsampwidth(cf.getsampwidth())
                    wf.setframerate(cf.getframerate())

                for clip in self.all_clips:
                    try:
                        with wave.open(clip, 'rb') as cf:
                            frames = cf.readframes(cf.getnframes())
                            wf.writeframes(frames)
                    except Exception as e:
                        logging.warning(f"Skipping bad clip {clip}: {e}")
        except Exception as e:
            logging.error(f"concatenate_clips failed: {e}")
            return None
        finally:
            # Reset only the per-clone counters so the next trigger is based on new speech
            self.total_duration_since_last_clone = 0.0
            self.clips_since_last_clone.clear()
            try:
                clone_progress.reset_collected()  # reset GUI progress (not whole session)
            except Exception:
                pass

        return output_path

    def reset_conversation(self):
        """Clear the entire conversation accumulation. Call this at end-of-call or on program shutdown."""
        self.clips_since_last_clone.clear()
        self.all_clips.clear()
        self.total_duration_since_last_clone = 0.0
        self.total_conversation_duration = 0.0
        try:
            clone_progress.reset_session()
        except Exception:
            pass
    
# ---------------- Clone progress tracking (for GUI) ----------------
class CloneProgressTracker:
    def __init__(self, required_seconds: float = 7.5):
        self.required_seconds = float(required_seconds)
        self.collected_seconds = 0.0
        self.clones_in_session = 0
        self.session_started_ts = time.time()
        self.lock = threading.Lock()
        self.pending_clip_paths: list[str] = []
        self.conversation_seconds = 0.0

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

    def note_conversation(self, seconds: float):
        with self.lock:
            try:
                self.conversation_seconds += max(0.0, float(seconds))
            except Exception:
                pass

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
            self.conversation_seconds = 0.0

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
                "conversation_seconds": self.conversation_seconds,
            }

# Global tracker instance
clone_progress = CloneProgressTracker(required_seconds=7.5)


def read_saved_info(limit_chars: int = 1000) -> str:
    """Return last up-to-`limit_chars` of savedInfo.txt if present.

    Uses a stable path anchored at the project root so GUI/API processes
    see the same file regardless of current working directory.
    """
    try:
        base_dir = Path(__file__).resolve().parent.parent  # repo root
    except Exception:
        base_dir = Path('.')
    path = base_dir / "savedInfo.txt"
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
        global RECORDING_ACTIVE, RECORDING_STARTED_TS
        os.makedirs("RecordedAudio", exist_ok=True)
        # Visibility: log input capture parameters for debugging
        try:
            logging.info("VAD recorder: input_rate=%d Hz, channels=%d, frame_samples=%d", self.rate, self.channels, self.frames_per_buffer)
        except Exception:
            pass
        stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.frames_per_buffer)
        audio_frames = []
        silent_frames = collections.deque(maxlen=int(self.rate / self.frames_per_buffer * max_silence_length))
        initial_timeout_frames = int(self.rate / self.frames_per_buffer * initial_silence_timeout)

        try:
            # Wait for initial speech up to timeout
            for _ in range(initial_timeout_frames):
                if ADMIN_FORCE_VAD_END.is_set() or ADMIN_END_CONVERSATION.is_set():
                    # director cut: stop waiting and return no audio
                    ADMIN_FORCE_VAD_END.clear()
                    return None
                try:
                    frame = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                except Exception as e:
                    logging.warning(f"Audio read error (initial): {e}")
                    continue
                if self.is_speech(frame):
                    audio_frames.append(frame)
                    # Mark start of active recording for GUI
                    RECORDING_ACTIVE = True
                    RECORDING_STARTED_TS = time.time()
                    break
                else:
                    silent_frames.append(frame)

            if not audio_frames:  # No speech detected
                return None

            # Continue recording until trailing silence reached
            while True:
                if ADMIN_FORCE_VAD_END.is_set() or ADMIN_END_CONVERSATION.is_set():
                    # director cut: end recording immediately
                    ADMIN_FORCE_VAD_END.clear()
                    break
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
            # Clear live-recording flag
            RECORDING_ACTIVE = False

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
        # Track last raw hook response and parsed state for debugging
        self._last_hook_raw: Optional[str] = None
        self._last_hold: Optional[bool] = None

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
                self._last_hold = True
                return True
            if hook_ovr == "offhook":
                self._last_hold = False
                return False
        except Exception:
            pass

        # 2) No serial? fall back to safe default (off-hook)
        if not getattr(self, "serial", None):
            logging.debug("onHold: no Arduino serial; defaulting to OFF-HOOK (False) – use GUI override to force ON-HOOK")
            # keep last known if we have it, else False
            return bool(self._last_hold) if self._last_hold is not None else False

        # 3) Query device
        resp = self.send_command("ISBUTTONPRESSED") or ""
        self._last_hook_raw = str(resp)
        r = str(resp).strip().upper()
        # Map common firmware responses
        if r in ("DOWN", "PRESSED", "LOW", "0", "ON", "ONHOOK", "HOOK:ON", "PRESSED:1", "BTN:PRESSED"):
            # switch pressed -> handset on the cradle (ON-HOOK)
            self._last_hold = True
            return True
        if r in ("UP", "RELEASED", "HIGH", "1", "OFF", "OFFHOOK", "HOOK:OFF", "PRESSED:0", "BTN:RELEASED"):
            # switch released -> handset lifted (OFF-HOOK)
            self._last_hold = False
            return False
        logging.debug(f"onHold: unexpected ISBUTTONPRESSED -> {resp!r}; using last known or OFF-HOOK")
        # If unknown response, prefer last known state to avoid flapping
        return bool(self._last_hold) if self._last_hold is not None else False
        
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

    def _wait_until_offhook(self, duration: float, clip_to_stop=None) -> bool:
        """
        Wait up to `duration` seconds for OFF-HOOK (handset lifted).
        If OFF-HOOK is detected, stop `clip_to_stop` (if given) and return True.
        Return False if timeout elapses with handset still ON-HOOK.
        """
        end_time = time.time() + duration
        while time.time() < end_time:
            try:
                if not self.onHold():  # OFF-HOOK
                    if clip_to_stop:
                        try:
                            clip_to_stop.stop()
                        except Exception:
                            pass
                    logging.info("OFF-HOOK detected – stopping hold cycle")
                    return True
            except Exception:
                pass
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
        logging.info(responce)   

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
        try:
            set_arduino_led_override("RECORDING")
        except Exception:
            pass

    def led_speaking(self):
        """led red blink"""
        command = "SETSTATE REPLYING"
        self.send_command(command)
        try:
            set_arduino_led_override("REPLYING")
        except Exception:
            pass

    def led_cloning(self):
        """led red blink"""
        command = "SETSTATE PROCESSING"
        self.send_command(command)
        try:
            set_arduino_led_override("PROCESSING")
        except Exception:
            pass

    def led_set_off(self):
        """led red blink"""
        command = "SETSTATE OFF"
        self.send_command(command)
        try:
            set_arduino_led_override("OFF")
        except Exception:
            pass

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


        tts_iter = self.client.tts(text=text,
                                   options=options,
                                   voice_engine=VOICE_CLONE_MODEL,
                                   protocol=_resolve_playht_protocol(VOICE_CLONE_MODEL, PROTOCOL))
        try:
            for chunk in tts_iter:
                header_buffer.extend(chunk)

                # Abort if handset is ON-HOOK or director override requests it
                if should_abort_playback():
                    logging.info("Abort requested – stopping live stream")
                    break

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
                            self.stream.write(bytes(pcm_chunk), exception_on_underflow=False)
                            # Abort if handset is on-hook or override
                            if should_abort_playback():
                                logging.info("Abort requested – stopping live stream")
                                break
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
                            # Abort if handset is on-hook or override
                            if should_abort_playback():
                                logging.info("Abort requested – stopping live stream")
                                break
                            self.stream.write(bytes(chunk), exception_on_underflow=False)
            # end for
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            logging.warning(f"PlayHT WebSocket closed during stream_audio: {e}")
        except Exception as e:
            logging.warning(f"PlayHT stream_audio error: {e}")
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
        except Exception:
            pass
        # Clear transient abort so next playback can proceed
        ADMIN_ABORT_TTS.clear()

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

        try:
            for chunk in out_stream:
                if should_abort_playback():
                    logging.info("Abort requested – stopping PlayDialog stream")
                    break
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
                        self._setup_audio_stream(sample_rate=ui_state.playht.sample_rate or RATE, channels=1)
                        self.stream.write(bytes(header_buffer))
                        continue

                # Already parsed header → stream every chunk
                # Abort if handset is ON-HOOK or director override
                if should_abort_playback():
                    logging.info("Abort requested – stopping PlayDialog stream")
                    break
                self.stream.write(bytes(chunk))
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            logging.warning(f"PlayHT WebSocket closed during _play_stream: {e}")
        except Exception as e:
            logging.warning(f"PlayHT _play_stream error: {e}")

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
            protocol=_resolve_playht_protocol(voice_engine, PROTOCOL),
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
        # Clear transient abort so next playback can proceed
        ADMIN_ABORT_TTS.clear()

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
                                     protocol=_resolve_playht_protocol(VOICE_CLONE_MODEL, PROTOCOL)):
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
            if should_abort_playback():
                logging.info("Abort requested – stopping buffered playback")
                break
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
        # Clear transient abort so next playback can proceed
        ADMIN_ABORT_TTS.clear()

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

        # When debugging, write stream to a WAV file instead of routing to BlackHole
        write_to_file = bool(BLACKHOLE_WRITE_TO_FILE)
        wav_out = None
        out_path = None


        for chunk in self.client.tts(text=text,
                                     options=options,
                                     voice_engine=VOICE_CLONE_MODEL,
                                     protocol=_resolve_playht_protocol(VOICE_CLONE_MODEL, PROTOCOL)):

            header_buffer.extend(chunk)
            if not header_parsed and len(header_buffer) >= 44:
                if header_buffer[:4] != b'RIFF' or header_buffer[8:12] != b'WAVE':
                    logging.error("Not a WAV stream – aborting BlackHole playback.")
                    # Close any file writer if opened
                    if wav_out:
                        try:
                            wav_out.close()
                        except Exception:
                            pass
                    return

                channels     = int.from_bytes(header_buffer[22:24], "little")
                sample_rate  = int.from_bytes(header_buffer[24:28], "little")

                data_pos = header_buffer.find(b'data')
                if data_pos != -1 and len(header_buffer) >= data_pos + 8:
                    pcm_start   = data_pos + 8
                    pcm_chunk   = header_buffer[pcm_start:]
                    header_parsed = True

                    # Open target: either WAV file (debug) or BlackHole device
                    if write_to_file:
                        try:
                            ts = time.strftime("%Y%m%d-%H%M%S")
                            out_path = f"RecordedAudio/blackhole_debug_playht_{ts}.wav"
                            wav_out = wave.open(out_path, 'wb')
                            wav_out.setnchannels(channels)
                            wav_out.setsampwidth(2)  # 16-bit PCM
                            wav_out.setframerate(sample_rate)
                        except Exception as e:
                            logging.error(f"Failed to open debug WAV for writing: {e}")
                            wav_out = None
                            write_to_file = False
                    else:
                        # open BlackHole at the correct format
                        self.setup_audio_stream(sample_rate, channels)

                    # Handle dangling byte alignment
                    pcm_chunk = self._tail_byte + pcm_chunk
                    self._tail_byte = b""
                    if len(pcm_chunk) & 1:
                        self._tail_byte = pcm_chunk[-1:]
                        pcm_chunk = pcm_chunk[:-1]

                    if pcm_chunk:
                        if should_abort_playback():
                            logging.info("Abort requested – stopping BlackHole stream")
                            if wav_out:
                                try:
                                    wav_out.close()
                                except Exception:
                                    pass
                            return
                        if write_to_file and wav_out:
                            try:
                                wav_out.writeframes(pcm_chunk)
                            except Exception as e:
                                logging.error(f"Failed to write debug WAV frames: {e}")
                        else:
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
                        if should_abort_playback():
                            logging.info("Abort requested – stopping BlackHole stream")
                            if wav_out:
                                try:
                                    wav_out.close()
                                except Exception:
                                    pass
                            return
                        if write_to_file and wav_out:
                            try:
                                wav_out.writeframes(chunk)
                            except Exception as e:
                                logging.error(f"Failed to write debug WAV frames: {e}")
                        else:
                            self.stream.write(bytes(chunk))

        # End of streaming loop – close WAV if used
        if write_to_file and wav_out:
            try:
                wav_out.close()
                logging.info(f"Wrote BlackHole debug WAV to {out_path}")
            except Exception:
                pass

    def cleanup(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.p is not None:
            self.p.terminate()
        # File writer, if used during stream, is closed inline after streaming

# ------------------------------------------------------------------
# ElevenLabs → Device (BlackHole/VB-CABLE) callback streamer
# ------------------------------------------------------------------
class ElevenToDeviceStreamer:
    """
    Pulls ElevenLabs streaming PCM on a producer thread into a byte ring buffer,
    while a PortAudio callback writes fixed-size frames to a specific output device
    (e.g., "BlackHole"), avoiding default-device routing and smoothing network jitter.
    """

    def __init__(self, client: 'ElevenLabs', device_name_substr: str = 'BlackHole',
                 rate: int | None = None, channels: int | None = None,
                 frames_per_buffer: int = 1024, prebuffer_ms: int | None = None):
        self.client = client
        self.dev_sub = device_name_substr
        self.request_rate = rate
        self.request_channels = channels
        self.frames = frames_per_buffer
        # Prebuffer window (ms) before starting device stream
        if prebuffer_ms is None:
            try:
                self.prebuffer_ms = int(os.getenv('E2D_PREBUFFER_MS', '3000'))
            except Exception:
                self.prebuffer_ms = 3000
        else:
            self.prebuffer_ms = max(0, int(prebuffer_ms))

        self.pa = None
        self.stream = None
        self.dev_index = None
        self.dev_rate = None
        self.dev_channels = None

        self._buf = bytearray()
        self._lock = threading.Lock()
        self._producer = None
        self._done = threading.Event()
        self._underruns = 0

    # --- helpers ---
    def _probe_device(self):
        pa = pyaudio.PyAudio()
        self.pa = pa
        idx = None
        info_m = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if self.dev_sub.lower() in str(info.get('name','')).lower():
                idx = i
                info_m = info
                break
        if idx is None:
            raise RuntimeError(f"Output device containing '{self.dev_sub}' not found")
        self.dev_index = idx
        rate = int(info_m.get('defaultSampleRate') or 44100)
        if rate not in (8000,16000,22050,24000,44100,48000):
            rate = 44100
        # Always track hardware/device rate; we'll resample if request differs
        self.hw_rate = rate
        self.dev_rate = self.hw_rate
        max_ch = int(info_m.get('maxOutputChannels') or 2)
        self.dev_channels = self.request_channels or (2 if max_ch >= 2 else 1)
        try:
            logging.info("ElevenToDeviceStreamer device='%s' index=%s hw_rate=%d ch=%d frames=%d",
                         info_m.get('name','?'), self.dev_index, self.hw_rate, self.dev_channels, self.frames)
        except Exception:
            pass

    def _upmix_mono_to_stereo(self, pcm: bytes) -> bytes:
        if self.dev_channels != 2:
            return pcm
        if not pcm:
            return pcm
        if len(pcm) & 1:
            pcm = pcm[:-1]
        # duplicate each 16-bit sample to L and R
        out = bytearray(len(pcm) * 2)
        mv = memoryview(out)
        mv[0::4] = pcm[0::2]
        mv[1::4] = pcm[1::2]
        mv[2::4] = pcm[0::2]
        mv[3::4] = pcm[1::2]
        return bytes(out)

    def _push(self, data: bytes):
        if not data:
            return
        with self._lock:
            self._buf.extend(data)
            # Cap buffer to ~10 seconds to avoid unbounded growth: 10s * rate * ch * 2
            cap = 10 * self.dev_rate * self.dev_channels * 2
            if len(self._buf) > cap:
                # drop oldest bytes
                drop = len(self._buf) - cap
                del self._buf[:drop]

    def _pull(self, nbytes: int) -> bytes:
        with self._lock:
            have = len(self._buf)
            if have == 0:
                self._underruns += 1
                return b"\x00" * nbytes
            if have >= nbytes:
                out = bytes(self._buf[:nbytes])
                del self._buf[:nbytes]
                return out
            out = bytes(self._buf)
            self._buf.clear()
        # pad with silence
        return out + (b"\x00" * (nbytes - len(out)))

    def _callback(self, in_data, frame_count, time_info, status):
        if should_abort_playback() and not self._buf:
            return (None, pyaudio.paComplete)
        nbytes = frame_count * self.dev_channels * 2
        data = self._pull(nbytes)
        return (data, pyaudio.paContinue)

    def _producer_main(self, text: str, voice_id: str, model_id: str, out_fmt: str, voice_settings):
        try:
            eleven_client_manager.ensure_fresh()
            audio_stream = eleven_client_manager.client.text_to_speech.stream(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format=out_fmt,
                voice_settings=voice_settings,
            )
            # Observe incoming raw PCM throughput to estimate actual sample rate (mono 16-bit)
            obs_start = time.time()
            obs_bytes = 0
            try:
                obs_interval = float(os.getenv('E2D_OBS_LOG_SEC', '2.0'))
            except Exception:
                obs_interval = 2.0
            next_log = obs_start + max(0.5, obs_interval)
            # Early check window to verify that ElevenLabs honors PCM rate; otherwise switch to buffered convert
            # Determine requested source rate from out_fmt: "pcm_<rate>"
            try:
                req_rate = int(str(out_fmt).split('_')[1])
            except Exception:
                req_rate = int(os.getenv('ELEVEN_BLACKHOLE_RATE', '44100'))
            expect_rate = req_rate
            early_deadline = obs_start + float(os.getenv('E2D_EARLY_CHECK_SEC', '1.0'))
            early_checked = False
            for chunk in audio_stream:
                if should_abort_playback():
                    break
                if not chunk:
                    continue
                # ensure 16-bit alignment
                if len(chunk) & 1:
                    chunk = chunk[:-1]
                # accumulate stats before upmix (EL stream is mono)
                obs_bytes += len(chunk)
                now = time.time()
                if now >= next_log:
                    dur = max(1e-3, now - obs_start)
                    inferred_rate = (obs_bytes / dur) / 2.0  # bytes/s ÷ 2 bytes/sample
                    try:
                        logging.info(
                            "ElevenLabs stream observed ~%.0f bytes/s → inferred %.0f Hz (mono)",
                            obs_bytes / dur, inferred_rate,
                        )
                    except Exception:
                        pass
                    obs_start = now
                    obs_bytes = 0
                    next_log = now + max(0.5, obs_interval)
                # Early verify: if incoming isn't PCM (e.g., compressed), switch to buffered convert
                if (not early_checked) and (now >= early_deadline):
                    early_checked = True
                    dur = max(1e-3, now - (early_deadline - float(os.getenv('E2D_EARLY_CHECK_SEC', '1.0'))))
                    inf_rate = (obs_bytes / dur) / 2.0 if dur > 0 else 0
                    if inf_rate < (0.6 * expect_rate):
                        try:
                            logging.info(
                                "ElevenToDeviceStreamer: inferred %.0f Hz < %.0f; switching to buffered convert %s",
                                inf_rate, expect_rate, out_fmt
                            )
                        except Exception:
                            pass
                        # Pull full PCM via buffered convert and feed the device buffer
                        try:
                            audio = eleven_client_manager.client.text_to_speech.convert(
                                text=text,
                                voice_id=voice_id,
                                model_id=model_id,
                                output_format=out_fmt,
                                voice_settings=voice_settings,
                            )
                            if isinstance(audio, (bytes, bytearray)):
                                pcm = bytes(audio)
                                # if ElevenLabs returned mono PCM at req_rate, resample to device rate
                                try:
                                    rr = req_rate
                                except Exception:
                                    rr = expect_rate
                                if rr and self.dev_rate and rr != self.dev_rate:
                                    try:
                                        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                                        y = resample_poly(a, self.dev_rate, rr)
                                        y = np.clip(y, -1.0, 1.0)
                                        pcm = (y * 32767.0).astype(np.int16).tobytes()
                                    except Exception:
                                        pass
                                # push in manageable chunks (output domain)
                                step = self.frames * self.dev_channels * 2
                                for i in range(0, len(pcm), step):
                                    if should_abort_playback():
                                        break
                                    self._push(self._upmix_mono_to_stereo(pcm[i:i+step]))
                            else:
                                for part in audio:
                                    if not part:
                                        continue
                                    if len(part) & 1:
                                        part = part[:-1]
                                    if should_abort_playback():
                                        break
                                    # resample chunk to device rate if needed
                                    try:
                                        rr = req_rate
                                    except Exception:
                                        rr = expect_rate
                                    if rr and self.dev_rate and rr != self.dev_rate:
                                        try:
                                            a = np.frombuffer(part, dtype=np.int16).astype(np.float32) / 32768.0
                                            y = resample_poly(a, self.dev_rate, rr)
                                            y = np.clip(y, -1.0, 1.0)
                                            part = (y * 32767.0).astype(np.int16).tobytes()
                                        except Exception:
                                            pass
                                    self._push(self._upmix_mono_to_stereo(part))
                        except Exception as ce:
                            logging.warning(f"Buffered convert failed: {ce}")
                        # End producer after buffered feed
                        break
                # ElevenLabs PCM is mono. Resample to hardware rate then upmix.
                data = chunk
                if req_rate and self.dev_rate and req_rate != self.dev_rate:
                    try:
                        import numpy as _np
                        from scipy.signal import resample_poly as _rp
                        a = _np.frombuffer(data, dtype=_np.int16).astype(_np.float32) / 32768.0
                        y = _rp(a, self.dev_rate, req_rate)
                        y = _np.clip(y, -1.0, 1.0)
                        data = (y * 32767.0).astype(_np.int16).tobytes()
                    except Exception:
                        pass
                self._push(self._upmix_mono_to_stereo(data))
        except Exception as e:
            logging.warning(f"ElevenToDeviceStreamer producer error: {e}")
        finally:
            self._done.set()

    def start(self, text: str, voice_id: str, *, model_id: str, voice_settings):
        self._probe_device()
        # Ensure no other active streams are using the device (e.g., test tone)
        try:
            _DEVICE_STREAMS.stop_all()
        except Exception:
            pass
        # Request rate from env (plan-supported), default 44100, else fall back to hardware rate
        try:
            forced_rate = int(os.getenv('ELEVEN_BLACKHOLE_RATE', '44100'))
        except Exception:
            forced_rate = 44100
        if forced_rate not in (8000,16000,22050,24000,32000,44100,48000):
            forced_rate = 44100
        out_fmt = f"pcm_{forced_rate}"
        try:
            logging.info("ElevenToDeviceStreamer requesting %s from ElevenLabs (device hw_rate=%d)", out_fmt, self.dev_rate)
        except Exception:
            pass
        # Start producer
        self._producer = threading.Thread(
            target=self._producer_main,
            args=(text, voice_id, model_id, out_fmt, voice_settings),
            daemon=True,
        )
        self._producer.start()

        # Wait for prebuffer before opening device (smoother start for long clips)
        if self.prebuffer_ms > 0:
            target_bytes = int(self.dev_rate * self.dev_channels * 2 * (self.prebuffer_ms / 1000.0))
            t0 = time.time()
            while True:
                with self._lock:
                    have = len(self._buf)
                if have >= target_bytes or self._done.is_set():
                    break
                if time.time() - t0 > 5.0:
                    break
                time.sleep(0.01)

        # Start device stream with callback
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.dev_channels,
            rate=self.dev_rate,
            output=True,
            output_device_index=self.dev_index,
            frames_per_buffer=self.frames,
            stream_callback=self._callback,
        )
        self.stream.start_stream()
        # Register a stop callback with the registry
        try:
            def _stop():
                try:
                    if self.stream:
                        self.stream.stop_stream(); self.stream.close()
                except Exception:
                    pass
                try:
                    if self.pa:
                        self.pa.terminate()
                except Exception:
                    pass
            _DEVICE_STREAMS.register(_stop)
        except Exception:
            pass

    def wait(self):
        # Wait for producer to finish and buffer to drain
        while True:
            if self._done.is_set():
                with self._lock:
                    empty = (len(self._buf) == 0)
                if empty:
                    break
            if should_abort_playback():
                break
            time.sleep(0.01)
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except Exception:
            pass
        try:
            if self.pa:
                self.pa.terminate()
        except Exception:
            pass
        if self._underruns:
            logging.info("ElevenToDeviceStreamer underruns=%d", self._underruns)

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
    def stream_text(self, text: str, audio_output_override: Optional[str] = None, streaming_override: Optional[bool] = None):
        raise NotImplementedError

    def clone_from_file(self, file_path: str, voice_name: str):
        """Clone a voice from a local audio file. Should return a voice id/url or None."""
        raise NotImplementedError

class PlayHTEngine(TTSEngine):
    """Adapter over existing PlayHT code paths (VoiceCloning + AudioStreamer)."""
    def stream_text(self, text: str, audio_output_override: Optional[str] = None, streaming_override: Optional[bool] = None):
        voice_url = voice_clone_manager.get_recent_clone_url()
        if not voice_url:
            logging.warning("PlayHTEngine.stream_text: no recent clone available; attempting ElevenLabs fallback")
            if AUTO_FALLBACK_TO_ELEVENLABS:
                try:
                    ElevenLabsEngine().stream_text(text, audio_output_override=audio_output_override, streaming_override=streaming_override)
                    return
                except Exception as e2:
                    logging.error(f"ElevenLabs fallback failed: {e2}")
            return
        try:
            use_streaming = ui_state.streaming if streaming_override is None else bool(streaming_override)
            out_route = ui_state.audio_output if audio_output_override is None else str(audio_output_override)
            if not use_streaming:
                # Buffered path for reference-quality WAV
                streamer = AudioStreamerBuffered()
                streamer.stream_audio(text, voice_url)
            else:
                # Live streaming path
                if out_route == "blackhole":
                    # Keep BlackHole device path (uses header parser) for virtual-audio routing
                    streamer = AudioStreamerBlackHole(api_key_manager.get_current_user_id(), api_key_manager.get_current_key())
                    streamer.stream_audio(text, voice_url)
                else:
                    # Use PlayHT stream-pair which yields clean PCM and aligns better with PlayDialog
                    pair = AudioStreamerPair()
                    pair.stream_audio(
                        text=text,
                        voice_url=voice_url,
                        voice_engine=os.getenv('PLAYHT_VOICE_ENGINE', 'PlayDialog')
                    )
        except Exception as e:
            logging.error(f"PlayHTEngine.stream_text failed: {e}")
            # If PlayHT auth fails (e.g., v4 sdk-auth 403 Forbidden),
            # switch to ElevenLabs for this utterance so the show continues.
            msg = str(e)
            if ("403" in msg or "Forbidden" in msg or "sdk-auth" in msg) and AUTO_FALLBACK_TO_ELEVENLABS:
                try:
                    ui_state.append_message("system", "PlayHT auth failed (403) — switching to ElevenLabs for this line.")
                except Exception:
                    pass
                logging.warning("PlayHT auth failure detected; falling back to ElevenLabs for this utterance.")
                try:
                    ElevenLabsEngine().stream_text(text)
                    return
                except Exception as e2:
                    logging.error(f"ElevenLabs fallback also failed: {e2}")

    def clone_from_file(self, file_path: str, voice_name: str):
        try:
            VoiceCloning.send_audio_for_cloning(fileLocation=file_path, voiceName=voice_name)
            return voice_name  # we store name->id mapping inside VoiceCloneManager
        except Exception as e:
            logging.error(f"PlayHTEngine.clone_from_file failed: {e}")
            return None

class ElevenLabsEngine(TTSEngine):
    """Streaming-first ElevenLabs TTS with buffered fallback."""

    def stream_text(self, text: str, audio_output_override: Optional[str] = None, streaming_override: Optional[bool] = None):
        """
        Prefer ElevenLabs streaming (low latency), fall back to buffered convert if needed.
        Uses most-recent ElevenLabs clone if available, else ELEVENLABS_VOICE_ID.
        """
        try:
            # 1) Choose a voice that exists for the CURRENT key
            recent_id = voice_clone_manager.get_recent_clone_id(engine="elevenlabs")
            env_id = os.getenv("ELEVENLABS_VOICE_ID")
            voice_id = None
            source = None
            eleven_client_manager.ensure_fresh()
            for cand, src in ((recent_id, "recent"), (env_id, "env")):
                if cand and eleven_voice_exists(cand):
                    voice_id = cand
                    source = src
                    break
            if not voice_id:
                # fall back to the most recent user-owned/deletable voice
                voices = fetch_elevenlabs_voices()
                if voices:
                    try:
                        voices.sort(key=lambda v: v.get('created_at') or 0)
                    except Exception:
                        pass
                    voice_id = voices[-1]["id"]
                    source = "list"
            if not voice_id:
                logging.warning("ElevenLabsEngine.stream_text: no accessible ElevenLabs voice for current key; skipping playback")
                return
            logging.info(f"ElevenLabsEngine.stream_text (streaming) using voice_id={voice_id} (source={source})")

            # 2) Low-latency model & compact format for streaming; override via env if desired
            els = ui_state.elevenlabs
            model_id = els.model_id_stream or os.getenv("ELEVENLABS_TTS_MODEL_STREAM", os.getenv("ELEVENLABS_TTS_MODEL", "eleven_v3"))
            # Default to MP3 for non-BlackHole routes (SDK player expects MP3). BlackHole path overrides below.
            output_format = els.format_stream or os.getenv("ELEVENLABS_TTS_FORMAT_STREAM", "mp3_44100_32")
            # If routing to BlackHole, request raw PCM at the device's native rate to avoid resampling
            route_blackhole = (str(audio_output_override).lower() == "blackhole") if audio_output_override is not None else (ui_state.audio_output == "blackhole")
            bh_rate = None
            bh_channels = 1
            if route_blackhole:
                pa_probe = pyaudio.PyAudio()
                try:
                    bh_index = None
                    for i in range(pa_probe.get_device_count()):
                        info = pa_probe.get_device_info_by_index(i)
                        if 'BlackHole' in info.get('name', ''):
                            bh_index = i
                            bh_rate = int(info.get('defaultSampleRate') or 48000)
                            # prefer stereo if available
                            max_ch = int(info.get('maxOutputChannels') or 1)
                            bh_channels = 2 if max_ch >= 2 else 1
                            break
                    if bh_rate is None:
                        bh_rate = 48000
                    # choose matching ElevenLabs PCM format
                    if bh_rate not in (8000, 16000, 22050, 24000, 44100, 48000):
                        bh_rate = 48000
                    output_format = f"pcm_{bh_rate}"
                finally:
                    pa_probe.terminate()
                # Visibility: confirm what device rate/channels and format we will stream
                try:
                    logging.info(
                        "ElevenLabs BlackHole route: device_rate=%d Hz, channels=%d, output_format=%s",
                        bh_rate, bh_channels, output_format
                    )
                except Exception:
                    pass

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
                # Use HTTP-based robust streamer for BlackHole (handles unexpected MP3)
                if route_blackhole:
                    device_hint = 'BlackHole'
                    try:
                        ao = (audio_output_override if audio_output_override is not None else ui_state.audio_output) or ''
                        if 'vb' in ao.lower():
                            device_hint = 'VB-Cable'
                    except Exception:
                        pass
                    try:
                        forced_rate = int(os.getenv('ELEVEN_BLACKHOLE_RATE', '44100'))
                    except Exception:
                        forced_rate = 44100
                    if forced_rate not in (8000,16000,22050,24000,32000,44100,48000):
                        forced_rate = 44100
                    frames = int(os.getenv('PYAUDIO_FRAMES', str(CHUNK)))
                    # Publish route for GUI
                    try:
                        with ui_state.lock:
                            ui_state.audio_route = {
                                "device": device_hint,
                                "requested_rate": forced_rate,
                                "device_rate": forced_rate,
                                "device_channels": 2,
                                "frames": frames,
                                "prebuffer_ms": int(os.getenv('E2D_PREBUFFER_MS', '3000')),
                            }
                    except Exception:
                        pass
                    # Do the HTTP stream
                    stream_eleven_http_to_blackhole(
                        voice_id=voice_id,
                        text=text,
                        model_id=model_id,
                        device_hint=device_hint,
                        device_rate=forced_rate,
                        frames_per_buffer=frames,
                        jitter_ms=int(os.getenv('E2D_JITTER_MS', '150')),
                        requested_of=f"pcm_{forced_rate}",
                    )
                    ADMIN_ABORT_TTS.clear()
                    return
                eleven_client_manager.ensure_fresh()
                audio_stream = eleven_client_manager.client.text_to_speech.stream(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format=output_format,
                    voice_settings=voice_settings,
                )
                # ElevenLabs PCM stream is mono 16-bit LE. When routing to stereo devices
                # (e.g., BlackHole 2ch → Ableton), upmix to interleaved stereo or open the
                # output stream as 1 channel. Mismatched channel counts cause pops/warping.
                if route_blackhole and output_format.startswith("pcm_"):
                    import array
                    pa = pyaudio.PyAudio()

                    def _probe_blackhole():
                        bh_index = None
                        bh_rate = 48000
                        bh_channels = 2
                        for i in range(pa.get_device_count()):
                            info = pa.get_device_info_by_index(i)
                            if 'BlackHole' in str(info.get('name', '')):
                                bh_index = i
                                try:
                                    r = int(info.get('defaultSampleRate') or 48000)
                                    if r in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
                                        bh_rate = r
                                except Exception:
                                    pass
                                try:
                                    max_ch = int(info.get('maxOutputChannels') or 2)
                                    bh_channels = 2 if max_ch >= 2 else 1
                                except Exception:
                                    pass
                                break
                        if bh_index is None:
                            raise RuntimeError("BlackHole output device not found")
                        return bh_index, bh_rate, bh_channels

                    def _mono16_to_stereo(pcm: bytes) -> bytes:
                        if not pcm:
                            return pcm
                        if len(pcm) & 1:
                            pcm = pcm[:-1]
                        a = array.array('h'); a.frombytes(pcm)
                        out = array.array('h', [0]) * (len(a) * 2)
                        j = 0
                        for s in a:
                            out[j] = s; out[j+1] = s; j += 2
                        return out.tobytes()

                    bh_index, bh_rate, bh_channels = _probe_blackhole()

                    desired_fmt = f"pcm_{bh_rate}"
                    if output_format != desired_fmt:
                        try:
                            audio_stream.close()
                        except Exception:
                            pass
                        audio_stream = eleven_client_manager.client.text_to_speech.stream(
                            text=text,
                            voice_id=voice_id,
                            model_id=model_id,
                            output_format=desired_fmt,
                            voice_settings=voice_settings,
                        )

                    if BLACKHOLE_WRITE_TO_FILE:
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        out_path = f"RecordedAudio/blackhole_debug_eleven_stream_{ts}.wav"
                        try:
                            wf = wave.open(out_path, 'wb')
                            wf.setnchannels(bh_channels)
                            wf.setsampwidth(2)
                            wf.setframerate(bh_rate)
                            for chunk in audio_stream:
                                if not chunk:
                                    continue
                                if should_abort_playback():
                                    logging.info("Abort requested – stopping ElevenLabs BlackHole stream (file)")
                                    break
                                if len(chunk) & 1:
                                    chunk = chunk[:-1]
                                if bh_channels == 2:
                                    chunk = _mono16_to_stereo(chunk)
                                wf.writeframes(chunk)
                        finally:
                            try:
                                wf.close()
                                logging.info(f"Wrote ElevenLabs BlackHole debug WAV to {out_path}")
                            except Exception:
                                pass
                        ADMIN_ABORT_TTS.clear()
                        return
                    else:
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=bh_channels,
                            rate=bh_rate,
                            output=True,
                            output_device_index=bh_index,
                            frames_per_buffer=CHUNK,
                        )

                        try:
                            for chunk in audio_stream:
                                if not chunk:
                                    continue
                                if should_abort_playback():
                                    logging.info("Abort requested – stopping ElevenLabs BlackHole stream")
                                    break
                                if len(chunk) & 1:
                                    chunk = chunk[:-1]
                                if bh_channels == 2:
                                    chunk = _mono16_to_stereo(chunk)
                                stream.write(chunk)
                        finally:
                            try:
                                stream.stop_stream(); stream.close()
                            except Exception:
                                pass
                            pa.terminate()
                            ADMIN_ABORT_TTS.clear()
                        return

                # Non-BlackHole: if the SDK's local streamer is available, use it for MP3 only
                if el_stream and callable(el_stream) and ("mp3" in (output_format or "")):
                    el_stream(audio_stream)
                    return

                # Fallback: write streamed chunks to a temp file then play
                import tempfile, wave
                if str(output_format or '').startswith('pcm_'):
                    # Wrap raw PCM to a valid WAV container for system playback
                    try:
                        sr = int(str(output_format).split('_')[1])
                    except Exception:
                        sr = 44100
                    with tempfile.NamedTemporaryFile(prefix="11l_stream_", suffix=".wav", delete=False) as tf:
                        tmp_path = tf.name
                    with contextlib.closing(wave.open(tmp_path, 'wb')) as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        for chunk in audio_stream:
                            if not chunk:
                                continue
                            if len(chunk) & 1:
                                chunk = chunk[:-1]
                            wf.writeframes(chunk)
                    _play_file_best_effort(tmp_path, codec_hint='wav')
                else:
                    suffix = ".mp3" if "mp3" in (output_format or '') else ".bin"
                    with tempfile.NamedTemporaryFile(prefix="11l_stream_", suffix=suffix, delete=False) as tf:
                        for chunk in audio_stream:
                            if chunk:
                                tf.write(chunk)
                        tmp_path = tf.name
                    _play_file_best_effort(tmp_path, codec_hint=('mp3' if suffix=='.mp3' else 'bin'))
                return
            except Exception as se:
                logging.warning(f"ElevenLabsEngine.stream_text: streaming failed ({se}); falling back to buffered convert")

            # 5) BUFFERED FALLBACK (previous behaviour)
            model_id_fb = els.model_id_buffered or "eleven_v3"
            output_format_fb = els.format_buffered or "pcm_44100"
            if route_blackhole:
                # Match device default rate if possible
                rate_fb = bh_rate or 48000
                if rate_fb not in (8000,16000,22050,24000,44100,48000):
                    rate_fb = 48000
                output_format_fb = f"pcm_{rate_fb}"
                try:
                    logging.info(
                        "ElevenLabs BlackHole buffered: device_rate=%d Hz, channels=%d, output_format=%s",
                        rate_fb, bh_channels, output_format_fb
                    )
                except Exception:
                    pass
            eleven_client_manager.ensure_fresh()
            audio = eleven_client_manager.client.text_to_speech.convert(
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

            # Try SDK play helper (non-BlackHole). If BlackHole, play via PyAudio.
            if route_blackhole and output_format_fb.startswith("pcm_"):
                # Buffered PCM → either write to WAV (debug) or play via BlackHole
                try:
                    if BLACKHOLE_WRITE_TO_FILE:
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        out_path = f"RecordedAudio/blackhole_debug_eleven_buffered_{ts}.wav"
                        try:
                            sample_rate = int(output_format_fb.split('_')[1])
                        except Exception:
                            sample_rate = bh_rate or 48000
                        channels = 2 if (bh_channels or 1) >= 2 else 1
                        try:
                            wf = wave.open(out_path, 'wb')
                            wf.setnchannels(channels)
                            wf.setsampwidth(2)
                            wf.setframerate(sample_rate)
                            pcm = buf
                            if len(pcm) & 1:
                                pcm = pcm[:-1]
                            if channels == 2:
                                out = bytearray(len(pcm)*2)
                                mv = memoryview(out)
                                mv[0::4] = pcm[0::2]
                                mv[1::4] = pcm[1::2]
                                mv[2::4] = pcm[0::2]
                                mv[3::4] = pcm[1::2]
                                pcm = bytes(out)
                            wf.writeframes(pcm)
                        finally:
                            try:
                                wf.close()
                                logging.info(f"Wrote ElevenLabs BlackHole debug WAV to {out_path}")
                            except Exception:
                                pass
                        return
                    else:
                        pa = pyaudio.PyAudio()
                        bh_index = None
                        bh_info = None
                        for i in range(pa.get_device_count()):
                            dev = pa.get_device_info_by_index(i)
                            if 'BlackHole' in dev.get('name', ''):
                                bh_index = i
                                bh_info = dev
                                break
                        if bh_index is None:
                            logging.error("ElevenLabsEngine: BlackHole device not found; falling back to system player")
                            raise RuntimeError("no_blackhole")
                        try:
                            sample_rate = int(output_format_fb.split('_')[1])
                        except Exception:
                            sample_rate = bh_rate or 48000
                        channels = 2 if int(bh_info.get('maxOutputChannels') or 1) >= 2 else 1
                        frames = int(os.getenv('PYAUDIO_FRAMES', str(CHUNK)))
                        stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, output=True,
                                         output_device_index=bh_index, frames_per_buffer=frames)
                        try:
                            pcm = buf
                            if len(pcm) & 1:
                                pcm = pcm[:-1]
                            if channels == 2:
                                # duplicate mono to stereo
                                out = bytearray(len(pcm)*2)
                                mv = memoryview(out)
                                mv[0::4] = pcm[0::2]
                                mv[1::4] = pcm[1::2]
                                mv[2::4] = pcm[0::2]
                                mv[3::4] = pcm[1::2]
                                pcm = bytes(out)
                            stream.write(pcm)
                        finally:
                            try:
                                stream.stop_stream(); stream.close()
                            except Exception:
                                pass
                            pa.terminate()
                        return
                except Exception:
                    # fall back to system player below
                    pass
            else:
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
                    import tempfile, wave
                    if str(output_format_fb or '').startswith('pcm_'):
                        try:
                            sr = int(str(output_format_fb).split('_')[1])
                        except Exception:
                            sr = 44100
                        with tempfile.NamedTemporaryFile(prefix='11l_tts_', suffix='.wav', delete=False) as tf:
                            tmp_path = tf.name
                        with contextlib.closing(wave.open(tmp_path, 'wb')) as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(sr)
                            wf.writeframes(buf if not (len(buf) & 1) else buf[:-1])
                        _play_file_best_effort(tmp_path, codec_hint='wav')
                    else:
                        suffix = '.mp3' if 'mp3' in (output_format_fb or '') else '.bin'
                        with tempfile.NamedTemporaryFile(prefix='11l_tts_', suffix=suffix, delete=False) as tf:
                            tf.write(buf)
                            tmp_path = tf.name
                        _play_file_best_effort(tmp_path, codec_hint=('mp3' if suffix=='.mp3' else 'bin'))

        except Exception as e:
            logging.error(f"ElevenLabsEngine.stream_text failed: {e}")

    def clone_from_file(self, file_path: str, voice_name: str):
        """Create an ElevenLabs instant voice clone from a local file and store the id."""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            voice = eleven_client_manager.client.voices.ivc.create(
                name=voice_name,
                files=[BytesIO(data)],
            )
            voice_id = getattr(voice, "voice_id", None) or getattr(voice, "id", None)
            if not voice_id:
                logging.error("ElevenLabsEngine.clone_from_file: no voice_id returned")
                return None

            voice_clone_manager.add_new_clone(voice_name, voice_id, engine="elevenlabs")
            try:
                eleven_client_manager.ivc_consumed(1)
            except Exception:
                pass
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

def tts_speak(text: str, *, audio_output_override: Optional[str] = None, streaming_override: Optional[bool] = None, engine_override: Optional[str] = None):
    try:
        engine: TTSEngine
        if engine_override:
            eo = engine_override.strip().lower()
            if eo == 'elevenlabs':
                engine = ElevenLabsEngine()
            else:
                engine = PlayHTEngine()
        else:
            engine = get_tts_engine()
        engine.stream_text(text, audio_output_override=audio_output_override, streaming_override=streaming_override)
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
        # Use current engine at execution time (not a module-level singleton)
        try:
            eng = get_tts_engine()
            eng.clone_from_file(concatenated_path, f'session_clone{ts}')
        except Exception as e:
            logging.error(f"tts_clone_from_session failed: {e}")

    if threaded:
        th = threading.Thread(target=_do_clone, daemon=True)
        th.start()
        return th
    else:
        _do_clone()
        return None


def _find_latest_consolidated_wav() -> Optional[str]:
    """Return the most recent consolidated WAV from RecordedAudio or None.
    Looks for files produced by concatenation helpers, e.g.:
    - RecordedAudio/concatenated_audio_*.wav
    - RecordedAudio/concatenated_pending_*.wav
    """
    try:
        candidates = []
        for pattern in (
            os.path.join("RecordedAudio", "concatenated_audio_*.wav"),
            os.path.join("RecordedAudio", "concatenated_pending_*.wav"),
        ):
            candidates.extend(glob.glob(pattern))
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: os.path.getmtime(p))
        return latest
    except Exception as e:
        logging.warning(f"Failed to scan for consolidated WAVs: {e}")
        return None


def startup_clone_from_recent_consolidated():
    """If a consolidated WAV exists, create a clone at startup using the selected engine.
    This seeds the session with a usable voice (PlayHT or ElevenLabs).
    """
    path = _find_latest_consolidated_wav()
    if not path:
        logging.info("No consolidated WAV found at startup; skipping initial clone")
        return

    try:
        # Ensure current clone cache is loaded
        try:
            voice_clone_manager.sync_state()
        except Exception:
            pass

        engine = (ui_state.engine or 'playht').strip().lower()
        ts = time.strftime('%Y%m%d-%H%M%S')
        name = f"startup_clone{ts}"

        if engine == 'elevenlabs':
            logging.info(f"Startup cloning (ElevenLabs) from {os.path.basename(path)}")
            el = ElevenLabsEngine()
            vid = el.clone_from_file(path, name)
            if vid:
                logging.info(f"Startup ElevenLabs clone ready: {name} -> {vid}")
            else:
                logging.warning("Startup ElevenLabs clone failed; TTS may fallback or use previous clones")
        else:
            logging.info(f"Startup cloning (PlayHT) from {os.path.basename(path)}")
            try:
                VoiceCloning.send_audio_for_cloning(fileLocation=path, voiceName=name)
                logging.info("Startup PlayHT clone queued/created")
            except Exception as e:
                logging.warning(f"Startup PlayHT clone failed: {e}")
    except Exception as e:
        logging.warning(f"startup_clone_from_recent_consolidated encountered an error: {e}")



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
            body = getattr(getattr(e, 'response', None), 'text', '')
            logging.error("send_audio_for_cloning error (status=%s): %s — %s", status, e, (body or '').strip()[:300])
            # If rate limited, wait and try once more, then move on without blocking TTS
            if status == 429:
                logging.warning("Cloning 429 – cooling down %ss then retrying once", VOICE_DELETE_COOLDOWN_SEC)
                time.sleep(max(1, VOICE_DELETE_COOLDOWN_SEC))
                try:
                    with open(fileLocation, "rb") as f:
                        files = {"sample_file": (os.path.basename(fileLocation), f, "audio/wav")}
                        response = requests.post(url, data=payload, files=files, headers=headers, timeout=15)
                    response.raise_for_status()
                    voice_id = errorCatching.extract_voice_id(response)
                    voice_clone_manager.add_new_clone(voiceName, voice_id, engine='playht')
                    logging.info("Cloned voice created on retry: %s -> %s", voiceName, voice_id)
                    return
                except Exception as e2:
                    logging.error("Cloning retry failed (still %s?): %s", status, e2)
                    # fall through to non-fatal behaviour
            if status == 403:
                voice_clone_manager.delete_oldest_clone(engine='playht')
                api_key_manager.switch_key()
            # Do not hard-fail the pipeline — allow TTS to continue via fallback
            logging.info("Proceeding without new clone; playback can use last known PlayHT clone or ElevenLabs fallback")
        except Exception as e:
            logging.error(f"send_audio_for_cloning unexpected error: {e}")

class VoiceCloneManager:
    def __init__(self, capacity_playht: int = 8, capacity_elevenlabs: int = 4, storage_file: Optional[str] = None):
        # Unbounded; we enforce caps ourselves
        self.clones = deque()
        self.capacity_map = {
            'playht': int(capacity_playht),
            'elevenlabs': int(capacity_elevenlabs),
        }
        # Use a consistent file path at project root so GUI launches (cwd=BigBird)
        # and direct runs (cwd=repo) share the same state file.
        if not storage_file:
            project_root = Path(__file__).resolve().parent.parent
            storage_file = str(project_root / 'voice_clones.json')
        self.storage_file = storage_file
        self.tombstoned_ids = set()
        self.load_clones()
        # Avoid blocking startup on network I/O; sync in background unless explicitly requested
        if _env_bool("SYNC_CLONES_ON_STARTUP", False):
            self.sync_with_api()
        else:
            try:
                threading.Thread(target=self.sync_with_api, daemon=True).start()
            except Exception:
                # Non-fatal; will sync later when needed
                pass

    def sync_with_api(self):
        # PlayHT
        try:
            ph = fetch_cloned_voices()
        except Exception as e:
            logging.warning(f"PlayHT sync failed: {e}")
            ph = []
        # Authoritative replace: drop local PlayHT entries, then add server list
        self.clones = deque([c for c in self.clones if c.get('engine') != 'playht'])
        for v in ph:
            norm = {
                'name': v.get('name') or v.get('id'),
                'id': v.get('id'),
                'engine': 'playht',
                'created_at': time.time()
            }
            if norm['id']:
                self.clones.append(norm)

        # ElevenLabs (filtered to your clones/generated)
        try:
            el = fetch_elevenlabs_voices()
        except Exception as e:
            logging.warning(f"ElevenLabs sync failed: {e}")
            el = None
        live_el_ids = {v.get("id") for v in (el or []) if v.get("id")}
        # Purge only when we fetched successfully; if el is None, keep existing local entries
        if el is not None:
            try:
                self.clones = deque([
                    c for c in self.clones
                    if not (c.get('engine') == 'elevenlabs' and c.get('id') and c.get('id') not in live_el_ids)
                ])
            except Exception:
                pass
        for v in (el or []):
            norm = {
                "name": v.get("name"),
                "id": v.get("id"),
                "engine": "elevenlabs",
                "created_at": v.get("created_at") or time.time(),
                "category": v.get("category"),
                "is_owner": v.get("is_owner"),
                "permission": v.get("permission"),
            }
            # Tag with the active alias for this sync
            try:
                data = ElevenKeyring.load()
                active = ElevenKeyring.get_active_record(data)
                alias = active.get("alias") if active else (data.get("last_active_alias") if data.get("keys") else None)
                norm["account_alias"] = alias
            except Exception:
                pass
            if not any(c.get("id") == norm["id"] for c in self.clones):
                self.clones.append(norm)

        # Dedupe and enforce caps

        # Post-load hygiene (do NOT enforce here — wait until sync)
        self._dedupe_in_place()
            
    def _count_engine(self, engine: str) -> int:
        return sum(1 for c in self.clones if c.get('engine','playht') == engine)

    def _enforce_capacity(self, engine: str):
        # Do not auto-delete remote ElevenLabs voices; only manage PlayHT capacity locally
        if engine == 'elevenlabs':
            return
        cap = self.capacity_map.get(engine, 999999)
        while self._count_engine(engine) > cap:
            # Choose the oldest that is not tombstoned
            candidates = [
                c for c in self.clones
                if c.get('engine','playht') == engine and c.get('id') not in self.tombstoned_ids
            ]
            if not candidates:
                break
            oldest = min(candidates, key=lambda c: c.get('created_at', 0))
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
            # Tag with the active ElevenLabs alias at creation time
            try:
                data = ElevenKeyring.load()
                active = ElevenKeyring.get_active_record(data)
                alias = active.get("alias") if active else (data.get("last_active_alias") if data.get("keys") else None)
            except Exception:
                alias = None
            meta.update({
                "category": "cloned",
                "is_owner": True,
                "account_alias": alias,
            })
        self.clones.append(meta)
        self._dedupe_in_place()
        self._enforce_capacity(engine)
        self.save_clones()
        self.recent_clone = self.get_recent_clone_id(engine=engine)
        logging.info(f"recent {engine} clone id {self.recent_clone}")
        
    def sync_state(self):
        #with self.lock:
            self.load_clones()  # Reload or re-synchronize the state   


    def resolve_playht_id(self, raw_id_or_hint: str) -> Optional[str]:
        """Map an S3-style id or fuzzy name to a canonical PlayHT id from the current cache."""
        hint = (raw_id_or_hint or '').strip()
        if not hint:
            return None
        keypart = hint
        try:
            if hint.startswith('s3://') and hint.endswith('/manifest.json'):
                folder = hint.split('/')[-2]
                keypart = folder
        except Exception:
            pass
        for c in list(self.clones):
            if c.get('engine') != 'playht':
                continue
            nm = (c.get('name') or '').strip()
            if not nm:
                continue
            if nm == keypart or keypart in nm:
                return c.get('id')
        return None

    def delete_oldest_clone(self, engine: Optional[str] = None):
        """Remove the oldest voice clone, optionally restricted to engine; also delete remotely."""
        if not self.clones:
            logging.info("delete_oldest_clone: no clones in memory")
            return
        candidates = [c for c in self.clones if (engine is None or c.get('engine','playht') == engine)]
        if not candidates:
            logging.info("delete_oldest_clone: no candidates for engine=%s", engine)
            return
        oldest = min(candidates, key=lambda c: c.get('created_at', 0))
        vid = oldest.get('id')
        eng = oldest.get('engine','playht')
        if not vid:
            logging.error("delete_oldest_clone: candidate missing id (engine=%s): %r", eng, oldest)
            return
        ok, reason = delete_clone_by_id(vid, engine=eng)
        if ok:
            try:
                self.clones.remove(oldest)
                self.save_clones()
            except ValueError:
                pass
        else:
            if reason == 'not_exist':
                # Remote confirms it doesn't exist — drop local entry to stop spikes
                try:
                    self.tombstoned_ids.add(vid)
                    self.clones.remove(oldest)
                    self.save_clones()
                    logging.info("Pruned stale local clone id=%s engine=%s (not_exist)", vid, eng)
                except ValueError:
                    pass
            else:
                logging.warning("delete_oldest_clone: remote delete failed for id=%s engine=%s; keeping local entry", vid, eng)
        
    def save_clones(self):
        try:
            with open(self.storage_file, 'w') as file:
                json.dump(list(self.clones), file)
        except Exception as e:
            logging.error(f"Error saving clones: {e}")

    def get_recent_clone_id(self, index: int = 1, engine: str = 'playht') -> Optional[str]:
        filtered = [c for c in self.clones if c.get('engine', 'playht') == engine]
        if engine == 'elevenlabs':
            # Prefer clones matching the active alias
            try:
                data = ElevenKeyring.load()
                active = ElevenKeyring.get_active_record(data)
                alias = active.get("alias") if active else (data.get("last_active_alias") if data.get("keys") else None)
            except Exception:
                alias = None
            if alias:
                filtered_by_alias = [c for c in filtered if c.get('account_alias') == alias]
                # Sort by created_at to ensure newest last
                filtered_by_alias.sort(key=lambda c: c.get('created_at') or 0)
                if len(filtered_by_alias) >= index:
                    return filtered_by_alias[-index]['id']
        # Fallback: sort all by created_at
        filtered.sort(key=lambda c: c.get('created_at') or 0)
        if len(filtered) >= index:
            return filtered[-index]['id']
        return None

    def get_recent_clone_url(self, index: int = 1, engine: str = 'playht') -> Optional[str]:
        # For PlayHT, the "url" we store is the id used by their API
        filtered = [c for c in self.clones if c.get('engine', 'playht') == engine]
        filtered.sort(key=lambda c: c.get('created_at') or 0)
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
                "account_alias": c.get("account_alias"),
                "deletable": _is_deletable(c),
            }
            for c in list(self.clones)
        ]

def delete_clone_by_id(voice_id: str, engine: str = 'playht') -> tuple[bool, str]:
    """
    Attempt to delete a cloned voice remotely.
    Returns (ok, reason) where ok is True on successful remote delete, and
    reason is one of: 'ok', 'not_exist', 'error'.
    """
    try:
        if not voice_id:
            logging.error("delete_clone_by_id: missing voice_id (engine=%s)", engine)
            return False, 'error'

        # ---------------- PLAYHT ----------------
        if engine.lower() == 'playht':
            # Path-style endpoint with canonical id (works when id is canonical)
            url_playht = f"https://api.play.ht/api/v2/cloned-voices/{urllib.parse.quote(str(voice_id))}"
            headers = {
                "Authorization": api_key_manager.get_current_key(),
                "X-USER-ID": api_key_manager.get_current_user_id(),
                "accept": "application/json",
            }
            try:
                resp = requests.delete(url_playht, headers=headers, timeout=15)
                if 200 <= resp.status_code < 300:
                    logging.info("Deleted PlayHT clone %s", voice_id)
                    return True, 'ok'

                # Rate limit → cool down and retry once
                if resp.status_code == 429:
                    logging.warning("PlayHT delete 429 for %s – cooling down %ss then retrying once", voice_id, VOICE_DELETE_COOLDOWN_SEC)
                    time.sleep(max(1, VOICE_DELETE_COOLDOWN_SEC))
                    resp2 = requests.delete(url_playht, headers=headers, timeout=15)
                    if 200 <= resp2.status_code < 300:
                        logging.info("Deleted PlayHT clone %s on retry", voice_id)
                        return True, 'ok'
                    # If still not good, continue with fallbacks using latest response
                    resp = resp2

                # Some tenants return S3 manifest paths as ids → path DELETE can 400/404.
                # 1) Try body-based delete on collection endpoint
                if resp.status_code in (404, 400):
                    try:
                        url_coll = "https://api.play.ht/api/v2/cloned-voices"
                        headers_json = dict(headers)
                        headers_json["content-type"] = "application/json"
                        payload = {"voice_id": voice_id}
                        resp_b = requests.delete(url_coll, headers=headers_json, json=payload, timeout=15)
                        if 200 <= resp_b.status_code < 300:
                            logging.info("Deleted PlayHT clone %s via body-based endpoint", voice_id)
                            return True, 'ok'
                    except requests.exceptions.RequestException as re2:
                        logging.warning("Body-based delete request error for %s: %s", voice_id, re2)

                    # 2) Map S3-style id/name to a canonical id and retry path-style DELETE
                    try:
                        canon = None
                        try:
                            canon = voice_clone_manager.resolve_playht_id(voice_id)
                        except Exception:
                            canon = None
                        if not canon:
                            try:
                                fresh = fetch_cloned_voices()
                                # refresh cache: drop playht, insert fresh
                                voice_clone_manager.clones = deque([c for c in voice_clone_manager.clones if c.get('engine') != 'playht'])
                                for v in fresh:
                                    if v.get('id'):
                                        voice_clone_manager.clones.append({'name': v.get('name') or v.get('id'), 'id': v.get('id'), 'engine': 'playht', 'created_at': time.time()})
                                canon = voice_clone_manager.resolve_playht_id(voice_id)
                            except Exception as _e:
                                logging.warning("Could not refresh PlayHT list for canonical mapping: %s", _e)
                        if canon and canon != voice_id:
                            url_canon = f"https://api.play.ht/api/v2/cloned-voices/{urllib.parse.quote(str(canon))}"
                            resp_canon = requests.delete(url_canon, headers=headers, timeout=15)
                            if 200 <= resp_canon.status_code < 300:
                                logging.info("Deleted PlayHT clone via canonical id %s (hint was %s)", canon, voice_id)
                                try:
                                    for c in list(voice_clone_manager.clones):
                                        if c.get('engine') == 'playht' and c.get('id') == voice_id:
                                            c['id'] = canon
                                except Exception:
                                    pass
                                return True, 'ok'
                            logging.error("Canonical delete failed for hint %s → %s: %s %s — %s",
                                          voice_id, canon, resp_canon.status_code, resp_canon.reason, (resp_canon.text or '').strip()[:200])
                    except requests.exceptions.RequestException as re3:
                        logging.error("Canonical id delete request error for %s: %s", voice_id, re3)
                    except Exception as e3:
                        logging.error("Canonical id resolution failed for %s: %s", voice_id, e3)

                # Final log for other failures
                logging.error("Error deleting PlayHT clone %s: %s %s — %s", voice_id, resp.status_code, resp.reason, (resp.text or '').strip()[:300])
                return False, 'error'

            except requests.exceptions.RequestException as re:
                status = getattr(getattr(re, 'response', None), 'status_code', None)
                body = getattr(getattr(re, 'response', None), 'text', '')
                logging.error("delete_clone_by_id PlayHT request error (id=%s): %s (status=%s) %s", voice_id, re, status, (body or '').strip()[:300])
                return False, 'error'

        # ---------------- ELEVENLABS ----------------
        elif engine.lower() == 'elevenlabs':
            try:
                # ElevenLabs SDK: success typically raises no exception
                eleven_client_manager.client.voices.delete(voice_id)
                logging.info("Deleted ElevenLabs voice %s", voice_id)
                return True, 'ok'
            except Exception as e:
                msg = getattr(e, 'message', None) or str(e)
                try:
                    body = getattr(getattr(e, 'response', None), 'json', lambda: {})()
                except Exception:
                    body = {}
                status = None
                if isinstance(body, dict):
                    detail = body.get('detail') or {}
                    status = detail.get('status')
                if status == 'voice_does_not_exist' or 'voice_does_not_exist' in msg:
                    return False, 'not_exist'
                logging.error("delete_clone_by_id ElevenLabs error (id=%s): %s", voice_id, msg)
                return False, 'error'

        else:
            logging.warning("delete_clone_by_id: unknown engine '%s' for id=%s", engine, voice_id)
            return False, 'error'

    except Exception as e:
        logging.error("delete_clone_by_id unexpected error (id=%s, engine=%s): %s", voice_id, engine, e)
        return False, 'error'

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
        voices = []
        for voice in voice_data:
            vid = voice.get('id') or voice.get('voice_id') or voice.get('url')
            name = voice.get('name') or voice.get('title') or vid
            if not vid:
                continue
            voices.append({'id': vid, 'name': name})
        return voices
    except requests.exceptions.Timeout:
        logging.error("The request timed out")
        return []
    except requests.exceptions.RequestException as e:
        # Handle other request exceptions (e.g., HTTPError, ConnectionError, etc.)
        logging.error(f"Failed to fetch cloned voices: {e}")
        return []

def fetch_elevenlabs_voices():
    api_key = eleven_client_manager.current_api_key() or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logging.warning("ELEVENLABS_API_KEY not set; cannot list ElevenLabs voices")
        return None
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
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch ElevenLabs voices: {e}")
        return None

def fetch_elevenlabs_voices_with_key(api_key: str):
    """Fetch ElevenLabs voices using a specific API key. Returns list or None on error."""
    if not api_key:
        return None
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key, "accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        out = []
        for v in data.get("voices", []):
            cat = (v.get("category") or "").lower()
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
        logging.error("ElevenLabs voices request timed out (alias-specific)")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch ElevenLabs voices (alias-specific): {e}")
        return None


# ---------------- ElevenLabs voice resolution helper ----------------
from typing import Optional

def resolve_active_eleven_voice_id(preferred: Optional[str] = None) -> Optional[str]:
    """Return a voice_id that exists for the CURRENT ElevenLabs API key.
    Order of precedence:
      1) If `preferred` is provided and exists (via /voices/{id}), return it.
      2) Else list voices for the current key (cloned/generated only) and return the newest.
      3) Else fall back to local cache filtered by active alias.
    """
    try:
        api_key = eleven_client_manager.current_api_key() or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            return None

        # 1) Honor preferred if it truly exists for the current key
        if preferred:
            try:
                if eleven_voice_exists(preferred):
                    return preferred
            except Exception:
                pass

        # 2) Ask ElevenLabs which voices this key can actually see
        voices = fetch_elevenlabs_voices() or []
        if voices:
            try:
                voices.sort(key=lambda v: v.get("created_at") or 0)
            except Exception:
                pass
            # newest matching entry (prefer cloned/generated already filtered by fetch_...)
            if voices:
                return voices[-1].get("id")

        # 3) Fallback: local cache filtered by account alias
        try:
            data = ElevenKeyring.load()
            active = ElevenKeyring.get_active_record(data)
            alias = active.get("alias") if active else (data.get("last_active_alias") if data.get("keys") else None)
        except Exception:
            alias = None
        try:
            candidates = [
                c for c in list(voice_clone_manager.clones)
                if c.get("engine") == "elevenlabs" and (alias is None or c.get("account_alias") == alias)
            ]
            candidates.sort(key=lambda c: c.get("created_at") or 0)
            if candidates:
                return candidates[-1].get("id")
        except Exception:
            pass
    except Exception:
        pass
    return None

def eleven_voice_exists(voice_id: str) -> bool:
    """Return True if the current ElevenLabs key can see `voice_id`."""
    if not voice_id:
        return False
    api_key = eleven_client_manager.current_api_key() or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return False
    url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
    headers = {"xi-api-key": api_key, "accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            return True
        return False
    except requests.exceptions.RequestException:
        return False

def delete_elevenlabs_voice(voice_id: str) -> bool:
    """Delete an ElevenLabs voice via REST; treat 'voice_does_not_exist' as success; optional SDK fallback."""
    api_key = eleven_client_manager.current_api_key() or os.getenv("ELEVENLABS_API_KEY")
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


"""def delete_clone_by_id(voice_id, engine: Optional[str] = None) -> bool:

    Delete a voice by id for the appropriate provider.
    If engine not supplied, infer from current list; fallback heuristic.

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
        resp = requests.delete(_url, headers=headers, timeout=15)
        if 200 <= resp.status_code < 300:
            logging.info("Deleted PlayHT clone %s", voice_id)
            return True
        if resp.status_code == 429:
            logging.warning("PlayHT delete 429 for %s – cooling down %ss then retrying once", voice_id, VOICE_DELETE_COOLDOWN_SEC)
            time.sleep(max(1, VOICE_DELETE_COOLDOWN_SEC))
            resp2 = requests.delete(_url, headers=headers, timeout=15)
            if 200 <= resp2.status_code < 300:
                logging.info("Deleted PlayHT clone %s on retry", voice_id)
                return True
            logging.error("Retry delete failed for %s: %s %s — %s", voice_id, resp2.status_code, resp2.reason, (resp2.text or '').strip()[:300])
            return False
        # Log body for debugging (400s often include a helpful message)
        logging.error("Error deleting PlayHT clone %s: %s %s — %s", voice_id, resp.status_code, resp.reason, (resp.text or '').strip()[:300])
        return False
    except requests.exceptions.Timeout:
        logging.error("The request timed out")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Error deleting PlayHT clone: {e}")
        return False"""


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
            logging.info(f"error in gen and strean: {e}")
                    
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
    try:
        set = live.Set(scan=True)
        # keep the code that immediately uses `set` below here, unchanged
    except LiveConnectionError as e:
        logging.warning("Ableton Live not available (LiveOSC not responding). Skipping music cycle: %s", e)
        return False
    except Exception as e:
        logging.warning("Unable to initialize Ableton Live. Skipping music cycle: %s", e)
        return False
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

    logging.info("music")

    try:
        set = live.Set(scan=True)
        # keep the code that immediately uses `set` below here, unchanged
    except LiveConnectionError as e:
        logging.warning("Ableton Live not available (LiveOSC not responding). Skipping music cycle: %s", e)
        return False
    except Exception as e:
        logging.warning("Unable to initialize Ableton Live. Skipping music cycle: %s", e)
        return False
    tempo = 60.0
    set.tempo = tempo
    track = set.tracks[0]
    logging.info("Track name '%s'" % track.name)
    clip = track.clips[1]
    logging.info("Clip name '%s', length %d beats" % (clip.name, clip.length))
    clip.play()

    clip_length_beats = clip.length
    clip_length_seconds = (clip_length_beats / tempo) * 60  # Convert beats to seconds
    logging.info("Clip length in seconds: %f" % clip_length_seconds)

    # Wait while on-hook; abort immediately once off-hook
    try:
        if arduino._wait_until_offhook(clip_length_seconds, clip_to_stop=clip):
            return
    except Exception as e:
        logging.info(e)
    finally:
        try:
            clip.stop()
        except Exception:
            pass
        try:
            arduino.double_ring()
        except Exception:
            pass

    logging.info("Clip has finished playing.")

def poem_cycle():

    logging.info("poem")
    
    set = live.Set(scan=True)
    tempo = 60.0
    set.tempo = tempo
    track = set.tracks[0]
    logging.info("Track name '%s'" % track.name)
    clip = track.clips[1]
    logging.info("Clip name '%s', length %d beats" % (clip.name, clip.length))
    clip.play()

    start_time = time.time()

    # Wait while on-hook, but abort immediately if handset lifted
    try:
        if arduino._wait_until_offhook(25, clip_to_stop=clip):
            return
    except Exception:
        time.sleep(25)

    try:
        # Guard thread to abort TTS if handset is lifted mid-stream
        def _abort_guard():
            while True:
                try:
                    if not arduino.onHold():
                        ADMIN_ABORT_TTS.set()
                        break
                except Exception:
                    break
                time.sleep(0.1)

        guard = threading.Thread(target=_abort_guard, daemon=True)
        guard.start()

        # Fake realtime: render each stanza to file and play with small delay
        try:
            forced_rate = int(os.getenv('ELEVEN_BLACKHOLE_RATE', '44100'))
        except Exception:
            forced_rate = 44100
        try:
            buffer_files = int(os.getenv('FAKE_STREAM_BUFFER_FILES', '2'))
        except Exception:
            buffer_files = 2
        lines = [poem[5], poem[6]]
        logging.info("Poem fake-stream: %d lines @ %d Hz, buffer_files=%d", len(lines), forced_rate, buffer_files)
        fs = ElevenFakeStreamer(device_hint='BlackHole', rate=forced_rate, buffer_files=buffer_files)
        fs.start(lines)
        # Do not block here; playback runs while the clip plays and until off-hook
        ADMIN_ABORT_TTS.clear()
    except Exception as e:
        logging.info(e)

    clip_length_beats = clip.length
    clip_length_seconds = (clip_length_beats / tempo) * 60  # Convert beats to seconds
    logging.info("Clip length in seconds: %f" % clip_length_seconds)

    elapsed_time = time.time() - start_time
    remaining_time = clip_length_seconds - elapsed_time
    if remaining_time > 0:
        if arduino._wait_until_offhook(remaining_time, clip_to_stop=clip):
            return

    logging.info("Clip has finished playing.")

def word_cycle(num_random: int = 3, inter_line_pause: float = 0.2):
    """Speak the first line, a few random middle lines, then the last line.
    Uses ElevenLabs HTTP streamer via streamToSpeakers().
    """
    print("word")
    try:
        # Validate poem
        if not isinstance(poem, (list, tuple)) or not poem:
            logging.warning("word_cycle: 'poem' is empty or invalid")
            return

        n = len(poem)
        # Choose up to num_random unique middle indices (1..n-2)
        middle_indices = list(range(1, n - 1)) if n > 2 else []
        k = max(0, min(num_random, len(middle_indices)))
        random_middle = random.sample(middle_indices, k=k) if k else []
        # Order: first, random middles (random order), last (if exists)
        indices = ([0] if n >= 1 else []) + random_middle + ([n - 1] if n >= 2 else [])

        # Resolve credentials/voice once (keychain + alias-aware voice resolution)
        api_key = (eleven_client_manager.current_api_key() or os.getenv("ELEVENLABS_API_KEY"))
        preferred = voice_clone_manager.get_recent_clone_id(engine='elevenlabs') or os.getenv('ELEVENLABS_VOICE_ID')
        voice_id = resolve_active_eleven_voice_id(preferred)
        if not api_key or not voice_id:
            logging.error("word_cycle: missing or invalid ElevenLabs api_key/voice_id (has_key=%s, preferred=%s, resolved=%s)", bool(api_key), preferred, voice_id)
            return
        else:
            if preferred and preferred != voice_id:
                logging.info("word_cycle: preferred voice %s not visible to active key; using %s", preferred, voice_id)
            else:
                logging.info("word_cycle: using ElevenLabs voice %s", voice_id)

        for idx in indices:
            if should_abort_playback():
                logging.info("word_cycle: aborted before line %s", idx)
                break
            try:
                text = str(poem[idx]).strip()
            except Exception:
                text = ""
            if not text:
                continue

            try:
                ok = streamToSpeakers(api_key=api_key, voice_id=voice_id, text=text)
            except Exception as e:
                logging.warning("word_cycle: streamToSpeakers raised on line %s: %s", idx, e)
                ok = False

            if not ok:
                logging.warning("word_cycle: streamToSpeakers failed for line %s", idx)

            if inter_line_pause > 0:
                try:
                    time.sleep(inter_line_pause)
                except Exception:
                    pass
    except Exception as e:
        logging.exception(e)

    print("Clip has finished playing.")

def test_full_interaction_cycle():

    session_manager = SessionAudioManager()
    openai_handler = OpenAIHandler()
    localAI = localAiHandler()
    session = ChatSession(system_message=GptTrainingText[2])
    # Lazy-load Whisper only when local transcription is actually used
    model = _load_whisper_module().load_model("base.en")
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
        logging.info(e)

    arduino.led_Off()
    arduino.led_speaking()
    
    tts_speak(gptGreetings[0])
    ui_state.append_message("assistant", gptGreetings[0])

    try:
        while loop:

            # Director can end conversation at any time
            if ADMIN_END_CONVERSATION.is_set():
                break
        
            # Step 1: Record audio with voice activity detection
            arduino.led_recording()
            logging.info("Recording... Speak into the microphone.")
            recorder = AudioRecorderWithVAD()
            audio_path = recorder.record_with_vad(
                initial_silence_timeout=ui_state.vad.initial_silence_timeout,
                max_silence_length=ui_state.vad.max_silence_length
)



            if arduino.onHold() or ADMIN_END_CONVERSATION.is_set():
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
                logging.info(f"Recording complete. Audio saved to {audio_path}")

            #Check if the voice cloning ai has enough data   

            if session_manager.should_send_for_cloning():
                arduino.led_cloning()
                logging.info("got plenty sending for cloning")
                session.add_system_message(GPTinstructionsCloned[j])
                if j < (len(GPTinstructionsCloned)-1):
                    j += 1
                aiThread = tts_clone_from_session(session_manager, threaded=True)
                if aiThread:
                    threads.append(aiThread)
            
            if arduino.onHold() or ADMIN_END_CONVERSATION.is_set():
                break
            
            arduino.led_cloning()
            transcription = localAI.whisper_audio(audio_path, model)

            #transcription = openai_handler.transcribe_audio(audio_path=audio_path)
            ui_state.append_message("user", transcription or "")
            logging.info('you said %s', transcription)
            if 'exit' in transcription.lower():
                logging.info("exiting program")
                loop = False
                word_cycle()
                poem_cycle()
                break

            if arduino.onHold() or ADMIN_END_CONVERSATION.is_set():
                break

            session.add_user_message(transcription)
            log.write_text(f"user: {transcription}")
            response = session.get_response()
            logging.info("AI says: %s", response)
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
        logging.info(e)
        
    finally:
        # Clear director end flag for next run
        ADMIN_END_CONVERSATION.clear()
        # Always run post-conversation cycles
        word_cycle()
        try:
            poem_cycle()
        except Exception as _e:
            logging.warning(f"poem_cycle failed: {_e}")
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
                logging.info(e)
            music_cycle()
        j = 0

        session.clear_messages()
        if True:
            for thread in threads:
                thread.join()

class FileManager:
    def __init__(self, filename):
        # Anchor relative filenames at the project root so all processes agree
        try:
            base_dir = Path(__file__).resolve().parent.parent
        except Exception:
            base_dir = Path('.')
        if not os.path.isabs(filename):
            self.filename = str(base_dir / filename)
        else:
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

    @app.post("/state/gui_overrides")
    def set_gui_overrides(payload: dict):
        try:
            enabled = bool(payload.get("enabled", False))
            ui_state.set_gui_overrides(enabled)
            return {"ok": True, "gui_overrides": ui_state.gui_overrides}
        except Exception as e:
            return {"ok": False, "error": str(e)}

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

    @app.post("/audio/test_tone")
    def audio_test_tone(payload: dict):
        """
        Generate a continuous sine tone to the BlackHole device to verify end-to-end routing.
        Payload: {"seconds": int, "freq": float, "rate": int, "device": str}
        Defaults: seconds=5, freq=1000.0 Hz, rate=44100, device contains "BlackHole".
        """
        seconds = int(payload.get("seconds", 5) or 5)
        freq = float(payload.get("freq", 1000.0) or 1000.0)
        try:
            rate = int(payload.get("rate", int(os.getenv('TEST_TONE_RATE', '44100'))))
        except Exception:
            rate = 44100
        dev_sub = str(payload.get("device", "BlackHole") or "BlackHole")
        frames = int(os.getenv('PYAUDIO_FRAMES', str(CHUNK)))
        mode = str(payload.get("mode", "blocking") or "blocking").strip().lower()

        try:
            pa = pyaudio.PyAudio()
            idx = None
            info_m = None
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if dev_sub.lower() in str(info.get('name','')).lower():
                    idx = i
                    info_m = info
                    break
            if idx is None:
                raise RuntimeError(f"Output device containing '{dev_sub}' not found")
            channels = 2 if int(info_m.get('maxOutputChannels') or 1) >= 2 else 1

            if mode == "callback":
                # Legacy callback mode (can be CPU-sensitive in Python)
                import math
                phase = 0.0
                two_pi = 2.0 * math.pi
                phase_inc = two_pi * freq / rate
                def cb(in_data, frame_count, time_info, status):
                    nonlocal phase
                    n = frame_count
                    buf = bytearray(n * 2)
                    for i in range(n):
                        s = int(32767 * (0.2 * math.sin(phase)))
                        buf[2*i] = s & 0xff
                        buf[2*i+1] = (s >> 8) & 0xff
                        phase += phase_inc
                        if phase >= two_pi:
                            phase -= two_pi
                    pcm = bytes(buf)
                    if channels == 2:
                        out = bytearray(len(pcm) * 2)
                        mv = memoryview(out)
                        mv[0::4] = pcm[0::2]
                        mv[1::4] = pcm[1::2]
                        mv[2::4] = pcm[0::2]
                        mv[3::4] = pcm[1::2]
                        pcm = bytes(out)
                    return (pcm, pyaudio.paContinue)
                stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=rate, output=True,
                                 output_device_index=idx, frames_per_buffer=frames, stream_callback=cb)
                stream.start_stream()
                t0 = time.time()
                while stream.is_active() and (time.time() - t0 < seconds):
                    time.sleep(0.05)
                try:
                    stream.stop_stream(); stream.close()
                except Exception:
                    pass
            else:
                # Blocking write mode with vectorized generation (more stable)
                import numpy as np
                t = np.arange(0, seconds, 1.0/rate, dtype=np.float32)
                mono = (0.2 * np.sin(2*np.pi*freq*t)).astype(np.float32)
                pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                if channels == 2:
                    out = bytearray(len(pcm) * 2)
                    mv = memoryview(out)
                    mv[0::4] = pcm[0::2]
                    mv[1::4] = pcm[1::2]
                    mv[2::4] = pcm[0::2]
                    mv[3::4] = pcm[1::2]
                    pcm = bytes(out)
                stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=rate, output=True,
                                 output_device_index=idx, frames_per_buffer=frames)
                # Write in large chunks (e.g., 0.5s) to reduce scheduling overhead
                chunk_bytes = int(rate * channels * 2 * 0.5)
                for i in range(0, len(pcm), chunk_bytes):
                    if should_abort_playback():
                        break
                    stream.write(pcm[i:i+chunk_bytes])
                try:
                    stream.stop_stream(); stream.close()
                except Exception:
                    pass
            pa.terminate()
            return {"ok": True, "seconds": seconds, "freq": freq, "rate": rate, "device": info_m.get('name','?'), "channels": channels, "mode": mode}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/eleven/fake_stream")
    def eleven_fake_stream(payload: dict):
        """
        Fake realtime: request ElevenLabs TTS for each line, save as WAV, and
        play them in order with a small initial file delay.

        Payload: {
          "lines": [str, ...],
          "device": "BlackHole"|"VB-CABLE"|...,   # substring match, default BlackHole
          "rate": 44100,                            # ElevenLabs PCM rate
          "buffer_files": 2,                        # initial delay in files
        }
        """
        lines = payload.get("lines") or []
        if not isinstance(lines, list) or not lines:
            return {"ok": False, "error": "lines must be a non-empty list"}
        dev = str(payload.get("device", "BlackHole") or "BlackHole")
        try:
            rate = int(payload.get("rate", int(os.getenv('ELEVEN_BLACKHOLE_RATE', '44100'))))
        except Exception:
            rate = 44100
        try:
            buf_files = int(payload.get("buffer_files", 2))
        except Exception:
            buf_files = 2
        try:
            streamer = ElevenFakeStreamer(device_hint=dev, rate=rate, buffer_files=buf_files)
            streamer.start(lines)
            # Non-blocking endpoint: return immediately
            return {"ok": True, "device": dev, "rate": rate, "buffer_files": buf_files}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/speak")
    def speak(payload: dict):
        txt = payload.get("text", "")
        if not txt:
            return {"ok": False, "error": "no text"}
        ui_state.append_message("assistant", txt)
        threading.Thread(target=tts_speak, args=(txt,), daemon=True).start()
        return {"ok": True}

    # ---- Director overrides ----
    @app.post("/control/vad/force-end")
    def control_vad_force_end():
        try:
            ADMIN_FORCE_VAD_END.set()
            return {"ok": True, "forced": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/control/abort-tts")
    def control_abort_tts():
        try:
            ADMIN_ABORT_TTS.set()
            return {"ok": True, "aborted": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/control/conversation/end")
    def control_end_conversation():
        try:
            ADMIN_END_CONVERSATION.set()
            return {"ok": True, "ended": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

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
            ok, reason = delete_clone_by_id(voice_id, engine=engine)
            if ok or reason == 'not_exist':
                # purge locally and resync
                try:
                    voice_clone_manager.clones = deque([c for c in voice_clone_manager.clones if c.get("id") != voice_id])
                    voice_clone_manager.save_clones()
                except Exception:
                    pass
                if reason == 'not_exist':
                    try:
                        voice_clone_manager.tombstoned_ids.add(voice_id)
                    except Exception:
                        pass
            return {"ok": ok or reason == 'not_exist', "engine": engine, "reason": reason}
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

    @app.post("/clones/sync-all")
    def clones_sync_all():
        """Aggregate ElevenLabs voices for all aliases in keyring; do not purge existing entries."""
        try:
            data = ElevenKeyring.load()
            keys = data.get("keys", [])
            aliases = [k.get("alias") for k in keys if k.get("api_key")]
            if not aliases:
                return {"ok": True, "count": len(voice_clone_manager.clones), "aliases": []}
            added = 0
            seen_ids = set()
            # Build a quick lookup of existing EL ids to avoid duplicates
            try:
                for c in list(voice_clone_manager.clones):
                    if c.get("engine") == "elevenlabs" and c.get("id"):
                        seen_ids.add(c.get("id"))
            except Exception:
                pass
            for rec in keys:
                alias = rec.get("alias")
                api_key = rec.get("api_key")
                if not api_key or not alias:
                    continue
                voices = fetch_elevenlabs_voices_with_key(api_key) or []
                for v in voices:
                    vid = v.get("id")
                    if not vid or vid in seen_ids:
                        continue
                    norm = {
                        "name": v.get("name") or vid,
                        "id": vid,
                        "engine": "elevenlabs",
                        "created_at": v.get("created_at") or time.time(),
                        "category": v.get("category"),
                        "is_owner": v.get("is_owner"),
                        "permission": v.get("permission"),
                        "account_alias": alias,
                    }
                    voice_clone_manager.clones.append(norm)
                    seen_ids.add(vid)
                    added += 1
            # Dedupe once
            voice_clone_manager._dedupe_in_place()
            voice_clone_manager.save_clones()
            return {"ok": True, "added": added, "count": len(voice_clone_manager.clones), "aliases": aliases}
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
    # Optional: disable control server for faster/lighter startupßß
    if not _env_bool("DISABLE_CONTROL_SERVER", False):
        start_control_server()
    # Optionally seed a voice clone from the most recent consolidated audio
    if _env_bool("SEED_CLONE_ON_STARTUP", False):
        try:
            threading.Thread(target=startup_clone_from_recent_consolidated, daemon=True).start()
        except Exception as _e:
            logging.warning(f"Startup clone seeding skipped due to error: {_e}")
    #filmPrint()
    while(True):

        arduino = ArduinoControl(arduinoLocation)

        try:
            while(True):
                #word_cycle()
                #poem_cycle()
                test_full_interaction_cycle()

        except Exception as e:
            print(e)

 
