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
from typing import AsyncGenerator, AsyncIterable, Generator, Iterable, Union, Optional
import struct, binascii   # add at top of the file if not already there
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
        """Add a new audio clip and update total duration."""
        with contextlib.closing(wave.open(file_path, 'rb')) as wf:
            duration = wf.getnframes() / wf.getframerate()
        self.clips.append(file_path)
        self.total_duration += duration

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

        return output_path

class AudioRecorderWithVAD:
    def __init__(self, vad_mode=3):
        self.vad = webrtcvad.Vad(vad_mode)
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
        return False

        isButton = self.send_command("ISBUTTONPRESSED")
        if isButton == "UP":
            return True
        elif isButton == "DOWN":
            return False
        else:
            print("Unexpected response:", isButton)
            return True
        
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
        options = TTSOptions(
            voice=voice_url,
            sample_rate=RATE,          # PlayHT ignores this for WAV; header wins
            format=Format.FORMAT_WAV,
            voice_guidance=1,
            style_guidance=1,
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
                        self.stream.write(bytes(chunk))

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
        options = TTSOptions(
            voice=voice_url,
            sample_rate=24000,
            format=Format.FORMAT_WAV,
            speed=0.7,
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
        # Use the same streamed playback path and the most recent clone
        voice_url = voice_clone_manager.get_recent_clone_url()
        if not voice_url:
            logging.warning("PlayHTEngine.stream_text: no recent clone available; text will be skipped")
            return
        try:
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
            model_id = os.getenv("ELEVENLABS_TTS_MODEL_STREAM", os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"))
            output_format = os.getenv("ELEVENLABS_TTS_FORMAT_STREAM", "mp3_22050_32")

            # 3) Build voice settings (env-overridable). We accept either dict or EL_VoiceSettings
            settings = {
                "use_speaker_boost": (os.getenv("ELEVENLABS_SPEAKER_BOOST", "false").lower() in ("1","true","yes","y")),
                "stability": float(os.getenv("ELEVENLABS_STABILITY", "0.5")),
                "similarity_boost": float(os.getenv("ELEVENLABS_SIMILARITY", "0.8")),
                "style": float(os.getenv("ELEVENLABS_STYLE", "0.1")),
                "speed": float(os.getenv("ELEVENLABS_SPEED", "1.0")),
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
            model_id_fb = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
            output_format_fb = os.getenv("ELEVENLABS_TTS_FORMAT", "mp3_44100_128")
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
    if TTS_ENGINE == 'elevenlabs':
        logging.info("TTS engine: ElevenLabs (scaffold)")
        return ElevenLabsEngine()
    logging.info("TTS engine: PlayHT")
    return PlayHTEngine()

# Global engine instance and convenience helpers
_tts_engine: TTSEngine = get_tts_engine()

def tts_speak(text: str):
    """Speak text using the selected engine (no behaviour change for PlayHT)."""
    try:
        _tts_engine.stream_text(text)
    except Exception as e:
        logging.error(f"tts_speak failed: {e}")

def tts_clone_from_session(session_manager: 'SessionAudioManager', threaded: bool = True):
    """Clone using accumulated session clips via the selected engine. Uses background thread when requested."""
    concatenated_path = session_manager.concatenate_clips()
    if not concatenated_path:
        logging.info("tts_clone_from_session: no audio to clone")
        return None

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

#Maaging Saved Clones
class VoiceCloneManager:
    def __init__(self, capacity: int = 8, storage_file: str = 'voice_clones.json'):
        self.clones = deque(maxlen=capacity)
        self.capacity = capacity
        self.storage_file = storage_file
        self.load_clones()  # Load clones from file on initialization
        self.sync_with_api()



    def sync_with_api(self):
        existing_clones = fetch_cloned_voices()
        for clone in existing_clones:
            normalized = {
                'name': clone.get('name'),
                'id': clone.get('id'),
                'engine': 'playht',
                'created_at': time.time()
            }
            if not any(c['id'] == normalized['id'] for c in self.clones):
                self.clones.append(normalized)
        self.save_clones()
            

    def add_new_clone(self, voice_name, voice_id, engine: str = 'playht'):
        current_time = time.time()
        try:
            if len(self.clones) >= self.capacity:
                # Evict within same engine pool if possible
                self.delete_oldest_clone(engine=engine)
        except Exception as e:
            logging.error(f"Capacity management error: {e}")

        self.clones.append({
            'name': voice_name,
            'id': voice_id,
            'engine': engine,
            'created_at': current_time
        })
        self.save_clones()
        self.recent_clone = self.get_recent_clone_id(engine=engine)
        logging.info(f"recent {engine} clone id {self.recent_clone}")
        
    def sync_state(self):
        #with self.lock:
            self.load_clones()  # Reload or re-synchronize the state   

    def delete_oldest_clone(self, engine: Optional[str] = None):
        """Remove the oldest voice clone, optionally restricted to an engine (playht/elevenlabs)."""
        if not self.clones:
            return
        candidates = [c for c in self.clones if (engine is None or c.get('engine', 'playht') == engine)]
        if not candidates:
            return
        oldest = min(candidates, key=lambda c: c.get('created_at', 0))
        try:
            if oldest.get('engine', 'playht') == 'playht':
                delete_clone_by_id(oldest['id'])
        except Exception as e:
            logging.error(f"Error deleting clone via API: {e}")
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
        """
        Load clones from disk and upgrade legacy entries (missing 'engine').
        We try to determine the correct engine by:
          1) Checking against current PlayHT clone IDs from API,
          2) Heuristics on the ID format as a fallback.
        This allows us to remember the last ElevenLabs clone on first load.
        """
        try:
            with open(self.storage_file, 'r') as file:
                clones_list = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Error loading clones: {e}")
            return
        # Build PlayHT id set (best-effort)
        playht_ids = set()
        try:
            playht_ids = {v['id'] for v in fetch_cloned_voices()}
        except Exception as e:
            logging.warning(f"Could not fetch PlayHT voices to upgrade legacy clones: {e}")
        def _guess_engine(vid: str) -> str:
            # If we know it's PlayHT, trust that.
            if vid in playht_ids:
                return 'playht'
            # Heuristic: ElevenLabs ids are often short (≈20-24) alphanumerics without dashes.
            # PlayHT ids are commonly UUID-like or longer/random; may contain dashes.
            if '-' in vid:
                return 'playht'
            if 18 <= len(vid) <= 24 and vid.isalnum():
                return 'elevenlabs'
            # Default to playht if uncertain
            return 'playht'
        upgraded = []
        upgraded_count = 0
        for c in clones_list:
            if 'engine' not in c or not c.get('engine'):
                c['engine'] = _guess_engine(c.get('id', ''))
                upgraded_count += 1
            upgraded.append(c)
        if upgraded_count:
            logging.info(f"Upgraded {upgraded_count} legacy clone entries with inferred engines")
        # Extend deque preserving order from disk (oldest→newest)
        self.clones.extend(upgraded)

    def get_clone_info(self):
        """Get information about all stored clones."""
        return list(self.clones)

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
        
voice_clone_manager = VoiceCloneManager()

def delete_clone_by_id(voice_id):
    url = "https://api.play.ht/api/v2/cloned-voices/"
    payload = { "voice_id": voice_id }
    headers = {
        "AUTHORIZATION": api_key_manager.get_current_key(),
        "X-USER-ID": api_key_manager.get_current_user_id(),
        "accept": "application/json",
        "content-type": "application/json"
    }

    try:
        response = requests.delete(url, json=payload, headers=headers, timeout=10)  # Timeout set to 10 seconds
        response.raise_for_status()  # Check if the request was successful
        print(response.text)
    except requests.exceptions.Timeout:
        logging.error("The request timed out")
    except requests.exceptions.RequestException as e:
        # Handle other request exceptions (e.g., HTTPError, ConnectionError, etc.)
        logging.error(f"Error in deleting clone: {e}")


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

    try:
        while loop:
        
            # Step 1: Record audio with voice activity detection
            arduino.led_recording()
            print("Recording... Speak into the microphone.")
            recorder = AudioRecorderWithVAD()
            audio_path = recorder.record_with_vad()



            if arduino.onHold():
                break

            #Check if any audio is returned
            if audio_path is None:
                afk = ["Are you still there?","Hello?","Anyone there?","Can you hear me?"]
                tts_speak(afk[1])
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
    #filmPrint()
    while(True):
        arduino = ArduinoControl(arduinoLocation)

        try:
            while(True):
    

                test_full_interaction_cycle()

        except errorCatching as e:
            print(e)

 