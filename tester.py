#!/usr/bin/env python3
import os, sys, time, argparse, requests, io
import numpy as np

try:
    import pyaudio
except Exception as e:
    print(f"[FATAL] PyAudio import failed: {e}\nInstall with: pip install pyaudio", file=sys.stderr)
    sys.exit(1)

try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False

API_URL_TMPL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

def find_output_device(p, name_substr="BlackHole"):
    idx = None
    info_m = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if name_substr.lower() in str(info.get('name','')).lower():
            idx = i
            info_m = info
            break
    if idx is None:
        raise RuntimeError(f"Output device containing '{name_substr}' not found")
    return idx, info_m

def upmix_mono_to_stereo(pcm_bytes: bytes) -> bytes:
    """Duplicate 16-bit mono samples into stereo without numpy copies where possible."""
    # Use numpy for clarity and speed
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    if arr.size == 0:
        return b""
    stereo = np.column_stack((arr, arr)).astype(np.int16).ravel()
    return stereo.tobytes()

def strip_wav_header_if_present(buf: bytearray):
    """
    If buffer starts with RIFF/WAVE, strip the header up to the start of the 'data' chunk.
    Returns (stripped_bytes, header_parsed_bool).
    Minimal parser: looks for 'RIFF....WAVE' and first 'data' occurrence.
    """
    if len(buf) < 32:
        return b"", False
    if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        # no header, likely raw PCM
        return bytes(buf), True
    # find 'data' chunk after 'WAVE'
    i = buf.find(b"data", 12)
    if i == -1 or i + 8 > len(buf):
        return b"", False  # header not fully received yet
    data_len = int.from_bytes(buf[i+4:i+8], "little", signed=False)
    # data begins at i+8
    payload = bytes(buf[i+8:])
    # clear consumed header bytes from the buffer
    del buf[:i+8]
    return payload, True

def decode_mp3_bytes_to_pcm(mp3_bytes: bytes, target_rate: int, target_chans: int) -> bytes:
    """Decode MP3 -> 16-bit PCM at target_rate/target_chans using pydub if available."""
    if not _HAS_PYDUB:
        raise RuntimeError("MP3 stream received and pydub is not installed. Install with: pip install pydub (requires ffmpeg)")
    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    seg = seg.set_frame_rate(int(target_rate)).set_channels(int(target_chans)).set_sample_width(2)
    return seg.raw_data

def stream_elevenlabs_to_blackhole(voice_id: str, text: str, model_id: str = "eleven_v3",
                                   device_hint: str = "BlackHole",
                                   device_rate: int = 44100,
                                   frames_per_buffer: int = 1024,
                                   jitter_ms: int = 150):
    api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY") or "sk_1d4288e8ac2e9faf8bb17e6b8f7f9ab5a0c4e562239b13e7"
    if not api_key:
        raise RuntimeError("Set ELEVEN_API_KEY (or ELEVENLABS_API_KEY) in env")

    url = API_URL_TMPL.format(voice_id=voice_id)
    requested_of = "pcm_44100"
    url = f"{url}?output_format={requested_of}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/wav",  # we’ll inspect Content-Type anyway
        "content-type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        # This is the key bit: ask for linear PCM at 44.1k
        "output_format": requested_of,
        # keep it minimal; omit voice_settings for now
    }

    # Open HTTP stream
    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    if resp.status_code != 200:
        ct = resp.headers.get("Content-Type", "?")
        body = (resp.text or "")[:400]
        raise RuntimeError(f"HTTP {resp.status_code} from ElevenLabs (CT={ct}): {body}")

    ct = resp.headers.get("Content-Type", "").lower()
    is_mp3 = ("mpeg" in ct) or ("mp3" in ct)

    # Setup PyAudio at device-native stereo if possible
    p = pyaudio.PyAudio()
    try:
        dev_idx, dev_info = find_output_device(p, name_substr=device_hint)
        chans = 2 if int(dev_info.get("maxOutputChannels") or 1) >= 2 else 1
        rate = int(dev_info.get("defaultSampleRate") or device_rate)
        if rate != device_rate:
            # We asked for 44100—prefer opening at 44100 (Ableton usually set to that).
            # If device reports a different default, we still open at 44100 here.
            rate = device_rate

        stream = p.open(format=pyaudio.paInt16,
                        channels=chans,
                        rate=rate,
                        output=True,
                        output_device_index=dev_idx,
                        frames_per_buffer=frames_per_buffer)

        if is_mp3:
            # Fallback: accumulate MP3 and decode to PCM, then play back
            mp3_buf = bytearray()
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    mp3_buf.extend(chunk)
            try:
                pcm = decode_mp3_bytes_to_pcm(bytes(mp3_buf), rate, chans)
            except Exception as e:
                raise RuntimeError(f"Received MP3 stream and failed to decode locally: {e}")

            # Write decoded PCM in frames_per_buffer-sized chunks
            step = frames_per_buffer * chans * 2
            for i in range(0, len(pcm), step):
                stream.write(pcm[i:i+step], exception_on_underflow=False)

            try:
                stream.stop_stream()
            finally:
                stream.close()
            return

        # --- Streaming read → small jitter buffer → write ---
        header_parsed = False
        tail = b""
        jitter_target = int(rate * (2 if chans==2 else 1) * 2 * (jitter_ms/1000.0))  # bytes
        jb = bytearray()

        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if not header_parsed:
                jb.extend(chunk)
                # Try to strip WAV header if present; otherwise accept raw bytes
                data, header_parsed = strip_wav_header_if_present(jb)
                if not header_parsed:
                    # Not enough header yet; continue accumulating
                    continue
                # If strip_wav_header_if_present returned data==jb (raw), we already have payload in jb
                if data:
                    jb = bytearray(data)
                # else jb already holds raw bytes post-header
                # Fill jitter buffer up to target before first write
                if len(jb) < jitter_target:
                    continue

            else:
                jb.extend(chunk)

            # Maintain 16-bit alignment between writes
            buf = tail + jb
            if len(buf) & 1:
                tail = buf[-1:]
                buf = buf[:-1]
            else:
                tail = b""
            jb.clear()

            if not buf:
                continue

            # ElevenLabs PCM is typically mono. If device is stereo, up-mix.
            if chans == 2:
                out = upmix_mono_to_stereo(buf)
            else:
                out = buf

            # Write; disable underflow exceptions so small network hiccups don’t crash
            stream.write(out, exception_on_underflow=False)

        # Flush any final bytes in tail (ignore if odd)
        # (If you *really* want to flush last odd byte, you’d need to buffer until next stream.)
        try:
            stream.stop_stream()
        finally:
            stream.close()

    finally:
        try:
            p.terminate()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser(description="Minimal ElevenLabs HTTP stream → BlackHole tester (PCM 44.1k).")
    ap.add_argument("--voice-id", required=True, help="ElevenLabs voice_id to use")
    ap.add_argument("--text", default="This is a clean streaming test into Ableton via BlackHole.",
                    help="Text to synthesize")
    ap.add_argument("--model-id", default="eleven_v3", help="Model id (e.g., eleven_v3)")
    ap.add_argument("--device", default="BlackHole", help="Substring of output device name (default: BlackHole)")
    ap.add_argument("--rate", type=int, default=44100, help="Playback sample rate (default 44100)")
    ap.add_argument("--buffer", type=int, default=1024, help="PyAudio frames_per_buffer (default 1024)")
    ap.add_argument("--jitter-ms", type=int, default=150, help="Initial jitter buffer in ms (default 150)")
    ap.add_argument("--output-format", default="pcm_44100", help="Requested ElevenLabs output_format (default pcm_44100)")
    args = ap.parse_args()

    stream_elevenlabs_to_blackhole(
        voice_id=args.voice_id,
        text=args.text,
        model_id=args.model_id,
        device_hint=args.device,
        device_rate=args.rate,
        frames_per_buffer=args.buffer,
        jitter_ms=args.jitter_ms,
    )
    print("Done.")

if __name__ == "__main__":
    main()