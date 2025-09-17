import numpy as np, pyaudio

SR = 44100
CH = 2
CHUNK = 1024          # try 512–1024 while testing
SECS = 5
F = 440.0

p = pyaudio.PyAudio()

# find BlackHole
idx = None
for i in range(p.get_device_count()):
    if 'BlackHole' in p.get_device_info_by_index(i)['name']:
        idx = i; break
assert idx is not None, "BlackHole device not found"

stream = p.open(format=pyaudio.paInt16, channels=CH, rate=SR,
                output=True, output_device_index=idx, frames_per_buffer=CHUNK)

omega = 2*np.pi*F / SR
phase = 0.0

frames = int(SR * SECS / CHUNK)
for _ in range(frames):
    t = phase + omega * np.arange(CHUNK)
    mono = 0.2 * np.sin(t)                         # float
    phase = (phase + omega * CHUNK) % (2*np.pi)    # keep continuity
    stereo = np.column_stack([mono, mono]).ravel()
    out = np.clip(stereo * 32767, -32768, 32767).astype(np.int16).tobytes()
    stream.write(out, exception_on_underflow=False)

stream.stop_stream(); stream.close(); p.terminate()