"""Synthesize the Chladni driving tones, in sync with timeline.MODES.
Pure tone at the mode frequency (+ a faint 2nd harmonic for body) and a very
soft sand-rustle that fades as the grains settle on the nodal lines."""
import numpy as np, wave, pathlib
from scipy.signal import butter, sosfilt
from timeline import MODES, DURATION, freq

SR = 48000
N = int(DURATION * SR)
t = np.arange(N) / SR
out = np.zeros(N)
rng = np.random.default_rng(3)
noise = sosfilt(butter(4, [2500, 9000], btype="band", fs=SR, output="sos"), rng.standard_normal(N))


def env(a, b, attack=0.12, release=0.35):
    e = np.clip((t - a) / attack, 0, 1) * np.clip((b - t) / release, 0, 1)
    return e * e * (3 - 2 * e)


for a, b, md in MODES:
    f = freq(md)
    e = env(a, b, release=0.35 if b < 29.9 else 0.6)
    # equal-loudness-ish: higher pitches a bit quieter
    g = 0.30 * (230 / f) ** 0.35
    ph = 2 * np.pi * f * (t - a)
    tone = np.sin(ph) + 0.12 * np.sin(2 * ph) + 0.04 * np.sin(3 * ph)
    out += g * e * tone
    # rustle: strong when the tone kicks in, decays as sand settles (~1.6 s)
    settle = np.exp(-np.clip(t - a, 0, None) / 0.9) * (t >= a) * env(a, b, 0.05, 0.2)
    out += 0.05 * settle * noise / np.std(noise)

out = np.clip(out, -1, 1)
pcm = (out * 0.9 * 32767).astype(np.int16)
stereo = np.repeat(pcm[:, None], 2, axis=1)
dst = pathlib.Path(__file__).parents[1] / "assets" / "audio" / "chladni_tones.wav"
with wave.open(str(dst), "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(stereo.tobytes())
print("wrote", dst)
