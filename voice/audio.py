"""Audio helpers: resampling and synthetic ward conditions.

Every noise source is generated or taken from other synthetic clips, so no outside sound
dataset is needed. Audio is float32 mono in [-1, 1].
"""

from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

STT_RATE = 16000


def resample(audio: np.ndarray, rate: int, target: int = STT_RATE) -> np.ndarray:
    if rate == target:
        return audio.astype(np.float32)
    g = gcd(rate, target)
    return resample_poly(audio, target // g, rate // g).astype(np.float32)


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    top = float(np.max(np.abs(audio))) or 1.0
    return (audio * (peak / top)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) or 1e-9


def _fit(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Loops or crops a noise clip to length n, from a random offset."""
    if len(noise) < n:
        noise = np.tile(noise, n // len(noise) + 1)
    start = rng.integers(0, len(noise) - n + 1)
    return noise[start:start + n]


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = _fit(noise, len(speech), rng)
    scale = _rms(speech) / (_rms(noise) * 10 ** (snr_db / 20))
    return speech + noise * scale


def room_hum(n: int, rng: np.random.Generator, rate: int = STT_RATE) -> np.ndarray:
    """Ventilation-like low noise: brown-ish noise plus 50 Hz mains hum."""
    brown = np.cumsum(rng.normal(size=n)).astype(np.float32)
    brown -= np.convolve(brown, np.ones(400) / 400, mode="same")  # drop the drift
    t = np.arange(n) / rate
    return brown / _rms(brown) + 0.3 * np.sin(2 * np.pi * 50 * t)


def monitor_beeps(n: int, rng: np.random.Generator, rate: int = STT_RATE) -> np.ndarray:
    """Patient-monitor and pump alarms: short tone bursts at a steady rhythm."""
    out = np.zeros(n, dtype=np.float32)
    freq = rng.choice([880.0, 1000.0, 1200.0])
    period, length = int(rate * rng.uniform(0.6, 1.2)), int(rate * 0.12)
    t = np.arange(length) / rate
    burst = np.sin(2 * np.pi * freq * t) * np.hanning(length)
    for start in range(int(rng.integers(0, period)), n - length, period):
        out[start:start + length] += burst
    return out


def reverb(audio: np.ndarray, rng: np.random.Generator, rate: int = STT_RATE) -> np.ndarray:
    """A hard-walled hospital room: exponentially decaying noise as the impulse response."""
    seconds = rng.uniform(0.25, 0.6)
    n = int(rate * seconds)
    impulse = rng.normal(size=n) * np.exp(-6.9 * np.arange(n) / n)  # -60 dB at the end
    impulse[0] = 1.0
    wet = np.convolve(audio, impulse)[:len(audio)]
    return 0.6 * audio + 0.4 * wet / (_rms(wet) / _rms(audio))


def phone_band(audio: np.ndarray, rate: int = STT_RATE) -> np.ndarray:
    """A cheap bedside microphone: 300-3400 Hz, like a phone line."""
    sos = butter(4, [300, 3400], btype="band", fs=rate, output="sos")
    return sosfilt(sos, audio)


CONDITIONS = ("clean", "ward", "hard")


def apply_condition(speech: np.ndarray, condition: str, babble: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """clean: as synthesized. ward: chatter, beeps and hum at moderate levels plus room echo.
    hard: the same much louder, through a poor microphone, spoken quietly."""
    if condition == "clean":
        return normalize(speech)
    n = len(speech)
    if condition == "ward":
        x = reverb(speech, rng)
        x = mix(x, babble, snr_db=15, rng=rng)
        x = mix(x, monitor_beeps(n, rng), snr_db=20, rng=rng)
        x = mix(x, room_hum(n, rng), snr_db=25, rng=rng)
        return normalize(x)
    if condition == "hard":
        x = reverb(speech * 0.4, rng)
        x = mix(x, babble, snr_db=6, rng=rng)
        x = mix(x, monitor_beeps(n, rng), snr_db=10, rng=rng)
        x = mix(x, room_hum(n, rng), snr_db=12, rng=rng)
        return normalize(phone_band(x), peak=0.5)
    raise ValueError(f"unknown condition {condition!r}")
