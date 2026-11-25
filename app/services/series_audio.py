"""Synthesized, license-free background music and sound effects for series episodes.

Everything here is generated from noise and oscillators, so the result carries
no third-party rights and needs no network access. All functions return mono
float32 arrays at ``SAMPLE_RATE``.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100


def _t(duration: float) -> np.ndarray:
    return np.arange(int(duration * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _lowpass(signal: np.ndarray, cutoff: float) -> np.ndarray:
    """One-pole low-pass filter (vectorised via FFT for speed)."""
    if signal.size == 0:
        return signal
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / SAMPLE_RATE)
    response = 1.0 / np.sqrt(1.0 + (freqs / cutoff) ** 4)
    return np.fft.irfft(spectrum * response, n=signal.size).astype(np.float32)


def _bandpass(signal: np.ndarray, low: float, high: float) -> np.ndarray:
    if signal.size == 0:
        return signal
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / SAMPLE_RATE)
    response = 1.0 / np.sqrt(1.0 + (freqs / high) ** 4)
    response *= 1.0 / np.sqrt(1.0 + (low / np.maximum(freqs, 1e-3)) ** 4)
    return np.fft.irfft(spectrum * response, n=signal.size).astype(np.float32)


def _normalize(signal: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = float(np.max(np.abs(signal))) if signal.size else 0.0
    return signal if m < 1e-9 else (signal / m * peak).astype(np.float32)


def _fade(signal: np.ndarray, fade_in: float = 0.05, fade_out: float = 0.2) -> np.ndarray:
    out = signal.copy()
    n_in = min(int(fade_in * SAMPLE_RATE), out.size)
    n_out = min(int(fade_out * SAMPLE_RATE), out.size)
    if n_in:
        out[:n_in] *= np.linspace(0, 1, n_in, dtype=np.float32)
    if n_out:
        out[-n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32)
    return out


# --------------------------------------------------------------------------
# Music
# --------------------------------------------------------------------------


def dark_drone(duration: float, seed: int = 7) -> np.ndarray:
    """A slow, ominous D-minor drone with sub pulses and wind."""
    t = _t(duration)
    rng = _rng(seed)
    out = np.zeros_like(t)
    # Detuned pad voices: D2, A2, F3, D3
    for freq, amp in ((73.42, 0.5), (110.0, 0.32), (174.61, 0.18), (146.83, 0.22)):
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.03 + rng.random() * 0.04) * t + rng.random() * 6)
        for detune in (-0.35, 0.0, 0.4):
            out += amp * lfo * np.sin(2 * np.pi * (freq + detune) * t + rng.random() * 6) / 3
    # Slow heartbeat-like sub pulse every ~2.4 s
    pulse_period = 2.4
    phase = (t % pulse_period) / pulse_period
    env = np.exp(-phase * 18) + 0.6 * np.exp(-np.clip(phase - 0.12, 0, None) * 22) * (phase > 0.12)
    out += 0.35 * env * np.sin(2 * np.pi * 46 * t)
    # Wind noise
    wind = _lowpass(rng.standard_normal(t.size).astype(np.float32), 500)
    wind *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t) ** 2
    out += 0.25 * _normalize(wind)
    return _fade(_normalize(out, 0.9), 2.0, 3.0)


# --------------------------------------------------------------------------
# Sound effects
# --------------------------------------------------------------------------


def waves(duration: float, seed: int = 1) -> np.ndarray:
    t = _t(duration)
    noise = _rng(seed).standard_normal(t.size).astype(np.float32)
    body = _lowpass(noise, 380)
    swell = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 4.5 - 1.5)) ** 2
    hiss = _bandpass(noise, 1500, 4000) * swell ** 3 * 0.25
    return _fade(_normalize(body * swell + hiss), 0.8, 1.0)


def creak(duration: float = 2.2, seed: int = 2) -> np.ndarray:
    """Groaning ship hull: resonant, slowly gliding tones with friction."""
    t = _t(duration)
    rng = _rng(seed)
    f0 = 70 + 25 * np.sin(2 * np.pi * 0.35 * t) + 15 * t
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    tone = np.sign(np.sin(phase)) * 0.4 + np.sin(2 * phase) * 0.4 + np.sin(3.01 * phase) * 0.2
    grit = rng.standard_normal(t.size).astype(np.float32)
    stick = (np.sin(2 * np.pi * 9 * t + 2 * np.sin(2 * np.pi * 0.7 * t)) > 0.2).astype(np.float32)
    sig = _bandpass(tone * (0.6 + 0.4 * stick) + 0.3 * grit * stick, 90, 1400)
    env = np.sin(np.pi * np.clip(t / duration, 0, 1)) ** 1.5
    return _normalize(sig * env)


def hum(duration: float) -> np.ndarray:
    t = _t(duration)
    sig = np.sin(2 * np.pi * 50 * t) + 0.4 * np.sin(2 * np.pi * 100 * t) + 0.15 * np.sin(2 * np.pi * 150 * t)
    return _fade(_normalize(sig.astype(np.float32)), 0.5, 0.5)


def boom(duration: float = 3.0, seed: int = 3) -> np.ndarray:
    t = _t(duration)
    rng = _rng(seed)
    f = 32 + 40 * np.exp(-t * 6)
    phase = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    body = np.sin(phase) * np.exp(-t * 1.6)
    rumble = _lowpass(rng.standard_normal(t.size).astype(np.float32), 160) * np.exp(-t * 1.1)
    metal = _bandpass(rng.standard_normal(t.size).astype(np.float32), 300, 1200) * np.exp(-t * 4)
    sig = body + 0.8 * _normalize(rumble) + 0.25 * _normalize(metal)
    return _fade(_normalize(sig), 0.005, 0.5)


def alarm(duration: float, far: bool = False) -> np.ndarray:
    """Two-tone ship alarm; ``far`` makes it muffled like behind walls."""
    t = _t(duration)
    period = 0.9
    freq = np.where((t % period) < period / 2, 680.0, 520.0)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    tone = np.clip(np.sin(phase) * 2.2, -1, 1) * 0.7 + 0.3 * np.sin(2 * phase)
    tone = _lowpass(tone.astype(np.float32), 700 if far else 2600)
    return _fade(_normalize(tone), 0.1, 0.6)


def water_rush(duration: float, seed: int = 4) -> np.ndarray:
    t = _t(duration)
    noise = _rng(seed).standard_normal(t.size).astype(np.float32)
    rush = _bandpass(noise, 120, 1800)
    gurgle = np.sin(2 * np.pi * (3 + 2 * np.sin(2 * np.pi * 0.4 * t)) * t) * 0.3 + 0.7
    rise = np.clip(t / max(duration, 0.1), 0, 1) * 0.6 + 0.4
    return _fade(_normalize(rush * gurgle * rise), 0.8, 0.8)


def static(duration: float, seed: int = 5) -> np.ndarray:
    t = _t(duration)
    rng = _rng(seed)
    noise = _bandpass(rng.standard_normal(t.size).astype(np.float32), 800, 5000)
    crackle = (rng.random(t.size) > 0.9985).astype(np.float32) * rng.standard_normal(t.size).astype(np.float32) * 6
    gate = np.ones_like(t)
    gate[int(min(duration * 0.6, duration) * SAMPLE_RATE):] = 0.15  # cuts off mid-sentence
    return _fade(_normalize((noise * 0.5 + crackle) * gate), 0.02, 0.3)


def knock(duration: float = 3.0) -> np.ndarray:
    out = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
    hit_t = _t(0.25)
    hit = (np.sin(2 * np.pi * 95 * hit_t) + 0.5 * np.sin(2 * np.pi * 210 * hit_t)) * np.exp(-hit_t * 28)
    for at in (0.3, 0.62, 0.94, 1.9, 2.2):
        i = int(at * SAMPLE_RATE)
        j = min(out.size, i + hit.size)
        if i < out.size:
            out[i:j] += hit[: j - i]
    return _normalize(_lowpass(out, 900))


def drip(duration: float, seed: int = 6) -> np.ndarray:
    rng = _rng(seed)
    out = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
    dt = _t(0.12)
    at = 0.4
    while at < duration - 0.2:
        f = 900 + rng.random() * 700
        d = np.sin(2 * np.pi * (f + 2500 * dt) * dt) * np.exp(-dt * 45)
        i = int(at * SAMPLE_RATE)
        out[i : i + d.size] += d[: max(0, min(d.size, out.size - i))]
        at += 0.9 + rng.random() * 1.4
    return _normalize(out)


def heartbeat(duration: float) -> np.ndarray:
    out = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
    bt = _t(0.18)
    beat = np.sin(2 * np.pi * 55 * bt) * np.exp(-bt * 30)
    at, interval = 0.2, 0.95
    while at < duration - 0.3:
        for off, amp in ((0.0, 1.0), (0.22, 0.7)):
            i = int((at + off) * SAMPLE_RATE)
            n = max(0, min(beat.size, out.size - i))
            out[i : i + n] += amp * beat[:n]
        at += interval
        interval = max(0.6, interval * 0.95)  # speeds up: rising tension
    return _normalize(out)


def riser(duration: float, seed: int = 8) -> np.ndarray:
    t = _t(duration)
    noise = _rng(seed).standard_normal(t.size).astype(np.float32)
    progress = np.clip(t / max(duration, 0.1), 0, 1)
    swell = _bandpass(noise, 200, 3000) * progress ** 2
    f = 110 + 220 * progress ** 2
    tone = np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE) * progress ** 1.5
    return _fade(_normalize(swell * 0.6 + tone * 0.4), 0.3, 0.4)


# name -> (generator(duration), gain, default length or None=whole scene)
SFX = {
    "waves": (waves, 0.55, None),
    "creak": (creak, 0.5, 2.4),
    "hum": (hum, 0.12, None),
    "boom": (boom, 0.9, 3.2),
    "alarm": (lambda d: alarm(d, far=False), 0.22, None),
    "alarm_far": (lambda d: alarm(d, far=True), 0.18, None),
    "water_rush": (water_rush, 0.45, None),
    "static": (static, 0.35, None),
    "knock": (knock, 0.6, 3.0),
    "drip": (drip, 0.35, None),
    "heartbeat": (heartbeat, 0.55, None),
    "riser": (riser, 0.3, None),
}


def render_sfx(name: str, scene_duration: float) -> tuple[np.ndarray, float]:
    """Return (samples, gain) for one named effect inside a scene."""
    if name not in SFX:
        raise ValueError(f"unknown sound effect: {name!r} (available: {', '.join(SFX)})")
    generator, gain, length = SFX[name]
    duration = length if length is not None else scene_duration
    return generator(max(0.2, duration)).astype(np.float32), gain
