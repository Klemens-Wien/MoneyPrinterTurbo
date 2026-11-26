"""Audio mix: narrator voice + ducked background music + weather ambience
(rain, wind, thunder synced to lightning flashes) + whooshes on cuts."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .narration import SAMPLE_RATE, read_wav

SR = SAMPLE_RATE
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SONG_DIR = os.path.join(ROOT, "resource", "songs")


def decode_audio(path: str, duration: float | None = None) -> np.ndarray:
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-loglevel", "error", "-i", path]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def band(x: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Zero-phase FFT band-pass with soft edges (works on mono or stereo)."""
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    gain = np.ones_like(freqs)
    if lo:
        gain *= 1.0 / (1.0 + (lo / np.maximum(freqs, 1e-3)) ** 4)
    if hi:
        gain *= 1.0 / (1.0 + (freqs / hi) ** 4)
    if x.ndim == 2:
        gain = gain[:, None]
    return np.fft.irfft(spec * gain, n=len(x), axis=0).astype(np.float32)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, window)
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = (c[window:] - c[:-window]) / window
    pad = len(x) - len(out)
    return np.concatenate([np.full(pad // 2, out[0] if len(out) else 0), out, np.full(pad - pad // 2, out[-1] if len(out) else 0)])


def _norm(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-9
    return x / peak


def thunder(rng: np.random.Generator, seconds: float = 4.5) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rumble = band(rng.standard_normal((n, 2)).astype(np.float32), 25, 160)
    rumble *= (np.exp(-t / 1.6) * (0.6 + 0.4 * np.sin(t * 7 + rng.uniform(0, 6)) ** 2))[:, None]
    crack = band(rng.standard_normal((n, 2)).astype(np.float32), 300, 3000) * np.exp(-t / 0.12)[:, None]
    attack = np.minimum(t / 0.03, 1.0)[:, None]
    return _norm(rumble * 1.0 + crack * 0.35) * attack


def stomp(rng: np.random.Generator, seconds: float = 1.2) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(32 + 45 * np.exp(-t * 14)) / SR) * np.exp(-t * 4.5)
    debris = band(rng.standard_normal((n, 2)).astype(np.float32), 60, 700) * np.exp(-t * 9)[:, None]
    x = boom[:, None] + _norm(debris) * 0.25
    return _norm(x)


def foley(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Small synthesized sound effects (stereo)."""
    if kind == "slam":
        n = int(0.7 * SR)
        t = np.arange(n) / SR
        knock = np.sin(2 * np.pi * np.cumsum(70 + 90 * np.exp(-t * 30)) / SR) * np.exp(-t * 9)
        crack = band(rng.standard_normal(n).astype(np.float32), 300, 2500) * np.exp(-t * 35)
        x = knock + _norm(crack) * 0.4
        pan = rng.uniform(-0.8, 0.8)
        return (_norm(x)[:, None] * np.array([1 - max(pan, 0), 1 + min(pan, 0)]) * 0.34).astype(np.float32)
    if kind == "step":
        n = int(0.45 * SR)
        t = np.arange(n) / SR
        thud = np.sin(2 * np.pi * np.cumsum(55 + 40 * np.exp(-t * 25)) / SR) * np.exp(-t * 12)
        wood = band(rng.standard_normal(n).astype(np.float32), 150, 900) * np.exp(-t * 30)
        return ((_norm(thud) + _norm(wood) * 0.3)[:, None] * 0.3).astype(np.float32)
    if kind == "creak":
        n = int(1.6 * SR)
        t = np.arange(n) / SR
        f = 180 + 120 * np.sin(t * 2.3) + 40 * np.sin(t * 17)
        tone = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.5
        grain = (rng.random(n) < 0.04).astype(float) * rng.standard_normal(n)
        x = band((tone + grain).astype(np.float32), 120, 2200) * np.sin(np.pi * np.minimum(t / 1.6, 1)) ** 0.5
        return (_norm(x)[:, None] * 0.16).astype(np.float32)
    if kind == "breath":
        n = int(2.6 * SR)
        t = np.arange(n) / SR
        env = np.sin(np.pi * t / 1.3) ** 2 * (t < 1.3) + np.sin(np.pi * (t - 1.3) / 1.3) ** 2 * (t >= 1.3) * 0.8
        x = band(rng.standard_normal(n).astype(np.float32), 80, 900) * env
        growl = np.sin(2 * np.pi * 48 * t) * env * 0.4
        return ((_norm(x) + growl)[:, None] * 0.22).astype(np.float32)
    if kind == "whisper":
        n = int(1.8 * SR)
        t = np.arange(n) / SR
        syll = np.clip(np.sin(2 * np.pi * 3.2 * t), 0, 1) ** 2 * np.sin(np.pi * t / 1.8)
        x = band(rng.standard_normal((n, 2)).astype(np.float32), 1500, 7000) * syll[:, None]
        return (_norm(x) * 0.12).astype(np.float32)
    if kind == "zap":  # being "switched off": electric buzz + power-down
        n = int(0.9 * SR)
        t = np.arange(n) / SR
        f = 120 * np.exp(-t * 2.5) + 40
        buzz = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t * 3.5)
        crackle = band(rng.standard_normal(n).astype(np.float32), 2000, 9000) * np.exp(-t * 9)
        x = buzz * 0.5 + _norm(crackle) * 0.5
        return (_norm(x)[:, None] * 0.3).astype(np.float32)
    if kind == "heartbeat":
        n = int(0.6 * SR)
        t = np.arange(n) / SR
        beat = lambda d: np.sin(2 * np.pi * 50 * np.maximum(t - d, 0)) * np.exp(-np.maximum(t - d, 0) * 18) * (t >= d)  # noqa: E731
        x = beat(0.0) + 0.7 * beat(0.22)
        return (x[:, None] * 0.28).astype(np.float32)
    return np.zeros((1, 2), dtype=np.float32)


def whoosh(rng: np.random.Generator, seconds: float = 0.45) -> np.ndarray:
    n = int(seconds * SR)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** 2
    x = band(rng.standard_normal((n, 2)).astype(np.float32), 400, 5000)
    return _norm(x * env[:, None])


@dataclass
class VoicePlacement:
    path: str
    start: float


def pick_song(seed: int) -> str | None:
    songs = sorted(glob.glob(os.path.join(SONG_DIR, "*.mp3")))
    if not songs:
        return None
    return songs[seed % len(songs)]


def mix(
    out_path: str,
    duration: float,
    voices: list[VoicePlacement],
    storm_spans: list[tuple[float, float]],
    flashes: list[float],
    cuts: list[float],
    music: str | None,
    seed: int = 0,
    music_volume: float = 0.30,
    thuds: list[float] | None = None,
    ticks: list[float] | None = None,
    dings: list[float] | None = None,
    sfx: list[tuple[float, str]] | None = None,
) -> str:
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    out = np.zeros((n, 2), dtype=np.float32)

    # voice
    voice = np.zeros(n, dtype=np.float32)
    for v in voices:
        data = read_wav(v.path)
        s = int(v.start * SR)
        e = min(n, s + len(data))
        if e > s:
            voice[s:e] += data[: e - s]
    out += voice[:, None] * 0.95

    # music with ducking under the voice
    if music and os.path.exists(music):
        try:
            track = decode_audio(music)
            reps = int(np.ceil(n / max(len(track), 1)))
            track = np.tile(track, (reps, 1))[:n]
            track = track / (np.max(np.abs(track)) + 1e-9)
            env = _smooth(np.abs(voice), int(0.3 * SR))
            duck = 1.0 - 0.55 * np.clip(env / 0.04, 0, 1)
            fade = np.ones(n)
            fi, fo = int(0.4 * SR), int(2.5 * SR)
            fade[:fi] = np.linspace(0, 1, fi)
            fade[-fo:] = np.linspace(1, 0, fo)
            out += track * (music_volume * duck * fade)[:, None]
        except Exception as exc:
            logger.warning(f"music could not be mixed: {exc}")

    # storm ambience
    if storm_spans:
        weather_env = np.zeros(n, dtype=np.float32)
        ramp = int(0.8 * SR)
        for a, b in storm_spans:
            s, e = int(a * SR), min(n, int(b * SR))
            weather_env[s:e] = 1.0
        weather_env = _smooth(weather_env, ramp).astype(np.float32)
        noise = rng.standard_normal((n, 2)).astype(np.float32)
        rain = _norm(band(noise, 900, 9000)) * 0.11
        wind_lfo = (0.55 + 0.45 * np.sin(np.arange(n) / SR * 0.7 + 1.0) * np.sin(np.arange(n) / SR * 0.23))[:, None]
        wind = _norm(band(noise[:, ::-1].copy(), 60, 450)) * 0.16 * wind_lfo
        out += (rain + wind) * weather_env[:, None]
        for t in flashes:
            clap = thunder(rng) * 0.5
            s = int((t + rng.uniform(0.15, 0.9)) * SR)
            e = min(n, s + len(clap))
            if e > s:
                out[s:e] += clap[: e - s]

    # monster footsteps
    for t in thuds or []:
        st = stomp(rng) * 0.32
        s = int(t * SR)
        e = min(n, s + len(st))
        if 0 <= s < e:
            out[s:e] += st[: e - s]

    # wheel ticks and the landing "ding"
    for t in ticks or []:
        nn = int(0.03 * SR)
        x = np.arange(nn) / SR
        click = (np.sin(2 * np.pi * 2400 * x) * 0.6 + rng.standard_normal(nn) * 0.4) * np.exp(-x * 220)
        s = int(t * SR)
        if 0 <= s < n:
            e = min(n, s + nn)
            out[s:e] += (click[: e - s] * 0.22)[:, None]
    for t in dings or []:
        nn = int(1.6 * SR)
        x = np.arange(nn) / SR
        bell = sum(np.sin(2 * np.pi * f * x) * a for f, a in ((880, 0.5), (1320, 0.35), (1760, 0.25), (2640, 0.1)))
        bell *= np.exp(-x * 2.8)
        boom = np.sin(2 * np.pi * np.cumsum(40 + 60 * np.exp(-x * 8)) / SR) * np.exp(-x * 2.5)
        s = int(t * SR)
        if 0 <= s < n:
            e = min(n, s + nn)
            out[s:e] += ((bell * 0.3 + boom * 0.45)[: e - s])[:, None]

    # horror foley: door slams, creaks, footsteps, breathing, whispers, heartbeat
    for t, kind in sfx or []:
        clip = foley(kind, rng)
        s = int(t * SR)
        if 0 <= s < n:
            e = min(n, s + len(clip))
            out[s:e] += clip[: e - s]

    # whooshes on cuts
    for t in cuts:
        w = whoosh(rng) * 0.07
        s = max(0, int((t - 0.22) * SR))
        e = min(n, s + len(w))
        out[s:e] += w[: e - s]

    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.98:
        out *= 0.98 / peak
    pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return out_path
