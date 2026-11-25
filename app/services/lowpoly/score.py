"""Procedural cinematic score (royalty-free, generated per video).

The music follows an "energy" curve derived from the storyboard: a dark drone
and pads for quiet tension, a pulsing bass + ticking hats when things move,
kicks and arpeggios for chases and the climax, a major-key lift for the
resolution, and big "braam" hits when the monster appears.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass

import numpy as np

SR = 44100
BPM = 100.0
BEAT = 60.0 / BPM

# chord roots (MIDI) and qualities
MINOR_PROGRESSION = [(57, "m"), (53, "M"), (48, "M"), (55, "M")]  # Am F C G
MAJOR_PROGRESSION = [(48, "M"), (55, "M"), (57, "m"), (53, "M")]  # C G Am F


def midi_hz(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12.0)


@dataclass
class Section:
    start: float
    end: float
    energy: float  # 0..1
    major: bool = False


def _chord(root: int, quality: str) -> list[int]:
    third = 4 if quality == "M" else 3
    return [root, root + third, root + 7]


def _add(buf: np.ndarray, start: float, sig: np.ndarray, pan: float = 0.0) -> None:
    s = int(start * SR)
    if s >= len(buf) or len(sig) == 0:
        return
    e = min(len(buf), s + len(sig))
    left, right = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    buf[s:e, 0] += sig[: e - s] * left
    buf[s:e, 1] += sig[: e - s] * right


def _saw(freq: float, n: int, harmonics: int = 7, detune: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    out = np.zeros(n)
    for h in range(1, harmonics + 1):
        out += np.sin(2 * np.pi * freq * (1 + detune) * h * t) / h
    return out


def _env(n: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(n)
    a, r = min(int(attack * SR), n), min(int(release * SR), n)
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] *= np.linspace(1, 0, r)
    return env


class Score:
    def __init__(self, duration: float, sections: list[Section], impacts: list[float], seed: int = 0):
        self.duration = duration
        self.sections = sections
        self.impacts = impacts
        self.rng = np.random.default_rng(seed)
        self.n = int(duration * SR) + SR
        grid = np.arange(0, duration + 1, 0.05)
        raw = np.array([self._raw_energy(t) for t in grid])
        kernel = np.ones(30) / 30  # 1.5 s smoothing
        self._grid = grid
        self._energy = np.convolve(np.pad(raw, 15, mode="edge"), kernel, mode="same")[15:-15]

    def _raw_energy(self, t: float) -> float:
        for s in self.sections:
            if s.start <= t < s.end:
                return s.energy
        return self.sections[-1].energy if self.sections else 0.3

    def energy(self, t: float) -> float:
        return float(np.interp(t, self._grid, self._energy))

    def major(self, t: float) -> bool:
        for s in self.sections:
            if s.start <= t < s.end:
                return s.major
        return False

    # ------------------------------------------------------------------ #
    def render(self) -> np.ndarray:
        buf = np.zeros((self.n, 2), dtype=np.float64)
        bar = BEAT * 4
        chord_len = bar * 2
        t = 0.0
        idx = 0
        while t < self.duration:
            prog = MAJOR_PROGRESSION if self.major(t + 0.1) else MINOR_PROGRESSION
            root, quality = prog[idx % len(prog)]
            e = self.energy(t)
            self._pad(buf, t, chord_len, root, quality, e)
            self._drone(buf, t, chord_len, root, e)
            self._bass(buf, t, chord_len, root, e)
            self._arp(buf, t, chord_len, root, quality)
            self._drums(buf, t, chord_len)
            t += chord_len
            idx += 1
        for hit in self.impacts:
            self._riser(buf, hit)
            self._braam(buf, hit)
        buf = np.tanh(buf * 1.2) / np.tanh(1.2)
        peak = np.max(np.abs(buf)) + 1e-9
        return (buf / peak * 0.9).astype(np.float32)

    def _pad(self, buf, t, length, root, quality, e):
        n = int((length + 1.2) * SR)
        env = _env(n, 1.0, 1.4)
        for i, note in enumerate(_chord(root, quality) + [root + 12]):
            f = midi_hz(note - 12 if i == 0 else note)
            sig = (_saw(f, n, 5, -0.003) + _saw(f, n, 5, 0.003)) * 0.5 * env
            _add(buf, t, sig * (0.035 + 0.02 * e), pan=(-0.5, 0.5, -0.2, 0.2)[i])

    def _drone(self, buf, t, length, root, e):
        n = int((length + 1.0) * SR)
        tt = np.arange(n) / SR
        f = midi_hz(root - 24)
        sig = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * f * 2.01 * tt)) * _env(n, 0.8, 1.0)
        sig *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.25 * tt)
        _add(buf, t, sig * 0.07)

    def _bass(self, buf, t, length, root, e0):
        step = BEAT / 2
        k = 0
        while k * step < length:
            tt = t + k * step
            e = self.energy(tt)
            if e > 0.3:
                n = int(step * 1.1 * SR)
                x = np.arange(n) / SR
                f = midi_hz(root - 24 + (12 if k % 4 == 3 else 0))
                sig = (np.sin(2 * np.pi * f * x) + 0.35 * np.sin(4 * np.pi * f * x)) * np.exp(-x * 9)
                _add(buf, tt, sig * 0.22 * min(1.0, (e - 0.3) * 2.2))
            k += 1

    def _arp(self, buf, t, length, root, quality):
        step = BEAT / 4
        notes = _chord(root + 12, quality) + [root + 24]
        pattern = [0, 1, 2, 3, 2, 1, 2, 3]
        k = 0
        while k * step < length:
            tt = t + k * step
            e = self.energy(tt)
            if e > 0.55:
                n = int(0.25 * SR)
                x = np.arange(n) / SR
                f = midi_hz(notes[pattern[k % len(pattern)]])
                sig = (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(6 * np.pi * f * x)) * np.exp(-x * 14)
                _add(buf, tt, sig * 0.05 * min(1.0, (e - 0.55) * 3), pan=0.35 if k % 2 else -0.35)
            k += 1

    def _drums(self, buf, t, length):
        beats = int(round(length / BEAT))
        for b in range(beats):
            tt = t + b * BEAT
            e = self.energy(tt)
            if e > 0.62 or (e > 0.45 and b % 2 == 0):
                n = int(0.4 * SR)
                x = np.arange(n) / SR
                freq = 45 + 80 * np.exp(-x * 30)
                kick = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-x * 7)
                _add(buf, tt, kick * 0.5 * min(1.0, e + 0.2))
            if e > 0.75 and b % 4 == 2:
                n = int(0.25 * SR)
                x = np.arange(n) / SR
                snare = self.rng.standard_normal(n) * np.exp(-x * 18) * 0.5 + np.sin(2 * np.pi * 190 * x) * np.exp(-x * 25)
                _add(buf, tt, snare * 0.16)
            for half in (0, 1):
                th = tt + half * BEAT / 2
                if self.energy(th) > 0.35:
                    n = int(0.05 * SR)
                    x = np.arange(n) / SR
                    noise = np.diff(self.rng.standard_normal(n + 1))  # crude high-pass
                    _add(buf, th, noise * np.exp(-x * 90) * 0.035, pan=0.4)

    def _riser(self, buf, hit):
        length = 2.2
        start = hit - length
        if start < 0:
            return
        n = int(length * SR)
        x = np.linspace(0, 1, n)
        noise = np.diff(self.rng.standard_normal(n + 1))
        tone = np.sin(2 * np.pi * np.cumsum(200 + 900 * x ** 2) / SR)
        _add(buf, start, (noise * 0.05 + tone * 0.03) * x ** 2.5)

    def _braam(self, buf, hit):
        n = int(3.2 * SR)
        x = np.arange(n) / SR
        env = np.minimum(x / 0.03, 1.0) * np.exp(-x * 1.1)
        sig = np.zeros(n)
        for note, det in ((33, -0.004), (33, 0.004), (45, 0.0), (40, 0.002)):
            sig += _saw(midi_hz(note), n, 9, det)
        boom = np.sin(2 * np.pi * np.cumsum(30 + 40 * np.exp(-x * 6)) / SR) * np.exp(-x * 1.6)
        _add(buf, hit, sig * env * 0.07 + boom * 0.5)


ACTION_ENERGY = {
    "chase": 0.85, "panic": 1.0, "monster": 0.9, "convoy": 0.6, "arrive": 0.5, "drive": 0.55, "bus": 0.5,
    "run": 0.8, "collapse": 0.85, "breakdown": 0.7, "standoff": 0.38, "push": 0.5, "friends": 0.7,
    "celebrate": 0.7, "crowd": 0.45, "argue": 0.55, "alone": 0.25, "walk": 0.4, "wave": 0.4,
    "build": 0.5, "landscape": 0.35, "house_leaves": 0.5, "faceoff": 0.4,
}


def sections_for(scenes) -> list[Section]:
    """scenes: PlannedScene list from the director."""
    out = []
    for ps in scenes:
        e = ACTION_ENERGY.get(ps.scene.action, 0.45)
        if ps.scene.mood == "storm":
            e = max(e, 0.75) if ps.scene.action != "standoff" else e
        major = ps.scene.mood in ("dawn", "sunset") and ps.scene.action not in ("monster", "panic", "chase")
        out.append(Section(ps.start, ps.end, e, major))
    return out


def write_score(path: str, duration: float, scenes, impacts: list[float], seed: int = 0) -> str:
    audio = Score(duration, sections_for(scenes), impacts, seed).render()
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return path
