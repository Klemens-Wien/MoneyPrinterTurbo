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
    "rampage": 0.95, "street_panic": 0.9, "hide": 0.3,
    "lockdown": 0.55, "watch": 0.5, "bed": 0.18, "sit_up": 0.25, "noises": 0.22, "doorway": 0.4,
    "empty_doorway": 0.2, "closet": 0.35, "doll_desk": 0.2, "doll_floor": 0.3, "doll_bed": 0.35,
    "doll_behind": 0.45, "slum_life": 0.35, "look_up": 0.3, "lift_ride": 0.45, "gate_guards": 0.5,
    "climb": 0.6, "climb_high": 0.8, "sky_view": 0.55, "rich_party": 0.6, "guards_chase": 0.95, "mirrors": 0.45,
    "edge_look": 0.35, "duo": 0.4,
    "bells": 0.7, "krampus_lane": 0.6, "krampus_pack": 0.85, "krampus_window": 0.45, "krampus_chase": 0.95,
    "krampus_door": 0.4, "krampus_leave": 0.3,
    "freeze_one": 0.6, "speak_freeze": 0.7,
    "lr_goodbye": 0.25, "lr_couch": 0.2, "lr_upstairs": 0.4, "lr_hold": 0.5, "lr_phone": 0.45, "lr_door": 0.55,
    "lr_two": 0.75, "lr_twins": 0.5, "zero_pride": 0.55, "hat_walk": 0.35, "hat_storm": 0.85,
    "zero_panic": 0.95, "hat_trial": 0.7, "hat_talk": 0.3, "hat_gift": 0.25, "lr_knock": 0.55, "new_kid": 0.5, "unfreeze": 0.55, "statues": 0.3,
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


class HorrorScore(Score):
    """Creepy score: dissonant drone, detuned music box lullaby, glassy tones, heartbeat bass,
    string-cluster swells, and a silence-then-hit on every scare."""

    LULLABY = [69, 72, 76, 74, 72, 71, 69, 64, 69, 72, 76, 77, 76, 74, 72, 71]  # A minor, slow

    def _fft_band(self, x, lo, hi):
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x), 1.0 / SR)
        g = 1.0 / (1.0 + (lo / np.maximum(f, 1e-3)) ** 4) / (1.0 + (f / hi) ** 4)
        return np.fft.irfft(spec * g, n=len(x))

    def render(self) -> np.ndarray:
        n = self.n
        t = np.arange(n) / SR
        energy = np.interp(t, self._grid, self._energy)
        buf = np.zeros((n, 2))
        # 1) dissonant low drone (A1 + Bb1), slowly breathing
        drone = np.zeros(n)
        for f0, det in ((55.0, 0.0), (58.27, 0.0), (55.0, 0.004), (82.4, -0.003)):
            drone += np.sign(np.sin(2 * np.pi * f0 * (1 + det) * t)) * 0.25
        drone = self._fft_band(drone, 30, 380)
        drone *= (0.55 + 0.45 * np.sin(2 * np.pi * 0.07 * t) ** 2) * (0.10 + 0.10 * energy)
        buf += np.stack([drone, np.roll(drone, 300)], axis=1)
        # 2) detuned music box lullaby in the quiet parts
        step = 0.62
        k = 0
        while k * step < self.duration:
            tt = k * step
            e = self.energy(tt)
            if e < 0.5:
                note = self.LULLABY[k % len(self.LULLABY)]
                f = midi_hz(note) * 2 ** (self.rng.uniform(-18, 18) / 1200)
                m = int(1.6 * SR)
                x = np.arange(m) / SR
                sig = sum(np.sin(2 * np.pi * f * r * x) * a * np.exp(-x * d)
                          for r, a, d in ((1.0, 1.0, 2.5), (2.76, 0.45, 5.0), (5.4, 0.2, 9.0)))
                _add(buf, tt, sig * 0.05 * (1 - e * 1.6), pan=float(self.rng.uniform(-0.5, 0.5)))
            k += 1
        # 3) glassy beating tones now and then
        tt = 1.5
        while tt < self.duration:
            f = float(self.rng.choice([1318.5, 1567.98, 1975.5]))
            m = int(5.0 * SR)
            x = np.arange(m) / SR
            env = np.minimum(x / 2.0, 1.0) * np.exp(-np.maximum(x - 2.0, 0) * 0.8)
            sig = (np.sin(2 * np.pi * f * x) + np.sin(2 * np.pi * (f + 6.5) * x)) * env
            _add(buf, tt, sig * 0.014, pan=float(self.rng.uniform(-0.8, 0.8)))
            tt += float(self.rng.uniform(5.0, 8.0))
        # 4) heartbeat sub pulse when tension rises
        tt = 0.0
        while tt < self.duration:
            e = self.energy(tt)
            if e > 0.33:
                m = int(0.55 * SR)
                x = np.arange(m) / SR
                beat = lambda d: np.sin(2 * np.pi * 48 * np.maximum(x - d, 0)) * np.exp(-np.maximum(x - d, 0) * 16) * (x >= d)  # noqa: E731
                _add(buf, tt, (beat(0.0) + 0.75 * beat(0.2)) * 0.35 * min(1.0, (e - 0.25) * 2))
                tt += 1.0 - 0.45 * e
            else:
                tt += 0.5
        # 5) dissonant string-cluster swells in the intense parts
        cluster = np.zeros(n)
        for note in (57, 58, 64, 65, 70):
            f = midi_hz(note)
            cluster += np.sign(np.sin(2 * np.pi * f * t + note)) * 0.2
        cluster = self._fft_band(cluster, 180, 2600)
        cluster *= np.clip((energy - 0.5) * 2.2, 0, 1) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t))
        buf += np.stack([cluster, np.roll(cluster, 500)], axis=1) * 0.05
        # 6) silence before each scare, then a hit + high shriek
        gate = np.ones(n)
        for hit in self.impacts:
            a, b = int((hit - 0.7) * SR), int(hit * SR)
            if 0 <= a < b <= n:
                gate[a:b] = np.linspace(1.0, 0.08, b - a)
            c = min(n, b + int(0.25 * SR))
            if b < c:
                gate[b:c] = np.linspace(0.08, 1.0, c - b)
        buf *= gate[:, None]
        for hit in self.impacts:
            self._braam(buf, hit)
            m = int(1.4 * SR)
            x = np.arange(m) / SR
            shriek = sum(np.sin(2 * np.pi * f * x + 3 * np.sin(2 * np.pi * 7 * x)) for f in (1760, 1864.7, 2093))
            _add(buf, hit, shriek * np.exp(-x * 3.0) * 0.02)
        buf = np.tanh(buf * 1.3) / np.tanh(1.3)
        peak = np.max(np.abs(buf)) + 1e-9
        return (buf / peak * 0.9).astype(np.float32)


class MysteryScore(HorrorScore):
    """Puzzle / "spot the fake" score: clock ticks, plucked pizzicato ostinato in D minor,
    celesta motif, low piano stabs and a rising tension pad."""

    MOTIF = [74, 77, 76, 73, 74, 81, 79, 77]

    def render(self) -> np.ndarray:
        n = self.n
        t = np.arange(n) / SR
        energy = np.interp(t, self._grid, self._energy)
        buf = np.zeros((n, 2))
        # soft pad (D minor add9), opening up with tension
        pad = np.zeros(n)
        for note in (50, 57, 62, 65, 64):
            f = midi_hz(note)
            pad += np.sin(2 * np.pi * f * t + np.sin(2 * np.pi * 0.3 * t) * 0.4) * 0.2
        pad = self._fft_band(pad, 80, 1800) * (0.05 + 0.07 * energy)
        buf += np.stack([pad, np.roll(pad, 400)], axis=1)
        # pizzicato ostinato (eighths at 100 bpm), louder with energy
        step = 0.3
        ost = [50, 57, 53, 57, 50, 58, 53, 57]
        k = 0
        while k * step < self.duration:
            tt = k * step
            e = self.energy(tt)
            if e > 0.22:
                m = int(0.35 * SR)
                x = np.arange(m) / SR
                f = midi_hz(ost[k % len(ost)])
                sig = (np.sin(2 * np.pi * f * x) + 0.5 * np.sin(4 * np.pi * f * x)) * np.exp(-x * 18)
                _add(buf, tt, sig * 0.09 * min(1.0, e + 0.2), pan=-0.3 if k % 2 else 0.3)
            # clock tick on every beat
            if k % 2 == 0:
                m = int(0.03 * SR)
                x = np.arange(m) / SR
                tick = np.diff(self.rng.standard_normal(m + 1)) * np.exp(-x * 250)
                _add(buf, tt, tick * (0.05 if k % 4 == 0 else 0.03), pan=0.5)
            k += 1
        # celesta motif in the calm parts
        step = 0.6
        k = 0
        while k * step < self.duration:
            tt = k * step
            if self.energy(tt) < 0.55 and (k // 8) % 2 == 0:
                f = midi_hz(self.MOTIF[k % len(self.MOTIF)])
                m = int(1.2 * SR)
                x = np.arange(m) / SR
                sig = (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(2 * np.pi * f * 4.0 * x) * np.exp(-x * 6)) * np.exp(-x * 3.2)
                _add(buf, tt, sig * 0.045, pan=float(self.rng.uniform(-0.6, 0.6)))
            k += 1
        # low piano stabs every 4.8 s when tense
        tt = 2.4
        while tt < self.duration:
            if self.energy(tt) > 0.4:
                m = int(2.5 * SR)
                x = np.arange(m) / SR
                sig = sum(np.sin(2 * np.pi * midi_hz(nn) * x) * np.exp(-x * 1.8) for nn in (26, 38, 45))
                _add(buf, tt, sig * 0.12)
            tt += 4.8
        for hit in self.impacts:
            self._braam(buf, hit)
        buf = np.tanh(buf * 1.3) / np.tanh(1.3)
        return (buf / (np.max(np.abs(buf)) + 1e-9) * 0.9).astype(np.float32)


class EpicScore(Score):
    """Hybrid trailer / trap score: 808 bass, rolling hi-hats, big toms, brass-like stabs and a choir pad.
    Builds with the story energy (hours counting up)."""

    PROG = [(57, "min"), (53, "maj"), (48, "maj"), (55, "maj")]

    def render(self) -> np.ndarray:
        n = self.n
        buf = np.zeros((n, 2))
        beat = 60 / 140
        bar = beat * 4
        t0 = 0.0
        idx = 0
        while t0 < self.duration:
            root, q = self.PROG[(idx // 2) % len(self.PROG)]
            e = self.energy(t0)
            # choir-ish pad: stacked detuned sines with vibrato
            m = int((bar + 1.0) * SR)
            x = np.arange(m) / SR
            env = _env(m, 0.4, 0.9)
            pad = np.zeros(m)
            for note in _chord(root, q) + [root + 12]:
                f = midi_hz(note)
                for det in (-0.004, 0.0, 0.004):
                    pad += np.sin(2 * np.pi * f * (1 + det) * x + 0.3 * np.sin(2 * np.pi * 5.2 * x))
            _add(buf, t0, pad * env * (0.012 + 0.012 * e))
            for b in range(4):
                tb = t0 + b * beat
                eb = self.energy(tb)
                # 808 on 1 and the "and" of 2
                for off in ((0.0,) if b % 2 else (0.0, beat * 0.5) if b == 2 else (0.0,)):
                    if b in (0, 2) and eb > 0.35:
                        mm = int(0.9 * SR)
                        xx = np.arange(mm) / SR
                        f = midi_hz(root - 24)
                        sig = np.sin(2 * np.pi * np.cumsum(f * (1 + 1.5 * np.exp(-xx * 30))) / SR) * np.exp(-xx * 2.5)
                        _add(buf, tb + off, np.tanh(sig * 2.2) * 0.32 * min(1.0, eb + 0.1))
                # clap / snare on 2 and 4
                if b in (1, 3) and eb > 0.5:
                    mm = int(0.3 * SR)
                    xx = np.arange(mm) / SR
                    noise = self.rng.standard_normal(mm) * (np.exp(-xx * 20) + 0.5 * np.exp(-np.maximum(xx - 0.012, 0) * 22))
                    _add(buf, tb, noise * 0.13)
                # hats: 8ths, 16th/32nd rolls when intense
                sub = 4 if eb > 0.8 else 2
                for h in range(sub):
                    th = tb + h * beat / sub
                    if self.energy(th) > 0.3:
                        mm = int(0.04 * SR)
                        xx = np.arange(mm) / SR
                        noise = np.diff(self.rng.standard_normal(mm + 1))
                        _add(buf, th, noise * np.exp(-xx * 120) * 0.03, pan=0.35)
                # big toms at bar starts when very intense
                if b == 0 and eb > 0.75:
                    mm = int(0.8 * SR)
                    xx = np.arange(mm) / SR
                    tom = np.sin(2 * np.pi * np.cumsum(70 + 60 * np.exp(-xx * 12)) / SR) * np.exp(-xx * 4)
                    _add(buf, tb, tom * 0.4)
            # brass-like stab on each new chord
            if idx % 2 == 0 and e > 0.45:
                mm = int(0.9 * SR)
                xx = np.arange(mm) / SR
                st = sum(_saw(midi_hz(nn - 12), mm, 8) for nn in _chord(root, q)) * _env(mm, 0.02, 0.5) * np.exp(-xx * 2)
                _add(buf, t0, st * 0.05)
            t0 += bar
            idx += 1
        for hit in self.impacts:
            self._riser(buf, hit)
            self._braam(buf, hit)
        buf = np.tanh(buf * 1.25) / np.tanh(1.25)
        return (buf / (np.max(np.abs(buf)) + 1e-9) * 0.9).astype(np.float32)


class DreadScore(HorrorScore):
    """The creepiest score: the horror bed plus a sub-bass rumble that beats against itself, whispers
    panning around the listener, metallic scrapes, creaking wood and reversed swells sucking into each scare."""

    def render(self) -> np.ndarray:
        base = super().render().astype(np.float64) * 0.75
        n = len(base)
        t = np.arange(n) / SR
        energy = np.interp(t, self._grid, self._energy)
        buf = np.zeros((n, 2))
        # 1) beating sub rumble (felt more than heard)
        sub = np.sin(2 * np.pi * 31.0 * t) + np.sin(2 * np.pi * 32.6 * t) + 0.4 * np.sin(2 * np.pi * 46.5 * t)
        sub *= (0.05 + 0.13 * energy) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.05 * t))
        buf += np.stack([sub, sub], axis=1)
        # 2) whispers: band-passed noise shaped like syllables, wandering in the stereo field
        tt = 2.0
        while tt < self.duration - 1.5:
            m = int(float(self.rng.uniform(1.0, 2.0)) * SR)
            x = np.arange(m) / SR
            noise = self._fft_band(self.rng.standard_normal(m), 1400, 4200)
            syll = np.clip(np.sin(2 * np.pi * float(self.rng.uniform(5.5, 8.5)) * x + float(self.rng.uniform(0, 6))), 0, 1) ** 2
            env = np.sin(np.pi * x / x[-1]) ** 1.5
            sig = noise * syll * env
            sig /= np.max(np.abs(sig)) + 1e-9
            pan = float(self.rng.uniform(-1, 1))
            _add(buf, tt, sig * 0.05, pan=pan)
            tt += float(self.rng.uniform(3.5, 6.5))
        # 3) metallic scrapes
        tt = 5.0
        while tt < self.duration - 2:
            m = int(1.8 * SR)
            x = np.arange(m) / SR
            f = float(self.rng.uniform(1800, 3200)) * (1 + 0.25 * x)
            sig = np.sin(2 * np.pi * np.cumsum(f) / SR + 4 * np.sin(2 * np.pi * 37 * x)) * np.exp(-x * 1.6) * np.minimum(x / 0.3, 1)
            _add(buf, tt, sig * 0.018, pan=float(self.rng.uniform(-0.7, 0.7)))
            tt += float(self.rng.uniform(9, 15))
        # 4) creaking wood (old house settling)
        tt = 3.5
        while tt < self.duration - 1:
            m = int(0.9 * SR)
            x = np.arange(m) / SR
            f0 = float(self.rng.uniform(90, 140))
            pulses = (np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.4 * x)) / SR) > 0.6).astype(float)
            sig = self._fft_band(pulses * (0.6 + 0.4 * self.rng.random(m)), 150, 1500) * np.sin(np.pi * x / x[-1])
            _add(buf, tt, sig * 0.05, pan=float(self.rng.uniform(-0.9, 0.9)))
            tt += float(self.rng.uniform(6, 11))
        # 5) reversed swell sucking into every scare
        for hit in self.impacts:
            m = int(1.6 * SR)
            x = np.arange(m) / SR
            chord = sum(np.sin(2 * np.pi * midi_hz(nn) * x) for nn in (45, 51, 56, 58)) * np.exp(-x * 2.2)
            rev = chord[::-1] * 0.08
            start = hit - 1.6
            if start > 0:
                _add(buf, start, rev)
        out = base + buf
        out = np.tanh(out * 1.2) / np.tanh(1.2)
        return (out / (np.max(np.abs(out)) + 1e-9) * 0.9).astype(np.float32)


STYLES = {"dread": DreadScore, "horror": HorrorScore, "mystery": MysteryScore, "epic": EpicScore}


def write_score(path: str, duration: float, scenes, impacts: list[float], seed: int = 0, style: str = "") -> str:
    cls = STYLES.get(style, Score)
    audio = cls(duration, sections_for(scenes), impacts, seed).render()
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return path
