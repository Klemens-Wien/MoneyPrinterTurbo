"""Narrator voice with per-word timings (for word-by-word captions).

Backends:
  edge   – Microsoft Edge neural voices via edge-tts (needs internet), exact word timings
  espeak – espeak-ng (offline, robotic), estimated word timings
  silent – no voice, estimated timings (useful for previews / your own voice-over)
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

SAMPLE_RATE = 44100

DEFAULT_VOICES = {
    "en": "en-US-AndrewNeural",
    "de": "de-DE-ConradNeural",
}
ESPEAK_VOICES = {"en": "en-us", "de": "de"}


@dataclass
class Word:
    text: str
    start: float
    end: float
    emph: bool = False


@dataclass
class Clip:
    path: str
    duration: float
    words: list[Word] = field(default_factory=list)


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def to_wav(src: str, dst: str, tempo: float = 1.0) -> None:
    filters = ["aresample=44100"]
    if abs(tempo - 1.0) > 1e-3:
        filters.append(f"atempo={tempo:.4f}")
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-af", ",".join(filters), "-ac", "1", "-ar", str(SAMPLE_RATE), dst],
        check=True,
    )


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
    return data


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _trim_silence(path: str, threshold: float = 0.01) -> tuple[float, float]:
    """Return (leading, trailing) silence in seconds."""
    data = np.abs(read_wav(path))
    idx = np.where(data > threshold)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    return idx[0] / SAMPLE_RATE, (len(data) - idx[-1]) / SAMPLE_RATE


def estimate_words(text: str, start: float, end: float) -> list[Word]:
    """Distribute word timings by syllable-ish weight (letters + punctuation pauses)."""
    tokens = text.split()
    if not tokens:
        return []
    weights = []
    for tok in tokens:
        w = 1.2 + len(re.sub(r"\W", "", tok)) * 0.55
        if re.search(r"[.!?]$", tok):
            w += 2.2
        elif re.search(r"[,;:]$", tok):
            w += 1.1
        weights.append(w)
    total = sum(weights)
    words, t = [], start
    for tok, w in zip(tokens, weights):
        dur = (end - start) * w / total
        words.append(Word(tok, t, t + dur * 0.92))
        t += dur
    return words


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _edge_clip(text: str, voice: str, rate_pct: int, out_wav: str, pitch: str = "+0Hz") -> Clip:
    import edge_tts

    mp3 = out_wav[:-4] + ".mp3"
    words: list[Word] = []

    async def run():
        rate = f"{rate_pct:+d}%"
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary", proxy=proxy)
        with open(mp3, "wb") as fh:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    fh.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / 1e7
                    words.append(Word(chunk["text"], start, start + chunk["duration"] / 1e7))

    for attempt in range(5):  # the service drops a request now and then: retry before giving up
        try:
            words.clear()
            asyncio.run(run())
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 4:
                raise
            logger.warning(f"edge-tts attempt {attempt + 1} failed ({exc}); retrying")
            time.sleep(1.5 * (attempt + 1))
    to_wav(mp3, out_wav)
    os.remove(mp3)
    return Clip(out_wav, wav_duration(out_wav), words)


def _espeak_clip(text: str, language: str, rate_pct: int, out_wav: str) -> Clip:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if not binary:
        raise RuntimeError("espeak-ng is not installed")
    raw = out_wav[:-4] + ".raw.wav"
    speed = int(165 * (1 + rate_pct / 100.0))
    subprocess.run([binary, "-v", ESPEAK_VOICES.get(language, "en-us"), "-s", str(speed), "-w", raw, text], check=True)
    to_wav(raw, out_wav)
    os.remove(raw)
    duration = wav_duration(out_wav)
    lead, trail = _trim_silence(out_wav)
    return Clip(out_wav, duration, estimate_words(text, lead, max(lead + 0.1, duration - trail)))


def _silent_clip(text: str, rate_pct: int, out_wav: str, language: str) -> Clip:
    wps = (2.55 if language == "en" else 2.2) * (1 + rate_pct / 100.0)
    duration = max(1.2, len(text.split()) / wps + 0.35)
    samples = np.zeros(int(duration * SAMPLE_RATE), dtype=np.int16)
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())
    return Clip(out_wav, duration, estimate_words(text, 0.1, duration - 0.25))


VOICE_FX = {
    # the thing that imitates mom: echo, a slight wobble and a muffled top end
    "eerie": "aecho=0.8:0.75:70|140:0.35|0.22,vibrato=f=4.5:d=0.12,lowpass=f=4200",
    # a voice coming through a phone speaker
    "phone": "highpass=f=350,lowpass=f=3200,acompressor",
    # a monster: formants and pitch lowered together (sounds big, not slowed down), warm low end, short room
    "monster": f"asetrate={SAMPLE_RATE}*0.8,aresample={SAMPLE_RATE},atempo=1.25,"
               "bass=g=5:f=140,aecho=0.6:0.45:35|70:0.18|0.1,acompressor=threshold=-18dB:ratio=3",
}


def _apply_fx(path: str, fx: str) -> None:
    chain = VOICE_FX.get(fx)
    if not chain:
        return
    tmp = path[:-4] + ".fx.wav"
    os.replace(path, tmp)
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", tmp, "-af", chain, "-ac", "1", "-ar", str(SAMPLE_RATE), path],
                   check=True)
    os.remove(tmp)


def _synthesize_once(texts: list[str], out_dir: str, backend: str, voice: str, language: str, rate_pct: int,
                     pitch: str = "+0Hz", specs: list[dict] | None = None) -> list[Clip]:
    clips = []
    for i, text in enumerate(texts):
        path = os.path.join(out_dir, f"voice_{i:02d}.wav")
        spec = (specs[i] if specs and i < len(specs) else None) or {}
        if backend == "edge":
            clip = _edge_clip(text, spec.get("voice") or voice, int(np.clip(rate_pct + int(spec.get("rate", 0)), -50, 60)),
                              path, spec.get("pitch") or pitch)
            if spec.get("fx"):
                _apply_fx(path, spec["fx"])
                clip.duration = wav_duration(path)
            if not clip.words:
                lead, trail = _trim_silence(path)
                clip.words = estimate_words(text, lead, clip.duration - trail)
        elif backend == "espeak":
            clip = _espeak_clip(text, language, rate_pct, path)
        else:
            clip = _silent_clip(text, rate_pct, path, language)
        clips.append(clip)
    return clips


def _resolve_backend(backend: str, voice: str) -> list[str]:
    if backend != "auto":
        return [backend]
    order = ["edge", "espeak", "silent"]
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        order.remove("espeak")
    return order


def synthesize(
    texts: list[str],
    out_dir: str,
    language: str = "en",
    voice: str | None = None,
    backend: str = "auto",
    rate_pct: int = 8,
    target_range: tuple[float, float] = (75.0, 95.0),
    gap: float = 0.18,
    pitch: str = "+0Hz",
    specs: list[dict] | None = None,
) -> tuple[list[Clip], str]:
    """Synthesize one clip per scene; re-time the speech so the total fits the target range."""
    os.makedirs(out_dir, exist_ok=True)
    voice = voice or DEFAULT_VOICES.get(language, DEFAULT_VOICES["en"])
    last_error: Exception | None = None
    for name in _resolve_backend(backend, voice):
        try:
            clips = _synthesize_once(texts, out_dir, name, voice, language, rate_pct, pitch, specs)
        except Exception as exc:
            last_error = exc
            logger.warning(f"TTS backend '{name}' failed: {exc}")
            if specs and name == "edge":  # a cast of real voices must never silently turn into a robot voice
                raise
            continue
        total = sum(c.duration for c in clips) + gap * len(clips)
        lo, hi = target_range
        ideal = (lo + hi) / 2
        if not lo + 2 <= total <= hi - 2:
            # adjust the speaking rate once, then fine-tune with time stretching
            # speed up when too long, but never drag a natural voice down by more than a few percent
            new_rate = int(np.clip(rate_pct + (total / ideal - 1) * 100, max(-30, rate_pct - 6), 40))
            if new_rate != rate_pct and name != "silent":
                logger.info(f"narration {total:.1f}s -> re-synthesizing at rate {new_rate:+d}%")
                clips = _synthesize_once(texts, out_dir, name, voice, language, new_rate, pitch, specs)
                total = sum(c.duration for c in clips) + gap * len(clips)
        if total > hi:
            tempo = min(total / (hi - 1.0), 1.35)
            logger.info(f"narration {total:.1f}s too long -> time-stretch x{tempo:.3f}")
            for clip in clips:
                tmp = clip.path[:-4] + ".tmp.wav"
                os.replace(clip.path, tmp)
                to_wav(tmp, clip.path, tempo=tempo)
                os.remove(tmp)
                clip.duration = wav_duration(clip.path)
                clip.words = [Word(w.text, w.start / tempo, w.end / tempo) for w in clip.words]
        return clips, name
    raise RuntimeError(f"no TTS backend worked: {last_error}")
