"""Episodic short-video series ("Du entscheidest" format).

Builds one vertical episode from an episode spec (JSON) and keeps the series
state (characters, story so far, pending viewer question) in ``series.json``.

It reuses the existing pipeline building blocks wherever possible:

* narration: ``voice.tts`` (Edge TTS) with an offline Piper fallback
* scene images: ``material.generate_images_openai`` (any OpenAI-compatible
  image endpoint, incl. local SD gateways) with offline placeholder art
* next-episode scripts: ``llm._generate_response`` with the configured LLM
* pop animation: ``video._get_subtitle_spring_scale``; FFmpeg via ``utils``

What it adds on top: karaoke subtitles (1-3 words, active word highlighted),
zoom/pan camera moves with transitions, synthesized music and sound effects,
an on-screen decision card at the end and the series bookkeeping.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.services import series_art, series_audio
from app.utils import utils

OUT_W, OUT_H, FPS = 1080, 1920, 30
ART_W, ART_H = series_art.W, series_art.H
SR = series_audio.SAMPLE_RATE

HIGHLIGHT = (255, 214, 10)
SUB_FONT = "BeVietnamPro-Bold.ttf"
PIPER_MODEL_NAME = "de_DE-thorsten-high"
PIPER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "vits-piper-de_DE-thorsten-high.tar.bz2"
)
CAMERA_MOVES = ("zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down")
TRANSITIONS = ("crossfade", "flash", "black")
EFFECTS = ("shake", "alarm_pulse", "flicker")


def series_root() -> str:
    return os.path.join(utils.root_dir(), "series")


def series_dir(slug: str) -> str:
    return os.path.join(series_root(), slug)


def series_file(slug: str) -> str:
    return os.path.join(series_dir(slug), "series.json")


def load_series(slug: str) -> dict:
    with open(series_file(slug), encoding="utf-8") as f:
        return json.load(f)


def save_series(slug: str, data: dict) -> None:
    path = series_file(slug)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_episode_spec(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate_episode_spec(spec: dict) -> list[str]:
    """Return a list of problems (empty when the spec is usable)."""
    problems = []
    scenes = spec.get("scenes") or []
    if not 6 <= len(scenes) <= 10:
        problems.append(f"6-10 Szenen erwartet, gefunden: {len(scenes)}")
    if scenes and not scenes[-1].get("decision"):
        problems.append("die letzte Szene muss \"decision\": true haben (Entscheidungsfrage)")
    q = spec.get("question") or {}
    for key in ("text", "option_1", "option_2"):
        if not str(q.get(key, "")).strip():
            problems.append(f"question.{key} fehlt")
    for i, s in enumerate(scenes, 1):
        if not str(s.get("narration", "")).strip():
            problems.append(f"Szene {i}: narration fehlt")
        if not str(s.get("image_prompt", "")).strip():
            problems.append(f"Szene {i}: image_prompt fehlt")
        for name in s.get("sfx", []):
            if name not in series_audio.SFX:
                problems.append(f"Szene {i}: unbekanntes Geräusch {name!r}")
        if s.get("camera", "zoom_in") not in CAMERA_MOVES:
            problems.append(f"Szene {i}: unbekannte Kamera {s.get('camera')!r}")
        if s.get("transition", "crossfade") not in TRANSITIONS:
            problems.append(f"Szene {i}: unbekannter Übergang {s.get('transition')!r}")
    if "episode" not in spec:
        problems.append("episode (Nummer) fehlt")
    return problems


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def expand_image_prompt(series: dict, scene: dict) -> str:
    """Insert the fixed character descriptions so every image shows the same people."""
    prompt = scene["image_prompt"]
    used = []
    for ch in series.get("characters", []):
        token = "{" + ch["id"] + "}"
        if token in prompt:
            prompt = prompt.replace(token, ch["image_prompt"])
            used.append(ch["name"])
    style = series.get("visual_style", "")
    consistency = (
        " Keep the exact same character designs in every image." if used else ""
    )
    return f"{prompt}. Style: {style}.{consistency}"


# --------------------------------------------------------------------------
# narration (TTS)
# --------------------------------------------------------------------------


def _ffmpeg() -> str:
    return utils.get_ffmpeg_binary() or "ffmpeg"


def _to_wav44k(src: str, dst: str, voice_filter: bool = True) -> None:
    af = "aresample=44100"
    if voice_filter:
        af += ",highpass=f=60,acompressor=threshold=-20dB:ratio=3:attack=5:release=120"
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-af", af, dst],
        check=True,
    )


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return (data.astype(np.float32) / 32768.0)


def piper_model_path() -> str:
    from app.config import config

    configured = str(config.app.get("series_piper_model", "") or "").strip()
    if configured:
        return configured
    return os.path.join(
        utils.storage_dir("series_models"),
        f"vits-piper-{PIPER_MODEL_NAME}",
        f"{PIPER_MODEL_NAME}.onnx",
    )


class Narrator:
    """Edge TTS when reachable, otherwise the local Piper voice."""

    def __init__(self, engine: str, edge_voice: str, speed: float = 0.9):
        self.engine = engine
        self.edge_voice = edge_voice
        self.speed = speed
        self._piper = None
        self.used = None

    def _edge(self, text: str, out_wav: str) -> bool:
        from app.services import voice

        mp3 = out_wav[:-4] + ".mp3"
        rate = self.speed if self.speed else 1.0
        try:
            sub_maker = voice.tts(text=text, voice_name=self.edge_voice, voice_rate=rate, voice_file=mp3)
        except Exception as exc:  # network/auth problems surface as exceptions
            logger.warning(f"edge tts failed: {exc}")
            return False
        if not sub_maker or not os.path.exists(mp3):
            return False
        _to_wav44k(mp3, out_wav)
        return True

    def _piper_tts(self, text: str, out_wav: str) -> None:
        if self._piper is None:
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise RuntimeError(
                    "Offline-Stimme nicht installiert: `uv pip install piper-tts`"
                ) from exc
            model = piper_model_path()
            if not os.path.exists(model):
                raise RuntimeError(
                    f"Piper-Stimmmodell fehlt: {model}\n"
                    f"Download: {PIPER_MODEL_URL} (entpacken nach storage/series_models/)"
                )
            self._piper = PiperVoice.load(model)
        from piper import SynthesisConfig

        raw = out_wav[:-4] + ".raw.wav"
        cfg = SynthesisConfig(length_scale=1.0 / max(self.speed, 0.5), noise_scale=0.6, noise_w_scale=0.7)
        with wave.open(raw, "wb") as w:
            self._piper.synthesize_wav(text, w, syn_config=cfg)
        _to_wav44k(raw, out_wav)
        os.remove(raw)

    def synth(self, text: str, out_wav: str) -> np.ndarray:
        if self.engine in ("auto", "edge") and self.used in (None, "edge"):
            if self._edge(text, out_wav):
                self.used = "edge"
                return _read_wav(out_wav)
            if self.engine == "edge":
                raise RuntimeError("Edge TTS nicht erreichbar")
            logger.warning("edge tts unavailable, switching to the offline piper voice")
        self.used = "piper"
        self._piper_tts(text, out_wav)
        return _read_wav(out_wav)


# --------------------------------------------------------------------------
# timeline
# --------------------------------------------------------------------------


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Scene:
    index: int
    spec: dict
    start: float = 0.0
    end: float = 0.0
    words: list[Word] = field(default_factory=list)
    image: Image.Image | None = None
    image_source: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _speech_bounds(samples: np.ndarray) -> tuple[float, float]:
    win = int(0.02 * SR)
    if samples.size < win * 2:
        return 0.0, samples.size / SR
    n = samples.size // win
    rms = np.sqrt((samples[: n * win].reshape(n, win) ** 2).mean(axis=1))
    thr = max(rms.max() * 0.06, 1e-4)
    active = np.where(rms > thr)[0]
    if active.size == 0:
        return 0.0, samples.size / SR
    return active[0] * win / SR, min(samples.size, (active[-1] + 1) * win) / SR


def distribute_words(sentence: str, start: float, end: float) -> list[Word]:
    """Spread the words of one spoken sentence across its speech interval."""
    tokens = sentence.split()
    weights = []
    for tok in tokens:
        core = re.sub(r"[^\wÄÖÜäöüß]", "", tok)
        w = max(len(core), 2) + 1.5
        if re.search(r"\d", core):
            w += len(core) * 2.5  # numbers are read out as long words
        if re.search(r"[,;:–-]$", tok):
            w += 2.5  # short pause after the comma
        weights.append(w)
    total = sum(weights) or 1
    out, t = [], start
    for tok, w in zip(tokens, weights):
        dur = (end - start) * w / total
        out.append(Word(tok, t, t + dur))
        t += dur
    return out


def group_words(words: list[Word], max_words: int = 3, max_chars: int = 16) -> list[list[Word]]:
    groups, cur = [], []
    for w in words:
        clean = display_word(w.text)
        cur_len = sum(len(display_word(x.text)) + 1 for x in cur)
        if cur and (len(cur) >= max_words or cur_len + len(clean) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?…:;,–]$", w.text):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def display_word(token: str) -> str:
    return re.sub(r"[.,;:…]+$", "", token).strip("„“\"")


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def group_timing(groups: list[list[Word]], total: float) -> list[tuple[float, float]]:
    spans = []
    for i, g in enumerate(groups):
        start = g[0].start
        end = g[-1].end + 0.25
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][0].start)
        spans.append((start, min(end, total)))
    return spans


def write_srt(groups, spans, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, (g, (a, b)) in enumerate(zip(groups, spans), 1):
            text = " ".join(display_word(w.text) for w in g)
            f.write(f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{text}\n\n")


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.LANCZOS)
    x = (img.width - w) // 2
    y = (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def scene_image(series: dict, scene: dict, images_mode: str, save_dir: str) -> tuple[Image.Image, str]:
    """AI image via the configured openai_image source, else offline placeholder art."""
    if images_mode in ("auto", "ai"):
        from app.models.schema import VideoAspect
        from app.services import material

        if material.is_openai_image_enabled():
            prompt = expand_image_prompt(series, scene)
            try:
                items = material.generate_images_openai(prompt, 5, VideoAspect.portrait, save_dir)
            except Exception as exc:
                logger.warning(f"image generation failed: {exc}")
                items = []
            if items:
                with Image.open(items[0].url) as im:
                    return _cover(im, ART_W, ART_H), "ai"
            if images_mode == "ai":
                raise RuntimeError("Bildgenerierung fehlgeschlagen (openai_image)")
        elif images_mode == "ai":
            raise RuntimeError(
                "Keine Bildquelle konfiguriert: openai_image_base_url / openai_image_model / "
                "openai_image_api_keys in config.toml eintragen"
            )
    return series_art.render(scene.get("art", "ship_night")), "placeholder"


# --------------------------------------------------------------------------
# overlays
# --------------------------------------------------------------------------


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(utils.font_dir(), SUB_FONT), size)


def _text_w(font, text) -> int:
    b = font.getbbox(text)
    return b[2] - b[0]


class SubtitleRenderer:
    """Big centered 1-3 word captions with the spoken word highlighted."""

    def __init__(self, size: int = 112, stroke: int = 10, max_width: int = 960):
        self.size, self.stroke, self.max_width = size, stroke, max_width
        self.cache: dict = {}

    def render(self, words: list[str], active: int) -> Image.Image:
        key = (tuple(words), active)
        if key in self.cache:
            return self.cache[key]
        size = self.size
        while True:
            font = _font(size)
            space = _text_w(font, " ")
            lines = [[]]
            for i, w in enumerate(words):
                trial = lines[-1] + [i]
                width = sum(_text_w(font, words[j]) for j in trial) + space * (len(trial) - 1)
                if lines[-1] and width > self.max_width:
                    lines.append([i])
                else:
                    lines[-1] = trial
            widest = max(sum(_text_w(font, words[j]) for j in ln) + space * (len(ln) - 1) for ln in lines)
            if widest <= self.max_width or size <= 64:
                break
            size -= 8
        line_h = int(size * 1.18)
        pad = self.stroke + 24
        w_img = widest + 2 * pad
        h_img = line_h * len(lines) + 2 * pad
        img = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
        shadow = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
        d, ds = ImageDraw.Draw(img), ImageDraw.Draw(shadow)
        stroke = max(6, int(self.stroke * size / self.size))
        for li, ln in enumerate(lines):
            lw = sum(_text_w(font, words[j]) for j in ln) + space * (len(ln) - 1)
            x = (w_img - lw) / 2
            y = pad + li * line_h
            for j in ln:
                color = HIGHLIGHT if j == active else (255, 255, 255)
                ds.text((x + 6, y + 8), words[j], font=font, fill=(0, 0, 0, 170), stroke_width=stroke, stroke_fill=(0, 0, 0, 170))
                d.text((x, y), words[j], font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0))
                x += _text_w(font, words[j]) + space
        shadow = shadow.filter(ImageFilter.GaussianBlur(8))
        out = Image.alpha_composite(shadow, img)
        self.cache[key] = out
        return out


def _wrap(font, text: str, max_width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cur and _text_w(font, trial) > max_width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def decision_card(question: dict) -> Image.Image:
    card = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    x0, x1, y0 = 60, OUT_W - 60, 300
    head = _font(76)
    opt_font = _font(54)
    num_font = _font(96)
    cta_font = _font(50)
    head_lines = _wrap(head, question["text"].upper(), x1 - x0 - 80)
    opts = [_wrap(opt_font, question[k], x1 - x0 - 260) for k in ("option_1", "option_2")]
    opt_heights = [max(190, 70 * len(o) + 70) for o in opts]
    y1 = y0 + 70 + 92 * len(head_lines) + 40 + sum(opt_heights) + 40 * 2 + 120
    d.rounded_rectangle((x0, y0, x1, y1), radius=44, fill=(6, 10, 18, 222), outline=HIGHLIGHT + (255,), width=6)
    y = y0 + 60
    for ln in head_lines:
        d.text((OUT_W / 2, y), ln, font=head, fill=HIGHLIGHT, anchor="mt", stroke_width=3, stroke_fill=(0, 0, 0))
        y += 92
    y += 30
    for n, (lines, oh) in enumerate(zip(opts, opt_heights), 1):
        bx0, bx1 = x0 + 40, x1 - 40
        d.rounded_rectangle((bx0, y, bx1, y + oh), radius=30, fill=(255, 255, 255, 30), outline=(255, 255, 255, 120), width=3)
        cx, cy = bx0 + 95, y + oh / 2
        d.ellipse((cx - 68, cy - 68, cx + 68, cy + 68), fill=HIGHLIGHT + (255,))
        d.text((cx, cy + 4), str(n), font=num_font, fill=(10, 10, 14), anchor="mm")
        ty = cy - 35 * len(lines)
        for ln in lines:
            d.text((bx0 + 200, ty), ln, font=opt_font, fill=(255, 255, 255))
            ty += 70
        y += oh + 40
    d.text((OUT_W / 2, y + 30), question.get("call_to_action", "Schreib 1 oder 2 in die Kommentare!"),
           font=cta_font, fill=(255, 255, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
    return card


def title_badge(series_title: str, episode: int) -> Image.Image:
    img = Image.new("RGBA", (OUT_W, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f1, f2 = _font(40), _font(30)
    d.text((OUT_W / 2, 92), series_title.upper(), font=f1, fill=(255, 255, 255, 230), anchor="mm",
           stroke_width=3, stroke_fill=(0, 0, 0, 200))
    tag = f"FOLGE {episode}"
    tw = _text_w(f2, tag) + 36
    d.rounded_rectangle((OUT_W / 2 - tw / 2, 128, OUT_W / 2 + tw / 2, 172), radius=22, fill=HIGHLIGHT + (235,))
    d.text((OUT_W / 2, 150), tag, font=f2, fill=(10, 10, 14), anchor="mm")
    return img


def ai_label() -> Image.Image:
    f = _font(28)
    text = "KI-generiert"
    img = Image.new("RGBA", (_text_w(f, text) + 40, 56), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, img.width - 1, 55), radius=26, fill=(0, 0, 0, 120))
    d.text((img.width / 2, 28), text, font=f, fill=(255, 255, 255, 200), anchor="mm")
    return img


def _blend(frame: np.ndarray, overlay: Image.Image, x: int, y: int, opacity: float = 1.0) -> None:
    ov = np.asarray(overlay, dtype=np.float32)
    h, w = ov.shape[:2]
    fx0, fy0 = max(0, x), max(0, y)
    fx1, fy1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    ov = ov[fy0 - y : fy1 - y, fx0 - x : fx1 - x]
    a = ov[..., 3:4] / 255.0 * opacity
    region = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (region * (1 - a) + ov[..., :3] * a).astype(np.uint8)


# --------------------------------------------------------------------------
# frame rendering
# --------------------------------------------------------------------------


def _ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def camera_box(move: str, p: float) -> tuple[float, float, float, float]:
    """Crop window (in art pixels) for camera progress p in [0, 1]."""
    e = _ease(p)
    if move == "zoom_out":
        zoom = 1.2 - 0.18 * e
        cx, cy = ART_W / 2, ART_H / 2
    elif move in ("pan_left", "pan_right", "pan_up", "pan_down"):
        zoom = 1.12
        mx = (ART_W - ART_W / zoom) / 2
        my = (ART_H - ART_H / zoom) / 2
        d = (e - 0.5) * 2
        cx, cy = ART_W / 2, ART_H / 2
        if move == "pan_left":
            cx -= d * mx * 0.95
        elif move == "pan_right":
            cx += d * mx * 0.95
        elif move == "pan_up":
            cy -= d * my * 0.95
        else:
            cy += d * my * 0.95
    else:  # zoom_in
        zoom = 1.02 + 0.18 * e
        cx, cy = ART_W / 2, ART_H / 2
    w, h = ART_W / zoom, ART_H / zoom
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class EpisodeRenderer:
    TRANSITION = 0.5

    def __init__(self, scenes: list[Scene], groups, spans, series_title: str, episode: int,
                 question: dict, total: float, seed: int = 0):
        self.scenes = scenes
        self.groups = groups
        self.spans = spans
        self.total = total
        self.subs = SubtitleRenderer()
        self.badge = title_badge(series_title, episode)
        self.ai = ai_label()
        self.card = decision_card(question)
        rng = np.random.default_rng(seed)
        self.grain = [rng.normal(0, 3, (OUT_H // 2, OUT_W // 2)).astype(np.int16) for _ in range(6)]
        self.flicker = rng.random(int(total * FPS) + FPS)

    def _scene_frame(self, sc: Scene, t: float) -> np.ndarray:
        half = self.TRANSITION / 2
        p = (t - (sc.start - half)) / max(sc.duration + self.TRANSITION, 0.1)
        box = camera_box(sc.spec.get("camera", "zoom_in"), p)
        effects = sc.spec.get("effects", [])
        if "shake" in effects:
            local = t - sc.start
            amp = 26 * math.exp(-max(local, 0) * 2.2)
            dx = amp * math.sin(t * 61.0)
            dy = amp * math.cos(t * 47.0)
            box = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        img = sc.image.transform((OUT_W, OUT_H), Image.EXTENT, box, Image.BILINEAR)
        frame = np.asarray(img, dtype=np.float32)
        if "alarm_pulse" in effects:
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.1 * t)
            frame = frame * (0.82 + 0.18 * pulse) + np.array([70, 0, 0], np.float32) * pulse * 0.6
        if "flicker" in effects:
            v = self.flicker[int(t * FPS) % self.flicker.size]
            if v < 0.18:
                frame *= 0.35 + v * 2
        return frame

    def frame(self, i: int) -> np.ndarray:
        t = i / FPS
        idx = next((k for k, s in enumerate(self.scenes) if t < s.end), len(self.scenes) - 1)
        sc = self.scenes[idx]
        frame = self._scene_frame(sc, t)
        half = self.TRANSITION / 2
        # incoming transition at the start of this scene
        if idx > 0 and t < sc.start + half:
            prev = self.scenes[idx - 1]
            a = (t - (sc.start - half)) / self.TRANSITION
            kind = sc.spec.get("transition", "crossfade")
            other = self._scene_frame(prev, t)
            frame = other * (1 - a) + frame * a
            if kind == "flash":
                frame = frame + 255 * max(0.0, 1 - abs(a - 0.5) * 2.2) * 0.9
            elif kind == "black":
                frame = frame * abs(a - 0.5) * 2
        elif idx + 1 < len(self.scenes) and t >= sc.end - half:
            nxt = self.scenes[idx + 1]
            a = (t - (sc.end - half)) / self.TRANSITION
            kind = nxt.spec.get("transition", "crossfade")
            other = self._scene_frame(nxt, t)
            frame = frame * (1 - a) + other * a
            if kind == "flash":
                frame = frame + 255 * max(0.0, 1 - abs(a - 0.5) * 2.2) * 0.9
            elif kind == "black":
                frame = frame * abs(a - 0.5) * 2
        # global fade from/to black
        if t < 0.25:
            frame *= t / 0.25
        if t > self.total - 0.6:
            frame *= max(0.0, (self.total - t) / 0.6)
        g = self.grain[i % len(self.grain)]
        frame += np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)[..., None]
        out = frame.clip(0, 255).astype(np.uint8)

        decision = sc.spec.get("decision")
        _blend(out, self.badge, 0, 40, 0.9)
        _blend(out, self.ai, OUT_W - self.ai.width - 40, OUT_H - 140, 1.0)
        if decision:
            local = t - sc.start
            a = _ease(local / 0.6)
            _blend(out, self.card, 0, int(60 * (1 - a)), a)
        self._draw_subtitle(out, t, OUT_H * (0.82 if decision else 0.5))
        return out

    def _draw_subtitle(self, out: np.ndarray, t: float, center_y: float) -> None:
        from app.services.video import _get_subtitle_spring_scale

        for g, (a, b) in zip(self.groups, self.spans):
            if a <= t < b:
                active = next((k for k, w in enumerate(g) if w.start <= t < w.end), len(g) - 1)
                img = self.subs.render([display_word(w.text) for w in g], active)
                scale = _get_subtitle_spring_scale(t - a, 0.3)
                if abs(scale - 1.0) > 0.01:
                    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
                _blend(out, img, int(OUT_W / 2 - img.width / 2), int(center_y - img.height / 2))
                return


# --------------------------------------------------------------------------
# audio mix
# --------------------------------------------------------------------------


def _place(track: np.ndarray, samples: np.ndarray, at: float, gain: float) -> None:
    i = int(at * SR)
    if i >= track.size:
        return
    n = min(samples.size, track.size - i)
    track[i : i + n] += samples[:n] * gain


def mix_audio(voice_track: np.ndarray, scenes: list[Scene], total: float, path: str,
              music_gain: float = 0.16, sfx_gain: float = 0.55) -> None:
    n = int(total * SR)
    voice = np.zeros(n, np.float32)
    voice[: min(n, voice_track.size)] = voice_track[:n]
    music = series_audio.dark_drone(total)[:n]
    sfx = np.zeros(n, np.float32)
    for sc in scenes:
        for name in sc.spec.get("sfx", []):
            samples, gain = series_audio.render_sfx(name, sc.duration + 0.4)
            _place(sfx, samples, max(0.0, sc.start - 0.1), gain)
    # duck music + effects slightly while the narrator speaks
    env = np.abs(voice)
    k = int(0.25 * SR)
    env = np.convolve(env, np.ones(k, np.float32) / k, mode="same")
    duck = 1.0 - 0.35 * np.clip(env / (env.max() + 1e-9) * 4, 0, 1)
    mix = voice * 1.0 + music[:n] * music_gain * duck + sfx * sfx_gain * (0.55 + 0.45 * duck)
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = np.tanh(mix / peak * 1.1) / math.tanh(1.1) * 0.94
    stereo = np.repeat((mix * 32767).astype(np.int16)[:, None], 2, axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def write_script(path: str, series: dict, spec: dict, scenes: list[Scene], total: float, narrator: str) -> None:
    lines = [
        f"# {series['series_title']} – Folge {spec['episode']}: {spec.get('title', '')}",
        "",
        f"Länge: {total:.1f} s · Format 1080x1920 (9:16) · Stimme: {narrator}",
        "",
    ]
    if spec.get("previous_choice"):
        lines += [f"**Zuschauer-Entscheidung aus der letzten Folge:** {spec['previous_choice']}", ""]
    lines += [f"**Hook (0–2 s):** {split_sentences(scenes[0].spec['narration'])[0]}", ""]
    for sc in scenes:
        s = sc.spec
        lines += [
            f"## Szene {sc.index} ({_fmt(sc.start)}–{_fmt(sc.end)})" + (" – Entscheidungsfrage" if s.get("decision") else ""),
            "",
            f"**Erzähler:** {s['narration']}",
            "",
            f"- Kamera: {s.get('camera', 'zoom_in')} · Übergang: {s.get('transition', 'crossfade')}"
            + (f" · Effekte: {', '.join(s['effects'])}" if s.get("effects") else ""),
            f"- Geräusche: {', '.join(s.get('sfx', [])) or '–'}",
            f"- Bild ({sc.image_source}): {expand_image_prompt(series, s)}",
            "",
        ]
    q = spec["question"]
    lines += [
        "## Zuschauerfrage",
        "",
        f"{q['text']}",
        f"1. {q['option_1']}",
        f"2. {q['option_2']}",
        "",
        q.get("call_to_action", "Schreib 1 oder 2 in die Kommentare!"),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_upload(path: str, series: dict, spec: dict) -> None:
    up = spec.get("upload", {})
    title = up.get("title") or f"{series['series_title']} – Folge {spec['episode']}"
    desc = up.get("description", "")
    tags = up.get("hashtags", [])
    q = spec["question"]
    question = f"{q['text']} 1️⃣ {q['option_1']} oder 2️⃣ {q['option_2']}?"
    ai_note = "Hinweis: KI-generiert – Stimme, Bilder und Ton sind computergeneriert. Die Geschichte ist frei erfunden."
    yt_title = title if len(title) <= 92 else title[:92].rstrip() + "…"
    text = f"""{series['series_title']} – Folge {spec['episode']}: {spec.get('title', '')}
================================================================

TIKTOK
------
Beschreibung:
{title}
{question} Schreib 1 oder 2 in die Kommentare! 👇
KI-generiert 🤖
{' '.join(tags[:6])} #KIgeneriert

Beim Hochladen: Einstellungen > "KI-generierte Inhalte" (AI-generated content) aktivieren.


YOUTUBE SHORTS
--------------
Titel:
{yt_title} #Shorts

Beschreibung:
{desc}

{question}
Schreib 1 oder 2 in die Kommentare – die meistgewählte Antwort bestimmt die nächste Folge!

{ai_note}

{' '.join(tags)} #Shorts #KIgeneriert

Beim Hochladen: "Veränderte oder synthetische Inhalte" mit "Ja" beantworten.


INSTAGRAM REELS
---------------
Caption:
{title}

{desc}

{question} 👇 1 oder 2?

{ai_note}

{' '.join(tags)} #reels #KIgeneriert

Beim Teilen: Erweiterte Einstellungen > "KI-Info" (Label "Mit KI erstellt") aktivieren.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------
# main entry points
# --------------------------------------------------------------------------


def render_episode(slug: str, spec: dict, output_root: str | None = None, *, voice_engine: str = "auto",
                   images_mode: str = "auto", update_series: bool = True) -> dict:
    """Render one episode into output/folge_XX/ and update series.json."""
    problems = validate_episode_spec(spec)
    if problems:
        raise ValueError("Episoden-Skript ungültig:\n- " + "\n- ".join(problems))
    series = load_series(slug)
    number = int(spec["episode"])
    output_root = output_root or os.path.join(utils.root_dir(), "output")
    out_dir = os.path.join(output_root, f"folge_{number:02d}")
    img_dir = os.path.join(out_dir, "bilder")
    os.makedirs(img_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="series-")
    try:
        narrator = Narrator(voice_engine, series.get("voice", {}).get("edge_voice", "de-DE-ConradNeural-Male"))
        scenes = [Scene(i + 1, s) for i, s in enumerate(spec["scenes"])]

        # 1) narration, sentence by sentence -> exact sentence timing
        logger.info("## narration")
        chunks, t = [], 0.15
        chunks.append(np.zeros(int(0.15 * SR), np.float32))
        for sc in scenes:
            sc.start = t if sc.index > 1 else 0.0
            sentences = split_sentences(sc.spec["narration"])
            for si, sentence in enumerate(sentences):
                samples = narrator.synth(sentence, os.path.join(work, f"s{sc.index:02d}_{si:02d}.wav"))
                a, b = _speech_bounds(samples)
                sc.words += distribute_words(sentence, t + a, t + b)
                chunks.append(samples)
                t += samples.size / SR
                gap = 0.22 if si < len(sentences) - 1 else (1.6 if sc.spec.get("decision") else 0.5)
                chunks.append(np.zeros(int(gap * SR), np.float32))
                t += gap
            sc.end = t
        total = t
        voice_track = np.concatenate(chunks)
        logger.info(f"episode length: {total:.1f}s (voice: {narrator.used})")
        if not 60 <= total <= 90:
            logger.warning(f"Episode ist {total:.1f}s lang (Ziel: 60–90 s)")

        # 2) subtitles
        words = [w for sc in scenes for w in sc.words]
        groups = group_words(words)
        spans = group_timing(groups, total)
        srt_path = os.path.join(out_dir, "untertitel.srt")
        write_srt(groups, spans, srt_path)

        # 3) scene images
        logger.info("## scene images")
        for sc in scenes:
            sc.image, sc.image_source = scene_image(series, sc.spec, images_mode, work)
            sc.image.save(os.path.join(img_dir, f"szene_{sc.index:02d}.jpg"), quality=92)

        # 4) audio
        logger.info("## audio mix")
        audio_path = os.path.join(work, "mix.wav")
        mix_audio(voice_track, scenes, total, audio_path)

        # 5) video
        logger.info("## rendering video")
        renderer = EpisodeRenderer(scenes, groups, spans, series["series_title"], number, spec["question"], total, seed=number)
        video_path = os.path.join(out_dir, f"folge_{number:02d}.mp4")
        tmp_video = video_path + ".part.mp4"
        cmd = [
            _ffmpeg(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-maxrate", "9M", "-bufsize", "18M",
            "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", tmp_video,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_frames = int(total * FPS)
        try:
            for i in range(n_frames):
                proc.stdin.write(renderer.frame(i).tobytes())
                if i % (FPS * 10) == 0:
                    logger.info(f"  frame {i}/{n_frames}")
            proc.stdin.close()
        except BrokenPipeError:
            pass
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg konnte das Video nicht schreiben")
        os.replace(tmp_video, video_path)

        # 6) documents + series state
        write_script(os.path.join(out_dir, "skript.md"), series, spec, scenes, total, narrator.used)
        write_upload(os.path.join(out_dir, "upload.txt"), series, spec)
        with open(os.path.join(out_dir, "episode.json"), "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        if update_series:
            record_episode(slug, spec, out_dir, total)
        return {
            "video": video_path,
            "duration": total,
            "voice": narrator.used,
            "images": sorted({sc.image_source for sc in scenes}),
            "output_dir": out_dir,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def record_episode(slug: str, spec: dict, out_dir: str, duration: float) -> None:
    series = load_series(slug)
    number = int(spec["episode"])
    entry = {
        "episode": number,
        "title": spec.get("title", ""),
        "danger": spec.get("danger", ""),
        "previous_choice": spec.get("previous_choice"),
        "summary": spec.get("summary", ""),
        "cliffhanger": spec.get("cliffhanger", ""),
        "question": spec["question"],
        "winning_answer": None,
        "duration_seconds": round(duration, 1),
        "output_dir": os.path.relpath(out_dir, utils.root_dir()),
    }
    episodes = [e for e in series.get("episodes", []) if e.get("episode") != number]
    episodes.append(entry)
    episodes.sort(key=lambda e: e["episode"])
    series["episodes"] = episodes
    series["story_so_far"] = [
        f"Folge {e['episode']}: "
        + (f"(Zuschauer wählten: {e['previous_choice']}) " if e.get("previous_choice") else "")
        + e["summary"]
        + (f" → Entscheidung: {e['winning_answer']}" if e.get("winning_answer") else "")
        for e in episodes
    ]
    series["pending_question"] = {"episode": number, **spec["question"]}
    save_series(slug, series)


def apply_answer(series: dict, answer: int) -> tuple[dict, str]:
    """Mark the viewers' winning option on the pending question; returns (series, choice text)."""
    pending = series.get("pending_question")
    if not pending:
        raise ValueError("Es gibt keine offene Zuschauerfrage in series.json")
    if answer not in (1, 2):
        raise ValueError("Antwort muss 1 oder 2 sein")
    choice = pending[f"option_{answer}"]
    series = copy.deepcopy(series)
    for e in series["episodes"]:
        if e["episode"] == pending["episode"]:
            e["winning_answer"] = f"{answer} – {choice}"
    return series, f"{answer} – {choice}"


def next_episode_prompt(series: dict, answer: int) -> str:
    """Prompt for the configured LLM to write the next episode spec as JSON."""
    _, choice = apply_answer(series, answer)
    number = series["pending_question"]["episode"] + 1
    characters = "\n".join(f"- {c['id']}: {c['name']} ({c['age']}), {c['role']}" for c in series["characters"])
    story = "\n".join(series.get("story_so_far", []))
    example = {
        "episode": number,
        "title": "…",
        "previous_choice": choice,
        "danger": "…",
        "summary": "2-4 Sätze, was in dieser Folge passiert",
        "cliffhanger": "…",
        "scenes": [
            {"id": 1, "art": "corridor_alarm", "camera": "zoom_in", "transition": "crossfade", "effects": [],
             "sfx": ["water_rush"], "narration": "…", "image_prompt": "English prompt with {JONAS} and {MIA} …"}
        ],
        "question": {"text": "…?", "option_1": "…", "option_2": "…", "call_to_action": "Schreib 1 oder 2 in die Kommentare!"},
        "upload": {"title": "…", "description": "…", "hashtags": ["#…"]},
    }
    return f"""Du schreibst Folge {number} der deutschen Kurzvideo-Serie "{series['series_title']}".

Prämisse: {series['premise']}
Ton: {series['tone']}
Figuren:
{characters}

Bisherige Handlung:
{story}

Die Zuschauer haben entschieden: Option {choice}.
Die Folge beginnt GENAU mit dieser Entscheidung und ihren Folgen.

Anforderungen:
- 60-90 Sekunden gesprochen (ca. 170-210 Wörter Erzählertext insgesamt), Deutsch, Erzählerstimme
- Szene 1: erster Satz ist ein Hook, maximal 7 Wörter, macht sofort neugierig
- kurze, spannende Sätze; neue Gefahr in dieser Folge; Cliffhanger am Ende
- keine blutige oder grafische Gewalt
- 7-10 Szenen; die letzte Szene hat "decision": true und spricht die Frage mit beiden Optionen und
  "Schreib eins oder zwei in die Kommentare!" (Zahlen als Wörter ausschreiben)
- image_prompt auf Englisch; verwende die Platzhalter {{JONAS}} und {{MIA}} statt die Figuren selbst zu beschreiben
- art (Offline-Platzhalterbild) aus: {', '.join(series_art.SCENES)}
- camera aus: {', '.join(CAMERA_MOVES)}; transition aus: {', '.join(TRANSITIONS)}; effects aus: {', '.join(EFFECTS)}
- sfx aus: {', '.join(series_audio.SFX)}

Antworte NUR mit JSON in genau dieser Struktur:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


def generate_next_episode_spec(series: dict, answer: int) -> dict:
    from app.services import llm

    prompt = next_episode_prompt(series, answer)
    response = llm._generate_response(prompt)
    if not response or response.startswith("Error"):
        raise RuntimeError(f"LLM-Antwort unbrauchbar: {response[:200] if response else 'leer'}")
    text = llm._strip_code_fence(response)
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        spec = llm._extract_embedded_json(text, "{")
    spec["episode"] = series["pending_question"]["episode"] + 1
    spec["previous_choice"] = apply_answer(series, answer)[1]
    problems = validate_episode_spec(spec)
    if problems:
        raise ValueError("LLM-Skript ungültig:\n- " + "\n- ".join(problems))
    return spec
