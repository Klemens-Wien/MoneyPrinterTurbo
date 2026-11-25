"""Word-by-word "pop" captions: huge rounded font, bright fill, thick black outline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .narration import Word

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_FONT = os.path.join(ROOT, "resource", "fonts", "LilitaOne-Regular.ttf")

FILL = (236, 255, 58)
EMPHASIS_FILL = (255, 255, 255)
STROKE = (12, 12, 16)


def clean_word(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\"'“”„«»(\[]+|[\"'“”„«»)\],.;:]+$", "", text)
    return text


def is_emphasis(word: str) -> bool:
    bare = re.sub(r"\W", "", word)
    return bool(bare) and (bare.isdigit() or (bare.isupper() and len(bare) > 1) or word.endswith("!"))


@dataclass
class TimedWord:
    text: str
    start: float
    end: float
    emphasis: bool = False


def build_timeline(words: list[Word], min_dur: float = 0.18) -> list[TimedWord]:
    out: list[TimedWord] = []
    for w in words:
        text = clean_word(w.text)
        if not text:
            continue
        out.append(TimedWord(text, w.start, max(w.end, w.start + min_dur), is_emphasis(w.text)))
    # each word stays until the next one starts (no flicker), capped for pauses
    for a, b in zip(out, out[1:]):
        if b.start - a.start < 1.6:
            a.end = b.start
        else:
            a.end = min(a.end + 0.25, b.start)
    if out:
        out[-1].end += 0.4
    return out


class CaptionRenderer:
    def __init__(self, width: int, height: int, font_path: str | None = None, size: int | None = None,
                 y_frac: float = 0.64, case: str = "lower"):
        self.width, self.height = width, height
        self.font_path = font_path or DEFAULT_FONT
        self.size = size or int(width * 0.135)
        self.y_frac = y_frac
        self.case = case
        self._cache: dict[tuple, np.ndarray] = {}

    def _font(self, size: int):
        try:
            return ImageFont.truetype(self.font_path, size)
        except OSError:
            return ImageFont.load_default(size)

    def _apply_case(self, text: str) -> str:
        if self.case == "upper":
            return text.upper()
        if self.case == "lower":
            return text if re.sub(r"\W", "", text).isupper() and len(text) > 1 else text.lower()
        return text

    def word_image(self, word: TimedWord) -> np.ndarray:
        key = (word.text, word.emphasis)
        if key in self._cache:
            return self._cache[key]
        text = self._apply_case(word.text)
        size = int(self.size * (1.12 if word.emphasis else 1.0))
        font = self._font(size)
        stroke = max(6, size // 9)
        max_w = self.width * 0.9
        while True:
            bbox = font.getbbox(text, stroke_width=stroke)
            if bbox[2] - bbox[0] <= max_w or size < 40:
                break
            size = int(size * 0.9)
            stroke = max(5, size // 9)
            font = self._font(size)
        pad = stroke * 3
        w, h = bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2
        origin = (pad - bbox[0], pad - bbox[1])
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).text((origin[0] + stroke * 0.5, origin[1] + stroke * 0.9), text, font=font,
                                    fill=(0, 0, 0, 150), stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
        shadow = shadow.filter(ImageFilter.GaussianBlur(stroke * 0.6))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fill = EMPHASIS_FILL if word.emphasis else FILL
        ImageDraw.Draw(layer).text(origin, text, font=font, fill=fill + (255,), stroke_width=stroke,
                                   stroke_fill=STROKE + (255,))
        img = Image.alpha_composite(shadow, layer)
        arr = np.asarray(img, dtype=np.uint8)
        self._cache[key] = arr
        return arr

    @staticmethod
    def pop_scale(age: float) -> float:
        """Overshoot pop-in: 0.6 -> 1.12 -> 1.0 within ~0.2s."""
        if age < 0:
            return 0.0
        if age < 0.09:
            return 0.6 + (1.12 - 0.6) * (age / 0.09)
        if age < 0.2:
            return 1.12 - 0.12 * ((age - 0.09) / 0.11)
        return 1.0

    def composite(self, frame: np.ndarray, word: TimedWord | None, t: float, y_frac: float | None = None) -> np.ndarray:
        if word is None:
            return frame
        img = self.word_image(word)
        s = self.pop_scale(t - word.start)
        if s <= 0:
            return frame
        if abs(s - 1.0) > 1e-3:
            h, w = img.shape[:2]
            img = np.asarray(Image.fromarray(img).resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR))
        h, w = img.shape[:2]
        cx, cy = self.width // 2, int(self.height * (y_frac if y_frac is not None else self.y_frac))
        x0, y0 = cx - w // 2, cy - h // 2
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(self.width, x0 + w), min(self.height, y0 + h)
        if fx1 <= fx0 or fy1 <= fy0:
            return frame
        sub = img[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0]
        alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
        region = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        frame = frame if frame.flags.writeable else frame.copy()
        frame[fy0:fy1, fx0:fx1] = (region * (1 - alpha) + sub[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        return frame


def active_word(words: list[TimedWord], t: float, cursor: int = 0) -> tuple[TimedWord | None, int]:
    """Find the word visible at time t (words sorted). Returns (word, new_cursor)."""
    i = cursor
    while i + 1 < len(words) and words[i + 1].start <= t:
        i += 1
    if words and words[i].start <= t < words[i].end:
        return words[i], i
    return None, i


def to_srt(words: list[TimedWord], path: str, per_line: int = 4) -> None:
    def ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    lines = []
    for n, i in enumerate(range(0, len(words), per_line), start=1):
        chunk = words[i : i + per_line]
        lines += [str(n), f"{ts(chunk[0].start)} --> {ts(chunk[-1].end)}", " ".join(w.text for w in chunk), ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


class HookBanner:
    """Short headline at the top of the first seconds ("In this town ...")."""

    def __init__(self, width: int, height: int, text: str, start: float = 0.0, end: float = 3.5,
                 font_path: str | None = None, y_frac: float = 0.15):
        self.width, self.height = width, height
        self.start, self.end = start, end
        self.y_frac = y_frac
        self.image = self._render(text.upper(), font_path or DEFAULT_FONT) if text else None

    def _render(self, text: str, font_path: str) -> np.ndarray:
        size = int(self.width * 0.085)
        font = ImageFont.truetype(font_path, size)
        stroke = max(5, size // 8)
        words, lines, line = text.split(), [], ""
        for w in words:
            trial = (line + " " + w).strip()
            if font.getbbox(trial, stroke_width=stroke)[2] > self.width * 0.86 and line:
                lines.append(line)
                line = w
            else:
                line = trial
        lines.append(line)
        line_h = int(size * 1.12)
        pad = stroke * 3
        img = Image.new("RGBA", (self.width, line_h * len(lines) + pad * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for i, ln in enumerate(lines):
            bbox = font.getbbox(ln, stroke_width=stroke)
            x = (self.width - (bbox[2] - bbox[0])) // 2 - bbox[0]
            draw.text((x, pad + i * line_h), ln, font=font, fill=(255, 255, 255, 255),
                      stroke_width=stroke, stroke_fill=STROKE + (255,))
        return np.asarray(img, dtype=np.uint8)

    def composite(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self.image is None or not (self.start <= t < self.end):
            return frame
        age, left = t - self.start, self.end - t
        alpha_k = min(1.0, age / 0.15, left / 0.3)
        img = self.image
        h = img.shape[0]
        y0 = int(self.height * self.y_frac) - h // 2 - int(30 * max(0.0, 1 - age / 0.2))
        y0 = max(0, y0)
        y1 = min(self.height, y0 + h)
        sub = img[: y1 - y0]
        alpha = sub[:, :, 3:4].astype(np.float32) / 255.0 * alpha_k
        region = frame[y0:y1].astype(np.float32)
        frame = frame if frame.flags.writeable else frame.copy()
        frame[y0:y1] = (region * (1 - alpha) + sub[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        return frame
