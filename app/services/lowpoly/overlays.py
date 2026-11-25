"""Game-show style overlays: monster wheel, survive-the-night clock, A/B question card."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .captions import DEFAULT_FONT

STRINGS = {
    "en": {"wheel_title": "TONIGHT'S MONSTER", "clock": "SURVIVE THE NIGHT", "cta": "COMMENT A OR B"},
    "de": {"wheel_title": "MONSTER DER NACHT", "clock": "ÜBERLEBE DIE NACHT", "cta": "KOMMENTIERE A ODER B"},
}


def strings(language: str) -> dict:
    return STRINGS.get(language, STRINGS["en"])


WHEEL_COLORS = ["#ff4d4d", "#ffb703", "#3ddc97", "#3d9bff", "#b04dff", "#ff6fb5", "#2ec4b6", "#f77f00"]


def _font(size: int, path: str | None = None):
    return ImageFont.truetype(path or DEFAULT_FONT, size)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def blend(frame: np.ndarray, img: np.ndarray, x0: int, y0: int, opacity: float = 1.0) -> np.ndarray:
    """Alpha-blend an RGBA image onto an RGB frame (clipped to the frame)."""
    h, w = img.shape[:2]
    H, W = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0 or opacity <= 0:
        return frame
    sub = img[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0 * opacity
    region = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame = frame if frame.flags.writeable else frame.copy()
    frame[fy0:fy1, fx0:fx1] = (region * (1 - a) + sub[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    return frame


def _text_img(text: str, size: int, fill=(255, 255, 255), stroke_ratio: float = 0.12, max_width: int | None = None):
    font = _font(size)
    stroke = max(3, int(size * stroke_ratio))
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = (line + " " + w).strip()
        if max_width and font.getbbox(trial, stroke_width=stroke)[2] > max_width and line:
            lines.append(line)
            line = w
        else:
            line = trial
    lines.append(line)
    widths = [font.getbbox(ln, stroke_width=stroke)[2] for ln in lines]
    line_h = int(size * 1.12)
    pad = stroke * 2
    img = Image.new("RGBA", (max(widths) + pad * 2, line_h * len(lines) + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text(((img.width - widths[i]) // 2, pad + i * line_h), ln, font=font, fill=fill + (255,),
               stroke_width=stroke, stroke_fill=(12, 12, 16, 255))
    return img


# --------------------------------------------------------------------------- #
# Wheel of monsters
# --------------------------------------------------------------------------- #


class MonsterWheel:
    def __init__(self, width: int, height: int, options: list[str], target: int, start: float = 0.3,
                 spin: float = 3.6, hold: float = 1.3, title: str = "TONIGHT'S MONSTER", turns: int = 5):
        self.width, self.height = width, height
        self.options, self.target = options, target
        self.start, self.spin, self.hold = start, spin, hold
        self.end = start + spin + hold
        n = len(options)
        self.seg = 360.0 / n
        self.final = turns * 360.0 + (target + 0.5) * self.seg
        self.size = int(width * 0.84)
        self.wheel = self._draw_wheel()
        self.highlight = self._draw_highlight()
        self.title = np.asarray(_text_img(title, int(width * 0.075)))
        self.result = np.asarray(_text_img(options[target] + "!", int(width * 0.11), fill=(255, 225, 60),
                                           max_width=int(width * 0.9)))
        self.pointer = self._draw_pointer()

    def _draw_wheel(self) -> Image.Image:
        s = self.size
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        rim = int(s * 0.035)
        d.ellipse((0, 0, s - 1, s - 1), fill=(30, 26, 44, 255))
        box = (rim, rim, s - rim, s - rim)
        for i, label in enumerate(self.options):
            a0 = -90 + i * self.seg
            d.pieslice(box, a0, a0 + self.seg, fill=_hex(WHEEL_COLORS[i % len(WHEEL_COLORS)]) + (255,),
                       outline=(255, 255, 255, 255), width=max(2, s // 220))
        # labels along the radius
        font_size = int(s * 0.055)
        for i, label in enumerate(self.options):
            mid = -90 + (i + 0.5) * self.seg
            txt = _text_img(label.upper(), font_size, max_width=int(s * 0.36))
            rot = txt.rotate(-mid, expand=True, resample=Image.BICUBIC)
            r = s * 0.29
            cx = s / 2 + math.cos(math.radians(mid)) * r
            cy = s / 2 + math.sin(math.radians(mid)) * r
            img.alpha_composite(rot, (int(cx - rot.width / 2), int(cy - rot.height / 2)))
        # rim lights
        for k in range(24):
            ang = math.radians(k * 15)
            cx, cy = s / 2 + math.cos(ang) * (s / 2 - rim / 2), s / 2 + math.sin(ang) * (s / 2 - rim / 2)
            rr = rim * 0.32
            d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(255, 240, 170, 255))
        hub = s * 0.08
        d.ellipse((s / 2 - hub, s / 2 - hub, s / 2 + hub, s / 2 + hub), fill=(255, 255, 255, 255),
                  outline=(30, 26, 44, 255), width=max(3, s // 120))
        return img

    def _draw_highlight(self) -> np.ndarray:
        s = self.size
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        rim = int(s * 0.035)
        a0 = -90 - self.seg / 2
        d.pieslice((rim, rim, s - rim, s - rim), a0, a0 + self.seg, fill=(255, 255, 255, 110))
        return np.asarray(img)

    def _draw_pointer(self) -> np.ndarray:
        w = int(self.size * 0.12)
        img = Image.new("RGBA", (w, int(w * 1.1)), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.polygon([(4, 4), (w - 4, 4), (w // 2, int(w * 1.05))], fill=(255, 255, 255, 255), outline=(20, 20, 30, 255))
        d.polygon([(w * 0.22, w * 0.14), (w * 0.78, w * 0.14), (w // 2, w * 0.82)], fill=(255, 60, 60, 255))
        return np.asarray(img)

    def angle(self, t: float) -> float:
        u = min(max((t - self.start) / self.spin, 0.0), 1.0)
        return self.final * (1 - (1 - u) ** 3.2)

    def tick_times(self, step: float = 1 / 120) -> list[float]:
        ticks, last = [], 0
        t = self.start
        while t <= self.start + self.spin:
            k = int(self.angle(t) // self.seg)
            if k != last:
                ticks.append(t)
                last = k
            t += step
        return ticks

    @property
    def land_time(self) -> float:
        return self.start + self.spin

    def composite(self, frame: np.ndarray, t: float) -> np.ndarray:
        if not (0 <= t < self.end):
            return frame
        fade_out = min(1.0, (self.end - t) / 0.35)
        fade_in = min(1.0, t / 0.2)
        k = min(fade_in, fade_out)
        frame = (frame.astype(np.float32) * (1 - 0.5 * k)).astype(np.uint8)
        cy = int(self.height * 0.43)
        cx = self.width // 2
        pop = 0.85 + 0.15 * min(1.0, t / 0.25)
        a = self.angle(t)
        rot = self.wheel.rotate(a, resample=Image.BILINEAR)
        size = int(self.size * pop)
        if size != self.size:
            rot = rot.resize((size, size), Image.BILINEAR)
        frame = blend(frame, np.asarray(rot), cx - size // 2, cy - size // 2, k)
        landed = t >= self.land_time
        if landed and int((t - self.land_time) * 8) % 2 == 0:
            hl = self.highlight if size == self.size else np.asarray(Image.fromarray(self.highlight).resize((size, size)))
            frame = blend(frame, hl, cx - size // 2, cy - size // 2, k)
        p = self.pointer
        frame = blend(frame, p, cx - p.shape[1] // 2, cy - size // 2 - int(p.shape[0] * 0.35), k)
        title = self.title
        frame = blend(frame, title, cx - title.shape[1] // 2, cy - size // 2 - title.shape[0] - int(self.height * 0.03), k)
        if landed:
            age = t - self.land_time
            s = 0.6 + 0.5 * min(1.0, age / 0.12) - 0.1 * min(1.0, max(0.0, age - 0.12) / 0.12)
            res = Image.fromarray(self.result)
            res = res.resize((max(1, int(res.width * s)), max(1, int(res.height * s))), Image.BILINEAR)
            r = np.asarray(res)
            frame = blend(frame, r, cx - r.shape[1] // 2, cy + size // 2 + int(self.height * 0.01), k)
        return frame


# --------------------------------------------------------------------------- #
# Survive-the-night clock
# --------------------------------------------------------------------------- #


class NightClock:
    def __init__(self, width: int, height: int, start: float, end: float, hold_end: float,
                 from_minutes: int = 22 * 60, to_minutes: int = 5 * 60 + 59, label: str = "SURVIVE THE NIGHT",
                 keyframes: list[tuple[float, int]] | None = None, h24: bool = False):
        self.width, self.height = width, height
        self.h24 = h24
        self.start, self.end, self.hold_end = start, end, hold_end
        self.from_m = from_minutes
        span = (to_minutes - from_minutes) % (24 * 60)
        self.span = span
        # keyframes: (video time, minutes since from_minutes) – the clock follows the narration
        keys = [(start, 0)] + [(t, (m - from_minutes) % (24 * 60)) for t, m in (keyframes or [])] + [(end, span)]
        keys.sort()
        cleaned = []
        for t, m in keys:
            if cleaned and m < cleaned[-1][1]:
                m = cleaned[-1][1]
            cleaned.append((t, m))
        self._kt = np.array([k[0] for k in cleaned])
        self._km = np.array([k[1] for k in cleaned], dtype=float)
        self.label = np.asarray(_text_img(label, int(width * 0.04), fill=(255, 220, 120)))
        self._cache: dict[tuple, np.ndarray] = {}

    def minutes(self, t: float) -> int:
        offset = float(np.interp(t, self._kt, self._km))
        return int(self.from_m + offset) % (24 * 60)

    def fmt(self, m: int) -> str:
        h, mm = divmod(m, 60)
        if self.h24:
            return f"{h:02d}:{mm:02d}"
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12}:{mm:02d} {suffix}"

    def _clock_img(self, text: str, urgent: bool) -> np.ndarray:
        key = (text, urgent)
        if key not in self._cache:
            size = int(self.width * 0.085)
            font = _font(size)
            tw = font.getbbox(text)[2]
            pad = int(size * 0.35)
            moon = int(size * 0.8)
            w, h = tw + moon + pad * 3, int(size * 1.35)
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            bg = (170, 20, 30, 215) if urgent else (14, 16, 36, 200)
            d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=bg, outline=(255, 255, 255, 90), width=3)
            my = (h - moon) // 2
            d.ellipse((pad, my, pad + moon, my + moon), fill=(255, 236, 160, 255))
            d.ellipse((pad + moon * 0.35, my - moon * 0.1, pad + moon * 1.25, my + moon * 0.8), fill=bg)
            d.text((pad * 2 + moon, (h - size) // 2 - size * 0.08), text, font=font, fill=(255, 255, 255, 255))
            self._cache[key] = np.asarray(img)
        return self._cache[key]

    def composite(self, frame: np.ndarray, t: float) -> np.ndarray:
        if not (self.start <= t < self.hold_end):
            return frame
        k = min(1.0, (t - self.start) / 0.3)
        m = self.minutes(t)
        urgent = t >= self.end and int(t * 3) % 2 == 0
        img = self._clock_img(self.fmt(m), urgent)
        cx, y = self.width // 2, int(self.height * 0.085)
        frame = blend(frame, self.label, cx - self.label.shape[1] // 2, y, k)
        return blend(frame, img, cx - img.shape[1] // 2, y + self.label.shape[0] - 4, k)


# --------------------------------------------------------------------------- #
# A/B decision card
# --------------------------------------------------------------------------- #


@dataclass
class Question:
    text: str
    a: str
    b: str


class QuestionCard:
    def __init__(self, width: int, height: int, question: Question, start: float, end: float,
                 cta: str = "COMMENT A OR B"):
        self.width, self.height = width, height
        self.start, self.end = start, end
        self.cta = cta
        self.card = self._draw(question)

    def _draw(self, q: Question) -> np.ndarray:
        W = self.width
        cw = int(W * 0.9)
        title = _text_img(q.text.upper(), int(W * 0.07), max_width=int(cw * 0.92))
        opt_size = int(W * 0.06)
        opts = []
        for letter, txt, color in (("A", q.a, "#ff4d4d"), ("B", q.b, "#3d9bff")):
            label = _text_img(f"{letter}: {txt.upper()}", opt_size, max_width=int(cw * 0.8))
            bw, bh = cw, label.height + int(opt_size * 0.5)
            box = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
            d = ImageDraw.Draw(box)
            d.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=bh // 3, fill=_hex(color) + (235,),
                                outline=(255, 255, 255, 255), width=5)
            box.alpha_composite(label, ((bw - label.width) // 2, (bh - label.height) // 2))
            opts.append(box)
        cta = _text_img(self.cta, int(W * 0.05), fill=(255, 225, 60))
        gap = int(W * 0.03)
        h = title.height + sum(o.height for o in opts) + cta.height + gap * 4
        card = Image.new("RGBA", (cw, h), (0, 0, 0, 0))
        y = 0
        card.alpha_composite(title, ((cw - title.width) // 2, y))
        y += title.height + gap
        for o in opts:
            card.alpha_composite(o, (0, y))
            y += o.height + gap
        card.alpha_composite(cta, ((cw - cta.width) // 2, y + gap // 2))
        return np.asarray(card)

    def composite(self, frame: np.ndarray, t: float) -> np.ndarray:
        if not (self.start <= t < self.end):
            return frame
        age = t - self.start
        k = min(1.0, age / 0.25)
        dim = min(1.0, age / 0.4) * 0.35
        frame = (frame.astype(np.float32) * (1 - dim)).astype(np.uint8)
        img = self.card
        x = (self.width - img.shape[1]) // 2
        y = int(self.height * 0.17) - int((1 - k) * 60)
        return blend(frame, img, x, y, k)
