"""Offline placeholder illustrations for series episodes.

Used when no text-to-image source is configured or reachable. Each scene is
drawn procedurally in one consistent, dark "cinematic silhouette" style, and
the main characters always share the same design:

* Jonas: tall, grey hoodie, black jeans, white sneakers, dark-brown hair
* Mia:   small, bright yellow raincoat, blond braids, pink pajamas, grey bunny

Images are rendered at 1.2x the 1080x1920 output so the renderer can zoom and
pan without upscaling.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1296, 2304

JONAS = {
    "hair": (52, 36, 26),
    "skin": (226, 200, 184),
    "hoodie": (112, 118, 126),
    "jeans": (28, 30, 36),
    "shoes": (236, 236, 236),
}
MIA = {
    "hair": (226, 192, 104),
    "skin": (236, 206, 188),
    "coat": (246, 198, 24),
    "pajama": (236, 156, 186),
    "bunny": (150, 150, 158),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _shade(color, f: float):
    return tuple(max(0, min(255, int(c * f))) for c in color)


def _mix(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _vgrad(top, bottom, w=W, h=H, power: float = 1.0) -> Image.Image:
    y = (np.linspace(0, 1, h, dtype=np.float32) ** power)[:, None, None]
    arr = np.array(top, np.float32) * (1 - y) + np.array(bottom, np.float32) * y
    arr = np.broadcast_to(arr, (h, w, 3))
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _glow(img: Image.Image, xy, radius: float, color, strength: float = 1.0, blur: float | None = None):
    layer = Image.new("RGB", img.size, (0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = xy
    d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_shade(color, strength))
    layer = layer.filter(ImageFilter.GaussianBlur(blur if blur is not None else radius * 0.8))
    return ImageChops.add(img, layer)


def _light_layer(img: Image.Image, draw_fn, blur: float, strength: float = 1.0) -> Image.Image:
    layer = Image.new("RGB", img.size, (0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    if strength != 1.0:
        layer = Image.eval(layer, lambda v: int(min(255, v * strength)))
    return ImageChops.add(img, layer)


def _vignette(img: Image.Image, strength: float = 0.75) -> Image.Image:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (xx - W / 2) / (W / 2)
    dy = (yy - H / 2) / (H / 2)
    r = np.sqrt(dx * dx * 1.1 + dy * dy * 0.9)
    mask = 1 - strength * np.clip((r - 0.45) / 0.9, 0, 1) ** 1.6
    arr = np.asarray(img, np.float32) * mask[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _finish(img: Image.Image, seed: int, tint=(1.0, 1.0, 1.0)) -> Image.Image:
    img = img.filter(ImageFilter.GaussianBlur(1.1))
    arr = np.asarray(img, np.float32) * np.array(tint, np.float32)
    # painterly texture: low-frequency brush noise + fine grain
    rng = np.random.default_rng(seed)
    low = Image.fromarray((rng.random((H // 24, W // 24)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    arr *= (0.92 + 0.16 * np.asarray(low, np.float32) / 255)[..., None]
    arr += rng.normal(0, 4, arr.shape[:2])[..., None]
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return _vignette(img)


def _fog(img: Image.Image, y0: int, y1: int, color, alpha: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = Image.fromarray((rng.random((40, 20)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    n = np.asarray(noise, np.float32) / 255
    yy = np.linspace(0, 1, H)[:, None]
    band = np.clip(1 - np.abs((yy * H - (y0 + y1) / 2) / ((y1 - y0) / 2 + 1)), 0, 1)
    a = (band * n * alpha)[..., None]
    arr = np.asarray(img, np.float32) * (1 - a) + np.array(color, np.float32) * a
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# --------------------------------------------------------------------------
# characters (same design in every scene)
# --------------------------------------------------------------------------


def draw_jonas(d: ImageDraw.ImageDraw, cx: float, foot: float, h: float, view: str = "back",
               light: float = 0.55, arm: str = "down", rim=None):
    c = {k: _shade(v, light) for k, v in JONAS.items()}
    head_r = 0.068 * h
    head_y = foot - 0.9 * h
    sh_y = foot - 0.79 * h
    hip_y = foot - 0.46 * h
    tw = 0.27 * h
    lw = 0.095 * h
    # legs
    for s in (-1, 1):
        x = cx + s * 0.065 * h
        d.rounded_rectangle((x - lw / 2, hip_y - 0.02 * h, x + lw / 2, foot - 0.04 * h), radius=lw * 0.4, fill=c["jeans"])
        d.rounded_rectangle((x - lw * 0.62 + s * 0.01 * h, foot - 0.055 * h, x + lw * 0.62 + s * 0.02 * h, foot), radius=lw * 0.3, fill=c["shoes"])
    # arms
    aw = 0.075 * h
    for s in (-1, 1):
        x = cx + s * (tw / 2 + aw * 0.15)
        if (arm == "reach_right" and s == 1) or (arm == "reach_left" and s == -1):
            d.line((x, sh_y + aw * 0.5, x + s * 0.2 * h, sh_y + 0.22 * h), fill=c["hoodie"], width=int(aw))
            d.ellipse((x + s * 0.2 * h - aw * 0.45, sh_y + 0.22 * h - aw * 0.45, x + s * 0.2 * h + aw * 0.45, sh_y + 0.22 * h + aw * 0.45), fill=c["skin"] if view == "front" else _shade(c["skin"], 0.8))
        elif arm == "up" and s == 1:
            d.line((x, sh_y + aw * 0.5, x + 0.05 * h, sh_y - 0.18 * h), fill=c["hoodie"], width=int(aw))
            d.ellipse((x + 0.05 * h - aw * 0.45, sh_y - 0.18 * h - aw * 0.5, x + 0.05 * h + aw * 0.45, sh_y - 0.18 * h + aw * 0.4), fill=_shade(c["skin"], 0.85))
        else:
            d.rounded_rectangle((x - aw / 2, sh_y + 0.01 * h, x + aw / 2, hip_y + 0.06 * h), radius=aw * 0.45, fill=_shade(c["hoodie"], 0.9))
            d.ellipse((x - aw * 0.45, hip_y + 0.03 * h, x + aw * 0.45, hip_y + 0.1 * h), fill=_shade(c["skin"], 0.85))
    # torso (hoodie)
    d.rounded_rectangle((cx - tw / 2, sh_y - 0.02 * h, cx + tw / 2, hip_y + 0.03 * h), radius=tw * 0.28, fill=c["hoodie"])
    d.rectangle((cx - tw / 2 + 2, hip_y - 0.03 * h, cx + tw / 2 - 2, hip_y + 0.03 * h), fill=_shade(c["hoodie"], 0.85))
    if view == "back":
        d.ellipse((cx - 0.11 * h, sh_y - 0.06 * h, cx + 0.11 * h, sh_y + 0.07 * h), fill=_shade(c["hoodie"], 0.85))  # hood
    else:
        d.polygon([(cx - 0.05 * h, sh_y - 0.01 * h), (cx + 0.05 * h, sh_y - 0.01 * h), (cx, sh_y + 0.07 * h)], fill=_shade((30, 44, 86), light))
        d.line((cx - 0.03 * h, sh_y + 0.02 * h, cx - 0.035 * h, sh_y + 0.15 * h), fill=_shade((220, 220, 220), light), width=3)
        d.line((cx + 0.03 * h, sh_y + 0.02 * h, cx + 0.035 * h, sh_y + 0.15 * h), fill=_shade((220, 220, 220), light), width=3)
    # neck + head
    d.rectangle((cx - 0.025 * h, head_y + head_r * 0.6, cx + 0.025 * h, sh_y), fill=_shade(c["skin"], 0.8))
    d.ellipse((cx - head_r * 0.9, head_y - head_r, cx + head_r * 0.9, head_y + head_r * 1.05), fill=c["skin"])
    if view == "back":
        d.ellipse((cx - head_r, head_y - head_r * 1.15, cx + head_r, head_y + head_r * 0.95), fill=c["hair"])
    else:
        d.chord((cx - head_r * 1.02, head_y - head_r * 1.25, cx + head_r * 1.02, head_y + head_r * 0.5), 180, 360, fill=c["hair"])
        for k in range(7):  # messy fringe
            x = cx - head_r * 0.9 + k * head_r * 0.3
            d.polygon([(x, head_y - head_r * 0.45), (x + head_r * 0.32, head_y - head_r * 0.45), (x + head_r * 0.12, head_y - head_r * 0.05)], fill=c["hair"])
        ey = head_y + head_r * 0.12
        for s in (-1, 1):
            d.ellipse((cx + s * head_r * 0.38 - 5, ey - 5, cx + s * head_r * 0.38 + 5, ey + 5), fill=(18, 18, 24))
    if rim:
        d.arc((cx - head_r, head_y - head_r * 1.15, cx + head_r, head_y + head_r), 200, 320, fill=rim, width=4)


def draw_mia(d: ImageDraw.ImageDraw, cx: float, foot: float, h: float, view: str = "back",
             light: float = 0.55, arm: str = "down", bunny: bool = True):
    c = {k: _shade(v, light) for k, v in MIA.items()}
    head_r = 0.1 * h
    head_y = foot - 0.84 * h
    sh_y = foot - 0.7 * h
    hem_y = foot - 0.3 * h
    lw = 0.1 * h
    for s in (-1, 1):
        x = cx + s * 0.07 * h
        d.rounded_rectangle((x - lw / 2, hem_y - 0.05 * h, x + lw / 2, foot - 0.04 * h), radius=lw * 0.4, fill=c["pajama"])
        d.rounded_rectangle((x - lw * 0.6, foot - 0.06 * h, x + lw * 0.6, foot), radius=lw * 0.3, fill=_shade((60, 60, 70), light))
    # raincoat (A-line)
    d.polygon([(cx - 0.15 * h, sh_y), (cx + 0.15 * h, sh_y), (cx + 0.25 * h, hem_y), (cx - 0.25 * h, hem_y)], fill=c["coat"])
    d.rounded_rectangle((cx - 0.16 * h, sh_y - 0.04 * h, cx + 0.16 * h, sh_y + 0.08 * h), radius=0.05 * h, fill=c["coat"])
    if view == "front":
        for k in range(3):
            y = sh_y + 0.08 * h + k * 0.09 * h
            d.ellipse((cx - 7, y - 7, cx + 7, y + 7), fill=_shade((40, 40, 40), light))
    aw = 0.085 * h
    for s in (-1, 1):
        x = cx + s * 0.19 * h
        if (arm == "reach_left" and s == -1) or (arm == "reach_right" and s == 1):
            d.line((cx + s * 0.13 * h, sh_y + 0.05 * h, cx + s * 0.32 * h, sh_y + 0.05 * h), fill=c["coat"], width=int(aw))
        else:
            d.line((cx + s * 0.13 * h, sh_y + 0.04 * h, x + s * 0.03 * h, sh_y + 0.3 * h), fill=_shade(c["coat"], 0.9), width=int(aw))
    if bunny:
        bx = cx + (-0.24 if arm != "reach_left" else 0.24) * h
        by = sh_y + 0.33 * h
        d.ellipse((bx - 0.07 * h, by - 0.05 * h, bx + 0.07 * h, by + 0.09 * h), fill=c["bunny"])
        d.ellipse((bx - 0.05 * h, by - 0.13 * h, bx + 0.05 * h, by - 0.02 * h), fill=c["bunny"])
        for s in (-1, 1):
            d.ellipse((bx + s * 0.03 * h - 0.017 * h, by - 0.27 * h, bx + s * 0.03 * h + 0.017 * h, by - 0.1 * h), fill=c["bunny"])
    # head + braids
    d.ellipse((cx - head_r * 0.92, head_y - head_r, cx + head_r * 0.92, head_y + head_r), fill=c["skin"])
    for s in (-1, 1):
        bx = cx + s * head_r * 0.95
        for k in range(4):
            y = head_y + head_r * (0.2 + k * 0.42)
            d.ellipse((bx - head_r * 0.24, y - head_r * 0.24, bx + head_r * 0.24, y + head_r * 0.24), fill=c["hair"])
    if view == "back":
        d.ellipse((cx - head_r, head_y - head_r * 1.05, cx + head_r, head_y + head_r * 0.95), fill=c["hair"])
        d.line((cx, head_y - head_r, cx, head_y + head_r * 0.9), fill=_shade(c["hair"], 0.75), width=3)
    else:
        d.chord((cx - head_r, head_y - head_r * 1.1, cx + head_r, head_y + head_r * 0.4), 180, 360, fill=c["hair"])
        ey = head_y + head_r * 0.15
        for s in (-1, 1):
            d.ellipse((cx + s * head_r * 0.38 - 5, ey - 6, cx + s * head_r * 0.38 + 5, ey + 6), fill=(18, 18, 24))


def _pair_back(d, cx, foot, h, light=0.5, holding=True):
    """Jonas and Mia seen from behind, holding hands."""
    draw_jonas(d, cx - 0.16 * h, foot, h, "back", light, arm="reach_right" if holding else "down")
    draw_mia(d, cx + 0.12 * h, foot, h * 0.62, "back", light, arm="reach_left" if holding else "down")


# --------------------------------------------------------------------------
# environments
# --------------------------------------------------------------------------


def _corridor(img: Image.Image, vp, wall, floor, ceil, door_color, light_color, n_doors=5, numbers=True):
    d = ImageDraw.Draw(img)
    vx, vy = vp
    bw, bh = W * 0.16, H * 0.1  # back wall half-size
    back = (vx - bw, vy - bh, vx + bw, vy + bh)
    d.polygon([(0, 0), (W, 0), (back[2], back[1]), (back[0], back[1])], fill=ceil)
    d.polygon([(0, H), (W, H), (back[2], back[3]), (back[0], back[3])], fill=floor)
    d.polygon([(0, 0), (back[0], back[1]), (back[0], back[3]), (0, H)], fill=wall)
    d.polygon([(W, 0), (back[2], back[1]), (back[2], back[3]), (W, H)], fill=_shade(wall, 0.85))
    d.rectangle(back, fill=_shade(wall, 0.45))
    font = _font(40)
    for side in (-1, 1):
        for k in range(n_doors):
            t0 = 0.12 + k * 0.17
            t1 = t0 + 0.08
            def pt(t, v):
                # t: 0 at viewer, 1 at back wall; v: 0 floor, 1 top of door
                x_edge = 0 if side < 0 else W
                x_back = back[0] if side < 0 else back[2]
                x = x_edge + (x_back - x_edge) * t
                yf = H + (back[3] - H) * t
                yc = 0 + (back[1] - 0) * t
                return x, yf + (yc - yf) * v * 0.72
            q = [pt(t0, 0), pt(t1, 0), pt(t1, 1), pt(t0, 1)]
            d.polygon(q, fill=door_color)
            if numbers and k < 3:
                x, y = pt((t0 + t1) / 2, 0.85)
                d.text((x, y), str(418 - side * 2 - k * 2 if side < 0 else 419 + k * 2), fill=_shade(light_color, 0.6), font=_font(int(44 * (1 - t0))), anchor="mm")
    for k in range(6):
        t = k / 6
        y = back[1] * t + 30 * (1 - t)
        x = vx
        size = 70 * (1 - t) + 12
        img.paste(_glow(img, (x, y + 40 * (1 - t)), size, light_color, 0.9), (0, 0))
    return img


def _font(size):
    from app.utils import utils
    return ImageFont.truetype(os.path.join(utils.font_dir(), "BeVietnamPro-Bold.ttf"), max(8, size))


def scene_ship_night(seed=1):
    img = _vgrad((2, 5, 14), (10, 26, 46), power=0.8)
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)
    for _ in range(220):
        x, y = rng.random() * W, rng.random() * H * 0.45
        r = rng.random() * 2.2
        v = int(120 + rng.random() * 120)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(v, v, int(v * 1.05)))
    img = _glow(img, (W * 0.74, H * 0.17), 90, (230, 235, 255), 1.0, blur=10)
    img = _glow(img, (W * 0.74, H * 0.17), 260, (60, 80, 120), 0.8)
    d = ImageDraw.Draw(img)
    horizon = H * 0.56
    d.rectangle((0, horizon, W, H), fill=(3, 9, 18))
    for k in range(160):  # moonlight on water
        y = horizon + (k / 160) ** 1.6 * (H - horizon)
        w = 30 + k * 2.2 + rng.random() * 60
        x = W * 0.74 + (rng.random() - 0.5) * 80 * (1 + k / 40)
        d.line((x - w / 2, y, x + w / 2, y), fill=(90 - k // 3, 110 - k // 3, 150 - k // 3), width=2)
    # ship, slightly tilted
    ship = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    s = ImageDraw.Draw(ship)
    base = horizon + 40
    hull = [(W * 0.03, base - 210), (W * 0.97, base - 250), (W * 0.9, base + 40), (W * 0.1, base + 40)]
    s.polygon(hull, fill=(20, 26, 36, 255))
    s.line([(W * 0.03, base - 210), (W * 0.97, base - 250)], fill=(150, 160, 175, 255), width=6)
    for deck in range(5):
        y0 = base - 250 - deck * 78
        x0 = W * (0.14 + deck * 0.06)
        x1 = W * (0.86 - deck * 0.05)
        s.rectangle((x0, y0 - 78, x1, y0), fill=(30 + deck * 4, 38 + deck * 4, 50 + deck * 4, 255))
        for k in range(int((x1 - x0) / 34)):
            if rng.random() < 0.72:
                wx = x0 + 14 + k * 34
                s.rectangle((wx, y0 - 52, wx + 16, y0 - 28), fill=(255, 205, 120, 255))
    s.polygon([(W * 0.58, base - 640), (W * 0.7, base - 640), (W * 0.67, base - 820), (W * 0.6, base - 820)], fill=(26, 30, 40, 255))
    s.rectangle((W * 0.6, base - 820, W * 0.67, base - 790), fill=(150, 30, 30, 255))
    for k in range(int(W * 0.8 / 40)):
        if rng.random() < 0.5:
            wx = W * 0.12 + k * 40
            s.ellipse((wx, base - 150, wx + 14, base - 136), fill=(255, 210, 140, 255))
    s.text((W * 0.2, base - 110), "AURORA", fill=(170, 175, 185, 255), font=_font(54))
    ship = ship.rotate(-2.2, center=(W / 2, base), resample=Image.BICUBIC)
    glow = ship.split()[0].point(lambda v: 255 if v > 200 else 0).filter(ImageFilter.GaussianBlur(14))
    img.paste(ship, (0, 0), ship)
    img = ImageChops.add(img, Image.merge("RGB", (glow, glow.point(lambda v: int(v * 0.7)), glow.point(lambda v: int(v * 0.35)))))
    # reflection of the ship lights
    refl = ship.transpose(Image.FLIP_TOP_BOTTOM).filter(ImageFilter.GaussianBlur(8))
    refl_shift = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    refl_shift.paste(refl, (0, int(2 * base - H + 80)))
    r, g, b, a = refl_shift.split()
    img.paste(refl_shift.convert("RGB"), (0, 0), a.point(lambda v: int(v * 0.25)))
    img = _fog(img, int(horizon - 250), int(horizon + 260), (60, 76, 100), 0.55, seed)
    return _finish(img, seed, (0.9, 0.97, 1.08))


def scene_cabin(seed=2):
    img = _vgrad((10, 16, 30), (6, 9, 18))
    d = ImageDraw.Draw(img)
    # porthole with moonlit sea
    px, py, pr = W * 0.62, H * 0.26, 170
    d.ellipse((px - pr - 26, py - pr - 26, px + pr + 26, py + pr + 26), fill=(60, 64, 72))
    d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(20, 36, 64))
    d.rectangle((px - pr, py + 40, px + pr, py + pr), fill=(8, 18, 34))
    img = _glow(img, (px + 60, py - 60), 36, (230, 235, 255), 1.0, blur=6)
    img = _light_layer(img, lambda l: l.polygon([(px - pr, py - pr), (px + pr, py - pr), (W * 0.2, H * 0.95), (-W * 0.2, H * 0.95)], fill=(40, 60, 95)), 90, 0.6)
    d = ImageDraw.Draw(img)
    # raincoat on hook (Mia's identity)
    hx, hy = W * 0.14, H * 0.3
    d.ellipse((hx - 10, hy - 10, hx + 10, hy + 10), fill=(120, 120, 120))
    d.polygon([(hx - 70, hy + 20), (hx + 70, hy + 20), (hx + 110, hy + 380), (hx - 110, hy + 380)], fill=_shade(MIA["coat"], 0.55))
    d.ellipse((hx - 72, hy - 10, hx + 72, hy + 90), fill=_shade(MIA["coat"], 0.5))
    # Mia's bed (right, sleeping)
    d.rounded_rectangle((W * 0.5, H * 0.62, W * 1.05, H * 0.74), radius=30, fill=(36, 44, 66))
    d.rounded_rectangle((W * 0.52, H * 0.585, W * 0.72, H * 0.635), radius=30, fill=(150, 156, 172))
    mh = 150
    d.ellipse((W * 0.6 - mh / 2, H * 0.565 - mh / 2, W * 0.6 + mh / 2, H * 0.565 + mh / 2 + 10), fill=_shade(MIA["skin"], 0.55))
    d.chord((W * 0.6 - mh / 2 - 6, H * 0.565 - mh / 2 - 8, W * 0.6 + mh / 2 + 6, H * 0.565 + mh / 2), 160, 380, fill=_shade(MIA["hair"], 0.6))
    for k in range(5):
        bx = W * 0.6 + mh / 2 + 10 + k * 34
        d.ellipse((bx - 20, H * 0.585 - 20, bx + 20, H * 0.585 + 20), fill=_shade(MIA["hair"], 0.6))
    d.rounded_rectangle((W * 0.55, H * 0.6, W * 1.02, H * 0.67), radius=50, fill=(70, 84, 120))
    d.ellipse((W * 0.66, H * 0.575, W * 0.74, H * 0.62), fill=_shade(MIA["bunny"], 0.6))
    # Jonas sitting at the edge of his bed, lit by his phone
    draw_jonas(d, W * 0.3, H * 0.86, H * 0.42, "front", 0.55, arm="reach_right")
    d.rounded_rectangle((-30, H * 0.72, W * 0.48, H * 0.83), radius=26, fill=(30, 36, 54))
    phone = (W * 0.37, H * 0.6)
    d.rounded_rectangle((phone[0] - 34, phone[1] - 58, phone[0] + 34, phone[1] + 58), radius=10, fill=(190, 220, 255))
    d.text(phone, "3:12", fill=(20, 30, 60), font=_font(26), anchor="mm")
    img = _glow(img, phone, 220, (70, 110, 170), 0.9)
    return _finish(img, seed, (0.85, 0.95, 1.12))


def scene_glass(seed=3):
    img = _vgrad((14, 18, 30), (4, 6, 12))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # table surface
    d.polygon([(-200, H * 0.62), (W + 200, H * 0.55), (W + 200, H * 1.1), (-200, H * 1.1)], fill=(58, 40, 30, 255))
    for k in range(14):
        y = H * 0.6 + k * 70
        d.line((-200, y + 40, W + 200, y - 40 + k * 2), fill=(70, 50, 38, 255), width=3)
    # glass
    gx, gy = W * 0.5, H * 0.585
    gw, gh = 210, 460
    d.polygon([(gx - gw, gy - gh), (gx + gw, gy - gh), (gx + gw * 0.8, gy), (gx - gw * 0.8, gy)], fill=(120, 150, 180, 70))
    d.line([(gx - gw, gy - gh), (gx - gw * 0.8, gy)], fill=(200, 220, 240, 200), width=6)
    d.line([(gx + gw, gy - gh), (gx + gw * 0.8, gy)], fill=(170, 190, 210, 160), width=4)
    d.ellipse((gx - gw, gy - gh - 30, gx + gw, gy - gh + 30), outline=(200, 220, 240, 200), width=5)
    layer = layer.rotate(9, center=(W / 2, H * 0.6), resample=Image.BICUBIC)
    img.paste(layer, (0, 0), layer)
    d = ImageDraw.Draw(img)
    # water stays level while the table tilts
    wy = gy - gh * 0.45
    d.polygon([(gx - 196, wy - 20), (gx + 205, wy - 20), (gx + 160, gy + 30), (gx - 150, gy - 10)], fill=(40, 80, 120))
    d.line((gx - 196, wy - 20, gx + 205, wy - 20), fill=(170, 210, 240), width=5)
    rng = random.Random(seed)
    for _ in range(14):
        x = gx + 180 + rng.random() * 260
        y = gy - gh * 0.6 + rng.random() * 500
        r = 6 + rng.random() * 12
        d.ellipse((x - r, y - r * 1.3, x + r, y + r), fill=(120, 170, 210))
    d.ellipse((gx + 120, gy + 40, gx + 520, gy + 130), fill=(30, 50, 70))
    img = _glow(img, (W * 0.85, H * 0.18), 360, (40, 60, 100), 0.8)
    return _finish(img, seed, (0.9, 0.98, 1.1))


def scene_corridor_alarm(seed=4):
    img = _vgrad((30, 6, 8), (12, 4, 6))
    img = _corridor(img, (W * 0.5, H * 0.44), (70, 22, 24), (34, 16, 18), (46, 14, 16), (96, 34, 34), (255, 60, 50))
    d = ImageDraw.Draw(img)
    floor_y = H * 0.93
    draw_jonas(d, W * 0.42, floor_y, H * 0.56, "front", 0.62, arm="reach_right", rim=(255, 80, 70))
    draw_mia(d, W * 0.7, floor_y - H * 0.02, H * 0.34, "front", 0.68, arm="reach_left")
    img = _light_layer(img, lambda l: l.rectangle((0, 0, W, H * 0.12), fill=(140, 20, 20)), 120, 0.8)
    return _finish(img, seed, (1.08, 0.9, 0.9))


def scene_stairs_water(seed=5):
    img = _vgrad((18, 10, 14), (3, 6, 10))
    d = ImageDraw.Draw(img)
    vx = W * 0.5
    for k in range(22):
        t = k / 22
        y = H * 0.18 + t ** 0.9 * H * 0.8
        half = W * (0.2 + 0.32 * t)
        step_h = 18 + 40 * t
        d.rectangle((vx - half, y, vx + half, y + step_h), fill=_mix((40, 30, 34), (70, 52, 54), t))
        d.line((vx - half, y, vx + half, y), fill=_mix((90, 60, 60), (160, 110, 100), t), width=3)
    for s in (-1, 1):
        d.line((vx + s * W * 0.2, H * 0.18, vx + s * W * 0.55, H * 1.0), fill=(110, 90, 90), width=10)
        for k in range(10):
            t = k / 10
            x = vx + s * (W * 0.2 + W * 0.35 * t)
            y = H * 0.18 + t * H * 0.82
            d.line((x, y, x, y - 120 - 140 * t), fill=(90, 70, 72), width=6)
    # black water rising from below (the bottom of the stairs is up in frame)
    water_top = H * 0.38
    wl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    w = ImageDraw.Draw(wl)
    w.rectangle((0, 0, W, water_top), fill=(2, 6, 10, 245))
    rng = random.Random(seed)
    for k in range(70):
        y = rng.random() * water_top
        x = rng.random() * W
        w.line((x, y, x + 40 + rng.random() * 160, y), fill=(160, 40, 40, 110), width=2)
    w.line((0, water_top, W, water_top), fill=(140, 60, 60, 160), width=4)
    img.paste(wl, (0, 0), wl)
    d = ImageDraw.Draw(img)
    # Jonas and Mia halfway up, seen from behind looking down
    _pair_back(d, W * 0.52, H * 0.98, H * 0.5, light=0.55)
    img = _light_layer(img, lambda l: l.ellipse((W * 0.2, -200, W * 0.8, 300), fill=(160, 30, 30)), 160, 0.7)
    return _finish(img, seed, (1.02, 0.92, 0.95))


def scene_speaker(seed=6):
    img = _vgrad((24, 28, 34), (8, 10, 14))
    d = ImageDraw.Draw(img)
    for k in range(9):  # wall panels
        x = k * W / 8
        d.line((x, 0, x, H), fill=(36, 40, 46), width=6)
    d.line((0, H * 0.82, W, H * 0.82), fill=(50, 54, 60), width=10)
    cx, cy = W * 0.5, H * 0.4
    d.rounded_rectangle((cx - 240, cy - 300, cx + 240, cy + 240), radius=40, fill=(66, 70, 74))
    d.ellipse((cx - 200, cy - 230, cx + 200, cy + 170), fill=(30, 32, 36))
    for k in range(-9, 10):
        d.line((cx - 190, cy - 30 + k * 20, cx + 190, cy - 30 + k * 20), fill=(52, 56, 60), width=6)
    d.rectangle((cx - 12, cy - 520, cx + 12, cy - 300), fill=(50, 52, 56))
    d.rounded_rectangle((cx - 170, cy + 330, cx + 170, cy + 430), radius=12, fill=(200, 180, 60))
    d.text((cx, cy + 380), "DECK 4", fill=(30, 30, 30), font=_font(58), anchor="mm")
    rng = random.Random(seed)
    for _ in range(16):
        x0, y0 = cx + 150 + rng.random() * 120, cy - 260 + rng.random() * 120
        pts = [(x0, y0)]
        for _ in range(4):
            x0 += (rng.random() - 0.3) * 60
            y0 += rng.random() * 60
            pts.append((x0, y0))
        d.line(pts, fill=(200, 220, 255), width=3)
    img = _glow(img, (cx + 200, cy - 200), 120, (120, 160, 255), 0.9)
    img = _glow(img, (W * 0.5, -100), 500, (60, 70, 90), 0.8)
    return _finish(img, seed, (0.95, 1.0, 1.08))


def scene_fire_door(seed=7):
    img = _vgrad((40, 12, 14), (12, 6, 8))
    d = ImageDraw.Draw(img)
    dx0, dy0, dx1, dy1 = W * 0.12, H * 0.14, W * 0.88, H * 0.84
    d.rectangle((dx0 - 60, dy0 - 60, dx1 + 60, dy1), fill=(60, 30, 30))
    stripe_w = 70
    for k in range(-30, 30):
        x = dx0 - 60 + k * stripe_w * 2
        d.polygon([(x, dy0 - 60), (x + stripe_w, dy0 - 60), (x + stripe_w - 60, dy0), (x - 60, dy0)], fill=(210, 170, 40))
    d.rounded_rectangle((dx0, dy0, dx1, dy1), radius=30, fill=(96, 98, 104))
    for k in range(1, 6):
        y = dy0 + k * (dy1 - dy0) / 6
        d.line((dx0 + 30, y, dx1 - 30, y), fill=(80, 82, 88), width=5)
    wx, wy = W * 0.5, H * 0.45
    d.ellipse((wx - 170, wy - 170, wx + 170, wy + 170), outline=(150, 150, 156), width=26)
    for a in range(0, 360, 60):
        d.line((wx, wy, wx + 170 * math.cos(math.radians(a)), wy + 170 * math.sin(math.radians(a))), fill=(140, 140, 146), width=20)
    d.ellipse((wx - 40, wy - 40, wx + 40, wy + 40), fill=(150, 150, 156))
    d.rounded_rectangle((W * 0.27, dy0 + 70, W * 0.73, dy0 + 170), radius=14, fill=(180, 30, 30))
    d.text((W * 0.5, dy0 + 120), "BRANDSCHUTZ", fill=(255, 240, 240), font=_font(60), anchor="mm")
    d.rectangle((0, dy1, W, H), fill=(26, 14, 16))
    draw_jonas(d, W * 0.42, H * 0.99, H * 0.52, "back", 0.5, arm="up")
    draw_mia(d, W * 0.62, H * 0.99, H * 0.32, "back", 0.52)
    img = _light_layer(img, lambda l: l.ellipse((W * 0.3, -150, W * 0.7, 150), fill=(255, 40, 40)), 90, 0.9)
    return _finish(img, seed, (1.08, 0.92, 0.92))


def scene_fork(seed=8):
    img = _vgrad((8, 14, 20), (4, 8, 12))
    d = ImageDraw.Draw(img)
    mid = W * 0.5
    # left: flooded passage glowing teal
    d.polygon([(0, H * 0.12), (mid - 40, H * 0.34), (mid - 40, H * 0.66), (0, H)], fill=(18, 40, 46))
    d.polygon([(mid - 420, H * 0.4), (mid - 120, H * 0.43), (mid - 120, H * 0.6), (mid - 420, H * 0.64)], fill=(6, 18, 22))
    d.polygon([(0, H * 0.72), (mid - 40, H * 0.58), (mid - 40, H * 0.68), (0, H)], fill=(20, 110, 120))
    rng = random.Random(seed)
    for _ in range(60):
        x = rng.random() * (mid - 60)
        y = H * 0.6 + rng.random() * H * 0.35
        d.line((x, y, x + 30 + rng.random() * 80, y - 10), fill=(90, 210, 210), width=2)
    d.rounded_rectangle((mid - 400, H * 0.36, mid - 140, H * 0.395), radius=6, fill=(30, 160, 80))
    d.text((mid - 270, H * 0.378), "RETTUNGSSTATION", fill=(220, 255, 230), font=_font(22), anchor="mm")
    img = _light_layer(img, lambda l: l.polygon([(0, H * 0.68), (mid - 40, H * 0.56), (mid - 40, H * 0.7), (0, H)], fill=(20, 140, 150)), 60, 0.8)
    d = ImageDraw.Draw(img)
    # right: narrow ladder shaft into darkness
    d.polygon([(W, H * 0.05), (mid + 40, H * 0.3), (mid + 40, H * 0.72), (W, H)], fill=(22, 24, 30))
    sx0, sx1 = mid + 200, mid + 400
    d.rectangle((sx0, 0, sx1, H * 0.7), fill=(4, 4, 6))
    for x in (sx0 + 30, sx1 - 30):
        d.line((x, 0, x, H * 0.7), fill=(120, 120, 126), width=10)
    for k in range(14):
        y = H * 0.7 - k * 110
        d.line((sx0 + 30, y, sx1 - 30, y), fill=(110, 110, 116), width=8)
    img = _glow(img, ((sx0 + sx1) / 2, 40), 90, (200, 190, 160), 0.6)
    d = ImageDraw.Draw(img)
    d.rectangle((mid - 40, H * 0.3, mid + 40, H * 0.72), fill=(30, 32, 36))
    d.polygon([(0, H), (W, H), (mid + 40, H * 0.72), (mid - 40, H * 0.72)], fill=(18, 20, 24))
    _pair_back(d, W * 0.5, H * 0.99, H * 0.42, light=0.55)
    return _finish(img, seed, (0.92, 1.0, 1.05))


def scene_water_shadow(seed=9):
    img = _vgrad((4, 16, 22), (1, 5, 8))
    rng = np.random.default_rng(seed)
    # the huge pale shape deep under the surface (blurred, never clearly visible)
    layer = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse((W * 0.18, H * 0.48, W * 0.95, H * 0.7), fill=(60, 100, 100))
    d.ellipse((W * 0.05, H * 0.55, W * 0.4, H * 0.63), fill=(50, 90, 90))
    layer = layer.filter(ImageFilter.GaussianBlur(70))
    img = ImageChops.add(img, layer)
    img = _glow(img, (W * 0.63, H * 0.56), 10, (150, 230, 220), 0.7, blur=8)
    img = _glow(img, (W * 0.71, H * 0.565), 10, (150, 230, 220), 0.7, blur=8)
    d = ImageDraw.Draw(img)
    for k in range(9):  # ripples
        r = 60 + k * 85
        d.ellipse((W * 0.6 - r, H * 0.56 - r * 0.4, W * 0.6 + r, H * 0.56 + r * 0.4), outline=(40 + k * 2, 90 - k * 4, 96 - k * 4), width=3)
    # reflections of Jonas and Mia at the top edge (flipped, rippled)
    refl = Image.new("RGB", (W, int(H * 0.36)), (0, 0, 0))
    rd = ImageDraw.Draw(refl)
    draw_jonas(rd, W * 0.36, refl.height * 0.02 + H * 0.33, H * 0.3, "front", 0.45)
    draw_mia(rd, W * 0.56, refl.height * 0.02 + H * 0.33, H * 0.19, "front", 0.5)
    refl = refl.transpose(Image.FLIP_TOP_BOTTOM)
    arr = np.asarray(refl)
    rows = np.arange(arr.shape[0])
    shift = (np.sin(rows / 9.0) * 6).astype(int)
    arr = np.stack([np.roll(arr[i], shift[i], axis=0) for i in rows])
    refl = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(2))
    refl = Image.eval(refl, lambda v: int(v * 0.6))
    img = ImageChops.add(img, Image.fromarray(np.pad(np.asarray(refl), ((0, H - refl.height), (0, 0), (0, 0)))))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.random() * W, rng.random() * H
        d.line((x, y, x + 60, y), fill=(30, 70, 76), width=2)
    return _finish(img, seed, (0.9, 1.02, 1.04))


def scene_decision(seed=10):
    img = scene_fork(seed)
    return Image.eval(img.filter(ImageFilter.GaussianBlur(3)), lambda v: int(v * 0.75))


SCENES = {
    "ship_night": scene_ship_night,
    "cabin": scene_cabin,
    "glass": scene_glass,
    "corridor_alarm": scene_corridor_alarm,
    "stairs_water": scene_stairs_water,
    "speaker": scene_speaker,
    "fire_door": scene_fire_door,
    "fork": scene_fork,
    "water_shadow": scene_water_shadow,
    "decision": scene_decision,
}


def render(name: str, seed: int = 0) -> Image.Image:
    """Render a named placeholder scene at W x H. Unknown names fall back to the ship."""
    fn = SCENES.get(name, scene_ship_night)
    return fn(seed + 1) if seed else fn()
