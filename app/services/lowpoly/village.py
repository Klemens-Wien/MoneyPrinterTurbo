"""Alpine mountain village (Tyrol style) for winter / Rauhnacht stories.

Snowy ground, chalets with dark wooden upper floors, balconies and flower boxes, an onion-dome church,
a fountain and a maypole on the square, lanterns along the lane, snowy fir forest and a ring of
snow-capped peaks. Like the other districts it sits far away from the main town.
"""

from __future__ import annotations

import math

import numpy as np

from .mesh import MeshBuilder, hex_color, rotate_y, translate, vary
from .props import PALETTE

VILLAGE = np.array([1500.0, 0.0, 0.0])
HOME_LOCAL = (32.0, 0.0, -16.0)

SNOW = hex_color("#eef3fb")
SNOW_SHADE = hex_color("#dfe7f4")
PATH = hex_color("#c9c3bc")
WOOD = hex_color("#6b4226")
WOOD_DARK = hex_color("#4a2c18")
PLASTER = hex_color("#f4efe6")
STONE = hex_color("#9a9488")
ROOF = hex_color("#4a3a32")
WINDOW = PALETTE["window"]


def world(local) -> np.ndarray:
    return VILLAGE + np.asarray(local, dtype=np.float64)


def _chalet(mb: MeshBuilder, x, z, rot, rng, obstacles, big=False, home=False):
    w = rng.uniform(10, 12) if not big else 16.0
    d = rng.uniform(8, 9.5) if not big else 12.0
    shutter = [hex_color("#2f7a3a"), hex_color("#b3202a"), hex_color("#2f5a8a")][rng.integers(3)]
    if home:
        shutter = hex_color("#b3202a")
    with mb.transform(translate(x, 0, z) @ rotate_y(rot)):
        mb.box((0, 1.6, 0), (w, 3.2, d), PLASTER)  # ground floor (plaster)
        mb.box((0, 0.3, 0), (w + 0.2, 0.6, d + 0.2), STONE)
        mb.box((0, 4.7, 0), (w + 0.1, 3.0, d + 0.1), WOOD)  # wooden upper floor
        for k in np.arange(-w / 2 + 0.4, w / 2, 0.8):  # plank lines
            mb.box((k, 4.7, d / 2 + 0.07), (0.06, 3.0, 0.04), WOOD_DARK)
        # windows with shutters
        for fy in (1.8, 4.8):
            for wx in np.linspace(-w / 2 + 2.0, w / 2 - 2.0, 3 if w < 14 else 4):
                mb.box((wx, fy, d / 2 + 0.06), (1.1, 1.2, 0.1), WINDOW)
                for s_ in (-1, 1):
                    mb.box((wx + s_ * 0.85, fy, d / 2 + 0.08), (0.5, 1.3, 0.08), shutter)
                if fy > 3:  # flower boxes
                    mb.box((wx, fy - 0.85, d / 2 + 0.25), (1.4, 0.35, 0.35), WOOD_DARK)
                    mb.box((wx, fy - 0.6, d / 2 + 0.25), (1.3, 0.25, 0.3), hex_color("#d6303a"))
        # balcony
        mb.box((0, 3.4, d / 2 + 0.9), (w * 0.85, 0.2, 1.6), WOOD)
        for k in np.arange(-w * 0.42, w * 0.42 + 0.1, 0.5):
            mb.box((k, 3.95, d / 2 + 1.65), (0.12, 0.9, 0.08), WOOD_DARK)
        mb.box((0, 4.45, d / 2 + 1.65), (w * 0.85, 0.12, 0.12), WOOD_DARK)
        mb.box((0, 1.25, d / 2 + 0.06), (1.4, 2.4, 0.12), WOOD_DARK)  # door
        # wide gable roof with snow on top (ridge along x)
        with mb.transform(rotate_y(math.pi / 2)):
            mb.gable_roof((0, 6.2, 0), d, w, d * 0.32, ROOF, overhang=1.4)
            mb.gable_roof((0, 6.35, 0), d, w, d * 0.32, SNOW, overhang=1.3)
        mb.box((w * 0.25, 8.4, -d * 0.1), (0.9, 1.8, 0.9), STONE)  # chimney
        if rng.random() < 0.6:  # woodpile
            mb.box((-w / 2 - 0.8, 0.8, 0), (1.2, 1.6, d * 0.6), hex_color("#8a5a34"))
            mb.box((-w / 2 - 0.8, 1.7, 0), (1.3, 0.2, d * 0.62), SNOW)
    obstacles.append((x, z, max(w, d) / 2 + 1.5, 10.0, -1.0))


def _church(mb: MeshBuilder, obstacles):
    with mb.transform(translate(0, 0, -34)):
        mb.box((0, 5.0, 0), (12, 10, 20), PLASTER)
        with mb.transform(rotate_y(0)):
            mb.gable_roof((0, 10, 0), 12, 20, 4.5, hex_color("#5a4a44"), overhang=0.6)
            mb.gable_roof((0, 10.15, 0), 12, 20, 4.5, SNOW, overhang=0.5)
        for zz in (-5, 0, 5):
            for s_ in (-1, 1):
                mb.box((s_ * 6.05, 6.0, zz), (0.1, 3.0, 1.2), hex_color("#5a7ab8"))
        # tower with onion dome
        mb.box((0, 9.0, 11.5), (5, 18, 5), PLASTER)
        mb.cylinder((0, 13.5, 14.05), 1.3, 0.12, hex_color("#f2c232"), segments=12, axis="z")
        mb.box((0, 13.9, 14.12), (0.15, 0.9, 0.06), hex_color("#222222"))
        mb.icosphere((0, 19.6, 11.5), 2.8, hex_color("#3f8a6a"), squash=(1.0, 1.15, 1.0), subdivisions=1)
        mb.cylinder((0, 22.5, 11.5), 0.8, 1.6, hex_color("#3f8a6a"), segments=8)
        mb.icosphere((0, 23.8, 11.5), 1.0, hex_color("#3f8a6a"), squash=(1, 1.2, 1))
        mb.cone((0, 24.6, 11.5), 0.5, 2.6, hex_color("#3f8a6a"), segments=6)
        mb.box((0, 27.6, 11.5), (0.12, 1.4, 0.12), hex_color("#f2c232"))
        mb.box((0, 27.8, 11.5), (0.8, 0.12, 0.12), hex_color("#f2c232"))
        mb.box((0, 1.6, 14.06), (1.8, 3.2, 0.1), WOOD_DARK)
    obstacles.append((0.0, -34.0, 12.0, 28.0, -1.0))


def _square(mb: MeshBuilder):
    mb.cylinder((0, 0.45, 6), 3.2, 0.9, STONE, segments=12)  # fountain
    mb.cylinder((0, 0.93, 6), 2.8, 0.06, hex_color("#bfe3ff"), segments=12)
    mb.cylinder((0, 1.6, 6), 0.35, 2.2, STONE, segments=8)
    mb.cylinder((0, 3.0, 6), 0.9, 0.3, STONE, segments=8)
    # maypole with blue-white spiral bands
    for k in range(16):
        mb.cylinder((-9, 1.0 + k * 1.2, 14), 0.25, 1.2, hex_color("#3d6bd6") if k % 2 else hex_color("#f4f6ff"), segments=6)
    mb.cone((-9, 20.2, 14), 1.5, 2.5, hex_color("#2f7a4a"), segments=7)


def _lantern(mb: MeshBuilder, x, z):
    mb.box((x, 1.6, z), (0.15, 3.2, 0.15), hex_color("#2a2a2a"))
    mb.box((x, 3.4, z), (0.55, 0.6, 0.55), WINDOW)
    mb.cone((x, 3.7, z), 0.45, 0.4, hex_color("#2a2a2a"), segments=4)


def _snow_pine(mb: MeshBuilder, x, z, size, rng):
    mb.cylinder((x, 0.6 * size, z), 0.25 * size, 1.2 * size, PALETTE["trunk"], segments=5, cap=False)
    green = vary(hex_color("#2c6b4a"), rng, 0.03)
    with mb.swaying(0.5):
        for i, (r, h) in enumerate(((1.7, 2.5), (1.3, 2.2), (0.9, 1.9))):
            y = (1.0 + i * 1.3) * size
            mb.cone((x, y, z), r * size, h * size, green, segments=7, rng=rng, jitter=0.1)
            mb.cone((x, y + h * size * 0.45, z), r * size * 0.55, h * size * 0.55, SNOW, segments=7, rng=rng, jitter=0.1)


def _peaks(mb: MeshBuilder, rng):
    for k in range(18):
        ang = k / 18 * 2 * math.pi + rng.uniform(-0.1, 0.1)
        dist = rng.uniform(420, 700)
        x, z = math.cos(ang) * dist, math.sin(ang) * dist
        h = rng.uniform(220, 480)
        r = rng.uniform(180, 300)
        mb.cone((x, -5, z), r, h, vary(hex_color("#8f8aa0"), rng, 0.04), segments=7, rng=rng, jitter=0.18)
        mb.cone((x, h * 0.55 - 5, z), r * 0.47, h * 0.47, SNOW, segments=7, rng=rng, jitter=0.12)


def build_static(seed: int = 9) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    local_obs: list = []
    with mb.transform(translate(*VILLAGE)):
        mb.box_min((-1100, -0.3, -1100), (1100, 0.0, 1100), SNOW)
        for _ in range(160):  # gentle snow drifts
            x, z = rng.uniform(-300, 300, 2)
            if abs(z) < 12 or math.hypot(x, z) < 40:
                continue
            mb.icosphere((x, -0.6, z), rng.uniform(2, 6), SNOW_SHADE, rng=rng, jitter=0.2, squash=(1.6, 0.35, 1.2))
        mb.box_min((-160, 0.0, -4), (160, 0.03, 4), PATH)  # main lane
        mb.box_min((-4, 0.0, -24), (4, 0.03, 4), PATH)
        mb.box_min((-16, 0.0, -14), (16, 0.03, 18), hex_color("#d8d2ca"))  # square
        _square(mb)
        _church(mb, local_obs)
        _peaks(mb, rng)
        for x in np.arange(-130, 131, 18.0):
            for side in (1, -1):
                if abs(x) < 20:
                    continue
                z = side * 16.0
                home = abs(x - HOME_LOCAL[0]) < 1 and side < 0
                big = abs(x + 22) < 1 and side > 0  # the Gasthaus
                _chalet(mb, x, z, 0.0 if side < 0 else math.pi, rng, local_obs, big=big, home=home)
                if rng.random() < 0.55 and not home:
                    _chalet(mb, x + rng.uniform(-2, 2), side * 38.0, 0.0 if side < 0 else math.pi, rng, local_obs)
        for x in np.arange(-125, 130, 14.0):
            for z in (5.2, -5.2):
                if abs(x) < 14:
                    continue
                _lantern(mb, x + (3 if z > 0 else -3), z)
        for _ in range(900):
            x, z = rng.uniform(-330, 330, 2)
            r = math.hypot(x, z)
            if r < 60 and abs(z) < 50 or abs(z) < 9 or (abs(x) < 150 and abs(z) < 52):
                continue
            if rng.random() > min(1.0, (r - 50) / 120):
                continue
            _snow_pine(mb, x, z, rng.uniform(1.0, 1.8), rng)
    obstacles = [(o[0] + VILLAGE[0], o[1] + VILLAGE[2], o[2], o[3], o[4]) for o in local_obs]
    return mb.build(), obstacles
