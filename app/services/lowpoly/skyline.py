""""Skyline" district: the rich live on a giant platform in the sky, the poor below in its shadow.

Lower city: cramped shacks, laundry lines, barrels, a dusty ground.
Upper city (70 m up on huge pillars): white-and-gold villas, gardens, pools, a golden palace tower and a
field of golden sun mirrors. One golden elevator connects both levels; an old maintenance ladder runs up
pillar 7. Like the bedroom, the district sits far outside the town beyond the fog.
"""

from __future__ import annotations

import math

import numpy as np

from . import props
from .mesh import MeshBuilder, hex_color, rotate_x, rotate_y, translate, vary
from .props import PALETTE

CITY = np.array([0.0, 0.0, -1500.0])
TOP = 70.0  # upper deck surface height
DECK = 100.0  # half size of the platform (x, z in [-DECK, DECK])
PILLARS = [(-70.0, -70.0), (-70.0, 0.0), (-70.0, 70.0), (0.0, -70.0), (0.0, 70.0), (70.0, -70.0), (70.0, 70.0),
           (70.0, 0.0)]  # the last one is "pillar 7" with the ladder
PILLAR_HALF = 5.0
LADDER_X = 70.0 + PILLAR_HALF + 0.25  # ladder on the +x face of pillar 7
LIFT_POS = np.array([-DECK - 7.0, 0.0, 0.0])  # elevator shaft just outside the deck edge

GOLD = hex_color("#f2c232")
GOLD_DARK = hex_color("#c99a1e")
CONCRETE = hex_color("#7d7f8c")
CONCRETE_DARK = hex_color("#5c5e6a")
WINDOW = PALETTE["window"]  # windows glow at night through the shader


def world(local) -> np.ndarray:
    return CITY + np.asarray(local, dtype=np.float64)


# --------------------------------------------------------------------------- #
# lower city
# --------------------------------------------------------------------------- #

SHACK_COLORS = [hex_color(c) for c in ("#8a6f5a", "#7a7f86", "#9a7b4f", "#6f7d6a", "#8c5f4f", "#6a7690", "#a08c6a")]
ROOF_COLORS = [hex_color(c) for c in ("#6b6f78", "#8a5a44", "#5a6a6a", "#7a7a5a")]
LAUNDRY = [hex_color(c) for c in ("#ff4d4d", "#3d9bff", "#ffd23f", "#3ddc97", "#ffffff", "#ff6fb5")]


def _shack(mb: MeshBuilder, x: float, z: float, rng: np.random.Generator, obstacles: list) -> None:
    w, d = rng.uniform(4, 7), rng.uniform(4, 6.5)
    floors = 1 if rng.random() < 0.6 else 2
    h = 3.0 * floors
    rot = rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.08, 0.08)
    wall = vary(SHACK_COLORS[rng.integers(len(SHACK_COLORS))], rng, 0.04)
    roof = vary(ROOF_COLORS[rng.integers(len(ROOF_COLORS))], rng, 0.03)
    with mb.transform(translate(x, 0, z) @ rotate_y(rot)):
        mb.box((0, h / 2, 0), (w, h, d), wall)
        with mb.transform(translate(0, h + 0.15, 0) @ rotate_x(rng.uniform(-0.12, 0.12))):
            mb.box((0, 0, 0), (w + 0.8, 0.2, d + 0.8), roof)  # corrugated sheet roof
        for f in range(floors):
            for side in (-1, 1):
                if rng.random() < 0.8:
                    mb.box((side * w * 0.25, 1.6 + f * 3.0, d / 2 + 0.03), (1.0, 0.9, 0.1), WINDOW)
        mb.box((0, 1.0, d / 2 + 0.04), (1.0, 2.0, 0.08), PALETTE["door"])
        if floors == 2 and rng.random() < 0.6:  # rooftop clutter
            mb.box((w * 0.2, h + 0.9, 0), (1.2, 1.4, 1.2), hex_color("#4a6fa5"))
        if rng.random() < 0.5:
            mb.cylinder((w / 2 + 0.6, 0.6, d * 0.3), 0.45, 1.2, hex_color("#4e7a52"), segments=7)  # barrel
    obstacles.append((x, z, max(w, d) / 2 + 0.8, h + 1.5, -1.0))


def _laundry(mb: MeshBuilder, a, b, rng: np.random.Generator) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = float(np.linalg.norm(d[[0, 2]]))
    if length < 2:
        return
    mid = (a + b) / 2
    ang = math.atan2(-d[2], d[0])
    with mb.transform(translate(*mid) @ rotate_y(ang)):
        mb.box((0, 0, 0), (length, 0.04, 0.04), hex_color("#2a2a2a"))
        for k in np.arange(-length / 2 + 0.6, length / 2 - 0.4, 0.9):
            if rng.random() < 0.7:
                mb.box((k, -0.45, 0), (0.6, 0.8, 0.05), LAUNDRY[rng.integers(len(LAUNDRY))])


# --------------------------------------------------------------------------- #
# upper city
# --------------------------------------------------------------------------- #


def _villa(mb: MeshBuilder, x: float, z: float, rng: np.random.Generator, obstacles: list) -> None:
    w, d = rng.uniform(10, 14), rng.uniform(8, 11)
    rot = rng.uniform(0, 2 * math.pi)
    white = hex_color("#fbfaf4")
    with mb.transform(translate(x, TOP, z) @ rotate_y(rot)):
        mb.box((0, 0.3, 0), (w + 2, 0.6, d + 2), hex_color("#e8e2d0"))
        mb.box((0, 2.6, 0), (w, 4.0, d), white)
        mb.box((-w * 0.15, 6.4, -d * 0.1), (w * 0.6, 3.6, d * 0.7), white)
        mb.box((0, 4.75, 0), (w + 0.4, 0.3, d + 0.4), GOLD)
        mb.box((-w * 0.15, 8.35, -d * 0.1), (w * 0.6 + 0.4, 0.3, d * 0.7 + 0.4), GOLD)
        for k in range(3):
            mb.box((-w / 2 + 2 + k * (w - 4) / 2, 2.8, d / 2 + 0.03), (2.0, 2.4, 0.1), WINDOW)
        mb.box((w * 0.3, 6.5, d * 0.25 + 0.03), (1.6, 1.8, 0.1), WINDOW)
        if rng.random() < 0.7:  # pool
            mb.box((w / 2 + 4, 0.35, 0), (5, 0.3, d * 0.8), hex_color("#e8e2d0"))
            mb.box((w / 2 + 4, 0.52, 0), (4.2, 0.1, d * 0.8 - 0.8), hex_color("#3fd0ff"))
        for side in (-1, 1):
            mb.cylinder((side * (w / 2 - 0.4), 2.6, d / 2 + 0.6), 0.25, 4.0, white, segments=8)
    obstacles.append((x, z, max(w, d) / 2 + 3.0, TOP + 10.0, TOP - 1.0))


def _palace(mb: MeshBuilder, obstacles: list) -> None:
    white = hex_color("#fbfaf4")
    with mb.transform(translate(0, TOP, 0)):
        mb.box((0, 0.5, 0), (34, 1.0, 34), hex_color("#e8e2d0"))
        mb.box((0, 5.0, 0), (26, 8.0, 26), white)
        mb.box((0, 9.2, 0), (27, 0.4, 27), GOLD)
        for lvl, s in enumerate((18, 13, 9)):
            mb.box((0, 12.0 + lvl * 6.0, 0), (s, 5.6, s), white)
            mb.box((0, 15.0 + lvl * 6.0, 0), (s + 0.6, 0.4, s + 0.6), GOLD)
            for side in (-1, 1):
                mb.box((0, 12.0 + lvl * 6.0, side * (s / 2 + 0.03)), (s * 0.7, 2.6, 0.1), WINDOW)
                mb.box((side * (s / 2 + 0.03), 12.0 + lvl * 6.0, 0), (0.1, 2.6, s * 0.7), WINDOW)
        mb.cone((0, 30.4, 0), 5.0, 9.0, GOLD, segments=4)
        mb.box((0, 41.0, 0), (0.3, 4.0, 0.3), GOLD_DARK)
    obstacles.append((0.0, 0.0, 19.0, TOP + 42.0, TOP - 1.0))


def _mirrors(mb: MeshBuilder, x0: float, z0: float, x1: float, z1: float) -> None:
    """A field of tilted golden sun mirrors."""
    for x in np.arange(x0, x1, 5.0):
        for z in np.arange(z0, z1, 4.0):
            with mb.transform(translate(x, TOP, z)):
                mb.box((0, 0.6, 0), (0.25, 1.2, 0.25), CONCRETE_DARK)
                with mb.transform(translate(0, 1.4, 0) @ rotate_x(-0.55)):
                    mb.box((0, 0, 0), (3.8, 0.12, 2.6), GOLD)
                    mb.box((0, 0.07, 0), (3.5, 0.02, 2.3), hex_color("#fff0a8"))


def _deck(mb: MeshBuilder, rng: np.random.Generator, obstacles: list) -> None:
    D = DECK
    mb.box_min((-D, TOP - 6.0, -D), (D, TOP, D), CONCRETE)
    mb.box_min((-D + 0.5, TOP - 6.2, -D + 0.5), (D - 0.5, TOP - 6.0, D - 0.5), CONCRETE_DARK)
    # underside lights (glow only at night through the window shader)
    for x in np.arange(-D + 10, D, 20):
        for z in np.arange(-D + 10, D, 20):
            mb.box((x, TOP - 6.3, z), (1.6, 0.15, 1.6), WINDOW)
    mb.box_min((-D, TOP, -D), (D, TOP + 0.05, D), hex_color("#9be27a"))  # lawn
    for x in np.arange(-D, D + 0.1, 2.0):  # golden railing
        for z in (-D, D):
            mb.box((x, TOP + 0.6, z), (0.12, 1.2, 0.12), GOLD)
    for z in np.arange(-D, D + 0.1, 2.0):
        for x in (-D, D):
            mb.box((x, TOP + 0.6, z), (0.12, 1.2, 0.12), GOLD)
    for c in (-D, D):
        mb.box_min((-D, TOP + 1.15, c - 0.1), (D, TOP + 1.3, c + 0.1), GOLD)
        mb.box_min((c - 0.1, TOP + 1.15, -D), (c + 0.1, TOP + 1.3, D), GOLD)
    # golden boulevards
    mb.box_min((-D, TOP + 0.05, -4), (D, TOP + 0.1, 4), hex_color("#f4e7c0"))
    mb.box_min((-4, TOP + 0.05, -D), (4, TOP + 0.1, D), hex_color("#f4e7c0"))
    # maintenance hatch where the ladder comes out
    mb.box((LADDER_X, TOP + 0.08, 0), (2.4, 0.1, 2.4), hex_color("#2a2c34"))
    mb.box((LADDER_X - 1.3, TOP + 0.5, 0), (0.15, 1.0, 2.4), hex_color("#8a8f98"))
    _palace(mb, obstacles)
    _mirrors(mb, -92, -92, -30, -30)
    _mirrors(mb, 30, -92, 92, -30)
    taken = []
    for _ in range(80):
        x, z = rng.uniform(-90, 90, 2)
        if abs(x) < 26 and abs(z) < 26 or abs(x) < 8 or abs(z) < 8 or (z < -28 and abs(x) > 28):
            continue
        if any(math.hypot(x - a, z - b) < 18 for a, b in taken):
            continue
        if math.hypot(x - 85, z - 10) < 14 or math.hypot(x - 20, z - 30) < 16 or math.hypot(x + 60, z + 22) < 14:
            continue  # keep story stages clear
        taken.append((x, z))
        _villa(mb, x, z, rng, obstacles)
    for _ in range(140):  # garden trees
        x, z = rng.uniform(-95, 95, 2)
        if any(math.hypot(x - a, z - b) < 10 for a, b in taken) or abs(x) < 20 and abs(z) < 20:
            continue
        if z < -28 and abs(x) > 28 or abs(x) < 5 or abs(z) < 5:
            continue
        props.tree_round(mb, (x, TOP, z), rng.uniform(0.9, 1.3), rng)


def _pillars_and_lift(mb: MeshBuilder, obstacles: list) -> None:
    for x, z in PILLARS:
        mb.box((x, (TOP - 6) / 2, z), (PILLAR_HALF * 2, TOP - 6, PILLAR_HALF * 2), CONCRETE)
        for y in np.arange(8, TOP - 8, 12):
            mb.box((x, y, z), (PILLAR_HALF * 2 + 0.4, 0.5, PILLAR_HALF * 2 + 0.4), CONCRETE_DARK)
        obstacles.append((x, z, PILLAR_HALF * 1.5, TOP - 6, -1.0))
    # number "7" painted on pillar 7 (blocky)
    px = 70.0 + PILLAR_HALF + 0.05
    mb.box((px, 9.0, -1.2), (0.1, 0.6, 2.4), hex_color("#ffd23f"))
    mb.box((px, 7.0, 0.9), (0.1, 4.0, 0.6), hex_color("#ffd23f"))
    # maintenance ladder up pillar 7
    for side in (-0.5, 0.5):
        mb.box((LADDER_X, (TOP - 6) / 2 + 3, side), (0.12, TOP, 0.12), hex_color("#8a8f98"))
    for y in np.arange(0.4, TOP, 0.45):
        mb.box((LADDER_X, y, 0), (0.1, 0.07, 1.0), hex_color("#9aa0a8"))
    # golden elevator shaft (cabin is animated separately)
    lx, _, lz = LIFT_POS
    for cx in (-3.5, 3.5):
        for cz in (-3.5, 3.5):
            mb.box((lx + cx, TOP / 2 + 4, lz + cz), (0.6, TOP + 8, 0.6), GOLD)
    for y in np.arange(0, TOP + 8, 8):
        mb.box_min((lx - 3.8, y, lz - 3.8), (lx + 3.8, y + 0.3, lz + 3.8), GOLD_DARK)
    mb.box_min((lx + 3.5, TOP - 0.2, lz - 3), (-DECK, TOP + 0.05, lz + 3), hex_color("#f4e7c0"))  # bridge
    mb.box((lx, TOP + 10, lz), (8.4, 1.2, 8.4), GOLD)
    # gate at the bottom
    mb.box_min((lx - 12, 0, lz - 9), (lx - 4.5, 0.25, lz + 9), hex_color("#c9c6d6"))
    for s in (-1, 1):
        mb.box((lx - 6.5, 2.5, lz + s * 5.5), (1.2, 5.0, 1.2), GOLD)
    mb.box((lx - 6.5, 5.3, lz), (1.2, 0.8, 12.2), GOLD)
    obstacles.append((lx, lz, 6.5, TOP + 12, -1.0))


def lift_cabin_mesh() -> np.ndarray:
    mb = MeshBuilder()
    mb.box((0, 0.2, 0), (6.6, 0.4, 6.6), GOLD)
    mb.box((0, 5.8, 0), (6.6, 0.4, 6.6), GOLD)
    mb.box((0, 3.0, 0), (6.4, 5.2, 6.4), hex_color("#bfe9ff"))
    for cx in (-3.2, 3.2):
        for cz in (-3.2, 3.2):
            mb.box((cx, 3.0, cz), (0.3, 5.6, 0.3), GOLD_DARK)
    return mb.build()


def build_static(seed: int = 3) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    obstacles: list = []
    mb = MeshBuilder()
    with mb.transform(translate(*CITY)):
        mb.box_min((-1200, -0.3, -1200), (1200, 0.0, 1200), hex_color("#6e6a64"))  # dusty ground
        mb.box_min((-170, 0.0, -4), (170, 0.03, 4), hex_color("#55545a"))  # main alley
        mb.box_min((-4, 0.0, -170), (4, 0.03, 170), hex_color("#55545a"))
    local_obs: list = []
    with mb.transform(translate(*CITY)):
        _pillars_and_lift(mb, local_obs)
        _deck(mb, rng, local_obs)
        clear = [(-20.0, 148.0, 12.0), (-40.0, 40.0, 9.0), (LADDER_X + 3, 0.0, 10.0), (LIFT_POS[0] - 12, 0.0, 16.0)]
        positions = []
        for x in np.arange(-165, 166, 8.5):
            for z in np.arange(-165, 166, 8.5):
                x2, z2 = x + rng.uniform(-1.5, 1.5), z + rng.uniform(-1.5, 1.5)
                if abs(x2) < 7 or abs(z2) < 7:
                    continue
                if any(abs(x2 - px) < 10 and abs(z2 - pz) < 10 for px, pz in PILLARS):
                    continue
                if any(math.hypot(x2 - cx, z2 - cz) < r for cx, cz, r in clear):
                    continue
                if math.hypot(x2 - LIFT_POS[0], z2 - LIFT_POS[2]) < 10:
                    continue
                if rng.random() < 0.12:
                    continue
                _shack(mb, x2, z2, rng, local_obs)
                positions.append((x2, z2))
        for _ in range(120):
            i, j = rng.integers(len(positions), size=2)
            a, b = positions[i], positions[j]
            if 6 < math.hypot(a[0] - b[0], a[1] - b[1]) < 12:
                _laundry(mb, (a[0], rng.uniform(3.5, 5.5), a[1]), (b[0], rng.uniform(3.5, 5.5), b[1]), rng)
    for x, z, r, top, bottom in local_obs:
        obstacles.append((x + CITY[0], z + CITY[2], r, top, bottom))
    return mb.build(), obstacles
