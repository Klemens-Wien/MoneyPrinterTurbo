"""Downtown of a big modern city ("Zero City") for city stories.

A wide avenue with lanes, crosswalks, traffic lights and street lamps, sidewalks lined with
skyscrapers (window bands, rooftop details, billboards), a plaza with a giant golden "0" monument,
parked cars, trees in planters and a ring of distant towers on the horizon. Like the other
districts it sits far away from the main town.
"""

from __future__ import annotations

import math

import numpy as np

from .mesh import MeshBuilder, hex_color, rotate_y, translate, vary
from .props import PALETTE

CITY = np.array([1500.0, 0.0, -1500.0])
MONUMENT_LOCAL = (0.0, 0.0, 40.0)
HOME_LOCAL = (96.0, 0.0, -13.0)  # the narrator's front door (south side of the avenue)

ASPHALT = hex_color("#3c3f48")
LANE = hex_color("#f2f2ea")
SIDEWALK = hex_color("#b9b6b0")
CURB = hex_color("#8d8a86")
PLAZA = hex_color("#d9d2c4")
PLAZA2 = hex_color("#c8bfae")
GOLD = hex_color("#f2c232")
GOLD_DARK = hex_color("#c9961c")
GLASS = hex_color("#7fa6c9")
WINDOW = PALETTE["window"]
FACADES = [hex_color(c) for c in ("#e7e1d6", "#c9d3df", "#9fb3c8", "#d8b49a", "#b7b2c9", "#8c97a8", "#e0cfa8",
                                  "#a8c3b8", "#cfa9a0", "#6f7d92")]
BILLBOARD = [hex_color(c) for c in ("#ff4d6d", "#3dd6c6", "#ffd84a", "#6c8cff", "#ff9f40", "#b06cff")]

AVENUE_HALF = 8.0  # half width of the road
WALK = 5.0  # sidewalk width


def world(local) -> np.ndarray:
    return CITY + np.asarray(local, dtype=np.float64)


def _tower(mb: MeshBuilder, x, z, w, d, h, rng, obstacles, face_dir):
    """A skyscraper centred at (x, z); face_dir = +1/-1: the side facing the avenue (z direction)."""
    col = vary(FACADES[int(rng.integers(len(FACADES)))], rng, 0.03)
    style = int(rng.integers(3))
    mb.box_min((x - w / 2, 0, z - d / 2), (x + w / 2, h, z + d / 2), col)
    # ground floor: shop windows and an awning
    fz = z + face_dir * (d / 2 + 0.05)
    mb.box((x, 1.7, fz), (w - 1.2, 2.6, 0.12), GLASS)
    mb.box((x, 3.3, z + face_dir * (d / 2 + 0.8)), (w - 1.0, 0.18, 1.6), BILLBOARD[int(rng.integers(len(BILLBOARD)))])
    # window bands on the street side and on the two flanks
    floor_h = 3.2
    for k in range(2, int(h / floor_h)):
        y = k * floor_h + 0.6
        if style == 0:  # continuous ribbon windows
            mb.box((x, y, fz), (w - 1.0, 1.3, 0.1), WINDOW if rng.random() < 0.35 else GLASS)
        else:  # punched windows
            n = max(2, int(w / 2.6))
            for i in range(n):
                wx = x - w / 2 + (i + 0.5) * w / n
                lit = rng.random() < 0.3
                mb.box((wx, y, fz), (w / n * 0.55, 1.4, 0.1), WINDOW if lit else GLASS)
        for sx in (-1, 1):
            mb.box((x + sx * (w / 2 + 0.05), y, z), (0.1, 1.3, d - 1.2), WINDOW if rng.random() < 0.25 else GLASS)
    # rooftop: parapet, machine room, antenna or water tank
    mb.box((x, h + 0.4, z), (w + 0.3, 0.8, d + 0.3), vary(col, rng, 0.06))
    mb.box((x + rng.uniform(-w / 5, w / 5), h + 1.8, z), (w * 0.35, 2.4, d * 0.35), hex_color("#8a8a90"))
    if rng.random() < 0.45:
        mb.box((x, h + 6.0, z), (0.25, 9.0, 0.25), hex_color("#4a4a52"))
        mb.box((x, h + 10.6, z), (0.45, 0.45, 0.45), hex_color("#ff4040"))
    elif rng.random() < 0.5:
        mb.cylinder((x - w * 0.2, h + 3.2, z + d * 0.15), 1.4, 2.6, hex_color("#8a5a3a"), segments=8)
    # big billboard on some facades
    if rng.random() < 0.35 and h > 30:
        by = rng.uniform(14, min(h - 6, 30))
        mb.box((x, by, z + face_dir * (d / 2 + 0.35)), (w * 0.7, 6.0, 0.3), BILLBOARD[int(rng.integers(len(BILLBOARD)))])
        mb.box((x, by, z + face_dir * (d / 2 + 0.52)), (w * 0.5, 3.2, 0.05), hex_color("#ffffff"))
    obstacles.append((x, z, max(w, d) / 2 + 1.0, h + 1.0, -1.0))


def _lamp(mb: MeshBuilder, x, z, side):
    mb.box((x, 3.0, z), (0.22, 6.0, 0.22), hex_color("#2e3138"))
    mb.box((x, 6.0, z - side * 1.0), (0.18, 0.18, 2.0), hex_color("#2e3138"))
    mb.box((x, 5.8, z - side * 1.9), (0.7, 0.25, 0.9), WINDOW)


def _traffic_light(mb: MeshBuilder, x, z):
    mb.box((x, 2.2, z), (0.25, 4.4, 0.25), hex_color("#2e3138"))
    mb.box((x, 4.8, z), (0.6, 1.6, 0.5), hex_color("#1c1e24"))
    for k, c in enumerate(("#ff3b30", "#ffcc00", "#34c759")):
        mb.box((x, 5.3 - k * 0.5, z + 0.27), (0.32, 0.32, 0.06), hex_color(c))


def _car(mb: MeshBuilder, x, z, rot, rng):
    col = vary([hex_color(c) for c in ("#d63a3a", "#2f6fd6", "#f2f2f2", "#222428", "#f2c232", "#3aa66a")][
        int(rng.integers(6))], rng, 0.04)
    with mb.transform(translate(x, 0, z) @ rotate_y(rot)):
        mb.box((0, 0.75, 0), (4.2, 0.9, 1.9), col)
        mb.box((-0.2, 1.55, 0), (2.4, 0.75, 1.75), col)
        mb.box((-0.2, 1.55, 0), (2.42, 0.55, 1.77), GLASS)
        for wx in (-1.35, 1.35):
            for wz in (-0.95, 0.95):
                mb.cylinder((wx, 0.42, wz), 0.42, 0.3, hex_color("#1a1a1a"), segments=8, axis="z")


def _tree(mb: MeshBuilder, x, z, rng):
    mb.box((x, 0.35, z), (1.6, 0.7, 1.6), hex_color("#8d8a86"))
    mb.cylinder((x, 1.6, z), 0.18, 2.4, PALETTE["trunk"], segments=5)
    with mb.swaying(0.4):
        mb.icosphere((x, 3.6, z), rng.uniform(1.4, 1.9), vary(hex_color("#4f9a4a"), rng, 0.05), rng=rng, jitter=0.15)


def _monument(mb: MeshBuilder, obstacles):
    """The giant golden zero on the plaza: the city's pride."""
    mx, _, mz = MONUMENT_LOCAL
    mb.cylinder((mx, 0.5, mz), 7.5, 1.0, hex_color("#e9e3d6"), segments=16)
    mb.cylinder((mx, 1.3, mz), 6.0, 0.6, hex_color("#cfc6b4"), segments=16)
    mb.box((mx, 2.6, mz), (7.0, 2.0, 2.6), hex_color("#2e3138"))  # plinth
    mb.box((mx, 2.6, mz - 1.32), (4.6, 0.9, 0.05), GOLD)  # plaque
    rx, ry, cy, th = 4.2, 6.2, 10.4, 1.5
    n = 28
    for i in range(n):
        a0, a1 = i / n * 2 * math.pi, (i + 1) / n * 2 * math.pi
        p0 = np.array([math.cos(a0) * rx, math.sin(a0) * ry])
        p1 = np.array([math.cos(a1) * rx, math.sin(a1) * ry])
        mid = (p0 + p1) / 2
        seg = p1 - p0
        length = float(np.linalg.norm(seg)) + 0.25
        ang = math.atan2(seg[1], seg[0])
        c, s = math.cos(ang), math.sin(ang)
        rot = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        with mb.transform(translate(mx + mid[0], cy + mid[1], mz) @ rot):
            mb.box((0, 0, 0), (length, th, 1.6), GOLD if i % 2 else GOLD_DARK)
    obstacles.append((mx, mz, 7.6, 17.0, -1.0))


def _plaza(mb: MeshBuilder, rng, obstacles):
    z0 = AVENUE_HALF + WALK
    mb.box_min((-34, 0.0, z0), (34, 0.04, 62), PLAZA)
    for k in range(-34, 34, 4):  # paving pattern
        mb.box_min((k, 0.041, z0), (k + 0.15, 0.05, 62), PLAZA2)
    _monument(mb, obstacles)
    for x in (-26, -14, 14, 26):
        for z in (24, 54):
            _tree(mb, x, z, rng)
            obstacles.append((x, z, 1.2, 5.0, -1.0))
    for x in (-20, 20):  # benches
        for z in (30, 48):
            mb.box((x, 0.5, z), (3.0, 0.2, 0.9), hex_color("#8a5a34"))
            mb.box((x, 0.9, z + 0.4), (3.0, 0.6, 0.15), hex_color("#8a5a34"))
    # flag poles with "zero" banners
    for x in (-8, 8):
        mb.box((x, 6.0, 60), (0.2, 12.0, 0.2), hex_color("#d8d8d8"))
        mb.box((x + 1.1, 10.5, 60), (2.0, 2.6, 0.06), GOLD)
    # city hall behind the plaza
    mb.box_min((-30, 0, 64), (30, 22, 84), hex_color("#efe9dc"))
    for x in np.arange(-26, 27, 4.0):
        mb.box((x, 8, 63.7), (1.1, 15, 1.1), hex_color("#ffffff"))  # columns
    mb.box((0, 23.5, 74), (40, 3.0, 16), hex_color("#e2dccc"))
    mb.cylinder((0, 27.0, 74), 6.0, 4.0, hex_color("#e2dccc"), segments=14)
    mb.icosphere((0, 30.0, 74), 6.0, GOLD, squash=(1.0, 0.7, 1.0), subdivisions=1)
    obstacles.append((0.0, 74.0, 31.0, 36.0, -1.0))


def build_static(seed: int = 11) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    local_obs: list = []
    with mb.transform(translate(*CITY)):
        mb.box_min((-290, -0.3, -290), (290, 0.0, 290), hex_color("#9a978f"))  # city blocks base
        # the avenue along x and two cross streets along z
        mb.box_min((-290, 0.0, -AVENUE_HALF), (290, 0.03, AVENUE_HALF), ASPHALT)
        for cx in (-60.0, 60.0):
            mb.box_min((cx - 7, 0.0, -290), (cx + 7, 0.03, 290), ASPHALT)
        for x in np.arange(-286, 286, 6.0):  # lane dashes
            if all(abs(x - cx) > 9 for cx in (-60, 60)):
                mb.box_min((x, 0.031, -0.12), (x + 3.0, 0.04, 0.12), LANE)
        for z in (-AVENUE_HALF / 2, AVENUE_HALF / 2):
            for x in np.arange(-286, 286, 6.0):
                if all(abs(x - cx) > 9 for cx in (-60, 60)):
                    mb.box_min((x, 0.031, z - 0.08), (x + 1.6, 0.04, z + 0.08), LANE)
        for cx in (-60.0, 60.0):  # crosswalks
            for side in (-1, 1):
                for k in range(-6, 7):
                    mb.box_min((cx + side * 9 - 1.4, 0.031, k * 1.2 - 0.35), (cx + side * 9 + 1.4, 0.04, k * 1.2 + 0.35),
                               LANE)
        for k in range(-6, 7):  # crosswalk in front of the plaza
            mb.box_min((k * 1.2 - 0.35, 0.031, -AVENUE_HALF + 0.4), (k * 1.2 + 0.35, 0.04, AVENUE_HALF - 0.4), LANE)
        # sidewalks with curbs
        for side in (-1, 1):
            z0, z1 = sorted((side * AVENUE_HALF, side * (AVENUE_HALF + WALK)))
            mb.box_min((-290, 0.0, z0), (290, 0.18, z1), SIDEWALK)
            zc = side * AVENUE_HALF
            mb.box_min((-290, 0.0, zc - 0.15), (290, 0.22, zc + 0.15), CURB)
        _plaza(mb, rng, local_obs)
        # skyscrapers along the avenue
        for side in (-1, 1):
            x = -280.0
            while x < 280:
                w = float(rng.uniform(14, 24))
                cx = x + w / 2
                if any(abs(cx - c) < w / 2 + 8 for c in (-60.0, 60.0)) or (side > 0 and abs(cx) < 34 + w / 2):
                    x += 4.0
                    continue
                d = float(rng.uniform(16, 24))
                cz = side * (AVENUE_HALF + WALK + 0.5 + d / 2)
                h = float(rng.uniform(26, 95)) if abs(cx) > 30 else float(rng.uniform(22, 40))
                if abs(cx - HOME_LOCAL[0]) < w / 2 + 2 and side < 0:  # the narrator's building: lower, warm
                    h = 24.0
                _tower(mb, cx, cz, w, d, h, rng, local_obs, -side)
                x += w + float(rng.uniform(1.0, 3.0))
        # the narrator's front door
        hx, _, hz = HOME_LOCAL
        mb.box((hx, 1.4, -(AVENUE_HALF + WALK + 0.45)), (1.8, 2.8, 0.2), hex_color("#7a3b2e"))
        mb.box((hx, 0.1, -(AVENUE_HALF + WALK - 0.4)), (2.6, 0.2, 1.2), hex_color("#9a8f80"))  # doorstep
        # second row of towers behind the first, and a ring of distant skyscrapers on the horizon
        for _ in range(70):
            x, z = rng.uniform(-260, 260), rng.choice([-1, 1]) * rng.uniform(62, 140)
            if abs(x - 60) < 12 or abs(x + 60) < 12 or (z > 0 and abs(x) < 40 and z < 100):
                continue
            w, d, h = rng.uniform(16, 26), rng.uniform(16, 26), rng.uniform(40, 140)
            _tower(mb, float(x), float(z), float(w), float(d), float(h), rng, local_obs, -1 if z > 0 else 1)
        for k in range(46):
            ang = k / 46 * 2 * math.pi + rng.uniform(-0.04, 0.04)
            r = rng.uniform(320, 520)
            x, z = math.cos(ang) * r, math.sin(ang) * r
            w, h = rng.uniform(22, 40), rng.uniform(80, 240)
            col = vary(hex_color("#8f9bb0"), rng, 0.05)
            mb.box_min((x - w / 2, -0.3, z - w / 2), (x + w / 2, h, z + w / 2), col)
            for y in np.arange(6, h - 4, 7.0):
                mb.box((x, y, z), (w + 0.1, 1.6, w + 0.1), WINDOW if rng.random() < 0.3 else hex_color("#6e7c94"))
        mb.box_min((-700, -0.35, -700), (700, -0.31, 700), hex_color("#7d7a74"))  # outskirts
        # street furniture
        for x in np.arange(-276, 277, 18.0):
            for side in (-1, 1):
                if all(abs(x - c) > 10 for c in (-60, 60)):
                    _lamp(mb, float(x), side * (AVENUE_HALF + 0.9), side)
        for cx in (-60.0, 60.0):
            for sx in (-1, 1):
                for sz in (-1, 1):
                    _traffic_light(mb, cx + sx * 8.5, sz * (AVENUE_HALF + 1.2))
        for x in np.arange(-270, 270, 11.0):  # parked cars along both curbs
            for side in (-1, 1):
                if all(abs(x - c) > 14 for c in (-60, 60)) and abs(x) > 16 and rng.random() < 0.55:
                    _car(mb, float(x), side * (AVENUE_HALF - 1.4), 0.0, rng)
        for x in np.arange(-268, 270, 24.0):  # sidewalk trees on the south side
            if all(abs(x - c) > 12 for c in (-60, 60)) and abs(x - HOME_LOCAL[0]) > 4:
                _tree(mb, float(x), -(AVENUE_HALF + 2.6), rng)
    obstacles = [(o[0] + CITY[0], o[1] + CITY[2], o[2], o[3], o[4]) for o in local_obs]
    return mb.build(), obstacles
