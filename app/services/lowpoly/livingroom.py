"""Living room at night with a staircase up to a dark landing (for "don't answer" horror).

Couch facing a TV, floor lamp, front door, window with moonlight, family photos and a staircase along
the back wall that leads up through an opening in the ceiling to a pitch-dark landing.
"""

from __future__ import annotations

import numpy as np

from .mesh import MeshBuilder, hex_color, translate

ROOM = np.array([-1500.0, 0.0, 0.0])
HX, HZ, H = 5.0, 4.0, 3.0

COUCH_LOCAL = np.array([0.6, 0.0, 1.6])
SEAT = 0.5
TV_LOCAL = np.array([-HX + 0.25, 1.35, 1.6])
LAMP_LOCAL = np.array([2.4, 1.75, 2.9])
DOOR_HINGE_LOCAL = np.array([-3.6, 0.0, HZ - 0.1])  # front door in the +z wall, panel along +x
DOOR_W, DOOR_H = 1.2, 2.2
LANDING_LOCAL = np.array([-0.4, 2.0, -3.3])  # halfway up the stairs, in the half-dark
STAIR_FOOT_LOCAL = np.array([-2.6, 0.0, -2.0])

BOUNDS_MIN = ROOM + np.array([-HX + 0.35, 0.35, -HZ + 0.4])
BOUNDS_MAX = ROOM + np.array([HX - 0.35, H - 0.2, HZ - 0.35])

WALL = hex_color("#5b5470")
WALL2 = hex_color("#4c465f")
FLOOR = hex_color("#8a6446")
WOOD = hex_color("#6b4a30")
DARK = hex_color("#0d0c12")
WINDOW = hex_color("#2c3e78")


def world(local) -> np.ndarray:
    return ROOM + np.asarray(local, dtype=np.float64)


def build_static() -> np.ndarray:
    mb = MeshBuilder()
    with mb.transform(translate(*ROOM)):
        t = 0.2
        mb.box_min((-HX - t, -0.2, -HZ - t), (HX + t, 0.0, HZ + t), FLOOR)
        for z in np.arange(-HZ, HZ, 0.55):
            mb.box_min((-HX, 0.0, z), (HX, 0.012, z + 0.03), hex_color("#73523a"))
        # ceiling with a big opening: a dark gallery (upstairs) above the back of the room
        mb.box_min((-HX - t, H, -0.5), (HX + t, H + t, HZ + t), hex_color("#4a4458"))
        mb.box_min((-HX - t, H, -HZ - t), (-0.6, H + t, -0.5), hex_color("#4a4458"))
        # walls
        mb.box_min((-HX - t, 0, -HZ - t), (HX + t, H, -HZ), WALL2)  # back wall (stairs side)
        mb.box_min((-HX - t, 0, -HZ), (-HX, H, HZ), WALL)  # TV wall
        mb.box_min((-HX, 0, HZ), (-3.6, H, HZ + t), WALL)  # front wall left of door
        mb.box_min((-2.4, 0, HZ), (HX + t, H, HZ + t), WALL)
        mb.box_min((-3.6, DOOR_H, HZ), (-2.4, H, HZ + t), WALL)
        mb.box_min((-3.7, 0, HZ - 0.06), (-3.6, DOOR_H + 0.1, HZ), hex_color("#e8e2d0"))  # door frame
        mb.box_min((-2.4, 0, HZ - 0.06), (-2.3, DOOR_H + 0.1, HZ), hex_color("#e8e2d0"))
        mb.box_min((-3.7, DOOR_H, HZ - 0.06), (-2.3, DOOR_H + 0.1, HZ), hex_color("#e8e2d0"))
        # +x wall with window
        mb.box_min((HX, 0, -HZ), (HX + t, 1.0, HZ), WALL)
        mb.box_min((HX, 2.3, -HZ), (HX + t, H, HZ), WALL)
        mb.box_min((HX, 1.0, -HZ), (HX + t, 2.3, 0.2), WALL)
        mb.box_min((HX, 1.0, 1.8), (HX + t, 2.3, HZ), WALL)
        mb.box_min((HX - 0.02, 1.62, 0.2), (HX + t, 1.68, 1.8), hex_color("#e8e2d0"))
        mb.box_min((HX - 0.02, 1.0, 0.97), (HX + t, 2.3, 1.03), hex_color("#e8e2d0"))
        # upstairs gallery: dark, with a railing you can see from the couch
        mb.box_min((1.0, H - 0.1, -HZ), (HX, H + 0.05, -0.5), hex_color("#2a2430"))  # gallery floor
        mb.box_min((-0.6, H, -HZ - t), (HX + t, H + 3.0, -HZ), DARK)
        mb.box_min((HX, H, -HZ), (HX + t, H + 3.0, -0.5), DARK)
        mb.box_min((-0.8, H, -HZ - t), (-0.6, H + 3.0, -0.5), DARK)
        mb.box_min((-0.8, H + 3.0, -HZ - t), (HX + t, H + 3.2, -0.5), DARK)
        mb.box_min((-0.8, H + 1.6, -0.6), (HX + t, H + 3.0, -0.5), DARK)  # wall above the gallery opening
        for x in np.arange(1.1, HX, 0.45):  # gallery railing
            mb.box((x, H + 0.5, -0.55), (0.06, 1.0, 0.06), hex_color("#e8e2d0"))
        mb.box((3.0, H + 1.0, -0.55), (4.0, 0.08, 0.08), hex_color("#e8e2d0"))
        mb.box_min((3.4, H, -HZ + 0.02), (4.4, H + 2.2, -HZ + 0.06), hex_color("#1a1620"))  # upstairs doorway
        # staircase along the back wall going up towards +x
        steps = 9
        for k in range(steps):
            x0 = -2.6 + k * (3.6 / steps)
            y1 = (k + 1) * H / steps
            mb.box_min((x0, 0.0, -HZ), (x0 + 3.6 / steps + 0.02, y1, -2.6), WOOD)
            mb.box_min((x0, y1, -HZ), (x0 + 0.06, y1 + 0.02, -2.6), hex_color("#e8e2d0"))
        mb.box_min((1.0, 0.0, -HZ), (HX, H - 0.1, -2.6), hex_color("#3e3650"))  # under the top of the stairs
        for k in range(7):  # banister
            x = -2.4 + k * 0.6
            y = (x + 2.6) / 3.6 * H
            mb.box((x, y + 0.5, -2.6), (0.06, 1.0, 0.06), hex_color("#e8e2d0"))
        mb.box((-0.8, 1.9, -2.6), (4.2, 0.08, 0.08), hex_color("#e8e2d0"))
        # furniture
        cx, _, cz = COUCH_LOCAL
        mb.box((cx, 0.25, cz), (1.0, 0.5, 2.6), hex_color("#3d6a8a"))  # seat (facing -x)
        mb.box((cx + 0.45, 0.75, cz), (0.3, 1.0, 2.6), hex_color("#335a76"))  # back
        for s in (-1, 1):
            mb.box((cx, 0.55, cz + s * 1.35), (1.0, 0.6, 0.25), hex_color("#335a76"))
        mb.box((cx - 0.05, 0.52, cz - 0.6), (0.7, 0.1, 0.6), hex_color("#ff9fb5"))  # blanket
        mb.box((-1.6, 0.25, cz), (1.0, 0.5, 1.4), WOOD)  # coffee table
        mb.box((-HX + 0.3, 0.4, cz), (0.6, 0.8, 2.4), WOOD)  # TV stand
        mb.box((-HX + 0.2, 1.35, cz), (0.12, 1.1, 1.9), hex_color("#141418"))  # TV frame
        mb.box((LAMP_LOCAL[0], 0.85, LAMP_LOCAL[2]), (0.08, 1.7, 0.08), hex_color("#2a2a2a"))
        mb.box((LAMP_LOCAL[0], 0.03, LAMP_LOCAL[2]), (0.5, 0.06, 0.5), hex_color("#2a2a2a"))
        mb.box_min((-2.4, 0.01, 0.4), (2.0, 0.025, 2.9), hex_color("#8a3b4a"))  # rug
        for x, c in ((-0.5, "#ffd23f"), (1.0, "#3d9bff"), (2.5, "#3ddc97")):  # family photos
            mb.box((x, 1.9, -HZ + 0.05), (0.7, 0.55, 0.05), hex_color("#e8e2d0"))
            mb.box((x, 1.9, -HZ + 0.08), (0.55, 0.4, 0.03), hex_color(c))
        mb.box((3.6, 0.9, 3.6), (1.2, 1.8, 0.5), WOOD)  # cupboard
        mb.box((3.6, 1.83, 3.6), (0.18, 0.06, 0.12), hex_color("#e8d27a"))
    return mb.build()


def door_mesh() -> np.ndarray:
    mb = MeshBuilder()
    mb.box((DOOR_W / 2, DOOR_H / 2, 0.0), (DOOR_W, DOOR_H, 0.07), hex_color("#7a3b2e"))
    mb.box((DOOR_W - 0.15, 1.05, -0.07), (0.08, 0.08, 0.06), hex_color("#e8d27a"))
    return mb.build()


def tv_mesh() -> np.ndarray:
    mb = MeshBuilder()
    mb.box((0.0, 0.0, 0.0), (0.04, 0.95, 1.75), hex_color("#9fd8ff"))
    return mb.build()


def lamp_mesh() -> np.ndarray:
    mb = MeshBuilder()
    mb.cone((0, 0, 0), 0.3, 0.35, hex_color("#ffd08a"), segments=8)
    return mb.build()
