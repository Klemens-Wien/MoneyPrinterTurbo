"""A kid's bedroom interior set (plus a dark hallway) for indoor night scenes.

The room sits far outside the town (beyond the fog) so outdoor shots never see
it. Moonlight comes in through the window via the normal shadow map; a small
night lamp adds a warm point light. Doors are separate animated meshes.
"""

from __future__ import annotations

import numpy as np

from .mesh import MeshBuilder, hex_color

ROOM = np.array([0.0, 0.0, 1500.0])  # world position of the room's floor centre
HALF_X, HALF_Z, HEIGHT = 3.5, 4.0, 3.0

BED_LOCAL = np.array([2.45, 0.0, -2.35])
BED_SURFACE = 0.72  # mattress top
DOOR_HINGE_LOCAL = np.array([-HALF_X + 0.1, 0.0, 1.2])
DOOR_WIDTH, DOOR_HEIGHT = 1.2, 2.2
CLOSET_LOCAL = np.array([0.0, 0.0, HALF_Z - 0.45])
LAMP_LOCAL = np.array([0.85, 1.05, -3.45])
HALLWAY_MONSTER_LOCAL = np.array([-5.4, 0.0, 1.8])

# camera must stay inside the room
BOUNDS_MIN = ROOM + np.array([-HALF_X + 0.35, 0.35, -HALF_Z + 0.35])
BOUNDS_MAX = ROOM + np.array([HALF_X - 0.35, HEIGHT - 0.25, HALF_Z - 0.35])

WALL = hex_color("#4e5d8f")
WALL_DARK = hex_color("#3d4a75")
FLOOR = hex_color("#9a6a45")
WOOD = hex_color("#7a5236")
WHITE = hex_color("#eef0f6")


def world(local) -> np.ndarray:
    return ROOM + np.asarray(local, dtype=np.float64)


def _room(mb: MeshBuilder) -> None:
    hx, hz, h, t = HALF_X, HALF_Z, HEIGHT, 0.2
    mb.box_min((-hx - t, -0.2, -hz - t), (hx + t, 0.0, hz + t), FLOOR)
    for i, z in enumerate(np.arange(-hz, hz, 0.5)):  # floor boards
        mb.box_min((-hx, 0.0, z), (hx, 0.012, z + 0.03), hex_color("#7d5536"))
    mb.box_min((-hx - t, h, -hz - t), (hx + t, h + t, hz + t), hex_color("#5a6796"))  # ceiling
    mb.box_min((-hx - t, 0, -hz - t), (hx + t, h, -hz), WALL)  # back wall (-z)
    mb.box_min((-hx - t, 0, hz), (hx + t, h, hz + t), WALL)  # front wall (+z)
    # +x wall with window opening z in [-1, 1], y in [1.0, 2.3]
    wx0, wx1 = hx, hx + t
    mb.box_min((wx0, 0, -hz), (wx1, 1.0, hz), WALL_DARK)
    mb.box_min((wx0, 2.3, -hz), (wx1, h, hz), WALL_DARK)
    mb.box_min((wx0, 1.0, -hz), (wx1, 2.3, -1.0), WALL_DARK)
    mb.box_min((wx0, 1.0, 1.0), (wx1, 2.3, hz), WALL_DARK)
    mb.box_min((wx0 - 0.02, 1.62, -1.0), (wx1, 1.68, 1.0), WHITE)  # window cross
    mb.box_min((wx0 - 0.02, 1.0, -0.03), (wx1, 2.3, 0.03), WHITE)
    mb.box_min((wx0 - 0.08, 0.95, -1.1), (wx0, 1.0, 1.1), WHITE)  # sill
    for z0, z1 in ((-1.55, -1.0), (1.0, 1.55)):  # curtains
        mb.box_min((wx0 - 0.12, 0.8, z0), (wx0 - 0.04, 2.5, z1), hex_color("#c85a6a"))
    # -x wall with the door opening z in [1.2, 2.4], y < 2.2
    dx0, dx1 = -hx - t, -hx
    mb.box_min((dx0, 0, -hz), (dx1, h, 1.2), WALL)
    mb.box_min((dx0, 0, 2.4), (dx1, h, hz), WALL)
    mb.box_min((dx0, 2.2, 1.2), (dx1, h, 2.4), WALL)
    mb.box_min((dx1, 0, 1.1), (dx1 + 0.05, 2.3, 1.2), WHITE)  # door frame
    mb.box_min((dx1, 0, 2.4), (dx1 + 0.05, 2.3, 2.5), WHITE)
    mb.box_min((dx1, 2.2, 1.1), (dx1 + 0.05, 2.3, 2.5), WHITE)
    # skirting boards
    mb.box_min((-hx, 0, -hz), (hx, 0.12, -hz + 0.04), WHITE)
    # hallway behind the door
    mb.box_min((-11.0, -0.2, 0.6), (-hx - t, 0.0, 3.0), hex_color("#5c4030"))
    mb.box_min((-11.0, h, 0.6), (-hx - t, h + t, 3.0), hex_color("#3a3f5c"))
    mb.box_min((-11.0, 0, 0.4), (-hx - t, h, 0.6), hex_color("#3a4466"))
    mb.box_min((-11.0, 0, 3.0), (-hx - t, h, 3.2), hex_color("#3a4466"))
    mb.box_min((-11.2, 0, 0.4), (-11.0, h, 3.2), hex_color("#2e3552"))
    mb.box_min((-9.8, 0.0, 2.9), (-8.6, 2.2, 3.0), WOOD)  # another door in the hall
    for k in range(4):  # stairs going down at the end of the hall
        mb.box_min((-10.8 + k * 0.0, -0.2 - k * 0.25, 0.7), (-10.0, -k * 0.25, 2.9), hex_color("#6a4a34"))


def _furniture(mb: MeshBuilder) -> None:
    bx, _, bz = BED_LOCAL
    # bed: head at -z
    mb.box((bx, 0.3, bz), (1.5, 0.3, 2.6), WOOD)
    mb.box((bx, 0.55, bz), (1.4, 0.24, 2.5), WHITE)  # mattress
    mb.box((bx, 0.72, bz + 0.45), (1.48, 0.12, 1.6), hex_color("#3d7be0"))  # blanket
    mb.box((bx, 0.75, bz - 0.95), (0.9, 0.16, 0.45), WHITE)  # pillow
    mb.box((bx, 0.75, bz - 1.33), (1.5, 1.3, 0.1), WOOD)  # headboard
    for cx in (-0.7, 0.7):
        for cz in (-1.25, 1.25):
            mb.box((bx + cx, 0.08, bz + cz), (0.1, 0.16, 0.1), WOOD)
    # nightstand + lamp base (the glowing shade is a separate emissive mesh)
    lx, _, lz = LAMP_LOCAL
    mb.box((lx, 0.35, lz), (0.6, 0.7, 0.5), WOOD)
    mb.box((lx, 0.75, lz), (0.12, 0.1, 0.12), hex_color("#e8d27a"))
    mb.box((lx + 0.15, 0.73, lz + 0.05), (0.18, 0.05, 0.12), hex_color("#ff5a5f"))  # alarm clock
    # closet against +z wall (doors are animated separately)
    cx, _, cz = CLOSET_LOCAL
    mb.box((cx, 1.25, cz + 0.225), (2.2, 2.5, 0.45), WOOD)
    mb.box((cx, 1.25, cz - 0.01), (2.0, 2.3, 0.02), hex_color("#120f18"))  # dark inside
    # desk + chair
    mb.box((-2.4, 0.75, -3.4), (1.6, 0.08, 0.8), WOOD)
    for dx in (-0.7, 0.7):
        mb.box((-2.4 + dx, 0.37, -3.4), (0.08, 0.74, 0.7), WOOD)
    mb.box((-2.4, 0.45, -2.7), (0.5, 0.08, 0.5), hex_color("#ffb703"))
    mb.box((-2.4, 0.8, -2.48), (0.5, 0.6, 0.06), hex_color("#ffb703"))
    # rug, toys, posters
    mb.box((0.0, 0.015, 0.0), (2.6, 0.02, 1.8), hex_color("#7a6cf0"))
    for i, (x, z, c) in enumerate(((-0.6, 0.4, "#ff4d4d"), (-0.3, 0.6, "#3ddc97"), (-0.45, 0.5, "#ffd23f"))):
        mb.box((x, 0.1 + 0.18 * (i == 2), z), (0.18, 0.18, 0.18), hex_color(c))
    for z, c in ((-2.6, "#ffb703"), (-0.2, "#3ddc97")):
        mb.box((-HALF_X + 0.03, 1.8, z), (0.04, 0.8, 0.6), hex_color(c))
    mb.box((1.2, 1.9, -HALF_Z + 0.03), (0.9, 0.6, 0.04), hex_color("#ff6fb5"))


def build_static() -> np.ndarray:
    mb = MeshBuilder()
    from .mesh import translate

    with mb.transform(translate(*ROOM)):
        _room(mb)
        _furniture(mb)
    return mb.build()


def door_mesh() -> np.ndarray:
    mb = MeshBuilder()
    mb.box((0.0, DOOR_HEIGHT / 2, DOOR_WIDTH / 2), (0.06, DOOR_HEIGHT, DOOR_WIDTH), hex_color("#c9a27a"))
    mb.box((0.06, 1.05, DOOR_WIDTH - 0.15), (0.06, 0.08, 0.08), hex_color("#e8d27a"))  # knob
    return mb.build()


def closet_door_mesh() -> np.ndarray:
    mb = MeshBuilder()
    mb.box((0.5, 1.2, 0.0), (1.0, 2.3, 0.05), hex_color("#8a5f3e"))
    mb.box((0.9, 1.2, -0.05), (0.05, 0.25, 0.05), hex_color("#e8d27a"))
    return mb.build()


def lamp_glow_mesh() -> np.ndarray:
    mb = MeshBuilder()
    mb.cone((0, 0, 0), 0.22, 0.3, hex_color("#ffd08a"), segments=8)
    return mb.build()


def eyes_mesh(color: str = "#ff3df2") -> np.ndarray:
    mb = MeshBuilder()
    for side in (-1, 1):
        mb.box((side * 0.16, 0, 0), (0.17, 0.08, 0.02), hex_color(color))
    return mb.build()
