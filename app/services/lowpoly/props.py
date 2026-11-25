"""Reusable low-poly props: trees, houses, vehicles, clouds and the toy figure.

All props are original, generic designs (blocky toy people, boxy trucks,
gable-roof houses) – no brand characters or trademarked shapes.
"""

from __future__ import annotations

import math

import numpy as np

from .mesh import MeshBuilder, hex_color, rotate_y, translate, vary

# Soft pastel palette tuned for the "toy world" look.
PALETTE = {
    "grass": hex_color("#6fd35a"),
    "grass_dark": hex_color("#5cc451"),
    "field": [hex_color(c) for c in ("#7fdc5a", "#a8e35c", "#e8e070", "#c8e65e", "#62d06e", "#b9a6f2", "#f0d873", "#93e07a")],
    "road": hex_color("#7a7f8c"),
    "road_line": hex_color("#f4f1e6"),
    "plaza": hex_color("#c9c3dd"),
    "sand": hex_color("#f5d98a"),
    "rock": hex_color("#a7a3b5"),
    "snow": hex_color("#f3f6ff"),
    "trunk": hex_color("#7a5236"),
    "leaf": [hex_color(c) for c in ("#6cc24a", "#86d35a", "#5bb04a", "#9bd65c", "#4fa64a")],
    "pine": [hex_color(c) for c in ("#2f8f5b", "#3a9e64", "#277a50")],
    "wall": [hex_color(c) for c in ("#f4f6ff", "#dceeff", "#fff1d6", "#ffe0e6", "#e3f6e8", "#f0e6ff")],
    "roof": [hex_color(c) for c in ("#5b6b8c", "#d0574a", "#4a5a7a", "#7a5a9c", "#3f7f8c")],
    "window": hex_color("#2c3e78"),
    "door": hex_color("#8a5a3c"),
    "vehicle": [hex_color(c) for c in ("#e84a4a", "#3db4f2", "#8fd14a", "#f2b632", "#ff7ab8", "#7a6cf0", "#2fc9a8")],
    "tire": hex_color("#26262e"),
    "glass": hex_color("#9fd8ff"),
    "chrome": hex_color("#d9dde8"),
    "skin": [hex_color(c) for c in ("#ffd6a8", "#f1c27d", "#c68642", "#8d5524", "#ffdbac")],
    "shirt": [hex_color(c) for c in ("#ff5a5f", "#3d9bff", "#ffd23f", "#2ec4b6", "#9b5de5", "#f15bb5", "#00bb7f")],
    "pants": [hex_color(c) for c in ("#2b3a67", "#3b3b4f", "#4a6fa5", "#5c4033", "#1f6f5c")],
    "hair": [hex_color(c) for c in ("#3b2a20", "#1d1d24", "#c8872b", "#7a3b1e", "#e8d27a")],
    "cloud": hex_color("#ffffff"),
    "water": hex_color("#4cc3e6"),
}


# --------------------------------------------------------------------------- #
# Nature
# --------------------------------------------------------------------------- #


def tree_round(mb: MeshBuilder, pos, size: float, rng: np.random.Generator) -> None:
    x, y, z = pos
    trunk_h = 2.2 * size
    mb.cylinder((x, y + trunk_h / 2, z), 0.28 * size, trunk_h, PALETTE["trunk"], segments=5, cap=False)
    leaf = vary(PALETTE["leaf"][rng.integers(len(PALETTE["leaf"]))], rng, 0.04)
    with mb.swaying(1.0):
        for _ in range(rng.integers(2, 4)):
            offset = rng.uniform(-0.7, 0.7, 3) * size
            offset[1] = abs(offset[1]) * 0.5
            r = rng.uniform(1.2, 1.7) * size
            mb.icosphere(
                (x + offset[0], y + trunk_h + r * 0.55 + offset[1], z + offset[2]),
                r, vary(leaf, rng, 0.03), rng=rng, jitter=0.12, squash=(1.0, 0.8, 1.0),
            )


def tree_pine(mb: MeshBuilder, pos, size: float, rng: np.random.Generator) -> None:
    x, y, z = pos
    mb.cylinder((x, y + 0.6 * size, z), 0.25 * size, 1.2 * size, PALETTE["trunk"], segments=5, cap=False)
    color = vary(PALETTE["pine"][rng.integers(len(PALETTE["pine"]))], rng, 0.03)
    with mb.swaying(0.7):
        for i, (r, h) in enumerate(((1.6, 2.4), (1.25, 2.1), (0.85, 1.8))):
            mb.cone((x, y + (1.0 + i * 1.25) * size, z), r * size, h * size, color, segments=6, rng=rng, jitter=0.1)


def rock(mb: MeshBuilder, pos, size: float, rng: np.random.Generator) -> None:
    mb.icosphere(pos, size, vary(PALETTE["rock"], rng, 0.05), rng=rng, jitter=0.25, squash=(1.2, 0.7, 1.0))


def cloud(mb: MeshBuilder, pos, size: float, rng: np.random.Generator) -> None:
    x, y, z = pos
    for _ in range(rng.integers(4, 7)):
        dx, dz = rng.uniform(-1.6, 1.6) * size, rng.uniform(-0.8, 0.8) * size
        r = rng.uniform(0.7, 1.2) * size
        mb.icosphere((x + dx, y + rng.uniform(-0.1, 0.3) * size, z + dz), r, PALETTE["cloud"],
                     rng=rng, jitter=0.12, squash=(1.2, 0.65, 1.0))


# --------------------------------------------------------------------------- #
# Buildings
# --------------------------------------------------------------------------- #


def _windows_on_face(mb, width, floors, floor_h, z_face, x_offset=0.0, axis="z", depth_sign=1):
    cols = max(1, int(width // 2.4))
    spacing = width / cols
    for f in range(floors):
        wy = 0.8 + f * floor_h + floor_h * 0.35
        for c in range(cols):
            wx = -width / 2 + spacing * (c + 0.5) + x_offset
            if axis == "z":
                mb.box((wx, wy, z_face), (spacing * 0.45, floor_h * 0.42, 0.16), PALETTE["window"])
            else:
                mb.box((z_face, wy, wx), (0.16, floor_h * 0.42, spacing * 0.45), PALETTE["window"])


def house(mb: MeshBuilder, pos, rng: np.random.Generator, width=None, depth=None, floors=None,
          rot: float = 0.0, wall=None, roof=None) -> dict:
    """A pastel house. Local front faces +z. Returns its dimensions."""
    width = width or rng.uniform(7, 11)
    depth = depth or rng.uniform(6, 9)
    floors = floors or int(rng.integers(1, 3))
    floor_h = 3.0
    wall = wall or vary(PALETTE["wall"][rng.integers(len(PALETTE["wall"]))], rng, 0.02)
    roof = roof or vary(PALETTE["roof"][rng.integers(len(PALETTE["roof"]))], rng, 0.03)
    wall_h = 0.8 + floors * floor_h
    with mb.transform(translate(*pos) @ rotate_y(rot)):
        mb.box((0, 0.4, 0), (width + 0.6, 0.8, depth + 0.6), PALETTE["rock"])  # foundation
        mb.box((0, wall_h / 2 + 0.2, 0), (width, wall_h - 0.4, depth), wall)
        _windows_on_face(mb, width, floors, floor_h, depth / 2 + 0.02)
        _windows_on_face(mb, width, floors, floor_h, -depth / 2 - 0.02)
        _windows_on_face(mb, depth, floors, floor_h, width / 2 + 0.02, axis="x")
        _windows_on_face(mb, depth, floors, floor_h, -width / 2 - 0.02, axis="x")
        mb.box((0, 1.9, depth / 2 + 0.05), (1.4, 2.4, 0.2), PALETTE["door"])
        if width >= depth:
            with mb.transform(rotate_y(math.pi / 2)):
                mb.gable_roof((0, wall_h, 0), depth, width, depth * 0.42, roof)
        else:
            mb.gable_roof((0, wall_h, 0), width, depth, width * 0.45, roof)
        if rng.random() < 0.5:
            mb.box((width * 0.25, wall_h + 1.6, -depth * 0.15), (0.8, 2.4, 0.8), hex_color("#b0605a"))  # chimney
    return {"width": width, "depth": depth, "height": wall_h + depth * 0.45}


def town_hall(mb: MeshBuilder, pos, rng: np.random.Generator, rot: float = 0.0) -> None:
    wall = hex_color("#fbf8ef")
    roof = hex_color("#5a6c94")
    with mb.transform(translate(*pos) @ rotate_y(rot)):
        mb.box((0, 0.4, 0), (25, 0.8, 13), PALETTE["rock"])
        mb.box((0, 5.6, 0), (24, 10, 12), wall)
        mb.box((0, 10.9, 0), (24.8, 0.6, 12.8), hex_color("#f2c94c"))
        _windows_on_face(mb, 24, 3, 3.0, 6.02)
        _windows_on_face(mb, 24, 3, 3.0, -6.02)
        with mb.transform(rotate_y(math.pi / 2)):
            mb.hip_roof((0, 11.2, 0), 12, 24, 3.4, roof)
        # clock tower
        mb.box((0, 14.5, 0), (5, 7, 5), wall)
        mb.cylinder((0, 15.5, 2.55), 1.6, 0.2, hex_color("#fff7d6"), segments=12, axis="z")
        mb.box((0.0, 15.9, 2.7), (0.18, 1.1, 0.1), hex_color("#2a2a2a"))
        mb.box((0.4, 15.5, 2.7), (0.9, 0.16, 0.1), hex_color("#2a2a2a"))
        mb.cone((0, 18, 0), 3.4, 5, roof, segments=4)
        mb.box((0, 1.9, 6.1), (3, 3, 0.3), PALETTE["door"])
        # flags
        for fx in (-10, 10):
            mb.box((fx, 13.5, 4), (0.2, 4, 0.2), PALETTE["chrome"])
            mb.box((fx + 0.9, 15, 4), (1.6, 1.0, 0.1), hex_color("#e84a4a"))


def shop(mb: MeshBuilder, pos, rng: np.random.Generator, rot: float = 0.0) -> None:
    """Flat-roof shop with a big rooftop sign shape (generic, no logos)."""
    wall = vary(PALETTE["wall"][rng.integers(len(PALETTE["wall"]))], rng, 0.02)
    accent = PALETTE["vehicle"][rng.integers(len(PALETTE["vehicle"]))]
    with mb.transform(translate(*pos) @ rotate_y(rot)):
        mb.box((0, 0.3, 0), (13, 0.6, 10), PALETTE["rock"])
        mb.box((0, 3.6, 0), (12, 6.6, 9), wall)
        mb.box((0, 7.1, 0), (12.6, 0.6, 9.6), accent)
        mb.box((0, 2.0, 4.55), (9, 2.6, 0.12), PALETTE["glass"])
        mb.box((0, 4.2, 5.0), (12.4, 0.5, 1.2), accent)  # awning
        mb.icosphere((0, 10.0, 0), 1.8, accent, squash=(1.4, 0.8, 1.0))
        mb.box((0, 8.4, 0), (0.4, 1.6, 0.4), PALETTE["chrome"])


def crane(mb: MeshBuilder, pos, rot: float = 0.0) -> None:
    yellow = hex_color("#f2c232")
    with mb.transform(translate(*pos) @ rotate_y(rot)):
        mb.box((0, 0.5, 0), (4, 1, 4), hex_color("#4a4a55"))
        for i in range(8):
            mb.box((0, 1.5 + i * 2.5, 0), (1.2, 2.3, 1.2), yellow)
        mb.box((5, 21.5, 0), (16, 0.9, 0.9), yellow)
        mb.box((-4, 20.8, 0), (2.5, 2, 2), hex_color("#4a4a55"))
        mb.box((11, 17, 0), (0.1, 9, 0.1), hex_color("#30303a"))
        mb.box((11, 12.3, 0), (1.4, 0.8, 1.4), hex_color("#30303a"))


# --------------------------------------------------------------------------- #
# Vehicles (local forward = +x, ground at y=0)
# --------------------------------------------------------------------------- #


def _wheels(mb, xs, half_track, radius=0.6):
    for x in xs:
        for zs in (-half_track, half_track):
            mb.cylinder((x, radius, zs), radius, 0.5, PALETTE["tire"], segments=8, axis="z")
            mb.cylinder((x, radius, zs + math.copysign(0.27, zs)), radius * 0.5, 0.06, PALETTE["chrome"], segments=6, axis="z")


def truck(mb: MeshBuilder, color, flatbed_length: float = 11.0) -> dict:
    """Semi truck with a long flatbed trailer. Returns the cargo platform anchor."""
    cab_x = flatbed_length / 2 + 1.6
    mb.box((cab_x, 1.9, 0), (3.2, 2.6, 2.7), color)
    mb.box((cab_x + 2.0, 1.4, 0), (1.4, 1.6, 2.6), color)  # hood
    mb.box((cab_x + 2.72, 1.1, 0), (0.1, 0.9, 2.0), PALETTE["chrome"])  # grille
    mb.box((cab_x + 0.7, 2.55, 0), (1.9, 1.0, 2.75), PALETTE["glass"])
    mb.box((cab_x + 1.63, 2.5, 0), (0.06, 0.9, 2.3), PALETTE["glass"])
    for z in (-1.4, 1.4):
        mb.box((cab_x + 2.68, 1.35, z * 0.75), (0.08, 0.35, 0.45), hex_color("#fff7c2"))
    mb.box((0, 1.15, 0), (flatbed_length, 0.35, 2.9), hex_color("#3d3f4a"))
    _wheels(mb, (cab_x + 1.6, cab_x - 0.6, -flatbed_length / 2 + 1.2, -flatbed_length / 2 + 2.6), 1.2, 0.55)
    return {"cargo_y": 1.35, "cargo_length": flatbed_length, "length": flatbed_length + 6.0}


def bus(mb: MeshBuilder, color=None) -> dict:
    color = color or hex_color("#f6c431")
    mb.box((0, 2.0, 0), (11, 2.8, 2.8), color)
    mb.box((0, 2.5, 0), (10.6, 0.9, 2.86), PALETTE["glass"])
    mb.box((5.52, 2.4, 0), (0.06, 1.4, 2.4), PALETTE["glass"])
    mb.box((0, 0.75, 0), (11.1, 0.3, 2.9), hex_color("#30303a"))
    _wheels(mb, (3.6, -3.4), 1.2, 0.6)
    return {"length": 11.0}


def car(mb: MeshBuilder, color) -> dict:
    mb.box((0, 1.0, 0), (4.4, 0.9, 2.0), color)
    mb.box((-0.3, 1.8, 0), (2.4, 0.8, 1.8), PALETTE["glass"])
    mb.box((-0.3, 2.25, 0), (2.4, 0.1, 1.8), color)
    _wheels(mb, (1.4, -1.4), 0.9, 0.45)
    return {"length": 4.4}


# --------------------------------------------------------------------------- #
# Toy figure (original blocky design). Local forward = +z, feet at y=0.
# Parts are built around their pivot so the animator can rotate limbs.
# --------------------------------------------------------------------------- #

FIGURE_PARTS = ("torso", "head", "arm_l", "arm_r", "leg_l", "leg_r")
FIGURE_PIVOTS = {
    "torso": (0.0, 0.95, 0.0),
    "head": (0.0, 1.9, 0.0),
    "arm_l": (-0.72, 1.78, 0.0),
    "arm_r": (0.72, 1.78, 0.0),
    "leg_l": (-0.26, 0.95, 0.0),
    "leg_r": (0.26, 0.95, 0.0),
}


def figure_colors(rng: np.random.Generator) -> dict:
    return {
        "skin": PALETTE["skin"][rng.integers(len(PALETTE["skin"]))],
        "shirt": PALETTE["shirt"][rng.integers(len(PALETTE["shirt"]))],
        "pants": PALETTE["pants"][rng.integers(len(PALETTE["pants"]))],
        "hair": PALETTE["hair"][rng.integers(len(PALETTE["hair"]))],
        "cap": rng.random() < 0.35,
    }


HERO_COLORS = {
    "skin": hex_color("#ffd6a8"),
    "shirt": hex_color("#ff7a1a"),
    "pants": hex_color("#2b3a67"),
    "hair": hex_color("#5a3420"),
    "cap": True,
    "cap_color": hex_color("#1fb6ff"),
}


SISTER_COLORS = {
    "skin": hex_color("#ffd6a8"),
    "shirt": hex_color("#ff6fb5"),
    "pants": hex_color("#7a6cf0"),
    "hair": hex_color("#e8c25a"),
    "cap": False,
}


def figure_parts(colors: dict) -> dict[str, MeshBuilder]:
    """Return one MeshBuilder per limb, each in pivot-local coordinates."""
    parts = {name: MeshBuilder() for name in FIGURE_PARTS}
    shoe = hex_color("#2a2a33")
    parts["torso"].box((0, 0.47, 0), (1.04, 0.95, 0.52), colors["shirt"])
    parts["torso"].box((0, 0.04, 0), (1.06, 0.12, 0.54), hex_color("#3a2a20"))  # belt
    head = parts["head"]
    head.box((0, 0.42, 0), (0.78, 0.78, 0.78), colors["skin"])
    head.box((-0.17, 0.5, 0.4), (0.12, 0.16, 0.04), hex_color("#1b1b22"))
    head.box((0.17, 0.5, 0.4), (0.12, 0.16, 0.04), hex_color("#1b1b22"))
    head.box((0, 0.27, 0.4), (0.28, 0.06, 0.04), hex_color("#7a3030"))
    if colors.get("cap"):
        cap = colors.get("cap_color", colors["shirt"])
        head.box((0, 0.86, -0.02), (0.84, 0.18, 0.84), cap)
        head.box((0, 0.8, 0.5), (0.7, 0.06, 0.4), cap)
    else:
        head.box((0, 0.86, -0.04), (0.84, 0.2, 0.86), colors["hair"])
        head.box((0, 0.55, -0.38), (0.84, 0.5, 0.12), colors["hair"])
    for side, name in ((-1, "arm_l"), (1, "arm_r")):
        parts[name].box((0, -0.32, 0), (0.38, 0.7, 0.42), colors["shirt"])
        parts[name].box((0, -0.8, 0), (0.34, 0.3, 0.38), colors["skin"])
    for name in ("leg_l", "leg_r"):
        parts[name].box((0, -0.4, 0), (0.46, 0.8, 0.48), colors["pants"])
        parts[name].box((0, -0.87, 0.06), (0.48, 0.16, 0.6), shoe)
    return parts


# --------------------------------------------------------------------------- #
# The Grey Giant – an original, blocky low-poly monster.
# Local forward = +z, feet at y=0, ~9.5 units tall (scaled up by the actor).
# --------------------------------------------------------------------------- #

MONSTER_PARTS = ("body", "head", "eyes", "arm_l", "arm_r", "leg_l", "leg_r")
MONSTER_HIP = (0.0, 3.0, 0.0)
MONSTER_OFFSETS = {  # relative to the hip pivot (head/eyes/arms ride on the body)
    "head": (0.0, 4.1, 0.55),
    "eyes": (0.0, 4.1, 0.55),
    "arm_l": (-2.75, 3.7, 0.0),
    "arm_r": (2.75, 3.7, 0.0),
}
MONSTER_LEGS = {"leg_l": (-1.3, 3.0, 0.0), "leg_r": (1.3, 3.0, 0.0)}


MONSTER_VARIANTS = {
    # name: skin, belly, dark, accent (moss/cracks), bone, eye, accent glows?, display name
    "grey": dict(skin="#5b6178", belly="#8a8fa6", dark="#3c4052", accent="#5f8f4e", bone="#f1ead2",
                 eye="#ffe14d", glow=False, name="Grey Giant"),
    "lava": dict(skin="#4a2620", belly="#6e3226", dark="#2a1512", accent="#ff7a1a", bone="#3a2a26",
                 eye="#ffd23f", glow=True, name="Lava Golem"),
    "ice": dict(skin="#8fc1e3", belly="#dff3ff", dark="#5a8fb8", accent="#ffffff", bone="#e8fbff",
                eye="#3ff6ff", glow=False, name="Ice Titan"),
    "shadow": dict(skin="#1f1b2e", belly="#2c2742", dark="#110f1a", accent="#b04dff", bone="#3a3450",
                   eye="#ff3df2", glow=True, name="Shadow Beast"),
    "swamp": dict(skin="#3f5a35", belly="#7a8a4a", dark="#26361f", accent="#2f6f3a", bone="#d8d0a0",
                  eye="#c6ff3d", glow=False, name="Swamp King"),
}
MONSTER_NAMES = {k: v["name"] for k, v in MONSTER_VARIANTS.items()}


def monster_parts(variant: str = "grey") -> dict[str, MeshBuilder]:
    """Blocky giant in one of several colour variants.

    Glowing details (eyes, lava cracks, shadow runes) live in separate "*_glow"
    meshes that the renderer draws unlit; they share the transform of their part.
    """
    spec = MONSTER_VARIANTS.get(variant, MONSTER_VARIANTS["grey"])
    skin, belly, dark = hex_color(spec["skin"]), hex_color(spec["belly"]), hex_color(spec["dark"])
    accent, bone, eye = hex_color(spec["accent"]), hex_color(spec["bone"]), hex_color(spec["eye"])
    glow = spec["glow"]
    names = list(MONSTER_PARTS) + ["body_glow", "arm_l_glow", "arm_r_glow", "leg_l_glow", "leg_r_glow"]
    parts = {name: MeshBuilder() for name in names}
    body = parts["body"]
    acc_body = parts["body_glow"] if glow else body
    body.box((0, 2.1, 0), (4.4, 4.2, 2.8), skin)
    body.box((0, 1.9, 1.42), (3.0, 3.2, 0.12), belly)
    acc_body.box((-1.6, 4.1, 0.2), (1.6, 0.5, 2.4), accent)
    acc_body.box((1.5, 4.15, -0.2), (1.4, 0.45, 2.0), accent)
    if glow:  # cracks / runes across chest and back
        for (x, y, w, h) in ((-0.8, 2.6, 0.18, 1.6), (0.6, 1.4, 1.4, 0.16), (0.2, 3.2, 0.16, 1.0), (-0.9, 0.9, 1.0, 0.16)):
            acc_body.box((x, y, 1.5), (w, h, 0.06), accent)
            acc_body.box((-x, y + 0.3, -1.43), (w, h, 0.06), accent)
    for i, z in enumerate((-0.9, -0.2, 0.5)):
        body.cone((0, 4.2, z - 0.4), 0.55 - i * 0.08, 1.4 - i * 0.2, bone, segments=4)
    for y in (1.0, 2.2, 3.3):
        body.box((0, y, -1.5), (0.5, 0.5, 0.4), bone)
    head = parts["head"]
    head.box((0, 0.95, 0.3), (2.4, 1.9, 2.2), skin)
    head.box((0, 0.2, 0.55), (2.2, 0.5, 2.0), dark)  # jaw
    for x in (-0.75, -0.25, 0.25, 0.75):
        head.box((x, 0.5, 1.48), (0.22, 0.32, 0.12), bone if variant != "lava" else hex_color("#ffb347"))
    head.box((0, 1.6, 1.41), (2.0, 0.22, 0.12), dark)  # brow
    for side in (-1, 1):
        head.cone((side * 1.0, 1.85, 0.0), 0.32, 1.1, bone, segments=4)
    eyes = parts["eyes"]
    for side in (-1, 1):
        eyes.box((side * 0.55, 1.18, 1.42), (0.5, 0.32, 0.1), eye)
    if variant == "lava":
        eyes.box((0, 0.42, 1.45), (1.9, 0.16, 0.05), accent)  # glowing mouth
    for side, name in ((-1, "arm_l"), (1, "arm_r")):
        arm = parts[name]
        acc = parts[name + "_glow"] if glow else arm
        arm.box((0, -2.0, 0), (1.2, 4.2, 1.25), skin)
        acc.box((side * 0.05, -0.4, 0), (1.35, 1.0, 1.4), accent)
        if glow:
            acc.box((0, -2.4, 0.64), (0.16, 2.2, 0.05), accent)
        arm.box((0, -4.35, 0.1), (1.55, 1.1, 1.55), dark)
        for k in (-0.45, 0.0, 0.45):
            arm.box((k, -4.95, 0.55), (0.25, 0.35, 0.25), bone)
    for name in ("leg_l", "leg_r"):
        leg = parts[name]
        leg.box((0, -1.4, 0), (1.7, 2.9, 1.8), skin)
        leg.box((0, -2.8, 0.25), (1.95, 0.45, 2.5), dark)
        if glow:
            parts[name + "_glow"].box((0.3, -1.2, 0.92), (0.16, 1.6, 0.05), accent)
    return parts


def school_bus(mb: MeshBuilder) -> dict:
    """A yellow bus chassis carrying a little school building."""
    yellow = hex_color("#f6c431")
    brick = hex_color("#d0674f")
    white = hex_color("#fbf8ef")
    mb.box((0, 1.25, 0), (14.5, 1.3, 3.0), yellow)
    mb.box((7.1, 1.9, 0), (0.6, 1.6, 2.9), yellow)
    mb.box((7.45, 2.3, 0), (0.06, 1.0, 2.5), PALETTE["glass"])
    mb.box((0, 0.65, 0), (14.6, 0.25, 3.05), hex_color("#30303a"))
    for z in (-1.2, 1.2):
        mb.box((7.45, 1.2, z), (0.08, 0.35, 0.45), hex_color("#fff7c2"))
    _wheels(mb, (5.2, 1.0, -4.8), 1.25, 0.62)
    # the school on top
    mb.box((-0.6, 4.4, 0), (12.5, 5.0, 5.6), brick)
    for x in np.arange(-5.6, 4.6, 2.0):
        for y in (3.4, 5.4):
            for zf in (2.82, -2.82):
                mb.box((x, y, zf), (1.1, 1.0, 0.1), white)
                mb.box((x, y, zf + (0.03 if zf > 0 else -0.03)), (0.85, 0.75, 0.08), PALETTE["window"])
    mb.box((-0.6, 7.0, 0), (13.0, 0.3, 6.1), white)
    with mb.transform(rotate_y(math.pi / 2)):
        mb.gable_roof((0, 7.1, 0.6), 6.0, 12.5, 2.2, hex_color("#4a5a7a"), overhang=0.3)
    mb.box((-0.6, 10.2, 0), (1.8, 2.4, 1.8), white)  # bell tower
    mb.cone((-0.6, 11.4, 0), 1.5, 2.2, hex_color("#4a5a7a"), segments=4)
    mb.cylinder((-0.6, 10.3, 0.95), 0.6, 0.12, hex_color("#fff7d6"), segments=10, axis="z")
    mb.box((5.0, 8.6, 0), (0.15, 3.2, 0.15), PALETTE["chrome"])
    mb.box((5.7, 9.7, 0), (1.3, 0.8, 0.08), hex_color("#3d9bff"))
    return {"length": 15.0}
