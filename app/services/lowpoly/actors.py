"""Animated actors: blocky toy figures, trucks carrying houses, buses,
buildings that rise or collapse. Each actor yields renderer draws per frame.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import props
from .mesh import MeshBuilder, hex_color, rotate_x, rotate_y, rotate_z, scale, translate
from .renderer import Draw

NPC_VARIANTS = 12
CARGO_VARIANTS = 6


class ActorLibrary:
    """Builds actor meshes lazily and remembers which ones the renderer needs."""

    def __init__(self, seed: int):
        self.seed = seed
        self.meshes: dict[str, np.ndarray] = {}

    def _store(self, key: str, builder: MeshBuilder) -> str:
        if key not in self.meshes:
            self.meshes[key] = builder.build()
        return key

    def figure(self, variant: str) -> dict[str, str]:
        """variant: "hero" or "npc<N>"; returns part -> mesh key."""
        keys = {f"fig_{variant}_{p}": p for p in props.FIGURE_PARTS}
        if all(k in self.meshes for k in keys):
            return {p: k for k, p in keys.items()}
        if variant == "hero":
            colors = props.HERO_COLORS
        else:
            idx = int(variant[3:]) if variant.startswith("npc") else 0
            colors = props.figure_colors(np.random.default_rng(self.seed * 1000 + idx))
        parts = props.figure_parts(colors)
        return {p: self._store(f"fig_{variant}_{p}", parts[p]) for p in props.FIGURE_PARTS}

    def truck(self, color_idx: int, cargo: str | None) -> str:
        key = f"truck_{color_idx}_{cargo}"
        if key in self.meshes:
            return key
        mb = MeshBuilder()
        color = props.PALETTE["vehicle"][color_idx % len(props.PALETTE["vehicle"])]
        info = props.truck(mb, color)
        if cargo:
            rng = np.random.default_rng(self.seed * 77 + zlib.crc32(cargo.encode()) % 1000)
            with mb.transform(translate(-0.4, info["cargo_y"], 0)):
                if cargo == "hero_house":
                    props.house(mb, (0, 0, 0), rng, width=10, depth=7.5, floors=1,
                                wall=hex_color("#fff1c4"), roof=hex_color("#d0574a"))
                elif cargo.startswith("house"):
                    props.house(mb, (0, 0, 0), rng, width=float(rng.uniform(9.5, 11.5)), depth=float(rng.uniform(6, 8)),
                                floors=int(rng.integers(1, 3)))
                elif cargo == "shop":
                    with mb.transform(scale(0.85)):
                        props.shop(mb, (0, 0, 0), rng)
                elif cargo == "hall":
                    with mb.transform(scale(0.5)):
                        props.town_hall(mb, (0, 0, 0), rng)
        return self._store(key, mb)

    def bus(self) -> str:
        if "bus" not in self.meshes:
            mb = MeshBuilder()
            props.bus(mb)
            self._store("bus", mb)
        return "bus"

    def car(self, color_idx: int) -> str:
        key = f"car_{color_idx}"
        if key not in self.meshes:
            mb = MeshBuilder()
            props.car(mb, props.PALETTE["vehicle"][color_idx % len(props.PALETTE["vehicle"])])
            self._store(key, mb)
        return key

    def building(self, kind: str) -> str:
        key = f"building_{kind}"
        if key not in self.meshes:
            mb = MeshBuilder()
            rng = np.random.default_rng(self.seed * 31 + len(kind))
            if kind == "hero_house":
                props.house(mb, (0, 0, 0), rng, width=9, depth=8, floors=1,
                            wall=hex_color("#fff1c4"), roof=hex_color("#d0574a"))
            elif kind == "tower":
                props.house(mb, (0, 0, 0), rng, width=10, depth=10, floors=5, wall=hex_color("#dceeff"))
            else:
                props.house(mb, (0, 0, 0), rng, width=11, depth=8, floors=2)
            self._store(key, mb)
        return key


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


@dataclass
class LinePath:
    start: np.ndarray
    direction: np.ndarray
    speed: float
    stop_at: float | None = None  # distance after which the actor stops (eased)

    def at(self, t: float):
        dist = self.speed * t
        if self.stop_at is not None:
            # decelerate smoothly into the stop distance
            total_time = 2 * self.stop_at / max(self.speed, 1e-6)
            u = min(t / total_time, 1.0)
            dist = self.stop_at * (1 - (1 - u) ** 2)
        pos = self.start + self.direction * dist
        return pos, self.direction


def heading_for_figure(direction) -> float:
    return math.atan2(direction[0], direction[2])


def heading_for_vehicle(direction) -> float:
    return math.atan2(-direction[2], direction[0])


# --------------------------------------------------------------------------- #
# Actors
# --------------------------------------------------------------------------- #


class Actor:
    def draws(self, t: float) -> list[Draw]:  # pragma: no cover - interface
        raise NotImplementedError

    def focus(self, t: float) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, t: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])


@dataclass
class Figure(Actor):
    parts: dict[str, str]
    position: np.ndarray
    heading: float = 0.0
    anim: str = "idle"
    phase: float = 0.0
    path: LinePath | None = None
    size: float = 1.0
    face_target: np.ndarray | None = None

    def _state(self, t: float):
        if self.path is not None:
            pos, direction = self.path.at(t)
            moving = self.path.stop_at is None or t < 2 * self.path.stop_at / max(self.path.speed, 1e-6)
            return pos, heading_for_figure(direction), moving
        heading = self.heading
        if self.face_target is not None:
            d = self.face_target - self.position
            heading = math.atan2(d[0], d[2])
        return self.position, heading, False

    def focus(self, t: float) -> np.ndarray:
        pos, _, _ = self._state(t)
        return pos + np.array([0.0, 1.6 * self.size, 0.0])

    def forward(self, t: float) -> np.ndarray:
        _, heading, _ = self._state(t)
        return np.array([math.sin(heading), 0.0, math.cos(heading)])

    def draws(self, t: float) -> list[Draw]:
        pos, heading, moving = self._state(t)
        tt = t + self.phase
        anim = self.anim
        if self.path is not None and not moving and anim in ("walk", "run"):
            anim = "idle"
        rot = {p: np.eye(4) for p in props.FIGURE_PARTS}
        lift = 0.0
        body_tilt = 0.0
        if anim in ("walk", "run"):
            freq, amp = (7.0, 0.65) if anim == "walk" else (11.0, 1.0)
            s = math.sin(tt * freq)
            rot["leg_l"], rot["leg_r"] = rotate_x(s * amp), rotate_x(-s * amp)
            rot["arm_l"], rot["arm_r"] = rotate_x(-s * amp * 0.9), rotate_x(s * amp * 0.9)
            lift = abs(math.cos(tt * freq)) * (0.08 if anim == "walk" else 0.18)
            body_tilt = 0.0 if anim == "walk" else 0.18
        elif anim in ("jump", "cheer"):
            j = abs(math.sin(tt * 5.0))
            lift = j * (0.9 if anim == "jump" else 0.6)
            wave = math.sin(tt * 10) * 0.25
            rot["arm_l"] = rotate_z(-2.6 + wave)
            rot["arm_r"] = rotate_z(2.6 - wave)
            rot["leg_l"], rot["leg_r"] = rotate_x(-0.3 * j), rotate_x(-0.3 * j)
        elif anim == "wave":
            rot["arm_r"] = rotate_z(2.5 + math.sin(tt * 9) * 0.35)
            rot["arm_l"] = rotate_x(math.sin(tt * 1.5) * 0.05)
        elif anim == "panic":
            rot["arm_l"] = rotate_z(-2.7 + math.sin(tt * 18) * 0.4)
            rot["arm_r"] = rotate_z(2.7 + math.sin(tt * 18 + 1) * 0.4)
            s = math.sin(tt * 14)
            rot["leg_l"], rot["leg_r"] = rotate_x(s * 0.6), rotate_x(-s * 0.6)
            lift = abs(math.cos(tt * 14)) * 0.12
        elif anim == "sad":
            rot["head"] = rotate_x(0.38)
            rot["arm_l"], rot["arm_r"] = rotate_z(0.05), rotate_z(-0.05)
            body_tilt = 0.12
        elif anim == "point":
            rot["arm_r"] = rotate_x(-1.55 + math.sin(tt * 3) * 0.08)
            rot["head"] = rotate_y(math.sin(tt * 1.3) * 0.15)
        elif anim == "talk":
            rot["head"] = rotate_x(math.sin(tt * 6) * 0.08) @ rotate_y(math.sin(tt * 1.7) * 0.2)
            rot["arm_r"] = rotate_x(-0.6 - math.sin(tt * 4) * 0.4)
            rot["arm_l"] = rotate_x(-0.3 - math.sin(tt * 3.3 + 1) * 0.3)
        else:  # idle / breathing
            b = math.sin(tt * 2.2)
            rot["arm_l"], rot["arm_r"] = rotate_z(-0.06 * b), rotate_z(0.06 * b)
            rot["head"] = rotate_y(math.sin(tt * 0.7) * 0.25)
            lift = 0.02 * (b + 1)
        base = translate(pos[0], pos[1] + lift, pos[2]) @ rotate_y(heading) @ scale(self.size)
        if body_tilt:
            base = base @ translate(0, 0.95, 0) @ rotate_x(body_tilt) @ translate(0, -0.95, 0)
        out = []
        for part, key in self.parts.items():
            px, py, pz = props.FIGURE_PIVOTS[part]
            out.append(Draw(key, base @ translate(px, py, pz) @ rot[part]))
        return out


@dataclass
class Vehicle(Actor):
    key: str
    path: LinePath
    length: float = 17.0
    bounce: float = 0.04
    phase: float = 0.0

    def _state(self, t):
        return self.path.at(t)

    def focus(self, t: float) -> np.ndarray:
        pos, _ = self._state(t)
        return pos + np.array([0.0, 3.0, 0.0])

    def forward(self, t: float) -> np.ndarray:
        return self.path.direction

    def draws(self, t: float) -> list[Draw]:
        pos, direction = self._state(t)
        bob = math.sin((t + self.phase) * 9) * self.bounce
        model = translate(pos[0], pos[1] + bob, pos[2]) @ rotate_y(heading_for_vehicle(direction))
        return [Draw(self.key, model)]


@dataclass
class Building(Actor):
    key: str
    position: np.ndarray
    heading: float = 0.0
    mode: str = "static"  # static | rise | collapse
    duration: float = 5.0

    def focus(self, t: float) -> np.ndarray:
        return self.position + np.array([0.0, 4.0, 0.0])

    def draws(self, t: float) -> list[Draw]:
        u = min(max(t / max(self.duration * 0.8, 0.1), 0.0), 1.0)
        base = translate(*self.position) @ rotate_y(self.heading)
        if self.mode == "rise":
            e = 1 - (1 - u) ** 3
            wobble = math.sin(t * 20) * 0.02 * (1 - e)
            model = base @ scale(1.0, max(e, 0.02), 1.0) @ rotate_z(wobble)
        elif self.mode == "collapse":
            e = u * u
            model = base @ translate(0, -e * 6.0, 0) @ rotate_z(e * 0.5) @ rotate_x(e * 0.25)
            model = model @ translate(math.sin(t * 40) * 0.08 * (1 - e), 0, 0)
        else:
            model = base
        return [Draw(self.key, model)]


@dataclass
class Group(Actor):
    members: list[Actor] = field(default_factory=list)
    focus_index: int = 0

    def draws(self, t: float) -> list[Draw]:
        out: list[Draw] = []
        for m in self.members:
            out.extend(m.draws(t))
        return out

    def focus(self, t: float) -> np.ndarray:
        return self.members[self.focus_index].focus(t)

    def forward(self, t: float) -> np.ndarray:
        return self.members[self.focus_index].forward(t)
