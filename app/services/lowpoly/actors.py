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
        self.materials: dict[str, int] = {"monster_eyes": 3}
        self.monster_variant = "grey"

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
        elif variant == "sister":
            colors = props.SISTER_COLORS
        elif variant == "doll":
            colors = props.DOLL_COLORS
        elif variant == "finn":
            colors = props.FINN_COLORS
        elif variant in ("mom", "mom_fake"):
            parts = props.figure_parts(props.MOM_COLORS if variant == "mom" else props.MOM_FAKE_COLORS)
            parts["head"].box((0, 0.2, -0.36), (0.86, 0.95, 0.16), (props.MOM_COLORS if variant == "mom" else props.MOM_FAKE_COLORS)["hair"])
            keys = {p_name: self._store(f"fig_{variant}_{p_name}", b) for p_name, b in parts.items()}
            if variant == "mom_fake":
                glow = MeshBuilder()
                for side in (-1, 1):
                    glow.box((side * 0.17, 0.5, 0.43), (0.17, 0.09, 0.03), hex_color("#ff3040"))
                keys["head_glow"] = self._store("fig_mom_fake_head_glow", glow)
                self.materials[keys["head_glow"]] = 3
            return keys
        elif variant == "krampus":
            parts = props.krampus_parts()
            keys = {}
            for p_name, builder in parts.items():
                keys[p_name] = self._store(f"fig_krampus_{p_name}", builder)
                if p_name.endswith("_glow"):
                    self.materials[keys[p_name]] = 3
            return keys
        elif variant.startswith(("rich", "poor", "guard")):
            kind = "rich" if variant.startswith("rich") else "poor" if variant.startswith("poor") else "guard"
            digits = variant[len(kind):]
            colors = props.class_colors(kind, int(digits) if digits.isdigit() else 0)
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

    def monster(self, variant: str | None = None) -> dict[str, str]:
        variant = variant or self.monster_variant
        prefix = "monster" if variant == "grey" else f"monster_{variant}"
        keys = {}
        parts = None
        for name in list(props.MONSTER_PARTS) + [p + "_glow" for p in ("body", "arm_l", "arm_r", "leg_l", "leg_r")]:
            key = f"{prefix}_{name}"
            if key not in self.meshes:
                parts = parts or props.monster_parts(variant)
                if len(parts[name]) == 0:
                    continue
                self._store(key, parts[name])
            if name == "eyes" or name.endswith("_glow"):
                self.materials[key] = 3
            keys[name] = key
        return keys

    def raw(self, key: str, vertices: np.ndarray, material: int = 0) -> str:
        if key not in self.meshes:
            self.meshes[key] = vertices
        if material:
            self.materials[key] = material
        return key

    def school_bus(self) -> str:
        if "school_bus" not in self.meshes:
            mb = MeshBuilder()
            props.school_bus(mb)
            self._store("school_bus", mb)
        return "school_bus"

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
    look_turn: float = 0.0  # head yaw offset for the "sit" pose
    label: str = ""  # text floating above the head
    label_style: str = "npc"
    label_locked: bool = False  # keep the label even if the story assigns labels to everyone
    freeze_at: float | None = None  # "switched off": turns grey and stops moving at this time
    unfreeze_at: float | None = None
    label_frozen: str = ""
    label_unfrozen: str = ""
    tint: tuple = (1.0, 1.0, 1.0)  # colour multiplier (e.g. a pale glow for ghosts)

    def frozen(self, t: float) -> bool:
        return self.freeze_at is not None and t >= self.freeze_at and (self.unfreeze_at is None or t < self.unfreeze_at)

    def current_label(self, t: float) -> str:
        if self.frozen(t):
            return self.label_frozen
        if self.unfreeze_at is not None and t >= self.unfreeze_at and self.label_unfrozen:
            return self.label_unfrozen
        return self.label

    def _state(self, t: float):
        if self.path is not None:
            pos, direction = self.path.at(t)
            moving = self.path.stop_at is None or t < 2 * self.path.stop_at / max(self.path.speed, 1e-6)
            if abs(direction[0]) + abs(direction[2]) < 1e-6:  # vertical motion (climbing, elevator)
                return pos, self.heading, moving
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
        frozen = self.frozen(t)
        t_pose = max(self.freeze_at, 0.0) if frozen else t
        pos, heading, moving = self._state(t_pose)
        tt = t_pose + self.phase
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
        elif anim == "climb":
            c = math.sin(tt * 5.0)
            rot["arm_l"], rot["arm_r"] = rotate_x(-2.7 + c * 0.35), rotate_x(-2.7 - c * 0.35)
            rot["leg_l"], rot["leg_r"] = rotate_x(-0.5 + c * 0.45), rotate_x(-0.5 - c * 0.45)
            rot["head"] = rotate_x(-0.35)
        elif anim == "look_up":
            b = math.sin(tt * 1.6)
            rot["head"] = rotate_x(-0.75 + 0.05 * b)
            rot["arm_l"], rot["arm_r"] = rotate_z(-0.08), rotate_z(0.08)
        elif anim == "lurk":  # hunched, head tilted, birch rod tapping
            b = math.sin(tt * 1.3)
            rot["head"] = rotate_z(0.25 + 0.05 * b) @ rotate_x(0.15)
            rot["arm_r"] = rotate_x(-0.5 + math.sin(tt * 4.0) * 0.25)
            rot["arm_l"] = rotate_x(-0.2) @ rotate_z(-0.15)
            body_tilt = 0.32
            lift = 0.02 * (b + 1)
        elif anim == "stomp":  # heavy walk, swinging the rod
            st = math.sin(tt * 4.5)
            rot["leg_l"], rot["leg_r"] = rotate_x(st * 0.55), rotate_x(-st * 0.55)
            rot["arm_r"] = rotate_x(-0.9 + st * 0.6)
            rot["arm_l"] = rotate_x(st * 0.45)
            rot["head"] = rotate_y(math.sin(tt * 1.1) * 0.35)
            body_tilt = 0.25
            lift = abs(math.cos(tt * 4.5)) * 0.1
        elif anim == "doll":  # unnaturally still, head tilted, arms reaching a little
            twitch = 0.25 if (tt % 4.3) < 0.12 else 0.0
            rot["head"] = rotate_z(0.38 + twitch) @ rotate_x(0.1)
            rot["arm_l"], rot["arm_r"] = rotate_x(-0.5), rotate_x(-0.45)
        elif anim == "lie":  # lying in bed, eyes darting around
            rot["head"] = rotate_y(math.sin(tt * 0.9) * 0.45 + math.sin(tt * 2.7) * 0.08)
            rot["arm_l"], rot["arm_r"] = rotate_z(-0.1), rotate_z(0.1)
        elif anim == "sit":  # sitting up in bed, staring at something
            rot["leg_l"], rot["leg_r"] = rotate_x(-1.5), rotate_x(-1.5)
            rot["arm_l"], rot["arm_r"] = rotate_x(-0.4), rotate_x(-0.35)
            rot["head"] = rotate_y(self.look_turn + math.sin(tt * 3.1) * 0.05)
            lift = 0.0
        elif anim == "crouch":
            rot["leg_l"], rot["leg_r"] = rotate_x(-1.2), rotate_x(-1.0)
            rot["arm_l"], rot["arm_r"] = rotate_x(-1.0), rotate_x(-1.1)
            rot["head"] = rotate_y(math.sin(tt * 2.2) * 0.35)  # nervously looking around
            lift = -0.42
            body_tilt = 0.35
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
        base = translate(pos[0], pos[1] + lift, pos[2]) @ rotate_y(heading)
        if anim == "lie":
            base = base @ rotate_x(-math.pi / 2)
        base = base @ scale(self.size)
        if body_tilt:
            base = base @ translate(0, 0.95, 0) @ rotate_x(body_tilt) @ translate(0, -0.95, 0)
        tint, grey = self.tint, 0.0
        if frozen:
            tint, grey = (0.92, 0.94, 1.0), 1.0
            since = t - max(self.freeze_at, 0.0)
            if since < 0.35:  # electric jolt when being switched off
                base = translate(math.sin(t * 90) * 0.06, 0, math.cos(t * 77) * 0.06) @ base
                tint, grey = (0.9, 1.0, 1.4), since / 0.35
        elif self.unfreeze_at is not None and 0 <= t - self.unfreeze_at < 0.5:
            grey = 1.0 - (t - self.unfreeze_at) / 0.5  # colour flows back
        out = []
        for part, key in self.parts.items():
            base_part = part[:-5] if part.endswith("_glow") else part
            px, py, pz = props.FIGURE_PIVOTS[base_part]
            out.append(Draw(key, base @ translate(px, py, pz) @ rot[base_part], tint, grey))
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
class Monster(Actor):
    """The Grey Giant. size scales the ~9.5 unit model (size 5 ~ 47 m tall)."""

    parts: dict[str, str]
    position: np.ndarray
    heading: float = 0.0
    size: float = 5.0
    anim: str = "walk"  # walk | roar | lean | push | idle | wave
    path: LinePath | None = None
    phase: float = 0.0
    ground: Callable[[float, float], float] | None = None
    stride: float = 1.6

    def _state(self, t):
        if self.path is not None:
            pos, d = self.path.at(t)
            heading = heading_for_figure(d)
        else:
            pos, heading = self.position, self.heading
        pos = np.array(pos, dtype=np.float64)
        if self.ground is not None:
            pos[1] = self.ground(float(pos[0]), float(pos[2]))
        return pos, heading

    def focus(self, t: float) -> np.ndarray:
        pos, _ = self._state(t)
        return pos + np.array([0.0, 6.0 * self.size, 0.0])

    def forward(self, t: float) -> np.ndarray:
        _, h = self._state(t)
        return np.array([math.sin(h), 0.0, math.cos(h)])

    def walking(self) -> bool:
        return self.anim in ("walk", "push")

    def step_times(self, duration: float) -> list[float]:
        """Local times at which a foot hits the ground."""
        if not self.walking():
            return []
        period = math.pi / self.stride
        first = (-self.phase) % period
        return [first + k * period for k in range(int(duration / period) + 2) if first + k * period < duration]

    def shake(self, t: float) -> float:
        if not self.walking():
            return 0.0
        period = math.pi / self.stride
        since = (t + self.phase) % period
        return math.exp(-since * 9.0)

    def draws(self, t: float) -> list[Draw]:
        pos, heading = self._state(t)
        tt = t + self.phase
        rot = {p: np.eye(4) for p in props.MONSTER_PARTS}
        body = np.eye(4)
        lift = 0.0
        if self.anim in ("walk", "push"):
            s = math.sin(tt * self.stride)
            rot["leg_l"], rot["leg_r"] = rotate_x(s * 0.45), rotate_x(-s * 0.45)
            lift = abs(math.cos(tt * self.stride)) * 0.25
            body = rotate_z(s * 0.06)
            if self.anim == "walk":
                rot["arm_l"], rot["arm_r"] = rotate_x(-s * 0.4), rotate_x(s * 0.4)
                body = body @ rotate_x(0.12)
            else:
                rot["arm_l"] = rot["arm_r"] = rotate_x(-1.35 + s * 0.05)
                body = body @ rotate_x(0.32)
            rot["head"] = rotate_y(math.sin(tt * 0.5) * 0.2)
        elif self.anim == "roar":
            shake = math.sin(tt * 30) * 0.04
            rot["arm_l"] = rotate_z(-2.2 + shake)
            rot["arm_r"] = rotate_z(2.2 - shake)
            rot["head"] = rotate_x(-0.45 + shake)
            body = rotate_x(-0.15)
        elif self.anim == "lean":
            k = min(t / 2.5, 1.0)
            body = rotate_x(0.55 * k)
            rot["head"] = rotate_x(0.35 * k) @ rotate_y(math.sin(tt * 0.6) * 0.15)
            rot["arm_l"] = rotate_x(-0.5 * k) @ rotate_z(-0.15)
            rot["arm_r"] = rotate_x(-0.5 * k) @ rotate_z(0.15)
            rot["leg_l"], rot["leg_r"] = rotate_x(-0.25 * k), rotate_x(0.1 * k)
        elif self.anim == "wave":
            rot["arm_r"] = rotate_z(2.3 + math.sin(tt * 3) * 0.3)
            rot["head"] = rotate_z(math.sin(tt * 1.2) * 0.15)
            body = rotate_z(math.sin(tt * 1.2) * 0.04)
        else:  # idle
            b = math.sin(tt * 1.1)
            body = rotate_x(0.05 + b * 0.02)
            rot["head"] = rotate_y(math.sin(tt * 0.4) * 0.3)
            rot["arm_l"], rot["arm_r"] = rotate_z(-0.05 * b), rotate_z(0.05 * b)
        base = translate(pos[0], pos[1] + lift * self.size, pos[2]) @ rotate_y(heading) @ scale(self.size)
        body_m = base @ translate(*props.MONSTER_HIP) @ body
        out = [Draw(self.parts["body"], body_m)]
        matrices = {"body": body_m}
        for name, off in props.MONSTER_OFFSETS.items():
            local = rot["head"] if name == "eyes" else rot[name]
            matrices[name] = body_m @ translate(*off) @ local
            out.append(Draw(self.parts[name], matrices[name]))
        for name, off in props.MONSTER_LEGS.items():
            matrices[name] = base @ translate(*off) @ rot[name]
            out.append(Draw(self.parts[name], matrices[name]))
        for name in ("body", "arm_l", "arm_r", "leg_l", "leg_r"):
            key = self.parts.get(name + "_glow")
            if key:
                out.append(Draw(key, matrices[name]))
        return out


@dataclass
class Prop(Actor):
    """A static or hinged mesh (doors, lamp shade, glowing eyes...).

    angle(t) rotates around the local y axis at `position` (a hinge);
    visible(t) can hide it (e.g. blinking eyes).
    """

    key: str
    position: np.ndarray
    heading: float = 0.0
    angle: Callable[[float], float] | None = None
    visible: Callable[[float], bool] | None = None

    def focus(self, t: float) -> np.ndarray:
        return self.position

    def draws(self, t: float) -> list[Draw]:
        if self.visible is not None and not self.visible(t):
            return []
        a = self.angle(t) if self.angle else 0.0
        return [Draw(self.key, translate(*self.position) @ rotate_y(self.heading + a))]


class Midpoint(Actor):
    """Invisible focus between two actors (e.g. the bus and the giant pushing it)."""

    def __init__(self, a: Actor, b: Actor, weight: float = 0.5, height: float = 8.0):
        self.a, self.b, self.weight, self.height = a, b, weight, height

    def draws(self, t: float) -> list[Draw]:
        return []

    def focus(self, t: float) -> np.ndarray:
        p = self.a.focus(t) * (1 - self.weight) + self.b.focus(t) * self.weight
        p[1] = self.height
        return p

    def forward(self, t: float) -> np.ndarray:
        return self.a.forward(t)


@dataclass
class Group(Actor):
    members: list[Actor] = field(default_factory=list)
    focus_index: int = 0
    look: Actor | None = None  # optional second subject the camera looks toward (e.g. the monster)
    look_ratio: float | None = None  # how far towards `look` staged shots aim (0 = focus, 1 = look target)

    def draws(self, t: float) -> list[Draw]:
        out: list[Draw] = []
        for m in self.members:
            out.extend(m.draws(t))
        return out

    def focus(self, t: float) -> np.ndarray:
        return self.members[self.focus_index].focus(t)

    def forward(self, t: float) -> np.ndarray:
        return self.members[self.focus_index].forward(t)

    def shake(self, t: float) -> float:
        return max((m.shake(t) for m in self.members if isinstance(m, Monster)), default=0.0)
