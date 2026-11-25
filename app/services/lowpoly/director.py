"""Turns a storyboard + scene durations into per-frame render state:
actors, camera, mood blending and lightning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import moods as mood_lib
from .actors import (
    NPC_VARIANTS,
    Actor,
    ActorLibrary,
    Building,
    Figure,
    Group,
    LinePath,
    Vehicle,
)
from .camera import shot_camera
from .mesh import translate
from .renderer import Camera, Draw
from .storyboard import Scene, Storyboard
from .world import SETTINGS, World

MOOD_BLEND_SECONDS = 1.2
PEOPLE_ACTIONS = {"walk", "run", "crowd", "celebrate", "argue", "alone", "wave"}
VEHICLE_ACTIONS = {"convoy", "arrive", "drive", "bus"}
ROAD_Y = 0.07


class Marker(Actor):
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float64)

    def draws(self, t):
        return []

    def focus(self, t):
        return self.position

    def forward(self, t):
        return np.array([1.0, 0.0, 0.0])


@dataclass
class PlannedScene:
    scene: Scene
    start: float
    end: float
    actor: Actor
    yaw: float
    scale: float
    mood_from: mood_lib.Mood
    mood_to: mood_lib.Mood
    flashes: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _road_frame(setting_name: str):
    s = SETTINGS[setting_name]
    lane = 2.2
    if s.road_axis == "x":
        direction = np.array([1.0, 0.0, 0.0])
        point = np.array([s.stage[0], ROAD_Y, s.road_line + lane])
    else:
        direction = np.array([0.0, 0.0, -1.0])
        point = np.array([s.road_line + lane, ROAD_Y, s.stage[1]])
    if setting_name == "coast":
        direction = -direction
        point = np.array([s.stage[0] + 18, ROAD_Y, s.road_line - lane])
    return point, direction


def _npc(lib: ActorLibrary, rng: np.random.Generator) -> dict:
    return lib.figure(f"npc{int(rng.integers(NPC_VARIANTS))}")


def build_actor(action: str, setting_name: str, duration: float, yaw: float, mood: str,
                lib: ActorLibrary, rng: np.random.Generator) -> tuple[Actor, float]:
    """Returns the actor group for a scene and a camera distance scale."""
    s = SETTINGS[setting_name]
    stage = np.array([s.stage[0], 0.0, s.stage[1]])
    hero = lib.figure("hero")
    facing_cam = yaw  # figure heading that looks at a camera placed at `yaw`
    road_point, road_dir = _road_frame(setting_name)

    if setting_name == "coast" and action == "convoy":
        action = "arrive"

    if action in ("convoy", "arrive"):
        n = 5
        spacing = 21.0
        speed = 10.0
        members = []
        cargos = ["house0", "house1", "shop", "house2", "hall", "house3"]
        for i in range(n):
            key = lib.truck(i + int(rng.integers(3)), cargos[i % len(cargos)])
            if action == "convoy":
                lead = road_point - road_dir * (speed * duration / 2) + road_dir * spacing * (n / 2)
                path = LinePath(lead - road_dir * spacing * i, road_dir, speed)
            else:
                stop = speed * duration * 0.42
                path = LinePath(road_point - road_dir * (stop + spacing * i), road_dir, speed, stop_at=stop)
            members.append(Vehicle(key, path, phase=float(rng.uniform(0, 6))))
        return Group(members, focus_index=n // 2), 1.0

    if action == "drive":
        speed = 9.0
        path = LinePath(road_point - road_dir * speed * duration / 2, road_dir, speed)
        truck = Vehicle(lib.truck(0, "hero_house"), path)
        car = Vehicle(lib.car(3), LinePath(path.start - road_dir * 16, road_dir, speed))
        return Group([truck, car], 0), 0.6

    if action == "bus":
        speed = 11.0
        path = LinePath(road_point - road_dir * speed * duration / 2, road_dir, speed)
        bus = Vehicle(lib.bus(), path, bounce=0.03)
        cars = [Vehicle(lib.car(i + 1), LinePath(path.start - road_dir * (14 + 9 * i), road_dir, speed)) for i in range(2)]
        return Group([bus] + cars, 0), 0.6

    if action in ("walk", "run"):
        speed = 1.5 if action == "walk" else 4.8
        heading = yaw + math.pi / 2 + 0.55
        d = np.array([math.sin(heading), 0.0, math.cos(heading)])
        start = stage - d * speed * duration / 2
        side = np.array([d[2], 0.0, -d[0]])
        members: list[Actor] = [Figure(hero, start, anim=action, path=LinePath(start, d, speed))]
        if action == "walk":
            members.append(Figure(_npc(lib, rng), start + side * 1.7, anim="walk", phase=0.4,
                                  path=LinePath(start + side * 1.7, d, speed)))
        else:
            for i in range(3):
                p = start - d * (4 + 3 * i) + side * rng.uniform(-2.5, 2.5)
                members.append(Figure(_npc(lib, rng), p, anim="run", phase=float(rng.uniform(0, 3)),
                                      path=LinePath(p, d, speed * rng.uniform(0.92, 1.0))))
        return Group(members, 0), 0.45

    if action in ("crowd", "celebrate", "argue"):
        hero_fig = Figure(hero, stage, heading=facing_cam + 0.35,
                          anim={"crowd": "talk", "celebrate": "cheer", "argue": "talk"}[action])
        members = [hero_fig]
        if action == "argue":
            for side in (-1, 1):
                ang = facing_cam + side * 1.1
                pos = stage + np.array([math.sin(ang), 0, math.cos(ang)]) * 3.2
                members.append(Figure(_npc(lib, rng), pos, anim="point", phase=float(rng.uniform(0, 3)), face_target=stage))
        else:
            count = 14 if action == "crowd" else 18
            for i in range(count):
                ang = facing_cam + math.pi + rng.uniform(-2.3, 2.3)
                r = rng.uniform(4.5, 10.0)
                pos = stage + np.array([math.sin(ang), 0, math.cos(ang)]) * r
                anim = "cheer" if action == "celebrate" else str(rng.choice(["idle", "talk", "point", "idle"]))
                members.append(Figure(_npc(lib, rng), pos, anim=anim, phase=float(rng.uniform(0, 5)), face_target=stage))
        return Group(members, 0), 0.5

    if action in ("alone", "wave"):
        sad = mood in ("storm", "night", "overcast")
        anim = "wave" if action == "wave" else ("sad" if sad else "idle")
        return Group([Figure(hero, stage, heading=facing_cam, anim=anim)], 0), 0.4

    if action in ("build", "collapse"):
        ang = facing_cam + math.pi + 0.5
        site = stage + np.array([math.sin(ang), 0, math.cos(ang)]) * 14
        key = lib.building("tower" if action == "build" else "house")
        building = Building(key, site, heading=float(rng.uniform(-0.3, 0.3)),
                            mode="rise" if action == "build" else "collapse", duration=duration)
        watcher = Figure(hero, stage, anim="point" if action == "build" else "panic", face_target=site)
        group = Group([building, watcher], 1)
        return group, 0.7

    # landscape
    return Group([Marker(stage + np.array([0.0, 2.0, 0.0]))], 0), 1.6


def flash_value(flashes: list[float], t: float) -> float:
    v = 0.0
    for f in flashes:
        dt = t - f
        if 0 <= dt < 0.7:
            v += math.exp(-dt * 16) + (0.7 * math.exp(-(dt - 0.13) * 14) if dt > 0.13 else 0.0)
    return min(v, 1.0) * 0.55


class Director:
    def __init__(self, world: World, storyboard: Storyboard, durations: list[float], seed: int = 0):
        self.world = world
        self.storyboard = storyboard
        self.library = ActorLibrary(seed)
        rng = np.random.default_rng(seed + 99)
        self.scenes: list[PlannedScene] = []
        t = 0.0
        prev_mood = mood_lib.get(storyboard.scenes[0].mood)
        for scene, dur in zip(storyboard.scenes, durations):
            setting = SETTINGS[scene.setting]
            yaw = setting.view_heading + float(rng.uniform(-0.6, 0.6))
            actor, cam_scale = build_actor(scene.action, scene.setting, dur, yaw, scene.mood, self.library, rng)
            target = mood_lib.get(scene.mood)
            flashes = []
            if target.lightning > 0:
                ft = t + float(rng.uniform(0.5, 1.6))
                while ft < t + dur - 0.3:
                    flashes.append(ft)
                    ft += float(rng.uniform(2.2, 4.5))
            self.scenes.append(PlannedScene(scene, t, t + dur, actor, yaw, cam_scale, prev_mood, target, flashes))
            prev_mood = target
            t += dur
        self.duration = t
        self._cursor = 0

    # ------------------------------------------------------------------ #
    @property
    def cut_times(self) -> list[float]:
        return [s.start for s in self.scenes[1:]]

    @property
    def flashes(self) -> list[float]:
        return [f for s in self.scenes for f in s.flashes]

    @property
    def storm_spans(self) -> list[tuple[float, float]]:
        return [(s.start, s.end) for s in self.scenes if s.mood_to.rain > 0]

    def scene_at(self, t: float) -> PlannedScene:
        while self._cursor + 1 < len(self.scenes) and self.scenes[self._cursor + 1].start <= t:
            self._cursor += 1
        while self._cursor > 0 and self.scenes[self._cursor].start > t:
            self._cursor -= 1
        return self.scenes[self._cursor]

    def static_draws(self, t: float) -> list[Draw]:
        return [
            Draw("static", np.eye(4)),
            Draw("water", np.eye(4)),
            Draw("clouds", translate(t * 1.8 - 60, 0, 0)),
        ]

    def _obstacles(self, ps: PlannedScene, local: float) -> np.ndarray:
        """Static buildings plus houses currently riding on trucks."""
        dynamic = []
        members = ps.actor.members if isinstance(ps.actor, Group) else [ps.actor]
        for m in members:
            if isinstance(m, Vehicle) and ("house" in m.key or "shop" in m.key or "hall" in m.key):
                p = m.focus(local)
                dynamic.append((p[0], p[2], 8.0, 13.0))
            elif isinstance(m, Building):
                dynamic.append((m.position[0], m.position[2], 7.5, 16.0))
        if not dynamic:
            return self.world.obstacles
        return np.concatenate([self.world.obstacles, np.array(dynamic, dtype=np.float64)])

    def frame(self, t: float):
        ps = self.scene_at(t)
        local = t - ps.start
        u = local / max(ps.duration, 1e-6)
        k = min(local / MOOD_BLEND_SECONDS, 1.0)
        mood = mood_lib.blend(ps.mood_from, ps.mood_to, k * k * (3 - 2 * k))
        cam: Camera = shot_camera(
            ps.scene.shot, u, local, ps.actor.focus, ps.actor.forward, ps.yaw, ps.scale, ground=self.world.ground,
            obstacles=self._obstacles(ps, local),
        )
        draws = self.static_draws(t) + ps.actor.draws(local)
        flash = flash_value(ps.flashes, t)
        dist = float(np.linalg.norm(np.asarray(cam.eye) - np.asarray(cam.target)))
        extent = float(np.clip(dist * 0.9 + 14, 22, 320))
        return cam, mood, draws, flash, extent
