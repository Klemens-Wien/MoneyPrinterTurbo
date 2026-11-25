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
    Midpoint,
    Monster,
    Prop,
    Vehicle,
)
from .camera import shot_camera
from .mesh import translate
from .renderer import Camera, Draw
from .storyboard import Scene, Storyboard
from . import interior
from .world import HOME_POS, SETTINGS, World

INTERIOR_ACTIONS = {"bed", "sit_up", "noises", "doorway", "closet", "empty_doorway"}

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
    shots: list[tuple[float, str, float]] = field(default_factory=list)  # (local start, shot, yaw)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shot_at(self, local: float) -> tuple[str, float, float, float]:
        """Return (shot, yaw, sub-shot progress 0..1, sub-shot local time)."""
        shots = self.shots or [(0.0, self.scene.shot, self.yaw)]
        idx = 0
        for i, (start, _, _) in enumerate(shots):
            if local >= start:
                idx = i
        start, shot, yaw = shots[idx]
        end = shots[idx + 1][0] if idx + 1 < len(shots) else self.duration
        return shot, yaw, (local - start) / max(end - start, 1e-6), local - start


# Shots that suit each kind of subject: used for the extra camera angles inside a scene
# (fast-paced shorts change the picture every ~2-3 seconds).
ALT_SHOTS = {
    "staged": ("low_angle", "closeup", "crane_up", "dolly_in"),
    "vehicle": ("drone_follow", "side_track", "chase_cam", "aerial_orbit"),
    "figure": ("closeup", "low_angle", "dolly_in", "pull_back", "aerial_orbit"),
    "other": ("flyover", "aerial_orbit", "crane_up", "top_down"),
}
SUBSHOT_MIN = 2.2


def _subject_kind(actor: Actor) -> str:
    if isinstance(actor, Group):
        if actor.look is not None:
            return "staged"
        subject = actor.members[actor.focus_index]
        if isinstance(subject, Vehicle):
            return "vehicle"
        if isinstance(subject, Figure):
            return "figure"
    return "other"


def plan_subshots(primary: str, yaw: float, duration: float, actor: Actor, rng: np.random.Generator,
                  fast_cuts: bool = True) -> list[tuple[float, str, float]]:
    shots = [(0.0, primary, yaw)]
    if not fast_cuts:
        return shots
    count = max(1, min(3, round(duration / 2.8)))
    while count > 1 and duration / count < SUBSHOT_MIN:
        count -= 1
    options = [s for s in ALT_SHOTS[_subject_kind(actor)] if s != primary]
    rng.shuffle(options)
    for i in range(1, count):
        alt = options[(i - 1) % len(options)] if options else primary
        side = 1 if i % 2 else -1
        shots.append((duration * i / count, alt, yaw + side * float(rng.uniform(0.7, 1.4))))
    return shots


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


def _flat(x: float, z: float) -> float:
    return 0.0


def build_actor(action: str, setting_name: str, duration: float, yaw: float, mood: str,
                lib: ActorLibrary, rng: np.random.Generator, ground=_flat) -> tuple[Actor, float]:
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

    if action == "house_leaves":
        speed = 8.0
        path = LinePath(road_point - road_dir * speed * duration / 2, road_dir, speed)
        truck = Vehicle(lib.truck(int(rng.integers(7)), str(rng.choice(["house0", "house1", "house2", "house3"]))), path)
        watcher = Figure(_npc(lib, rng), road_point + np.array([road_dir[2], 0, -road_dir[0]]) * 9 + road_dir * 4,
                         anim="panic", face_target=road_point)
        return Group([truck, watcher], 0), 0.6

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
            buddy, size = (lib.figure("sister"), 0.72) if setting_name == "home" else (_npc(lib, rng), 1.0)
            members.append(Figure(buddy, start + side * 1.7, anim="walk", phase=0.4, size=size,
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

    toward_cam = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    away = -toward_cam

    def giant(pos, heading=0.0, anim="walk", path=None, size=5.0):
        pos = np.zeros(3) if pos is None else pos
        return Monster(lib.monster(), np.asarray(pos, dtype=np.float64), heading=heading, size=size,
                       anim=anim, path=path, phase=float(rng.uniform(0, 3)), ground=ground)

    def face(src, dst) -> float:
        d = np.asarray(dst) - np.asarray(src)
        return math.atan2(d[0], d[2])

    if action == "chase":
        n, spacing, speed = 5, 21.0, 10.0
        lead = road_point - road_dir * (speed * duration / 2) + road_dir * spacing * (n / 2)
        members: list[Actor] = []
        cargos = ["house0", "house1", "shop", "house2", "hall"]
        for i in range(n):
            members.append(Vehicle(lib.truck(i + int(rng.integers(3)), cargos[i]),
                                   LinePath(lead - road_dir * spacing * i, road_dir, speed), phase=float(rng.uniform(0, 6))))
        members.append(giant(None, path=LinePath(lead - road_dir * (spacing * n + 75), road_dir, speed)))
        return Group(members, focus_index=n // 2), 1.0

    if action == "monster":
        pos = stage + away * 95
        monster = giant(pos, heading=face(pos, stage), anim="roar")
        viewer = Marker(stage + np.array([0.0, 1.6, 0.0]))
        head = Marker(np.array([pos[0], ground(pos[0], pos[2]) + 38.0, pos[2]]))
        return Group([monster, viewer, head], 1, look=head), 1.6

    bus_heading_dir = road_dir
    if action in ("breakdown", "panic", "standoff"):
        bus = Vehicle(lib.school_bus(), LinePath(road_point.copy(), bus_heading_dir, 0.0), bounce=0.0)
        if action == "breakdown":
            hero_fig = Figure(hero, road_point + toward_cam * 5 + bus_heading_dir * 3, anim="point",
                              face_target=road_point)
            m_pos = road_point + away * 230
            monster = giant(m_pos, path=LinePath(m_pos, -away, 3.0))
            return Group([bus, hero_fig, monster], 0), 1.3
        if action == "panic":
            members = [bus]
            hero_pos = road_point + toward_cam * 6
            members.append(Figure(hero, hero_pos, anim="panic", face_target=road_point + away * 100))
            for i in range(16):
                ang = rng.uniform(0, 2 * math.pi)
                d = np.array([math.sin(ang), 0.0, math.cos(ang)])
                start = road_point + d * rng.uniform(3, 9) + toward_cam * 3
                members.append(Figure(_npc(lib, rng), start, anim="panic", phase=float(rng.uniform(0, 3)),
                                      path=LinePath(start, d, float(rng.uniform(2.5, 4.5)))))
            m_pos = road_point + away * 120
            monster = giant(m_pos, path=LinePath(m_pos, toward_cam, 4.0))
            members.append(monster)
            return Group(members, 0, look=monster), 0.9
        # standoff: the hero stands on the road in front of the school, the giant faces both
        hero_pos = road_point + bus_heading_dir * 11 + toward_cam * 1.5
        m_pos = hero_pos + bus_heading_dir * 40 + away * 8
        hero_fig = Figure(hero, hero_pos, heading=face(hero_pos, m_pos), anim="idle")
        monster = giant(m_pos, heading=face(m_pos, hero_pos), anim="lean")
        return Group([bus, hero_fig, monster], 1, look=monster), 1.2

    if setting_name == "bedroom" or action in INTERIOR_ACTIONS:
        return _bedroom_scene(action, duration, lib, rng, hero)

    if action == "lockdown":
        return Group([Marker(np.array([0.0, 6.0, 10.0]))], 0), 1.5

    if action == "watch":
        home = np.array(HOME_POS)
        m_pos = home + np.array([0.0, 0.0, 34.0])
        monster = giant(m_pos, heading=math.pi, anim="idle")
        window = Marker(home + np.array([0.0, 3.5, 4.5]))
        return Group([monster, window], 1, look=monster), 1.0

    if action == "rampage":
        speed = 5.5
        start = road_point - road_dir * (speed * duration / 2 + 40)
        monster = giant(start, path=LinePath(start, road_dir, speed))
        monster.stride = 1.9
        members: list[Actor] = []
        crowd_start = start + road_dir * 48
        hero_fig = Figure(hero, crowd_start + np.array([0.0, 0.0, 1.5]), anim="run",
                          path=LinePath(crowd_start + np.array([0.0, 0.0, 1.5]), road_dir, speed * 1.08))
        members.append(hero_fig)
        side = np.array([road_dir[2], 0.0, -road_dir[0]])
        for i in range(10):
            p = crowd_start + road_dir * rng.uniform(-4, 14) + side * rng.uniform(-4, 4)
            members.append(Figure(_npc(lib, rng), p, anim="run", phase=float(rng.uniform(0, 3)),
                                  path=LinePath(p, road_dir, speed * float(rng.uniform(1.0, 1.2)))))
        members.append(monster)
        return Group(members, 0, look=monster), 1.0

    if action == "street_panic":
        members = [Figure(hero, stage.copy(), anim="panic", face_target=stage + away * 100)]
        for i in range(14):
            ang = rng.uniform(0, 2 * math.pi)
            d = np.array([math.sin(ang), 0.0, math.cos(ang)])
            p = stage + d * rng.uniform(2, 8)
            members.append(Figure(_npc(lib, rng), p, anim="panic", phase=float(rng.uniform(0, 3)),
                                  path=LinePath(p, d, float(rng.uniform(2.5, 4.5)))))
        m_pos = stage + away * 110
        monster = giant(m_pos, path=LinePath(m_pos, toward_cam, 4.0))
        members.append(monster)
        return Group(members, 0, look=monster), 0.9

    # at the narrator's home the threat always comes from the street in front of the door (+z)
    threat = np.array([0.0, 0.0, 1.0]) if setting_name == "home" else away

    if action == "hide":
        hero_pos = stage.copy()
        sister = Figure(lib.figure("sister"), hero_pos + np.array([1.1, 0.0, 0.4]), anim="crouch", size=0.72,
                        face_target=hero_pos + threat * 50 + np.array([1.1, 0, 0.4]), phase=1.3)
        hero_fig = Figure(hero, hero_pos, anim="crouch", face_target=hero_pos + threat * 50)
        side = np.array([threat[2], 0.0, -threat[0]])
        m_start = stage + threat * 70 - side * 40
        monster = giant(m_start, path=LinePath(m_start, side, 3.5))
        return Group([hero_fig, sister, monster], 0, look=monster), 0.45

    if action == "faceoff":
        hero_pos = stage.copy()
        m_pos = hero_pos + threat * 36
        hero_fig = Figure(hero, hero_pos, heading=face(hero_pos, m_pos), anim="idle" if mood != "storm" else "panic")
        monster = giant(m_pos, heading=face(m_pos, hero_pos), anim="lean")
        return Group([hero_fig, monster], 0, look=monster), 1.2

    if action == "push":
        speed = 2.4
        start = road_point - road_dir * speed * duration / 2
        bus = Vehicle(lib.school_bus(), LinePath(start, road_dir, speed), bounce=0.02)
        side = np.array([road_dir[2], 0.0, -road_dir[0]]) * np.sign(np.dot(np.array([road_dir[2], 0, -road_dir[0]]), toward_cam) or 1)
        monster = giant(None, anim="push", path=LinePath(start - road_dir * 25.0, road_dir, speed))
        hero_fig = Figure(hero, start + side * 4.5, anim="walk", path=LinePath(start + side * 4.5 + road_dir * 3, road_dir, speed))
        return Group([bus, monster, hero_fig, Midpoint(bus, monster, 0.4, 9.0)], 3), 1.8

    if action == "friends":
        group, _ = build_actor("celebrate", setting_name, duration, yaw, mood, lib, rng, ground)
        m_pos = stage + away * 32
        group.members.append(giant(m_pos, heading=face(m_pos, stage), anim="wave"))
        group.members.append(Marker(stage + np.array([0.0, 9.0, 0.0])))
        group.focus_index = len(group.members) - 1
        return group, 1.6

    # landscape
    return Group([Marker(stage + np.array([0.0, 2.0, 0.0]))], 0), 1.6


def _bedroom_scene(action: str, duration: float, lib: ActorLibrary, rng: np.random.Generator, hero):
    w = interior.world
    door_key = lib.raw("door", interior.door_mesh())
    closet_key = lib.raw("closet_door", interior.closet_door_mesh())
    lamp_key = lib.raw("lamp_glow", interior.lamp_glow_mesh(), material=3)
    eyes_key = lib.raw("eyes_glow", interior.eyes_mesh(), material=3)

    def ramp(a0, a1, t0=0.15, t1=0.85):
        def f(t):
            u = min(max((t / max(duration, 1e-6) - t0) / (t1 - t0), 0.0), 1.0)
            return a0 + (a1 - a0) * (u * u * (3 - 2 * u))
        return f

    door_angle = {"noises": ramp(0.0, 0.45), "doorway": ramp(0.55, 1.2, 0.0, 0.5),
                  "empty_doorway": ramp(1.2, 1.35)}.get(action, lambda t: 0.0)
    closet_angle = ramp(0.0, 0.55, 0.1, 0.6) if action == "closet" else (lambda t: 0.0)
    members: list[Actor] = [
        # hinge on the far side so the opening door swings away from the bed's line of sight
        Prop(door_key, w(interior.DOOR_HINGE_LOCAL + np.array([0.0, 0.0, interior.DOOR_WIDTH])), heading=math.pi,
             angle=lambda t: -door_angle(t)),
        Prop(closet_key, w(interior.CLOSET_LOCAL + np.array([-1.0, 0.0, -0.15])), angle=closet_angle),
        Prop(closet_key, w(interior.CLOSET_LOCAL + np.array([1.0, 0.0, -0.15])), heading=math.pi),
        Prop(lamp_key, w(interior.LAMP_LOCAL)),
    ]
    bed = interior.BED_LOCAL
    sitting = action in ("sit_up", "doorway", "empty_doorway")
    if sitting:
        hero_fig = Figure(hero, w(bed + np.array([0.0, interior.BED_SURFACE - 0.95 * 0.8 + 0.05, -0.55])),
                          anim="sit", size=0.8, look_turn=-0.95)
    else:
        hero_fig = Figure(hero, w(bed + np.array([0.0, interior.BED_SURFACE + 0.2, 1.05])), anim="lie", size=0.8)
    members.insert(0, hero_fig)
    look = None
    if action == "doorway":
        m = Monster(lib.monster(), w(interior.HALLWAY_MONSTER_LOCAL), heading=math.pi / 2, size=0.25,
                    anim="lean", phase=float(rng.uniform(0, 3)))
        members.append(m)
        look = m
    if action == "closet":
        blink = lambda t: (t % 2.7) > 0.15  # noqa: E731
        eyes = Prop(eyes_key, w(interior.CLOSET_LOCAL + np.array([-0.45, 1.75, -0.04])), heading=math.pi,
                    visible=lambda t: t > duration * 0.35 and blink(t))
        members.append(eyes)
        look = eyes
    group = Group(members, 0, look=look)
    return group, 0.12


def flash_value(flashes: list[float], t: float) -> float:
    v = 0.0
    for f in flashes:
        dt = t - f
        if 0 <= dt < 0.7:
            v += math.exp(-dt * 16) + (0.7 * math.exp(-(dt - 0.13) * 14) if dt > 0.13 else 0.0)
    return min(v, 1.0) * 0.55


class Director:
    def __init__(self, world: World, storyboard: Storyboard, durations: list[float], seed: int = 0,
                 fast_cuts: bool = True):
        self.world = world
        self.storyboard = storyboard
        self.seed = seed
        self.library = ActorLibrary(seed)
        self.library.monster_variant = getattr(storyboard, "monster", "grey") or "grey"
        rng = np.random.default_rng(seed + 99)
        self.scenes: list[PlannedScene] = []
        t = 0.0
        prev_mood = mood_lib.get(storyboard.scenes[0].mood)
        for scene, dur in zip(storyboard.scenes, durations):
            setting = SETTINGS[scene.setting]
            yaw = setting.view_heading + float(rng.uniform(-0.6, 0.6))
            actor, cam_scale = build_actor(scene.action, scene.setting, dur, yaw, scene.mood, self.library, rng,
                                           ground=world.ground)
            target = mood_lib.get(scene.mood)
            flashes = []
            if target.lightning > 0:
                ft = t + float(rng.uniform(0.5, 1.6))
                while ft < t + dur - 0.3:
                    flashes.append(ft)
                    ft += float(rng.uniform(2.2, 4.5))
            planned = PlannedScene(scene, t, t + dur, actor, yaw, cam_scale, prev_mood, target, flashes)
            planned.shots = plan_subshots(scene.shot, yaw, dur, actor, rng, fast_cuts)
            self.scenes.append(planned)
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
    def sfx(self) -> list[tuple[float, str]]:
        """Sound effects implied by the scenes: (time, kind)."""
        rng = np.random.default_rng(self.seed + 5)
        out: list[tuple[float, str]] = []
        for s in self.scenes:
            a, t0, d = s.scene.action, s.start, s.duration
            if a == "lockdown":
                for k in range(14):
                    out.append((t0 + 0.8 + k * (d - 1.6) / 14 + float(rng.uniform(-0.1, 0.1)), "slam"))
            if a == "noises":
                for k in range(int(d / 0.95)):
                    out.append((t0 + 0.3 + k * 0.95, "step"))
                out.append((t0 + d * 0.2, "creak"))
            if a in ("doorway", "empty_doorway"):
                out.append((t0 + 0.1, "creak"))
            if a == "doorway":
                out.append((t0 + 0.9, "breath"))
            if a == "closet":
                out.append((t0 + d * 0.12, "creak"))
                out.append((t0 + d * 0.4, "whisper"))
            if a in ("noises", "doorway", "closet", "sit_up"):
                for k in range(int(d / 0.8)):
                    out.append((t0 + k * 0.8, "heartbeat"))
        return sorted(out)

    @property
    def monster_steps(self) -> list[float]:
        out = []
        for s in self.scenes:
            members = s.actor.members if isinstance(s.actor, Group) else [s.actor]
            for m in members:
                if isinstance(m, Monster):
                    out += [s.start + t for t in m.step_times(s.duration)]
        return out

    @property
    def impacts(self) -> list[float]:
        """Moments for a dramatic musical hit (monster reveals / confrontations)."""
        hits, prev = [], None
        for s in self.scenes:
            if s.scene.action in ("monster", "standoff", "panic", "faceoff", "rampage", "street_panic",
                                  "doorway", "closet", "watch") and \
                    prev not in ("monster", "standoff", "faceoff", "rampage", "street_panic", "doorway", "closet"):
                hits.append(s.start)
            prev = s.scene.action
        return hits

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
            elif isinstance(m, Vehicle) and m.key == "school_bus":
                p = m.focus(local)
                dynamic.append((p[0], p[2], 8.5, 13.5))
            elif isinstance(m, Monster):
                p = m.focus(local)
                dynamic.append((p[0], p[2], 2.6 * m.size, 9.5 * m.size))
            elif isinstance(m, Building):
                dynamic.append((m.position[0], m.position[2], 7.5, 16.0))
        if not dynamic:
            return self.world.obstacles
        return np.concatenate([self.world.obstacles, np.array(dynamic, dtype=np.float64)])

    def frame(self, t: float):
        ps = self.scene_at(t)
        local = t - ps.start
        shot, yaw, u, _ = ps.shot_at(local)
        indoor = ps.scene.setting == "bedroom" or ps.scene.action in INTERIOR_ACTIONS
        k = min(local / MOOD_BLEND_SECONDS, 1.0)
        mood = mood_lib.blend(ps.mood_from, ps.mood_to, k * k * (3 - 2 * k))
        cam: Camera = shot_camera(
            shot, u, local, ps.actor.focus, ps.actor.forward, yaw, ps.scale, ground=self.world.ground,
            look=ps.actor.look.focus if isinstance(ps.actor, Group) and ps.actor.look is not None else None,
            obstacles=None if indoor else self._obstacles(ps, local),
            bounds=(interior.BOUNDS_MIN, interior.BOUNDS_MAX) if indoor else None,
        )
        shake = ps.actor.shake(local) if hasattr(ps.actor, "shake") else 0.0
        if shake > 0.01:
            amp = shake * 0.35 * max(1.0, ps.scale)
            jitter = np.array([math.sin(t * 91.0), math.sin(t * 77.0 + 1.0), math.sin(t * 63.0 + 2.0)]) * amp
            cam.eye = tuple(np.asarray(cam.eye) + jitter)
            cam.target = tuple(np.asarray(cam.target) + jitter * 0.6)
        draws = self.static_draws(t) + ps.actor.draws(local)
        flash = flash_value(ps.flashes, t)
        dist = float(np.linalg.norm(np.asarray(cam.eye) - np.asarray(cam.target)))
        extent = float(np.clip(dist * 0.9 + 14, 22, 320))
        extras: dict = {"lights_out": self._lights_out(ps, local)}
        if indoor:
            extent = 14.0
            flicker = 1.0
            if ps.scene.action in ("noises", "doorway", "closet"):
                flicker = 0.75 + 0.25 * math.sin(t * 23.0) * math.sin(t * 7.3)
            extras["point_light"] = (interior.world(interior.LAMP_LOCAL + np.array([0.0, 0.15, 0.0])),
                                     (1.0 * flicker, 0.72 * flicker, 0.4 * flicker), 4.5)
        return cam, mood, draws, flash, extent, extras

    def _lights_out(self, ps: PlannedScene, local: float) -> float:
        """Radius around the town centre in which all windows are dark (-1: none)."""
        idx = self.scenes.index(ps)
        if ps.scene.action == "lockdown":
            return max(-1.0, (local - 0.8) * 70.0)
        if any(s.scene.action == "lockdown" for s in self.scenes[:idx]):
            return 1e5
        return -1.0
