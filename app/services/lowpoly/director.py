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
from . import interior, livingroom, skyline
from .world import HOME_POS, SETTINGS, World

INTERIOR_ACTIONS = {"bed", "sit_up", "noises", "doorway", "closet", "empty_doorway",
                    "doll_desk", "doll_floor", "doll_bed", "doll_behind"}
# where the doll is in each doll scene (room-local position, facing target)
DOLL_SPOTS = {
    "doll_desk": ((-2.25, 0.79, -3.35), (2.45, 0.0, -2.35)),
    "doll_floor": ((0.4, 0.02, -0.9), (2.45, 0.0, -2.35)),
    "doll_bed": ((2.45, 0.78, -1.15), (2.45, 0.0, -3.5)),
    "doll_behind": ((2.45, 0.83, -3.45), (2.45, 0.0, 0.0)),
}

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
    "figure": ("medium", "player_cam", "closeup", "low_angle", "dolly_in"),
    "hold": (),
    "other": ("flyover", "aerial_orbit", "crane_up", "top_down"),
}
SUBSHOT_MIN = 1.8  # viral hits cut every ~1.4-2.6 s


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
    count = max(1, min(4, round(duration / 2.3)))
    while count > 1 and duration / count < SUBSHOT_MIN:
        count -= 1
    if primary in ("medium_low", "pov"):  # keep this framing for the whole scene
        return shots
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
                lib: ActorLibrary, rng: np.random.Generator, ground=_flat, scene: Scene | None = None) -> tuple[Actor, float]:
    """Returns the actor group for a scene and a camera distance scale."""
    s = SETTINGS[setting_name]
    stage = np.array([s.stage[0], s.stage_y, s.stage[1]])
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

    if action == "duo":
        friend_pos = stage + np.array([math.sin(facing_cam + 1.4), 0.0, math.cos(facing_cam + 1.4)]) * 2.4
        hero_fig = Figure(hero, stage.copy(), anim="talk", face_target=friend_pos)
        partner = scene.partner if scene is not None else ""
        friend = Figure(lib.figure(partner) if partner else _npc(lib, rng), friend_pos, anim="talk", phase=1.1,
                        face_target=stage, size=0.74 if partner in ("sister", "doll") else 1.0)
        if scene is not None and scene.say:
            friend.label, friend.label_locked, friend.label_style = scene.say, True, "speech_hero"
        extras = [Figure(_npc(lib, rng), stage + np.array([math.sin(a), 0, math.cos(a)]) * 9, anim="walk",
                         path=LinePath(stage + np.array([math.sin(a), 0, math.cos(a)]) * 9,
                                       np.array([math.cos(a), 0, -math.sin(a)]), 1.3), phase=float(i))
                  for i, a in enumerate(rng.uniform(0, 2 * math.pi, 3))]
        return Group([hero_fig, friend] + extras, 0), 0.5

    if action in KRAMPUS_ACTIONS:
        return _krampus_scene(action, stage, duration, yaw, mood, lib, rng, hero, scene)

    if action == "speak_freeze":  # the narrator speaks a word and the person in front of them turns to stone
        say = (scene.say if scene is not None else "") or "HELLO"
        hero_fig = Figure(hero, stage.copy(), anim="talk", label=say, label_style="speech_hero", label_locked=True,
                          heading=facing_cam + math.pi)
        fw = np.array([math.sin(facing_cam + math.pi), 0.0, math.cos(facing_cam + math.pi)])
        victim = Figure(_npc(lib, rng), stage + fw * 3.2, anim="talk", face_target=stage, label="HUSS",
                        label_style="speech", label_locked=True, freeze_at=duration * 0.5)
        crowd = []
        for i in range(10):
            ang = facing_cam + math.pi + rng.uniform(-1.6, 1.6)
            p = stage + np.array([math.sin(ang), 0, math.cos(ang)]) * rng.uniform(4.0, 10.0)
            crowd.append(Figure(_npc(lib, rng), p, anim=str(rng.choice(["idle", "talk"])), phase=float(rng.uniform(0, 3)),
                                face_target=stage))
        return Group([hero_fig, victim] + crowd, 0, look=victim, look_ratio=0.6), 0.45

    if action in ("freeze_one", "new_kid", "unfreeze", "statues"):
        say = (scene.say if scene is not None else "") or "HELLO"
        reply = (scene.reply if scene is not None else "") or "THANK YOU"

        def bystanders(n, center, r0, r1, look_at=None, anims=("idle", "talk", "idle")):
            out = []
            for i in range(n):
                ang = facing_cam + math.pi + rng.uniform(-2.4, 2.4)
                p = center + np.array([math.sin(ang), 0.0, math.cos(ang)]) * rng.uniform(r0, r1)
                out.append(Figure(_npc(lib, rng), p, anim=str(rng.choice(anims)), phase=float(rng.uniform(0, 4)),
                                  face_target=center if look_at is None else look_at))
            return out

        if action == "freeze_one":
            breaker = Figure(_npc(lib, rng), stage.copy(), heading=facing_cam, anim="talk", label=say,
                             label_style="danger", label_locked=True, freeze_at=duration * 0.45)
            hero_fig = Figure(hero, stage + np.array([math.sin(facing_cam + 1.2), 0, math.cos(facing_cam + 1.2)]) * 4.0,
                              anim="idle", face_target=stage)
            return Group([breaker, hero_fig] + bystanders(13, stage, 3.5, 9.0), 0), 0.45
        if action == "new_kid":
            d = np.array([math.sin(facing_cam + math.pi / 2), 0.0, math.cos(facing_cam + math.pi / 2)])
            start = stage - d * 7.0
            finn = Figure(lib.figure("finn"), start, anim="walk", label=say, label_style="speech_hero",
                          label_locked=True, path=LinePath(start, d, 1.6, stop_at=6.0))
            hero_fig = Figure(hero, stage + d * 3.5 + np.array([math.sin(facing_cam), 0, math.cos(facing_cam)]) * 1.5,
                              anim="idle", face_target=stage)
            crowd = bystanders(12, stage, 4.0, 9.5, look_at=stage)
            return Group([finn, hero_fig] + crowd, 0), 0.45
        if action == "unfreeze":
            statue = Figure(_npc(lib, rng), stage.copy(), heading=facing_cam + 0.6, anim="point", label_locked=True,
                            freeze_at=-1.0, unfreeze_at=duration * 0.55, label_unfrozen=reply, label_style="speech")
            side = np.array([math.sin(facing_cam - 1.1), 0, math.cos(facing_cam - 1.1)])
            finn = Figure(lib.figure("finn"), stage + side * 1.8, anim="talk", label=say, label_style="speech_hero",
                          label_locked=True, face_target=stage)
            hero_fig = Figure(hero, stage + side * 4.2 + np.array([math.sin(facing_cam), 0, math.cos(facing_cam)]) * 2,
                              anim="idle", face_target=stage)
            return Group([statue, finn, hero_fig] + bystanders(6, stage, 7.0, 11.0), 0), 0.4
        # statues: a square of switched-off people
        members: list[Actor] = [Figure(hero, stage.copy(), anim="walk", path=LinePath(
            stage - np.array([math.sin(facing_cam + 1.57), 0, math.cos(facing_cam + 1.57)]) * 4,
            np.array([math.sin(facing_cam + 1.57), 0, math.cos(facing_cam + 1.57)]), 1.0))]
        for i in range(9):
            ang = rng.uniform(0, 2 * math.pi)
            p = stage + np.array([math.sin(ang), 0, math.cos(ang)]) * rng.uniform(3.0, 10.0)
            members.append(Figure(_npc(lib, rng), p, heading=float(rng.uniform(0, 6.3)),
                                  anim=str(rng.choice(["talk", "point", "wave", "walk"])), freeze_at=-1.0,
                                  label_locked=True, phase=float(rng.uniform(0, 3))))
        members += bystanders(5, stage, 10.0, 14.0, anims=("walk", "idle"))
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
        variant = (scene.variant if scene is not None and scene.variant else None)
        return Monster(lib.monster(variant), np.asarray(pos, dtype=np.float64), heading=heading, size=size,
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

    if setting_name == "livingroom" or action in LR_ACTIONS:
        return _livingroom_scene(action, duration, lib, rng, hero, scene)

    if setting_name == "bedroom" or action in INTERIOR_ACTIONS:
        return _bedroom_scene(action, duration, lib, rng, hero)

    if action in SKYLINE_ACTIONS:
        return _skyline_scene(action, setting_name, stage, duration, yaw, mood, lib, rng, hero)

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


SKYLINE_ACTIONS = {"slum_life", "look_up", "lift_ride", "gate_guards", "climb", "climb_high", "sky_view",
                   "rich_party", "guards_chase", "mirrors", "edge_look"}


def _skyline_scene(action, setting_name, stage, duration, yaw, mood, lib, rng, hero):
    sw = skyline.world
    toward_cam = np.array([math.sin(yaw), 0.0, math.cos(yaw)])

    def deck_point(p, height=skyline.TOP + 4.0):
        """A point on the sky deck to look up at: its edge, or (under the deck) its underside further away."""
        local = p - skyline.CITY
        x = float(np.clip(local[0], -skyline.DECK, skyline.DECK))
        z = float(np.clip(local[2], -skyline.DECK, skyline.DECK))
        target = sw((x, height, z))
        if np.hypot(target[0] - p[0], target[2] - p[2]) < 8.0:  # we are under the deck
            target = p - toward_cam * 30.0
            target[1] = skyline.TOP - 7.0
        return target

    def crowd(kind, n, center, radius, anims, speed=(0.0, 0.0)):
        out = []
        for i in range(n):
            ang = rng.uniform(0, 2 * math.pi)
            p = center + np.array([math.sin(ang), 0.0, math.cos(ang)]) * rng.uniform(2.5, radius)
            anim = str(rng.choice(anims))
            path = None
            if speed[1] > 0:
                d_ang = rng.uniform(0, 2 * math.pi)
                path = LinePath(p, np.array([math.sin(d_ang), 0.0, math.cos(d_ang)]), float(rng.uniform(*speed)))
            out.append(Figure(lib.figure(f"{kind}{int(rng.integers(8))}"), p, anim=anim, phase=float(rng.uniform(0, 4)),
                              path=path, face_target=None if path else center))
        return out

    if action == "slum_life":
        members = [Figure(hero, stage.copy(), anim="look_up", heading=yaw)]
        members += crowd("poor", 12, stage, 12.0, ["walk"], speed=(0.8, 1.4))
        members += crowd("poor", 5, stage, 7.0, ["talk", "idle", "sad"])
        return Group(members, 0, look=Marker(deck_point(stage)), look_ratio=0.12), 0.5

    if action == "look_up":
        target = deck_point(stage, skyline.TOP + 12.0)
        d = target - stage
        hero_fig = Figure(hero, stage.copy(), anim="look_up", heading=math.atan2(d[0], d[2]))
        return Group([hero_fig, Marker(target)], 0, look=Marker(target), look_ratio=0.22), 0.6

    if action in ("lift_ride", "gate_guards"):
        base = sw(skyline.LIFT_POS)
        guards = [Figure(lib.figure(f"guard{i}"), base + np.array([-9.0, 0.0, s_ * 4.0]), anim="idle",
                         face_target=base + np.array([-30.0, 0.0, 0.0])) for i, s_ in enumerate((-1, 1))]
        if action == "lift_ride":
            cabin_key = lib.raw("lift_cabin", skyline.lift_cabin_mesh())
            speed = (skyline.TOP - 0.2) / max(duration * 0.85, 1.0)
            cabin = Vehicle(cabin_key, LinePath(base.copy(), np.array([0.0, 1.0, 0.0]), speed), bounce=0.0)
            rich = Figure(lib.figure("rich1"), base + np.array([0.0, 0.4, 0.0]), anim="idle", heading=-math.pi / 2,
                          path=LinePath(base + np.array([0.0, 0.4, 0.0]), np.array([0.0, 1.0, 0.0]), speed))
            hero_fig = Figure(hero, stage.copy(), anim="look_up", face_target=base)
            return Group([cabin, rich, hero_fig] + guards, 2, look=cabin), 0.8
        hero_start = stage + np.array([-8.0, 0.0, 0.0])
        hero_fig = Figure(hero, hero_start, anim="walk",
                          path=LinePath(hero_start, np.array([1.0, 0.0, 0.0]), 1.4, stop_at=5.0))
        guards[0].anim = "point"
        guards[0].face_target = None
        guards[0].heading = -math.pi / 2
        return Group([hero_fig] + guards, 0, look=guards[0]), 0.5

    if action in ("climb", "climb_high"):
        x = skyline.LADDER_X + 0.45
        y0 = 0.2 if action == "climb" else 34.0
        speed = 0.9 if action == "climb" else 1.0
        start = sw((x, y0, 0.0))
        hero_fig = Figure(hero, start, anim="climb", heading=-math.pi / 2,
                          path=LinePath(start, np.array([0.0, 1.0, 0.0]), speed))
        return Group([hero_fig], 0), 0.45 if action == "climb" else 0.6

    if action == "sky_view":
        palace = sw((0.0, skyline.TOP + 28.0, 0.0))
        hero_fig = Figure(hero, stage.copy(), anim="look_up", face_target=palace)
        rich = crowd("rich", 6, stage + np.array([-18.0, 0.0, 6.0]), 8.0, ["walk"], speed=(0.6, 1.0))
        return Group([hero_fig] + rich, 0, look=Marker(palace)), 0.8

    if action == "rich_party":
        members = [Figure(hero, stage + toward_cam * 7.0, anim="crouch", face_target=stage)]
        members += crowd("rich", 12, stage, 7.0, ["cheer", "talk", "idle", "talk"])
        return Group(members, 0, look=Marker(stage + np.array([0.0, 1.6, 0.0]))), 0.6

    if action == "guards_chase":
        d = np.array([math.sin(yaw + math.pi / 2), 0.0, math.cos(yaw + math.pi / 2)])
        start = stage - d * 4.5 * duration / 2
        members: list[Actor] = [Figure(hero, start, anim="run", path=LinePath(start, d, 4.8))]
        for i in range(3):
            p = start - d * (7 + 3 * i) + np.array([d[2], 0.0, -d[0]]) * rng.uniform(-2, 2)
            members.append(Figure(lib.figure(f"guard{i}"), p, anim="run", phase=float(rng.uniform(0, 2)),
                                  path=LinePath(p, d, 4.6)))
        return Group(members, 0), 0.6

    if action == "mirrors":
        field_c = sw((-60.0, skyline.TOP + 1.5, -60.0))
        hero_fig = Figure(hero, stage.copy(), anim="idle", face_target=field_c)
        return Group([hero_fig, Marker(field_c)], 0, look=Marker(field_c)), 0.8

    # edge_look: at the railing, looking down at the lower city
    out_dir = np.array([1.0, 0.0, 0.0])
    edge = sw((skyline.DECK - 1.2, skyline.TOP, 6.0))
    down = edge + out_dir * 90.0
    down[1] = 0.0
    hero_fig = Figure(hero, edge, anim="sad" if mood in ("night", "storm") else "idle", face_target=edge + out_dir * 10)
    return Group([hero_fig, Marker(down)], 0, look=Marker(down)), 0.8


KRAMPUS_ACTIONS = {"bells", "krampus_lane", "krampus_pack", "krampus_window", "krampus_chase", "krampus_door",
                   "krampus_leave"}


def _krampus_scene(action, stage, duration, yaw, mood, lib, rng, hero, scene):
    kp = lib.figure("krampus")
    toward = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    away = -toward
    side = np.array([toward[2], 0.0, -toward[0]])

    def krampus(pos, **kw):
        kw.setdefault("phase", float(rng.uniform(0, 3)))
        return Figure(kp, np.asarray(pos, dtype=np.float64), size=1.25, **kw)

    if action == "bells":  # bells in the distance: everyone hurries home
        members: list[Actor] = [Figure(hero, stage.copy(), anim="run", path=LinePath(stage.copy(), toward, 3.8))]
        for i in range(7):
            p = stage + side * rng.uniform(-6, 6) + away * rng.uniform(0, 8)
            d = toward + side * rng.uniform(-0.6, 0.6)
            members.append(Figure(_npc(lib, rng), p, anim="run", phase=float(rng.uniform(0, 3)),
                                   path=LinePath(p, d / np.linalg.norm(d), float(rng.uniform(3.0, 4.2)))))
        for i in range(2):
            p = stage + away * (55 + 8 * i) + side * (i * 4 - 2)
            members.append(krampus(p, anim="stomp", path=LinePath(p, toward, 1.4)))
        return Group(members, 0), 0.6
    if action == "krampus_lane":  # Krampus stomps down the lane, the narrator hides behind a woodpile
        start = stage + away * 13.0
        k = krampus(start, anim="stomp", path=LinePath(start, toward, 1.0))
        hide = Figure(hero, stage + side * 3.0 + toward * 2.0, anim="crouch", face_target=start)
        return Group([hide, k], 0, look=k, look_ratio=0.75), 0.45
    if action == "krampus_pack":  # a whole pack of Krampusse circling the square
        members = [Figure(hero, stage + toward * 9.0, anim="crouch", face_target=stage)]
        for i in range(6):
            ang = i / 6 * 2 * math.pi
            p = stage + np.array([math.sin(ang), 0, math.cos(ang)]) * 7.0
            d = np.array([math.cos(ang), 0, -math.sin(ang)])
            members.append(krampus(p, anim="stomp", path=LinePath(p, d, 1.1)))
        members.append(Marker(stage + np.array([0.0, 2.0, 0.0])))
        return Group(members, len(members) - 1), 0.7
    if action == "krampus_window":  # outside the narrator's chalet, staring at the window
        from . import village as vil
        home = vil.world(vil.HOME_LOCAL)
        window = home + np.array([0.0, 4.8, 4.8])
        k = krampus(home + np.array([1.5, 0.0, 9.5]), anim="lurk", face_target=window)
        return Group([k, Marker(window)], 0, look=Marker(window), look_ratio=0.4), 0.5
    if action == "krampus_chase":
        d = side
        start = stage - d * 6.0
        runner = Figure(hero, start, anim="run", path=LinePath(start, d, 4.6))
        k = krampus(start - d * 7.0, anim="stomp", path=LinePath(start - d * 7.0, d, 4.4))
        return Group([runner, k], 0), 0.55
    if action == "krampus_door":  # face to face at the front door
        hero_fig = Figure(hero, stage.copy(), anim="idle", face_target=stage + toward * 4.0)
        k = krampus(stage + toward * 4.2, anim="point", face_target=stage)
        # shot over Krampus' shoulder towards the narrator in the doorway (the house is behind the narrator)
        return Group([k, hero_fig], 0, look=hero_fig, look_ratio=0.7), 0.5
    # krampus_leave: walking away into the forest at dawn
    k = krampus(stage.copy(), anim="stomp", path=LinePath(stage.copy(), away, 1.3))
    watcher = Figure(hero, stage + toward * 10.0, anim="idle", face_target=stage)
    return Group([watcher, k], 0, look=k, look_ratio=0.5), 0.6


LR_ACTIONS = {"lr_goodbye", "lr_couch", "lr_upstairs", "lr_hold", "lr_phone", "lr_door", "lr_two", "lr_knock",
              "lr_twins"}


def _livingroom_scene(action, duration, lib, rng, hero, scene):
    w = livingroom.world
    say = (scene.say if scene is not None else "") or ""
    reply = (scene.reply if scene is not None else "") or ""
    door_key = lib.raw("lr_door", livingroom.door_mesh())
    tv_key = lib.raw("lr_tv", livingroom.tv_mesh(), material=3)
    lamp_key = lib.raw("lr_lamp", livingroom.lamp_mesh(), material=3)
    partner = (scene.partner if scene is not None and scene.partner else "sister")
    sister_parts = lib.figure(partner)
    psize = 0.72 if partner in ("sister", "doll") else 1.0
    landing = w(livingroom.LANDING_LOCAL)

    def ramp(a0, a1, t0=0.1, t1=0.6):
        def f(t):
            u = min(max((t / max(duration, 1e-6) - t0) / (t1 - t0), 0.0), 1.0)
            return a0 + (a1 - a0) * u * u * (3 - 2 * u)
        return f

    door_angle = ramp(0.0, 1.35) if action in ("lr_door", "lr_two") else (lambda t: 0.0)
    if action == "lr_two":
        door_angle = lambda t: 1.35  # noqa: E731
    members: list[Actor] = [
        Prop(door_key, w(livingroom.DOOR_HINGE_LOCAL), angle=door_angle),
        Prop(tv_key, w(livingroom.TV_LOCAL), visible=lambda t: (t * 11.0) % 1.0 > 0.12),
        Prop(lamp_key, w(livingroom.LAMP_LOCAL)),
    ]
    couch = livingroom.COUCH_LOCAL
    middle = w((-0.6, 0.0, 0.6))
    look: Actor | None = None
    focus = 0

    def fake_mom(label=""):
        f = Figure(lib.figure("mom_fake"), landing.copy(), heading=0.35, anim="doll", phase=1.0, tint=(2.3, 2.3, 2.6))
        if label:
            f.label, f.label_style, f.label_locked = label, "danger", True
        return f

    if action in ("lr_couch", "lr_phone"):
        hero_fig = Figure(hero, w(couch + np.array([0.0, livingroom.SEAT - 0.95 + 0.05, 0.5])), anim="sit",
                          heading=-math.pi / 2, look_turn=0.0 if action == "lr_couch" else 0.5)
        sis = Figure(sister_parts, w(couch + np.array([0.0, livingroom.SEAT - 0.95 * psize + 0.05, -0.6])),
                     anim="sit", heading=-math.pi / 2, size=psize, look_turn=-0.35)
        members = [hero_fig, sis] + members
        if action == "lr_couch" and say:
            members.append(fake_mom(say))
    elif action == "lr_goodbye":
        mom = Figure(lib.figure("mom"), w((-2.7, 0.0, 2.9)), anim="talk", face_target=middle,
                     label=say, label_style="speech_hero", label_locked=True)
        hero_fig = Figure(hero, middle + np.array([0.7, 0.0, -0.2]), anim="idle", face_target=w((-2.7, 0, 2.9)))
        sis = Figure(sister_parts, middle + np.array([-1.0, 0.0, -0.7]), anim="idle", size=psize,
                     face_target=w((-2.7, 0, 2.9)))
        members = [hero_fig, sis, mom] + members
        look, focus = mom, 0
    elif action in ("lr_upstairs", "lr_hold"):
        stair = w(livingroom.STAIR_FOOT_LOCAL)
        hero_fig = Figure(hero, middle.copy(), anim="look_up", face_target=landing)
        if action == "lr_hold":  # little sister starts walking to the stairs, the narrator grabs her
            sis_start = middle + np.array([0.6, 0.0, -0.5])
            d = stair - sis_start
            d[1] = 0.0
            d /= np.linalg.norm(d)
            sis = Figure(sister_parts, sis_start, anim="walk", size=psize,
                         path=LinePath(sis_start, d, 0.7, stop_at=1.2))
            hero_fig.anim = "point"
            hero_fig.face_target = stair
        else:
            sis = Figure(sister_parts, middle + np.array([-0.6, 0.0, 0.5]), anim="look_up", size=psize,
                         face_target=landing)
        members = [hero_fig, sis, fake_mom(say)] + members
        look, focus = members[2], 0
    elif action == "lr_twins":  # two identical moms side by side: spot the fake
        left, right = w((-1.2, 0.0, -1.2)), w((0.6, 0.0, -1.2))
        fake_left = False  # same side in every scene, so the puzzle is fair
        real = Figure(lib.figure("mom"), right if fake_left else left, anim="idle", heading=0.0, phase=0.7)
        fake = Figure(lib.figure("mom_fake"), left if fake_left else right, anim="idle", heading=0.0, phase=1.9,
                      tint=(1.12, 1.1, 1.16))
        members = [real, fake, Marker(w((-0.3, 1.9, -1.2)))]  # just the two of them, nobody in the way
        look, focus = None, 2
    elif action == "lr_knock":  # someone knocks at the closed front door; the kids stare at it
        door = w((-3.0, 0.0, 3.6))
        hero_fig = Figure(hero, middle.copy(), anim="idle", face_target=door)
        sis = Figure(sister_parts, middle + np.array([-0.7, 0.0, -0.4]), anim="idle", size=psize, face_target=door)
        members = [hero_fig, sis, Marker(door + np.array([0.0, 1.3, 0.0]))] + members
        look, focus = members[2], 0
    elif action in ("lr_door", "lr_two"):
        start = w((-3.0, 0.0, 5.2))
        mom = Figure(lib.figure("mom"), start, anim="walk", label=say if action == "lr_door" else say,
                     label_style="speech_hero", label_locked=True,
                     path=LinePath(start, np.array([0.0, 0.0, -1.0]), 1.2, stop_at=2.4 if action == "lr_door" else 0.01))
        if action == "lr_two":
            mom.path = None
            mom.position = w((-3.0, 0.0, 2.8))
            mom.anim = "idle"
            mom.face_target = middle
        hero_fig = Figure(hero, middle.copy(), anim="idle", face_target=w((-3.0, 0.0, 2.8)))
        sis = Figure(sister_parts, middle + np.array([-0.5, 0.0, -0.5]), anim="idle", size=psize,
                     face_target=w((-3.0, 0.0, 2.8)))
        members = [hero_fig, sis, mom] + members
        if action == "lr_two":
            members.append(fake_mom(reply))
            look = members[-1]
        else:
            look = mom
    return Group(members, focus, look=look, look_ratio=0.55), 0.3


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
    sitting = action in ("sit_up", "doorway", "empty_doorway", "doll_behind")
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
    if action in DOLL_SPOTS:
        spot, target = DOLL_SPOTS[action]
        doll = Figure(lib.figure("doll"), w(spot), anim="doll", size=0.21, face_target=w(target))
        members.append(doll)
        look = doll
        if action == "doll_behind":
            # frame the narrator's face with the doll peeking over the shoulder (no over-the-shoulder staging)
            look = None
            members.append(Marker((hero_fig.focus(0.0) + doll.focus(0.0)) / 2 + np.array([0.0, 0.1, 0.0])))
            return Group(members, len(members) - 1), 0.35
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
        self.library.hero_variant = getattr(storyboard, "hero", "") or ""
        rng = np.random.default_rng(seed + 99)
        self.scenes: list[PlannedScene] = []
        t = 0.0
        prev_mood = mood_lib.get(storyboard.scenes[0].mood)
        for scene, dur in zip(storyboard.scenes, durations):
            setting = SETTINGS[scene.setting]
            yaw = setting.view_heading + float(rng.uniform(-0.6, 0.6))
            actor, cam_scale = build_actor(scene.action, scene.setting, dur, yaw, scene.mood, self.library, rng,
                                           ground=world.ground, scene=scene)
            target = mood_lib.get(scene.mood)
            flashes = []
            if target.lightning > 0:
                ft = t + float(rng.uniform(0.5, 1.6))
                while ft < t + dur - 0.3:
                    flashes.append(ft)
                    ft += float(rng.uniform(2.2, 4.5))
            self._assign_labels(actor, scene, storyboard)
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
            if a in ("doll_floor", "doll_bed", "doll_behind"):
                out.append((t0 + 0.05, "creak"))  # a tiny wooden creak each time the doll has moved
            if a == "closet":
                out.append((t0 + d * 0.12, "creak"))
                out.append((t0 + d * 0.4, "whisper"))
            members = s.actor.members if isinstance(s.actor, Group) else [s.actor]
            if a in KRAMPUS_ACTIONS:
                for k in range(int(d / 1.1)):
                    out.append((t0 + 0.2 + k * 1.1 + float(rng.uniform(-0.1, 0.1)), "bells"))
                out.append((t0 + d * 0.3, "chains"))
            if a == "lr_upstairs":
                for k in range(int(d / 1.0)):
                    out.append((t0 + 0.4 + k * 1.0, "step"))
            if a in ("lr_upstairs", "lr_hold", "lr_two"):
                for k in range(int(d / 0.75)):
                    out.append((t0 + k * 0.75, "heartbeat"))
            if a == "lr_phone" and not s.scene.chat:
                out.append((t0 + 0.2, "buzz"))
                out.append((t0 + 0.9, "buzz"))
            if s.scene.chat:  # one buzz per incoming message
                nmsg = len(s.scene.chat)
                span = max(d - 0.2 - 0.6, 0.5)
                for i, (side, _) in enumerate(s.scene.chat):
                    if side.lower() not in ("me", "i", "ich"):
                        out.append((t0 + 0.35 + i * span / nmsg, "buzz"))
            if a == "lr_door":
                out.append((t0 + d * 0.12, "creak"))
            if a == "lr_knock":
                for k in range(3):
                    out.append((t0 + 0.4 + k * 0.55, "knock"))
            if a in ("noises",) and "KRAMPUS" in getattr(self.storyboard, "wheel_pick", "").upper():
                out.append((t0 + d * 0.5, "bells"))
            for m in members:
                if isinstance(m, Figure) and m.freeze_at is not None and m.freeze_at >= 0:
                    out.append((t0 + m.freeze_at, "zap"))
                if isinstance(m, Figure) and m.unfreeze_at is not None:
                    out.append((t0 + m.unfreeze_at, "creak"))
            if a in ("noises", "doorway", "closet", "sit_up", "doll_floor", "doll_bed", "doll_behind"):
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
                                  "doorway", "closet", "watch", "doll_behind", "krampus_lane", "krampus_door",
                                  "lr_upstairs", "lr_two") and \
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
                dynamic.append((p[0], p[2], 8.0, 13.0, -1.0))
            elif isinstance(m, Vehicle) and m.key == "school_bus":
                p = m.focus(local)
                dynamic.append((p[0], p[2], 8.5, 13.5, -1.0))
            elif isinstance(m, Monster):
                p = m.focus(local)
                dynamic.append((p[0], p[2], 2.6 * m.size, 9.5 * m.size, -1.0))
            elif isinstance(m, Building):
                dynamic.append((m.position[0], m.position[2], 7.5, 16.0, -1.0))
        if not dynamic:
            return self.world.obstacles
        return np.concatenate([self.world.obstacles, np.array(dynamic, dtype=np.float64)])

    def frame(self, t: float):
        ps = self.scene_at(t)
        local = t - ps.start
        shot, yaw, u, _ = ps.shot_at(local)
        living = ps.scene.setting == "livingroom" or ps.scene.action in LR_ACTIONS
        indoor = living or ps.scene.setting == "bedroom" or ps.scene.action in INTERIOR_ACTIONS
        k = min(local / MOOD_BLEND_SECONDS, 1.0)
        mood = mood_lib.blend(ps.mood_from, ps.mood_to, k * k * (3 - 2 * k))
        cam: Camera = shot_camera(
            shot, u, local, ps.actor.focus, ps.actor.forward, yaw, ps.scale, ground=self.world.ground,
            look=ps.actor.look.focus if isinstance(ps.actor, Group) and ps.actor.look is not None else None,
            look_ratio=ps.actor.look_ratio if isinstance(ps.actor, Group) else None,
            obstacles=None if indoor else self._obstacles(ps, local),
            bounds=((livingroom.BOUNDS_MIN, livingroom.BOUNDS_MAX) if living else
                    (interior.BOUNDS_MIN, interior.BOUNDS_MAX) if indoor else self._cam_bounds(ps)),
        )
        if shot == "pov":
            cam = self._pov_camera(ps, local, u) or cam
        if living and ps.scene.action in ("lr_upstairs", "lr_hold", "lr_two") and shot != "medium":
            cam = self._stairs_camera(shot, u, local) or cam
        if living and ps.scene.action == "lr_twins":
            cam = self._twins_camera(shot, u)
        if living and ps.scene.action in ("lr_goodbye", "lr_knock"):
            cam = self._door_camera(shot, u, ps.scene.action == "lr_goodbye")
        shake = ps.actor.shake(local) if hasattr(ps.actor, "shake") else 0.0
        if shake > 0.01:
            amp = shake * 0.35 * max(1.0, ps.scale)
            jitter = np.array([math.sin(t * 91.0), math.sin(t * 77.0 + 1.0), math.sin(t * 63.0 + 2.0)]) * amp
            cam.eye = tuple(np.asarray(cam.eye) + jitter)
            cam.target = tuple(np.asarray(cam.target) + jitter * 0.6)
        draws = self.static_draws(t) + ps.actor.draws(local)
        if shot == "pov":  # we are looking through someone's eyes: hide their body
            hide = f"fig_{ps.scene.pov_of}_" if ps.scene.pov_of else "fig_hero_"
            draws = [d for d in draws if not d.key.startswith(hide)]
        flash = flash_value(ps.flashes, t)
        members = ps.actor.members if isinstance(ps.actor, Group) else [ps.actor]
        for m in members:
            if isinstance(m, Figure) and m.freeze_at is not None and 0 <= local - m.freeze_at < 0.25:
                flash = max(flash, 0.35 * (1 - (local - m.freeze_at) / 0.25))
        dist = float(np.linalg.norm(np.asarray(cam.eye) - np.asarray(cam.target)))
        extent = float(np.clip(dist * 0.9 + 14, 22, 320))
        extras: dict = {"lights_out": self._lights_out(ps, local), "labels": self._figure_labels(ps, local)}
        if ps.scene.reveal:  # where the answer is (for the circle overlay)
            parts = self.library.figure(ps.scene.reveal)
            mem = ps.actor.members if isinstance(ps.actor, Group) else [ps.actor]
            tgt = next((m for m in mem if isinstance(m, Figure) and m.parts == parts), None)
            if tgt is not None:
                extras["reveal"] = tgt.focus(local) + np.array([0.0, 0.55 * tgt.size, 0.0])
        if shot == "pov" and not ps.scene.pov_of:  # what "you" say can't float above your own head
            hero_parts = self.library.figure("hero")
            mem = ps.actor.members if isinstance(ps.actor, Group) else [ps.actor]
            me = next((m for m in mem if isinstance(m, Figure) and m.parts == hero_parts), None)
            if me is not None:
                extras["labels"] = [l for l in extras["labels"] if l[1] != me.current_label(local) or l[2] != me.label_style]
                if me.current_label(local):
                    extras["pov_say"] = (me.current_label(local), me.label_style)
        if indoor:
            extent = 14.0
            flicker = 1.0
            if ps.scene.action in ("noises", "doorway", "closet"):
                flicker = 0.75 + 0.25 * math.sin(t * 23.0) * math.sin(t * 7.3)
            lamp = interior.world(interior.LAMP_LOCAL + np.array([0.0, 0.15, 0.0]))
            radius = 4.5
            if living:
                lamp, radius = livingroom.world(livingroom.LAMP_LOCAL + np.array([0.0, 0.1, 0.0])), 10.0
                if ps.scene.action in ("lr_upstairs", "lr_two", "lr_hold"):
                    flicker = 0.7 + 0.3 * math.sin(t * 19.0) * math.sin(t * 5.3)
                    if (t * 3.1) % 4.0 < 0.12:
                        flicker = 0.15  # the lights cut out for a moment
            extras["point_light"] = (lamp, (1.0 * flicker, 0.72 * flicker, 0.4 * flicker), radius)
            if living:  # the TV throws cold flickering light across the room
                tv = 0.75 + 0.25 * math.sin(t * 13.0) * math.sin(t * 3.7 + 1.0)
                extras["point_light2"] = (livingroom.world(livingroom.TV_LOCAL + np.array([0.6, 0.0, 0.0])),
                                          (0.45 * tv, 0.62 * tv, 0.95 * tv), 7.5)
        return cam, mood, draws, flash, extent, extras

    def _assign_labels(self, actor: Actor, scene: Scene, board: Storyboard) -> None:
        hero_label = scene.hero_label or getattr(board, "hero_label", "")
        npc_labels = getattr(board, "npc_labels", {}) or {}
        hero_parts = self.library.figure("hero")
        members = actor.members if isinstance(actor, Group) else [actor]
        speech = getattr(board, "label_style", "") == "speech"
        for i, m in enumerate(members):
            if not isinstance(m, Figure) or m.label_locked:
                continue
            if m.parts == hero_parts:
                m.label = hero_label
                m.label_style = "danger" if hero_label.startswith("!") else "hero"
                m.label = m.label.lstrip("!")
                if speech and m.label_style == "hero":
                    m.label_style = "speech_hero"
            elif scene.npc_label and m.size >= 0.7:
                m.label, m.label_style = scene.npc_label, "speech" if speech else "npc"
            elif npc_labels and m.size >= 0.7:
                key = next(iter(m.parts.values()))  # e.g. "fig_rich3_torso"
                kind = next((k for k in ("rich", "poor", "guard") if f"fig_{k}" in key), "default")
                pool = npc_labels.get(kind) or npc_labels.get("default") or []
                if pool:
                    m.label = pool[(i * 7 + len(scene.narration)) % len(pool)]
                    if speech:
                        m.label_style = "speech"

    @staticmethod
    def _figure_labels(ps: PlannedScene, local: float) -> list:
        out = []
        members = ps.actor.members if isinstance(ps.actor, Group) else [ps.actor]
        for m in members:
            if isinstance(m, Figure):
                text = m.current_label(local)
                if text:
                    out.append((m.focus(local) + np.array([0.0, 1.55 * m.size, 0.0]), text, m.label_style))
        return out

    # ------------------------------------------------------------------ #
    # automatic shot QA: no black frames, no camera stuck inside a figure or wall
    # ------------------------------------------------------------------ #
    @staticmethod
    def frame_penalty(img: np.ndarray) -> float:
        a = img.astype(np.float32)
        lum = float(a.mean()) / 255.0
        g = a.mean(axis=2)
        grad = np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]
        # a big smooth bright area = the camera is inside / right against a head, shoulder or wall
        blob = float(((grad < 1.5) & (a.max(axis=2)[:-1, :-1] > 110)).mean())
        return max(0.0, 0.035 - lum) * 25.0 + max(0.0, blob - 0.56) * 4.0

    def qa_fix(self, render, log=None) -> int:
        """render(t) -> RGB frame. Replaces failing sub-shots; returns the number of shots fixed."""
        candidates = ("medium", "player_cam", "low_angle", "dolly_in", "closeup", "crane_up", "aerial_orbit",
                      "side_track", "pull_back", "top_down")
        fixed = 0
        for ps in self.scenes:
            for i, (st, shot, yaw) in enumerate(list(ps.shots)):
                if shot in ("pov", "medium_low"):
                    continue
                end = ps.shots[i + 1][0] if i + 1 < len(ps.shots) else ps.duration

                def score(sh, yw):
                    ps.shots[i] = (st, sh, yw)
                    self._cursor = 0
                    return max(self.frame_penalty(render(ps.start + st + (end - st) * f)) for f in (0.15, 0.55, 0.92))

                best = (score(shot, yaw), shot, yaw)
                if best[0] <= 0:
                    continue
                for k, alt in enumerate(c for c in candidates if c != shot):
                    for dy in (0.0, 0.9, -0.9):
                        p = score(alt, yaw + dy)
                        if p < best[0]:
                            best = (p, alt, yaw + dy)
                        if p <= 0:
                            break
                    if best[0] <= 0:
                        break
                ps.shots[i] = (st, best[1], best[2])
                fixed += 1
                if log:
                    log(f"QA: scene '{ps.scene.action}' shot {i}: {shot} -> {best[1]} (penalty {best[0]:.2f})")
        self._cursor = 0
        return fixed

    @staticmethod
    def _twins_camera(shot: str, u: float):
        e = u * u * (3 - 2 * u)
        tgt = livingroom.world((-0.3, 1.45, -1.2))
        if shot in ("closeup", "dolly_in"):
            return Camera(tuple(livingroom.world((-0.3, 1.95, 3.5 - 0.5 * e))), tuple(livingroom.world((-0.3, 1.6, -1.2))), 54)
        return Camera(tuple(livingroom.world((-0.3 + 0.25 * math.sin(u * 3), 1.6, 3.7 - 0.3 * e))), tuple(tgt), 62)

    @staticmethod
    def _door_camera(shot: str, u: float, goodbye: bool = False):
        """Looking from the room towards the front door, the kids in the foreground."""
        e = u * u * (3 - 2 * u)
        door = livingroom.world((-2.9, 1.45, 3.4))
        if goodbye:  # behind the boys, mom at the door facing us
            mom_head = livingroom.world((-2.7, 2.05, 2.9))
            eye = livingroom.world((-0.2 - 0.3 * e, 2.05, -1.4 + 0.4 * e))
            return Camera(tuple(eye), tuple(mom_head), 34 if shot in ("closeup", "dolly_in") else 50)
        if shot in ("closeup", "dolly_in"):  # from beside the TV, tight on the door
            eye = livingroom.world((-4.2, 1.85, 0.0 + 0.5 * e))
            return Camera(tuple(eye), tuple(door + np.array([0.3, 0.0, 0.0])), 46)
        eye = livingroom.world((-4.55, 2.3, -3.5 + 0.4 * e))  # wide: the door and the boys staring at it
        return Camera(tuple(eye), tuple(livingroom.world((-1.95, 1.3, 1.9))), 66)

    @staticmethod
    def _stairs_camera(shot: str, u: float, local: float):
        """Fixed compositions for the thing on the stairs: wide from the couch, or close on its face."""
        head = livingroom.world(livingroom.LANDING_LOCAL + np.array([0.0, 2.35, 0.0]))
        e = u * u * (3 - 2 * u)
        if shot == "closeup":  # her face, slowly pushing in
            eye = head + np.array([0.45 - 0.1 * e, -0.45, 3.4 - 0.8 * e])
            return Camera(tuple(eye), tuple(head + np.array([0.0, -0.35, 0.0])), 50)
        if shot == "crane_up":  # from the foot of the stairs, looking up at her
            eye = livingroom.world((-2.2 + 0.4 * e, 0.7, -1.6))
            return Camera(tuple(eye), tuple(head + np.array([0.0, -0.7, 0.0])), 58)
        eye = livingroom.world((2.3 - 0.5 * e, 1.25, 2.9 - 0.7 * e))
        return Camera(tuple(eye), tuple(head + np.array([0.0, -0.9, 0.0])), 62)

    def _pov_camera(self, ps: PlannedScene, local: float, u: float):
        hero_parts = self.library.figure(ps.scene.pov_of or "hero")
        members = ps.actor.members if isinstance(ps.actor, Group) else [ps.actor]
        hero = next((m for m in members if isinstance(m, Figure) and m.parts == hero_parts), None)
        if hero is None:
            return None
        head = hero.focus(local) + np.array([0.0, 0.45 * hero.size, 0.0])
        fw = hero.forward(local)
        look = None
        if isinstance(ps.actor, Group) and ps.actor.look is not None:
            look = ps.actor.look.focus(local)
        elif isinstance(ps.actor, Group):
            others = [m for m in members if m is not hero and isinstance(m, Figure)]
            if others:
                look = others[0].focus(local)
        eye = head - fw * 0.6
        if look is not None and np.linalg.norm(look - eye) > 1.0:
            target = look
        else:
            target = eye + fw * 10.0 + np.array([0.0, -1.2, 0.0])
        # a little handheld head bob
        eye = eye + np.array([math.sin(local * 2.1) * 0.03, math.sin(local * 4.2) * 0.04, 0.0])
        return Camera(tuple(eye), tuple(target), 68)

    @staticmethod
    def _cam_bounds(ps: PlannedScene):
        cy = SETTINGS[ps.scene.setting].cam_y
        if not cy:
            return None
        return (np.array([-1e9, cy[0], -1e9]), np.array([1e9, cy[1], 1e9]))

    def _lights_out(self, ps: PlannedScene, local: float) -> float:
        """Radius around the town centre in which all windows are dark (-1: none)."""
        idx = self.scenes.index(ps)
        if ps.scene.action == "lockdown":
            return max(-1.0, (local - 0.8) * 70.0)
        if any(s.scene.action == "lockdown" for s in self.scenes[:idx]):
            return 1e5
        return -1.0
