"""Procedural low-poly world: terrain, patchwork fields, roads, a toy town,
forests, snowy mountains, a coastline and clouds.

The world is built once per video (seeded) so every scene of a story plays
in the same consistent place, just like a game map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import props
from .mesh import MeshBuilder, hex_color, vary
from .props import PALETTE

WORLD_HALF = 640.0
WATER_LEVEL = -0.6


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #


def _hash2(ix: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 374761393 + iz.astype(np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def value_noise(x: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    x0, z0 = np.floor(x), np.floor(z)
    fx, fz = x - x0, z - z0
    ux, uz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    a = _hash2(x0, z0, seed)
    b = _hash2(x0 + 1, z0, seed)
    c = _hash2(x0, z0 + 1, seed)
    d = _hash2(x0 + 1, z0 + 1, seed)
    return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uz


def fbm(x, z, seed: int, octaves: int = 4) -> np.ndarray:
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        total = total + value_noise(x * freq, z * freq, seed + o * 17) * amp
        norm += amp
        amp *= 0.5
        freq *= 2.03
    return total / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- #
# Locations ("settings") the storyboard can refer to
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Setting:
    name: str
    stage: tuple[float, float]  # where people stand / action happens (x, z)
    road_axis: str  # "x": road runs along x at z=road_line ; "z": along z at x=road_line
    road_line: float
    view_heading: float  # preferred camera yaw (radians) looking at stage
    description: str
    clear_radius: float = 24.0  # no buildings are placed this close to the stage
    stage_y: float = 0.0  # height of the stage (e.g. the sky deck)
    cam_y: tuple | None = None  # (min, max) camera height for this place


SETTINGS: dict[str, Setting] = {
    "town": Setting("town", (8.0, 11.0), "x", 0.0, 0.6, "town square in front of the town hall"),
    "home": Setting("home", (40.0, -21.0), "x", 0.0, 0.25, "the narrator's own little house"),
    "fields": Setting("fields", (170.0, 40.0), "x", 0.0, 0.9, "colourful patchwork farm fields"),
    "road": Setting("road", (-150.0, 9.0), "x", 0.0, 0.3, "the long country road"),
    "coast": Setting("coast", (-292.0, 22.0), "x", 0.0, -1.2, "sandy beach by the sea"),
    "mountains": Setting("mountains", (8.0, -300.0), "z", 0.0, 3.4, "foot of the snowy mountains"),
    "forest": Setting("forest", (210.0, -120.0), "x", 0.0, 2.2, "a dense green forest"),
    "construction": Setting("construction", (72.0, 34.0), "x", 0.0, 0.7, "an empty building lot with a crane"),
    "bedroom": Setting("bedroom", (2.45, 1497.65), "x", 0.0, -0.8,
                       "the narrator's dark bedroom at night (bed, door to the hallway, closet, window)", clear_radius=0.0),
    "slums": Setting("slums", (-20.0, -1352.0), "x", 0.0, 0.15,
                     "(skyline) the poor lower city in the shadow of the rich sky platform", clear_radius=0.0),
    "under_deck": Setting("under_deck", (-40.0, -1460.0), "x", 0.0, 0.6,
                          "(skyline) dark alleys right under the sky platform", clear_radius=0.0, cam_y=(0.6, 55.0)),
    "lift_gate": Setting("lift_gate", (-121.0, -1500.0), "x", 0.0, -1.15,
                         "(skyline) the guarded golden elevator, the only way up", clear_radius=0.0),
    "ladder": Setting("ladder", (78.5, -1500.0), "x", 0.0, 1.35,
                      "(skyline) the old maintenance ladder up pillar 7", clear_radius=0.0, cam_y=(0.6, 62.0)),
    "sky_edge": Setting("sky_edge", (79.0, -1496.0), "x", 0.0, 1.9,
                        "(skyline) the edge of the rich sky city where the ladder comes out", clear_radius=0.0,
                        stage_y=70.0, cam_y=(72.0, 400.0)),
    "sky_city": Setting("sky_city", (20.0, -1470.0), "x", 0.0, 0.7,
                        "(skyline) golden villas, pools and gardens of the rich", clear_radius=0.0,
                        stage_y=70.0, cam_y=(72.0, 400.0)),
    "sky_mirrors": Setting("sky_mirrors", (-60.0, -1522.0), "x", 0.0, 2.6,
                           "(skyline) a field of golden sun mirrors on the rich platform", clear_radius=0.0,
                           stage_y=70.0, cam_y=(72.0, 400.0)),
    "village_square": Setting("village_square", (1506.0, 7.0), "x", 0.0, 0.45,
                              "(alpine village) the snowy square with church, fountain and maypole", clear_radius=0.0),
    "village_lane": Setting("village_lane", (1460.0, 1.5), "x", 0.0, 1.3,
                            "(alpine village) the narrow lane between wooden chalets and lanterns", clear_radius=0.0),
    "chalet_home": Setting("chalet_home", (1532.0, -5.5), "x", 0.0, 0.25,
                           "(alpine village) in front of the narrator's chalet", clear_radius=0.0),
    "village_edge": Setting("village_edge", (1600.0, 30.0), "x", 0.0, -2.2,
                            "(alpine village) where the village ends and the dark snowy forest begins", clear_radius=0.0),
    "livingroom": Setting("livingroom", (-1499.4, 1.6), "x", 0.0, -1.2,
                          "(indoor) living room at night: couch, TV, front door, stairs up to a dark landing",
                          clear_radius=0.0),
    "city_plaza": Setting("city_plaza", (1500.0, -1476.0), "x", -1500.0, 2.95,
                          "(big city) the plaza with the giant golden zero monument and the city hall",
                          clear_radius=0.0),
    "city_avenue": Setting("city_avenue", (1410.0, -1503.0), "x", -1500.0, 1.9,
                           "(big city) the busy avenue between the skyscrapers", clear_radius=0.0),
    "city_crossing": Setting("city_crossing", (1560.0, -1509.0), "x", -1500.0, 0.8,
                             "(big city) the crossroads with traffic lights and crosswalks", clear_radius=0.0),
    "city_home": Setting("city_home", (1596.0, -1509.5), "x", -1500.0, 0.25,
                         "(big city) the sidewalk in front of the narrator's front door", clear_radius=0.0),
    "main_street": Setting("main_street", (-62.0, 7.0), "x", 0.0, 0.45,
                           "the main street of the town, houses on both sides", clear_radius=6.0),
}

HOME_POS = (40.0, 0.0, -34.0)
TOWN_HALL_POS = (0.0, 0.0, 36.0)
BUILD_SITE = (72.0, 0.0, 34.0)


# --------------------------------------------------------------------------- #
# Height field
# --------------------------------------------------------------------------- #


class Terrain:
    def __init__(self, seed: int):
        self.seed = seed

    def height(self, x, z):
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        r = np.hypot(x, z)
        hills = fbm(x / 140.0, z / 140.0, self.seed)
        h = smoothstep(270, 470, r) * (hills * 1.4 - 0.25).clip(0) * 34
        ridge = 1.0 - np.abs(fbm(x / 95.0, z / 95.0, self.seed + 7) * 2 - 1)
        north = smoothstep(330, 500, -z)
        h = h + north * (25 + 150 * ridge ** 2 * fbm(x / 60.0, z / 60.0, self.seed + 3))
        # flatten roads and stages
        flat = smoothstep(12, 40, np.abs(z)) * 1.0
        flat = np.maximum(flat, (x < -330).astype(float))
        north_road = np.where((z < 0) & (z > -335), smoothstep(12, 40, np.abs(x)), 1.0)
        flat = flat * north_road
        for s in SETTINGS.values():
            sx, sz = s.stage
            flat = flat * smoothstep(30, 70, np.hypot(x - sx, z - sz))
        h = h * flat
        west = smoothstep(300, 350, -x)
        h = h * (1 - west) + west * (-7.0 + 3.0 * fbm(x / 30, z / 30, self.seed + 11))
        return h


# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #


@dataclass
class World:
    seed: int
    terrain: Terrain
    static: np.ndarray  # vertex array (terrain + props)
    water: np.ndarray
    clouds: np.ndarray
    house_specs: list = field(default_factory=list)
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))  # x, z, radius, top, bottom

    def ground(self, x: float, z: float) -> float:
        if abs(z) > 1000 or abs(x) > 1000:  # interior sets / skyline / village districts: flat ground at 0
            return 0.0
        return max(float(self.terrain.height(x, z)), 0.0) if x > -300 else float(self.terrain.height(x, z))


def _terrain_mesh(terrain: Terrain, rng: np.random.Generator) -> MeshBuilder:
    n = 160
    xs = np.linspace(-WORLD_HALF, WORLD_HALF, n + 1)
    gx, gz = np.meshgrid(xs, xs, indexing="xy")
    # jitter interior vertices a little for an organic low-poly look
    cell = xs[1] - xs[0]
    jitter = rng.uniform(-0.28, 0.28, gx.shape + (2,)) * cell
    far = smoothstep(260, 330, np.hypot(gx, gz))
    gx = gx + jitter[..., 0] * far
    gz = gz + jitter[..., 1] * far
    gy = terrain.height(gx, gz)

    p = np.stack([gx, gy, gz], axis=-1)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]
    tri1 = np.stack([a, b, c], axis=2).reshape(-1, 3, 3)
    tri2 = np.stack([a, c, d], axis=2).reshape(-1, 3, 3)
    tris = np.concatenate([tri1, tri2], axis=0)

    centroid = tris.mean(axis=1)
    hy = centroid[:, 1]
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
    slope = 1 - np.abs(normal[:, 1])

    hilly = (np.abs(hy) > 0.4)[:, None]
    shade = rng.uniform(-0.03, 0.03, (len(tris), 1)) * hilly
    grass = np.array(PALETTE["grass"])
    grass_dark = np.array(PALETTE["grass_dark"])
    patch = fbm(centroid[:, 0] / 120, centroid[:, 2] / 120, 5)[:, None]
    colors = grass_dark + (grass - grass_dark) * np.clip((patch - 0.35) * 3, 0, 1)
    colors = colors + shade
    rock_mask = (slope > 0.45) | (hy > 55)
    colors[rock_mask] = np.array(PALETTE["rock"]) + shade[rock_mask]
    snow_line = 80 + 20 * fbm(centroid[:, 0] / 50, centroid[:, 2] / 50, 9)
    snow_mask = (hy > snow_line) & (slope < 0.62)
    colors[snow_mask] = np.array(PALETTE["snow"]) + shade[snow_mask] * 0.3
    sand_mask = (centroid[:, 0] < -285) & (hy < 1.4)
    colors[sand_mask] = np.array(PALETTE["sand"]) + shade[sand_mask]
    deep = hy < -2.0
    colors[deep] = np.array(hex_color("#d9c98a"))
    colors = colors.clip(0, 1)

    mb = MeshBuilder()
    mb.triangles_colored(tris.reshape(-1, 3), colors)
    return mb


def _roads(mb: MeshBuilder) -> None:
    y = 0.06
    mb.box_min((-302, -0.2, -4.5), (WORLD_HALF, y, 4.5), PALETTE["road"])
    mb.box_min((-4.5, -0.2, -335), (4.5, y, 4.5), PALETTE["road"])
    for x in np.arange(-298, WORLD_HALF, 12):
        mb.box_min((x, y, -0.18), (x + 5, y + 0.03, 0.18), PALETTE["road_line"])
    for z in np.arange(-330, -6, 12):
        mb.box_min((-0.18, y, z), (0.18, y + 0.03, z + 5), PALETTE["road_line"])
    # town plaza
    mb.box_min((-30, -0.2, 5), (30, 0.08, 26), PALETTE["plaza"])
    # sidewalks
    mb.box_min((-130, -0.2, 4.5), (130, 0.1, 6.5), hex_color("#c9c6d6"))
    mb.box_min((-130, -0.2, -6.5), (130, 0.1, -4.5), hex_color("#c9c6d6"))


def _fields(mb: MeshBuilder, terrain: Terrain, rng: np.random.Generator) -> list[tuple[float, float, float, float]]:
    rects = []
    size = 38.0
    for gx in np.arange(-300, 330, size):
        for gz in np.arange(-300, 300, size):
            cx, cz = gx + size / 2, gz + size / 2
            if abs(cx) < 140 and abs(cz) < 80:
                continue
            if abs(cz) < 14 + size / 2 or (abs(cx) < 14 + size / 2 and cz < 0):
                continue
            if cx < -270:
                continue
            if any(math.hypot(cx - s.stage[0], cz - s.stage[1]) < 30 for s in SETTINGS.values() if s.name not in ("fields",)):
                continue
            if math.hypot(cx - 210, cz + 120) < 100:  # forest
                continue
            heights = terrain.height(np.array([gx, gx + size, gx, gx + size]), np.array([gz, gz, gz + size, gz + size]))
            if np.max(np.abs(heights)) > 0.3:
                continue
            splits = [(gx, gz, gx + size, gz + size)]
            if rng.random() < 0.55:
                mid = gx + size * rng.uniform(0.35, 0.65)
                splits = [(gx, gz, mid, gz + size), (mid, gz, gx + size, gz + size)]
            for x0, z0, x1, z1 in splits:
                color = vary(PALETTE["field"][rng.integers(len(PALETTE["field"]))], rng, 0.02)
                inset = 1.1
                mb.box_min((x0 + inset, -0.2, z0 + inset), (x1 - inset, 0.05, z1 - inset), color)
                if rng.random() < 0.3:  # furrow stripes
                    darker = tuple(max(c - 0.06, 0) for c in color)
                    for sx in np.arange(x0 + 3, x1 - 2, 3.0):
                        mb.box_min((sx, 0.05, z0 + inset), (sx + 0.9, 0.09, z1 - inset), darker)
                rects.append((x0, z0, x1, z1))
    return rects


def _town(mb: MeshBuilder, rng: np.random.Generator, obstacles: list) -> list[dict]:
    specs = []

    def add_house(pos, **kw):
        info = props.house(mb, pos, rng, **kw)
        obstacles.append((pos[0], pos[2], math.hypot(info["width"], info["depth"]) / 2 + 0.6, info["height"] + 1.5))
        return info

    def near_stage(x, z):
        return any(math.hypot(x - st.stage[0], z - st.stage[1]) < st.clear_radius for st in SETTINGS.values())

    props.town_hall(mb, TOWN_HALL_POS, rng, rot=math.pi)
    obstacles.append((TOWN_HALL_POS[0], TOWN_HALL_POS[2], 14.0, 24.0))
    hero_house = add_house(
        HOME_POS, width=9, depth=8, floors=1, rot=0.0,
        wall=hex_color("#fff1c4"), roof=hex_color("#d0574a"),
    )
    specs.append({"pos": HOME_POS, **hero_house})
    for x in np.arange(-112, 120, 17.0):
        for side in (1, -1):
            for row in (0, 1):
                if row == 1 and rng.random() < 0.45:
                    continue
                z = side * (17 + row * 19 + rng.uniform(-1.5, 1.5))
                if side > 0 and abs(x) < 24:
                    continue  # town hall
                if side < 0 and abs(x) < 14:
                    continue  # north road
                if math.hypot(x - HOME_POS[0], z - HOME_POS[2]) < 15:
                    continue
                if math.hypot(x - BUILD_SITE[0], z - BUILD_SITE[2]) < 22:
                    continue
                if abs(x - SETTINGS["town"].stage[0]) < 30 and 0 < z < 30:
                    continue
                if near_stage(x, z):
                    continue
                rot = math.pi if side > 0 else 0.0
                if rng.random() < 0.12:
                    props.shop(mb, (x, 0, z), rng, rot=rot)
                    obstacles.append((x, z, 8.5, 13.0))
                else:
                    specs.append({"pos": (x, 0, z), **add_house((x, 0, z), rot=rot)})
    for z in np.arange(-26, -130, -17.0):
        for side in (1, -1):
            x = side * 16
            if math.hypot(x - HOME_POS[0], z - HOME_POS[2]) < 15 or near_stage(x, z):
                continue
            add_house((x, 0, z), rot=-side * math.pi / 2)
    props.crane(mb, (BUILD_SITE[0] + 12, 0, BUILD_SITE[2] + 6), rot=math.pi * 0.85)
    obstacles.append((BUILD_SITE[0] + 12, BUILD_SITE[2] + 6, 3.5, 24.0))
    # little fences around the build site
    for i in range(10):
        mb.box((BUILD_SITE[0] - 14 + i * 3, 0.6, BUILD_SITE[2] - 13), (2.6, 1.2, 0.2), hex_color("#ff9f1c"))
    return specs


def _trees(mb: MeshBuilder, terrain: Terrain, rng: np.random.Generator, fields: list) -> None:
    def blocked(x, z, h):
        if x > -335 and abs(z) < 12:
            return True
        if abs(x) < 12 and -340 < z < 0:
            return True
        if abs(x) < 135 and abs(z) < 70:
            return True
        if h < 0.4 and x < -280:
            return True
        for s in SETTINGS.values():
            if math.hypot(x - s.stage[0], z - s.stage[1]) < (22 if s.name != "forest" else 10):
                return True
        return False

    def inside_field(x, z):
        for x0, z0, x1, z1 in fields:
            if x0 + 1 < x < x1 - 1 and z0 + 1 < z < z1 - 1:
                return True
        return False

    candidates = rng.uniform(-WORLD_HALF + 20, WORLD_HALF - 20, (7000, 2))
    hs = terrain.height(candidates[:, 0], candidates[:, 1])
    density = fbm(candidates[:, 0] / 90, candidates[:, 1] / 90, 21)
    placed = 0
    for (x, z), h, dens in zip(candidates, hs, density):
        forest = math.hypot(x - 210, z + 120) < 95
        if blocked(x, z, h) or h > 78:
            continue
        p = 0.9 if forest else (dens - 0.3) * 2.2
        if rng.random() > p:
            continue
        if inside_field(x, z):
            continue
        size = rng.uniform(0.9, 1.5)
        if h > 18 or z < -280 or (forest and rng.random() < 0.3):
            props.tree_pine(mb, (x, h - 0.2, z), size * 1.3, rng)
        else:
            props.tree_round(mb, (x, h - 0.2, z), size, rng)
        placed += 1
    # rows of trees along some field edges (like wind breaks)
    for x0, z0, x1, z1 in fields:
        if rng.random() < 0.4:
            for x in np.arange(x0 + 2, x1 - 1, rng.uniform(4, 7)):
                props.tree_round(mb, (x + rng.uniform(-1, 1), 0, z1 + rng.uniform(-0.3, 0.3)), rng.uniform(0.8, 1.2), rng)
    # town greenery
    for x in np.arange(-120, 125, 11):
        for z in (7.6, -7.6):
            if abs(x) < 8 or rng.random() < 0.35:
                continue
            props.tree_round(mb, (x, 0, z), 0.65, rng)
    # rocks scattered on the slopes
    for _ in range(140):
        x, z = rng.uniform(-WORLD_HALF, WORLD_HALF, 2)
        h = float(terrain.height(x, z))
        if h > 8 and not blocked(x, z, h):
            props.rock(mb, (x, h, z), rng.uniform(1.5, 4), rng)


def _clouds(rng: np.random.Generator) -> MeshBuilder:
    mb = MeshBuilder()
    for _ in range(34):
        x, z = rng.uniform(-900, 900, 2)
        props.cloud(mb, (x, rng.uniform(120, 190), z), rng.uniform(9, 18), rng)
    return mb


def _water() -> MeshBuilder:
    mb = MeshBuilder()
    s = 2600
    mb.quad((-s, WATER_LEVEL, -s), (s, WATER_LEVEL, -s), (s, WATER_LEVEL, s), (-s, WATER_LEVEL, s), PALETTE["water"])
    return mb


def build_world(seed: int = 7) -> World:
    rng = np.random.default_rng(seed)
    terrain = Terrain(seed)
    static = MeshBuilder()
    static.extend(_terrain_mesh(terrain, rng))
    _roads(static)
    fields = _fields(static, terrain, rng)
    obstacles: list = []
    specs = _town(static, rng, obstacles)
    _trees(static, terrain, rng, fields)
    from . import city, interior, livingroom, skyline, village

    sky_static, sky_obstacles = skyline.build_static(seed)
    vil_static, vil_obstacles = village.build_static(seed)
    city_static, city_obstacles = city.build_static(seed)
    obstacles = [(o[0], o[1], o[2], o[3], -1.0) for o in obstacles] + sky_obstacles + vil_obstacles + city_obstacles
    return World(
        seed=seed,
        terrain=terrain,
        static=np.concatenate([static.build(), interior.build_static(), livingroom.build_static(), sky_static,
                               vil_static, city_static]),
        water=_water().build(),
        clouds=_clouds(rng).build(),
        house_specs=specs,
        obstacles=np.array(obstacles, dtype=np.float64).reshape(-1, 5),
    )
