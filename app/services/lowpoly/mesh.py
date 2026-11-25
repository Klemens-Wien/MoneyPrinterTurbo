"""Low-poly mesh construction.

Every primitive is emitted as plain (non-indexed) triangles with a colour and a
"sway" weight per vertex. Normals are not stored: the fragment shader derives a
flat face normal from screen-space derivatives, which gives the faceted toy look
for free and makes triangle winding irrelevant.

Vertex layout (7 floats): x, y, z, r, g, b, sway
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Sequence

import numpy as np

VERTEX_FLOATS = 7


def hex_color(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def vary(color: Sequence[float], rng: np.random.Generator, amount: float = 0.06):
    """Return a slightly jittered colour so repeated props do not look cloned."""
    jitter = rng.uniform(-amount, amount)
    return tuple(float(np.clip(c + jitter + rng.uniform(-amount, amount) * 0.4, 0, 1)) for c in color)


# --------------------------------------------------------------------------- #
# Matrix helpers (column vectors, row-major numpy arrays)
# --------------------------------------------------------------------------- #


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = (x, y, z)
    return m


def scale(x: float, y: float | None = None, z: float | None = None) -> np.ndarray:
    y = x if y is None else y
    z = x if z is None else z
    return np.diag([x, y, z, 1.0])


def rotate_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rotate_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rotate_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

_BOX_FACES = (
    (0, 1, 2, 3),  # -z
    (5, 4, 7, 6),  # +z
    (4, 0, 3, 7),  # -x
    (1, 5, 6, 2),  # +x
    (3, 2, 6, 7),  # +y
    (4, 5, 1, 0),  # -y
)


class MeshBuilder:
    """Accumulates triangles under a transform stack."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._matrix = np.eye(4)
        self.sway = 0.0

    # -- transform stack ---------------------------------------------------- #
    @contextmanager
    def transform(self, matrix: np.ndarray):
        previous = self._matrix
        self._matrix = previous @ matrix
        try:
            yield self
        finally:
            self._matrix = previous

    @contextmanager
    def swaying(self, weight: float):
        previous = self.sway
        self.sway = weight
        try:
            yield self
        finally:
            self.sway = previous

    # -- raw emission ------------------------------------------------------- #
    def triangles(self, verts: np.ndarray, color: Sequence[float], sway: float | None = None) -> None:
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        if len(verts) == 0:
            return
        homo = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
        world = (self._matrix @ homo.T).T[:, :3]
        chunk = np.empty((len(verts), VERTEX_FLOATS), dtype=np.float32)
        chunk[:, :3] = world
        chunk[:, 3:6] = color[:3]
        chunk[:, 6] = self.sway if sway is None else sway
        self._chunks.append(chunk)

    def triangles_colored(self, verts: np.ndarray, colors: np.ndarray, sway: np.ndarray | float = 0.0) -> None:
        """Emit triangles with one colour per triangle (colors shape: (n_tris, 3))."""
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        homo = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
        world = (self._matrix @ homo.T).T[:, :3]
        chunk = np.empty((len(verts), VERTEX_FLOATS), dtype=np.float32)
        chunk[:, :3] = world
        chunk[:, 3:6] = np.repeat(np.asarray(colors, dtype=np.float32), 3, axis=0)
        chunk[:, 6] = np.repeat(np.asarray(sway, dtype=np.float32), 3) if np.ndim(sway) else sway
        self._chunks.append(chunk)

    def quad(self, a, b, c, d, color) -> None:
        self.triangles(np.array([a, b, c, a, c, d]), color)

    # -- primitives --------------------------------------------------------- #
    def box(self, center, size, color, rot_y: float = 0.0) -> None:
        cx, cy, cz = center
        sx, sy, sz = (s / 2.0 for s in size)
        corners = np.array(
            [
                (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
                (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
            ]
        )
        if rot_y:
            c, s = math.cos(rot_y), math.sin(rot_y)
            x, z = corners[:, 0].copy(), corners[:, 2].copy()
            corners[:, 0], corners[:, 2] = c * x + s * z, -s * x + c * z
        corners += (cx, cy, cz)
        tris = []
        for a, b, c_, d in _BOX_FACES:
            tris += [corners[a], corners[b], corners[c_], corners[a], corners[c_], corners[d]]
        self.triangles(np.array(tris), color)

    def box_min(self, minimum, maximum, color) -> None:
        """Axis-aligned box from two corners."""
        mn, mx = np.asarray(minimum, float), np.asarray(maximum, float)
        self.box((mn + mx) / 2, mx - mn, color)

    def gable_roof(self, center, width, depth, height, color, overhang: float = 0.4) -> None:
        """Triangular prism roof; ridge runs along the z axis."""
        cx, cy, cz = center
        w = width / 2 + overhang
        d = depth / 2 + overhang
        l0, r0, t0 = (cx - w, cy, cz - d), (cx + w, cy, cz - d), (cx, cy + height, cz - d)
        l1, r1, t1 = (cx - w, cy, cz + d), (cx + w, cy, cz + d), (cx, cy + height, cz + d)
        self.triangles(np.array([l0, r0, t0, l1, t1, r1]), color)
        self.quad(l0, t0, t1, l1, color)
        self.quad(r0, r1, t1, t0, color)
        self.quad(l0, l1, r1, r0, color)

    def hip_roof(self, center, width, depth, height, color, overhang: float = 0.4) -> None:
        cx, cy, cz = center
        w = width / 2 + overhang
        d = depth / 2 + overhang
        ridge = max(d - w, 0.2)
        a, b = (cx - w, cy, cz - d), (cx + w, cy, cz - d)
        c, e = (cx + w, cy, cz + d), (cx - w, cy, cz + d)
        t0, t1 = (cx, cy + height, cz - ridge), (cx, cy + height, cz + ridge)
        self.triangles(np.array([a, b, t0, c, e, t1]), color)
        self.quad(b, c, t1, t0, color)
        self.quad(e, a, t0, t1, color)

    def cylinder(self, center, radius, height, color, segments: int = 8, cap: bool = True, axis: str = "y") -> None:
        cx, cy, cz = center
        angles = np.linspace(0, 2 * math.pi, segments + 1)
        ring = np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1)
        tris = []
        h = height / 2
        for i in range(segments):
            (x0, z0), (x1, z1) = ring[i], ring[i + 1]
            a, b = (x0, -h, z0), (x1, -h, z1)
            c, d = (x1, h, z1), (x0, h, z0)
            tris += [a, b, c, a, c, d]
            if cap:
                tris += [(0, h, 0), d, c, (0, -h, 0), b, a]
        verts = np.array(tris, dtype=np.float64)
        if axis == "x":
            verts = verts[:, [1, 0, 2]]
        elif axis == "z":
            verts = verts[:, [0, 2, 1]]
        self.triangles(verts + (cx, cy, cz), color)

    def cone(self, base_center, radius, height, color, segments: int = 7, rng=None, jitter: float = 0.0) -> None:
        bx, by, bz = base_center
        angles = np.linspace(0, 2 * math.pi, segments + 1)
        radii = np.full(segments + 1, radius)
        if rng is not None and jitter:
            radii[:-1] *= rng.uniform(1 - jitter, 1 + jitter, segments)
            radii[-1] = radii[0]
        apex = (bx, by + height, bz)
        tris = []
        for i in range(segments):
            p0 = (bx + math.cos(angles[i]) * radii[i], by, bz + math.sin(angles[i]) * radii[i])
            p1 = (bx + math.cos(angles[i + 1]) * radii[i + 1], by, bz + math.sin(angles[i + 1]) * radii[i + 1])
            tris += [p0, p1, apex, p1, p0, base_center]
        self.triangles(np.array(tris), color)

    def icosphere(self, center, radius, color, rng=None, jitter: float = 0.0, squash=(1.0, 1.0, 1.0), subdivisions: int = 0) -> None:
        verts, faces = _icosahedron()
        for _ in range(subdivisions):
            verts, faces = _subdivide(verts, faces)
        verts = verts * radius
        if rng is not None and jitter:
            verts = verts * rng.uniform(1 - jitter, 1 + jitter, (len(verts), 1))
        verts = verts * np.asarray(squash) + np.asarray(center)
        self.triangles(verts[faces.reshape(-1)], color)

    # -- output ------------------------------------------------------------- #
    def extend(self, other: "MeshBuilder") -> None:
        self._chunks.extend(other._chunks)

    def build(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros((0, VERTEX_FLOATS), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(self._chunks, axis=0), dtype=np.float32)

    def __len__(self) -> int:
        return sum(len(c) for c in self._chunks)


_ICO_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _icosahedron():
    global _ICO_CACHE
    if _ICO_CACHE is None:
        t = (1 + 5 ** 0.5) / 2
        v = np.array(
            [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)],
            dtype=np.float64,
        )
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        f = np.array(
            [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)],
            dtype=np.int64,
        )
        _ICO_CACHE = (v, f)
    return _ICO_CACHE


def _subdivide(verts: np.ndarray, faces: np.ndarray):
    verts = list(map(tuple, verts))
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = (np.asarray(verts[a]) + np.asarray(verts[b])) / 2
            m /= np.linalg.norm(m)
            verts.append(tuple(m))
            cache[key] = len(verts) - 1
        return cache[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts), np.array(new_faces, dtype=np.int64)


def merge(builders: Iterable[MeshBuilder]) -> np.ndarray:
    parts = [b.build() for b in builders]
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros((0, VERTEX_FLOATS), dtype=np.float32)
    return np.concatenate(parts, axis=0)
