"""Camera moves ("shots") relative to a moving focus point."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .renderer import Camera

SHOTS = {
    "aerial_orbit": "high drone shot slowly circling the action (establishing shot)",
    "flyover": "drone flies forward over the landscape toward the action",
    "drone_follow": "drone follows behind/above a moving vehicle or person",
    "side_track": "camera glides sideways alongside the moving action",
    "dolly_in": "camera pushes in from medium distance to a close view (tension)",
    "pull_back": "starts close on the narrator and pulls far back to reveal the scene (big reveal)",
    "low_angle": "camera near the ground looking up at the action (heroic / dramatic)",
    "top_down": "bird's-eye view straight down, slowly rotating",
    "crane_up": "camera rises from ground level to a high view",
    "closeup": "close view on the narrator's face and upper body",
}

SHOT_NAMES = tuple(SHOTS)


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _offset(yaw: float, dist: float, height: float) -> np.ndarray:
    return np.array([math.sin(yaw) * dist, height, math.cos(yaw) * dist])


def shot_camera(
    shot: str,
    u: float,
    t: float,
    focus: Callable[[float], np.ndarray],
    forward: Callable[[float], np.ndarray],
    yaw: float,
    scale: float = 1.0,
    ground: Callable[[float, float], float] | None = None,
    obstacles: np.ndarray | None = None,
) -> Camera:
    """u: normalised scene progress 0..1, t: scene time in seconds."""
    f = np.asarray(focus(t), dtype=np.float64)
    e = ease(u)
    s = scale
    if shot == "aerial_orbit":
        eye = f + _offset(yaw + 0.35 * u, 95 * s, 70 * s)
        cam = Camera(eye, f, 42)
    elif shot == "flyover":
        d = np.array([math.sin(yaw + math.pi), 0.0, math.cos(yaw + math.pi)])
        eye = f - d * (150 * s - 90 * s * u) + np.array([0.0, 60 * s - 15 * s * u, 0.0])
        cam = Camera(eye, f + d * 30 * s, 50)
    elif shot == "drone_follow":
        fw = np.asarray(forward(t), dtype=np.float64)
        fw = fw / (np.linalg.norm(fw) + 1e-9)
        side = np.array([fw[2], 0.0, -fw[0]])
        eye = f - fw * 30 * s + side * 10 * s + np.array([0.0, 17 * s, 0.0])
        cam = Camera(eye, f + fw * 8 * s, 50)
    elif shot == "side_track":
        fw = np.asarray(forward(t), dtype=np.float64)
        side = np.array([fw[2], 0.0, -fw[0]])
        if np.dot(side, _offset(yaw, 1, 0)) < 0:
            side = -side
        eye = f + side * 26 * s - fw * (8 - 10 * u) * s + np.array([0.0, 11 * s, 0.0])
        cam = Camera(eye, f, 48)
    elif shot == "dolly_in":
        dist = (42 - 30 * e) * s
        eye = f + _offset(yaw + 0.15 * u, dist, (16 - 12 * e) * s)
        cam = Camera(eye, f, 48)
    elif shot == "pull_back":
        k = e ** 1.6
        dist = 6 + (150 * s - 6) * k
        height = 1.2 + (95 * s - 1.2) * k
        eye = f + _offset(yaw - 0.5 * u, dist, height)
        cam = Camera(eye, f + np.array([0, 0.4 * (1 - k), 0]), 46)
    elif shot == "low_angle":
        eye = f + _offset(yaw + 0.25 * u, 10 * s, -1.0)
        eye[1] = max(eye[1], 0.5)
        cam = Camera(eye, f + np.array([0.0, 0.8 + 2.0 * s, 0.0]), 58)
    elif shot == "top_down":
        eye = f + _offset(yaw + 0.6 * u, 6 * s, (110 - 25 * e) * s)
        cam = Camera(eye, f, 45, roll=0.0)
    elif shot == "crane_up":
        eye = f + _offset(yaw, (18 + 30 * e) * s, 1.5 + 50 * e * s)
        cam = Camera(eye, f + np.array([0, 1.5 * (1 - e), 0]), 50)
    elif shot == "closeup":
        eye = f + _offset(yaw + 0.12 * u, 4.6 - 0.8 * e, 0.6)
        cam = Camera(eye, f + np.array([0.0, 0.4, 0.0]), 40)
    else:
        eye = f + _offset(yaw, 40 * s, 20 * s)
        cam = Camera(eye, f, 50)
    if ground is not None:
        g = ground(float(cam.eye[0]), float(cam.eye[2]))
        if cam.eye[1] < g + 0.6:
            cam.eye = (cam.eye[0], g + 0.6, cam.eye[2])
    if obstacles is not None and len(obstacles):
        cam.eye = avoid_obstacles(np.asarray(cam.eye, dtype=np.float64), np.asarray(cam.target, dtype=np.float64), obstacles)
    # subtle handheld drift keeps static moments alive
    drift = np.array([math.sin(t * 0.9), math.sin(t * 1.3 + 1.0) * 0.6, math.cos(t * 0.7)]) * 0.12 * (1 + 0.02 * np.linalg.norm(np.asarray(cam.eye) - f))
    cam.eye = tuple(np.asarray(cam.eye, dtype=np.float64) + drift)
    cam.target = tuple(np.asarray(cam.target, dtype=np.float64))
    return cam


def avoid_obstacles(eye: np.ndarray, target: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    """Lift the camera out of / above buildings that would block the view.

    obstacles: (n, 4) array of x, z, radius, height (cylindrical proxies).
    """
    eye = eye.copy()
    near = obstacles[np.hypot(obstacles[:, 0] - eye[0], obstacles[:, 1] - eye[2]) < 400]
    if not len(near):
        return eye
    samples = np.linspace(0.0, 0.92, 24)[:, None]
    for _ in range(12):
        pts = eye + (target - eye) * samples  # (s, 3)
        dx = pts[:, None, 0] - near[None, :, 0]
        dz = pts[:, None, 2] - near[None, :, 1]
        inside = (np.hypot(dx, dz) < near[None, :, 2]) & (pts[:, None, 1] < near[None, :, 3])
        if not inside.any():
            break
        # never lift for an obstacle the target itself stands in (e.g. a house on a truck)
        tgt_inside = (np.hypot(target[0] - near[:, 0], target[2] - near[:, 1]) < near[:, 2])
        inside &= ~tgt_inside[None, :]
        if not inside.any():
            break
        eye[1] += 3.0
    return eye
