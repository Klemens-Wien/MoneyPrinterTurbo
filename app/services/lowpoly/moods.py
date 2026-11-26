"""Lighting / weather presets ("Stimmungen") and blending between them."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np

from .mesh import hex_color


@dataclass(frozen=True)
class Mood:
    sun_elevation: float  # degrees
    sun_azimuth: float  # degrees
    sun_color: tuple
    sun_strength: float
    sky_ambient: tuple
    ground_ambient: tuple
    shadow_tint: tuple
    sky_zenith: tuple
    sky_horizon: tuple
    fog_color: tuple
    fog_density: float
    cloud_tint: tuple
    exposure: float = 1.0
    saturation: float = 1.1
    rain: float = 0.0
    wind: float = 0.15
    lightning: float = 0.0  # probability weight of flashes
    stars: float = 0.0
    water_color: tuple = hex_color("#4cc3e6")
    window_glow: float = 0.0
    snow: float = 0.0

    def sun_direction(self) -> np.ndarray:
        el, az = math.radians(self.sun_elevation), math.radians(self.sun_azimuth)
        return np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])


MOODS: dict[str, Mood] = {
    "day": Mood(
        sun_elevation=50, sun_azimuth=150, sun_color=hex_color("#fff4dc"), sun_strength=1.05,
        sky_ambient=hex_color("#bcd8ff"), ground_ambient=hex_color("#9bb08a"), shadow_tint=hex_color("#a89be0"),
        sky_zenith=hex_color("#5fb4f5"), sky_horizon=hex_color("#d6f1ff"), fog_color=hex_color("#cfe9fb"),
        fog_density=0.0008, cloud_tint=hex_color("#ffffff"), exposure=1.0, saturation=1.22,
    ),
    "dawn": Mood(
        sun_elevation=14, sun_azimuth=100, sun_color=hex_color("#ffd2b0"), sun_strength=0.85,
        sky_ambient=hex_color("#c9c2f0"), ground_ambient=hex_color("#a09090"), shadow_tint=hex_color("#9a8fd6"),
        sky_zenith=hex_color("#7a8fe0"), sky_horizon=hex_color("#ffd1c1"), fog_color=hex_color("#f2d4da"),
        fog_density=0.0016, cloud_tint=hex_color("#ffd8e0"), exposure=1.0, saturation=1.1,
    ),
    "sunset": Mood(
        sun_elevation=7, sun_azimuth=-110, sun_color=hex_color("#ff9a4a"), sun_strength=1.25,
        sky_ambient=hex_color("#b58fc9"), ground_ambient=hex_color("#8a6a70"), shadow_tint=hex_color("#7a5fb8"),
        sky_zenith=hex_color("#4a4aa8"), sky_horizon=hex_color("#ffae63"), fog_color=hex_color("#f0a58a"),
        fog_density=0.0015, cloud_tint=hex_color("#ffb08a"), exposure=1.02, saturation=1.2,
        water_color=hex_color("#c77a8a"), window_glow=0.35,
    ),
    "storm": Mood(
        sun_elevation=40, sun_azimuth=20, sun_color=hex_color("#9aa4b8"), sun_strength=0.22,
        sky_ambient=hex_color("#7d8aa8"), ground_ambient=hex_color("#4f5a5a"), shadow_tint=hex_color("#5a6080"),
        sky_zenith=hex_color("#2e3648"), sky_horizon=hex_color("#69748a"), fog_color=hex_color("#5d687c"),
        fog_density=0.0030, cloud_tint=hex_color("#4a5060"), exposure=1.18, saturation=0.75,
        rain=1.0, wind=1.0, lightning=1.0, water_color=hex_color("#3e5a6a"), window_glow=0.6,
    ),
    "overcast": Mood(
        sun_elevation=45, sun_azimuth=30, sun_color=hex_color("#e6e8f0"), sun_strength=0.45,
        sky_ambient=hex_color("#b8c2d6"), ground_ambient=hex_color("#7f8a80"), shadow_tint=hex_color("#8c90b0"),
        sky_zenith=hex_color("#8a98b0"), sky_horizon=hex_color("#d0d6e0"), fog_color=hex_color("#c4cad6"),
        fog_density=0.0022, cloud_tint=hex_color("#c8ccd6"), exposure=1.0, saturation=0.95, wind=0.45,
    ),
    "night": Mood(
        sun_elevation=38, sun_azimuth=-40, sun_color=hex_color("#8fa6ff"), sun_strength=0.42,
        sky_ambient=hex_color("#3a4a8c"), ground_ambient=hex_color("#1c2438"), shadow_tint=hex_color("#28305e"),
        sky_zenith=hex_color("#0b1030"), sky_horizon=hex_color("#2a3a78"), fog_color=hex_color("#1e2a5a"),
        fog_density=0.0018, cloud_tint=hex_color("#3a4470"), exposure=1.1, saturation=1.0,
        stars=1.0, water_color=hex_color("#1a2a5a"), window_glow=1.0,
    ),
    "snow_dusk": Mood(
        sun_elevation=6, sun_azimuth=-100, sun_color=hex_color("#ffb08a"), sun_strength=0.75,
        sky_ambient=hex_color("#9aa8e0"), ground_ambient=hex_color("#6a6f9a"), shadow_tint=hex_color("#6a6cc0"),
        sky_zenith=hex_color("#3a4a9a"), sky_horizon=hex_color("#f2a6a0"), fog_color=hex_color("#b4a8d0"),
        fog_density=0.0016, cloud_tint=hex_color("#e8b8c8"), exposure=1.05, saturation=1.1,
        window_glow=0.8, snow=0.55,
    ),
    "snow_night": Mood(
        sun_elevation=40, sun_azimuth=-40, sun_color=hex_color("#a8bcff"), sun_strength=0.5,
        sky_ambient=hex_color("#4a5a9c"), ground_ambient=hex_color("#2a3058"), shadow_tint=hex_color("#2a3270"),
        sky_zenith=hex_color("#0b1030"), sky_horizon=hex_color("#2e3e80"), fog_color=hex_color("#2a3466"),
        fog_density=0.0022, cloud_tint=hex_color("#3a4470"), exposure=1.2, saturation=1.0,
        stars=0.7, window_glow=1.0, snow=0.85,
    ),
    "indoor": Mood(
        sun_elevation=26, sun_azimuth=90, sun_color=hex_color("#9fb8ff"), sun_strength=1.1,
        sky_ambient=hex_color("#2c3566"), ground_ambient=hex_color("#1a1f3a"), shadow_tint=hex_color("#1e2450"),
        sky_zenith=hex_color("#0b1030"), sky_horizon=hex_color("#24326c"), fog_color=hex_color("#10142c"),
        fog_density=0.004, cloud_tint=hex_color("#3a4470"), exposure=1.45, saturation=1.0,
        stars=1.0, water_color=hex_color("#1a2a5a"),
    ),
}

MOOD_NAMES = tuple(MOODS)


def _lerp(a, b, t):
    if isinstance(a, tuple):
        return tuple(x + (y - x) * t for x, y in zip(a, b))
    return a + (b - a) * t


def blend(a: Mood, b: Mood, t: float) -> Mood:
    if t <= 0:
        return a
    if t >= 1:
        return b
    values = {}
    for f in fields(Mood):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if f.name == "sun_azimuth":
            diff = (vb - va + 180) % 360 - 180
            values[f.name] = va + diff * t
        else:
            values[f.name] = _lerp(va, vb, t)
    return replace(a, **values)


def get(name: str) -> Mood:
    return MOODS.get(name, MOODS["day"])
