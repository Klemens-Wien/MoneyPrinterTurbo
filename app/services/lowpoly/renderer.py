"""Headless OpenGL renderer (moderngl + EGL) for the low-poly world.

Passes per frame:
  1. shadow map from the sun (orthographic, follows the camera focus)
  2. sky gradient with sun disc / stars
  3. scene: flat-shaded geometry, hemisphere ambient, soft tinted shadows, fog
  4. post: exposure, saturation, vignette, lightning flash, screen-space rain
The post pass flips the image so the read-back buffer is top-down RGB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .moods import Mood

SCENE_VS = """
#version 330
in vec3 in_pos;
in vec3 in_col;
in float in_sway;
uniform mat4 u_model;
uniform mat4 u_vp;
uniform mat4 u_light_vp;
uniform float u_time;
uniform vec2 u_wind;
uniform vec3 u_cam;
out vec3 v_world;
out vec3 v_rel;
out vec3 v_col;
out vec4 v_light;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    float h = max(w.y, 0.0);
    float gust = 0.55 + 0.45 * sin(u_time * 2.1 + w.x * 0.13 + w.z * 0.09);
    w.xz += u_wind * in_sway * gust * 0.06 * h;
    v_world = w.xyz;
    v_rel = w.xyz - u_cam;  // camera-relative: keeps derivative normals precise far from the origin
    v_col = in_col;
    v_light = u_light_vp * w;
    gl_Position = u_vp * w;
}
"""

SHADOW_VS = """
#version 330
in vec3 in_pos;
in vec3 in_col;
in float in_sway;
uniform mat4 u_model;
uniform mat4 u_light_vp;
uniform float u_time;
uniform vec2 u_wind;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    float h = max(w.y, 0.0);
    float gust = 0.55 + 0.45 * sin(u_time * 2.1 + w.x * 0.13 + w.z * 0.09);
    w.xz += u_wind * in_sway * gust * 0.06 * h;
    gl_Position = u_light_vp * w;
}
"""

SHADOW_FS = """
#version 330
void main() {}
"""

SCENE_FS = """
#version 330
in vec3 v_world;
in vec3 v_rel;
in vec3 v_col;
in vec4 v_light;
uniform sampler2DShadow u_shadow;
uniform vec3 u_cam;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_sky_amb;
uniform vec3 u_ground_amb;
uniform vec3 u_shadow_tint;
uniform vec3 u_fog_col;
uniform float u_fog_density;
uniform vec3 u_tint;
uniform int u_material;
uniform float u_time;
uniform vec3 u_water_col;
uniform vec3 u_sky_horizon;
uniform float u_flash;
uniform float u_alpha;
uniform float u_window_glow;
uniform float u_grey;
uniform float u_lights_out;   // windows within this radius of the town centre are dark (-1 = none)
uniform vec3 u_pl_pos;        // optional point light (night lamp)
uniform vec3 u_pl_col;
uniform float u_pl_radius;
uniform vec3 u_pl2_pos;       // second point light (e.g. a flickering TV)
uniform vec3 u_pl2_col;
uniform float u_pl2_radius;
out vec4 f_color;

float shadow_at(vec3 N) {
    vec3 p = v_light.xyz / v_light.w * 0.5 + 0.5;
    if (p.x <= 0.0 || p.x >= 1.0 || p.y <= 0.0 || p.y >= 1.0 || p.z >= 1.0) return 1.0;
    float ndl = max(dot(N, u_sun_dir), 0.0);
    float bias = 0.00025 + 0.0012 * (1.0 - ndl);
    vec2 texel = 1.0 / vec2(textureSize(u_shadow, 0));
    float sum = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            sum += texture(u_shadow, vec3(p.xy + vec2(x, y) * texel * 1.3, p.z - bias));
    return sum / 9.0;
}

void main() {
    vec3 N = normalize(cross(dFdx(v_rel), dFdy(v_rel)));
    vec3 V = -v_rel;
    if (dot(N, V) < 0.0) N = -N;
    vec3 albedo = v_col * u_tint;
    if (u_grey > 0.0) albedo = mix(albedo, vec3(dot(albedo, vec3(0.3, 0.59, 0.11)) * 0.85 + 0.08), u_grey);
    if (u_material == 1) {
        N = normalize(vec3(0.05 * sin(v_world.x * 0.21 + u_time * 1.4) + 0.04 * sin(v_world.z * 0.33 - u_time * 1.1),
                           1.0,
                           0.05 * cos(v_world.z * 0.19 + u_time * 1.2) + 0.03 * cos(v_world.x * 0.27 + u_time)));
        albedo = u_water_col;
    }
    float ndl = max(dot(N, u_sun_dir), 0.0);
    float sh = (u_material == 2) ? 1.0 : shadow_at(N);
    vec3 amb = mix(u_ground_amb, u_sky_amb, N.y * 0.5 + 0.5);
    float lit = ndl * sh;
    vec3 light = amb * 0.48 + u_sun_col * lit * 0.85 + u_shadow_tint * 0.3 * (1.0 - lit);
    if (u_pl_radius > 0.0) {
        vec3 Lp = u_pl_pos - v_world;
        float dp = length(Lp);
        float att = pow(clamp(1.0 - dp / u_pl_radius, 0.0, 1.0), 2.0);
        light += u_pl_col * att * (0.35 + 0.65 * max(dot(N, Lp / max(dp, 1e-4)), 0.0));
    }
    if (u_pl2_radius > 0.0) {
        vec3 Lq = u_pl2_pos - v_world;
        float dq = length(Lq);
        float att2 = pow(clamp(1.0 - dq / u_pl2_radius, 0.0, 1.0), 2.0);
        light += u_pl2_col * att2 * (0.3 + 0.7 * max(dot(N, Lq / max(dq, 1e-4)), 0.0));
    }
    vec3 col = albedo * light;
    if (u_material == 1) {
        vec3 Vn = normalize(V);
        float fres = pow(1.0 - max(dot(N, Vn), 0.0), 4.0);
        col = mix(col, u_sky_horizon, fres * 0.55);
        vec3 H = normalize(Vn + u_sun_dir);
        col += u_sun_col * pow(max(dot(N, H), 0.0), 140.0) * 0.9 * sh;
    }
    if (u_material == 2) {
        col = albedo * (amb * 0.75 + u_sun_col * (0.35 + 0.35 * ndl));
    }
    if (u_material == 3) {
        col = albedo * 1.12;  // glowing (monster eyes, lava cracks)
    }
    if (u_window_glow > 0.0 && distance(v_col, vec3(0.1725, 0.2431, 0.4706)) < 0.03) {
        float lit_window = step(0.3, fract(sin(dot(floor(v_world * 0.7), vec3(12.9898, 78.233, 37.719))) * 43758.5453));
        if (u_lights_out >= 0.0 && length(v_world.xz) < u_lights_out) lit_window = 0.0;
        col = mix(col, vec3(1.0, 0.78, 0.42) * 1.25, u_window_glow * lit_window);
    }
    col += albedo * u_flash * 0.9;
    float d = length(V);
    float fog = 1.0 - exp(-pow(d * u_fog_density, 1.35));
    if (u_material == 2) fog *= 0.6;
    if (u_material == 3) fog *= 0.35;
    col = mix(col, u_fog_col, clamp(fog, 0.0, 1.0));
    f_color = vec4(col, u_alpha);
}
"""

FULLSCREEN_VS = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

SKY_FS = """
#version 330
in vec2 v_uv;
uniform mat4 u_inv_vp;
uniform vec3 u_cam;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform vec3 u_fog_col;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform float u_sun_vis;
uniform float u_stars;
uniform float u_flash;
out vec4 f_color;

float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }

void main() {
    vec4 p = u_inv_vp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = normalize(p.xyz / p.w - u_cam);
    float h = dir.y;
    vec3 col = mix(u_horizon, u_zenith, pow(clamp(h, 0.0, 1.0), 0.5));
    float sd = max(dot(dir, u_sun_dir), 0.0);
    col += u_sun_col * (pow(sd, 1400.0) * 4.0 + pow(sd, 40.0) * 0.25 + pow(sd, 6.0) * 0.12) * u_sun_vis;
    if (u_stars > 0.0) {
        vec3 cell = floor(dir * 260.0);
        float s = hash(cell);
        col += vec3(step(0.9965, s)) * u_stars * smoothstep(0.05, 0.3, h);
    }
    col = mix(u_fog_col, col, smoothstep(-0.03, 0.22, h));
    col += vec3(0.75, 0.8, 1.0) * u_flash;
    f_color = vec4(col, 1.0);
}
"""

POST_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_exposure;
uniform float u_saturation;
uniform float u_rain;
uniform float u_snow;
uniform float u_time;
uniform float u_flash;
uniform float u_aspect;
out vec4 f_color;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float rain(vec2 uv, float t) {
    float acc = 0.0;
    for (int l = 0; l < 3; l++) {
        float fl = float(l);
        float scale = 34.0 + fl * 26.0;
        vec2 p = vec2(uv.x * scale + uv.y * scale * 0.12, uv.y * scale * 0.16 * u_aspect + t * (5.5 + fl * 1.5));
        vec2 cell = floor(p);
        vec2 f = fract(p);
        float h = hash(cell + fl * 17.13);
        if (h > 0.45) {
            float offs = 0.15 + 0.7 * hash(cell * 1.7 + 3.1);
            float len0 = 0.2 + 0.5 * hash(cell + 9.2);
            float line = smoothstep(0.07, 0.0, abs(f.x - offs)) * smoothstep(len0, len0 + 0.25, f.y) * smoothstep(1.0, 0.75, f.y);
            acc += line * (0.5 - fl * 0.12);
        }
    }
    return acc;
}

float snow(vec2 uv, float t) {
    float acc = 0.0;
    for (int l = 0; l < 3; l++) {
        float fl = float(l);
        float scale = 9.0 + fl * 7.0;
        vec2 p = vec2(uv.x * scale + sin(t * 0.7 + fl) * 0.6, uv.y * scale * u_aspect * 0.55 + t * (0.9 + fl * 0.35));
        vec2 cell = floor(p);
        vec2 f = fract(p);
        float h = hash(cell + fl * 31.7);
        if (h > 0.35) {
            vec2 c = vec2(0.2 + 0.6 * hash(cell * 1.3 + 2.1), 0.2 + 0.6 * hash(cell * 2.7 + 5.3));
            c.x += sin(t * 1.5 + h * 6.28) * 0.12;
            float r = (0.05 + 0.05 * hash(cell + 7.7)) * (1.0 - fl * 0.22);
            acc += smoothstep(r, r * 0.3, length((f - c) * vec2(1.0, 1.0))) * (0.9 - fl * 0.2);
        }
    }
    return acc;
}

void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec3 col = texture(u_tex, uv).rgb * u_exposure;
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(lum), col, u_saturation);
    col = col * 1.04 - 0.01;
    col = col / (1.0 + max(col - 0.85, 0.0) * 0.9);
    if (u_rain > 0.0) {
        float r = rain(uv, u_time);
        col = mix(col, vec3(0.86, 0.9, 1.0), clamp(r * u_rain, 0.0, 0.55));
        col = mix(col, col * vec3(0.92, 0.95, 1.05), u_rain * 0.5);
    }
    if (u_snow > 0.0) {
        col = mix(col, vec3(0.97, 0.98, 1.0), clamp(snow(uv, u_time) * u_snow, 0.0, 0.9));
    }
    vec2 q = v_uv - 0.5;
    col *= 1.0 - dot(q * vec2(1.0, 0.8), q * vec2(1.0, 0.8)) * 0.55;
    col += vec3(0.25, 0.27, 0.3) * u_flash;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


# --------------------------------------------------------------------------- #
# Matrices
# --------------------------------------------------------------------------- #


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def orthographic(half: float, near: float, far: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 0] = m[1, 1] = 1.0 / half
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, (0.0, 0.0, 1.0))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _gl(m: np.ndarray) -> bytes:
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


@dataclass
class Camera:
    eye: tuple
    target: tuple
    fov: float = 50.0
    roll: float = 0.0


@dataclass
class Draw:
    key: str
    model: np.ndarray
    tint: tuple = (1.0, 1.0, 1.0)
    grey: float = 0.0  # 1 = fully desaturated (stone-like)


class Renderer:
    def __init__(self, width: int, height: int, samples: int = 4, shadow_size: int = 2048):
        import moderngl

        self.moderngl = moderngl
        self.width, self.height = width, height
        try:
            self.ctx = moderngl.create_standalone_context(backend="egl")
        except Exception:
            self.ctx = moderngl.create_standalone_context()
        ctx = self.ctx
        self.scene_prog = ctx.program(vertex_shader=SCENE_VS, fragment_shader=SCENE_FS)
        self.shadow_prog = ctx.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
        self.sky_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=SKY_FS)
        self.post_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=POST_FS)

        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self._fs_vbo = ctx.buffer(quad.tobytes())
        self.sky_vao = ctx.vertex_array(self.sky_prog, [(self._fs_vbo, "2f", "in_pos")])
        self.post_vao = ctx.vertex_array(self.post_prog, [(self._fs_vbo, "2f", "in_pos")])

        self.shadow_size = shadow_size
        self.shadow_tex = ctx.depth_texture((shadow_size, shadow_size))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)

        max_samples = ctx.max_samples
        samples = min(samples, max_samples) if samples > 1 else 0
        self.msaa_fbo = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((width, height), 4, samples=samples, dtype="f1")],
            depth_attachment=ctx.depth_renderbuffer((width, height), samples=samples),
        )
        self.resolve_tex = ctx.texture((width, height), 4)
        self.resolve_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.resolve_fbo = ctx.framebuffer(color_attachments=[self.resolve_tex])
        self.out_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((width, height), 3)])

        self.meshes: dict[str, tuple] = {}
        self.materials: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def register_mesh(self, key: str, vertices: np.ndarray, material: int = 0) -> None:
        if key in self.meshes or len(vertices) == 0:
            return
        vbo = self.ctx.buffer(np.ascontiguousarray(vertices, dtype="f4").tobytes())
        layout = (vbo, "3f 3f 1f", "in_pos", "in_col", "in_sway")
        scene_vao = self.ctx.vertex_array(self.scene_prog, [layout])
        shadow_vao = self.ctx.vertex_array(self.shadow_prog, [(vbo, "3f 12x 1f", "in_pos", "in_sway")])
        self.meshes[key] = (scene_vao, shadow_vao, vbo)
        self.materials[key] = material

    def has_mesh(self, key: str) -> bool:
        return key in self.meshes

    # ------------------------------------------------------------------ #
    def _set(self, prog, name, value):
        if name in prog:
            if isinstance(value, bytes):
                prog[name].write(value)
            else:
                prog[name].value = value

    def render(self, camera: Camera, mood: Mood, draws: list[Draw], time: float, flash: float = 0.0,
               shadow_focus=None, shadow_extent: float = 120.0, extras: dict | None = None) -> np.ndarray:
        extras = extras or {}
        ctx = self.ctx
        mgl = self.moderngl
        aspect = self.width / self.height
        eye = np.asarray(camera.eye, dtype=np.float64)
        target = np.asarray(camera.target, dtype=np.float64)
        dist = float(np.linalg.norm(target - eye))
        near = max(0.15, min(2.0, dist * 0.02))
        proj = perspective(camera.fov, aspect, near, 2600.0)
        up = (math.sin(camera.roll), math.cos(camera.roll), 0.0)
        view = look_at(eye, target, up)
        vp = proj @ view
        self.last_vp = vp

        sun = mood.sun_direction()
        focus = np.asarray(shadow_focus if shadow_focus is not None else target, dtype=np.float64)
        texel = shadow_extent * 2 / self.shadow_size
        focus = np.round(focus / texel) * texel
        light_view = look_at(focus + sun * 900.0, focus, (0.0, 1.0, 0.0) if abs(sun[1]) < 0.99 else (0.0, 0.0, 1.0))
        light_vp = orthographic(shadow_extent, 1.0, 1900.0) @ light_view

        wind_dir = np.array([0.8, 0.6]) * mood.wind
        wind = tuple(float(v) for v in wind_dir)

        # 1. shadow pass
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        ctx.polygon_offset = (2.0, 4.0)
        self._set(self.shadow_prog, "u_light_vp", _gl(light_vp))
        self._set(self.shadow_prog, "u_time", time)
        self._set(self.shadow_prog, "u_wind", wind)
        for d in draws:
            if self.materials.get(d.key, 0) != 0:
                continue
            _, shadow_vao, _ = self.meshes[d.key]
            self._set(self.shadow_prog, "u_model", _gl(d.model))
            shadow_vao.render()
        ctx.polygon_offset = (0.0, 0.0)

        # 2. sky
        self.msaa_fbo.use()
        self.msaa_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.disable(mgl.DEPTH_TEST)
        sp = self.sky_prog
        self._set(sp, "u_inv_vp", _gl(np.linalg.inv(vp)))
        self._set(sp, "u_cam", tuple(eye))
        self._set(sp, "u_zenith", mood.sky_zenith)
        self._set(sp, "u_horizon", mood.sky_horizon)
        self._set(sp, "u_fog_col", mood.fog_color)
        self._set(sp, "u_sun_dir", tuple(sun))
        self._set(sp, "u_sun_col", mood.sun_color)
        self._set(sp, "u_sun_vis", float(min(1.0, mood.sun_strength)) * (1.0 - mood.rain))
        self._set(sp, "u_stars", mood.stars)
        self._set(sp, "u_flash", flash)
        self.sky_vao.render()

        # 3. scene
        ctx.enable(mgl.DEPTH_TEST)
        p = self.scene_prog
        self.shadow_tex.use(0)
        self._set(p, "u_shadow", 0)
        self._set(p, "u_vp", _gl(vp))
        self._set(p, "u_light_vp", _gl(light_vp))
        self._set(p, "u_time", time)
        self._set(p, "u_wind", wind)
        self._set(p, "u_cam", tuple(eye))
        self._set(p, "u_sun_dir", tuple(sun))
        self._set(p, "u_sun_col", tuple(c * mood.sun_strength for c in mood.sun_color))
        self._set(p, "u_sky_amb", mood.sky_ambient)
        self._set(p, "u_ground_amb", mood.ground_ambient)
        self._set(p, "u_shadow_tint", mood.shadow_tint)
        self._set(p, "u_fog_col", mood.fog_color)
        self._set(p, "u_fog_density", mood.fog_density)
        self._set(p, "u_water_col", mood.water_color)
        self._set(p, "u_sky_horizon", mood.sky_horizon)
        self._set(p, "u_flash", flash)
        self._set(p, "u_alpha", 1.0)
        self._set(p, "u_window_glow", float(mood.window_glow))
        self._set(p, "u_lights_out", float(extras.get("lights_out", -1.0)))
        light = extras.get("point_light")
        if light:
            self._set(p, "u_pl_pos", tuple(float(v) for v in light[0]))
            self._set(p, "u_pl_col", tuple(float(v) for v in light[1]))
            self._set(p, "u_pl_radius", float(light[2]))
        else:
            self._set(p, "u_pl_radius", 0.0)
        light2 = extras.get("point_light2")
        if light2:
            self._set(p, "u_pl2_pos", tuple(float(v) for v in light2[0]))
            self._set(p, "u_pl2_col", tuple(float(v) for v in light2[1]))
            self._set(p, "u_pl2_radius", float(light2[2]))
        else:
            self._set(p, "u_pl2_radius", 0.0)
        for d in draws:
            scene_vao, _, _ = self.meshes[d.key]
            material = self.materials.get(d.key, 0)
            tint = d.tint
            if material == 2:
                tint = tuple(a * b for a, b in zip(tint, mood.cloud_tint))
            self._set(p, "u_material", material)
            self._set(p, "u_model", _gl(d.model))
            self._set(p, "u_tint", tuple(float(t) for t in tint))
            self._set(p, "u_grey", float(d.grey))
            scene_vao.render()

        # 4. resolve + post
        ctx.copy_framebuffer(self.resolve_fbo, self.msaa_fbo)
        self.out_fbo.use()
        ctx.disable(mgl.DEPTH_TEST)
        self.resolve_tex.use(0)
        pp = self.post_prog
        self._set(pp, "u_tex", 0)
        self._set(pp, "u_exposure", mood.exposure)
        self._set(pp, "u_saturation", mood.saturation)
        self._set(pp, "u_rain", mood.rain)
        self._set(pp, "u_snow", float(getattr(mood, "snow", 0.0)))
        self._set(pp, "u_time", time)
        self._set(pp, "u_flash", flash)
        self._set(pp, "u_aspect", self.height / self.width)
        self.post_vao.render()
        data = self.out_fbo.read(components=3, alignment=1)
        return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:
            pass
