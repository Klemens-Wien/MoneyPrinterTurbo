"""End-to-end: theme -> storyboard -> narration -> 3D render -> captions -> MP4."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import zlib
from dataclasses import dataclass

from loguru import logger

from . import captions as cap
from . import narration, overlays, score, soundtrack
from . import storyboard as sb
from .director import Director
from .narration import Word
from .renderer import Renderer
from .world import build_world

DURATION_RANGE = (75.0, 95.0)


@dataclass
class Options:
    theme: str
    out_dir: str
    language: str = "en"
    voice: str | None = None
    tts: str = "auto"  # auto | edge | espeak | silent
    story_file: str | None = None
    use_llm: bool = True
    seed: int | None = None
    music: str | None = "auto"  # auto (generated score) | songs | none | <path>
    width: int = 1080
    height: int = 1920
    fps: int = 30
    samples: int = 4
    crf: int = 18
    preset: str = "medium"
    caption_case: str = "lower"
    max_seconds: float | None = None  # render only the first N seconds (quick look)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "story"


def _scene_durations(clips: list[narration.Clip], gap: float) -> list[float]:
    durations = [c.duration + gap for c in clips]
    total = sum(durations)
    lo, hi = DURATION_RANGE
    if total < lo + 1:
        extra = lo + 1 - total
        per = min(extra / len(durations), 1.0)
        durations = [d + per for d in durations]
        extra -= per * len(durations)
        if extra > 0:
            durations[-1] += extra
    durations[-1] += 1.0  # breathing room at the end
    return durations


def _ffmpeg_writer(path: str, opts: Options, audio_path: str) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{opts.width}x{opts.height}", "-r", str(opts.fps), "-i", "-",
        "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf), "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-r", str(opts.fps), "-g", str(opts.fps * 2),
        "-c:a", "aac", "-b:a", "192k", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-shortest", "-movflags", "+faststart", path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def generate(opts: Options) -> dict:
    started = time.time()
    seed = opts.seed if opts.seed is not None else zlib.crc32(opts.theme.encode("utf-8")) % 100000
    work = os.path.join(opts.out_dir, f"{_slug(opts.theme)}-{seed}")
    os.makedirs(work, exist_ok=True)
    logger.info(f"output folder: {work}")

    # 1. story
    if opts.story_file:
        board = sb.load(opts.story_file, opts.language)
    else:
        board = sb.create(opts.theme, opts.language, use_llm=opts.use_llm)
    with open(os.path.join(work, "storyboard.json"), "w", encoding="utf-8") as fh:
        json.dump(board.to_dict(), fh, ensure_ascii=False, indent=2)
    logger.info(f"storyboard: '{board.title}' – {len(board.scenes)} scenes, {board.word_count} words")

    # 2. narration
    gap = 0.18
    clips, backend = narration.synthesize(
        [s.narration for s in board.scenes], os.path.join(work, "voice"), board.language, opts.voice, opts.tts,
        target_range=DURATION_RANGE, gap=gap,
    )
    durations = _scene_durations(clips, gap)
    logger.info(f"narration via {backend}: {sum(durations):.1f}s")

    # 3. direction
    world = build_world(seed)
    director = Director(world, board, durations, seed)
    wheel = clock = question = None
    if board.wheel:
        from .props import MONSTER_VARIANTS

        names = list(MONSTER_VARIANTS)
        wheel = overlays.MonsterWheel(opts.width, opts.height, [MONSTER_VARIANTS[n]["name"] for n in names],
                                      names.index(board.monster), start=0.25,
                                      spin=max(2.6, min(3.6, durations[0] - 1.6)), hold=1.2)
    if board.night_clock and director.scenes:
        clock_start = director.scenes[1].start if wheel and len(director.scenes) > 1 else 0.0
        clock_end = director.scenes[-1].start if board.question else director.duration
        keys = []
        for ps in director.scenes:
            if ps.scene.clock:
                hh, mm = map(int, ps.scene.clock.split(":"))
                keys.append((ps.start, hh * 60 + mm))
        start_m = keys[0][1] if keys else 22 * 60
        clock = overlays.NightClock(opts.width, opts.height, clock_start, clock_end, director.duration,
                                    from_minutes=start_m, keyframes=keys)
    if board.question and director.scenes:
        q = board.question
        question = overlays.QuestionCard(opts.width, opts.height, overlays.Question(q["text"], q["a"], q["b"]),
                                         director.scenes[-1].start + 0.3, director.duration + 1)
    starts = [s.start for s in director.scenes]

    words: list[Word] = []
    for clip, start in zip(clips, starts):
        words += [Word(w.text, w.start + start, w.end + start) for w in clip.words]
    timed = cap.build_timeline(words)
    cap.to_srt(timed, os.path.join(work, "captions.srt"))

    # 4. sound
    music = None
    music_volume = 0.30
    if opts.music == "auto":
        music = score.write_score(os.path.join(work, "score.wav"), director.duration, director.scenes,
                                  director.impacts, seed)
        music_volume = 0.42
    elif opts.music == "songs":
        music = soundtrack.pick_song(seed)
    elif opts.music and opts.music != "none":
        music = opts.music
    total = director.duration if not opts.max_seconds else min(director.duration, opts.max_seconds)
    audio_path = os.path.join(work, "mix.wav")
    soundtrack.mix(
        audio_path, director.duration,
        [soundtrack.VoicePlacement(c.path, s) for c, s in zip(clips, starts)],
        director.storm_spans, director.flashes, director.cut_times, music, seed,
        music_volume=music_volume, thuds=director.monster_steps,
        ticks=wheel.tick_times() if wheel else None, dings=[wheel.land_time] if wheel else None,
        sfx=director.sfx,
    )

    # 5. render
    renderer = Renderer(opts.width, opts.height, samples=opts.samples)
    renderer.register_mesh("static", world.static)
    renderer.register_mesh("water", world.water, material=1)
    renderer.register_mesh("clouds", world.clouds, material=2)
    for key, verts in director.library.meshes.items():
        renderer.register_mesh(key, verts, material=director.library.materials.get(key, 0))

    captioner = cap.CaptionRenderer(opts.width, opts.height, case=opts.caption_case)
    hook_end = min(director.scenes[0].end, 4.0) if director.scenes else 3.5
    banner = cap.HookBanner(opts.width, opts.height, "" if wheel else board.hook, 0.0, hook_end)
    video_path = os.path.join(work, f"{_slug(board.title)}.mp4")
    writer = _ffmpeg_writer(video_path, opts, audio_path)
    frames = int(total * opts.fps)
    cursor = 0
    render_started = time.time()
    try:
        for i in range(frames):
            t = i / opts.fps
            cam, mood, draws, flash, extent, extras = director.frame(t)
            frame = renderer.render(cam, mood, draws, t, flash=flash, shadow_extent=extent, extras=extras)
            word, cursor = cap.active_word(timed, t, cursor)
            if wheel is not None:
                frame = wheel.composite(frame, t)
            if clock is not None:
                frame = clock.composite(frame, t)
            if question is not None:
                frame = question.composite(frame, t)
            low = (wheel is not None and t < wheel.end)
            frame = captioner.composite(frame, word, t, y_frac=0.85 if low else None)
            frame = banner.composite(frame, t)
            writer.stdin.write(frame.tobytes())
            if i % (opts.fps * 5) == 0:
                elapsed = time.time() - render_started
                eta = elapsed / max(i, 1) * (frames - i)
                logger.info(f"frame {i}/{frames}  ({t:5.1f}s)  eta {eta:5.0f}s")
    finally:
        writer.stdin.close()
        code = writer.wait()
        renderer.release()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with code {code}")

    # 6. upload metadata
    tags = " ".join("#" + h.replace(" ", "") for h in board.hashtags)
    with open(os.path.join(work, "upload.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"{board.title}\n\n{board.description}\n\n{tags}\n")
    result = {
        "video": video_path,
        "duration": round(total, 2),
        "scenes": len(board.scenes),
        "tts": backend,
        "music": music,
        "folder": work,
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(work, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    logger.success(f"done: {video_path} ({total:.1f}s) in {result['seconds']}s")
    return result
