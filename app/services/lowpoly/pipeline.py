"""End-to-end: theme -> storyboard -> narration -> 3D render -> captions -> MP4."""

from __future__ import annotations

import dataclasses
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
    pitch: str | None = None  # e.g. "+10Hz" (younger) / "-8Hz" (deeper)
    base_rate: int = 15  # speaking rate in percent: brisk but still natural (no pitch tricks on the narrator)
    duration_range: tuple | None = None  # (min, max) seconds; default: story setting or 75-95


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "story"


def _scene_durations(clips: list[narration.Clip], gap: float, duration_range=DURATION_RANGE) -> list[float]:
    durations = [c.duration + gap for c in clips]
    total = sum(durations)
    lo, hi = duration_range
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
    duration_range = tuple(opts.duration_range or board.duration or DURATION_RANGE)
    # *word* marks a stressed word: spoken normally, shown in yellow
    spoken = [s.narration.replace("*", "") for s in board.scenes]
    stressed = [{re.sub(r"\W", "", w).lower() for w in re.findall(r"\*([^*]+)\*", s.narration) for w in w.split()}
                for s in board.scenes]
    specs = [dict(board.voices.get(s.speaker or "narrator", {})) for s in board.scenes]
    clips, backend = narration.synthesize(
        spoken, os.path.join(work, "voice"), board.language,
        opts.voice or board.voice or None, opts.tts,
        target_range=duration_range, gap=gap, pitch=opts.pitch or board.pitch or "+0Hz", rate_pct=opts.base_rate,
        specs=specs if board.voices else None,
    )
    durations = _scene_durations(clips, gap, duration_range)
    logger.info(f"narration via {backend}: {sum(durations):.1f}s")

    # 3. direction
    world = build_world(seed)
    director = Director(world, board, durations, seed)
    wheel = clock = question = None
    text = overlays.strings(board.language)
    if board.wheel:
        from .props import MONSTER_VARIANTS

        if board.wheel_options:
            options = board.wheel_options
            pick = board.wheel_pick if board.wheel_pick in options else options[0]
        else:
            names = list(MONSTER_VARIANTS)
            options = [MONSTER_VARIANTS[n]["name"] for n in names]
            pick = MONSTER_VARIANTS[board.monster]["name"]
        wheel = overlays.MonsterWheel(opts.width, opts.height, options, options.index(pick), start=0.25,
                                      spin=max(2.6, min(3.6, durations[0] - 1.6)), hold=1.2,
                                      title=text["wheel_title"], compact=board.wheel_compact)
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
                                    from_minutes=start_m, keyframes=keys, label=text["clock"],
                                    h24=board.language == "de")
    if board.question and director.scenes:
        q = board.question
        question = overlays.QuestionCard(opts.width, opts.height, overlays.Question(q["text"], q["a"], q["b"]),
                                         director.scenes[-1].start + 0.3, director.duration + 1, cta=text["cta"])
    starts = [s.start for s in director.scenes]

    words: list[Word] = []
    for clip, start, marks in zip(clips, starts, stressed):
        words += [Word(w.text, w.start + start, w.end + start, re.sub(r"\W", "", w.text).lower() in marks)
                  for w in clip.words]
    timed = cap.build_timeline(words)
    if board.voices:  # colour the captions by who is speaking
        for ps in director.scenes:
            color = board.voices.get(ps.scene.speaker or "narrator", {}).get("color")
            if color:
                rgb = tuple(int(str(color).lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                for w in timed:
                    if ps.start <= w.start < ps.end:
                        w.color = rgb
    cap.to_srt(timed, os.path.join(work, "captions.srt"))

    countdowns = [overlays.Countdown(opts.width, opts.height, ps.scene.countdown, ps.start + 0.2, ps.end,
                                     title="" if (ps.scene.wyr or ps.scene.doors) else (ps.scene.card or ""),
                                     y_frac=0.41 if (ps.scene.wyr or ps.scene.doors) else 0.16)
                  for ps in director.scenes if ps.scene.countdown]
    wyrs = [overlays.WouldYouRather(opts.width, opts.height, str(ps.scene.wyr.get("a", "")),
                                    str(ps.scene.wyr.get("b", "")), ps.start + 0.05, ps.end,
                                    pct=ps.scene.wyr.get("pct"))
            for ps in director.scenes if ps.scene.wyr]
    doors = [overlays.DoorPicker(opts.width, opts.height, ps.start + 0.05, ps.end,
                                 opened=int(ps.scene.doors.get("opened", 0)), stamp=str(ps.scene.doors.get("stamp", "")),
                                 dead=list(ps.scene.doors.get("dead", [])))
             for ps in director.scenes if ps.scene.doors]
    door_hits = [d.start + 1.2 for d in doors if d.stamp]
    notes = [overlays.NoteCard(opts.width, opts.height, board.note_title or "RULES", ps.scene.note, ps.start + 0.1,
                               ps.end) for ps in director.scenes if ps.scene.note]
    reveals = [(ps.start + max(0.3, (ps.end - ps.start) * 0.35), ps.end) for ps in director.scenes if ps.scene.reveal]
    tick_list = (wheel.tick_times() if wheel else []) + [x for c in countdowns for x in c.tick_times()]
    ding_list = ([wheel.land_time] if wheel else []) + [x for nt in notes for x in nt.line_times()] + \
        [a for a, _ in reveals] + door_hits + [w.start + 0.4 for w in wyrs if w.pct is not None]

    # 4. sound
    music = None
    music_volume = 0.30
    if opts.music == "auto":
        music = score.write_score(os.path.join(work, "score.wav"), director.duration, director.scenes,
                                  director.impacts, seed, style=board.music_style)
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
        ticks=tick_list or None, dings=ding_list or None,
        sfx=director.sfx,
    )

    # 5. automatic shot QA on a small preview renderer
    qa = Renderer(216, 384, samples=0, shadow_size=1024)
    qa.register_mesh("static", world.static)
    qa.register_mesh("water", world.water, material=1)
    qa.register_mesh("clouds", world.clouds, material=2)
    for key, verts in director.library.meshes.items():
        qa.register_mesh(key, verts, material=director.library.materials.get(key, 0))

    def _qa_render(tq):
        cq, mq, dq, fq, eq, xq = director.frame(tq)
        mq = dataclasses.replace(mq, rain=0.0, snow=0.0)  # weather streaks would hide a camera stuck in a wall
        return qa.render(cq, mq, dq, tq, flash=0.0, shadow_extent=eq, extras=xq)

    fixed = director.qa_fix(_qa_render, log=logger.info)
    qa.release()
    logger.info(f"shot QA: {fixed} shot(s) replaced")

    # 6. render
    renderer = Renderer(opts.width, opts.height, samples=opts.samples)
    renderer.register_mesh("static", world.static)
    renderer.register_mesh("water", world.water, material=1)
    renderer.register_mesh("clouds", world.clouds, material=2)
    for key, verts in director.library.meshes.items():
        renderer.register_mesh(key, verts, material=director.library.materials.get(key, 0))

    captioner = cap.CaptionRenderer(opts.width, opts.height, case=opts.caption_case)
    hook_end = min(director.scenes[0].end, 4.0) if director.scenes else 3.5
    banner = cap.HookBanner(opts.width, opts.height, "" if wheel else board.hook, 0.0, hook_end)
    labels = overlays.LabelLayer(opts.width, opts.height, scale=board.label_scale, max_labels=board.max_labels)
    stickers = [overlays.StickerCard(opts.width, opts.height, ps.scene.card, ps.start + 0.05)
                for ps in director.scenes if ps.scene.card and not ps.scene.countdown and not ps.scene.wyr
                and not ps.scene.doors]
    counter = overlays.CounterBadge(opts.width, opts.height,
                                    [(ps.start, ps.end, ps.scene.counter) for ps in director.scenes if ps.scene.counter])
    chat_name = board.chat_name or ("Mom" if board.language != "de" else "Mama")
    chats = [overlays.PhoneChat(opts.width, opts.height, chat_name, ps.scene.chat, ps.start + 0.2, ps.end)
             for ps in director.scenes if ps.scene.chat]
    phones = [overlays.PhoneCard(opts.width, opts.height, "Mom" if board.language != "de" else "Mama",
                                 ps.scene.phone, ps.start + 0.3, ps.end) for ps in director.scenes if ps.scene.phone]
    ranks = [overlays.RankCard(opts.width, opts.height, ps.scene.rank, ps.start, ps.end, board.language)
             for ps in director.scenes if ps.scene.rank]
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
            frame = labels.composite(frame, extras.get("labels", []), getattr(renderer, "last_vp", None), cam.eye)
            frame = overlays.pov_say(labels, frame, extras.get("pov_say"))
            frame = counter.composite(frame, t)
            for card in ranks:
                frame = card.composite(frame, t)
            for card in phones:
                frame = card.composite(frame, t)
            for card in chats:
                frame = card.composite(frame, t)
            for sticker in stickers:
                frame = sticker.composite(frame, t)
            for nt in notes:
                frame = nt.composite(frame, t)
            asking = question is not None and t >= question.start  # the final question card owns the top
            for card in wyrs + doors:
                if not asking:
                    frame = card.composite(frame, t)
            for cd in countdowns:
                frame = cd.composite(frame, t)
            for a, b in reveals:
                if a <= t < b:
                    frame = overlays.reveal_circle(frame, extras.get("reveal"), getattr(renderer, "last_vp", None), t, a)
            if wheel is not None:
                frame = wheel.composite(frame, t)
            if clock is not None:
                frame = clock.composite(frame, t)
            if question is not None:
                frame = question.composite(frame, t)
            low = wheel is not None and t < wheel.end
            on_phone = any(c.start <= t < c.end for c in chats + phones)
            low = low or any(nt.start <= t < nt.end for nt in notes)
            frame = captioner.composite(frame, word, t, y_frac=0.84 if low else 0.2 if on_phone else None)
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
