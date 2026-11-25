#!/usr/bin/env python3
"""Low-poly story shorts: one command from a story theme to an upload-ready MP4.

Examples:
  python lowpoly_video.py "I moved my entire town on trucks"
  python lowpoly_video.py "Ich habe meine Stadt auf LKWs umgezogen" --lang de
  python lowpoly_video.py "..." --preview            # fast 540x960 draft
  python lowpoly_video.py --story-file my_story.json  # your own storyboard
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("theme", nargs="?", default="", help="story theme / idea")
    parser.add_argument("--lang", default="en", choices=["en", "de"], help="narration language")
    parser.add_argument("--voice", help="edge-tts voice, e.g. en-US-AndrewNeural or de-DE-ConradNeural")
    parser.add_argument("--tts", default="auto", choices=["auto", "edge", "espeak", "silent"])
    parser.add_argument("--story-file", help="storyboard JSON instead of generating one")
    parser.add_argument("--no-llm", action="store_true", help="use the built-in story template")
    parser.add_argument("--music", default="auto", help="'auto' (generated royalty-free score), 'songs' (resource/songs), 'none' or a path")
    parser.add_argument("--seed", type=int, help="world/camera seed (default: derived from theme)")
    parser.add_argument("--out", default=os.path.join("storage", "lowpoly"), help="output folder")
    parser.add_argument("--preview", action="store_true", help="540x960, faster encode (draft)")
    parser.add_argument("--max-seconds", type=float, help="render only the first N seconds")
    parser.add_argument("--caption-case", default="lower", choices=["lower", "upper", "keep"])
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    if not args.theme and not args.story_file:
        parser.error("give a story theme or --story-file")

    from app.services.lowpoly.pipeline import Options, generate

    opts = Options(
        theme=args.theme or os.path.splitext(os.path.basename(args.story_file))[0],
        out_dir=args.out,
        language=args.lang,
        voice=args.voice,
        tts=args.tts,
        story_file=args.story_file,
        use_llm=not args.no_llm,
        seed=args.seed,
        music=args.music,
        fps=args.fps,
        caption_case=args.caption_case,
        max_seconds=args.max_seconds,
    )
    if args.preview:
        opts.width, opts.height, opts.samples, opts.preset, opts.crf = 540, 960, 2, "veryfast", 23
    result = generate(opts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
