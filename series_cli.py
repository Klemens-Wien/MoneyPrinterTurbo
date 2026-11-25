"""Kurzvideo-Serie mit Zuschauer-Entscheidungen.

Beispiele:
  # Folge aus einem fertigen Episoden-Skript rendern
  python series_cli.py render --series aurora --episode series/aurora/folgen/folge_01.json

  # Nächste Folge mit der Gewinner-Antwort der Zuschauer (Skript schreibt das konfigurierte LLM)
  python series_cli.py next --series aurora --answer 2

  # ... oder mit einem selbst geschriebenen / von Claude geschriebenen Skript
  python series_cli.py next --series aurora --answer 2 --episode series/aurora/folgen/folge_02.json

  # Nur den LLM-Prompt für die nächste Folge ausgeben
  python series_cli.py prompt --series aurora --answer 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _render(args, spec: dict) -> int:
    from app.services import series

    result = series.render_episode(
        args.series,
        spec,
        args.output,
        voice_engine=args.voice,
        images_mode=args.images,
    )
    print()
    print(f"Fertig: {result['video']}")
    print(f"Länge: {result['duration']:.1f} s · Stimme: {result['voice']} · Bilder: {', '.join(result['images'])}")
    if "placeholder" in result["images"]:
        print(
            "Hinweis: Es wurden Offline-Platzhalterbilder verwendet. Für KI-Szenenbilder "
            "openai_image_base_url / openai_image_model / openai_image_api_keys in config.toml eintragen."
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--series", default="aurora", help="Serien-Ordner unter series/")
        p.add_argument("--output", default=None, help="Ausgabe-Wurzel (Standard: output/)")
        p.add_argument("--voice", choices=("auto", "edge", "piper"), default="auto",
                       help="auto = Edge TTS, bei Fehler Offline-Stimme Piper")
        p.add_argument("--images", choices=("auto", "ai", "placeholder"), default="auto",
                       help="auto = openai_image falls konfiguriert, sonst Platzhalter")

    p_render = sub.add_parser("render", help="eine Folge aus einem Episoden-Skript rendern")
    common(p_render)
    p_render.add_argument("--episode", required=True, help="Pfad zur Episoden-JSON")

    p_next = sub.add_parser("next", help="nächste Folge mit der Zuschauer-Antwort erstellen")
    common(p_next)
    p_next.add_argument("--answer", type=int, choices=(1, 2), required=True)
    p_next.add_argument("--episode", default=None, help="optional: fertiges Episoden-Skript statt LLM")

    p_prompt = sub.add_parser("prompt", help="LLM-Prompt für die nächste Folge ausgeben")
    p_prompt.add_argument("--series", default="aurora")
    p_prompt.add_argument("--answer", type=int, choices=(1, 2), required=True)

    args = parser.parse_args(argv)
    from app.services import series

    if args.command == "render":
        return _render(args, series.load_episode_spec(args.episode))

    data = series.load_series(args.series)
    if args.command == "prompt":
        print(series.next_episode_prompt(data, args.answer))
        return 0

    # next
    updated, choice = series.apply_answer(data, args.answer)
    number = data["pending_question"]["episode"] + 1
    if args.episode:
        spec = series.load_episode_spec(args.episode)
        spec.setdefault("previous_choice", choice)
    else:
        try:
            spec = series.generate_next_episode_spec(data, args.answer)
        except Exception as exc:
            prompt_path = os.path.join(series.series_dir(args.series), "folgen", f"folge_{number:02d}_prompt.txt")
            os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(series.next_episode_prompt(data, args.answer))
            print(f"Das LLM konnte das Skript nicht schreiben: {exc}", file=sys.stderr)
            print(f"Prompt gespeichert: {prompt_path}", file=sys.stderr)
            print("Lass das Skript z. B. von Claude schreiben und starte dann mit --episode <datei>.", file=sys.stderr)
            return 1
        spec_path = os.path.join(series.series_dir(args.series), "folgen", f"folge_{number:02d}.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        print(f"Skript gespeichert: {spec_path}")
    # record the viewers' answer only once the new episode is about to be rendered
    series.save_series(args.series, updated)
    return _render(args, spec)


if __name__ == "__main__":
    sys.exit(main())
