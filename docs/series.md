# Kurzvideo-Serien mit Zuschauer-Entscheidung

`series_cli.py` erzeugt Folgen einer Serie im Format 9:16 (1080x1920, MP4, 60–90 s).
Jede Folge endet mit einer Frage, die zwei Optionen hat. Die Antwort mit den meisten
Stimmen bestimmt die nächste Folge.

## Dateien

| Pfad | Inhalt |
|---|---|
| `series/<slug>/series.json` | Figuren (feste Bild-Beschreibung), Stil, bisherige Handlung, offene Zuschauerfrage |
| `series/<slug>/folgen/folge_XX.json` | Episoden-Skript: Szenen mit Erzählertext, Bild-Prompt, Kamera, Geräuschen und der Frage |
| `output/folge_XX/` | `folge_XX.mp4`, `skript.md`, `untertitel.srt`, `upload.txt`, `bilder/`, `episode.json` |

## Befehle

```bash
# eine Folge rendern
python series_cli.py render --series aurora --episode series/aurora/folgen/folge_01.json

# nächste Folge mit der Gewinner-Antwort (das Skript schreibt das in config.toml konfigurierte LLM)
python series_cli.py next --series aurora --answer 2

# nächste Folge mit einem fertigen Skript (z. B. von Claude geschrieben)
python series_cli.py next --series aurora --answer 2 --episode series/aurora/folgen/folge_02.json
```

Kann das LLM kein Skript schreiben, speichert `next` den fertigen Prompt unter
`series/<slug>/folgen/folge_XX_prompt.txt`.

## Was bereits vorhandene Teile übernehmen

* Stimme: Edge TTS (`voice.tts`, Stimme `de-DE-ConradNeural-Male`). Ist Edge TTS nicht
  erreichbar, wird die Offline-Stimme Piper `de_DE-thorsten-high` verwendet:
  `uv pip install piper-tts`, dann das Modell von
  <https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-thorsten-high.tar.bz2>
  nach `storage/series_models/` entpacken (oder `series_piper_model = "<pfad>.onnx"` in config.toml).
* Szenenbilder: die Materialquelle `openai_image` (`openai_image_base_url`, `openai_image_model`,
  `openai_image_api_keys` in config.toml). Sie funktioniert mit OpenAI oder einem lokalen
  SD-Server (`docs/local-image-server`). Die Figurenbeschreibungen aus `series.json` werden in
  jeden Prompt eingesetzt (`{JONAS}`, `{MIA}`). Ohne Bildquelle zeichnet `series_art.py`
  einheitliche Platzhalter-Illustrationen.
* Skript der nächsten Folge: der konfigurierte LLM-Provider (`llm._generate_response`).

## Was neu ergänzt wurde

* Untertitel mit 1–3 Wörtern in der Bildmitte; das gesprochene Wort ist gelb
* Zoom und Schwenk (`camera`), Übergänge (`crossfade`, `flash`, `black`) und Effekte
  (`shake`, `alarm_pulse`, `flicker`)
* synthetisch erzeugte, lizenzfreie Musik und Geräusche (`series_audio.py`): `waves`, `creak`, `hum`,
  `boom`, `alarm`, `alarm_far`, `water_rush`, `static`, `knock`, `drip`, `heartbeat`, `riser`
* eine Entscheidungskarte in der letzten Szene, ein Label „KI-generiert“ und Upload-Texte für
  TikTok, YouTube Shorts und Instagram Reels
