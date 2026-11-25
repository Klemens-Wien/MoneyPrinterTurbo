# Low-Poly Story Shorts

Ein Befehl macht aus einem Story-Thema ein fertiges, hochladbares Hochformat-Video
(1080×1920, 30 fps, 75–95 s, H.264/AAC, −14 LUFS) im Stil „Spielzeug-Welt in Low-Poly“:

* eigene, blockige Spielzeugfiguren (Ich-Erzähler mit oranger Jacke und blauer Kappe, dazu Stadtbewohner),
  Lastwagen mit ganzen Häusern, Busse, Autos – keine Marken-Charaktere
* bunte Low-Poly-Welt: Flickenteppich-Felder, Stadt mit Rathaus, Wald, Schneeberge, Küste, Wolken
* weiches Licht mit lila getönten Schatten, wechselnde Stimmungen: `day`, `dawn`, `sunset`, `storm`
  (Regen, Wind, Blitze + Donner), `overcast`, `night`
* Ich-Erzähler-Stimme, große Wort-für-Wort-Untertitel (gelb, dicke schwarze Kontur, Pop-Animation)
* eigene, generierte Filmmusik (lizenzfrei): Spannung folgt der Story – ruhige Drones, treibender Bass
  bei Verfolgungen, Dur-Auflösung am Ende, „Braam“-Schläge wenn das Monster auftaucht
* der „Grey Giant“: ein eigenes, 50 m großes Low-Poly-Monster mit leuchtenden Augen (laufen, brüllen,
  sich herunterbeugen, schieben, winken), stampfende Schritte mit Kamerawackeln und Bass-Wumms
* Schule auf einem Bus, die liegen bleiben kann
* Musik wird unter der Stimme geduckt, dazu Wetter-Atmo und Whooshes auf den Schnitten

## Format: „In This Town“

Die Pipeline folgt dem viralen „In This Town“-Format (Roblox-artige Story-Shorts):

* **Hook = Regel:** Der erste Satz stellt die seltsame Regel der Stadt auf („In this town, if you lie, your
  house drives away.“). Zusätzlich steht oben im Bild eine kurze Schlagzeile (`"hook"` im Storyboard).
* **Aufbau:** Regel → Alltag mit der Regel (2–3 lustige Beispiele) → persönlicher Einsatz → erster Riss →
  Eskalation → Enthüllung → Wendung, die die Regel auflöst. Der letzte Satz zahlt auf die Regel ein.
* **Tempo:** Jede Szene bekommt automatisch 2–3 Kamerawinkel, also etwa alle 2,5 s ein neues Bild.
* **Vielfalt:** YouTube nimmt seit Juli 2026 Videos von der Monetarisierung aus, die sich nur minimal
  unterscheiden („Generic or Repetitive Content“). Jede Story sollte eine eigene Regel und Wendung haben.

Beispiele: `docs/lowpoly-example-liars-house.json`, `docs/lowpoly-example-grey-giant.json`.

## Format: „Survive the Night“ (Glücksrad + Entscheidungsfrage)

Optionale Felder im Storyboard machen aus der Story eine Art Spiel-Show:

```json
{
  "monster": "lava",            // grey | lava | ice | shadow | swamp
  "wheel": true,                // Glücksrad am Anfang, landet auf diesem Monster (mit Tick-Sound + Gong)
  "night_clock": true,          // Uhr oben im Bild: 10:00 PM -> 5:59 AM
  "question": {"text": "Should I open the door?", "a": "Open it", "b": "Wait one more minute"},
  "scenes": [ {"narration": "...", "clock": "02:00", ...} ]   // "clock" stellt die Uhr passend zur Erzählung
}
```

* Nachts leuchten die Fenster der Stadt, Lava-Golem und Schattenbestie haben glühende Risse.
* Die Frage-Karte erscheint in der letzten Szene („COMMENT A OR B“) und bringt Kommentare und Stoff für Teil 2.
* Neue Aktionen dafür: `rampage` (Monster stampft durch die Hauptstraße, alle rennen), `street_panic`,
  `hide` (Erzähler und kleine Schwester verstecken sich), Ort `main_street`.

Beispiel: `docs/lowpoly-example-lava-night.json`.

## Installation

```bash
pip install -r requirements.txt -r requirements-lowpoly.txt
# Linux-Server ohne Bildschirm: Mesa-EGL für Headless-OpenGL
sudo apt-get install -y libegl1 libgl1-mesa-dri
# optional: Offline-Stimme als Fallback
sudo apt-get install -y espeak-ng
```

macOS und Windows brauchen nichts extra (moderngl nutzt dort das System-OpenGL).

## Benutzung

```bash
# Englisch (Standard), Story vom in config.toml eingestellten LLM
python lowpoly_video.py "I moved my entire town on trucks"

# Deutsch
python lowpoly_video.py "Ich habe meine ganze Stadt auf Lastwagen umgezogen" --lang de

# schneller Entwurf (540×960) bzw. nur die ersten 15 Sekunden
python lowpoly_video.py "..." --preview
python lowpoly_video.py "..." --max-seconds 15

# eigene Story statt LLM
python lowpoly_video.py --story-file meine_story.json --lang de
```

Ergebnis in `storage/lowpoly/<thema>-<seed>/`:

| Datei | Inhalt |
|---|---|
| `<titel>.mp4` | fertiges Video zum Hochladen |
| `upload.txt` | Titel, Beschreibung, Hashtags |
| `captions.srt` | Untertitel (falls die Plattform eigene Untertitel will) |
| `storyboard.json` | die verwendete Story – editierbar und mit `--story-file` neu renderbar |
| `mix.wav`, `voice/` | Tonspur und Einzelaufnahmen |

### Optionen

| Option | Bedeutung |
|---|---|
| `--lang en\|de` | Sprache von Story und Stimme |
| `--voice` | Edge-TTS-Stimme, z. B. `en-US-AndrewNeural`, `en-US-ChristopherNeural`, `de-DE-ConradNeural`, `de-DE-KatjaNeural` |
| `--tts auto\|edge\|espeak\|silent` | `auto` = Edge (Internet) → espeak-ng → stumm |
| `--no-llm` | eingebaute Vorlagen-Story statt LLM |
| `--music auto\|songs\|none\|<datei>` | `auto` = generierter lizenzfreier Score, `songs` = Track aus `resource/songs` (Lizenz selbst prüfen) |
| `--seed N` | andere Welt / Kamerawinkel bei gleichem Thema |
| `--caption-case lower\|upper\|keep` | Schreibweise der Untertitel |

## Wie es funktioniert

```
Thema ──► storyboard.py   LLM schreibt 12–16 Szenen (Ich-Perspektive, Hook, Eskalation, Twist, Payoff)
          │               und wählt pro Szene: Ort, Aktion, Kamerafahrt, Stimmung (validiert)
          ▼
          narration.py    Stimme pro Szene + Wort-Zeitstempel; Sprechtempo wird automatisch
          │               so angepasst, dass das Video 75–95 s lang wird
          ▼
          director.py     baut pro Szene Figuren/Fahrzeuge, Kamerafahrt, Stimmungs-Überblendung, Blitze
          ▼
          renderer.py     Headless-OpenGL: Schattenkarte, Himmel, flache Low-Poly-Schattierung, Nebel,
          │               Regen, Vignette  ──►  captions.py legt das aktuelle Wort darüber
          ▼
          soundtrack.py + ffmpeg   Stimme + Musik (geduckt) + Wetter-Atmo  ──►  MP4 (x264, AAC, faststart)
```

### Vokabular des Storyboards

* **setting**: `town`, `home`, `fields`, `road`, `coast`, `mountains`, `forest`, `construction`
* **action**: `convoy` (Lastwagen mit Häusern), `arrive`, `drive` (das eigene Haus auf dem Laster), `bus`, `walk`,
  `run`, `crowd`, `celebrate`, `argue`, `alone`, `build` (Haus wächst aus dem Boden), `collapse`, `wave`, `landscape`,
  Monster: `chase` (Konvoi flieht, Riese stampft hinterher), `monster` (Riese brüllt), `breakdown` (Schulbus liegen
  geblieben), `panic`, `standoff` (Riese bleibt vor dem Erzähler stehen), `push` (Riese schiebt den Bus), `friends`
* **shot**: `aerial_orbit`, `flyover`, `drone_follow`, `side_track`, `dolly_in`, `pull_back`, `low_angle`,
  `top_down`, `crane_up`, `closeup`, `chase_cam` (vor dem Konvoi, Blick zurück auf den Verfolger)
* **mood**: `day`, `dawn`, `sunset`, `storm`, `overcast`, `night`

Eine fertige Monster-Story liegt als Beispiel in `docs/lowpoly-example-grey-giant.json`:

```bash
python lowpoly_video.py --story-file docs/lowpoly-example-grey-giant.json
```

Beispiel `meine_story.json`:

```json
{
  "title": "Ich habe meine Stadt umgezogen",
  "description": "Niemand hat mir geglaubt …",
  "hashtags": ["lowpoly", "story"],
  "scenes": [
    {"narration": "Alle haben gelacht, als ich gesagt habe: Ich ziehe die ganze Stadt um.",
     "setting": "town", "action": "crowd", "shot": "dolly_in", "mood": "day"},
    {"narration": "Dann kam der Sturm.",
     "setting": "fields", "action": "convoy", "shot": "drone_follow", "mood": "storm"}
  ]
}
```

## Renderzeit

Mit GPU dauert es grob so lange wie das Video selbst. Auf einer reinen CPU-Maschine
(Mesa llvmpipe, 4 Kerne) braucht ein Bild bei 1080×1920 etwa 0,15 s; ein 85-s-Video ist dann
nach etwa 10–12 Minuten fertig. `--preview` ist rund dreimal schneller.
