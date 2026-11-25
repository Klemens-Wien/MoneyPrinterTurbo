"""Storyboard: first-person ("Ich-Erzähler") story split into shots.

The storyboard is produced by the project's configured LLM (app.services.llm)
and validated against the vocabulary the renderer understands. Without a
working LLM a built-in template story is used so the pipeline always runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from loguru import logger

from .camera import SHOTS
from .moods import MOOD_NAMES
from .world import SETTINGS

ACTIONS = {
    "convoy": "a line of colourful trucks carrying whole houses drives down the road",
    "arrive": "trucks carrying houses slow down and arrive at the place",
    "drive": "one truck carries the narrator's own house down the road",
    "bus": "a yellow bus drives by",
    "walk": "the narrator walks, a friend next to them",
    "run": "the narrator runs, other people run behind",
    "crowd": "many townspeople gather around the narrator and talk",
    "celebrate": "everyone jumps and cheers",
    "argue": "the narrator argues with two people who point at them",
    "alone": "the narrator stands alone, sad",
    "build": "a new house rises out of the ground while the narrator watches",
    "collapse": "a house shakes, tilts and sinks into the ground, the narrator panics",
    "wave": "the narrator waves at the camera",
    "landscape": "no people, just the scenery",
    "chase": "the truck convoy flees while the giant grey monster stomps after it",
    "monster": "the giant grey monster with glowing eyes appears and roars",
    "breakdown": "the school (a school building on a bus) has broken down on the road, the monster far behind",
    "panic": "people run around screaming next to the broken-down school bus, the monster approaching",
    "standoff": "the narrator stands alone in front of the school bus, the monster stops and leans down",
    "push": "the monster gently pushes the school bus along the road, the narrator walks beside it",
    "friends": "the monster stands calmly and waves while everyone cheers around it",
}
ACTION_NAMES = tuple(ACTIONS)
SETTING_NAMES = tuple(SETTINGS)

WORD_TARGETS = {"en": (195, 235), "de": (165, 205)}


@dataclass
class Scene:
    narration: str
    setting: str = "town"
    mood: str = "day"
    action: str = "landscape"
    shot: str = "aerial_orbit"


@dataclass
class Storyboard:
    title: str
    scenes: list[Scene]
    language: str = "en"
    description: str = ""
    hashtags: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(s.narration.split()) for s in self.scenes)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict, language: str | None = None) -> "Storyboard":
        scenes = [normalize_scene(s) for s in data.get("scenes", []) if str(s.get("narration", "")).strip()]
        if not scenes:
            raise ValueError("storyboard has no scenes with narration")
        _vary_shots(scenes)
        return cls(
            title=str(data.get("title") or scenes[0].narration[:60]).strip(),
            scenes=scenes,
            language=language or data.get("language") or "en",
            description=str(data.get("description") or "").strip(),
            hashtags=[str(h).lstrip("#") for h in data.get("hashtags", []) if str(h).strip()][:8],
        )


def _pick(value, allowed, default):
    value = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if value in allowed:
        return value
    for name in allowed:
        if name in value or value in name and value:
            return name
    return default


def normalize_scene(raw: dict) -> Scene:
    return Scene(
        narration=re.sub(r"\s+", " ", str(raw.get("narration", ""))).strip(),
        setting=_pick(raw.get("setting"), SETTING_NAMES, "town"),
        mood=_pick(raw.get("mood"), MOOD_NAMES, "day"),
        action=_pick(raw.get("action"), ACTION_NAMES, "landscape"),
        shot=_pick(raw.get("shot"), tuple(SHOTS), "aerial_orbit"),
    )


def _vary_shots(scenes: list[Scene]) -> None:
    """Never use the same camera move twice in a row – keeps the pacing lively."""
    fallback = ("dolly_in", "aerial_orbit", "side_track", "low_angle", "crane_up", "pull_back")
    for i in range(1, len(scenes)):
        if scenes[i].shot == scenes[i - 1].shot:
            for alt in fallback:
                if alt != scenes[i - 1].shot and (i + 1 >= len(scenes) or alt != scenes[i + 1].shot):
                    scenes[i].shot = alt
                    break


# --------------------------------------------------------------------------- #
# LLM prompt
# --------------------------------------------------------------------------- #


def build_prompt(theme: str, language: str = "en") -> str:
    lo, hi = WORD_TARGETS.get(language, WORD_TARGETS["en"])
    lang_name = {"en": "English", "de": "German"}.get(language, language)
    settings = "\n".join(f'  - "{k}": {v.description}' for k, v in SETTINGS.items())
    actions = "\n".join(f'  - "{k}": {v}' for k, v in ACTIONS.items())
    shots = "\n".join(f'  - "{k}": {v}' for k, v in SHOTS.items())
    moods = ", ".join(f'"{m}"' for m in MOOD_NAMES)
    return f"""You write scripts for viral vertical short videos (TikTok, Reels, Shorts).
The video shows a colourful low-poly toy world with blocky toy people, trucks and houses.
A first-person narrator ("I") tells a gripping, slightly absurd personal story.

STORY THEME: {theme}

RULES
- Language of the narration: {lang_name}. First person ("I" / "ich"), spoken, simple, punchy.
- Total narration: {lo}-{hi} words (about 80-90 seconds when spoken). Count carefully.
- 12-16 scenes. Each scene: 1-2 short sentences (6-24 words).
- Scene 1 is the HOOK: a bold, curious statement that makes people stay (no greeting, no intro).
- Escalate: plan -> first success -> bigger stakes -> crisis / twist (around 65-75 %) -> payoff.
- The last line is a punchline or a loop back to the hook. No call to action, no emojis, no hashtags.
- The weather/mood follows the emotion: e.g. calm "day", dramatic "storm" at the crisis,
  warm "sunset" or "dawn" at the resolution, "night" for secret moments. Change it at least 3 times.
- Use varied camera shots. Never the same shot twice in a row. Start with a strong visual.

For every scene choose exactly one value for each field:
setting:
{settings}
action:
{actions}
shot:
{shots}
mood: {moods}

Return ONLY valid JSON, no markdown:
{{"title": "...", "description": "one-line video caption", "hashtags": ["...", "..."],
 "scenes": [{{"narration": "...", "setting": "...", "action": "...", "shot": "...", "mood": "..."}}]}}"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in LLM response")
    return json.loads(text[start : end + 1])


def from_llm(theme: str, language: str = "en", attempts: int = 2) -> Storyboard:
    from app.services import llm  # imported lazily: pulls in the whole provider stack

    prompt = build_prompt(theme, language)
    last_error: Exception | None = None
    for attempt in range(attempts):
        response = llm._generate_response(prompt)
        if not response or response.startswith("Error:"):
            last_error = RuntimeError(response or "empty LLM response")
            logger.warning(f"storyboard LLM attempt {attempt + 1} failed: {response[:200] if response else ''}")
            continue
        try:
            board = Storyboard.from_dict(_extract_json(response), language)
        except Exception as exc:  # malformed JSON – ask again
            last_error = exc
            logger.warning(f"storyboard JSON invalid (attempt {attempt + 1}): {exc}")
            continue
        lo, hi = WORD_TARGETS.get(language, WORD_TARGETS["en"])
        if board.word_count < lo * 0.75 and attempt + 1 < attempts:
            prompt += f"\n\nYour last answer had only {board.word_count} words. Write {lo}-{hi} words."
            last_error = ValueError("story too short")
            continue
        return board
    raise RuntimeError(f"LLM storyboard failed: {last_error}")


# --------------------------------------------------------------------------- #
# Offline template (used when no LLM is configured / reachable)
# --------------------------------------------------------------------------- #

_TEMPLATE = {
    "en": [
        ("Everybody in my town laughed when I told them my plan: {theme}.", "town", "day", "crowd", "dolly_in"),
        ("I'm just one tiny guy in a tiny blocky world. But I was dead serious.", "home", "day", "wave", "pull_back"),
        ("So the next morning I got up before the sun and started walking.", "home", "dawn", "walk", "crane_up"),
        ("First stop: the truck yard. I rented every single truck in the county. All seven of them.", "road", "day", "convoy", "aerial_orbit"),
        ("Then came the crazy part. We loaded the houses. One by one. Onto the trucks.", "town", "day", "convoy", "side_track"),
        ("My neighbors came out in their pajamas and just stared at me.", "town", "day", "argue", "low_angle"),
        ("And then we rolled out. A whole street, driving through the fields.", "fields", "day", "convoy", "drone_follow"),
        ("For a moment, everything was perfect. I honestly thought I had done it.", "fields", "day", "walk", "flyover"),
        ("Then the sky turned black.", "fields", "storm", "landscape", "top_down"),
        ("The storm hit us halfway to the mountains. Rain, wind, and the trucks started sliding.", "mountains", "storm", "convoy", "drone_follow"),
        ("One of the houses started shaking. And it was mine.", "mountains", "storm", "collapse", "dolly_in"),
        ("I ran faster than I have ever run in my life.", "mountains", "storm", "run", "side_track"),
        ("We held it together with ropes, hope, and a lot of yelling, until the storm finally gave up.", "coast", "overcast", "crowd", "aerial_orbit"),
        ("By sunset, the whole town was standing on a brand new spot right by the sea.", "coast", "sunset", "arrive", "crane_up"),
        ("And the people who laughed at me? They were the first ones cheering.", "coast", "sunset", "celebrate", "low_angle"),
        ("So yes. {theme_cap}. And I would do it again tomorrow.", "coast", "sunset", "alone", "pull_back"),
    ],
    "de": [
        ("Alle in meiner Stadt haben gelacht, als ich ihnen meinen Plan erzählt habe: {theme}.", "town", "day", "crowd", "dolly_in"),
        ("Ich bin nur ein kleiner Typ in einer kleinen Klötzchenwelt. Aber ich meinte es todernst.", "home", "day", "wave", "pull_back"),
        ("Also bin ich am nächsten Morgen vor Sonnenaufgang losgelaufen.", "home", "dawn", "walk", "crane_up"),
        ("Erster Halt: der Lastwagenhof. Ich habe jeden einzelnen Lastwagen gemietet. Alle sieben.", "road", "day", "convoy", "aerial_orbit"),
        ("Dann kam der verrückte Teil. Wir haben die Häuser verladen. Eins nach dem anderen.", "town", "day", "convoy", "side_track"),
        ("Meine Nachbarn kamen im Schlafanzug raus und haben mich einfach nur angestarrt.", "town", "day", "argue", "low_angle"),
        ("Und dann sind wir losgefahren. Eine ganze Straße, quer durch die Felder.", "fields", "day", "convoy", "drone_follow"),
        ("Für einen Moment war alles perfekt. Ich dachte wirklich, ich hätte es geschafft.", "fields", "day", "walk", "flyover"),
        ("Dann wurde der Himmel schwarz.", "fields", "storm", "landscape", "top_down"),
        ("Der Sturm hat uns auf halbem Weg zu den Bergen erwischt. Regen, Wind, und die Laster kamen ins Rutschen.", "mountains", "storm", "convoy", "drone_follow"),
        ("Eins der Häuser fing an zu wackeln. Und es war meins.", "mountains", "storm", "collapse", "dolly_in"),
        ("Ich bin so schnell gerannt wie noch nie in meinem Leben.", "mountains", "storm", "run", "side_track"),
        ("Mit Seilen, Hoffnung und viel Geschrei haben wir durchgehalten, bis der Sturm aufgegeben hat.", "coast", "overcast", "crowd", "aerial_orbit"),
        ("Bei Sonnenuntergang stand die ganze Stadt an einem neuen Platz, direkt am Meer.", "coast", "sunset", "arrive", "crane_up"),
        ("Und die Leute, die mich ausgelacht haben? Die haben als Erste gejubelt.", "coast", "sunset", "celebrate", "low_angle"),
        ("Also ja. {theme_cap}. Und ich würde es morgen wieder tun.", "coast", "sunset", "alone", "pull_back"),
    ],
}


def from_template(theme: str, language: str = "en") -> Storyboard:
    rows = _TEMPLATE.get(language, _TEMPLATE["en"])
    theme = theme.strip().rstrip(".!?") or ("moving my whole town" if language == "en" else "meine ganze Stadt umziehen")
    theme_cap = theme[:1].upper() + theme[1:]
    scenes = [
        Scene(narration=text.format(theme=theme, theme_cap=theme_cap), setting=s, mood=m, action=a, shot=sh)
        for text, s, m, a, sh in rows
    ]
    tags = ["lowpoly", "story", "animation", "shorts", "storytime"]
    return Storyboard(title=theme_cap, scenes=scenes, language=language, description=theme_cap, hashtags=tags)


def create(theme: str, language: str = "en", use_llm: bool = True) -> Storyboard:
    if use_llm:
        try:
            return from_llm(theme, language)
        except Exception as exc:
            logger.warning(f"falling back to the built-in story template: {exc}")
    return from_template(theme, language)


def load(path: str, language: str | None = None) -> Storyboard:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return Storyboard.from_dict(data, language)
