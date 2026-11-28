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
from .props import MONSTER_VARIANTS
from .world import SETTINGS

MONSTER_VARIANT_NAMES = tuple(MONSTER_VARIANTS)

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
    "duo": "the narrator and a friend talk face to face, people walk past",
    "lr_goodbye": "(living room) mom says goodbye at the front door and warns the kids (say = her words)",
    "lr_couch": "(living room) the kids sit on the couch in front of the flickering TV",
    "lr_upstairs": "(living room) a voice calls from the dark landing upstairs (say = what it says)",
    "lr_hold": "(living room) the little sister walks towards the stairs, the narrator grabs her",
    "lr_phone": "(living room) a text message arrives on the phone (phone = the message)",
    "lr_door": "(living room) the front door opens and mom walks in (say = her words)",
    "zero_pride": "(big city) proud citizens cheer in front of the giant golden zero monument",
    "hat_walk": "(big city) the man who never takes off his hat walks by, everyone stares",
    "hat_storm": "(big city) a gust of wind blows the man's hat off and his number appears (reply = the number)",
    "zero_panic": "(big city) everyone runs away from the man with the number",
    "hat_trial": "(big city) the whole city surrounds the man in front of the monument",
    "hat_talk": "(big city) the narrator talks face to face with the man",
    "hat_gift": "(big city) the man's hat lies on the narrator's doorstep",
    "mayor_speech": "(big city) the mayor gives a speech in front of the golden zero (reply = his number)",
    "krampus_interview": "(alpine village) the narrator interviews Krampus face to face in the snowy lane",
    "lr_twins": "(living room) two identical moms stand side by side, one of them is fake",
    "lr_knock": "(living room) someone knocks on the closed front door, the kids stare at it",
    "lr_two": "(living room) mom at the door (say) and the voice upstairs (reply) at the same time",
    "speak_freeze": "the narrator says a word and the person in front of them turns to stone",
    "freeze_one": "in a crowd, one person says a forbidden word and gets switched off (turns grey and freezes)",
    "new_kid": "a new kid walks into the crowd talking normally; everyone stares",
    "unfreeze": "a switched-off grey person is talked to and switches back on",
    "bells": "(village) Krampus bells ring in the distance, everyone runs home through the snow",
    "krampus_lane": "(village) Krampus stomps down the lane, the narrator hides and watches",
    "krampus_pack": "(village) a pack of Krampusse circles the snowy square",
    "krampus_window": "(village) Krampus stands outside the narrator's chalet, staring at the window",
    "krampus_chase": "(village) Krampus chases the narrator through the snow",
    "krampus_door": "(village) face to face with Krampus at the front door",
    "krampus_leave": "(village) Krampus walks away into the forest",
    "statues": "a square full of switched-off grey people, others walk past in silence",
    "landscape": "no people, just the scenery",
    "chase": "the truck convoy flees while the giant grey monster stomps after it",
    "monster": "the giant grey monster with glowing eyes appears and roars",
    "breakdown": "the school (a school building on a bus) has broken down on the road, the monster far behind",
    "panic": "people run around screaming next to the broken-down school bus, the monster approaching",
    "standoff": "the narrator stands alone in front of the school bus, the monster stops and leans down",
    "push": "the monster gently pushes the school bus along the road, the narrator walks beside it",
    "friends": "the monster stands calmly and waves while everyone cheers around it",
    "house_leaves": "somebody's house drives away on a truck while its owner panics",
    "faceoff": "the narrator stands alone face to face with the giant monster, which leans down to look at them",
    "rampage": "the monster stomps down the street while the narrator and townspeople run ahead of it",
    "street_panic": "people scatter screaming in the street while the monster approaches",
    "hide": "the narrator and their little sister crouch and hide while the monster walks past in the distance",
    "lockdown": "9 PM: every door in town slams shut and all the lights go out, one house after another",
    "watch": "outside, the monster stands in the dark street staring at the narrator's house",
    "bed": "(bedroom) the narrator lies awake in bed, eyes darting around",
    "sit_up": "(bedroom) the narrator sits up in bed and stares at the door",
    "noises": "(bedroom) footsteps in the house, the bedroom door creaks open a little",
    "doorway": "(bedroom) glowing eyes of a creature in the dark hallway behind the open door",
    "empty_doorway": "(bedroom) the door is wide open, the hallway is empty",
    "closet": "(bedroom) the closet door creaks open and glowing eyes appear inside",
    "slum_life": "(skyline, lower city) poor people shuffle through the dark alleys, the narrator looks up",
    "look_up": "(skyline) the narrator stares up at the giant rich platform in the sky",
    "lift_ride": "(skyline) the golden elevator carries a rich person up while guards watch the gate",
    "gate_guards": "(skyline) the narrator walks to the elevator gate, a guard points them away",
    "climb": "(skyline) the narrator climbs the maintenance ladder up the pillar (low part)",
    "climb_high": "(skyline) the narrator climbs the ladder high above the city",
    "sky_view": "(skyline, top) the narrator sees the golden sky city for the first time",
    "rich_party": "(skyline, top) rich people party while the narrator hides and watches",
    "guards_chase": "(skyline, top) guards chase the narrator across the sky city",
    "mirrors": "(skyline, top) the narrator discovers the field of golden sun mirrors",
    "edge_look": "(skyline, top) the narrator stands at the railing looking down at the poor city",
    "doll_desk": "(bedroom) a creepy porcelain doll sits on the desk, the narrator stares at it from bed",
    "doll_floor": "(bedroom) the doll has moved: now it stands on the floor in the middle of the room",
    "doll_bed": "(bedroom) the doll has moved again: now it sits at the foot of the bed",
    "doll_behind": "(bedroom) the narrator sits up, the doll is right behind them on the pillow",
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
    clock: str = ""  # optional time of night shown when this scene starts, e.g. "02:00"
    card: str = ""  # comic sticker shown when the scene starts, e.g. "3 DAYS LATER"
    hero_label: str = ""  # text floating above the narrator in this scene (overrides the story default)
    npc_label: str = ""  # one text above all other people in this scene (e.g. their only word)
    say: str = ""  # what the special character of the scene says (rule breaker / partner)
    reply: str = ""  # answer of a second character (e.g. after being switched back on)
    partner: str = ""  # figure variant of the narrator's partner in this scene, e.g. "finn"
    counter: str = ""  # persistent badge at the top during the scene, e.g. "NIGHT 41"
    phone: str = ""  # text message shown on a phone ("MOM: I'm still in the car")
    chat: list = field(default_factory=list)  # phone chat thread: [["me"|"them", "text"], ...] revealed one by one
    speaker: str = ""  # who speaks this line (key into the story's "voices"); default: the narrator
    parent: str = ""  # who is "mom" in the living room scenes ("dad" swaps in the father)
    hero_as: str = ""  # draw the narrator as another figure in this scene (e.g. "hero_hat")
    pov_of: str = ""  # POV shots through the eyes of this figure variant (e.g. "mom_fake") instead of the narrator
    note: list = field(default_factory=list)  # lines on a paper note, revealed one by one
    countdown: int = 0  # big countdown (seconds) during this scene, e.g. "spot the fake in 10 seconds"
    reveal: str = ""  # figure variant to circle at the end of the scene (the answer)
    variant: str = ""  # monster variant for this scene (grey, lava, ice, shadow, swamp)
    rank: dict | None = None  # ranking card: {"n": 3, "name": "...", "stats": {"SPEED": 7, ...}}


@dataclass
class Storyboard:
    title: str
    scenes: list[Scene]
    language: str = "en"
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    hook: str = ""  # short on-screen text shown at the top during the first seconds
    monster: str = "grey"  # monster variant (grey, lava, ice, shadow, swamp)
    wheel: bool = False  # open with a spinning "tonight's monster" wheel that lands on `monster`
    night_clock: bool = False  # "survive the night" clock 10:00 PM -> 5:59 AM
    question: dict | None = None  # {"text": ..., "a": ..., "b": ...} shown at the end (comment bait)
    wheel_options: list[str] = field(default_factory=list)  # custom wheel labels (default: monster names)
    wheel_pick: str = ""  # which label the wheel lands on
    voice: str = ""  # optional edge-tts voice for this story
    pitch: str = ""  # optional pitch shift, e.g. "+10Hz" for a younger narrator
    hero_label: str = ""  # the rule made visible: text floating above the narrator (e.g. "3 DAYS")
    npc_labels: dict = field(default_factory=dict)  # {"default"|"rich"|"poor"|"guard": [texts above people]}
    label_style: str = ""  # "speech" renders labels as speech bubbles
    wheel_compact: bool = False  # small wheel at the top so the hero stays visible underneath
    voices: dict = field(default_factory=dict)  # {"speaker": {"voice", "pitch", "rate", "fx", "color"}}
    music_style: str = ""  # "horror" | "mystery" | "epic" (default: generic score)
    hero: str = ""  # alternative narrator figure, e.g. "babysitter" or "rich_kid"
    note_title: str = ""
    chat_name: str = ""  # name shown at the top of the phone chat
    duration: list = field(default_factory=list)  # optional [min, max] seconds for this story

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
            hook=str(data.get("hook") or "").strip()[:80],
            monster=_pick(data.get("monster"), MONSTER_VARIANT_NAMES, "grey"),
            wheel=bool(data.get("wheel", False)),
            night_clock=bool(data.get("night_clock", False)),
            question=_parse_question(data.get("question")),
            wheel_options=[str(o)[:24] for o in data.get("wheel_options", []) if str(o).strip()][:8],
            wheel_pick=str(data.get("wheel_pick") or "").strip(),
            voice=str(data.get("voice") or "").strip(),
            hero_label=str(data.get("hero_label") or "").strip()[:24],
            npc_labels=_parse_npc_labels(data.get("npc_labels")),
            label_style=str(data.get("label_style") or "").strip(),
            wheel_compact=bool(data.get("wheel_compact", False)),
            voices={str(k): dict(v) for k, v in (data.get("voices") or {}).items() if isinstance(v, dict)},
            music_style=str(data.get("music_style") or "").strip(),
            hero=str(data.get("hero") or "").strip(),
            note_title=str(data.get("note_title") or "").strip()[:30],
            chat_name=str(data.get("chat_name") or "").strip()[:20],
            duration=[float(x) for x in data.get("duration", [])][:2] if isinstance(data.get("duration"), list) else [],
            pitch=str(data.get("pitch") or "").strip() if re.match(r"^[+-]\d{1,3}Hz$", str(data.get("pitch") or "").strip()) else "",
        )


def _parse_npc_labels(raw) -> dict:
    if isinstance(raw, list):
        raw = {"default": raw}
    if not isinstance(raw, dict):
        return {}
    return {str(k): [str(x)[:24] for x in v if str(x).strip()][:40] for k, v in raw.items() if isinstance(v, list)}


def _parse_question(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    text, a, b = (str(raw.get(k) or "").strip() for k in ("text", "a", "b"))
    if not (text and a and b):
        return None
    return {"text": text[:90], "a": a[:40], "b": b[:40]}


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
        clock=_parse_clock(raw.get("clock")),
        card=str(raw.get("card") or "").strip()[:40],
        hero_label=str(raw.get("hero_label") or "").strip()[:24],
        npc_label=str(raw.get("npc_label") or "").strip()[:24],
        say=str(raw.get("say") or "").strip()[:32],
        reply=str(raw.get("reply") or "").strip()[:32],
        partner=str(raw.get("partner") or "").strip()[:16],
        counter=str(raw.get("counter") or "").strip()[:24],
        phone=str(raw.get("phone") or "").strip()[:120],
        chat=[[str(m[0]), str(m[1])[:90]] for m in raw.get("chat", []) if isinstance(m, (list, tuple)) and len(m) == 2][:6],
        speaker=str(raw.get("speaker") or "").strip()[:16],
        pov_of=str(raw.get("pov_of") or "").strip()[:16],
        parent=str(raw.get("parent") or "").strip()[:16],
        hero_as=str(raw.get("hero_as") or "").strip()[:16],
        note=[str(x)[:60] for x in raw.get("note", [])][:7],
        countdown=int(raw.get("countdown") or 0),
        reveal=str(raw.get("reveal") or "").strip()[:16],
        variant=str(raw.get("variant") or "").strip()[:12],
        rank=raw.get("rank") if isinstance(raw.get("rank"), dict) else None,
    )


def _parse_clock(value) -> str:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", str(value or ""))
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        return ""
    return f"{int(m.group(1)):02d}:{m.group(2)}"


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

RULES (the viral "In this town" format)
- Language of the narration: {lang_name}. One calm first-person narrator ("I" / "ich"), spoken, simple, punchy.
- Total narration: {lo}-{hi} words (about 80-90 seconds when spoken). Count carefully.
- 13-17 scenes. Each scene: 1-2 short sentences (5-20 words). Every sentence must be something we can SEE.
- Scene 1 is the HOOK and states the town's one strange rule: "In this town, ..." / "In dieser Stadt, ..."
  It must be instantly understandable and make people need to know what happens.
- Structure: rule -> how normal life works with it (2-3 funny concrete examples) -> the narrator's personal
  stake -> first crack -> escalation -> reveal -> a twist that pays the rule off in a surprising way.
- Open curiosity loops ("until the day ...", "and that's when ...") so viewers keep watching.
- The last line is a punchline that ties back to the rule (it should make people rewatch). No call to action,
  no emojis, no hashtags in the narration.
- The weather/mood follows the emotion: calm "day" at the start, "night" or "storm" at the crisis,
  "dawn" or "sunset" for the resolution. Change it at least 3 times.
- Use the giant monster (actions chase/monster/standoff/push/friends) only if it fits the story.
- Varied camera shots, never the same shot twice in a row.
- "hook": 3-6 word on-screen headline for the first seconds (e.g. "IN THIS TOWN, LIARS LOSE THEIR HOUSE").

For every scene choose exactly one value for each field:
setting:
{settings}
action:
{actions}
shot:
{shots}
mood: {moods}

Return ONLY valid JSON, no markdown:
{{"title": "...", "hook": "...", "description": "one-line video caption", "hashtags": ["...", "..."],
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
