import json
from pathlib import Path

import numpy as np
import pytest

from app.services import series, series_audio

ROOT = Path(__file__).resolve().parents[2]
EPISODE_1 = ROOT / "series" / "aurora" / "folgen" / "folge_01.json"


def _spec():
    return json.loads(EPISODE_1.read_text(encoding="utf-8"))


def test_episode_one_spec_is_valid():
    assert series.validate_episode_spec(_spec()) == []


def test_validation_requires_final_decision_scene():
    spec = _spec()
    spec["scenes"][-1]["decision"] = False
    assert any("decision" in p for p in series.validate_episode_spec(spec))


def test_validation_rejects_unknown_sound_effect():
    spec = _spec()
    spec["scenes"][0]["sfx"] = ["explosion"]
    assert any("explosion" in p for p in series.validate_episode_spec(spec))


def test_image_prompt_inserts_fixed_character_descriptions():
    data = series.load_series("aurora")
    scene = {"image_prompt": "{JONAS} holds the hand of {MIA}"}
    prompt = series.expand_image_prompt(data, scene)
    assert "grey hoodie" in prompt
    assert "yellow raincoat" in prompt
    assert "{" not in prompt


def test_words_are_grouped_into_one_to_three_words():
    words = series.distribute_words("Dann ein dumpfer Schlag. Tief unten im Rumpf.", 0.0, 3.0)
    groups = series.group_words(words)
    assert all(1 <= len(g) <= 3 for g in groups)
    # a sentence end always closes a group
    assert series.display_word(groups[1][-1].text) == "Schlag" or groups[0][-1].text.endswith(".")
    assert words[0].start == 0.0 and abs(words[-1].end - 3.0) < 1e-6


def test_srt_time_format():
    assert series._srt_time(62.5) == "00:01:02,500"


def test_apply_answer_marks_winner():
    data = {
        "episodes": [{"episode": 1, "winning_answer": None}],
        "pending_question": {"episode": 1, "text": "?", "option_1": "Tauchen", "option_2": "Klettern"},
    }
    updated, choice = series.apply_answer(data, 2)
    assert choice == "2 – Klettern"
    assert updated["episodes"][0]["winning_answer"] == "2 – Klettern"
    assert data["episodes"][0]["winning_answer"] is None


def test_apply_answer_without_question_fails():
    with pytest.raises(ValueError):
        series.apply_answer({"episodes": [], "pending_question": None}, 1)


def test_camera_box_stays_inside_the_artwork():
    for move in series.CAMERA_MOVES:
        for p in (0.0, 0.5, 1.0):
            x0, y0, x1, y1 = series.camera_box(move, p)
            assert x0 >= -1 and y0 >= -1
            assert x1 <= series.ART_W + 1 and y1 <= series.ART_H + 1


@pytest.mark.parametrize("name", sorted(series_audio.SFX))
def test_sound_effects_are_finite_and_bounded(name):
    samples, gain = series_audio.render_sfx(name, 1.5)
    assert samples.size > 0
    assert np.isfinite(samples).all()
    assert np.max(np.abs(samples)) <= 1.0 + 1e-5
    assert 0 < gain <= 1
