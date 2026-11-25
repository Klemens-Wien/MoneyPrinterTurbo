import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.lowpoly import captions, moods, storyboard
from app.services.lowpoly.actors import ActorLibrary
from app.services.lowpoly.camera import SHOT_NAMES, avoid_obstacles, shot_camera
from app.services.lowpoly.director import Director, build_actor, flash_value
from app.services.lowpoly.mesh import VERTEX_FLOATS, MeshBuilder
from app.services.lowpoly.narration import Word, estimate_words
from app.services.lowpoly.pipeline import _scene_durations
from app.services.lowpoly.narration import Clip
from app.services.lowpoly.world import SETTINGS, build_world


class TestMesh(unittest.TestCase):
    def test_box_emits_twelve_triangles(self):
        mb = MeshBuilder()
        mb.box((0, 0, 0), (1, 2, 3), (1, 0, 0))
        verts = mb.build()
        self.assertEqual(verts.shape, (36, VERTEX_FLOATS))
        np.testing.assert_allclose(verts[:, 1].min(), -1.0)
        np.testing.assert_allclose(verts[:, 1].max(), 1.0)

    def test_transform_is_applied(self):
        from app.services.lowpoly.mesh import translate

        mb = MeshBuilder()
        with mb.transform(translate(10, 0, 0)):
            mb.box((0, 0, 0), (2, 2, 2), (1, 1, 1))
        self.assertAlmostEqual(float(mb.build()[:, 0].min()), 9.0, places=5)


class TestStoryboard(unittest.TestCase):
    def test_normalizes_unknown_values(self):
        board = storyboard.Storyboard.from_dict(
            {"title": "T", "scenes": [
                {"narration": "I did it.", "setting": "Mountains", "mood": "STORM", "action": "nope", "shot": "dolly-in"},
                {"narration": "Again.", "setting": "x", "mood": "y", "action": "walk", "shot": "dolly_in"},
            ]}
        )
        s0, s1 = board.scenes
        self.assertEqual((s0.setting, s0.mood, s0.action, s0.shot), ("mountains", "storm", "landscape", "dolly_in"))
        self.assertEqual(s1.setting, "town")
        self.assertNotEqual(s1.shot, s0.shot, "consecutive shots must differ")

    def test_llm_json_is_parsed_from_code_fence(self):
        payload = {"title": "Truck town", "description": "d", "hashtags": ["#a", "b"],
                   "scenes": [{"narration": " ".join(["word"] * 16), "setting": "road", "mood": "day",
                               "action": "convoy", "shot": "aerial_orbit"}] * 14}
        fake = "```json\n" + json.dumps(payload) + "\n```"
        with patch("app.services.llm._generate_response", return_value=fake):
            board = storyboard.from_llm("theme", "en")
        self.assertEqual(board.title, "Truck town")
        self.assertEqual(board.hashtags, ["a", "b"])
        self.assertEqual(len(board.scenes), 14)

    def test_llm_failure_falls_back_to_template(self):
        with patch("app.services.llm._generate_response", return_value="Error: no key"):
            board = storyboard.create("building a tower", "de")
        self.assertEqual(board.language, "de")
        self.assertIn("building a tower", board.scenes[0].narration)

    def test_template_is_in_duration_budget(self):
        for lang in ("en", "de"):
            words = storyboard.from_template("x", lang).word_count
            self.assertGreater(words, 150)
            self.assertLess(words, 260)

    def test_prompt_lists_vocabulary(self):
        prompt = storyboard.build_prompt("theme", "de")
        for name in list(SETTINGS) + list(storyboard.ACTIONS) + list(SHOT_NAMES):
            self.assertIn(f'"{name}"', prompt)
        self.assertIn("German", prompt)


class TestCaptions(unittest.TestCase):
    def test_timeline_strips_punctuation_and_bridges_gaps(self):
        words = [Word('"Hello,', 0.0, 0.2), Word("world!", 0.5, 0.8), Word("5", 3.0, 3.2)]
        tl = captions.build_timeline(words)
        self.assertEqual([w.text for w in tl], ["Hello", "world!", "5"])
        self.assertEqual(tl[0].end, 0.5)
        self.assertTrue(tl[1].emphasis and tl[2].emphasis)
        self.assertLess(tl[1].end, 3.0)

    def test_active_word_and_composite(self):
        tl = captions.build_timeline([Word("a", 0, 0.3), Word("b", 0.3, 0.6)])
        word, cursor = captions.active_word(tl, 0.45)
        self.assertEqual(word.text, "b")
        frame = np.zeros((320, 180, 3), dtype=np.uint8)
        out = captions.CaptionRenderer(180, 320).composite(frame, word, 0.6)
        self.assertGreater(int(out.sum()), 0)

    def test_srt_output(self):
        tl = captions.build_timeline([Word(str(i), i * 0.4, i * 0.4 + 0.3) for i in range(6)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.srt")
            captions.to_srt(tl, path)
            text = open(path, encoding="utf-8").read()
        self.assertIn("00:00:00,000 --> ", text)
        self.assertIn("\n2\n", text)


class TestTimingAndDirection(unittest.TestCase):
    def test_estimated_words_cover_span(self):
        words = estimate_words("One two, three.", 0.5, 2.5)
        self.assertEqual(len(words), 3)
        self.assertAlmostEqual(words[0].start, 0.5)
        self.assertLessEqual(words[-1].end, 2.5)

    def test_short_narration_is_padded_to_minimum(self):
        clips = [Clip("x", 4.0) for _ in range(12)]
        durations = _scene_durations(clips, 0.2)
        self.assertGreaterEqual(sum(durations), 75.0)

    def test_mood_blend_endpoints(self):
        a, b = moods.get("day"), moods.get("storm")
        self.assertEqual(moods.blend(a, b, 0.0), a)
        self.assertEqual(moods.blend(a, b, 1.0), b)
        mid = moods.blend(a, b, 0.5)
        self.assertAlmostEqual(mid.rain, 0.5)

    def test_flash_is_bounded(self):
        self.assertEqual(flash_value([], 1.0), 0.0)
        self.assertLessEqual(flash_value([1.0, 1.05], 1.1), 1.0)

    def test_camera_lifts_over_buildings(self):
        obstacles = np.array([[0.0, 0.0, 6.0, 12.0]])
        eye = avoid_obstacles(np.array([0.0, 2.0, 0.0]), np.array([30.0, 1.0, 0.0]), obstacles)
        self.assertGreaterEqual(eye[1], 12.0)

    def test_every_action_and_shot_produces_a_frame_state(self):
        world = build_world(3)
        lib = ActorLibrary(3)
        rng = np.random.default_rng(0)
        for action in storyboard.ACTION_NAMES:
            for setting in SETTINGS:
                actor, scale = build_actor(action, setting, 5.0, 0.4, "day", lib, rng)
                for shot in SHOT_NAMES:
                    cam = shot_camera(shot, 0.5, 2.5, actor.focus, actor.forward, 0.4, scale,
                                      ground=world.ground, obstacles=world.obstacles)
                    self.assertTrue(np.all(np.isfinite(cam.eye)), (action, setting, shot))
                    self.assertGreater(np.linalg.norm(np.subtract(cam.eye, cam.target)), 0.5)
                for draw in actor.draws(2.5):
                    self.assertIn(draw.key, lib.meshes)

    def test_director_timeline(self):
        board = storyboard.from_template("x", "en")
        world = build_world(5)
        director = Director(world, board, [5.0] * len(board.scenes), seed=5)
        self.assertAlmostEqual(director.duration, 5.0 * len(board.scenes))
        self.assertEqual(len(director.cut_times), len(board.scenes) - 1)
        self.assertTrue(director.storm_spans)
        self.assertTrue(director.flashes)
        cam, mood, draws, flash, extent = director.frame(47.3)
        self.assertGreater(len(draws), 3)
        self.assertGreater(extent, 0)


if __name__ == "__main__":
    unittest.main()
