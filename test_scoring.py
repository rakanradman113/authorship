"""
Tests for Step 4. Expected answers are worked out by hand in the comments.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_loader import load_text
from collection import analyze_collection, METHODS
from scoring import (delta, delta_to_score, method_scores, DELTA_SCALE,
                     overall_score, pair_score, WEIGHTS, display_points)


class TestDelta(unittest.TestCase):
    def test_hand_example(self):
        # A: 1.2, -0.5, 0.3     B: 1.0, -0.2, 0.3
        # |differences| = 0.2, 0.3, 0  -> Delta = 0.5 / 3 = 0.16667
        a = {"x": 1.2, "y": -0.5, "z": 0.3}
        b = {"x": 1.0, "y": -0.2, "z": 0.3}
        self.assertAlmostEqual(delta(a, b), 0.16667, places=5)

    def test_identical_is_zero(self):
        a = {"x": 1.5, "y": -2.0}
        self.assertEqual(delta(a, dict(a)), 0.0)

    def test_only_listed_features(self):
        # Same as the hand example but averaged over x and y only:
        # (0.2 + 0.3) / 2 = 0.25
        a = {"x": 1.2, "y": -0.5, "z": 0.3}
        b = {"x": 1.0, "y": -0.2, "z": 0.3}
        self.assertAlmostEqual(delta(a, b, ["x", "y"]), 0.25)

    def test_no_features_gives_none(self):
        self.assertIsNone(delta({"x": 0.0}, {"x": 0.0}, []))

    def test_order_does_not_matter(self):
        a = {"x": 1.0, "y": 2.0}
        b = {"x": -1.0, "y": 0.5}
        self.assertEqual(delta(a, b), delta(b, a))


class TestScore(unittest.TestCase):
    def test_hand_values(self):
        # 100 * (1 - Delta/2.4)
        self.assertAlmostEqual(delta_to_score(0), 100.0)
        self.assertAlmostEqual(delta_to_score(0.6), 75.0)    # 1 - 0.25
        self.assertAlmostEqual(delta_to_score(1.2), 50.0)    # 1 - 0.5
        self.assertAlmostEqual(delta_to_score(2.4), 0.0)

    def test_never_below_zero(self):
        self.assertEqual(delta_to_score(3.0), 0.0)
        self.assertEqual(delta_to_score(100), 0.0)

    def test_scale_is_2_4(self):
        # Changed from 2.2 in Phase 4 (with extreme values left out of the statistics)
        self.assertEqual(DELTA_SCALE, 2.4)

    def test_random_unrelated_texts_score_about_50(self):
        # Randomly produced cases: give two "texts" random bell-curve z-scores
        # for 434 features (like ours); their Delta should be about 1.13, so
        # their score about 100 * (1 - 1.13/2.4) = 53. Averaged over 300 pairs.
        rng = random.Random(11)
        scores = []
        for _ in range(300):
            a = {f: rng.gauss(0, 1) for f in range(434)}
            b = {f: rng.gauss(0, 1) for f in range(434)}
            scores.append(delta_to_score(delta(a, b)))
        avg = sum(scores) / len(scores)
        self.assertAlmostEqual(avg, 53.0, delta=1.0)


def text(name, sentence, repeats=80):
    return load_text(name, (sentence + " ") * repeats)


class TestMethodScores(unittest.TestCase):
    def setUp(self):
        self.texts = [
            text("a", "The cat sat upon the mat, and the dog lay by the fire."),
            text("a_copy", "The cat sat upon the mat, and the dog lay by the fire."),
            text("b", "Whilst it rained; we read of kings and queens."),
            text("c", "Thou art fair (said he) -- yet I am not."),
            text("d", "The ship sailed upon the sea at dawn and dusk."),
            text("e", "It was cold, and she was tired of it all!"),
        ]
        self.c = analyze_collection(self.texts)

    def test_identical_texts_score_100_on_every_method(self):
        s = method_scores(self.c, 0, 1)
        for m in METHODS:
            self.assertAlmostEqual(s[m]["score"], 100.0)
            self.assertAlmostEqual(s[m]["delta"], 0.0)

    def test_all_five_methods_returned(self):
        self.assertEqual(set(method_scores(self.c, 0, 2)), set(METHODS))

    def test_pair_order_does_not_matter(self):
        self.assertEqual(method_scores(self.c, 2, 3), method_scores(self.c, 3, 2))

    def test_unused_features_left_out(self):
        # Found in the bug hunt: 101 of 119 function words were unused by every
        # text, which pushed every pair's function-word score to 90-97.
        # Only words that vary may be averaged.
        used = self.c.varying_features("function_words")
        self.assertNotIn("betwixt", used)
        self.assertIn("upon", used)
        s = method_scores(self.c, 0, 2)
        self.assertEqual(s["function_words"]["features_used"], len(used))
        self.assertLess(len(used), 119)

    def test_unused_features_no_longer_inflate(self):
        # a and b share few function words; in the bug-hunt collection they
        # scored 94 before the fix (60 after).
        s = method_scores(self.c, 0, 2)
        self.assertLess(s["function_words"]["score"], 80)

    def test_scores_between_0_and_100(self):
        n = len(self.c.names)
        for i in range(n):
            for j in range(n):
                for m, v in method_scores(self.c, i, j).items():
                    self.assertGreaterEqual(v["score"], 0.0)
                    self.assertLessEqual(v["score"], 100.0)

    def test_copy_scores_higher_than_different_text(self):
        same = method_scores(self.c, 0, 1)
        diff = method_scores(self.c, 0, 2)
        for m in METHODS:
            self.assertGreater(same[m]["score"], diff[m]["score"])


def per_method(**scores):
    return {m: {"score": s, "features_used": 1} for m, s in scores.items()}


class TestOverallScore(unittest.TestCase):
    def test_weights_add_to_100(self):
        self.assertEqual(sum(WEIGHTS.values()), 100)
        # Phase 4 decision B (first 30 / 20 / 20 / 17 / 13)
        self.assertEqual(WEIGHTS, {"function_words": 35, "ngrams": 35, "length": 10,
                                   "punctuation": 10, "vocabulary": 10})

    def test_hand_example(self):
        # 90*35% + 80*35% + 70*10% + 65*10% + 75*10%
        # = 31.5 + 28 + 7 + 6.5 + 7.5 = 80.5
        score, rows = overall_score(per_method(function_words=90, ngrams=80, length=70,
                                               punctuation=65, vocabulary=75))
        self.assertAlmostEqual(score, 80.5)
        points = {r["method"]: r["points"] for r in rows}
        self.assertAlmostEqual(points["function_words"], 31.5)
        self.assertAlmostEqual(points["punctuation"], 6.5)

    def test_breakdown_adds_up_to_score(self):
        score, rows = overall_score(per_method(function_words=12, ngrams=99, length=3,
                                               punctuation=50, vocabulary=71))
        self.assertAlmostEqual(sum(r["points"] for r in rows), score)

    def test_breakdown_same_order_for_every_pair(self):
        # Symmetry: rows always in weight order, largest first
        _, rows = overall_score(per_method(function_words=1, ngrams=2, length=3,
                                           punctuation=4, vocabulary=5))
        self.assertEqual([r["method"] for r in rows],
                         ["function_words", "ngrams", "length", "punctuation", "vocabulary"])

    def test_left_out_method_weight_is_shared(self):
        # Decision b: vocabulary has no score -> other weights add to 90.
        #   function words 35/90 = 38.89%, punctuation 10/90 = 11.11%
        #   score = (90*35 + 80*35 + 70*10 + 65*10) / 90 = 7300 / 90 = 81.111
        score, rows = overall_score(per_method(function_words=90, ngrams=80, length=70,
                                               punctuation=65, vocabulary=None))
        self.assertAlmostEqual(score, 81.111, places=3)
        w = {r["method"]: r["weight"] for r in rows}
        self.assertAlmostEqual(w["function_words"], 38.889, places=3)
        self.assertAlmostEqual(w["vocabulary"], 0.0)
        self.assertAlmostEqual(sum(w.values()), 100.0)

    def test_left_out_method_still_shown_with_note(self):
        _, rows = overall_score(per_method(function_words=90, ngrams=80, length=70,
                                           punctuation=65, vocabulary=None))
        vocab = [r for r in rows if r["method"] == "vocabulary"][0]
        self.assertIn("Left out", vocab["note"])

    def test_all_equal_scores_give_that_score(self):
        score, _ = overall_score(per_method(function_words=60, ngrams=60, length=60,
                                            punctuation=60, vocabulary=60))
        self.assertAlmostEqual(score, 60.0)

    def test_no_method_scored_means_identical(self):
        # Decision (found in Step 9): if no feature varies anywhere, every text
        # is identical in every measured feature -> perfect match, 100.
        score, rows = overall_score(per_method(function_words=None, ngrams=None, length=None,
                                               punctuation=None, vocabulary=None))
        self.assertEqual(score, 100.0)
        self.assertEqual([r["points"] for r in rows], [35.0, 35.0, 10.0, 10.0, 10.0])
        self.assertTrue(all(r["identical"] for r in rows))

    def test_random_score_always_between_min_and_max(self):
        # Randomly produced cases: a weighted average can never be below the
        # lowest method score or above the highest.
        rng = random.Random(5)
        for _ in range(300):
            vals = {m: (None if rng.random() < 0.2 else rng.uniform(0, 100)) for m in WEIGHTS}
            score, rows = overall_score(per_method(**vals))
            present = [v for v in vals.values() if v is not None]
            if not present:
                self.assertEqual(score, 100.0)
                continue
            self.assertGreaterEqual(score, min(present) - 1e-9)
            self.assertLessEqual(score, max(present) + 1e-9)
            self.assertAlmostEqual(sum(r["weight"] for r in rows), 100.0)

    def test_pair_score_on_real_collection(self):
        texts = [text("a", "The cat sat upon the mat, and the dog lay by the fire."),
                 text("a_copy", "The cat sat upon the mat, and the dog lay by the fire."),
                 text("b", "Whilst it rained; we read of kings and queens."),
                 text("c", "Thou art fair (said he) -- yet I am not."),
                 text("d", "The ship sailed upon the sea at dawn and dusk.")]
        c = analyze_collection(texts)
        same, _ = pair_score(c, 0, 1)
        diff, _ = pair_score(c, 0, 2)
        self.assertAlmostEqual(same, 100.0)
        self.assertLess(diff, same)


class TestDisplayPoints(unittest.TestCase):
    def test_hand_example(self):
        # Weights 35/35/10/10/10. Scores 90, 80, 70, 65, 75 give exact points
        # 31.5, 28, 7, 6.5, 7.5 = 80.5; all already one decimal, so the shown
        # points are the exact ones and add up to 80.5.
        score, rows = overall_score(per_method(function_words=90, ngrams=80, length=70,
                                               punctuation=65, vocabulary=75))
        total, shown = display_points(score, rows)
        self.assertEqual(total, 80.5)
        self.assertEqual(shown, [31.5, 28.0, 7.0, 6.5, 7.5])

    def test_rounding_needs_adjusting(self):
        # Scores 61, 47, 33, 29, 81: exact points 21.35, 16.45, 3.3, 2.9, 8.1
        # = 52.1. Rounding each down: 21.3 + 16.4 + 3.3 + 2.9 + 8.1 = 52.0, so
        # one step (0.1) is missing; it goes to a row that lost 0.05.
        score, rows = overall_score(per_method(function_words=61, ngrams=47, length=33,
                                               punctuation=29, vocabulary=81))
        total, shown = display_points(score, rows)
        self.assertEqual(total, 52.1)
        self.assertAlmostEqual(sum(shown), 52.1)

    def test_random_always_adds_up(self):
        # Randomly produced cases: 1,000 random breakdowns. The shown points
        # must add up exactly to the shown total, and each must be within
        # 0.1 of its exact value.
        rng = random.Random(9)
        for _ in range(1000):
            vals = {m: (None if rng.random() < 0.15 else rng.uniform(0, 100)) for m in WEIGHTS}
            score, rows = overall_score(per_method(**vals))
            if score is None:
                continue
            total, shown = display_points(score, rows)
            self.assertAlmostEqual(sum(shown), total, places=9)
            self.assertAlmostEqual(total, round(score, 1), places=9)
            for exact, s in zip((r["points"] for r in rows), shown):
                self.assertLess(abs(exact - s), 0.1 + 1e-9)

    def test_left_out_method_shows_zero(self):
        score, rows = overall_score(per_method(function_words=90, ngrams=80, length=70,
                                               punctuation=65, vocabulary=None))
        _, shown = display_points(score, rows)
        self.assertEqual(shown[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
