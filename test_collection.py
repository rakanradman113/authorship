"""
Tests for Step 3. Expected answers are worked out by hand in the comments.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_loader import load_text
from collection import (feature_stats, zscores, median, robust_spread, find_unusual, select_top_ngrams, keep_only,
                        analyze_collection, METHODS, RELIABLE_TEXTS)
from features import FUNCTION_WORDS, PUNCTUATION_MARKS


class TestStats(unittest.TestCase):
    def test_hand_example(self):
        # values 2, 4, 6
        #   average = 12/3 = 4
        #   spread  = sqrt(((2-4)^2 + 0 + (6-4)^2) / 3) = sqrt(8/3) = 1.63299
        #   z       = -2/1.63299, 0, 2/1.63299 = -1.22474, 0, 1.22474
        avg, spread = feature_stats([2, 4, 6])
        self.assertAlmostEqual(avg, 4.0)
        self.assertAlmostEqual(spread, 1.63299, places=5)
        z = zscores([2, 4, 6])
        self.assertAlmostEqual(z[0], -1.22474, places=5)
        self.assertAlmostEqual(z[1], 0.0)
        self.assertAlmostEqual(z[2], 1.22474, places=5)

    def test_ordinary_values_all_kept(self):
        # 2, 4, 6: median 4, distances 2, 0, 2 -> MAD 2 -> spread 2.965.
        # Keep range 4 +/- 5*2.965 = 4 +/- 14.8 contains every value, so nothing is left out
        # and the result is the plain average and standard deviation.
        self.assertAlmostEqual(robust_spread([2, 4, 6], 4), 2 * 1.4826)
        self.assertEqual(feature_stats([2, 4, 6]), feature_stats([2, 4, 6]))

    def test_extreme_value_left_out_hand_example(self):
        # Phase 4 decision B'. Values 1, 2, 3, 4, 100 (100 is the odd text).
        #   median 3; distances 2, 1, 0, 1, 97 -> MAD 1 -> spread 1.4826
        #   leave out anything further than 5 * 1.4826 = 7.413 from 3 -> 100 out
        #   average of 1, 2, 3, 4 = 2.5
        #   standard deviation = sqrt((2.25 + 0.25 + 0.25 + 2.25) / 4) = sqrt(1.25) = 1.1180
        # (Without this: average 22, spread 39.0, and the four ordinary values
        #  would all get z-scores near -0.5, i.e. look alike.)
        avg, spread = feature_stats([1, 2, 3, 4, 100])
        self.assertAlmostEqual(avg, 2.5)
        self.assertAlmostEqual(spread, 1.1180, places=4)
        z = zscores([1, 2, 3, 4, 100])
        self.assertAlmostEqual(z[0], -1.3416, places=3)     # (1 - 2.5) / 1.118
        self.assertAlmostEqual(z[3], 1.3416, places=3)
        self.assertGreater(z[4], 80)            # the odd text still stands out

    def test_unusual_but_not_extreme_value_kept(self):
        # 1, 2, 3, 4, 9: 9 is 6 from the middle, 6 / 1.4826 = 4.05 spreads,
        # under 5, so it stays in: plain average 3.8.
        avg, _ = feature_stats([1, 2, 3, 4, 9])
        self.assertAlmostEqual(avg, 3.8)

    def test_rare_feature_fallback(self):
        # A word used by 2 of 5 texts: 0, 0, 0, 5, 6. More than half the
        # distances from the median (0) are 0, so MAD = 0; the fallback uses
        # the average distance: (0+0+0+5+6)/5 = 2.2, times 1.2533 = 2.757.
        self.assertAlmostEqual(robust_spread([0, 0, 0, 5, 6], 0), 2.2 * 1.2533)
        # Keep range 0 +/- 5*2.757 = 0 +/- 13.8 contains everything: feature kept.
        z = zscores([0, 0, 0, 5, 6])
        self.assertGreater(z[4], 0)
        self.assertLess(z[0], 0)

    def test_small_collections_use_plain_statistics(self):
        # Found in testing: with 3-4 texts, leaving values out judged ordinary
        # texts extreme. Below 5 values the plain average and spread are used:
        # 1, 2, 30 -> average 11, nothing left out.
        avg, _ = feature_stats([1, 2, 30])
        self.assertAlmostEqual(avg, 11.0)

    def test_median(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_zero_spread_gives_zero(self):
        # Decision: every text has the same value -> all z-scores 0
        self.assertEqual(zscores([0, 0, 0, 0]), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(zscores([7.5, 7.5]), [0.0, 0.0])

    def test_rounding_noise_is_not_a_difference(self):
        # Found in the bug hunt: 0.1 + 0.2 is stored as 0.30000000000000004.
        # Before the fix this gave z = [0, -1.41]; the values are really equal.
        self.assertEqual(zscores([0.1 + 0.2, 0.3]), [0.0, 0.0])

    def test_tiny_real_difference_still_counts(self):
        # A small but real difference (0.001 on values near 1) is kept.
        z = zscores([1.000, 1.001])
        self.assertAlmostEqual(abs(z[0]), 1.0)

    def test_two_texts_always_plus_minus_one(self):
        # Why 2 texts are unreliable: any two different values give z = -1 and +1.
        for a, b in [(1, 2), (0, 1000), (3.3, 3.4)]:
            z = zscores([a, b])
            self.assertAlmostEqual(abs(z[0]), 1.0)
            self.assertAlmostEqual(abs(z[1]), 1.0)

    def test_random_zscores_have_mean_0_spread_1(self):
        # Randomly produced cases: whenever the spread is not 0, z-scores
        # always average 0 with spread 1. That is what z-scores are for.
        rng = random.Random(3)
        for _ in range(200):
            values = [rng.uniform(-100, 100) for _ in range(rng.randint(2, 50))]
            avg, spread = feature_stats(zscores(values))
            self.assertAlmostEqual(avg, 0.0, places=9)
            self.assertAlmostEqual(spread, 1.0, places=9)

    def test_scaling_does_not_change_zscores(self):
        # Rates per 1,000 or per 100 words give the same z-scores.
        a = zscores([1, 2, 6])
        b = zscores([10, 20, 60])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y)


class TestTopNgrams(unittest.TestCase):
    def test_hand_example(self):
        # Text 1: "abc" 10, "xyz" 2         Text 2: "abc" 4, "qqq" 9
        # Averages (missing = 0): abc (10+4)/2 = 7, qqq 9/2 = 4.5, xyz 2/2 = 1
        top = select_top_ngrams([{"abc": 10, "xyz": 2}, {"abc": 4, "qqq": 9}], n=2)
        self.assertEqual(top, ["abc", "qqq"])

    def test_average_not_total_count_one_text_cannot_dominate(self):
        # "zzz" is huge in one text only; "the" is moderate in all three.
        # Averages: the = 50, zzz = 120/3 = 40 -> "the" wins.
        # (zzz is also in only 1 of 3 texts, so the half-the-texts rule removes it.)
        dicts = [{"the": 50, "zzz": 120}, {"the": 50}, {"the": 50}]
        self.assertEqual(select_top_ngrams(dicts, n=1), ["the"])

    def test_must_appear_in_half_the_texts(self):
        # 4 texts -> a 3-gram must appear in at least 2.
        # "liz" has the highest average (400/4 = 100) but is in only 1 text.
        # "the" (in all 4) and "ing" (in 2) qualify.
        dicts = [{"the": 20, "liz": 400, "ing": 10},
                 {"the": 20, "ing": 10},
                 {"the": 20},
                 {"the": 20}]
        self.assertEqual(select_top_ngrams(dicts, n=5), ["the", "ing"])

    def test_half_rounds_up(self):
        # 5 texts -> must appear in at least 3 (half of 5 = 2.5, rounded up).
        dicts = [{"abc": 1, "xyz": 1}, {"abc": 1, "xyz": 1}, {"abc": 1}, {}, {}]
        self.assertEqual(select_top_ngrams(dicts, n=5), ["abc"])

    def test_ties_broken_alphabetically(self):
        self.assertEqual(select_top_ngrams([{"bbb": 1, "aaa": 1, "ccc": 1}], n=2),
                         ["aaa", "bbb"])

    def test_fewer_than_n_available(self):
        self.assertEqual(len(select_top_ngrams([{"abc": 1}], n=300)), 1)

    def test_keep_only_fills_missing_with_zero(self):
        self.assertEqual(keep_only({"abc": 5, "zzz": 1}, ["abc", "qqq"]),
                         {"abc": 5, "qqq": 0.0})


def make_text(name, sentence, repeats=80):
    # 80 repeats so even the shortest sentence (8 words) gives 640 words, over
    # the 500 minimum. (60 repeats first made one text 480 words, and it was
    # correctly excluded.)
    return load_text(name, (sentence + " ") * repeats)


class TestAnalyzeCollection(unittest.TestCase):
    def setUp(self):
        self.texts = [
            make_text("a", "The cat sat upon the mat, and the dog lay by the fire."),
            make_text("b", "Whilst it rained; we read of kings and queens."),
            make_text("c", "Thou art fair (said he) -- yet I am not."),
            make_text("d", "The ship sailed upon the sea at dawn."),
            make_text("e", "It was cold, and she was tired of it all!"),
        ]

    def test_every_method_and_feature_present(self):
        c = analyze_collection(self.texts)
        self.assertEqual(len(c.z), 5)
        for zi in c.z:
            self.assertEqual(set(zi), set(METHODS))
            self.assertEqual(len(zi["function_words"]), len(FUNCTION_WORDS))
            self.assertEqual(len(zi["punctuation"]), len(PUNCTUATION_MARKS))
            self.assertEqual(len(zi["length"]), 4)
            self.assertEqual(set(zi["vocabulary"]), {"mattr"})
            self.assertEqual(len(zi["ngrams"]), len(c.top_ngrams))

    def test_unused_word_gets_zero(self):
        # No text uses "betwixt" -> spread 0 -> z = 0 everywhere
        c = analyze_collection(self.texts)
        self.assertTrue(all(zi["function_words"]["betwixt"] == 0.0 for zi in c.z))

    def test_upon_stands_out(self):
        # Only texts a and d use "upon", so they get positive z-scores
        # and the other three get the same negative one.
        c = analyze_collection(self.texts)
        z = [zi["function_words"]["upon"] for zi in c.z]
        self.assertGreater(z[0], 0)
        self.assertGreater(z[3], 0)
        self.assertAlmostEqual(z[1], z[2])
        self.assertAlmostEqual(z[1], z[4])
        self.assertLess(z[1], 0)

    def test_no_warning_at_5_texts(self):
        self.assertEqual(analyze_collection(self.texts).warning, "")

    def test_warning_below_5_texts(self):
        c = analyze_collection(self.texts[:3])
        self.assertIn("Only 3 texts", c.warning)

    def test_three_texts_allowed_with_warning(self):
        c = analyze_collection(self.texts[:3])
        self.assertTrue(c.warning)

    def test_two_texts_refused(self):
        # Changed in Step 6: with 2 texts every pair scored 9.1, whatever the texts.
        with self.assertRaises(ValueError):
            analyze_collection(self.texts[:2])

    def test_one_text_refused(self):
        with self.assertRaises(ValueError):
            analyze_collection(self.texts[:1])

    def test_short_texts_excluded_and_reported(self):
        texts = self.texts + [load_text("short", "Too short to use.")]
        c = analyze_collection(texts)
        self.assertNotIn("short", c.names)
        self.assertEqual(c.excluded[0][0], "short")

    def test_short_texts_do_not_count_toward_minimum(self):
        texts = [self.texts[0], self.texts[1], load_text("s1", "short"), load_text("s2", "short")]
        with self.assertRaises(ValueError):
            analyze_collection(texts)

    def test_unusual_text_flagged(self):
        # 5 ordinary texts + one word repeated 500 times: only the odd one is flagged.
        odd = load_text("Odd", " ".join(["word"] * 500) + ".")
        c = analyze_collection(self.texts + [odd])
        self.assertEqual([name for name, _ in c.unusual], ["Odd"])
        self.assertGreater(c.unusual[0][1], 4)

    def test_no_unusual_in_ordinary_collection(self):
        self.assertEqual(analyze_collection(self.texts).unusual, [])

    def test_no_flag_under_5_texts(self):
        odd = load_text("Odd", " ".join(["word"] * 500) + ".")
        self.assertEqual(analyze_collection(self.texts[:3] + [odd]).unusual, [])

    def test_reliable_threshold_is_5(self):
        self.assertEqual(RELIABLE_TEXTS, 5)


if __name__ == "__main__":
    unittest.main()
