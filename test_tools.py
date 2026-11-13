"""
Tests for Step 10: the random-text generator and the evaluation tool.
A testing tool with a bug would give misleading results, so they are tested too.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import true_author, pair_counts, group_counts, evaluate
from generator import generate, write_folder
from grouping import Group
from text_loader import load_folder, load_text, MIN_WORDS


class P:
    def __init__(self, i, j, score):
        self.i, self.j, self.score = i, j, score


class TestGenerator(unittest.TestCase):
    def test_counts_and_names(self):
        items = generate(5, 3, seed=1)
        self.assertEqual(len(items), 15)
        self.assertEqual(items[0][:2], ("Author 1", "Text 1"))
        self.assertEqual(items[-1][:2], ("Author 5", "Text 3"))

    def test_same_seed_same_texts(self):
        self.assertEqual(generate(2, 2, seed=7), generate(2, 2, seed=7))
        self.assertNotEqual(generate(2, 2, seed=7), generate(2, 2, seed=8))

    def test_texts_long_enough_to_use(self):
        for _, _, body in generate(5, 3, seed=2):
            self.assertGreaterEqual(load_text("t", body).word_count, MIN_WORDS)

    def test_writes_folder_with_author_in_name(self):
        with tempfile.TemporaryDirectory() as d:
            write_folder(d, generate(2, 2, seed=3))
            names = sorted(os.listdir(d))
            self.assertEqual(names[0], "Author 1 - Text 1.txt")
            self.assertEqual(len(load_folder(d)), 4)


class TestEvaluationCounts(unittest.TestCase):
    def test_true_author(self):
        self.assertEqual(true_author("Jane Austen - Emma"), "Jane Austen")
        self.assertEqual(true_author("Author 3 - Text 2"), "Author 3")

    def test_pair_counts_hand_example(self):
        # Lines 60 / 65. Authors: A A B. Pairs: (0,1) same 85 -> likely;
        # (0,2) diff 62 -> wrongly possible; (1,2) diff 40 -> correct.
        authors = ["A", "A", "B"]
        c = pair_counts([P(0, 1, 85), P(0, 2, 62), P(1, 2, 40)], authors)
        self.assertEqual(c, {"same_total": 1, "same_likely": 1, "same_possible": 0,
                             "same_missed": 0, "diff_total": 2, "diff_likely": 0,
                             "diff_possible": 1})

    def test_pair_counts_other_lines(self):
        authors = ["A", "A"]
        self.assertEqual(pair_counts([P(0, 1, 57)], authors)["same_missed"], 1)
        self.assertEqual(pair_counts([P(0, 1, 57)], authors, possible=55)["same_possible"], 1)

    def test_group_counts_hand_example(self):
        # Authors A A A B. One group {0, 1, 3}: 0-1 together (correct),
        # 0-2 and 1-2 split, 0-3 and 1-3 false.
        authors = ["A", "A", "A", "B"]
        g = Group(likely=[[0, 1]], possible=[3], lowest_join=70)
        self.assertEqual(group_counts([g], authors), {"false": 2, "split": 2, "together": 1})

    def test_no_groups(self):
        self.assertEqual(group_counts([], ["A", "A", "B"]), {"false": 0, "split": 1, "together": 0})


class TestEvaluationEndToEnd(unittest.TestCase):
    def test_copies_are_perfect(self):
        # 3 authors x 2 identical copies + 1 more text each: copies must be likely.
        items = generate(3, 1, seed=4)
        texts = []
        for a, t, body in items:
            texts.append(load_text(f"{a} - {t}", body))
            texts.append(load_text(f"{a} - {t} copy", body))
        r = evaluate(texts)
        self.assertEqual(r["pair_counts"]["same_likely"], 3)
        self.assertEqual(r["groups"]["complete"]["split"], 0)

    def test_easy_generated_set_mostly_right(self):
        # Easy setting: authors differ strongly. No different-author pair should
        # be called likely, and most same-author pairs should be found.
        items = generate(5, 3, seed=1, difference=0.8)
        r = evaluate([load_text(f"{a} - {t}", b) for a, t, b in items])
        c = r["pair_counts"]
        self.assertEqual(c["diff_likely"], 0)
        self.assertGreaterEqual(c["same_likely"] + c["same_possible"], 12)   # of 15

    def test_one_odd_text_does_not_create_false_matches(self):
        # Found in Phase 4 with the edge-case samples: one extremely unusual
        # text (a word repeated 500 times) made 5 different-author pairs reach
        # "possible". With extreme values left out (decision B') none should.
        # (Option B, clipping at 3 spreads, still let one pair reach 65.4 in
        #  this small collection; this test is why B was changed to B'.)
        items = generate(4, 2, seed=5, difference=0.6)
        texts = [load_text(f"{a} - {t}", b) for a, t, b in items]
        texts.append(load_text("Odd - Boundary", " ".join(["word"] * 500) + "."))
        c = evaluate(texts)
        wrong = [p for p in c["pairs"] if "Odd" not in p.name_a + p.name_b
                 and true_author(p.name_a) != true_author(p.name_b) and p.score >= 65]
        self.assertEqual(wrong, [])

    def test_line_table_includes_current(self):
        items = generate(3, 2, seed=5)
        r = evaluate([load_text(f"{a} - {t}", b) for a, t, b in items])
        self.assertIn((60, 65), [(row["possible"], row["likely"]) for row in r["lines"]])


if __name__ == "__main__":
    unittest.main()
