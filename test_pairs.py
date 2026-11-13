"""
Tests for Step 6. Expected answers are worked out by hand in the comments.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_loader import load_text
from collection import analyze_collection
from pairs import (all_pairs, label, LIKELY, POSSIBLE, UNLIKELY,
                   POSSIBLE_LINE, LIKELY_LINE)


class TestLabels(unittest.TestCase):
    def test_lines(self):
        # Decision after the Phase 4 real-book test (option A); first 65 / 80.
        self.assertEqual((POSSIBLE_LINE, LIKELY_LINE), (60.0, 65.0))

    def test_hand_examples(self):
        self.assertEqual(label(92), LIKELY)
        self.assertEqual(label(65), LIKELY)        # exactly on the line counts
        self.assertEqual(label(64.99), POSSIBLE)
        self.assertEqual(label(60), POSSIBLE)
        self.assertEqual(label(59.99), UNLIKELY)   # a hair below the line
        self.assertEqual(label(30), UNLIKELY)

    def test_missing_score_is_unlikely(self):
        self.assertEqual(label(None), UNLIKELY)


def text(name, sentence, repeats=80):
    return load_text(name, (sentence + " ") * repeats)


SENTENCES = {
    "a": "The cat sat upon the mat, and the dog lay by the fire.",
    "b": "Whilst it rained; we read of kings and queens.",
    "c": "Thou art fair (said he) -- yet I am not.",
    "d": "The ship sailed upon the sea at dawn and dusk.",
    "e": "It was cold, and she was tired of it all!",
    "f": "Our garden grew roses, lilies and tall grasses.",
}


class TestAllPairs(unittest.TestCase):
    def setUp(self):
        texts = [text(k, v) for k, v in SENTENCES.items()]
        texts.append(text("a_copy", SENTENCES["a"]))
        self.c = analyze_collection(texts)
        self.pairs = all_pairs(self.c)

    def test_number_of_pairs(self):
        # 7 texts -> 7*6/2 = 21 pairs
        self.assertEqual(len(self.pairs), 21)

    def test_each_pair_once_no_self_pairs(self):
        seen = {(p.i, p.j) for p in self.pairs}
        self.assertEqual(len(seen), 21)
        self.assertTrue(all(p.i < p.j for p in self.pairs))

    def test_copy_is_likely(self):
        copy = [p for p in self.pairs if {p.name_a, p.name_b} == {"a", "a_copy"}][0]
        self.assertAlmostEqual(copy.score, 100.0)
        self.assertEqual(copy.label, LIKELY)

    def test_labels_match_lines(self):
        for p in self.pairs:
            self.assertEqual(p.label, label(p.score))

    def test_copy_is_the_strongest_match(self):
        # The copy pair scores 100, higher than every other pair. (With the
        # 60 / 65 lines, some unrelated pairs of these one-sentence test texts
        # also reach "likely" at about 66; that is a limit of the test texts,
        # which repeat one sentence 80 times. Real books: see Phase 4 notes.)
        top = max(self.pairs, key=lambda p: p.score)
        self.assertEqual({top.name_a, top.name_b}, {"a", "a_copy"})
        self.assertTrue(all(p.score < 100 for p in self.pairs if p is not top))


class TestRandomPairCounts(unittest.TestCase):
    def test_pair_count_formula(self):
        # Randomly produced cases: for n texts, always n(n-1)/2 pairs.
        rng = random.Random(4)
        words = ("the of and upon whilst thou cat ship rain garden rose sea "
                 "fire king queen cold tired dawn").split()
        for _ in range(5):
            n = rng.randint(3, 8)
            texts = [load_text(f"t{k}", " ".join(rng.choice(words) for _ in range(600)) + ".")
                     for k in range(n)]
            pairs = all_pairs(analyze_collection(texts))
            self.assertEqual(len(pairs), n * (n - 1) // 2)


class TestManyMatches(unittest.TestCase):
    """Found in the bug hunt: with relative lines, collections with many real
    matches pushed the likely line above 100, so copies were only "possible"."""

    def check(self, spec):
        texts = [text(f"{k}{n}", SENTENCES[k]) for n, k in enumerate(spec)]
        pairs = all_pairs(analyze_collection(texts))
        return [p for p in pairs if p.name_a[0] == p.name_b[0]]

    def test_three_texts_one_match(self):
        for p in self.check(["a", "a", "b"]):
            self.assertEqual(p.label, LIKELY)

    def test_two_authors_three_texts_each(self):
        matches = self.check(["a", "a", "a", "b", "b", "b"])
        self.assertEqual(len(matches), 6)
        for p in matches:
            self.assertEqual(p.label, LIKELY)


if __name__ == "__main__":
    unittest.main()
