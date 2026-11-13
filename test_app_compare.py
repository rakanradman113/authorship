"""
Tests for Step 9, Part 2: the Compare texts page.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as webapp
from text_loader import load_text

SENTENCES = {
    "Ash": "The cat sat upon the mat, and the dog lay by the fire.",
    "Birch": "Whilst it rained; we read of kings and queens.",
    "Cedar": "Thou art fair (said he) -- yet I am not.",
    "Dogwood": "The ship sailed upon the sea at dawn and dusk.",
    "Elm": "It was cold, and she was tired of it all!",
}


def add(name, sentence, repeats=80):
    webapp.texts.append(load_text(name, (sentence + " ") * repeats))


class ComparePageTests(unittest.TestCase):
    def setUp(self):
        webapp.texts.clear()
        webapp._cache.update(key=None, results=None)
        self.client = webapp.app.test_client()

    def page(self):
        return self.client.get("/compare").get_data(as_text=True)

    def load_five_plus_copy(self):
        for k, v in SENTENCES.items():
            add(k, v)
        add("Ash copy", SENTENCES["Ash"])

    def test_too_few_texts(self):
        add("Ash", SENTENCES["Ash"])
        html = self.page()
        self.assertIn("Add at least 3 texts", html)
        self.assertNotIn("<details", html)

    def test_number_of_pairs(self):
        self.load_five_plus_copy()            # 6 texts -> 15 pairs
        html = self.page()
        self.assertEqual(html.count('<details class="pair'), 15)
        self.assertIn("15 pairs", html)

    def test_copy_first_and_likely(self):
        self.load_five_plus_copy()
        html = self.page()
        first = html[html.index('<details class="pair'):]
        first = first[:first.index("</details>")]
        self.assertIn("Ash", first)
        self.assertIn("Ash copy", first)
        self.assertIn("100.0", first)
        self.assertIn("Likely same author", first)

    def test_breakdown_rows_in_fixed_order(self):
        self.load_five_plus_copy()
        html = self.page()
        first = html[:html.index("</details>")]
        order = [first.index(n) for n in ["Function words", "Character 3-grams",
                                          "Sentence/word length", "Punctuation",
                                          "Vocabulary richness"]]
        self.assertEqual(order, sorted(order))

    def test_shown_points_add_up_to_shown_total(self):
        # Step 5 display rule, checked on the actual page for every pair
        self.load_five_plus_copy()
        html = self.page()
        for block in html.split('<details class="pair')[1:]:
            block = block[:block.index("</details>")]
            points = [float(x) for x in re.findall(
                r'<td class="right">([\d.]+)</td>\s*</tr>', block)]
            *rows, total = points
            self.assertAlmostEqual(sum(rows), total, places=6)

    def test_score_times_weight_matches_points(self):
        # Found in the screenshot check: whole-number scores made rows look
        # wrong (60 x 30% shown as 18.1). With one decimal, every row's
        # score x weight is within 0.1 of its shown points.
        self.load_five_plus_copy()
        html = self.page()
        rows = re.findall(r'<td class="right">([\d.]+)</td>\s*<td class="right">([\d.]+)%</td>'
                          r'\s*<td class="right">([\d.]+)</td>', html)
        self.assertTrue(rows)
        for score, weight, points in rows:
            self.assertLessEqual(abs(float(score) * float(weight) / 100 - float(points)), 0.11)

    def test_unlikely_pairs_folded_away(self):
        # Decision: unlikely pairs sit under one "Show unlikely pairs (N)" toggle.
        self.load_five_plus_copy()
        html = self.page()
        n_unlikely = len(re.findall(r'<details class="pair unlikely"', html))
        self.assertIn(f"Show unlikely pairs ({n_unlikely})", html)
        fold = html[html.index('<details class="fold">'):]
        self.assertNotIn('<details class="pair likely"', fold)        # likely stays outside
        self.assertEqual(fold.count('<details class="pair unlikely"'), n_unlikely)

    def test_no_fold_when_nothing_unlikely(self):
        # Three copies: every pair scores 100, nothing unlikely, no toggle.
        # (This test first exposed a bug: identical texts scored 0 and "Unlikely".)
        for k in ["Ash", "Ash 2", "Ash 3"]:
            add(k, SENTENCES["Ash"])
        html = self.page()
        self.assertNotIn("Show unlikely pairs", html)
        self.assertEqual(html.count("Likely same author"), 3)
        self.assertEqual(html.count("100.0<span"), 3)
        self.assertIn("identical in every measured feature", html)

    def test_labels_use_plain_words(self):
        self.load_five_plus_copy()
        html = self.page()
        for word in ["likely", "possible", "unlikely"]:
            self.assertNotIn(f">{word}<", html)      # never the raw internal word
        self.assertIn("Unlikely", html)

    def test_sorted_likely_then_possible_then_unlikely(self):
        self.load_five_plus_copy()
        html = self.page()
        labels = re.findall(r'<details class="pair (\w+)"', html)
        rank = {"likely": 0, "possible": 1, "unlikely": 2}
        self.assertEqual(labels, sorted(labels, key=rank.get))

    def test_short_text_listed_as_left_out(self):
        self.load_five_plus_copy()
        webapp.texts.append(load_text("Postcard", "Wish you were here."))
        html = self.page()
        self.assertIn("Left out (too short): Postcard", html)
        self.assertNotIn("Postcard <span", html)       # not in any pair

    def test_warning_shown_for_3_to_4_texts(self):
        for k in ["Ash", "Birch", "Cedar"]:
            add(k, SENTENCES[k])
        self.assertIn("Only 3 texts are ready", self.page())

    def test_results_update_when_texts_change(self):
        self.load_five_plus_copy()
        self.assertIn("15 pairs", self.page())
        add("Zeta", "Our garden grew roses, lilies and tall grasses.")
        self.assertIn("21 pairs", self.page())        # 7 texts -> 21 pairs

    def test_results_reused_when_nothing_changes(self):
        self.load_five_plus_copy()
        self.page()
        first = webapp._cache["results"]
        self.page()
        self.assertIs(webapp._cache["results"], first)

    def test_no_technical_terms(self):
        # (The test texts were first named Alpha, Beta, Gamma, Delta... and the
        #  text name "Delta" set this off; renamed to trees.)
        self.load_five_plus_copy()
        html = self.page().lower()
        for term in ["delta", "z-score", "linkage", "spread"]:
            self.assertNotIn(term, html, msg=f"technical term on page: {term}")


if __name__ == "__main__":
    unittest.main()
