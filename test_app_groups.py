"""
Tests for Step 9, Part 3: the Authorship groups page.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as webapp
from grouping import Group, CrossLink
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


class GroupsPageTests(unittest.TestCase):
    def setUp(self):
        webapp.texts.clear()
        webapp._cache.update(key=None, results=None)
        self.client = webapp.app.test_client()

    def page(self):
        return self.client.get("/groups").get_data(as_text=True)

    def test_too_few_texts(self):
        add("Ash", SENTENCES["Ash"])
        html = self.page()
        self.assertIn("Add at least 3 texts", html)
        self.assertNotIn("Group 1", html)

    def test_two_pairs_of_copies_make_two_groups(self):
        # Ash + copy and Birch + copy are exact matches (100), so each pair is
        # a likely group. (This test first assumed the other texts would all
        # stand alone, but Dogwood scores 68+ with both Birch texts, so it
        # correctly joins Birch's group as "possible". The checks below follow
        # the real pair scores instead of guesses.)
        for k in SENTENCES:
            add(k, SENTENCES[k])
        add("Ash copy", SENTENCES["Ash"])
        add("Birch copy", SENTENCES["Birch"])
        html = self.page()
        self.assertIn("2 groups", html)
        g1 = html[html.index("Group 1"):html.index("Group 2")]
        g2 = html[html.index("Group 2"):html.index("Not in any group")]
        self.assertIn("<li>Ash</li><li>Ash copy</li>", g1)
        self.assertIn("<li>Birch</li><li>Birch copy</li>", g2)
        self.assertIn("Likely same author", g1)
        self.assertIn("Likely same author", g2)

    def test_page_agrees_with_pair_scores(self):
        # Every pair inside a group, as shown on the page, must score at least
        # the group's stated lowest score, and at least 65.
        for k in SENTENCES:
            add(k, SENTENCES[k])
        add("Ash copy", SENTENCES["Ash"])
        add("Birch copy", SENTENCES["Birch"])
        self.page()
        r = webapp._cache["results"]
        names = r["collection"].names
        score = {frozenset((p.name_a, p.name_b)): p.score for p in r["pairs"]}
        groups, _, _ = webapp.groups_view(r["groups"], r["ungrouped"], r["links"], names)
        for g in groups:
            members = [n for s in g["sets"] for n in s] + g["possible"]
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    self.assertGreaterEqual(round(score[frozenset((a, b))], 1), g["lowest"])
                    self.assertGreaterEqual(score[frozenset((a, b))], 65)

    def test_no_groups_message(self):
        # Rendered from an empty result (with these one-sentence test texts some
        # unrelated pairs reach the mid-60s, so a no-group collection is easier
        # to build by hand).
        with webapp.app.test_request_context():
            html = webapp.render_template("groups.html", groups=[], alone=["A", "B", "C"],
                                          notes=[], level="ok", message="", ready=True,
                                          excluded=[])
        self.assertIn("No texts in this collection are similar enough to group", html)
        self.assertIn("0 groups", html)
        self.assertIn("3 texts not in any group", html)

    def test_short_texts_left_out(self):
        for k in SENTENCES:
            add(k, SENTENCES[k])
        webapp.texts.append(load_text("Postcard", "Wish you were here."))
        self.assertIn("Left out (too short): Postcard", self.page())

    def test_nav_links_all_active(self):
        add("Ash", SENTENCES["Ash"])
        html = self.page()
        for path in ['href="/"', 'href="/compare"', 'href="/groups"']:
            self.assertIn(path, html)
        self.assertNotIn("nav-later", html)

    def test_no_technical_terms(self):
        for k in SENTENCES:
            add(k, SENTENCES[k])
        add("Ash copy", SENTENCES["Ash"])
        html = self.page().lower()
        for term in ["delta", "z-score", "linkage", "spread", "cluster", "merge"]:
            self.assertNotIn(term, html, msg=term)


class GroupsViewTests(unittest.TestCase):
    """The page's wording for hand-built groups (exact cases are hard to
    produce from real texts)."""
    names = ["A", "B", "C", "D", "E", "F", "G"]

    def test_possible_member(self):
        g = Group(likely=[[2, 3]], possible=[5], lowest_join=66.04, likely_scores=[86])
        groups, alone, _ = webapp.groups_view([g], [6], [], self.names)
        self.assertEqual(groups[0]["sets"], [["C", "D"]])
        self.assertEqual(groups[0]["possible"], ["F"])
        self.assertEqual(groups[0]["lowest"], 66.0)
        self.assertFalse(groups[0]["connected"])
        self.assertEqual(alone, ["G"])

    def test_possibly_connected_sets(self):
        g = Group(likely=[[0, 1], [2, 3]], possible=[], lowest_join=66, likely_scores=[86, 84])
        groups, _, _ = webapp.groups_view([g], [], [], self.names)
        self.assertTrue(groups[0]["connected"])
        self.assertEqual(groups[0]["sets"], [["A", "B"], ["C", "D"]])

    def test_notes_wording(self):
        links = [CrossLink(0, 2, 84.3), CrossLink(1, 4, 62.0)]   # lines 60 / 65
        _, _, notes = webapp.groups_view([], [], links, self.names)
        self.assertEqual(notes[0], {"a": "A", "b": "C", "score": 84.3, "words": "Likely same author"})
        self.assertEqual(notes[1]["words"], "Possibly same author")

    def test_connected_and_possible_render(self):
        # Render a hand-built result through the real template.
        g1 = Group(likely=[[0, 1], [2, 3]], possible=[4], lowest_join=66, likely_scores=[86, 84])
        groups, alone, notes = webapp.groups_view([g1], [5, 6], [CrossLink(4, 5, 62.0)], self.names)
        with webapp.app.test_request_context():
            html = webapp.render_template("groups.html", groups=groups, alone=alone, notes=notes,
                                          level="ok", message="", ready=True, excluded=[])
        self.assertIn("These 2 sets are possibly by the same author too.", html)
        self.assertEqual(html.count('<span class="tag likely">'), 2)
        self.assertIn('<span class="tag possible">', html)
        self.assertIn("at least 66.0 / 100", html)
        self.assertIn("Also worth checking", html)
        self.assertIn("62.0 / 100 · Possibly same author", html)


if __name__ == "__main__":
    unittest.main()
