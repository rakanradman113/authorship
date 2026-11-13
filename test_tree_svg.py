"""
Tests for Step 9, Part 4: the tree picture.
"""

import os
import random
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clustering import Merge, cluster, members
from tree_svg import (leaf_order, x_for, colour, short, tree_svg, GREEN, AMBER, GREY,
                      LABEL_W, PLOT_W, LEAF_STUB)


def merges_from(n, joins):
    result, sizes = [], {i: 1 for i in range(n)}
    for k, (a, b, s) in enumerate(joins):
        new = n + k
        sizes[new] = sizes[a] + sizes[b]
        result.append(Merge(a, b, new, s, sizes[new]))
    return result


class TestLayout(unittest.TestCase):
    def test_x_positions(self):
        # 100 at the left edge of the plot, 0 at the right, 50 halfway.
        self.assertEqual(x_for(100), LABEL_W)
        self.assertEqual(x_for(0), LABEL_W + PLOT_W)
        self.assertEqual(x_for(50), LABEL_W + PLOT_W / 2)

    def test_colours(self):
        # Lines 60 / 65 (Phase 4 decision)
        self.assertEqual(colour(65), GREEN)
        self.assertEqual(colour(64.9), AMBER)
        self.assertEqual(colour(60), AMBER)
        self.assertEqual(colour(59.9), GREY)

    def test_leaf_order_hand_example(self):
        # A=0 B=1 C=2 D=3: A+C 90, B+D 85, then the two groups.
        # Groups must sit together: A, C, B, D (not A, B, C, D).
        m = merges_from(4, [(0, 2, 90), (1, 3, 85), (4, 5, 40)])
        self.assertEqual(leaf_order(m, 4), [0, 2, 1, 3])

    def test_random_groups_always_side_by_side(self):
        # Randomly produced cases: in 300 random trees, every group's texts
        # sit in one unbroken run, so no branches cross.
        rng = random.Random(51)
        for _ in range(300):
            n = rng.randint(3, 15)
            s = [[100.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    s[i][j] = s[j][i] = rng.uniform(0, 100)
            m = cluster(s)
            order = leaf_order(m, n)
            self.assertEqual(sorted(order), list(range(n)))
            pos = {t: k for k, t in enumerate(order)}
            for x in m:
                spots = sorted(pos[t] for t in members(m, n, x.new))
                self.assertEqual(spots, list(range(spots[0], spots[0] + len(spots))))

    def test_short_titles(self):
        self.assertEqual(short("Letter"), "Letter")
        long = "A very long title that goes on and on and on and on"
        self.assertTrue(short(long).endswith("…"))
        self.assertLessEqual(len(short(long)), 40)


class TestPicture(unittest.TestCase):
    def setUp(self):
        # The planning example: A+B 91, C+D 86, {A,B}+E 80, {C,D}+F 66, ... G 40
        self.m = merges_from(7, [(0, 1, 91), (2, 3, 86), (7, 4, 80), (8, 5, 66),
                                 (9, 10, 45), (11, 6, 40)])
        self.names = list("ABCDEFG")
        self.svg = tree_svg(self.m, self.names)

    def test_one_bracket_per_join(self):
        # 7 texts -> 6 joins -> 6 brackets of 3 lines each
        self.assertEqual(self.svg.count('<g class="join"'), 6)
        self.assertEqual(len(re.findall(r'<g class="join".*?</g>', self.svg)), 6)
        for g in re.findall(r'<g class="join".*?</g>', self.svg):
            self.assertEqual(g.count("<line"), 3)

    def test_every_title_once(self):
        self.assertEqual(self.svg.count('class="leaf"'), 7)

    def test_join_at_right_place_and_colour(self):
        # The 91 join: vertical line at x_for(91), green
        x = f"{x_for(91):.1f}"
        bracket = re.findall(r'<g class="join".*?</g>', self.svg)[0]
        self.assertIn(f'x2="{x}"', bracket)
        self.assertIn(GREEN, bracket)
        self.assertIn("Joined at 91.0 / 100", bracket)
        self.assertIn(GREEN, re.findall(r'<g class="join".*?</g>', self.svg)[3])   # 66 >= 65
        self.assertIn(GREY, re.findall(r'<g class="join".*?</g>', self.svg)[5])    # 40

    def test_line_labels_follow_settings(self):
        # Were typed in as "Likely (80)" / "Possibly (65)" at first; now read
        # from the settings so they cannot go stale.
        self.assertIn("Likely (65)", self.svg)
        self.assertIn("Possibly (60)", self.svg)
        self.assertNotIn("(80)", self.svg)
        self.assertEqual(self.svg.count('class="cut"'), 2)

    def test_line_labels_at_different_heights(self):
        # Decision a: the labels overlapped when the lines were 60 / 65.
        ys = {m.group(2): float(m.group(1)) for m in
              re.finditer(r'<text x="[\d.]+" y="([\d.]+)"[^>]*>(Likely|Possibly) \(', self.svg)}
        self.assertEqual(set(ys), {"Likely", "Possibly"})
        self.assertGreaterEqual(abs(ys["Likely"] - ys["Possibly"]), 14)   # font is 11 high

    def test_join_at_100_is_visible(self):
        # Found in the screenshot check: identical texts join at 100, where the
        # titles also started, so the bracket had arms of length 0.
        svg = tree_svg(merges_from(3, [(0, 1, 100), (3, 2, 50)]), ["A", "B", "C"])
        bracket = re.findall(r'<g class="join".*?</g>', svg)[0]
        arms = re.findall(r'<line x1="([\d.]+)" y1="[\d.]+" x2="([\d.]+)"', bracket)[:2]
        for x1, x2 in arms:
            self.assertAlmostEqual(float(x2) - float(x1), LEAF_STUB, places=1)

    def test_titles_are_escaped(self):
        svg = tree_svg(merges_from(3, [(0, 1, 90), (3, 2, 50)]),
                       ["<script>x</script>", "Tom & Jerry", "C"])
        self.assertNotIn("<script>x</script>", svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("Tom &amp; Jerry", svg)

    def test_long_title_full_on_hover(self):
        long = "Letter to the honourable committee of the exhibition society, 1887"
        svg = tree_svg(merges_from(3, [(0, 1, 90), (3, 2, 50)]), [long, "B", "C"])
        self.assertIn(f"<title>{long}</title>", svg)
        self.assertIn("…", svg)


if __name__ == "__main__":
    unittest.main()
