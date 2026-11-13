"""
Tests for Step 8. Expected answers are worked out by hand in the comments.
Most tests hand-build a merge list so the joins are exactly known.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clustering import Merge, cluster, members, AVERAGE, COMPLETE
from grouping import find_groups, cross_links
from pairs import POSSIBLE_LINE, LIKELY_LINE

# Logic tests below were worked out by hand with the first lines (65 / 80).
# They test HOW grouping works, so they pass those lines explicitly; the
# real settings (60 / 65) are checked in TestCurrentLines.
OLD = {"possible": 65, "likely": 80}
from clustering import DEFAULT_LINKAGE


def merges_from(n, joins):
    """joins: list of (a, b, score). New group ids are n, n+1, ... in order."""
    result, sizes = [], {i: 1 for i in range(n)}
    for k, (a, b, s) in enumerate(joins):
        new = n + k
        sizes[new] = sizes[a] + sizes[b]
        result.append(Merge(a, b, new, s, sizes[new]))
    return result


class TestPlanningExample(unittest.TestCase):
    # The cutoff example from planning, with the fixed lines (possible 65, likely 80):
    # texts A=0 B=1 C=2 D=3 E=4 F=5 G=6
    #   A+B 91 -> 7     C+D 86 -> 8     {A,B}+E 80 -> 9     {C,D}+F 66 -> 10
    #   {A,B,E}+{C,D,F} 45 -> 11        G joins 40 -> 12
    # Expected:
    #   Group 1: A, B, E likely (all joined at 80+)
    #   Group 2: C, D likely, F possible (F joined at 66)
    #   G: no group (joined at 40)
    def setUp(self):
        m = merges_from(7, [(0, 1, 91), (2, 3, 86), (7, 4, 80), (8, 5, 66),
                            (9, 10, 45), (11, 6, 40)])
        self.groups, self.ungrouped = find_groups(m, 7, **OLD)

    def test_two_groups(self):
        self.assertEqual(len(self.groups), 2)

    def test_group_1_all_likely(self):
        g = self.groups[0]
        self.assertEqual(g.likely, [[0, 1, 4]])
        self.assertEqual(g.possible, [])

    def test_group_2_possible_member(self):
        g = self.groups[1]
        self.assertEqual(g.likely, [[2, 3]])
        self.assertEqual(g.possible, [5])
        self.assertEqual(g.lowest_join, 66)

    def test_g_ungrouped(self):
        self.assertEqual(self.ungrouped, [6])


class TestLines(unittest.TestCase):
    def test_exactly_65_joins(self):
        m = merges_from(3, [(0, 1, 65), (3, 2, 30)])
        groups, ungrouped = find_groups(m, 3, **OLD)
        self.assertEqual(groups[0].possible, [0, 1])
        self.assertEqual(groups[0].likely, [])
        self.assertEqual(ungrouped, [2])

    def test_just_below_65_does_not(self):
        m = merges_from(3, [(0, 1, 64.99), (3, 2, 30)])
        groups, ungrouped = find_groups(m, 3, **OLD)
        self.assertEqual(groups, [])
        self.assertEqual(ungrouped, [0, 1, 2])

    def test_exactly_80_is_likely(self):
        m = merges_from(3, [(0, 1, 80), (3, 2, 30)])
        groups, _ = find_groups(m, 3, **OLD)
        self.assertEqual(groups[0].likely, [[0, 1]])

    def test_no_groups_when_nothing_matches(self):
        m = merges_from(4, [(0, 1, 50), (2, 3, 48), (4, 5, 30)])
        groups, ungrouped = find_groups(m, 4, **OLD)
        self.assertEqual(groups, [])
        self.assertEqual(ungrouped, [0, 1, 2, 3])

    def test_everything_one_likely_group(self):
        m = merges_from(3, [(0, 1, 95), (3, 2, 90)])
        groups, ungrouped = find_groups(m, 3, **OLD)
        self.assertEqual(groups[0].likely, [[0, 1, 2]])
        self.assertEqual(ungrouped, [])


class TestPossiblyConnected(unittest.TestCase):
    def test_two_likely_subgroups_joined_weakly(self):
        # Decision 2a: {C,D} at 86 and {E,F} at 84 join at 66.
        # Shown as two likely groups, possibly connected.
        m = merges_from(4, [(0, 1, 86), (2, 3, 84), (4, 5, 66)])
        groups, _ = find_groups(m, 4, **OLD)
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g.likely, [[0, 1], [2, 3]])
        self.assertTrue(g.possibly_connected)
        self.assertEqual(g.possible, [])

    def test_single_likely_subgroup_not_marked(self):
        m = merges_from(3, [(0, 1, 86), (3, 2, 70)])
        groups, _ = find_groups(m, 3, **OLD)
        self.assertFalse(groups[0].possibly_connected)


class TestOrder(unittest.TestCase):
    def test_tightest_group_first(self):
        # Sorted by each group's lowest join. (First version sorted by the
        # strongest inner pair, which listed C, D, F (lowest join 66) ahead of
        # A, B, E (all 80+); changed so the tightest whole group comes first.)
        # {0,1} at 70, {2,3} at 95 -> {2,3} listed first
        m = merges_from(4, [(2, 3, 95), (0, 1, 70), (4, 5, 20)])
        groups, _ = find_groups(m, 4, **OLD)
        self.assertEqual(groups[0].all_members, [2, 3])
        self.assertEqual(groups[1].all_members, [0, 1])


class TestRandom(unittest.TestCase):
    """Randomly produced cases: 300 random collections of 3-15 texts."""

    def test_every_text_exactly_once(self):
        rng = random.Random(31)
        for _ in range(300):
            n = rng.randint(3, 15)
            s = [[100.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    s[i][j] = s[j][i] = rng.uniform(0, 100)
            m = cluster(s, rng.choice([AVERAGE, COMPLETE]))
            groups, ungrouped = find_groups(m, n)
            placed = [t for g in groups for t in g.all_members] + ungrouped
            self.assertEqual(sorted(placed), list(range(n)))

    def test_groups_respect_lines(self):
        # Every group came together at 65+; every likely subgroup at 80+.
        rng = random.Random(32)
        for _ in range(300):
            n = rng.randint(3, 15)
            s = [[100.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    s[i][j] = s[j][i] = rng.uniform(40, 100)
            m = cluster(s)
            groups, _ = find_groups(m, n)          # current settings
            for g in groups:
                self.assertGreaterEqual(g.lowest_join, POSSIBLE_LINE)
                self.assertTrue(all(x >= LIKELY_LINE for x in g.likely_scores))
                self.assertGreaterEqual(len(g.all_members), 2)


class P:
    """A minimal stand-in for a Step 6 pair result."""
    def __init__(self, i, j, score):
        self.i, self.j, self.score = i, j, score


def table_and_pairs(n, scores, default=30.0):
    s = [[100.0 if i == j else default for j in range(n)] for i in range(n)]
    for (i, j), v in scores.items():
        s[i][j] = s[j][i] = v
    pairs = [P(i, j, s[i][j]) for i in range(n) for j in range(i + 1, n)]
    return s, pairs


class TestCompleteDefaultAndNotes(unittest.TestCase):
    def test_example_1_split_pair_becomes_note(self):
        # A=0 B=1 C=2: A-B 90, A-C 70, B-C 50
        # Complete: A+B at 90; C joins at min(70, 50) = 50 -> C not grouped.
        # A-C (70) is "possible" but split -> listed as a note.
        s, pairs = table_and_pairs(3, {(0, 1): 90, (0, 2): 70, (1, 2): 50})
        groups, ungrouped = find_groups(cluster(s), 3, **OLD)
        self.assertEqual(groups[0].likely, [[0, 1]])
        self.assertEqual(ungrouped, [2])
        links = cross_links(groups, pairs, OLD['possible'])
        self.assertEqual([(x.a, x.b, x.score) for x in links], [(0, 2, 70)])

    def test_example_2_weak_pair_kept_out(self):
        # A-B 90, A-C 85, B-C 60
        # Average would put C in the group at 72.5 (with B-C only 60).
        # Complete: C joins at min(85, 60) = 60 -> not grouped; A-C (85) is a note.
        s, pairs = table_and_pairs(3, {(0, 1): 90, (0, 2): 85, (1, 2): 60})
        avg_groups, _ = find_groups(cluster(s, AVERAGE), 3, **OLD)
        self.assertEqual(avg_groups[0].possible, [2])            # the old problem
        groups, ungrouped = find_groups(cluster(s), 3, **OLD)
        self.assertEqual(ungrouped, [2])
        self.assertEqual([(x.a, x.b) for x in cross_links(groups, pairs, OLD['possible'])], [(0, 2)])

    def test_pairs_inside_same_group_are_not_notes(self):
        s, pairs = table_and_pairs(3, {(0, 1): 90, (0, 2): 85, (1, 2): 82})
        groups, _ = find_groups(cluster(s), 3, **OLD)
        self.assertEqual(cross_links(groups, pairs, OLD['possible']), [])

    def test_unlikely_pairs_are_not_notes(self):
        s, pairs = table_and_pairs(3, {(0, 1): 90})
        groups, _ = find_groups(cluster(s), 3, **OLD)
        self.assertEqual(cross_links(groups, pairs, OLD['possible']), [])

    def test_random_nothing_is_lost(self):
        # Randomly produced cases: with complete linkage (default), for 1,000
        # random collections:
        #   - no pair under 65 inside a group, no pair under 80 inside a likely subgroup
        #   - every 65+ pair is either inside one group or listed as a note
        rng = random.Random(41)
        for _ in range(1000):
            n = rng.randint(3, 10)
            scores = {(i, j): round(rng.uniform(40, 100), 1)
                      for i in range(n) for j in range(i + 1, n)}
            s, pairs = table_and_pairs(n, scores)
            groups, _ = find_groups(cluster(s), n)          # current settings
            group_of = {t: k for k, g in enumerate(groups) for t in g.all_members}
            noted = {(x.a, x.b) for x in cross_links(groups, pairs)}
            for (i, j), v in scores.items():
                same = i in group_of and group_of.get(i) == group_of.get(j)
                if same:
                    self.assertGreaterEqual(v, POSSIBLE_LINE)
                if any(i in sub and j in sub for g in groups for sub in g.likely):
                    self.assertGreaterEqual(v, LIKELY_LINE)
                if v >= POSSIBLE_LINE:
                    self.assertTrue(same or (i, j) in noted)


class TestCurrentLines(unittest.TestCase):
    """The real settings (60 / 65)."""

    def test_exactly_60_joins_as_possible(self):
        groups, ungrouped = find_groups(merges_from(3, [(0, 1, 60), (3, 2, 30)]), 3)
        self.assertEqual(groups[0].possible, [0, 1])
        self.assertEqual(ungrouped, [2])

    def test_just_below_60_does_not(self):
        groups, _ = find_groups(merges_from(3, [(0, 1, 59.99), (3, 2, 30)]), 3)
        self.assertEqual(groups, [])

    def test_exactly_65_is_likely(self):
        groups, _ = find_groups(merges_from(3, [(0, 1, 65), (3, 2, 30)]), 3)
        self.assertEqual(groups[0].likely, [[0, 1]])


if __name__ == "__main__":
    unittest.main()
