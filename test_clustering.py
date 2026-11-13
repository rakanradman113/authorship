"""
Tests for Step 7. Expected answers are worked out by hand in the comments.
"""

import os
import random
import sys
import unittest
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clustering import cluster, members, similarity_matrix, AVERAGE, COMPLETE, DEFAULT_LINKAGE


def matrix(n, scores, default=30.0):
    """Build a similarity table; `scores` maps (i, j) -> score."""
    s = [[100.0 if i == j else default for j in range(n)] for i in range(n)]
    for (i, j), v in scores.items():
        s[i][j] = s[j][i] = v
    return s


# The example from planning: A=0 B=1 C=2 D=3 E=4
EXAMPLE = matrix(5, {(0, 1): 90, (3, 4): 85, (0, 2): 80, (1, 2): 75, (2, 3): 70})


class TestHandExample(unittest.TestCase):
    def test_average(self):
        # 1. A+B at 90 (highest)                      -> group 5
        # 2. Now {A,B}-C = (80+75)/2 = 77.5, D-E = 85  -> D+E at 85, group 6
        # 3. {A,B}+C at 77.5                           -> group 7
        # 4. {A,B,C}-{D,E} = average of A-D 30, A-E 30, B-D 30, B-E 30,
        #    C-D 70, C-E 30 = 220/6 = 36.667            -> group 8
        m = cluster(EXAMPLE, AVERAGE)
        self.assertEqual([(x.a, x.b) for x in m], [(0, 1), (3, 4), (2, 5), (6, 7)])
        self.assertEqual([round(x.score, 3) for x in m], [90, 85, 77.5, 36.667])
        self.assertEqual([x.size for x in m], [2, 2, 3, 5])

    def test_complete(self):
        # Same first two joins. {A,B}-C = lowest of 80, 75 = 75.
        # Last join = lowest of all 6 pairs between the groups = 30.
        m = cluster(EXAMPLE, COMPLETE)
        self.assertEqual([round(x.score, 3) for x in m], [90, 85, 75, 30])

    def test_members(self):
        m = cluster(EXAMPLE, AVERAGE)
        self.assertEqual(members(m, 5, 7), [0, 1, 2])      # A, B, C
        self.assertEqual(members(m, 5, 6), [3, 4])         # D, E
        self.assertEqual(members(m, 5, 8), [0, 1, 2, 3, 4])
        self.assertEqual(members(m, 5, 3), [3])

    def test_default_is_complete(self):
        # Decision in Step 8: complete linkage never puts a weak pair in a group.
        self.assertEqual(DEFAULT_LINKAGE, COMPLETE)


class TestLinkageDifference(unittest.TestCase):
    def test_complete_refuses_outsider_average_accepts(self):
        # Example 2 from planning: A, B, C one author; X matches A only.
        #   A-B 90, A-C 85, B-C 85; X-A 85, X-B 50, X-C 50
        # X joining {A,B,C}: average = (85+50+50)/3 = 61.67, complete = 50
        s = matrix(4, {(0, 1): 90, (0, 2): 85, (1, 2): 85,
                       (3, 0): 85, (3, 1): 50, (3, 2): 50})
        self.assertAlmostEqual(cluster(s, AVERAGE)[-1].score, 61.667, places=3)
        self.assertAlmostEqual(cluster(s, COMPLETE)[-1].score, 50.0)


class TestEdgeCases(unittest.TestCase):
    def test_three_texts(self):
        m = cluster(matrix(3, {(0, 1): 80}))
        self.assertEqual(len(m), 2)
        self.assertEqual((m[0].a, m[0].b, m[0].score), (0, 1, 80))

    def test_all_equal_scores_ties_broken_by_id(self):
        m = cluster(matrix(4, {}, default=50))
        self.assertEqual((m[0].a, m[0].b), (0, 1))
        self.assertTrue(all(x.score == 50 for x in m))

    def test_bad_linkage(self):
        with self.assertRaises(ValueError):
            cluster(EXAMPLE, "single")

    def test_same_result_every_run(self):
        self.assertEqual(cluster(EXAMPLE), cluster(EXAMPLE))


def brute_force(sim, group_a, group_b, linkage):
    """Slow, obviously-correct group similarity: look at every pair."""
    values = [sim[i][j] for i, j in product(group_a, group_b)]
    return sum(values) / len(values) if linkage == AVERAGE else min(values)


class TestRandom(unittest.TestCase):
    """Randomly produced cases: 300 random score tables of 3-15 texts."""

    def random_matrix(self, rng):
        n = rng.randint(3, 15)
        s = [[100.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s[i][j] = s[j][i] = round(rng.uniform(0, 100), 2)
        return s

    def test_join_scores_match_brute_force(self):
        # Every join's score must equal the slow look-at-every-pair answer.
        rng = random.Random(21)
        for _ in range(300):
            s = self.random_matrix(rng)
            n = len(s)
            for linkage in (AVERAGE, COMPLETE):
                m = cluster(s, linkage)
                for x in m:
                    expected = brute_force(s, members(m, n, x.a), members(m, n, x.b), linkage)
                    self.assertAlmostEqual(x.score, expected, places=9)

    def test_structure(self):
        # n-1 joins; last group holds every text; each group is used once.
        rng = random.Random(22)
        for _ in range(300):
            s = self.random_matrix(rng)
            n = len(s)
            m = cluster(s, rng.choice([AVERAGE, COMPLETE]))
            self.assertEqual(len(m), n - 1)
            self.assertEqual(members(m, n, m[-1].new), list(range(n)))
            used = [x.a for x in m] + [x.b for x in m]
            self.assertEqual(len(used), len(set(used)))

    def test_join_scores_never_go_up(self):
        # With average or complete linkage, each join is no more similar
        # than the one before. (This is what lets Step 8 cut the tree at a
        # single score.)
        rng = random.Random(23)
        for _ in range(300):
            s = self.random_matrix(rng)
            for linkage in (AVERAGE, COMPLETE):
                scores = [x.score for x in cluster(s, linkage)]
                for earlier, later in zip(scores, scores[1:]):
                    self.assertGreaterEqual(earlier + 1e-9, later)


class TestFromPairs(unittest.TestCase):
    def test_similarity_matrix(self):
        class P:
            def __init__(self, i, j, score):
                self.i, self.j, self.score = i, j, score
        s = similarity_matrix(3, [P(0, 1, 88.0), P(0, 2, None), P(1, 2, 40.0)])
        self.assertEqual(s, [[100, 88, 0], [88, 100, 40], [0, 40, 100]])


if __name__ == "__main__":
    unittest.main()
