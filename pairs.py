"""
Step 6: Score every pair of texts, then label each pair.

1. all_pairs()    - scores all n(n-1)/2 pairs (Steps 4-5 for each)
2. label()        - gives each pair its label

Label lines are FIXED scores (decision, changed after testing):
    possible line = 60
    likely line   = 65
(Started at 65 / 80. Changed after the Phase 4 real-book test of 17 novels by
6 authors: no different-author pair scored above 59.7, and no same-author
pair reached 80. At 60 / 65: 9 of 16 same-author pairs found, 0 of 120
different-author pairs wrongly flagged.)
Lines relative to the collection (average + 1 or 2 spreads) were tried first,
but when many pairs really match (e.g. 3 texts by one author and 3 by
another) the matches raise the average and spread so much that the lines
pass 100 and even identical texts could not be "likely". The scores are
already relative to the collection through the z-scores, so fixed lines are
enough.
The same lines are used for grouping in Step 8, so a pair's label and its
group's label always agree.

Labels always use the exact score, never the rounded one shown to the user.

Running time: n(n-1)/2 pairs, each O(k) (k = 434 features), so O(n^2 * k).
Labels: one comparison per pair, O(n^2).
"""

from dataclasses import dataclass

from scoring import pair_score

POSSIBLE_LINE = 60.0   # Decision (Phase 4, real-book test, option A); first 65
LIKELY_LINE = 65.0     # Decision (Phase 4, real-book test, option A); first 80

LIKELY, POSSIBLE, UNLIKELY = "likely", "possible", "unlikely"


@dataclass
class PairResult:
    i: int
    j: int
    name_a: str
    name_b: str
    score: float
    breakdown: list
    label: str = UNLIKELY


def all_pairs(collection):
    """Score every pair once (A-B, never also B-A or A-A), then label them."""
    n = len(collection.names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score, breakdown = pair_score(collection, i, j)
            pairs.append(PairResult(i, j, collection.names[i], collection.names[j],
                                    score, breakdown))
    for p in pairs:
        p.label = label(p.score)
    return pairs


def label(score):
    """Exact score -> likely / possible / unlikely. A score exactly on a
    line counts as reaching it."""
    if score is None:
        return UNLIKELY
    if score >= LIKELY_LINE:
        return LIKELY
    if score >= POSSIBLE_LINE:
        return POSSIBLE
    return UNLIKELY
