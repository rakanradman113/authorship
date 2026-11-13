"""
Step 7: Hierarchical clustering (written from scratch, no clustering library).

Start with every text in its own group. Repeatedly join the two most
similar groups, until one group is left. The list of joins (the "merge
list") is the tree that Step 8 cuts and Step 9 draws.

How similar are two GROUPS? That is the linkage setting (both are
available; complete is the default, see below):
  average  - the average score over every text-to-text pair between them
  complete - the LOWEST score over every text-to-text pair between them
             (every member must match every other member)

Example (scores out of 100):
  A-B 90, D-E 85, A-C 80, B-C 75, C-D 70, all other pairs 30
  average:  A+B at 90, D+E at 85, {A,B}+C at 77.5, then the two groups at 36.7
  complete: A+B at 90, D+E at 85, {A,B}+C at 75,   then the two groups at 30

Running time for n texts:
  n-1 joins; each join searches the current groups' similarities, O(n^2),
  and updates one row, O(n). Total O(n^3). (Faster versions exist, around
  O(n^2 log n), but n is the number of texts, which is small for a museum.)
"""

from dataclasses import dataclass

AVERAGE, COMPLETE = "average", "complete"
# Decision (Step 8): complete linkage is the default. Every pair inside a
# group then scores at least as high as the group's lowest join, so a group
# never contains a weak pair. Average stays available as a setting.
DEFAULT_LINKAGE = COMPLETE


@dataclass
class Merge:
    a: int          # group joined (ids 0..n-1 are single texts;
    b: int          #   n, n+1, ... are groups made by earlier joins)
    new: int        # id of the group this join creates
    score: float    # similarity of a and b when they were joined
    size: int       # number of texts in the new group


def similarity_matrix(n, pairs):
    """n x n table of pair scores from Step 6. A pair with no score counts as 0."""
    s = [[100.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for p in pairs:
        score = p.score if p.score is not None else 0.0
        s[p.i][p.j] = s[p.j][p.i] = score
    return s


def cluster(sim, linkage=DEFAULT_LINKAGE):
    """Join groups until one is left. Returns the merge list (n-1 Merges)."""
    if linkage not in (AVERAGE, COMPLETE):
        raise ValueError(f"linkage must be '{AVERAGE}' or '{COMPLETE}'")
    n = len(sim)
    size = {i: 1 for i in range(n)}
    # Similarity between every two current groups, keyed by (smaller id, larger id)
    between = {(i, j): sim[i][j] for i in range(n) for j in range(i + 1, n)}
    merges = []
    next_id = n

    while len(size) > 1:
        # 1. The most similar pair of groups. Ties: smallest ids first, so the
        #    result never changes from run to run.
        (a, b), best = max(between.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        new = next_id
        next_id += 1

        # 2. Similarity of the new group to every other remaining group
        for k in size:
            if k in (a, b):
                continue
            s_a = between[_key(a, k)]
            s_b = between[_key(b, k)]
            if linkage == AVERAGE:
                # Weighted by group size, this equals the average over all
                # text-to-text pairs between the new group and k.
                value = (size[a] * s_a + size[b] * s_b) / (size[a] + size[b])
            else:
                value = min(s_a, s_b)
            between[_key(new, k)] = value

        # 3. Remove the two old groups
        for k in list(size):
            if k not in (a, b):
                between.pop(_key(a, k), None)
                between.pop(_key(b, k), None)
        between.pop(_key(a, b))
        size[new] = size.pop(a) + size.pop(b)
        merges.append(Merge(a, b, new, best, size[new]))
    return merges


def members(merges, n, group_id):
    """The text indices in a group (single texts are their own group)."""
    if group_id < n:
        return [group_id]
    m = merges[group_id - n]
    return sorted(members(merges, n, m.a) + members(merges, n, m.b))


def _key(x, y):
    return (x, y) if x < y else (y, x)
