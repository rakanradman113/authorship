"""
Step 8: Cut the tree into groups of possible common authorship.

Decisions:
  - Cut at the "possible" line (60; set in pairs.py): every join at or above
    it stays inside a group. (Changed from "biggest gap": with fixed label
    lines, the gap could cut above the line and drop "possible" texts.)
  - Inside a group, joins at or above the "likely" line (65) form LIKELY
    subgroups. Texts that only joined between the lines are POSSIBLE members.
  - If a group holds two or more likely subgroups that only joined each
    other between the lines, they are shown as separate likely groups, marked as
    possibly connected.
  - With complete linkage (the default) a group never contains a weak pair.
    But a "possible" pair can still end up split across groups, since
    one text may match one member of a group and not the others. Those pairs
    are listed as notes (cross_links), so nothing is hidden.

Join scores never go up as the tree is built (tested in Step 7), so
"every join above a line" is always a set of whole branches of the tree.

Running time: one walk over the n-1 joins, O(n), plus listing members,
O(n) per group. Total O(n^2) at most, small next to Steps 6-7.
"""

from dataclasses import dataclass, field

from clustering import members
from pairs import POSSIBLE_LINE, LIKELY_LINE


@dataclass
class Group:
    likely: list                    # list of likely subgroups (each a list of text indices)
    possible: list                  # text indices that only reached "possible"
    lowest_join: float              # score at which the whole group came together
    likely_scores: list = field(default_factory=list)   # lowest join inside each likely subgroup

    @property
    def all_members(self):
        return sorted([t for sub in self.likely for t in sub] + self.possible)

    @property
    def possibly_connected(self):
        """True when two or more likely subgroups only joined between the lines."""
        return len(self.likely) >= 2


def _score(merges, n, node):
    return merges[node - n].score


def _top_nodes(merges, n, node, line):
    """Highest joins at or above `line` inside `node` (single texts are not groups)."""
    if node < n:
        return []
    if _score(merges, n, node) >= line:
        return [node]
    m = merges[node - n]
    return _top_nodes(merges, n, m.a, line) + _top_nodes(merges, n, m.b, line)


def find_groups(merges, n, possible=POSSIBLE_LINE, likely=LIKELY_LINE):
    """Returns (groups, ungrouped text indices).
    Groups are ordered by how tightly the WHOLE group holds together
    (its lowest join), tightest first."""
    if not merges:
        return [], list(range(n))
    root = merges[-1].new
    groups = []
    for node in _top_nodes(merges, n, root, possible):
        likely_nodes = _top_nodes(merges, n, node, likely)
        likely_subs = [members(merges, n, x) for x in likely_nodes]
        in_likely = {t for sub in likely_subs for t in sub}
        groups.append(Group(
            likely=likely_subs,
            possible=[t for t in members(merges, n, node) if t not in in_likely],
            lowest_join=_score(merges, n, node),
            likely_scores=[_score(merges, n, x) for x in likely_nodes],
        ))
    grouped = {t for g in groups for t in g.all_members}
    ungrouped = [t for t in range(n) if t not in grouped]
    groups.sort(key=lambda g: (-g.lowest_join, g.all_members))
    return groups, ungrouped


@dataclass
class CrossLink:
    a: int
    b: int
    score: float


def cross_links(groups, pairs, possible=POSSIBLE_LINE):
    """Pairs at or above the possible line whose two texts are NOT in the same group.
    Shown to the user as notes, e.g. "C also possibly matches A (70)".
    Strongest first. One pass over the pairs: O(n^2)."""
    group_of = {}
    for k, g in enumerate(groups):
        for t in g.all_members:
            group_of[t] = k
    links = []
    for p in pairs:
        if p.score is None or p.score < possible:
            continue
        ga, gb = group_of.get(p.i), group_of.get(p.j)
        if ga is None or gb is None or ga != gb:
            links.append(CrossLink(p.i, p.j, p.score))
    links.sort(key=lambda x: (-x.score, x.a, x.b))
    return links
