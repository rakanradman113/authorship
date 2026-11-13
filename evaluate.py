"""
Step 10, tool 2: Evaluation (a testing tool, not part of the program).

Runs the whole program on texts whose authors are KNOWN and counts its
mistakes. The true author comes from the file name: "Author - Title.txt".

Reports:
  Pairs  - same-author pairs found (likely / possible) and missed;
           different-author pairs wrongly labelled likely or possible
  Groups - false groupings (different-author pairs put in the same group)
           and split authors (same-author pairs not in the same group),
           for both linkages
  Lines  - the same counts for other "possible" and "likely" line values,
           so the line values can be tuned

Usage:
  python3 evaluate.py FOLDER
"""

import os
import sys

from clustering import AVERAGE, COMPLETE, cluster, similarity_matrix
from collection import analyze_collection
from grouping import find_groups
from pairs import LIKELY_LINE, POSSIBLE_LINE, all_pairs
from text_loader import load_folder


def true_author(name):
    """'Jane Austen - Emma' -> 'Jane Austen'."""
    return name.split(" - ")[0].strip()


def pair_counts(pairs, authors, possible=POSSIBLE_LINE, likely=LIKELY_LINE):
    c = {"same_total": 0, "same_likely": 0, "same_possible": 0, "same_missed": 0,
         "diff_total": 0, "diff_likely": 0, "diff_possible": 0}
    for p in pairs:
        score = p.score if p.score is not None else 0.0
        same = authors[p.i] == authors[p.j]
        level = "likely" if score >= likely else "possible" if score >= possible else "missed"
        if same:
            c["same_total"] += 1
            c["same_" + level] += 1
        else:
            c["diff_total"] += 1
            if level != "missed":
                c["diff_" + level] += 1
    return c


def group_counts(groups, authors):
    """Count, over all pairs of texts:
       false   - different authors, same group
       split   - same author, not in the same group
       together- same author, same group (correct)"""
    group_of = {t: k for k, g in enumerate(groups) for t in g.all_members}
    n = len(authors)
    c = {"false": 0, "split": 0, "together": 0}
    for i in range(n):
        for j in range(i + 1, n):
            same_group = i in group_of and group_of.get(i) == group_of.get(j)
            if authors[i] == authors[j]:
                c["together" if same_group else "split"] += 1
            elif same_group:
                c["false"] += 1
    return c


def evaluate(texts):
    """All results as a dictionary (also used by the tests)."""
    c = analyze_collection(texts)
    authors = [true_author(name) for name in c.names]
    pairs = all_pairs(c)
    sim = similarity_matrix(len(c.names), pairs)
    out = {"names": c.names, "authors": authors, "pairs": pairs,
           "pair_counts": pair_counts(pairs, authors), "groups": {}, "lines": []}
    merges = {lk: cluster(sim, lk) for lk in (COMPLETE, AVERAGE)}
    for lk, m in merges.items():
        groups, _ = find_groups(m, len(c.names))
        out["groups"][lk] = group_counts(groups, authors)
    for possible in (55, 60, 65, 70, 75):
        for likely in (60, 65, 70, 75, 80, 85, 90):
            if likely <= possible:
                continue
            row = {"possible": possible, "likely": likely,
                   "pairs": pair_counts(pairs, authors, possible, likely)}
            for lk, m in merges.items():
                groups, _ = find_groups(m, len(c.names), possible, likely)
                row[lk] = group_counts(groups, authors)
            out["lines"].append(row)
    return out


def report(r):
    p = r["pair_counts"]
    lines = [f"{len(r['names'])} texts by {len(set(r['authors']))} authors, "
             f"{p['same_total'] + p['diff_total']} pairs\n",
             f"PAIRS (lines: possible {POSSIBLE_LINE:g}, likely {LIKELY_LINE:g})",
             f"  Same author ({p['same_total']} pairs): {p['same_likely']} likely, "
             f"{p['same_possible']} possible, {p['same_missed']} missed",
             f"  Different authors ({p['diff_total']} pairs): {p['diff_likely']} wrongly likely, "
             f"{p['diff_possible']} wrongly possible\n",
             "GROUPS (pairs of texts)"]
    for lk, g in r["groups"].items():
        lines.append(f"  {lk:9} together correctly {g['together']:3}   split authors {g['split']:3}"
                     f"   false groupings {g['false']:3}")
    lines.append("\nOTHER LINE VALUES (mistakes = missed + wrongly possible/likely pairs;"
                 " groups: split + false)")
    lines.append("  possible likely | pair mistakes | complete split+false | average split+false")
    for row in r["lines"]:
        pc = row["pairs"]
        pm = pc["same_missed"] + pc["diff_likely"] + pc["diff_possible"]
        cg, ag = row["complete"], row["average"]
        mark = "  <- current" if (row["possible"], row["likely"]) == (POSSIBLE_LINE, LIKELY_LINE) else ""
        lines.append(f"  {row['possible']:8} {row['likely']:6} | {pm:13} | "
                     f"{cg['split']:5} + {cg['false']:<12} | {ag['split']:5} + {ag['false']}{mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 evaluate.py FOLDER")
    texts = load_folder(sys.argv[1])
    print(report(evaluate(texts)))
