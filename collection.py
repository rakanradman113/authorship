"""
Step 3: Collection averages, spreads and z-scores.

Up to Step 2, each text is measured on its own. Z-scores compare each text
to the whole collection:

    z = (text's value - collection average) / collection spread
  (truly extreme values are left out of the average and spread; see feature_stats)

  z =  0   -> exactly average for this collection
  z = +2   -> far above average (e.g. uses "upon" much more than the others)
  z = -1   -> somewhat below average

What this file does, in order:
  1. build_profiles()    - runs the five Step 2 features on every included text
  2. select_top_ngrams() - keeps the 300 3-grams with the highest average rate
  3. feature_stats()     - average and spread of every feature (extremes left out)
  4. zscores()           - turns every value into a z-score
  5. analyze_collection()- runs 1-4 and checks the collection size

Running time, for n texts of at most m characters:
  profiles        O(n*m)   (Step 2 features, one pass per text)
  top 3-grams     O(n*G + G log G), G = distinct 3-grams (a few thousand at
                  most for letters and spaces, so effectively a constant)
  stats, z-scores O(n*k),  k = number of features (4 + 119 + 10 + 1 + 300)
Total: O(n*m + n*k).
"""

import math
from dataclasses import dataclass, field

from features import (length_features, function_word_features,
                      punctuation_features, vocabulary_features,
                      ngram_features, TOP_NGRAMS)

METHODS = ["length", "function_words", "punctuation", "vocabulary", "ngrams"]

MIN_TEXTS = 3            # Decision: with 2 texts every z-score is +1 or -1,
                         # so any two different texts score the same (9.1)
RELIABLE_TEXTS = 5       # Decision: allow 3+, but warn below 5


# --- 1. Profiles ------------------------------------------------------------

def build_profiles(texts):
    """One dictionary per included text: method -> {feature: value}."""
    profiles = []
    for t in texts:
        profiles.append({
            "length": length_features(t),
            "function_words": function_word_features(t),
            "punctuation": punctuation_features(t),
            "vocabulary": vocabulary_features(t),
            "ngrams": ngram_features(t),       # all 3-grams for now
        })
    return profiles


# --- 2. Top 3-grams -----------------------------------------------------------

def select_top_ngrams(ngram_dicts, n=TOP_NGRAMS):
    """The n 3-grams with the highest AVERAGE rate across texts.
    - Average rate (not raw count) is used so one very long text cannot
      decide the list on its own. A 3-gram missing from a text counts as 0.
    - Decision: a 3-gram must appear in at least half the texts. Character
      names ("Elizabeth" -> eli, liz, iza ...) usually appear only in the
      texts about that character, while style patterns appear everywhere,
      so this keeps names from making two texts look like the same author.
    - Ties are broken alphabetically so the result never changes from run
      to run."""
    totals, present = {}, {}
    for d in ngram_dicts:
        for g, rate in d.items():
            totals[g] = totals.get(g, 0.0) + rate
            if rate > 0:
                present[g] = present.get(g, 0) + 1
    count = len(ngram_dicts)
    needed = math.ceil(count / 2)
    candidates = [g for g in totals if present.get(g, 0) >= needed]
    ranked = sorted(candidates, key=lambda g: (-totals[g] / count, g))
    return ranked[:n]


def keep_only(ngram_dict, chosen):
    """Restrict one text's 3-grams to the chosen list (missing = 0)."""
    return {g: ngram_dict.get(g, 0.0) for g in chosen}


# --- 3. Averages and spreads ----------------------------------------------------

def median(values):
    v = sorted(values)
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def robust_spread(values, mid):
    """A spread measure that one extreme value barely moves: the median
    distance from the middle (MAD), times 1.4826 to match the standard
    deviation on ordinary bell-curve data.
    For rare features (e.g. a word only 2 of 15 texts use) more than half the
    distances are 0, so the MAD is 0 even though texts differ; then the
    average distance from the median is used (times 1.2533, same reason)."""
    distances = [abs(v - mid) for v in values]
    spread = median(distances) * 1.4826
    if spread == 0:
        spread = sum(distances) / len(distances) * 1.2533
    return spread


LEAVE_OUT_SPREADS = 5   # values further than this from the middle are left out


def feature_stats(values):
    """Average and spread (standard deviation) of a feature, leaving out
    values that are truly extreme.

    Decision (Phase 4, option B'): with the plain average and standard
    deviation, one very unusual text (e.g. a word repeated 500 times)
    stretched the spread so much that every other text looked alike.
    Now, for each feature:
      1. find the middle (median) and an outlier-resistant spread
      2. leave out any value more than 5 of those spreads from the middle
         (far beyond what a normal text produced in testing)
      3. take the ordinary average and standard deviation of the rest
    Ordinary collections are unaffected (nothing is that far out).

    Tried first (option B): pulling extreme values in to 3 spreads instead
    of leaving them out. In small collections the pulled-in value still
    widened the spread, and unrelated texts reached the 65 line. Leaving
    values out already at 3 spreads also dropped normal texts' unusual values
    and caused many more missed matches. 5 spreads did best in testing.

    The z-score of the odd text itself still uses its real value, so it
    still shows as very different.

    Collections of fewer than 5 texts use the plain average and spread."""
    # Only with 5+ texts: with 3-4 values the median and spread rest on too
    # few texts, and an ordinary text could be judged "extreme" (found in
    # testing: ordinary texts then got average distances of up to 410).
    if len(values) < RELIABLE_TEXTS:
        avg = sum(values) / len(values)
        return avg, (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
    mid = median(values)
    rs = robust_spread(values, mid)
    if rs > 0:
        kept = [v for v in values if abs(v - mid) <= LEAVE_OUT_SPREADS * rs]
        if len(kept) >= 2:
            values = kept
    avg = sum(values) / len(values)
    spread = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
    return avg, spread


# --- 4. Z-scores ----------------------------------------------------------------

# Computers store decimals slightly imprecisely (0.1 + 0.2 = 0.30000000000000004).
# A spread this small compared to the values is rounding noise, not a real
# difference between texts.
TOLERANCE = 1e-9


def is_constant(avg, spread):
    """True when a feature has (effectively) the same value in every text."""
    return spread <= TOLERANCE * max(1.0, abs(avg))


def zscores(values):
    """Z-score of every value in the list.
    Decision: if the spread is 0 (every text has the same value, e.g. no text
    uses "betwixt"), every z-score is 0. The feature cannot tell texts apart,
    so it is effectively ignored. A spread smaller than one billionth of the
    values' size counts as 0, so rounding noise is not mistaken for a
    difference."""
    avg, spread = feature_stats(values)
    if is_constant(avg, spread):
        return [0.0] * len(values)
    return [(v - avg) / spread for v in values]


# --- 5. Whole collection ----------------------------------------------------------

@dataclass
class Collection:
    names: list                        # included texts, in order
    excluded: list                     # (name, reason) for texts left out
    raw: list                          # raw[i][method][feature]  = value
    z: list                            # z[i][method][feature]    = z-score
    stats: dict                        # stats[method][feature]   = (avg, spread)
    top_ngrams: list = field(default_factory=list)
    warning: str = ""
    unusual: list = field(default_factory=list)   # (name, average distance)

    def varying_features(self, method):
        """Features of a method whose value differs between texts.
        Features every text shares (e.g. a word no text uses) are left out,
        since they cannot tell texts apart."""
        return [f for f, (avg, spread) in self.stats[method].items()
                if not is_constant(avg, spread)]


def analyze_collection(texts):
    """Run Steps 2 and 3 on a list of Text objects from Step 1."""
    included = [t for t in texts if t.included]
    excluded = [(t.name, t.reason) for t in texts if not t.included]

    if len(included) < MIN_TEXTS:
        raise ValueError(
            f"At least {MIN_TEXTS} texts of 500+ words are needed to compare; "
            f"{len(included)} found.")
    warning = ""
    if len(included) < RELIABLE_TEXTS:
        warning = (f"Only {len(included)} texts. Similarity scores compare each text "
                   f"to the collection, so with fewer than {RELIABLE_TEXTS} texts the "
                   f"results are unreliable.")

    raw = build_profiles(included)
    top = select_top_ngrams([p["ngrams"] for p in raw])
    for p in raw:
        p["ngrams"] = keep_only(p["ngrams"], top)

    n = len(included)
    z = [{m: {} for m in METHODS} for _ in range(n)]
    stats = {m: {} for m in METHODS}
    for m in METHODS:
        for feature in raw[0][m]:
            values = [raw[i][m][feature] for i in range(n)]
            stats[m][feature] = feature_stats(values)
            for i, zi in enumerate(zscores(values)):
                z[i][m][feature] = zi

    c = Collection(names=[t.name for t in included], excluded=excluded,
                   raw=raw, z=z, stats=stats, top_ngrams=top, warning=warning)
    c.unusual = find_unusual(c)
    return c


UNUSUAL_SPREADS = 4   # Decision (Phase 4): flag texts further than this on average


def find_unusual(c):
    """Texts that are very unlike every other text (e.g. a list, a table, a
    different language). Decision (Phase 4, part a): flag them so the museum
    can decide whether to remove them, since one such text makes the others
    look more alike than they are.

    Measure: the text's average distance from the collection, in spreads
    (average |z| over every feature that varies). In testing, ordinary texts
    in collections of 5+ were at most 1.57; a word repeated 500 times was
    7.7 to 12.3. Only for 5+ texts: with 3-4 texts one odd text dominates the
    collection and cannot be told apart (it measured 1.27 vs 0.75)."""
    n = len(c.names)
    if n < RELIABLE_TEXTS:
        return []
    features = [(m, f) for m in METHODS for f in c.varying_features(m)]
    if not features:
        return []
    flagged = []
    for i in range(n):
        distance = sum(abs(c.z[i][m][f]) for m, f in features) / len(features)
        if distance > UNUSUAL_SPREADS:
            flagged.append((c.names[i], distance))
    return flagged
