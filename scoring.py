"""
Step 4: Similarity score for one pair of texts, per method.

For each of the five methods:
  1. Delta = average of |z of text A - z of text B| over the method's features
     that VARY in the collection (Burrows' Delta). 0 means identical; bigger
     means more different. Features every text shares are left out: they
     would add differences of 0 and make every pair look more similar.
  2. Score = 100 * (1 - Delta / 2.4), never below 0.

Why 2.4: first chosen as 2.2, so that two unrelated texts (typical Delta
about 1.1) scored about 50. In Phase 4 the z-scores changed to leave out extreme
values (see collection.py), which slightly raised typical Deltas; testing on
made-up authors at three difficulty levels gave the fewest missed matches,
with no wrong ones, at 2.4. Unrelated texts now score about 44-53.
    Delta 0    -> 100   (identical style)
    Delta 1.2  ->  50
    Delta 2.4+ ->   0   (far more different than usual)
To be rechecked with real books.

Running time for one pair: O(k), k = number of features (434).
"""

from collection import METHODS

DELTA_SCALE = 2.4   # Decision: straight line; 2.2 at first, 2.4 after Phase 4 testing


def delta(z_a, z_b, features=None):
    """Burrows' Delta between two {feature: z-score} dictionaries,
    averaged over `features` (default: all of them).
    Returns None when there are no features to compare."""
    if features is None:
        features = list(z_a)
    if not features:
        return None
    return sum(abs(z_a[f] - z_b[f]) for f in features) / len(features)


def delta_to_score(d):
    """Turn a Delta (0 = identical) into a 0-100 similarity score."""
    return max(0.0, 100 * (1 - d / DELTA_SCALE))


def method_scores(collection, i, j):
    """Per-method Delta and score for texts i and j of an analyzed collection.
    Returns {method: {"delta": ..., "score": ...}}."""
    result = {}
    for m in METHODS:
        features = collection.varying_features(m)
        d = delta(collection.z[i][m], collection.z[j][m], features)
        score = None if d is None else delta_to_score(d)
        result[m] = {"delta": d, "score": score, "features_used": len(features)}
    return result


# --- Step 5: Weighted overall score and breakdown ------------------------------

# Decision (Phase 4, option B): measured weights (total 100).
# First judgment-based: 30 / 20 / 20 / 17 / 13. On the 17-novel real-book
# test, each method alone put this many same-author pairs above every
# different-author pair: function words 10, 3-grams 10, punctuation 5,
# length 0, vocabulary 0 (of 16). So the two methods that separate authors
# best get most of the weight, and the other three share the rest. With
# these weights: 14 of 16 clean separations (was 10), 12 of 16 found at the
# 60 line (was 9), still 0 different-author pairs flagged.
# Dropping length and vocabulary entirely did worse (12 clean, 8 found).
WEIGHTS = {
    "function_words": 35,
    "ngrams": 35,
    "length": 10,
    "punctuation": 10,
    "vocabulary": 10,
}

METHOD_NAMES = {   # how each method is shown to the user
    "function_words": "Function words",
    "ngrams": "Character 3-grams",
    "length": "Sentence/word length",
    "punctuation": "Punctuation",
    "vocabulary": "Vocabulary richness",
}


def overall_score(per_method):
    """Weighted average of the method scores, plus a breakdown.

    Decision: a method with no score (no feature varies in the collection)
    is left out, and its weight is shared among the remaining methods in
    proportion to their own weights. Example: if vocabulary (13) is left out,
    the other weights add to 87, so function words count 30/87 = 34.5%.

    Returns (score, breakdown). Each breakdown row has the method's score,
    its effective weight (%) and the points it contributed; the points add
    up to the overall score. Rows are in weight order, largest first."""
    used = {m: w for m, w in WEIGHTS.items() if per_method[m]["score"] is not None}
    total_weight = sum(used.values())
    if total_weight == 0:
        # No feature varies anywhere in the collection: every text is identical
        # in every measured feature. Decision: a perfect match (100), shown with
        # every method at 100 and its normal weight. (Found when three copies
        # of one text scored 0 and "Unlikely".)
        breakdown = [{"method": m, "name": METHOD_NAMES[m], "score": 100.0,
                      "weight": float(w), "points": float(w), "features_used": 0,
                      "identical": True} for m, w in WEIGHTS.items()]
        return 100.0, breakdown
    breakdown = []
    for m in WEIGHTS:               # fixed order, same for every pair
        s = per_method[m]["score"]
        if m in used:
            share = used[m] / total_weight
            breakdown.append({"method": m, "name": METHOD_NAMES[m], "score": s,
                              "weight": share * 100, "points": s * share,
                              "features_used": per_method[m]["features_used"]})
        else:
            breakdown.append({"method": m, "name": METHOD_NAMES[m], "score": None,
                              "weight": 0.0, "points": 0.0, "features_used": 0,
                              "note": "Left out: no feature differs between texts"})
    score = sum(row["points"] for row in breakdown)
    return score, breakdown


def pair_score(collection, i, j):
    """Overall score and breakdown for texts i and j."""
    return overall_score(method_scores(collection, i, j))


def display_points(score, breakdown, decimals=1):
    """Round the breakdown for display so the shown points always add up to
    the shown total (decision b).

    Plain rounding can disagree: 27.0 + 16.0 + 14.0 + 11.1 + 9.8 = 77.9,
    but the total 77.8 rounds to 77.8. Method ("largest remainder"):
      1. round the total
      2. round every row DOWN
      3. hand out the missing steps (0.1 each) to the rows that lost the
         most when rounded down
    Each shown point is then within one step of its exact value.

    Note: labels and cutoffs always use the exact score, never the rounded
    one; rounding is only for what the user sees."""
    step = 10 ** decimals
    total = round(score * step)
    floors = [int(r["points"] * step) for r in breakdown]
    missing = total - sum(floors)
    by_loss = sorted(range(len(breakdown)),
                     key=lambda k: -(breakdown[k]["points"] * step - floors[k]))
    for k in by_loss[:missing]:
        floors[k] += 1
    return total / step, [f / step for f in floors]
