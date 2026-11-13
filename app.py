"""
Step 9: The Flask interface.

Run it with:   python3 app.py      then open http://127.0.0.1:5000

Pages (following the design mock):
  /            Text collection - add texts, see which are ready     (Part 1)
  /compare     Compare texts   - pair scores and breakdowns         (Part 2, next)
  /groups      Authorship groups - groups and the tree               (Parts 3-4)

Texts are kept in memory while the program runs (one collection at a time,
for one user on this computer). Closing the program clears them.
"""

from flask import Flask, redirect, render_template, request, url_for

from clustering import cluster, similarity_matrix
from collection import MIN_TEXTS, RELIABLE_TEXTS, analyze_collection
from grouping import cross_links, find_groups
from pairs import all_pairs, LIKELY_LINE
from scoring import display_points
from text_loader import load_text, MIN_WORDS
from tree_svg import tree_svg

# Developer setting (hidden from museum staff): "complete" or "average".
LINKAGE = "complete"

MAX_UPLOAD_MB = 50

LABEL_WORDS = {   # decision: wording shown to museum staff
    "likely": "Likely same author",
    "possible": "Possibly same author",
    "unlikely": "Unlikely",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# The current collection: list of Text objects from Step 1, in the order added.
texts = []

# Results are worked out once and reused until the collection changes.
_cache = {"key": None, "results": None}


def analysis():
    """Run Steps 2-6 on the current collection, or reuse the last results if
    the texts have not changed. Returns None if there are too few texts."""
    key = tuple((t.name, len(t.clean)) for t in texts)
    if _cache["key"] != key:
        ready = [t for t in texts if t.included]
        if len(ready) < MIN_TEXTS:
            _cache.update(key=key, results=None)
        else:
            c = analyze_collection(texts)
            pairs = all_pairs(c)
            n = len(c.names)
            merges = cluster(similarity_matrix(n, pairs), LINKAGE)
            groups, ungrouped = find_groups(merges, n)
            _cache.update(key=key, results={
                "collection": c, "pairs": pairs, "merges": merges,
                "groups": groups, "ungrouped": ungrouped,
                "links": cross_links(groups, pairs)})
    return _cache["results"]


def pair_view(p):
    """What the Compare page shows for one pair: rounded numbers that always
    add up (Step 5 display rule), label in plain words."""
    total, shown = display_points(p.score, p.breakdown) if p.score is not None else (0.0, [])
    rows = []
    for row, points in zip(p.breakdown, shown):
        rows.append({"name": row["name"],
                     # One decimal, so score x weight visibly matches the points
                     # (whole numbers made 60 x 30% show as 18.1).
                     "score": None if row["score"] is None else round(row["score"], 1),
                     "weight": round(row["weight"], 1), "points": points,
                     "note": row.get("note", "")})
    identical = any(row.get("identical") for row in p.breakdown)
    return {"a": p.name_a, "b": p.name_b, "score": total, "label": p.label,
            "label_words": LABEL_WORDS[p.label], "rows": rows, "identical": identical}


def decode(raw_bytes):
    """Read uploaded bytes as text. Most files are UTF-8; older Windows files
    are often cp1252, so that is tried next. utf-8-sig also removes the
    invisible marker some editors put at the start of a file."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def unique_name(name):
    """Two texts must not share a name, or results would be confusing."""
    taken = {t.name for t in texts}
    if name not in taken:
        return name
    k = 2
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


def collection_status():
    """What to tell the user about the collection as a whole."""
    ready = sum(1 for t in texts if t.included)
    if ready < MIN_TEXTS:
        return ready, "blocked", (f"Add at least {MIN_TEXTS} texts of {MIN_WORDS}+ words "
                                  f"to compare them ({ready} ready so far).")
    if ready < RELIABLE_TEXTS:
        return ready, "warning", (f"Only {ready} texts are ready. Scores compare each text "
                                  f"to the rest of the collection, so results are more "
                                  f"reliable with {RELIABLE_TEXTS} or more.")
    return ready, "ok", ""


def unusual_names():
    """Texts flagged as very unlike all the others (Phase 4 decision a)."""
    results = analysis()
    return [name for name, _ in results["collection"].unusual] if results else []


@app.route("/")
def collection_page():
    ready, level, message = collection_status()
    return render_template("collection.html", texts=texts, ready=ready,
                           level=level, message=message, min_words=MIN_WORDS,
                           error=request.args.get("error", ""), unusual=unusual_names())


@app.route("/compare")
def compare_page():
    results = analysis()
    ready, level, message = collection_status()
    if results is None:
        return render_template("compare.html", pairs=[], level=level, message=message,
                               excluded=[])
    order = {"likely": 0, "possible": 1, "unlikely": 2}
    pairs = sorted(results["pairs"], key=lambda p: (order[p.label], -(p.score or 0), p.name_a, p.name_b))
    return render_template("compare.html", pairs=[pair_view(p) for p in pairs],
                           level=level, message=message,
                           excluded=results["collection"].excluded, unusual=unusual_names())


def groups_view(groups, ungrouped, links, names):
    """What the Groups page shows, in plain words. Scores shown with one
    decimal; the rules themselves always use exact scores."""
    out = []
    for k, g in enumerate(groups, start=1):
        out.append({
            "number": k,
            "sets": [[names[t] for t in sub] for sub in g.likely],
            "possible": [names[t] for t in g.possible],
            "connected": g.possibly_connected,
            "lowest": round(g.lowest_join, 1),
        })
    notes = [{"a": names[x.a], "b": names[x.b], "score": round(x.score, 1),
              "words": LABEL_WORDS["likely" if x.score >= LIKELY_LINE else "possible"]}
             for x in links]
    return out, [names[t] for t in ungrouped], notes


@app.route("/groups")
def groups_page():
    results = analysis()
    ready, level, message = collection_status()
    if results is None:
        return render_template("groups.html", groups=[], alone=[], notes=[],
                               level=level, message=message, excluded=[], ready=False,
                               tree="")
    names = results["collection"].names
    groups, alone, notes = groups_view(results["groups"], results["ungrouped"],
                                       results["links"], names)
    return render_template("groups.html", groups=groups, alone=alone, notes=notes,
                           level=level, message=message, ready=True,
                           excluded=results["collection"].excluded,
                           tree=tree_svg(results["merges"], names), unusual=unusual_names())


@app.route("/add/files", methods=["POST"])
def add_files():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return redirect(url_for("collection_page", error="Choose at least one .txt file."))
    skipped = []
    for f in files:
        if not f.filename.lower().endswith(".txt"):
            skipped.append(f.filename)
            continue
        name = unique_name(f.filename.rsplit(".", 1)[0])
        texts.append(load_text(name, decode(f.read())))
    error = ""
    if skipped:
        error = "Only .txt files can be added. Skipped: " + ", ".join(skipped)
    return redirect(url_for("collection_page", error=error))


@app.route("/add/paste", methods=["POST"])
def add_paste():
    title = request.form.get("title", "").strip()
    body = request.form.get("body", "")
    if not title:
        return redirect(url_for("collection_page", error="Give the text a title."))
    if not body.strip():
        return redirect(url_for("collection_page", error="Paste the text before adding it."))
    texts.append(load_text(unique_name(title), body))
    return redirect(url_for("collection_page"))


@app.route("/remove/<int:index>", methods=["POST"])
def remove(index):
    if 0 <= index < len(texts):
        texts.pop(index)
    return redirect(url_for("collection_page"))


@app.route("/clear", methods=["POST"])
def clear():
    texts.clear()
    return redirect(url_for("collection_page"))


@app.errorhandler(413)
def too_large(_):
    return redirect(url_for("collection_page",
                            error=f"Upload too large (limit {MAX_UPLOAD_MB} MB at a time)."))


if __name__ == "__main__":
    app.run(debug=False)
