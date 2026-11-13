"""
Step 9, Part 4: Draw the tree (dendrogram) as an SVG picture, by hand.

Decision: sideways. Titles down the left; joins branch to the right.
The horizontal axis is the score, 100 on the left to 0 on the right, so
strong matches join close to the titles and weak ones far to the right.
Dashed lines mark the likely and possibly lines (from pairs.py).

How the picture is built from the Step 7 merge list:
  1. Leaf order: walk the tree so every group's texts sit next to each
     other (otherwise branches would cross).
  2. Each text gets a row (y). Each join sits halfway between the two
     things it joins (y), at the x for its score.
  3. Each join is drawn as a bracket: two horizontal lines from the
     joined things out to the join's score, and one vertical line
     connecting them.
Branches are coloured by the join's score: green at the likely line or above,
amber between the two lines, grey below.

Running time: one walk of the tree, O(n).
"""

from html import escape

from pairs import LIKELY_LINE, POSSIBLE_LINE

ROW = 30            # height of one text's row
LABEL_W = 300       # space for titles on the left
PLOT_W = 520        # width of the score axis (100 -> 0)
TOP = 56            # room for the two line labels, one above the other
BOTTOM = 44         # room for the axis numbers
MAX_TITLE = 40      # longer titles are shortened (full title shown on hover)
LEAF_STUB = 16      # each title's line starts this far left of 100, so even a
                    # join at exactly 100 (identical texts) has visible arms

GREEN, AMBER, GREY = "#234d3a", "#c08a1e", "#b9b6ad"


def leaf_order(merges, n):
    """Texts in an order where every group's members are side by side."""
    if not merges:
        return list(range(n))
    order = []

    def walk(node):
        if node < n:
            order.append(node)
        else:
            m = merges[node - n]
            walk(m.a)
            walk(m.b)

    walk(merges[-1].new)
    return order


def x_for(score):
    """Score 100 is at the left edge of the plot, 0 at the right."""
    return LABEL_W + (100 - score) / 100 * PLOT_W


def colour(score):
    if score >= LIKELY_LINE:
        return GREEN
    if score >= POSSIBLE_LINE:
        return AMBER
    return GREY


def short(title):
    return title if len(title) <= MAX_TITLE else title[:MAX_TITLE - 1].rstrip() + "…"


def tree_svg(merges, names):
    n = len(names)
    order = leaf_order(merges, n)
    height = TOP + n * ROW + BOTTOM
    width = LABEL_W + PLOT_W + 30
    y = {}
    x = {}
    for row, t in enumerate(order):
        y[t] = TOP + row * ROW + ROW / 2
        x[t] = x_for(100) - LEAF_STUB

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
             f'width="100%" role="img" aria-label="Tree of how the texts join, by score">']

    # The likely and possibly lines, with labels (values come from pairs.py;
    # they were typed in as 80 and 65 at first, which would have gone stale)
    plot_bottom = TOP + n * ROW
    # The two labels sit at different heights (decision a): with lines 60 / 65
    # they are only 5 points apart and the labels overlapped at one height.
    for line, label, col, label_y in [
            (LIKELY_LINE, f"Likely ({LIKELY_LINE:g})", GREEN, TOP - 32),
            (POSSIBLE_LINE, f"Possibly ({POSSIBLE_LINE:g})", AMBER, TOP - 14)]:
        lx = x_for(line)
        parts.append(f'<line class="cut" x1="{lx:.1f}" y1="{label_y + 4}" x2="{lx:.1f}" y2="{plot_bottom}" '
                     f'stroke="{col}" stroke-dasharray="4 4" stroke-width="1"/>')
        parts.append(f'<text x="{lx:.1f}" y="{label_y}" text-anchor="middle" font-size="11" '
                     f'fill="{col}">{label}</text>')

    # Axis
    parts.append(f'<line x1="{x_for(100):.1f}" y1="{plot_bottom + 6}" x2="{x_for(0):.1f}" '
                 f'y2="{plot_bottom + 6}" stroke="#d6d2c7"/>')
    for tick in sorted({100, LIKELY_LINE, POSSIBLE_LINE, 25, 0}, reverse=True):
        tx = x_for(tick)
        parts.append(f'<text x="{tx:.1f}" y="{plot_bottom + 22}" text-anchor="middle" '
                     f'font-size="11" fill="#6b736d">{tick:g}</text>')
    parts.append(f'<text x="{x_for(50):.1f}" y="{plot_bottom + 38}" text-anchor="middle" '
                 f'font-size="11" fill="#6b736d">Score where texts join (higher = more similar)</text>')

    # Titles
    for t in order:
        full = escape(names[t])
        parts.append(f'<text class="leaf" x="{LABEL_W - LEAF_STUB - 8}" y="{y[t] + 4:.1f}" text-anchor="end" '
                     f'font-size="13" fill="#1f2a24"><title>{full}</title>{escape(short(names[t]))}</text>')

    # Joins, in the order they happened
    for m in merges:
        jx = x_for(m.score)
        col = colour(m.score)
        ya, yb = y[m.a], y[m.b]
        parts.append(f'<g class="join" stroke="{col}" stroke-width="2" fill="none">'
                     f'<title>Joined at {m.score:.1f} / 100</title>'
                     f'<line x1="{x[m.a]:.1f}" y1="{ya:.1f}" x2="{jx:.1f}" y2="{ya:.1f}"/>'
                     f'<line x1="{x[m.b]:.1f}" y1="{yb:.1f}" x2="{jx:.1f}" y2="{yb:.1f}"/>'
                     f'<line x1="{jx:.1f}" y1="{ya:.1f}" x2="{jx:.1f}" y2="{yb:.1f}"/></g>')
        x[m.new] = jx
        y[m.new] = (ya + yb) / 2

    parts.append("</svg>")
    return "\n".join(parts)
