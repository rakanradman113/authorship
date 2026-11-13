"""
Step 2: Style features for one text.

Each function takes a Text from text_loader.py and returns a dictionary of
numbers. Every function makes one pass over the text, so each is O(m) for a
text of m characters.

Features built so far:
  2a. length_features()         - sentence and word length (averages + spread)
  2b. function_word_features()  - rate of each of 119 function words
  2c. punctuation_features()    - rate of each punctuation mark
  2d. vocabulary_features()     - vocabulary richness (MATTR)
  2e. ngram_features()          - rate of every character 3-gram
"""

import re
from collections import Counter

from text_loader import split_words


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _spread(values):
    """Standard deviation: how far values typically sit from their average.
    0 means every value is the same. Uses the population formula
    (divide by n), since we measure the whole text, not a sample of it."""
    if not values:
        return 0.0
    avg = _mean(values)
    return (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5


# --- 2a. Sentence and word length -------------------------------------------

def length_features(text):
    """Rhythm of the writing.
    sentence length = number of words in a sentence
    word length     = number of letters in a word (apostrophes not counted,
                      so "don't" has 4 letters)
    Returns the average and spread of each."""
    sentence_lengths = [len(split_words(s)) for s in text.sentences]
    word_lengths = [len(w.replace("'", "")) for w in text.words]
    return {
        "avg_sentence_length": _mean(sentence_lengths),
        "spread_sentence_length": _spread(sentence_lengths),
        "avg_word_length": _mean(word_lengths),
        "spread_word_length": _spread(word_lengths),
    }


# --- 2b. Function words -----------------------------------------------------

# 119 words, approved list. Grouped by type for readability; the order does
# not affect any result.
FUNCTION_WORDS = [
    # Articles / determiners (8)
    "the", "a", "an", "this", "that", "these", "those", "such",
    # Pronouns (22)
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "who", "whom",
    "whose", "which",
    # Prepositions (18)
    "of", "to", "in", "for", "on", "with", "at", "by", "from", "upon",
    "into", "about", "through", "without", "within", "against", "among",
    "between",
    # Conjunctions (16)
    "and", "but", "or", "nor", "if", "as", "than", "because", "though",
    "although", "while", "whilst", "yet", "so", "unless", "whether",
    # Helping verbs (21)
    "be", "is", "am", "are", "was", "were", "been", "have", "has", "had",
    "do", "did", "shall", "should", "will", "would", "may", "might", "must",
    "can", "could",
    # Common adverbs / quantifiers (18)
    "not", "no", "very", "all", "any", "some", "more", "most", "only",
    "then", "there", "here", "now", "too", "also", "even", "ever", "never",
    # Old-fashioned (16)
    "thee", "thou", "thy", "thine", "ye", "hath", "doth", "ere", "oft",
    "hence", "thus", "wherein", "whereby", "perchance", "amongst", "betwixt",
]


def function_word_features(text):
    """How often each function word appears, per 1,000 words.
    rate = (times the word appears / total words) * 1000
    Counting is one pass over the words (O(m)); looking up 119 words in
    the count afterwards is a fixed amount of extra work."""
    counts = Counter(text.words)
    total = len(text.words)
    if total == 0:
        return {w: 0.0 for w in FUNCTION_WORDS}
    return {w: counts[w] / total * 1000 for w in FUNCTION_WORDS}


# --- 2c. Punctuation --------------------------------------------------------

from text_loader import _ends_in_abbreviation, is_number_abbreviation

# Extended set. Several ways of writing a dash all count as "dash", and an
# ellipsis counts as its own mark so it does not add three periods.
PUNCTUATION_MARKS = ["comma", "period", "semicolon", "colon", "exclamation",
                     "question", "dash", "parenthesis", "quotation", "ellipsis"]

# Order matters: longer patterns are tried first, so "..." is one ellipsis
# (not three periods) and "--" is one dash (not two hyphens).
_PUNCT = re.compile(
    r"(?P<ellipsis>\.\.\.|\u2026|\.(?: \.){2})"     # ...  or ... as one character, or . . .
    r"|(?P<dash>--+|\u2014|\u2013|(?:^|(?<=\s))-(?=\s))"  # --, em/en dash, spaced or line-start -
    r"|(?P<comma>,)|(?P<period>\.)|(?P<semicolon>;)|(?P<colon>:)"
    r"|(?P<exclamation>!)|(?P<question>\?)"
    r"|(?P<parenthesis>\()"                     # only "(" so each pair counts once
    r"|(?P<quotation>(?:^|(?<=[\s(\[\u2014-]))[\"'](?=\S))",  # opening quotes only
    re.MULTILINE)

# Old contractions that start with an apostrophe. These are not quotations.
_APOSTROPHE_WORDS = ("'tis", "'twas", "'twere", "'twill", "'twould", "'em",
                     "'gainst", "'neath", "'tween", "'twixt", "'mid", "'midst",
                     "'til", "'round", "'bout", "'cause", "'ere", "'ll", "'s")


def _is_counted(match, text):
    """Rules that decide whether a found mark really counts."""
    mark, i = match.lastgroup, match.start()
    before = text[i - 1] if i > 0 else ""
    after = text[i + 1] if i + 1 < len(text) else ""
    if mark in ("period", "colon") and before.isdigit() and after.isdigit():
        return False                    # 3.50 or 10:30 is a number, not punctuation
    if mark == "period" and (_ends_in_abbreviation(text[max(0, i - 20):i + 1])
                             or is_number_abbreviation(text, i)):
        return False                    # Mr. Dr. St. J. No. 5 are not sentence periods
    if mark == "quotation" and match.group() == "'":
        rest = text[i:i + 8].lower()
        if rest.startswith(_APOSTROPHE_WORDS):
            return False                # 'tis, 'twas ... are apostrophes
    return True


def punctuation_features(text):
    """How often each punctuation mark appears, per 1,000 words.
    - Hyphens inside words (well-known) are not dashes.
    - Parentheses count once per pair.
    - Quotations count once per quoted passage (the opening mark), for both
      double quotes ("...") and single quotes ('...'), so texts printed in
      either style are measured the same way.
    - Periods after abbreviations (Mr., Dr., initials) are not counted.
    - . and : between digits (3.50, 10:30) are not counted.
    One pass over the text: O(m)."""
    clean = text.clean
    counts = Counter(m.lastgroup for m in _PUNCT.finditer(clean) if _is_counted(m, clean))
    total = len(text.words)
    if total == 0:
        return {p: 0.0 for p in PUNCTUATION_MARKS}
    return {p: counts[p] / total * 1000 for p in PUNCTUATION_MARKS}


# --- 2d. Vocabulary richness (MATTR) -----------------------------------------

MATTR_WINDOW = 500   # Decision: same as the 500-word minimum text length


def mattr(words, window=MATTR_WINDOW):
    """Moving-Average Type-Token Ratio.
    Slide a window of `window` words across the text one word at a time.
    For each position: unique words in the window / window size.
    MATTR = the average of those ratios.

    Sliding window (O(m)): instead of recounting all 500 words at every
    position, keep a count of each word in the window. Moving one step adds
    the new word and removes the old one, so each step is a fixed amount of
    work. (Recounting every window would be O(m * 500), about 500 times
    slower.)

    If the text is shorter than the window, the whole text is used as one
    window (plain type-token ratio). Texts under 500 words are excluded in
    Step 1, so this only matters for tests."""
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)

    counts = Counter(words[:window])        # the first window
    unique = len(counts)
    total_ratio = unique / window
    positions = 1
    for i in range(window, len(words)):
        new, old = words[i], words[i - window]
        counts[new] += 1                    # word entering the window
        if counts[new] == 1:
            unique += 1
        counts[old] -= 1                    # word leaving the window
        if counts[old] == 0:
            unique -= 1
            del counts[old]
        total_ratio += unique / window
        positions += 1
    return total_ratio / positions


def vocabulary_features(text):
    return {"mattr": mattr(text.words)}


# --- 2e. Character 3-grams ----------------------------------------------------

NGRAM_SIZE = 3
TOP_NGRAMS = 300   # Decision: keep the 300 most common across the collection.
                   # The choice needs every text, so it is made in Step 3;
                   # here each text gets the rates of all its 3-grams.


def ngram_text(text):
    """The text as lowercase letters and single spaces only.
    Built from the Step 1 word list, so punctuation and numbers are already
    gone. Apostrophes are dropped (don't -> dont). A space is added at each
    end so the first and last words are treated like every other word."""
    return " " + " ".join(w.replace("'", "") for w in text.words) + " "


def ngram_features(text):
    """Rate of every 3-character sequence, per 1,000 words.
    Slide a 3-character window across the letters-and-spaces text.
    One pass: O(m)."""
    s = ngram_text(text)
    total = len(text.words)
    if total == 0:
        return {}
    counts = Counter(s[i:i + NGRAM_SIZE] for i in range(len(s) - NGRAM_SIZE + 1))
    return {g: c / total * 1000 for g, c in counts.items()}
