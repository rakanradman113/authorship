"""
Step 1: Load and clean texts.

What this file does, in order:
  1. strip_gutenberg()   - removes Project Gutenberg license headers/footers
     strip_editor_tags() - removes [Illustration: ...] and [Footnote: ...]
  2. normalize()         - joins hard-wrapped lines and tidies whitespace
  3. split_words()       - breaks the text into lowercase words
  4. split_sentences()   - breaks the text into sentences
  5. load_text()         - runs 1-4 on one text and applies the 500-word minimum
  6. load_folder()       - runs load_text() on every .txt file in a folder

Running time: every function makes a constant number of passes over the
text, so loading one text of m characters is O(m), and n texts is O(n*m).
"""

import os
import re
from dataclasses import dataclass, field

MIN_WORDS = 500  # Decision: texts under 500 words are excluded

# --- 1. Gutenberg headers and footers --------------------------------------

# Gutenberg files wrap the book between lines like:
#   *** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***
#   *** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***
_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                    re.IGNORECASE)
_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK",
                  re.IGNORECASE)


def strip_gutenberg(raw):
    """Keep only the text between the Gutenberg START and END markers.
    If the markers are missing, the text is returned unchanged."""
    start = _START.search(raw)
    if start:
        raw = raw[start.end():]
    end = _END.search(raw)
    if end:
        raw = raw[:end.start()]
    return raw


# Gutenberg editors insert tags like "[Illustration: The garden]" and
# "[Footnote 3: See chapter two.]". Their words are not the author's, so the
# whole tag is removed. A tag may run over several lines.
_EDITOR_TAG = re.compile(r"\[(?:Illustration|Footnote)[^\]]*\]", re.IGNORECASE)


def strip_editor_tags(raw):
    return _EDITOR_TAG.sub(" ", raw)


# --- 2. Normalizing ---------------------------------------------------------

def normalize(text):
    """Make the text uniform without changing words or punctuation.
    - Windows/Mac line endings become plain newlines
    - Curly quotes and apostrophes become straight ones, so they are
      counted the same way later
    - Lines that were hard-wrapped (single newlines) are joined back up;
      blank lines (paragraph breaks) are kept
    - Runs of spaces become one space
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"'))
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    # Drop paragraphs with no letters at all, such as scene breaks "* * *"
    # or a lone page number.
    return "\n\n".join(p for p in paragraphs if re.search(r"[^\W\d_]", p))


# --- 3. Words ---------------------------------------------------------------

# A word is a run of letters, optionally with an inner apostrophe
# (don't, o'er, 'tis is counted as tis). Hyphenated words are split:
# "to-day" -> "to", "day". Numbers are not words.
# [^\W\d_] means "any letter", including accented ones (naïve, rôle, café).
_LETTER = r"[^\W\d_]"
_WORD = re.compile(rf"{_LETTER}+(?:'{_LETTER}+)*")


# Older texts hyphenate to-day, to-morrow, to-night. Without this, each one
# would add a fake "to" to the function-word counts. They are joined into
# today, tomorrow, tonight. Other hyphenated words are still split.
_TO_DAY = re.compile(r"\bto-(day|morrow|night)\b")

# Roman numerals in headings ("CHAPTER I", "Book IV.", "Act I, Scene II")
# are numbers, and numbers are not words. Without this, the "I" in
# "CHAPTER I" would be counted as the pronoun "I".
# Only a capitalized heading word followed by a capital numeral and then
# punctuation or the end of a line counts, so "the book I lent" is untouched.
_HEADING_NUMERAL = re.compile(
    r"\b(Chapter|CHAPTER|Book|BOOK|Part|PART|Volume|VOLUME|Vol|VOL|"
    r"Section|SECTION|Act|ACT|Scene|SCENE)(\.?)\s+[IVXLCDM]+\b(?=\s*(?:[.,:;\-]|$))",
    re.MULTILINE)


def split_words(text):
    """Return the list of lowercase words in the text."""
    text = _HEADING_NUMERAL.sub(r"\1\2", text)
    text = _TO_DAY.sub(r"to\1", text.lower())
    return _WORD.findall(text)


# --- 4. Sentences -----------------------------------------------------------

# Words that end in a period but usually do not end a sentence.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "rev", "esq", "capt", "col", "gen", "lt",
    "messrs", "mt", "vs", "etc", "jr", "sr", "prof", "hon", "sq",
    "i.e", "e.g", "viz", "ch", "vol",
}

# A sentence ends with . ! or ? (possibly followed by closing quotes or
# brackets), then whitespace.
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s+")


def split_sentences(text):
    """Return a list of sentences.
    A paragraph break always ends a sentence, so a heading such as
    "CHAPTER I" stays separate (it counts as a short sentence).
    Inside a paragraph, a break is made after . ! or ? followed by a space,
    unless:
      - the next word starts with a lowercase letter
        ('"Who is there?" he asked.' and 'He paused... and left.' stay whole)
      - the word before a period is a known abbreviation (Mr., Dr., St., ...)
      - the period is in "No. 5" ("no" followed by a number)
      - the word before a period is a single capital letter (an initial,
        like the J. in J. Smith)

    Known limitation (accepted): when an abbreviation really does end a
    sentence ("He lived on Baker St. Then he moved."), no break is made.
    Fixing this fully is not possible with simple rules, and it only makes
    a few sentences slightly longer."""
    sentences = []
    for paragraph in text.split("\n\n"):
        sentences.extend(_split_paragraph(paragraph))
    # Drop "sentences" with no words at all (e.g. a stray "* * *").
    return [s for s in sentences if _WORD.search(s.lower())]


def _split_paragraph(paragraph):
    sentences = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        end = match.end()
        if _next_word_is_lowercase(paragraph, end):
            continue
        piece = paragraph[start:match.start() + 1]
        if match.group().startswith(".") and (
                _ends_in_abbreviation(piece)
                or is_number_abbreviation(paragraph, match.start())):
            continue
        sentence = paragraph[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    last = paragraph[start:].strip()
    if last:
        sentences.append(last)
    return sentences


def _next_word_is_lowercase(paragraph, i):
    """Look past any opening quotes or brackets to the next letter."""
    while i < len(paragraph) and paragraph[i] in "\"'([":
        i += 1
    return i < len(paragraph) and paragraph[i].islower()


def is_number_abbreviation(text, period_index):
    """True for the period in "No. 5": the word before is "no" and a number
    follows. "No." on its own ("I said no.") is an ordinary sentence end."""
    before = text[max(0, period_index - 4):period_index]
    if not re.search(r"(?:^|[^\w])no$", before, re.IGNORECASE):
        return False
    return re.match(r"\s*\d", text[period_index + 1:period_index + 4]) is not None


def _ends_in_abbreviation(piece):
    words = piece.rstrip(".").split()
    if not words:
        return False
    last = words[-1].strip("\"'([")
    # An initial (J. Smith). "I" is left out on purpose: "So said I." is far
    # more common in older writing than an initial I. (as in "I. Newton").
    if len(last) == 1 and last.isupper() and last != "I":
        return True
    return last.lower() in ABBREVIATIONS


# --- 5. One text --------------------------------------------------------------

@dataclass
class Text:
    name: str
    clean: str
    words: list = field(repr=False)
    sentences: list = field(repr=False)
    word_count: int = 0
    included: bool = True
    reason: str = ""          # why a text was excluded, shown to the user


def load_text(name, raw):
    """Clean one text and decide whether it is long enough to analyze."""
    clean = normalize(strip_editor_tags(strip_gutenberg(raw)))
    words = split_words(clean)
    sentences = split_sentences(clean)
    text = Text(name=name, clean=clean, words=words, sentences=sentences,
                word_count=len(words))
    if text.word_count < MIN_WORDS:
        text.included = False
        text.reason = (f"Too short: {text.word_count} words "
                       f"(at least {MIN_WORDS} needed for a reliable comparison)")
    return text


# --- 6. A folder ------------------------------------------------------------

def load_folder(folder):
    """Load every .txt file in a folder, sorted by file name."""
    texts = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, encoding="utf-8", errors="replace") as f:
                texts.append(load_text(filename, f.read()))
    return texts


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    for t in load_folder(folder):
        status = "included" if t.included else f"EXCLUDED - {t.reason}"
        print(f"{t.name}: {t.word_count} words, "
              f"{len(t.sentences)} sentences, {status}")
