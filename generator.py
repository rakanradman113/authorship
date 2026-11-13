"""
Step 10, tool 1: Random-text generator (a testing tool, not part of the program).

Makes up "authors", each with their own writing habits, and writes several
texts for each. Because we know who wrote what, the program's answers can be
checked automatically by evaluate.py.

Each made-up author has:
  - favourite and avoided function words (each word's normal rate is
    multiplied by a random factor for this author)
  - favourite ordinary words, from a shared made-up vocabulary
  - a typical sentence length and how much it varies
  - punctuation habits (commas, semicolons, dashes, ...)
`difference` controls how strongly authors differ: small values make a hard
test, larger values an easy one.

The ordinary words are invented ("varnel", "toshing") so no real text is
copied; they behave like words for every feature the program measures.

Usage:
  python3 generator.py OUTPUT_FOLDER [--authors 5] [--texts 3] [--seed 1]
Files are named "Author 1 - Text 1.txt", so evaluate.py knows the true author.
"""

import argparse
import math
import os
import random

from features import FUNCTION_WORDS

# Rough share of each function word in ordinary English prose (per word),
# so made-up texts have realistic proportions. Words not listed get 0.001.
BASE_RATES = {
    "the": 0.055, "of": 0.032, "and": 0.03, "to": 0.025, "a": 0.021, "in": 0.017,
    "i": 0.012, "that": 0.011, "it": 0.011, "he": 0.01, "was": 0.01, "his": 0.008,
    "you": 0.008, "her": 0.008, "with": 0.007, "as": 0.007, "for": 0.007,
    "had": 0.006, "not": 0.006, "be": 0.006, "she": 0.006, "but": 0.006,
    "at": 0.005, "on": 0.005, "by": 0.004, "is": 0.004, "which": 0.004,
    "have": 0.004, "my": 0.004, "they": 0.003, "all": 0.003, "this": 0.003,
    "so": 0.003, "were": 0.003, "from": 0.003, "would": 0.003, "no": 0.002,
    "been": 0.002, "if": 0.002, "or": 0.002, "their": 0.002, "upon": 0.0015,
    "could": 0.002, "very": 0.0015, "there": 0.002, "an": 0.002,
}
SYLLABLES = ["ar", "en", "or", "al", "mon", "tar", "ves", "lin", "dor", "ish", "ing",
             "el", "um", "ra", "ber", "cot", "fen", "gal", "hap", "ket", "lo", "mir",
             "nal", "pe", "quin", "ros", "sel", "tum", "va", "wel", "yar", "zo", "th",
             "ste", "ple", "ard", "ous", "ly", "ment", "ness"]


def make_vocabulary(rng, size=1500):
    """Invented ordinary words, with a few common and many rare (like real
    language, where frequency falls off roughly as 1 / rank). The +10 keeps
    the very top words from dominating (first version repeated one invented
    word over and over)."""
    words = set()
    while len(words) < size:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(rng.choice([1, 2, 2, 3, 3, 4]))))
    words = sorted(words)
    rng.shuffle(words)
    return words, [1 / (rank + 10) for rank in range(size)]


class Author:
    def __init__(self, name, rng, vocab, vocab_weights, difference):
        self.name = name
        spread = 0.25 + 0.6 * difference
        self.fw_weights = [BASE_RATES.get(w, 0.001) * math.exp(rng.gauss(0, spread))
                           for w in FUNCTION_WORDS]
        self.vocab = vocab
        self.vocab_weights = [v * math.exp(rng.gauss(0, spread * 1.5)) for v in vocab_weights]
        self.function_share = min(0.65, max(0.35, rng.gauss(0.5, 0.04 + 0.04 * difference)))
        self.sentence_mean = max(6, rng.gauss(20, 3 + 5 * difference))
        self.sentence_sd = max(2, self.sentence_mean * rng.uniform(0.25, 0.6))
        self.comma = max(0.01, rng.gauss(0.06, 0.02 + 0.02 * difference))
        self.semicolon = max(0.0, rng.gauss(0.006, 0.004 + 0.004 * difference))
        self.dash = max(0.0, rng.gauss(0.004, 0.003 + 0.003 * difference))
        self.colon = max(0.0, rng.gauss(0.002, 0.0015))
        self.question = max(0.0, rng.uniform(0.02, 0.12 * (0.5 + difference)))
        self.exclaim = max(0.0, rng.uniform(0.0, 0.08 * (0.5 + difference)))

    def sentence(self, rng):
        n = max(3, int(rng.gauss(self.sentence_mean, self.sentence_sd)))
        words = []
        for k in range(n):
            if rng.random() < self.function_share:
                words.append(rng.choices(FUNCTION_WORDS, self.fw_weights)[0])
            else:
                words.append(rng.choices(self.vocab, self.vocab_weights)[0])
            if k < n - 1:
                r = rng.random()
                if r < self.comma:
                    words[-1] += ","
                elif r < self.comma + self.semicolon:
                    words[-1] += ";"
                elif r < self.comma + self.semicolon + self.colon:
                    words[-1] += ":"
                elif r < self.comma + self.semicolon + self.colon + self.dash:
                    words[-1] += " --"
        words[0] = words[0].capitalize()
        r = rng.random()
        end = "?" if r < self.question else "!" if r < self.question + self.exclaim else "."
        return " ".join(words) + end

    def text(self, rng, n_words):
        sentences, count = [], 0
        while count < n_words:
            s = self.sentence(rng)
            sentences.append(s)
            count += len(s.split())
        paragraphs = []
        while sentences:
            k = rng.randint(3, 8)
            paragraphs.append(" ".join(sentences[:k]))
            sentences = sentences[k:]
        return "\n\n".join(paragraphs)


def generate(n_authors=5, texts_per_author=3, seed=1, difference=0.5,
             min_words=1000, max_words=3000):
    """Returns a list of (true author, title, text)."""
    rng = random.Random(seed)
    vocab, weights = make_vocabulary(rng)
    result = []
    for a in range(1, n_authors + 1):
        author = Author(f"Author {a}", rng, vocab, weights, difference)
        for t in range(1, texts_per_author + 1):
            body = author.text(rng, rng.randint(min_words, max_words))
            result.append((author.name, f"Text {t}", body))
    return result


def write_folder(folder, items):
    os.makedirs(folder, exist_ok=True)
    for author, title, body in items:
        with open(os.path.join(folder, f"{author} - {title}.txt"), "w", encoding="utf-8") as f:
            f.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write made-up texts by made-up authors.")
    ap.add_argument("folder")
    ap.add_argument("--authors", type=int, default=5)
    ap.add_argument("--texts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--difference", type=float, default=0.5,
                    help="how strongly authors differ (0.1 hard ... 1.0 easy)")
    args = ap.parse_args()
    items = generate(args.authors, args.texts, args.seed, args.difference)
    write_folder(args.folder, items)
    print(f"Wrote {len(items)} texts to {args.folder}")
