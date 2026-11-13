"""
Tests for Step 2. Every expected answer is worked out by hand in the comments.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_loader import load_text
from features import (length_features, function_word_features, FUNCTION_WORDS,
                      punctuation_features, PUNCTUATION_MARKS,
                      mattr, vocabulary_features, MATTR_WINDOW,
                      ngram_text, ngram_features, TOP_NGRAMS)
import random


class TestLengthFeatures(unittest.TestCase):
    def test_hand_example(self):
        # "I came. I saw the city. Did I conquer?"
        # Sentence lengths: "I came" = 2, "I saw the city" = 4, "Did I conquer" = 3
        #   average = (2+4+3)/3 = 3
        #   spread  = sqrt(((2-3)^2 + (4-3)^2 + (3-3)^2) / 3) = sqrt(2/3) = 0.8165
        # Word lengths: I=1 came=4 I=1 saw=3 the=3 city=4 Did=3 I=1 conquer=7
        #   total = 27 letters over 9 words, average = 3
        #   squared distances from 3: 4,1,4,0,0,1,0,4,16 = 30
        #   spread = sqrt(30/9) = 1.8257
        f = length_features(load_text("t", "I came. I saw the city. Did I conquer?"))
        self.assertAlmostEqual(f["avg_sentence_length"], 3.0)
        self.assertAlmostEqual(f["spread_sentence_length"], 0.8165, places=4)
        self.assertAlmostEqual(f["avg_word_length"], 3.0)
        self.assertAlmostEqual(f["spread_word_length"], 1.8257, places=4)

    def test_equal_lengths_have_zero_spread(self):
        # Two sentences of 3 words, every word 3 letters.
        f = length_features(load_text("t", "The cat sat. The dog ran."))
        self.assertAlmostEqual(f["avg_sentence_length"], 3.0)
        self.assertAlmostEqual(f["spread_sentence_length"], 0.0)
        self.assertAlmostEqual(f["avg_word_length"], 3.0)
        self.assertAlmostEqual(f["spread_word_length"], 0.0)

    def test_apostrophe_not_counted_as_letter(self):
        # "don't" = 4 letters, "go" = 2, average = 3
        f = length_features(load_text("t", "Don't go."))
        self.assertAlmostEqual(f["avg_word_length"], 3.0)

    def test_same_average_different_spread(self):
        # Both texts average 3 words per sentence:
        #   steady: 3, 3          -> spread 0
        #   mixed:  1, 5          -> spread 2
        steady = length_features(load_text("a", "One two three. Four five six."))
        mixed = length_features(load_text("b", "One. Two three four five six."))
        self.assertAlmostEqual(steady["avg_sentence_length"], 3.0)
        self.assertAlmostEqual(mixed["avg_sentence_length"], 3.0)
        self.assertAlmostEqual(steady["spread_sentence_length"], 0.0)
        self.assertAlmostEqual(mixed["spread_sentence_length"], 2.0)

    def test_empty_text_gives_zeros(self):
        f = length_features(load_text("t", ""))
        self.assertEqual(f, {"avg_sentence_length": 0.0, "spread_sentence_length": 0.0,
                             "avg_word_length": 0.0, "spread_word_length": 0.0})

    def test_sample_letter(self):
        # sample_texts/hartley_letter.txt repeats 2 sentences 40 times:
        #   "It is perhaps worth observing ... in its extent" = 17 words
        #   "Mr Hartley wrote to Dr Brown ... I think satisfied" = 14 words
        #   average = 15.5, spread = |17 - 15.5| = 1.5
        # (This test first expected 16 and 15 words; recounting by hand showed
        #  the code was right and the hand count was wrong.)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sample_texts", "hartley_letter.txt")
        with open(path, encoding="utf-8") as fh:
            f = length_features(load_text("h", fh.read()))
        self.assertAlmostEqual(f["avg_sentence_length"], 15.5)
        self.assertAlmostEqual(f["spread_sentence_length"], 1.5)


class TestFunctionWords(unittest.TestCase):
    def test_list_has_119_unique_words(self):
        self.assertEqual(len(FUNCTION_WORDS), 119)
        self.assertEqual(len(set(FUNCTION_WORDS)), 119)   # no duplicates

    def test_list_is_lowercase(self):
        # Text words are lowercased, so list words must be too or they never match.
        self.assertTrue(all(w == w.lower() for w in FUNCTION_WORDS))

    def test_hand_example(self):
        # "The cat sat upon the mat and the dog sat by it."
        # 12 words. the=3, upon=1, and=1, by=1, it=1; cat/sat/mat/dog are not on the list.
        #   the  = 3/12 * 1000 = 250
        #   upon = 1/12 * 1000 = 83.333
        #   of   = 0
        f = function_word_features(load_text("t", "The cat sat upon the mat and the dog sat by it."))
        self.assertAlmostEqual(f["the"], 250.0)
        self.assertAlmostEqual(f["upon"], 83.3333, places=3)
        self.assertAlmostEqual(f["and"], 83.3333, places=3)
        self.assertEqual(f["of"], 0.0)

    def test_returns_every_word(self):
        f = function_word_features(load_text("t", "Hello there."))
        self.assertEqual(set(f), set(FUNCTION_WORDS))

    def test_capitals_still_count(self):
        # "THE", "The", "the" are all the same word.
        f = function_word_features(load_text("t", "THE The the cat"))
        self.assertAlmostEqual(f["the"], 750.0)   # 3 of 4 words

    def test_whole_words_only(self):
        # "there" must not count as "the"; "isle" must not count as "is".
        f = function_word_features(load_text("t", "There is an isle"))
        self.assertAlmostEqual(f["the"], 0.0)
        self.assertAlmostEqual(f["there"], 250.0)
        self.assertAlmostEqual(f["is"], 250.0)

    def test_old_fashioned_words(self):
        # "Thou hast thy sword" -> thou = 1/4, thy = 1/4 ("hast" is not on the list)
        f = function_word_features(load_text("t", "Thou hast thy sword"))
        self.assertAlmostEqual(f["thou"], 250.0)
        self.assertAlmostEqual(f["thy"], 250.0)

    def test_rate_does_not_depend_on_length(self):
        # The same text repeated 10 times has the same rates.
        short = function_word_features(load_text("a", "The cat and the dog."))
        long = function_word_features(load_text("b", "The cat and the dog. " * 10))
        for w in FUNCTION_WORDS:
            self.assertAlmostEqual(short[w], long[w])

    def test_empty_text_gives_zeros(self):
        f = function_word_features(load_text("t", ""))
        self.assertTrue(all(v == 0.0 for v in f.values()))


class TestPunctuation(unittest.TestCase):
    def rates(self, s):
        return punctuation_features(load_text("t", s))

    def test_hand_example(self):
        # 'Yes, I came; I saw: did I win? No!'
        # words: yes i came i saw did i win no = 9
        # comma 1, semicolon 1, colon 1, question 1, exclamation 1
        #   each = 1/9 * 1000 = 111.11
        f = self.rates("Yes, I came; I saw: did I win? No!")
        for mark in ["comma", "semicolon", "colon", "question", "exclamation"]:
            self.assertAlmostEqual(f[mark], 111.111, places=2)
        self.assertEqual(f["period"], 0.0)

    def test_ellipsis_is_not_three_periods(self):
        # "Wait... no." -> 2 words, 1 ellipsis, 1 period
        f = self.rates("Wait... no.")
        self.assertAlmostEqual(f["ellipsis"], 500.0)
        self.assertAlmostEqual(f["period"], 500.0)

    def test_ellipsis_single_character(self):
        f = self.rates("Wait\u2026 no")
        self.assertAlmostEqual(f["ellipsis"], 500.0)

    def test_dash_forms_all_count_as_dash(self):
        # 4 words, 3 dashes: "--", em dash, spaced hyphen
        f = self.rates("a -- b\u2014c - d")
        self.assertAlmostEqual(f["dash"], 750.0)

    def test_hyphen_inside_word_is_not_a_dash(self):
        f = self.rates("A well-known man.")
        self.assertEqual(f["dash"], 0.0)

    def test_long_dash_counts_once(self):
        # "----" is one dash, not two
        f = self.rates("a ---- b")
        self.assertAlmostEqual(f["dash"], 500.0)

    def test_parenthesis_counts_once_per_pair(self):
        # "a (b) c" -> 3 words, 1 pair
        f = self.rates("a (b) c")
        self.assertAlmostEqual(f["parenthesis"], 333.333, places=2)

    def test_quotes_count_once_per_passage(self):
        # '"Go," he said.' -> 3 words, 1 quoted passage, 1 comma, 1 period
        f = self.rates('"Go," he said.')
        self.assertAlmostEqual(f["quotation"], 333.333, places=2)
        self.assertAlmostEqual(f["comma"], 333.333, places=2)
        self.assertAlmostEqual(f["period"], 333.333, places=2)

    def test_curly_quotes_count_too(self):
        f = self.rates("\u201cGo\u201d")
        self.assertAlmostEqual(f["quotation"], 1000.0)   # 1 passage, 1 word

    # Fixes chosen after the first bug hunt
    def test_spaced_ellipsis_is_one(self):
        # "Wait . . . go" -> 2 words, 1 ellipsis, no periods
        f = self.rates("Wait . . . go")
        self.assertAlmostEqual(f["ellipsis"], 500.0)
        self.assertEqual(f["period"], 0.0)

    def test_single_quote_dialogue_counts(self):
        # "'Go home,' she said." -> 4 words, 1 quoted passage
        self.assertAlmostEqual(self.rates("'Go home,' she said.")["quotation"], 250.0)

    def test_single_and_double_quote_styles_match(self):
        a = self.rates('"Go home," she said.')
        b = self.rates("'Go home,' she said.")
        self.assertEqual(a, b)

    def test_old_contractions_are_not_quotes(self):
        self.assertEqual(self.rates("'Tis late, 'twas said.")["quotation"], 0.0)

    def test_plural_possessive_is_not_a_quote(self):
        self.assertEqual(self.rates("the boys' hats")["quotation"], 0.0)

    def test_abbreviation_periods_not_counted(self):
        # "Mr. Smith met Dr. Brown." -> 5 words, only the final period counts
        self.assertAlmostEqual(self.rates("Mr. Smith met Dr. Brown.")["period"], 200.0)

    def test_initial_period_not_counted(self):
        # "J. Smith left." -> 3 words, 1 real period
        self.assertAlmostEqual(self.rates("J. Smith left.")["period"], 333.333, places=2)

    def test_numbers_do_not_count(self):
        # "It cost 3.50 at 10:30." -> 3 words (numbers are not words), 1 period
        f = self.rates("It cost 3.50 at 10:30.")
        self.assertAlmostEqual(f["period"], 333.333, places=2)
        self.assertEqual(f["colon"], 0.0)

    def test_no_period_counts_only_without_number(self):
        # "I said no." -> 3 words, 1 period
        self.assertAlmostEqual(self.rates("I said no.")["period"], 333.333, places=2)
        # "At No. 5 he stopped." -> 4 words (5 is a number), 1 period (the last)
        self.assertAlmostEqual(self.rates("At No. 5 he stopped.")["period"], 250.0)

    def test_dash_at_line_start(self):
        self.assertAlmostEqual(self.rates("- Item one")["dash"], 500.0)

    def test_returns_every_mark(self):
        self.assertEqual(set(self.rates("Hi.")), set(PUNCTUATION_MARKS))

    def test_empty_text_gives_zeros(self):
        self.assertTrue(all(v == 0.0 for v in self.rates("").values()))

    def test_sample_letter(self):
        # Each 31-word block, counted by hand:
        #   "It is, perhaps, worth ... extent."   -> 2 commas, 1 period
        #   "Mr. Hartley wrote to Dr. Brown upon the matter; he was, I think,
        #    satisfied."                          -> 2 commas, 1 period (Mr. and Dr.
        #                                             are abbreviations), 1 semicolon
        #   per block: 4 commas, 2 periods, 1 semicolon over 31 words
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sample_texts", "hartley_letter.txt")
        with open(path, encoding="utf-8") as fh:
            f = punctuation_features(load_text("h", fh.read()))
        self.assertAlmostEqual(f["comma"], 4 / 31 * 1000)
        self.assertAlmostEqual(f["period"], 2 / 31 * 1000)
        self.assertAlmostEqual(f["semicolon"], 1 / 31 * 1000)
        self.assertEqual(f["dash"], 0.0)   # the "* * *" breaks were removed in Step 1


def slow_mattr(words, window):
    """Obviously-correct reference: recount every window from scratch."""
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = [len(set(words[i:i + window])) / window
              for i in range(len(words) - window + 1)]
    return sum(ratios) / len(ratios)


class TestMATTR(unittest.TestCase):
    def test_hand_example_window_3(self):
        # words: a b a c, window 3
        #   [a b a] -> 2 unique / 3 = 0.6667
        #   [b a c] -> 3 unique / 3 = 1.0
        #   MATTR = (0.6667 + 1.0) / 2 = 0.8333
        self.assertAlmostEqual(mattr("a b a c".split(), window=3), 0.8333, places=4)

    def test_hand_example_window_2(self):
        # words: a a b b, window 2
        #   [a a]=1/2  [a b]=2/2  [b b]=1/2  -> (0.5 + 1 + 0.5) / 3 = 0.6667
        self.assertAlmostEqual(mattr("a a b b".split(), window=2), 0.6667, places=4)

    def test_all_different_is_1(self):
        self.assertAlmostEqual(mattr("a b c d e f".split(), window=3), 1.0)

    def test_all_same_is_minimum(self):
        # every window holds 1 unique word out of 3
        self.assertAlmostEqual(mattr("a a a a a".split(), window=3), 1 / 3)

    def test_shorter_than_window_uses_whole_text(self):
        # 4 words, 3 unique -> 0.75
        self.assertAlmostEqual(mattr("a b a c".split(), window=10), 0.75)

    def test_exactly_one_window(self):
        self.assertAlmostEqual(mattr("a b a".split(), window=3), 2 / 3)

    def test_empty(self):
        self.assertEqual(mattr([], window=3), 0.0)

    def test_window_is_500(self):
        self.assertEqual(MATTR_WINDOW, 500)

    def test_length_does_not_change_mattr_much(self):
        # Plain type-token ratio drops as a text gets longer; MATTR should not.
        rng = random.Random(1)
        vocab = [f"w{i}" for i in range(2000)]
        short = [rng.choice(vocab) for _ in range(1000)]
        long = [rng.choice(vocab) for _ in range(20000)]
        plain = lambda ws: len(set(ws)) / len(ws)
        self.assertGreater(plain(short) - plain(long), 0.3)       # plain TTR collapses
        self.assertLess(abs(mattr(short) - mattr(long)), 0.02)    # MATTR stays steady

    def test_random_texts_match_slow_version(self):
        # Randomly produced cases: 200 random texts with random sizes,
        # vocabularies and windows. Fast sliding version must equal the slow one.
        rng = random.Random(42)
        for _ in range(200):
            vocab = [f"w{i}" for i in range(rng.randint(1, 30))]
            words = [rng.choice(vocab) for _ in range(rng.randint(1, 120))]
            window = rng.randint(1, 40)
            self.assertAlmostEqual(mattr(words, window), slow_mattr(words, window),
                                   msg=f"window={window}, words={words}")

    def test_vocabulary_features_on_sample_letter(self):
        # The sample letter repeats one 31-word block. Only "is" and "the"
        # repeat inside it, so it has 31 - 2 = 29 unique words. Any 500-word
        # window covers the block 16+ times, so every window holds all 29:
        # MATTR = 29/500 = 0.058.
        # (This test first expected 28, wrongly counting "it" and "its" as the
        #  same word; the program was right.)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sample_texts", "hartley_letter.txt")
        with open(path, encoding="utf-8") as fh:
            f = vocabulary_features(load_text("h", fh.read()))
        self.assertAlmostEqual(f["mattr"], 29 / 500)


class TestNgrams(unittest.TestCase):
    def grams(self, s):
        return ngram_features(load_text("t", s))

    def test_hand_example(self):
        # "The cat." -> text " the cat " (9 characters) -> 7 three-grams:
        #   " th", "the", "he ", "e c", " ca", "cat", "at "
        # each appears once; 2 words -> each rate = 1/2 * 1000 = 500
        f = self.grams("The cat.")
        self.assertEqual(set(f), {" th", "the", "he ", "e c", " ca", "cat", "at "})
        self.assertTrue(all(v == 500.0 for v in f.values()))

    def test_repeated_gram(self):
        # "the then" -> " the then " -> " th" twice, "the" twice, "he " once,
        # "e t" once, "hen" once, "en " once; 2 words
        f = self.grams("the then")
        self.assertAlmostEqual(f[" th"], 1000.0)
        self.assertAlmostEqual(f["the"], 1000.0)
        self.assertAlmostEqual(f["hen"], 500.0)

    def test_lowercase_and_letters_only(self):
        # Case, punctuation and numbers do not matter
        self.assertEqual(self.grams("THE CAT!!"), self.grams("the cat"))
        self.assertEqual(self.grams("the 42 cat"), self.grams("the cat"))

    def test_apostrophe_dropped(self):
        self.assertEqual(ngram_text(load_text("t", "Don't go")), " dont go ")

    def test_accented_letters_kept(self):
        self.assertIn("caf\u00e9"[1:], self.grams("caf\u00e9"))   # "afé"

    def test_total_matches_formula(self):
        # Randomly produced cases: for any text, the number of 3-grams is
        # (length of the letters-and-spaces text - 2). Check the rates add up.
        rng = random.Random(7)
        letters = "abcde "
        for _ in range(100):
            raw = "".join(rng.choice(letters) for _ in range(rng.randint(5, 300)))
            t = load_text("r", raw)
            if not t.words:
                continue
            expected_total = (len(ngram_text(t)) - 2) / len(t.words) * 1000
            self.assertAlmostEqual(sum(ngram_features(t).values()), expected_total)

    def test_empty(self):
        self.assertEqual(self.grams(""), {})

    def test_top_ngrams_setting(self):
        self.assertEqual(TOP_NGRAMS, 300)


if __name__ == "__main__":
    unittest.main()
