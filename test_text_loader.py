"""
Tests for Step 1. Every expected answer here is small enough to check by hand.
Run from the project folder with:   python -m unittest discover tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_loader import (strip_gutenberg, strip_editor_tags, normalize, split_words,
                         split_sentences, load_text, load_folder, MIN_WORDS)


class TestGutenberg(unittest.TestCase):
    def test_strips_header_and_footer(self):
        raw = ("License stuff\n*** START OF THE PROJECT GUTENBERG EBOOK EMMA ***\n"
               "The real book.\n*** END OF THE PROJECT GUTENBERG EBOOK EMMA ***\nMore license")
        self.assertEqual(strip_gutenberg(raw).strip(), "The real book.")

    def test_no_markers_leaves_text_alone(self):
        self.assertEqual(strip_gutenberg("Just a letter."), "Just a letter.")


class TestEditorTags(unittest.TestCase):
    def test_illustration_removed(self):
        t = load_text("t", "She smiled. [Illustration: A lady in the garden] He left.")
        self.assertEqual(t.words, ["she", "smiled", "he", "left"])

    def test_footnote_over_several_lines_removed(self):
        t = load_text("t", "True.[Footnote 1: See the\nsecond chapter.] Yes.")
        self.assertEqual(t.words, ["true", "yes"])

    def test_other_brackets_kept(self):
        # Only Illustration and Footnote tags are removed.
        t = load_text("t", "He [the vicar] spoke.")
        self.assertEqual(t.words, ["he", "the", "vicar", "spoke"])


class TestNormalize(unittest.TestCase):
    def test_joins_wrapped_lines_keeps_paragraphs(self):
        raw = "It was a\nfine day.\n\nThe next\nparagraph."
        self.assertEqual(normalize(raw), "It was a fine day.\n\nThe next paragraph.")

    def test_curly_quotes_become_straight(self):
        self.assertEqual(normalize("“He’s here”"), "\"He's here\"")

    def test_windows_line_endings(self):
        self.assertEqual(normalize("a\r\nb"), "a b")


class TestWords(unittest.TestCase):
    def test_basic(self):
        # Hand count: the, cat, sat, upon, the, mat -> 6 words
        self.assertEqual(split_words("The cat sat upon the mat."),
                         ["the", "cat", "sat", "upon", "the", "mat"])

    def test_apostrophes_kept_inside_words(self):
        self.assertEqual(split_words("Don't o'er"), ["don't", "o'er"])

    def test_accented_letters_are_letters(self):
        # Found while checking: these used to split into "na", "ve", "caf"...
        self.assertEqual(split_words("a naïve café rôle"),
                         ["a", "naïve", "café", "rôle"])

    def test_hyphens_split_and_numbers_ignored(self):
        self.assertEqual(split_words("A well-known man, in 1887."),
                         ["a", "well", "known", "man", "in"])

    # Found while testing function words: these added fake words.
    def test_to_day_family_joined(self):
        self.assertEqual(split_words("I shall go to-day, not to-morrow or To-Night."),
                         ["i", "shall", "go", "today", "not", "tomorrow", "or", "tonight"])

    def test_to_with_other_hyphen_still_splits(self):
        # Only day/morrow/night are joined.
        self.assertEqual(split_words("to-be"), ["to", "be"])

    def test_heading_numeral_is_not_a_word(self):
        self.assertEqual(split_words("CHAPTER I\n\nIt was dark."),
                         ["chapter", "it", "was", "dark"])
        self.assertEqual(split_words("Book IV. The end"), ["book", "the", "end"])
        self.assertEqual(split_words("ACT I, SCENE II."), ["act", "scene"])

    def test_pronoun_i_after_noun_is_kept(self):
        # Not a heading: lowercase "book", and "I" is followed by a word.
        self.assertEqual(split_words("the book I lent him"),
                         ["the", "book", "i", "lent", "him"])
        self.assertEqual(split_words("The Part I played was small."),
                         ["the", "part", "i", "played", "was", "small"])


class TestSentences(unittest.TestCase):
    def test_three_sentences(self):
        s = split_sentences("I came. I saw! Did I conquer?")
        self.assertEqual(s, ["I came.", "I saw!", "Did I conquer?"])

    def test_abbreviations_do_not_split(self):
        s = split_sentences("Mr. Smith met Dr. Brown at St. Paul's. They spoke.")
        self.assertEqual(len(s), 2)

    def test_initials_do_not_split(self):
        s = split_sentences("I met J. R. Hartley today. He was kind.")
        self.assertEqual(len(s), 2)

    def test_pronoun_i_ends_sentence(self):
        # Found while checking: "I." used to be mistaken for an initial.
        self.assertEqual(split_sentences("So said I. The next day we left."),
                         ["So said I.", "The next day we left."])

    def test_known_limitation_abbreviation_ending_sentence(self):
        # Accepted limitation: an abbreviation that really ends a sentence
        # does not cause a break. This test documents that behavior.
        self.assertEqual(len(split_sentences("He lived on Baker St. Then he moved.")), 1)

    # Found while testing Step 2: these used to split in the wrong places.
    def test_dialogue_question_then_lowercase(self):
        self.assertEqual(split_sentences('"Who is there?" he asked. Nobody answered.'),
                         ['"Who is there?" he asked.', "Nobody answered."])

    def test_dialogue_exclamation_then_lowercase(self):
        self.assertEqual(split_sentences('"Alas!" cried she, and wept.'),
                         ['"Alas!" cried she, and wept.'])

    def test_ellipsis_mid_sentence(self):
        self.assertEqual(split_sentences("He paused... and then he left the room."),
                         ["He paused... and then he left the room."])

    def test_heading_is_its_own_sentence(self):
        clean = normalize("CHAPTER I\n\nIt was a dark night. The wind blew.")
        self.assertEqual(split_sentences(clean),
                         ["CHAPTER I", "It was a dark night.", "The wind blew."])

    # "no" is an abbreviation only when a number follows.
    def test_no_at_sentence_end_splits(self):
        self.assertEqual(split_sentences("I said no. Then he left."),
                         ["I said no.", "Then he left."])

    def test_no_with_number_does_not_split(self):
        self.assertEqual(len(split_sentences("He lived at No. 5 Baker Street. It was small.")), 2)

    def test_word_ending_in_no_is_not_no(self):
        # "Mono." / "piano." must not be mistaken for "No."
        self.assertEqual(len(split_sentences("He played the piano. 5 men listened.")), 2)

    def test_quote_after_period(self):
        s = split_sentences('"I shall go." She left.')
        self.assertEqual(s, ['"I shall go."', "She left."])

    def test_no_final_punctuation(self):
        self.assertEqual(split_sentences("No ending here"), ["No ending here"])

    def test_scene_break_dropped(self):
        # Scene breaks sit on their own line in real books.
        clean = normalize("Hello.\n\n* * *\n\nBye.")
        self.assertEqual(split_sentences(clean), ["Hello.", "Bye."])

    def test_lone_page_number_dropped(self):
        self.assertEqual(normalize("End of page.\n\n42\n\nNext page."),
                         "End of page.\n\nNext page.")


class TestMinimumLength(unittest.TestCase):
    def test_exactly_minimum_is_included(self):
        t = load_text("a.txt", "word " * MIN_WORDS)
        self.assertTrue(t.included)
        self.assertEqual(t.word_count, 500)

    def test_one_below_minimum_is_excluded(self):
        t = load_text("b.txt", "word " * (MIN_WORDS - 1))
        self.assertFalse(t.included)
        self.assertIn("499 words", t.reason)


class TestEdgeCases(unittest.TestCase):
    def test_empty_text(self):
        t = load_text("empty.txt", "")
        self.assertEqual(t.word_count, 0)
        self.assertEqual(t.sentences, [])
        self.assertFalse(t.included)

    def test_whitespace_only(self):
        t = load_text("blank.txt", "   \n\n  \t ")
        self.assertEqual(t.word_count, 0)

    def test_no_punctuation_is_one_sentence(self):
        t = load_text("x.txt", "a b c d e")
        self.assertEqual(len(t.sentences), 1)


class TestFolder(unittest.TestCase):
    def test_loads_only_txt_files_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            for name, body in [("b.txt", "Two."), ("a.txt", "One."), ("notes.md", "skip")]:
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write(body)
            names = [t.name for t in load_folder(d)]
        self.assertEqual(names, ["a.txt", "b.txt"])


if __name__ == "__main__":
    unittest.main()
