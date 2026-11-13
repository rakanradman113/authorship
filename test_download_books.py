"""
Tests for the book-download tool. Real downloads cannot run here, so a fake
download function stands in for Gutenberg.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from download_books import BOOKS, check, file_name, main, title_line

HEADER = "The Project Gutenberg eBook of {t}\n\nTitle: {t}\n\nAuthor: Someone\n\n*** START OF THE PROJECT GUTENBERG EBOOK ***\nText."


class TestDownloadBooks(unittest.TestCase):
    def test_list(self):
        self.assertEqual(len(BOOKS), 17)
        self.assertEqual(len({b[2] for b in BOOKS}), 17)               # no number twice
        authors = {b[0] for b in BOOKS}
        self.assertEqual(len(authors), 6)

    def test_file_name(self):
        self.assertEqual(file_name("Jane Austen", "Emma"), "Jane Austen - Emma.txt")

    def test_title_check(self):
        self.assertTrue(check(HEADER.format(t="Emma"), "emma"))
        self.assertFalse(check(HEADER.format(t="Moby Dick"), "emma"))
        self.assertEqual(title_line("no header here"), "")

    def test_saves_right_book_refuses_wrong_one(self):
        books = [("Jane Austen", "Emma", 158, "emma"),
                 ("Jane Austen", "Persuasion", 105, "persuasion")]
        fake = {158: HEADER.format(t="Emma"), 105: HEADER.format(t="Moby Dick")}
        with tempfile.TemporaryDirectory() as d:
            ok, problems = main(d, books, get=fake.get, pause=0)
            self.assertEqual(ok, 1)
            self.assertEqual(os.listdir(d), ["Jane Austen - Emma.txt"])
            self.assertIn("WRONG BOOK", problems[0])

    def test_failed_download_reported(self):
        def broken(n):
            raise OSError("no connection")
        with tempfile.TemporaryDirectory() as d:
            ok, problems = main(d, [("A", "B", 1, "b")], get=broken, pause=0)
            self.assertEqual(ok, 0)
            self.assertIn("download failed", problems[0])

    def test_rerun_skips_existing(self):
        books = [("Jane Austen", "Emma", 158, "emma")]
        calls = []
        def fake(n):
            calls.append(n)
            return HEADER.format(t="Emma")
        with tempfile.TemporaryDirectory() as d:
            main(d, books, get=fake, pause=0)
            main(d, books, get=fake, pause=0)
            self.assertEqual(calls, [158])          # downloaded only once


if __name__ == "__main__":
    unittest.main()
