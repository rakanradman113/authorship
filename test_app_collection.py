"""
Tests for Step 9, Part 1: the Text collection page.
Uses Flask's test client, which sends requests like a browser would.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as webapp


LONG = "The cat sat upon the mat. " * 120      # 720 words
SHORT = "A short note of thanks."               # 5 words


def upload(client, *files):
    data = {"files": [(io.BytesIO(body), name) for name, body in files]}
    return client.post("/add/files", data=data, content_type="multipart/form-data",
                       follow_redirects=True)


class CollectionPageTests(unittest.TestCase):
    def setUp(self):
        webapp.texts.clear()
        self.client = webapp.app.test_client()

    def page(self):
        return self.client.get("/").get_data(as_text=True)

    def test_empty_page(self):
        html = self.page()
        self.assertIn("No texts yet", html)
        self.assertIn("Add at least 3 texts", html)

    def test_upload_one_file(self):
        upload(self.client, ("letter.txt", LONG.encode()))
        self.assertEqual([t.name for t in webapp.texts], ["letter"])   # .txt dropped
        html = self.page()
        self.assertIn("720", html)
        self.assertIn("Ready", html)

    def test_upload_several_files(self):
        upload(self.client, ("a.txt", LONG.encode()), ("b.txt", LONG.encode()),
               ("c.txt", LONG.encode()))
        self.assertEqual(len(webapp.texts), 3)

    def test_short_text_flagged(self):
        upload(self.client, ("note.txt", SHORT.encode()))
        html = self.page()
        self.assertIn("Too short", html)
        self.assertIn("5 words", html)          # reason shown on hover

    def test_paste(self):
        self.client.post("/add/paste", data={"title": "Pasted letter", "body": LONG})
        self.assertEqual(webapp.texts[0].name, "Pasted letter")
        self.assertEqual(webapp.texts[0].word_count, 720)

    def test_paste_needs_title_and_body(self):
        r = self.client.post("/add/paste", data={"title": "", "body": LONG}, follow_redirects=True)
        self.assertIn("Give the text a title", r.get_data(as_text=True))
        r = self.client.post("/add/paste", data={"title": "X", "body": "   "}, follow_redirects=True)
        self.assertIn("Paste the text", r.get_data(as_text=True))
        self.assertEqual(webapp.texts, [])

    def test_non_txt_file_skipped(self):
        r = upload(self.client, ("photo.jpg", b"\xff\xd8\xff"), ("a.txt", LONG.encode()))
        self.assertIn("Skipped: photo.jpg", r.get_data(as_text=True))
        self.assertEqual([t.name for t in webapp.texts], ["a"])

    def test_no_file_chosen(self):
        r = self.client.post("/add/files", data={}, content_type="multipart/form-data",
                             follow_redirects=True)
        self.assertIn("Choose at least one", r.get_data(as_text=True))

    def test_duplicate_names_get_numbers(self):
        upload(self.client, ("letter.txt", LONG.encode()))
        upload(self.client, ("letter.txt", LONG.encode()))
        self.client.post("/add/paste", data={"title": "letter", "body": LONG})
        self.assertEqual([t.name for t in webapp.texts], ["letter", "letter (2)", "letter (3)"])

    def test_remove_and_clear(self):
        upload(self.client, ("a.txt", LONG.encode()), ("b.txt", LONG.encode()))
        self.client.post("/remove/0")
        self.assertEqual([t.name for t in webapp.texts], ["b"])
        self.client.post("/remove/99")                     # out of range: nothing happens
        self.assertEqual(len(webapp.texts), 1)
        self.client.post("/clear")
        self.assertEqual(webapp.texts, [])

    def test_remove_all_asks_first(self):
        # Decision: "Remove all texts" opens an "are you sure?" step; only the
        # second button actually removes.
        upload(self.client, ("a.txt", LONG.encode()), ("b.txt", LONG.encode()))
        html = self.page()
        self.assertIn("Are you sure? This removes all 2 texts.", html)
        self.assertIn("Yes, remove all", html)
        self.assertEqual(len(webapp.texts), 2)          # showing the page removes nothing

    def test_remove_all_needs_post(self):
        upload(self.client, ("a.txt", LONG.encode()))
        self.client.get("/clear")                        # a plain link visit must not clear
        self.assertEqual(len(webapp.texts), 1)

    def test_collection_messages(self):
        upload(self.client, ("a.txt", LONG.encode()), ("b.txt", LONG.encode()))
        self.assertIn("Add at least 3 texts", self.page())          # 2 ready: blocked
        upload(self.client, ("c.txt", LONG.encode()))
        self.assertIn("Only 3 texts are ready", self.page())        # 3-4: warning
        upload(self.client, ("d.txt", LONG.encode()), ("e.txt", LONG.encode()))
        html = self.page()
        self.assertNotIn("Only", html)                              # 5+: no message
        self.assertNotIn("Add at least", html)

    def test_short_texts_do_not_count_as_ready(self):
        upload(self.client, ("a.txt", LONG.encode()), ("b.txt", LONG.encode()),
               ("c.txt", SHORT.encode()))
        self.assertIn("Add at least 3 texts", self.page())

    def test_old_windows_file(self):
        # cp1252 file with curly quotes and an accented letter
        body = ("“Café” " + LONG).encode("cp1252")
        upload(self.client, ("old.txt", body))
        self.assertIn("café", webapp.texts[0].words)

    def test_file_with_utf8_marker(self):
        upload(self.client, ("bom.txt", "﻿".encode() + LONG.encode()))
        self.assertFalse(webapp.texts[0].clean.startswith("﻿"))

    def test_title_with_html_is_shown_as_text(self):
        # A title like <b>x</b> must appear as text, not change the page.
        self.client.post("/add/paste", data={"title": "<script>alert(1)</script>", "body": LONG})
        html = self.page()
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_linkage_hidden_from_page(self):
        upload(self.client, ("a.txt", LONG.encode()))
        html = self.page().lower()
        self.assertNotIn("linkage", html)
        self.assertEqual(webapp.LINKAGE, "complete")

    def test_unusual_text_flagged(self):
        # Phase 4 decision a: a text very unlike all others is flagged.
        from generator import generate
        for a, t, body in generate(5, 1, seed=1):
            webapp.texts.append(webapp.load_text(f"{a} - {t}", body))
        webapp.texts.append(webapp.load_text("Odd list", " ".join(["word"] * 500) + "."))
        webapp._cache.update(key=None, results=None)
        html = self.page()
        self.assertIn("<strong>Odd list</strong> is very unlike every other text", html)
        self.assertIn(">Unusual</span>", html)
        self.assertEqual(html.count(">Unusual</span>"), 1)
        for path in ["/compare", "/groups"]:
            self.assertIn("very unlike every other text", self.client.get(path).get_data(as_text=True))

    def test_no_flag_for_ordinary_texts(self):
        from generator import generate
        for a, t, body in generate(5, 1, seed=1):
            webapp.texts.append(webapp.load_text(f"{a} - {t}", body))
        webapp._cache.update(key=None, results=None)
        self.assertNotIn("very unlike", self.page())

    def test_label_wording(self):
        self.assertEqual(webapp.LABEL_WORDS, {"likely": "Likely same author",
                                              "possible": "Possibly same author",
                                              "unlikely": "Unlikely"})


if __name__ == "__main__":
    unittest.main()
