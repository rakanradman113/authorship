"""
Testing tool (Phase 4): download the real known-author book set from
Project Gutenberg. Not part of the program itself.

What it does, for each book in BOOKS:
  1. downloads the plain-text file from gutenberg.org
  2. checks the file's "Title:" line matches the book, so a wrong number
     in the list cannot silently put the wrong book in the test
  3. saves it as "Author - Title.txt" in the output folder
Books already in the folder are skipped, so it can be run again safely.
It waits 2 seconds between downloads to be polite to Gutenberg's servers.

The Gutenberg license header and footer are removed later by the program
itself (Step 1). You still need to remove each book's CONTENTS page by hand.

Usage (on your own computer, in the project folder):
    python3 download_books.py real_books
"""

import os
import sys
import time
import urllib.request

# (author, title, Gutenberg book number, word that must appear in the Title: line)
BOOKS = [
    ("Jane Austen", "Pride and Prejudice", 1342, "pride"),
    ("Jane Austen", "Emma", 158, "emma"),
    ("Jane Austen", "Persuasion", 105, "persuasion"),
    ("Charles Dickens", "Oliver Twist", 730, "oliver twist"),
    ("Charles Dickens", "Great Expectations", 1400, "great expectations"),
    ("Charles Dickens", "A Tale of Two Cities", 98, "two cities"),
    ("George Eliot", "Silas Marner", 550, "silas marner"),
    ("George Eliot", "Middlemarch", 145, "middlemarch"),
    ("George Eliot", "The Mill on the Floss", 6688, "mill on the floss"),
    ("Thomas Hardy", "Far from the Madding Crowd", 107, "madding crowd"),
    ("Thomas Hardy", "Tess of the d'Urbervilles", 110, "tess"),
    ("Thomas Hardy", "Jude the Obscure", 153, "jude"),
    ("Charlotte Bronte", "Jane Eyre", 1260, "jane eyre"),
    ("Charlotte Bronte", "Villette", 9182, "villette"),
    ("Charlotte Bronte", "Shirley", 30486, "shirley"),
    ("Anne Bronte", "Agnes Grey", 767, "agnes grey"),
    ("Anne Bronte", "The Tenant of Wildfell Hall", 969, "wildfell"),
]

URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"


def file_name(author, title):
    # "Tess of the d'Urbervilles" is fine in a file name; only / and : are not.
    return f"{author} - {title}.txt".replace("/", "-").replace(":", "")


def title_line(text):
    """The 'Title:' line from the Gutenberg header, lowercased ('' if none)."""
    for line in text[:5000].splitlines():
        if line.lower().startswith("title:"):
            return line[6:].strip().lower()
    return ""


def check(text, keyword):
    """True if the downloaded file's Title: line contains the expected word."""
    return keyword in title_line(text)


def fetch(n):
    req = urllib.request.Request(URL.format(n=n), headers={"User-Agent": "authorship-class-project"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8-sig", errors="replace")


def main(folder, books=BOOKS, get=fetch, pause=2):
    os.makedirs(folder, exist_ok=True)
    ok, problems = 0, []
    for k, (author, title, n, keyword) in enumerate(books, start=1):
        path = os.path.join(folder, file_name(author, title))
        label = f"[{k}/{len(books)}] {author} - {title}"
        if os.path.exists(path):
            print(f"{label}: already there, skipped")
            ok += 1
            continue
        try:
            text = get(n)
        except Exception as e:
            problems.append(f"{label}: download failed ({e})")
            print(problems[-1])
            continue
        if not check(text, keyword):
            problems.append(f"{label}: WRONG BOOK? Title line says '{title_line(text)}' (not saved)")
            print(problems[-1])
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        ok += 1
        print(f"{label}: saved ({len(text.split()):,} words incl. header)")
        if pause:
            time.sleep(pause)
    print(f"\n{ok} of {len(books)} books ready in '{folder}'.")
    if problems:
        print("Problems:")
        for p in problems:
            print("  " + p)
    print("Next: open each file and delete its CONTENTS page (and any preface by an editor).")
    return ok, problems


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 download_books.py FOLDER")
    main(sys.argv[1])
