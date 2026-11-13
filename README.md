# Common Authorship Project (MTH 4322, Project 6)

Finds texts that may share an author by comparing writing style.

## Current status
Step 1 done (load and clean texts).
Step 2 done: sentence/word length, function words, punctuation, MATTR, character 3-grams.
Step 3 done: collection averages, spreads, z-scores, top 300 3-grams.
Steps 4-5 done: per-method scores, weighted overall score, breakdown.
Step 6 done: all pairs scored and labeled (likely 65+, possible 60+; set after the real-book test).
Step 7 done: hierarchical clustering (complete linkage by default, average available).
Step 8 done: groups (cut at the possible line, likely subgroups at the likely line) and notes for split pairs.
Step 9 built: Text collection, Compare texts, Authorship groups (with the tree).
Step 10: testing tools built (random-text generator, evaluation); real-book set comes in Phase 4.

## Folder layout
```
authorship/
  text_loader.py        Step 1: load, clean, split into words and sentences
  features.py           Step 2: style features (length, function words, punctuation, MATTR, 3-grams)
  collection.py         Step 3: compare each text to the collection (z-scores)
  scoring.py            Steps 4-5: pair scores, weighted score, breakdown
  pairs.py              Step 6: all pairs and their labels
  clustering.py         Step 7: hierarchical clustering, builds the merge list
  grouping.py           Step 8: cuts the tree into groups, lists split pairs
  app.py                Step 9: the web interface (Flask)
  tree_svg.py           Step 9: draws the tree as an SVG picture
  generator.py          Step 10 (testing tool): made-up texts by made-up authors
  evaluate.py           Step 10 (testing tool): counts mistakes on texts with known authors
  download_books.py     Phase 4 (testing tool): downloads the real known-author books
  templates/            Step 9: page layouts (HTML)
  static/style.css      Step 9: page styling
  tests/
    test_text_loader.py 38 tests for Step 1
    test_features.py    56 tests for Step 2
    test_collection.py  33 tests for Step 3
    test_scoring.py     29 tests for Steps 4-5
    test_pairs.py       11 tests for Step 6
    test_clustering.py  13 tests for Step 7
    test_grouping.py    19 tests for Step 8
    test_app_collection.py 21 tests for Step 9, Part 1
    test_app_compare.py 15 tests for Step 9, Part 2
    test_app_groups.py  11 tests for Step 9, Part 3
    test_tree_svg.py    12 tests for Step 9, Part 4
    test_tools.py       13 tests for Step 10
    test_download_books.py 6 tests for the download tool
  sample_texts/         two small sample files to try the loader on
```

## Requirements
Python 3.9 or newer, and Flask for the web interface. On a Mac with Homebrew
Python, install Flask inside a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate      (again in every new terminal)
python3 -m pip install -r requirements.txt
```

## Run the tests
From inside the `authorship` folder:
```
python -m unittest discover tests -v
```
Expected: `Ran 283 tests ... OK`

## Run the program
```
python3 app.py
```
Then open http://127.0.0.1:5000 in a browser. Press Ctrl+C in the terminal to stop it.

## Try the loader on a folder of .txt files
```
python text_loader.py sample_texts
```
Expected:
```
hartley_letter.txt: 1240 words, 80 sentences, included
postcard.txt: 8 words, 2 sentences, EXCLUDED - Too short: 8 words (at least 500 needed for a reliable comparison)
```
To test a real book, put a Project Gutenberg .txt file in a folder and run the same command on that folder.

## Testing tools (Step 10)
Make 5 made-up authors with 3 texts each, then check the program's answers:
```
python3 generator.py test_texts --authors 5 --texts 3 --seed 1
python3 evaluate.py test_texts
```
`--difference` (0.1 hard ... 1.0 easy) sets how strongly the made-up authors differ.

For real books, download the 17-book test set (needs internet; run on your own computer):
```
python3 download_books.py real_books
```
Then open each file and delete its contents page (the Gutenberg header and footer
are removed by the program automatically), and run `python3 evaluate.py real_books`.
