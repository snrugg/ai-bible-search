"""Bible indexer — canonical structure of the KJV.

The KJV has 66 books across the Old and New Testaments with ~1,189 chapters total.
This module holds the stable metadata (name, abbreviation, testament, chapter count)
for each book as a pure-data lookup — no I/O required. Book names match the
canonical order used by bible-api.com and other standard KJV sources.
"""

from __future__ import annotations

BIBLE_BOOKS: tuple[dict[str, str | int], ...] = (
    # ─── Old Testament (39 books) ───────────────────────────────────────────
    {"name": "Genesis",           "abbrev": "GEN", "testament": "OT", "chapter_count": 50},
    {"name": "Exodus",            "abbrev": "EXO", "testament": "OT", "chapter_count": 40},
    {"name": "Leviticus",         "abbrev": "LEV", "testament": "OT", "chapter_count": 27},
    {"name": "Numbers",           "abbrev": "NUM", "testament": "OT", "chapter_count": 36},
    {"name": "Deuteronomy",       "abbrev": "DEU", "testament": "OT", "chapter_count": 34},
    {"name": "Joshua",            "abbrev": "JOS", "testament": "OT", "chapter_count": 24},
    {"name": "Judges",            "abbrev": "JDG", "testament": "OT", "chapter_count": 21},
    {"name": "Ruth",              "abbrev": "RUT", "testament": "OT", "chapter_count":  4},
    {"name": "1 Samuel",          "abbrev": "1SA", "testament": "OT", "chapter_count": 31},
    {"name": "2 Samuel",          "abbrev": "2SA", "testament": "OT", "chapter_count": 24},
    {"name": "1 Kings",           "abbrev": "1KI", "testament": "OT", "chapter_count": 22},
    {"name": "2 Kings",           "abbrev": "2KI", "testament": "OT", "chapter_count": 25},
    {"name": "1 Chronicles",      "abbrev": "1CH", "testament": "OT", "chapter_count": 29},
    {"name": "2 Chronicles",      "abbrev": "2CH", "testament": "OT", "chapter_count": 36},
    {"name": "Ezra",              "abbrev": "EZR", "testament": "OT", "chapter_count": 10},
    {"name": "Nehemiah",          "abbrev": "NEH", "testament": "OT", "chapter_count": 13},
    {"name": "Esther",            "abbrev": "EST", "testament": "OT", "chapter_count": 10},
    {"name": "Job",               "abbrev": "JOB", "testament": "OT", "chapter_count": 42},
    {"name": "Psalms",            "abbrev": "PSA", "testament": "OT", "chapter_count": 150},
    {"name": "Proverbs",          "abbrev": "PRO", "testament": "OT", "chapter_count": 31},
    {"name": "Ecclesiastes",      "abbrev": "ECC", "testament": "OT", "chapter_count": 12},
    {"name": "Song of Solomon",   "abbrev": "SOS", "testament": "OT", "chapter_count":  8},
    {"name": "Isaiah",            "abbrev": "ISA", "testament": "OT", "chapter_count": 66},
    {"name": "Jeremiah",          "abbrev": "JER", "testament": "OT", "chapter_count": 52},
    {"name": "Lamentations",      "abbrev": "LAM", "testament": "OT", "chapter_count":  5},
    {"name": "Ezekiel",           "abbrev": "EZK", "testament": "OT", "chapter_count": 48},
    {"name": "Daniel",            "abbrev": "DAN", "testament": "OT", "chapter_count": 12},
    {"name": "Hosea",             "abbrev": "HOS", "testament": "OT", "chapter_count": 14},
    {"name": "Joel",              "abbrev": "JOL", "testament": "OT", "chapter_count":  3},
    {"name": "Amos",              "abbrev": "AMO", "testament": "OT", "chapter_count":  9},
    {"name": "Obadiah",           "abbrev": "OBA", "testament": "OT", "chapter_count":  1},
    {"name": "Jonah",             "abbrev": "JON", "testament": "OT", "chapter_count":  4},
    {"name": "Micah",             "abbrev": "MIC", "testament": "OT", "chapter_count":  7},
    {"name": "Nahum",             "abbrev": "NAH", "testament": "OT", "chapter_count":  3},
    {"name": "Habakkuk",          "abbrev": "HAB", "testament": "OT", "chapter_count":  3},
    {"name": "Zephaniah",         "abbrev": "ZEP", "testament": "OT", "chapter_count":  3},
    {"name": "Haggai",            "abbrev": "HAG", "testament": "OT", "chapter_count":  2},
    {"name": "Zechariah",         "abbrev": "ZEC", "testament": "OT", "chapter_count": 14},
    {"name": "Malachi",           "abbrev": "MAL", "testament": "OT", "chapter_count":  4},

    # ─── New Testament (27 books) ───────────────────────────────────────────
    {"name": "Matthew",           "abbrev": "MAT", "testament": "NT", "chapter_count": 28},
    {"name": "Mark",              "abbrev": "MRK", "testament": "NT", "chapter_count": 16},
    {"name": "Luke",              "abbrev": "LUK", "testament": "NT", "chapter_count": 24},
    {"name": "John",              "abbrev": "JHN", "testament": "NT", "chapter_count": 21},
    {"name": "Acts",              "abbrev": "ACT", "testament": "NT", "chapter_count": 28},
    {"name": "Romans",            "abbrev": "ROM", "testament": "NT", "chapter_count": 16},
    {"name": "1 Corinthians",     "abbrev": "1CO", "testament": "NT", "chapter_count": 16},
    {"name": "2 Corinthians",     "abbrev": "2CO", "testament": "NT", "chapter_count": 13},
    {"name": "Galatians",         "abbrev": "GAL", "testament": "NT", "chapter_count":  6},
    {"name": "Ephesians",         "abbrev": "EPH", "testament": "NT", "chapter_count":  6},
    {"name": "Philippians",       "abbrev": "PHP", "testament": "NT", "chapter_count":  4},
    {"name": "Colossians",        "abbrev": "COL", "testament": "NT", "chapter_count":  4},
    {"name": "1 Thessalonians",   "abbrev": "1TH", "testament": "NT", "chapter_count":  5},
    {"name": "2 Thessalonians",   "abbrev": "2TH", "testament": "NT", "chapter_count":  3},
    {"name": "1 Timothy",         "abbrev": "1TI", "testament": "NT", "chapter_count":  6},
    {"name": "2 Timothy",         "abbrev": "2TI", "testament": "NT", "chapter_count":  4},
    {"name": "Titus",             "abbrev": "TIT", "testament": "NT", "chapter_count":  3},
    {"name": "Philemon",          "abbrev": "PHM", "testament": "NT", "chapter_count":  1},
    {"name": "Hebrews",           "abbrev": "HEB", "testament": "NT", "chapter_count": 13},
    {"name": "James",             "abbrev": "JAS", "testament": "NT", "chapter_count":  5},
    {"name": "1 Peter",           "abbrev": "1PE", "testament": "NT", "chapter_count":  5},
    {"name": "2 Peter",           "abbrev": "2PE", "testament": "NT", "chapter_count":  3},
    {"name": "1 John",            "abbrev": "1JH", "testament": "NT", "chapter_count":  5},
    {"name": "2 John",            "abbrev": "2JH", "testament": "NT", "chapter_count":  1},
    {"name": "3 John",            "abbrev": "3JH", "testament": "NT", "chapter_count":  1},
    {"name": "Jude",              "abbrev": "JUD", "testament": "NT", "chapter_count":  1},
    {"name": "Revelation",        "abbrev": "REV", "testament": "NT", "chapter_count": 22},
)

# ─── Public API ─────────────────────────────────────────────────────────────


def total_books() -> int:
    """Return the number of books in the Bible."""
    return len(BIBLE_BOOKS)


def total_chapters() -> int:
    """Return the total number of chapters across all 66 books."""
    return sum(b["chapter_count"] for b in BIBLE_BOOKS)


def book_names(testament: str | None = None) -> list[str]:
    """Return ordered list of book names, optionally filtered by testament ('OT'/'NT')."""
    if testament is None:
        return [b["name"] for b in BIBLE_BOOKS]
    return [b["name"] for b in BIBLE_BOOKS if b["testament"] == testament]


def get_book(name: str) -> dict[str, str | int] | None:
    """Return the book dict for *name*, or ``None`` if not found (case-insensitive)."""
    name_lower = name.lower()
    for book in BIBLE_BOOKS:
        if book["name"].lower() == name_lower:
            return book
    return None


def get_chapters(name: str) -> list[int]:
    """Return a list of chapter numbers for *name* (1-based, contiguous)."""
    book = get_book(name)
    if book is None:
        msg = f"Unknown book: {name!r}"
        raise ValueError(msg)
    return list(range(1, book["chapter_count"] + 1))


def iter_books(testament: str | None = None):
    """Yield book dicts in canonical order, optionally filtered by testament."""
    for book in BIBLE_BOOKS:
        if testament is None or book["testament"] == testament:
            yield book
