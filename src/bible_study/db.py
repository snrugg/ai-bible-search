"""SQLite storage for Bible verses, chapter summaries, and book summaries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

INIT_SQL = """
CREATE TABLE IF NOT EXISTS verses (
    id INTEGER PRIMARY KEY,
    book_name TEXT NOT NULL,
    book_abbrev TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(book_name, chapter, verse)
);

CREATE INDEX IF NOT EXISTS idx_verses_book_chapter
    ON verses(book_name, chapter);

CREATE TABLE IF NOT EXISTS chapter_summaries (
    id INTEGER PRIMARY KEY,
    book_name TEXT NOT NULL,
    book_abbrev TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    summary TEXT NOT NULL,
    prompt_used TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(book_name, chapter)
);

CREATE TABLE IF NOT EXISTS book_summaries (
    id INTEGER PRIMARY KEY,
    book_name TEXT UNIQUE NOT NULL,
    book_abbrev TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(path: Path) -> None:
    """Create the database schema if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(INIT_SQL)


def upsert_verses(
    path: Path,
    book_name: str,
    chapter_num: int,
    verses: list[tuple[int, str]],
) -> None:
    """Insert or replace verse texts for a book/chapter pair."""
    abbrev = book_name[:3].upper()
    with sqlite3.connect(str(path)) as conn:
        for verse_num, text in verses:
            conn.execute(
                  "INSERT OR REPLACE INTO verses "
                  "(book_name, book_abbrev, chapter, verse, text) "
                  "VALUES (?,?,?,?,?)",
                  (book_name, abbrev, chapter_num, verse_num, text),
              )
        conn.commit()


def get_verses(
    path: Path, book_name: str, chapter_num: int
) -> list[dict]:
    """Return all verses for a book/chapter."""
    with sqlite3.connect(str(path)) as conn:
        cursor = conn.execute(
              "SELECT verse, text FROM verses "
              "WHERE book_name=? AND chapter=? "
              "ORDER BY verse",
              (book_name, chapter_num),
          )
        return [{"verse": row[0], "text": row[1]} for row in cursor.fetchall()]


def get_chapter_text(path: Path, book_name: str, chapter_num: int) -> str:
    """Concatenate all verse texts for a chapter."""
    verses = get_verses(path, book_name, chapter_num)
    return " ".join(v["text"] for v in verses)


def verse_count(path: Path, book_name: str | None = None) -> int:
    """Count total verses; optionally filtered to book."""
    with sqlite3.connect(str(path)) as conn:
        if book_name is not None:
            cursor = conn.execute(
                  "SELECT COUNT(*) FROM verses WHERE book_name=?",
                  (book_name,),
              )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM verses")
        return cursor.fetchone()[0]


def has_summary(path: Path, book_name: str, chapter_num: int) -> bool:
    """Return True if a summary exists for this chapter."""
    with sqlite3.connect(str(path)) as conn:
        cursor = conn.execute(
              "SELECT COUNT(*) FROM chapter_summaries "
              "WHERE book_name=? AND chapter=?",
              (book_name, chapter_num),
          )
        return cursor.fetchone()[0] > 0


def save_summary(
    path: Path,
    book_name: str,
    chapter_num: int,
    summary_text: str,
    prompt_used: str = '',
) -> None:
    """Store (or replace) a chapter summary."""
    abbrev = book_name[:3].upper()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
              "INSERT OR REPLACE INTO chapter_summaries "
              "(book_name, book_abbrev, chapter, summary, prompt_used) "
              "VALUES (?,?,?,?,?)",
              (book_name, abbrev, chapter_num, summary_text, prompt_used),
          )
        conn.commit()


def get_summary(
    path: Path, book_name: str, chapter_num: int
) -> str | None:
    """Return a stored summary, or None if missing."""
    with sqlite3.connect(str(path)) as conn:
        cursor = conn.execute(
              "SELECT summary FROM chapter_summaries "
              "WHERE book_name=? AND chapter=?",
              (book_name, chapter_num),
          )
        row = cursor.fetchone()
        return row[0] if row is not None else None


def get_unsummarized_chapters(
    path: Path, book_names: list[str],
) -> list[tuple[str, int]]:
    """Return (book, chapter) pairs that have no summary yet."""
    with sqlite3.connect(str(path)) as conn:
        verses = conn.execute(
              "SELECT DISTINCT book_name, chapter FROM verses",
          ).fetchall()
        summarized = {
              (row[0], row[1])
             for row in conn.execute(
                   "SELECT book_name, chapter FROM chapter_summaries",
               ).fetchall()
          }
        result = [
              (v[0], v[1])
             for v in verses
             if (v[0], v[1]) not in summarized
             and v[0] in book_names
          ]
    return sorted(result)


def save_book_summary(
    path: Path, book_name: str, abbrev: str, summary: str,
) -> None:
    """Store a book-level aggregate summary."""
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
              "INSERT OR REPLACE INTO book_summaries "
              "(book_name, book_abbrev, summary) VALUES (?,?,?)",
              (book_name, abbrev, summary),
          )
        conn.commit()


def get_all_book_names(path: Path) -> list[str]:
    """Return distinct book names that have verses stored."""
    with sqlite3.connect(str(path)) as conn:
        cursor = conn.execute(
              "SELECT DISTINCT book_name FROM verses ORDER BY book_name",
          )
        return [row[0] for row in cursor.fetchall()]


def get_chapter_summaries_for_book(
    path: Path, book_name: str,
) -> list[dict]:
    """Return chapter summaries for book, ordered by chapter."""
    with sqlite3.connect(str(path)) as conn:
        cursor = conn.execute(
              "SELECT chapter, summary FROM chapter_summaries "
              "WHERE book_name=? ORDER BY chapter",
              (book_name,),
          )
        return [{"chapter": row[0], "summary": row[1]}
                for row in cursor.fetchall()]


def get_chapter_progress(
    path: Path, book_names: list[str],
) -> tuple[int, int]:
    """Return (total_chapters, summarized_chapters) for books."""
    total = 0
    summed = 0
    with sqlite3.connect(str(path)) as conn:
        verses = conn.execute(
              "SELECT DISTINCT book_name, chapter FROM verses",
          ).fetchall()
        for bname, chap in verses:
            if bname not in book_names:
                continue
            total += 1
            cursor = conn.execute(
                  "SELECT COUNT(*) FROM chapter_summaries "
                  "WHERE book_name=? AND chapter=?",
                  (bname, chap),
              )
            if cursor.fetchone()[0] > 0:
                summed += 1
    return total, summed


# Aliases for test compatibility
get_verses_for_chapter = get_verses


def get_saved_books(path: Path) -> list[str]:
    """Return distinct book names from book_summaries table."""
    with sqlite3.connect(str(path)) as conn:
        cursor = conn.execute(
                'SELECT DISTINCT book_name FROM book_summaries ORDER BY book_name',
            )
        return [row[0] for row in cursor.fetchall()]
