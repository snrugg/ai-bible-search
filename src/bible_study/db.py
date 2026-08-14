"""SQLite storage for Bible verses, chapter summaries, and book summaries."""

from __future__ import annotations

import sqlite3
from contextlib import closing
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

-- Vector-search chunk metadata.  Deliberately kept here, in plain SQLite,
-- rather than inside the sqlite-vec virtual tables: init_db() runs on a
-- connection with no extension loaded, and every read below this line has
-- to keep working when sqlite-vec is unavailable.  Only the vectors
-- themselves live in vec0 tables (see vectors.py), joined back on chunks.id.
--
-- chapter/verse_start/verse_end are NOT NULL DEFAULT 0 rather than nullable
-- because SQLite treats NULLs as distinct in a UNIQUE index -- a nullable
-- verse_start would let every re-index insert duplicate summary rows
-- instead of updating them.
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    tier TEXT NOT NULL,
    book_name TEXT NOT NULL,
    chapter INTEGER NOT NULL DEFAULT 0,
    verse_start INTEGER NOT NULL DEFAULT 0,
    verse_end INTEGER NOT NULL DEFAULT 0,
    citation TEXT NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    embedded_hash TEXT NOT NULL DEFAULT '',
    embed_model TEXT NOT NULL DEFAULT '',
    embed_dims INTEGER NOT NULL DEFAULT 0,
    embedded_at TIMESTAMP,
    UNIQUE(tier, book_name, chapter, verse_start)
);

CREATE INDEX IF NOT EXISTS idx_chunks_tier ON chunks(tier);

CREATE INDEX IF NOT EXISTS idx_chunks_book_chapter
    ON chunks(book_name, chapter);

CREATE TABLE IF NOT EXISTS vec_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def init_db(path: Path) -> None:
    """Create the database schema if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(path))) as conn:
        conn.executescript(INIT_SQL)
        conn.commit()


def upsert_verses(
    path: Path,
    book_name: str,
    chapter_num: int,
    verses: list[tuple[int, str]],
) -> None:
    """Insert or replace verse texts for a book/chapter pair."""
    abbrev = book_name[:3].upper()
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
        conn.execute(
              "INSERT OR REPLACE INTO book_summaries "
              "(book_name, book_abbrev, summary) VALUES (?,?,?)",
              (book_name, abbrev, summary),
          )
        conn.commit()


def get_book_summary(path: Path, book_name: str) -> str | None:
    """Return the stored book-level summary, or None if missing."""
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(
              "SELECT summary FROM book_summaries WHERE book_name=?",
              (book_name,),
          )
        row = cursor.fetchone()
        return row[0] if row is not None else None


def get_stored_chapters(path: Path, book_name: str) -> list[int]:
    """Return chapter numbers that have verse text stored, in order."""
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(
              "SELECT DISTINCT chapter FROM verses "
              "WHERE book_name=? ORDER BY chapter",
              (book_name,),
          )
        return [row[0] for row in cursor.fetchall()]


def clear_chapter_summaries(path: Path, book_name: str | None = None) -> int:
    """Delete chapter summaries (all, or just one book's). Returns row count."""
    with closing(sqlite3.connect(str(path))) as conn:
        if book_name is not None:
            cursor = conn.execute(
                  "DELETE FROM chapter_summaries WHERE book_name=?",
                  (book_name,),
              )
        else:
            cursor = conn.execute("DELETE FROM chapter_summaries")
        conn.commit()
        return cursor.rowcount


def clear_book_summaries(path: Path, book_name: str | None = None) -> int:
    """Delete book-level summaries (all, or just one book's). Returns row count."""
    with closing(sqlite3.connect(str(path))) as conn:
        if book_name is not None:
            cursor = conn.execute(
                  "DELETE FROM book_summaries WHERE book_name=?",
                  (book_name,),
              )
        else:
            cursor = conn.execute("DELETE FROM book_summaries")
        conn.commit()
        return cursor.rowcount


def get_all_book_names(path: Path) -> list[str]:
    """Return distinct book names that have verses stored."""
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(
              "SELECT DISTINCT book_name FROM verses ORDER BY book_name",
          )
        return [row[0] for row in cursor.fetchall()]


def get_chapter_summaries_for_book(
    path: Path, book_name: str,
) -> list[dict]:
    """Return chapter summaries for book, ordered by chapter."""
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
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
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(
                'SELECT DISTINCT book_name FROM book_summaries ORDER BY book_name',
            )
        return [row[0] for row in cursor.fetchall()]


# -- Vector-search chunk storage ------------------------------------------ #

#: Columns returned by the chunk read helpers, in order.
_CHUNK_COLUMNS = (
    "id", "tier", "book_name", "chapter", "verse_start", "verse_end",
    "citation", "text", "text_hash",
)

_CHUNK_SELECT = (
    "SELECT id, tier, book_name, chapter, verse_start, verse_end, "
    "citation, text, text_hash FROM chunks"
)


def _chunk_row(row) -> dict:
    """Turn a ``_CHUNK_SELECT`` row into a dict."""
    return dict(zip(_CHUNK_COLUMNS, row, strict=True))


def upsert_chunk(
    path: Path,
    tier: str,
    book_name: str,
    chapter: int,
    verse_start: int,
    verse_end: int,
    citation: str,
    text: str,
    text_hash: str,
) -> int:
    """Insert or update one chunk, returning its stable row id.

    Deliberately NOT ``INSERT OR REPLACE``, unlike every other write in this
    module: REPLACE deletes the conflicting row and inserts a new one with a
    fresh rowid, which would orphan the matching vec0 vector and silently
    leave the index pointing at nothing.  ``ON CONFLICT ... DO UPDATE``
    keeps the id stable.  ``embedded_hash`` is left untouched so that
    unchanged text stays marked as embedded -- that is what makes
    re-embedding incremental.
    """
    with closing(sqlite3.connect(str(path))) as conn:
        conn.execute(
              "INSERT INTO chunks "
              "(tier, book_name, chapter, verse_start, verse_end, "
              " citation, text, text_hash) "
              "VALUES (?,?,?,?,?,?,?,?) "
              "ON CONFLICT(tier, book_name, chapter, verse_start) "
              "DO UPDATE SET verse_end=excluded.verse_end, "
              "citation=excluded.citation, text=excluded.text, "
              "text_hash=excluded.text_hash",
              (tier, book_name, chapter, verse_start, verse_end,
               citation, text, text_hash),
          )
        conn.commit()
        cursor = conn.execute(
              "SELECT id FROM chunks WHERE tier=? AND book_name=? "
              "AND chapter=? AND verse_start=?",
              (tier, book_name, chapter, verse_start),
          )
        return cursor.fetchone()[0]


def get_stale_chunks(
    path: Path,
    embed_model: str,
    embed_dims: int,
    limit: int | None = None,
) -> list[dict]:
    """Return chunks that need embedding, oldest id first.

    A chunk is stale when it has never been embedded, when its text changed
    since it was embedded, or when it was embedded by a different model or
    at a different width.
    """
    sql = (
        f"{_CHUNK_SELECT} WHERE embedded_hash='' OR embedded_hash<>text_hash "
        "OR embed_model<>? OR embed_dims<>? ORDER BY id"
    )
    params: list = [embed_model, embed_dims]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(sql, tuple(params))
        return [_chunk_row(row) for row in cursor.fetchall()]


def get_chunks_by_ids(path: Path, ids: list[int]) -> dict[int, dict]:
    """Return ``{chunk_id: chunk_dict}`` for the given ids."""
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(
              f"{_CHUNK_SELECT} WHERE id IN ({placeholders})",
              tuple(int(i) for i in ids),
          )
        return {row[0]: _chunk_row(row) for row in cursor.fetchall()}


def mark_chunks_embedded(
    path: Path,
    ids: list[int],
    embed_model: str,
    embed_dims: int,
) -> int:
    """Stamp chunks as embedded by *embed_model* at *embed_dims*.

    Call this only after the vectors themselves are committed: a crash
    between the two leaves the rows stale, so the next run redoes them.
    """
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(
              f"UPDATE chunks SET embedded_hash=text_hash, embed_model=?, "
              f"embed_dims=?, embedded_at=CURRENT_TIMESTAMP "
              f"WHERE id IN ({placeholders})",
              (embed_model, embed_dims, *[int(i) for i in ids]),
          )
        conn.commit()
        return cursor.rowcount


def chunk_counts(path: Path) -> dict[str, tuple[int, int]]:
    """Return ``{tier: (total_chunks, embedded_chunks)}``."""
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(
              "SELECT tier, COUNT(*), "
              "SUM(CASE WHEN embedded_hash<>'' AND embedded_hash=text_hash "
              "THEN 1 ELSE 0 END) "
              "FROM chunks GROUP BY tier ORDER BY tier",
          )
        return {row[0]: (row[1], row[2] or 0) for row in cursor.fetchall()}


def clear_chunks(path: Path) -> int:
    """Delete every chunk row. Returns the number deleted."""
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute("DELETE FROM chunks")
        conn.commit()
        return cursor.rowcount


def get_meta(path: Path, key: str) -> str | None:
    """Return a vec_meta value, or None when unset."""
    with closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute(
              "SELECT value FROM vec_meta WHERE key=?", (key,),
          )
        row = cursor.fetchone()
        return row[0] if row else None


def set_meta(path: Path, key: str, value: str) -> None:
    """Insert or overwrite a vec_meta value."""
    with closing(sqlite3.connect(str(path))) as conn:
        conn.execute(
              "INSERT OR REPLACE INTO vec_meta (key, value) VALUES (?,?)",
              (key, str(value)),
          )
        conn.commit()
