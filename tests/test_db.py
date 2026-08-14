"""Tests for bible_study/db -- SQLite schema and CRUD operations."""

import pytest

from bible_study.db import (
    get_all_book_names,
    get_chapter_progress,
    get_chapter_summaries_for_book,
    get_chapter_text,
    get_summary,
    get_unsummarized_chapters,
    get_verses,
    get_verses_for_chapter,
    has_summary,
    init_db,
    save_book_summary,
    save_summary,
    upsert_verses,
    verse_count,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "bible.db"
    init_db(path)
    return path


class TestSchema:

    def test_init_creates_all_tables(self, db_path):
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            ).fetchall()
        table_names = {t[0] for t in tables}
        assert "verses" in table_names
        assert "chapter_summaries" in table_names
        assert "book_summaries" in table_names

    def test_init_is_idempotent(self, db_path):
        init_db(db_path)
        init_db(db_path)


class TestVerseCRUD:

    def test_upsert_and_get_verses(self, db_path):
        upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning..."), (2, "And the earth...")])
        verses = get_verses(db_path, "Genesis", 1)
        assert len(verses) == 2
        assert verses[0]["verse"] == 1
        assert verses[1]["verse"] == 2

    def test_upsert_overwrites_same_chapter(self, db_path):
        upsert_verses(db_path, "Genesis", 1, [(1, "first")])
        upsert_verses(db_path, "Genesis", 1, [(1, "second")])
        verses = get_verses(db_path, "Genesis", 1)
        assert len(verses) == 1
        assert verses[0]["text"] == "second"

    def test_get_chapter_text_concats_all(self, db_path):
        upsert_verses(db_path, "Genesis", 1, [(1, "hello"), (2, "world")])
        text = get_chapter_text(db_path, "Genesis", 1)
        assert "hello" in text
        assert "world" in text

    def test_verse_count_returns_total(self, db_path):
        upsert_verses(db_path, "Genesis", 1, [(1, "x")])
        upsert_verses(db_path, "Genesis", 1, [(2, "y")])
        assert verse_count(db_path, "Genesis") == 2


class TestChapterSummaries:

    def test_has_summary_false_before_save(self, db_path):
        assert has_summary(db_path, "Genesis", 1) is False

    def test_has_summary_true_after_save(self, db_path):
        save_summary(db_path, "Genesis", 1, "summary text", "prompt used")
        assert has_summary(db_path, "Genesis", 1) is True

    def test_save_and_get_summary(self, db_path):
        save_summary(db_path, "Genesis", 1, "my summary", "prompt")
        summary = get_summary(db_path, "Genesis", 1)
        assert summary == "my summary"

    def test_get_summary_returns_none_when_missing(self, db_path):
        assert get_summary(db_path, "Exodus", 99) is None

    def test_get_unsummarized_chapters(self, tmp_path):
        path = tmp_path / "test.db"
        init_db(path)
        for chap in range(1, 4):
            upsert_verses(path, "Genesis", chap, [(chap, f"text {chap}")])
        books = ["Genesis"]
        uns = get_unsummarized_chapters(path, books)
        assert len(uns) == 3
        chapters_in_result = [c for _, c in uns]
        for c in [1, 2, 3]:
            assert c in chapters_in_result


class TestBookSummaries:

    def test_save_and_get_book_summary(self, db_path):
        from bible_study.db import get_saved_books
        save_book_summary(db_path, "Genesis", "GEN", "book summary text")
        summary = get_saved_books(db_path)
        assert "Genesis" in summary

    def test_get_chapter_summaries_for_book(self, db_path):
        save_summary(db_path, "Genesis", 1, "summary one")
        save_summary(db_path, "Genesis", 2, "summary two")
        sums = get_chapter_summaries_for_book(db_path, "Genesis")
        assert len(sums) == 2
        assert all(k in sums[0] for k in ("chapter", "summary"))


class TestProgress:

    def test_get_chapter_progress(self, tmp_path):
        path = tmp_path / "test.db"
        init_db(path)
        upsert_verses(path, "Genesis", 1, [(1, "text")])
        save_summary(path, "Genesis", 1, "summary")
        books = ["Genesis"]
        total, summed = get_chapter_progress(path, books)
        assert total == 1
        assert summed == 1


class TestClearSummaries:

    def _seed(self, path):
        for book, abbrev in (("Genesis", "GEN"), ("Exodus", "EXO")):
            for chap in (1, 2):
                upsert_verses(path, book, chap, [(1, "verse text")])
                save_summary(path, book, chap, f"{book} {chap}")
            save_book_summary(path, book, abbrev, f"{book} overview")

    def test_clear_chapter_summaries_removes_all(self, db_path):
        from bible_study.db import clear_chapter_summaries
        self._seed(db_path)
        assert clear_chapter_summaries(db_path) == 4
        assert get_chapter_summaries_for_book(db_path, "Genesis") == []
        assert get_chapter_summaries_for_book(db_path, "Exodus") == []

    def test_clear_chapter_summaries_scoped_to_book(self, db_path):
        from bible_study.db import clear_chapter_summaries
        self._seed(db_path)
        assert clear_chapter_summaries(db_path, "Genesis") == 2
        assert get_chapter_summaries_for_book(db_path, "Genesis") == []
        assert len(get_chapter_summaries_for_book(db_path, "Exodus")) == 2

    def test_clear_chapter_summaries_keeps_verses(self, db_path):
        from bible_study.db import clear_chapter_summaries
        self._seed(db_path)
        clear_chapter_summaries(db_path)
        assert verse_count(db_path) == 4

    def test_clear_book_summaries_removes_all(self, db_path):
        from bible_study.db import clear_book_summaries, get_saved_books
        self._seed(db_path)
        assert clear_book_summaries(db_path) == 2
        assert get_saved_books(db_path) == []

    def test_clear_book_summaries_scoped_to_book(self, db_path):
        from bible_study.db import clear_book_summaries, get_saved_books
        self._seed(db_path)
        assert clear_book_summaries(db_path, "Genesis") == 1
        assert get_saved_books(db_path) == ["Exodus"]

    def test_clearing_empty_tables_returns_zero(self, db_path):
        from bible_study.db import clear_book_summaries, clear_chapter_summaries
        assert clear_chapter_summaries(db_path) == 0
        assert clear_book_summaries(db_path, "Genesis") == 0
