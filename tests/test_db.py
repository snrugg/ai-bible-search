"""Tests for bible_study/db -- SQLite schema and CRUD operations."""

from contextlib import closing

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
        with closing(sqlite3.connect(str(db_path))) as conn:
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


class TestChunkSchema:
    """The chunks and vec_meta tables ship in INIT_SQL, extension-free."""

    def test_init_db_creates_chunks_table(self, db_path):
        import sqlite3
        with closing(sqlite3.connect(str(db_path))) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'",
            ).fetchone()
        assert row is not None

    def test_init_db_creates_vec_meta_table(self, db_path):
        import sqlite3
        with closing(sqlite3.connect(str(db_path))) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_meta'",
            ).fetchone()
        assert row is not None

    def test_init_db_is_idempotent_for_chunks(self, db_path):
        from bible_study.db import upsert_chunk
        upsert_chunk(db_path, "book", "Genesis", 0, 0, 0, "Genesis", "t", "h")
        init_db(db_path)
        from bible_study.db import chunk_counts
        assert chunk_counts(db_path)["book"] == (1, 0)


class TestChunkUpsert:

    def test_upsert_returns_an_id(self, db_path):
        from bible_study.db import upsert_chunk
        cid = upsert_chunk(
            db_path, "verse", "Genesis", 1, 1, 5, "Genesis 1:1-5", "text", "h1",
        )
        assert isinstance(cid, int)

    def test_upsert_keeps_the_id_stable_on_update(self, db_path):
        """INSERT OR REPLACE would allocate a new rowid and orphan the vector."""
        from bible_study.db import get_chunks_by_ids, upsert_chunk
        first = upsert_chunk(
            db_path, "verse", "Genesis", 1, 1, 5, "Genesis 1:1-5", "old", "h1",
        )
        second = upsert_chunk(
            db_path, "verse", "Genesis", 1, 1, 5, "Genesis 1:1-5", "new", "h2",
        )
        assert second == first
        assert get_chunks_by_ids(db_path, [first])[first]["text"] == "new"

    def test_book_tier_chunk_is_unique(self, db_path):
        """Sentinel 0 columns, not NULL -- NULLs are distinct in a UNIQUE index."""
        from bible_study.db import chunk_counts, upsert_chunk
        upsert_chunk(db_path, "book", "Genesis", 0, 0, 0, "Genesis", "a", "h1")
        upsert_chunk(db_path, "book", "Genesis", 0, 0, 0, "Genesis", "b", "h2")
        assert chunk_counts(db_path)["book"] == (1, 0)

    def test_chapter_tier_chunk_is_unique(self, db_path):
        from bible_study.db import chunk_counts, upsert_chunk
        upsert_chunk(db_path, "chapter", "Genesis", 1, 0, 0, "Genesis 1", "a", "h1")
        upsert_chunk(db_path, "chapter", "Genesis", 1, 0, 0, "Genesis 1", "b", "h2")
        assert chunk_counts(db_path)["chapter"] == (1, 0)

    def test_different_tiers_do_not_collide(self, db_path):
        from bible_study.db import chunk_counts, upsert_chunk
        upsert_chunk(db_path, "book", "Genesis", 0, 0, 0, "Genesis", "a", "h1")
        upsert_chunk(db_path, "chapter", "Genesis", 0, 0, 0, "Genesis", "b", "h2")
        counts = chunk_counts(db_path)
        assert counts["book"][0] == 1
        assert counts["chapter"][0] == 1


class TestStaleChunks:

    def _chunk(self, db_path, text_hash="h1", verse_start=1):
        from bible_study.db import upsert_chunk
        return upsert_chunk(
            db_path, "verse", "Genesis", 1, verse_start, 5,
            "Genesis 1:1-5", "text", text_hash,
        )

    def test_never_embedded_is_stale(self, db_path):
        from bible_study.db import get_stale_chunks
        self._chunk(db_path)
        assert len(get_stale_chunks(db_path, "m", 4)) == 1

    def test_current_chunk_is_not_stale(self, db_path):
        from bible_study.db import get_stale_chunks, mark_chunks_embedded
        cid = self._chunk(db_path)
        mark_chunks_embedded(db_path, [cid], "m", 4)
        assert get_stale_chunks(db_path, "m", 4) == []

    def test_changed_text_is_stale(self, db_path):
        from bible_study.db import get_stale_chunks, mark_chunks_embedded
        cid = self._chunk(db_path, text_hash="h1")
        mark_chunks_embedded(db_path, [cid], "m", 4)
        self._chunk(db_path, text_hash="h2")
        assert len(get_stale_chunks(db_path, "m", 4)) == 1

    def test_model_change_is_stale(self, db_path):
        from bible_study.db import get_stale_chunks, mark_chunks_embedded
        cid = self._chunk(db_path)
        mark_chunks_embedded(db_path, [cid], "old-model", 4)
        assert len(get_stale_chunks(db_path, "new-model", 4)) == 1

    def test_dims_change_is_stale(self, db_path):
        from bible_study.db import get_stale_chunks, mark_chunks_embedded
        cid = self._chunk(db_path)
        mark_chunks_embedded(db_path, [cid], "m", 4)
        assert len(get_stale_chunks(db_path, "m", 8)) == 1

    def test_limit_is_honoured(self, db_path):
        from bible_study.db import get_stale_chunks
        for start in (1, 4, 7):
            self._chunk(db_path, verse_start=start)
        assert len(get_stale_chunks(db_path, "m", 4, limit=2)) == 2

    def test_stale_chunks_are_ordered_by_id(self, db_path):
        from bible_study.db import get_stale_chunks
        for start in (1, 4, 7):
            self._chunk(db_path, verse_start=start)
        ids = [c["id"] for c in get_stale_chunks(db_path, "m", 4)]
        assert ids == sorted(ids)

    def test_mark_embedded_returns_row_count(self, db_path):
        from bible_study.db import mark_chunks_embedded
        cid = self._chunk(db_path)
        assert mark_chunks_embedded(db_path, [cid], "m", 4) == 1

    def test_mark_embedded_with_no_ids(self, db_path):
        from bible_study.db import mark_chunks_embedded
        assert mark_chunks_embedded(db_path, [], "m", 4) == 0


class TestChunkReads:

    def test_get_chunks_by_ids_returns_a_dict(self, db_path):
        from bible_study.db import get_chunks_by_ids, upsert_chunk
        cid = upsert_chunk(
            db_path, "verse", "Genesis", 1, 1, 5, "Genesis 1:1-5", "text", "h",
        )
        found = get_chunks_by_ids(db_path, [cid])
        assert found[cid]["citation"] == "Genesis 1:1-5"
        assert found[cid]["book_name"] == "Genesis"
        assert found[cid]["verse_end"] == 5

    def test_get_chunks_by_ids_with_empty_list(self, db_path):
        from bible_study.db import get_chunks_by_ids
        assert get_chunks_by_ids(db_path, []) == {}

    def test_get_chunks_by_ids_ignores_unknown_ids(self, db_path):
        from bible_study.db import get_chunks_by_ids
        assert get_chunks_by_ids(db_path, [9999]) == {}

    def test_chunk_counts_splits_embedded(self, db_path):
        from bible_study.db import chunk_counts, mark_chunks_embedded, upsert_chunk
        a = upsert_chunk(db_path, "verse", "Genesis", 1, 1, 5, "c", "t", "h1")
        upsert_chunk(db_path, "verse", "Genesis", 1, 4, 8, "c", "t", "h2")
        mark_chunks_embedded(db_path, [a], "m", 4)
        assert chunk_counts(db_path)["verse"] == (2, 1)

    def test_chunk_counts_on_empty_db(self, db_path):
        from bible_study.db import chunk_counts
        assert chunk_counts(db_path) == {}

    def test_clear_chunks_removes_everything(self, db_path):
        from bible_study.db import chunk_counts, clear_chunks, upsert_chunk
        upsert_chunk(db_path, "verse", "Genesis", 1, 1, 5, "c", "t", "h")
        assert clear_chunks(db_path) == 1
        assert chunk_counts(db_path) == {}


class TestVecMeta:

    def test_get_meta_returns_none_when_unset(self, db_path):
        from bible_study.db import get_meta
        assert get_meta(db_path, "embed_model") is None

    def test_set_then_get_meta(self, db_path):
        from bible_study.db import get_meta, set_meta
        set_meta(db_path, "embed_model", "qwen3-embedding:0.6b")
        assert get_meta(db_path, "embed_model") == "qwen3-embedding:0.6b"

    def test_set_meta_overwrites(self, db_path):
        from bible_study.db import get_meta, set_meta
        set_meta(db_path, "embed_dims", "1024")
        set_meta(db_path, "embed_dims", "768")
        assert get_meta(db_path, "embed_dims") == "768"

    def test_set_meta_stringifies(self, db_path):
        from bible_study.db import get_meta, set_meta
        set_meta(db_path, "embed_dims", 1024)
        assert get_meta(db_path, "embed_dims") == "1024"
