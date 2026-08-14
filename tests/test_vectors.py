"""Tests for bible_study/vectors -- chunking, embedding storage, and KNN.

These run against a real sqlite-vec extension (a declared dependency) but
with 4-dimensional toy vectors, so KNN executes real extension code with
no network and no embedding model.
"""

import sqlite3
from contextlib import closing

import pytest

from bible_study.db import (
    init_db,
    save_book_summary,
    save_summary,
    upsert_verses,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "bible.db"
    init_db(path)
    return path


@pytest.fixture
def seeded_db(db_path):
    """Two books: Genesis 1 (10 verses + summaries), Exodus 1 (verses only)."""
    upsert_verses(
        db_path, "Genesis", 1,
        [(i, f"Genesis verse {i}") for i in range(1, 11)],
    )
    save_summary(db_path, "Genesis", 1, "God creates the heavens and earth.")
    save_book_summary(db_path, "Genesis", "GEN", "The book of beginnings.")
    upsert_verses(db_path, "Exodus", 1, [(1, "Now these are the names.")])
    return db_path


def _verses(count, start=1):
    return [{"verse": i, "text": f"t{i}"} for i in range(start, start + count)]


def _fake_embed(vector):
    """Return an ollama.embed stand-in yielding *vector* for every input."""
    def _embed(texts, **kwargs):
        items = [texts] if isinstance(texts, str) else list(texts)
        return [list(vector) for _ in items]
    return _embed


def _tier_embed(mapping, default):
    """Return an embed stand-in that picks a vector by marker substring."""
    def _embed(texts, **kwargs):
        items = [texts] if isinstance(texts, str) else list(texts)
        out = []
        for text in items:
            for marker, vector in mapping.items():
                if marker in text:
                    out.append(list(vector))
                    break
            else:
                out.append(list(default))
        return out
    return _embed


class TestConnect:
    """Extension loading, and the two ways it can be unavailable."""

    def test_connect_loads_the_extension(self, db_path):
        from bible_study.vectors import _connect
        conn = _connect(db_path)
        try:
            version = conn.execute("SELECT vec_version()").fetchone()[0]
        finally:
            conn.close()
        assert isinstance(version, str)

    def test_connect_reports_a_missing_package(self, db_path, monkeypatch):
        import builtins
        from bible_study.vectors import VectorSupportError, _connect
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sqlite_vec":
                raise ImportError("no module named sqlite_vec")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(VectorSupportError, match="not installed"):
            _connect(db_path)

    def test_connect_reports_a_build_without_extension_support(
        self, db_path, mocker,
    ):
        from bible_study.vectors import VectorSupportError, _connect
        conn = mocker.MagicMock()
        conn.enable_load_extension.side_effect = AttributeError("nope")
        mocker.patch("sqlite3.connect", return_value=conn)
        with pytest.raises(VectorSupportError, match="loadable-extension"):
            _connect(db_path)
        conn.close.assert_called_once()

    def test_connect_reports_a_failed_load(self, db_path, mocker):
        import sqlite_vec
        from bible_study.vectors import VectorSupportError, _connect
        mocker.patch.object(
            sqlite_vec, "load", side_effect=RuntimeError("bad dylib"),
        )
        with pytest.raises(VectorSupportError, match="Failed to load"):
            _connect(db_path)


class TestInitVec:

    def test_creates_a_table_per_tier(self, db_path):
        from bible_study.vectors import _VEC_TABLES, init_vec
        init_vec(db_path, dims=4, model="fake")
        with closing(sqlite3.connect(str(db_path))) as conn:
            names = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                )
            }
        for table in _VEC_TABLES.values():
            assert table in names

    def test_is_idempotent(self, db_path):
        from bible_study.vectors import init_vec
        init_vec(db_path, dims=4, model="fake")
        init_vec(db_path, dims=4, model="fake")

    def test_records_model_and_dims(self, db_path):
        from bible_study.vectors import init_vec, vec_status
        init_vec(db_path, dims=4, model="fake")
        status = vec_status(db_path)
        assert status["embed_dims"] == "4"
        assert status["embed_model"] == "fake"
        assert status["distance_metric"] == "cosine"

    def test_rejects_a_dims_change(self, db_path):
        from bible_study.vectors import VectorIndexError, init_vec
        init_vec(db_path, dims=4, model="fake")
        with pytest.raises(VectorIndexError, match="--rebuild"):
            init_vec(db_path, dims=8, model="fake")

    def test_falls_back_when_cosine_is_unsupported(self, db_path, mocker):
        """Older sqlite-vec builds reject the distance_metric keyword."""
        from bible_study import vectors
        real_connect = vectors._connect

        class _NoCosine:
            """Proxy that rejects cosine DDL; Connection.execute is read-only."""

            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args):
                if "distance_metric=cosine" in sql:
                    raise sqlite3.OperationalError("unknown keyword")
                return self._conn.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        mocker.patch.object(
            vectors, "_connect", side_effect=lambda p: _NoCosine(real_connect(p)),
        )
        vectors.init_vec(db_path, dims=4, model="fake")
        assert vectors.vec_status(db_path)["distance_metric"] == "l2"

    def test_reads_dims_from_config_when_omitted(self, db_path, mocker):
        from bible_study import vectors
        mocker.patch("bible_study.prompts.get_embed_dims", return_value=4)
        mocker.patch("bible_study.prompts.get_embed_model", return_value="cfg")
        vectors.init_vec(db_path)
        assert vectors.vec_status(db_path)["embed_model"] == "cfg"


class TestCitation:

    def test_book_tier(self):
        from bible_study.vectors import citation
        assert citation("book", "Genesis") == "Genesis"

    def test_chapter_tier(self):
        from bible_study.vectors import citation
        assert citation("chapter", "Genesis", 3) == "Genesis 3"

    def test_verse_range(self):
        from bible_study.vectors import citation
        assert citation("verse", "Genesis", 1, 1, 5) == "Genesis 1:1-5"

    def test_single_verse(self):
        from bible_study.vectors import citation
        assert citation("verse", "Genesis", 1, 7, 7) == "Genesis 1:7"

    def test_multiword_book_name(self):
        from bible_study.vectors import citation
        assert citation("verse", "1 Samuel", 2, 1, 5) == "1 Samuel 2:1-5"


class TestBuildWindows:

    def test_empty_chapter_yields_nothing(self):
        from bible_study.vectors import build_windows
        assert build_windows("Genesis", 1, []) == []

    def test_short_chapter_yields_one_chunk(self):
        from bible_study.vectors import build_windows
        chunks = build_windows("Genesis", 1, _verses(3))
        assert len(chunks) == 1
        assert chunks[0]["verse_start"] == 1
        assert chunks[0]["verse_end"] == 3

    def test_exactly_one_window_yields_one_chunk(self):
        from bible_study.vectors import build_windows
        assert len(build_windows("Genesis", 1, _verses(5))) == 1

    def test_stride_produces_overlap(self):
        from bible_study.vectors import build_windows
        chunks = build_windows("Genesis", 1, _verses(10))
        assert [c["citation"] for c in chunks] == [
            "Genesis 1:1-5", "Genesis 1:4-8", "Genesis 1:6-10",
        ]

    def test_every_window_is_full_width_when_possible(self):
        from bible_study.vectors import build_windows
        for count in (6, 10, 11, 12, 33, 176):
            chunks = build_windows("Genesis", 1, _verses(count))
            widths = {c["verse_end"] - c["verse_start"] + 1 for c in chunks}
            assert widths == {5}, f"{count} verses produced widths {widths}"

    def test_last_chunk_ends_on_the_last_verse(self):
        from bible_study.vectors import build_windows
        chunks = build_windows("Genesis", 1, _verses(12))
        assert chunks[-1]["verse_end"] == 12

    def test_all_verses_are_covered(self):
        from bible_study.vectors import build_windows
        covered = set()
        for chunk in build_windows("Genesis", 1, _verses(23)):
            covered.update(range(chunk["verse_start"], chunk["verse_end"] + 1))
        assert covered == set(range(1, 24))

    def test_starts_strictly_increase(self):
        from bible_study.vectors import build_windows
        for count in range(1, 40):
            starts = [c["verse_start"]
                      for c in build_windows("Genesis", 1, _verses(count))]
            assert starts == sorted(set(starts))

    def test_verse_numbers_come_from_rows_not_positions(self):
        from bible_study.vectors import build_windows
        rows = [{"verse": v, "text": "x"} for v in (1, 2, 4, 5, 6)]
        chunks = build_windows("Genesis", 1, rows)
        assert chunks[0]["citation"] == "Genesis 1:1-6"

    def test_text_embeds_the_citation(self):
        from bible_study.vectors import build_windows
        chunk = build_windows("Genesis", 1, _verses(5))[0]
        assert chunk["text"].startswith("Genesis 1:1-5")

    def test_stride_of_one_still_terminates(self):
        from bible_study.vectors import build_windows
        chunks = build_windows("Genesis", 1, _verses(8), window=5, stride=1)
        assert chunks[-1]["verse_end"] == 8

    def test_invalid_window_raises(self):
        from bible_study.vectors import build_windows
        with pytest.raises(ValueError, match="window must be"):
            build_windows("Genesis", 1, _verses(5), window=0)

    def test_invalid_stride_raises(self):
        from bible_study.vectors import build_windows
        with pytest.raises(ValueError, match="stride must be"):
            build_windows("Genesis", 1, _verses(5), stride=0)


class TestRebuildChunks:

    def test_creates_all_three_tiers(self, seeded_db):
        from bible_study.db import chunk_counts
        from bible_study.vectors import rebuild_chunks
        rebuild_chunks(seeded_db, window=5, stride=3)
        counts = chunk_counts(seeded_db)
        assert counts["verse"][0] == 4       # 3 from Genesis, 1 from Exodus
        assert counts["chapter"][0] == 1
        assert counts["book"][0] == 1

    def test_returns_the_number_written(self, seeded_db):
        from bible_study.vectors import rebuild_chunks
        assert rebuild_chunks(seeded_db, window=5, stride=3) == 6

    def test_skips_books_without_summaries(self, seeded_db):
        from bible_study.db import get_chunks_by_ids, get_stale_chunks
        from bible_study.vectors import rebuild_chunks
        rebuild_chunks(seeded_db, window=5, stride=3)
        chunks = get_stale_chunks(seeded_db, "m", 4)
        books = {c["book_name"] for c in chunks if c["tier"] == "book"}
        assert books == {"Genesis"}
        assert get_chunks_by_ids(seeded_db, []) == {}

    def test_is_idempotent(self, seeded_db):
        from bible_study.db import chunk_counts, get_stale_chunks
        from bible_study.vectors import rebuild_chunks
        rebuild_chunks(seeded_db, window=5, stride=3)
        first = {c["id"] for c in get_stale_chunks(seeded_db, "m", 4)}
        rebuild_chunks(seeded_db, window=5, stride=3)
        second = {c["id"] for c in get_stale_chunks(seeded_db, "m", 4)}
        assert first == second
        assert chunk_counts(seeded_db)["verse"][0] == 4

    def test_updates_text_when_a_summary_changes(self, seeded_db):
        from bible_study.db import get_stale_chunks, save_summary
        from bible_study.vectors import rebuild_chunks
        rebuild_chunks(seeded_db, window=5, stride=3)
        save_summary(seeded_db, "Genesis", 1, "A completely new summary.")
        rebuild_chunks(seeded_db, window=5, stride=3)
        chapter = [c for c in get_stale_chunks(seeded_db, "m", 4)
                   if c["tier"] == "chapter"][0]
        assert "completely new summary" in chapter["text"]

    def test_honours_an_explicit_book_list(self, seeded_db):
        from bible_study.db import chunk_counts
        from bible_study.vectors import rebuild_chunks
        rebuild_chunks(seeded_db, ["Exodus"], window=5, stride=3)
        assert chunk_counts(seeded_db)["verse"][0] == 1

    def test_reads_window_and_stride_from_config(self, seeded_db, mocker):
        from bible_study.db import chunk_counts
        from bible_study.vectors import rebuild_chunks
        mocker.patch("bible_study.prompts.get_chunk_window", return_value=5)
        mocker.patch("bible_study.prompts.get_chunk_stride", return_value=5)
        rebuild_chunks(seeded_db)
        assert chunk_counts(seeded_db)["verse"][0] == 3


class TestNormalize:

    def test_returns_unit_length(self):
        import math
        from bible_study.vectors import normalize
        out = normalize([3.0, 4.0])
        assert math.isclose(math.sqrt(sum(x * x for x in out)), 1.0)

    def test_handles_a_zero_vector(self):
        from bible_study.vectors import normalize
        assert normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_preserves_direction(self):
        from bible_study.vectors import normalize
        out = normalize([2.0, 0.0])
        assert out == [1.0, 0.0]

    def test_is_idempotent_on_unit_vectors(self):
        from bible_study.vectors import normalize
        once = normalize([1.0, 1.0])
        assert normalize(once) == pytest.approx(once)

    def test_serialize_round_trips_as_float32(self):
        import struct
        from bible_study.vectors import _serialize
        blob = _serialize([1.0, 0.5, 0.0, -1.0])
        assert struct.unpack("<4f", blob) == (1.0, 0.5, 0.0, -1.0)

    def test_text_hash_is_stable_and_short(self):
        from bible_study.vectors import text_hash
        assert text_hash("abc") == text_hash("abc")
        assert text_hash("abc") != text_hash("abd")
        assert len(text_hash("abc")) == 16


class TestEmbedAll:

    def _prepare(self, db_path, mocker, vector=(1.0, 0.0, 0.0, 0.0)):
        from bible_study.vectors import init_vec, rebuild_chunks
        init_vec(db_path, dims=4, model="fake")
        rebuild_chunks(db_path, window=5, stride=3)
        mocker.patch("bible_study.ollama.embed", side_effect=_fake_embed(vector))
        return db_path

    def test_writes_one_vector_per_chunk(self, seeded_db, mocker):
        from bible_study.db import chunk_counts
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker)
        assert embed_all(seeded_db, batch_size=2) == 6
        counts = chunk_counts(seeded_db)
        assert counts["verse"] == (4, 4)
        assert counts["book"] == (1, 1)

    def test_is_incremental(self, seeded_db, mocker):
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker)
        embed_all(seeded_db, batch_size=2)
        assert embed_all(seeded_db, batch_size=2) == 0

    def test_re_embeds_changed_text(self, seeded_db, mocker):
        from bible_study.db import save_summary
        from bible_study.vectors import embed_all, rebuild_chunks
        self._prepare(seeded_db, mocker)
        embed_all(seeded_db, batch_size=10)
        save_summary(seeded_db, "Genesis", 1, "Rewritten summary text.")
        rebuild_chunks(seeded_db, window=5, stride=3)
        assert embed_all(seeded_db, batch_size=10) == 1

    def test_batches_the_requests(self, seeded_db, mocker):
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker)
        mock_embed = mocker.patch(
            "bible_study.ollama.embed",
            side_effect=_fake_embed((1.0, 0.0, 0.0, 0.0)),
        )
        embed_all(seeded_db, batch_size=2)
        assert mock_embed.call_count == 3

    def test_honours_limit(self, seeded_db, mocker):
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker)
        assert embed_all(seeded_db, batch_size=10, limit=2) == 2

    def test_returns_zero_when_nothing_pending(self, seeded_db, mocker):
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker)
        embed_all(seeded_db, batch_size=10)
        assert embed_all(seeded_db, batch_size=10) == 0

    def test_survives_a_failing_batch(self, seeded_db, mocker):
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker)
        good = _fake_embed((1.0, 0.0, 0.0, 0.0))
        calls = {"n": 0}

        def flaky(texts, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("transient")
            return good(texts, **kwargs)

        mocker.patch("bible_study.ollama.embed", side_effect=flaky)
        done = embed_all(seeded_db, batch_size=2)
        assert done == 4
        log = (seeded_db.parent / "EMBED_PROGRESS.md").read_text()
        assert "FAILED" in log

    def test_writes_a_progress_file(self, seeded_db, mocker, tmp_path):
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker)
        log = tmp_path / "custom.md"
        embed_all(seeded_db, batch_size=10, progress_file=log)
        assert "Bible Embedding Progress" in log.read_text()

    def test_rejects_a_wrong_width_vector(self, seeded_db, mocker):
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker, vector=(1.0, 0.0))
        assert embed_all(seeded_db, batch_size=10) == 0
        log = (seeded_db.parent / "EMBED_PROGRESS.md").read_text()
        assert "dimensions but the index expects" in log

    def test_uses_the_index_width_over_config(self, seeded_db, mocker):
        """A config change must not write mismatched vectors into vec0."""
        from bible_study.vectors import embed_all
        self._prepare(seeded_db, mocker)
        mocker.patch("bible_study.prompts.get_embed_dims", return_value=999)
        assert embed_all(seeded_db, batch_size=10) == 6


class TestSearch:

    def _indexed(self, db_path, mocker):
        from bible_study.vectors import embed_all, init_vec, rebuild_chunks
        init_vec(db_path, dims=4, model="fake")
        rebuild_chunks(db_path, window=5, stride=3)
        mocker.patch("bible_study.ollama.embed", side_effect=_tier_embed(
            {"(book summary)": (0.0, 0.0, 1.0, 0.0),
             "(summary)": (0.0, 1.0, 0.0, 0.0)},
            default=(1.0, 0.0, 0.0, 0.0),
        ))
        embed_all(db_path, batch_size=10)
        return db_path

    def test_returns_nearest_first(self, seeded_db, mocker):
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        hits = search(seeded_db, [0.0, 1.0, 0.0, 0.0],
                      {"verse": 2, "chapter": 2, "book": 2})
        assert hits[0]["tier"] == "chapter"

    def test_respects_k_per_tier(self, seeded_db, mocker):
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        hits = search(seeded_db, [1.0, 0.0, 0.0, 0.0],
                      {"verse": 2, "chapter": 1, "book": 0})
        tiers = [h["tier"] for h in hits]
        assert tiers.count("verse") == 2
        assert tiers.count("chapter") == 1
        assert tiers.count("book") == 0

    def test_hydrates_citation_and_text(self, seeded_db, mocker):
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        hit = search(seeded_db, [1.0, 0.0, 0.0, 0.0], {"verse": 1})[0]
        assert hit["citation"].startswith("Genesis") or \
            hit["citation"].startswith("Exodus")
        assert hit["text"]
        assert "distance" in hit and "rank" in hit

    def test_rank_is_dense_per_tier(self, seeded_db, mocker):
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        hits = search(seeded_db, [1.0, 0.0, 0.0, 0.0],
                      {"verse": 4, "chapter": 1, "book": 1})
        ranks = [h["rank"] for h in hits if h["tier"] == "verse"]
        assert ranks == list(range(len(ranks)))

    def test_is_deterministic_across_runs(self, seeded_db, mocker):
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        query = [1.0, 0.0, 0.0, 0.0]
        first = [h["id"] for h in search(seeded_db, query, {"verse": 4})]
        second = [h["id"] for h in search(seeded_db, query, {"verse": 4})]
        assert first == second

    def test_skips_tiers_with_k_below_one(self, seeded_db, mocker):
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        hits = search(seeded_db, [1.0, 0.0, 0.0, 0.0], {"verse": 0})
        assert hits == []

    def test_ignores_unknown_tiers(self, seeded_db, mocker):
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        assert search(seeded_db, [1.0, 0.0, 0.0, 0.0], {"nope": 5}) == []

    def test_empty_index_returns_nothing(self, db_path):
        from bible_study.vectors import init_vec, search
        init_vec(db_path, dims=4, model="fake")
        assert search(db_path, [1.0, 0.0, 0.0, 0.0]) == []

    def test_missing_index_raises(self, db_path):
        from bible_study.vectors import VectorIndexError, search
        with pytest.raises(VectorIndexError, match="bible-study embed"):
            search(db_path, [1.0, 0.0, 0.0, 0.0])

    def test_k_binds_as_a_parameter(self, seeded_db, mocker):
        """sqlite-vec KNN requires k; verify a bound parameter works."""
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        assert len(search(seeded_db, [1.0, 0.0, 0.0, 0.0], {"verse": 1})) == 1
        assert len(search(seeded_db, [1.0, 0.0, 0.0, 0.0], {"verse": 3})) == 3

    def test_default_k_is_used_when_omitted(self, seeded_db, mocker):
        from bible_study.vectors import search
        self._indexed(seeded_db, mocker)
        assert len(search(seeded_db, [1.0, 0.0, 0.0, 0.0])) == 6


class TestEmbedQuery:

    def test_wraps_the_question_as_a_query(self, mocker):
        from bible_study.vectors import embed_query
        mock_embed = mocker.patch(
            "bible_study.ollama.embed",
            side_effect=_fake_embed((3.0, 4.0)),
        )
        out = embed_query("what is grace?")
        assert mock_embed.call_args.kwargs["is_query"] is True
        assert out == pytest.approx([0.6, 0.8])

    def test_uses_the_configured_model(self, mocker):
        from bible_study.vectors import embed_query
        mocker.patch("bible_study.prompts.get_embed_model", return_value="cfg")
        mock_embed = mocker.patch(
            "bible_study.ollama.embed", side_effect=_fake_embed((1.0, 0.0)),
        )
        embed_query("q")
        assert mock_embed.call_args.kwargs["model"] == "cfg"

    def test_explicit_model_wins(self, mocker):
        from bible_study.vectors import embed_query
        mock_embed = mocker.patch(
            "bible_study.ollama.embed", side_effect=_fake_embed((1.0, 0.0)),
        )
        embed_query("q", ollama_kwargs={"model": "explicit"})
        assert mock_embed.call_args.kwargs["model"] == "explicit"


class TestClearVectors:

    def test_removes_vectors_and_resets_staleness(self, seeded_db, mocker):
        from bible_study.db import chunk_counts
        from bible_study.vectors import (
            clear_vectors,
            embed_all,
            init_vec,
            rebuild_chunks,
        )
        init_vec(seeded_db, dims=4, model="fake")
        rebuild_chunks(seeded_db, window=5, stride=3)
        mocker.patch(
            "bible_study.ollama.embed",
            side_effect=_fake_embed((1.0, 0.0, 0.0, 0.0)),
        )
        embed_all(seeded_db, batch_size=10)
        assert clear_vectors(seeded_db) == 6
        assert chunk_counts(seeded_db)["verse"] == (4, 0)

    def test_is_safe_before_init_vec(self, db_path):
        from bible_study.vectors import clear_vectors
        assert clear_vectors(db_path) == 0


class TestVecStatus:

    def test_reports_counts_and_build_metadata(self, seeded_db):
        from bible_study.vectors import init_vec, rebuild_chunks, vec_status
        init_vec(seeded_db, dims=4, model="fake")
        rebuild_chunks(seeded_db, window=5, stride=3)
        status = vec_status(seeded_db)
        assert status["chunks"]["verse"] == (4, 0)
        assert status["embed_dims"] == "4"

    def test_works_before_any_index_exists(self, db_path):
        from bible_study.vectors import vec_status
        status = vec_status(db_path)
        assert status["chunks"] == {}
        assert status["embed_model"] is None
