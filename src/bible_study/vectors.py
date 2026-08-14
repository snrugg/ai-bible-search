"""Vector search over verses and summaries, backed by sqlite-vec.

Storage and retrieval only -- no LLM orchestration.  Chunk *metadata*
lives in the plain ``chunks`` table created by :mod:`bible_study.db`, so
it stays readable when sqlite-vec is unavailable; only the vectors
themselves live in vec0 virtual tables, joined back on ``chunks.id``.

The pipeline on top of this (retrieve, expand, answer) is in
:mod:`bible_study.rag`.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import bible_study.ollama as _ol

#: Verses per chunk, and how far each window advances.  Stride < window
#: makes consecutive chunks overlap, so an idea spanning a boundary is
#: still embedded as a unit.  At 5/3 the KJV yields 9,969 verse chunks;
#: stride 5 (no overlap) yields 6,704 and embeds about a third faster.
CHUNK_WINDOW = 5
CHUNK_STRIDE = 3

#: One vec0 table per tier, all keyed on ``chunks.id``.
#:
#: Three tables rather than one with a filterable metadata column: the
#: retrieval design needs top-k *per tier*, and verse chunks outnumber
#: chapter summaries roughly 8:1, so a single global top-k is routinely
#: all verse chunks with zero chapter coverage.  Per-tier tables also
#: need only the oldest, most widely supported vec0 syntax.
_VEC_TABLES = {
    "verse": "vec_verse_chunks",
    "chapter": "vec_chapter_chunks",
    "book": "vec_book_chunks",
}

#: Order used to break ranking ties deterministically.
_TIER_ORDER = {"verse": 0, "chapter": 1, "book": 2}

#: Default hits requested per tier.
DEFAULT_K = {"verse": 8, "chapter": 4, "book": 2}

_META_MODEL = "embed_model"
_META_DIMS = "embed_dims"
_META_METRIC = "distance_metric"

_DDL_COSINE = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
    "embedding float[{dims}] distance_metric=cosine)"
)
_DDL_PLAIN = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
    "embedding float[{dims}])"
)


class VectorSupportError(RuntimeError):
    """Raised when sqlite-vec cannot be loaded at all.

    Distinct from a missing *index*: no amount of re-running ``embed``
    fixes this, because the extension itself is unavailable.
    """


class VectorIndexError(RuntimeError):
    """Raised when the vector index is absent, empty, or the wrong width."""


def _connect(path: Path) -> sqlite3.Connection:
    """Open *path* with the sqlite-vec extension loaded.

    Every vector query goes through here.  Plain metadata reads use
    :mod:`bible_study.db` and never load the extension, which is the whole
    point of keeping ``chunks`` in ``INIT_SQL``.
    """
    try:
        import sqlite_vec
    except ImportError as exc:
        msg = (
            "sqlite-vec is not installed, so vector search is unavailable. "
            "Run `uv sync` -- sqlite-vec is a declared project dependency."
        )
        raise VectorSupportError(msg) from exc

    conn = sqlite3.connect(str(path))
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.OperationalError,
            sqlite3.NotSupportedError) as exc:
        conn.close()
        msg = (
            "This Python was built without SQLite loadable-extension "
            "support, so sqlite-vec cannot be loaded. Use a uv-managed "
            "interpreter, or rebuild CPython with "
            "--enable-loadable-sqlite-extensions."
        )
        raise VectorSupportError(msg) from exc

    try:
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        conn.close()
        msg = f"Failed to load the sqlite-vec extension: {exc}"
        raise VectorSupportError(msg) from exc
    return conn


def text_hash(text: str) -> str:
    """Return the short content hash used for staleness detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize(vector: list[float]) -> list[float]:
    """Scale *vector* to unit length.

    On unit vectors, L2 distance and cosine distance are monotonically
    related, so ranking is identical under either metric.  Normalising
    both stored and query vectors therefore keeps results correct even if
    the installed sqlite-vec build ignores ``distance_metric=cosine``.

    Ollama already returns unit vectors, so this is normally a no-op --
    it is here to remove the dependency on that staying true.

    The corollary: rank on *order*, never on an absolute distance
    threshold, because the scale differs between the two metrics.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def _serialize(vector: list[float]) -> bytes:
    """Pack floats into sqlite-vec's little-endian float32 blob format."""
    return struct.pack(f"<{len(vector)}f", *vector)


def citation(
    tier: str,
    book_name: str,
    chapter: int = 0,
    verse_start: int = 0,
    verse_end: int = 0,
) -> str:
    """Render a human-readable reference for a chunk."""
    if tier == "book":
        return book_name
    if tier == "chapter":
        return f"{book_name} {chapter}"
    if verse_start == verse_end:
        return f"{book_name} {chapter}:{verse_start}"
    return f"{book_name} {chapter}:{verse_start}-{verse_end}"


def init_vec(
    path: Path,
    dims: int | None = None,
    model: str | None = None,
) -> None:
    """Create the per-tier vec0 tables and record what built them.

    A vec0 column's width is fixed at creation and cannot be altered, so a
    dimension change is refused here rather than allowed to mix two vector
    spaces in one index.

    Raises
    ------
    VectorIndexError
        If *dims* differs from the width the existing index was built at.
    """
    from bible_study import db as _db
    from bible_study.prompts import get_embed_dims, get_embed_model

    dims = int(dims if dims is not None else get_embed_dims())
    model = model if model is not None else get_embed_model()

    stored = _db.get_meta(path, _META_DIMS)
    if stored is not None and int(stored) != dims:
        msg = (
            f"This index was built at {stored} dimensions but {dims} was "
            f"requested. A vec0 column's width cannot be altered -- run "
            f"`bible-study embed --rebuild` to discard the old vectors and "
            f"start over."
        )
        raise VectorIndexError(msg)

    metric = "cosine"
    with closing(_connect(path)) as conn:
        for table in _VEC_TABLES.values():
            try:
                conn.execute(_DDL_COSINE.format(table=table, dims=dims))
            except sqlite3.OperationalError:
                # Older builds reject the distance_metric keyword.  Our
                # vectors are unit length, so plain L2 ranks identically.
                metric = "l2"
                conn.execute(_DDL_PLAIN.format(table=table, dims=dims))
        conn.commit()

    _db.set_meta(path, _META_DIMS, str(dims))
    _db.set_meta(path, _META_MODEL, model)
    _db.set_meta(path, _META_METRIC, metric)


def clear_vectors(path: Path) -> int:
    """Drop every vector and forget which chunks were embedded.

    Returns the number of vectors removed.  Chunk rows and their text are
    kept; only the embeddings and their staleness stamps are discarded.
    """
    from bible_study import db as _db

    removed = 0
    with closing(_connect(path)) as conn:
        for table in _VEC_TABLES.values():
            try:
                removed += conn.execute(
                    f"SELECT COUNT(*) FROM {table}",
                ).fetchone()[0]
                conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                continue  # table not created yet
        conn.commit()

    with closing(sqlite3.connect(str(path))) as conn:
        conn.execute(
            "UPDATE chunks SET embedded_hash='', embed_model='', "
            "embed_dims=0, embedded_at=NULL",
        )
        conn.commit()
    _db.set_meta(path, _META_DIMS, _db.get_meta(path, _META_DIMS) or "0")
    return removed


def build_windows(
    book_name: str,
    chapter: int,
    verses: list[dict],
    window: int = CHUNK_WINDOW,
    stride: int = CHUNK_STRIDE,
) -> list[dict]:
    """Slice a chapter's verses into overlapping windows.

    *verses* is the ``[{"verse", "text"}, ...]`` list that
    ``db.get_verses`` returns.  Verse numbers are read from the rows
    rather than inferred from list positions, so a chapter with a gap
    still cites the range it actually covers.

    Chapters shorter than *window* yield exactly one chunk covering the
    whole chapter.  Otherwise every chunk is exactly *window* verses: a
    final window that would overhang the end is pulled back to end on the
    last verse, so it overlaps its predecessor more rather than trailing a
    one- or two-verse runt.  That pull-back always advances past the
    previous start, so it never duplicates a chunk.
    """
    if window < 1:
        msg = f"window must be >= 1, got {window}"
        raise ValueError(msg)
    if stride < 1:
        msg = f"stride must be >= 1, got {stride}"
        raise ValueError(msg)

    chunks: list[dict] = []
    total = len(verses)
    if total == 0:
        return chunks

    start = 0
    while True:
        if start + window > total and total >= window:
            start = total - window
        group = verses[start:start + window]
        first = group[0]["verse"]
        last = group[-1]["verse"]
        body = " ".join(v["text"].strip() for v in group)
        ref = citation("verse", book_name, chapter, first, last)
        chunks.append({
            "tier": "verse",
            "book_name": book_name,
            "chapter": chapter,
            "verse_start": first,
            "verse_end": last,
            "citation": ref,
            # The reference rides along in the embedded text so that the
            # citation itself is searchable.
            "text": f"{ref}\n\n{body}",
        })
        if start + window >= total:
            break
        start += stride
    return chunks


def rebuild_chunks(
    path: Path,
    book_names_list: list[str] | None = None,
    window: int | None = None,
    stride: int | None = None,
) -> int:
    """Populate ``chunks`` from verses and stored summaries.

    Idempotent: re-running updates text in place and keeps chunk ids
    stable, so vectors for unchanged text stay valid.  Returns the number
    of chunks written.
    """
    from bible_study import db as _db
    from bible_study.prompts import get_chunk_stride, get_chunk_window

    window = int(window if window is not None else get_chunk_window())
    stride = int(stride if stride is not None else get_chunk_stride())

    books = book_names_list
    if books is None:
        books = _db.get_all_book_names(path)

    written = 0
    for book_name in books:
        for chapter in _db.get_stored_chapters(path, book_name):
            verses = _db.get_verses(path, book_name, chapter)
            for chunk in build_windows(
                book_name, chapter, verses, window, stride,
            ):
                _db.upsert_chunk(
                    path, chunk["tier"], chunk["book_name"], chunk["chapter"],
                    chunk["verse_start"], chunk["verse_end"],
                    chunk["citation"], chunk["text"],
                    text_hash(chunk["text"]),
                )
                written += 1

            summary = _db.get_summary(path, book_name, chapter)
            if summary:
                ref = citation("chapter", book_name, chapter)
                body = f"{ref} (summary)\n\n{summary}"
                _db.upsert_chunk(
                    path, "chapter", book_name, chapter, 0, 0,
                    ref, body, text_hash(body),
                )
                written += 1

        book_summary = _db.get_book_summary(path, book_name)
        if book_summary:
            body = f"{book_name} (book summary)\n\n{book_summary}"
            _db.upsert_chunk(
                path, "book", book_name, 0, 0, 0,
                book_name, body, text_hash(body),
            )
            written += 1
    return written


def embed_query(question: str, ollama_kwargs: dict[str, Any] | None = None):
    """Embed *question* with the query-side instruction prefix."""
    from bible_study.prompts import get_embed_model

    kwargs = dict(ollama_kwargs or {})
    kwargs.setdefault("model", get_embed_model())
    kwargs["is_query"] = True
    return normalize(_ol.embed([question], **kwargs)[0])


def embed_all(
    path: Path,
    batch_size: int | None = None,
    limit: int | None = None,
    progress_file: Path | None = None,
    ollama_kwargs: dict[str, Any] | None = None,
) -> int:
    """Embed every stale chunk, returning how many were embedded.

    Batches survive individually: one failing batch is logged and the run
    continues, matching ``summary.generate_all_chapters``.  Chunks are
    stamped as embedded only *after* their vectors commit, so a crash
    leaves them stale and the next run redoes them.
    """
    from bible_study import db as _db
    from bible_study.prompts import get_embed_dims, get_embed_model

    kwargs = dict(ollama_kwargs or {})
    # The model is the user's intent, so it comes from config -- switching
    # it marks every chunk stale and re-embeds.  The width is a physical
    # property of the existing vec0 columns, so the index's own record wins
    # over config; init_vec() is what refuses a genuine width change.
    model = kwargs.pop("model", None) or get_embed_model()
    dims = kwargs.pop("dims", None)
    if dims is None:
        stored = _db.get_meta(path, _META_DIMS)
        dims = stored if stored is not None else get_embed_dims()
    dims = int(dims)
    batch_size = int(batch_size or _ol.EMBED_BATCH)

    pending = _db.get_stale_chunks(path, model, dims, limit=limit)
    if not pending:
        print("All chunks already embedded.")  # noqa: T201
        return 0

    log_path = progress_file or (path.parent / "EMBED_PROGRESS.md")
    batches = [pending[i:i + batch_size]
               for i in range(0, len(pending), batch_size)]
    done = 0

    with open(log_path, "w") as log_fh:
        log_fh.write("# Bible Embedding Progress\n\n")
        log_fh.write(f"Model: {model} ({dims} dims)\n\n")
        log_fh.write(f"Chunks to embed: {len(pending)}\n\n")

        for i, batch in enumerate(batches):
            label = f"{i + 1}/{len(batches)}"
            try:
                vectors = _ol.embed(
                    [chunk["text"] for chunk in batch], model=model, **kwargs,
                )
                _store_vectors(path, batch, vectors, dims)
                _db.mark_chunks_embedded(
                    path, [chunk["id"] for chunk in batch], model, dims,
                )
                done += len(batch)
                log_fh.write(
                    f"- **{label}**: embedded {len(batch)} chunks "
                    f"({done}/{len(pending)})\n",
                )
            except Exception as exc:  # noqa: BLE001
                log_fh.write(f"- **{label}**: **FAILED** ({exc})\n")
            log_fh.flush()
            time.sleep(0.2)

    return done


def _store_vectors(
    path: Path,
    batch: list[dict],
    vectors: list[list[float]],
    dims: int,
) -> None:
    """Write one vector per chunk into the tier's vec0 table."""
    with closing(_connect(path)) as conn:
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != dims:
                msg = (
                    f"Embedding for chunk {chunk['id']} has {len(vector)} "
                    f"dimensions but the index expects {dims} -- refusing to "
                    f"mix vector spaces"
                )
                raise RuntimeError(msg)
            table = _VEC_TABLES[chunk["tier"]]
            blob = _serialize(normalize(vector))
            # DELETE + INSERT rather than UPDATE or upsert: vec0's support
            # for those has varied across 0.1.x, this works everywhere.
            conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (chunk["id"],))
            conn.execute(
                f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
                (chunk["id"], blob),
            )
        conn.commit()


def search(
    path: Path,
    query_vector: list[float],
    k_by_tier: dict[str, int] | None = None,
) -> list[dict]:
    """Return KNN hits across tiers, hydrated with their chunk metadata.

    Results carry ``distance`` and ``rank`` (position within that tier).
    Rank on order, never on an absolute distance -- see :func:`normalize`.
    """
    from bible_study import db as _db

    k_by_tier = dict(k_by_tier if k_by_tier is not None else DEFAULT_K)
    blob = _serialize(normalize(query_vector))

    raw: list[dict] = []
    with closing(_connect(path)) as conn:
        for tier, k in k_by_tier.items():
            if tier not in _VEC_TABLES or int(k) < 1:
                continue
            # Table name comes from a module-level literal keyed by a
            # validated tier; there is no path from user input.
            table = _VEC_TABLES[tier]
            try:
                rows = conn.execute(
                    f"SELECT rowid, distance FROM {table} "
                    f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (blob, int(k)),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                msg = (
                    "No vector index found -- run `bible-study embed` to "
                    "build it."
                )
                raise VectorIndexError(msg) from exc
            for row in rows:
                raw.append({
                    "tier": tier,
                    "chunk_id": row[0],
                    "distance": row[1],
                })

    meta = _db.get_chunks_by_ids(path, [hit["chunk_id"] for hit in raw])
    hits = [dict(hit, **meta[hit["chunk_id"]])
            for hit in raw if hit["chunk_id"] in meta]
    # Sort before ranking, and break ties on id: vec0 returns equidistant
    # rows in an arbitrary order, which would otherwise make `rank` -- and
    # therefore the assembled prompt -- vary between identical runs.
    hits.sort(key=lambda h: (h["distance"], _TIER_ORDER[h["tier"]], h["id"]))
    seen: dict[str, int] = {}
    for hit in hits:
        rank = seen.get(hit["tier"], 0)
        hit["rank"] = rank
        seen[hit["tier"]] = rank + 1
    return hits


def vec_status(path: Path) -> dict:
    """Return index counts plus what the index was built with.

    Chunk counts come from plain SQLite, so this works even when
    sqlite-vec is unavailable.
    """
    from bible_study import db as _db

    return {
        "chunks": _db.chunk_counts(path),
        "embed_model": _db.get_meta(path, _META_MODEL),
        "embed_dims": _db.get_meta(path, _META_DIMS),
        "distance_metric": _db.get_meta(path, _META_METRIC),
    }
