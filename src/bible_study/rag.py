"""Grounded question answering over the vector index.

Retrieve, expand, rank, budget, then one ``generate()`` call.  Everything
between retrieval and generation is plain Python -- there is no agentic
loop and no second LLM pass, so the assembled prompt is deterministic and
assertable.

Storage and KNN live in :mod:`bible_study.vectors`; this module never
touches SQL or the sqlite-vec extension directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bible_study.ollama as _ol
import bible_study.vectors as _vec
from bible_study.prompts import (
    build_ask_prompt,
    get_model,
    get_num_ctx,
    load_config,
)

#: Hits requested per tier at the default ``top_k``.
DEFAULT_K_VERSE = 8
DEFAULT_K_CHAPTER = 4
DEFAULT_K_BOOK = 2

#: Relative weight of each tier when merging the three ranked lists.
#:
#: Distances across tiers are numerically comparable -- one vector space --
#: but not *semantically* comparable: chapter summaries are long, abstract,
#: LLM-written prose that scores closer to an abstract question than terse
#: 17th-century verse text does.  Sorting the merged pool by raw distance
#: therefore returns almost all summaries and almost no scripture, which
#: discards the point of indexing verses at all.  Weighted reciprocal rank
#: guarantees the head of every tier survives into the context.
_TIER_WEIGHT = {"verse": 1.0, "chapter": 0.9, "book": 0.8}

_TIER_ORDER = {"verse": 0, "chapter": 1, "book": 2}

#: Slack above check_prompt_fits' own reserve.  CHARS_PER_TOKEN = 4 is
#: optimistic on KJV English, whose short archaic words ("thee", "unto",
#: "hath") tokenize worse than modern prose.
BUDGET_SAFETY_TOKENS = 512


def _block(kind, chunk, score, is_expansion):
    """Build one context block from a chunk-shaped mapping."""
    return {
        "kind": kind,
        "citation": chunk["citation"],
        "text": chunk["text"],
        "book_name": chunk["book_name"],
        "chapter": chunk.get("chapter", 0),
        "verse_start": chunk.get("verse_start", 0),
        "verse_end": chunk.get("verse_end", 0),
        "score": score,
        "is_expansion": is_expansion,
    }


def _score(tier: str, rank: int) -> float:
    """Weighted reciprocal rank; see :data:`_TIER_WEIGHT`.

    Expansions inherit their parent's score rather than taking a penalty.
    Ordering within a score is settled by the ``is_expansion`` flag in
    :func:`rank`, which is what actually places a summary directly after
    the passage it belongs to.  A scalar penalty cannot do that: for an
    expansion of rank *r* to sit above the rank *r+1* parent, the penalty
    would have to exceed (r+1)/(r+2), which tends to 1 as *r* grows, so no
    single constant works for every rank.
    """
    return _TIER_WEIGHT[tier] / (rank + 1)


def retrieve(
    db_path: Path,
    question: str,
    k_verse: int = DEFAULT_K_VERSE,
    k_chapter: int = DEFAULT_K_CHAPTER,
    k_book: int = DEFAULT_K_BOOK,
    ollama_kwargs: dict[str, Any] | None = None,
) -> list[dict]:
    """Embed *question* and return KNN hits across all three tiers."""
    vector = _vec.embed_query(question, ollama_kwargs=ollama_kwargs)
    return _vec.search(db_path, vector, {
        "verse": k_verse,
        "chapter": k_chapter,
        "book": k_book,
    })


def expand(db_path: Path, hits: list[dict]) -> list[dict]:
    """Turn hits into context blocks, pulling in one level of surrounding text.

    A verse hit pulls its chapter summary; a chapter hit pulls its book
    summary; a book hit pulls nothing.  Expansion is deliberately one level
    deep -- two would let a single verse hit drag in several kilobytes of
    summary and crowd out every other hit.

    Duplicates are collapsed on ``(kind, book_name, chapter, verse_start)``
    keeping the best score, so six verse hits in one chapter attach that
    chapter's summary once rather than six times.  A block reached both
    directly and as an expansion is kept as the direct hit.
    """
    from bible_study import db as _db

    best: dict[tuple, dict] = {}

    def add(block):
        key = (block["kind"], block["book_name"], block["chapter"],
               block["verse_start"])
        current = best.get(key)
        if current is None:
            best[key] = block
            return
        better = (block["score"], not block["is_expansion"])
        incumbent = (current["score"], not current["is_expansion"])
        if better > incumbent:
            best[key] = block

    for hit in hits:
        tier = hit["tier"]
        score = _score(tier, hit["rank"])
        if tier == "verse":
            add(_block("verses", hit, score, False))
            summary = _db.get_summary(db_path, hit["book_name"], hit["chapter"])
            if summary:
                ref = _vec.citation("chapter", hit["book_name"], hit["chapter"])
                add(_block("chapter-summary", {
                    "citation": f"{ref} (summary)",
                    "text": summary,
                    "book_name": hit["book_name"],
                    "chapter": hit["chapter"],
                    "verse_start": 0,
                    "verse_end": 0,
                }, score, True))
        elif tier == "chapter":
            add(_block("chapter-summary", hit, score, False))
            book_summary = _db.get_book_summary(db_path, hit["book_name"])
            if book_summary:
                add(_block("book-summary", {
                    "citation": f"{hit['book_name']} (book summary)",
                    "text": book_summary,
                    "book_name": hit["book_name"],
                    "chapter": 0,
                    "verse_start": 0,
                    "verse_end": 0,
                }, score, True))
        else:
            add(_block("book-summary", hit, score, False))

    return list(best.values())


def rank(blocks: list[dict]) -> list[dict]:
    """Sort blocks best-first, breaking every tie deterministically.

    Expansions share their parent's score and sort immediately after it on
    the ``is_expansion`` flag, so the context reads "passage, that
    passage's chapter summary, next passage".

    Determinism is not cosmetic: it is what makes the assembled prompt
    stable between identical runs, and therefore assertable in tests.
    """
    return sorted(blocks, key=lambda b: (
        -b["score"],
        b["is_expansion"],
        _TIER_ORDER.get(_kind_tier(b["kind"]), 9),
        b["book_name"],
        b["chapter"],
        b["verse_start"],
    ))


def _kind_tier(kind: str) -> str:
    """Map a block kind back to its tier name."""
    if kind == "verses":
        return "verse"
    if kind == "chapter-summary":
        return "chapter"
    return "book"


def build_context(
    blocks: list[dict],
    budget_chars: int,
) -> tuple[str, list[dict], int]:
    """Assemble source text within *budget_chars*, dropping the weakest.

    Blocks arrive pre-sorted, so anything dropped is by construction the
    lowest-ranked material.  The single best block is always kept even if
    it alone exceeds the budget -- an oversized top hit should still get a
    chance rather than yielding an empty context.

    Returns ``(context_text, kept_blocks, dropped_count)``.
    """
    kept: list[dict] = []
    used = 0
    dropped = 0
    for block in blocks:
        rendered = f"[{block['citation']}]\n{block['text']}\n"
        if kept and used + len(rendered) > budget_chars:
            # continue, not break: a small high-value block after a large
            # one should still fit.
            dropped += 1
            continue
        kept.append(block)
        used += len(rendered)
    text = "\n".join(
        f"[{b['citation']}]\n{b['text']}\n" for b in kept
    )
    return text, kept, dropped


def answer_question(
    question: str,
    db_path: Path,
    k_verse: int = DEFAULT_K_VERSE,
    k_chapter: int = DEFAULT_K_CHAPTER,
    k_book: int = DEFAULT_K_BOOK,
    ollama_kwargs: dict[str, Any] | None = None,
    config: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Answer *question* from the indexed corpus with one generate() call.

    Returns ``{"question", "answer", "sources", "dropped", "prompt"}``.

    Raises
    ------
    RuntimeError
        If retrieval comes back empty.  Generating anyway would produce a
        prompt whose sources section is blank, which the model answers
        from memory -- an ungrounded answer indistinguishable from a
        grounded one.
    """
    ollama_kwargs = dict(ollama_kwargs or {})
    if config is None:
        config = load_config()

    hits = retrieve(
        db_path, question, k_verse, k_chapter, k_book,
        ollama_kwargs=ollama_kwargs.pop("embed_kwargs", None),
    )
    blocks = rank(expand(db_path, hits))
    if not blocks:
        msg = (
            f"Nothing in the vector index matched {question!r}. Run "
            f"`bible-study embed` to build the index, or rephrase the "
            f"question."
        )
        raise RuntimeError(msg)

    num_ctx = get_num_ctx(config)
    overhead = _ol.estimate_tokens(build_ask_prompt(config, question, ""))
    budget_tokens = (
        num_ctx - _ol.RESPONSE_RESERVE_TOKENS - overhead - BUDGET_SAFETY_TOKENS
    )
    budget_chars = max(0, budget_tokens) * _ol.CHARS_PER_TOKEN

    context, kept, dropped = build_context(blocks, budget_chars)
    prompt = build_ask_prompt(config, question, context)

    ollama_kwargs.setdefault("model", get_model(config))
    ollama_kwargs.setdefault("num_ctx", num_ctx)
    answer = _ol.generate(prompt, **ollama_kwargs)

    return {
        "question": question,
        "answer": str(answer),
        "sources": [
            {"citation": b["citation"], "kind": b["kind"],
             "book_name": b["book_name"], "chapter": b["chapter"]}
            for b in kept
        ],
        "dropped": dropped,
        "prompt": prompt,
    }
