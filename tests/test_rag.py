"""Tests for bible_study/rag -- expansion, ranking, budgeting, answering.

Every test patches ``bible_study.vectors.search`` and
``bible_study.ollama.generate``, so nothing here touches the sqlite-vec
extension, an embedding model, or the network.
"""

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
    upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning.")])
    upsert_verses(db_path, "Genesis", 2, [(1, "Thus the heavens.")])
    save_summary(db_path, "Genesis", 1, "Creation summary.")
    save_summary(db_path, "Genesis", 2, "Rest summary.")
    save_book_summary(db_path, "Genesis", "GEN", "Book of beginnings.")
    upsert_verses(db_path, "Exodus", 1, [(1, "Now these are the names.")])
    return db_path


def _hit(tier, book="Genesis", chapter=1, verse_start=1, verse_end=5, rank=0):
    from bible_study.vectors import citation
    return {
        "tier": tier,
        "chunk_id": hash((tier, book, chapter, verse_start)) % 10000,
        "id": hash((tier, book, chapter, verse_start)) % 10000,
        "distance": 0.1 * (rank + 1),
        "rank": rank,
        "book_name": book,
        "chapter": chapter if tier != "book" else 0,
        "verse_start": verse_start if tier == "verse" else 0,
        "verse_end": verse_end if tier == "verse" else 0,
        "citation": citation(tier, book, chapter, verse_start, verse_end),
        "text": f"{tier} text for {book} {chapter}",
    }


class TestExpand:

    def test_verse_hit_pulls_its_chapter_summary(self, seeded_db):
        from bible_study.rag import expand
        blocks = expand(seeded_db, [_hit("verse")])
        kinds = {b["kind"] for b in blocks}
        assert kinds == {"verses", "chapter-summary"}

    def test_verse_hit_without_a_summary_yields_one_block(self, seeded_db):
        from bible_study.rag import expand
        blocks = expand(seeded_db, [_hit("verse", book="Exodus", chapter=1)])
        assert [b["kind"] for b in blocks] == ["verses"]

    def test_chapter_hit_pulls_the_book_summary(self, seeded_db):
        from bible_study.rag import expand
        blocks = expand(seeded_db, [_hit("chapter")])
        kinds = {b["kind"] for b in blocks}
        assert kinds == {"chapter-summary", "book-summary"}

    def test_book_hit_has_no_expansion(self, seeded_db):
        from bible_study.rag import expand
        blocks = expand(seeded_db, [_hit("book")])
        assert [b["kind"] for b in blocks] == ["book-summary"]

    def test_verse_hit_does_not_pull_the_book_summary(self, seeded_db):
        """Expansion is one level deep on purpose."""
        from bible_study.rag import expand
        blocks = expand(seeded_db, [_hit("verse")])
        assert all(b["kind"] != "book-summary" for b in blocks)

    def test_repeated_chapter_summary_appears_once(self, seeded_db):
        from bible_study.rag import expand
        hits = [
            _hit("verse", verse_start=1, rank=0),
            _hit("verse", verse_start=4, rank=1),
            _hit("verse", verse_start=7, rank=2),
        ]
        blocks = expand(seeded_db, hits)
        summaries = [b for b in blocks if b["kind"] == "chapter-summary"]
        assert len(summaries) == 1

    def test_book_summary_reached_twice_appears_once(self, seeded_db):
        from bible_study.rag import expand
        hits = [
            _hit("chapter", chapter=1, rank=0),
            _hit("chapter", chapter=2, rank=1),
        ]
        blocks = expand(seeded_db, hits)
        books = [b for b in blocks if b["kind"] == "book-summary"]
        assert len(books) == 1

    def test_dedupe_keeps_the_best_score(self, seeded_db):
        from bible_study.rag import expand
        hits = [
            _hit("verse", verse_start=4, rank=3),
            _hit("verse", verse_start=1, rank=0),
        ]
        blocks = expand(seeded_db, hits)
        summary = [b for b in blocks if b["kind"] == "chapter-summary"][0]
        assert summary["score"] == pytest.approx(1.0)

    def test_higher_scoring_direct_hit_wins(self, seeded_db):
        """book rank 0 (0.80) beats the expansion of chapter rank 1 (0.45)."""
        from bible_study.rag import expand
        hits = [
            _hit("chapter", chapter=1, rank=1),
            _hit("book", rank=0),
        ]
        blocks = expand(seeded_db, hits)
        book = [b for b in blocks if b["kind"] == "book-summary"][0]
        assert book["is_expansion"] is False
        assert book["score"] == pytest.approx(0.8)

    def test_higher_scoring_expansion_wins(self, seeded_db):
        """The text is identical either way, so the better score should rank it."""
        from bible_study.rag import expand
        hits = [
            _hit("chapter", chapter=1, rank=0),
            _hit("book", rank=0),
        ]
        blocks = expand(seeded_db, hits)
        book = [b for b in blocks if b["kind"] == "book-summary"][0]
        assert book["score"] == pytest.approx(0.9)

    def test_empty_hits_yield_no_blocks(self, seeded_db):
        from bible_study.rag import expand
        assert expand(seeded_db, []) == []


class TestRank:

    def _blocks(self, seeded_db, hits):
        from bible_study.rag import expand, rank
        return rank(expand(seeded_db, hits))

    def test_expansion_follows_its_parent(self, seeded_db):
        blocks = self._blocks(seeded_db, [
            _hit("verse", chapter=1, rank=0),
            _hit("verse", chapter=2, rank=1),
        ])
        kinds = [b["kind"] for b in blocks]
        assert kinds == [
            "verses", "chapter-summary", "verses", "chapter-summary",
        ]

    def test_verse_tier_outranks_chapter_at_equal_rank(self, seeded_db):
        blocks = self._blocks(seeded_db, [
            _hit("chapter", chapter=1, rank=0),
            _hit("verse", chapter=2, rank=0),
        ])
        assert blocks[0]["kind"] == "verses"

    def test_ordering_is_deterministic(self, seeded_db):
        hits = [
            _hit("verse", chapter=1, rank=0),
            _hit("chapter", chapter=2, rank=0),
            _hit("book", rank=0),
        ]
        first = [b["citation"] for b in self._blocks(seeded_db, hits)]
        second = [b["citation"] for b in self._blocks(seeded_db, hits)]
        assert first == second

    def test_scores_are_non_increasing(self, seeded_db):
        blocks = self._blocks(seeded_db, [
            _hit("verse", chapter=1, rank=0),
            _hit("verse", chapter=2, rank=1),
            _hit("book", rank=0),
        ])
        scores = [b["score"] for b in blocks]
        assert scores == sorted(scores, reverse=True)


class TestBuildContext:

    def _blocks(self, count, size=100):
        return [
            {"kind": "verses", "citation": f"Genesis 1:{i}",
             "text": "x" * size, "book_name": "Genesis", "chapter": 1,
             "verse_start": i, "verse_end": i, "score": 1.0 / (i + 1),
             "is_expansion": False}
            for i in range(count)
        ]

    def test_all_blocks_fit_a_large_budget(self):
        from bible_study.rag import build_context
        text, kept, dropped = build_context(self._blocks(3), 100000)
        assert len(kept) == 3
        assert dropped == 0
        assert "Genesis 1:0" in text

    def test_lowest_ranked_blocks_are_dropped(self):
        from bible_study.rag import build_context
        _, kept, dropped = build_context(self._blocks(5), 300)
        assert dropped == 5 - len(kept)
        assert kept[0]["citation"] == "Genesis 1:0"

    def test_top_block_survives_an_impossible_budget(self):
        from bible_study.rag import build_context
        _, kept, dropped = build_context(self._blocks(3), 0)
        assert len(kept) == 1
        assert dropped == 2

    def test_a_small_block_after_a_dropped_one_still_fits(self):
        from bible_study.rag import build_context
        blocks = self._blocks(1, size=50) + self._blocks(1, size=5000)
        blocks[1]["citation"] = "Genesis 2:1"
        small = self._blocks(1, size=10)[0]
        small["citation"] = "Genesis 3:1"
        _, kept, dropped = build_context([*blocks, small], 200)
        citations = [b["citation"] for b in kept]
        assert "Genesis 3:1" in citations
        assert dropped == 1

    def test_empty_blocks_yield_empty_context(self):
        from bible_study.rag import build_context
        text, kept, dropped = build_context([], 100)
        assert (text, kept, dropped) == ("", [], 0)

    def test_context_labels_every_block_with_its_citation(self):
        from bible_study.rag import build_context
        text, _, _ = build_context(self._blocks(2), 100000)
        assert "[Genesis 1:0]" in text
        assert "[Genesis 1:1]" in text


class TestAnswerQuestion:

    @pytest.fixture
    def wired(self, seeded_db, mocker):
        mocker.patch(
            "bible_study.vectors.embed_query", return_value=[1.0, 0.0],
        )
        mocker.patch(
            "bible_study.vectors.search",
            return_value=[_hit("verse", rank=0), _hit("book", rank=0)],
        )
        gen = mocker.patch(
            "bible_study.ollama.generate", return_value="An answer.",
        )
        return seeded_db, gen

    def test_returns_answer_and_sources(self, wired):
        from bible_study.rag import answer_question
        db_path, _ = wired
        result = answer_question("why?", db_path, config={})
        assert result["answer"] == "An answer."
        assert result["sources"]
        assert result["question"] == "why?"

    def test_makes_exactly_one_generate_call(self, wired):
        """No agentic loop, no second pass."""
        from bible_study.rag import answer_question
        db_path, gen = wired
        answer_question("why?", db_path, config={})
        gen.assert_called_once()

    def test_embeds_the_question_as_a_query(self, seeded_db, mocker):
        from bible_study.rag import answer_question
        embed_query = mocker.patch(
            "bible_study.vectors.embed_query", return_value=[1.0, 0.0],
        )
        mocker.patch(
            "bible_study.vectors.search", return_value=[_hit("verse")],
        )
        mocker.patch("bible_study.ollama.generate", return_value="A")
        answer_question("why?", seeded_db, config={})
        embed_query.assert_called_once()

    def test_context_contains_every_kept_citation(self, wired):
        from bible_study.rag import answer_question
        db_path, _ = wired
        result = answer_question("why?", db_path, config={})
        for source in result["sources"]:
            assert source["citation"] in result["prompt"]

    def test_passes_k_through_to_search(self, seeded_db, mocker):
        from bible_study.rag import answer_question
        mocker.patch("bible_study.vectors.embed_query", return_value=[1.0])
        search = mocker.patch(
            "bible_study.vectors.search", return_value=[_hit("verse")],
        )
        mocker.patch("bible_study.ollama.generate", return_value="A")
        answer_question(
            "q", seeded_db, k_verse=3, k_chapter=2, k_book=1, config={},
        )
        assert search.call_args[0][2] == {
            "verse": 3, "chapter": 2, "book": 1,
        }

    def test_raises_when_retrieval_is_empty(self, seeded_db, mocker):
        """An empty context invites an ungrounded answer from memory."""
        from bible_study.rag import answer_question
        mocker.patch("bible_study.vectors.embed_query", return_value=[1.0])
        mocker.patch("bible_study.vectors.search", return_value=[])
        gen = mocker.patch("bible_study.ollama.generate", return_value="A")
        with pytest.raises(RuntimeError, match="bible-study embed"):
            answer_question("why?", seeded_db, config={})
        gen.assert_not_called()

    def test_explicit_model_overrides_config(self, wired):
        from bible_study.rag import answer_question
        db_path, gen = wired
        answer_question(
            "q", db_path, ollama_kwargs={"model": "explicit"},
            config={"ollama_model": "from-config"},
        )
        assert gen.call_args.kwargs["model"] == "explicit"

    def test_model_comes_from_config(self, wired):
        from bible_study.rag import answer_question
        db_path, gen = wired
        answer_question("q", db_path, config={"ollama_model": "from-config"})
        assert gen.call_args.kwargs["model"] == "from-config"

    def test_num_ctx_comes_from_config(self, wired):
        from bible_study.rag import answer_question
        db_path, gen = wired
        answer_question("q", db_path, config={"ollama_num_ctx": "65536"})
        assert gen.call_args.kwargs["num_ctx"] == 65536

    def test_uses_the_config_ask_template(self, wired):
        from bible_study.rag import answer_question
        db_path, _ = wired
        result = answer_question(
            "why?", db_path,
            config={"ask": "CUSTOM {question} :: {context}"},
        )
        assert result["prompt"].startswith("CUSTOM why? :: ")

    def test_loads_config_when_omitted(self, wired, mocker):
        from bible_study.rag import answer_question
        db_path, _ = wired
        load = mocker.patch(
            "bible_study.rag.load_config", return_value={},
        )
        answer_question("q", db_path)
        load.assert_called_once()

    def test_propagates_prompt_too_long(self, wired, mocker):
        from bible_study.ollama import PromptTooLongError
        from bible_study.rag import answer_question
        db_path, gen = wired
        gen.side_effect = PromptTooLongError("too long")
        with pytest.raises(PromptTooLongError):
            answer_question("q", db_path, config={})

    def test_question_with_backslashes_is_safe(self, wired):
        from bible_study.rag import answer_question
        db_path, _ = wired
        result = answer_question(r"what is \1?", db_path, config={})
        assert r"what is \1?" in result["prompt"]

    def test_reports_dropped_sources(self, wired):
        from bible_study.rag import answer_question
        db_path, _ = wired
        result = answer_question(
            "q", db_path, config={"ollama_num_ctx": "2560"},
        )
        assert result["dropped"] >= 1
