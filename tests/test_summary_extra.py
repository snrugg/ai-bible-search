"""Additional tests for bible_study/summary -- batch runs, markdown, edge cases."""

from pathlib import Path

import pytest

from bible_study.db import init_db, save_summary, upsert_verses


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "bible.db"
    init_db(path)
    return path


class TestSummarizeChapterPaths:
    """Cover the fetch-from-API and error branches of summarize_chapter."""

    def test_fetches_from_api_when_db_empty(self, db_path, mocker):
        from bible_study.summary import summarize_chapter
        mocker.patch("bible_study.ollama.generate", return_value="Summary text.")
        mock_fetch = mocker.patch(
            "bible_study.api.fetch_chapter",
            return_value=[{"verse": 1, "text": "In the beginning"}],
        )
        result = summarize_chapter("Genesis", 1, db_path=db_path)
        assert result == "Summary text."
        mock_fetch.assert_called_once_with("Genesis", 1)

    def test_api_fetched_verses_are_persisted(self, db_path, mocker):
        from bible_study.db import get_verses
        from bible_study.summary import summarize_chapter
        mocker.patch("bible_study.ollama.generate", return_value="Summary.")
        mocker.patch(
            "bible_study.api.fetch_chapter",
            return_value=[{"verse": 1, "text": "In the beginning"}],
        )
        summarize_chapter("Genesis", 1, db_path=db_path)
        stored = get_verses(db_path, "Genesis", 1)
        assert stored[0]["text"] == "In the beginning"

    def test_summary_is_saved_to_db(self, db_path, mocker):
        from bible_study.db import get_summary
        from bible_study.summary import summarize_chapter
        mocker.patch("bible_study.ollama.generate", return_value="Stored summary.")
        upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
        summarize_chapter("Genesis", 1, db_path=db_path)
        assert get_summary(db_path, "Genesis", 1) == "Stored summary."

    def test_raises_when_api_returns_no_verses(self, db_path, mocker):
        from bible_study.summary import summarize_chapter
        mocker.patch("bible_study.api.fetch_chapter", return_value=[])
        with pytest.raises(RuntimeError, match="No verse text"):
            summarize_chapter("Genesis", 1, db_path=db_path)

    def test_works_without_db_path(self, mocker):
        from bible_study.summary import summarize_chapter
        mocker.patch("bible_study.ollama.generate", return_value="No-DB summary.")
        mocker.patch(
            "bible_study.api.fetch_chapter",
            return_value=[{"verse": 1, "text": "text"}],
        )
        assert summarize_chapter("Genesis", 1) == "No-DB summary."

    def test_passes_ollama_kwargs_through(self, db_path, mocker):
        from bible_study.summary import summarize_chapter
        mock_gen = mocker.patch("bible_study.ollama.generate", return_value="x")
        upsert_verses(db_path, "Genesis", 1, [(1, "text")])
        summarize_chapter(
            "Genesis", 1, db_path=db_path, ollama_kwargs={"model": "custom"},
        )
        assert mock_gen.call_args.kwargs["model"] == "custom"


class TestSummarizeBookPaths:
    """Cover summarize_book error and persistence branches."""

    def test_requires_db_path(self):
        from bible_study.summary import summarize_book
        with pytest.raises(ValueError, match="db_path is required"):
            summarize_book("Genesis")

    def test_raises_when_no_chapter_summaries(self, db_path):
        from bible_study.summary import summarize_book
        with pytest.raises(RuntimeError, match="No chapter summaries"):
            summarize_book("Genesis", db_path=db_path)

    def test_saves_book_summary_to_db(self, db_path, mocker):
        from bible_study.db import get_saved_books
        from bible_study.summary import summarize_book
        mocker.patch("bible_study.ollama.generate", return_value="Book summary.")
        save_summary(db_path, "Genesis", 1, "chapter one summary")
        summarize_book("Genesis", db_path=db_path)
        assert "Genesis" in get_saved_books(db_path)

    def test_prompt_includes_aggregated_chapters(self, db_path, mocker):
        from bible_study.summary import summarize_book
        mock_gen = mocker.patch("bible_study.ollama.generate", return_value="ok")
        save_summary(db_path, "Genesis", 1, "first chapter")
        save_summary(db_path, "Genesis", 2, "second chapter")
        summarize_book("Genesis", db_path=db_path)
        prompt = mock_gen.call_args[0][0]
        assert "Genesis" in prompt

    def test_unknown_book_is_not_persisted(self, db_path, mocker):
        from bible_study.db import get_saved_books
        from bible_study.summary import summarize_book
        mocker.patch("bible_study.ollama.generate", return_value="Made up.")
        save_summary(db_path, "Fakebook", 1, "chapter summary")
        result = summarize_book("Fakebook", db_path=db_path)
        assert result == "Made up."
        assert get_saved_books(db_path) == []


class TestGenerateAllChapters:
    """Cover the batch runner and its progress log."""

    def test_returns_empty_when_nothing_to_do(self, db_path):
        from bible_study.summary import generate_all_chapters
        assert generate_all_chapters(db_path) == []

    def test_summarizes_pending_chapters(self, db_path, mocker):
        from bible_study.summary import generate_all_chapters
        mocker.patch("bible_study.summary.time.sleep")
        mocker.patch("bible_study.ollama.generate", return_value="Chapter summary.")
        upsert_verses(db_path, "Genesis", 1, [(1, "verse one")])
        upsert_verses(db_path, "Genesis", 2, [(1, "verse two")])
        results = generate_all_chapters(db_path)
        assert sorted(results) == [("Genesis", 1), ("Genesis", 2)]

    def test_writes_progress_file(self, db_path, tmp_path, mocker):
        from bible_study.summary import generate_all_chapters
        mocker.patch("bible_study.summary.time.sleep")
        mocker.patch("bible_study.ollama.generate", return_value="Summary.")
        upsert_verses(db_path, "Genesis", 1, [(1, "verse one")])
        progress = tmp_path / "progress.md"
        generate_all_chapters(db_path, progress_file=progress)
        text = progress.read_text()
        assert "# Bible Summary Progress" in text
        assert "Genesis Ch.1" in text
        assert "summarised" in text

    def test_defaults_progress_file_next_to_db(self, db_path, mocker):
        from bible_study.summary import generate_all_chapters
        mocker.patch("bible_study.summary.time.sleep")
        mocker.patch("bible_study.ollama.generate", return_value="Summary.")
        upsert_verses(db_path, "Genesis", 1, [(1, "verse one")])
        generate_all_chapters(db_path)
        assert (db_path.parent / "SUMMARY_PROGRESS.md").exists()

    def test_records_failures_without_aborting(self, db_path, tmp_path, mocker):
        from bible_study.summary import generate_all_chapters
        mocker.patch("bible_study.summary.time.sleep")
        mocker.patch(
            "bible_study.ollama.generate",
            side_effect=[RuntimeError("ollama down"), "Second summary."],
        )
        upsert_verses(db_path, "Genesis", 1, [(1, "verse one")])
        upsert_verses(db_path, "Genesis", 2, [(1, "verse two")])
        progress = tmp_path / "progress.md"
        results = generate_all_chapters(db_path, progress_file=progress)
        assert results == [("Genesis", 2)]
        text = progress.read_text()
        assert "**FAILED**" in text
        assert "ollama down" in text

    def test_skips_already_summarized_chapters(self, db_path, mocker):
        from bible_study.summary import generate_all_chapters
        mocker.patch("bible_study.summary.time.sleep")
        mocker.patch("bible_study.ollama.generate", return_value="New summary.")
        upsert_verses(db_path, "Genesis", 1, [(1, "verse one")])
        upsert_verses(db_path, "Genesis", 2, [(1, "verse two")])
        save_summary(db_path, "Genesis", 1, "already done")
        results = generate_all_chapters(db_path)
        assert results == [("Genesis", 2)]


class TestRenderChapterMarkdown:
    """Cover cross-link and verse-text branches of render_chapter_markdown."""

    def test_first_chapter_links_forward_only(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown("Genesis", 1, "Summary")
        assert "Genesis Ch. 2" in md
        assert "Genesis Ch. 0" not in md

    def test_middle_chapter_links_both_ways(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown("Genesis", 25, "Summary")
        assert "Genesis Ch. 24" in md
        assert "Genesis Ch. 26" in md

    def test_last_chapter_links_backward_only(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown("Genesis", 50, "Summary")
        assert "Genesis Ch. 49" in md
        assert "Genesis Ch. 51" not in md

    def test_single_chapter_book_has_no_links(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown("Obadiah", 1, "Summary")
        assert "(first or last chapter)" in md

    def test_verses_text_rendered_in_details_block(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown(
            "Genesis", 1, "Summary", verses_text="In the beginning",
        )
        assert "<details><summary>Original KJV Text</summary>" in md
        assert "In the beginning" in md
        assert "</details>" in md

    def test_verses_text_omitted_when_absent(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown("Genesis", 1, "Summary")
        assert "<details>" not in md

    def test_links_use_lowercase_abbreviation(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown("Genesis", 1, "Summary")
        assert "gen/chapter-02.md" in md


class TestExportMarkdowns:
    """Cover export_markdowns output structure."""

    def test_requires_output_dir(self, db_path):
        from bible_study.summary import export_markdowns
        with pytest.raises(ValueError, match="output_dir is required"):
            export_markdowns(db_path)

    def test_returns_counts_per_book(self, db_path, tmp_path):
        from bible_study.summary import export_markdowns
        save_summary(db_path, "Genesis", 1, "summary one")
        save_summary(db_path, "Genesis", 2, "summary two")
        results = export_markdowns(db_path, output_dir=tmp_path / "out")
        assert results == {"GEN": 2}

    def test_writes_chapter_files(self, db_path, tmp_path):
        from bible_study.summary import export_markdowns
        out = tmp_path / "out"
        save_summary(db_path, "Genesis", 1, "summary one")
        export_markdowns(db_path, output_dir=out)
        chapter_file = out / "gen" / "chapter-01.md"
        assert chapter_file.exists()
        assert "summary one" in chapter_file.read_text()

    def test_writes_master_index(self, db_path, tmp_path):
        from bible_study.summary import export_markdowns
        out = tmp_path / "out"
        save_summary(db_path, "Genesis", 1, "summary one")
        export_markdowns(db_path, output_dir=out)
        index = (out / "index.md").read_text()
        assert "Bible Study -- Summary Index" in index
        assert "(1/50 chapters)" in index
        assert "(not summarised)" in index

    def test_writes_per_book_index(self, db_path, tmp_path):
        from bible_study.summary import export_markdowns
        out = tmp_path / "out"
        save_summary(db_path, "Genesis", 1, "summary one")
        export_markdowns(db_path, output_dir=out)
        book_index = (out / "gen" / "index.md").read_text()
        assert "# Genesis" in book_index
        assert "chapter-01.md" in book_index

    def test_empty_db_produces_index_only(self, db_path, tmp_path):
        from bible_study.summary import export_markdowns
        out = tmp_path / "out"
        results = export_markdowns(db_path, output_dir=out)
        assert results == {}
        assert (out / "index.md").exists()
        assert not list(out.glob("*/chapter-*.md"))
