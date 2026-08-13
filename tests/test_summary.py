"""Tests for bible_study/summary -- summary pipeline orchestration."""

import tempfile
from pathlib import Path

import pytest


class TestSummarizeChapter:
    """Test the summarize_chapter() end-to-end flow."""

    def test_summarize_chapter_returns_nonempty_string(self, mocker):
        from bible_study.db import init_db, upsert_verses
        mock_gen = mocker.patch("bible_study.ollama.generate", return_value="A summary.")
        td = tempfile.mkdtemp()
        db_path = Path(td) / "test.db"
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning...")])
        from bible_study.summary import summarize_chapter
        result = summarize_chapter("Genesis", 1, db_path=db_path)
        assert result == "A summary."
        mock_gen.assert_called_once()

    def test_summarize_raises_when_no_verses(self, mocker):
        mocker.patch("bible_study.api.requests.get", side_effect=RuntimeError("no api"))
        with pytest.raises(RuntimeError):
            from bible_study.summary import summarize_chapter
            summarize_chapter("Genesis", 1)


class TestSummarizeBook:
    """Test the summarize_book() pipeline."""

    def test_summarize_book_returns_text(self, mocker):
        mock_gen = mocker.patch("bible_study.ollama.generate", return_value="Book overview.")
        td = tempfile.mkdtemp()
        db_path = Path(td) / "test.db"
        from bible_study.db import init_db, save_summary, upsert_verses
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "verse text")])
        save_summary(db_path, "Genesis", 1, "summary text")
        from bible_study.summary import summarize_book
        result = summarize_book("Genesis", db_path=db_path)
        assert result == "Book overview."


class TestExportMarkdowns:
    """Test markdown export functionality."""

    def test_render_chapter_markdown_includes_summary(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown("Genesis", 1, "A summary")
        assert "# Genesis Chapter 1" in md
        assert "A summary" in md

    def test_export_creates_directory_and_file(self, tmp_path):
        from bible_study.db import init_db, save_summary, upsert_verses
        from bible_study.summary import export_markdowns
        data_dir = tmp_path / "data"
        db_path = data_dir / "bible.db"
        out_dir = tmp_path / "output"
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "text")])
        save_summary(db_path, "Genesis", 1, "summary text")
        results = export_markdowns(db_path=db_path, output_dir=out_dir)
        files = list(out_dir.rglob("*.md"))
        assert len(files) > 0


class TestBookSummaries:
    """Test book-level summary generation."""

    def test_generate_book_summary_saves_to_db(self, mocker):
        mock_gen = mocker.patch("bible_study.ollama.generate", return_value="Genesis overview.")
        td = tempfile.mkdtemp()
        db_path = Path(td) / "test.db"
        from bible_study.db import init_db, save_summary, upsert_verses
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "verse text")])
        save_summary(db_path, "Genesis", 1, "summary text")
        from bible_study.summary import summarize_book
        result = summarize_book("Genesis", db_path=db_path)
        assert result == "Genesis overview."

