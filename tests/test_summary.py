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


class TestContextWindowPlumbing:
    """The configured num_ctx reaches every Ollama call."""

    def _seeded_db(self, tmp_path):
        from bible_study.db import init_db, save_summary, upsert_verses
        db_path = tmp_path / "test.db"
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "verse text")])
        save_summary(db_path, "Genesis", 1, "summary text")
        return db_path

    def test_chapter_summary_passes_configured_num_ctx(
        self, tmp_path, monkeypatch, mocker,
    ):
        mock_gen = mocker.patch("bible_study.ollama.generate", return_value="S")
        db_path = self._seeded_db(tmp_path)
        (tmp_path / "config.yaml").write_text(
            "chapter_summary: |\n  {chapter_text}\nollama_num_ctx: 65536\n",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIBLE_STUDY_CONFIG", raising=False)
        from bible_study.summary import summarize_chapter
        summarize_chapter("Genesis", 1, db_path=db_path)
        assert mock_gen.call_args.kwargs["num_ctx"] == 65536

    def test_book_summary_passes_configured_num_ctx(
        self, tmp_path, monkeypatch, mocker,
    ):
        mock_gen = mocker.patch("bible_study.ollama.generate", return_value="S")
        db_path = self._seeded_db(tmp_path)
        (tmp_path / "config.yaml").write_text(
            "book_summary: |\n  {chapter_summaries}\nollama_num_ctx: 65536\n",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIBLE_STUDY_CONFIG", raising=False)
        from bible_study.summary import summarize_book
        summarize_book("Genesis", db_path=db_path)
        assert mock_gen.call_args.kwargs["num_ctx"] == 65536

    def test_explicit_kwarg_overrides_config(self, tmp_path, monkeypatch, mocker):
        mock_gen = mocker.patch("bible_study.ollama.generate", return_value="S")
        db_path = self._seeded_db(tmp_path)
        (tmp_path / "config.yaml").write_text(
            "book_summary: |\n  {chapter_summaries}\nollama_num_ctx: 65536\n",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIBLE_STUDY_CONFIG", raising=False)
        from bible_study.summary import summarize_book
        summarize_book("Genesis", db_path=db_path, ollama_kwargs={"num_ctx": 4096})
        assert mock_gen.call_args.kwargs["num_ctx"] == 4096

    def test_oversized_book_summary_raises_instead_of_truncating(
        self, tmp_path, monkeypatch,
    ):
        """The Psalms failure mode: too many chapter summaries to fit."""
        from bible_study.db import init_db, save_summary, upsert_verses
        from bible_study.ollama import PromptTooLongError
        db_path = tmp_path / "test.db"
        init_db(db_path)
        for chapter in range(1, 51):
            upsert_verses(db_path, "Genesis", chapter, [(1, "verse text")])
            save_summary(db_path, "Genesis", chapter, "x" * 2000)
        (tmp_path / "config.yaml").write_text(
            "book_summary: |\n  {chapter_summaries}\nollama_num_ctx: 4096\n",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIBLE_STUDY_CONFIG", raising=False)
        from bible_study.summary import summarize_book
        with pytest.raises(PromptTooLongError):
            summarize_book("Genesis", db_path=db_path)


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

