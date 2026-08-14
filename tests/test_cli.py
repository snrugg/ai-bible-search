"""Tests for bible_study/cli -- Click CLI commands."""

from click.testing import CliRunner

import pytest


@pytest.fixture
def runner():
    """Return a Click CliRunner."""
    return CliRunner()


class TestCliGroup:
    """Root group wiring."""

    def test_cli_group_exists(self):
        from bible_study.cli import cli
        assert callable(cli)

    def test_help_lists_every_command(self, runner):
        from bible_study.cli import cli
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for name in ("init", "summarize", "summarize-book", "view", "status"):
            assert name in result.output

    def test_no_args_exits_cleanly(self, runner):
        from bible_study.cli import cli
        result = runner.invoke(cli, [])
        assert result.exit_code in (0, 2)

    def test_all_commands_registered(self):
        from bible_study.cli import cli
        assert set(cli.commands) == {
            "init", "summarize", "summarize-book", "view", "status",
            "export", "clear-summaries", "clear-book-summaries", "config-edit",
        }


class TestInitCommand:
    """bible-study init."""

    def test_init_creates_database(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.api.download_all", return_value=[])
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "Indexed 66 books." in result.output

    def test_init_reports_downloaded_chapters(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch(
            "bible_study.api.download_all",
            return_value=[("Genesis", 1), ("Genesis", 2)],
        )
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "Downloaded 2 chapters." in result.output

    def test_init_with_custom_data_dir(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.api.download_all", return_value=[])
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "-d", "custom"])
            assert result.exit_code == 0
            assert "custom" in result.output

    def test_init_accepts_force_flag(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.api.download_all", return_value=[])
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "-f"])
            assert result.exit_code == 0


class TestSummarizeCommand:
    """bible-study summarize."""

    def test_summarize_reports_done(self, runner, mocker):
        from bible_study.cli import cli
        mock_gen = mocker.patch(
            "bible_study.cli.generate_all_chapters", return_value=[],
        )
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["summarize"])
            assert result.exit_code == 0
            assert "Done!" in result.output
            mock_gen.assert_called_once()

    def test_summarize_uses_custom_data_dir(self, runner, mocker):
        from bible_study.cli import cli
        mock_gen = mocker.patch(
            "bible_study.cli.generate_all_chapters", return_value=[],
        )
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["summarize", "-d", "mydata"])
            assert result.exit_code == 0
            assert "mydata" in str(mock_gen.call_args[0][0])


class TestSummarizeBookCommand:
    """bible-study summarize-book."""

    def test_summarize_book_with_no_books(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.cli.summarize_book", return_value="x")
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["summarize-book"])
            assert result.exit_code == 0
            assert "All book summaries generated." in result.output

    def test_summarize_book_iterates_stored_books(self, runner, mocker):
        from bible_study.cli import cli
        mock_sum = mocker.patch(
            "bible_study.cli.summarize_book", return_value="summary",
        )
        mocker.patch(
            "bible_study.db.get_all_book_names", return_value=["Genesis", "Exodus"],
        )
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["summarize-book"])
            assert result.exit_code == 0
            assert "Summarizing Genesis" in result.output
            assert "Summarizing Exodus" in result.output
            assert mock_sum.call_count == 2


class TestViewCommand:
    """bible-study view."""

    def test_view_starts_server_on_default_port(self, runner, mocker):
        from bible_study.cli import cli
        mock_serve = mocker.patch("bible_study.cli.browse")
        result = runner.invoke(cli, ["view"])
        assert result.exit_code == 0
        mock_serve.assert_called_once_with(port=8080)

    def test_view_accepts_custom_port(self, runner, mocker):
        from bible_study.cli import cli
        mock_serve = mocker.patch("bible_study.cli.browse")
        result = runner.invoke(cli, ["view", "--port", "9090"])
        assert result.exit_code == 0
        mock_serve.assert_called_once_with(port=9090)


class TestStatusCommand:
    """bible-study status."""

    def test_status_reports_zero_progress(self, runner):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "Total chapters: 0" in result.output
            assert "Summarized:     0" in result.output

    def test_status_counts_summarized_chapters(self, runner):
        from bible_study.cli import cli
        from bible_study.db import init_db, save_summary, upsert_verses
        with runner.isolated_filesystem():
            from pathlib import Path
            db_path = Path("data/bible.db")
            init_db(db_path)
            upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
            upsert_verses(db_path, "Genesis", 2, [(1, "Thus the heavens")])
            save_summary(db_path, "Genesis", 1, "a summary")
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "Total chapters: 2" in result.output
            assert "Summarized:     1" in result.output
            assert "Remaining:     1" in result.output


class TestClearSummariesCommand:
    """bible-study clear-summaries."""

    def _seed(self):
        """Create a db with two summarised books; returns its path."""
        from pathlib import Path

        from bible_study.db import (
            init_db,
            save_book_summary,
            save_summary,
            upsert_verses,
        )
        db_path = Path("data/bible.db")
        init_db(db_path)
        for book, abbrev in (("Genesis", "GEN"), ("Exodus", "EXO")):
            for chap in (1, 2):
                upsert_verses(db_path, book, chap, [(1, "verse text")])
                save_summary(db_path, book, chap, f"{book} {chap} summary")
            save_book_summary(db_path, book, abbrev, f"{book} overview")
        return db_path

    def test_clears_everything_with_yes_flag(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_chapter_progress, get_saved_books
        from bible_study.indexer import book_names
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(cli, ["clear-summaries", "--yes"])
            assert result.exit_code == 0
            assert "Cleared 4 chapter summaries and 2 book summaries" in result.output
            assert get_chapter_progress(db_path, book_names())[1] == 0
            assert get_saved_books(db_path) == []

    def test_keeps_verse_text(self, runner):
        from bible_study.cli import cli
        from bible_study.db import verse_count
        with runner.isolated_filesystem():
            db_path = self._seed()
            runner.invoke(cli, ["clear-summaries", "--yes"])
            assert verse_count(db_path) == 4

    def test_book_option_scopes_the_delete(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_chapter_summaries_for_book, get_saved_books
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(cli, ["clear-summaries", "--book", "genesis", "-y"])
            assert result.exit_code == 0
            assert get_chapter_summaries_for_book(db_path, "Genesis") == []
            assert len(get_chapter_summaries_for_book(db_path, "Exodus")) == 2
            assert get_saved_books(db_path) == ["Exodus"]

    def test_scope_chapters_leaves_book_summaries(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_saved_books
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(
                cli, ["clear-summaries", "--scope", "chapters", "-y"],
            )
            assert "Cleared 4 chapter summaries and 0 book summaries" in result.output
            assert len(get_saved_books(db_path)) == 2

    def test_scope_books_leaves_chapter_summaries(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_chapter_summaries_for_book, get_saved_books
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(cli, ["clear-summaries", "--scope", "books", "-y"])
            assert "Cleared 0 chapter summaries and 2 book summaries" in result.output
            assert get_saved_books(db_path) == []
            assert len(get_chapter_summaries_for_book(db_path, "Genesis")) == 2

    def test_prompts_for_confirmation_and_aborts(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_chapter_summaries_for_book
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(cli, ["clear-summaries"], input="n\n")
            assert result.exit_code != 0
            assert "Delete chapter and book summaries for all 66 books?" in result.output
            assert len(get_chapter_summaries_for_book(db_path, "Genesis")) == 2

    def test_confirmation_accepted_deletes(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_chapter_summaries_for_book
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(cli, ["clear-summaries"], input="y\n")
            assert result.exit_code == 0
            assert get_chapter_summaries_for_book(db_path, "Genesis") == []

    def test_rejects_unknown_book(self, runner):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["clear-summaries", "--book", "nope", "-y"])
            assert result.exit_code != 0
            assert "Unknown book: nope" in result.output

    def test_errors_when_database_missing(self, runner):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["clear-summaries", "-y"])
            assert result.exit_code != 0
            assert "nothing to clear" in result.output

    def test_honours_data_dir(self, runner, tmp_path):
        from bible_study.cli import cli
        from bible_study.db import init_db, save_summary
        db_path = tmp_path / "alt" / "bible.db"
        init_db(db_path)
        save_summary(db_path, "Genesis", 1, "a summary")
        result = runner.invoke(
            cli, ["clear-summaries", "--data-dir", str(tmp_path / "alt"), "-y"],
        )
        assert result.exit_code == 0
        assert "Cleared 1 chapter summaries" in result.output


class TestClearBookSummariesCommand:
    """bible-study clear-book-summaries."""

    def _seed(self):
        """Create a db with two summarised books; returns its path."""
        from pathlib import Path

        from bible_study.db import (
            init_db,
            save_book_summary,
            save_summary,
            upsert_verses,
        )
        db_path = Path("data/bible.db")
        init_db(db_path)
        for book, abbrev in (("Genesis", "GEN"), ("Exodus", "EXO")):
            for chap in (1, 2):
                upsert_verses(db_path, book, chap, [(1, "verse text")])
                save_summary(db_path, book, chap, f"{book} {chap} summary")
            save_book_summary(db_path, book, abbrev, f"{book} overview")
        return db_path

    def test_clears_book_summaries_with_yes_flag(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_saved_books
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(cli, ["clear-book-summaries", "--yes"])
            assert result.exit_code == 0
            assert "Cleared 2 book summaries." in result.output
            assert get_saved_books(db_path) == []

    def test_keeps_chapter_summaries_and_verses(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_chapter_summaries_for_book, verse_count
        with runner.isolated_filesystem():
            db_path = self._seed()
            runner.invoke(cli, ["clear-book-summaries", "-y"])
            assert len(get_chapter_summaries_for_book(db_path, "Genesis")) == 2
            assert verse_count(db_path) == 4

    def test_book_option_scopes_the_delete(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_saved_books
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(
                cli, ["clear-book-summaries", "--book", "genesis", "-y"],
            )
            assert result.exit_code == 0
            assert get_saved_books(db_path) == ["Exodus"]

    def test_prompts_for_confirmation_and_aborts(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_saved_books
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(cli, ["clear-book-summaries"], input="n\n")
            assert result.exit_code != 0
            assert "Delete book summaries for all 66 books?" in result.output
            assert len(get_saved_books(db_path)) == 2

    def test_confirmation_accepted_deletes(self, runner):
        from bible_study.cli import cli
        from bible_study.db import get_saved_books
        with runner.isolated_filesystem():
            db_path = self._seed()
            result = runner.invoke(cli, ["clear-book-summaries"], input="y\n")
            assert result.exit_code == 0
            assert get_saved_books(db_path) == []

    def test_rejects_unknown_book(self, runner):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(
                cli, ["clear-book-summaries", "--book", "nope", "-y"],
            )
            assert result.exit_code != 0
            assert "Unknown book: nope" in result.output

    def test_errors_when_database_missing(self, runner):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["clear-book-summaries", "-y"])
            assert result.exit_code != 0
            assert "nothing to clear" in result.output

    def test_honours_data_dir(self, runner, tmp_path):
        from bible_study.cli import cli
        from bible_study.db import init_db, save_book_summary
        db_path = tmp_path / "alt" / "bible.db"
        init_db(db_path)
        save_book_summary(db_path, "Genesis", "GEN", "an overview")
        result = runner.invoke(
            cli, ["clear-book-summaries", "--data-dir", str(tmp_path / "alt"), "-y"],
        )
        assert result.exit_code == 0
        assert "Cleared 1 book summaries." in result.output


class TestConfigEditCommand:
    """bible-study config-edit."""

    def test_config_edit_opens_existing_file(self, runner, mocker):
        from bible_study.cli import cli
        mock_open = mocker.patch("webbrowser.open")
        with runner.isolated_filesystem():
            from pathlib import Path
            Path("config.yaml").write_text("chapter_summary: |\n  hi\n")
            result = runner.invoke(cli, ["config-edit"])
            assert result.exit_code == 0
            mock_open.assert_called_once()

    def test_config_edit_reports_missing_file(self, runner, mocker):
        from bible_study.cli import cli
        mock_open = mocker.patch("webbrowser.open")
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["config-edit"])
            assert result.exit_code == 0
            assert "Config file not found" in result.output
            mock_open.assert_not_called()
