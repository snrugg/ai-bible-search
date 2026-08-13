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

    def test_all_six_commands_registered(self):
        from bible_study.cli import cli
        assert set(cli.commands) == {
            "init", "summarize", "summarize-book", "view", "status",
            "export", "config-edit",
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
