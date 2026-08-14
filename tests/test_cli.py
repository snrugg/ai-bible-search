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
        for name in ("init", "summarize", "summarize-book", "view", "status",
                     "embed", "ask", "search"):
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
            "embed", "ask", "search",
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
        assert mock_serve.call_args.kwargs["port"] == 8080

    def test_view_accepts_custom_port(self, runner, mocker):
        from bible_study.cli import cli
        mock_serve = mocker.patch("bible_study.cli.browse")
        result = runner.invoke(cli, ["view", "--port", "9090"])
        assert result.exit_code == 0
        assert mock_serve.call_args.kwargs["port"] == 9090

    def test_view_passes_data_dir_db_to_server(self, runner, mocker, tmp_path):
        from pathlib import Path

        from bible_study.cli import cli
        from bible_study.db import init_db
        mock_serve = mocker.patch("bible_study.cli.browse")
        data_dir = tmp_path / "mydata"
        init_db(data_dir / "bible.db")
        result = runner.invoke(cli, ["view", "--data-dir", str(data_dir)])
        assert result.exit_code == 0
        assert mock_serve.call_args.kwargs["db_path"] == Path(data_dir) / "bible.db"
        assert "Warning" not in result.output

    def test_view_warns_when_database_missing(self, runner, mocker, tmp_path):
        from bible_study.cli import cli
        mocker.patch("bible_study.cli.browse")
        result = runner.invoke(cli, ["view", "--data-dir", str(tmp_path / "empty")])
        assert result.exit_code == 0
        assert "run `init` first" in result.output


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
            assert "Remaining:      1" in result.output


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


class TestEmbedCommand:
    """bible-study embed."""

    def _seed(self):
        from pathlib import Path
        from bible_study.db import init_db, save_summary, upsert_verses
        db_path = Path("data/bible.db")
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
        save_summary(db_path, "Genesis", 1, "A summary.")
        return db_path

    def test_embeds_and_reports(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.ollama.embed",
            side_effect=lambda texts, **kw: [[0.1] * 1024 for _ in texts],
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["embed"])
            assert result.exit_code == 0, result.output
            assert "Chunked 2" in result.output
            assert "Embedded 2 chunks." in result.output
            assert "Done!" in result.output

    def test_echoes_the_embedding_model(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.ollama.embed",
            side_effect=lambda texts, **kw: [[0.1] * 1024 for _ in texts],
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["embed"])
            assert "Using embedding model:" in result.output

    def test_warns_when_model_missing(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=False)
        mocker.patch(
            "bible_study.ollama.embed",
            side_effect=lambda texts, **kw: [[0.1] * 1024 for _ in texts],
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["embed"])
            assert "ollama pull" in result.output

    def test_rebuild_clears_vectors(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.ollama.embed",
            side_effect=lambda texts, **kw: [[0.1] * 1024 for _ in texts],
        )
        with runner.isolated_filesystem():
            self._seed()
            runner.invoke(cli, ["embed"])
            result = runner.invoke(cli, ["embed", "--rebuild"])
            assert result.exit_code == 0, result.output
            assert "Cleared 2 vectors." in result.output

    def test_limit_is_honoured(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.ollama.embed",
            side_effect=lambda texts, **kw: [[0.1] * 1024 for _ in texts],
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["embed", "--limit", "1"])
            assert "Embedded 1 chunks." in result.output

    def test_errors_when_ollama_is_down(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=False)
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["embed"])
            assert result.exit_code != 0
            assert "Cannot reach Ollama" in result.output

    def test_errors_when_database_missing(self, runner):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["embed"])
            assert result.exit_code != 0
            assert "run `init` first" in result.output

    def test_reports_total_failure(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.ollama.embed", side_effect=RuntimeError("down"),
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["embed"])
            assert result.exit_code != 0
            assert "failed to embed" in result.output

    def test_honours_data_dir(self, runner, mocker, tmp_path):
        from bible_study.cli import cli
        from bible_study.db import chunk_counts, init_db, upsert_verses
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.ollama.embed",
            side_effect=lambda texts, **kw: [[0.1] * 1024 for _ in texts],
        )
        db_path = tmp_path / "alt" / "bible.db"
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
        result = runner.invoke(cli, ["embed", "-d", str(tmp_path / "alt")])
        assert result.exit_code == 0, result.output
        assert chunk_counts(db_path)["verse"] == (1, 1)

    def test_surfaces_a_dims_change_as_a_clean_error(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.ollama.embed",
            side_effect=lambda texts, **kw: [[0.1] * 1024 for _ in texts],
        )
        with runner.isolated_filesystem():
            self._seed()
            runner.invoke(cli, ["embed"])
            mocker.patch("bible_study.prompts.get_embed_dims", return_value=512)
            result = runner.invoke(cli, ["embed"])
            assert result.exit_code != 0
            assert "--rebuild" in result.output


class TestAskCommand:
    """bible-study ask."""

    @pytest.fixture
    def wired(self, mocker):
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        return mocker.patch(
            "bible_study.rag.answer_question",
            return_value={
                "question": "q",
                "answer": "Abraham left Ur because God called him.",
                "sources": [
                    {"citation": "Genesis 11:27-32", "kind": "verses",
                     "book_name": "Genesis", "chapter": 11},
                    {"citation": "Genesis 12 (summary)",
                     "kind": "chapter-summary",
                     "book_name": "Genesis", "chapter": 12},
                ],
                "dropped": 0,
                "prompt": "P",
            },
        )

    def _seed(self):
        from pathlib import Path
        from bible_study.db import init_db, upsert_verses
        db_path = Path("data/bible.db")
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
        return db_path

    def test_prints_the_answer(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "why did Abraham leave Ur?"])
            assert result.exit_code == 0, result.output
            assert "Abraham left Ur because God called him." in result.output

    def test_lists_sources(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "why?"])
            assert "Sources:" in result.output
            assert "- Genesis 11:27-32" in result.output

    def test_no_show_sources_hides_them(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "why?", "--no-show-sources"])
            assert "Sources:" not in result.output

    def test_accepts_an_unquoted_question(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "why", "did", "Abraham", "go"])
            assert result.exit_code == 0
            assert wired.call_args[0][0] == "why did Abraham go"

    def test_top_k_reaches_answer_question(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            runner.invoke(cli, ["ask", "why?", "-k", "12"])
            assert wired.call_args.kwargs["k_verse"] == 12
            assert wired.call_args.kwargs["k_chapter"] == 6
            assert wired.call_args.kwargs["k_book"] == 3

    def test_reports_dropped_sources(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.rag.answer_question",
            return_value={"question": "q", "answer": "A", "sources": [],
                          "dropped": 3, "prompt": "P"},
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "why?"])
            assert "3 more omitted" in result.output

    def test_errors_when_database_missing(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["ask", "why?"])
            assert result.exit_code != 0
            assert "run `init` first" in result.output

    def test_errors_when_ollama_is_down(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=False)
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "why?"])
            assert result.exit_code != 0
            assert "Cannot reach Ollama" in result.output

    def test_missing_index_becomes_a_clean_error(self, runner, mocker):
        from bible_study.cli import cli
        from bible_study.vectors import VectorIndexError
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.rag.answer_question",
            side_effect=VectorIndexError("No vector index found"),
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "why?"])
            assert result.exit_code != 0
            assert "No vector index found" in result.output
            assert "Traceback" not in result.output

    def test_prompt_too_long_becomes_a_clean_error(self, runner, mocker):
        from bible_study.cli import cli
        from bible_study.ollama import PromptTooLongError
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=True)
        mocker.patch(
            "bible_study.rag.answer_question",
            side_effect=PromptTooLongError("Prompt is ~9 tokens"),
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "why?"])
            assert result.exit_code != 0
            assert "Prompt is ~9 tokens" in result.output

    def test_blank_question_is_rejected(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["ask", "   "])
            assert result.exit_code != 0
            assert "Ask a question" in result.output


class TestSearchCommand:
    """bible-study search."""

    def _seed(self):
        from pathlib import Path
        from bible_study.db import init_db, save_book_summary, save_summary, upsert_verses
        db_path = Path("data/bible.db")
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
        save_summary(db_path, "Genesis", 1, "Creation summary.")
        save_book_summary(db_path, "Genesis", "GEN", "Book of beginnings.")
        return db_path

    def _hit(self, tier="verse", citation="Genesis 1:1-5"):
        return {
            "tier": tier, "book_name": "Genesis", "chapter": 1,
            "verse_start": 1, "verse_end": 5, "id": 1, "chunk_id": 1,
            "distance": 0.1234, "rank": 0,
            "citation": citation, "text": "In the beginning God created.",
        }

    @pytest.fixture
    def wired(self, mocker):
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.vectors.embed_query", return_value=[1.0])
        return mocker.patch(
            "bible_study.vectors.search", return_value=[self._hit()],
        )

    def test_prints_ranked_hits(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["search", "beginning"])
            assert result.exit_code == 0, result.output
            assert "Genesis 1:1-5" in result.output
            assert "0.1234" in result.output

    def test_json_output_parses(self, runner, wired):
        import json
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["search", "beginning", "--json"])
            assert result.exit_code == 0, result.output
            rows = json.loads(result.output)
            assert rows[0]["citation"] == "Genesis 1:1-5"
            assert rows[0]["kind"] == "verse"
            assert "text" in rows[0]

    def test_expand_returns_ranked_blocks(self, runner, wired):
        import json
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(
                cli, ["search", "beginning", "--expand", "--json"],
            )
            assert result.exit_code == 0, result.output
            rows = json.loads(result.output)
            kinds = {row["kind"] for row in rows}
            assert "verses" in kinds
            assert "chapter-summary" in kinds
            assert all("score" in row for row in rows)
            assert any(row["is_expansion"] for row in rows)

    def test_top_k_reaches_search(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            runner.invoke(cli, ["search", "x", "-k", "12"])
            assert wired.call_args[0][2] == {
                "verse": 12, "chapter": 6, "book": 3,
            }

    def test_accepts_an_unquoted_query(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["search", "covenant", "with", "Abraham"])
            assert result.exit_code == 0

    def test_reports_no_matches(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.vectors.embed_query", return_value=[1.0])
        mocker.patch("bible_study.vectors.search", return_value=[])
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["search", "nothing"])
            assert "No matches." in result.output

    def test_empty_json_is_still_valid(self, runner, mocker):
        import json
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.vectors.embed_query", return_value=[1.0])
        mocker.patch("bible_study.vectors.search", return_value=[])
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["search", "nothing", "--json"])
            assert json.loads(result.output) == []

    def test_errors_when_database_missing(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["search", "x"])
            assert result.exit_code != 0
            assert "run `init` first" in result.output

    def test_errors_when_ollama_is_down(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=False)
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["search", "x"])
            assert result.exit_code != 0
            assert "Cannot reach Ollama" in result.output

    def test_missing_index_becomes_a_clean_error(self, runner, mocker):
        from bible_study.cli import cli
        from bible_study.vectors import VectorIndexError
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch(
            "bible_study.vectors.embed_query",
            side_effect=VectorIndexError("No vector index found"),
        )
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["search", "x"])
            assert result.exit_code != 0
            assert "No vector index found" in result.output
            assert "Traceback" not in result.output

    def test_blank_query_is_rejected(self, runner, wired):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            self._seed()
            result = runner.invoke(cli, ["search", "   "])
            assert result.exit_code != 0
            assert "something to search for" in result.output

    def test_honours_data_dir(self, runner, mocker, tmp_path):
        from bible_study.cli import cli
        from bible_study.db import init_db, upsert_verses
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.vectors.embed_query", return_value=[1.0])
        mock_search = mocker.patch(
            "bible_study.vectors.search", return_value=[self._hit()],
        )
        db_path = tmp_path / "alt" / "bible.db"
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
        result = runner.invoke(cli, ["search", "x", "-d", str(tmp_path / "alt")])
        assert result.exit_code == 0, result.output
        assert mock_search.call_args[0][0] == db_path


class TestStatusChunkCounts:
    """status reports chunk/embedded counts once `embed` has run."""

    def test_reports_chunk_counts(self, runner):
        from pathlib import Path
        from bible_study.cli import cli
        from bible_study.db import init_db, upsert_chunk, upsert_verses
        with runner.isolated_filesystem():
            db_path = Path("data/bible.db")
            init_db(db_path)
            upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
            upsert_chunk(
                db_path, "verse", "Genesis", 1, 1, 5, "Genesis 1:1-5", "t", "h",
            )
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "Chunks:         1" in result.output
            assert "Embedded:       0" in result.output

    def test_omits_chunk_counts_before_embedding(self, runner):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "Chunks:" not in result.output


class TestAskModelWarnings:
    """ask soft-warns for each model Ollama does not list."""

    def test_warns_for_both_models(self, runner, mocker):
        from pathlib import Path
        from bible_study.cli import cli
        from bible_study.db import init_db, upsert_verses
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.ollama.check_model_available", return_value=False)
        mocker.patch(
            "bible_study.rag.answer_question",
            return_value={"question": "q", "answer": "A", "sources": [],
                          "dropped": 0, "prompt": "P"},
        )
        with runner.isolated_filesystem():
            db_path = Path("data/bible.db")
            init_db(db_path)
            upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning")])
            result = runner.invoke(cli, ["ask", "why?"])
            assert result.output.count("was not listed by Ollama") == 2
