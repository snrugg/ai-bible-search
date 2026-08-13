"""Tests for the end-to-end pipeline wiring: init -> summarize -> status."""

import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fake_api(mocker):
    """Stub bible-api.com with a two-verse chapter."""
    resp = mocker.MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"verses": [
        {"verse": 1, "text": "The vision of Obadiah."},
        {"verse": 2, "text": "Behold, I have made thee small."},
    ]}
    mocker.patch("bible_study.api.time.sleep")
    return mocker.patch("bible_study.api.requests.get", return_value=resp)


@pytest.fixture
def one_book(mocker):
    """Restrict the canon to a single book so tests stay fast."""
    return mocker.patch("bible_study.indexer.book_names", return_value=["Obadiah"])


class TestSingleChapterBooks:
    """The API reads a trailing 1 as a verse for one-chapter books."""

    def test_single_chapter_books_are_detected(self):
        from bible_study.api import _is_single_chapter_book
        for name in ("Obadiah", "Philemon", "2 John", "3 John", "Jude"):
            assert _is_single_chapter_book(name) is True

    def test_multi_chapter_books_are_not(self):
        from bible_study.api import _is_single_chapter_book
        for name in ("Genesis", "Psalms", "Revelation"):
            assert _is_single_chapter_book(name) is False

    def test_unknown_book_is_not_single_chapter(self):
        from bible_study.api import _is_single_chapter_book
        assert _is_single_chapter_book("Fakebook") is False

    def test_uses_verse_range_for_single_chapter_book(self, mocker):
        from bible_study.api import fetch_chapter
        resp = mocker.MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"verses": [{"verse": 1, "text": "x"}]}
        mock_get = mocker.patch("bible_study.api.requests.get", return_value=resp)
        fetch_chapter("Obadiah", 1)
        assert "1:1-21" in mock_get.call_args[0][0]

    def test_uses_plain_chapter_for_multi_chapter_book(self, mocker):
        from bible_study.api import fetch_chapter
        resp = mocker.MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"verses": [{"verse": 1, "text": "x"}]}
        mock_get = mocker.patch("bible_study.api.requests.get", return_value=resp)
        fetch_chapter("Genesis", 1)
        url = mock_get.call_args[0][0]
        assert url.endswith("Genesis%201")
        assert ":" not in url.split("/")[-1]

    def test_probes_downward_for_unrecorded_book(self, mocker):
        import requests

        from bible_study.api import _fetch_single_chapter_book
        ok = mocker.MagicMock()
        ok.status_code = 200
        ok.raise_for_status.return_value = None
        ok.json.return_value = {"verses": [{"verse": 1, "text": "x"}]}
        bad = mocker.MagicMock()
        bad.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get = mocker.patch(
            "bible_study.api.requests.get", side_effect=[bad, bad, ok],
        )
        result = _fetch_single_chapter_book("Mystery", 1, 30)
        assert len(result) == 1
        assert mock_get.call_count == 3

    def test_raises_when_every_probe_fails(self, mocker):
        import requests

        from bible_study.api import _fetch_single_chapter_book
        bad = mocker.MagicMock()
        bad.raise_for_status.side_effect = requests.HTTPError("404")
        mocker.patch("bible_study.api.requests.get", return_value=bad)
        with pytest.raises(requests.HTTPError):
            _fetch_single_chapter_book("Mystery", 1, 30)


class TestInitPersistsVerses:
    """init must load verses into SQLite, not just the JSON cache."""

    def test_init_writes_verses_to_db(self, runner, fake_api, one_book):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            con = sqlite3.connect("data/bible.db")
            count = con.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
            con.close()
            assert count == 2

    def test_init_reports_stored_verse_count(self, runner, fake_api, one_book):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])
            assert "Stored 2 verses" in result.output

    def test_init_writes_json_cache(self, runner, fake_api, one_book):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            assert (Path("data/api-cache") / "obadiah-1.json").exists()

    def test_rerunning_init_makes_no_new_http_calls(
        self, runner, fake_api, one_book,
    ):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            calls_after_first = fake_api.call_count
            runner.invoke(cli, ["init"])
            assert fake_api.call_count == calls_after_first

    def test_download_all_without_db_path_skips_persistence(
        self, tmp_path, fake_api,
    ):
        from bible_study.api import download_all
        results = download_all(
            book_names_list=["Obadiah"], cache_dir=tmp_path / "cache",
        )
        assert results == [("Obadiah", 1)]


class TestSummarizeReportsFailures:
    """summarize must not report success when nothing was summarised."""

    def test_aborts_when_ollama_unreachable(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=False)
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["summarize"])
            assert result.exit_code == 1
            assert "Cannot reach Ollama" in result.output

    def test_errors_when_every_chapter_fails(self, runner, mocker, one_book):
        from bible_study.cli import cli
        from bible_study.db import init_db, upsert_verses
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.summary.time.sleep")
        mocker.patch(
            "bible_study.ollama.generate", side_effect=RuntimeError("ollama down"),
        )
        with runner.isolated_filesystem():
            db = Path("data/bible.db")
            init_db(db)
            upsert_verses(db, "Obadiah", 1, [(1, "verse text")])
            result = runner.invoke(cli, ["summarize"])
            assert result.exit_code == 1
            assert "failed to summarise" in result.output

    def test_reports_count_on_success(self, runner, mocker, one_book):
        from bible_study.cli import cli
        from bible_study.db import init_db, upsert_verses
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.summary.time.sleep")
        mocker.patch("bible_study.ollama.generate", return_value="A summary.")
        with runner.isolated_filesystem():
            db = Path("data/bible.db")
            init_db(db)
            upsert_verses(db, "Obadiah", 1, [(1, "verse text")])
            result = runner.invoke(cli, ["summarize"])
            assert result.exit_code == 0
            assert "Summarised 1 chapters." in result.output

    def test_succeeds_when_nothing_pending(self, runner, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["summarize"])
            assert result.exit_code == 0
            assert "Summarised 0 chapters." in result.output


class TestFullPipeline:
    """init -> summarize -> status across one book."""

    def test_pipeline_end_to_end(self, runner, fake_api, one_book, mocker):
        from bible_study.cli import cli
        mocker.patch("bible_study.ollama.health_check", return_value=True)
        mocker.patch("bible_study.summary.time.sleep")
        mocker.patch("bible_study.ollama.generate", return_value="Edom's downfall.")
        with runner.isolated_filesystem():
            assert runner.invoke(cli, ["init"]).exit_code == 0
            assert runner.invoke(cli, ["summarize"]).exit_code == 0
            status = runner.invoke(cli, ["status"])
            assert "Total chapters: 1" in status.output
            assert "Summarized:     1" in status.output
            con = sqlite3.connect("data/bible.db")
            stored = con.execute("SELECT summary FROM chapter_summaries").fetchone()
            con.close()
            assert stored[0] == "Edom's downfall."


class TestConfigDiscovery:
    """load_config must work regardless of the working directory."""

    def test_finds_repo_config_from_any_cwd(self, tmp_path, monkeypatch):
        from bible_study.prompts import load_config
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIBLE_STUDY_CONFIG", raising=False)
        config = load_config()
        assert "chapter_summary" in config

    def test_env_var_overrides_discovery(self, tmp_path, monkeypatch):
        from bible_study.prompts import load_config
        custom = tmp_path / "mine.yaml"
        custom.write_text("chapter_summary: |\n  CUSTOM {book_name}\n")
        monkeypatch.setenv("BIBLE_STUDY_CONFIG", str(custom))
        assert "CUSTOM" in load_config()["chapter_summary"]

    def test_cwd_config_wins_over_repo_config(self, tmp_path, monkeypatch):
        from bible_study.prompts import load_config
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIBLE_STUDY_CONFIG", raising=False)
        (tmp_path / "config.yaml").write_text("chapter_summary: |\n  LOCAL\n")
        assert load_config()["chapter_summary"].strip() == "LOCAL"


class TestExportCommand:
    """bible-study export."""

    def test_export_reports_zero_for_empty_db(self, runner):
        from bible_study.cli import cli
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["export"])
            assert result.exit_code == 0
            assert "Exported 0 chapters across 0 books" in result.output

    def test_export_writes_chapter_files(self, runner):
        from bible_study.cli import cli
        from bible_study.db import init_db, save_summary
        with runner.isolated_filesystem():
            db = Path("data/bible.db")
            init_db(db)
            save_summary(db, "Genesis", 1, "A summary of Genesis 1.")
            result = runner.invoke(cli, ["export"])
            assert result.exit_code == 0
            assert "Exported 1 chapters across 1 books" in result.output
            chapter = Path("output/gen/chapter-01.md")
            assert chapter.exists()
            assert "A summary of Genesis 1." in chapter.read_text()

    def test_export_honours_custom_output_dir(self, runner):
        from bible_study.cli import cli
        from bible_study.db import init_db, save_summary
        with runner.isolated_filesystem():
            db = Path("data/bible.db")
            init_db(db)
            save_summary(db, "Genesis", 1, "Summary text.")
            result = runner.invoke(cli, ["export", "-o", "docs"])
            assert result.exit_code == 0
            assert (Path("docs") / "index.md").exists()

    def test_export_is_registered(self):
        from bible_study.cli import cli
        assert "export" in cli.commands
