"""Tests covering remaining branches in api, ollama, prompts, and db."""

import json

import pytest

from bible_study.db import init_db, save_summary, upsert_verses


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "bible.db"
    init_db(path)
    return path


class TestDbRemainingBranches:
    """Cover the untaken branches in db.py."""

    def test_verse_count_without_book_filter(self, db_path):
        from bible_study.db import verse_count
        upsert_verses(db_path, "Genesis", 1, [(1, "a"), (2, "b")])
        upsert_verses(db_path, "Exodus", 1, [(1, "c")])
        assert verse_count(db_path) == 3

    def test_verse_count_with_book_filter(self, db_path):
        from bible_study.db import verse_count
        upsert_verses(db_path, "Genesis", 1, [(1, "a"), (2, "b")])
        upsert_verses(db_path, "Exodus", 1, [(1, "c")])
        assert verse_count(db_path, "Genesis") == 2

    def test_progress_ignores_books_outside_filter(self, db_path):
        from bible_study.db import get_chapter_progress
        upsert_verses(db_path, "Genesis", 1, [(1, "a")])
        upsert_verses(db_path, "Exodus", 1, [(1, "b")])
        total, summed = get_chapter_progress(db_path, ["Genesis"])
        assert total == 1
        assert summed == 0

    def test_unsummarized_ignores_books_outside_filter(self, db_path):
        from bible_study.db import get_unsummarized_chapters
        upsert_verses(db_path, "Genesis", 1, [(1, "a")])
        upsert_verses(db_path, "Exodus", 1, [(1, "b")])
        assert get_unsummarized_chapters(db_path, ["Genesis"]) == [("Genesis", 1)]

    def test_get_chapter_text_joins_verses(self, db_path):
        from bible_study.db import get_chapter_text
        upsert_verses(db_path, "Genesis", 1, [(1, "hello"), (2, "world")])
        assert get_chapter_text(db_path, "Genesis", 1) == "hello world"

    def test_saved_books_empty_before_any_save(self, db_path):
        from bible_study.db import get_saved_books
        assert get_saved_books(db_path) == []


class TestOllamaRemainingBranches:
    """Cover check_model_available failure and generate retry exhaustion."""

    def test_check_model_returns_false_on_request_error(self, mocker):
        from bible_study.ollama import check_model_available
        mocker.patch("requests.get", side_effect=Exception("connection refused"))
        assert check_model_available() is False

    def test_check_model_returns_false_on_bad_json(self, mocker):
        from bible_study.ollama import check_model_available
        resp = mocker.MagicMock()
        resp.json.side_effect = ValueError("not json")
        mocker.patch("requests.get", return_value=resp)
        assert check_model_available() is False

    def test_generate_raises_after_exhausting_retries(self, mocker):
        from bible_study.ollama import generate
        mocker.patch("bible_study.ollama.time.sleep")
        mocker.patch("requests.post", side_effect=Exception("timeout"))
        with pytest.raises(RuntimeError, match="Ollama generation failed"):
            generate("prompt")

    def test_generate_error_message_names_retry_count(self, mocker):
        from bible_study.ollama import generate
        mocker.patch("bible_study.ollama.time.sleep")
        mocker.patch("requests.post", side_effect=Exception("boom"))
        with pytest.raises(RuntimeError, match="after 2 retries"):
            generate("prompt", max_retries=2)

    def test_generate_returns_empty_when_response_key_absent(self, mocker):
        from bible_study.ollama import generate
        resp = mocker.MagicMock(json=lambda: {})
        mocker.patch("requests.post", return_value=resp)
        assert generate("prompt") == ""

    def test_health_check_false_on_non_200(self, mocker):
        from bible_study.ollama import health_check
        mocker.patch("requests.get", return_value=mocker.MagicMock(status_code=500))
        assert health_check() is False


class TestPromptsRemainingBranches:
    """Cover load_config discovery, render fallbacks, and inline templates."""

    def test_load_config_with_explicit_path(self, tmp_path):
        from bible_study.prompts import load_config
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("chapter_summary: |\n  Book {book_name}\n")
        assert "chapter_summary" in load_config(cfg)

    def test_load_config_discovers_cwd_file(self, tmp_path, monkeypatch):
        from bible_study.prompts import load_config
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text("chapter_summary: |\n  hi\n")
        assert load_config()["chapter_summary"].strip() == "hi"

    def test_load_config_returns_empty_for_non_mapping(self, tmp_path):
        from bible_study.prompts import load_config
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- one\n- two\n")
        assert load_config(cfg) == {}

    def test_render_leaves_unmatched_placeholders(self):
        from bible_study.prompts import render
        config = {"tpl": "Book: {book_name}, Chapter: {chapter_number}"}
        result = render("tpl", config, book_name="Genesis")
        assert "Genesis" in result
        assert "{chapter_number}" in result

    def test_render_substitutes_none_as_empty_string(self):
        from bible_study.prompts import render
        config = {"tpl": "Value: [{thing}]"}
        assert render("tpl", config, thing=None) == "Value: []"

    def test_render_raises_keyerror_for_missing_template(self):
        from bible_study.prompts import render
        with pytest.raises(KeyError):
            render("nope", {"tpl": "x"})

    def test_chapter_prompt_falls_back_to_inline(self):
        from bible_study.prompts import build_chapter_prompt
        prompt = build_chapter_prompt({}, "Genesis", "verse text", 1)
        assert "Genesis" in prompt
        assert "verse text" in prompt

    def test_book_prompt_falls_back_to_inline(self):
        from bible_study.prompts import build_book_summary_prompt
        prompt = build_book_summary_prompt({}, "Genesis", 50)
        assert "Genesis" in prompt
        assert "50" in prompt

    def test_chapter_prompt_uses_config_template(self):
        from bible_study.prompts import build_chapter_prompt
        config = {"chapter_summary": "CFG {book_name} {chapter_number}"}
        assert build_chapter_prompt(config, "Genesis", "text", 3) == "CFG Genesis 3"

    def test_book_prompt_uses_config_template(self):
        from bible_study.prompts import build_book_summary_prompt
        config = {"book_summary": "CFG {book_name} has {chapter_count}"}
        result = build_book_summary_prompt(config, "Genesis", 50)
        assert result == "CFG Genesis has 50"

    def test_book_prompt_fills_chapter_summaries_placeholder(self):
        from bible_study.prompts import build_book_summary_prompt
        config = {"book_summary": "CFG {book_name}: {chapter_summaries}"}
        result = build_book_summary_prompt(config, "Genesis", 50, "Chapter 1: light")
        assert result == "CFG Genesis: Chapter 1: light"

    def test_book_prompt_drops_summaries_when_template_omits_placeholder(self):
        from bible_study.prompts import build_book_summary_prompt
        config = {"book_summary": "CFG {book_name} has {chapter_count}"}
        result = build_book_summary_prompt(config, "Genesis", 50, "Chapter 1: light")
        assert result == "CFG Genesis has 50"

    def test_inline_book_prompt_appends_source_block(self):
        from bible_study.prompts import build_inline_book_prompt
        bare = build_inline_book_prompt("Genesis", 50)
        with_sources = build_inline_book_prompt("Genesis", 50, "Chapter 1: light")
        assert "Chapter 1: light" in with_sources
        assert with_sources.startswith(bare)

    def test_inline_book_prompt_omits_block_for_blank_summaries(self):
        from bible_study.prompts import build_inline_book_prompt
        assert build_inline_book_prompt("Genesis", 50, "   ") == (
            build_inline_book_prompt("Genesis", 50)
        )


class TestApiRemainingBranches:
    """Cover download_all defaults, failures, and cache-scan edge cases."""

    def test_download_all_uses_full_canon_by_default(self, tmp_path, mocker):
        from bible_study.api import download_all
        mocker.patch("bible_study.api.time.sleep")
        mocker.patch("bible_study.api.save_chapter", return_value={"verses": []})
        mocker.patch(
            "bible_study.indexer.book_names", return_value=["Obadiah"],
        )
        results = download_all(cache_dir=tmp_path / "cache")
        assert results == [("Obadiah", 1)]

    def test_download_all_reports_failed_chapters(self, tmp_path, mocker, capsys):
        from bible_study.api import download_all
        mocker.patch("bible_study.api.time.sleep")
        mocker.patch(
            "bible_study.api.save_chapter", side_effect=RuntimeError("network down"),
        )
        results = download_all(
            book_names_list=["Obadiah"], cache_dir=tmp_path / "cache",
        )
        assert results == []
        assert "failed to fetch Obadiah 1" in capsys.readouterr().out

    def test_download_all_creates_cache_dir(self, tmp_path, mocker):
        from bible_study.api import download_all
        mocker.patch("bible_study.api.time.sleep")
        mocker.patch("bible_study.api.save_chapter", return_value={"verses": []})
        cache = tmp_path / "brand-new"
        download_all(book_names_list=["Obadiah"], cache_dir=cache)
        assert cache.is_dir()

    def test_last_cached_chapter_ignores_non_numeric_suffix(self, tmp_path):
        from bible_study.api import _last_cached_chapter
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "genesis-abc.json").touch()
        assert _last_cached_chapter("Genesis", cache) is None

    def test_last_cached_chapter_returns_none_for_missing_dir(self, tmp_path):
        from bible_study.api import _last_cached_chapter
        assert _last_cached_chapter("Genesis", tmp_path / "does-not-exist") is None

    def test_last_cached_chapter_ignores_other_books(self, tmp_path):
        from bible_study.api import _last_cached_chapter
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "exodus-9.json").touch()
        (cache / "genesis-2.json").touch()
        assert _last_cached_chapter("Genesis", cache) == 2

    def test_save_chapter_returns_cached_payload(self, tmp_path, mocker):
        from bible_study.api import save_chapter
        cache = tmp_path / "cache"
        cache.mkdir()
        payload = [{"verse": 1, "text": "cached text"}]
        (cache / "genesis-1.json").write_text(json.dumps(payload))
        mock_get = mocker.patch("bible_study.api.requests.get")
        result = save_chapter("Genesis", 1, cache_dir=cache)
        # A legacy bare-list cache is normalised to the mapping shape.
        assert result == {"verses": payload}
        mock_get.assert_not_called()

    def test_save_chapter_reads_mapping_shaped_cache(self, tmp_path, mocker):
        from bible_study.api import save_chapter
        cache = tmp_path / "cache"
        cache.mkdir()
        payload = {"verses": [{"verse": 1, "text": "cached text"}]}
        (cache / "genesis-1.json").write_text(json.dumps(payload))
        mock_get = mocker.patch("bible_study.api.requests.get")
        assert save_chapter("Genesis", 1, cache_dir=cache) == payload
        mock_get.assert_not_called()


class TestFinalBranches:
    """Cover the last defensive branches across modules."""

    def test_render_markdown_handles_unknown_book(self):
        from bible_study.summary import render_chapter_markdown
        md = render_chapter_markdown("Fakebook", 1, "Summary")
        assert "# Fakebook Chapter 1" in md
        assert "unk/" in md or "(first or last chapter)" in md

    def test_load_config_raises_when_no_file_anywhere(self, tmp_path, monkeypatch):
        from pathlib import Path

        import bible_study.prompts as prompts
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        with pytest.raises(FileNotFoundError, match="No config.yaml found"):
            prompts.load_config()

    def test_generate_returns_empty_with_zero_retries(self, mocker):
        from bible_study.ollama import generate
        mock_post = mocker.patch("requests.post")
        assert generate("prompt", max_retries=0) == ""
        mock_post.assert_not_called()

    def test_handler_factory_passes_db_path_through(self, mocker, tmp_path):
        import bible_study.browser as browser
        captured = {}

        class FakeHandler:
            def __init__(self, *args, db_path=None, **kwargs):
                captured["db_path"] = db_path
                captured["args"] = args

        mocker.patch.object(browser, "_SQLiteHandler", FakeHandler)
        mocker.patch.object(browser.webbrowser, "open")
        server = mocker.patch.object(browser, "HTTPServer")
        db = tmp_path / "bible.db"
        browser.serve(port=8099, db_path=db)
        factory = server.call_args[0][1]
        factory("req", ("127.0.0.1", 1234), None)
        assert captured["db_path"] == db
        assert captured["args"][0] == "req"

    def test_handler_init_stores_db_path(self, mocker, tmp_path):
        from bible_study.browser import _SQLiteHandler
        mocker.patch(
            "http.server.SimpleHTTPRequestHandler.__init__", return_value=None,
        )
        db = tmp_path / "bible.db"
        handler = _SQLiteHandler(db_path=db)
        assert handler.db_path == db

    def test_handler_init_defaults_db_path(self, mocker):
        from pathlib import Path as _P

        from bible_study.browser import _SQLiteHandler
        mocker.patch(
            "http.server.SimpleHTTPRequestHandler.__init__", return_value=None,
        )
        assert _SQLiteHandler().db_path == _P("data/bible.db")
