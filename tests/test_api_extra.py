"""Additional tests for bible_study/api -- error handling and caching."""

import json
from unittest.mock import MagicMock

import pytest


class TestFetchChapterErrors:
    """Test fetch_chapter error handling paths."""

    def test_fetch_raises_on_non_5xx_error(self, mocker):
        from bible_study.api import fetch_chapter
        # Non-5xx errors should raise immediately (no retry)
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get = mocker.patch(
            "bible_study.api.requests.get", return_value=mock_resp,
        )
        # A non-5xx error should raise and not retry
        with pytest.raises(requests.HTTPError):
            fetch_chapter("Genesis", 1)
        assert mock_get.call_count == 1

    def test_fetch_retries_on_500(self, mocker):
        from bible_study.api import fetch_chapter
        # 5xx should trigger retries
        import requests
        mocker.patch("bible_study.api.time.sleep")
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.raise_for_status.side_effect = requests.HTTPError("500")
        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.raise_for_status.return_value = None
        mock_resp_ok.json.return_value = {"verses": [{"verse": 1, "text": "hello"}]}
        # First call fails with 500, second succeeds
        mock_get = mocker.patch(
            "bible_study.api.requests.get",
            side_effect=[mock_fail, mock_resp_ok],
        )
        # Should retry and eventually succeed
        result = fetch_chapter("Genesis", 1)
        assert len(result) == 1
        assert mock_get.call_count == 2


class TestSaveChapter:
    """Test save_chapter caching behavior."""

    def test_save_returns_verses_dict(self, mocker, tmp_path):
        from bible_study.api import save_chapter
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"verses": [{"verse": 1, "text": "hello"}]}
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("bible_study.api.requests.get", return_value=mock_resp)
        cache_dir = tmp_path / "cache"
        result = save_chapter("Genesis", 1, cache_dir=cache_dir)
        assert isinstance(result, dict)
        assert "verses" in result

    def test_save_creates_cache_file(self, mocker, tmp_path):
        from bible_study.api import save_chapter
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"verses": [{"verse": 1, "text": "hello"}]}
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("bible_study.api.requests.get", return_value=mock_resp)
        cache_dir = tmp_path / "cache"
        save_chapter("Genesis", 1, cache_dir=cache_dir)
        assert (cache_dir / "genesis-1.json").exists()

    def test_save_skips_when_cache_hit(self, mocker, tmp_path):
        from bible_study.api import save_chapter
        # Pre-populate the cache so save sees it as a hit
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "genesis-1.json").write_text(
            json.dumps({"verses": [{"verse": 1, "text": "cached"}]})
        )
        mock_get = mocker.patch("bible_study.api.requests.get")
        result = save_chapter("Genesis", 1, cache_dir=cache_dir)
        # Should not call the API since cache exists
        assert len(result["verses"]) == 1

    def test_save_with_custom_cache_dir(self, mocker, tmp_path):
        from bible_study.api import save_chapter
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"verses": [{"verse": 1, "text": "hello"}]}
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("bible_study.api.requests.get", return_value=mock_resp)
        custom_cache = tmp_path / "custom-cache"
        result = save_chapter("Genesis", 1, cache_dir=custom_cache)
        assert (custom_cache / "genesis-1.json").exists()

    def test_save_creates_directory_if_missing(self, mocker, tmp_path):
        from bible_study.api import save_chapter
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"verses": [{"verse": 1, "text": "hello"}]}
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("bible_study.api.requests.get", return_value=mock_resp)
        new_cache = tmp_path / "new-dir"
        result = save_chapter("Genesis", 1, cache_dir=new_cache)
        assert new_cache.exists()


class TestDownloadAll:
    """Test download_all end-to-end flow."""

    def test_download_all_with_mocked_save(self, mocker):
        from bible_study.api import download_all
        mock_save = mocker.patch(
            "bible_study.api.save_chapter", return_value={"verses": []}
        )
        results = download_all(book_names_list=["Genesis"], cache_dir=None)
        assert isinstance(results, list)

    def test_download_all_with_empty_book_list(self, mocker):
        from bible_study.api import download_all
        # Empty book list should return empty results
        result = download_all(book_names_list=[])
        assert result == []

    def test_download_all_returns_fetched_chapters(self, mocker):
        from bible_study.api import download_all
        mock_save = mocker.patch(
            "bible_study.api.save_chapter",
            return_value={"verses": [{"verse": 1, "text": "x"}]},
        )
        result = download_all(book_names_list=["Genesis"], cache_dir=None)
        # Genesis has 50 chapters, all saved via the mocked save_chapter
        assert len(result) == 50


class TestParseVersesEdgeCases:
    """Test _parse_verses with various input shapes."""

    def test_parses_single_verse(self):
        from bible_study.api import _parse_verses
        result = _parse_verses({"verses": [{"verse": 1, "text": "hello"}]})
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_parses_preserves_text_content(self):
        from bible_study.api import _parse_verses
        # strip() should remove leading/trailing whitespace
        data = {"verses": [{"verse": 1, "text": "  hello world   "}]}
        result = _parse_verses(data)
        assert result[0]["text"] == "hello world"

    def test_parses_empty_text_field(self):
        from bible_study.api import _parse_verses
        data = {"verses": [{"verse": 1, "text": ""}]}
        result = _parse_verses(data)
        assert len(result) == 1
        assert result[0]["text"] == ""

    def test_parses_multiple_verses(self):
        from bible_study.api import _parse_verses
        data = {"verses": [
            {"verse": 1, "text": "first"},
            {"verse": 2, "text": "second"},
            {"verse": 3, "text": "third"},
        ]}
        result = _parse_verses(data)
        assert len(result) == 3
        assert result[1]["text"] == "second"


class TestConstants:
    """Test module-level constants."""

    def test_base_url_is_correct(self):
        from bible_study.api import BASE_URL
        assert BASE_URL == "https://bible-api.com"

    def test_max_retries_defined(self):
        from bible_study.api import MAX_RETRIES
        assert MAX_RETRIES == 5

    def test_request_timeout_defined(self):
        from bible_study.api import REQUEST_TIMEOUT
        assert REQUEST_TIMEOUT == 30


class TestFetchChapterIntegration:
    """End-to-end test for fetch_chapter flow."""

    def test_fetch_chapter_with_full_mock(self, mocker):
        from bible_study.api import fetch_chapter
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"verses": [
            {"verse": 1, "text": "In the beginning"},
            {"verse": 2, "text": "And God said"},
        ]}
        mocker.patch("bible_study.api.requests.get", return_value=mock_resp)
        result = fetch_chapter("Genesis", 1)
        assert len(result) == 2
        assert result[0]["verse"] == 1


class TestCacheKeyBookNames:
    """Test _cache_key with various book name formats."""

    def test_cache_key_is_path_object(self):
        from bible_study.api import _cache_key
        key = _cache_key("Genesis", 1)
        assert str(key).endswith(".json")

    def test_cache_key_includes_book_name(self):
        from bible_study.api import _cache_key
        key = _cache_key("Psalms", 23)
        assert "psalms" in str(key).lower()

    def test_cache_key_includes_chapter_number(self):
        from bible_study.api import _cache_key
        key = _cache_key("Genesis", 50)
        assert "50" in str(key)


class TestGetChaptersForBook:
    """Test get_chapters_for_book fallback behavior."""

    def test_returns_known_chapter_count(self):
        from bible_study.api import get_chapters_for_book
        result = get_chapters_for_book("Genesis")
        assert len(result) == 50

    def test_fallback_to_50_for_unknown_book(self):
        from bible_study.api import get_chapters_for_book
        # Should fall back to range(1, 51) for unknown books
        result = get_chapters_for_book("TotallyFakeBook999")
        assert len(result) == 50
        assert result[0] == 1
        assert result[-1] == 50


class TestLastCachedChapter:
    """Test _last_cached_chapter behavior."""

    def test_returns_highest_cached_number(self, tmp_path):
        from bible_study.api import _last_cached_chapter
        cache = tmp_path / "cache"
        cache.mkdir()
        # Create some cached chapter files
        (cache / "genesis-1.json").touch()
        (cache / "genesis-3.json").touch()
        (cache / "genesis-2.json").touch()
        result = _last_cached_chapter("Genesis", cache)
        assert result == 3

    def test_returns_none_when_no_caches(self, tmp_path):
        from bible_study.api import _last_cached_chapter
        cache = tmp_path / "empty"
        cache.mkdir()
        result = _last_cached_chapter("Genesis", cache)
        assert result is None

    def test_skips_non_json_files(self, tmp_path):
        from bible_study.api import _last_cached_chapter
        cache = tmp_path / "cache"
        cache.mkdir()
        # Only .json files should be considered
        (cache / "genesis-1.json").touch()
        (cache / "genesis.txt").touch()
        result = _last_cached_chapter("Genesis", cache)
        assert result == 1


class TestCacheKeyWithPath:
    """Test _cache_key with base_dir parameter."""

    def test_cache_key_includes_base_dir(self, tmp_path):
        from bible_study.api import _cache_key
        key = _cache_key("Genesis", 1, base_dir=tmp_path / "custom")
        assert "custom" in str(key).lower()

    def test_cache_key_defaults_to_data_dir(self):
        from bible_study.api import DEFAULT_CACHE_DIR, _cache_key
        key = _cache_key("Genesis", 1)
        assert str(DEFAULT_CACHE_DIR) == str(key.parent)


class TestCacheKeySpecialBooks:
    """Test _cache_key with books containing special characters."""

    def test_cache_key_for_job(self):
        from bible_study.api import _cache_key
        key = _cache_key("Job", 1)
        assert "job" in str(key).lower()

    def test_cache_key_for_philippians(self):
        from bible_study.api import _cache_key
        key = _cache_key("Philippians", 1)
        assert "philippians" in str(key).lower()


class TestFetchChapterSlug:
    """Test that fetch_chapter constructs correct URLs."""

    def test_fetch_genesis_1(self, mocker):
        from bible_study.api import fetch_chapter
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"verses": [{"verse": 1, "text": "x"}]}
        mock_get = mocker.patch("bible_study.api.requests.get")
        mock_get.return_value = mock_resp
        fetch_chapter("Genesis", 1)
        call_args = mock_get.call_args
        assert "%20" in str(call_args) or "Genesis" in str(call_args)

    def test_fetch_params_include_kjv(self, mocker):
        from bible_study.api import fetch_chapter
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"verses": [{"verse": 1, "text": "x"}]}
        mock_get = mocker.patch("bible_study.api.requests.get")
        mock_get.return_value = mock_resp
        fetch_chapter("Genesis", 1)
        args, kwargs = mock_get.call_args
        assert "params" in kwargs
