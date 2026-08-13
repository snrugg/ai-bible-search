"""Tests for bible_study/api -- bible-api.com client."""

from unittest.mock import MagicMock, patch

import pytest


class TestParseVerses:
    """Test the internal _parse_verses helper."""

    def test_parses_simple_response(self):
        from bible_study.api import _parse_verses
        data = {"verses": [
               {"verse": 1, "text": "hello"},
               {"verse": 2, "text": " world\n"},
           ]}
        result = _parse_verses(data)
        assert len(result) == 2
        assert result[0]["verse"] == 1
        assert result[0]["text"].strip() == "hello"
        assert result[1]["text"].strip() == "world"

    def test_parses_empty_response(self):
        from bible_study.api import _parse_verses
        assert _parse_verses({"verses": []}) == []

    def test_ignores_missing_verses_key(self):
        from bible_study.api import _parse_verses
        result = _parse_verses({})
        assert result == []


class TestFetchChapter:
    """Test fetching a chapter with mocked HTTP."""

    @patch("bible_study.api.requests.get")
    def test_returns_parsed_verses(self, mock_get, mock_chapter_response):
        from bible_study.api import fetch_chapter
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_chapter_response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        verses = fetch_chapter("Genesis", 1)
        assert len(verses) == 3
        assert verses[0]["verse"] == 1
        assert "beginning" in verses[0]["text"].lower()


class TestCacheKey:
    """Test the _cache_key helper."""

    def test_cache_key_returns_path_with_book_and_number(self):
        from bible_study.api import _cache_key
        key = _cache_key("Genesis", 1)
        assert "genesis" in str(key).lower()
        assert "1.json" in str(key)

    def test_cache_key_handles_spaces_in_book_name(self):
        from bible_study.api import _cache_key
        key = _cache_key("1 Samuel", 1)
        assert "1-samuel" in str(key).lower()


class TestCaching:
    """Test chapter data caching to local JSON files."""

    @patch("bible_study.api.requests.get")
    def test_saves_to_cache(self, mock_get, tmp_data, mock_chapter_response):
        from bible_study.api import save_chapter
        cache_dir = tmp_data / "api-cache"
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_chapter_response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = save_chapter("Genesis", 1, cache_dir=cache_dir)
        assert isinstance(result, dict)

        # Verify cache file was created
        cache_file = cache_dir / "genesis-1.json"
        assert cache_file.exists()
