"""Integration tests that exercise real I/O.

Run with:  pytest --run-integration
"""

import pytest


@pytest.mark.skip(reason="Requires live Ollama; run manually")
class TestRealOllamaHealth:
    """Test against a real local Ollama instance."""

    def test_health_check_passes(self):
        from bible_study.ollama import health_check
        assert health_check() is True


@pytest.mark.skip(reason="Requires live API; run manually")
class TestRealAPI:
    """Test against the real bible-api.com."""

    def test_fetch_genesis_1(self):
        from bible_study.api import fetch_chapter
        verses = fetch_chapter("Genesis", 1)
        assert len(verses) > 0
        assert "beginning" in verses[0]["text"].lower()


@pytest.mark.skip(reason="Requires full data setup; run manually")
class TestEndToEnd:
    """Full end-to-end test."""

    def test_summary_pipeline(self):
        from bible_study.db import init_db, upsert_verses
        from bible_study.api import fetch_chapter
        from bible_study.ollama import health_check
        from bible_study.summary import summarize_chapter
        from bible_study.prompts import load_config
        import tempfile
        from pathlib import Path

        # Verify all components are available
        assert health_check() is True

        # Fetch a chapter from the real API
        verses = fetch_chapter("Genesis", 1)
        assert len(verses) > 0

        # Set up a temp DB with the fetched data
        tmp_db = Path(tempfile.mkdtemp()) / "bible.db"
        init_db(tmp_db)
        upsert_verses(tmp_db, "Genesis", 1, [(v["verse"], v["text"]) for v in verses])

        # Summarize the chapter (calls Ollama)
        prompt = load_config()
        summary = summarize_chapter("Genesis", 1, db_path=tmp_db)
        assert len(summary) > 0

