"""Shared fixtures for Bible Study tests."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_data(tmp_path: Path) -> Path:
     return tmp_path


@pytest.fixture
def yaml_config(tmp_data: Path) -> Path:
     cfg = tmp_data / "config.yaml"
     cfg.write_text("""chapter_summary: |
   Summarize the following chapter of the KJV Bible.

   Book: {book_name}
   Chapter: {chapter_number}

   Text:
       {chapter_text}

   Provide your summary below.

book_summary: |
   Aggregate summary for book {book_name} with {chapter_count} chapters.
""")
     return cfg


@pytest.fixture
def mock_chapter_response() -> dict:
     return {"reference": "Genesis 1", "translation_name": "King James Version",
               "verses": [{"verse": 1, "text": "In the beginning..."},
                           {"verse": 2, "text": "And the earth was formless..."},
                           {"verse": 3, "text": "And God said..."}]}


@pytest.fixture
def mock_ollama_response() -> str:
     return "Genesis 1 presents a structured account of creation."

