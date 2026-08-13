"""Tests for bible_study/prompts -- YAML config loading + template rendering."""

from pathlib import Path

import pytest


class TestInlineTemplates:
    """Test prompt building without needing a config.yaml file."""

    def test_build_inline_chapter_prompt_contains_book_name(self):
        from bible_study.prompts import build_inline_chapter_prompt
        prompt = build_inline_chapter_prompt("Genesis", "verse text", 1)
        assert "Genesis" in prompt
        assert "1" in prompt

    def test_build_inline_book_prompt_contains_book_info(self):
        from bible_study.prompts import build_inline_book_prompt
        prompt = build_inline_book_prompt("Genesis", 50)
        assert "Genesis" in prompt
        assert "50" in prompt


class TestYamlConfig:
    """Test loading and rendering from a YAML config file."""

    def test_load_config_returns_dict(self, tmp_path):
        from bible_study.prompts import load_config
        cfg = tmp_path / "config.yaml"
        cfg.write_text("chapter_summary: |\n  Summarize {book_name}\n")
        result = load_config(cfg)
        assert isinstance(result, dict)

    def test_load_has_expected_keys(self, tmp_path):
        from bible_study.prompts import load_config
        cfg = tmp_path / "config.yaml"
        cfg.write_text("chapter_summary: |\n  Test\nbook_summary: |\n  Test\n")
        result = load_config(cfg)
        assert "chapter_summary" in result
        assert "book_summary" in result

    def test_render_replaces_placeholders(self, tmp_path):
        from bible_study.prompts import load_config, render
        cfg = tmp_path / "config.yaml"
        cfg.write_text("chapter_summary: |\n  Book: {book_name}\n")
        config = load_config(cfg)
        result = render("chapter_summary", config, book_name="Genesis")
        assert "{book_name}" not in result
        assert "Genesis" in result

    def test_render_ignores_unknown_keys(self, tmp_path):
        from bible_study.prompts import load_config, render
        cfg = tmp_path / "config.yaml"
        cfg.write_text("chapter_summary: |\n  Test\n")
        config = load_config(cfg)
        result = render("chapter_summary", config, unknown_key="ignored")
        assert isinstance(result, str)

    def test_render_raises_on_missing_template_name(self, tmp_path):
        from bible_study.prompts import load_config, render
        cfg = tmp_path / "config.yaml"
        cfg.write_text("chapter_summary: |\n  Test\n")
        config = load_config(cfg)
        with pytest.raises(KeyError):
            render("nonexistent_key", config)
