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


class TestModelConfig:
    """Test resolution of the ollama_model config key."""

    def test_get_model_reads_configured_value(self, tmp_path):
        from bible_study.prompts import get_model, load_config
        cfg = tmp_path / "config.yaml"
        cfg.write_text("ollama_model: llama3:8b\n")
        assert get_model(load_config(cfg)) == "llama3:8b"

    def test_get_model_falls_back_to_default(self):
        from bible_study.ollama import MODEL
        from bible_study.prompts import get_model
        assert get_model({}) == MODEL

    def test_get_model_ignores_blank_value(self):
        from bible_study.ollama import MODEL
        from bible_study.prompts import get_model
        assert get_model({"ollama_model": "   "}) == MODEL

    def test_get_model_loads_config_when_omitted(self, tmp_path, monkeypatch):
        from bible_study.prompts import get_model
        (tmp_path / "config.yaml").write_text("ollama_model: mistral:7b\n")
        monkeypatch.chdir(tmp_path)
        assert get_model() == "mistral:7b"

    def test_get_model_falls_back_when_no_config_file(self, tmp_path, monkeypatch):
        from bible_study import prompts
        from bible_study.ollama import MODEL
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIBLE_STUDY_CONFIG", raising=False)
        monkeypatch.setattr(
            prompts, "load_config",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
        )
        assert prompts.get_model() == MODEL


class TestNumCtxConfig:
    """Test resolution of the ollama_num_ctx config key."""

    def test_get_num_ctx_reads_configured_value(self, tmp_path):
        from bible_study.prompts import get_num_ctx, load_config
        cfg = tmp_path / "config.yaml"
        cfg.write_text("ollama_num_ctx: 131072\n")
        assert get_num_ctx(load_config(cfg)) == 131072

    def test_get_num_ctx_falls_back_to_default(self):
        from bible_study.ollama import NUM_CTX
        from bible_study.prompts import get_num_ctx
        assert get_num_ctx({}) == NUM_CTX

    def test_get_num_ctx_ignores_blank_value(self):
        from bible_study.ollama import NUM_CTX
        from bible_study.prompts import get_num_ctx
        assert get_num_ctx({"ollama_num_ctx": "   "}) == NUM_CTX

    def test_get_num_ctx_ignores_non_numeric_value(self):
        from bible_study.ollama import NUM_CTX
        from bible_study.prompts import get_num_ctx
        assert get_num_ctx({"ollama_num_ctx": "big"}) == NUM_CTX

    def test_get_num_ctx_ignores_non_positive_value(self):
        from bible_study.ollama import NUM_CTX
        from bible_study.prompts import get_num_ctx
        assert get_num_ctx({"ollama_num_ctx": "0"}) == NUM_CTX
        assert get_num_ctx({"ollama_num_ctx": "-1"}) == NUM_CTX

    def test_get_num_ctx_loads_config_when_omitted(self, tmp_path, monkeypatch):
        from bible_study.prompts import get_num_ctx
        (tmp_path / "config.yaml").write_text("ollama_num_ctx: 8192\n")
        monkeypatch.chdir(tmp_path)
        assert get_num_ctx() == 8192

    def test_get_num_ctx_falls_back_when_no_config_file(self, tmp_path, monkeypatch):
        from bible_study import prompts
        from bible_study.ollama import NUM_CTX
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIBLE_STUDY_CONFIG", raising=False)
        monkeypatch.setattr(
            prompts, "load_config",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
        )
        assert prompts.get_num_ctx() == NUM_CTX
