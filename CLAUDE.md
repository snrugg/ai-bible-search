# Bible Study — CLAUDE.md

## Project Overview

An indexed, summarized version of the entire KJV Bible (66 books, ~1,189 chapters). Fetches text from bible-api.com, generates chapter summaries via local Ollama (`qwen3.6:35b-a3b-nvfp4`), stores everything in SQLite, and provides an HTTP browser viewer. Built with Python 3.14, uv package management, and a hatchling build backend.

## Key Directories

- `src/bible_study/` — All production code
- `tests/` — Unit tests (all mocked); `tests/integration/` — live I/O (skipped)
- `config.yaml` — User-editable prompt templates
- `data/` — SQLite database created at runtime

## Coding Standards

### Indentation Convention

Always use **4 spaces** for module-level code, **8 spaces** inside functions/methods (including docstrings inside functions). Never mix 6-space indentation — this has been a source of persistent corruption in every file-writing method. Verify `python -c "compile(open('file.py').read(), 'file.py', 'exec')"` after writing any file.

### Import Style

Summary module uses `import bible_study.ollama as _ol` (not `from X import Y`) so mocks work at the test level via `mocker.patch("bible_study.ollama.generate", ...)`. When adding new external imports to summary.py, always use the `import X as _x` alias style.

### SQLite Patterns

All DB functions open/close connections within the function (context managers). Use `sqlite3.connect(str(path))` — never share connections across threads. Schema is in the `INIT_SQL` string at the top of `db.py`. Add UNIQUE constraints where logically appropriate.

### Test Writing Rules

Every test must call `init_db(db_path)` before touching SQLite tables, and `upsert_verses(...)` to populate data before calling summary functions. Use `mocker.patch("bible_study.ollama.generate", return_value="...")` for Ollama mocks. Never use `from X import Y` inside tests when you need to patch — the patched module attribute must be the target.

### Naming Conventions

- Functions: `snake_case`
- Classes: `PascalCase`
- Internal helpers: `_private_name`
- DB abbreviations: first 3 uppercase chars (`GEN`, `EXO`)
- Test classes: `Test<ModuleOrFeature>`
- Test methods: `test_<action>_<condition>`

## Architecture Summary

```
cli.py ──▶ summary.py ──▶ ollama.py (local LLM)
                    │       db.py (SQLite storage)
                    │       api.py (bible-api.com fetcher)
                    │       prompts.py (YAML template loader)
```

- `init`: downloads KJV via `api.download_all()` → SQLite
- `summarize`: iterates unsummarized chapters → Ollama → stores in DB
- `summarize-book`: aggregates book-level summaries from chapter summaries
- `view`: starts HTTP server backed by SQLite
- `status`: shows progress (total vs summarized chapters)

## Running Tests

```bash
uv run pytest                          # Full suite (all mocked)
uv run pytest --cov=bible_study       # Coverage report
uv run pytest --no-cov                # Fast, no coverage overhead
```

All 64+ tests currently pass. 5 are skipped by design (3 integration + 2 browser integration). Target: 100% test coverage with 90% mutation passing rate.

## Common Tasks

### Adding a new CLI command
Edit `src/bible_study/cli.py` — add a `@cli.command()` decorated function with any needed options. The Click group auto-discovers all decorated methods.

### Adding a new DB field
Update `INIT_SQL` in `db.py`, add the column to every `CREATE TABLE` statement, and update existing CRUD functions. Then add corresponding tests.

### Customising prompts
Edit `config.yaml`. No code change needed — `prompts.py` reads this at runtime and substitutes variables via simple `.replace()`.

### Adding a new module to summary pipeline
1. Implement the function in a new module under `src/bible_study/`
2. Import it in `summary.py` using the `_alias` pattern (`import X as _x`)
3. Add tests in `tests/test_<module>.py` with proper mocks and `init_db()` calls

## File Map

| File | Role |
|------|------|
| `__init__.py` | Entry point; delegates to Click CLI |
| `cli.py` | Click commands: init, summarize, summarize-book, view, status, config-edit |
| `api.py` | bible-api.com client with caching + chapter enumeration |
| `db.py` | SQLite schema & all CRUD helpers (verses, chapter_summaries, book_summaries) |
| `indexer.py` | Hardcoded 66-book structure, book names, chapter counts |
| `ollama.py` | Local Ollama API client: health_check, check_model_available, generate |
| `prompts.py` | YAML config loader + prompt builders (chapter_summary, book_summary) |
| `summary.py` | Pipeline orchestration: summarize_chapter, summarize_book, export_markdowns |
| `browser.py` | SQLite-backed HTTP server with _SQLiteHandler class and serve() entry point |
