# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An indexed, summarized version of the entire KJV Bible (66 books, ~1,189 chapters). Fetches text from bible-api.com, generates chapter summaries via local Ollama (model set by `ollama_model` in `config.yaml`, defaulting to `qwen3.6:35b-a3b-nvfp4`), stores everything in SQLite, and provides an HTTP browser viewer. Built with Python 3.14, uv package management, and a hatchling build backend.

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
- `summarize-book`: aggregates book-level summaries from chapter summaries. The stored chapter summaries are concatenated as `Chapter N: <summary>` and passed into the `{chapter_summaries}` placeholder of the `book_summary` template, so the model summarises your generated text rather than writing from memory
- `view`: starts HTTP server backed by SQLite. Routes: `/` (book index with per-book progress), `/book/<slug>` (chapter grid + book summary), `/book/<slug>/<n>` (chapter summary + KJV verses). Slugs are lowercase with hyphens (`1-samuel`); `--data-dir` selects the database
- `status`: shows progress (total vs summarized chapters)
- `export`: writes per-chapter markdown with cross-links plus book and master indexes
- `config-edit`: hands `config.yaml` to `webbrowser.open()` (the OS default handler, not `$EDITOR`)

Every command except `config-edit` takes `--data-dir` / `-d` via the shared `_data_dir_option` decorator at the top of `cli.py`; reuse that decorator rather than redeclaring the option.
- `clear-summaries`: deletes stored summaries (never verse text) so they can be regenerated; `--book` scopes to one book, `--scope chapters|books|all` picks the level, `-y` skips the confirm prompt
- `clear-book-summaries`: deletes only book-level summaries, leaving chapter summaries and verses intact so `summarize-book` can re-aggregate; same `--book` / `-y` options. Equivalent to `clear-summaries --scope books`

## Running Tests

```bash
uv sync --extra dev                            # Install dev deps (pytest, mutmut)
uv run pytest                                  # Full suite; coverage is on by default
uv run pytest --no-cov                         # Fast, no coverage overhead
uv run pytest tests/test_summary.py --no-cov   # One file
uv run pytest tests/test_summary.py::TestSummarizeBook::test_summarize_book_returns_text --no-cov
uv run pytest -k "book_summary" --no-cov       # By name pattern
uv run mutmut run                              # Mutation testing (src/bible_study)
```

`addopts` in `pyproject.toml` already passes `--cov=bible_study`, so a bare `uv run pytest` produces a coverage report and fails under 90%. 351 tests pass; 5 are skipped by design (3 in `tests/integration/`, 2 over-the-wire browser tests). Target: 100% test coverage with 90% mutation passing rate. No linter or formatter is configured — the `# noqa` comments in the source are vestigial.

## Ollama Context Window

**Ollama never errors on an over-long prompt.** llama.cpp runs with `--context-shift`, which evicts the *oldest* tokens and answers anyway. Since every template puts its instructions first and `{chapter_text}` / `{chapter_summaries}` last, the instruction header is the first thing discarded — the model returns a confident, well-formed answer to a question it only partially saw. This is invisible unless you check the output against the template it was supposed to follow.

It bit `summarize_book()`, which concatenates every chapter summary with no size cap. Psalms is 150 chapters ≈ 276 KB ≈ 70k tokens; the server had autodetected a 32k window (`OLLAMA_CONTEXT_LENGTH` defaults to 4k/32k/256k by VRAM), so stored Psalms summaries cover only ~chapters 107–150 and ignore the output template. Isaiah (~36k tokens) was affected too; Genesis (~28k) sat just under the line.

Two mechanisms now guard this:

- `ollama.generate()` always sends `options.num_ctx`, resolved from `ollama_num_ctx` in `config.yaml` via `prompts.get_num_ctx()` (same fallback chain as `get_model()` → `ollama.NUM_CTX`). The window no longer depends on how the server happened to start.
- `ollama.check_prompt_fits()` runs *before* the request and raises `PromptTooLongError` when the estimate exceeds `num_ctx - RESPONSE_RESERVE_TOKENS`. Sizing uses a `CHARS_PER_TOKEN = 4` heuristic — no tokenizer is shipped, so it is approximate and deliberately conservative.

So over-long prompts now fail loudly instead of being silently truncated. If `summarize-book` raises `PromptTooLongError`, that is the guard working: raise `ollama_num_ctx`, or shorten the input. Note that a bigger window only prevents *truncation* — feeding ~70k tokens to a small model still yields a shallow summary, so chunked map-reduce in `summarize_book()` remains the real fix for long books.

`claude.sh` is a personal launcher, not part of the package. It exports `OLLAMA_CONTEXT_LENGTH` in front of `ollama launch` — a client command — so it has never affected the server this tool talks to.

## Common Tasks

### Adding a new CLI command
Edit `src/bible_study/cli.py` — add a `@cli.command()` decorated function with any needed options. The Click group auto-discovers all decorated methods. Click derives the command name from the function name: underscores become dashes and a trailing `_cmd` is stripped, so `clear_book_summaries_cmd` is invoked as `clear-book-summaries`. Use that suffix when the function name would otherwise shadow an imported helper. Add the new name to the command-set assertion in `tests/test_cli.py`.

### Adding a new DB field
Update `INIT_SQL` in `db.py`, add the column to every `CREATE TABLE` statement, and update existing CRUD functions. Then add corresponding tests.

### Customising prompts
Edit `config.yaml`. No code change needed — `prompts.py` reads this at runtime and substitutes variables via simple `.replace()`.

### Changing the Ollama model
Set `ollama_model` in `config.yaml` (tag included, e.g. `llama3:8b`). `prompts.get_model()` resolves it, `summary.py` passes it into every `ollama.generate()` call, and `ollama.MODEL` is only the fallback when the key is absent or the config file is missing. An explicit `ollama_kwargs={"model": ...}` still overrides config.

### Changing the context window
Set `ollama_num_ctx` in `config.yaml`. `prompts.get_num_ctx()` resolves it (ignoring blank, non-numeric, and non-positive values), `summary.py` passes it into every `ollama.generate()` call, and `ollama.NUM_CTX` is the fallback. An explicit `ollama_kwargs={"num_ctx": ...}` still overrides config. Verify a change landed with `ollama ps` — the `CONTEXT` column reflects what the last request actually requested.

### Adding a new module to summary pipeline
1. Implement the function in a new module under `src/bible_study/`
2. Import it in `summary.py` using the `_alias` pattern (`import X as _x`)
3. Add tests in `tests/test_<module>.py` with proper mocks and `init_db()` calls

## File Map

| File | Role |
|------|------|
| `__init__.py` | Entry point; delegates to Click CLI |
| `cli.py` | Click commands: init, summarize, summarize-book, view, status, export, clear-summaries, clear-book-summaries, config-edit |
| `api.py` | bible-api.com client: JSON cache under `data/api-cache/`, chapter enumeration, and retry/backoff for `RETRYABLE_STATUS` (429/5xx) honouring `Retry-After`. Cached files may hold either a bare verse list (older) or a mapping (newer) — `fetch_chapter` normalises both, so keep that branch when touching the cache format |
| `db.py` | SQLite schema & all CRUD helpers (verses, chapter_summaries, book_summaries) |
| `indexer.py` | Hardcoded 66-book structure, book names, chapter counts |
| `ollama.py` | Local Ollama API client: health_check, check_model_available, generate, plus the context-window guard (check_prompt_fits, PromptTooLongError) |
| `prompts.py` | YAML config loader + prompt builders (chapter_summary, book_summary) |
| `summary.py` | Pipeline orchestration: summarize_chapter, summarize_book, export_markdowns |
| `browser.py` | SQLite-backed HTTP server with _SQLiteHandler class and serve() entry point |

## Browser Handler Rules

Every response must go through `_send_html()` or `_send_error()` — both send a status line, `Content-Type`, and `Content-Length`. Writing straight to `self.wfile` produces a headerless reply that clients reject as HTTP/0.9. Handlers hold a `db_path` (not a shared connection) and read through `db.py` helpers, which open and close per call. All stored text (verses, summaries) is passed through `html.escape()` before rendering.

Handler tests built with `_SQLiteHandler.__new__` must mock `send_response`, `send_header`, and `end_headers` alongside `wfile` — see `_handler_for()` in `tests/test_browser.py`. Mocked-handler tests cannot catch protocol errors, so `TestOverTheWire` runs a real `HTTPServer` on an ephemeral port and asserts parsed status codes.
