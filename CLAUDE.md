# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An indexed, summarized version of the entire KJV Bible (66 books, ~1,189 chapters). Fetches text from bible-api.com, generates chapter summaries via local Ollama (model set by `ollama_model` in `config.yaml`, defaulting to `qwen3.6:35b-a3b-nvfp4`), stores everything in SQLite, indexes verses and summaries into a sqlite-vec vector database for semantic search and grounded question answering, and provides an HTTP browser viewer. Built with Python 3.14, uv package management, and a hatchling build backend.

## Key Directories

- `src/bible_study/` — All production code
- `tests/` — Unit tests (all mocked); `tests/integration/` — live I/O (skipped)
- `config.yaml` — User-editable prompt templates, model names, and chunking settings
- `.githooks/` — Tracked git hooks; enable with `git config core.hooksPath .githooks`
- `.claude/skills/bible-ask/` — Claude Code skill: same retrieval as `ask`, answered by Claude instead of the local model
- `data/` — SQLite database created at runtime; git-ignored, since the vector index pushes it past 80 MB

## Coding Standards

### Indentation Convention

Always use **4 spaces** for module-level code, **8 spaces** inside functions/methods (including docstrings inside functions). Never mix 6-space indentation — this has been a source of persistent corruption in every file-writing method. Verify `python -c "compile(open('file.py').read(), 'file.py', 'exec')"` after writing any file.

### Import Style

Summary module uses `import bible_study.ollama as _ol` (not `from X import Y`) so mocks work at the test level via `mocker.patch("bible_study.ollama.generate", ...)`. When adding new external imports to summary.py, always use the `import X as _x` alias style.

### SQLite Patterns

All DB functions open *and close* their own connection within the function — never share one across threads, and never hold one on an object (`browser.py` handlers keep a `Path`, not a connection).

Always use `with closing(sqlite3.connect(str(path))) as conn:`. **`with sqlite3.connect(...)` alone does not close anything** — sqlite3's context manager is a *transaction* manager that commits or rolls back and leaves the connection open, so the bare form leaks one connection per call to the garbage collector. That produced ~13,000 `ResourceWarning`s across the test suite until it was fixed.

Because `closing()` provides no implicit commit, **every write must call `conn.commit()` explicitly** before the block ends. Readers need nothing.

Schema is in the `INIT_SQL` string at the top of `db.py`. Add UNIQUE constraints where logically appropriate — and remember NULLs are distinct in a UNIQUE index, so use `NOT NULL DEFAULT 0` sentinels for columns that do not apply to every row.

### Test Writing Rules

Every test must call `init_db(db_path)` before touching SQLite tables, and `upsert_verses(...)` to populate data before calling summary functions. Use `mocker.patch("bible_study.ollama.generate", return_value="...")` for Ollama mocks. Never use `from X import Y` inside tests when you need to patch — the patched module attribute must be the target.

**Any CLI test that invokes a command with an Ollama preflight must mock `bible_study.ollama.health_check`** — `summarize`, `embed`, `ask`, and `search` all call it. Without the mock the test passes only where Ollama happens to be running, so it goes green locally and red in CI. Two `TestSummarizeCommand` tests did exactly that and failed on the first CI run. To reproduce a no-Ollama environment locally, clone to a scratch directory and point `ollama.OLLAMA_BASE` at a dead port (`http://localhost:1`) before running the suite.

Vector tests run against the **real** sqlite-vec extension (a declared dependency) but with 4-dimensional toy vectors: `init_vec(db_path, dims=4)`, then `mocker.patch("bible_study.ollama.embed", ...)`. Faking vec0 would test our SQL strings against a mock and prove nothing about the two things most likely to break — whether the DDL parses and whether `k = ?` binds. `rag` tests instead patch `bible_study.vectors.search`, so they need no extension at all.

### Naming Conventions

- Functions: `snake_case`
- Classes: `PascalCase`
- Internal helpers: `_private_name`
- DB abbreviations: first 3 uppercase chars (`GEN`, `EXO`)
- Test classes: `Test<ModuleOrFeature>`
- Test methods: `test_<action>_<condition>`

## Architecture Summary

```
cli.py ──▶ summary.py ──▶ ollama.py (local LLM: generate + embed)
           rag.py     ──▶ db.py (SQLite storage)
           vectors.py ──▶ api.py (bible-api.com fetcher)
                          prompts.py (YAML template loader)
```

- `init`: downloads KJV via `api.download_all()` → SQLite
- `summarize`: iterates unsummarized chapters → Ollama → stores in DB
- `summarize-book`: aggregates book-level summaries from chapter summaries. The stored chapter summaries are concatenated as `Chapter N: <summary>` and passed into the `{chapter_summaries}` placeholder of the `book_summary` template, so the model summarises your generated text rather than writing from memory
- `embed`: chunks verses and summaries into `chunks`, then embeds every stale chunk into the sqlite-vec index. Incremental — re-running embeds only what changed. `--rebuild` discards existing vectors, `--limit` is a smoke test
- `ask "question"`: retrieves across all three tiers, expands, budgets, and makes one `generate()` call. `-k` sets verse hits (summary tiers scale down from it); `--no-show-sources` hides the citation list
- `search "query"`: retrieval only, no LLM call. `--json` for machine-readable output, `--expand` to apply the same expand/rank step `ask` uses — so `search --expand --json` prints exactly the context `ask` would send to the model. This is the retrieval-debugging surface, and what the `bible-ask` skill consumes
- `view`: starts HTTP server backed by SQLite. Routes: `/` (book index with per-book progress), `/book/<slug>` (chapter grid + book summary), `/book/<slug>/<n>` (chapter summary + KJV verses), `/search?q=` (ranked hits, no LLM call), `/ask?q=` (answer + linked sources). Slugs are lowercase with hyphens (`1-samuel`); `--data-dir` selects the database
- `status`: shows progress (total vs summarized chapters, plus chunk and embedded counts once `embed` has run)
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

`addopts` in `pyproject.toml` already passes `--cov=bible_study`, so a bare `uv run pytest` produces a coverage report and fails under 90%. 586 tests pass; 9 are skipped by design (7 in `tests/integration/`, 2 over-the-wire browser tests). Target: 100% test coverage with 90% mutation passing rate. No linter or formatter is configured — the `# noqa` comments in the source are vestigial.

## Git Hooks

Hooks live in the tracked `.githooks/` directory, not in `.git/hooks/`, so they are versioned with the repo. They are active only when `core.hooksPath` points at them:

```bash
git config core.hooksPath .githooks     # once per clone
```

`pre-commit` does four things, cheapest first:

1. **Refuses generated data** — `data/bible.db`, `data/api-cache/`, `*PROGRESS.md`, `output/`, `htmlcov/`, any `*.db`. These are in `data/.gitignore`, but gitignore does not apply to a file staged with `git add -f`, which is exactly how they get committed by accident.
2. **Refuses any blob over 5 MB** — a backstop for whatever the name patterns above do not anticipate.
3. **Byte-compiles staged Python** — catches the indentation corruption called out under Coding Standards before the slower test run.
4. **Runs the full suite**, including the 90% coverage floor. About 19 seconds.

Escape hatches, for when you mean it:

```bash
SKIP_TESTS=1 git commit -m "wip"   # skip only step 4
git commit --no-verify             # skip the hook entirely
```

The test run checks the **working tree, not the staged snapshot**. Stashing unstaged changes to test the index exactly is more likely to lose work than to catch a bug, so it deliberately does not. Failing test output goes to `/tmp/bible-study-precommit.log`.

There is no `pre-push` hook: the repo has no remote, so it would never fire.

`.github/workflows/tests.yml` runs the same suite on GitHub for push and pull requests, so the coverage floor is enforced for contributors who never enable the hook. It also verifies sqlite-vec loads before running tests — on a runner that fails to load the extension, that turns a mass of confusing test failures into one clear message.

## Ollama Context Window

**Ollama never errors on an over-long prompt.** llama.cpp runs with `--context-shift`, which evicts the *oldest* tokens and answers anyway. Since every template puts its instructions first and `{chapter_text}` / `{chapter_summaries}` last, the instruction header is the first thing discarded — the model returns a confident, well-formed answer to a question it only partially saw. This is invisible unless you check the output against the template it was supposed to follow.

It bit `summarize_book()`, which concatenates every chapter summary with no size cap. Psalms is 150 chapters ≈ 276 KB ≈ 70k tokens; the server had autodetected a 32k window (`OLLAMA_CONTEXT_LENGTH` defaults to 4k/32k/256k by VRAM), so stored Psalms summaries cover only ~chapters 107–150 and ignore the output template. Isaiah (~36k tokens) was affected too; Genesis (~28k) sat just under the line.

Two mechanisms now guard this:

- `ollama.generate()` always sends `options.num_ctx`, resolved from `ollama_num_ctx` in `config.yaml` via `prompts.get_num_ctx()` (same fallback chain as `get_model()` → `ollama.NUM_CTX`). The window no longer depends on how the server happened to start.
- `ollama.check_prompt_fits()` runs *before* the request and raises `PromptTooLongError` when the estimate exceeds `num_ctx - RESPONSE_RESERVE_TOKENS`. Sizing uses a `CHARS_PER_TOKEN = 4` heuristic — no tokenizer is shipped, so it is approximate and deliberately conservative.

So over-long prompts now fail loudly instead of being silently truncated. If `summarize-book` raises `PromptTooLongError`, that is the guard working: raise `ollama_num_ctx`, or shorten the input. Note that a bigger window only prevents *truncation* — feeding ~70k tokens to a small model still yields a shallow summary, so chunked map-reduce in `summarize_book()` remains the real fix for long books.

`claude.sh` is a personal launcher, not part of the package. It exports `OLLAMA_CONTEXT_LENGTH` in front of `ollama launch` — a client command — so it has never affected the server this tool talks to.

## Vector Search and `ask`

`embed` builds a semantic index over three tiers, all keyed on `book_name`
(never `book_abbrev` — `verses.book_abbrev` is `book_name[:3].upper()`, which
disagrees with `indexer.BIBLE_BOOKS`: Ezekiel is `EZE` there but `EZK` here).

| tier | count | source |
|------|-------|--------|
| `verse` | 9,969 | 5-verse windows advancing 3 verses, so chunks overlap |
| `chapter` | 1,189 | each stored chapter summary, whole |
| `book` | 66 | each stored book summary |

11,224 vectors at 1024 dims. Building takes about 5 minutes on GPU (~36
chunks/sec) and grows `data/bible.db` from 16 MB to roughly 81 MB — which is
why the database is no longer tracked in git.

Chunk *metadata* lives in the plain `chunks` table in `INIT_SQL`, so it stays
readable when sqlite-vec is unavailable. Only vectors live in the three vec0
tables, joined back on `chunks.id`.

### Things that will silently corrupt the index

- **Never use `INSERT OR REPLACE` on `chunks`.** Every other write in `db.py`
  does, so this is the easy mistake. `REPLACE` deletes the row and reinserts
  with a *new rowid*, orphaning the vec0 vector. Use `ON CONFLICT DO UPDATE`.
  `test_upsert_keeps_the_id_stable_on_update` is the guard.
- **Sentinel `0`, not `NULL`, for non-applicable columns.** SQLite treats
  NULLs as distinct in a UNIQUE index, so nullable `chapter`/`verse_start`
  would let every re-index insert duplicate summary rows.
- **Rank on order, never on an absolute distance threshold.** Vectors are
  normalised on write and query, so L2 and cosine rank identically — but they
  are different functions of the same similarity, so a hard cutoff means
  different things depending on which DDL the installed build accepted.
- **Do not filter after KNN.** Asking for the top 10 and then keeping only
  Genesis can return nothing even when Genesis matches well. Query the tier
  tables separately instead, which is why there are three.
- **Stamp `embedded_hash` only after vectors commit**, so a crash leaves rows
  stale and the next run redoes them.

### Retrieval design

Three vec0 tables rather than one with a filterable metadata column, because
retrieval needs top-k *per tier*: verse chunks outnumber chapter summaries
8:1, so a single global top-k is routinely all verses and no summaries.

Merging uses weighted reciprocal rank, not raw distance. Distances across
tiers are numerically comparable — one vector space — but not semantically:
chapter summaries are long, abstract, LLM-written prose that scores closer to
an abstract question than terse 17th-century verse text does. Sorting the
merged pool by distance returns nearly all summaries and nearly no scripture.

Expansion is one level deep: a verse hit pulls its chapter summary, a chapter
hit pulls its book summary, a book hit pulls nothing. Two levels would let a
single verse hit drag in kilobytes of summary and crowd out everything else.
Expansions inherit their parent's score and sort immediately after it on the
`is_expansion` flag — no scalar penalty can place an expansion of rank *r*
above the rank *r+1* parent for every *r*, since the required penalty tends
to 1 as *r* grows.

`answer_question` raises on empty retrieval rather than generating. A blank
sources section produces a fluent answer from parametric memory that is
indistinguishable from a grounded one.

### Changing the embedding model

Set `embed_model` and `embed_dims` in `config.yaml`. A vec0 column's width is
fixed at creation and cannot be altered, so `init_vec` refuses a width change
and tells you to run `embed --rebuild`. Changing only the model (same width)
marks every chunk stale and re-embeds on the next `embed`.

Note the split: the **model** comes from config (your intent), the **width**
comes from the index's own `vec_meta` record (a physical property of the
existing columns). `embed_all` reads the stored width, so a stray config edit
cannot write mismatched vectors.

Qwen3-Embedding is **asymmetric** — queries get an `Instruct: ...\nQuery: ...`
prefix, documents are embedded raw. Both sides are handled inside
`ollama.embed()` via `is_query`. Wrapping documents too, or leaving queries
bare, degrades retrieval invisibly; `test_documents_are_left_raw` guards it.

### Answer quality

Retrieval is strong: "a woman who gleaned in the fields of her kinsman"
returns Ruth 2 without the word Ruth appearing in the query. Grounding is only
as good as the answering model. `gemma3:4b` will still extrapolate from
loosely related passages when asked something Scripture does not address — it
leads with the disclaimer but then answers anyway. If groundedness matters
more than latency, point `ollama_model` at a larger model.

## The `bible-ask` Claude Code Skill

`.claude/skills/bible-ask/SKILL.md` answers corpus questions with **Claude** rather than the local Ollama model, while reusing this project's retrieval unchanged. It is not an API integration: there is no `anthropic` dependency, no API key, and no per-query cost — the answering model is the Claude Code session itself.

The split is the point. Retrieval is the part that needs this repo (the vector index, the chunking, the expansion rules); answering is the part a stronger model does better. The skill runs:

```bash
uv run bible-study search "<question>" --expand --json -k 8
```

`--expand` returns the assembled, ranked context `ask` itself would send — each hit plus its parent summary, deduped — so the skill and `ask` are grounded in identical material and only the answering model differs. That is also why `search` gained `--json`: the skill needs a stable machine-readable interface, and coupling it to `vectors.search` internals would break on refactor.

**Ollama must still be running.** The query is embedded locally with `qwen3-embedding:0.6b` even though Claude writes the answer.

The skill's grounding rules mirror the `ask` template in `config.yaml` — cite every claim, prefer KJV text over generated summaries, and decline rather than reason by analogy when the retrieved sources do not address the question. If you change one, change the other.

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
| `cli.py` | Click commands: init, summarize, summarize-book, embed, ask, search, view, status, export, clear-summaries, clear-book-summaries, config-edit |
| `api.py` | bible-api.com client: JSON cache under `data/api-cache/`, chapter enumeration, and retry/backoff for `RETRYABLE_STATUS` (429/5xx) honouring `Retry-After`. Cached files may hold either a bare verse list (older) or a mapping (newer) — `fetch_chapter` normalises both, so keep that branch when touching the cache format |
| `db.py` | SQLite schema & all CRUD helpers (verses, chapter_summaries, book_summaries, chunks, vec_meta) |
| `indexer.py` | Hardcoded 66-book structure, book names, chapter counts |
| `ollama.py` | Local Ollama API client: health_check, check_model_available, generate, embed, plus the context-window guard (check_prompt_fits, PromptTooLongError) |
| `prompts.py` | YAML config loader + prompt builders (chapter_summary, book_summary, ask) |
| `summary.py` | Pipeline orchestration: summarize_chapter, summarize_book, export_markdowns |
| `browser.py` | SQLite-backed HTTP server with _SQLiteHandler class and serve() entry point. Uses `ThreadingHTTPServer` so a slow `/ask` does not block every other request |
| `vectors.py` | sqlite-vec layer: extension loading, vec0 tables, chunking, embed_all, KNN search |
| `rag.py` | Grounded question answering: retrieve, expand, rank, budget, one generate() call |

## Browser Handler Rules

Every response must go through `_send_html()` or `_send_error()` — both send a status line, `Content-Type`, and `Content-Length`. Writing straight to `self.wfile` produces a headerless reply that clients reject as HTTP/0.9. Handlers hold a `db_path` (not a shared connection) and read through `db.py` helpers, which open and close per call. All stored text (verses, summaries) is passed through `html.escape()` before rendering.

Handler tests built with `_SQLiteHandler.__new__` must mock `send_response`, `send_header`, and `end_headers` alongside `wfile` — see `_handler_for()` in `tests/test_browser.py`. Mocked-handler tests cannot catch protocol errors, so `TestOverTheWire` runs a real `HTTPServer` on an ephemeral port and asserts parsed status codes.
