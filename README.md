# Bible Study — KJV Index, Summary & Semantic Search

An indexed, summarized, semantically searchable version of the entire King James
Version (KJV) Bible, built locally with SQLite and a local LLM (Ollama). Fetches
text from [bible-api.com](https://bible-api.com), generates chapter and book
summaries via Ollama, embeds everything into a
[sqlite-vec](https://github.com/asg017/sqlite-vec) vector index, and answers
questions from the indexed corpus with citations. Nothing leaves your machine.

## Features

- **Full KJV corpus** — 66 books, 1,189 chapters, 31,102 verses indexed automatically
- **AI-powered summaries** — chapter-level and book-level summaries from local Ollama models
- **Semantic search** — 11,224 vectors over verse passages *and* generated summaries
- **Grounded question answering** — `ask` retrieves, expands, and answers with references
- **SQLite storage** — everything in one file, no external services
- **YAML-configurable** — prompts, models, and chunking in `config.yaml`, no code changes
- **Browser viewer** — HTTP server with clickable navigation, search, and ask
- **Markdown export** — every summary as linked markdown files
- **Tested** — 569 tests, 99% coverage, enforced by a pre-commit hook

## Quick Start

### Prerequisites

- **Python 3.14+** (managed via `uv`)
- **[uv](https://github.com/astral-sh/uv)** — Python package installer and resolver
- **[Ollama](https://ollama.com)** running locally, with two models:

```bash
ollama pull gemma3:4b               # generation (or any chat model you prefer)
ollama pull qwen3-embedding:0.6b    # embeddings -- 1024 dims, ~640 MB
```

The generation model is set by `ollama_model` in `config.yaml`; the embedding
model by `embed_model`. Both must be **embedding** and **chat** models
respectively — they are not interchangeable.

### Installation

```bash
git clone <repo-url> && cd bible-study
uv sync
git config core.hooksPath .githooks   # enable the pre-commit hook
```

## Usage

```bash
# 1. Download the KJV text from bible-api.com into SQLite  (~40 min, resumable)
uv run bible-study init

# 2. Generate chapter summaries via Ollama                  (resumable)
uv run bible-study summarize

# 3. Generate book-level aggregate summaries                (optional)
uv run bible-study summarize-book

# 4. Build the vector search index                          (~5 min, incremental)
uv run bible-study embed

# Ask a question, grounded in the indexed corpus
uv run bible-study ask "Why did Abraham leave Ur?"

# Browse, search, and ask at http://localhost:8080
uv run bible-study view

# Check progress
uv run bible-study status

# Export all summaries as linked markdown to output/
uv run bible-study export
```

Steps 1–3 must run before step 4: `embed` indexes verses *and* whatever
summaries exist at the time. Re-run `embed` after generating more summaries —
it only embeds what changed.

### Data directory

All commands except `config-edit` accept `--data-dir DIR` (or `-d DIR`) to
override the default `data/` location.

### What gets written to disk

| Path | Contents |
|------|----------|
| `data/api-cache/<book>-<chapter>.json` | Raw API response, one file per chapter (1,189 files, ~7 MB) |
| `data/bible.db` | SQLite: verses, summaries, chunks, and vectors (~81 MB once embedded) |
| `data/SUMMARY_PROGRESS.md` | Per-chapter log written by `summarize` |
| `data/EMBED_PROGRESS.md` | Per-batch log written by `embed` |

**All of these are git-ignored.** The database reaches ~81 MB once the vector
index is built, and binary blobs do not delta-compress — committing it would add
tens of megabytes to history on every re-index. The pre-commit hook refuses to
commit them even with `git add -f`.

## Semantic Search and `ask`

`embed` builds an index over three tiers, all searched together:

| Tier | Count | Source |
|------|-------|--------|
| Verse passages | 9,969 | 5-verse windows advancing 3 verses, so chunks overlap |
| Chapter summaries | 1,189 | each generated chapter summary, whole |
| Book summaries | 66 | each generated book summary |

11,224 vectors at 1024 dimensions. Building takes about **5 minutes** on GPU
(~36 chunks/sec) and is **incremental** — re-running embeds only chunks whose
text changed, whose model changed, or that were never embedded. Interrupting it
is safe; the next run picks up where it stopped.

Overlapping windows mean a passage spanning a chunk boundary is still embedded
as a unit. Set `chunk_stride: 5` in `config.yaml` for non-overlapping windows —
6,704 verse chunks instead of 9,969, about a third faster to build, at some cost
in recall near the seams.

### How `ask` works

```
question ──▶ embed as query ──▶ KNN across all three tiers
                                        │
                                        ▼
                             expand one level (plain Python, no LLM)
                               a passage  ─▶ + its chapter summary
                               a chapter  ─▶ + its book summary
                                        │
                                        ▼
                        dedupe ─▶ rank ─▶ fit to context window
                                        │
                                        ▼
                              ONE generate() call ──▶ answer + citations
```

There is no agentic loop and no second LLM pass, so the assembled prompt is
deterministic. Retrieval that comes back empty raises rather than generating: a
blank sources section produces a fluent answer from the model's own memory that
is indistinguishable from a grounded one.

```bash
uv run bible-study ask "What does Scripture say about covenant?" -k 20
uv run bible-study ask "Who was Ruth's mother-in-law?" --no-show-sources
```

`-k` sets how many verse passages to retrieve; the summary tiers scale down from
it (`-k 8` gives 8 verse, 4 chapter, 2 book hits).

### Debugging retrieval

The browser's `/search` route runs retrieval **without** an LLM call, so it
shows exactly what `ask` is working from. If an answer looks wrong, look there
first — the problem is usually retrieval, not generation.

### A caveat on groundedness

Retrieval is strong: *"a woman who gleaned in the fields of her kinsman"*
returns Ruth 2 without "Ruth" appearing in the query. Grounding, though, is only
as good as the answering model. `gemma3:4b` will still extrapolate from loosely
related passages when asked something Scripture does not address — it leads with
a disclaimer, then answers anyway. If groundedness matters more than latency,
point `ollama_model` at a larger model.

## Browser Viewer

`bible-study view` serves on `http://localhost:8080`:

| Route | Page |
|-------|------|
| `/` | 66-book index with per-book download and summary progress |
| `/book/<slug>` | Chapter grid plus the book summary |
| `/book/<slug>/<n>` | Chapter summary plus the KJV verses |
| `/search?q=` | Ranked semantic hits with distances, linked to their chapters |
| `/ask?q=` | Grounded answer with linked sources |

Slugs are lowercase with hyphens: `1-samuel`, `song-of-solomon`.

## CLI Reference

| Command | Description |
|---------|-------------|
| `bible-study init` | Download all KJV books to SQLite |
| `bible-study summarize` | Generate chapter summaries for all unsummarized chapters |
| `bible-study summarize-book` | Generate book-level aggregate summaries |
| `bible-study embed [--rebuild] [--limit N] [--batch-size N]` | Build the vector index |
| `bible-study ask "QUESTION" [-k N] [--no-show-sources]` | Answer a question from the index |
| `bible-study view [--port PORT]` | Launch the browser viewer |
| `bible-study status` | Show indexing, summarization, and embedding progress |
| `bible-study export [-o DIR]` | Export summaries as linked markdown (default `output/`) |
| `bible-study clear-summaries [--book B] [--scope all\|chapters\|books] [-y]` | Delete summaries so they can be regenerated |
| `bible-study clear-book-summaries [--book B] [-y]` | Delete only book-level summaries |
| `bible-study config-edit` | Open `config.yaml` in the OS default handler |

Every command except `config-edit` also accepts `--data-dir DIR` / `-d DIR`.

## Configuration

Everything user-tunable lives in `config.yaml` at the repo root — **no code
change is needed**. `prompts.py` reads the file at runtime.

### Settings

| Key | Default | Purpose |
|-----|---------|---------|
| `ollama_model` | `qwen3.6:35b-a3b-nvfp4` | Chat model used for summaries and answers |
| `ollama_num_ctx` | `32768` | Context window requested on every call (see below) |
| `embed_model` | `qwen3-embedding:0.6b` | Embedding model for chunks and questions |
| `embed_dims` | `1024` | Vector width emitted by `embed_model` |
| `chunk_window` | `5` | Verses per chunk |
| `chunk_stride` | `3` | Verses each window advances; `< window` means overlap |

Defaults come from `bible_study.ollama` and apply when a key is absent, blank,
or invalid. Model names must match `ollama list` exactly, tag included.

### Prompt templates

| Key | Used by | Placeholders |
|-----|---------|--------------|
| `chapter_summary` | `summarize` | `{book_name}`, `{chapter_number}`, `{chapter_text}` |
| `book_summary` | `summarize-book` | `{book_name}`, `{chapter_count}`, `{chapter_summaries}` |
| `ask` | `ask`, `/ask` | `{question}`, `{context}` |

Placeholders are substituted by plain text replacement. Any placeholder you
leave out is simply left alone; unknown keys are ignored. Values are inserted
literally, so a question containing `\1` or `\g<0>` is safe.

### Where the config is found

`load_config()` resolves in this order, first match wins:

1. `$BIBLE_STUDY_CONFIG` — an explicit path, if set
2. `./config.yaml` — relative to the current working directory
3. `config.yaml` at the repo root
4. `config.yaml` beside the installed package

So you can keep experiment variants side by side:

```bash
BIBLE_STUDY_CONFIG=prompts/terse.yaml uv run bible-study summarize
```

### Seeing the prompt

```bash
uv run python -c "
from bible_study.prompts import load_config, build_chapter_prompt
print(build_chapter_prompt(load_config(), 'Genesis', '<chapter text>', 1))
"
```

Chapter prompts are also stored per chapter in `chapter_summaries.prompt_used`,
so you can audit exactly what produced any given summary. `ask` returns the full
prompt it used under the `prompt` key.

### Re-running after a prompt change

`summarize` skips chapters that already have a summary, so clear the old ones
first. Verse text is never touched, so nothing is re-downloaded:

```bash
uv run bible-study clear-summaries                       # everything, both levels
uv run bible-study clear-summaries --book Genesis        # one book
uv run bible-study clear-summaries --scope books         # book summaries only
uv run bible-study summarize && uv run bible-study embed
```

Re-run `embed` afterwards — changed summary text is detected by content hash and
re-embedded automatically.

### Changing the embedding model

Set `embed_model` and `embed_dims` together. A vector column's width is fixed
when the index is created and cannot be altered, so a width change is **refused**
rather than allowed to mix two vector spaces:

```bash
uv run bible-study embed --rebuild   # discard old vectors, start over
```

Changing only the model at the same width needs no rebuild — every chunk is
marked stale and re-embedded on the next `embed`.

## The Ollama Context Window

**Ollama never errors on an over-long prompt.** llama.cpp runs with
`--context-shift`, which evicts the *oldest* tokens and answers anyway. Since
every template puts its instructions first and the bulk text last, the
instruction header is the first thing discarded — the model returns a confident,
well-formed answer to a question it only partially saw.

This silently corrupted book summaries for long books. Psalms is 150 chapters,
roughly 70k tokens of chapter summaries, against a server that had autodetected
a 32k window; the stored summary covered only about chapters 107–150.

Two guards now prevent it:

- `ollama.generate()` always sends `options.num_ctx` from `ollama_num_ctx`, so
  the window no longer depends on how the server happened to start.
- `ollama.check_prompt_fits()` runs *before* the request and raises
  `PromptTooLongError` when the estimate exceeds the window minus a response
  reserve.

If `summarize-book` raises `PromptTooLongError`, the guard is working: raise
`ollama_num_ctx` or shorten the input. Verify a change landed with `ollama ps` —
the `CONTEXT` column reflects what the last request actually asked for.

`ask` sizes its retrieved context against the same budget and drops the
lowest-ranked sources to fit, reporting how many it dropped rather than
truncating silently.

## Project Structure

```
bible-study/
├── pyproject.toml                      # uv/hatchling config; pytest, coverage, mutmut
├── config.yaml                         # Prompts, models, chunking (user-editable)
├── CLAUDE.md                           # Guidance for Claude Code
├── README.md                           # You are here
├── .githooks/pre-commit                # Guards + compile check + test suite
│
├── src/bible_study/
│     ├── __init__.py                   # Entry point; delegates to Click CLI
│     ├── cli.py                        # Click commands
│     ├── api.py                        # bible-api.com client with file cache + backoff
│     ├── indexer.py                    # Hardcoded 66-book structure + chapter counts
│     ├── db.py                         # SQLite schema and CRUD helpers
│     ├── ollama.py                     # Ollama client: generate, embed, context guard
│     ├── prompts.py                    # YAML config loader + prompt builders
│     ├── summary.py                    # Summary pipeline orchestration
│     ├── vectors.py                    # sqlite-vec: chunking, embedding, KNN search
│     ├── rag.py                        # Retrieve, expand, rank, budget, answer
│     └── browser.py                    # SQLite-backed HTTP server
│
├── tests/                              # 569 tests, all mocked by default
│     ├── conftest.py                   # Shared fixtures
│     ├── test_api.py  test_api_extra.py  test_rate_limit.py
│     ├── test_db.py   test_indexer.py    test_ollama.py
│     ├── test_prompts.py                test_pipeline.py
│     ├── test_summary.py                test_summary_extra.py
│     ├── test_vectors.py                test_rag.py
│     ├── test_cli.py                    test_browser.py
│     ├── test_coverage_gaps.py         # Tests organised by uncovered branch
│     └── integration/                  # Live I/O, skipped by default
│
├── data/                               # Generated at runtime; git-ignored
└── output/                             # Markdown exports from `bible-study export`
```

## Architecture

```
cli.py (Click CLI)
    │
    ├── init            ─▶ api.download_all() ─▶ db.upsert_verses()
    ├── summarize       ─▶ summary.generate_all_chapters()
    │                        ├── ollama.generate()
    │                        └── db.save_summary()
    ├── summarize-book  ─▶ summary.summarize_book()
    ├── embed           ─▶ vectors.rebuild_chunks() ─▶ db.chunks
    │                        └── vectors.embed_all() ─▶ ollama.embed() ─▶ vec0 tables
    ├── ask             ─▶ rag.answer_question()
    │                        ├── vectors.embed_query() + vectors.search()
    │                        ├── expand / rank / budget   (plain Python)
    │                        └── ollama.generate()
    ├── view            ─▶ browser.serve()   (ThreadingHTTPServer)
    ├── status          ─▶ db.get_chapter_progress() + db.chunk_counts()
    └── export          ─▶ summary.export_markdowns()
```

Chunk **metadata** lives in an ordinary `chunks` table, so it stays readable
when sqlite-vec is unavailable. Only the vectors live in virtual tables, joined
back on `chunks.id` — one table per tier, because retrieval needs top-k *per
tier* and verse chunks outnumber chapter summaries 8:1.

## Testing

All tests are mocked by default — no live API or LLM calls.

```bash
uv sync --extra dev                            # Install dev deps
uv run pytest                                  # Full suite; coverage on by default
uv run pytest --no-cov                         # Fast, no coverage overhead
uv run pytest tests/test_vectors.py --no-cov   # One file
uv run pytest -k "expand" --no-cov             # By name pattern
uv run mutmut run                              # Mutation testing
```

`addopts` in `pyproject.toml` already passes `--cov=bible_study`, so a bare
`uv run pytest` produces a coverage report and **fails under 90%**. Current
state: **569 passing, 9 skipped by design** (7 live-I/O integration tests, 2
over-the-wire browser tests), **99% coverage**.

Integration tests are skipped with class-level `@pytest.mark.skip`. Run them by
hand after removing the marker; they need a live Ollama and network access.

### Test Matrix

| Module | Purpose |
|--------|---------|
| `test_api.py`, `test_api_extra.py`, `test_rate_limit.py` | API client — caching, retries, 429 backoff |
| `test_db.py` | SQLite schema and CRUD, including chunk rowid stability |
| `test_indexer.py` | 66-book structure, chapter counts |
| `test_ollama.py` | Ollama client — generate, embed, retries, query/document asymmetry |
| `test_prompts.py` | Config loading, prompt assembly, literal substitution |
| `test_summary.py`, `test_summary_extra.py`, `test_pipeline.py` | Summary pipeline |
| `test_vectors.py` | Chunking, vec0 storage, KNN — real extension, 4-dim toy vectors |
| `test_rag.py` | Expansion, ranking, budgeting, answering — fully mocked |
| `test_cli.py` | Every command via `CliRunner` |
| `test_browser.py` | HTTP server, including real-socket route tests |
| `test_coverage_gaps.py` | Organised by uncovered branch rather than by module |

Vector tests run against the **real** sqlite-vec extension with 4-dimensional
toy vectors. Faking it would test our SQL against a mock and prove nothing about
the two things most likely to break: whether the DDL parses, and whether the
`k = ?` KNN constraint binds as a parameter.

## Git Hooks

Hooks live in `.githooks/` so they are versioned with the repo. Enable them once
per clone:

```bash
git config core.hooksPath .githooks
```

`pre-commit` refuses to commit the generated database, the API cache, progress
logs, or any blob over 5 MB; byte-compiles staged Python; then runs the full
test suite including the coverage floor.

```bash
SKIP_TESTS=1 git commit -m "wip"   # skip only the test run
git commit --no-verify             # skip every check
```

The test run checks the working tree, not the staged snapshot — stashing
unstaged changes to test the index exactly is more likely to lose work than to
catch a bug.

## Architecture Decisions

- **SQLite** over a full database — zero external services, single-file portable storage
- **sqlite-vec** over a dedicated vector database — the corpus is 11,224 vectors; brute-force KNN in-process is faster than a network round trip, and there is nothing else to run
- **bible-api.com** free tier for KJV text — no API key required
- **Local Ollama** for generation and embeddings — data stays on disk, offline-capable
- **Deterministic retrieve-then-expand** over an agentic loop — one LLM call, a reproducible prompt, and a pipeline that is testable without a model
- **Click CLI** for command orchestration — cross-platform, composable subcommands
- **YAML config** over inline constants — prompts and models editable without code changes
- **Plain `str.replace()` templating** instead of Jinja2 — no extra dependency, and values are inserted literally

## License

Private project — all rights reserved.
