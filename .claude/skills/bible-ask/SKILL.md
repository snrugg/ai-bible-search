---
name: bible-ask
description: Answer a question about the KJV Bible from this repo's indexed corpus — 31,102 verses plus generated chapter and book summaries, retrieved by semantic search. Use whenever the user asks something answerable from Scripture or from the summaries in this project ("what does the Bible say about X", "who was Y", "where does Z happen", "trace the theme of W"), or explicitly asks to search or ask the corpus. Retrieval is local; you write the answer.
---

# Answering from the indexed corpus

This repo ships a semantic index over the whole KJV: verse passages, per-chapter
summaries, and per-book summaries. `bible-study ask` answers such questions with
a local Ollama model. This skill does the same retrieval and lets **you** write
the answer instead — same grounding rules, better prose and judgment.

## Retrieve first

Run this before answering. Never answer from memory — the point is to ground the
answer in *this* corpus, with citations the user can click through to.

```bash
uv run bible-study search "<the user's question, verbatim>" --expand --json -k 8
```

- `--expand` returns the assembled context `ask` itself would use: each hit plus
  its parent summary, deduped and ranked. This is what you want by default.
- `-k` sets verse passages retrieved; the summary tiers scale down from it.
  Raise to 16–20 for broad thematic questions ("trace the theme of covenant"),
  drop to 4–6 for a specific factual lookup.
- Drop `--expand` when you only want the raw ranked hits with distances — useful
  when the user is asking *what the index contains* rather than a question about
  Scripture.

Each JSON row has `kind` (`verses` / `chapter-summary` / `book-summary`),
`citation`, `book_name`, `chapter`, `score`, `is_expansion`, and `text`.

If the first search is thin or off-target, search again with a rephrasing before
concluding the corpus lacks an answer — retrieval is cheap and takes about a
second.

## Then answer

Ground everything in what came back:

- Cite the reference for each claim, using the `citation` field as given
  (`Genesis 17:7-11`, `Genesis 17 (summary)`).
- Prefer the KJV text over the summaries where both cover the same ground. The
  summaries are generated study notes, not Scripture — say so if you lean on
  them for something contested.
- The rows are the nearest matches by similarity, so they are *always*
  populated. That is not evidence they are relevant. Use the ones that bear on
  the question and ignore the rest.
- If none of them bear on it, say the corpus does not address the question and
  stop. Don't reason by analogy from loosely related passages, and don't
  substitute general principles for an answer the sources can't give.
- Where the sources genuinely disagree or are ambiguous, say so rather than
  resolving it silently.
- Match length to the question: a couple of sentences for a factual lookup, more
  for a thematic trace across books.

Keep the tone of a study-guide entry — neutral and informative. The project's
own prompts frame the voice as a conservative Christian Bible scholar working
only from the text in front of them; match that, and leave your own theological
commentary out unless asked for it.

## Prerequisites, and what breaks

Retrieval embeds the query locally with `qwen3-embedding:0.6b`, so **Ollama must
be running** even though you are the one answering.

| Error | Fix |
|---|---|
| `Cannot reach Ollama` | Start Ollama |
| `No vector index found` | `uv run bible-study embed` |
| `No database at data/bible.db` | `uv run bible-study init`, then `summarize`, then `embed` |
| Empty results on a reasonable question | Rephrase and retry before concluding the corpus lacks it |

Add `-d <dir>` to `search` if the database is not at the default `data/`.

## Related

- `uv run bible-study ask "..."` — the same pipeline answered by the local model
  instead of you. Slower and less capable, but works with no Claude Code session.
- `uv run bible-study view` — browser at `localhost:8080`; `/search` shows the
  ranked hits and `/ask` runs the local pipeline, both linked to chapter pages.
