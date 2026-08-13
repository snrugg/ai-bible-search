"""Summary pipeline -- orchestrate fetch, prompt, generate, store.

Ties together the Bible API client, Ollama client, SQLite storage, and
YAML-configured prompt templates into a coherent summarisation workflow.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from bible_study.indexer import book_names, iter_books
import bible_study.ollama as _ol
from bible_study.prompts import (
    build_book_summary_prompt,
    build_chapter_prompt,
    load_config,
)


def summarize_chapter(
    book_name: str,
    chapter_num: int,
    db_path: Path | None = None,
    ollama_kwargs: dict[str, Any] | None = None,
) -> str:
    """Fetch a chapter, build a prompt, call Ollama, and store the result.

    Returns the generated summary text.  If verses are missing from the DB
    they are fetched fresh from bible-api.com first.

    Raises ``RuntimeError`` if no config file can be found.
    """
    ollama_kwargs = ollama_kwargs or {}

    # -- 1. Get verse text -------------------------------------------------
    verses_text: str = ""
    _db_module: Any | None = None
    if db_path is not None:
        from bible_study import db as _db_mod
        _db_module = _db_mod
        if _db_mod.verse_count(db_path, book_name) > 0:
            verses = _db_mod.get_verses(db_path, book_name, chapter_num)
            verses_text = " ".join(v["text"] for v in verses)

    if not verses_text.strip():
        # Fresh fetch via API
        from bible_study.api import fetch_chapter as _fetch
        raw = _fetch(book_name, chapter_num)
        verses_text = " ".join(v.get("text", "") for v in raw)
        if db_path is not None:
            _db_module.upsert_verses(  # type: ignore[attr-defined]
                db_path, book_name, chapter_num,
                [(v["verse"], v["text"]) for v in raw],
            )

    if not verses_text.strip():
        msg = f"No verse text returned for {book_name} chapter {chapter_num}"
        raise RuntimeError(msg)

    # -- 2. Build prompt ---------------------------------------------------
    config = load_config()
    prompt = build_chapter_prompt(config, book_name, verses_text, chapter_num)

    # -- 3. Call Ollama ----------------------------------------------------
    summary = _ol.generate(prompt, **ollama_kwargs)

    # -- 4. Store in DB ----------------------------------------------------
    if db_path is not None:
        _db_module.save_summary(  # type: ignore[attr-defined]
            db_path, book_name, chapter_num, str(summary), prompt,
        )

    return str(summary)


def summarize_book(
    book_name: str,
    db_path: Path | None = None,
    ollama_kwargs: dict[str, Any] | None = None,
) -> str:
    """Aggregate a book-level summary from its chapter summaries via Ollama.

    Returns the generated book-level summary text.
    """
    ollama_kwargs = ollama_kwargs or {}

    if db_path is None:
        raise ValueError("db_path is required to retrieve chapter summaries")

    from bible_study import db as _db_module
    chapter_sums = _db_module.get_chapter_summaries_for_book(db_path, book_name)
    if not chapter_sums:
        raise RuntimeError(
            f"No chapter summaries found for {book_name} -- summarize chapters first",
        )

    # Build aggregated text from individual chapter summaries
    agg_parts: list[str] = []
    for cs in chapter_sums:
        agg_parts.append(f"Chapter {cs['chapter']}: {cs['summary']}")
    agg_text = "\n\n".join(agg_parts)

    # Get the book's total chapter count
    from bible_study.indexer import get_book as _get_book
    book_info = _get_book(book_name)
    num_chapters = book_info["chapter_count"] if book_info else len(chapter_sums)

    # Build and call prompt
    config = load_config()
    prompt = build_book_summary_prompt(config, book_name, num_chapters)
    summary = _ol.generate(prompt, **ollama_kwargs)

    # Store book-level summary
    if book_info:
        _db_module.save_book_summary(  # type: ignore[attr-defined]
            db_path, book_name, book_info["abbrev"], str(summary),
        )

    return str(summary)


def generate_all_chapters(
    db_path: Path,
    progress_file: Path | None = None,
    ollama_kwargs: dict[str, Any] | None = None,
) -> list[tuple[str, int]]:
    """Generate summaries for all unsummarized chapters.

    Returns the list of ``(book_name, chapter_num)`` pairs that were newly
    summarised in this run.
    """
    ol_kwargs = ollama_kwargs or {}
    books = book_names()
    from bible_study import db as _db_module
    unsummarized = _db_module.get_unsummarized_chapters(db_path, books)

    if not unsummarized:
        print("All chapters already summarized.")  # noqa: T201
        return []

    summary_file = progress_file or (db_path.parent / "SUMMARY_PROGRESS.md")
    results: list[tuple[str, int]] = []

    with open(summary_file, "w") as log_fh:
        log_fh.write("# Bible Summary Progress\n\n")
        log_fh.write(f"Total chapters to summarise: {len(unsummarized)}\n\n")

        for i, (bname, chap) in enumerate(unsummarized):
            try:
                summary = summarize_chapter(
                    bname, chap, db_path=db_path, ollama_kwargs=ol_kwargs,
                )
                results.append((bname, chap))
                elapsed = f"{i + 1}/{len(unsummarized)}"
                log_fh.write(
                    f"- **{elapsed}**: {bname} Ch.{chap} -- summarised "
                    f"({len(str(summary))} chars)\n",
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = f"{i + 1}/{len(unsummarized)}"
                log_fh.write(
                    f"- **{elapsed}**: {bname} Ch.{chap} -- **FAILED** ({exc})\n",
                )

            # Brief pause between Ollama calls to avoid rate limits
            time.sleep(0.2)

    return results


def render_chapter_markdown(
    book_name: str,
    chapter_num: int,
    summary: str,
    *,
    verses_text: str | None = None,
) -> str:
    """Render a chapter's markdown with cross-links and optional verse text."""
    from bible_study.indexer import get_book as _get_book

    book_info = _get_book(book_name)
    if book_info is None:
        abbrev = "UNK"
        total_chapters = chapter_num
    else:
        abbrev = book_info["abbrev"]
        total_chapters = book_info["chapter_count"]

    prev_chap = chapter_num - 1 if chapter_num > 1 else None
    next_chap = chapter_num + 1 if chapter_num < total_chapters else None

    lines: list[str] = []
    lines.append(f"# {book_name} Chapter {chapter_num}")
    lines.append("")
    lines.append(summary)
    lines.append("")
    lines.append("---")
    lines.append("")

    # Optional original verse text (rendered in a collapsible details block)
    if verses_text:
        lines.append('<details><summary>Original KJV Text</summary>')
        lines.append(verses_text)
        lines.append("</details>")
        lines.append("")

    # Cross-links to adjacent chapters in the same book
    lines.append("**See also:**")
    links: list[str] = []
    if prev_chap is not None:
        links.append(
            f"[{book_name} Ch. {prev_chap}]({abbrev.lower()}/chapter-{prev_chap:02d}.md)",
        )
    if next_chap is not None:
        links.append(
            f"[{book_name} Ch. {next_chap}]({abbrev.lower()}/chapter-{next_chap:02d}.md)",
        )
    lines.append(" ".join(links) if links else " *(first or last chapter)*")

    return "\n".join(lines) + "\n"


def export_markdowns(
    db_path: Path,
    book_names_list: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, int]:
    """Export every chapter's summary as a markdown file.

    Structure::

        {output_dir}/
        index.md                        # Master index with book links
        gen/                            # Genesis chapters
            chapter-01.md               # First chapter
                ...
        exo/                           # Exodus
            ...

    Returns ``{book_abbrev: chapter_count}``.
    """
    if output_dir is None:
        raise ValueError("output_dir is required")
    results: dict[str, int] = defaultdict(int)

    for book in iter_books():
        abbrev = book["abbrev"]
        book_dir = output_dir / abbrev.lower()
        book_dir.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [f"# {book['name']}", "", "## Chapters", ""]

        for chap_num in range(1, book["chapter_count"] + 1):
            summary = _get_summary_from_db(db_path, book["name"], chap_num)
            if summary is None:
                continue

            # Build markdown with cross-links
            md = render_chapter_markdown(book["name"], chap_num, summary)
            results[abbrev] += 1

            # Write individual chapter file
            fname = f"chapter-{chap_num:02d}.md"
            (book_dir / fname).write_text(md)

            # Add to book-level index line
            lines.append(f"- [{fname}](./{fname})")

        # Write book index page
        (book_dir / "index.md").write_text("\n".join(lines))

    # Write master index linking all books
    idx_lines: list[str] = ["# Bible Study -- Summary Index", ""]
    for book in iter_books():
        abbrev = book["abbrev"].lower()
        count = results.get(book["abbrev"], 0)
        status = (f" ({count}/{book['chapter_count']} chapters)"
                  if count > 0 else " (not summarised)")
        idx_lines.append(f"- [{book['name']}](./{abbrev}/index.md){status}")

    (output_dir / "index.md").write_text("\n".join(idx_lines))

    return dict(results)


# -- Internal helpers ------------------------------------------------------ #


def _get_summary_from_db(db_path: Path, book_name: str, chapter: int) -> str | None:
    """Get a stored summary from the SQLite DB for export."""
    from bible_study import db as _m
    return _m.get_summary(db_path, book_name, chapter)
