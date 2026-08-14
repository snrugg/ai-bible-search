"""Click-based CLI for Bible Study indexing and summarization."""

from pathlib import Path

import click

from bible_study.browser import serve as browse
from bible_study.summary import generate_all_chapters, summarize_book

_data_dir_option = click.option(
    "--data-dir",
    "-d",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Directory for the SQLite database.",
)

_force_option = click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing data.",
)


@click.group()
def cli():
    """Bible Study -- Index the KJV Bible and generate chapter summaries."""
    pass


@cli.command()
@_data_dir_option
@_force_option
def init(data_dir: Path | None, force: bool) -> None:
    """Download the entire KJV Bible corpus into SQLite."""
    from bible_study.api import download_all
    from bible_study.db import init_db
    from bible_study.indexer import book_names

    data_dir = Path(data_dir) if data_dir else Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "bible.db"

    click.echo(f"Initializing database at {db_path} ...")
    init_db(db_path)
    books = book_names()
    click.echo(f"Indexed {len(books)} books.")
    fetched = download_all(cache_dir=data_dir / "api-cache", db_path=db_path)
    click.echo(f"Downloaded {len(fetched)} chapters.")

    from bible_study.db import verse_count
    click.echo(f"Stored {verse_count(db_path)} verses in {db_path}.")


@cli.command()
@_data_dir_option
def summarize(data_dir: Path | None) -> None:
    """Generate chapter summaries for all unsummarized chapters."""
    from bible_study.db import get_unsummarized_chapters, init_db
    from bible_study.indexer import book_names

    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    init_db(db_path)

    from bible_study.ollama import check_model_available, health_check
    from bible_study.prompts import get_model
    if not health_check():
        raise click.ClickException(
            "Cannot reach Ollama at http://localhost:11434 -- start it first.",
        )

    model = get_model()
    click.echo(f"Using Ollama model: {model}")
    if not check_model_available(model_name=model):
        click.echo(
            f"Warning: '{model}' was not listed by Ollama. "
            "Check `ollama list` or the ollama_model key in config.yaml.",
        )

    pending = get_unsummarized_chapters(db_path, book_names())
    done = generate_all_chapters(db_path)

    if pending and not done:
        progress = db_path.parent / "SUMMARY_PROGRESS.md"
        raise click.ClickException(
            f"All {len(pending)} chapters failed to summarise. "
            f"Check that Ollama is running and see {progress} for details.",
        )

    click.echo(f"Summarised {len(done)} chapters.")
    click.echo("Done!")


@cli.command()
@_data_dir_option
def summarize_book_cmd(data_dir: Path | None) -> None:
    """Generate book-level aggregate summaries."""
    from bible_study.db import init_db, get_all_book_names

    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    init_db(db_path)

    books = get_all_book_names(db_path)
    for book in books:
        click.echo(f"Summarizing {book} ...")
        summarize_book(book, db_path)
        click.echo(f"  => done: {book}")
    click.echo("All book summaries generated.")


@cli.command()
@_data_dir_option
@click.option("--port", "-p", type=int, default=8080, help="Port for the viewer.")
def view(data_dir: Path | None, port: int) -> None:
    """Launch a lightweight browser to view verses and summaries."""
    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    if not db_path.exists():
        click.echo(f"Warning: no database at {db_path} -- run `init` first.")
    click.echo(f"Serving {db_path} at http://localhost:{port}")
    browse(port=port, db_path=db_path)


@cli.command()
@_data_dir_option
def status(data_dir: Path | None) -> None:
    """Show indexing and summarization progress."""
    from bible_study.db import get_chapter_progress, init_db
    from bible_study.indexer import book_names

    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    init_db(db_path)

    total, summed = get_chapter_progress(db_path, book_names())
    click.echo(f"Total chapters: {total}")
    click.echo(f"Summarized:     {summed}")
    click.echo(f"Remaining:      {total - summed}")

    from bible_study.db import chunk_counts
    counts = chunk_counts(db_path)
    if counts:
        chunks = sum(c[0] for c in counts.values())
        embedded = sum(c[1] for c in counts.values())
        click.echo(f"Chunks:         {chunks}")
        click.echo(f"Embedded:       {embedded}")


@cli.command()
@_data_dir_option
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output",
    help="Directory to write markdown files into.",
)
def export(data_dir: Path | None, output_dir: str) -> None:
    """Export every stored summary as linked markdown files."""
    from bible_study.db import init_db
    from bible_study.summary import export_markdowns

    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    init_db(db_path)

    out = Path(output_dir)
    results = export_markdowns(db_path, output_dir=out)
    total = sum(results.values())
    click.echo(f"Exported {total} chapters across {len(results)} books to {out}.")


@cli.command()
@_data_dir_option
@click.option(
    "--book",
    "-b",
    default=None,
    help="Only clear summaries for this book (default: every book).",
)
@click.option(
    "--scope",
    type=click.Choice(["all", "chapters", "books"]),
    default="all",
    help="Which summaries to clear: chapter-level, book-level, or both.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
def clear_summaries(
    data_dir: Path | None, book: str | None, scope: str, yes: bool,
) -> None:
    """Delete stored summaries so they can be regenerated.

    Verse text is never touched -- only summaries are removed, so
    `summarize` will re-generate them without re-downloading anything.
    """
    from bible_study.db import (
        clear_book_summaries,
        clear_chapter_summaries,
        init_db,
    )
    from bible_study.indexer import get_book

    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    if not db_path.exists():
        raise click.ClickException(f"No database at {db_path} -- nothing to clear.")
    init_db(db_path)

    book_name = None
    if book is not None:
        book_info = get_book(book)
        if book_info is None:
            raise click.ClickException(f"Unknown book: {book}")
        book_name = book_info["name"]

    target = book_name or "all 66 books"
    what = {
        "all": "chapter and book summaries",
        "chapters": "chapter summaries",
        "books": "book summaries",
    }[scope]
    if not yes:
        click.confirm(f"Delete {what} for {target}?", abort=True)

    chapters = books = 0
    if scope in ("all", "chapters"):
        chapters = clear_chapter_summaries(db_path, book_name)
    if scope in ("all", "books"):
        books = clear_book_summaries(db_path, book_name)

    click.echo(f"Cleared {chapters} chapter summaries and {books} book summaries.")


@cli.command()
@_data_dir_option
@click.option(
    "--book",
    "-b",
    default=None,
    help="Only clear the summary for this book (default: every book).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
def clear_book_summaries_cmd(
    data_dir: Path | None, book: str | None, yes: bool,
) -> None:
    """Delete book-level summaries so they can be regenerated.

    Chapter summaries and verse text are left alone, so `summarize-book`
    can re-aggregate immediately.  Equivalent to
    `clear-summaries --scope books`.
    """
    from bible_study.db import clear_book_summaries, init_db
    from bible_study.indexer import get_book

    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    if not db_path.exists():
        raise click.ClickException(f"No database at {db_path} -- nothing to clear.")
    init_db(db_path)

    book_name = None
    if book is not None:
        book_info = get_book(book)
        if book_info is None:
            raise click.ClickException(f"Unknown book: {book}")
        book_name = book_info["name"]

    target = book_name or "all 66 books"
    if not yes:
        click.confirm(f"Delete book summaries for {target}?", abort=True)

    books = clear_book_summaries(db_path, book_name)
    click.echo(f"Cleared {books} book summaries.")


@cli.command()
def config_edit() -> None:
    """Open config.yaml in the default text editor."""
    import webbrowser
    from pathlib import Path as _P
    cfg = _P("config.yaml")
    if cfg.exists():
        webbrowser.open(str(cfg.resolve()))
    else:
        click.echo(f"Config file not found: {cfg}")

@cli.command()
@_data_dir_option
@click.option(
    "--rebuild",
    is_flag=True,
    default=False,
    help="Discard existing vectors and embed everything from scratch.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Chunks sent per Ollama request.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Only embed the first N pending chunks (useful as a smoke test).",
)
def embed(
    data_dir: Path | None,
    rebuild: bool,
    batch_size: int | None,
    limit: int | None,
) -> None:
    """Build the vector search index over verses and summaries."""
    from bible_study import vectors as _vec
    from bible_study.db import chunk_counts, init_db
    from bible_study.ollama import check_model_available, health_check
    from bible_study.prompts import get_embed_dims, get_embed_model

    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    if not db_path.exists():
        raise click.ClickException(
            f"No database at {db_path} -- run `init` first.",
        )
    init_db(db_path)

    if not health_check():
        raise click.ClickException(
            "Cannot reach Ollama at http://localhost:11434 -- start it first.",
        )

    model = get_embed_model()
    dims = get_embed_dims()
    click.echo(f"Using embedding model: {model} ({dims} dims)")
    if not check_model_available(model_name=model):
        click.echo(
            f"Warning: '{model}' was not listed by Ollama. "
            f"Run `ollama pull {model}`, or check the embed_model key "
            f"in config.yaml.",
        )

    try:
        _vec.init_vec(db_path, dims=dims, model=model)
        if rebuild:
            click.echo(f"Cleared {_vec.clear_vectors(db_path)} vectors.")
        made = _vec.rebuild_chunks(db_path)
        click.echo(f"Chunked {made} passages and summaries.")
        pending = sum(
            total - done for total, done in chunk_counts(db_path).values()
        )
        done = _vec.embed_all(db_path, batch_size=batch_size, limit=limit)
    except (_vec.VectorSupportError, _vec.VectorIndexError) as exc:
        raise click.ClickException(str(exc)) from exc

    if pending and not done:
        progress = data_dir / "EMBED_PROGRESS.md"
        raise click.ClickException(
            f"All {pending} chunks failed to embed. "
            f"Check that Ollama is running and see {progress} for details.",
        )

    click.echo(f"Embedded {done} chunks.")
    click.echo("Done!")


@cli.command()
@click.argument("question", nargs=-1, required=True)
@_data_dir_option
@click.option(
    "-k",
    "--top-k",
    type=int,
    default=8,
    help="Verse passages to retrieve; the summary tiers scale down from this.",
)
@click.option(
    "--show-sources/--no-show-sources",
    default=True,
    help="List the references used to build the answer.",
)
def ask(
    question: tuple[str, ...],
    data_dir: Path | None,
    top_k: int,
    show_sources: bool,
) -> None:
    """Answer a question using the indexed Bible text and summaries."""
    from bible_study import vectors as _vec
    from bible_study.ollama import (
        PromptTooLongError,
        check_model_available,
        health_check,
    )
    from bible_study.prompts import get_embed_model, get_model
    from bible_study.rag import answer_question

    text = " ".join(question).strip()
    if not text:
        raise click.ClickException("Ask a question, e.g. `ask \"who is Ruth?\"`.")

    data_dir = Path(data_dir) if data_dir else Path("data")
    db_path = data_dir / "bible.db"
    if not db_path.exists():
        raise click.ClickException(
            f"No database at {db_path} -- run `init` first.",
        )

    if not health_check():
        raise click.ClickException(
            "Cannot reach Ollama at http://localhost:11434 -- start it first.",
        )

    model = get_model()
    embed_model = get_embed_model()
    click.echo(f"Using Ollama model: {model}")
    click.echo(f"Using embedding model: {embed_model}")
    for name in (model, embed_model):
        if not check_model_available(model_name=name):
            click.echo(f"Warning: '{name}' was not listed by Ollama.")

    try:
        result = answer_question(
            text,
            db_path,
            k_verse=top_k,
            k_chapter=max(1, top_k // 2),
            k_book=max(1, top_k // 4),
        )
    except (_vec.VectorSupportError, _vec.VectorIndexError,
            PromptTooLongError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("")
    click.echo(result["answer"].strip())

    if show_sources:
        click.echo("")
        click.echo("Sources:")
        for source in result["sources"]:
            click.echo(f"  - {source['citation']}")
        if result["dropped"]:
            click.echo(
                f"  ({result['dropped']} more omitted to fit the "
                f"context window)",
            )
