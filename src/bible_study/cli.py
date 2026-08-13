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

    from bible_study.ollama import health_check
    if not health_check():
        raise click.ClickException(
            "Cannot reach Ollama at http://localhost:11434 -- start it first.",
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
    browse(port=port)


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
    click.echo(f"Remaining:     {total - summed}")


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
def config_edit() -> None:
    """Open config.yaml in the default text editor."""
    import webbrowser
    from pathlib import Path as _P
    cfg = _P("config.yaml")
    if cfg.exists():
        webbrowser.open(str(cfg.resolve()))
    else:
        click.echo(f"Config file not found: {cfg}")