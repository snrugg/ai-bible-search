"""Bible Study — Index the KJV Bible and generate chapter summaries via local LLMs."""

from bible_study.cli import cli


def main():
    """Entry point wrapper for Click's CLI group."""
    cli()


__all__ = ["cli", "main"]
