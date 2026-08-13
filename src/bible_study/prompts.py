"""Prompt configuration -- YAML-based templates for LLM calls."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_FILENAME = "config.yaml"


def load_config(path: str | None = None) -> dict[str, str]:
    """Load prompt templates from a YAML file.

    Returns a mapping of template name (e.g. ``"chapter_summary"``) to
    the raw template string.  If *path* is given it must point to an
    existing ``.yaml`` / ``.yml`` file; otherwise the default filename
    in the project root is used.

    Raises
    ------
    FileNotFoundError
        If no config file can be located.
    """
    if path is not None:
        return _load_file(Path(path))
    # Walk up from this file looking for config.yaml
    candidates = [
        Path("config.yaml"),
        Path(__file__).parent.parent / "config.yaml",
        Path(__file__).parent / "config.yaml",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return _load_file(resolved)
    raise FileNotFoundError("No config.yaml found")


def _load_file(path: Path) -> dict[str, str]:
    """Read a YAML file and return its top-level mapping."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return {}
    return {k: str(v) for k, v in data.items()}


def render(template_name: str, config: dict[str, str], **kwargs: Any) -> str:
    """Fill a template from *config* with the given keyword arguments.

    Placeholders follow the ``{key}`` pattern and are replaced using a
    simple regular-expression substitution so that every missing key
    silently becomes ``""`` (empty string).  Unknown keys are ignored.
    """
    try:
        template = config[template_name]
    except KeyError as exc:
        msg = f"Template '{template_name}' not found in config"
        raise KeyError(template_name) from exc
    result = template
    for key, value in kwargs.items():
        placeholder = "{" + str(key) + "}"
        if placeholder in result:
            result = re.sub(re.escape(placeholder), str(value or ""), result)
    return result


def _render_template(text: str, **kwargs: Any) -> str:
    """Replace ``{key}`` placeholders in *text* with kwargs values."""
    for key, value in kwargs.items():
        placeholder = "{" + str(key) + "}"
        if placeholder in text:
            text = re.sub(re.escape(placeholder), str(value or ""), text)
    return text


# -- Inline builders (no config file required) ---------------------------- #


def build_chapter_prompt(
    config: dict[str, str],
    book_name: str,
    chapter_text: str,
    chapter_number: int,
) -> str:
    """Build a chapter summary prompt using the stored template.

    Falls back to an inline template when the config has no
    ``chapter_summary`` key.
    """
    if "chapter_summary" in config:
        return render(
            "chapter_summary",
            config,
            book_name=book_name,
            chapter_number=chapter_number,
            chapter_text=chapter_text,
        )
    return _inline_chapter_prompt(book_name, chapter_number, chapter_text)


def build_book_summary_prompt(
    config: dict[str, str],
    book_name: str,
    chapter_count: int,
) -> str:
    """Build a book-level aggregate prompt using the stored template.

    Falls back to an inline template when the config has no
    ``book_summary`` key.
    """
    if "book_summary" in config:
        return render(
            "book_summary",
            config,
            book_name=book_name,
            chapter_count=chapter_count,
        )
    return _inline_book_prompt(book_name, chapter_count)


# -- Inline fallback templates -------------------------------------------- #

_inline_chapter_template = (
    "Summarize the following chapter of the KJV Bible in 1-2 paragraphs.\n"
    "\n"
    "Book: {book_name}\n"
    "Chapter: {chapter_number}\n"
    "\n"
    "Text:\n"
    "{chapter_text}\n"
    "\n"
    "Provide your summary below."
)


_inline_book_template = (
    "Create an aggregate summary for the entire book of {book_name}.\n"
    "The book has {chapter_count} chapters total.\n"
    "\n"
    "Use 2-3 paragraphs. Highlight major narrative arcs and\n"
    "theological themes."
)


def _inline_chapter_prompt(
    book_name: str,
    chapter_number: int,
    chapter_text: str,
) -> str:
    """Inline fallback for chapter summarisation."""
    return _render_template(
        _inline_chapter_template,
        book_name=book_name,
        chapter_number=chapter_number,
        chapter_text=chapter_text,
    )


def _inline_book_prompt(book_name: str, chapter_count: int) -> str:
    """Inline fallback for book-level summarisation."""
    return _render_template(
        _inline_book_template,
        book_name=book_name,
        chapter_count=chapter_count,
    )


# -- Convenience helpers (used by tests) ---------------------------------- #


def build_inline_chapter_prompt(
    book_name: str,
    chapter_text: str,
    chapter_number: int,
) -> str:
    """Build a chapter prompt without needing any config file."""
    return _inline_chapter_prompt(book_name, chapter_number, chapter_text)


def build_inline_book_prompt(book_name: str, chapter_count: int) -> str:
    """Build a book-level prompt without needing any config file."""
    return _inline_book_prompt(book_name, chapter_count)
