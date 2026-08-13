"""Bible API client -- fetch KJV chapter text from bible-api.com."""

from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import requests

BASE_URL = "https://bible-api.com"
DEFAULT_CACHE_DIR: Path = Path("data/api-cache")
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1.0
BETWEEN_REQUEST_DELAY = 0.5


def fetch_chapter(
    book_name: str, chapter_num: int, timeout: int = REQUEST_TIMEOUT,
) -> list[dict[str, Any]]:
    """Fetch a single chapter from bible-api.com and return parsed verses."""
    slug = quote(book_name) + f"%20{chapter_num}"
    url = f"{BASE_URL}/{slug}"
    params = {"format": "json", "translation": "kjv"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if getattr(resp, "status_code", 0) >= 500 and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
                continue
            raise

        data = resp.json()
        return _parse_verses(data)

    msg = f"Failed after {MAX_RETRIES} retries for {book_name} chapter {chapter_num}"
    raise RuntimeError(msg)    # pragma: no cover


def save_chapter(
    book_name: str, chapter_num: int, cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch a chapter and save the raw JSON response to local cache."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    key = _cache_key(book_name, chapter_num, base_dir=cache_dir)

    if key.exists():
        with key.open("r") as fh:
            return cast(dict[str, Any], json.load(fh))

    data = fetch_chapter(book_name, chapter_num)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with key.open("w") as fh:
        json.dump(data, fh)
    return {"verses": data}


def download_all(
    book_names_list: list[str] | None = None,
    cache_dir: Path | None = None,
) -> list[tuple[str, int]]:
    """Download every chapter across books to the API cache."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if book_names_list is None:
        from bible_study.indexer import book_names as _bn
        book_names_list = _bn()

    newly_fetched: list[tuple[str, int]] = []

    for bname in book_names_list:
        last_cached = _last_cached_chapter(bname, cache_dir)
        chapters = get_chapters_for_book(bname)
        for chap in chapters:
            try:
                save_chapter(bname, chap, cache_dir=cache_dir)
                newly_fetched.append((bname, chap))
            except Exception:   # noqa: BLE001
                print(f"Warning: failed to fetch {bname} {chap}")

        time.sleep(BETWEEN_REQUEST_DELAY)

    return newly_fetched


def _cache_key(
    book_name: str, chapter_num: int, *, base_dir: Path | None = None,
) -> Path:
    """Return the cache-file path for a (book, chapter) pair."""
    safe = book_name.lower().replace(" ", "-")
    return (base_dir or DEFAULT_CACHE_DIR) / f"{safe}-{chapter_num}.json"


def _last_cached_chapter(book_name: str, cache_dir: Path) -> int | None:
    """Return the highest cached chapter number, or None."""
    safe = book_name.lower().replace(" ", "-")
    prefix = f"{safe}-"
    best: int | None = None
    try:
        for p in cache_dir.iterdir():
            if p.name.startswith(prefix) and p.name.endswith(".json"):
                try:
                    num = int(p.name[len(prefix):-5])
                    if best is None or num > best:
                        best = num
                except ValueError:
                    continue
    except OSError:
        return None
    return best


def _parse_verses(data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract a clean verses list from the bible-api.com response."""
    verses = data.get("verses", [])
    return [{"verse": v["verse"], "text": v["text"].strip()} for v in verses]


def get_chapters_for_book(book_name: str) -> list[int]:
    """Return a list of chapter numbers for book."""
    from bible_study.indexer import get_chapters as _get_chapters
    try:
        return _get_chapters(book_name)
    except ValueError:
        return list(range(1, 51))
