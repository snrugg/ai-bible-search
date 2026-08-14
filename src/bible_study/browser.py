"""Lightweight SQLite-backed HTTP server for Bible Study viewer."""

from __future__ import annotations

import webbrowser
from html import escape
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

_DEFAULT_PORT = 8080
_DEFAULT_DB = Path("data/bible.db")

_STYLE = """
body { max-width: 46rem; margin: 2rem auto; padding: 0 1rem;
       font: 16px/1.6 -apple-system, Segoe UI, Roboto, sans-serif; color: #222; }
a { color: #1a5490; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { margin-bottom: 0.2rem; }
h2 { margin: 1.8rem 0 0.4rem; font-size: 1.1rem; color: #555;
     border-bottom: 1px solid #e2e2e2; padding-bottom: 0.2rem; }
nav { margin: 1rem 0; font-size: 0.9rem; }
.totals { color: #555; margin: 0.2rem 0 0; }
table.books { width: 100%; border-collapse: collapse; }
table.books th { text-align: right; font-size: 0.75rem; text-transform: uppercase;
                 letter-spacing: 0.04em; color: #888; font-weight: 600;
                 padding: 0.2rem 0.5rem; }
table.books th:first-child { text-align: left; }
table.books td { padding: 0.25rem 0.5rem; text-align: right;
                 border-top: 1px solid #f0f0f0; white-space: nowrap; }
table.books td:first-child { text-align: left; width: 99%; }
table.books tr:hover td { background: #fafaf7; }
.ok { color: #2b7a2b; }
.chapters a { display: inline-block; min-width: 2.4rem; padding: 0.25rem 0.4rem;
              margin: 0.1rem; text-align: center; border: 1px solid #ccd; }
.chapters a.done { background: #eef6ee; border-color: #9c9; }
.summary { background: #f7f7f4; border-left: 3px solid #9ab; padding: 0.8rem 1rem;
           margin: 1rem 0; white-space: pre-wrap; }
.verse { margin: 0.4rem 0; }
.verse b { color: #888; font-size: 0.8rem; vertical-align: super; }
.muted { color: #777; }
"""


def serve(port: int = _DEFAULT_PORT, db_path: Path | None = None) -> None:
    """Start the SQLite-backed HTTP browser server."""
    db_path = Path(db_path) if db_path is not None else _DEFAULT_DB

    def handler(*args, **kwargs):
        return _SQLiteHandler(*args, db_path=db_path, **kwargs)

    server = HTTPServer(("127.0.0.1", port), handler)
    webbrowser.open(f"http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def _slug(book_name: str) -> str:
    """Turn a book name into a URL path segment (``1 Samuel`` -> ``1-samuel``)."""
    return book_name.lower().replace(" ", "-")


def _book_from_slug(slug: str):
    """Resolve a URL segment back to a book dict, or None if unknown."""
    from bible_study.indexer import get_book as _get_book
    return _get_book(unquote(slug).replace("-", " ").strip())


class _SQLiteHandler(SimpleHTTPRequestHandler):
    """Handle HTTP requests using SQLite data."""

    def __init__(self, *args, db_path=None, **kwargs):
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DB
        super().__init__(*args, **kwargs)

    # -- Routing ---------------------------------------------------------- #

    def do_GET(self):
        """Route incoming GET requests."""
        path = urlparse(self.path).path.strip("/")
        if not path:
            self._book_list()
        elif path.startswith("book/"):
            parts = [p for p in path[5:].split("/") if p]
            if len(parts) == 1:
                self._chapter_list(parts[0])
            elif len(parts) == 2 and parts[1].isdigit():
                self._chapter_view(parts[0], int(parts[1]))
            else:
                self._send_error(404, f"No such route: /{path}")
        else:
            self._fallback_page(path)

    # -- Pages ------------------------------------------------------------ #

    def _book_list(self):
        """Show all 66 books with per-book download and summary progress."""
        from bible_study.indexer import iter_books, total_chapters

        if not self.db_path.exists():
            self._send_html(self._page(
                "Bible Study",
                "<p class='muted'>No database found at "
                f"{escape(str(self.db_path))} -- run <code>bible-study init</code> "
                "first.</p>",
            ))
            return

        summarized = self._summary_counts()
        stored = self._verse_counts()
        with_book_summary = self._book_summary_names()

        parts = []
        totals = (sum(stored.values()), sum(summarized.values()),
                  len(with_book_summary))
        parts.append(
            "<p class='totals'>"
            f"{totals[0]}/{total_chapters()} chapters downloaded &middot; "
            f"{totals[1]} summarised &middot; "
            f"{totals[2]}/66 book summaries</p>",
        )

        for testament, heading in (("OT", "Old Testament"),
                                   ("NT", "New Testament")):
            parts.append(f"<h2>{heading}</h2>")
            parts.append(
                "<table class='books'><thead><tr><th>Book</th>"
                "<th>Downloaded</th><th>Summarised</th><th>Book summary</th>"
                "</tr></thead><tbody>",
            )
            for book in iter_books(testament):
                parts.append(self._book_row(book, stored, summarized,
                                            with_book_summary))
            parts.append("</tbody></table>")

        self._send_html(self._page("Bible Study", "".join(parts)))

    def _book_row(self, book, stored, summarized, with_book_summary):
        """Render one book's row for the index table."""
        name = book["name"]
        total = book["chapter_count"]
        have = stored.get(name, 0)
        done = summarized.get(name, 0)

        if have >= total:
            downloaded = "<span class='ok'>&#9989;</span>"
        elif have:
            downloaded = f"{have}/{total}"
        else:
            downloaded = "<span class='muted'>&mdash;</span>"

        if done >= total:
            summaries = f"<span class='ok'>&#9989; {done}/{total}</span>"
        elif done:
            summaries = f"{done}/{total}"
        else:
            summaries = f"<span class='muted'>0/{total}</span>"

        book_summary = (
            "<span class='ok'>&#9989;</span>" if name in with_book_summary
            else "<span class='muted'>&mdash;</span>"
        )
        return (
            f"<tr><td><a href='/book/{_slug(name)}'>{escape(name)}</a></td>"
            f"<td>{downloaded}</td><td>{summaries}</td>"
            f"<td>{book_summary}</td></tr>"
        )

    def _chapter_list(self, slug):
        """Show a book's chapters, plus its aggregate summary when present."""
        book = _book_from_slug(slug)
        if book is None:
            self._send_error(404, f"Book not found: {unquote(slug)}")
            return

        from bible_study import db as _db
        name = book["name"]
        has_db = self.db_path.exists()
        summarized = set()
        stored = set()
        book_summary = None
        if has_db:
            summarized = {
                cs["chapter"]
                for cs in _db.get_chapter_summaries_for_book(self.db_path, name)
            }
            stored = set(_db.get_stored_chapters(self.db_path, name))
            book_summary = _db.get_book_summary(self.db_path, name)

        parts = ["<nav><a href='/'>&larr; All books</a></nav>"]
        if book_summary:
            parts.append(f"<div class='summary'>{escape(book_summary)}</div>")
        total = book["chapter_count"]
        tick = " <span class='ok'>&#9989;</span>" if len(summarized) >= total else ""
        parts.append(f"<p class='muted'>{len(summarized)} of {total} chapters "
                     f"summarised, {len(stored)} of {total} downloaded.{tick}</p>")
        parts.append("<div class='chapters'>")
        for chap in range(1, book["chapter_count"] + 1):
            cls = " class='done'" if chap in summarized else ""
            title = ("summarised" if chap in summarized
                     else "text only" if chap in stored else "no data")
            parts.append(
                f"<a href='/book/{_slug(name)}/{chap}'{cls} "
                f"title='{title}'>{chap}</a>",
            )
        parts.append("</div>")
        self._send_html(self._page(name, "".join(parts)))

    def _chapter_view(self, slug, chapter):
        """Show a chapter's summary and its KJV verses."""
        book = _book_from_slug(slug)
        if book is None:
            self._send_error(404, f"Book not found: {unquote(slug)}")
            return
        name = book["name"]
        if not 1 <= chapter <= book["chapter_count"]:
            self._send_error(
                404, f"{name} has no chapter {chapter} "
                     f"(1-{book['chapter_count']})",
            )
            return

        summary = None
        verses = []
        if self.db_path.exists():
            from bible_study import db as _db
            summary = _db.get_summary(self.db_path, name, chapter)
            verses = _db.get_verses(self.db_path, name, chapter)

        parts = [
            f"<nav><a href='/'>All books</a> &middot; "
            f"<a href='/book/{_slug(name)}'>{escape(name)}</a></nav>",
        ]
        if summary:
            parts.append(f"<div class='summary'>{escape(summary)}</div>")
        else:
            parts.append("<p class='muted'>No summary yet -- "
                         "run <code>bible-study summarize</code>.</p>")

        if verses:
            parts.append("<h2>King James Version</h2>")
            for v in verses:
                parts.append(
                    f"<p class='verse'><b>{v['verse']}</b> "
                    f"{escape(v['text'].strip())}</p>",
                )
        else:
            parts.append("<p class='muted'>No verse text stored for this "
                         "chapter.</p>")

        links = []
        if chapter > 1:
            links.append(
                f"<a href='/book/{_slug(name)}/{chapter - 1}'>&larr; "
                f"Chapter {chapter - 1}</a>",
            )
        if chapter < book["chapter_count"]:
            links.append(
                f"<a href='/book/{_slug(name)}/{chapter + 1}'>Chapter "
                f"{chapter + 1} &rarr;</a>",
            )
        parts.append(f"<nav>{' &middot; '.join(links)}</nav>")

        self._send_html(self._page(f"{name} {chapter}", "".join(parts)))

    def _fallback_page(self, path):
        """Show a 404 for unknown routes."""
        self._send_error(404, f"No such route: /{unquote(path)}")

    # -- Helpers ---------------------------------------------------------- #

    def _counts(self, sql):
        """Run a ``(book_name, count)`` query and return it as a dict."""
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            return dict(conn.execute(sql).fetchall())

    def _summary_counts(self):
        """Return ``{book_name: summarised_chapter_count}``."""
        return self._counts(
            "SELECT book_name, COUNT(*) FROM chapter_summaries "
            "GROUP BY book_name",
        )

    def _verse_counts(self):
        """Return ``{book_name: chapter_count_with_verses}``."""
        return self._counts(
            "SELECT book_name, COUNT(DISTINCT chapter) FROM verses "
            "GROUP BY book_name",
        )

    def _book_summary_names(self):
        """Return the set of book names that have an aggregate summary."""
        from bible_study import db as _db
        return set(_db.get_saved_books(self.db_path))

    def _page(self, title, body):
        """Wrap *body* in a complete HTML document."""
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
            f"<body><h1>{escape(title)}</h1>{body}</body></html>"
        )

    def _send_html(self, html, code=200):
        """Send a complete HTTP response with *html* as the body.

        Every response must go through here -- writing straight to
        ``wfile`` skips the status line and headers, which makes clients
        fall back to HTTP/0.9 (``curl`` refuses such responses outright).
        """
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        """Helper to send a plain-text error page."""
        body = message.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        """Suppress request logs during tests."""
        pass


# Public alias so tests can reference the class name directly
Handler = _SQLiteHandler

if __name__ == "__main__":
    serve()
