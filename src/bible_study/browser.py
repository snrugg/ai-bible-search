"""Lightweight SQLite-backed HTTP server for Bible Study viewer."""

from __future__ import annotations

import webbrowser
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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
form.search { margin: 1rem 0; display: flex; gap: 0.4rem; }
form.search input { flex: 1; padding: 0.4rem 0.5rem; font-size: 1rem;
                    border: 1px solid #ccd; }
form.search button { padding: 0.4rem 0.9rem; font-size: 1rem; cursor: pointer;
                     border: 1px solid #9ab; background: #f7f7f4; }
.hit { margin: 0.9rem 0; }
.hit .ref { font-weight: 600; }
.hit .dist { color: #999; font-size: 0.8rem; }
.hit .snippet { color: #444; margin: 0.15rem 0 0; }
.tier { color: #888; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.04em; }
"""


def serve(port: int = _DEFAULT_PORT, db_path: Path | None = None) -> None:
    """Start the SQLite-backed HTTP browser server."""
    db_path = Path(db_path) if db_path is not None else _DEFAULT_DB

    def handler(*args, **kwargs):
        return _SQLiteHandler(*args, db_path=db_path, **kwargs)

    # Threading, not plain HTTPServer: an /ask request holds the socket for
    # the whole generate() call -- tens of seconds -- during which a
    # single-threaded server serves nothing at all, not even a favicon.
    # Safe because handlers hold a Path and every read opens its own
    # connection; no sqlite3 connection is ever shared across threads.
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
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


def _query_param(query: str, name: str) -> str:
    """Return the first value of *name* in a raw query string, or ''."""
    return parse_qs(query).get(name, [""])[0].strip()


def _search_form(action: str, query: str, label: str, hint: str = "") -> str:
    """Render a search form that echoes *query* back safely."""
    placeholder = escape(hint or "Search the KJV ...")
    return (
        f"<form class='search' action='{action}' method='get'>"
        f"<input type='text' name='q' value='{escape(query)}' "
        f"placeholder='{placeholder}' autofocus>"
        f"<button type='submit'>{escape(label)}</button></form>"
    )


def _nav() -> str:
    """Render the shared navigation strip."""
    return (
        "<nav><a href='/'>Books</a> &middot; "
        "<a href='/search'>Search</a> &middot; "
        "<a href='/ask'>Ask</a></nav>"
    )


def _hit_link(hit: dict) -> str:
    """Link a search hit back to the page that holds it."""
    slug = _slug(hit["book_name"])
    if hit["tier"] == "book":
        return f"/book/{slug}"
    return f"/book/{slug}/{hit['chapter']}"


_TIER_LABEL = {
    "verse": "passage",
    "chapter": "chapter summary",
    "book": "book summary",
}


class _SQLiteHandler(SimpleHTTPRequestHandler):
    """Handle HTTP requests using SQLite data."""

    def __init__(self, *args, db_path=None, **kwargs):
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DB
        super().__init__(*args, **kwargs)

    # -- Routing ---------------------------------------------------------- #

    def do_GET(self):
        """Route incoming GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.strip("/")
        if not path:
            self._book_list()
        elif path == "search":
            self._search_page(_query_param(parsed.query, "q"))
        elif path == "ask":
            self._ask_page(_query_param(parsed.query, "q"))
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

        parts = [_search_form("/search", "", "Search")]
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

    def _search_page(self, query):
        """Rank indexed passages and summaries against *query*.

        No LLM call, so this doubles as the retrieval-debugging surface:
        if `ask` returns something odd, look here first.
        """
        parts = [
            "<h1>Search</h1>",
            _nav(),
            _search_form("/search", query, "Search"),
        ]
        if not query:
            parts.append("<p class='muted'>Type a phrase or question. "
                         "Results are ranked by meaning, not keyword.</p>")
            self._send_html(self._page("Search", "".join(parts)))
            return

        try:
            hits = self._search_hits(query)
        except Exception as exc:  # noqa: BLE001
            parts.append(f"<p class='muted'>Search unavailable: "
                         f"{escape(str(exc))}</p>")
            self._send_html(self._page("Search", "".join(parts)))
            return

        if not hits:
            parts.append("<p class='muted'>Nothing matched. Run "
                         "<code>bible-study embed</code> if the index has "
                         "not been built.</p>")
        for hit in hits:
            snippet = hit["text"][:280]
            if len(hit["text"]) > 280:
                snippet += " ..."
            parts.append(
                f"<div class='hit'>"
                f"<a class='ref' href='{_hit_link(hit)}'>"
                f"{escape(hit['citation'])}</a> "
                f"<span class='tier'>"
                f"{escape(_TIER_LABEL.get(hit['tier'], hit['tier']))}</span> "
                f"<span class='dist'>{hit['distance']:.3f}</span>"
                f"<p class='snippet'>{escape(snippet)}</p>"
                f"</div>",
            )
        self._send_html(self._page("Search", "".join(parts)))

    def _search_hits(self, query):
        """Embed *query* and return ranked hits across all tiers."""
        from bible_study import vectors as _vec
        return _vec.search(
            self.db_path,
            _vec.embed_query(query),
            {"verse": 8, "chapter": 4, "book": 2},
        )

    def _ask_page(self, query):
        """Answer *query* from the index, with linked sources."""
        parts = [
            "<h1>Ask</h1>",
            _nav(),
            _search_form(
                "/ask", query, "Ask",
                hint="Ask a question about the KJV ...",
            ),
        ]
        if not query:
            parts.append("<p class='muted'>Ask a question in plain English. "
                         "This runs a local model, so an answer takes a few "
                         "seconds.</p>")
            self._send_html(self._page("Ask", "".join(parts)))
            return

        try:
            from bible_study.rag import answer_question
            result = answer_question(query, self.db_path)
        except Exception as exc:  # noqa: BLE001
            parts.append(f"<p class='muted'>Could not answer: "
                         f"{escape(str(exc))}</p>")
            self._send_html(self._page("Ask", "".join(parts)))
            return

        parts.append(f"<div class='summary'>{escape(result['answer'].strip())}"
                     f"</div>")
        parts.append("<h2>Sources</h2>")
        for source in result["sources"]:
            slug = _slug(source["book_name"])
            href = (f"/book/{slug}/{source['chapter']}"
                    if source["chapter"] else f"/book/{slug}")
            parts.append(
                f"<div class='hit'><a class='ref' href='{href}'>"
                f"{escape(source['citation'])}</a></div>",
            )
        if result["dropped"]:
            parts.append(
                f"<p class='muted'>{result['dropped']} more omitted to fit "
                f"the context window.</p>",
            )
        self._send_html(self._page("Ask", "".join(parts)))

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
