"""Lightweight SQLite-backed HTTP server for Bible Study viewer."""

from __future__ import annotations

import sqlite3
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import webbrowser

_DEFAULT_PORT = 8080


def serve(port: int = _DEFAULT_PORT) -> None:
    """Start the SQLite-backed HTTP browser server."""
    db_path = Path("data/bible.db")
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
    else:
        conn = None
    handler = _SQLiteHandler(conn)
    server = HTTPServer(("127.0.0.1", port), handler)
    webbrowser.open(f"http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


class _SQLiteHandler(SimpleHTTPRequestHandler):
    """Handle HTTP requests using SQLite data."""

    def __init__(self, *args, conn=None, **kwargs):
        self.conn = conn
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Route incoming GET requests."""
        path = urlparse(self.path).path.strip("/")
        if not path:
            self._book_list()
        elif path.startswith("book/"):
            book_name = path[5:]  # strip "book/" prefix
            self._chapter_view(book_name)
        else:
            self._fallback_page(path)

    def _book_list(self):
        """Show the list of all 66 Bible books."""
        from bible_study.indexer import book_names
        parts = [f"<h1>Bible Study</h1><ul>"]
        for name in book_names():
            href = f"/book/{name.lower()}"
            parts.append(f'<li><a href="{href}">{name}</a></li>')
        parts.append("</ul>")
        self.wfile.write("".join(parts).encode("utf-8"))

    def _chapter_view(self, book_name):
        """Show verses + summary for a chapter."""
        from bible_study.indexer import get_book as _get_book
        book_info = _get_book(book_name.title())
        if book_info is None:
            self._send_error(404, f"Book not found: {book_name}")
            return
        html = f"<h1>{book_info['name']}</h1>"
        html += "<p>Loading chapters...</p>"
        self.wfile.write(html.encode("utf-8"))

    def _fallback_page(self, path):
        """Show a placeholder for unknown routes."""
        html = f"<h1>{path}</h1><p>Loading...</p>"
        self.wfile.write(html.encode("utf-8"))

    def _send_error(self, code, message):
        """Helper to send an HTML error page."""
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))

    def log_message(self, fmt, *args):
        """Suppress request logs during tests."""
        pass


# Public alias so tests can reference the class name directly
Handler = _SQLiteHandler

if __name__ == "__main__":
    serve()
