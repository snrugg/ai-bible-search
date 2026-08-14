"""Tests for bible_study/browser -- lightweight HTTP server."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bible_study.db import (
    init_db,
    save_book_summary,
    save_summary,
    upsert_verses,
)


@pytest.fixture
def populated_db(tmp_path) -> Path:
    """A database with one summarised chapter and one text-only chapter."""
    db_path = tmp_path / "data" / "bible.db"
    init_db(db_path)
    upsert_verses(db_path, "Genesis", 1, [(1, "In the beginning God created.")])
    upsert_verses(db_path, "Genesis", 2, [(1, "Thus the heavens were finished.")])
    save_summary(db_path, "Genesis", 1, "God creates the heavens and the earth.")
    save_book_summary(db_path, "Genesis", "GEN", "The book of beginnings.")
    # Downloaded but never summarised -- a third display state on the index.
    upsert_verses(db_path, "Exodus", 1, [(1, "Now these are the names.")])
    return db_path


def _handler_for(db_path):
    """Build a handler bound to *db_path* with response plumbing mocked."""
    from bible_study.browser import _SQLiteHandler
    handler = _SQLiteHandler.__new__(_SQLiteHandler)
    handler.db_path = Path(db_path)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    return handler


def _body(handler) -> str:
    """Decode the last body written by a mocked handler."""
    return handler.wfile.write.call_args[0][0].decode("utf-8")


def _row_for(html: str, book_name: str) -> str:
    """Return the index-table row whose first cell links to *book_name*."""
    for row in html.split("<tr>"):
        if f">{book_name}</a>" in row:
            return "<tr>" + row.split("</tr>")[0] + "</tr>"
    raise AssertionError(f"no row for {book_name}")


class TestServer:
    """Structural checks on serve() and the handler class."""

    def test_serve_function_exists(self):
        from bible_study.browser import serve
        assert callable(serve)

    def test_serve_handler_class_exists(self):
        from bible_study.browser import _SQLiteHandler
        assert hasattr(_SQLiteHandler, "do_GET")

    def test_handler_is_subclass_of_simple_request_handler(self):
        from http.server import SimpleHTTPRequestHandler

        from bible_study.browser import _SQLiteHandler
        assert issubclass(_SQLiteHandler, SimpleHTTPRequestHandler)

    def test_handler_public_alias_matches_private_class(self):
        from bible_study.browser import Handler, _SQLiteHandler
        assert Handler is _SQLiteHandler

    def test_serve_accepts_default_port(self):
        import inspect

        from bible_study.browser import serve
        sig = inspect.signature(serve)
        assert sig.parameters["port"].default == 8080


class TestServeMock:
    """Exercise serve() with HTTPServer and webbrowser mocked out."""

    def test_serve_calls_httpserver(self):
        from bible_study.browser import serve
        with patch("bible_study.browser.webbrowser.open"):
            with patch("bible_study.browser.HTTPServer") as ms:
                serve(port=8080)
                assert ms.call_args[0][0] == ("127.0.0.1", 8080)

    def test_serve_opens_browser_with_custom_port(self):
        from bible_study.browser import serve
        with patch("bible_study.browser.webbrowser.open") as mo:
            with patch("bible_study.browser.HTTPServer"):
                serve(port=9000)
                mo.assert_called_once()
                assert "9000" in mo.call_args[0][0]

    def test_serve_uses_default_port_when_no_arg(self):
        from bible_study.browser import serve
        with patch("bible_study.browser.webbrowser.open"):
            with patch("bible_study.browser.HTTPServer") as ms:
                serve()
                assert ms.call_args[0][0] == ("127.0.0.1", 8080)

    def test_serve_calls_serve_forever(self):
        from bible_study.browser import serve
        with patch("bible_study.browser.webbrowser.open"):
            with patch("bible_study.browser.HTTPServer") as ms:
                serve(port=8081)
                ms.return_value.serve_forever.assert_called_once()

    def test_serve_swallows_keyboard_interrupt(self):
        from bible_study.browser import serve
        with patch("bible_study.browser.webbrowser.open"):
            with patch("bible_study.browser.HTTPServer") as ms:
                ms.return_value.serve_forever.side_effect = KeyboardInterrupt
                serve(port=8083)

    def test_serve_passes_custom_db_path_to_handler(self, tmp_path):
        from bible_study.browser import serve
        db = tmp_path / "custom.db"
        with patch("bible_study.browser.webbrowser.open"):
            with patch("bible_study.browser.HTTPServer") as ms:
                serve(port=8084, db_path=db)
                factory = ms.call_args[0][1]
        with patch("bible_study.browser.SimpleHTTPRequestHandler.__init__",
                   return_value=None):
            handler = factory()
        assert handler.db_path == db

    def test_serve_defaults_to_data_bible_db(self):
        from bible_study.browser import serve
        with patch("bible_study.browser.webbrowser.open"):
            with patch("bible_study.browser.HTTPServer") as ms:
                serve(port=8085)
                factory = ms.call_args[0][1]
        with patch("bible_study.browser.SimpleHTTPRequestHandler.__init__",
                   return_value=None):
            handler = factory()
        assert handler.db_path == Path("data/bible.db")


class TestSlugs:
    """Book-name <-> URL-segment round-tripping."""

    def test_slug_lowercases_and_hyphenates(self):
        from bible_study.browser import _slug
        assert _slug("1 Samuel") == "1-samuel"
        assert _slug("Song of Solomon") == "song-of-solomon"

    def test_book_from_slug_resolves_multiword_names(self):
        from bible_study.browser import _book_from_slug
        assert _book_from_slug("song-of-solomon")["name"] == "Song of Solomon"
        assert _book_from_slug("1-samuel")["name"] == "1 Samuel"

    def test_book_from_slug_accepts_encoded_spaces(self):
        from bible_study.browser import _book_from_slug
        assert _book_from_slug("1%20samuel")["name"] == "1 Samuel"

    def test_book_from_slug_returns_none_for_unknown(self):
        from bible_study.browser import _book_from_slug
        assert _book_from_slug("notabook") is None


class TestHandlerMethods:
    """Page rendering against a real SQLite database."""

    def test_log_message_is_silenced(self):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        assert handler.log_message("%s", "ignored") is None

    def test_send_error_writes_message(self, populated_db):
        handler = _handler_for(populated_db)
        handler._send_error(404, "Book not found: nope")
        handler.send_response.assert_called_once_with(404)
        handler.end_headers.assert_called_once()
        assert b"Book not found" in handler.wfile.write.call_args[0][0]

    def test_fallback_page_returns_404(self, populated_db):
        handler = _handler_for(populated_db)
        handler._fallback_page("some/route")
        handler.send_response.assert_called_once_with(404)
        assert b"some/route" in handler.wfile.write.call_args[0][0]

    def test_book_list_writes_all_66_books(self, populated_db):
        handler = _handler_for(populated_db)
        handler._book_list()
        written = _body(handler)
        assert "Genesis" in written
        assert "Revelation" in written
        assert written.count("<tr><td>") == 66

    def test_book_list_splits_the_testaments(self, populated_db):
        handler = _handler_for(populated_db)
        handler._book_list()
        written = _body(handler)
        assert "Old Testament" in written
        assert "New Testament" in written
        assert written.index("Malachi") < written.index("New Testament")

    def test_book_list_shows_overall_totals(self, populated_db):
        handler = _handler_for(populated_db)
        handler._book_list()
        written = _body(handler)
        assert "3/1189 chapters downloaded" in written
        assert "1 summarised" in written
        assert "1/66 book summaries" in written

    def test_book_list_shows_partial_progress(self, populated_db):
        handler = _handler_for(populated_db)
        handler._book_list()
        written = _body(handler)
        # Genesis: 2 of 50 chapters downloaded, 1 of them summarised.
        assert "<td>2/50</td><td>1/50</td>" in written

    def test_book_list_checks_off_fully_downloaded_books(self, tmp_path):
        db_path = tmp_path / "data" / "bible.db"
        init_db(db_path)
        for chap in (1, 2, 3, 4):          # Jonah has exactly 4 chapters
            upsert_verses(db_path, "Jonah", chap, [(1, "text")])
        handler = _handler_for(db_path)
        handler._book_list()
        row = _row_for(_body(handler), "Jonah")
        assert "&#9989;" in row            # downloaded: checkmark, no fraction
        assert "0/4" in row                # summarised: none yet

    def test_book_list_checks_off_fully_summarized_books(self, tmp_path):
        db_path = tmp_path / "data" / "bible.db"
        init_db(db_path)
        for chap in (1, 2, 3, 4):
            upsert_verses(db_path, "Jonah", chap, [(1, "text")])
            save_summary(db_path, "Jonah", chap, "a summary")
        handler = _handler_for(db_path)
        handler._book_list()
        row = _row_for(_body(handler), "Jonah")
        assert row.count("&#9989;") == 2   # downloaded and summarised
        assert "4/4" in row

    def test_book_list_marks_books_with_an_aggregate_summary(self, populated_db):
        handler = _handler_for(populated_db)
        handler._book_list()
        written = _body(handler)
        # Genesis has a book summary; Exodus has verses but no book summary.
        assert _row_for(written, "Genesis").endswith(
            "<td><span class='ok'>&#9989;</span></td></tr>",
        )
        assert _row_for(written, "Exodus").endswith(
            "<td><span class='muted'>&mdash;</span></td></tr>",
        )

    def test_book_list_dashes_undownloaded_books(self, populated_db):
        handler = _handler_for(populated_db)
        handler._book_list()
        row = _row_for(_body(handler), "Revelation")
        assert "&mdash;" in row
        assert "0/22" in row

    def test_book_list_reports_missing_database(self, tmp_path):
        handler = _handler_for(tmp_path / "nope.db")
        handler._book_list()
        assert "No database found" in _body(handler)

    def test_chapter_list_marks_summarized_chapters(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_list("genesis")
        written = _body(handler)
        assert "1 of 50 chapters summarised, 2 of 50 downloaded." in written
        assert "class='done'" in written
        assert "The book of beginnings." in written

    def test_chapter_list_links_every_chapter(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_list("genesis")
        written = _body(handler)
        assert written.count("/book/genesis/") == 50

    def test_chapter_list_works_without_database(self, tmp_path):
        handler = _handler_for(tmp_path / "nope.db")
        handler._chapter_list("genesis")
        assert "0 of 50 chapters summarised" in _body(handler)

    def test_chapter_list_ticks_a_fully_summarized_book(self, tmp_path):
        db_path = tmp_path / "data" / "bible.db"
        init_db(db_path)
        save_summary(db_path, "Obadiah", 1, "Edom's downfall.")  # 1 chapter
        handler = _handler_for(db_path)
        handler._chapter_list("obadiah")
        assert "&#9989;" in _body(handler)

    def test_chapter_list_404_for_unknown_book(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_list("notabook")
        handler.send_response.assert_called_once_with(404)

    def test_chapter_view_renders_summary_and_verses(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_view("genesis", 1)
        written = _body(handler)
        assert "God creates the heavens and the earth." in written
        assert "In the beginning God created." in written

    def test_chapter_view_notes_missing_summary(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_view("genesis", 2)
        written = _body(handler)
        assert "No summary yet" in written
        assert "Thus the heavens were finished." in written

    def test_chapter_view_notes_missing_verses(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_view("genesis", 3)
        assert "No verse text stored" in _body(handler)

    def test_chapter_view_without_database(self, tmp_path):
        handler = _handler_for(tmp_path / "nope.db")
        handler._chapter_view("genesis", 1)
        assert "No verse text stored" in _body(handler)

    def test_chapter_view_has_prev_and_next_links(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_view("genesis", 2)
        written = _body(handler)
        assert "/book/genesis/1" in written
        assert "/book/genesis/3" in written

    def test_chapter_view_omits_prev_on_first_chapter(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_view("genesis", 1)
        assert "Chapter 0" not in _body(handler)

    def test_chapter_view_omits_next_on_last_chapter(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_view("genesis", 50)
        assert "Chapter 51" not in _body(handler)

    def test_chapter_view_404_for_out_of_range_chapter(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_view("genesis", 99)
        handler.send_response.assert_called_once_with(404)
        assert b"no chapter 99" in handler.wfile.write.call_args[0][0]

    def test_chapter_view_404_for_unknown_book(self, populated_db):
        handler = _handler_for(populated_db)
        handler._chapter_view("notabook", 1)
        handler.send_response.assert_called_once_with(404)

    def test_html_in_stored_text_is_escaped(self, tmp_path):
        db_path = tmp_path / "data" / "bible.db"
        init_db(db_path)
        upsert_verses(db_path, "Genesis", 1, [(1, "<script>alert(1)</script>")])
        handler = _handler_for(db_path)
        handler._chapter_view("genesis", 1)
        written = _body(handler)
        assert "<script>alert(1)</script>" not in written
        assert "&lt;script&gt;" in written


class TestResponseHeaders:
    """Every response must carry a status line and headers."""

    def test_send_html_sets_status_and_headers(self, populated_db):
        handler = _handler_for(populated_db)
        handler._send_html("<h1>hi</h1>")
        handler.send_response.assert_called_once_with(200)
        headers = dict(c[0] for c in handler.send_header.call_args_list)
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert headers["Content-Length"] == str(len(b"<h1>hi</h1>"))
        handler.end_headers.assert_called_once()

    def test_send_html_accepts_custom_status(self, populated_db):
        handler = _handler_for(populated_db)
        handler._send_html("<h1>gone</h1>", code=410)
        handler.send_response.assert_called_once_with(410)

    @pytest.mark.parametrize(
        "method, args",
        [("_book_list", ()), ("_chapter_list", ("genesis",)),
         ("_chapter_view", ("genesis", 1))],
    )
    def test_every_page_sends_a_status_line(self, method, args, populated_db):
        handler = _handler_for(populated_db)
        getattr(handler, method)(*args)
        handler.send_response.assert_called_once_with(200)
        handler.end_headers.assert_called_once()


class TestDoGetRouting:
    """Exercise do_GET routing across all branches."""

    def _handler(self, path):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.path = path
        handler.wfile = MagicMock()
        handler._book_list = MagicMock()
        handler._chapter_list = MagicMock()
        handler._chapter_view = MagicMock()
        handler._fallback_page = MagicMock()
        handler._send_error = MagicMock()
        return handler

    def test_root_path_shows_book_list(self):
        handler = self._handler("/")
        handler.do_GET()
        handler._book_list.assert_called_once()

    def test_book_path_shows_chapter_list(self):
        handler = self._handler("/book/genesis")
        handler.do_GET()
        handler._chapter_list.assert_called_once_with("genesis")

    def test_chapter_path_shows_chapter_view(self):
        handler = self._handler("/book/genesis/3")
        handler.do_GET()
        handler._chapter_view.assert_called_once_with("genesis", 3)

    def test_trailing_slash_still_lists_chapters(self):
        handler = self._handler("/book/genesis/")
        handler.do_GET()
        handler._chapter_list.assert_called_once_with("genesis")

    def test_non_numeric_chapter_is_rejected(self):
        handler = self._handler("/book/genesis/three")
        handler.do_GET()
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 404

    def test_too_many_segments_are_rejected(self):
        handler = self._handler("/book/genesis/1/2")
        handler.do_GET()
        handler._send_error.assert_called_once()

    def test_unknown_path_shows_fallback(self):
        handler = self._handler("/something-else")
        handler.do_GET()
        handler._fallback_page.assert_called_once_with("something-else")

    def test_query_string_is_stripped(self):
        handler = self._handler("/book/genesis?chapter=2")
        handler.do_GET()
        handler._chapter_list.assert_called_once_with("genesis")


class TestOverTheWire:
    """Drive a real HTTPServer on a loopback socket."""

    @pytest.fixture
    def server_port(self, populated_db):
        import threading
        from http.server import HTTPServer

        from bible_study.browser import _SQLiteHandler

        def handler(*args, **kwargs):
            return _SQLiteHandler(*args, db_path=populated_db, **kwargs)

        server = HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def _get(self, port, path):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, resp.getheader("Content-Type"), resp.read()
        finally:
            conn.close()

    def test_root_returns_parseable_http_response(self, server_port):
        status, ctype, body = self._get(server_port, "/")
        assert status == 200
        assert ctype == "text/html; charset=utf-8"
        assert b"Genesis" in body

    def test_book_route_lists_chapters(self, server_port):
        status, _ctype, body = self._get(server_port, "/book/genesis")
        assert status == 200
        assert b"/book/genesis/50" in body

    def test_chapter_route_serves_summary_and_verses(self, server_port):
        status, _ctype, body = self._get(server_port, "/book/genesis/1")
        assert status == 200
        assert b"God creates the heavens and the earth." in body
        assert b"In the beginning God created." in body

    def test_unknown_book_returns_404(self, server_port):
        status, _ctype, body = self._get(server_port, "/book/notabook")
        assert status == 404
        assert b"Book not found" in body

    def test_unknown_route_returns_404(self, server_port):
        status, _ctype, _body = self._get(server_port, "/nope")
        assert status == 404


class TestIntegration:
    """Integration tests -- skipped by default."""

    @pytest.mark.skip(reason="integration test -- runs only on demand")
    def test_server_responds_with_book_list(self):
        pass

    @pytest.mark.skip(reason="integration test -- runs only on demand")
    def test_chapter_view_returns_verses(self):
        pass
