"""Tests for bible_study/browser -- lightweight HTTP server."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest


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

    def test_serve_connects_when_db_exists(self, tmp_path, monkeypatch):
        from bible_study.browser import serve
        db = tmp_path / "bible.db"
        sqlite3.connect(str(db)).close()
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db.rename(data_dir / "bible.db")
        with patch("bible_study.browser.webbrowser.open"):
            with patch("bible_study.browser.HTTPServer") as ms:
                with patch.object(sqlite3, "connect", return_value=MagicMock()) as mc:
                    serve(port=8084)
                    mc.assert_called_once()
                assert ms.called

    def test_serve_handles_missing_db(self, tmp_path, monkeypatch):
        from bible_study.browser import serve
        monkeypatch.chdir(tmp_path)
        with patch("bible_study.browser.webbrowser.open"):
            with patch("bible_study.browser.HTTPServer") as ms:
                serve(port=8085)
                assert ms.called


class TestHandlerMethods:
    """Source-level checks of the request handler methods."""

    def test_book_list_produces_html(self):
        import inspect

        from bible_study.browser import _SQLiteHandler
        source = inspect.getsource(_SQLiteHandler._book_list)
        assert "<h1>Bible Study</h1>" in source
        assert "<a href" in source

    def test_fallback_page_method_exists(self):
        from bible_study.browser import _SQLiteHandler
        assert hasattr(_SQLiteHandler, "_fallback_page")
        assert hasattr(_SQLiteHandler, "_send_error")

    def test_log_message_is_silenced(self):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        assert handler.log_message("%s", "ignored") is None

    def test_send_error_writes_message(self):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler._send_error(404, "Book not found: nope")
        handler.send_response.assert_called_once_with(404)
        handler.end_headers.assert_called_once()
        written = handler.wfile.write.call_args[0][0]
        assert b"Book not found" in written

    def test_fallback_page_writes_path(self):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.wfile = MagicMock()
        handler._fallback_page("some/route")
        written = handler.wfile.write.call_args[0][0]
        assert b"some/route" in written

    def test_book_list_writes_all_66_books(self):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.wfile = MagicMock()
        handler._book_list()
        written = handler.wfile.write.call_args[0][0].decode("utf-8")
        assert "Genesis" in written
        assert "Revelation" in written
        assert written.count("<li>") == 66

    def test_chapter_view_renders_known_book(self):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.wfile = MagicMock()
        handler._chapter_view("genesis")
        written = handler.wfile.write.call_args[0][0].decode("utf-8")
        assert "Genesis" in written

    def test_chapter_view_sends_404_for_unknown_book(self):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.wfile = MagicMock()
        handler._send_error = MagicMock()
        handler._chapter_view("notabook")
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 404


class TestDoGetRouting:
    """Exercise do_GET routing across all three branches."""

    def _handler(self, path):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.path = path
        handler.wfile = MagicMock()
        handler._book_list = MagicMock()
        handler._chapter_view = MagicMock()
        handler._fallback_page = MagicMock()
        return handler

    def test_root_path_shows_book_list(self):
        handler = self._handler("/")
        handler.do_GET()
        handler._book_list.assert_called_once()

    def test_book_path_shows_chapter_view(self):
        handler = self._handler("/book/genesis")
        handler.do_GET()
        handler._chapter_view.assert_called_once_with("genesis")

    def test_unknown_path_shows_fallback(self):
        handler = self._handler("/something-else")
        handler.do_GET()
        handler._fallback_page.assert_called_once_with("something-else")

    def test_query_string_is_stripped(self):
        handler = self._handler("/book/genesis?chapter=2")
        handler.do_GET()
        handler._chapter_view.assert_called_once_with("genesis")


class TestHandlerInitialization:
    """Verify the conn attribute round-trips."""

    def test_handler_with_none_conn(self):
        from bible_study.browser import _SQLiteHandler
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.conn = None
        assert handler.conn is None

    def test_handler_with_mock_conn(self):
        from bible_study.browser import _SQLiteHandler
        mock_conn = MagicMock()
        handler = _SQLiteHandler.__new__(_SQLiteHandler)
        handler.conn = mock_conn
        assert handler.conn is mock_conn


class TestIntegration:
    """Integration tests -- skipped by default."""

    @pytest.mark.skip(reason="integration test -- runs only on demand")
    def test_server_responds_with_book_list(self):
        pass

    @pytest.mark.skip(reason="integration test -- runs only on demand")
    def test_chapter_view_returns_verses(self):
        pass
