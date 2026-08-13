"""Tests for rate-limit handling, throttling, and resumable downloads."""

from pathlib import Path

import pytest
import requests


def _resp(mocker, status, headers=None, verses=None):
    """Build a mock response with the given status."""
    r = mocker.MagicMock()
    r.status_code = status
    r.headers = headers or {}
    if status == 200:
        r.raise_for_status.return_value = None
        r.json.return_value = {
            "verses": verses or [{"verse": 1, "text": "text"}],
        }
    else:
        err = requests.HTTPError(f"{status}")
        err.response = r
        r.raise_for_status.side_effect = err
    return r


class TestRetryableStatuses:
    """429 and 5xx are retried; 4xx client errors are not."""

    def test_429_is_retryable(self):
        from bible_study.api import RETRYABLE_STATUS
        assert 429 in RETRYABLE_STATUS

    def test_server_errors_are_retryable(self):
        from bible_study.api import RETRYABLE_STATUS
        for code in (500, 502, 503, 504):
            assert code in RETRYABLE_STATUS

    def test_404_is_not_retryable(self, mocker):
        from bible_study.api import fetch_chapter
        mocker.patch("bible_study.api.time.sleep")
        mock_get = mocker.patch(
            "bible_study.api.requests.get", return_value=_resp(mocker, 404),
        )
        with pytest.raises(requests.HTTPError):
            fetch_chapter("Genesis", 1)
        assert mock_get.call_count == 1

    def test_429_then_success(self, mocker):
        from bible_study.api import fetch_chapter
        mocker.patch("bible_study.api.time.sleep")
        mock_get = mocker.patch(
            "bible_study.api.requests.get",
            side_effect=[_resp(mocker, 429), _resp(mocker, 200)],
        )
        assert len(fetch_chapter("Genesis", 1)) == 1
        assert mock_get.call_count == 2

    def test_429_raises_after_max_retries(self, mocker):
        from bible_study.api import MAX_RETRIES, fetch_chapter
        mocker.patch("bible_study.api.time.sleep")
        mock_get = mocker.patch(
            "bible_study.api.requests.get", return_value=_resp(mocker, 429),
        )
        with pytest.raises(requests.HTTPError):
            fetch_chapter("Genesis", 1)
        assert mock_get.call_count == MAX_RETRIES


class TestRetryDelay:
    """Backoff honours Retry-After and treats 429 differently from 5xx."""

    def test_honours_retry_after_header(self, mocker):
        from bible_study.api import _retry_delay
        assert _retry_delay(_resp(mocker, 429, {"Retry-After": "12"}), 1) == 12.0

    def test_ignores_unparseable_retry_after(self, mocker):
        from bible_study.api import RATE_LIMIT_BACKOFF, _retry_delay
        r = _resp(mocker, 429, {"Retry-After": "soon"})
        assert _retry_delay(r, 1) == RATE_LIMIT_BACKOFF

    def test_rate_limit_uses_long_backoff(self, mocker):
        from bible_study.api import RATE_LIMIT_BACKOFF, _retry_delay
        assert _retry_delay(_resp(mocker, 429), 2) == RATE_LIMIT_BACKOFF * 2

    def test_server_error_uses_exponential_backoff(self, mocker):
        from bible_study.api import RETRY_DELAY, _retry_delay
        assert _retry_delay(_resp(mocker, 500), 1) == RETRY_DELAY
        assert _retry_delay(_resp(mocker, 500), 3) == RETRY_DELAY * 4

    def test_rate_limit_backoff_exceeds_server_backoff(self):
        from bible_study.api import RATE_LIMIT_BACKOFF, RETRY_DELAY
        assert RATE_LIMIT_BACKOFF > RETRY_DELAY * 8


class TestThrottling:
    """Sleep between live fetches, but never between cache hits."""

    def test_delay_is_at_least_two_seconds(self):
        from bible_study.api import BETWEEN_REQUEST_DELAY
        assert BETWEEN_REQUEST_DELAY >= 2.0

    def test_sleeps_after_each_live_fetch(self, tmp_path, mocker):
        from bible_study.api import download_all
        mocker.patch(
            "bible_study.api.requests.get", return_value=_resp(mocker, 200),
        )
        sleep = mocker.patch("bible_study.api.time.sleep")
        download_all(book_names_list=["Jude"], cache_dir=tmp_path / "c")
        assert sleep.call_count == 1

    def test_does_not_sleep_on_cache_hits(self, tmp_path, mocker):
        from bible_study.api import download_all
        mocker.patch(
            "bible_study.api.requests.get", return_value=_resp(mocker, 200),
        )
        cache = tmp_path / "c"
        mocker.patch("bible_study.api.time.sleep")
        download_all(book_names_list=["Jude"], cache_dir=cache)

        # Second pass reads from cache, so it must not throttle at all.
        sleep = mocker.patch("bible_study.api.time.sleep")
        results = download_all(book_names_list=["Jude"], cache_dir=cache)
        assert results == [("Jude", 1)]
        sleep.assert_not_called()


class TestRateLimitAbort:
    """A 429 stops the run instead of failing every remaining chapter."""

    def test_stops_on_rate_limit(self, tmp_path, mocker, capsys):
        from bible_study.api import download_all
        mocker.patch("bible_study.api.time.sleep")
        err = requests.HTTPError("429")
        err.response = _resp(mocker, 429)
        mocker.patch("bible_study.api.save_chapter", side_effect=err)
        results = download_all(
            book_names_list=["Genesis"], cache_dir=tmp_path / "c",
        )
        assert results == []
        out = capsys.readouterr().out
        assert "Rate limited" in out
        assert "resume" in out

    def test_keeps_chapters_fetched_before_the_limit(self, tmp_path, mocker):
        from bible_study.api import download_all
        mocker.patch("bible_study.api.time.sleep")
        err = requests.HTTPError("429")
        err.response = _resp(mocker, 429)
        mocker.patch(
            "bible_study.api.save_chapter",
            side_effect=[
                {"verses": [{"verse": 1, "text": "a"}], "_fetched": True},
                {"verses": [{"verse": 1, "text": "b"}], "_fetched": True},
                err,
            ],
        )
        results = download_all(
            book_names_list=["Genesis"], cache_dir=tmp_path / "c",
        )
        assert results == [("Genesis", 1), ("Genesis", 2)]

    def test_non_rate_limit_errors_continue(self, tmp_path, mocker, capsys):
        from bible_study.api import download_all
        mocker.patch("bible_study.api.time.sleep")
        mocker.patch(
            "bible_study.api.save_chapter",
            side_effect=RuntimeError("transient blip"),
        )
        results = download_all(
            book_names_list=["Jude", "Obadiah"], cache_dir=tmp_path / "c",
        )
        assert results == []
        out = capsys.readouterr().out
        assert "Rate limited" not in out
        assert "2 chapter(s) failed" in out

    def test_failure_summary_truncates(self, tmp_path, mocker, capsys):
        from bible_study.api import download_all
        mocker.patch("bible_study.api.time.sleep")
        mocker.patch(
            "bible_study.api.save_chapter", side_effect=RuntimeError("nope"),
        )
        download_all(book_names_list=["Genesis"], cache_dir=tmp_path / "c")
        out = capsys.readouterr().out
        assert "50 chapter(s) failed" in out
        assert "and 40 more" in out

    def test_is_rate_limit_detects_429(self, mocker):
        from bible_study.api import _is_rate_limit
        err = requests.HTTPError("429")
        err.response = _resp(mocker, 429)
        assert _is_rate_limit(err) is True

    def test_is_rate_limit_ignores_other_errors(self, mocker):
        from bible_study.api import _is_rate_limit
        err = requests.HTTPError("500")
        err.response = _resp(mocker, 500)
        assert _is_rate_limit(err) is False
        assert _is_rate_limit(RuntimeError("boom")) is False


class TestResume:
    """A partial download resumes without refetching or corrupting data."""

    def test_resume_skips_cached_and_fetches_rest(self, tmp_path, mocker):
        from bible_study.api import download_all
        from bible_study.db import init_db, verse_count
        cache = tmp_path / "c"
        db = tmp_path / "bible.db"
        init_db(db)
        mocker.patch("bible_study.api.time.sleep")
        mocker.patch(
            "bible_study.api.get_chapters_for_book", return_value=[1, 2, 3],
        )

        # First pass: only chapter 1 succeeds.
        err = requests.HTTPError("500")
        err.response = _resp(mocker, 500)
        get = mocker.patch(
            "bible_study.api.requests.get",
            side_effect=[_resp(mocker, 200), err, err, err, err, err, err, err,
                         err, err, err],
        )
        download_all(book_names_list=["Genesis"], cache_dir=cache, db_path=db)
        assert verse_count(db) == 1

        # Second pass: chapter 1 comes from cache, 2 and 3 now succeed.
        get.side_effect = [_resp(mocker, 200), _resp(mocker, 200)]
        results = download_all(
            book_names_list=["Genesis"], cache_dir=cache, db_path=db,
        )
        assert results == [("Genesis", 1), ("Genesis", 2), ("Genesis", 3)]
        assert verse_count(db) == 3

    def test_reingests_cached_chapters_into_empty_db(self, tmp_path, mocker):
        from bible_study.api import download_all
        from bible_study.db import init_db, verse_count
        cache = tmp_path / "c"
        mocker.patch("bible_study.api.time.sleep")
        mocker.patch("bible_study.api.get_chapters_for_book", return_value=[1])
        mocker.patch(
            "bible_study.api.requests.get", return_value=_resp(mocker, 200),
        )
        download_all(book_names_list=["Genesis"], cache_dir=cache)

        # A fresh DB is repopulated from cache without any network calls.
        db = tmp_path / "new.db"
        init_db(db)
        get = mocker.patch("bible_study.api.requests.get")
        download_all(book_names_list=["Genesis"], cache_dir=cache, db_path=db)
        assert verse_count(db) == 1
        get.assert_not_called()

    def test_upsert_is_idempotent_across_runs(self, tmp_path, mocker):
        from bible_study.api import download_all
        from bible_study.db import init_db, verse_count
        cache = tmp_path / "c"
        db = tmp_path / "bible.db"
        init_db(db)
        mocker.patch("bible_study.api.time.sleep")
        mocker.patch("bible_study.api.get_chapters_for_book", return_value=[1])
        mocker.patch(
            "bible_study.api.requests.get", return_value=_resp(mocker, 200),
        )
        for _ in range(3):
            download_all(
                book_names_list=["Genesis"], cache_dir=cache, db_path=db,
            )
        assert verse_count(db) == 1
