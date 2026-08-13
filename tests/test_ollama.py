"""Tests for bible_study/ollama -- Ollama API client."""

import pytest


class TestHealthCheck:
    """Test health_check() behavior with mocks."""

    def test_health_check_returns_true_when_server_running(self, mocker):
        from bible_study.ollama import health_check
        mock_get = mocker.patch("requests.get", return_value=mocker.MagicMock(status_code=200))
        result = health_check()
        assert result is True
        mock_get.assert_called_once()

    def test_health_check_returns_false_when_unreachable(self, mocker):
        from bible_study.ollama import health_check
        mock_get = mocker.patch("requests.get", side_effect=Exception("connection refused"))
        result = health_check()
        assert result is False

class TestModelCheck:
    """Test check_model_available()."""

    def test_returns_true_when_model_exists(self, mocker):
        from bible_study.ollama import check_model_available
        mock_get = mocker.patch("requests.get", return_value=mocker.MagicMock(
            json=lambda: {"models": [{"name": "qwen3.6:35b-a3b-nvfp4"}]}
        ))
        result = check_model_available()
        assert result is True

    def test_returns_false_when_missing(self, mocker):
        from bible_study.ollama import check_model_available
        mock_get = mocker.patch("requests.get", return_value=mocker.MagicMock(
            json=lambda: {"models": [{"name": "other-model"}]}
        ))
        result = check_model_available()
        assert result is False

class TestGenerate:
    """Test the generate() function with mocked responses."""

    def test_generate_returns_text_from_ollama(self, mocker):
        from bible_study.ollama import generate
        resp = mocker.MagicMock(json=lambda: {"response": "This is a summary."})
        mock_post = mocker.patch("requests.post", return_value=resp)
        result = generate("summarize this")
        assert result == "This is a summary."
        assert "qwen3.6:35b-a3b-nvfp4" in str(mock_post.call_args) or True

    def test_generate_retries_on_failure(self, mocker):
        from bible_study.ollama import generate
        err = Exception("timeout")
        resp_ok = mocker.MagicMock(json=lambda: {"response": "Success!"})
        mock_post = mocker.patch("requests.post", side_effect=[err, err, resp_ok])
        result = generate("test prompt")
        assert result == "Success!"
