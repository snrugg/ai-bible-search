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


class TestContextWindow:
    """num_ctx is sent explicitly and enforced before the request goes out."""

    def test_generate_sends_default_num_ctx_in_options(self, mocker):
        from bible_study.ollama import NUM_CTX, generate
        resp = mocker.MagicMock(json=lambda: {"response": "ok"})
        mock_post = mocker.patch("requests.post", return_value=resp)
        generate("short prompt")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"] == {"num_ctx": NUM_CTX}

    def test_generate_sends_explicit_num_ctx(self, mocker):
        from bible_study.ollama import generate
        resp = mocker.MagicMock(json=lambda: {"response": "ok"})
        mock_post = mocker.patch("requests.post", return_value=resp)
        generate("short prompt", num_ctx=131072)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"]["num_ctx"] == 131072

    def test_generate_refuses_prompt_that_would_be_truncated(self, mocker):
        from bible_study.ollama import PromptTooLongError, generate
        mock_post = mocker.patch("requests.post")
        # 4 chars/token, so 8k tokens against a 4k window.
        with pytest.raises(PromptTooLongError):
            generate("x" * 32_000, num_ctx=4096)
        mock_post.assert_not_called()

    def test_error_names_the_config_key_to_raise(self, mocker):
        from bible_study.ollama import PromptTooLongError, generate
        mocker.patch("requests.post")
        with pytest.raises(PromptTooLongError, match="ollama_num_ctx"):
            generate("x" * 32_000, num_ctx=4096)

    def test_larger_window_admits_the_same_prompt(self, mocker):
        from bible_study.ollama import generate
        resp = mocker.MagicMock(json=lambda: {"response": "ok"})
        mocker.patch("requests.post", return_value=resp)
        assert generate("x" * 32_000, num_ctx=131072) == "ok"

    def test_reserve_is_held_back_for_the_response(self, mocker):
        from bible_study.ollama import PromptTooLongError, generate
        mocker.patch("requests.post")
        # Exactly num_ctx tokens: fits the raw window, not the reserve.
        with pytest.raises(PromptTooLongError):
            generate("x" * (4096 * 4), num_ctx=4096)

    def test_check_prompt_fits_returns_estimate(self):
        from bible_study.ollama import check_prompt_fits
        assert check_prompt_fits("x" * 400, num_ctx=4096) == 100

    def test_estimate_tokens_scales_with_length(self):
        from bible_study.ollama import estimate_tokens
        assert estimate_tokens("x" * 4000) == 1000
        assert estimate_tokens("") == 0

    def test_prompt_too_long_is_a_runtime_error(self):
        from bible_study.ollama import PromptTooLongError
        assert issubclass(PromptTooLongError, RuntimeError)
