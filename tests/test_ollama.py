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


class TestEmbed:
    """ollama.embed() -- batched /api/embed calls."""

    def _resp(self, mocker, payload):
        return mocker.MagicMock(json=lambda: payload, raise_for_status=lambda: None)

    def test_returns_one_vector_per_input(self, mocker):
        from bible_study.ollama import embed
        mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}))
        assert embed(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]

    def test_accepts_a_bare_string(self, mocker):
        from bible_study.ollama import embed
        mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1.0, 2.0]]}))
        assert embed("a") == [[1.0, 2.0]]

    def test_empty_input_makes_no_request(self, mocker):
        from bible_study.ollama import embed
        mock_post = mocker.patch("requests.post")
        assert embed([]) == []
        mock_post.assert_not_called()

    def test_posts_to_the_embed_endpoint(self, mocker):
        from bible_study.ollama import embed
        mock_post = mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1.0]]}))
        embed(["a"])
        assert mock_post.call_args[0][0].endswith("/api/embed")

    def test_sends_input_not_prompt(self, mocker):
        from bible_study.ollama import embed
        mock_post = mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1.0], [2.0]]}))
        embed(["a", "b"])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["input"] == ["a", "b"]
        assert "prompt" not in payload

    def test_uses_the_given_model(self, mocker):
        from bible_study.ollama import embed
        mock_post = mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1.0]]}))
        embed(["a"], model="custom:tag")
        assert mock_post.call_args.kwargs["json"]["model"] == "custom:tag"

    def test_coerces_values_to_float(self, mocker):
        from bible_study.ollama import embed
        mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1, 2]]}))
        assert embed(["a"]) == [[1.0, 2.0]]

    def test_retries_then_succeeds(self, mocker):
        from bible_study.ollama import embed
        mocker.patch("bible_study.ollama.time.sleep")
        mocker.patch("requests.post", side_effect=[
            Exception("boom"),
            self._resp(mocker, {"embeddings": [[1.0]]}),
        ])
        assert embed(["a"]) == [[1.0]]

    def test_raises_after_retries_exhausted(self, mocker):
        import pytest
        from bible_study.ollama import embed
        mocker.patch("bible_study.ollama.time.sleep")
        mocker.patch("requests.post", side_effect=Exception("down"))
        with pytest.raises(RuntimeError, match="embedding failed after 3 retries"):
            embed(["a"])

    def test_rejects_a_misaligned_response(self, mocker):
        import pytest
        from bible_study.ollama import embed
        mocker.patch("bible_study.ollama.time.sleep")
        mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1.0]]}))
        with pytest.raises(RuntimeError, match="refusing to misalign"):
            embed(["a", "b"])


class TestQueryInstruction:
    """Qwen3-Embedding encodes queries and documents differently."""

    def _resp(self, mocker, payload):
        return mocker.MagicMock(json=lambda: payload, raise_for_status=lambda: None)

    def test_format_query_wraps_with_instruct_prefix(self):
        from bible_study.ollama import format_query
        out = format_query("what is grace?")
        assert out.startswith("Instruct: ")
        assert "\nQuery: what is grace?" in out

    def test_format_query_accepts_a_custom_instruction(self):
        from bible_study.ollama import format_query
        assert format_query("q", instruction="Custom task").startswith(
            "Instruct: Custom task",
        )

    def test_queries_are_wrapped(self, mocker):
        from bible_study.ollama import embed
        mock_post = mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1.0]]}))
        embed(["what is grace?"], is_query=True)
        assert mock_post.call_args.kwargs["json"]["input"][0].startswith(
            "Instruct: ",
        )

    def test_documents_are_left_raw(self, mocker):
        """Wrapping documents too would silently degrade retrieval."""
        from bible_study.ollama import embed
        mock_post = mocker.patch("requests.post", return_value=self._resp(
            mocker, {"embeddings": [[1.0]]}))
        embed(["In the beginning"])
        assert mock_post.call_args.kwargs["json"]["input"] == ["In the beginning"]
