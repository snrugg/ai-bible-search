"""Local Ollama API client with retry logic."""

from __future__ import annotations

import time
from typing import Any

import requests

OLLAMA_BASE = 'http://localhost:11434'
MODEL = 'qwen3.6:35b-a3b-nvfp4'
MAX_RETRIES = 3
RETRY_DELAY = 1.0
TIMEOUT = 60

#: Context window requested from Ollama on every call.  Sent as
#: ``options.num_ctx`` so the window never depends on how the server
#: happened to be started -- Ollama otherwise autodetects 4k/32k/256k from
#: available VRAM, and silently evicts the *oldest* tokens (llama.cpp runs
#: with ``--context-shift``) when a prompt overflows.  Truncation from the
#: front drops the instruction header first, so an over-long prompt returns
#: a confident answer to a question the model never fully saw.
NUM_CTX = 32768

#: Tokens held back from ``num_ctx`` for the model's own response.
RESPONSE_RESERVE_TOKENS = 2048

#: Rough characters-per-token ratio used to size prompts before sending
#: them.  Deliberately conservative: real tokenizers vary per model and we
#: do not ship one, so this only has to be good enough to catch prompts
#: that are wildly too big.
CHARS_PER_TOKEN = 4


class PromptTooLongError(RuntimeError):
    """Raised when a prompt cannot fit in the requested context window.

    Ollama itself never reports this -- it truncates and answers anyway --
    so the check has to happen before the request goes out.
    """


def estimate_tokens(text: str) -> int:
    """Approximate the token count of *text* (see :data:`CHARS_PER_TOKEN`)."""
    return len(text) // CHARS_PER_TOKEN


def check_prompt_fits(
    prompt: str,
    num_ctx: int = NUM_CTX,
    reserve: int = RESPONSE_RESERVE_TOKENS,
) -> int:
    """Return the estimated token count, or raise if it will not fit.

    Raises
    ------
    PromptTooLongError
        If the estimate exceeds ``num_ctx - reserve``.
    """
    estimated = estimate_tokens(prompt)
    budget = num_ctx - reserve
    if estimated > budget:
        msg = (
            f'Prompt is ~{estimated} tokens ({len(prompt)} chars) but only '
            f'{budget} fit in a {num_ctx}-token context window (reserving '
            f'{reserve} for the response). Ollama would silently drop the '
            f'oldest tokens -- including the instructions -- so this is '
            f'refused. Raise ollama_num_ctx in config.yaml or shorten the '
            f'prompt.'
        )
        raise PromptTooLongError(msg)
    return estimated


def health_check(base_url: str = OLLAMA_BASE) -> bool:
    """Return True if the local Ollama instance is reachable."""
    try:
        resp = requests.get(f'{base_url}/api/tags', timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def check_model_available(
    base_url: str = OLLAMA_BASE,
    model_name: str = MODEL,
) -> bool:
    """Check if a specific model is available."""
    try:
        resp = requests.get(f'{base_url}/api/tags', timeout=5)
        resp.raise_for_status()
        models = resp.json().get('models', [])
        return any(m.get('name') == model_name for m in models)
    except Exception:
        return False


def generate(
    prompt: str,
    base_url: str = OLLAMA_BASE,
    model: str = MODEL,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
    num_ctx: int = NUM_CTX,
) -> str:
    """Generate a text response from Ollama for the given prompt.

    *num_ctx* is sent to the server rather than inherited from it, and the
    prompt is measured against it first.

    Raises
    ------
    PromptTooLongError
        If the prompt would overflow *num_ctx* and be truncated.
    """
    check_prompt_fits(prompt, num_ctx)

    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {'num_ctx': num_ctx},
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f'{base_url}/api/generate',
                json=payload,
                timeout=timeout,
               )
            resp.raise_for_status()
            return resp.json().get('response', '')
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(
                f'Ollama generation failed after {max_retries} retries: {exc}',
               ) from exc
    return ''
