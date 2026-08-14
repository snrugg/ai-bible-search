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

#: Model used to embed chunks and questions for vector search.  Must be an
#: embedding model, not a chat model.
EMBED_MODEL = 'qwen3-embedding:0.6b'

#: Vector width emitted by :data:`EMBED_MODEL`.  Baked into the vec0
#: virtual tables when they are created and not alterable afterwards.
EMBED_DIMS = 1024

#: Inputs sent per /api/embed request.
EMBED_BATCH = 32

#: Embedding gets its own, longer timeout: the first request loads a second
#: model into VRAM, which can take far longer than a warm generate call.
EMBED_TIMEOUT = 120

#: Qwen3-Embedding is *asymmetric*.  Queries are wrapped in a one-line
#: instruction; documents are embedded raw.  Both sides must stay encoded
#: this way -- wrapping documents too, or leaving queries bare, silently
#: costs retrieval accuracy and is invisible in the output.
QUERY_INSTRUCTION = (
    'Given a question about the Bible, retrieve the passages and study '
    'summaries that answer it'
)


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


def format_query(text: str, instruction: str = QUERY_INSTRUCTION) -> str:
    """Wrap *text* in the Qwen3-Embedding query-instruction prefix."""
    return f'Instruct: {instruction}\nQuery: {text}'


def embed(
    texts: str | list[str],
    base_url: str = OLLAMA_BASE,
    model: str = EMBED_MODEL,
    timeout: int = EMBED_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    is_query: bool = False,
    instruction: str = QUERY_INSTRUCTION,
) -> list[list[float]]:
    """Embed one or more strings, returning one vector per input.

    Always returns a list of vectors -- a bare string yields a one-element
    list -- so callers never have to branch on the input type.

    Set *is_query* for search queries only; see :data:`QUERY_INSTRUCTION`.

    Uses ``/api/embed``, which takes a batch under ``input`` and returns
    ``{"embeddings": [[...], ...]}``.  The legacy ``/api/embeddings``
    endpoint takes a single ``prompt`` and returns ``{"embedding": [...]}``
    -- do not mix the two.

    Raises
    ------
    RuntimeError
        If the request keeps failing, or if Ollama returns a different
        number of vectors than there were inputs.
    """
    items = [texts] if isinstance(texts, str) else list(texts)
    if not items:
        return []
    if is_query:
        items = [format_query(t, instruction) for t in items]

    payload = {'model': model, 'input': items}

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f'{base_url}/api/embed',
                json=payload,
                timeout=timeout,
               )
            resp.raise_for_status()
            vectors = resp.json().get('embeddings', [])
            if len(vectors) != len(items):
                msg = (
                    f'Ollama returned {len(vectors)} embeddings for '
                    f'{len(items)} inputs -- refusing to misalign vectors '
                    f'with their chunks'
                )
                raise RuntimeError(msg)
            return [[float(x) for x in vec] for vec in vectors]
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(
                f'Ollama embedding failed after {max_retries} retries: {exc}',
               ) from exc
    return []
