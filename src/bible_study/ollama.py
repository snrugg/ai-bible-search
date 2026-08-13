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
) -> str:
    """Generate a text response from Ollama for the given prompt."""
    payload = {
         'model': model,
         'prompt': prompt,
         'stream': False,
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
