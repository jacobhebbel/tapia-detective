"""Minimal OpenRouter chat client for the top-level Tapia Detective game.

OpenRouter is OpenAI-API-compatible, so this drives it with the `openai`
Python SDK pointed at OpenRouter's base URL. Deliberately independent of
``backend/llm_client.py`` -- the simple text game loop (main.py/agents.py)
does not depend on the ``backend/`` package.

Environment (loaded via python-dotenv from a local .env if present):
  OPENROUTER_API_KEY   required for any live call
  OPENROUTER_MODEL     optional; defaults to `anthropic/claude-3.5-sonnet`
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


class MissingAPIKeyError(RuntimeError):
    """Raised when OPENROUTER_API_KEY isn't set for a live call."""


def get_client() -> OpenAI:
    """Construct an OpenAI SDK client pointed at OpenRouter.

    Raises MissingAPIKeyError if OPENROUTER_API_KEY is not set.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "OPENROUTER_API_KEY is not set. Add it to your .env to enable "
            "live dialogue."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.8) -> str:
    """Send a chat-completion request to OpenRouter and return the reply text.

    Raises:
        MissingAPIKeyError: if OPENROUTER_API_KEY is not set.
    """
    client = get_client()
    model = model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        extra_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "Tapia Detective"),
        },
    )
    return completion.choices[0].message.content or ""
