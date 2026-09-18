"""Thin OpenRouter LLM client for the Nautilus House game.

OpenRouter is OpenAI-API-compatible, so we drive it with the official `openai`
Python SDK pointed at OpenRouter's base URL.

Environment (loaded via python-dotenv from a local .env if present):
  OPENROUTER_API_KEY   required for any live call
  OPENROUTER_MODEL     optional; defaults to `anthropic/claude-3.5-sonnet`

Design notes
------------
* Importing this module never crashes, even with no API key set. The key is
  only checked when a client is actually constructed (`get_client`) / a call is
  made (`chat`). Missing-key raises `MissingAPIKeyError`, which callers/tests
  can catch to skip live calls.
* `chat(...)` returns the full text when `stream=False`, or a generator that
  yields text chunks when `stream=True`.
"""

from __future__ import annotations

import os
from typing import Iterator

from dotenv import load_dotenv
from openai import OpenAI

# Load a local .env (if any) once, at import time. This is side-effect-free
# when no .env exists and never raises on a missing key.
load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


class MissingAPIKeyError(RuntimeError):
    """Raised when OPENROUTER_API_KEY is not available for a live call.

    Catchable by callers/tests so they can skip live LLM calls gracefully
    instead of crashing.
    """


def get_api_key() -> str:
    """Return the OpenRouter API key or raise MissingAPIKeyError."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "OPENROUTER_API_KEY is not set. Add it to your environment or a "
            ".env file to make live LLM calls. (Tests can catch "
            "MissingAPIKeyError and skip.)"
        )
    return key


def get_default_model() -> str:
    """Return the configured model id, falling back to the default."""
    return os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL


def get_client() -> OpenAI:
    """Construct an OpenAI SDK client pointed at OpenRouter.

    Raises MissingAPIKeyError if OPENROUTER_API_KEY is not set.
    """
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=get_api_key())


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.8,
    stream: bool = False,
) -> "str | Iterator[str]":
    """Send a chat-completion request to OpenRouter.

    Args:
        messages: OpenAI-style message dicts, e.g.
            [{"role": "system", "content": ...}, {"role": "user", ...}].
        model: OpenRouter model id; defaults to OPENROUTER_MODEL or
            `anthropic/claude-3.5-sonnet`.
        temperature: sampling temperature.
        stream: if False (default), returns the full response text as a str.
            If True, returns a generator yielding text chunks as they arrive.

    Returns:
        The full response text (stream=False) or an iterator of text chunks
        (stream=True).

    Raises:
        MissingAPIKeyError: if OPENROUTER_API_KEY is not set.
    """
    client = get_client()
    model = model or get_default_model()

    # Optional attribution headers OpenRouter recommends; harmless if ignored.
    extra_headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Nautilus House Mystery"),
    }

    if stream:
        return _chat_stream(client, model, messages, temperature, extra_headers)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        extra_headers=extra_headers,
    )
    return completion.choices[0].message.content or ""


def _chat_stream(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    extra_headers: dict,
) -> Iterator[str]:
    """Yield text chunks from a streaming chat completion."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        extra_headers=extra_headers,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            yield text


if __name__ == "__main__":
    # Minimal self-check: never make a live call unless a key is present.
    print(f"default model: {get_default_model()}")
    try:
        get_api_key()
        print("OPENROUTER_API_KEY detected (live calls possible).")
    except MissingAPIKeyError:
        print("No OPENROUTER_API_KEY set (live calls will be skipped).")
