"""
Shared LLM client for all OpenRouter API calls in the pipeline.

All LLM calls should go through this module so that:
  - The model is configured in one place (OPENROUTER_MODEL env var)
  - Every call gets structured logging (model, latency, tokens, parse result)
  - Parse failures log the full raw response for debugging
"""

import json
import logging
import os
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-4o"


class LLMResponseError(Exception):
    """Raised when the LLM API returns an unparseable, empty, or failed response."""
    pass


class LLMClient:
    """
    Thin wrapper around the OpenRouter chat completions API.

    Args:
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        model:   OpenRouter model slug. Falls back to OPENROUTER_MODEL env var,
                 then to "openai/gpt-4o".

    To A/B test models: instantiate with an explicit model= argument, or set
    OPENROUTER_MODEL in the environment before starting the server.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
        # OPENROUTER_MAX_TOKENS in .env controls output length for all LLM calls.
        # If unset, the API uses its own default (typically 4096).
        _env_max = os.environ.get("OPENROUTER_MAX_TOKENS")
        self.max_tokens: int | None = max_tokens or (int(_env_max) if _env_max else None)

    def chat(
        self,
        messages: list[dict[str, Any]],
        response_format: dict | None = None,
        timeout: int = 120,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Call OpenRouter chat completions and return parsed JSON.

        Returns:
            (parsed_result, usage) where usage contains token counts from the API.

        Raises:
            LLMResponseError: If the HTTP request fails, the response content is
                              empty or None, or the content is not valid JSON.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if response_format:
            payload["response_format"] = response_format

        t_start = time.perf_counter()
        try:
            response = requests.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise LLMResponseError(f"HTTP request failed: {e}") from e

        latency = time.perf_counter() - t_start

        if not response.ok:
            logger.warning(
                "[LLMClient] HTTP error model=%s status=%d latency=%.2fs body=%s",
                self.model, response.status_code, latency, response.text[:500],
            )
            raise LLMResponseError(
                f"HTTP {response.status_code}: {response.text[:200]}"
            )

        response_data = response.json()
        usage: dict[str, Any] = response_data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", "?")
        tokens_out = usage.get("completion_tokens", "?")

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.warning(
                "[LLMClient] Unexpected response structure model=%s latency=%.2fs",
                self.model, latency,
            )
            logger.debug("[LLMClient] Full response_data: %s", response_data)
            raise LLMResponseError(
                f"Missing choices[0].message.content: {e}"
            ) from e

        if not content:
            logger.warning(
                "[LLMClient] Empty content model=%s latency=%.2fs tokens_in=%s tokens_out=%s",
                self.model, latency, tokens_in, tokens_out,
            )
            logger.debug("[LLMClient] Full response_data (empty content): %s", response_data)
            raise LLMResponseError("LLM returned empty content field")

        # Some models wrap their JSON in markdown code fences despite response_format.
        # Strip them before parsing.
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
            stripped = re.sub(r"\n?```\s*$", "", stripped)

        try:
            result = json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.warning(
                "[LLMClient] JSON parse failed model=%s latency=%.2fs content_preview=%r error=%s",
                self.model, latency, content[:300], e,
            )
            logger.debug("[LLMClient] Full content that failed to parse: %s", content)
            raise LLMResponseError(f"JSON decode error: {e}") from e

        logger.info(
            "[LLMClient] model=%s latency=%.2fs tokens_in=%s tokens_out=%s parsed=ok",
            self.model, latency, tokens_in, tokens_out,
        )
        return result, usage
