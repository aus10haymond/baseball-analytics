"""
LLM client for the Orchestrator Agent.

Provides a unified interface to OpenAI and Anthropic APIs via httpx,
with exponential-backoff retry logic and a simple rate limiter.
No optional SDK dependencies required — calls are made over raw HTTP.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx


class LLMConfigError(Exception):
    """Raised when the LLM client is mis-configured (e.g. missing API key)."""


class LLMRateLimitError(Exception):
    """Raised when the upstream API returns a 429 and retries are exhausted."""


class LLMError(Exception):
    """Generic LLM call failure."""


# ------------------------------------------------------------------
# Per-provider HTTP helpers
# ------------------------------------------------------------------

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_HF_URL = "https://api-inference.huggingface.co/v1/chat/completions"


def _build_openai_request(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _parse_openai_response(data: Dict[str, Any]) -> str:
    return data["choices"][0]["message"]["content"]


def _build_anthropic_request(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt
    return payload


def _parse_anthropic_response(data: Dict[str, Any]) -> str:
    return data["content"][0]["text"]


# ------------------------------------------------------------------
# Rate limiter — enforces a minimum interval between calls
# ------------------------------------------------------------------

class _RateLimiter:
    """Ensures at least ``min_interval_seconds`` between calls."""

    def __init__(self, min_interval_seconds: float = 0.5):
        self._min_interval = min_interval_seconds
        self._last_call: float = 0.0

    async def acquire(self) -> None:
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()


# ------------------------------------------------------------------
# LLMClient
# ------------------------------------------------------------------

class LLMClient:
    """
    Async LLM client supporting OpenAI, Anthropic, HuggingFace, and Ollama via raw HTTP.

    Usage::

        client = LLMClient.from_settings(settings)
        response = await client.call("Classify this text: ...", system_prompt="...")
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 500,
        max_retries: int = 3,
        min_interval_seconds: float = 0.5,
        base_url: Optional[str] = None,
        fallback: Optional["LLMClient"] = None,
    ):
        provider = provider.lower()
        if not api_key and provider not in ("ollama",):
            raise LLMConfigError(
                f"No API key provided for LLM provider {provider!r}. "
                "Set DM_LLM_API_KEY in your environment."
            )
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_url = base_url
        self.fallback = fallback
        self._rate_limiter = _RateLimiter(min_interval_seconds)

    @classmethod
    def from_settings(cls, settings) -> "LLMClient":
        """Create an LLMClient (with optional fallback) from the shared settings object."""
        fallback = None
        if settings.llm_fallback_provider:
            fallback_base_url = settings.llm_fallback_base_url or "http://localhost:11434"
            fallback = cls(
                provider=settings.llm_fallback_provider,
                model=settings.llm_fallback_model or "",
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                base_url=fallback_base_url,
            )
        return cls(
            provider=settings.llm_provider,
            model=settings.llm_model,
            api_key=settings.llm_api_key or "",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            fallback=fallback,
        )

    async def call(self, prompt: str, system_prompt: str = "") -> str:
        """
        Send ``prompt`` to the configured LLM and return the text response.

        Retries on transient errors (5xx, network errors) with exponential backoff.
        If all retries fail and a fallback client is configured, delegates to it.
        """
        last_error: Exception = LLMError("No attempts made")
        for attempt in range(self.max_retries + 1):
            try:
                await self._rate_limiter.acquire()
                return await self._dispatch(prompt, system_prompt)
            except (LLMRateLimitError, LLMError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                await asyncio.sleep(2 ** attempt)

        if self.fallback is not None:
            return await self.fallback.call(prompt, system_prompt)
        raise last_error

    async def _dispatch(self, prompt: str, system_prompt: str) -> str:
        if self.provider == "openai":
            return await self._call_openai_compat(
                _OPENAI_URL, f"Bearer {self.api_key}", prompt, system_prompt
            )
        if self.provider == "anthropic":
            return await self._call_anthropic(prompt, system_prompt)
        if self.provider == "huggingface":
            return await self._call_openai_compat(
                _HF_URL, f"Bearer {self.api_key}", prompt, system_prompt
            )
        if self.provider == "ollama":
            base = (self.base_url or "http://localhost:11434").rstrip("/")
            url = f"{base}/v1/chat/completions"
            return await self._call_openai_compat(url, "", prompt, system_prompt)
        raise LLMConfigError(
            f"Unknown LLM provider: {self.provider!r}. "
            "Use 'openai', 'anthropic', 'huggingface', or 'ollama'."
        )

    async def _call_openai_compat(
        self, url: str, auth_header: str, prompt: str, system_prompt: str
    ) -> str:
        payload = _build_openai_request(
            self.model, system_prompt, prompt, self.temperature, self.max_tokens
        )
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header
        return await self._post(url, headers, payload, _parse_openai_response)

    async def _call_openai(self, prompt: str, system_prompt: str) -> str:
        payload = _build_openai_request(
            self.model, system_prompt, prompt, self.temperature, self.max_tokens
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return await self._post(_OPENAI_URL, headers, payload, _parse_openai_response)

    async def _call_anthropic(self, prompt: str, system_prompt: str) -> str:
        payload = _build_anthropic_request(
            self.model, system_prompt, prompt, self.temperature, self.max_tokens
        )
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        return await self._post(_ANTHROPIC_URL, headers, payload, _parse_anthropic_response)

    async def _post(self, url: str, headers: dict, payload: dict, parser) -> str:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            raise LLMError(f"Network error calling {url}: {exc}") from exc

        if response.status_code == 429:
            raise LLMRateLimitError(f"Rate limited by {url}")
        if response.status_code >= 500:
            raise LLMError(f"Server error {response.status_code} from {url}: {response.text[:200]}")
        if response.status_code >= 400:
            raise LLMError(
                f"Client error {response.status_code} from {url}: {response.text[:200]}"
            )

        try:
            return parser(response.json())
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(f"Failed to parse LLM response: {exc}") from exc


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from ``text``.

    Handles responses wrapped in markdown code fences (```json ... ```).
    Raises ``LLMError`` if no valid JSON is found.
    """
    # Strip markdown code fences
    if "```" in text:
        start = text.find("{", text.find("```"))
        end = text.rfind("}") + 1
    else:
        start = text.find("{")
        end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise LLMError(f"No JSON object found in LLM response: {text[:200]!r}")

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Invalid JSON in LLM response: {exc}") from exc
