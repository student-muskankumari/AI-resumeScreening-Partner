"""Groq adapter (primary model). Uses Groq's OpenAI-compatible endpoint."""

from __future__ import annotations

import httpx

from ..config import Settings
from .base import LLMProvider, ProviderError


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.model = settings.groq_model
        self._settings = settings
        self._client = client

    async def complete_json(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_completion_tokens": self._settings.llm_max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._settings.groq_reasoning_effort:
            payload["reasoning_effort"] = self._settings.groq_reasoning_effort
        try:
            response = await self._client.post(
                f"{self._settings.groq_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self._settings.groq_api_key}"},
                json=payload,
                timeout=httpx.Timeout(self._settings.llm_timeout_seconds, connect=10.0),
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"groq request failed: {type(exc).__name__}") from exc

        if response.status_code in (401, 403):
            raise ProviderError(f"groq rejected the API key (HTTP {response.status_code})", fatal=True)
        if response.status_code == 429:
            raise ProviderError("groq rate limit reached (HTTP 429)", retry_after=_retry_after(response))
        if response.status_code >= 400:
            raise ProviderError(f"groq returned HTTP {response.status_code}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("groq returned an unexpected response shape") from exc
        if not content:
            raise ProviderError("groq returned an empty message")
        return content
