"""Gemini adapter (fallback model). Uses the REST generateContent endpoint."""

from __future__ import annotations

import httpx

from ..config import Settings
from .base import LLMProvider, ProviderError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.model = settings.gemini_model
        self._settings = settings
        self._client = client

    async def complete_json(self, system: str, user: str) -> str:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self._settings.llm_max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        url = f"{self._settings.gemini_base_url.rstrip('/')}/models/{self.model}:generateContent"
        try:
            response = await self._client.post(
                url,
                headers={"x-goog-api-key": self._settings.gemini_api_key},
                json=payload,
                timeout=httpx.Timeout(self._settings.llm_timeout_seconds, connect=10.0),
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"gemini request failed: {type(exc).__name__}") from exc

        if response.status_code in (401, 403):
            raise ProviderError(f"gemini rejected the API key (HTTP {response.status_code})", fatal=True)
        if response.status_code == 429:
            raise ProviderError("gemini rate limit reached (HTTP 429)", retry_after=None)
        if response.status_code >= 400:
            raise ProviderError(f"gemini returned HTTP {response.status_code}")
        try:
            parts = response.json()["candidates"][0]["content"]["parts"]
            content = "".join(part.get("text", "") for part in parts)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("gemini returned an unexpected response shape") from exc
        if not content:
            raise ProviderError("gemini returned an empty message")
        return content
