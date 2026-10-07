"""The small interface every model provider implements.

Business logic only ever sees `LLMProvider.complete_json`. Swapping or adding
a provider means writing one class like `GroqProvider`; nothing else changes.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque


class ProviderError(Exception):
    """A provider call failed. `fatal` means retrying cannot help (bad key)."""

    def __init__(self, message: str, *, retry_after: float | None = None, fatal: bool = False) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.fatal = fatal


class LLMProvider(ABC):
    name: str = "provider"
    model: str = ""

    @abstractmethod
    async def complete_json(self, system: str, user: str) -> str:
        """Return the model's raw text, which should be one JSON object."""


def estimate_tokens(*texts: str) -> int:
    """Rough token count (about four characters per token)."""
    return sum(len(text) for text in texts) // 4 + 1


class TokenRateLimiter:
    """Keeps requests under a tokens-per-minute budget (free tiers are tight)."""

    def __init__(self, tokens_per_minute: int, *, clock=time.monotonic, sleep=asyncio.sleep) -> None:
        self.tokens_per_minute = max(1, tokens_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._events and now - self._events[0][0] >= 60:
                    self._events.popleft()
                used = sum(count for _, count in self._events)
                if not self._events or used + tokens <= self.tokens_per_minute:
                    self._events.append((now, tokens))
                    return
                await self._sleep(max(0.05, 60 - (now - self._events[0][0])))
