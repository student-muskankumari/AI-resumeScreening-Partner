"""Evidence extraction with a fallback chain: Groq -> Gemini -> rules.

Guarantees:
  * one model call per resume at most per provider attempt, never a loop
  * a failed, slow, rate-limited or malformed model answer never fails the
    resume: the next source takes over, and rules always succeed
  * model output is validated against the Pydantic `Evidence` schema
  * every evidence quote is checked against the resume text; anything the
    model cannot back with a real quote earns no points
  * results are cached on disk by resume text, prompt version and model
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import config
from .cache import DiskCache
from .config import Settings
from .llm.base import LLMProvider, ProviderError, TokenRateLimiter, estimate_tokens
from .llm.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .llm.rules import build_evidence
from .models import Evidence, LevelSignal, StrengthSignal
from .sections import TaggedLine

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class AnalysisOutcome:
    evidence: Evidence
    source: str                       # "groq" | "gemini" | "rules"
    warnings: list[str] = field(default_factory=list)
    from_cache: bool = False


def parse_model_json(raw: str) -> dict:
    """Pull the JSON object out of a model reply (tolerates code fences)."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def _normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def quote_is_grounded(quote: str, resume_norm: str, resume_tokens: set[str]) -> bool:
    """True if the quote appears in the resume (allowing small edits)."""
    quote_norm = _normalise(quote)
    if not quote_norm:
        return False
    if quote_norm in resume_norm:
        return True
    tokens = [token for token in quote_norm.split() if len(token) >= 3]
    if len(tokens) < 3:
        return False
    return sum(token in resume_tokens for token in tokens) / len(tokens) >= 0.85


def verify_evidence(evidence: Evidence, resume_text: str) -> int:
    """Zero out any positive signal the model did not back with a real quote.

    Returns the number of signals dropped.
    """
    resume_norm = _normalise(resume_text)
    resume_tokens = set(resume_norm.split())
    dropped = 0
    for group in (evidence.ai, evidence.backend, evidence.cloud, evidence.engineering):
        for name in type(group).model_fields:
            signal = getattr(group, name)
            if isinstance(signal, StrengthSignal):
                if signal.strength and not (signal.evidence and quote_is_grounded(signal.evidence, resume_norm, resume_tokens)):
                    signal.strength, signal.evidence = 0, None
                    dropped += 1
            elif isinstance(signal, LevelSignal):
                if signal.level != "absent" and not (signal.evidence and quote_is_grounded(signal.evidence, resume_norm, resume_tokens)):
                    signal.level, signal.evidence = "absent", None
                    dropped += 1
    return dropped


class Analyzer:
    def __init__(self, settings: Settings, providers: list[LLMProvider] | None = None,
                 cache: DiskCache | None = None, *, sleep=asyncio.sleep) -> None:
        self.settings = settings
        self.providers = list(providers or [])
        self.cache = cache
        self._sleep = sleep
        self._limiters = {p.name: TokenRateLimiter(settings.llm_tokens_per_minute, sleep=sleep)
                          for p in self.providers}
        self._failures: dict[str, int] = {p.name: 0 for p in self.providers}
        self._disabled: dict[str, str] = {}
        self.calls: dict[str, int] = {p.name: 0 for p in self.providers}

    # -- cache ---------------------------------------------------------------
    @staticmethod
    def _cache_key(text: str, provider: LLMProvider) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        model = re.sub(r"[^A-Za-z0-9.-]", "-", provider.model)
        return f"{digest}_{PROMPT_VERSION}_{provider.name}_{model}"

    # -- one provider ----------------------------------------------------------
    async def _try_provider(self, provider: LLMProvider, text: str, warnings: list[str]) -> Evidence | None:
        system, user = SYSTEM_PROMPT, build_user_prompt(text[: config.MAX_RESUME_CHARS_FOR_LLM])
        attempts = 1 + max(0, self.settings.llm_retries_per_provider)
        for attempt in range(1, attempts + 1):
            await self._limiters[provider.name].acquire(
                estimate_tokens(system, user) + self.settings.llm_expected_output_tokens
            )
            self.calls[provider.name] += 1
            try:
                raw = await provider.complete_json(system, user)
                evidence = Evidence.model_validate(parse_model_json(raw))
            except ProviderError as exc:
                warnings.append(f"{provider.name} attempt {attempt}: {exc}")
                if exc.fatal:
                    self._disabled[provider.name] = str(exc)
                    return None
                if attempt < attempts and exc.retry_after:
                    await self._sleep(min(exc.retry_after, 30.0))
                continue
            except (ValueError, ValidationError) as exc:
                reason = "invalid JSON" if isinstance(exc, ValueError) and not isinstance(exc, ValidationError) \
                    else "output did not match the schema"
                warnings.append(f"{provider.name} attempt {attempt}: {reason}")
                continue
            self._failures[provider.name] = 0
            return evidence
        self._failures[provider.name] += 1
        if self._failures[provider.name] >= self.settings.llm_max_consecutive_failures:
            self._disabled[provider.name] = "too many consecutive failures; skipped for the rest of the batch"
        return None

    # -- public ----------------------------------------------------------------
    async def analyze(self, text: str, lines: list[TaggedLine]) -> AnalysisOutcome:
        warnings: list[str] = []
        for provider in self.providers:
            if provider.name in self._disabled:
                continue
            key = self._cache_key(text, provider)
            cached = self.cache.get("llm", key) if self.cache else None
            if cached is not None:
                try:
                    return AnalysisOutcome(Evidence.model_validate(cached), provider.name, warnings, from_cache=True)
                except ValidationError:
                    pass   # stale cache entry; ask again
            evidence = await self._try_provider(provider, text, warnings)
            if evidence is None:
                continue
            dropped = verify_evidence(evidence, text)
            if dropped:
                warnings.append(f"{provider.name}: {dropped} signal(s) dropped because the quote was not found in the resume")
            if self.cache:
                self.cache.set("llm", key, evidence.model_dump())
            return AnalysisOutcome(evidence, provider.name, warnings)

        if self.providers:
            warnings.append("all model providers unavailable; evidence extracted by rules")
        return AnalysisOutcome(build_evidence(lines), "rules", warnings)

    @property
    def disabled(self) -> dict[str, str]:
        return dict(self._disabled)
