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
from .models import EligibilityResult, Evidence, LevelSignal, StrengthSignal
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


class _ResumeIndex:
    """Normalised resume text for checking quotes.

    Narrow columns split words across lines ("per-" / "sonalization"); a
    model quoting that line writes the whole word. Both spellings are indexed
    so a faithful quote is not mistaken for an invented one.
    """

    def __init__(self, resume_text: str) -> None:
        joined = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", resume_text)
        self.variants = [_normalise(resume_text), _normalise(joined)]
        self.tokens = set(self.variants[0].split()) | set(self.variants[1].split())

    def contains(self, quote: str) -> bool:
        quote_norm = _normalise(quote)
        if not quote_norm:
            return False
        if any(quote_norm in variant for variant in self.variants):
            return True
        tokens = [token for token in quote_norm.split() if len(token) >= 3]
        if len(tokens) < 3:
            return False
        return sum(token in self.tokens for token in tokens) / len(tokens) >= 0.85


def quote_is_grounded(quote: str, resume_text: str) -> bool:
    """True if the quote appears in the resume (allowing small edits)."""
    return _ResumeIndex(resume_text).contains(quote)


def verify_evidence(evidence: Evidence, resume_text: str) -> list[str]:
    """Zero out any positive signal the model did not back with a real quote.

    Returns a description of each signal dropped, for the output warnings.
    """
    index = _ResumeIndex(resume_text)
    dropped: list[str] = []

    def problem(quote: str | None) -> str | None:
        if not quote:
            return "no quote given"
        if not index.contains(quote):
            snippet = " ".join(quote.split())
            return f"quote not in resume: '{snippet[:60]}'"
        return None

    for group_name, group in (("ai", evidence.ai), ("backend", evidence.backend),
                              ("cloud", evidence.cloud), ("engineering", evidence.engineering)):
        for name in type(group).model_fields:
            signal = getattr(group, name)
            if isinstance(signal, StrengthSignal) and signal.strength:
                reason = problem(signal.evidence)
                if reason:
                    signal.strength, signal.evidence = 0, None
                    dropped.append(f"{group_name}.{name} ({reason})")
            elif isinstance(signal, LevelSignal) and signal.level != "absent":
                reason = problem(signal.evidence)
                if reason:
                    signal.level, signal.evidence = "absent", None
                    dropped.append(f"{group_name}.{name} ({reason})")
    return dropped


_DEPTH_RULES = ("retrieval", "agents_tools", "state_orchestration", "data_pipeline", "evaluation")
_LEVEL_ORDER = {"absent": 0, "listed": 1, "claimed": 2, "used": 3}


def reconcile_with_filter(evidence: Evidence, eligibility: EligibilityResult) -> list[str]:
    """Keep model evidence consistent with itself and with the hard filter.

    The rule-based filter is authoritative about *whether* Python and AI
    evidence exist; the model adds depth on top and may not contradict it.
    Returns notes for the output warnings.
    """
    notes: list[str] = []

    # 1. The filter saw Python in use / listed; the model may not rate it lower.
    python = evidence.backend.python
    if _LEVEL_ORDER[python.level] < _LEVEL_ORDER[eligibility.python_level]:
        notes.append(f"Python level raised from '{python.level}' to '{eligibility.python_level}' "
                     "to match the eligibility filter")
        python.level, python.evidence = eligibility.python_level, eligibility.python_evidence

    # 2. The filter found AI work in a project or job; it cannot score nothing.
    if eligibility.ai_kind != "none":
        if evidence.ai.kind == "none":
            evidence.ai.kind = eligibility.ai_kind
            notes.append(f"AI kind set to '{eligibility.ai_kind}' to match the eligibility filter")
        if evidence.ai.use_case.strength == 0:
            evidence.ai.use_case.strength = 1
            evidence.ai.use_case.evidence = eligibility.ai_evidence
            notes.append("AI use case credited at the basic level from the eligibility filter's evidence")

    # 3. A "thin wrapper" flag that the same evidence contradicts is ignored.
    severity = evidence.ai.wrapper_severity
    if severity:
        strong = sum(1 for rule in _DEPTH_RULES if getattr(evidence.ai, rule).strength == 2)
        if strong >= config.WRAPPER_FLAG_IGNORED_AT[severity]:
            notes.append(f"thin-wrapper flag (severity {severity}) ignored: {strong} depth signals "
                         "are clearly implemented with quotes")
            evidence.ai.wrapper_severity, evidence.ai.wrapper_reason = 0, None
    return notes


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
    def _checked(self, evidence: Evidence, provider: LLMProvider, text: str, warnings: list[str]) -> Evidence:
        """Apply the quote check and describe anything it dropped."""
        dropped = verify_evidence(evidence, text)
        if dropped:
            shown = "; ".join(dropped[:8]) + (f"; and {len(dropped) - 8} more" if len(dropped) > 8 else "")
            warnings.append(f"{provider.name}: {len(dropped)} signal(s) dropped by the quote check: {shown}")
        return evidence

    async def analyze(self, text: str, lines: list[TaggedLine]) -> AnalysisOutcome:
        warnings: list[str] = []
        for provider in self.providers:
            if provider.name in self._disabled:
                continue
            key = self._cache_key(text, provider)
            cached = self.cache.get("llm", key) if self.cache else None
            if cached is not None:
                try:
                    evidence = Evidence.model_validate(cached)
                    return AnalysisOutcome(self._checked(evidence, provider, text, warnings),
                                           provider.name, warnings, from_cache=True)
                except ValidationError:
                    pass   # stale cache entry; ask again
            evidence = await self._try_provider(provider, text, warnings)
            if evidence is None:
                continue
            # Cache the model's answer as given, so the quote check (and any
            # later improvement to it) runs on every load.
            if self.cache:
                self.cache.set("llm", key, evidence.model_dump())
            return AnalysisOutcome(self._checked(evidence, provider, text, warnings), provider.name, warnings)

        if self.providers:
            warnings.append("all model providers unavailable; evidence extracted by rules")
        return AnalysisOutcome(build_evidence(lines), "rules", warnings)

    @property
    def disabled(self) -> dict[str, str]:
        return dict(self._disabled)
