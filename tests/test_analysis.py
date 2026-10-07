"""Model adapters and the Groq -> Gemini -> rules fallback chain (all mocked)."""

import asyncio
import dataclasses
import json

import httpx
import pytest

from screener.analysis import Analyzer, parse_model_json, verify_evidence
from screener.cache import DiskCache
from screener.llm.base import LLMProvider, ProviderError, TokenRateLimiter
from screener.llm.gemini import GeminiProvider
from screener.llm.groq import GroqProvider
from screener.models import Evidence
from screener.sections import tag_lines

from conftest import STRONG_AGENTIC

LINES = tag_lines(STRONG_AGENTIC)

GOOD = {
    "candidate_name": "Asha Rao",
    "project_summary": "Stateful LangGraph multi-agent workflow with RAG.",
    "ai": {
        "kind": "genai",
        "use_case": {"strength": 2, "evidence": "Built a stateful LangGraph multi-agent workflow with tool calling"},
        "retrieval": {"strength": 2, "evidence": "Implemented a RAG pipeline with embeddings in pgvector"},
        "evaluation": {"strength": 2, "evidence": "This quote is invented and is not in the resume at all"},
    },
    "backend": {"python": {"level": "used", "evidence": "Developed an async FastAPI backend with PostgreSQL"},
                "redis": {"level": "used", "evidence": None}},
    "strengths": ["Agentic workflow"],
    "unknown_extra_field": "ignored",
}


class FakeProvider(LLMProvider):
    def __init__(self, name, replies):
        self.name, self.model, self.replies, self.calls = name, f"{name}-model", list(replies), 0

    async def complete_json(self, system, user):
        self.calls += 1
        assert "RESUME" in user and "untrusted data" in system
        reply = self.replies.pop(0) if self.replies else ProviderError("no more replies")
        if isinstance(reply, Exception):
            raise reply
        return reply


async def no_sleep(_seconds):
    return None


def analyze(settings, providers, cache=None):
    analyzer = Analyzer(settings, providers, cache, sleep=no_sleep)
    return analyzer, asyncio.run(analyzer.analyze(STRONG_AGENTIC, LINES))


def test_primary_success_uses_the_model_and_validates_schema(settings):
    groq = FakeProvider("groq", [json.dumps(GOOD)])
    gemini = FakeProvider("gemini", [])
    _, outcome = analyze(settings, [groq, gemini])
    assert outcome.source == "groq" and gemini.calls == 0
    assert isinstance(outcome.evidence, Evidence) and outcome.evidence.ai.retrieval.strength == 2


def test_unverifiable_quotes_earn_nothing(settings):
    _, outcome = analyze(settings, [FakeProvider("groq", [json.dumps(GOOD)])])
    assert outcome.evidence.ai.evaluation.strength == 0          # invented quote
    assert outcome.evidence.backend.redis.level == "absent"      # positive signal with no quote
    assert outcome.evidence.backend.python.level == "used"       # real quote kept
    assert any("dropped" in w for w in outcome.warnings)


def test_invalid_json_is_retried_once_then_succeeds(settings):
    groq = FakeProvider("groq", ["Sure! here you go: not json", "```json\n" + json.dumps(GOOD) + "\n```"])
    _, outcome = analyze(settings, [groq])
    assert outcome.source == "groq" and groq.calls == 2
    assert any("invalid JSON" in w for w in outcome.warnings)


def test_schema_violation_counts_as_failure(settings):
    bad = json.dumps({"ai": {"kind": "wizardry"}})
    groq = FakeProvider("groq", [bad, bad])
    gemini = FakeProvider("gemini", [json.dumps(GOOD)])
    _, outcome = analyze(settings, [groq, gemini])
    assert outcome.source == "gemini" and groq.calls == 2 and gemini.calls == 1
    assert any("schema" in w for w in outcome.warnings)


@pytest.mark.parametrize("failure", [
    ProviderError("groq request failed: ReadTimeout"),
    ProviderError("groq rate limit reached (HTTP 429)", retry_after=2),
    ProviderError("groq returned HTTP 500"),
])
def test_primary_failure_falls_back_to_secondary(settings, failure):
    groq = FakeProvider("groq", [failure, failure])
    gemini = FakeProvider("gemini", [json.dumps(GOOD)])
    _, outcome = analyze(settings, [groq, gemini])
    assert outcome.source == "gemini" and groq.calls == 2      # one retry, no loop


def test_all_providers_failing_falls_back_to_rules(settings):
    groq = FakeProvider("groq", [ProviderError("down")] * 2)
    gemini = FakeProvider("gemini", ["{broken", "{broken"])
    _, outcome = analyze(settings, [groq, gemini])
    assert outcome.source == "rules" and outcome.evidence.ai.kind == "genai"
    assert outcome.warnings[-1].startswith("all model providers unavailable")


def test_no_providers_means_rules_without_warnings(settings):
    _, outcome = analyze(settings, [])
    assert outcome.source == "rules" and outcome.warnings == []


def test_bad_api_key_disables_the_provider_for_the_batch(settings):
    groq = FakeProvider("groq", [ProviderError("groq rejected the API key (HTTP 401)", fatal=True)])
    gemini = FakeProvider("gemini", [json.dumps(GOOD), json.dumps(GOOD)])
    analyzer = Analyzer(settings, [groq, gemini], sleep=no_sleep)
    asyncio.run(analyzer.analyze(STRONG_AGENTIC, LINES))
    asyncio.run(analyzer.analyze(STRONG_AGENTIC + " second", LINES))
    assert groq.calls == 1 and gemini.calls == 2 and "groq" in analyzer.disabled


def test_repeated_failures_skip_the_provider(settings):
    s = dataclasses.replace(settings, llm_max_consecutive_failures=2)
    groq = FakeProvider("groq", [ProviderError("down")] * 10)
    analyzer = Analyzer(s, [groq], sleep=no_sleep)
    for i in range(4):
        outcome = asyncio.run(analyzer.analyze(f"{STRONG_AGENTIC} {i}", LINES))
        assert outcome.source == "rules"
    assert groq.calls == 4 and "groq" in analyzer.disabled     # 2 resumes x 2 attempts, then skipped


def test_cache_prevents_a_second_model_call(settings):
    cache = DiskCache(settings.cache_dir)
    groq = FakeProvider("groq", [json.dumps(GOOD)])
    analyze(settings, [groq], cache)
    groq_again = FakeProvider("groq", [])
    _, outcome = analyze(settings, [groq_again], cache)
    assert outcome.source == "groq" and outcome.from_cache and groq_again.calls == 0


def test_parse_model_json_variants():
    assert parse_model_json('{"a": 1}') == {"a": 1}
    assert parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_model_json('Here: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    for bad in ("", "nothing here", "[1, 2]"):
        with pytest.raises(ValueError):
            parse_model_json(bad)


def test_verify_evidence_accepts_small_edits_but_not_inventions():
    evidence = Evidence.model_validate({"ai": {"kind": "genai", "retrieval": {
        "strength": 2, "evidence": "Implemented a RAG pipeline with embeddings in pgvector, hybrid search"}}})
    assert verify_evidence(evidence, STRONG_AGENTIC) == 0
    evidence = Evidence.model_validate({"ai": {"kind": "genai", "agents_tools": {
        "strength": 2, "evidence": "Designed autonomous swarm planners using reinforcement learning"}}})
    assert verify_evidence(evidence, STRONG_AGENTIC) == 1 and evidence.ai.agents_tools.strength == 0


def test_rate_limiter_waits_when_the_minute_budget_is_used():
    clock = {"t": 0.0}
    slept = []
    async def fake_sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds
    limiter = TokenRateLimiter(1000, clock=lambda: clock["t"], sleep=fake_sleep)
    async def scenario():
        await limiter.acquire(600)
        await limiter.acquire(600)      # would exceed 1000 in the same minute
    asyncio.run(scenario())
    assert slept and sum(slept) >= 59


# ---- provider adapters against a mocked HTTP transport --------------------
def test_groq_adapter_request_and_response(settings):
    s = dataclasses.replace(settings, groq_api_key="gsk_test", groq_model="some-model")
    seen = {}
    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})
    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await GroqProvider(s, client).complete_json("system text", "user text")
    assert asyncio.run(call()) == '{"ok": true}'
    assert seen["url"].endswith("/openai/v1/chat/completions") and seen["auth"] == "Bearer gsk_test"
    assert seen["body"]["model"] == "some-model" and seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


def test_gemini_adapter_request_and_response(settings):
    s = dataclasses.replace(settings, gemini_api_key="g_test", gemini_model="gem-model")
    seen = {}
    def handler(request):
        seen["url"], seen["key"] = str(request.url), request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]})
    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await GeminiProvider(s, client).complete_json("system text", "user text")
    assert asyncio.run(call()) == '{"ok": 1}'
    assert seen["url"].endswith("/models/gem-model:generateContent") and "g_test" not in seen["url"]
    assert seen["key"] == "g_test"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.parametrize("provider_cls, key_field", [(GroqProvider, "groq_api_key"), (GeminiProvider, "gemini_api_key")])
@pytest.mark.parametrize("status, fatal", [(401, True), (429, False), (500, False)])
def test_adapters_turn_http_errors_into_provider_errors(settings, provider_cls, key_field, status, fatal):
    s = dataclasses.replace(settings, **{key_field: "k"})
    async def call():
        transport = httpx.MockTransport(lambda request: httpx.Response(status, headers={"retry-after": "7"}, json={}))
        async with httpx.AsyncClient(transport=transport) as client:
            await provider_cls(s, client).complete_json("s", "u")
    with pytest.raises(ProviderError) as info:
        asyncio.run(call())
    assert info.value.fatal is fatal


@pytest.mark.parametrize("provider_cls", [GroqProvider, GeminiProvider])
def test_adapters_handle_timeouts_and_odd_payloads(settings, provider_cls):
    def timeout(request):
        raise httpx.ConnectTimeout("no route", request=request)
    for handler in (timeout, lambda request: httpx.Response(200, json={"unexpected": []})):
        async def call():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                await provider_cls(settings, client).complete_json("s", "u")
        with pytest.raises(ProviderError):
            asyncio.run(call())
