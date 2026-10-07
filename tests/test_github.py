"""GitHub scoring and failure handling, with the API fully mocked."""

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from screener.cache import DiskCache
from screener.github import GitHubEnricher, score_repos

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def repo(name, days_ago, language="Python", fork=False, archived=False, description="", topics=None):
    return {"name": name, "pushed_at": (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "language": language, "fork": fork, "archived": archived, "description": description,
            "topics": topics or [], "stargazers_count": 0, "html_url": f"https://github.com/u/{name}", "size": 10}


def test_active_python_ai_profile_scores_full_marks():
    repos = [repo("rag-bot", 3), repo("agent-kit", 20), repo("fastapi-svc", 60), repo("ml-notes", 100)]
    activity, relevance, detail = score_repos(repos, NOW)
    assert (activity, relevance) == (5, 5)
    assert detail["days_since_latest_push"] == 3 and detail["relevant_python_ai_repos"] == 4


def test_stale_profile_scores_zero():
    activity, relevance, _ = score_repos([repo("old", 900), repo("older", 1200)], NOW)
    assert (activity, relevance) == (0, 0)


def test_forks_and_archived_repos_are_not_credited():
    repos = [repo("forked-llm", 2, fork=True), repo("archived-ai", 5, archived=True)]
    activity, relevance, detail = score_repos(repos, NOW)
    assert detail["own_repos"] == 1 and relevance == 0 and activity == 3


def test_relevance_uses_language_topics_and_description():
    repos = [repo("site", 10, language="TypeScript", description="LLM agent playground"),
             repo("thing", 10, language="JavaScript", topics=["langchain"]),
             repo("portfolio", 10, language="HTML")]
    _, relevance, detail = score_repos(repos, NOW)
    assert detail["relevant_python_ai_repos"] == 2 and relevance == 3


def test_empty_profile():
    assert score_repos([], NOW)[:2] == (0, 0)


def enricher(handler, settings, cache=None):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GitHubEnricher(settings, client, cache, now=NOW), client


def run(coro):
    return asyncio.run(coro)


def test_ok_profile(settings):
    def handler(request):
        assert request.url.path == "/users/asha/repos"
        assert "authorization" not in request.headers          # no token configured
        return httpx.Response(200, json=[repo("rag-bot", 3), repo("api", 40)])
    e, _ = enricher(handler, settings)
    result = run(e.enrich(["asha"]))
    assert result.status == "ok" and result.username == "asha" and result.score > 0
    assert "Score" in result.summary and e.api_calls == 1


def test_token_is_sent_from_settings_when_present(settings):
    import dataclasses
    seen = {}
    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=[])
    e, _ = enricher(handler, dataclasses.replace(settings, github_token="tok123"))
    run(e.enrich(["asha"]))
    assert seen["auth"] == "Bearer tok123"


def test_rejected_token_falls_back_to_unauthenticated_calls(settings):
    import dataclasses
    seen = []
    def handler(request):
        seen.append("authorization" in request.headers)
        if "authorization" in request.headers:
            return httpx.Response(401, json={"message": "Bad credentials"})
        return httpx.Response(200, json=[repo("rag-bot", 3)])
    e, _ = enricher(handler, dataclasses.replace(settings, github_token="ghp_invalid"))
    first = run(e.enrich(["asha"]))
    second = run(e.enrich(["ravi"]))
    assert first.status == second.status == "ok" and e.token_rejected
    assert seen == [True, False, False]          # token tried once, then dropped for the run


def test_rejected_token_with_concurrent_lookups_still_scores_everyone(settings):
    import dataclasses
    def handler(request):
        if "authorization" in request.headers:
            return httpx.Response(401, json={"message": "Bad credentials"})
        return httpx.Response(200, json=[repo("rag-bot", 3)])
    e, _ = enricher(handler, dataclasses.replace(settings, github_token="ghp_invalid"))
    async def together():
        return await asyncio.gather(*(e.enrich([f"user{i}"]) for i in range(6)))
    assert [r.status for r in run(together())] == ["ok"] * 6


def test_missing_username_is_no_profile_without_any_call(settings):
    e, _ = enricher(lambda request: httpx.Response(500), settings)
    result = run(e.enrich([]))
    assert result.status == "no_profile" and result.score == 0 and e.api_calls == 0


def test_404_tries_the_next_candidate(settings):
    def handler(request):
        if "/users/wrong/" in request.url.path:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(200, json=[repo("rag-bot", 3)])
    e, _ = enricher(handler, settings)
    result = run(e.enrich(["wrong", "right"]))
    assert result.status == "ok" and result.username == "right" and result.tried_usernames == ["wrong", "right"]


def test_all_candidates_missing_is_not_found(settings):
    e, _ = enricher(lambda request: httpx.Response(404, json={}), settings)
    result = run(e.enrich(["ghost"]))
    assert result.status == "not_found" and result.score == 0


def test_two_real_accounts_uses_the_more_active_one(settings):
    def handler(request):
        if "/users/old-account/" in request.url.path:
            return httpx.Response(200, json=[repo("x", 800, language="HTML")])
        return httpx.Response(200, json=[repo("rag-bot", 3), repo("agent", 9)])
    e, _ = enricher(handler, settings)
    result = run(e.enrich(["old-account", "new-account"]))
    assert result.username == "new-account" and "more than one GitHub account" in result.detail["note"]


def test_rate_limit_is_recorded_and_stops_further_calls(settings):
    def handler(request):
        return httpx.Response(403, headers={"x-ratelimit-remaining": "0"},
                              json={"message": "API rate limit exceeded"})
    e, _ = enricher(handler, settings)
    first = run(e.enrich(["a"]))
    second = run(e.enrich(["b"]))
    assert first.status == second.status == "rate_limited"
    assert first.score == 0 and e.api_calls == 1            # the second lookup made no request


def test_server_error_and_timeout_are_recorded_not_raised(settings):
    e, _ = enricher(lambda request: httpx.Response(500), settings)
    assert run(e.enrich(["a"])).status == "error"

    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)
    e, _ = enricher(boom, settings)
    result = run(e.enrich(["a"]))
    assert result.status == "error" and "timed out" in result.summary


def test_malformed_response_is_an_error(settings):
    e, _ = enricher(lambda request: httpx.Response(200, json={"unexpected": True}), settings)
    assert run(e.enrich(["a"])).status == "error"


def test_results_are_cached_in_the_run_and_on_disk(settings):
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json=[repo("rag-bot", 3)])
    cache = DiskCache(settings.cache_dir)
    e, _ = enricher(handler, settings, cache)
    run(e.enrich(["Asha"]))
    run(e.enrich(["asha"]))                       # same user, different case: in-run cache
    assert calls["n"] == 1
    e2, _ = enricher(handler, settings, cache)    # a new run reads the disk cache
    assert run(e2.enrich(["asha"])).status == "ok" and calls["n"] == 1
