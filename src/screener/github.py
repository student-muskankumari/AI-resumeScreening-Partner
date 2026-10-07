"""GitHub enrichment (assignment section 5).

One public API call per candidate: the list of the user's own repositories,
sorted by last push. That single response carries everything the score uses
(push dates, languages, descriptions, topics, fork/archived flags), so a
whole batch fits inside even the unauthenticated rate limit.

GitHub is a positive signal only. A missing profile, a private account, a
rate limit or any API error gives 0 points, is recorded in the output, and
never changes eligibility or stops the batch.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import httpx

from . import config, taxonomy
from .cache import DiskCache
from .config import Settings
from .models import GitHubResult

_REPO_FIELDS = ("name", "description", "language", "fork", "archived", "pushed_at",
                "stargazers_count", "topics", "html_url", "size")
_PYTHON_LANGUAGES = {"python", "jupyter notebook"}
_AI_REPO_TERMS = taxonomy.rx(
    r"ai", r"ml", r"llms?", r"rag", r"gpt", r"genai", r"agents?", r"agentic", r"chatbot", r"langchain",
    r"langgraph", r"llamaindex", r"openai", r"gemini", r"ollama", r"embeddings?", r"vector", r"nlp",
    r"machine learning", r"deep learning", r"neural", r"transformers?", r"copilot", r"mcp",
    r"detection", r"classification", r"prediction", r"recognition", r"mlops", r"pytorch", r"tensorflow",
)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_relevant(repo: dict) -> bool:
    if (repo.get("language") or "").lower() in _PYTHON_LANGUAGES:
        return True
    text = " ".join([
        re.sub(r"[-_.]", " ", repo.get("name") or ""),
        repo.get("description") or "",
        " ".join(repo.get("topics") or []).replace("-", " "),
    ])
    return bool(_AI_REPO_TERMS.search(text))


def score_repos(repos: list[dict], now: datetime) -> tuple[int, int, dict]:
    """Return (recent activity 0-5, maintained/relevant 0-5, supporting facts)."""
    own = [repo for repo in repos if not repo.get("fork")]
    dated = [(repo, _parse_time(repo.get("pushed_at"))) for repo in own]
    dated = [(repo, pushed) for repo, pushed in dated if pushed is not None]

    detail: dict = {"public_repos": len(repos), "own_repos": len(own)}
    if not dated:
        detail["activity_basis"] = "No public repositories of their own"
        detail["relevance_basis"] = "No public repositories of their own"
        return 0, 0, detail

    latest = max(pushed for _, pushed in dated)
    days_since = max(0, (now - latest).days)
    active = [repo for repo, pushed in dated if (now - pushed).days <= config.GITHUB_ACTIVE_WINDOW_DAYS]
    maintained = [repo for repo, pushed in dated
                  if (now - pushed).days <= config.GITHUB_MAINTAINED_DAYS and not repo.get("archived")]
    relevant = [repo for repo in maintained if _is_relevant(repo)]

    recent_30, recent_90, recent_365 = config.GITHUB_RECENT_DAYS
    activity = 3 if days_since <= recent_30 else 2 if days_since <= recent_90 else 1 if days_since <= recent_365 else 0
    for threshold, bonus in config.GITHUB_ACTIVE_BONUS:
        if len(active) >= threshold:
            activity += bonus
            break
    activity = min(activity, config.GITHUB_RUBRIC["recent_activity"])

    relevance = 0
    for threshold, points in config.GITHUB_RELEVANT_POINTS:
        if len(relevant) >= threshold:
            relevance = points
            break
    if relevance == 0 and len(maintained) >= 3:
        relevance = 1

    names = [repo.get("name") for repo in relevant[:5]]
    detail.update({
        "latest_push": latest.date().isoformat(),
        "days_since_latest_push": days_since,
        "repos_pushed_last_180_days": len(active),
        "maintained_repos_last_12_months": len(maintained),
        "relevant_python_ai_repos": len(relevant),
        "relevant_repo_names": names,
        "activity_basis": (f"Latest public push {days_since} day(s) ago; "
                           f"{len(active)} repo(s) pushed in the last {config.GITHUB_ACTIVE_WINDOW_DAYS} days"),
        "relevance_basis": (f"{len(relevant)} maintained Python/AI repo(s) out of "
                            f"{len(maintained)} maintained in the last 12 months"
                            + (f": {', '.join(n for n in names if n)}" if names else "")),
    })
    return activity, relevance, detail


def _summary(activity: int, relevance: int, detail: dict) -> str:
    if not detail.get("latest_push"):
        return "Profile exists but has no public repositories of its own."
    days = detail["days_since_latest_push"]
    recency = ("Recently active" if days <= 30 else "Active in the last 3 months" if days <= 90
               else "Some activity in the last year" if days <= 365 else "No public activity in the last year")
    relevant = detail["relevant_python_ai_repos"]
    repos = (f"{relevant} maintained Python/AI repositor{'y' if relevant == 1 else 'ies'}"
             if relevant else "no maintained Python/AI repositories")
    return f"{recency} (last push {days} day(s) ago); {repos}. Score {activity + relevance}/10."


class GitHubEnricher:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, cache: DiskCache | None = None,
                 *, now: datetime | None = None) -> None:
        self.settings = settings
        self._client = client
        self._cache = cache
        self._now = now
        self._memo: dict[str, tuple[str, list[dict] | None, str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._rate_limited = False
        self.token_rejected = False     # set when GitHub answers 401 to our token
        self.api_calls = 0

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "resume-screener",
        }
        if self.settings.github_token and not self.token_rejected:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        return headers

    async def _fetch(self, username: str) -> tuple[str, list[dict] | None, str]:
        """Return (status, repos, note). Status: ok | not_found | rate_limited | error."""
        key = username.lower()
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._memo:                       # in-run cache
                return self._memo[key]
            if self._cache:                             # disk cache
                cached = self._cache.get("github", key, self.settings.github_cache_ttl_hours * 3600)
                if cached and cached.get("status") in {"ok", "not_found"}:
                    self._memo[key] = (cached["status"], cached.get("repos"), "from cache")
                    return self._memo[key]
            if self._rate_limited:
                return "rate_limited", None, "skipped: GitHub rate limit already reached in this run"

            url = f"{self.settings.github_base_url.rstrip('/')}/users/{username}/repos"
            params = {"per_page": 100, "sort": "pushed", "type": "owner"}
            try:
                self.api_calls += 1
                headers = self._headers()
                response = await self._client.get(url, params=params, headers=headers,
                                                  timeout=self.settings.github_timeout_seconds)
                if response.status_code == 401 and "Authorization" in headers:
                    # A bad or expired token must not cost every candidate
                    # their GitHub score: carry on without it.
                    self.token_rejected = True
                    self.api_calls += 1
                    response = await self._client.get(url, params=params, headers=self._headers(),
                                                      timeout=self.settings.github_timeout_seconds)
            except httpx.TimeoutException:
                return "error", None, "GitHub request timed out"
            except httpx.HTTPError as exc:
                return "error", None, f"GitHub request failed: {type(exc).__name__}"

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    return "error", None, "GitHub returned invalid JSON"
                if not isinstance(payload, list):
                    return "error", None, "GitHub returned an unexpected response"
                repos = [{field: repo.get(field) for field in _REPO_FIELDS}
                         for repo in payload if isinstance(repo, dict)]
                result = ("ok", repos, "")
            elif response.status_code == 404:
                result = ("not_found", None, "GitHub user does not exist")
            elif response.status_code in (403, 429):
                remaining = response.headers.get("x-ratelimit-remaining")
                if remaining == "0" or response.status_code == 429 or "rate limit" in response.text.lower():
                    self._rate_limited = True
                    return "rate_limited", None, "GitHub API rate limit reached"
                return "error", None, f"GitHub refused the request (HTTP {response.status_code})"
            elif response.status_code == 401:
                return "error", None, "GitHub rejected the token (HTTP 401)"
            else:
                return "error", None, f"GitHub returned HTTP {response.status_code}"

            self._memo[key] = result
            if self._cache:
                self._cache.set("github", key, {"status": result[0], "repos": result[1]})
            return result

    async def enrich(self, usernames: list[str]) -> GitHubResult:
        """Validate the candidate usernames and score the profile.

        Most resumes give one username. When a resume points at more than one
        account (for example a link and a different URL in the text), each is
        checked and the more active one is used; all are recorded.
        """
        if not usernames:
            return GitHubResult(status="no_profile")
        tried: list[str] = []
        now = self._now or datetime.now(timezone.utc)
        best: GitHubResult | None = None
        failure: GitHubResult | None = None
        for username in usernames[: config.GITHUB_MAX_USERNAME_ATTEMPTS]:
            tried.append(username)
            status, repos, note = await self._fetch(username)
            if status == "ok":
                activity, relevance, detail = score_repos(repos or [], now)
                detail["checked_at"] = now.date().isoformat()
                result = GitHubResult(
                    status="ok", username=username, profile_url=f"https://github.com/{username}",
                    recent_activity_points=activity, relevant_repos_points=relevance,
                    summary=_summary(activity, relevance, detail), detail=detail,
                )
                if best is None or result.score > best.score:
                    best = result
            elif status != "not_found" and failure is None:
                failure = GitHubResult(
                    status=status, username=username, profile_url=f"https://github.com/{username}",
                    summary=f"GitHub could not be checked: {note}. Scored 0; eligibility unaffected.",
                    detail={"error": note},
                )
        chosen = best or failure
        if chosen is None:
            chosen = GitHubResult(
                status="not_found", username=tried[0], profile_url=f"https://github.com/{tried[0]}",
                summary=f"GitHub profile not found (tried: {', '.join(tried)}). Scored 0; eligibility unaffected.",
                detail={"error": "no such GitHub user"},
            )
        chosen.tried_usernames = tried
        if best is not None and len(tried) > 1:
            chosen.detail["note"] = (f"Resume points at more than one GitHub account ({', '.join(tried)}); "
                                     f"the more active one ({best.username}) was scored")
        return chosen
