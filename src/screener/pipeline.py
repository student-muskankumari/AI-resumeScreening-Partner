"""Batch orchestration.

    discover -> parse -> extract -> eligibility gate -> evidence (model/rules)
             -> GitHub enrichment -> score -> rank

Every resume is processed inside its own error boundary, so one unreadable
file, one failed model call or one GitHub error never stops the batch.
Network work (model calls, GitHub) runs concurrently with small, explicit
limits; everything else is plain synchronous code.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import extract
from .analysis import Analyzer
from .cache import DiskCache
from .config import Settings
from .eligibility import evaluate
from .github import GitHubEnricher
from .ingest import discover
from .llm.base import LLMProvider
from .llm.gemini import GeminiProvider
from .llm.groq import GroqProvider
from .llm.rules import build_evidence
from .models import CandidateResult, GitHubResult
from .parsing import UnreadableResume, parse_resume
from .scoring import assign_ranks, derive_notes, score_candidate
from .sections import TaggedLine, tag_lines

log = logging.getLogger("screener")


@dataclass
class BatchResult:
    candidates: list[CandidateResult]
    summary: dict
    input_dir: str = ""


@dataclass
class _Prepared:
    """A parsed resume waiting for the network stage."""

    result: CandidateResult
    text: str = ""
    lines: list[TaggedLine] = field(default_factory=list)
    github_usernames: list[str] = field(default_factory=list)


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(re.sub(r"\W+", "", text.lower()).encode("utf-8")).hexdigest()


def _prepare(input_dir: Path, settings: Settings) -> list[_Prepared]:
    """Local, deterministic stages: parse, extract and run the hard filter."""
    prepared: list[_Prepared] = []
    seen_text: dict[str, str] = {}
    seen_email: dict[str, str] = {}

    for resume_file in discover(input_dir):
        name = resume_file.path.name
        if resume_file.duplicate_of:
            prepared.append(_Prepared(CandidateResult(
                status="duplicate", source_file=name, file_hash=resume_file.file_hash,
                duplicate_of=resume_file.duplicate_of,
                warnings=[f"Identical file to {resume_file.duplicate_of}; skipped"],
            )))
            continue
        try:
            parsed = parse_resume(resume_file.path)
            parsed.file_hash = resume_file.file_hash

            fingerprint = _text_fingerprint(parsed.text)
            if fingerprint in seen_text:
                prepared.append(_Prepared(CandidateResult(
                    status="duplicate", source_file=name, file_hash=resume_file.file_hash,
                    duplicate_of=seen_text[fingerprint],
                    warnings=[f"Same content as {seen_text[fingerprint]}; skipped"],
                )))
                continue
            seen_text[fingerprint] = name

            lines = tag_lines(parsed.text)
            email = extract.extract_email(parsed)
            warnings = list(parsed.warnings)
            if email:
                if email.lower() in seen_email:
                    warnings.append(f"Same email address as {seen_email[email.lower()]}")
                else:
                    seen_email[email.lower()] = name
            eligibility = evaluate(lines, settings)
            result = CandidateResult(
                status="eligible" if eligibility.eligible else "rejected",
                source_file=name,
                file_hash=resume_file.file_hash,
                candidate_name=extract.extract_name(parsed, email),
                email=email,
                matched_skills=extract.extract_skills(parsed.text),
                eligibility=eligibility,
                parser=parsed.parser,
                page_count=parsed.page_count,
                warnings=warnings,
            )
            prepared.append(_Prepared(result, parsed.text, lines, extract.github_candidates(parsed)))
        except UnreadableResume as exc:
            log.warning("unreadable resume %s: %s", name, exc)
            prepared.append(_Prepared(CandidateResult(
                status="failed", source_file=name, file_hash=resume_file.file_hash, error=str(exc))))
        except Exception as exc:   # never let one file stop the batch
            log.exception("unexpected error while reading %s", name)
            prepared.append(_Prepared(CandidateResult(
                status="failed", source_file=name, file_hash=resume_file.file_hash,
                error=f"{type(exc).__name__}: {exc}")))
    return prepared


def build_providers(settings: Settings, client: httpx.AsyncClient) -> list[LLMProvider]:
    """Primary then fallback model. A provider without a key is left out."""
    providers: list[LLMProvider] = []
    if settings.groq_api_key:
        providers.append(GroqProvider(settings, client))
    if settings.gemini_api_key:
        providers.append(GeminiProvider(settings, client))
    return providers


async def run_batch(
    input_dir: Path,
    settings: Settings,
    *,
    use_llm: bool = True,
    use_github: bool = True,
    providers: list[LLMProvider] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> BatchResult:
    started = time.monotonic()
    prepared = _prepare(Path(input_dir), settings)
    eligible = [item for item in prepared if item.result.status == "eligible"]
    log.info("found %d file(s); %d eligible after the hard filter", len(prepared), len(eligible))

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=True)
    cache = DiskCache(settings.cache_dir, settings.use_disk_cache)
    try:
        if not use_llm:
            active_providers: list[LLMProvider] = []
        elif providers is not None:
            active_providers = providers
        else:
            active_providers = build_providers(settings, client)
        analyzer = Analyzer(settings, active_providers, cache)
        enricher = GitHubEnricher(settings, client, cache)
        llm_gate = asyncio.Semaphore(max(1, settings.llm_concurrency))
        github_gate = asyncio.Semaphore(max(1, settings.github_concurrency))

        async def analyse(item: _Prepared) -> None:
            try:
                async with llm_gate:
                    outcome = await analyzer.analyze(item.text, item.lines)
                item.result.evidence = outcome.evidence
                item.result.evidence_source = outcome.source
                item.result.warnings.extend(outcome.warnings)
            except Exception as exc:   # last line of defence: rules always work
                log.exception("analysis failed for %s", item.result.source_file)
                item.result.evidence = build_evidence(item.lines)
                item.result.evidence_source = "rules"
                item.result.warnings.append(f"analysis error ({type(exc).__name__}); used rules")

        async def enrich(item: _Prepared) -> None:
            if not use_github:
                item.result.github = GitHubResult(
                    status="skipped", username=(item.github_usernames or [None])[0],
                    summary="GitHub enrichment was turned off for this run.")
                return
            try:
                async with github_gate:
                    item.result.github = await enricher.enrich(item.github_usernames)
            except Exception as exc:
                log.exception("GitHub enrichment failed for %s", item.result.source_file)
                item.result.github = GitHubResult(
                    status="error", username=(item.github_usernames or [None])[0],
                    summary=f"GitHub could not be checked ({type(exc).__name__}). Scored 0; eligibility unaffected.")

        await asyncio.gather(*(analyse(item) for item in eligible), *(enrich(item) for item in eligible))
    finally:
        if owns_client:
            await client.aclose()

    for item in prepared:
        result = item.result
        if result.status == "rejected":
            username = (item.github_usernames or [None])[0]
            result.github = GitHubResult(
                status="skipped" if username else "no_profile",
                username=username,
                profile_url=f"https://github.com/{username}" if username else None,
                summary=("Not checked: the candidate did not pass the eligibility filter."
                         if username else "No GitHub profile found in the resume."),
            )
        if result.status != "eligible" or result.evidence is None or result.eligibility is None:
            continue
        result.score = score_candidate(result.evidence, result.github)
        result.project_summary = result.evidence.project_summary
        result.strengths, result.concerns = derive_notes(
            result.evidence, result.score, result.eligibility, result.github)
        if not result.candidate_name and result.evidence.candidate_name:
            result.candidate_name = result.evidence.candidate_name.strip()

    for item in prepared:
        if not item.result.candidate_name and item.result.status in {"eligible", "rejected"}:
            item.result.candidate_name = Path(item.result.source_file).stem

    assign_ranks([item.result for item in prepared])
    candidates = [item.result for item in prepared]

    def count(status: str) -> int:
        return sum(1 for c in candidates if c.status == status)

    sources: dict[str, int] = {}
    github_statuses: dict[str, int] = {}
    for candidate in candidates:
        if candidate.status == "eligible":
            sources[candidate.evidence_source or "rules"] = sources.get(candidate.evidence_source or "rules", 0) + 1
            github_statuses[candidate.github.status] = github_statuses.get(candidate.github.status, 0) + 1

    summary = {
        "total_resumes": len(candidates),
        "successfully_parsed": count("eligible") + count("rejected"),
        "eligible": count("eligible"),
        "rejected": count("rejected"),
        "failed_unreadable": count("failed"),
        "duplicates_skipped": count("duplicate"),
        "evidence_sources": sources,
        "llm_calls": analyzer.calls,
        "llm_providers_skipped": analyzer.disabled,
        "github_status_of_eligible": github_statuses,
        "github_api_calls": enricher.api_calls,
        "github_token_rejected": enricher.token_rejected,
        "policy": {
            "require_python_usage": settings.require_python_usage,
            "allow_classical_ml": settings.allow_classical_ml,
        },
        "run_seconds": round(time.monotonic() - started, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return BatchResult(candidates=candidates, summary=summary, input_dir=str(input_dir))
