"""Output: results.json, the terminal summary and the `--explain` view."""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .models import CandidateResult
from .pipeline import BatchResult
from .scoring import RULE_LABELS, rubric_description

_BREAKDOWN_KEYS = ("ai_project_depth", "python_backend", "cloud_fullstack", "github", "engineering_depth")


def _eligibility_basis(candidate: CandidateResult) -> dict | None:
    e = candidate.eligibility
    if e is None:
        return None
    python_met = e.python_level != "absent" and not any("Python" in r for r in e.rejection_reasons)
    return {
        "rule": "Eligible only when BOTH Python evidence and AI/agentic evidence are present",
        "python": {"met": python_met, "level": e.python_level, "evidence": e.python_evidence},
        "ai": {"met": e.ai_kind != "none", "kind": e.ai_kind, "evidence": e.ai_evidence},
        "flags": e.flags,
    }


def _github_block(candidate: CandidateResult) -> dict:
    g = candidate.github
    return {
        "status": g.status,
        "username": g.username,
        "profile_url": g.profile_url,
        "score": g.score,
        "tried_usernames": g.tried_usernames,
        "detail": g.detail,
    }


def candidate_to_dict(candidate: CandidateResult) -> dict:
    base = {
        "rank": candidate.rank,
        "candidate_name": candidate.candidate_name,
        "source_file": candidate.source_file,
        "status": candidate.status,
        "email": candidate.email,
        "eligible": candidate.status == "eligible",
    }
    processing = {
        "parser": candidate.parser,
        "pages": candidate.page_count,
        "evidence_source": candidate.evidence_source,
        "warnings": candidate.warnings,
    }

    if candidate.status == "failed":
        return {**base, "rejection_reasons": [f"Resume could not be read: {candidate.error}"],
                "total_score": None, "processing": processing}
    if candidate.status == "duplicate":
        return {**base, "rejection_reasons": [f"Duplicate of {candidate.duplicate_of}; not processed again"],
                "total_score": None, "duplicate_of": candidate.duplicate_of, "processing": processing}

    eligibility = candidate.eligibility
    record = {
        **base,
        "rejection_reasons": eligibility.rejection_reasons if eligibility else [],
        "total_score": None,
        "matched_skills": candidate.matched_skills,
    }
    if candidate.status == "rejected" or candidate.score is None:
        record.update({
            "github_summary": candidate.github.summary,
            "github": _github_block(candidate),
            "grading_basis": {"eligibility": _eligibility_basis(candidate)},
            "processing": processing,
        })
        return record

    score = candidate.score
    record.update({
        "total_score": score.total_score,
        "score_breakdown": {key: score.categories[key].score for key in _BREAKDOWN_KEYS},
        "penalties": [p.model_dump() for p in score.penalties],
        "project_summary": candidate.project_summary,
        "github_summary": candidate.github.summary,
        "strengths": candidate.strengths,
        "concerns": candidate.concerns,
        "github": _github_block(candidate),
        "grading_basis": {
            "eligibility": _eligibility_basis(candidate),
            **{key: score.categories[key].model_dump(exclude_none=True) for key in _BREAKDOWN_KEYS},
            "penalties": [p.model_dump() for p in score.penalties],
            "base_score": score.base_score,
            "total_score": score.total_score,
            "evidence_source": candidate.evidence_source,
        },
        "processing": processing,
    })
    return record


def build_output(batch: BatchResult) -> dict:
    order = {"eligible": 0, "rejected": 1, "failed": 2, "duplicate": 3}
    ordered = sorted(
        batch.candidates,
        key=lambda c: (order[c.status], c.rank if c.rank is not None else 10**6, c.source_file.lower()),
    )
    return {
        "batch_summary": batch.summary,
        "scoring_rubric": rubric_description(),
        "candidates": [candidate_to_dict(c) for c in ordered],
    }


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Terminal views
# --------------------------------------------------------------------------
def format_summary(data: dict, top: int = 10) -> str:
    s = data["batch_summary"]
    lines = [
        "",
        "Batch summary",
        f"  Total resumes        {s['total_resumes']}",
        f"  Successfully parsed  {s['successfully_parsed']}",
        f"  Eligible             {s['eligible']}",
        f"  Rejected             {s['rejected']}",
        f"  Failed / unreadable  {s['failed_unreadable']}",
        f"  Duplicates skipped   {s['duplicates_skipped']}",
        f"  Evidence source      {s.get('evidence_sources', {})}",
        f"  GitHub (eligible)    {s.get('github_status_of_eligible', {})}",
        "",
    ]
    ranked = [c for c in data["candidates"] if c["eligible"]]
    if ranked:
        lines.append(f"Top {min(top, len(ranked))} of {len(ranked)} eligible candidates")
        lines.append(f"  {'#':>2}  {'Score':>5}  {'AI':>2} {'Py':>2} {'Cl':>2} {'GH':>2} {'En':>2}  {'Pen':>3}  Candidate (file)")
        for c in ranked[:top]:
            b = c["score_breakdown"]
            penalty = sum(p["points"] for p in c["penalties"])
            lines.append(
                f"  {c['rank']:>2}  {c['total_score']:>5}  {b['ai_project_depth']:>2} {b['python_backend']:>2} "
                f"{b['cloud_fullstack']:>2} {b['github']:>2} {b['engineering_depth']:>2}  {penalty:>3}  "
                f"{c['candidate_name']} ({c['source_file']})"
            )
        lines.append("  Columns: AI depth /40, Python+backend /30, Cloud /15, GitHub /10, Engineering /5, penalties")
        lines.append("")
    rejected = [c for c in data["candidates"] if c["status"] == "rejected"]
    if rejected:
        lines.append(f"Rejected ({len(rejected)})")
        for c in rejected:
            lines.append(f"  {c['candidate_name']} ({c['source_file']}): {'; '.join(c['rejection_reasons'])}")
        lines.append("")
    for c in data["candidates"]:
        if c["status"] in {"failed", "duplicate"}:
            lines.append(f"  {c['status'].upper()} {c['source_file']}: {'; '.join(c['rejection_reasons'])}")
    return "\n".join(lines)


def find_candidate(data: dict, query: str) -> dict | None:
    needle = query.strip().lower()
    for c in data["candidates"]:
        if c["source_file"].lower() == needle or Path(c["source_file"]).stem.lower() == needle:
            return c
    for c in data["candidates"]:
        if needle and needle in (c.get("candidate_name") or "").lower():
            return c
    return None


def format_explanation(candidate: dict) -> str:
    """How one candidate was graded, and on what basis."""
    lines = ["", f"{candidate.get('candidate_name')}  ({candidate['source_file']})"]
    basis = candidate.get("grading_basis") or {}
    eligibility = basis.get("eligibility")

    if candidate["status"] in {"failed", "duplicate"}:
        lines.append(f"  Status: {candidate['status']} - {'; '.join(candidate['rejection_reasons'])}")
        return "\n".join(lines)

    lines.append("  Eligibility (both conditions must be met)")
    if eligibility:
        for key, label in (("python", "Python evidence"), ("ai", "AI / agentic evidence")):
            part = eligibility[key]
            mark = "met    " if part["met"] else "NOT met"
            extra = part.get("level") or part.get("kind")
            lines.append(f"    {label:<22} {mark} [{extra}]  {part.get('evidence') or '-'}")
        if eligibility.get("flags"):
            lines.append(f"    Flags: {', '.join(eligibility['flags'])}")
    if not candidate["eligible"]:
        lines.append("  Result: REJECTED")
        for reason in candidate["rejection_reasons"]:
            lines.append(f"    - {reason}")
        return "\n".join(lines)

    lines.append(f"  Result: ELIGIBLE, rank {candidate['rank']}, total {candidate['total_score']}/100 "
                 f"(evidence from: {basis.get('evidence_source')})")
    titles = {
        "ai_project_depth": "AI / agentic / RAG project depth",
        "python_backend": "Python & backend engineering",
        "cloud_fullstack": "Cloud / deployment / full stack",
        "github": "GitHub activity",
        "engineering_depth": "Engineering depth signals",
    }
    for key in _BREAKDOWN_KEYS:
        category = basis[key]
        lines.append(f"  {titles[key]}: {category['score']}/{category['max']}")
        for item in category["items"]:
            label = RULE_LABELS.get(item["rule"], item["rule"].replace("_", " "))
            detail = item.get("evidence") or item.get("note") or "no evidence"
            if item.get("evidence") and item.get("note"):
                detail = f"{item['evidence']}  [{item['note']}]"
            lines.append(f"    {item['points']:>3}/{item['max']:<2} {label}: {detail}")
    if basis.get("penalties"):
        lines.append("  Penalties")
        for penalty in basis["penalties"]:
            lines.append(f"    {penalty['points']:>3}    {penalty['type']}: {penalty['reason']}")
    lines.append(f"  Total = {basis['base_score']} (categories) "
                 f"{sum(p['points'] for p in basis.get('penalties', [])):+d} (penalties) "
                 f"= {basis['total_score']}/100")
    lines.append(f"  GitHub: {candidate.get('github_summary')}")
    if candidate.get("strengths"):
        lines.append("  Strengths: " + "; ".join(candidate["strengths"]))
    if candidate.get("concerns"):
        lines.append("  Concerns:  " + "; ".join(candidate["concerns"]))
    return "\n".join(lines)


def weights_line() -> str:
    return ", ".join(f"{name} {points}" for name, points in config.CATEGORY_WEIGHTS.items())
