"""Candidate scoring (assignment section 4).

All arithmetic happens here, in code. The evidence source (model or rules)
only says what the resume shows; this module turns that into points using
the rubric in `config.py`. Every point awarded is recorded as a `ScoreItem`
with the rule, the points, the maximum and the resume line it relied on, so
a score can always be traced back to its basis.

Invariants (covered by tests):
  * each category's items add up to the category score
  * no category exceeds its weight
  * total = sum of categories + penalties, clamped to 0..100
"""

from __future__ import annotations

import math

from . import config
from .models import (
    AIEvidence,
    BackendEvidence,
    CandidateResult,
    CategoryScore,
    CloudEvidence,
    EligibilityResult,
    EngineeringEvidence,
    Evidence,
    GitHubResult,
    LevelSignal,
    Penalty,
    ScoreItem,
    ScoreResult,
)

RULE_LABELS: dict[str, str] = {
    "use_case": "Real AI system in a project or job",
    "implementation_depth": "AI/LLM implementation depth",
    "retrieval": "RAG / retrieval / vector search",
    "agents_tools": "Agents / tool calling",
    "state_orchestration": "State / orchestration",
    "data_pipeline": "Data / context pipeline",
    "evaluation": "Evaluation / reliability",
    "business_logic": "Real-world product or business logic",
    "python": "Python",
    "fastapi": "FastAPI",
    "async": "Async programming",
    "postgresql": "PostgreSQL",
    "redis": "Redis",
    "cloud_platform": "Cloud platform (GCP preferred)",
    "docker": "Docker",
    "deployment": "Deployment / CI-CD",
    "frontend_e2e": "React / Next.js in an end-to-end system",
    "testing": "Testing",
    "architecture": "Architecture",
    "caching": "Caching",
    "queues": "Queues / messaging",
    "observability": "Observability",
    "concurrency": "Concurrency",
    "failure_handling": "Failure handling",
    "recent_activity": "Recent public GitHub activity",
    "relevant_repos": "Maintained / relevant repositories",
}


def strength_points(max_points: int, strength: int) -> int:
    """0 -> nothing, 1 -> half (rounded up), 2 -> full."""
    if strength <= 0:
        return 0
    return max_points if strength >= 2 else math.ceil(max_points / 2)


def level_points(max_points: int, level: str) -> int:
    return int(math.floor(max_points * config.LEVEL_MULTIPLIER.get(level, 0.0) + 0.5))


def _category(name: str, items: list[ScoreItem]) -> CategoryScore:
    return CategoryScore(
        score=sum(item.points for item in items),
        max=config.CATEGORY_WEIGHTS[name],
        items=items,
    )


# --------------------------------------------------------------------------
# AI / agentic / RAG project depth (40)
# --------------------------------------------------------------------------
def score_ai(ai: AIEvidence) -> CategoryScore:
    items: list[ScoreItem] = []
    for rule, max_points in config.AI_RUBRIC.items():
        signal = getattr(ai, rule)
        strength = 0 if ai.kind == "none" else signal.strength
        points = strength_points(max_points, strength)
        items.append(ScoreItem(rule=rule, points=points, max=max_points,
                               evidence=signal.evidence if points else None))
    subtotal = sum(item.points for item in items)
    if ai.kind == "classical_ml" and subtotal > config.CLASSICAL_ML_AI_CAP:
        items.append(ScoreItem(
            rule="classical_ml_cap",
            points=config.CLASSICAL_ML_AI_CAP - subtotal,
            max=0,
            note=(f"AI evidence is classical ML only (no LLM, RAG, embeddings or agents); "
                  f"category capped at {config.CLASSICAL_ML_AI_CAP}"),
        ))
    return _category("ai_project_depth", items)


# --------------------------------------------------------------------------
# Python & backend engineering (30)
# --------------------------------------------------------------------------
def score_backend(backend: BackendEvidence) -> CategoryScore:
    rubric = config.BACKEND_RUBRIC
    python_in_use = "used" in {
        backend.python.level, backend.fastapi.level, backend.other_python_framework.level,
    }
    items: list[ScoreItem] = []

    python_level = "used" if python_in_use else backend.python.level
    python_note = None
    if python_level == "listed":
        python_note = "Listed as a skill only; no project or job shows it in use"
    elif python_level == "claimed":
        python_note = "Claimed in the summary; no project or job shows it in use"
    items.append(ScoreItem(rule="python", points=level_points(rubric["python"], python_level),
                           max=rubric["python"], evidence=backend.python.evidence, note=python_note))

    # FastAPI is the named framework; Django/Flask earn partial credit. The
    # better of the two options is used, never both.
    other = backend.other_python_framework
    fastapi_points = level_points(rubric["fastapi"], backend.fastapi.level)
    other_points = level_points(config.OTHER_PYTHON_FRAMEWORK_MAX, other.level)
    if fastapi_points >= other_points and fastapi_points > 0:
        items.append(ScoreItem(rule="fastapi", points=fastapi_points, max=rubric["fastapi"],
                               evidence=backend.fastapi.evidence,
                               note=None if backend.fastapi.level == "used" else f"FastAPI {backend.fastapi.level} only"))
    elif other_points > 0:
        note = ("No FastAPI; partial credit for Django/Flask" if backend.fastapi.level == "absent"
                else f"FastAPI {backend.fastapi.level} only; partial credit for Django/Flask in use")
        items.append(ScoreItem(rule="fastapi", points=other_points, max=rubric["fastapi"],
                               evidence=other.evidence, note=note))
    else:
        items.append(ScoreItem(rule="fastapi", points=0, max=rubric["fastapi"]))

    def stack_item(rule: str, signal: LevelSignal, fallback: LevelSignal | None = None,
                   fallback_max: int = 0, fallback_note: str = "") -> ScoreItem:
        max_points = rubric[rule]
        note = None
        if signal.level != "absent":
            points, evidence = level_points(max_points, signal.level), signal.evidence
            if signal.level != "used":
                note = f"{signal.level.capitalize()} only"
        elif fallback is not None and fallback.level != "absent":
            points, evidence, note = level_points(fallback_max, fallback.level), fallback.evidence, fallback_note
        else:
            points, evidence = 0, None
        if points and not python_in_use:
            points = int(points * config.NON_PYTHON_BACKEND_FACTOR)
            note = ((note + "; ") if note else "") + "shown outside a Python stack, counted at half"
        return ScoreItem(rule=rule, points=points, max=max_points, evidence=evidence if points else None, note=note)

    items.append(stack_item("async", backend.async_programming))
    items.append(stack_item("postgresql", backend.postgresql, backend.other_sql,
                            config.OTHER_SQL_MAX, "No PostgreSQL; partial credit for another SQL database"))
    items.append(stack_item("redis", backend.redis))
    return _category("python_backend", items)


# --------------------------------------------------------------------------
# Cloud / deployment / full stack (15)
# --------------------------------------------------------------------------
def score_cloud(cloud: CloudEvidence) -> CategoryScore:
    rubric = config.CLOUD_RUBRIC
    items: list[ScoreItem] = []
    gcp_points = level_points(rubric["cloud_platform"], cloud.gcp.level)
    other_points = level_points(config.OTHER_CLOUD_MAX, cloud.other_cloud.level)
    if gcp_points >= other_points and gcp_points > 0:
        items.append(ScoreItem(rule="cloud_platform", points=gcp_points, max=rubric["cloud_platform"],
                               evidence=cloud.gcp.evidence,
                               note=None if cloud.gcp.level == "used" else f"GCP {cloud.gcp.level} only"))
    elif other_points > 0:
        items.append(ScoreItem(rule="cloud_platform", points=other_points, max=rubric["cloud_platform"],
                               evidence=cloud.other_cloud.evidence,
                               note="No GCP; partial credit for another cloud or hosting platform"))
    else:
        items.append(ScoreItem(rule="cloud_platform", points=0, max=rubric["cloud_platform"]))

    for rule, signal in (("docker", cloud.docker), ("deployment", cloud.deployment),
                         ("frontend_e2e", cloud.frontend_e2e)):
        points = level_points(rubric[rule], signal.level)
        note = None if signal.level in {"used", "absent"} else f"{signal.level.capitalize()} only"
        items.append(ScoreItem(rule=rule, points=points, max=rubric[rule],
                               evidence=signal.evidence if points else None, note=note))
    return _category("cloud_fullstack", items)


# --------------------------------------------------------------------------
# Engineering depth (5)
# --------------------------------------------------------------------------
def score_engineering(engineering: EngineeringEvidence) -> CategoryScore:
    items: list[ScoreItem] = []
    awarded = 0
    for rule in config.ENGINEERING_SIGNALS:
        signal: LevelSignal = getattr(engineering, rule)
        shown = signal.level == "used"
        points = 1 if shown and awarded < config.ENGINEERING_MAX else 0
        awarded += points
        note = None
        if shown and not points:
            note = "Shown, but the category cap was already reached"
        elif signal.level in {"claimed", "listed"}:
            note = f"{signal.level.capitalize()} only; needs a project or job to count"
        items.append(ScoreItem(rule=rule, points=points, max=1,
                               evidence=signal.evidence if shown else None, note=note))
    return _category("engineering_depth", items)


# --------------------------------------------------------------------------
# GitHub (10)
# --------------------------------------------------------------------------
def score_github(github: GitHubResult) -> CategoryScore:
    rubric = config.GITHUB_RUBRIC
    note = None if github.status == "ok" else f"GitHub status: {github.status}"
    items = [
        ScoreItem(rule="recent_activity", points=min(github.recent_activity_points, rubric["recent_activity"]),
                  max=rubric["recent_activity"], evidence=github.detail.get("activity_basis"), note=note),
        ScoreItem(rule="relevant_repos", points=min(github.relevant_repos_points, rubric["relevant_repos"]),
                  max=rubric["relevant_repos"], evidence=github.detail.get("relevance_basis"), note=note),
    ]
    return _category("github", items)


# --------------------------------------------------------------------------
# Project-quality penalties
# --------------------------------------------------------------------------
def compute_penalties(ai: AIEvidence) -> list[Penalty]:
    penalties: list[Penalty] = []
    if ai.kind == "genai" and ai.wrapper_severity > 0:
        penalties.append(Penalty(
            type="thin_llm_wrapper",
            points=-config.WRAPPER_PENALTY[ai.wrapper_severity],
            reason=ai.wrapper_reason or "AI project is a thin wrapper around an LLM/API call",
        ))
    if ai.kind != "none" and ai.tutorial_style:
        penalties.append(Penalty(
            type="tutorial_style_project",
            points=-config.TUTORIAL_PENALTY,
            reason=ai.tutorial_reason or "AI project is listed without implementation detail or ownership evidence",
        ))
    # Keep the combined deduction inside the 5-15 band the assignment sets.
    excess = -sum(p.points for p in penalties) - config.MAX_TOTAL_PENALTY
    if excess > 0 and penalties:
        last = penalties[-1]
        penalties[-1] = Penalty(type=last.type, points=last.points + excess, reason=last.reason)
        penalties = [p for p in penalties if p.points < 0]
    return penalties


def score_candidate(evidence: Evidence, github: GitHubResult) -> ScoreResult:
    categories = {
        "ai_project_depth": score_ai(evidence.ai),
        "python_backend": score_backend(evidence.backend),
        "cloud_fullstack": score_cloud(evidence.cloud),
        "github": score_github(github),
        "engineering_depth": score_engineering(evidence.engineering),
    }
    penalties = compute_penalties(evidence.ai)
    base = sum(category.score for category in categories.values())
    total = max(0, min(100, base + sum(p.points for p in penalties)))
    return ScoreResult(categories=categories, penalties=penalties, base_score=base, total_score=total)


# --------------------------------------------------------------------------
# Ranking and human-readable notes
# --------------------------------------------------------------------------
def ranking_key(candidate: CandidateResult) -> tuple:
    score = candidate.score
    assert score is not None
    cats = score.categories
    return (
        -score.total_score,
        -cats["ai_project_depth"].score,
        -cats["python_backend"].score,
        -cats["engineering_depth"].score,
        -cats["github"].score,
        candidate.source_file.lower(),
    )


def assign_ranks(candidates: list[CandidateResult]) -> list[CandidateResult]:
    """Rank eligible candidates, highest score first, with stable tie-breaks."""
    eligible = sorted((c for c in candidates if c.status == "eligible" and c.score), key=ranking_key)
    for position, candidate in enumerate(eligible, start=1):
        candidate.rank = position
    return eligible


def derive_notes(evidence: Evidence, score: ScoreResult, eligibility: EligibilityResult,
                 github: GitHubResult) -> tuple[list[str], list[str]]:
    """Short strengths and concerns, each tied to a scored rule."""
    strengths: list[str] = []
    concerns: list[str] = []

    for name in ("ai_project_depth", "python_backend", "cloud_fullstack"):
        for item in score.categories[name].items:
            if item.max and item.points == item.max and item.rule in RULE_LABELS:
                strengths.append(RULE_LABELS[item.rule])
    engineering = [RULE_LABELS[i.rule].lower() for i in score.categories["engineering_depth"].items if i.points]
    if len(engineering) >= 3:
        strengths.append("Engineering depth: " + ", ".join(engineering))
    if score.categories["github"].score >= 7:
        strengths.append("Active GitHub with relevant repositories")

    if "python_listed_only" in eligibility.flags or "python_claimed_only" in eligibility.flags:
        concerns.append("Python is named as a skill but no project or job shows it in use")
    if evidence.ai.kind == "classical_ml":
        concerns.append("AI work is classical ML only; no LLM, RAG or agentic implementation")
    for penalty in score.penalties:
        concerns.append(penalty.reason)
    by_rule = {item.rule: item for cat in score.categories.values() for item in cat.items}
    if evidence.ai.kind == "genai":
        for rule, text in (("retrieval", "No retrieval/RAG evidence"),
                           ("agents_tools", "No agent or tool-calling evidence"),
                           ("evaluation", "No evaluation evidence")):
            if by_rule[rule].points == 0:
                concerns.append(text)
    for rule, text in (("fastapi", "No FastAPI evidence"), ("postgresql", "No PostgreSQL evidence"),
                       ("redis", "No Redis evidence"), ("docker", "No Docker evidence")):
        if by_rule[rule].points == 0:
            concerns.append(text)
    if github.status == "no_profile":
        concerns.append("No GitHub profile in the resume")
    elif github.status != "ok":
        concerns.append(f"GitHub could not be checked ({github.status})")
    elif score.categories["github"].score <= 2:
        concerns.append("Little recent or relevant public GitHub activity")

    model_strengths = [s.strip() for s in evidence.strengths if isinstance(s, str) and s.strip()]
    model_concerns = [c.strip() for c in evidence.concerns if isinstance(c, str) and c.strip()]
    strengths = _dedupe(model_strengths[:3] + strengths)[:5]
    concerns = _dedupe(concerns + model_concerns[:2])[:6]
    return strengths, concerns


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def rubric_description() -> dict:
    """The published grading basis, written once into the output file."""
    return {
        "total": 100,
        "categories": {
            "ai_project_depth": {
                "max": config.CATEGORY_WEIGHTS["ai_project_depth"],
                "how": "Each rule is graded 0 (no evidence), 1 (mentioned or basic: half points, "
                       "rounded up) or 2 (clearly implemented: full points).",
                "rules": {rule: {"max": pts, "meaning": RULE_LABELS[rule]} for rule, pts in config.AI_RUBRIC.items()},
                "cap": f"Classical-ML-only evidence is capped at {config.CLASSICAL_ML_AI_CAP}.",
            },
            "python_backend": {
                "max": config.CATEGORY_WEIGHTS["python_backend"],
                "how": "Full points when the technology is used in a project or job, 60% when only "
                       "claimed in a summary, one third when only listed as a skill.",
                "rules": {rule: {"max": pts, "meaning": RULE_LABELS[rule]} for rule, pts in config.BACKEND_RUBRIC.items()},
                "partial_credit": {
                    "django_or_flask_without_fastapi": config.OTHER_PYTHON_FRAMEWORK_MAX,
                    "other_sql_without_postgresql": config.OTHER_SQL_MAX,
                    "backend_items_outside_a_python_stack": f"x{config.NON_PYTHON_BACKEND_FACTOR}",
                },
            },
            "cloud_fullstack": {
                "max": config.CATEGORY_WEIGHTS["cloud_fullstack"],
                "how": "Same used / claimed / listed scale as the backend category.",
                "rules": {rule: {"max": pts, "meaning": RULE_LABELS[rule]} for rule, pts in config.CLOUD_RUBRIC.items()},
                "partial_credit": {"other_cloud_without_gcp": config.OTHER_CLOUD_MAX},
            },
            "github": {
                "max": config.CATEGORY_WEIGHTS["github"],
                "how": "0-5 for recent public activity plus 0-5 for maintained Python/AI repositories. "
                       "Missing, private or unreachable GitHub scores 0 and never affects eligibility.",
                "rules": {rule: {"max": pts, "meaning": RULE_LABELS[rule]} for rule, pts in config.GITHUB_RUBRIC.items()},
            },
            "engineering_depth": {
                "max": config.CATEGORY_WEIGHTS["engineering_depth"],
                "how": "One point per distinct signal shown in a project or job, capped at the category maximum.",
                "rules": {rule: {"max": 1, "meaning": RULE_LABELS[rule]} for rule in config.ENGINEERING_SIGNALS},
            },
        },
        "penalties": {
            "thin_llm_wrapper": "Minus 5, 10 or 15 when the AI project is mostly a model/API call with little "
                                "workflow, retrieval, state, data processing or evaluation around it.",
            "tutorial_style_project": f"Minus {config.TUTORIAL_PENALTY} when AI work is listed without "
                                      "implementation detail or evidence of ownership.",
            "combined_cap": config.MAX_TOTAL_PENALTY,
        },
        "ranking": "Total score descending; ties broken by AI depth, Python/backend, engineering depth, "
                   "GitHub, then file name.",
        "formula": "total = clamp(sum of category scores + penalties, 0, 100)",
    }
