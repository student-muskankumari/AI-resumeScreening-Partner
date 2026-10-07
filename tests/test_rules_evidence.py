"""The rule-based fallback must fill the same schema the models fill."""

from screener.llm.rules import build_evidence
from screener.scoring import score_candidate
from screener.models import GitHubResult
from screener.sections import tag_lines

from conftest import STRONG_AGENTIC, THIN_WRAPPER, PYTHON_ONLY


def test_agentic_rag_resume_gets_depth_signals_with_quotes():
    evidence = build_evidence(tag_lines(STRONG_AGENTIC))
    assert evidence.ai.kind == "genai"
    for rule in ("retrieval", "agents_tools", "state_orchestration", "evaluation"):
        signal = getattr(evidence.ai, rule)
        assert signal.strength >= 1 and signal.evidence, rule
    assert evidence.ai.wrapper_severity == 0 and not evidence.ai.tutorial_style
    assert evidence.backend.python.level == "used"
    assert evidence.backend.fastapi.level == "used"
    assert evidence.backend.redis.level == "used"
    assert evidence.cloud.gcp.level == "used" and evidence.cloud.docker.level == "used"
    assert evidence.engineering.testing.level == "used"
    assert evidence.project_summary


def test_thin_wrapper_is_flagged_and_scores_far_below_the_agentic_resume():
    thin = build_evidence(tag_lines(THIN_WRAPPER))
    assert thin.ai.kind == "genai" and thin.ai.wrapper_severity >= 2 and thin.ai.wrapper_reason
    strong = build_evidence(tag_lines(STRONG_AGENTIC))
    thin_score = score_candidate(thin, GitHubResult())
    strong_score = score_candidate(strong, GitHubResult())
    assert any(p.type == "thin_llm_wrapper" for p in thin_score.penalties)
    assert strong_score.total_score - thin_score.total_score >= 40


def test_no_ai_resume_has_no_ai_evidence_and_no_penalty():
    evidence = build_evidence(tag_lines(PYTHON_ONLY))
    assert evidence.ai.kind == "none"
    result = score_candidate(evidence, GitHubResult())
    assert result.categories["ai_project_depth"].score == 0 and result.penalties == []
    assert result.categories["python_backend"].score >= 25


def test_skills_list_only_is_marked_listed_not_used():
    text = """Skills
Backend: Python, FastAPI, Redis, PostgreSQL, Docker
Projects
Notes | Java
- Built a notes app.
"""
    evidence = build_evidence(tag_lines(text))
    assert evidence.backend.python.level == "listed"
    assert evidence.backend.redis.level == "listed"
    assert evidence.cloud.docker.level == "listed"
