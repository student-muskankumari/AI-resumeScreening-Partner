"""Scoring invariants, priorities and penalties."""

import itertools

import pytest

from screener import config
from screener.models import (
    AIEvidence, BackendEvidence, CandidateResult, CloudEvidence, EngineeringEvidence, Evidence,
    GitHubResult, LevelSignal, StrengthSignal,
)
from screener.scoring import assign_ranks, compute_penalties, score_candidate


def strong(level="used"):
    return LevelSignal(level=level, evidence="quote")


def ai(kind="genai", wrapper=0, tutorial=False, **strengths):
    fields = {rule: StrengthSignal(strength=strengths.get(rule, 0), evidence="quote" if strengths.get(rule) else None)
              for rule in config.AI_RUBRIC}
    return AIEvidence(kind=kind, wrapper_severity=wrapper, tutorial_style=tutorial, **fields)


def full_ai(kind="genai"):
    return ai(kind, **{rule: 2 for rule in config.AI_RUBRIC})


FULL_BACKEND = BackendEvidence(python=strong(), fastapi=strong(), async_programming=strong(),
                               postgresql=strong(), redis=strong())
FULL_CLOUD = CloudEvidence(gcp=strong(), docker=strong(), deployment=strong(), frontend_e2e=strong())
FULL_ENGINEERING = EngineeringEvidence(**{name: strong() for name in config.ENGINEERING_SIGNALS})
FULL_GITHUB = GitHubResult(status="ok", recent_activity_points=5, relevant_repos_points=5)


def evidence(ai_part, backend=None, cloud=None, engineering=None):
    return Evidence(ai=ai_part, backend=backend or BackendEvidence(), cloud=cloud or CloudEvidence(),
                    engineering=engineering or EngineeringEvidence())


def test_perfect_candidate_scores_exactly_100():
    result = score_candidate(evidence(full_ai(), FULL_BACKEND, FULL_CLOUD, FULL_ENGINEERING), FULL_GITHUB)
    assert result.total_score == 100
    assert {k: v.score for k, v in result.categories.items()} == config.CATEGORY_WEIGHTS


def test_empty_evidence_scores_zero():
    result = score_candidate(Evidence(), GitHubResult())
    assert result.total_score == 0 and result.penalties == []


CASES = [
    evidence(full_ai(), FULL_BACKEND, FULL_CLOUD, FULL_ENGINEERING),
    evidence(full_ai("classical_ml"), FULL_BACKEND),
    evidence(ai(use_case=2, implementation_depth=1, wrapper=3, tutorial=True)),
    evidence(ai(retrieval=1, agents_tools=2, evaluation=1),
             BackendEvidence(python=strong("listed"), postgresql=strong(), redis=strong("claimed")),
             CloudEvidence(other_cloud=strong(), docker=strong("listed")),
             EngineeringEvidence(testing=strong(), caching=strong("listed"))),
    evidence(ai("none"), BackendEvidence(python=strong(), other_python_framework=strong(), other_sql=strong())),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("github", [GitHubResult(), FULL_GITHUB,
                                    GitHubResult(status="rate_limited"),
                                    GitHubResult(status="ok", recent_activity_points=3, relevant_repos_points=2)])
def test_scores_always_add_up(case, github):
    result = score_candidate(case, github)
    for name, category in result.categories.items():
        assert category.score == sum(item.points for item in category.items)
        assert 0 <= category.score <= config.CATEGORY_WEIGHTS[name] == category.max
    assert result.base_score == sum(c.score for c in result.categories.values())
    assert result.total_score == max(0, min(100, result.base_score + sum(p.points for p in result.penalties)))
    assert all(p.points < 0 for p in result.penalties)
    assert -sum(p.points for p in result.penalties) <= config.MAX_TOTAL_PENALTY


def test_listed_scores_below_claimed_below_used():
    def backend_score(level):
        return score_candidate(evidence(ai("none"), BackendEvidence(
            python=strong(level), fastapi=strong(level), postgresql=strong(level), redis=strong(level),
            async_programming=strong(level))), GitHubResult()).categories["python_backend"].score
    assert backend_score("listed") < backend_score("claimed") < backend_score("used") == 30


def test_framework_in_skills_section_never_earns_full_points():
    listed = score_candidate(evidence(ai("none"), BackendEvidence(python=strong(), fastapi=strong("listed"))),
                             GitHubResult())
    item = next(i for i in listed.categories["python_backend"].items if i.rule == "fastapi")
    assert 0 < item.points < config.BACKEND_RUBRIC["fastapi"]


def test_backend_outside_a_python_stack_counts_for_half():
    java = BackendEvidence(python=strong("listed"), postgresql=strong(), redis=strong(), async_programming=strong())
    python = BackendEvidence(python=strong(), postgresql=strong(), redis=strong(), async_programming=strong())
    java_score = score_candidate(evidence(ai("none"), java), GitHubResult()).categories["python_backend"].score
    python_score = score_candidate(evidence(ai("none"), python), GitHubResult()).categories["python_backend"].score
    assert java_score < python_score / 2


def test_gcp_beats_other_clouds_and_django_gets_partial_fastapi_credit():
    gcp = score_candidate(evidence(ai("none"), cloud=CloudEvidence(gcp=strong())), GitHubResult())
    aws = score_candidate(evidence(ai("none"), cloud=CloudEvidence(other_cloud=strong())), GitHubResult())
    assert gcp.categories["cloud_fullstack"].score == 5 > aws.categories["cloud_fullstack"].score == 3
    django = score_candidate(evidence(ai("none"), BackendEvidence(python=strong(), other_python_framework=strong())),
                             GitHubResult())
    fastapi_item = next(i for i in django.categories["python_backend"].items if i.rule == "fastapi")
    assert fastapi_item.points == config.OTHER_PYTHON_FRAMEWORK_MAX


def test_project_quality_ordering_wrapper_below_rag_below_agentic_rag():
    wrapper = score_candidate(evidence(ai(use_case=2, implementation_depth=1, wrapper=2)), GitHubResult())
    rag = score_candidate(evidence(ai(use_case=2, implementation_depth=1, retrieval=2, data_pipeline=1)), GitHubResult())
    agentic = score_candidate(evidence(ai(use_case=2, implementation_depth=2, retrieval=2, agents_tools=2,
                                          state_orchestration=2, evaluation=2)), GitHubResult())
    assert wrapper.total_score < rag.total_score < agentic.total_score


@pytest.mark.parametrize("severity, points", [(0, 0), (1, -5), (2, -10), (3, -15)])
def test_thin_wrapper_penalty_is_between_5_and_15(severity, points):
    penalties = compute_penalties(ai(use_case=2, wrapper=severity))
    assert sum(p.points for p in penalties) == points
    if severity:
        assert penalties[0].type == "thin_llm_wrapper" and penalties[0].reason


def test_tutorial_penalty_applies_and_combined_penalty_is_capped():
    assert [p.points for p in compute_penalties(ai(use_case=1, tutorial=True))] == [-config.TUTORIAL_PENALTY]
    combined = compute_penalties(ai(use_case=2, wrapper=3, tutorial=True))
    assert sum(p.points for p in combined) == -config.MAX_TOTAL_PENALTY


def test_no_wrapper_penalty_without_genai_work():
    assert compute_penalties(ai("classical_ml", wrapper=3)) == []
    assert compute_penalties(ai("none", wrapper=3, tutorial=True)) == []


def test_classical_ml_is_capped_and_the_cap_is_visible():
    result = score_candidate(evidence(full_ai("classical_ml")), GitHubResult())
    category = result.categories["ai_project_depth"]
    assert category.score == config.CLASSICAL_ML_AI_CAP
    assert any(item.rule == "classical_ml_cap" and item.points < 0 for item in category.items)


def test_strong_python_without_meaningful_ai_cannot_rank_near_the_top():
    python_star = score_candidate(evidence(ai("classical_ml", use_case=1, implementation_depth=1),
                                           FULL_BACKEND, FULL_CLOUD, FULL_ENGINEERING), FULL_GITHUB)
    ai_builder = score_candidate(evidence(full_ai(), BackendEvidence(python=strong(), fastapi=strong())),
                                 GitHubResult())
    assert python_star.categories["ai_project_depth"].score <= config.CLASSICAL_ML_AI_CAP
    assert python_star.total_score <= 100 - (40 - config.CLASSICAL_ML_AI_CAP)
    mid_ai = score_candidate(evidence(full_ai(), FULL_BACKEND), GitHubResult())
    assert mid_ai.total_score > python_star.total_score
    assert ai_builder.categories["ai_project_depth"].score == 40


def test_github_failure_scores_zero_but_does_not_break_scoring():
    for status in ("no_profile", "not_found", "rate_limited", "error"):
        result = score_candidate(evidence(full_ai()), GitHubResult(status=status))
        assert result.categories["github"].score == 0 and result.total_score == 40


def test_engineering_depth_is_capped_at_five_signals():
    result = score_candidate(evidence(ai("none"), engineering=FULL_ENGINEERING), GitHubResult())
    assert result.categories["engineering_depth"].score == 5
    listed_only = EngineeringEvidence(testing=strong("listed"), caching=strong("claimed"))
    assert score_candidate(evidence(ai("none"), engineering=listed_only), GitHubResult()).total_score == 0


def candidate(name, total_evidence, github=None):
    c = CandidateResult(status="eligible", source_file=name)
    c.score = score_candidate(total_evidence, github or GitHubResult())
    return c


def test_ranking_is_deterministic_with_tie_breaks():
    a = candidate("b_file.pdf", evidence(ai(use_case=2, retrieval=2)))                        # 11, AI 11
    b = candidate("a_file.pdf", evidence(ai(use_case=2, retrieval=2)))                        # same score, earlier name
    c = candidate("c_file.pdf", evidence(ai(use_case=2), BackendEvidence(python=strong(level="claimed"))))  # 11, AI 5
    d = candidate("d_file.pdf", evidence(full_ai()))
    rejected = CandidateResult(status="rejected", source_file="z.pdf")
    for order in itertools.permutations([a, b, c, d, rejected]):
        ranked = assign_ranks(list(order))
        assert [x.source_file for x in ranked] == ["d_file.pdf", "a_file.pdf", "b_file.pdf", "c_file.pdf"]
        assert [x.rank for x in ranked] == [1, 2, 3, 4]
    assert rejected.rank is None
