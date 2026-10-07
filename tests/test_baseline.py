"""Regression check of the hard filter against the provided 50 resumes.

The expected outcomes come from reading every resume by hand. The test is
skipped when the resume folder is not present (the files hold personal data
and may not travel with the code).
"""

from pathlib import Path

import pytest

from screener.eligibility import evaluate
from screener.extract import extract_email, github_candidates
from screener.parsing import parse_resume
from screener.sections import tag_lines

RESUMES = Path(__file__).resolve().parents[1] / "resumes"
FILES = sorted(RESUMES.glob("candidate_*.pdf")) if RESUMES.exists() else []
pytestmark = pytest.mark.skipif(len(FILES) != 50, reason="the provided 50-resume set is not in ./resumes")

REJECTED = {1, 2, 3, 11, 18, 27, 31, 32, 42, 45, 46}
NO_PYTHON = {1, 2, 11, 18, 32, 42, 46}                 # includes strong-AI profiles with no Python
PYTHON_LISTED_ONLY = {9}
CLASSICAL_ML_ONLY = {4, 5, 14, 20, 21, 34, 48}
NO_GITHUB_USERNAME = {2, 5, 15, 34, 42, 43, 46}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    from screener.config import Settings
    settings = Settings(cache_dir=tmp_path_factory.mktemp("cache"))
    out = {}
    for path in FILES:
        parsed = parse_resume(path)
        number = int(path.stem.split("_")[1])
        out[number] = (parsed, evaluate(tag_lines(parsed.text), settings))
    return out


def test_all_fifty_parse_with_an_email(results):
    assert len(results) == 50
    assert all(extract_email(parsed) for parsed, _ in results.values())


def test_eligibility_matches_the_hand_checked_baseline(results):
    rejected = {n for n, (_, r) in results.items() if not r.eligible}
    assert rejected == REJECTED


def test_rejection_reasons(results):
    for number in NO_PYTHON:
        assert any(r.startswith("No evidence of Python stack") for r in results[number][1].rejection_reasons), number
    for number in REJECTED - NO_PYTHON:
        reasons = results[number][1].rejection_reasons
        assert len(reasons) == 1 and reasons[0].startswith("No AI/agentic project evidence"), number


def test_flags(results):
    eligible = {n: r for n, (_, r) in results.items() if r.eligible}
    assert {n for n, r in eligible.items() if "python_listed_only" in r.flags} == PYTHON_LISTED_ONLY
    assert {n for n, r in eligible.items() if "classical_ml_only" in r.flags} == CLASSICAL_ML_ONLY


def test_github_usernames(results):
    missing = {n for n, (parsed, _) in results.items() if not github_candidates(parsed)}
    assert missing == NO_GITHUB_USERNAME
    assert github_candidates(results[38][0]) == ["Pavani-A"]            # repo links only
    assert github_candidates(results[23][0])[0] == "annisha-atdoor"     # link and text disagree
    assert len(github_candidates(results[23][0])) == 2
