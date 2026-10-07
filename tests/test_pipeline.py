"""End-to-end runs on a small synthetic resume folder (no network)."""

import asyncio
import json
import shutil

import httpx
import pytest

import main as cli
from screener import report
from screener.llm.base import LLMProvider, ProviderError
from screener.pipeline import run_batch

from conftest import (AI_NO_PYTHON, FRONTEND_ONLY, PYTHON_ONLY, STRONG_AGENTIC, THIN_WRAPPER, make_pdf)


@pytest.fixture
def resume_dir(tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    make_pdf(folder / "strong_agentic.pdf", "ASHA RAO", STRONG_AGENTIC)
    make_pdf(folder / "python_only.pdf", "Ravi Kumar", PYTHON_ONLY)
    make_pdf(folder / "ai_no_python.pdf", "Meera Joshi", AI_NO_PYTHON)
    make_pdf(folder / "frontend_only.pdf", "Dev Shah", FRONTEND_ONLY)
    make_pdf(folder / "thin_wrapper.pdf", "Kiran T", THIN_WRAPPER)
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01\x02")
    make_pdf(folder / "empty.pdf", "", "")
    shutil.copy(folder / "strong_agentic.pdf", folder / "strong_agentic_copy.pdf")
    (folder / "notes.xyz").write_text("not a resume")
    (folder / "plain.txt").write_text("Lena Park\nlena@example.com\n" + STRONG_AGENTIC, encoding="utf-8")
    return folder


def run(folder, settings, **kwargs):
    kwargs.setdefault("use_llm", False)
    kwargs.setdefault("use_github", False)
    return asyncio.run(run_batch(folder, settings, **kwargs))


def by_file(batch):
    return {c.source_file: c for c in batch.candidates}


def test_batch_survives_bad_files_and_counts_add_up(resume_dir, settings):
    batch = run(resume_dir, settings)
    files = by_file(batch)
    assert set(files) == {"strong_agentic.pdf", "python_only.pdf", "ai_no_python.pdf", "frontend_only.pdf",
                          "thin_wrapper.pdf", "broken.pdf", "empty.pdf", "strong_agentic_copy.pdf", "plain.txt"}
    assert files["broken.pdf"].status == "failed" and files["broken.pdf"].error
    assert files["empty.pdf"].status == "failed"
    assert files["strong_agentic_copy.pdf"].status == "duplicate"
    assert files["strong_agentic_copy.pdf"].duplicate_of == "strong_agentic.pdf"

    s = batch.summary
    assert s["total_resumes"] == 9
    assert s["eligible"] + s["rejected"] == s["successfully_parsed"]
    assert s["successfully_parsed"] + s["failed_unreadable"] + s["duplicates_skipped"] == s["total_resumes"]
    assert (s["eligible"], s["rejected"], s["failed_unreadable"], s["duplicates_skipped"]) == (3, 3, 2, 1)


def test_filter_and_ranking_outcomes(resume_dir, settings):
    files = by_file(run(resume_dir, settings))
    assert files["strong_agentic.pdf"].status == "eligible"
    assert files["thin_wrapper.pdf"].status == "eligible"
    assert files["plain.txt"].status == "eligible"                 # TXT bonus format
    for name in ("python_only.pdf", "ai_no_python.pdf", "frontend_only.pdf"):
        assert files[name].status == "rejected" and files[name].eligibility.rejection_reasons
        assert files[name].score is None and files[name].rank is None

    strong, thin = files["strong_agentic.pdf"], files["thin_wrapper.pdf"]
    assert strong.rank < thin.rank and strong.score.total_score > thin.score.total_score + 30
    assert any(p.type == "thin_llm_wrapper" for p in thin.score.penalties)
    assert strong.candidate_name == "Asha Rao" and strong.email == "asha.rao@example.com"
    assert {"Python", "FastAPI", "LangGraph", "RAG"} <= set(strong.matched_skills)
    assert strong.evidence_source == "rules"


def test_output_has_the_fields_the_assignment_asks_for(resume_dir, settings):
    data = report.build_output(run(resume_dir, settings))
    assert set(data) == {"batch_summary", "scoring_rubric", "candidates"}
    top = data["candidates"][0]
    for key in ("rank", "candidate_name", "eligible", "total_score", "score_breakdown", "matched_skills",
                "project_summary", "github_summary", "strengths", "concerns", "grading_basis"):
        assert key in top, key
    assert top["rank"] == 1 and top["eligible"] is True
    assert set(top["score_breakdown"]) == {"ai_project_depth", "python_backend", "cloud_fullstack",
                                           "github", "engineering_depth"}
    assert sum(top["score_breakdown"].values()) + sum(p["points"] for p in top["penalties"]) == top["total_score"]

    # eligible first by rank, highest score first
    ranked = [c for c in data["candidates"] if c["eligible"]]
    assert [c["rank"] for c in ranked] == list(range(1, len(ranked) + 1))
    assert [c["total_score"] for c in ranked] == sorted((c["total_score"] for c in ranked), reverse=True)

    rejected = next(c for c in data["candidates"] if c["source_file"] == "ai_no_python.pdf")
    assert rejected["eligible"] is False and rejected["rejection_reasons"] == ["No evidence of Python stack"]
    assert rejected["matched_skills"] and rejected["total_score"] is None

    basis = top["grading_basis"]
    assert basis["eligibility"]["python"]["met"] and basis["eligibility"]["ai"]["met"]
    item = basis["ai_project_depth"]["items"][0]
    assert {"rule", "points", "max"} <= set(item)
    json.dumps(data)   # serialisable


def test_explain_view_shows_rules_points_and_evidence(resume_dir, settings):
    data = report.build_output(run(resume_dir, settings))
    text = report.format_explanation(report.find_candidate(data, "strong_agentic.pdf"))
    assert "ELIGIBLE" in text and "RAG / retrieval / vector search" in text and "Total =" in text
    rejected = report.format_explanation(report.find_candidate(data, "Meera"))
    assert "REJECTED" in rejected and "No evidence of Python stack" in rejected
    assert report.find_candidate(data, "nobody") is None


class DownProvider(LLMProvider):
    name, model = "groq", "m"

    async def complete_json(self, system, user):
        raise ProviderError("service unavailable")


def test_model_outage_and_github_outage_do_not_fail_the_batch(resume_dir, settings):
    def github_down(request):
        return httpx.Response(503)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(github_down)) as client:
            return await run_batch(resume_dir, settings, use_llm=True, use_github=True,
                                   providers=[DownProvider()], http_client=client)
    batch = asyncio.run(go())
    strong = by_file(batch)["strong_agentic.pdf"]
    assert strong.status == "eligible" and strong.evidence_source == "rules"
    assert strong.github.status == "error" and strong.score.categories["github"].score == 0
    assert any("service unavailable" in w for w in strong.warnings)
    assert batch.summary["eligible"] == 3


class WorkingProvider(LLMProvider):
    """Stands in for a healthy model: real quotes, plus one invented one."""

    name, model = "groq", "m"

    def __init__(self):
        self.calls = 0

    async def complete_json(self, system, user):
        self.calls += 1
        return json.dumps({
            "candidate_name": "Someone",
            "project_summary": "LangGraph multi-agent workflow with RAG.",
            "ai": {
                "kind": "genai",
                "use_case": {"strength": 2, "evidence": "Built a stateful LangGraph multi-agent workflow with tool calling"},
                "agents_tools": {"strength": 2, "evidence": "LangGraph multi-agent workflow with tool calling and a supervisor"},
                "evaluation": {"strength": 2, "evidence": "Published a peer reviewed benchmark at NeurIPS"},
            },
            "backend": {"python": {"level": "used", "evidence": "Developed an async FastAPI backend with PostgreSQL"}},
            "strengths": ["Agentic workflow in production"],
        })


def test_model_evidence_flows_into_scores_and_only_eligible_resumes_are_sent(resume_dir, settings):
    provider = WorkingProvider()
    batch = run(resume_dir, settings, use_llm=True, providers=[provider])
    files = by_file(batch)
    strong = files["strong_agentic.pdf"]
    assert strong.evidence_source == "groq" and batch.summary["evidence_sources"] == {"groq": 3}
    assert provider.calls == 3                      # rejected, failed and duplicate files never reach the model
    items = {i.rule: i for i in strong.score.categories["ai_project_depth"].items}
    assert items["agents_tools"].points == 6        # real quote: full points
    assert items["evaluation"].points == 0          # invented quote: dropped
    assert "Agentic workflow in production" in strong.strengths
    assert files["python_only.pdf"].status == "rejected"     # the gate is unaffected by the model

    # second run is served from the disk cache
    again = WorkingProvider()
    run(resume_dir, settings, use_llm=True, providers=[again])
    assert again.calls == 0


def test_github_enrichment_feeds_the_score(resume_dir, settings):
    from datetime import datetime, timezone
    pushed = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def handler(request):
        assert "/users/asha-rao-dev/repos" in request.url.path
        return httpx.Response(200, json=[{"name": f"rag-{i}", "pushed_at": pushed, "language": "Python",
                                          "fork": False, "archived": False} for i in range(4)])

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_batch(resume_dir, settings, use_llm=False, use_github=True, http_client=client)
    files = by_file(asyncio.run(go()))
    strong = files["strong_agentic.pdf"]
    assert strong.github.status == "ok" and strong.score.categories["github"].score == 10
    assert files["thin_wrapper.pdf"].github.status == "no_profile"
    assert files["python_only.pdf"].github.status in {"skipped", "no_profile"}   # rejected: not looked up


def test_missing_input_folder_is_a_clear_error(tmp_path, settings):
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "nope", settings)


def test_cli_writes_results_and_explains(resume_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    output = tmp_path / "out" / "results.json"
    code = cli.main(["--input", str(resume_dir), "--output", str(output), "--no-github",
                     "--explain", "strong_agentic.pdf"])
    assert code == 0 and output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["batch_summary"]["total_resumes"] == 9
    printed = capsys.readouterr().out
    assert "Batch summary" in printed and "Asha Rao" in printed and "Total =" in printed

    assert cli.main(["--output", str(output), "--explain", "thin_wrapper"]) == 0
    assert "thin_llm_wrapper" in capsys.readouterr().out
    assert cli.main(["--input", str(tmp_path / "missing"), "--output", str(output)]) == 2
