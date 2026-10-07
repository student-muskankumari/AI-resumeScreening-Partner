"""Bonus FastAPI interface. Skipped when FastAPI is not installed."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from screener import api  # noqa: E402

from conftest import AI_NO_PYTHON, STRONG_AGENTIC, make_pdf  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    api._state["results"] = None
    return TestClient(api.app)


def test_results_before_any_run_is_404(client):
    assert client.get("/results").status_code == 404


def test_screen_uploaded_resumes_then_fetch_results(client, tmp_path):
    strong = make_pdf(tmp_path / "strong.pdf", "ASHA RAO", STRONG_AGENTIC).read_bytes()
    no_python = make_pdf(tmp_path / "nopy.pdf", "Meera Joshi", AI_NO_PYTHON).read_bytes()
    response = client.post(
        "/screen?use_github=false&use_llm=false",
        files=[("files", ("strong.pdf", strong, "application/pdf")),
               ("files", ("../../nopy.pdf", no_python, "application/pdf")),
               ("files", ("bad.pdf", b"not a pdf", "application/pdf")),
               ("files", ("ignored.exe", b"binary", "application/octet-stream"))],
    )
    assert response.status_code == 200
    summary = response.json()["batch_summary"]
    assert (summary["total_resumes"], summary["eligible"], summary["rejected"], summary["failed_unreadable"]) == (3, 1, 1, 1)
    again = client.get("/results")
    assert again.status_code == 200 and again.json()["candidates"][0]["candidate_name"] == "Asha Rao"


def test_screen_without_files_uses_the_server_folder(client, tmp_path, monkeypatch):
    folder = tmp_path / "resumes"
    folder.mkdir()
    make_pdf(folder / "strong.pdf", "ASHA RAO", STRONG_AGENTIC)
    monkeypatch.setenv("RESUME_DIR", str(folder))
    response = client.post("/screen?use_github=false&use_llm=false")
    assert response.status_code == 200 and response.json()["batch_summary"]["eligible"] == 1
    monkeypatch.setenv("RESUME_DIR", str(tmp_path / "missing"))
    assert client.post("/screen?use_github=false").status_code == 400
