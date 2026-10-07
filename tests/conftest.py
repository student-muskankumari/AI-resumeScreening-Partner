"""Shared test helpers. No test here touches the network or a real API key."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from screener.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(cache_dir=tmp_path / "cache", use_disk_cache=True, llm_tokens_per_minute=10**9)


def make_pdf(path: Path, name: str, body: str) -> Path:
    """Write a small text PDF: a large name line, then the body."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), name, fontsize=20)
    page.insert_textbox(pymupdf.Rect(50, 80, 560, 800), body, fontsize=9)
    doc.save(path)
    doc.close()
    return path


STRONG_AGENTIC = """asha.rao@example.com | github.com/asha-rao-dev
Summary
Backend engineer focused on Python and agentic AI systems.
Skills
Languages: Python, SQL, JavaScript
Frameworks: FastAPI, LangGraph, LangChain, React
Experience
Acme AI - Software Engineer Intern
- Built a stateful LangGraph multi-agent workflow with tool calling and a supervisor that routes each request.
- Implemented a RAG pipeline with embeddings in pgvector, hybrid search and reranking over 20,000 documents.
- Developed an async FastAPI backend with PostgreSQL and a Redis cache, cutting latency by 40%.
- Added a 60 case eval suite to catch hallucination, with retry and fallback handling, run in CI with pytest.
Projects
Support Copilot | Python, FastAPI, LangGraph, Docker, GCP
- Deployed on GCP Cloud Run with Docker and GitHub Actions; React dashboard for agents.
Education
B.Tech Computer Science, 2026
"""

PYTHON_ONLY = """ravi.k@example.com
Summary
Python backend developer.
Skills
Languages: Python, SQL
Frameworks: FastAPI, Django
Experience
Shopline - Backend Intern
- Built REST APIs in Python with FastAPI and PostgreSQL for the order service.
- Added Redis caching and wrote pytest unit tests for the payment module.
Projects
Inventory Service | Python, Django, MySQL
- Developed CRUD endpoints, authentication and reporting for a warehouse team.
Education
B.Tech Information Technology, 2025
"""

AI_NO_PYTHON = """meera.j@example.com
Summary
Full stack TypeScript developer.
Skills
Languages: TypeScript, JavaScript, SQL
Frameworks: Node.js, React, Next.js
Experience
Nimbus - Software Engineer
- Engineered a RAG platform in Node.js using pgvector embeddings and the OpenAI API for semantic search.
- Built a document ingestion pipeline and an LLM answer service with streaming responses.
Projects
Resume Ranker | Node.js, React, PostgreSQL, OpenAI API
- Built LLM based candidate scoring with schema validation and a retry loop.
Education
B.E. Computer Science, 2024
"""

FRONTEND_ONLY = """dev.s@example.com
Summary
Frontend engineer who uses GitHub Copilot and AI-assisted development daily.
Skills
Languages: JavaScript, TypeScript, Java
Frameworks: React, Next.js, Spring Boot
Experience
Pixel Labs - Frontend Developer
- Built reusable React components and Next.js pages for an admin dashboard.
- Integrated REST APIs and Razorpay payments; improved load time by 30%.
Projects
Shop UI | React, Redux, Tailwind
- Developed a responsive storefront with cart, search and checkout flows.
Education
BCA, 2023
"""

THIN_WRAPPER = """kiran.t@example.com
Skills
Languages: Python, C++
Tools: Streamlit, Git
Projects
Chatbot Using Gemini API | Python, Streamlit
- Built a chatbot that sends the user question to the Gemini API and shows the reply in a Streamlit page.
Notes App | Python, Flask
- Developed a simple notes application with login and search.
Education
B.Tech Computer Science, 2026
"""
