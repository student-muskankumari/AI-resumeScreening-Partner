"""The hard filter is the most heavily weighted part of the assignment."""

import dataclasses

import pytest

from screener.eligibility import REASON_NO_AI, REASON_NO_PYTHON, evaluate
from screener.sections import tag_lines

from conftest import AI_NO_PYTHON, FRONTEND_ONLY, PYTHON_ONLY, STRONG_AGENTIC, THIN_WRAPPER


def gate(text, settings):
    return evaluate(tag_lines(text), settings)


def test_python_plus_agentic_project_is_eligible(settings):
    result = gate(STRONG_AGENTIC, settings)
    assert result.eligible and result.rejection_reasons == []
    assert result.python_level == "used" and result.ai_kind == "genai"


def test_strong_ai_without_python_is_rejected(settings):
    result = gate(AI_NO_PYTHON, settings)
    assert not result.eligible
    assert result.rejection_reasons == [REASON_NO_PYTHON]


def test_python_backend_without_ai_is_rejected(settings):
    result = gate(PYTHON_ONLY, settings)
    assert not result.eligible
    assert len(result.rejection_reasons) == 1 and result.rejection_reasons[0].startswith(REASON_NO_AI)


def test_frontend_profile_using_ai_coding_tools_is_rejected_on_both_counts(settings):
    result = gate(FRONTEND_ONLY, settings)
    assert not result.eligible
    assert any(r.startswith(REASON_NO_PYTHON) for r in result.rejection_reasons)
    assert any(r.startswith(REASON_NO_AI) for r in result.rejection_reasons)


def test_thin_llm_wrapper_still_passes_the_gate(settings):
    # Thin projects are handled by scoring penalties, not by the hard filter.
    assert gate(THIN_WRAPPER, settings).eligible


def test_other_languages_never_cause_rejection(settings):
    text = STRONG_AGENTIC.replace("Languages: Python, SQL, JavaScript",
                                  "Languages: Python, Java, JavaScript, TypeScript, React, Next.js")
    assert gate(text, settings).eligible


SKILLS_ONLY_PYTHON = """Skills
Languages: Java, Python, SQL
Projects
Resume Screener | Java, Spring Boot, Spring AI, OpenAI
- Built a Spring Boot backend that calls the OpenAI API to rank candidates.
"""


def test_python_only_in_skills_list_passes_with_a_flag_by_default(settings):
    result = gate(SKILLS_ONLY_PYTHON, settings)
    assert result.eligible and "python_listed_only" in result.flags and result.python_level == "listed"


def test_python_only_in_skills_list_is_rejected_under_the_strict_switch(settings):
    strict = dataclasses.replace(settings, require_python_usage=True)
    result = gate(SKILLS_ONLY_PYTHON, strict)
    assert not result.eligible and result.rejection_reasons[0].startswith(REASON_NO_PYTHON)


CLASSICAL_ML = """Skills
Languages: Python
Projects
Waste Image Classification | Python, TensorFlow
- Developed a convolutional neural network to classify waste images and deployed it with Streamlit.
"""


def test_classical_ml_only_passes_with_a_flag_by_default(settings):
    result = gate(CLASSICAL_ML, settings)
    assert result.eligible and result.ai_kind == "classical_ml" and "classical_ml_only" in result.flags


def test_classical_ml_only_is_rejected_when_the_switch_is_off(settings):
    strict = dataclasses.replace(settings, allow_classical_ml=False)
    result = gate(CLASSICAL_ML, strict)
    assert not result.eligible and "classical ML" in result.rejection_reasons[0]


@pytest.mark.parametrize(
    "text, expected_fragment",
    [
        # AI frameworks named in a skills list, but no AI project anywhere
        ("""Skills
Languages: Python
AI: LangChain, RAG, LLMs
Projects
Todo API | Python, FastAPI
- Built CRUD endpoints with authentication.
""", "skills list"),
        # only marketing wording
        ("""Skills
Languages: Python
Projects
Shop Assistant | Python, Flask
- Built an AI-powered shopping platform with payments and search.
""", "generic"),
        # only AI coding tools
        ("""Skills
Languages: Python
Experience
- Built Flask services in Python; used GitHub Copilot and AI-assisted development for faster delivery.
""", "coding assistants"),
    ],
)
def test_weak_ai_mentions_do_not_count(settings, text, expected_fragment):
    result = gate(text, settings)
    assert not result.eligible
    assert expected_fragment in result.rejection_reasons[0]


def test_python_only_in_a_certificate_is_not_python_evidence(settings):
    text = """Skills
Languages: Java, C++
Projects
Support Bot | Node.js, LangChain, OpenAI
- Built a RAG chatbot over support articles with embeddings.
Certifications
Crash Course on Python - Google
"""
    result = gate(text, settings)
    assert not result.eligible
    assert "certifications" in result.rejection_reasons[0]


def test_resume_with_no_section_headings_is_still_judged(settings):
    text = """Jane Doe
jane@example.com
Built a RAG assistant in Python with FastAPI, embeddings and a Qdrant vector store for internal documents.
Python, FastAPI, Docker, Qdrant
"""
    assert gate(text, settings).eligible


def test_domain_names_ending_in_ai_are_not_ai_evidence(settings):
    text = """Experience
Osfin.ai - Automation Engineer
- Developed Playwright test scripts in Python for the billing product.
"""
    result = gate(text, settings)
    assert not result.eligible and result.rejection_reasons == [REASON_NO_AI]
