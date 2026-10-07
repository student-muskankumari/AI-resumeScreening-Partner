from screener import extract
from screener.models import ParsedResume
from screener.sections import detect_heading, normalise_text, tag_lines


def parsed(text: str, links=None, name_hint=None) -> ParsedResume:
    return ParsedResume(source_file="x.pdf", path="x.pdf", file_hash="h", text=normalise_text(text),
                        links=links or [], name_hint=name_hint)


def test_headings_are_recognised_whatever_the_wording():
    assert detect_heading("TECHNICAL SKILLS") == "skills"
    assert detect_heading("Work Experience:") == "experience"
    assert detect_heading("Internship / Training") == "experience"
    assert detect_heading("Key Projects") == "projects"
    assert detect_heading("Certifications & Achievements") == "certifications"
    assert detect_heading("P R O J E C T S") == "projects"
    assert detect_heading("Built a RAG pipeline with LangChain.") is None


def test_levels_follow_sections():
    text = """Summary
Engineer who knows Python.
Skills
Languages: Python, Java
Projects
Search Bot | Python, LangChain
- Built a retrieval pipeline over product manuals.
Certifications
Machine Learning with Python - IBM
"""
    levels = {line.text: line.level for line in tag_lines(text)}
    assert levels["Engineer who knows Python."] == "claimed"
    assert levels["Languages: Python, Java"] == "listed"
    assert levels["Search Bot | Python, LangChain"] == "used"
    assert levels["Machine Learning with Python - IBM"] == "mention"


def test_sentence_about_building_counts_as_usage_under_an_unknown_heading():
    text = """Skills
Python, Java, React
THINGS I HAVE MADE
Built a multi-agent LangGraph workflow in Python that answers support tickets from a vector store.
"""
    used = [line for line in tag_lines(text) if line.level == "used"]
    assert len(used) == 1 and "LangGraph" in used[0].text


def test_resume_without_any_headings_falls_back_to_line_shape():
    text = """Jane Doe
Python, Java, React, Docker
Built a RAG service in Python with FastAPI and pgvector for internal documents search.
"""
    by_text = {line.text: line.level for line in tag_lines(text)}
    assert by_text["Python, Java, React, Docker"] == "listed"
    assert by_text["Built a RAG service in Python with FastAPI and pgvector for internal documents search."] == "used"


def test_ligatures_are_normalised_and_hidden_duplicate_lines_dropped():
    assert "fixes" in normalise_text("critical ﬁxes")
    line = "Developed backend services using Python FastAPI and PostgreSQL for a school platform."
    text = f"Experience\n{line}\n{line}\n"
    assert sum(1 for l in tag_lines(text) if l.text == line) == 1


def test_email_and_name():
    p = parsed("PRIYA R\nEmail: priya.r29@example.com | Mobile: 99999", name_hint="PRIYA R")
    assert extract.extract_email(p) == "priya.r29@example.com"
    assert extract.extract_name(p, "priya.r29@example.com") == "Priya R"
    p = parsed("+91-9000000000\nAgam Jain\njain@example.com")
    assert extract.extract_name(p) == "Agam Jain"
    p = parsed("Curriculum Vitae\nphone 12345\nx.y@example.com")
    assert extract.extract_name(p, "john.smith42@example.com") == "John Smith"


def test_surname_on_its_own_line_is_joined():
    p = parsed("Prathamesh\nPatil\n+91 12345 | Bengaluru", name_hint="Prathamesh")
    assert extract.extract_name(p) == "Prathamesh Patil"


def test_github_username_from_hidden_link_beats_text():
    p = parsed("LinkedIn | GitHub", links=["https://linkedin.com/in/x", "https://github.com/ZyanHere"])
    assert extract.github_candidates(p) == ["ZyanHere"]


def test_github_username_from_repo_links_only():
    p = parsed("Projects", links=["https://github.com/Pavani-A/Auto_Eval", "https://github.com/Pavani-A/pantry"])
    assert extract.github_candidates(p) == ["Pavani-A"]


def test_bare_github_link_gives_no_username():
    assert extract.github_candidates(parsed("GitHub", links=["https://github.com/"])) == []


def test_padded_and_mixed_case_urls_are_deduplicated():
    p = parsed("github.com/kiran-naregal", links=[" https://github.com/Kiran-naregal/code-review-agent "])
    assert [u.lower() for u in extract.github_candidates(p)] == ["kiran-naregal"]


def test_username_wrapped_onto_next_line_is_rejoined():
    p = parsed("https://github.com/annishasaravan\nan\nportfolio.example.com")
    assert extract.github_candidates(p) == ["annishasaravanan"]


def test_link_and_text_disagreeing_yields_both_candidates_link_first():
    p = parsed("https://github.com/textuser", links=["https://github.com/link-user"])
    assert extract.github_candidates(p) == ["link-user", "textuser"]


def test_heading_after_url_is_not_treated_as_a_wrapped_username():
    p = parsed("GitHub: github.com/KS1745\nSummary\nFull stack developer")
    assert extract.github_candidates(p) == ["KS1745"]


def test_github_pages_site_is_a_last_resort():
    p = parsed("Portfolio", links=["https://nidhi-km.github.io/Portfolio/"])
    assert extract.github_candidates(p) == ["nidhi-km"]


def test_skills_ignore_ai_coding_tools_and_substrings():
    skills = extract.extract_skills("Uses GitHub Copilot and Cursor. Skills: JavaScript, HTML, React.js")
    assert "JavaScript" in skills and "React" in skills
    assert "Java" not in skills and "Machine Learning" not in skills and "LLM" not in skills
