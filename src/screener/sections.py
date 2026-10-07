"""Turn raw resume text into logical lines tagged with an evidence level.

Resumes do not share a layout, so this does not depend on any one section
name. It recognises a wide vocabulary of headings when they exist and falls
back to the shape of each line when they do not:

    used    -> text inside a project / experience section, or any sentence
               that describes building something
    claimed -> summary / profile statements and achievements
    listed  -> skills sections and bare comma-separated lists
    mention -> education, coursework and certifications (never evidence)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

LEVEL_RANK = {"used": 3, "claimed": 2, "listed": 1, "mention": 0}

_HEADINGS: dict[str, tuple[str, ...]] = {
    "experience": (
        "experience", "work experience", "professional experience", "internship", "internships",
        "internship experience", "internship training", "internships work experience",
        "employment", "employment history", "work history", "freelancing", "freelance",
        "open source", "research", "research experience", "industry experience",
        "relevant experience", "experience and projects",
    ),
    "projects": (
        "projects", "project", "key projects", "selected projects", "project experience",
        "academic projects", "personal projects", "hackathon experience", "projects and research",
        "notable projects", "side projects", "technical projects", "major projects",
    ),
    "skills": (
        "skills", "technical skills", "skills summary", "core competencies", "core technical expertise",
        "technical expertise", "tools", "technologies", "tech stack", "languages", "soft skills",
        "area of interest", "areas of interest", "ats keyword index", "skill set", "key skills",
        "skills and tools", "technical proficiency", "additional skills",
    ),
    "summary": (
        "summary", "professional summary", "profile", "profile summary", "objective", "objectives",
        "career objective", "career aspiration", "about", "about me", "resume summary",
        "profile objective", "executive summary",
    ),
    "education": (
        "education", "education qualification", "academic background", "academics",
        "relevant coursework", "coursework", "academic qualifications",
    ),
    "certifications": (
        "certifications", "certificates", "certification", "courses", "certifications and courses",
        "certifications and professional development", "certifications and achievements",
        "certificates and achievements", "credentials", "licenses and certifications",
    ),
    "other": (
        "achievements", "achievement", "awards", "publications", "publication", "extracurricular",
        "extra curricular activities", "co curricular activities", "leadership", "languages known",
        "personal details", "links", "find me online", "key achievements", "achievements and leadership",
        "achievements and activities", "achievements and open source", "key achievements and metrics",
        "involvement", "online coding profiles", "interests", "hobbies", "declaration",
        "data structures and algorithm", "get in touch", "certifications achievements",
        "research under review",
    ),
}
_HEADING_TO_SECTION = {name: section for section, names in _HEADINGS.items() for name in names}

_SECTION_LEVEL = {
    "experience": "used",
    "projects": "used",
    "skills": "listed",
    "summary": "claimed",
    "top": "claimed",
    "other": "claimed",
    "education": "mention",
    "certifications": "mention",
}

_BULLETS = "•●◦▪■◆○∙·‣⁃–—-*"
_BULLET_RE = re.compile(rf"^\s*[{re.escape(_BULLETS)}]+\s*")
_ACTION_VERB = re.compile(
    r"\b(built|build|developed|designed|implemented|engineered|architected|created|deployed|"
    r"integrated|trained|shipped|delivered|automated|optimi[sz]ed|constructed|launched|"
    r"orchestrated|fine-tuned|authored|contributed|wrote|led|migrated|refactored|prototyped|"
    r"configured|established|introduced|operationali[sz]ed|extended|exposed)\b",
    re.I,
)


@dataclass(frozen=True)
class TaggedLine:
    text: str
    section: str
    level: str   # used | claimed | listed | mention


def normalise_text(text: str) -> str:
    """Fold ligatures and odd spacing so patterns match what a reader sees."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("​", " ").replace("­", "").replace("﻿", "")
    text = re.sub(r"[-\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n"))


def _heading_key(line: str) -> str:
    key = re.sub(r"\([^)]*\)", " ", line.lower())
    key = key.replace("&", " and ").replace("/", " ")
    key = re.sub(r"[^a-z ]", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def detect_heading(line: str) -> str | None:
    """Return the section a heading line opens, or None if it is not one."""
    stripped = line.strip().strip(":").strip()
    if not stripped or len(stripped) > 48 or stripped.endswith("."):
        return None
    # "E X P E R I E N C E" style headings
    if re.fullmatch(r"(?:[A-Za-z&] ){3,}[A-Za-z&]", stripped):
        stripped = stripped.replace(" ", "")
    key = _heading_key(stripped)
    if not key or len(key.split()) > 6:
        return None
    return _HEADING_TO_SECTION.get(key)


def _is_list_like(text: str) -> bool:
    """A bare list of technologies rather than a sentence."""
    separators = len(re.findall(r"[,|;•·/]", text))
    words = len(text.split())
    if _ACTION_VERB.search(text):
        return False
    if ":" in text[:40] and separators >= 1 and words <= 40:
        return True
    return separators >= 2 and words / (separators + 1) <= 3.5


def is_narrative(text: str) -> bool:
    return len(text.split()) >= 8 and bool(_ACTION_VERB.search(text))


def _join_lines(raw_lines: list[str]) -> list[tuple[str, str | None]]:
    """Merge wrapped lines into logical lines. Returns (text, heading)."""
    logical: list[tuple[str, str | None]] = []
    buffer = ""
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        heading = detect_heading(line)
        if heading:
            if buffer:
                logical.append((buffer, None))
                buffer = ""
            logical.append((line, heading))
            continue
        starts_bullet = bool(_BULLET_RE.match(line))
        line = _BULLET_RE.sub("", line).strip()
        if not line:          # a bullet glyph on its own line
            if buffer:
                logical.append((buffer, None))
                buffer = ""
            continue
        previous_open = (
            buffer
            and not starts_bullet
            and len(buffer) < 700
            and buffer[-1] not in ".!?:"
            and len(buffer) >= 45
            and (line[0].islower() or buffer[-1] in ",-–(/&" or len(buffer) >= 60)
        )
        if previous_open:
            # keep the hyphen of a wrapped compound ("LLM-" + "powered")
            joiner = "" if buffer.endswith("-") else " "
            buffer = buffer + joiner + line
        else:
            if buffer:
                logical.append((buffer, None))
            buffer = line
    if buffer:
        logical.append((buffer, None))
    return logical


def tag_lines(text: str) -> list[TaggedLine]:
    """Split text into logical lines and give each an evidence level."""
    raw_lines = normalise_text(text).split("\n")
    logical = _join_lines(raw_lines)
    has_headings = any(heading for _, heading in logical)

    tagged: list[TaggedLine] = []
    section = "top"
    seen: set[str] = set()
    for line, heading in logical:
        if heading:
            section = heading
            continue
        # Drop exact repeats (some PDFs carry a hidden duplicate text layer).
        fingerprint = re.sub(r"\W+", "", line.lower())
        if len(fingerprint) >= 30:
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

        if not has_headings:
            level = "listed" if _is_list_like(line) else ("used" if is_narrative(line) else "claimed")
        else:
            level = _SECTION_LEVEL.get(section, "claimed")
            # A sentence that describes building something is usage evidence
            # even if the heading above it was not recognised.
            if level in {"listed", "mention"} and is_narrative(line):
                level = "used"
        tagged.append(TaggedLine(text=line, section=section, level=level))
    return tagged
