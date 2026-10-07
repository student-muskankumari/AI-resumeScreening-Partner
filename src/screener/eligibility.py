"""Hard eligibility filter (assignment section 3).

Rule-based on purpose: no LLM is involved, so the same resume always gets the
same decision and every decision can be unit-tested.

A candidate is eligible only when BOTH hold:
  1. Python evidence  - Python is a real skill, project technology, work
                        technology or implementation language.
  2. AI evidence      - at least one meaningful AI / LLM / RAG / agentic
                        project, framework use or implementation.

Other languages (JavaScript, Java, React, Next.js ...) never cause rejection.
AI coding assistants (Copilot, Cursor, "AI-assisted development") never count
as AI evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import taxonomy
from .config import Settings
from .models import EligibilityResult
from .sections import LEVEL_RANK, TaggedLine

REASON_NO_PYTHON = "No evidence of Python stack"
REASON_NO_AI = "No AI/agentic project evidence"


@dataclass
class Hit:
    """Best evidence found for a group of patterns."""

    level: str = "absent"            # used | claimed | listed | mention | absent
    evidence: str | None = None
    labels: set[str] = field(default_factory=set)        # labels at best level
    used_labels: set[str] = field(default_factory=set)   # labels at 'used'

    @property
    def rank(self) -> int:
        return LEVEL_RANK.get(self.level, -1)

    @property
    def scoring_level(self) -> str:
        """Level on the four-step scale used by scoring."""
        return self.level if self.level in {"used", "claimed", "listed"} else "absent"


def clip(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def strip_ai_tools(text: str) -> str:
    return taxonomy.AI_TOOL_PHRASES.sub(" ", text)


def find(lines: list[TaggedLine], patterns: dict[str, re.Pattern[str]] | re.Pattern[str],
         *, ai_context: bool = False) -> Hit:
    """Scan tagged lines and keep the strongest evidence level found."""
    if isinstance(patterns, re.Pattern):
        patterns = {"match": patterns}
    hit = Hit()
    best_count = 0   # the evidence line is the one naming the most terms
    for line in lines:
        text = strip_ai_tools(line.text) if ai_context else line.text
        matched = {label for label, pattern in patterns.items() if pattern.search(text)}
        if not matched:
            continue
        rank = LEVEL_RANK.get(line.level, -1)
        if line.level == "used":
            hit.used_labels |= matched
        if rank > hit.rank:
            hit.level, hit.evidence, hit.labels = line.level, clip(line.text), set(matched)
            best_count = len(matched)
        elif rank == hit.rank:
            hit.labels |= matched
            if len(matched) > best_count:
                hit.evidence, best_count = clip(line.text), len(matched)
    return hit


def evaluate(lines: list[TaggedLine], settings: Settings) -> EligibilityResult:
    python = find(lines, taxonomy.PYTHON)
    genai = find(lines, taxonomy.GENAI, ai_context=True)
    classical = find(lines, taxonomy.CLASSICAL_ML, ai_context=True)
    generic = find(lines, taxonomy.GENERIC_AI, ai_context=True)
    tool_lines = [line for line in lines if taxonomy.AI_TOOL_PHRASES.search(line.text)]

    reasons: list[str] = []
    flags: list[str] = []

    # ---- Condition 1: Python -------------------------------------------
    if python.level == "used":
        python_ok = True
    elif python.level in {"claimed", "listed"}:
        python_ok = not settings.require_python_usage
        if python_ok:
            flags.append("python_listed_only" if python.level == "listed" else "python_claimed_only")
        else:
            reasons.append(f"{REASON_NO_PYTHON} (Python is named as a skill but never used in a project or job)")
    else:
        python_ok = False
        detail = " (Python appears only under education or certifications)" if python.level == "mention" else ""
        reasons.append(REASON_NO_PYTHON + detail)

    # ---- Condition 2: AI / agentic -------------------------------------
    ai_kind, ai_level, ai_evidence = "none", "absent", None
    if genai.level == "used":
        ai_ok, ai_kind, ai_level, ai_evidence = True, "genai", "used", genai.evidence
    elif classical.level == "used" and settings.allow_classical_ml:
        ai_ok, ai_kind, ai_level, ai_evidence = True, "classical_ml", "used", classical.evidence
        flags.append("classical_ml_only")
    else:
        ai_ok = False
        if classical.level == "used":
            detail = "only classical ML work; no LLM, RAG, embedding or agent implementation"
            ai_evidence = classical.evidence
        elif max(genai.rank, classical.rank) >= LEVEL_RANK["listed"]:
            detail = "AI terms appear only in the summary or a skills list, not in any project or job"
            best = genai if genai.rank >= classical.rank else classical
            ai_level, ai_evidence = best.scoring_level, best.evidence
        elif generic.rank >= LEVEL_RANK["listed"]:
            detail = "only generic 'AI-powered' wording; no named model, framework or technique"
            ai_evidence = generic.evidence
        elif tool_lines:
            detail = ("only AI coding assistants are mentioned (Copilot, Cursor, AI-assisted development); "
                      "using them is not AI project evidence")
            ai_evidence = clip(tool_lines[0].text)
        else:
            detail = ""
        reasons.append(f"{REASON_NO_AI} ({detail})" if detail else REASON_NO_AI)

    return EligibilityResult(
        eligible=python_ok and ai_ok,
        rejection_reasons=reasons,
        python_level=python.scoring_level,
        python_evidence=python.evidence,
        ai_kind=ai_kind,
        ai_level=ai_level,
        ai_evidence=ai_evidence,
        flags=flags,
    )
