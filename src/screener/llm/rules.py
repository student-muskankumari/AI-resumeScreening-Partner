"""Rule-based evidence extraction.

This is the last step of the fallback chain (Groq -> Gemini -> rules) and the
only one that needs no network or API key. It fills the same `Evidence`
schema the models fill, using the term lists in `taxonomy.py` and the
used / claimed / listed level of each line.

It is weaker than a model at judging how deep a project really is, so every
candidate scored this way is labelled `evidence_source: "rules"`.
"""

from __future__ import annotations

from .. import config, taxonomy
from ..eligibility import clip, find, strip_ai_tools
from ..models import (
    AIEvidence,
    BackendEvidence,
    CloudEvidence,
    EngineeringEvidence,
    Evidence,
    LevelSignal,
    StrengthSignal,
)
from ..sections import TaggedLine, is_narrative


def _level(lines: list[TaggedLine], pattern, *, ai_context: bool = False) -> LevelSignal:
    hit = find(lines, pattern, ai_context=ai_context)
    level = hit.scoring_level
    return LevelSignal(level=level, evidence=hit.evidence if level != "absent" else None)


def _matches_any(text: str, patterns: dict) -> set[str]:
    return {label for label, pattern in patterns.items() if pattern.search(text)}


def _ai_evidence(lines: list[TaggedLine]) -> tuple[AIEvidence, str]:
    used = [(index, line) for index, line in enumerate(lines) if line.level == "used"]
    genai_idx: list[int] = []
    ml_idx: list[int] = []
    genai_labels: set[str] = set()
    ml_labels: set[str] = set()
    label_count: dict[int, int] = {}
    for index, line in used:
        text = strip_ai_tools(line.text)
        genai = _matches_any(text, taxonomy.GENAI)
        ml = _matches_any(text, taxonomy.CLASSICAL_ML)
        if genai:
            genai_idx.append(index)
            genai_labels |= genai
        if ml:
            ml_idx.append(index)
            ml_labels |= ml
        label_count[index] = len(genai) * 2 + len(ml)

    if genai_idx:
        kind, ai_idx, labels = "genai", genai_idx, genai_labels
    elif ml_idx:
        kind, ai_idx, labels = "classical_ml", ml_idx, ml_labels
    else:
        return AIEvidence(kind="none"), ""

    # Lines next to an AI line (same section) usually describe the same
    # project, so depth signals are read from that neighbourhood too. For
    # LLM-era work only the LLM lines count: a separate classical-ML project
    # must not lend its "97% accuracy" to an unrelated chatbot.
    context: set[int] = set(ai_idx)
    for index in ai_idx:
        for neighbour in range(index - 2, index + 3):
            if 0 <= neighbour < len(lines) and lines[neighbour].level == "used" \
                    and lines[neighbour].section == lines[index].section:
                context.add(neighbour)
    context_lines = [lines[i] for i in sorted(context)]

    def depth(rule: str) -> StrengthSignal:
        groups = taxonomy.AI_DEPTH_GROUPS[rule]
        required = taxonomy.AI_DEPTH_REQUIRED.get(rule)
        matched_groups: set[int] = set()
        best_line, best_hits = None, 0
        for line in context_lines:
            text = strip_ai_tools(line.text)
            hits = {g for g, pattern in enumerate(groups) if pattern.search(text)}
            if not hits:
                continue
            matched_groups |= hits
            if len(hits) > best_hits:
                best_line, best_hits = line.text, len(hits)
        if required is not None and not (matched_groups & required):
            return StrengthSignal()          # only supporting words, no core term
        strength = 2 if len(matched_groups) >= 2 else len(matched_groups)
        return StrengthSignal(strength=strength, evidence=clip(best_line) if best_line else None)

    # The line quoted as the main evidence: a sentence beats a bare tech list.
    best_index = max(ai_idx, key=lambda i: (is_narrative(lines[i].text), label_count.get(i, 0), -i))
    best_line = clip(lines[best_index].text, 240)

    retrieval = depth("retrieval")
    agents = depth("agents_tools")
    state = depth("state_orchestration")
    data = depth("data_pipeline")
    evaluation = depth("evaluation")

    # "Deep" needs both breadth (several distinct AI technologies in use) and
    # a substantial description, so a long list of names is not enough.
    context_words = sum(len(line.text.split()) for line in context_lines)
    deep_labels = 5 if kind == "genai" else 4
    deep = len(labels) >= deep_labels and context_words >= config.DEEP_AI_MIN_WORDS
    implementation = StrengthSignal(strength=2 if deep else 1, evidence=best_line)
    use_case = StrengthSignal(strength=2 if kind == "genai" else 1, evidence=best_line)

    work_lines = [lines[i] for i in ai_idx if lines[i].section == "experience"]
    metric_lines = [line for line in context_lines if taxonomy.METRIC.search(line.text)]
    if work_lines:
        strongest = max(work_lines, key=lambda line: len(line.text.split()))
        business = StrengthSignal(strength=2, evidence=clip(strongest.text))
    elif metric_lines:
        business = StrengthSignal(strength=1, evidence=clip(metric_lines[0].text))
    else:
        business = StrengthSignal()

    # Thin-wrapper check: how much real work surrounds the model call?
    severity, reason = 0, None
    if kind == "genai":
        signals = (retrieval, agents, state, data, evaluation)
        total = sum(s.strength for s in signals)
        if total == 0:
            severity = 3
            reason = ("LLM/API use is shown with no retrieval, agent, state, data-processing "
                      "or evaluation work around it")
        elif total == 1:
            severity = 2
            reason = "LLM/API use with only one weak supporting signal; little workflow or backend logic"
        elif total == 2 and not any(s.strength == 2 for s in signals):
            severity = 1
            reason = "Two weak supporting signals around the model call; workflow details are thin"

    tutorial = context_words < config.TUTORIAL_MAX_AI_WORDS and not metric_lines
    tutorial_reason = (
        f"AI work is described in only {context_words} words with no implementation detail or results"
        if tutorial else None
    )

    evidence = AIEvidence(
        kind=kind,
        use_case=use_case,
        implementation_depth=implementation,
        retrieval=retrieval,
        agents_tools=agents,
        state_orchestration=state,
        data_pipeline=data,
        evaluation=evaluation,
        business_logic=business,
        wrapper_severity=severity,
        wrapper_reason=reason,
        tutorial_style=tutorial,
        tutorial_reason=tutorial_reason,
    )
    return evidence, best_line


def build_evidence(lines: list[TaggedLine]) -> Evidence:
    """Build an Evidence object from tagged resume lines, with no model."""
    ai, summary = _ai_evidence(lines)
    backend = BackendEvidence(
        python=_level(lines, taxonomy.PYTHON),
        fastapi=_level(lines, taxonomy.FASTAPI),
        other_python_framework=_level(lines, taxonomy.OTHER_PY_FRAMEWORK),
        async_programming=_level(lines, taxonomy.ASYNC),
        postgresql=_level(lines, taxonomy.POSTGRESQL),
        other_sql=_level(lines, taxonomy.OTHER_SQL),
        redis=_level(lines, taxonomy.REDIS),
    )
    cloud = CloudEvidence(
        gcp=_level(lines, taxonomy.GCP),
        other_cloud=_level(lines, taxonomy.OTHER_CLOUD),
        docker=_level(lines, taxonomy.DOCKER),
        deployment=_level(lines, taxonomy.DEPLOYMENT),
        frontend_e2e=_level(lines, taxonomy.FRONTEND),
    )
    engineering = EngineeringEvidence(
        **{name: _level(lines, pattern) for name, pattern in taxonomy.ENGINEERING.items()}
    )
    return Evidence(
        project_summary=summary,
        ai=ai,
        backend=backend,
        cloud=cloud,
        engineering=engineering,
    )
