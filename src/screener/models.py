"""Data models shared across the pipeline.

`Evidence` is the one schema every analysis source must return: the Groq
adapter, the Gemini adapter and the rule-based fallback all produce it, and
scoring consumes nothing else. That keeps scoring identical whichever source
answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Level = Literal["used", "claimed", "listed", "absent"]


class _Lenient(BaseModel):
    """Ignore unknown keys so a chatty model cannot break validation."""

    model_config = ConfigDict(extra="ignore")


class LevelSignal(_Lenient):
    """Where a technology shows up in the resume.

    used    -> in a project or job description
    claimed -> stated in a summary / profile sentence
    listed  -> only in a skills list
    absent  -> not found
    """

    level: Level = "absent"
    evidence: Optional[str] = None

    @field_validator("level", mode="before")
    @classmethod
    def _normalise_level(cls, value: object) -> object:
        if value is None:
            return "absent"
        if isinstance(value, str):
            return value.strip().lower() or "absent"
        return value


class StrengthSignal(_Lenient):
    """How strongly a depth signal is shown: 0 none, 1 basic, 2 clear."""

    strength: int = Field(default=0, ge=0, le=2)
    evidence: Optional[str] = None

    @field_validator("strength", mode="before")
    @classmethod
    def _coerce_strength(cls, value: object) -> object:
        if value is None:
            return 0
        if isinstance(value, bool):
            return 2 if value else 0
        if isinstance(value, (int, float)):
            return max(0, min(2, int(value)))
        return value


class AIEvidence(_Lenient):
    kind: Literal["genai", "classical_ml", "none"] = "none"
    use_case: StrengthSignal = Field(default_factory=StrengthSignal)
    implementation_depth: StrengthSignal = Field(default_factory=StrengthSignal)
    retrieval: StrengthSignal = Field(default_factory=StrengthSignal)
    agents_tools: StrengthSignal = Field(default_factory=StrengthSignal)
    state_orchestration: StrengthSignal = Field(default_factory=StrengthSignal)
    data_pipeline: StrengthSignal = Field(default_factory=StrengthSignal)
    evaluation: StrengthSignal = Field(default_factory=StrengthSignal)
    business_logic: StrengthSignal = Field(default_factory=StrengthSignal)
    # 0 = real system, 1 = somewhat thin, 2 = thin wrapper, 3 = bare API call
    wrapper_severity: int = Field(default=0, ge=0, le=3)
    wrapper_reason: Optional[str] = None
    tutorial_style: bool = False
    tutorial_reason: Optional[str] = None

    @field_validator("wrapper_severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value: object) -> object:
        if value is None:
            return 0
        if isinstance(value, bool):
            return 2 if value else 0
        if isinstance(value, (int, float)):
            return max(0, min(3, int(value)))
        return value


class BackendEvidence(_Lenient):
    python: LevelSignal = Field(default_factory=LevelSignal)
    fastapi: LevelSignal = Field(default_factory=LevelSignal)
    other_python_framework: LevelSignal = Field(default_factory=LevelSignal)
    async_programming: LevelSignal = Field(default_factory=LevelSignal)
    postgresql: LevelSignal = Field(default_factory=LevelSignal)
    other_sql: LevelSignal = Field(default_factory=LevelSignal)
    redis: LevelSignal = Field(default_factory=LevelSignal)


class CloudEvidence(_Lenient):
    gcp: LevelSignal = Field(default_factory=LevelSignal)
    other_cloud: LevelSignal = Field(default_factory=LevelSignal)
    docker: LevelSignal = Field(default_factory=LevelSignal)
    deployment: LevelSignal = Field(default_factory=LevelSignal)
    frontend_e2e: LevelSignal = Field(default_factory=LevelSignal)


class EngineeringEvidence(_Lenient):
    testing: LevelSignal = Field(default_factory=LevelSignal)
    architecture: LevelSignal = Field(default_factory=LevelSignal)
    caching: LevelSignal = Field(default_factory=LevelSignal)
    queues: LevelSignal = Field(default_factory=LevelSignal)
    observability: LevelSignal = Field(default_factory=LevelSignal)
    concurrency: LevelSignal = Field(default_factory=LevelSignal)
    failure_handling: LevelSignal = Field(default_factory=LevelSignal)


class Evidence(_Lenient):
    """Structured evidence extracted from one resume."""

    candidate_name: Optional[str] = None
    project_summary: str = ""
    ai: AIEvidence = Field(default_factory=AIEvidence)
    backend: BackendEvidence = Field(default_factory=BackendEvidence)
    cloud: CloudEvidence = Field(default_factory=CloudEvidence)
    engineering: EngineeringEvidence = Field(default_factory=EngineeringEvidence)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    @field_validator("project_summary", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return "" if value is None else value

    @field_validator("strengths", "concerns", mode="before")
    @classmethod
    def _none_to_list(cls, value: object) -> object:
        return [] if value is None else value


# --------------------------------------------------------------------------
# Pipeline records
# --------------------------------------------------------------------------
@dataclass
class ParsedResume:
    source_file: str
    path: str
    file_hash: str
    text: str = ""
    links: list[str] = field(default_factory=list)
    page_count: int = 0
    parser: str = ""
    name_hint: Optional[str] = None   # largest text on page one, if known
    warnings: list[str] = field(default_factory=list)


class EligibilityResult(BaseModel):
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    python_level: Level = "absent"
    python_evidence: Optional[str] = None
    ai_kind: Literal["genai", "classical_ml", "none"] = "none"
    ai_level: Level = "absent"
    ai_evidence: Optional[str] = None
    flags: list[str] = Field(default_factory=list)


class ScoreItem(BaseModel):
    rule: str
    points: int
    max: int
    evidence: Optional[str] = None
    note: Optional[str] = None


class CategoryScore(BaseModel):
    score: int
    max: int
    items: list[ScoreItem] = Field(default_factory=list)


class Penalty(BaseModel):
    type: str
    points: int          # negative
    reason: str


class ScoreResult(BaseModel):
    categories: dict[str, CategoryScore]
    penalties: list[Penalty] = Field(default_factory=list)
    base_score: int
    total_score: int


class GitHubResult(BaseModel):
    status: Literal[
        "ok", "no_profile", "not_found", "rate_limited", "error", "skipped"
    ] = "no_profile"
    username: Optional[str] = None
    profile_url: Optional[str] = None
    recent_activity_points: int = 0
    relevant_repos_points: int = 0
    summary: str = "No GitHub profile found in the resume."
    detail: dict = Field(default_factory=dict)
    tried_usernames: list[str] = Field(default_factory=list)

    @property
    def score(self) -> int:
        return self.recent_activity_points + self.relevant_repos_points


class CandidateResult(BaseModel):
    status: Literal["eligible", "rejected", "failed", "duplicate"]
    source_file: str
    file_hash: str = ""
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    matched_skills: list[str] = Field(default_factory=list)
    eligibility: Optional[EligibilityResult] = None
    evidence: Optional[Evidence] = None
    evidence_source: Optional[str] = None     # groq | gemini | rules
    score: Optional[ScoreResult] = None
    github: GitHubResult = Field(default_factory=GitHubResult)
    project_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    parser: str = ""
    page_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    duplicate_of: Optional[str] = None
    rank: Optional[int] = None
