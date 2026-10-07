"""Central configuration.

Everything tunable lives here: category weights, the scoring rubric, penalty
sizes, thresholds, model names, concurrency limits and policy switches.
Business logic imports from this module and hard-codes none of it.

Secrets and per-machine settings come from environment variables (a `.env`
file in the working directory is loaded if present).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Scoring model (assignment section 4). Category maxima sum to 100.
# --------------------------------------------------------------------------
CATEGORY_WEIGHTS: dict[str, int] = {
    "ai_project_depth": 40,
    "python_backend": 30,
    "cloud_fullstack": 15,
    "github": 10,
    "engineering_depth": 5,
}

# AI / Agentic / RAG project depth (40). Each rule is graded 0, 1 or 2:
# 0 = no evidence, 1 = mentioned or basic, 2 = clearly implemented.
AI_RUBRIC: dict[str, int] = {
    "use_case": 5,              # a real AI system exists in a project or job
    "implementation_depth": 7,  # how much AI/LLM engineering is described
    "retrieval": 6,             # RAG, embeddings, vector search, reranking
    "agents_tools": 6,          # agents, tool calling, MCP, multi-agent
    "state_orchestration": 5,   # state, memory, graphs, multi-step workflows
    "data_pipeline": 4,         # ingestion, parsing, chunking, preprocessing
    "evaluation": 4,            # evals, benchmarks, hallucination checks
    "business_logic": 3,        # shipped in real work / real product logic
}

# Python & backend engineering (30). The assignment names these five items.
BACKEND_RUBRIC: dict[str, int] = {
    "python": 10,
    "fastapi": 6,
    "async": 4,
    "postgresql": 5,
    "redis": 5,
}

# Cloud / deployment / full stack (15).
CLOUD_RUBRIC: dict[str, int] = {
    "cloud_platform": 5,   # GCP gets full credit, other clouds partial
    "docker": 4,
    "deployment": 3,       # deployment, CI/CD, Kubernetes, IaC
    "frontend_e2e": 3,     # React / Next.js as part of an end-to-end system
}

# Engineering depth (5): one point per distinct signal, capped at 5.
ENGINEERING_SIGNALS: tuple[str, ...] = (
    "testing",
    "architecture",
    "caching",
    "queues",
    "observability",
    "concurrency",
    "failure_handling",
)
ENGINEERING_MAX = CATEGORY_WEIGHTS["engineering_depth"]

# GitHub (10): 0-5 recent activity + 0-5 maintained / relevant repositories.
GITHUB_RUBRIC: dict[str, int] = {"recent_activity": 5, "relevant_repos": 5}

# How much of a rule's points each evidence level earns. Evidence inside a
# project or job ("used") beats a summary claim, which beats a skills list.
LEVEL_MULTIPLIER: dict[str, float] = {
    "used": 1.0,
    "claimed": 0.6,
    "listed": 1 / 3,
    "absent": 0.0,
}

# Partial credit for near-equivalents of the named technologies.
OTHER_PYTHON_FRAMEWORK_MAX = 3   # Django / Flask when FastAPI is absent
OTHER_SQL_MAX = 2                # MySQL / SQLite / generic SQL, no PostgreSQL
OTHER_CLOUD_MAX = 3              # AWS / Azure / other hosting when no GCP

# Backend items shown only in a non-Python stack count for half.
NON_PYTHON_BACKEND_FACTOR = 0.5

# A candidate whose only AI evidence is classical ML / CV (no LLM, RAG,
# agents or embeddings) passes the gate but cannot score above this on AI.
CLASSICAL_ML_AI_CAP = 10

# Project-quality penalties (assignment: deduct 5-15 for thin LLM wrappers).
WRAPPER_PENALTY: dict[int, int] = {0: 0, 1: 5, 2: 10, 3: 15}
TUTORIAL_PENALTY = 5
# A model's thin-wrapper flag is ignored when the same evidence shows this
# many depth signals clearly implemented (severity -> signals needed).
WRAPPER_FLAG_IGNORED_AT: dict[int, int] = {1: 2, 2: 3, 3: 3}
MAX_TOTAL_PENALTY = 15
TUTORIAL_MAX_AI_WORDS = 40   # AI work described in fewer words than this
DEEP_AI_MIN_WORDS = 120      # "deep" implementation needs at least this much description

# GitHub scoring thresholds.
GITHUB_RECENT_DAYS = (30, 90, 365)        # latest push within -> 3 / 2 / 1
GITHUB_ACTIVE_WINDOW_DAYS = 180           # repos pushed inside this window
GITHUB_ACTIVE_BONUS = ((4, 2), (2, 1))    # (repo count, bonus points)
GITHUB_MAINTAINED_DAYS = 365              # a repo counts as maintained
GITHUB_RELEVANT_POINTS = ((4, 5), (3, 4), (2, 3), (1, 2))
GITHUB_MAX_USERNAME_ATTEMPTS = 3

# Parsing.
MIN_TEXT_CHARS = 200          # less text than this means "unreadable"
MAX_RESUME_CHARS_FOR_LLM = 14000
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

assert sum(CATEGORY_WEIGHTS.values()) == 100
assert sum(AI_RUBRIC.values()) == CATEGORY_WEIGHTS["ai_project_depth"]
assert sum(BACKEND_RUBRIC.values()) == CATEGORY_WEIGHTS["python_backend"]
assert sum(CLOUD_RUBRIC.values()) == CATEGORY_WEIGHTS["cloud_fullstack"]
assert sum(GITHUB_RUBRIC.values()) == CATEGORY_WEIGHTS["github"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Build with `Settings.from_env()` or pass values
    directly in tests."""

    # LLM providers (primary, then fallback). Empty key = provider skipped.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    groq_reasoning_effort: str = "low"   # for reasoning models; "" to omit
    llm_timeout_seconds: float = 60.0
    llm_max_output_tokens: int = 2500
    llm_expected_output_tokens: int = 1200   # used only for pacing estimates
    llm_max_consecutive_failures: int = 3    # then the provider is skipped
    llm_retries_per_provider: int = 1
    llm_concurrency: int = 1
    llm_tokens_per_minute: int = 7000   # stay under Groq's free 8,000 TPM

    # GitHub.
    github_token: str = ""
    github_base_url: str = "https://api.github.com"
    github_timeout_seconds: float = 20.0
    github_concurrency: int = 5
    github_cache_ttl_hours: int = 24

    # Caching.
    cache_dir: Path = Path(".cache")
    use_disk_cache: bool = True

    # Eligibility policy switches (assignment section 3).
    # True  -> Python must be used in a project or job; a skills-list-only
    #          mention is rejected.
    # False -> a declared Python skill passes the gate and scores low.
    require_python_usage: bool = False
    # True  -> classical ML / CV work passes the gate (AI depth is capped).
    # False -> only LLM / RAG / agentic / embedding work passes.
    allow_classical_ml: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", cls.groq_model).strip(),
            groq_base_url=os.getenv("GROQ_BASE_URL", cls.groq_base_url).strip(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model).strip(),
            gemini_base_url=os.getenv("GEMINI_BASE_URL", cls.gemini_base_url).strip(),
            groq_reasoning_effort=os.getenv("GROQ_REASONING_EFFORT", cls.groq_reasoning_effort).strip(),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", cls.llm_timeout_seconds),
            llm_max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", cls.llm_max_output_tokens),
            llm_retries_per_provider=_env_int("LLM_RETRIES_PER_PROVIDER", cls.llm_retries_per_provider),
            llm_concurrency=_env_int("LLM_CONCURRENCY", cls.llm_concurrency),
            llm_tokens_per_minute=_env_int("LLM_TOKENS_PER_MINUTE", cls.llm_tokens_per_minute),
            github_token=os.getenv("GITHUB_TOKEN", "").strip(),
            github_base_url=os.getenv("GITHUB_BASE_URL", cls.github_base_url).strip(),
            github_timeout_seconds=_env_float("GITHUB_TIMEOUT_SECONDS", cls.github_timeout_seconds),
            github_concurrency=_env_int("GITHUB_CONCURRENCY", cls.github_concurrency),
            github_cache_ttl_hours=_env_int("GITHUB_CACHE_TTL_HOURS", cls.github_cache_ttl_hours),
            cache_dir=Path(os.getenv("CACHE_DIR", str(cls.cache_dir)).strip()),
            use_disk_cache=_env_bool("USE_DISK_CACHE", cls.use_disk_cache),
            require_python_usage=_env_bool("REQUIRE_PYTHON_USAGE", cls.require_python_usage),
            allow_classical_ml=_env_bool("ALLOW_CLASSICAL_ML", cls.allow_classical_ml),
        )
