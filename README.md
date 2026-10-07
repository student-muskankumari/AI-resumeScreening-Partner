# AI Resume Screening & Ranking

Reads a folder of resumes, rejects candidates who do not meet the minimum
Python + AI requirement, scores the rest out of 100, adds a small public
GitHub signal, and writes a ranked, explainable result file.

```
resumes/ ─► parse ─► extract ─► HARD FILTER ─► evidence ─► GitHub ─► score ─► rank ─► results.json
            (PDF,    (name,     (Python AND     (Groq →     (0–10)    (rubric
            links)   email,      AI, in code)    Gemini →              in code)
                     GitHub)                     rules)
```

## Quick start

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # optional: add keys (see below)

python main.py --input ./resumes --output ./output/results.json
```

Other commands:

```bash
python main.py --explain candidate_07.pdf        # how one candidate was graded, and why
python main.py --input ./resumes --no-llm        # rules only, no model calls
python main.py --input ./resumes --no-github     # skip GitHub enrichment
python -m pytest                                 # 133 tests, no network, no keys
```

### Environment variables

All are optional. With none set the pipeline still runs end to end.

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY`, `GROQ_MODEL` | Primary model (Groq free tier). |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Fallback model (Gemini free tier). |
| `GITHUB_TOKEN` | Raises the GitHub limit from 60 to 5,000 requests/hour. A token with no scopes is enough. |
| `REQUIRE_PYTHON_USAGE`, `ALLOW_CLASSICAL_ML` | Eligibility policy switches (see Design Decisions). |
| `LLM_TOKENS_PER_MINUTE`, `LLM_CONCURRENCY`, `GITHUB_CONCURRENCY`, `CACHE_DIR` | Pacing, concurrency and cache location. |

A first run with model keys takes roughly 15–20 minutes for 50 resumes,
because calls are paced under the free tier's tokens-per-minute limit.
Results are cached in `.cache/`, so later runs take seconds.

## Output

`output/results.json` has three parts:

- `batch_summary`: total, successfully parsed, eligible, rejected,
  failed/unreadable, duplicates, which evidence source was used, GitHub
  lookup statuses.
- `scoring_rubric`: the full grading basis (every rule and its points),
  written once so each candidate is graded on the same published rules.
- `candidates`: eligible candidates first, ranked highest score first, then
  rejected candidates with explicit reasons, then failed files.

Each eligible candidate carries the fields from the assignment (`rank`,
`candidate_name`, `eligible`, `total_score`, `score_breakdown`,
`matched_skills`, `project_summary`, `github_summary`, `strengths`,
`concerns`) plus:

- `penalties`: each deduction with its type, points and reason.
- `grading_basis`: for every rule, the points given, the maximum, and the
  resume line it relied on. Rules that scored zero are listed too.
- `processing`: parser used, evidence source (`groq`, `gemini` or `rules`)
  and any warnings (for example a model call that failed and fell back).

`python main.py --explain <file or name>` prints the same basis in the terminal.

## Design Decisions

### Filtering strategy

The hard filter is plain code, not an LLM, so the same resume always gets
the same decision and the rules can be unit-tested (`src/screener/eligibility.py`).

1. The text is split into lines and each line is tagged by where it sits:
   **used** (project or job description), **claimed** (summary), **listed**
   (skills list) or **mention** (education, certificates). Section headings
   are matched against a wide vocabulary, and a sentence that describes
   building something counts as usage even under an unrecognised heading, so
   the filter does not depend on one resume layout.
2. **Python evidence**: Python or a Python-only framework appears as a
   skill, project technology or work technology.
3. **AI evidence**: a named LLM / RAG / agent / embedding technology or
   technique is *used in a project or job*. A framework name in a skills
   list, generic "AI-powered" wording, and AI coding assistants (Copilot,
   Cursor, "AI-assisted development") do not count.
4. Both must hold. Other languages (Java, JavaScript, React, Next.js) are
   never a reason to reject.

Two judgment calls, both switchable in `.env`:

- **Python only in a skills list** passes the gate (the assignment lists
  "genuine skill" as sufficient) but earns one third of the Python points
  and a visible concern. `REQUIRE_PYTHON_USAGE=true` rejects it instead.
- **Classical ML / CV only** (no LLM, RAG or agents) passes the gate, but its
  AI-depth score is capped at 10 of 40, so it cannot rank near the top.
  `ALLOW_CLASSICAL_ML=false` rejects it instead.

On the provided 50 resumes: 39 eligible, 11 rejected. Two of the rejected
candidates describe real LLM/RAG work but show no Python at all; they are
rejected by rule, as the assignment requires.

### Scoring strategy

The evidence source only reports *what the resume shows*; all arithmetic is
in code (`src/screener/scoring.py`) with the rubric in `src/screener/config.py`.

| Category | Max | Rules |
|---|---|---|
| AI / agentic / RAG depth | 40 | real AI system 5, implementation depth 7, retrieval/RAG 6, agents/tool calling 6, state/orchestration 5, data pipeline 4, evaluation 4, business logic 3 |
| Python & backend | 30 | Python 10, FastAPI 6, async 4, PostgreSQL 5, Redis 5 |
| Cloud / deployment / full stack | 15 | cloud platform 5 (GCP full, other clouds 3), Docker 4, deployment/CI-CD 3, React/Next.js in an end-to-end system 3 |
| GitHub | 10 | recent activity 0–5, maintained/relevant repositories 0–5 |
| Engineering depth | 5 | 1 point per signal: testing, architecture, caching, queues, observability, concurrency, failure handling |

- AI rules are graded 0 / half / full (no evidence, basic, clearly implemented).
- Backend and cloud rules pay in full only when the technology is used in a
  project or job: 60% when only claimed in a summary, one third when only
  listed. A skills list alone can never earn full points.
- Backend items shown only outside a Python stack count for half.
- **Penalties**: a thin LLM wrapper loses 5, 10 or 15 points depending on how
  little surrounds the model call; a project listed with no implementation
  detail loses 5. Combined deductions are capped at 15.
- `total = clamp(categories + penalties, 0, 100)`. Ties are broken by AI
  depth, then Python/backend, engineering depth, GitHub and file name, so
  the order is stable between runs.

Tests assert that items add up to their category, categories never exceed
their weight, and a wrapper scores below RAG, which scores below agentic RAG
with evaluation.

### LLM usage

- One call per eligible resume, at temperature 0, asking for a single JSON
  object that is validated against a Pydantic schema (`Evidence` in
  `src/screener/models.py`). The model never decides eligibility and never
  produces a number that goes into the score directly.
- Every signal must come with a short quote from the resume. Quotes are
  checked against the resume text; a signal the model cannot back with a
  real quote is dropped. The resume is passed as delimited data and the
  prompt tells the model to ignore instructions inside it.
- Providers sit behind one small interface (`src/screener/llm/base.py`).
  Order: **Groq → Gemini → rule-based extraction**, one retry each, no loops.
  A bad key or three consecutive failures skips that provider for the rest
  of the batch. The rule-based fallback fills the same schema, so scoring is
  identical whichever source answered, and `evidence_source` records which
  one it was.
- Rejected resumes never reach the model, calls are paced under the free
  tier's token limit, and answers are cached by resume text, prompt version
  and model.

The rule-based fallback is weaker than a model at judging how deep a project
really is: it works from term lists and line context, so a keyword-dense
resume can score higher than it should. Candidates scored that way are
labelled `rules` in the output.

### GitHub scoring

- The username comes from the PDF's link annotations first (many resumes hide
  the URL behind the word "GitHub"), then from linked repositories, then
  from the text. If a resume points at two accounts, both are checked and
  the more active one is used.
- One API call per candidate (their own repositories sorted by last push)
  provides every signal, which keeps a 50-resume batch inside even the
  unauthenticated limit.
- **Recent activity (0–5)**: 3 / 2 / 1 for a push in the last 30 / 90 / 365
  days, plus up to 2 for several repositories pushed in the last 180 days.
- **Maintained and relevant (0–5)**: non-fork, non-archived repositories
  pushed in the last 12 months that are Python or AI-related.
- Missing profile, unknown user, rate limit, timeout or API error all score
  0, are recorded with a status, and never affect eligibility or stop the
  batch. After the first rate-limit response no further calls are made.
- If GitHub rejects the token, the run continues without it and reports
  `github_token_rejected: true` in the batch summary.

### Reliability

- Each resume is processed inside its own error boundary: a corrupt, empty
  or password-protected file is reported as failed and the batch continues.
- Byte-identical files and files with identical text are reported as
  duplicates and processed once.
- PyMuPDF reads text in content order (needed for multi-column resumes) and
  switches to position order only when headings are stranded from their
  content (some design-tool exports). pypdf is the fallback parser. DOCX
  and TXT are also read.

## Bonus items included

- DOCX and TXT parsing.
- FastAPI interface: `uvicorn screener.api:app --app-dir src`, then
  `POST /screen` (upload files, or use the server folder in `RESUME_DIR`)
  and `GET /results`. It calls the same pipeline as the CLI.
- Bounded async for model and GitHub calls, with disk caching.
- Terminal summary and per-candidate explanation.
- Integration tests on synthetic resumes.

## Limitations

- The model adapters are tested against mocked HTTP responses; a live run
  depends on the keys, models and quotas of your accounts.
- No OCR: a scanned, image-only resume is reported as unreadable.
- Technologies are credited per resume, not per project, on the rule-based
  path. Redis used in a Node service still counts as Redis.
- Resume claims are not verified beyond checking that GitHub activity exists.

## If I Had More Time

1. **Calibrate against human labels.** Have a reviewer rank 20–30 resumes
   and tune rubric points and thresholds to match, then track agreement
   between model evidence and rule evidence.
2. **Verify claims against linked repositories.** Check that the projects a
   resume describes exist in the candidate's GitHub and contain the stated
   stack, instead of scoring activity alone.
3. **OCR and stronger layout handling** for scanned and heavily designed PDFs.
4. **Per-project attribution**, so backend and cloud credit is tied to the
   project that used the technology.

## Project layout

```
main.py                    CLI entry point
src/screener/
  config.py                weights, rubric, thresholds, settings from env
  models.py                Pydantic schemas (Evidence, results)
  taxonomy.py              term lists for the rule engine
  ingest.py                file discovery, hashing, duplicates
  parsing.py               PDF / DOCX / TXT to text and links
  sections.py              lines tagged used / claimed / listed
  extract.py               name, email, GitHub username, skills
  eligibility.py           the hard filter
  analysis.py              Groq -> Gemini -> rules chain, quote check, cache
  llm/                     provider interface, Groq, Gemini, rules, prompt
  scoring.py               rubric scoring, penalties, ranking
  github.py                GitHub enrichment
  pipeline.py              batch orchestration
  report.py                results.json, summary, explain view
  api.py                   optional FastAPI interface
tests/                     133 tests
resumes/                   input resumes
output/results.json        generated results for the provided set
```

The resumes and `results.json` contain real people's contact details. Keep
the repository private.
