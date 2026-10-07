"""AI resume screening and ranking - command line entry point.

    python main.py --input ./resumes --output ./output/results.json
    python main.py --explain candidate_07.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from screener.config import Settings          # noqa: E402
from screener.pipeline import run_batch       # noqa: E402
from screener import report                   # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen a folder of resumes: hard-filter on Python + AI evidence, "
                    "score eligible candidates out of 100, enrich with public GitHub activity, rank.")
    parser.add_argument("--input", type=Path, help="Folder containing the resumes (PDF; DOCX/TXT also read).")
    parser.add_argument("--output", type=Path, default=Path("output/results.json"),
                        help="Where to write the JSON results (default: output/results.json).")
    parser.add_argument("--explain", metavar="FILE_OR_NAME",
                        help="Show how one candidate was graded and on what basis. "
                             "With --input it runs first; without it, it reads the existing --output file.")
    parser.add_argument("--no-llm", action="store_true", help="Skip model calls; extract evidence by rules only.")
    parser.add_argument("--no-github", action="store_true", help="Skip GitHub enrichment.")
    parser.add_argument("--top", type=int, default=10, help="How many ranked candidates to print (default 10).")
    parser.add_argument("--quiet", action="store_true", help="Do not print the summary table.")
    parser.add_argument("--verbose", action="store_true", help="Show progress logs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Names and quotes can contain characters a legacy Windows console cannot
    # encode; never let printing crash the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    # httpx logs full request URLs at INFO; keep them out of normal output.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.input is None:
        if not args.explain:
            print("error: --input is required (or use --explain with an existing results file)", file=sys.stderr)
            return 2
        if not args.output.exists():
            print(f"error: {args.output} not found; run with --input first", file=sys.stderr)
            return 2
        data = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        settings = Settings.from_env()
        if not args.no_llm and not (settings.groq_api_key or settings.gemini_api_key):
            print("note: no GROQ_API_KEY or GEMINI_API_KEY set; evidence will be extracted by rules.",
                  file=sys.stderr)
        try:
            batch = asyncio.run(run_batch(args.input, settings,
                                          use_llm=not args.no_llm, use_github=not args.no_github))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        data = report.build_output(batch)
        report.write_json(args.output, data)
        if not args.quiet:
            print(report.format_summary(data, top=args.top))
            print(f"Results written to {args.output}")

    if args.explain:
        candidate = report.find_candidate(data, args.explain)
        if candidate is None:
            print(f"error: no candidate matching '{args.explain}'", file=sys.stderr)
            return 1
        print(report.format_explanation(candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
