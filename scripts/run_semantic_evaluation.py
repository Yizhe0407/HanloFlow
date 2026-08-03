from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.semantic_evaluation import (
    SEMANTIC_SPLITS,
    SemanticEvaluationCase,
    build_semantic_summary,
    deterministic_json,
    load_semantic_cases,
    run_semantic_cases,
)
from taigi_converter import TaigiConverter

DEFAULT_CASES_PATH = REPO_ROOT / "data" / "semantic_eval_cases.jsonl"


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    cases: Sequence[SemanticEvaluationCase] | None = None,
    converter: TaigiConverter | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="執行獨立 semantic evaluation exact-match baseline")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--split", action="append", choices=SEMANTIC_SPLITS, default=[])
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--mismatch-limit", type=int, default=20)
    parser.add_argument(
        "--include-latency",
        action="store_true",
        help="在非 deterministic 的診斷輸出中包含 latency",
    )
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if args.mismatch_limit < 0:
        parser.error("--mismatch-limit 不可小於 0")

    selected_cases = list(cases) if cases is not None else load_semantic_cases(args.cases)
    if args.split:
        requested = set(args.split)
        selected_cases = [case for case in selected_cases if case.split in requested]
    results = run_semantic_cases(selected_cases, converter=converter)
    summary = build_semantic_summary(
        selected_cases,
        results,
        mismatch_limit=args.mismatch_limit,
        include_latency=args.include_latency,
    )
    rendered = deterministic_json(summary)
    stdout.write(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    return int(args.fail_on_mismatch and summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
