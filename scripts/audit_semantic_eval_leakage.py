from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regression_runner import load_all_regression_cases
from scripts.semantic_evaluation import (
    SemanticEvaluationCase,
    canonicalize_semantic_source,
    deterministic_json,
    load_semantic_cases,
)

DEFAULT_CASES_PATH = REPO_ROOT / "data" / "semantic_eval_cases.jsonl"
DEFAULT_LEXICON_PATH = REPO_ROOT / "data" / "lexicon_entries.jsonl"
DEFAULT_SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_active_exact_entries(path: Path) -> dict[str, list[dict[str, Any]]]:
    entries: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            if payload.get("status", "active") != "active" or payload.get("level") not in {"phrase", "sentence"}:
                continue
            src = payload.get("src")
            if not isinstance(src, str) or not src:
                continue
            entries.setdefault(src, []).append(
                {
                    "entry_id": payload.get("entry_id", ""),
                    "level": payload.get("level", ""),
                    "line": line_number,
                    "source": src,
                }
            )
    return entries


def _canonical_source_index(sources: Sequence[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for source in sorted(sources):
        index.setdefault(canonicalize_semantic_source(source), []).append(source)
    return index


def _canonical_entry_index(
    active_exact_entries: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for source in sorted(active_exact_entries):
        canonical_source = canonicalize_semantic_source(source)
        for entry in active_exact_entries[source]:
            normalized_entry = dict(entry)
            normalized_entry.setdefault("source", source)
            index.setdefault(canonical_source, []).append(normalized_entry)
    return index


def audit_semantic_leakage(
    cases: Sequence[SemanticEvaluationCase],
    *,
    regression_sources: set[str],
    active_exact_entries: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    regression_index = _canonical_source_index(tuple(regression_sources))
    exact_entry_index = _canonical_entry_index(active_exact_entries)

    for case in cases:
        canonical_source = canonicalize_semantic_source(case.source)
        matched_regression_sources = regression_index.get(canonical_source, [])
        if matched_regression_sources:
            findings.append(
                {
                    "kind": "regression_source_overlap",
                    "match_type": "raw" if case.source in matched_regression_sources else "canonical",
                    "case_id": case.case_id,
                    "split": case.split,
                    "source": case.source,
                    "canonical_source": canonical_source,
                    "matched_sources": matched_regression_sources,
                }
            )

        matched_entries = exact_entry_index.get(canonical_source, [])
        approved_sentence_entry_ids = set(case.sentence_override_entry_ids)
        overridden_sentence_entries = [
            entry
            for entry in matched_entries
            if entry.get("level") == "sentence"
            and entry.get("entry_id") in approved_sentence_entry_ids
        ]
        actionable_entries = [
            entry for entry in matched_entries if entry not in overridden_sentence_entries
        ]
        if actionable_entries:
            matched_sources = sorted(
                {str(entry.get("source", "")) for entry in actionable_entries}
            )
            finding: dict[str, Any] = {
                "kind": "exact_runtime_entry_overlap",
                "match_type": "raw" if case.source in matched_sources else "canonical",
                "case_id": case.case_id,
                "split": case.split,
                "source": case.source,
                "canonical_source": canonical_source,
                "matched_sources": matched_sources,
                "entries": actionable_entries,
            }
            if overridden_sentence_entries:
                finding["overridden_sentence_entries"] = overridden_sentence_entries
                finding["sentence_override_reason"] = case.sentence_override_reason
            findings.append(finding)

    counts = Counter(finding["kind"] for finding in findings)
    match_counts = Counter(finding["match_type"] for finding in findings)
    return {
        "summary": {
            "case_count": len(cases),
            "finding_count": len(findings),
            "clean": not findings,
            "counts_by_kind": dict(sorted(counts.items())),
            "counts_by_match_type": dict(sorted(match_counts.items())),
        },
        "findings": findings,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    cases: Sequence[SemanticEvaluationCase] | None = None,
    regression_sources: set[str] | None = None,
    active_exact_entries: dict[str, list[dict[str, Any]]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="稽核 semantic evaluation corpus 的資料洩漏與 exact override 重疊")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--scripts-dir", type=Path, default=DEFAULT_SCRIPTS_DIR)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args(argv)

    loaded_cases = list(cases) if cases is not None else load_semantic_cases(args.cases)
    loaded_regression_sources = regression_sources
    if loaded_regression_sources is None:
        loaded_regression_sources = {
            located.case.source for located in load_all_regression_cases(args.scripts_dir)
        }
    loaded_entries = active_exact_entries
    if loaded_entries is None:
        loaded_entries = load_active_exact_entries(args.lexicon)

    report = audit_semantic_leakage(
        loaded_cases,
        regression_sources=loaded_regression_sources,
        active_exact_entries=loaded_entries,
    )
    rendered = deterministic_json(report)
    stdout.write(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    return int(args.fail_on_findings and not report["summary"]["clean"])


if __name__ == "__main__":
    raise SystemExit(main())
