from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regression_runner import (
    LocatedRegressionCase,
    load_all_regression_cases,
)
from taigi_converter import TaigiConverter
from taigi_converter.models import ConversionResult, RuntimeLexiconEntry

REPORT_SCHEMA_VERSION = 2


def _json_value(value: Any) -> Any:
    """Convert frozen runtime metadata into deterministic JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_value(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    return value


def _entry_metadata(entry: RuntimeLexiconEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "src": entry.src,
        "tgt": entry.tgt,
        "level": entry.level,
        "tier": entry.tier,
        "priority": entry.priority,
        "context": _json_value(entry.context),
        "score": entry.score,
        "status": entry.status,
        "provenance": entry.source,
        "trust": entry.trust,
    }


def _active_entries(converter: TaigiConverter) -> list[RuntimeLexiconEntry]:
    return sorted(
        (entry for entry in converter.entries_by_index if entry.status == "active"),
        key=lambda entry: (entry.src, entry.entry_id),
    )


def runtime_active_unique_sources(converter: TaigiConverter) -> list[str]:
    """Return every unique source present in the active runtime, without exclusions."""
    return sorted({entry.src for entry in _active_entries(converter)})


def _convert_with_trace(converter: TaigiConverter, text: str) -> ConversionResult:
    # Deliberately use the normal runtime path. Do not pass suppression, allowlist,
    # review, or audit-specific profiles that could hide fixed-point failures.
    result = converter.convert(text, trace=True)
    if not isinstance(result, ConversionResult):
        raise TypeError("converter.convert(..., trace=True) must return ConversionResult")
    return result


def _phase_report(
    *,
    input_text: str,
    result: ConversionResult,
    entries_by_id: Mapping[str, RuntimeLexiconEntry],
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    matched_entry_ids: list[str] = []
    for match in result.matches:
        matched_entry_ids.append(match.entry_id)
        entry = entries_by_id.get(match.entry_id)
        row = match.to_dict()
        if entry is not None:
            row.update(
                {
                    "trust": entry.trust,
                    "context": _json_value(entry.context),
                    "provenance": entry.source,
                    "status": entry.status,
                }
            )
        else:
            row.update(
                {
                    "trust": None,
                    "context": None,
                    "provenance": None,
                    "status": None,
                }
            )
        matches.append(row)

    rules = [rule.to_dict() for rule in result.rules_applied]
    return {
        "input": input_text,
        "output": result.output,
        "entry_ids": matched_entry_ids,
        "rule_ids": [rule["rule_id"] for rule in rules],
        "matches": matches,
        "rules_applied": rules,
        "warnings": list(result.warnings),
    }


def _regression_occurrence_metadata(occurrence: LocatedRegressionCase) -> dict[str, Any]:
    case = occurrence.case
    return {
        "suite": occurrence.suite,
        "script": occurrence.script,
        "index": occurrence.index,
        "category": case.category,
        "expected": case.expected,
        "oracle_kind": case.oracle_kind,
        "provenance": case.provenance,
        "reviewed_by": case.reviewed_by,
        "reviewed_at": case.reviewed_at,
    }


def audit_runtime_fixed_points(
    converter: TaigiConverter,
    *,
    regression_occurrences: Sequence[LocatedRegressionCase] | None = None,
) -> dict[str, Any]:
    if regression_occurrences is None:
        regression_occurrences = load_all_regression_cases(REPO_ROOT / "scripts")

    active_entries = _active_entries(converter)
    entries_by_id = {entry.entry_id: entry for entry in active_entries}
    source_entries: dict[str, list[RuntimeLexiconEntry]] = {}
    for entry in active_entries:
        source_entries.setdefault(entry.src, []).append(entry)

    regression_sources: dict[str, list[LocatedRegressionCase]] = {}
    for occurrence in regression_occurrences:
        regression_sources.setdefault(occurrence.case.source, []).append(occurrence)

    runtime_unique_sources = set(source_entries)
    regression_unique_sources = set(regression_sources)
    unique_sources = sorted(runtime_unique_sources | regression_unique_sources)
    findings: list[dict[str, Any]] = []
    changed_first_pass_count = 0

    for source in unique_sources:
        producer_result = _convert_with_trace(converter, source)
        first = producer_result.output
        if first != source:
            changed_first_pass_count += 1

        consumer_result = _convert_with_trace(converter, first)
        second = consumer_result.output
        if second == first:
            continue

        findings.append(
            {
                "source": source,
                "first": first,
                "second": second,
                "source_entries": [_entry_metadata(entry) for entry in source_entries.get(source, ())],
                "regression_cases": [
                    _regression_occurrence_metadata(occurrence) for occurrence in regression_sources.get(source, ())
                ],
                "producer": _phase_report(
                    input_text=source,
                    result=producer_result,
                    entries_by_id=entries_by_id,
                ),
                "consumer": _phase_report(
                    input_text=first,
                    result=consumer_result,
                    entries_by_id=entries_by_id,
                ),
            }
        )

    non_idempotent_count = len(findings)
    non_idempotent_sources = {finding["source"] for finding in findings}
    unique_source_count = len(unique_sources)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "summary": {
            "runtime_active_entry_count": len(active_entries),
            "runtime_unique_source_count": len(runtime_unique_sources),
            "regression_case_count": len(regression_occurrences),
            "regression_unique_source_count": len(regression_unique_sources),
            "regression_only_unique_source_count": len(regression_unique_sources - runtime_unique_sources),
            "audited_unique_source_count": unique_source_count,
            "first_pass_changed_source_count": changed_first_pass_count,
            "idempotent_source_count": unique_source_count - non_idempotent_count,
            "non_idempotent_source_count": non_idempotent_count,
            "non_idempotent_runtime_source_count": len(non_idempotent_sources & runtime_unique_sources),
            "non_idempotent_regression_source_count": len(non_idempotent_sources & regression_unique_sources),
            "non_idempotent_regression_only_source_count": len(
                non_idempotent_sources & (regression_unique_sources - runtime_unique_sources)
            ),
            "non_idempotent_source_rate": (
                round(non_idempotent_count / unique_source_count, 6) if unique_source_count else 0.0
            ),
        },
        "findings": findings,
    }


def serialize_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _print_human(
    report: Mapping[str, Any],
    *,
    sample_limit: int,
    stdout: TextIO,
) -> None:
    summary = report["summary"]
    print("Runtime fixed-point audit", file=stdout)
    print(
        f"active entries: {summary['runtime_active_entry_count']}",
        file=stdout,
    )
    print(
        f"unique active sources: {summary['runtime_unique_source_count']}",
        file=stdout,
    )
    print(
        f"regression cases: {summary['regression_case_count']}",
        file=stdout,
    )
    print(
        f"unique regression sources: {summary['regression_unique_source_count']}",
        file=stdout,
    )
    print(
        f"audited unique sources: {summary['audited_unique_source_count']}",
        file=stdout,
    )
    print(
        f"first-pass changed: {summary['first_pass_changed_source_count']}",
        file=stdout,
    )
    print(
        f"non-idempotent: {summary['non_idempotent_source_count']} ({summary['non_idempotent_source_rate']:.6f})",
        file=stdout,
    )

    findings = report["findings"]
    if sample_limit <= 0 or not findings:
        return

    sample_count = min(sample_limit, len(findings))
    print(f"\nfindings sample: {sample_count}/{len(findings)}", file=stdout)
    for finding in findings[:sample_count]:
        producer_ids = ",".join(finding["producer"]["entry_ids"]) or "-"
        consumer_ids = ",".join(finding["consumer"]["entry_ids"]) or "-"
        print(
            f"- {finding['source']} -> {finding['first']} -> {finding['second']}",
            file=stdout,
        )
        print(
            f"  producer={producer_ids} consumer={consumer_ids}",
            file=stdout,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    converter: TaigiConverter | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "掃描所有 runtime active unique sources 與 regression sources，檢查 convert(src) 是否為 runtime fixed point"
        ),
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="只掃 runtime active sources，不載入 regression sources",
    )
    parser.add_argument("--json", action="store_true", help="輸出完整 deterministic JSON report")
    parser.add_argument("--output", type=Path, help="將完整 JSON report 寫入指定路徑")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="人類可讀摘要最多顯示幾筆 non-idempotent findings",
    )
    parser.add_argument(
        "--fail-on-non-idempotent",
        action="store_true",
        help="若發現任一 non-idempotent source，回傳 exit code 1",
    )
    args = parser.parse_args(argv)

    output_stream = stdout if stdout is not None else sys.stdout
    runtime_converter = converter if converter is not None else TaigiConverter()
    regression_occurrences = [] if args.runtime_only else None
    report = audit_runtime_fixed_points(
        runtime_converter,
        regression_occurrences=regression_occurrences,
    )
    payload = serialize_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")

    if args.json:
        print(payload, end="", file=output_stream)
    else:
        _print_human(
            report,
            sample_limit=max(args.sample_limit, 0),
            stdout=output_stream,
        )

    non_idempotent_count = report["summary"]["non_idempotent_source_count"]
    return 1 if args.fail_on_non_idempotent and non_idempotent_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
