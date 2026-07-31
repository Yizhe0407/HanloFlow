from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_lexicon_risks import audit_entries, load_entries, observe_runtime_sources
from scripts.regression_runner import load_all_regression_cases
from taigi_converter import ConversionResult, TaigiConverter

REGISTRY_SCHEMA_VERSION = 1
VALID_DISPOSITIONS = frozenset({"accepted_legacy", "exclude", "context_gate", "promote", "shadowed"})
RESOLVED_DISPOSITIONS = frozenset({"exclude", "context_gate", "promote", "shadowed"})
_REQUIRED_FIELDS = frozenset(
    {
        "entry_id",
        "finding_fingerprint",
        "signals",
        "disposition",
        "reason_code",
        "evidence",
        "reviewed_by",
        "reviewed_at",
        "source_snapshot",
    }
)
_FINGERPRINT_ENTRY_FIELDS = (
    "entry_id",
    "src",
    "tgt",
    "level",
    "tier",
    "priority",
    "trust",
    "source",
    "context",
)


def finding_fingerprint(finding: Mapping[str, Any]) -> str:
    payload = {
        "entry": {field: finding.get(field) for field in _FINGERPRINT_ENTRY_FIELDS},
        "signals": sorted(str(signal["kind"]) for signal in finding.get("signals", ())),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_snapshot(finding: Mapping[str, Any]) -> dict[str, Any]:
    return {field: finding.get(field) for field in _FINGERPRINT_ENTRY_FIELDS if field != "entry_id"}


def collect_regression_evidence(
    finding_ids: set[str],
    *,
    scripts_dir: Path = REPO_ROOT / "scripts",
    converter: TaigiConverter | None = None,
) -> dict[str, dict[str, Any]]:
    runtime = converter or TaigiConverter()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_sets: dict[str, set[str]] = defaultdict(set)
    case_counts: Counter[str] = Counter()
    for located in load_all_regression_cases(scripts_dir):
        result = runtime.convert(located.case.source, trace=True)
        if not isinstance(result, ConversionResult):
            raise TypeError("trace=True 必須回傳 ConversionResult")
        matched_ids = finding_ids.intersection(match.entry_id for match in result.matches)
        for entry_id in matched_ids:
            case_counts[entry_id] += 1
            source_sets[entry_id].add(located.case.source)
            counts[entry_id][located.case.oracle_kind] += 1
    return {
        entry_id: {
            "regression_trace_count": case_counts[entry_id],
            "regression_source_count": len(source_sets[entry_id]),
            "oracle_counts": dict(sorted(counts[entry_id].items())),
        }
        for entry_id in sorted(finding_ids)
    }


def load_registry(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"disposition registry 不存在: {path}"]
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"registry line {line_number}: JSON 無法解析: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"registry line {line_number}: row 必須是 object")
                continue
            rows.append(row)
    return rows, errors


def validate_registry(
    rows: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    *,
    source_entries: Sequence[Any] = (),
    regression_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    registry_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        missing = sorted(_REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"registry row {index}: 缺少欄位 {missing}")
        entry_id = row.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append(f"registry row {index}: entry_id 必須是非空字串")
            continue
        if entry_id in registry_by_id:
            errors.append(f"registry row {index}: entry_id 重複: {entry_id}")
        registry_by_id[entry_id] = row
        if row.get("disposition") not in VALID_DISPOSITIONS:
            errors.append(f"registry row {index}: disposition 不合法: {row.get('disposition')!r}")
        if not isinstance(row.get("signals"), list) or not all(
            isinstance(item, str) for item in row.get("signals", [])
        ):
            errors.append(f"registry row {index}: signals 必須是字串陣列")
        for field in ("finding_fingerprint", "reason_code", "reviewed_by", "reviewed_at"):
            if not isinstance(row.get(field), str) or not row.get(field):
                errors.append(f"registry row {index}: {field} 必須是非空字串")
        if not isinstance(row.get("source_snapshot"), dict):
            errors.append(f"registry row {index}: source_snapshot 必須是 object")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            errors.append(f"registry row {index}: evidence 必須是 object")

    findings_by_id = {str(finding["entry_id"]): finding for finding in findings}
    unclassified = sorted(set(findings_by_id) - set(registry_by_id))
    if unclassified:
        errors.append(f"目前 findings 有 {len(unclassified)} 筆未分類: {unclassified[:20]}")

    source_by_id = {located.entry.entry_id: located for located in source_entries}
    stale: list[str] = []
    resolved: list[str] = []
    disposition_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for entry_id, row in registry_by_id.items():
        disposition = str(row.get("disposition"))
        disposition_counts[disposition] += 1
        reason_counts[str(row.get("reason_code"))] += 1
        current = findings_by_id.get(entry_id)
        if disposition == "accepted_legacy":
            if current is None:
                errors.append(f"{entry_id}: accepted_legacy 已不再是目前 finding，必須改為 resolved disposition")
                continue
            expected_fingerprint = finding_fingerprint(current)
            if row.get("finding_fingerprint") != expected_fingerprint:
                stale.append(entry_id)
            current_signals = sorted(str(signal["kind"]) for signal in current.get("signals", ()))
            if sorted(row.get("signals", ())) != current_signals:
                errors.append(f"{entry_id}: signals 與目前 finding 不一致")
        elif disposition in RESOLVED_DISPOSITIONS:
            resolved.append(entry_id)
            if current is not None:
                errors.append(f"{entry_id}: disposition={disposition} 但危險 finding 仍在 runtime 生效")
            located = source_by_id.get(entry_id)
            if located is None:
                errors.append(f"{entry_id}: resolved disposition 找不到原始詞條")
            elif located.entry.status == "active" and located.runtime_eligible:
                errors.append(f"{entry_id}: resolved disposition 的原始詞條仍 active 且 runtime eligible")
            resolution_ids = row.get("resolution_entry_ids", [])
            if not isinstance(resolution_ids, list):
                errors.append(f"{entry_id}: resolution_entry_ids 必須是陣列")
                resolution_ids = []
            elif disposition in {"context_gate", "promote"} and not resolution_ids:
                errors.append(f"{entry_id}: {disposition} 必須提供 resolution_entry_ids")

            for resolution_id in resolution_ids:
                if not isinstance(resolution_id, str) or not resolution_id:
                    errors.append(f"{entry_id}: resolution entry ID 必須是非空字串: {resolution_id!r}")
                    continue
                resolution = source_by_id.get(resolution_id)
                if resolution is None:
                    errors.append(f"{entry_id}: resolution entry 不存在: {resolution_id}")
                    continue
                if resolution.entry.status != "active" or not resolution.runtime_eligible:
                    errors.append(f"{entry_id}: resolution entry 未 active/runtime eligible: {resolution_id}")
                if located is not None and disposition in {"context_gate", "promote", "exclude"}:
                    original_src = located.entry.src
                    resolution_src = resolution.entry.src
                    if original_src not in resolution_src:
                        errors.append(
                            f"{entry_id}: resolution entry source 未涵蓋原 source: "
                            f"{resolution_id} ({resolution_src!r} 不含 {original_src!r})"
                        )

        if regression_evidence is not None:
            expected = regression_evidence.get(
                entry_id,
                {"regression_trace_count": 0, "regression_source_count": 0, "oracle_counts": {}},
            )
            if row.get("evidence") != expected:
                errors.append(f"{entry_id}: regression evidence 漂移")

    if stale:
        errors.append(f"{len(stale)} 筆 finding fingerprint 漂移: {stale[:20]}")

    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "summary": {
            "registry_count": len(rows),
            "current_finding_count": len(findings),
            "unclassified_count": len(unclassified),
            "resolved_count": len(resolved),
            "stale_fingerprint_count": len(stale),
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "error_count": len(errors),
        },
        "errors": errors,
    }


def audit_registry(
    *,
    data_path: Path,
    registry_path: Path,
    verify_regression_evidence: bool = True,
) -> dict[str, Any]:
    entries = load_entries(data_path)
    observations = observe_runtime_sources(entries)
    risk_report = audit_entries(entries, runtime_observations=observations, limit=None)
    findings = risk_report["findings"]
    rows, load_errors = load_registry(registry_path)
    evidence = (
        collect_regression_evidence({str(finding["entry_id"]) for finding in findings})
        if verify_regression_evidence
        else None
    )
    report = validate_registry(
        rows,
        findings,
        source_entries=entries,
        regression_evidence=evidence,
    )
    report["errors"] = load_errors + report["errors"]
    report["summary"]["error_count"] = len(report["errors"])
    report["risk_summary"] = risk_report["summary"]
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="稽核低信任詞典風險的逐筆 disposition registry")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "lexicon_entries.jsonl")
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "data" / "lexicon_risk_dispositions.jsonl",
    )
    parser.add_argument("--skip-regression-evidence", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true", help="有 registry 錯誤、漂移或未分類時失敗")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout) -> int:
    args = _parse_args(argv)
    report = audit_registry(
        data_path=args.data,
        registry_path=args.registry,
        verify_regression_evidence=not args.skip_regression_evidence,
    )
    if args.json or args.output:
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        summary = report["summary"]
        text = (
            f"registry: {summary['registry_count']}\n"
            f"current findings: {summary['current_finding_count']}\n"
            f"unclassified: {summary['unclassified_count']}\n"
            f"resolved: {summary['resolved_count']}\n"
            f"errors: {summary['error_count']}\n"
        )
        if report["errors"]:
            text += "\n".join(f"- {error}" for error in report["errors"][:50]) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        stdout.write(text)
    return 1 if args.fail_on_findings and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
