from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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

# These are review signals, not proof that an expected answer is wrong. Some forms can
# be legitimate Taiwanese Hokkien vocabulary, quotations, or proper names.
MANDARIN_SURFACE_MARKERS = (
    "哪裡",
    "什麼",
    "沒有",
    "有沒有",
    "可以",
    "需要",
    "不要",
    "讓",
    "我們",
    "覺得",
    "如果",
    "所以",
    "時候",
    "一起",
    "東西",
)


LocatedCase = LocatedRegressionCase


@dataclass(frozen=True)
class AuditFinding:
    kind: str
    severity: str
    suite: str
    script: str
    index: int
    category: str
    source: str
    expected: str
    detail: str


def load_all_cases(scripts_dir: Path) -> list[LocatedCase]:
    return load_all_regression_cases(scripts_dir)


def audit_cases(
    located_cases: list[LocatedCase],
    *,
    converter: TaigiConverter | None = None,
) -> dict[str, Any]:
    findings: list[AuditFinding] = []
    exact_duplicates: dict[tuple[str, str, str], list[LocatedCase]] = defaultdict(list)
    expected_by_source: dict[str, set[str]] = defaultdict(set)
    locations_by_source: dict[str, list[LocatedCase]] = defaultdict(list)
    identity_observation_count = 0
    surface_marker_observation_count = 0

    for located in located_cases:
        case = located.case
        exact_duplicates[(case.category, case.source, case.expected)].append(located)
        expected_by_source[case.source].add(case.expected)
        locations_by_source[case.source].append(located)

        if case.source == case.expected:
            identity_observation_count += 1
            if case.oracle_kind in {"unreviewed", "verified_translation"}:
                findings.append(
                    _finding(
                        "identity_expected",
                        "review",
                        located,
                        "source 與 expected 完全相同；只能證明 passthrough 相容性，不能單獨證明翻譯正確",
                    )
                )

        markers = [marker for marker in MANDARIN_SURFACE_MARKERS if marker in case.expected]
        if markers:
            surface_marker_observation_count += 1
            if case.oracle_kind == "unreviewed":
                findings.append(
                    _finding(
                        "mandarin_surface_marker",
                        "review",
                        located,
                        "未分類 expected 含需人工判讀的華語表面詞：" + "、".join(markers),
                    )
                )

        if case.oracle_kind == "unreviewed":
            findings.append(
                _finding(
                    "unreviewed_oracle",
                    "governance",
                    located,
                    "尚未標示人工驗證、相容性快照、專名保留或格式正規化等 oracle 類型",
                )
            )

        if converter is not None:
            reconverted = converter.convert(case.expected)
            assert isinstance(reconverted, str)
            if reconverted != case.expected:
                findings.append(
                    _finding(
                        "non_idempotent_expected",
                        "risk",
                        located,
                        f"expected 再轉換後變成：{reconverted}",
                    )
                )

    exact_duplicate_location_count = 0
    explained_duplicate_location_count = 0
    for duplicates in exact_duplicates.values():
        if len(duplicates) < 2:
            continue
        exact_duplicate_location_count += len(duplicates)
        groups = {located.case.duplicate_group for located in duplicates}
        reasons = {located.case.duplicate_reason for located in duplicates}
        if len(groups) == 1 and len(reasons) == 1 and "" not in groups and "" not in reasons:
            explained_duplicate_location_count += len(duplicates)
            continue
        for located in duplicates:
            findings.append(
                _finding(
                    "duplicate_case",
                    "governance",
                    located,
                    f"同 category/source/expected 共重複 {len(duplicates)} 次，且未提供一致的 duplicate_group/duplicate_reason",
                )
            )

    for source, expected_values in expected_by_source.items():
        if len(expected_values) < 2:
            continue
        detail = "同一 source 存在互相衝突的 expected：" + " | ".join(sorted(expected_values))
        for located in locations_by_source[source]:
            findings.append(_finding("conflicting_expected", "error", located, detail))

    finding_counts = Counter(finding.kind for finding in findings)
    severity_counts = Counter(finding.severity for finding in findings)
    suite_counts = Counter(located.suite for located in located_cases)
    identity_counts = Counter(
        located.suite for located in located_cases if located.case.source == located.case.expected
    )
    oracle_counts = Counter(located.case.oracle_kind for located in located_cases)

    return {
        "summary": {
            "case_count": len(located_cases),
            "suite_count": len(suite_counts),
            "suite_case_counts": dict(sorted(suite_counts.items())),
            "identity_expected_count": identity_observation_count,
            "identity_expected_rate": _rate(identity_observation_count, len(located_cases)),
            "unreviewed_identity_expected_count": finding_counts["identity_expected"],
            "identity_by_suite": dict(sorted(identity_counts.items())),
            "non_idempotent_expected_count": finding_counts["non_idempotent_expected"],
            "mandarin_surface_marker_case_count": surface_marker_observation_count,
            "unreviewed_mandarin_surface_marker_case_count": finding_counts["mandarin_surface_marker"],
            "exact_duplicate_location_count": exact_duplicate_location_count,
            "explained_duplicate_location_count": explained_duplicate_location_count,
            "duplicate_case_location_count": finding_counts["duplicate_case"],
            "conflicting_expected_location_count": finding_counts["conflicting_expected"],
            "oracle_kind_counts": dict(sorted(oracle_counts.items())),
            "human_verified_translation_count": oracle_counts["verified_translation"],
            "ai_semantic_review_count": oracle_counts["ai_semantic_review"],
            "finding_counts": dict(sorted(finding_counts.items())),
            "severity_counts": dict(sorted(severity_counts.items())),
        },
        "findings": [asdict(finding) for finding in findings],
    }


def _finding(kind: str, severity: str, located: LocatedCase, detail: str) -> AuditFinding:
    return AuditFinding(
        kind=kind,
        severity=severity,
        suite=located.suite,
        script=located.script,
        index=located.index,
        category=located.case.category,
        source=located.case.source,
        expected=located.case.expected,
        detail=detail,
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _print_human(
    report: dict[str, Any],
    *,
    sample_limit: int,
    stdout: TextIO = sys.stdout,
) -> None:
    summary = report["summary"]
    print(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        file=stdout,
    )
    if sample_limit <= 0:
        return

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in report["findings"]:
        grouped[finding["kind"]].append(finding)
    for kind in sorted(grouped):
        rows = grouped[kind]
        print(
            f"\n[{kind}] total={len(rows)} sample={min(sample_limit, len(rows))}",
            file=stdout,
        )
        for row in rows[:sample_limit]:
            print(
                f"- {row['script']}:{row['index']} [{row['category']}] {row['source']} -> {row['expected']}",
                file=stdout,
            )
            print(f"  {row['detail']}", file=stdout)


def main(
    argv: Sequence[str] | None = None,
    *,
    located_cases: list[LocatedCase] | None = None,
    converter: TaigiConverter | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="稽核 regression expected 的 oracle 品質；此工具找 review signals，不宣稱自動判定語意正誤",
    )
    parser.add_argument("--json", action="store_true", help="輸出完整 machine-readable JSON")
    parser.add_argument("--output", type=Path, help="將完整 JSON report 寫入指定路徑")
    parser.add_argument("--sample-limit", type=int, default=5, help="人類可讀輸出每類最多顯示幾筆")
    parser.add_argument(
        "--skip-idempotency",
        action="store_true",
        help="略過 expected 再轉換檢查（可避免載入 converter）",
    )
    parser.add_argument(
        "--fail-on-conflict",
        action="store_true",
        help="若同一 source 有互相衝突 expected，回傳非零 exit code",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="若存在任何未解釋的 oracle 品質 finding，回傳非零 exit code",
    )
    args = parser.parse_args(argv)

    output_stream = stdout if stdout is not None else sys.stdout
    cases = located_cases if located_cases is not None else load_all_cases(REPO_ROOT / "scripts")
    runtime_converter = None
    if not args.skip_idempotency:
        runtime_converter = converter if converter is not None else TaigiConverter()
    report = audit_cases(cases, converter=runtime_converter)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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

    conflicts = report["summary"]["conflicting_expected_location_count"]
    if args.fail_on_findings and report["findings"]:
        return 1
    return 1 if args.fail_on_conflict and conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
