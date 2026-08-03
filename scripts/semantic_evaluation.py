from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from taigi_converter import TaigiConverter

SEMANTIC_SPLITS = ("train", "development", "holdout")
SEMANTIC_FAILURE_TYPES = (
    "semantic_correct",
    "under_conversion",
    "over_conversion",
    "wrong_sense",
    "proper_noun_damage",
    "grammar_error",
    "acceptable_identity",
    "formatting",
)
SEMANTIC_ORACLE_KINDS = (
    "verified_translation",
    "ai_semantic_review",
    "protected_proper_noun",
    "protected_technical_term",
)
CASE_ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+\Z")
_REQUIRED_REVIEW_FIELDS = ("provenance", "reviewed_by", "reviewed_at")
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "source",
        "expected",
        "category",
        "failure_type",
        "focus_terms",
        "oracle_kind",
        "provenance",
        "reviewed_by",
        "reviewed_at",
        "split",
        "allow_sentence_override",
        "sentence_override_reason",
    }
)


@dataclass(frozen=True, slots=True)
class SemanticEvaluationCase:
    case_id: str
    source: str
    expected: str
    category: str
    failure_type: str
    focus_terms: tuple[str, ...]
    oracle_kind: str
    provenance: str
    reviewed_by: str
    reviewed_at: str
    split: str
    allow_sentence_override: bool = False
    sentence_override_reason: str = ""

    def __post_init__(self) -> None:
        text_fields = (
            ("case_id", self.case_id),
            ("source", self.source),
            ("expected", self.expected),
            ("category", self.category),
            ("failure_type", self.failure_type),
            ("oracle_kind", self.oracle_kind),
            ("provenance", self.provenance),
            ("reviewed_by", self.reviewed_by),
            ("reviewed_at", self.reviewed_at),
            ("split", self.split),
            ("sentence_override_reason", self.sentence_override_reason),
        )
        invalid_types = [name for name, value in text_fields if not isinstance(value, str)]
        if invalid_types:
            raise ValueError("semantic case 文字欄位型別錯誤: " + ", ".join(invalid_types))
        if not CASE_ID_PATTERN.fullmatch(self.case_id):
            raise ValueError(f"無效 case_id: {self.case_id!r}")
        for name, value in (("source", self.source), ("expected", self.expected), ("category", self.category)):
            if not value.strip():
                raise ValueError(f"{name} 不可為空")
        if self.failure_type not in SEMANTIC_FAILURE_TYPES:
            raise ValueError(f"未知 failure_type: {self.failure_type}")
        if self.oracle_kind not in SEMANTIC_ORACLE_KINDS:
            raise ValueError(f"未知 semantic oracle_kind: {self.oracle_kind}")
        if self.split not in SEMANTIC_SPLITS:
            raise ValueError(f"未知 semantic split: {self.split}")
        if not isinstance(self.focus_terms, tuple) or not self.focus_terms:
            raise ValueError("focus_terms 必須是至少含一項的陣列")
        if len(set(self.focus_terms)) != len(self.focus_terms):
            raise ValueError("focus_terms 不可重複")
        for term in self.focus_terms:
            if not isinstance(term, str) or not term:
                raise ValueError("focus_terms 只能包含非空字串")
            if term not in self.source:
                raise ValueError(f"focus term 不在 source 中: {term!r}")
        missing = [name for name in _REQUIRED_REVIEW_FIELDS if not getattr(self, name).strip()]
        if missing:
            raise ValueError(f"{self.oracle_kind} 必須提供審查 metadata: " + ", ".join(missing))
        try:
            date.fromisoformat(self.reviewed_at)
        except ValueError as exc:
            raise ValueError("reviewed_at 必須是 YYYY-MM-DD") from exc
        if not isinstance(self.allow_sentence_override, bool):
            raise ValueError("allow_sentence_override 必須是 boolean")
        has_reason = bool(self.sentence_override_reason.strip())
        if self.allow_sentence_override != has_reason:
            raise ValueError("allow_sentence_override 與 sentence_override_reason 必須同時設定")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SemanticEvaluationCase:
        unknown = sorted(set(payload) - _CASE_FIELDS)
        missing = sorted(_CASE_FIELDS - set(payload))
        if unknown:
            raise ValueError(f"semantic case 含未知欄位: {unknown}")
        if missing:
            raise ValueError(f"semantic case 缺少欄位: {missing}")
        focus_terms = payload["focus_terms"]
        if not isinstance(focus_terms, list):
            raise ValueError("focus_terms 必須是 JSON array")
        return cls(
            case_id=payload["case_id"],
            source=payload["source"],
            expected=payload["expected"],
            category=payload["category"],
            failure_type=payload["failure_type"],
            focus_terms=tuple(focus_terms),
            oracle_kind=payload["oracle_kind"],
            provenance=payload["provenance"],
            reviewed_by=payload["reviewed_by"],
            reviewed_at=payload["reviewed_at"],
            split=payload["split"],
            allow_sentence_override=payload["allow_sentence_override"],
            sentence_override_reason=payload["sentence_override_reason"],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["focus_terms"] = list(self.focus_terms)
        return payload


@dataclass(frozen=True, slots=True)
class SemanticEvaluationResult:
    case: SemanticEvaluationCase
    output: str
    passed: bool
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case.case_id,
            "source": self.case.source,
            "expected": self.case.expected,
            "output": self.output,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 4),
            "split": self.case.split,
            "category": self.case.category,
            "failure_type": self.case.failure_type,
            "oracle_kind": self.case.oracle_kind,
        }


def load_semantic_cases(path: Path) -> list[SemanticEvaluationCase]:
    cases: list[SemanticEvaluationCase] = []
    seen_ids: dict[str, int] = {}
    seen_sources: dict[str, tuple[int, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 格式錯誤: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: semantic case 必須是 JSON object")
            try:
                case = SemanticEvaluationCase.from_dict(payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if case.case_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate case_id {case.case_id!r}；首次出現在第 {seen_ids[case.case_id]} 行"
                )
            if case.source in seen_sources:
                previous_line, previous_split = seen_sources[case.source]
                raise ValueError(
                    f"{path}:{line_number}: duplicate source 跨 semantic cases/splits；"
                    f"首次出現在第 {previous_line} 行 ({previous_split})"
                )
            seen_ids[case.case_id] = line_number
            seen_sources[case.source] = (line_number, case.split)
            cases.append(case)
    return cases


def run_semantic_cases(
    cases: Sequence[SemanticEvaluationCase],
    *,
    converter: TaigiConverter | None = None,
) -> list[SemanticEvaluationResult]:
    active_converter = converter or TaigiConverter()
    results: list[SemanticEvaluationResult] = []
    for case in cases:
        started = time.perf_counter()
        output = active_converter.convert(case.source)
        latency_ms = (time.perf_counter() - started) * 1000
        results.append(
            SemanticEvaluationResult(
                case=case,
                output=output,
                passed=output == case.expected,
                latency_ms=latency_ms,
            )
        )
    return results


def _counter(items: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def _latency_summary(results: Sequence[SemanticEvaluationResult]) -> dict[str, float]:
    values = sorted(result.latency_ms for result in results)
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    p95_index = max(int(len(values) * 0.95) - 1, 0)
    return {
        "mean_ms": round(mean(values), 4),
        "p95_ms": round(values[p95_index], 4),
        "max_ms": round(max(values), 4),
    }


def build_semantic_summary(
    cases: Sequence[SemanticEvaluationCase],
    results: Sequence[SemanticEvaluationResult],
    *,
    mismatch_limit: int = 20,
) -> dict[str, Any]:
    passed = sum(result.passed for result in results)
    failed_results = [result for result in results if not result.passed]
    case_count = len(cases)
    if len(results) != case_count:
        raise ValueError("cases 與 results 數量不一致")
    return {
        "case_count": case_count,
        "passed": passed,
        "failed": len(failed_results),
        "pass_rate": round(passed / case_count, 6) if case_count else 0.0,
        "counts_by_split": _counter(case.split for case in cases),
        "counts_by_category": _counter(case.category for case in cases),
        "counts_by_failure_type": _counter(case.failure_type for case in cases),
        "counts_by_oracle_kind": _counter(case.oracle_kind for case in cases),
        "failures_by_split": _counter(result.case.split for result in failed_results),
        "failures_by_category": _counter(result.case.category for result in failed_results),
        "failures_by_failure_type": _counter(result.case.failure_type for result in failed_results),
        "latency": _latency_summary(results),
        "mismatches": [result.to_dict() for result in failed_results[:mismatch_limit]],
        "mismatches_truncated": max(len(failed_results) - mismatch_limit, 0),
    }


def deterministic_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
