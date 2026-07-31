from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_runtime_fixed_points import (
    audit_runtime_fixed_points,
    main,
    runtime_active_unique_sources,
    serialize_report,
)
from scripts.regression_runner import LocatedRegressionCase, RegressionCase
from taigi_converter.models import ConversionResult, MatchTrace, RuntimeLexiconEntry


def _entry(
    entry_id: str,
    src: str,
    tgt: str,
    *,
    status: str = "active",
    trust: str = "human",
    context: dict[str, str] | None = None,
) -> RuntimeLexiconEntry:
    return RuntimeLexiconEntry(
        entry_id=entry_id,
        src=src,
        tgt=tgt,
        level="phrase",
        tier="manual",
        priority=1000,
        context=context,
        score=1.0,
        status=status,
        source=f"test:{entry_id}",
        trust=trust,
        updated_by="test",
        updated_at="2026-07-28T00:00:00+08:00",
    )


def _match(entry: RuntimeLexiconEntry) -> MatchTrace:
    return MatchTrace(
        entry_id=entry.entry_id,
        src=entry.src,
        tgt=entry.tgt,
        level=entry.level,
        tier=entry.tier,
        start=0,
        end=len(entry.src),
        priority=entry.priority,
        score=entry.score,
    )


class _FakeConverter:
    def __init__(self) -> None:
        self.producer = _entry(
            "lx_producer",
            "甲",
            "台",
            context={"right_regex": "^(?!仔)"},
        )
        self.shadowed = _entry("lx_shadowed", "甲", "臺", trust="seed")
        self.consumer = _entry(
            "lx_consumer",
            "台",
            "臺",
            trust="machine",
            context={"left_regex": "^$"},
        )
        self.stable = _entry("lx_stable", "乙", "乙")
        self.disabled = _entry("lx_disabled", "丙", "丁", status="disabled")
        self.entries_by_index = (
            self.stable,
            self.disabled,
            self.shadowed,
            self.consumer,
            self.producer,
        )
        self.calls: list[tuple[str, bool]] = []

    def convert(self, text: str, trace: bool = False) -> str | ConversionResult:
        self.calls.append((text, trace))
        if text == "甲":
            result = ConversionResult(output="台", matches=[_match(self.producer)])
        elif text == "台":
            result = ConversionResult(output="臺", matches=[_match(self.consumer)])
        else:
            result = ConversionResult(output=text)
        return result if trace else result.output


class RuntimeFixedPointAuditTests(unittest.TestCase):
    def test_function_api_includes_canonical_regressions_by_default(self) -> None:
        occurrence = LocatedRegressionCase(
            suite="conversation",
            script="run_conversation_regression.py",
            index=1,
            case=RegressionCase("default_coverage", "乙", "乙"),
        )
        with patch(
            "scripts.audit_runtime_fixed_points.load_all_regression_cases",
            return_value=[occurrence],
        ) as load_all:
            report = audit_runtime_fixed_points(_FakeConverter())

        load_all.assert_called_once_with(Path(__file__).resolve().parents[1] / "scripts")
        self.assertEqual(report["summary"]["regression_case_count"], 1)
        self.assertEqual(report["summary"]["regression_unique_source_count"], 1)
        self.assertEqual(report["summary"]["audited_unique_source_count"], 3)

    def test_scans_all_active_unique_sources_without_suppression(self) -> None:
        converter = _FakeConverter()

        self.assertEqual(runtime_active_unique_sources(converter), ["乙", "台", "甲"])
        report = audit_runtime_fixed_points(converter, regression_occurrences=[])

        self.assertEqual(
            converter.calls,
            [
                ("乙", True),
                ("乙", True),
                ("台", True),
                ("臺", True),
                ("甲", True),
                ("台", True),
            ],
        )
        self.assertEqual(
            report["summary"],
            {
                "runtime_active_entry_count": 4,
                "runtime_unique_source_count": 3,
                "regression_case_count": 0,
                "regression_unique_source_count": 0,
                "regression_only_unique_source_count": 0,
                "audited_unique_source_count": 3,
                "first_pass_changed_source_count": 2,
                "idempotent_source_count": 2,
                "non_idempotent_source_count": 1,
                "non_idempotent_runtime_source_count": 1,
                "non_idempotent_regression_source_count": 0,
                "non_idempotent_regression_only_source_count": 0,
                "non_idempotent_source_rate": 0.333333,
            },
        )

    def test_finding_contains_outputs_and_enriched_producer_consumer_trace(self) -> None:
        finding = audit_runtime_fixed_points(_FakeConverter(), regression_occurrences=[])["findings"][0]

        self.assertEqual(
            {key: finding[key] for key in ("source", "first", "second")},
            {"source": "甲", "first": "台", "second": "臺"},
        )
        self.assertEqual(
            [row["entry_id"] for row in finding["source_entries"]],
            ["lx_producer", "lx_shadowed"],
        )
        self.assertEqual(finding["producer"]["entry_ids"], ["lx_producer"])
        self.assertEqual(finding["consumer"]["entry_ids"], ["lx_consumer"])
        self.assertEqual(finding["producer"]["matches"][0]["trust"], "human")
        self.assertEqual(
            finding["producer"]["matches"][0]["context"],
            {"right_regex": "^(?!仔)"},
        )
        self.assertEqual(finding["consumer"]["matches"][0]["trust"], "machine")
        self.assertEqual(
            finding["consumer"]["matches"][0]["context"],
            {"left_regex": "^$"},
        )
        self.assertEqual(finding["regression_cases"], [])
        self.assertNotIn("latency_ms", finding["producer"])
        self.assertNotIn("latency_ms", finding["consumer"])

    def test_scans_regression_only_sentence_sources_and_reports_origin(self) -> None:
        occurrence = LocatedRegressionCase(
            suite="conversation",
            script="run_conversation_regression.py",
            index=7,
            case=RegressionCase(
                "sentence_fixed_point",
                "甲句",
                "台句",
                oracle_kind="compatibility_snapshot",
            ),
        )
        converter = _FakeConverter()
        original_convert = converter.convert

        def convert(text: str, trace: bool = False) -> str | ConversionResult:
            if text == "甲句":
                result = ConversionResult(output="甲", matches=[])
                converter.calls.append((text, trace))
                return result if trace else result.output
            return original_convert(text, trace=trace)

        converter.convert = convert  # type: ignore[method-assign]
        duplicate_occurrence = LocatedRegressionCase(
            suite="family",
            script="run_family_regression.py",
            index=3,
            case=occurrence.case,
        )
        report = audit_runtime_fixed_points(
            converter,
            regression_occurrences=[occurrence, duplicate_occurrence],
        )

        self.assertEqual(
            report["summary"],
            {
                "runtime_active_entry_count": 4,
                "runtime_unique_source_count": 3,
                "regression_case_count": 2,
                "regression_unique_source_count": 1,
                "regression_only_unique_source_count": 1,
                "audited_unique_source_count": 4,
                "first_pass_changed_source_count": 3,
                "idempotent_source_count": 2,
                "non_idempotent_source_count": 2,
                "non_idempotent_runtime_source_count": 1,
                "non_idempotent_regression_source_count": 1,
                "non_idempotent_regression_only_source_count": 1,
                "non_idempotent_source_rate": 0.5,
            },
        )
        finding = next(row for row in report["findings"] if row["source"] == "甲句")
        self.assertEqual((finding["first"], finding["second"]), ("甲", "台"))
        self.assertEqual(finding["source_entries"], [])
        self.assertEqual(
            finding["regression_cases"],
            [
                {
                    "suite": "conversation",
                    "script": "run_conversation_regression.py",
                    "index": 7,
                    "category": "sentence_fixed_point",
                    "expected": "台句",
                    "oracle_kind": "compatibility_snapshot",
                    "provenance": "",
                    "reviewed_by": "",
                    "reviewed_at": "",
                },
                {
                    "suite": "family",
                    "script": "run_family_regression.py",
                    "index": 3,
                    "category": "sentence_fixed_point",
                    "expected": "台句",
                    "oracle_kind": "compatibility_snapshot",
                    "provenance": "",
                    "reviewed_by": "",
                    "reviewed_at": "",
                },
            ],
        )

    def test_json_serialization_is_deterministic(self) -> None:
        first = serialize_report(audit_runtime_fixed_points(_FakeConverter(), regression_occurrences=[]))
        second = serialize_report(audit_runtime_fixed_points(_FakeConverter(), regression_occurrences=[]))

        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["schema_version"], 2)
        self.assertTrue(first.endswith("\n"))

    def test_cli_fail_flag_and_output_report(self) -> None:
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "audit.json"
            exit_code = main(
                [
                    "--json",
                    "--output",
                    str(output_path),
                    "--fail-on-non-idempotent",
                    "--runtime-only",
                ],
                converter=_FakeConverter(),
                stdout=stdout,
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout.getvalue(), output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(stdout.getvalue())["summary"]["non_idempotent_source_count"],
                1,
            )

        self.assertEqual(
            main(["--runtime-only"], converter=_FakeConverter(), stdout=io.StringIO()),
            0,
        )


if __name__ == "__main__":
    unittest.main()
