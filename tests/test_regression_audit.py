from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_regression_expectations import LocatedCase, audit_cases, main
from scripts.regression_runner import (
    RegressionCase,
    compatibility_snapshot_case,
    load_all_regression_cases,
    validate_regression_script_registry,
)


class _FakeConverter:
    def convert(self, text: str) -> str:
        return {"會議": "表決"}.get(text, text)


class RegressionAuditTests(unittest.TestCase):
    def test_canonical_loader_preserves_normalized_suite_and_script_locations(self) -> None:
        first = RegressionCase("first", "甲句", "台句")
        second = RegressionCase("second", "乙句", "臺句")
        with tempfile.TemporaryDirectory() as temp:
            scripts_dir = Path(temp)
            for name in (
                "run_alpha_regression.py",
                "run_beta_regression.py",
                "run_package_parity_regression.py",
            ):
                (scripts_dir / name).touch()
            with (
                patch(
                    "scripts.regression_runner.REGRESSION_SCRIPT_NAMES",
                    ("run_alpha_regression.py", "run_beta_regression.py"),
                ),
                patch(
                    "scripts.regression_runner.load_suite_cases",
                    side_effect=[[first], [second]],
                ) as load_cases,
            ):
                located = load_all_regression_cases(scripts_dir)

        self.assertEqual(
            [(row.suite, row.script, row.index, row.case.source) for row in located],
            [
                ("alpha", "run_alpha_regression.py", 1, "甲句"),
                ("beta", "run_beta_regression.py", 1, "乙句"),
            ],
        )
        self.assertEqual(
            [call.args[0].name for call in load_cases.call_args_list],
            ["run_alpha_regression.py", "run_beta_regression.py"],
        )

    def test_registry_validation_rejects_unregistered_and_missing_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            scripts_dir = Path(temp)
            (scripts_dir / "run_alpha_regression.py").touch()
            (scripts_dir / "run_new_regression.py").touch()
            with patch(
                "scripts.regression_runner.REGRESSION_SCRIPT_NAMES",
                ("run_alpha_regression.py", "run_missing_regression.py"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "run_new_regression.py.*run_missing_regression.py",
                ):
                    validate_regression_script_registry(scripts_dir)

    def test_audit_distinguishes_review_signals_from_conflicts(self) -> None:
        located = [
            LocatedCase(
                suite="sample",
                script="run_sample_regression.py",
                index=1,
                case=RegressionCase("chat", "你有沒有空？", "你有沒有空？"),
            ),
            LocatedCase(
                suite="sample",
                script="run_sample_regression.py",
                index=2,
                case=RegressionCase("work", "會議", "會議"),
            ),
            LocatedCase(
                suite="other",
                script="run_other_regression.py",
                index=1,
                case=RegressionCase("work", "會議", "表決"),
            ),
        ]

        report = audit_cases(located, converter=_FakeConverter())
        summary = report["summary"]

        self.assertEqual(summary["case_count"], 3)
        self.assertEqual(summary["identity_expected_count"], 2)
        self.assertEqual(summary["mandarin_surface_marker_case_count"], 1)
        self.assertEqual(summary["non_idempotent_expected_count"], 1)
        self.assertEqual(summary["conflicting_expected_location_count"], 2)
        self.assertEqual(summary["human_verified_translation_count"], 0)
        self.assertEqual(summary["ai_semantic_review_count"], 0)

    def test_classified_snapshot_keeps_observations_without_review_findings(self) -> None:
        located = [
            LocatedCase(
                suite="sample",
                script="run_sample_regression.py",
                index=1,
                case=compatibility_snapshot_case("chat", "你有沒有空？", "你有沒有空？"),
            )
        ]

        report = audit_cases(located)
        summary = report["summary"]
        self.assertEqual(summary["identity_expected_count"], 1)
        self.assertEqual(summary["mandarin_surface_marker_case_count"], 1)
        self.assertEqual(summary["unreviewed_identity_expected_count"], 0)
        self.assertEqual(summary["unreviewed_mandarin_surface_marker_case_count"], 0)
        self.assertEqual(report["findings"], [])

    def test_compatibility_snapshot_factory_does_not_claim_human_review(self) -> None:
        case = compatibility_snapshot_case("chat", "你好", "你好")
        self.assertEqual(case.oracle_kind, "compatibility_snapshot")
        self.assertEqual(case.provenance, "")
        self.assertEqual(case.reviewed_by, "")
        self.assertEqual(case.reviewed_at, "")

    def test_verified_translation_requires_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "provenance"):
            RegressionCase("chat", "你好", "你好", oracle_kind="verified_translation")

        case = RegressionCase(
            "chat",
            "你好",
            "你好",
            oracle_kind="verified_translation",
            provenance="教育部辭典與人工語境審查",
            reviewed_by="reviewer",
            reviewed_at="2026-07-28",
        )
        self.assertEqual(case.oracle_kind, "verified_translation")

    def test_ai_semantic_review_is_separate_and_requires_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "provenance"):
            RegressionCase("chat", "你好", "你好", oracle_kind="ai_semantic_review")

        case = RegressionCase(
            "chat",
            "請放涼",
            "請囥冷",
            oracle_kind="ai_semantic_review",
            provenance="Codex AI 語義審查；教育部臺灣台語常用詞辭典例句查證",
            reviewed_by="codex_ai_semantic_audit",
            reviewed_at="2026-07-28",
        )
        summary = audit_cases([LocatedCase("sample", "sample.py", 1, case)])["summary"]
        self.assertEqual(summary["human_verified_translation_count"], 0)
        self.assertEqual(summary["ai_semantic_review_count"], 1)

    def test_duplicate_metadata_must_be_complete(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate_group"):
            RegressionCase("chat", "你好", "你好", duplicate_group="shared")

    def test_explained_duplicate_is_counted_but_not_reported(self) -> None:
        case = RegressionCase(
            "chat",
            "你好",
            "你好",
            duplicate_group="shared-greeting",
            duplicate_reason="同一句刻意覆蓋兩個產品入口的共用行為",
        )
        located = [
            LocatedCase("a", "a.py", 1, case),
            LocatedCase("b", "b.py", 1, case),
        ]

        summary = audit_cases(located)["summary"]
        self.assertEqual(summary["exact_duplicate_location_count"], 2)
        self.assertEqual(summary["explained_duplicate_location_count"], 2)
        self.assertEqual(summary["duplicate_case_location_count"], 0)

    def test_duplicate_case_is_reported_without_becoming_a_conflict(self) -> None:
        case = RegressionCase("chat", "你好", "你好")
        located = [
            LocatedCase("a", "a.py", 1, case),
            LocatedCase("a", "a.py", 2, case),
        ]

        summary = audit_cases(located)["summary"]

        self.assertEqual(summary["duplicate_case_location_count"], 2)
        self.assertEqual(summary["conflicting_expected_location_count"], 0)

    def test_cli_fail_flags_preserve_legacy_conflict_behavior(self) -> None:
        clean_cases = [
            LocatedCase(
                suite="sample",
                script="run_sample_regression.py",
                index=1,
                case=compatibility_snapshot_case("chat", "你好", "你好"),
            )
        ]
        finding_cases = [
            LocatedCase(
                suite="sample",
                script="run_sample_regression.py",
                index=1,
                case=RegressionCase("chat", "你好", "你好"),
            )
        ]
        conflict_cases = [
            LocatedCase(
                suite="sample",
                script="run_sample_regression.py",
                index=1,
                case=RegressionCase("chat", "會議", "會議"),
            ),
            LocatedCase(
                suite="other",
                script="run_other_regression.py",
                index=1,
                case=RegressionCase("chat", "會議", "表決"),
            ),
        ]

        self.assertEqual(
            main(
                ["--json", "--skip-idempotency", "--fail-on-findings"],
                located_cases=clean_cases,
                stdout=io.StringIO(),
            ),
            0,
        )
        self.assertEqual(
            main(
                ["--json", "--skip-idempotency"],
                located_cases=finding_cases,
                stdout=io.StringIO(),
            ),
            0,
        )
        finding_stdout = io.StringIO()
        self.assertEqual(
            main(
                ["--json", "--skip-idempotency", "--fail-on-findings"],
                located_cases=finding_cases,
                stdout=finding_stdout,
            ),
            1,
        )
        self.assertEqual(
            json.loads(finding_stdout.getvalue())["summary"]["unreviewed_identity_expected_count"],
            1,
        )
        self.assertEqual(
            main(
                ["--json", "--skip-idempotency", "--fail-on-conflict"],
                located_cases=conflict_cases,
                stdout=io.StringIO(),
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
