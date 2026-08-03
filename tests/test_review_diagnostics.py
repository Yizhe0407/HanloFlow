from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from taigi_converter import TaigiConverter
from taigi_converter.converter import _linear_identity_ratio
from taigi_converter.review_queue import append_review_item, export_pending_reviews


class RuleTraceProbeConverter(TaigiConverter):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.collect_trace_flags: list[bool] = []

    def _apply_rules(
        self,
        text: str,
        *,
        collect_trace: bool,
        skip_passes: set[str] | None = None,
    ):
        self.collect_trace_flags.append(collect_trace)
        return super()._apply_rules(text, collect_trace=collect_trace, skip_passes=skip_passes)


def test_normal_fast_path_does_not_collect_rule_trace() -> None:
    converter = RuleTraceProbeConverter()

    assert converter.convert("食飽了沒") == "食飽未"
    assert converter.collect_trace_flags
    assert not any(converter.collect_trace_flags)


def test_enqueue_review_opt_in_collects_rule_trace_without_changing_return_type() -> None:
    with TemporaryDirectory() as temp:
        converter = RuleTraceProbeConverter(review_data_dir=Path(temp))
        output = converter.convert("食飽了沒", profile={"enqueue_review": True})

    assert output == "食飽未"
    assert isinstance(output, str)
    assert converter.collect_trace_flags
    assert all(converter.collect_trace_flags)


def test_unknown_input_enqueues_structured_confidence_evidence() -> None:
    with TemporaryDirectory() as temp:
        converter = TaigiConverter(review_data_dir=Path(temp))
        with patch("taigi_converter.converter.append_review_item") as append:
            output = converter.convert(
                "完全未知內容",
                profile={"enqueue_review": True, "owner": "diagnostic-test"},
            )

    assert output == "完全未知內容"
    append.assert_called_once()
    payload = append.call_args.args[1]
    evidence = payload["evidence"]
    assert payload["priority"] == evidence["review_priority"]
    assert payload["owner"] == "diagnostic-test"
    assert evidence["low_confidence_reasons"] == [
        "no_transform_evidence",
        "sparse_conversion_coverage",
    ]
    assert evidence["matched_span_ratio"] == 0.0
    assert evidence["identity_ratio"] == 1.0
    assert evidence["confidence_score"] < 0.5
    assert evidence["residual_terms"] == []
    assert evidence["protected_terms"] == []
    assert evidence["matches"] == []
    assert evidence["rules_applied"] == []


def test_sparse_partial_conversion_is_enqueued_even_without_residual_warning() -> None:
    with TemporaryDirectory() as temp:
        converter = TaigiConverter(review_data_dir=Path(temp))
        with patch("taigi_converter.converter.append_review_item") as append:
            converter.convert("今天完全未知內容", profile={"enqueue_review": True})

    append.assert_called_once()
    evidence = append.call_args.args[1]["evidence"]
    assert "sparse_conversion_coverage" in evidence["low_confidence_reasons"]
    assert evidence["warnings"] == []
    assert evidence["match_count"] == 1
    assert evidence["matches"][0]["src"] == "今天"
    assert 0.0 < evidence["matched_span_ratio"] < 0.35


def test_residual_review_contains_matches_rules_and_protected_terms() -> None:
    source = "民眾說：「就列個東西告示牌，清楚告知使用規範。」"
    with TemporaryDirectory() as temp:
        converter = TaigiConverter(review_data_dir=Path(temp))
        with patch("taigi_converter.converter.append_review_item") as append:
            converter.convert(source, profile={"enqueue_review": True})

    append.assert_called_once()
    evidence = append.call_args.args[1]["evidence"]
    assert evidence["residual_terms"] == ["東西"]
    assert evidence["protected_terms"] == ["使用規範"]
    assert evidence["match_entry_ids"] == ["lx_531000000012"]
    assert evidence["matches"][0]["entry_id"] == "lx_531000000012"
    assert evidence["rule_count"] == 1
    assert evidence["rule_ids"] == ["rl_8e67ee3e3752"]
    assert evidence["rules_applied"][0]["hit_count"] == 1
    assert evidence["rules_applied"][0]["matched_chars"] > 0
    assert evidence["rule_span_ratio"] > 0.0
    assert evidence["evidence_span_ratio"] >= evidence["matched_span_ratio"]


def test_fully_protected_or_rule_only_input_is_not_low_confidence() -> None:
    with TemporaryDirectory() as temp:
        converter = TaigiConverter(review_data_dir=Path(temp))
        with patch("taigi_converter.converter.append_review_item") as append:
            assert converter.convert("周到", profile={"enqueue_review": True}) == "周到"
            assert converter.convert("食飽了沒", profile={"enqueue_review": True}) == "食飽未"
            assert (
                converter.convert("我的卡片到期了。", profile={"enqueue_review": True})
                == "我的卡片到期矣。"
            )

    append.assert_not_called()


def test_same_target_duplicate_entries_are_not_reported_as_ambiguous() -> None:
    with TemporaryDirectory() as temp:
        converter = TaigiConverter(review_data_dir=Path(temp))
        with patch("taigi_converter.converter.append_review_item") as append:
            assert converter.convert("今天", profile={"enqueue_review": True}) == "今仔日"

    append.assert_not_called()


def test_equal_span_alternatives_are_reported_as_ambiguous() -> None:
    with TemporaryDirectory() as temp:
        converter = TaigiConverter(review_data_dir=Path(temp))
        with patch("taigi_converter.converter.append_review_item") as append:
            assert converter.convert("左轉", profile={"enqueue_review": True}) == "倒斡"

    append.assert_called_once()
    evidence = append.call_args.args[1]["evidence"]
    assert evidence["low_confidence_reasons"] == ["ambiguous_candidates"]
    assert evidence["ambiguous_candidate_count"] >= 1
    assert evidence["matched_span_ratio"] == 1.0


def test_pending_review_export_orders_priority_then_confidence() -> None:
    with TemporaryDirectory() as temp:
        data_dir = Path(temp)
        output_path = data_dir / "pending.jsonl"
        append_review_item(
            data_dir,
            {"kind": "low", "priority": 10, "evidence": {"confidence_score": 0.1}},
        )
        append_review_item(
            data_dir,
            {"kind": "high-confident", "priority": 80, "evidence": {"confidence_score": 0.8}},
        )
        append_review_item(
            data_dir,
            {"kind": "high-uncertain", "priority": 80, "evidence": {"confidence_score": 0.2}},
        )

        assert export_pending_reviews(data_dir, output_path, limit=2) == 2
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert [row["kind"] for row in rows] == ["high-uncertain", "high-confident"]


def test_identity_ratio_handles_long_repeated_input_without_alignment() -> None:
    source = "甲乙" * 50_000
    assert _linear_identity_ratio(source, source) == 1.0
    assert _linear_identity_ratio(source, source[:-1] + "丙") > 0.99
