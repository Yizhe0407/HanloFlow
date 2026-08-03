from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.audit_semantic_eval_leakage import audit_semantic_leakage
from scripts.audit_semantic_eval_leakage import main as audit_main
from scripts.run_semantic_evaluation import main as runner_main
from scripts.semantic_evaluation import (
    SemanticEvaluationCase,
    SemanticEvaluationResult,
    _latency_summary,
    build_semantic_summary,
    load_semantic_cases,
    run_semantic_cases,
)


class FakeConverter:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs

    def convert(self, source: str) -> str:
        return self.outputs.get(source, source)


def case_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_id": "sem_conversation_0001",
        "source": "我還沒找到他。",
        "expected": "我猶未揣著伊。",
        "category": "conversation",
        "failure_type": "wrong_sense",
        "focus_terms": ["還", "找到", "他"],
        "oracle_kind": "ai_semantic_review",
        "provenance": "Codex AI semantic corpus review；非人工翻譯認證",
        "reviewed_by": "openai_codex",
        "reviewed_at": "2026-08-03",
        "split": "holdout",
        "allow_sentence_override": False,
        "sentence_override_reason": "",
        "sentence_override_entry_ids": [],
    }
    payload.update(overrides)
    return payload


def write_cases(path: Path, *payloads: dict[str, object]) -> None:
    path.write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def make_case(**overrides: object) -> SemanticEvaluationCase:
    return SemanticEvaluationCase.from_dict(case_payload(**overrides))


def test_load_valid_semantic_case() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "cases.jsonl"
        write_cases(path, case_payload())
        cases = load_semantic_cases(path)

    assert len(cases) == 1
    assert cases[0].focus_terms == ("還", "找到", "他")
    assert cases[0].oracle_kind == "ai_semantic_review"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "test", "未知 semantic split"),
        ("failure_type", "lexical", "未知 failure_type"),
        ("oracle_kind", "compatibility_snapshot", "未知 semantic oracle_kind"),
        ("reviewed_at", "2026/08/03", "YYYY-MM-DD"),
    ],
)
def test_schema_rejects_invalid_enums_and_date(field: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_case(**{field: value})


def test_schema_requires_truthful_review_metadata() -> None:
    with pytest.raises(ValueError, match="provenance"):
        make_case(provenance="")


def test_schema_requires_focus_term_in_source() -> None:
    with pytest.raises(ValueError, match="不在 source"):
        make_case(focus_terms=["不存在"])


def test_sentence_override_exception_requires_reason() -> None:
    with pytest.raises(ValueError, match="必須同時設定"):
        make_case(allow_sentence_override=True)
    with pytest.raises(ValueError, match="必須同時設定"):
        make_case(sentence_override_reason="人工核准")


def test_loader_rejects_duplicate_case_id() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "cases.jsonl"
        write_cases(
            path,
            case_payload(),
            case_payload(source="另一句。", expected="另外一句。", focus_terms=["另一句"]),
        )
        with pytest.raises(ValueError, match="duplicate case_id"):
            load_semantic_cases(path)


def test_loader_rejects_source_reuse_across_splits() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "cases.jsonl"
        write_cases(
            path,
            case_payload(),
            case_payload(case_id="sem_conversation_0002", split="train"),
        )
        with pytest.raises(ValueError, match="duplicate raw source"):
            load_semantic_cases(path)


def test_loader_rejects_runtime_canonical_source_reuse() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "cases.jsonl"
        write_cases(
            path,
            case_payload(
                source="臺北車站。",
                expected="臺北車站。",
                focus_terms=["臺北車站"],
            ),
            case_payload(
                case_id="sem_conversation_0002",
                source="台北車站。 ",
                expected="台北車站。",
                focus_terms=["台北車站"],
                split="train",
            ),
        )
        with pytest.raises(ValueError, match="duplicate canonical source") as exc_info:
            load_semantic_cases(path)

    assert "raw source='台北車站。 '" in str(exc_info.value)
    assert "canonical source='台北車站。'" in str(exc_info.value)


def test_leakage_audit_reports_regression_and_exact_runtime_overlap() -> None:
    case = make_case()
    report = audit_semantic_leakage(
        [case],
        regression_sources={case.source},
        active_exact_entries={case.source: [{"entry_id": "lx_exact", "level": "sentence", "line": 1}]},
    )

    assert report["summary"] == {
        "case_count": 1,
        "finding_count": 2,
        "clean": False,
        "counts_by_kind": {"exact_runtime_entry_overlap": 1, "regression_source_overlap": 1},
        "counts_by_match_type": {"raw": 2},
    }
    assert {finding["canonical_source"] for finding in report["findings"]} == {case.source}
    assert all(finding["matched_sources"] == [case.source] for finding in report["findings"])


def test_leakage_audit_reports_canonical_overlap_with_raw_sources() -> None:
    case = make_case(
        source="臺北車站。 ",
        expected="臺北車站。",
        focus_terms=["臺北車站"],
    )
    report = audit_semantic_leakage(
        [case],
        regression_sources={"台北車站。"},
        active_exact_entries={
            "台北車站。": [{"entry_id": "lx_exact", "level": "sentence", "line": 1}]
        },
    )

    assert report["summary"]["counts_by_match_type"] == {"canonical": 2}
    for finding in report["findings"]:
        assert finding["source"] == "臺北車站。 "
        assert finding["canonical_source"] == "台北車站。"
        assert finding["matched_sources"] == ["台北車站。"]


def test_leakage_audit_honors_documented_sentence_override() -> None:
    case = make_case(
        allow_sentence_override=True,
        sentence_override_reason="此案例專門驗證已核准的完整句 override",
        sentence_override_entry_ids=["lx_exact"],
    )
    report = audit_semantic_leakage(
        [case],
        regression_sources=set(),
        active_exact_entries={case.source: [{"entry_id": "lx_exact", "level": "sentence", "line": 1}]},
    )
    assert report["summary"]["clean"] is True


def test_sentence_override_never_exempts_phrase_overlap() -> None:
    case = make_case(
        allow_sentence_override=True,
        sentence_override_reason="只核准指定的完整句 override，不核准 phrase",
        sentence_override_entry_ids=["lx_sentence"],
    )
    report = audit_semantic_leakage(
        [case],
        regression_sources=set(),
        active_exact_entries={
            case.source: [
                {"entry_id": "lx_sentence", "level": "sentence", "line": 1},
                {"entry_id": "lx_phrase", "level": "phrase", "line": 2},
            ]
        },
    )

    assert report["summary"]["counts_by_kind"] == {"exact_runtime_entry_overlap": 1}
    finding = report["findings"][0]
    assert [entry["entry_id"] for entry in finding["entries"]] == ["lx_phrase"]
    assert [entry["entry_id"] for entry in finding["overridden_sentence_entries"]] == [
        "lx_sentence"
    ]
    assert finding["sentence_override_reason"] == case.sentence_override_reason


def test_sentence_override_only_exempts_approved_entry_ids() -> None:
    case = make_case(
        allow_sentence_override=True,
        sentence_override_reason="只核准 lx_approved",
        sentence_override_entry_ids=["lx_approved"],
    )
    report = audit_semantic_leakage(
        [case],
        regression_sources=set(),
        active_exact_entries={
            case.source: [
                {"entry_id": "lx_approved", "level": "sentence", "line": 1},
                {"entry_id": "lx_unapproved", "level": "sentence", "line": 2},
            ]
        },
    )

    finding = report["findings"][0]
    assert [entry["entry_id"] for entry in finding["entries"]] == ["lx_unapproved"]
    assert [entry["entry_id"] for entry in finding["overridden_sentence_entries"]] == [
        "lx_approved"
    ]


def test_audit_cli_fail_flag_and_deterministic_json() -> None:
    case = make_case()
    first = io.StringIO()
    second = io.StringIO()
    kwargs = {
        "cases": [case],
        "regression_sources": {case.source},
        "active_exact_entries": {},
    }
    assert audit_main(["--fail-on-findings"], stdout=first, **kwargs) == 1
    assert audit_main([], stdout=second, **kwargs) == 0
    assert first.getvalue() == second.getvalue()
    assert json.loads(first.getvalue())["summary"]["finding_count"] == 1


def test_runner_summary_and_baseline_exit_contract() -> None:
    cases = [
        make_case(),
        make_case(
            case_id="sem_news_0001",
            source="市府公布新的交通措施。",
            expected="市府公布新的交通措施。",
            category="news",
            failure_type="acceptable_identity",
            focus_terms=["市府", "交通措施"],
            split="development",
        ),
    ]
    converter = FakeConverter({cases[0].source: cases[0].expected})
    stdout = io.StringIO()

    assert runner_main([], stdout=stdout, cases=cases, converter=converter) == 0
    summary = json.loads(stdout.getvalue())
    assert summary["case_count"] == 2
    assert summary["passed"] == 2
    assert summary["counts_by_split"] == {"development": 1, "holdout": 1}


def test_runner_fail_on_mismatch_is_opt_in_and_split_filter_works() -> None:
    cases = [
        make_case(),
        make_case(
            case_id="sem_news_0001",
            source="市府公布新的交通措施。",
            expected="市府公布新的交通措施。",
            category="news",
            failure_type="acceptable_identity",
            focus_terms=["市府", "交通措施"],
            split="development",
        ),
    ]
    converter = FakeConverter({})
    baseline_stdout = io.StringIO()
    enforced_stdout = io.StringIO()

    assert runner_main(["--split", "holdout"], stdout=baseline_stdout, cases=cases, converter=converter) == 0
    assert (
        runner_main(
            ["--split", "holdout", "--fail-on-mismatch"],
            stdout=enforced_stdout,
            cases=cases,
            converter=converter,
        )
        == 1
    )
    summary = json.loads(baseline_stdout.getvalue())
    assert summary["case_count"] == 1
    assert summary["failed"] == 1


def test_runner_json_is_deterministic_across_two_executions() -> None:
    cases = [make_case()]
    first = io.StringIO()
    second = io.StringIO()

    assert runner_main([], stdout=first, cases=cases, converter=FakeConverter({})) == 0
    assert runner_main([], stdout=second, cases=cases, converter=FakeConverter({})) == 0

    assert first.getvalue() == second.getvalue()
    summary = json.loads(first.getvalue())
    assert "latency" not in summary
    assert "latency_ms" not in summary["mismatches"][0]


def test_runner_latency_is_opt_in_and_mismatch_limit_is_applied() -> None:
    cases = [make_case()]
    diagnostic_stdout = io.StringIO()
    limited_stdout = io.StringIO()

    assert (
        runner_main(
            ["--include-latency"],
            stdout=diagnostic_stdout,
            cases=cases,
            converter=FakeConverter({}),
        )
        == 0
    )
    diagnostic_summary = json.loads(diagnostic_stdout.getvalue())
    assert set(diagnostic_summary["latency"]) == {"mean_ms", "p95_ms", "max_ms"}
    assert "latency_ms" in diagnostic_summary["mismatches"][0]

    assert (
        runner_main(
            ["--include-latency", "--mismatch-limit", "0"],
            stdout=limited_stdout,
            cases=cases,
            converter=FakeConverter({}),
        )
        == 0
    )
    limited_summary = json.loads(limited_stdout.getvalue())
    assert limited_summary["mismatches"] == []
    assert limited_summary["mismatches_truncated"] == 1



def test_latency_summary_uses_nearest_rank_p95() -> None:
    case = make_case()
    results = [
        SemanticEvaluationResult(
            case=case,
            output=case.expected,
            passed=True,
            latency_ms=float(index),
        )
        for index in range(1, 22)
    ]

    assert _latency_summary(results)["p95_ms"] == 20.0

def test_build_semantic_summary_rejects_result_identity_or_order_mismatch() -> None:
    cases = [
        make_case(),
        make_case(
            case_id="sem_news_0001",
            source="市府公布新的交通措施。",
            expected="市府公布新的交通措施。",
            category="news",
            failure_type="acceptable_identity",
            focus_terms=["市府", "交通措施"],
            split="development",
        ),
    ]
    results = run_semantic_cases(cases, converter=FakeConverter({}))

    with pytest.raises(ValueError, match="identity/order"):
        build_semantic_summary(list(reversed(cases)), results)


def test_repository_semantic_corpus_distribution_and_zero_overlap() -> None:
    from collections import Counter

    from scripts.audit_semantic_eval_leakage import load_active_exact_entries
    from scripts.regression_runner import load_all_regression_cases

    repo_root = Path(__file__).resolve().parents[1]
    cases = load_semantic_cases(repo_root / "data" / "semantic_eval_cases.jsonl")
    report = audit_semantic_leakage(
        cases,
        regression_sources={
            located.case.source for located in load_all_regression_cases(repo_root / "scripts")
        },
        active_exact_entries=load_active_exact_entries(repo_root / "data" / "lexicon_entries.jsonl"),
    )

    assert len(cases) == 300
    assert Counter(case.category for case in cases) == {
        "conversation": 50,
        "news": 50,
        "transport_travel": 50,
        "medical_public_service": 50,
        "polysemy_adversarial": 50,
        "proper_nouns_technical": 50,
    }
    assert Counter(case.split for case in cases) == {
        "train": 150,
        "development": 90,
        "holdout": 60,
    }
    assert {case.oracle_kind for case in cases} == {"ai_semantic_review"}
    assert report["summary"]["clean"] is True
