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
    build_semantic_summary,
    deterministic_json,
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
        with pytest.raises(ValueError, match="duplicate source"):
            load_semantic_cases(path)


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
    }


def test_leakage_audit_honors_documented_override_but_not_regression_overlap() -> None:
    case = make_case(
        allow_sentence_override=True,
        sentence_override_reason="此案例專門驗證已核准的完整句 override",
    )
    report = audit_semantic_leakage(
        [case],
        regression_sources=set(),
        active_exact_entries={case.source: [{"entry_id": "lx_exact", "level": "sentence", "line": 1}]},
    )
    assert report["summary"]["clean"] is True


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


def test_summary_json_is_deterministic_and_mismatch_limit_is_applied() -> None:
    cases = [make_case()]
    results = run_semantic_cases(cases, converter=FakeConverter({}))
    summary = build_semantic_summary(cases, results, mismatch_limit=0)

    assert summary["mismatches"] == []
    assert summary["mismatches_truncated"] == 1
    assert deterministic_json(summary) == deterministic_json(summary)
