from __future__ import annotations

import io
import json
from pathlib import Path

from scripts.audit_lexicon_risks import (
    LocatedEntry,
    RuntimeObservation,
    audit_entries,
    main,
    serialize_report,
)
from taigi_converter.lexicon_policy import runtime_exclusion_reason
from taigi_converter.models import LexiconEntry


def _located(
    entry_id: str,
    src: str,
    tgt: str,
    *,
    line: int,
    trust: str = "machine",
    tier: str = "manual_hotfix",
    context=None,
    status: str = "active",
) -> LocatedEntry:
    entry = LexiconEntry(
        entry_id=entry_id,
        src=src,
        tgt=tgt,
        level="phrase",
        tier=tier,
        priority=100,
        context=context,
        score=1.0,
        status=status,
        source="review_queue",
        trust=trust,
        updated_by="itaigi_full",
        updated_at="2026-07-29T00:00:00+08:00",
    )
    return LocatedEntry(
        line=line,
        raw=entry.to_dict(),
        entry=entry,
        runtime_exclusion=runtime_exclusion_reason(entry),
    )


def test_audit_prioritizes_composed_structural_risks_deterministically() -> None:
    entries = [
        _located("lx_reverse_a", "甲方", "乙方", line=2, tier="base", trust="seed"),
        _located("lx_reverse_b", "乙方", "甲方", line=3, tier="base", trust="seed"),
        _located("lx_function", "但是", "毋過", line=4),
        _located("lx_safe", "垃圾", "糞埽", line=5, tier="base", trust="seed"),
        _located("lx_ascii", "票號", "票ticket#7", line=6),
    ]

    first = audit_entries(entries, limit=None)
    second = audit_entries(list(reversed(entries)), limit=None)

    assert serialize_report(first) == serialize_report(second)
    assert first["findings"][0]["entry_id"] == "lx_ascii"
    by_id = {finding["entry_id"]: finding for finding in first["findings"]}
    assert {signal["kind"] for signal in by_id["lx_function"]["signals"]} >= {
        "broad_context_free_function_word",
        "short_context_free_rewrite",
        "runtime_machine_override",
    }
    assert "reverse_rewrite_edge" in {signal["kind"] for signal in by_id["lx_reverse_a"]["signals"]}
    assert "lx_safe" in by_id
    assert {signal["kind"] for signal in by_id["lx_safe"]["signals"]} == {"short_context_free_rewrite"}


def test_extension_g_target_is_not_misclassified_as_non_hanji() -> None:
    entry = _located("lx_extension_g", "骯髒", "癩𰣻", line=1, tier="base", trust="seed")

    report = audit_entries([entry], limit=None)

    signals = {signal["kind"] for signal in report["findings"][0]["signals"]}
    assert "non_hanji_or_empty_target" not in signals


def test_runtime_excluded_entries_are_counted_but_not_ranked_by_default() -> None:
    excluded = _located("lx_excluded", "甲", "甲", line=1)
    included = _located("lx_included", "但是", "毋過", line=2)
    assert excluded.runtime_exclusion == "noop_manual_hotfix"

    default_report = audit_entries([excluded, included], limit=None)
    inclusive_report = audit_entries(
        [excluded, included],
        include_runtime_excluded=True,
        limit=None,
    )

    assert default_report["summary"]["selected_runtime_excluded_entry_count"] == 1
    assert {finding["entry_id"] for finding in default_report["findings"]} == {"lx_included"}
    assert inclusive_report["summary"]["candidate_entry_count"] == 2


def test_active_conflicts_consider_all_trust_levels_but_only_selected_rows_are_reported() -> None:
    machine = _located("lx_machine", "何時", "當時", line=1)
    human = _located(
        "lx_human",
        "何時",
        "啥物時陣",
        line=2,
        trust="human",
        tier="manual",
    )

    report = audit_entries([machine, human], trusts=["machine"], limit=None)

    assert [finding["entry_id"] for finding in report["findings"]] == ["lx_machine"]
    kinds = {signal["kind"] for signal in report["findings"][0]["signals"]}
    assert "competing_active_targets" in kinds


def test_runtime_probe_filters_shadowed_rows_and_records_exact_winner() -> None:
    shadowed = _located("lx_shadowed", "何時", "當時", line=1)
    winner = _located("lx_winner", "但是", "毋過", line=2)
    observations = {
        "何時": RuntimeObservation(
            output="啥物時陣",
            winner_entry_ids=("lx_human",),
            match_entry_ids=("lx_human",),
        ),
        "但是": RuntimeObservation(
            output="毋過",
            winner_entry_ids=("lx_winner",),
            match_entry_ids=("lx_winner",),
        ),
    }

    report = audit_entries(
        [shadowed, winner],
        runtime_observations=observations,
        limit=None,
    )
    assert [finding["entry_id"] for finding in report["findings"]] == ["lx_winner"]
    assert report["findings"][0]["runtime_exact_winner"] is True
    assert report["findings"][0]["runtime_output"] == "毋過"
    assert report["summary"]["runtime_exact_winner_entry_count"] == 1
    assert report["summary"]["runtime_shadowed_entry_count"] == 1

    inclusive = audit_entries(
        [shadowed, winner],
        runtime_observations=observations,
        include_runtime_shadowed=True,
        limit=None,
    )
    by_id = {finding["entry_id"]: finding for finding in inclusive["findings"]}
    assert by_id["lx_shadowed"]["runtime_exact_winner"] is False


def test_cli_writes_deterministic_json_with_limit(tmp_path: Path) -> None:
    data_path = tmp_path / "lexicon.jsonl"
    output_path = tmp_path / "report.json"
    rows = [
        _located("lx_1", "但是", "毋過", line=1).raw,
        _located("lx_2", "票號", "票ABC-7", line=2).raw,
    ]
    data_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--data",
                str(data_path),
                "--json",
                "--limit",
                "1",
                "--output",
                str(output_path),
            ],
            stdout=io.StringIO(),
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["risk_finding_count"] == 2
    assert report["summary"]["reported_finding_count"] == 1
    assert report["findings"][0]["entry_id"] == "lx_2"
