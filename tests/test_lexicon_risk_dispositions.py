from __future__ import annotations

from scripts.audit_lexicon_risk_dispositions import finding_fingerprint, source_snapshot, validate_registry
from scripts.audit_lexicon_risks import LocatedEntry
from taigi_converter.models import LexiconEntry


def _finding(entry_id: str = "lx_risk") -> dict[str, object]:
    return {
        "entry_id": entry_id,
        "src": "故意",
        "tgt": "刁工",
        "level": "phrase",
        "tier": "manual_hotfix",
        "priority": 1000,
        "trust": "machine",
        "source": "review_queue",
        "context": None,
        "signals": [
            {"kind": "short_context_free_rewrite"},
            {"kind": "runtime_machine_override"},
        ],
    }


def _row(finding: dict[str, object], *, disposition: str = "accepted_legacy") -> dict[str, object]:
    return {
        "entry_id": finding["entry_id"],
        "finding_fingerprint": finding_fingerprint(finding),
        "signals": sorted(signal["kind"] for signal in finding["signals"]),  # type: ignore[index]
        "disposition": disposition,
        "reason_code": "test_reason",
        "evidence": {
            "regression_trace_count": 0,
            "regression_source_count": 0,
            "oracle_counts": {},
        },
        "reviewed_by": "test",
        "reviewed_at": "2026-07-29",
        "source_snapshot": source_snapshot(finding),
        "resolution_entry_ids": [],
    }


def _located(*, entry_id: str, status: str, context: dict[str, str] | None = None) -> LocatedEntry:
    raw = {
        "entry_id": entry_id,
        "src": "故意",
        "tgt": "刁工",
        "level": "phrase",
        "tier": "manual" if context else "manual_hotfix",
        "priority": 1000,
        "context": context,
        "score": 1.0,
        "status": status,
        "source": "test",
        "trust": "ai_reviewed" if context else "machine",
        "updated_by": "test",
        "updated_at": "2026-07-29",
    }
    return LocatedEntry(line=1, raw=raw, entry=LexiconEntry.from_dict(raw), runtime_exclusion=None)


def test_registry_accepts_exact_classified_finding() -> None:
    finding = _finding()
    report = validate_registry([_row(finding)], [finding])
    assert report["summary"]["unclassified_count"] == 0
    assert report["summary"]["error_count"] == 0


def test_registry_fails_closed_on_unclassified_or_fingerprint_drift() -> None:
    finding = _finding()
    unclassified = validate_registry([], [finding])
    assert unclassified["summary"]["unclassified_count"] == 1
    assert unclassified["errors"]

    row = _row(finding)
    row["finding_fingerprint"] = "0" * 64
    drifted = validate_registry([row], [finding])
    assert drifted["summary"]["stale_fingerprint_count"] == 1
    assert drifted["errors"]


def test_context_gate_requires_inactive_original_and_active_resolution() -> None:
    finding = _finding()
    row = _row(finding, disposition="context_gate")
    row["resolution_entry_ids"] = ["lx_resolution"]
    original = _located(entry_id="lx_risk", status="disabled")
    resolution = _located(
        entry_id="lx_resolution",
        status="active",
        context={"right_regex": "^(?!殺人罪)"},
    )
    report = validate_registry([row], [], source_entries=[original, resolution])
    assert report["summary"]["resolved_count"] == 1
    assert report["summary"]["error_count"] == 0


def test_context_gate_rejects_unrelated_resolution_source() -> None:
    finding = _finding()
    row = _row(finding, disposition="context_gate")
    row["resolution_entry_ids"] = ["lx_resolution"]
    original = _located(entry_id="lx_risk", status="disabled")
    resolution = _located(
        entry_id="lx_resolution",
        status="active",
        context={"right_regex": "^(?!殺人罪)"},
    )
    resolution_raw = dict(resolution.raw)
    resolution_raw["src"] = "無關詞"
    unrelated = LocatedEntry(
        line=resolution.line,
        raw=resolution_raw,
        entry=LexiconEntry.from_dict(resolution_raw),
        runtime_exclusion=None,
    )

    report = validate_registry([row], [], source_entries=[original, unrelated])

    assert any("source 未涵蓋原 source" in error for error in report["errors"])


def test_exclude_validates_optional_resolution_ids() -> None:
    finding = _finding()
    row = _row(finding, disposition="exclude")
    row["resolution_entry_ids"] = ["lx_missing"]
    original = _located(entry_id="lx_risk", status="disabled")

    report = validate_registry([row], [], source_entries=[original])

    assert any("resolution entry 不存在" in error for error in report["errors"])


def test_accepted_legacy_cannot_silently_disappear() -> None:
    finding = _finding()
    report = validate_registry([_row(finding)], [])
    assert any("必須改為 resolved disposition" in error for error in report["errors"])
