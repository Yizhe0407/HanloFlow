from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from taigi_converter.lexicon_policy import (
    TRUST_MACHINE,
    TRUST_SEED,
    VALID_TRUSTS,
    runtime_exclusion_reason,
)
from taigi_converter.models import LexiconEntry
from taigi_converter.unicode_policy import contains_han_ideograph

REPORT_SCHEMA_VERSION = 2
DEFAULT_TRUSTS = (TRUST_MACHINE, TRUST_SEED)

_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./#:+-]*")

# A context-free rewrite of these short, high-frequency forms can alter many
# unrelated sentences. This is intentionally narrow: it is a ranking signal,
# not a declaration that the target is linguistically wrong.
BROAD_FUNCTION_WORDS = frozenset(
    {
        "但",
        "但是",
        "不但",
        "不然",
        "以及",
        "而且",
        "如果",
        "所以",
        "因為",
        "只是",
        "還",
        "才",
        "就",
        "讓",
        "把",
        "被",
        "我們",
        "你們",
        "他們",
        "您",
        "哪裡",
        "什麼",
        "怎麼",
        "何時",
    }
)

# Mandarin-looking surface forms are weak evidence only. They help prioritize
# low-trust outputs that may be untranslated or semantically drifted.
MANDARIN_TARGET_MARKERS = (
    "哪裡",
    "什麼",
    "沒有",
    "可以",
    "需要",
    "不要",
    "我們",
    "你們",
    "他們",
    "如果",
    "所以",
    "時候",
)

SIGNAL_WEIGHTS: Mapping[str, int] = {
    "non_hanji_or_empty_target": 80,
    "ascii_in_target": 55,
    "reverse_rewrite_edge": 45,
    "broad_context_free_function_word": 40,
    "competing_active_targets": 35,
    "severe_contraction": 30,
    "short_context_free_rewrite": 25,
    "severe_expansion": 20,
    "runtime_machine_override": 18,
    "target_is_low_trust_source": 15,
    "mandarin_surface_in_target": 12,
    "high_target_fan_in": 10,
}


@dataclass(frozen=True)
class RuntimeObservation:
    output: str
    winner_entry_ids: tuple[str, ...]
    match_entry_ids: tuple[str, ...]


@dataclass(frozen=True)
class LocatedEntry:
    line: int
    raw: Mapping[str, Any]
    entry: LexiconEntry
    runtime_exclusion: str | None

    @property
    def runtime_eligible(self) -> bool:
        return self.runtime_exclusion is None


def load_entries(path: Path) -> list[LocatedEntry]:
    entries: list[LocatedEntry] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            entry = LexiconEntry.from_dict(raw)
            entries.append(
                LocatedEntry(
                    line=line_number,
                    raw=raw,
                    entry=entry,
                    runtime_exclusion=runtime_exclusion_reason(entry),
                )
            )
    return entries


def observe_runtime_sources(entries: Sequence[LocatedEntry]) -> dict[str, RuntimeObservation]:
    """Probe exact-source runtime winners using the currently built artifacts."""

    from taigi_converter import ConversionResult, TaigiConverter

    converter = TaigiConverter()
    sources = sorted(
        {located.entry.src for located in entries if located.entry.status == "active" and located.runtime_eligible}
    )
    observations: dict[str, RuntimeObservation] = {}
    for source in sources:
        result = converter.convert(source, trace=True)
        if not isinstance(result, ConversionResult):
            raise TypeError("trace=True 必須回傳 ConversionResult")
        exact_winners = tuple(
            match.entry_id
            for match in result.matches
            if match.src == source and match.start == 0 and match.end == len(source)
        )
        observations[source] = RuntimeObservation(
            output=result.output,
            winner_entry_ids=exact_winners,
            match_entry_ids=tuple(match.entry_id for match in result.matches),
        )
    return observations


def _text_len(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _has_hanji(text: str) -> bool:
    return contains_han_ideograph(text)


def _entry_payload(located: LocatedEntry) -> dict[str, Any]:
    entry = located.entry
    return {
        "entry_id": entry.entry_id,
        "line": located.line,
        "src": entry.src,
        "tgt": entry.tgt,
        "level": entry.level,
        "tier": entry.tier,
        "priority": entry.priority,
        "trust": entry.trust,
        "source": entry.source,
        "updated_by": entry.updated_by,
        "updated_at": entry.updated_at,
        "context": entry.context,
        "runtime_eligible": located.runtime_eligible,
        "runtime_exclusion_reason": located.runtime_exclusion,
    }


def _signal_details(
    located: LocatedEntry,
    *,
    active_targets_by_source: Mapping[str, set[str]],
    low_trust_sources: set[str],
    low_trust_edges: set[tuple[str, str]],
    sources_by_target: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    entry = located.entry
    src = entry.src
    tgt = entry.tgt
    src_len = _text_len(src)
    tgt_len = _text_len(tgt)
    signals: list[tuple[str, str]] = []

    if not tgt or not _has_hanji(tgt):
        signals.append(("non_hanji_or_empty_target", "target 為空或不含漢字"))
    if _ASCII_WORD_RE.search(tgt):
        signals.append(("ascii_in_target", "target 含 ASCII 單字或識別符"))
    if src != tgt and (tgt, src) in low_trust_edges:
        signals.append(("reverse_rewrite_edge", "低信任詞條存在反向 rewrite edge"))
    if entry.context is None and src in BROAD_FUNCTION_WORDS and src != tgt:
        signals.append(("broad_context_free_function_word", "高頻功能詞以無 context 規則全域改寫"))
    competing_targets = active_targets_by_source.get(src, set())
    if len(competing_targets) > 1:
        signals.append(("competing_active_targets", f"同一 active source 有 {len(competing_targets)} 個 targets"))
    if src_len >= 4 and tgt_len <= 2 and tgt_len * 2 <= src_len:
        signals.append(("severe_contraction", f"長度由 {src_len} 大幅縮為 {tgt_len}"))
    if entry.context is None and src != tgt and src_len <= 2:
        signals.append(("short_context_free_rewrite", f"長度 {src_len} 的短 source 無 context 全域改寫"))
    if src_len <= 3 and tgt_len >= src_len * 2 + 2:
        signals.append(("severe_expansion", f"長度由 {src_len} 大幅擴為 {tgt_len}"))
    if entry.trust == TRUST_MACHINE and entry.tier == "manual_hotfix" and located.runtime_eligible:
        signals.append(("runtime_machine_override", "machine manual_hotfix 仍可進入 runtime"))
    if src != tgt and tgt in low_trust_sources:
        signals.append(("target_is_low_trust_source", "target 同時是另一個低信任 active source"))
    markers = sorted(marker for marker in MANDARIN_TARGET_MARKERS if marker in tgt and marker not in src)
    if markers:
        signals.append(("mandarin_surface_in_target", "target 新增華語表面詞：「" + "、".join(markers) + "」"))
    fan_in = len(sources_by_target.get(tgt, set()))
    if src != tgt and fan_in >= 8:
        signals.append(("high_target_fan_in", f"共有 {fan_in} 個低信任 sources 聚合到此 target"))

    return [
        {
            "kind": kind,
            "weight": SIGNAL_WEIGHTS[kind],
            "detail": detail,
        }
        for kind, detail in sorted(signals, key=lambda item: (-SIGNAL_WEIGHTS[item[0]], item[0]))
    ]


def audit_entries(
    entries: Sequence[LocatedEntry],
    *,
    trusts: Iterable[str] = DEFAULT_TRUSTS,
    include_runtime_excluded: bool = False,
    include_runtime_shadowed: bool = False,
    runtime_observations: Mapping[str, RuntimeObservation] | None = None,
    limit: int | None = 100,
) -> dict[str, Any]:
    selected_trusts = tuple(sorted(set(trusts)))
    active = [located for located in entries if located.entry.status == "active"]
    selected = [located for located in active if located.entry.trust in selected_trusts]
    policy_candidates = [located for located in selected if include_runtime_excluded or located.runtime_eligible]

    def is_runtime_candidate(located: LocatedEntry) -> bool:
        if runtime_observations is None or include_runtime_shadowed:
            return True
        if not located.runtime_eligible:
            return include_runtime_excluded
        observation = runtime_observations.get(located.entry.src)
        return observation is not None and located.entry.entry_id in observation.winner_entry_ids

    candidates = [located for located in policy_candidates if is_runtime_candidate(located)]

    active_targets_by_source: dict[str, set[str]] = defaultdict(set)
    for located in active:
        active_targets_by_source[located.entry.src].add(located.entry.tgt)

    low_trust_sources = {located.entry.src for located in selected}
    low_trust_edges = {(located.entry.src, located.entry.tgt) for located in selected}
    sources_by_target: dict[str, set[str]] = defaultdict(set)
    for located in selected:
        sources_by_target[located.entry.tgt].add(located.entry.src)

    findings: list[dict[str, Any]] = []
    signal_counts: Counter[str] = Counter()
    for located in candidates:
        signals = _signal_details(
            located,
            active_targets_by_source=active_targets_by_source,
            low_trust_sources=low_trust_sources,
            low_trust_edges=low_trust_edges,
            sources_by_target=sources_by_target,
        )
        if not signals:
            continue
        signal_counts.update(signal["kind"] for signal in signals)
        payload = _entry_payload(located)
        observation = runtime_observations.get(located.entry.src) if runtime_observations is not None else None
        if observation is not None:
            payload["runtime_output"] = observation.output
            payload["runtime_winner_entry_ids"] = list(observation.winner_entry_ids)
            payload["runtime_match_entry_ids"] = list(observation.match_entry_ids)
            payload["runtime_exact_winner"] = located.entry.entry_id in observation.winner_entry_ids
        payload["risk_score"] = sum(signal["weight"] for signal in signals)
        payload["signals"] = signals
        findings.append(payload)

    findings.sort(
        key=lambda item: (
            -item["risk_score"],
            item["src"],
            item["tgt"],
            item["entry_id"],
            item["line"],
        )
    )
    total_finding_count = len(findings)
    if limit is not None and limit > 0:
        findings = findings[:limit]

    runtime_exclusion_counts = Counter(
        located.runtime_exclusion for located in selected if located.runtime_exclusion is not None
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "configuration": {
            "trusts": list(selected_trusts),
            "include_runtime_excluded": include_runtime_excluded,
            "include_runtime_shadowed": include_runtime_shadowed,
            "runtime_probe_enabled": runtime_observations is not None,
            "limit": 0 if limit is None else limit,
        },
        "summary": {
            "source_entry_count": len(entries),
            "active_entry_count": len(active),
            "selected_active_entry_count": len(selected),
            "selected_runtime_eligible_entry_count": sum(located.runtime_eligible for located in selected),
            "selected_runtime_excluded_entry_count": sum(not located.runtime_eligible for located in selected),
            "policy_candidate_entry_count": len(policy_candidates),
            "runtime_exact_winner_entry_count": sum(
                located.runtime_eligible
                and runtime_observations is not None
                and (observation := runtime_observations.get(located.entry.src)) is not None
                and located.entry.entry_id in observation.winner_entry_ids
                for located in policy_candidates
            ),
            "runtime_shadowed_entry_count": (
                sum(
                    located.runtime_eligible
                    and (observation := runtime_observations.get(located.entry.src)) is not None
                    and located.entry.entry_id not in observation.winner_entry_ids
                    for located in policy_candidates
                )
                if runtime_observations is not None
                else 0
            ),
            "candidate_entry_count": len(candidates),
            "risk_finding_count": total_finding_count,
            "reported_finding_count": len(findings),
            "signal_counts": dict(sorted(signal_counts.items())),
            "runtime_exclusion_counts": {
                str(reason): count for reason, count in sorted(runtime_exclusion_counts.items())
            },
        },
        "findings": findings,
    }


def serialize_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _print_human(report: Mapping[str, Any], *, stdout: TextIO) -> None:
    summary = report["summary"]
    print("Low-trust lexicon risk audit", file=stdout)
    print(f"selected active entries: {summary['selected_active_entry_count']}", file=stdout)
    print(f"runtime eligible: {summary['selected_runtime_eligible_entry_count']}", file=stdout)
    print(f"runtime excluded: {summary['selected_runtime_excluded_entry_count']}", file=stdout)
    if report["configuration"]["runtime_probe_enabled"]:
        print(f"runtime exact winners: {summary['runtime_exact_winner_entry_count']}", file=stdout)
        print(f"runtime shadowed: {summary['runtime_shadowed_entry_count']}", file=stdout)
    print(
        f"risk findings: {summary['risk_finding_count']} (reported {summary['reported_finding_count']})",
        file=stdout,
    )
    for finding in report["findings"]:
        kinds = ",".join(signal["kind"] for signal in finding["signals"])
        print(
            f"- score={finding['risk_score']:>3} {finding['entry_id']} {finding['src']} -> {finding['tgt']} [{kinds}]",
            file=stdout,
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank active low-trust lexicon entries for manual semantic review.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "lexicon_entries.jsonl",
        help="lexicon JSONL path",
    )
    parser.add_argument(
        "--trust",
        action="append",
        choices=sorted(VALID_TRUSTS),
        dest="trusts",
        help="trust label to include; repeatable (default: machine and seed)",
    )
    parser.add_argument(
        "--include-runtime-excluded",
        action="store_true",
        help="also rank rows already excluded by runtime policy",
    )
    parser.add_argument(
        "--include-runtime-shadowed",
        action="store_true",
        help="also rank policy-eligible rows that do not win an exact-source runtime probe",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="maximum findings in output; 0 emits all findings",
    )
    parser.add_argument("--json", action="store_true", help="print deterministic JSON")
    parser.add_argument("--output", type=Path, help="write output to this path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout) -> int:
    args = _parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit 不可小於 0")

    entries = load_entries(args.data)
    default_data = (REPO_ROOT / "data" / "lexicon_entries.jsonl").resolve()
    runtime_observations = observe_runtime_sources(entries) if args.data.resolve() == default_data else None
    report = audit_entries(
        entries,
        trusts=args.trusts or DEFAULT_TRUSTS,
        include_runtime_excluded=args.include_runtime_excluded,
        include_runtime_shadowed=args.include_runtime_shadowed,
        runtime_observations=runtime_observations,
        limit=None if args.limit == 0 else args.limit,
    )
    if args.json or args.output:
        text = serialize_report(report)
    else:
        from io import StringIO

        buffer = StringIO()
        _print_human(report, stdout=buffer)
        text = buffer.getvalue()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
