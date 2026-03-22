from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lexicon_policy import VALID_TRUSTS, is_sentence_manual_override, is_trusted_manual_entry, runtime_exclusion_reason
from models import PASS_ORDER, TIER_ORDER, LexiconEntry, RuleEntry


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR_NAME = "artifacts"
CORE_LEXICON_FILE = "core_lexicon.json"

TIER_INDEX = {tier: i for i, tier in enumerate(TIER_ORDER)}
PASS_INDEX = {name: i for i, name in enumerate(PASS_ORDER)}
RESIDUAL_MANDARIN_TERMS = ["東西", "什麼", "為什麼", "為何"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _entry_id(src: str, tgt: str, level: str, tier: str, source: str) -> str:
    raw = f"{src}|{tgt}|{level}|{tier}|{source}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"lx_{digest}"


def _rule_id(pass_name: str, pattern: str, replacement: str) -> str:
    raw = f"{pass_name}|{pattern}|{replacement}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"rl_{digest}"


def _entry_sort_key(entry: LexiconEntry) -> tuple[int, int, int, float, str]:
    return (
        TIER_INDEX.get(entry.tier, 999),
        -entry.priority,
        -len(entry.src),
        -entry.score,
        entry.entry_id,
    )


def _load_core_lexicon_entries(data_dir: Path) -> list[LexiconEntry]:
    core_path = data_dir / CORE_LEXICON_FILE
    if not core_path.exists():
        return []

    data = _read_json(core_path)
    if not isinstance(data, list):
        raise ValueError("core_lexicon.json 必須是 list")

    rows: list[LexiconEntry] = []
    now = _now_iso()
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"core_lexicon.json 第 {idx} 筆必須是 object")
        src = str(item.get("src", "")).strip()
        tgt = str(item.get("tgt", "")).strip()
        if not src or not tgt:
            raise ValueError(f"core_lexicon.json 第 {idx} 筆 src/tgt 不可為空")

        level = str(item.get("level", "phrase"))
        if level not in {"sentence", "phrase", "char"}:
            raise ValueError(f"core_lexicon.json 第 {idx} 筆 level 不合法: {level}")

        priority = int(item.get("priority", 800 if level != "char" else 80))
        source = str(item.get("source", "core:lexicon"))
        updated_by = str(item.get("updated_by", "core_lexicon"))
        updated_at = str(item.get("updated_at", now))

        rows.append(
            LexiconEntry(
                entry_id=str(item.get("entry_id", _entry_id(src, tgt, level, "core", source))),
                src=src,
                tgt=tgt,
                level=level,
                tier="core",
                priority=priority,
                context=item.get("context"),
                score=float(item.get("score", 1.0)),
                status=str(item.get("status", "active")),
                source=source,
                trust=str(item.get("trust", "human")),
                updated_by=updated_by,
                updated_at=updated_at,
            )
        )
    return rows


def default_rule_entries() -> list[RuleEntry]:
    raw_rules = [
        {
            "pass_name": "normalization",
            "type": "literal",
            "pattern": "臺",
            "replacement": "台",
            "priority": 100,
            "note": "統一常見字形",
        },
        {
            "pass_name": "normalization",
            "type": "regex",
            "pattern": r"\s+",
            "replacement": "",
            "priority": 90,
            "note": "移除多餘空白",
        },
        {
            "pass_name": "grammar",
            "type": "regex",
            "pattern": r"食飽了沒",
            "replacement": "食飽未",
            "priority": 100,
            "note": "固定句型",
        },
        {
            "pass_name": "grammar",
            "type": "regex",
            "pattern": r"了沒",
            "replacement": "未",
            "priority": 95,
            "note": "了沒句型",
        },
        {
            "pass_name": "grammar",
            "type": "regex",
            "pattern": r"嗎\??$",
            "replacement": "無",
            "priority": 80,
            "note": "疑問語氣",
        },
        {
            "pass_name": "fluency",
            "type": "regex",
            "pattern": r"真好吃",
            "replacement": "真好食",
            "priority": 90,
            "note": "口語優化",
        },
        {
            "pass_name": "fluency",
            "type": "regex",
            "pattern": r"這馬咧做啥",
            "replacement": "這馬咧創啥",
            "priority": 85,
            "note": "固定句型",
        },
        {
            "pass_name": "fluency",
            "type": "regex",
            "pattern": r"咧做啥",
            "replacement": "咧創啥",
            "priority": 80,
            "note": "動詞口語化",
        },
        {
            "pass_name": "fluency",
            "type": "literal",
            "pattern": "為何",
            "replacement": "按怎",
            "priority": 75,
            "note": "華語殘留修正",
        },
        {
            "pass_name": "fluency",
            "type": "literal",
            "pattern": "為什麼",
            "replacement": "按怎",
            "priority": 75,
            "note": "華語殘留修正",
        },
        {
            "pass_name": "fluency",
            "type": "literal",
            "pattern": "東西",
            "replacement": "物件",
            "priority": 60,
            "note": "華語殘留修正",
        },
        {
            "pass_name": "fluency",
            "type": "literal",
            "pattern": "什麼",
            "replacement": "啥",
            "priority": 60,
            "note": "華語殘留修正",
        },
    ]

    return [
        RuleEntry(
            rule_id=_rule_id(r["pass_name"], r["pattern"], r["replacement"]),
            pass_name=r["pass_name"],
            type=r["type"],
            pattern=r["pattern"],
            replacement=r["replacement"],
            priority=r["priority"],
            enabled=True,
            note=r["note"],
        )
        for r in raw_rules
    ]


def _load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    items: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.add(line)
    return items


def migrate_legacy_data(data_dir: Path = DATA_DIR) -> dict[str, int]:
    phrase_path = data_dir / "phrase_lexicon.json"
    char_path = data_dir / "char_lexicon.json"
    if not phrase_path.exists() or not char_path.exists():
        raise FileNotFoundError("找不到 legacy 詞典檔（phrase_lexicon.json / char_lexicon.json）")

    allowlist = _load_allowlist(data_dir / "char_verified_allowlist.txt")
    phrase_lexicon = _read_json(phrase_path)
    char_lexicon = _read_json(char_path)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    now = _now_iso()

    for src, tgt in sorted(phrase_lexicon.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        level = "sentence" if len(src) >= 12 else "phrase"
        priority = 60 if level == "sentence" else 40
        key = (src, tgt, level, "base", "active")
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            LexiconEntry(
                entry_id=_entry_id(src, tgt, level, "base", "legacy:phrase_lexicon"),
                src=src,
                tgt=tgt,
                level=level,
                tier="base",
                priority=priority,
                context=None,
                score=0.0,
                status="active",
                source="legacy:phrase_lexicon",
                trust="seed",
                updated_by="migration",
                updated_at=now,
            ).to_dict()
        )

    for src, tgt in sorted(char_lexicon.items(), key=lambda kv: kv[0]):
        status = "active" if src in allowlist else "disabled"
        key = (src, tgt, "char", "base", status)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            LexiconEntry(
                entry_id=_entry_id(src, tgt, "char", "base", "legacy:char_lexicon"),
                src=src,
                tgt=tgt,
                level="char",
                tier="base",
                priority=10,
                context=None,
                score=0.0,
                status=status,
                source="legacy:char_lexicon",
                trust="seed",
                updated_by="migration",
                updated_at=now,
            ).to_dict()
        )

    write_jsonl(data_dir / "lexicon_entries.jsonl", rows)

    rule_path = data_dir / "rule_entries.jsonl"
    if not rule_path.exists():
        write_jsonl(rule_path, [r.to_dict() for r in default_rule_entries()])

    review_queue_path = data_dir / "review_queue.jsonl"
    review_queue_path.touch(exist_ok=True)

    return {
        "lexicon_entries": len(rows),
        "phrase_entries": len(phrase_lexicon),
        "char_entries": len(char_lexicon),
        "char_active": sum(1 for r in rows if r["level"] == "char" and r["status"] == "active"),
    }


def _build_phrase_trie(entries: list[LexiconEntry]) -> dict[str, Any]:
    root: dict[str, Any] = {"children": {}}
    for entry in entries:
        if entry.level not in {"phrase", "sentence"}:
            continue
        if is_sentence_manual_override(entry):
            continue
        # context-aware entries are matched in contextual pass only,
        # so we keep them out of trie to avoid bypassing context checks.
        if entry.context:
            continue
        if entry.status != "active":
            continue
        node = root
        for ch in entry.src:
            node = node["children"].setdefault(ch, {"children": {}})
        node.setdefault("entry_ids", []).append(entry.entry_id)

    def normalize_node(node: dict[str, Any]) -> dict[str, Any]:
        out = {"children": {}}
        children = node.get("children", {})
        for ch in sorted(children.keys()):
            out["children"][ch] = normalize_node(children[ch])
        if "entry_ids" in node:
            out["entry_ids"] = node["entry_ids"]
        return out

    return normalize_node(root)


def detect_masked_rules(rules: list[RuleEntry]) -> list[str]:
    warnings: list[str] = []
    grouped: dict[str, list[RuleEntry]] = {}
    for rule in rules:
        if not rule.enabled:
            continue
        grouped.setdefault(rule.pass_name, []).append(rule)

    for pass_name, pass_rules in grouped.items():
        ordered = sorted(
            pass_rules,
            key=lambda r: (PASS_INDEX.get(r.pass_name, 999), -r.priority, r.rule_id),
        )
        for i, earlier in enumerate(ordered):
            if earlier.type != "literal" or not earlier.pattern:
                continue
            for later in ordered[i + 1 :]:
                if later.type != "literal" or not later.pattern:
                    continue
                if later.pattern.startswith(earlier.pattern):
                    warnings.append(
                        (
                            f"[{pass_name}] rule {earlier.rule_id} ({earlier.pattern!r} -> {earlier.replacement!r}) "
                            f"可能遮蔽 {later.rule_id} ({later.pattern!r} -> {later.replacement!r})"
                        )
                    )
    return warnings


def compile_runtime_artifacts(data_dir: Path = DATA_DIR, fail_on_mask: bool = False) -> dict[str, Any]:
    lexicon_path = data_dir / "lexicon_entries.jsonl"
    rule_path = data_dir / "rule_entries.jsonl"

    if not lexicon_path.exists():
        raise FileNotFoundError("找不到 data/lexicon_entries.jsonl")
    if not rule_path.exists():
        raise FileNotFoundError("找不到 data/rule_entries.jsonl")

    source_rows = load_jsonl(lexicon_path)
    invalid_trust_rows: list[str] = []
    for idx, row in enumerate(source_rows, start=1):
        trust = row.get("trust")
        if trust not in VALID_TRUSTS:
            entry_id = row.get("entry_id", f"line_{idx}")
            invalid_trust_rows.append(f"{entry_id}:{trust!r}")
    if invalid_trust_rows:
        sample = ", ".join(invalid_trust_rows[:10])
        raise ValueError(
            "lexicon_entries.jsonl 存在非法 trust（需為 human/machine/seed）: "
            f"{sample}"
        )

    source_entries = [LexiconEntry.from_dict(row) for row in source_rows]
    core_entries = _load_core_lexicon_entries(data_dir)
    entries = list({entry.entry_id: entry for entry in (source_entries + core_entries)}.values())
    rules = [RuleEntry.from_dict(row) for row in load_jsonl(rule_path)]

    entries_by_id = {entry.entry_id: entry for entry in entries}

    runtime_entries: list[LexiconEntry] = []
    runtime_excluded: dict[str, str] = {}
    for entry in entries:
        reason = runtime_exclusion_reason(entry)
        if reason:
            runtime_excluded[entry.entry_id] = reason
            continue
        runtime_entries.append(entry)

    active_entries = [entry for entry in runtime_entries if entry.status == "active"]
    active_rules = [rule for rule in rules if rule.enabled]

    sentence_override_map: dict[str, list[str]] = {}
    contextual_override_ids: list[str] = []
    char_map: dict[str, list[str]] = {}

    for entry in sorted(active_entries, key=_entry_sort_key):
        if is_sentence_manual_override(entry):
            sentence_override_map.setdefault(entry.src, []).append(entry.entry_id)
        if is_trusted_manual_entry(entry) and entry.context:
            contextual_override_ids.append(entry.entry_id)
        if entry.level == "char":
            char_map.setdefault(entry.src, []).append(entry.entry_id)

    phrase_trie = _build_phrase_trie(active_entries)

    sorted_rules = sorted(
        active_rules,
        key=lambda r: (PASS_INDEX.get(r.pass_name, 999), -r.priority, r.rule_id),
    )

    mask_warnings = detect_masked_rules(sorted_rules)
    if fail_on_mask and mask_warnings:
        joined = "\n".join(mask_warnings)
        raise ValueError(f"偵測到 rule masking：\n{joined}")

    artifacts_dir = data_dir / ARTIFACT_DIR_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        artifacts_dir / "entry_table.json",
        {
            "version": 1,
            "entries": {entry_id: entry.to_dict() for entry_id, entry in entries_by_id.items()},
            "runtime_excluded": runtime_excluded,
        },
    )

    _write_json(
        artifacts_dir / "phrase_trie.json",
        {
            "version": 1,
            "trie": phrase_trie,
        },
    )

    _write_json(
        artifacts_dir / "char_map.json",
        {
            "version": 1,
            "map": char_map,
        },
    )

    rules_by_pass: dict[str, list[dict[str, Any]]] = {name: [] for name in PASS_ORDER}
    for rule in sorted_rules:
        rules_by_pass.setdefault(rule.pass_name, []).append(rule.to_dict())

    _write_json(
        artifacts_dir / "rule_plan.json",
        {
            "version": 1,
            "pass_order": PASS_ORDER,
            "rules": rules_by_pass,
            "mask_warnings": mask_warnings,
            "residual_terms": RESIDUAL_MANDARIN_TERMS,
        },
    )

    _write_json(
        artifacts_dir / "override_index.json",
        {
            "version": 1,
            "tier_order": TIER_ORDER,
            "sentence_override_map": sentence_override_map,
            "contextual_override_ids": contextual_override_ids,
        },
    )

    manifest = {
        "version": 1,
        "generated_at": _now_iso(),
        "entry_count": len(entries),
        "runtime_entry_count": len(runtime_entries),
        "runtime_excluded_entry_count": len(runtime_excluded),
        "runtime_excluded_reasons": dict(Counter(runtime_excluded.values())),
        "core_entry_count": len(core_entries),
        "active_entry_count": len(active_entries),
        "rule_count": len(rules),
        "active_rule_count": len(active_rules),
        "mask_warning_count": len(mask_warnings),
    }
    _write_json(artifacts_dir / "manifest.json", manifest)

    return manifest


def ensure_runtime_ready(data_dir: Path = DATA_DIR, fail_on_mask: bool = False) -> dict[str, Any]:
    lexicon_entries = data_dir / "lexicon_entries.jsonl"
    rule_entries = data_dir / "rule_entries.jsonl"
    core_lexicon = data_dir / CORE_LEXICON_FILE
    manifest_path = data_dir / ARTIFACT_DIR_NAME / "manifest.json"

    migrated_stats: dict[str, int] | None = None
    if not lexicon_entries.exists():
        migrated_stats = migrate_legacy_data(data_dir)
    elif not rule_entries.exists():
        write_jsonl(rule_entries, [r.to_dict() for r in default_rule_entries()])

    must_compile = not manifest_path.exists()
    if not must_compile:
        source_mtimes = [lexicon_entries.stat().st_mtime, rule_entries.stat().st_mtime]
        if core_lexicon.exists():
            source_mtimes.append(core_lexicon.stat().st_mtime)
        latest_source_mtime = max(source_mtimes)
        must_compile = manifest_path.stat().st_mtime < latest_source_mtime

    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    if must_compile:
        manifest = compile_runtime_artifacts(data_dir=data_dir, fail_on_mask=fail_on_mask)

    if migrated_stats:
        manifest["migration"] = migrated_stats
    return manifest
