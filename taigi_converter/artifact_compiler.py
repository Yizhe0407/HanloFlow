from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text, exclusive_file_lock
from .lexicon_policy import (
    VALID_TRUSTS,
    is_sentence_manual_override,
    is_trusted_manual_entry,
    runtime_exclusion_reason,
    runtime_layer_rank,
)
from .models import PASS_ORDER, TIER_ORDER, LexiconEntry, RuleEntry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR_NAME = "artifacts"
CORE_LEXICON_FILE = "core_lexicon.json"
CHAR_ALLOWLIST_FILE = "char_verified_allowlist.txt"
RUNTIME_FILTER_CORE_IDENTITY_PROTECTED = "core_identity_protected_term"
RUNTIME_FILTER_IDENTITY_PASSTHROUGH_MASKED = "identity_passthrough_masked"
LEXICON_STAGE = "split_char_after_rules"
COMPILER_VERSION = 3
MANIFEST_VERSION = 3
SOURCE_FILES = ("lexicon_entries.jsonl", "rule_entries.jsonl", CORE_LEXICON_FILE, CHAR_ALLOWLIST_FILE)
ARTIFACT_FILES = (
    "entry_table.json",
    "phrase_trie.json",
    "char_map.json",
    "rule_plan.json",
    "override_index.json",
)
VALID_LEVELS = {"sentence", "phrase", "char"}
VALID_STATUSES = {"active", "disabled"}
VALID_RULE_TYPES = {"literal", "regex"}
VALID_CONTEXT_FIELDS = {"left_regex", "right_regex", "full_regex", "left_literal", "right_literal"}

TIER_INDEX = {tier: i for i, tier in enumerate(TIER_ORDER)}
PASS_INDEX = {name: i for i, name in enumerate(PASS_ORDER)}
RESIDUAL_MANDARIN_TERMS = ["東西", "什麼", "為什麼", "為何"]
RULE_TOKEN_MAP = {
    "{{PRONOUN}}": r"(?:我|你|伊|恁|怹|咱|阮|他|她|你們|他們|逐家)",
}
REGEX_DOT_GREEDY_RE = re.compile(r"(?<!\\)\.(?:\*|\+)")
REGEX_UNBOUNDED_NEG_CLASS_RE = re.compile(r"\[\^[^\]]+\]\+")
ENTRY_ID_PREFIX = "lx_"
RULE_ID_PREFIX = "rl_"
RUNTIME_ID_SUFFIX_LEN = 12
RUNTIME_LEVELS = ("sentence", "phrase", "char")
RUNTIME_LEVEL_INDEX = {name: idx for idx, name in enumerate(RUNTIME_LEVELS)}
RUNTIME_TIER_INDEX = {name: idx for idx, name in enumerate(TIER_ORDER)}
RUNTIME_TRUSTS = ("human", "machine", "seed")
RUNTIME_TRUST_INDEX = {name: idx for idx, name in enumerate(RUNTIME_TRUSTS)}
RUNTIME_CONTEXT_RIGHT_REGEX = "r"
RUNTIME_CONTEXT_LEFT_LITERAL = "l"
MANIFEST_SHORT_KEYS = {
    "version": "v",
    "compiler_version": "cv",
    "source_digest": "sd",
    "entry_count": "e",
    "runtime_entry_count": "re",
    "runtime_excluded_entry_count": "rx",
    "runtime_excluded_reasons": "rr",
    "core_entry_count": "c",
    "active_entry_count": "a",
    "rule_count": "r",
    "active_rule_count": "ar",
    "mask_warning_count": "mw",
    "regex_hazard_count": "rh",
    "pipeline_conflict_count": "pc",
    "protected_term_count": "pt",
    "residual_core_term_count": "rc",
    "lexicon_stage": "ls",
    "core_identity_protected_entry_count": "ci",
    "identity_passthrough_protected_entry_count": "ip",
    "artifact_hashes": "ah",
}
MANIFEST_LONG_KEYS = {short_key: long_key for long_key, short_key in MANIFEST_SHORT_KEYS.items()}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    atomic_write_json(path, data, indent=indent, mode=0o644)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 格式錯誤: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL 每列必須是 object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, rows)


def _source_digest(data_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(f"compiler={COMPILER_VERSION}\n".encode("ascii"))
    for name in SOURCE_FILES:
        path = data_dir / name
        if not path.exists():
            digest.update(f"{name}:missing\n".encode())
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_lexicon_rows(rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        prefix = f"lexicon line {index}"
        for field in ("entry_id", "src", "level", "tier"):
            if not isinstance(row.get(field), str) or not row.get(field):
                errors.append(f"{prefix}: {field} 必須是非空字串")
        if not isinstance(row.get("tgt", ""), str):
            errors.append(f"{prefix}: tgt 必須是字串")
        if row.get("level") not in VALID_LEVELS:
            errors.append(f"{prefix}: level={row.get('level')!r} 不合法")
        if row.get("level") == "char" and len(row.get("src", "")) != 1:
            errors.append(f"{prefix}: char 詞條的 src 必須恰為一個字元")
        if row.get("tier") not in TIER_INDEX:
            errors.append(f"{prefix}: tier={row.get('tier')!r} 不合法")
        if row.get("status", "active") not in VALID_STATUSES:
            errors.append(f"{prefix}: status={row.get('status')!r} 不合法")
        if row.get("trust") not in VALID_TRUSTS:
            errors.append(f"{prefix}: trust={row.get('trust')!r} 不合法")

        score_value = row.get("score", 0.0)
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            errors.append(f"{prefix}: score 必須是數字")
        else:
            score = float(score_value)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                errors.append(f"{prefix}: score 必須是 0 到 1 的有限數字")

        priority = row.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            errors.append(f"{prefix}: priority 必須是整數")

        context = row.get("context")
        if context is not None and not isinstance(context, dict):
            errors.append(f"{prefix}: context 必須是 object 或 null")
        elif isinstance(context, dict):
            unknown = set(context) - VALID_CONTEXT_FIELDS
            if unknown:
                errors.append(f"{prefix}: context 含未知欄位 {sorted(unknown)}")
            for field, value in context.items():
                if not isinstance(value, str) or not value:
                    errors.append(f"{prefix}: context.{field} 必須是非空字串")
                    continue
                if field.endswith("_regex"):
                    try:
                        re.compile(value)
                    except re.error as exc:
                        errors.append(f"{prefix}: context.{field} regex 無法編譯: {exc}")
    if errors:
        raise ValueError("詞典 schema 驗證失敗:\n" + "\n".join(errors[:50]))


def _validate_rule_rows(rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_patterns: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows, start=1):
        prefix = f"rule line {index}"
        for field in ("rule_id", "pass_name", "pattern"):
            if not isinstance(row.get(field), str) or not row.get(field):
                errors.append(f"{prefix}: {field} 必須是非空字串")
        if not isinstance(row.get("replacement", ""), str):
            errors.append(f"{prefix}: replacement 必須是字串")
        if row.get("pass_name") not in PASS_INDEX:
            errors.append(f"{prefix}: pass_name={row.get('pass_name')!r} 不合法")
        if row.get("type", "literal") not in VALID_RULE_TYPES:
            errors.append(f"{prefix}: type={row.get('type')!r} 不合法")
        if not isinstance(row.get("enabled", True), bool):
            errors.append(f"{prefix}: enabled 必須是 boolean")
        priority = row.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            errors.append(f"{prefix}: priority 必須是整數")
        if not isinstance(row.get("note", ""), str):
            errors.append(f"{prefix}: note 必須是字串")
        rule_id = row.get("rule_id")
        if isinstance(rule_id, str):
            if rule_id in seen_ids:
                errors.append(f"{prefix}: rule_id={rule_id!r} 重複")
            seen_ids.add(rule_id)
        key = (str(row.get("pass_name")), str(row.get("type", "literal")), str(row.get("pattern")))
        if key in seen_patterns:
            errors.append(f"{prefix}: pass/type/pattern 重複: {key!r}")
        seen_patterns.add(key)
        if row.get("type", "literal") == "regex" and isinstance(row.get("pattern"), str):
            try:
                re.compile(_expand_rule_tokens(row["pattern"]))
            except re.error as exc:
                errors.append(f"{prefix}: regex 無法編譯: {exc}")
    if errors:
        raise ValueError("規則 schema 驗證失敗:\n" + "\n".join(errors[:50]))


def _validate_entry_target_conflicts(entries: list[LexiconEntry]) -> None:
    grouped: dict[tuple[Any, ...], list[LexiconEntry]] = {}
    for entry in entries:
        if entry.status != "active":
            continue
        key = (
            entry.src,
            runtime_layer_rank(entry),
            entry.priority,
            entry.score,
        )
        grouped.setdefault(key, []).append(entry)
    conflicts = [items for items in grouped.values() if len({entry.tgt for entry in items}) > 1]
    if not conflicts:
        return
    details = []
    for items in conflicts[:20]:
        details.append(f"{items[0].src!r}: " + ", ".join(f"{item.entry_id}->{item.tgt!r}" for item in items))
    raise ValueError("詞典存在同順位但不同目標的 active 詞條:\n" + "\n".join(details))

def _entry_id(src: str, tgt: str, level: str, tier: str, source: str) -> str:
    raw = f"{src}|{tgt}|{level}|{tier}|{source}".encode()
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"lx_{digest}"


def _rule_id(pass_name: str, pattern: str, replacement: str) -> str:
    raw = f"{pass_name}|{pattern}|{replacement}".encode()
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"rl_{digest}"


def _expand_rule_tokens(text: str) -> str:
    expanded = text
    for token, token_pattern in RULE_TOKEN_MAP.items():
        expanded = expanded.replace(token, token_pattern)
    return expanded


def _expand_rules_with_tokens(rules: list[RuleEntry]) -> list[RuleEntry]:
    expanded_rules: list[RuleEntry] = []
    for rule in rules:
        expanded_rules.append(
            RuleEntry(
                rule_id=rule.rule_id,
                pass_name=rule.pass_name,
                type=rule.type,
                pattern=_expand_rule_tokens(rule.pattern),
                replacement=_expand_rule_tokens(rule.replacement),
                priority=rule.priority,
                enabled=rule.enabled,
                note=rule.note,
            )
        )
    return expanded_rules


def _pack_runtime_context(context: dict[str, Any] | None) -> list[str] | dict[str, Any] | None:
    if not context:
        return None
    if set(context.keys()) == {"right_regex"}:
        return [RUNTIME_CONTEXT_RIGHT_REGEX, str(context["right_regex"])]
    if set(context.keys()) == {"left_literal"}:
        return [RUNTIME_CONTEXT_LEFT_LITERAL, str(context["left_literal"])]
    return context


def _serialize_runtime_rule(rule: RuleEntry) -> list[Any]:
    row: list[Any] = [rule.pattern, rule.replacement]
    if rule.type == "regex" or rule.priority:
        row.append(1 if rule.type == "regex" else 0)
        if rule.priority:
            row.append(rule.priority)
    return row


def _serialize_runtime_sentence_overrides(
    sentence_override_map: dict[str, list[str]],
    *,
    entry_index_by_id: dict[str, int],
) -> dict[str, int | list[int]]:
    runtime_map: dict[str, int | list[int]] = {}
    for src, entry_ids in sentence_override_map.items():
        indexes = [entry_index_by_id[entry_id] for entry_id in entry_ids if entry_id in entry_index_by_id]
        if not indexes:
            continue
        if len(indexes) == 1:
            runtime_map[src] = indexes[0]
        else:
            runtime_map[src] = indexes
    return runtime_map


def _serialize_runtime_char_map(
    char_map: dict[str, list[str]],
    *,
    entry_index_by_id: dict[str, int],
) -> dict[str, list[int]]:
    return {
        src: [entry_index_by_id[entry_id] for entry_id in entry_ids if entry_id in entry_index_by_id]
        for src, entry_ids in char_map.items()
    }


def _serialize_runtime_entry_table(active_entries: list[LexiconEntry]) -> tuple[dict[str, Any], dict[str, int]]:
    ordered_entries = sorted(active_entries, key=lambda entry: entry.entry_id)
    entry_ids = [entry.entry_id for entry in ordered_entries]
    entry_index_by_id = {entry_id: index for index, entry_id in enumerate(entry_ids)}

    combo_entries: dict[tuple[str, str, str], list[LexiconEntry]] = {}
    for entry in ordered_entries:
        combo = (entry.level, entry.tier, entry.trust)
        combo_entries.setdefault(combo, []).append(entry)

    combos = list(combo_entries.keys())
    combo_index = {combo: index for index, combo in enumerate(combos)}
    combo_defaults: list[list[Any]] = []
    for combo in combos:
        bucket = combo_entries[combo]
        default_priority = Counter(item.priority for item in bucket).most_common(1)[0][0]
        default_score = Counter(item.score for item in bucket).most_common(1)[0][0]
        combo_defaults.append(
            [
                RUNTIME_LEVEL_INDEX[combo[0]],
                RUNTIME_TIER_INDEX[combo[1]],
                RUNTIME_TRUST_INDEX[combo[2]],
                default_priority,
                default_score,
            ]
        )

    rows: list[list[Any]] = []
    for entry in ordered_entries:
        combo = (entry.level, entry.tier, entry.trust)
        kind_index = combo_index[combo]
        default_priority = combo_defaults[kind_index][3]
        default_score = combo_defaults[kind_index][4]
        packed_context = _pack_runtime_context(entry.context)
        row: list[Any] = [entry.src, entry.tgt, kind_index]
        if entry.priority != default_priority or entry.score != default_score or packed_context is not None:
            row.append(entry.priority)
            if entry.score != default_score or packed_context is not None:
                row.append(entry.score)
                if packed_context is not None:
                    row.append(packed_context)
        rows.append(row)

    table_doc: dict[str, Any] = {
        "v": 7,
        "k": combo_defaults,
        "e": rows,
    }
    if all(entry_id.startswith(ENTRY_ID_PREFIX) for entry_id in entry_ids):
        id_chunks: list[str] = []
        id_exceptions: dict[str, str] = {}
        for index, entry_id in enumerate(entry_ids):
            if len(entry_id) == len(ENTRY_ID_PREFIX) + RUNTIME_ID_SUFFIX_LEN:
                id_chunks.append(entry_id[len(ENTRY_ID_PREFIX) :])
            else:
                id_chunks.append("0" * RUNTIME_ID_SUFFIX_LEN)
                id_exceptions[str(index)] = entry_id
        table_doc["ih"] = "".join(id_chunks)
        if id_exceptions:
            table_doc["ix"] = id_exceptions
    else:
        table_doc["i"] = entry_ids
    return table_doc, entry_index_by_id


def _compress_phrase_trie_node(node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "" in node:
        out[""] = node[""]
    for label in sorted(key for key in node if key):
        child = node[label]
        merged_label = label
        merged_child = child
        while "" not in merged_child and len(merged_child) == 1:
            (next_label, next_child), = merged_child.items()
            merged_label += next_label
            merged_child = next_child
        out[merged_label] = _compress_phrase_trie_node(merged_child)
    return out


def _serialize_runtime_phrase_trie(
    entries: list[LexiconEntry],
    *,
    entry_index_by_id: dict[str, int],
) -> dict[str, Any]:
    root: dict[str, Any] = {"children": {}}
    for entry in entries:
        if entry.level not in {"phrase", "sentence"}:
            continue
        if is_sentence_manual_override(entry):
            continue
        if entry.context or entry.status != "active":
            continue
        node = root
        for ch in entry.src:
            node = node["children"].setdefault(ch, {"children": {}})
        node.setdefault("entry_indexes", []).append(entry_index_by_id[entry.entry_id])

    def normalize_node(raw_node: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if "entry_indexes" in raw_node:
            out[""] = raw_node["entry_indexes"]
        for ch in sorted(raw_node.get("children", {}).keys()):
            out[ch] = normalize_node(raw_node["children"][ch])
        return out

    return _compress_phrase_trie_node(normalize_node(root))


def _serialize_runtime_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        MANIFEST_SHORT_KEYS.get(key, key): value
        for key, value in manifest.items()
    }


def _inflate_runtime_manifest(manifest_doc: dict[str, Any]) -> dict[str, Any]:
    if not manifest_doc:
        return {}
    if "version" in manifest_doc:
        return manifest_doc
    return {
        MANIFEST_LONG_KEYS.get(key, key): value
        for key, value in manifest_doc.items()
    }


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
            "pattern": r"[ \t\u3000\xA0]+",
            "replacement": " ",
            "priority": 90,
            "note": "只壓縮空白/Tab/全形空白/NBSP（保留換行）",
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
            "replacement": "是按怎",
            "priority": 75,
            "note": "華語殘留修正",
        },
        {
            "pass_name": "fluency",
            "type": "literal",
            "pattern": "為什麼",
            "replacement": "是按怎",
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


def _collect_protected_terms(
    core_entries: list[LexiconEntry],
    allowlist_items: set[str],
    identity_passthrough_terms: set[str] | None = None,
) -> tuple[list[str], set[str]]:
    protected_terms = {item for item in allowlist_items if len(item) > 1}
    if identity_passthrough_terms:
        protected_terms.update(term for term in identity_passthrough_terms if len(term) > 1)
    core_identity_entry_ids: set[str] = set()
    for entry in core_entries:
        if entry.status != "active":
            continue
        if entry.level not in {"phrase", "sentence"}:
            continue
        if len(entry.src) <= 1:
            continue
        if entry.src != entry.tgt:
            continue
        core_identity_entry_ids.add(entry.entry_id)
        protected_terms.add(entry.src)
    return sorted(protected_terms, key=lambda item: (-len(item), item)), core_identity_entry_ids


def _collect_identity_passthrough_entries(
    entries: list[LexiconEntry],
) -> tuple[set[str], set[str]]:
    terms: set[str] = set()
    entry_ids: set[str] = set()
    for entry in entries:
        if entry.status != "active":
            continue
        if entry.tier == "blocked":
            continue
        if entry.level not in {"phrase", "sentence"}:
            continue
        if entry.context:
            continue
        if not entry.src or entry.src != entry.tgt:
            continue
        entry_ids.add(entry.entry_id)
        if len(entry.src) > 1:
            terms.add(entry.src)
    return terms, entry_ids


def _collect_residual_core_terms(
    residual_terms: list[str],
    active_entries: list[LexiconEntry],
    active_rules: list[RuleEntry],
) -> list[str]:
    if not residual_terms:
        return []

    direct_lexicon_terms = {
        entry.src
        for entry in active_entries
        if entry.src and entry.src != entry.tgt
    }
    direct_literal_rule_terms = {
        rule.pattern
        for rule in active_rules
        if rule.type == "literal" and rule.pattern and rule.pattern != rule.replacement
    }
    overlap = {
        term
        for term in residual_terms
        if term in direct_lexicon_terms or term in direct_literal_rule_terms
    }
    return sorted(overlap)


def migrate_legacy_data(data_dir: Path = DATA_DIR) -> dict[str, int]:
    phrase_path = data_dir / "phrase_lexicon.json"
    char_path = data_dir / "char_lexicon.json"
    if not phrase_path.exists() or not char_path.exists():
        raise FileNotFoundError("找不到 legacy 詞典檔（phrase_lexicon.json / char_lexicon.json）")

    allowlist = _load_allowlist(data_dir / CHAR_ALLOWLIST_FILE)
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

                            f"[{pass_name}] rule {earlier.rule_id} ({earlier.pattern!r} -> {earlier.replacement!r}) "
                            f"可能遮蔽 {later.rule_id} ({later.pattern!r} -> {later.replacement!r})"

                    )
    return warnings


def detect_regex_hazards(rules: list[RuleEntry]) -> list[str]:
    warnings: list[str] = []
    for rule in rules:
        if not rule.enabled or rule.type != "regex" or not rule.pattern:
            continue
        pattern = rule.pattern
        if REGEX_DOT_GREEDY_RE.search(pattern):
            warnings.append(f"[{rule.pass_name}] rule {rule.rule_id} 使用 dot-greedy，可能造成過度捕獲或回溯放大。")
        if REGEX_UNBOUNDED_NEG_CLASS_RE.search(pattern):
            warnings.append(f"[{rule.pass_name}] rule {rule.rule_id} 使用未設上限的 neg-charclass +，建議改成 bounded quantifier。")
        if pattern.count("(?!") >= 8:
            warnings.append(f"[{rule.pass_name}] rule {rule.rule_id} 含大量連鎖 negative lookahead，長文掃描成本偏高。")
        if rule.pass_name != "normalization" and r"\s+" in pattern and rule.replacement == "":
            warnings.append(f"[{rule.pass_name}] rule {rule.rule_id} 直接清除 \\s+，可能造成英數與中文黏連。")
    return warnings


def detect_pipeline_conflicts(rules: list[RuleEntry]) -> list[str]:
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
            if not earlier.pattern:
                continue
            for later in ordered[i + 1 :]:
                if not later.pattern:
                    continue
                if (
                    earlier.type == "literal"
                    and later.type == "literal"
                    and len(earlier.pattern) < len(later.pattern)
                    and earlier.pattern in later.pattern
                    and earlier.replacement != later.replacement
                ):
                    warnings.append(

                            f"[{pass_name}] rule {earlier.rule_id}（短詞 {earlier.pattern!r}）"
                            f"可能先命中，導致長詞規則 {later.rule_id}（{later.pattern!r}）失效。"

                    )
                if (
                    earlier.replacement
                    and earlier.replacement == later.pattern
                    and earlier.replacement != later.replacement
                ):
                    warnings.append(

                            f"[{pass_name}] rule {earlier.rule_id} 的 replacement 會觸發"
                            f" {later.rule_id}，可能形成連鎖改寫：{earlier.replacement!r}。"

                    )
    return warnings


def _validate_unique_entry_ids(entries: list[LexiconEntry]) -> None:
    entries_by_id: dict[str, list[LexiconEntry]] = {}
    for entry in entries:
        entries_by_id.setdefault(entry.entry_id, []).append(entry)

    duplicates = {
        entry_id: grouped_entries
        for entry_id, grouped_entries in entries_by_id.items()
        if len(grouped_entries) > 1
    }
    if not duplicates:
        return

    details: list[str] = []
    for entry_id, grouped_entries in sorted(duplicates.items())[:10]:
        sources = ", ".join(repr(entry.src) for entry in grouped_entries[:3])
        details.append(f"{entry_id}: {sources}")
    raise ValueError(
        "詞典存在重複 entry_id；每筆詞條必須使用唯一 ID: "
        + "; ".join(details)
    )


def _compile_runtime_artifacts_unlocked(
    data_dir: Path = DATA_DIR,
    fail_on_mask: bool = False,
    *,
    output_data_dir: Path | None = None,
) -> dict[str, Any]:
    source_digest_before = _source_digest(data_dir)
    lexicon_path = data_dir / "lexicon_entries.jsonl"
    rule_path = data_dir / "rule_entries.jsonl"

    if not lexicon_path.exists():
        raise FileNotFoundError("找不到 data/lexicon_entries.jsonl")
    if not rule_path.exists():
        raise FileNotFoundError("找不到 data/rule_entries.jsonl")

    source_rows = load_jsonl(lexicon_path)
    _validate_lexicon_rows(source_rows)
    rule_rows = load_jsonl(rule_path)
    _validate_rule_rows(rule_rows)

    source_entries = [LexiconEntry.from_dict(row) for row in source_rows]
    allowlist_items = _load_allowlist(data_dir / CHAR_ALLOWLIST_FILE)
    core_entries = _load_core_lexicon_entries(data_dir)
    identity_passthrough_terms, identity_passthrough_entry_ids = _collect_identity_passthrough_entries(source_entries)
    protected_terms, core_identity_entry_ids = _collect_protected_terms(
        core_entries,
        allowlist_items,
        identity_passthrough_terms=identity_passthrough_terms,
    )
    entries = source_entries + core_entries
    _validate_unique_entry_ids(entries)
    _validate_entry_target_conflicts(entries)
    rules = _expand_rules_with_tokens([RuleEntry.from_dict(row) for row in rule_rows])

    runtime_entries: list[LexiconEntry] = []
    runtime_excluded: dict[str, str] = {}
    for entry in entries:
        if entry.entry_id in core_identity_entry_ids:
            runtime_excluded[entry.entry_id] = RUNTIME_FILTER_CORE_IDENTITY_PROTECTED
            continue
        if entry.entry_id in identity_passthrough_entry_ids and not is_sentence_manual_override(entry):
            runtime_excluded[entry.entry_id] = RUNTIME_FILTER_IDENTITY_PASSTHROUGH_MASKED
            continue
        reason = runtime_exclusion_reason(entry)
        if reason:
            runtime_excluded[entry.entry_id] = reason
            continue
        runtime_entries.append(entry)

    active_entries = [entry for entry in runtime_entries if entry.status == "active"]
    active_rules = [rule for rule in rules if rule.enabled]
    residual_core_terms = _collect_residual_core_terms(
        RESIDUAL_MANDARIN_TERMS,
        active_entries,
        active_rules,
    )

    ordered_active_entries = sorted(active_entries, key=_entry_sort_key)
    entry_table_doc, entry_index_by_id = _serialize_runtime_entry_table(ordered_active_entries)

    sentence_override_map: dict[str, list[str]] = {}
    contextual_override_ids: list[str] = []
    char_map: dict[str, list[str]] = {}

    for entry in ordered_active_entries:
        if is_sentence_manual_override(entry):
            sentence_override_map.setdefault(entry.src, []).append(entry.entry_id)
        if is_trusted_manual_entry(entry) and entry.context:
            contextual_override_ids.append(entry.entry_id)
        if entry.level == "char" and not entry.context:
            char_map.setdefault(entry.src, []).append(entry.entry_id)

    phrase_trie = _serialize_runtime_phrase_trie(
        ordered_active_entries,
        entry_index_by_id=entry_index_by_id,
    )

    sorted_rules = sorted(
        active_rules,
        key=lambda r: (PASS_INDEX.get(r.pass_name, 999), -r.priority, r.rule_id),
    )

    mask_warnings = detect_masked_rules(sorted_rules)
    regex_hazards = detect_regex_hazards(sorted_rules)
    pipeline_conflicts = detect_pipeline_conflicts(sorted_rules)
    if fail_on_mask and (mask_warnings or regex_hazards or pipeline_conflicts):
        sections: list[str] = []
        if mask_warnings:
            sections.append("rule masking:\n" + "\n".join(mask_warnings))
        if regex_hazards:
            sections.append("regex hazards:\n" + "\n".join(regex_hazards))
        if pipeline_conflicts:
            sections.append("pipeline conflicts:\n" + "\n".join(pipeline_conflicts))
        raise ValueError("偵測到規則風險：\n" + "\n\n".join(sections))

    target_data_dir = output_data_dir or data_dir
    artifacts_dir = target_data_dir / ARTIFACT_DIR_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rules_by_pass: list[list[list[Any]]] = [[] for _ in PASS_ORDER]
    for rule in sorted_rules:
        rules_by_pass[PASS_INDEX.get(rule.pass_name, 0)].append(_serialize_runtime_rule(rule))

    rule_plan_doc: dict[str, Any] = {
        "v": 6,
        "r": rules_by_pass,
        "rt": RESIDUAL_MANDARIN_TERMS,
        "rc": residual_core_terms,
        "pt": "\n".join(protected_terms),
    }
    if mask_warnings:
        rule_plan_doc["mw"] = mask_warnings
    if regex_hazards:
        rule_plan_doc["rh"] = regex_hazards
    if pipeline_conflicts:
        rule_plan_doc["pc"] = pipeline_conflicts

    artifact_documents: dict[str, Any] = {
        "entry_table.json": entry_table_doc,
        "phrase_trie.json": {"v": 5, "t": phrase_trie},
        "char_map.json": {
            "v": 2,
            "m": _serialize_runtime_char_map(char_map, entry_index_by_id=entry_index_by_id),
        },
        "rule_plan.json": rule_plan_doc,
        "override_index.json": {
            "v": 3,
            "s": _serialize_runtime_sentence_overrides(
                sentence_override_map,
                entry_index_by_id=entry_index_by_id,
            ),
            "c": [
                entry_index_by_id[entry_id]
                for entry_id in contextual_override_ids
                if entry_id in entry_index_by_id
            ],
        },
    }
    artifact_texts = {
        name: json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        for name, document in artifact_documents.items()
    }
    artifact_hashes = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in artifact_texts.items()
    }

    source_digest_after = _source_digest(data_dir)
    if source_digest_after != source_digest_before:
        raise RuntimeError("source data 在 artifact 建置期間發生變更；請重試")

    manifest = {
        "version": MANIFEST_VERSION,
        "compiler_version": COMPILER_VERSION,
        "source_digest": source_digest_after,
        "artifact_hashes": artifact_hashes,
        "entry_count": len(entries),
        "runtime_entry_count": len(runtime_entries),
        "runtime_excluded_entry_count": len(runtime_excluded),
        "runtime_excluded_reasons": dict(Counter(runtime_excluded.values())),
        "core_entry_count": len(core_entries),
        "active_entry_count": len(active_entries),
        "rule_count": len(rules),
        "active_rule_count": len(active_rules),
        "mask_warning_count": len(mask_warnings),
        "regex_hazard_count": len(regex_hazards),
        "pipeline_conflict_count": len(pipeline_conflicts),
        "protected_term_count": len(protected_terms),
        "residual_core_term_count": len(residual_core_terms),
        "lexicon_stage": LEXICON_STAGE,
        "core_identity_protected_entry_count": len(core_identity_entry_ids),
        "identity_passthrough_protected_entry_count": len(identity_passthrough_entry_ids),
    }

    # Data files are replaced first and the checksummed manifest is the commit
    # marker. Readers reject/retry any mixed generation instead of silently
    # loading a partially rebuilt runtime.
    for name, text in artifact_texts.items():
        atomic_write_text(artifacts_dir / name, text, mode=0o644)
    _write_json(
        artifacts_dir / "manifest.json",
        _serialize_runtime_manifest(manifest),
        indent=None,
    )

    return manifest


def _artifact_build_lock_path(target_data_dir: Path) -> Path:
    return target_data_dir / ".artifact-build.lock"


def compile_runtime_artifacts(
    data_dir: Path = DATA_DIR,
    fail_on_mask: bool = False,
    *,
    output_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Compile one checksummed runtime generation under a process lock."""

    target_data_dir = output_data_dir or data_dir
    with exclusive_file_lock(_artifact_build_lock_path(target_data_dir)):
        return _compile_runtime_artifacts_unlocked(
            data_dir=data_dir,
            output_data_dir=target_data_dir,
            fail_on_mask=fail_on_mask,
        )


def ensure_runtime_ready(
    data_dir: Path = DATA_DIR,
    fail_on_mask: bool = False,
    *,
    output_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Prepare development artifacts using locked, content-hash validation."""

    target_data_dir = output_data_dir or data_dir
    manifest_path = target_data_dir / ARTIFACT_DIR_NAME / "manifest.json"

    def current_manifest() -> dict[str, Any]:
        if not manifest_path.exists():
            return {}
        try:
            document = _read_json(manifest_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(document, dict):
            return {}
        return _inflate_runtime_manifest(document)

    def artifact_hashes_match(candidate: dict[str, Any]) -> bool:
        hashes = candidate.get("artifact_hashes")
        if not isinstance(hashes, dict):
            return False
        for name in ARTIFACT_FILES:
            expected_hash = hashes.get(name)
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                return False
            try:
                payload = (target_data_dir / ARTIFACT_DIR_NAME / name).read_bytes()
            except OSError:
                return False
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                return False
        return True

    migrated_stats: dict[str, int] | None = None
    with exclusive_file_lock(_artifact_build_lock_path(target_data_dir)):
        lexicon_entries = data_dir / "lexicon_entries.jsonl"
        rule_entries = data_dir / "rule_entries.jsonl"
        if not lexicon_entries.exists():
            migrated_stats = migrate_legacy_data(data_dir)
        elif not rule_entries.exists():
            write_jsonl(rule_entries, [rule.to_dict() for rule in default_rule_entries()])

        expected_digest = _source_digest(data_dir)
        manifest = current_manifest()
        is_current = bool(manifest) and (
            manifest.get("version") == MANIFEST_VERSION
            and manifest.get("compiler_version") == COMPILER_VERSION
            and manifest.get("source_digest") == expected_digest
            and manifest.get("lexicon_stage") == LEXICON_STAGE
            and artifact_hashes_match(manifest)
        )
        if not is_current:
            manifest = _compile_runtime_artifacts_unlocked(
                data_dir=data_dir,
                output_data_dir=target_data_dir,
                fail_on_mask=fail_on_mask,
            )

    if migrated_stats:
        manifest["migration"] = migrated_stats
    return manifest
