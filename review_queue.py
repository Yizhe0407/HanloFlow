from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from artifact_compiler import load_jsonl, write_jsonl
from lexicon_policy import normalize_trust

ALLOWED_DECISIONS = {"add_override", "disable_base_entry", "reject"}
DEFAULT_REVIEW_OWNER = "reviewer"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_review_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"rq_{digest}"


def _new_entry_id(src: str, tgt: str, level: str, tier: str, source: str) -> str:
    raw = f"{src}|{tgt}|{level}|{tier}|{source}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"lx_{digest}"


def _queue_path(data_dir: Path) -> Path:
    return data_dir / "review_queue.jsonl"


def _lexicon_path(data_dir: Path) -> Path:
    return data_dir / "lexicon_entries.jsonl"


def _audit_path(data_dir: Path) -> Path:
    return data_dir / "review_audit.jsonl"


def append_review_item(data_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    queue_path = _queue_path(data_dir)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "review_id": item.get("review_id") or _new_review_id(item),
        "created_at": _now_iso(),
        "status": "pending",
        **item,
    }

    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")

    return payload


def import_unresolved_entries(
    data_dir: Path,
    unresolved_path: Path,
    owner: str = "migration",
    reason: str = "offline_unresolved",
) -> int:
    if not unresolved_path.exists():
        return 0

    imported = 0
    with unresolved_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            append_review_item(
                data_dir,
                {
                    "kind": "offline_unresolved",
                    "action": "add_override",
                    "owner": owner,
                    "reason": reason,
                    "evidence": row,
                },
            )
            imported += 1
    return imported


def ensure_review_ids(data_dir: Path) -> int:
    queue_path = _queue_path(data_dir)
    if not queue_path.exists():
        queue_path.touch(exist_ok=True)
        return 0

    rows = load_jsonl(queue_path)
    changed = 0
    for row in rows:
        if row.get("review_id"):
            continue
        row["review_id"] = _new_review_id(row)
        changed += 1

    if changed:
        write_jsonl(queue_path, rows)
    return changed


def load_review_queue(data_dir: Path) -> list[dict[str, Any]]:
    ensure_review_ids(data_dir)
    return load_jsonl(_queue_path(data_dir))


def export_pending_reviews(
    data_dir: Path,
    output_path: Path,
    limit: int = 200,
) -> int:
    rows = load_review_queue(data_dir)
    pending = [row for row in rows if row.get("status", "pending") == "pending"]
    pending = pending[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in pending:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return len(pending)


def _load_decisions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _apply_add_override(
    lexicon_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    owner: str,
) -> dict[str, Any]:
    src = decision.get("src")
    tgt = decision.get("tgt")
    if not src or tgt is None:
        raise ValueError("add_override 需要 src 與 tgt")

    level = decision.get("level", "phrase")
    tier = decision.get("tier", "manual")
    priority = int(decision.get("priority", 100))
    context = decision.get("context")
    score = float(decision.get("score", 1.0))
    status = decision.get("status", "active")
    source = decision.get("source", "review_queue")
    trust = normalize_trust(
        trust=decision.get("trust"),
        source=source,
        updated_by=owner,
        tier=tier,
    )
    updated_at = _now_iso()

    entry_id = decision.get("entry_id") or _new_entry_id(src, tgt, level, tier, source)

    replaced = False
    for row in lexicon_rows:
        if row.get("entry_id") != entry_id:
            continue
        row.update(
            {
                "src": src,
                "tgt": tgt,
                "level": level,
                "tier": tier,
                "priority": priority,
                "context": context,
                "score": score,
                "status": status,
                "source": source,
                "trust": trust,
                "updated_by": owner,
                "updated_at": updated_at,
            }
        )
        replaced = True
        break

    if not replaced:
        lexicon_rows.append(
            {
                "entry_id": entry_id,
                "src": src,
                "tgt": tgt,
                "level": level,
                "tier": tier,
                "priority": priority,
                "context": context,
                "score": score,
                "status": status,
                "source": source,
                "trust": trust,
                "updated_by": owner,
                "updated_at": updated_at,
            }
        )

    return {"entry_id": entry_id, "op": "add_or_update_override"}


def _apply_disable_base_entry(
    lexicon_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    owner: str,
) -> dict[str, Any]:
    target_entry_id = decision.get("entry_id")
    target_src = decision.get("src")
    target_level = decision.get("level")

    disabled_count = 0
    for row in lexicon_rows:
        if row.get("tier") != "base":
            continue
        if row.get("status") == "disabled":
            continue

        id_match = target_entry_id and row.get("entry_id") == target_entry_id
        src_match = target_src and row.get("src") == target_src
        level_match = (not target_level) or row.get("level") == target_level

        if not (id_match or src_match):
            continue
        if not level_match:
            continue

        row["status"] = "disabled"
        row["updated_by"] = owner
        row["updated_at"] = _now_iso()
        disabled_count += 1

    if disabled_count == 0:
        raise ValueError("disable_base_entry 沒有命中任何 active base 詞條")

    return {"disabled_count": disabled_count, "op": "disable_base_entry"}


def apply_review_decisions(
    data_dir: Path,
    decisions_path: Path,
    *,
    dry_run: bool = False,
    owner: str = DEFAULT_REVIEW_OWNER,
) -> dict[str, Any]:
    ensure_review_ids(data_dir)

    queue_rows = load_jsonl(_queue_path(data_dir))
    lexicon_rows = load_jsonl(_lexicon_path(data_dir))
    decisions = _load_decisions(decisions_path)

    queue_index = {row.get("review_id"): row for row in queue_rows if row.get("review_id")}

    applied = 0
    add_override = 0
    disable_base = 0
    rejected = 0
    errors: list[str] = []
    audit_rows: list[dict[str, Any]] = []

    for idx, decision in enumerate(decisions, start=1):
        review_id = decision.get("review_id")
        if not review_id:
            errors.append(f"line {idx}: 缺少 review_id")
            continue

        queue_item = queue_index.get(review_id)
        if not queue_item:
            errors.append(f"line {idx}: 找不到 review_id={review_id}")
            continue
        if queue_item.get("status") != "pending":
            errors.append(f"line {idx}: review_id={review_id} 已非 pending")
            continue

        action = decision.get("decision")
        if action not in ALLOWED_DECISIONS:
            errors.append(f"line {idx}: decision={action!r} 不合法")
            continue

        actor = decision.get("owner") or owner
        reason = decision.get("reason", "")

        try:
            if action == "add_override":
                info = _apply_add_override(lexicon_rows, decision, actor)
                add_override += 1
            elif action == "disable_base_entry":
                info = _apply_disable_base_entry(lexicon_rows, decision, actor)
                disable_base += 1
            else:
                info = {"op": "reject"}
                rejected += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"line {idx}: review_id={review_id} 套用失敗: {exc}")
            continue

        queue_item["status"] = "resolved"
        queue_item["resolved_at"] = _now_iso()
        queue_item["resolved_by"] = actor
        queue_item["decision"] = action
        queue_item["decision_reason"] = reason

        audit_rows.append(
            {
                "review_id": review_id,
                "action": action,
                "owner": actor,
                "reason": reason,
                "applied_at": _now_iso(),
                "result": info,
            }
        )
        applied += 1

    summary = {
        "total_decisions": len(decisions),
        "applied": applied,
        "add_override": add_override,
        "disable_base_entry": disable_base,
        "reject": rejected,
        "errors": errors,
    }

    if dry_run:
        return summary

    write_jsonl(_queue_path(data_dir), queue_rows)
    write_jsonl(_lexicon_path(data_dir), lexicon_rows)

    if audit_rows:
        audit_path = _audit_path(data_dir)
        with audit_path.open("a", encoding="utf-8") as f:
            for row in audit_rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

    return summary
