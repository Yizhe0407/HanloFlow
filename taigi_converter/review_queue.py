from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifact_compiler import load_jsonl, validate_lexicon_rows
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    durable_unlink,
    exclusive_file_lock,
    fsync_directory,
)
from .lexicon_policy import VALID_TRUSTS, normalize_trust

ALLOWED_DECISIONS = {"add_override", "disable_base_entry", "reject"}
DEFAULT_REVIEW_OWNER = "reviewer"
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
IMMUTABLE_GENERATION_DIRECTORY_MODE = 0o500
IMMUTABLE_GENERATION_FILE_MODE = 0o400
_REVIEW_METADATA_FIELDS = {
    "coalesced_review_ids",
    "created_at",
    "decision",
    "decision_reason",
    "fingerprint",
    "last_seen_at",
    "occurrence_count",
    "resolved_at",
    "resolved_by",
    "review_id",
    "status",
}
_STATE_DIR_NAME = ".review_state"
_GENERATIONS_DIR_NAME = "generations"
_POINTER_NAME = "current.json"
_MANIFEST_NAME = "manifest.json"
_MIRROR_DIRTY_NAME = "mirror_dirty.json"
_SNAPSHOT_VERSION = 1
_TRANSACTION_FILE_NAMES = {
    "lexicon_entries.jsonl",
    "review_audit.jsonl",
    "review_queue.jsonl",
}


class ReviewCommitDurabilityError(OSError):
    """The pointer switched, but durable persistence could not be confirmed."""

    committed = True


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _chmod_existing(path: Path, mode: int) -> None:
    """Harden an existing mutable state file without an exists/chmod race."""
    try:
        os.chmod(path, mode)
    except FileNotFoundError:
        # Another process may atomically replace or remove transient state between
        # discovery and chmod. Writers create replacements with the requested mode.
        pass


def _prepare_state_dir(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    state_path = data_dir / _STATE_DIR_NAME
    generations_path = state_path / _GENERATIONS_DIR_NAME
    state_path.mkdir(exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    generations_path.mkdir(exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    if os.name != "posix":
        return
    for directory in (data_dir, state_path, generations_path):
        os.chmod(directory, PRIVATE_DIRECTORY_MODE)
    for path in (
        data_dir / ".review_queue.lock",
        data_dir / ".review_transaction.json",
        data_dir / "review_audit.jsonl",
        data_dir / "review_queue.jsonl",
        data_dir / "lexicon_entries.jsonl",
        state_path / _POINTER_NAME,
        state_path / _MIRROR_DIRTY_NAME,
    ):
        _chmod_existing(path, PRIVATE_FILE_MODE)


def _new_review_id() -> str:
    return f"rq_{uuid.uuid4().hex}"


def _review_content(item: dict[str, Any]) -> dict[str, Any]:
    if any(not isinstance(key, str) for key in item):
        raise ValueError("review item 的欄位名稱必須是字串")
    return {key: value for key, value in item.items() if key not in _REVIEW_METADATA_FIELDS}


def _review_fingerprint(item: dict[str, Any]) -> str:
    raw = json.dumps(
        _review_content(item),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"rfp_{hashlib.sha256(raw).hexdigest()}"


def _new_entry_id(src: str, tgt: str, level: str, tier: str, source: str) -> str:
    raw = f"{src}|{tgt}|{level}|{tier}|{source}".encode()
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"lx_{digest}"


def _queue_path(data_dir: Path) -> Path:
    return data_dir / "review_queue.jsonl"


def _lexicon_path(data_dir: Path) -> Path:
    return data_dir / "lexicon_entries.jsonl"


def _audit_path(data_dir: Path) -> Path:
    return data_dir / "review_audit.jsonl"


def _transaction_path(data_dir: Path) -> Path:
    return data_dir / ".review_transaction.json"


def _lock_path(data_dir: Path) -> Path:
    return data_dir / ".review_queue.lock"


def _state_path(data_dir: Path) -> Path:
    return data_dir / _STATE_DIR_NAME


def _generations_path(data_dir: Path) -> Path:
    return _state_path(data_dir) / _GENERATIONS_DIR_NAME


def _pointer_path(data_dir: Path) -> Path:
    return _state_path(data_dir) / _POINTER_NAME


def _mirror_dirty_path(data_dir: Path) -> Path:
    return _state_path(data_dir) / _MIRROR_DIRTY_NAME


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_jsonl(payload: bytes, *, path: Path) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"review snapshot file 不是合法 UTF-8: {path}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"review snapshot file JSON 格式錯誤: {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"review snapshot file 每列必須是 object: {path}:{line_number}")
        rows.append(row)
    return rows


def _validate_transaction_files(files: dict[str, list[dict[str, Any]]]) -> None:
    if set(files) != _TRANSACTION_FILE_NAMES:
        raise ValueError("review transaction 必須原子更新 queue、lexicon 與 audit")
    for name, rows in files.items():
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"review transaction 的 {name} 必須是 object list")


def _read_pointer(data_dir: Path) -> tuple[str, str]:
    path = _pointer_path(data_dir)
    try:
        pointer = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"review snapshot pointer 損毀: {path}") from exc
    if not isinstance(pointer, dict) or pointer.get("version") != _SNAPSHOT_VERSION:
        raise RuntimeError(f"review snapshot pointer 格式錯誤: {path}")
    generation = pointer.get("generation")
    manifest_digest = pointer.get("manifest_sha256")
    suffix = generation.removeprefix("gen_") if isinstance(generation, str) else ""
    if (
        not isinstance(generation, str)
        or not generation.startswith("gen_")
        or len(suffix) != 32
        or any(char not in "0123456789abcdef" for char in suffix)
        or not isinstance(manifest_digest, str)
        or len(manifest_digest) != 64
        or any(char not in "0123456789abcdef" for char in manifest_digest)
    ):
        raise RuntimeError(f"review snapshot pointer 格式錯誤: {path}")
    return generation, manifest_digest


def _load_snapshot(data_dir: Path) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    # Resolve the atomic pointer exactly once. Old generations are retained, so
    # concurrent commits cannot invalidate the selected immutable snapshot.
    generation, expected_manifest_digest = _read_pointer(data_dir)
    generation_path = _generations_path(data_dir) / generation
    manifest_path = generation_path / _MANIFEST_NAME
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"review snapshot manifest 損毀: {manifest_path}") from exc
    if _digest(manifest_payload) != expected_manifest_digest:
        raise RuntimeError(f"review snapshot manifest checksum 不符: {manifest_path}")
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != _SNAPSHOT_VERSION
        or manifest.get("generation") != generation
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != _TRANSACTION_FILE_NAMES
    ):
        raise RuntimeError(f"review snapshot manifest 格式錯誤: {manifest_path}")
    files: dict[str, list[dict[str, Any]]] = {}
    for name in _TRANSACTION_FILE_NAMES:
        metadata = manifest["files"][name]
        path = generation_path / name
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"review snapshot generation 不完整: {path}") from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("size") != len(payload)
            or metadata.get("sha256") != _digest(payload)
        ):
            raise RuntimeError(f"review snapshot file checksum 不符: {path}")
        files[name] = _decode_jsonl(payload, path=path)
    return generation, files


def _sync_flat_mirrors(data_dir: Path, encoded: dict[str, bytes]) -> None:
    """Update non-authoritative legacy mirrors after the pointer commit."""
    for name in sorted(_TRANSACTION_FILE_NAMES):
        path = data_dir / name
        payload = encoded[name]
        if name != _queue_path(Path()).name and not payload and not path.exists():
            continue
        atomic_write_text(path, payload.decode("utf-8"), mode=PRIVATE_FILE_MODE)


def _repair_flat_mirrors_unlocked(data_dir: Path, files: dict[str, list[dict[str, Any]]]) -> None:
    if not _mirror_dirty_path(data_dir).exists():
        return
    encoded = {name: _jsonl_bytes(rows) for name, rows in files.items()}
    try:
        _sync_flat_mirrors(data_dir, encoded)
    except OSError:
        return
    durable_unlink(_mirror_dirty_path(data_dir), missing_ok=True)


def _publish_pointer(data_dir: Path, generation: str, manifest_digest: str) -> None:
    atomic_write_json(
        _pointer_path(data_dir),
        {
            "version": _SNAPSHOT_VERSION,
            "generation": generation,
            "manifest_sha256": manifest_digest,
        },
        indent=None,
        mode=PRIVATE_FILE_MODE,
    )


def _commit_transaction_unlocked(
    data_dir: Path,
    files: dict[str, list[dict[str, Any]]],
) -> None:
    """Publish an immutable generation; pointer replacement is the commit point."""
    _prepare_state_dir(data_dir)
    _validate_transaction_files(files)
    encoded = {name: _jsonl_bytes(rows) for name, rows in files.items()}
    generation = f"gen_{uuid.uuid4().hex}"
    generations_path = _generations_path(data_dir)
    staging = generations_path / f".staging_{generation}"
    final = generations_path / generation
    staging.mkdir(mode=PRIVATE_DIRECTORY_MODE)
    published = False
    try:
        metadata: dict[str, dict[str, Any]] = {}
        for name, payload in encoded.items():
            path = staging / name
            path.write_bytes(payload)
            if os.name == "posix":
                os.chmod(path, IMMUTABLE_GENERATION_FILE_MODE)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            metadata[name] = {"sha256": _digest(payload), "size": len(payload)}
        try:
            parent, _ = _read_pointer(data_dir)
        except FileNotFoundError:
            parent = None
        manifest = {
            "version": _SNAPSHOT_VERSION,
            "generation": generation,
            "parent_generation": parent,
            "created_at": _now_iso(),
            "files": metadata,
        }
        manifest_path = staging / _MANIFEST_NAME
        atomic_write_json(manifest_path, manifest, indent=None, mode=PRIVATE_FILE_MODE)
        if os.name == "posix":
            os.chmod(manifest_path, IMMUTABLE_GENERATION_FILE_MODE)
            with manifest_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.chmod(staging, IMMUTABLE_GENERATION_DIRECTORY_MODE)
        fsync_directory(staging)
        os.replace(staging, final)
        fsync_directory(generations_path)
        manifest_digest = _digest((final / _MANIFEST_NAME).read_bytes())
        atomic_write_json(
            _mirror_dirty_path(data_dir),
            {"version": _SNAPSHOT_VERSION, "generation": generation},
            indent=None,
            mode=PRIVATE_FILE_MODE,
        )
        try:
            _publish_pointer(data_dir, generation, manifest_digest)
        except OSError as exc:
            try:
                committed = _read_pointer(data_dir) == (generation, manifest_digest)
            except (OSError, RuntimeError):
                committed = False
            if committed:
                published = True
                raise ReviewCommitDurabilityError("review snapshot pointer 已切換，但無法確認 durable fsync") from exc
            raise
        published = True
    finally:
        if not published:
            if os.name == "posix" and staging.exists():
                os.chmod(staging, PRIVATE_DIRECTORY_MODE)
            shutil.rmtree(staging, ignore_errors=True)
            fsync_directory(generations_path)
    # Mirrors are explicitly non-authoritative. Once the pointer is published,
    # mirror I/O must not turn a committed batch into an ambiguous exception.
    try:
        _sync_flat_mirrors(data_dir, encoded)
    except OSError:
        return
    durable_unlink(_mirror_dirty_path(data_dir), missing_ok=True)


def _legacy_transaction_files(data_dir: Path) -> dict[str, list[dict[str, Any]]] | None:
    journal_path = _transaction_path(data_dir)
    if not journal_path.exists():
        return None
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"review transaction journal 損毀: {journal_path}") from exc
    files = journal.get("files") if isinstance(journal, dict) else None
    if (
        not isinstance(journal, dict)
        or journal.get("version") != 1
        or not isinstance(files, dict)
        or set(files) != _TRANSACTION_FILE_NAMES
    ):
        raise RuntimeError(f"review transaction journal 格式錯誤: {journal_path}")
    if any(not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows) for rows in files.values()):
        raise RuntimeError(f"review transaction journal 內容不合法: {journal_path}")
    return files


def _recover_transaction_unlocked(data_dir: Path) -> None:
    files = _legacy_transaction_files(data_dir)
    if files is None:
        return
    original_files = files
    normalized, _ = _normalize_review_rows(original_files[_queue_path(Path()).name])
    files = copy.deepcopy(original_files)
    files[_queue_path(Path()).name] = normalized
    if files != original_files:
        atomic_write_json(
            _transaction_path(data_dir),
            {"version": 1, "files": files},
            mode=PRIVATE_FILE_MODE,
        )
    try:
        _, current_files = _load_snapshot(data_dir)
    except FileNotFoundError:
        _commit_transaction_unlocked(data_dir, files)
    else:
        if files != current_files:
            raise RuntimeError("legacy review transaction 與目前 generation 衝突；拒絕覆蓋較新的 snapshot")
    durable_unlink(_transaction_path(data_dir), missing_ok=True)


def _occurrence_count(row: dict[str, Any]) -> int:
    value = row.get("occurrence_count", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def _review_id_aliases(row: dict[str, Any]) -> list[str]:
    aliases = row.get("coalesced_review_ids")
    if not isinstance(aliases, list):
        return []
    return [alias for alias in aliases if isinstance(alias, str) and alias]


def _merge_review_ids(
    target: dict[str, Any],
    source: dict[str, Any],
    reserved_ids: set[str],
) -> None:
    aliases = [alias for alias in _review_id_aliases(target) if alias != target.get("review_id")]
    source_ids = [source.get("review_id"), *_review_id_aliases(source)]
    for review_id in source_ids:
        if (
            isinstance(review_id, str)
            and review_id
            and review_id != target.get("review_id")
            and review_id not in aliases
            and review_id not in reserved_ids
        ):
            aliases.append(review_id)
            reserved_ids.add(review_id)
    if aliases:
        target["coalesced_review_ids"] = aliases


def _normalize_review_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    pending_by_fingerprint: dict[str, dict[str, Any]] = {}
    for original in rows:
        row = dict(original)
        fingerprint = _review_fingerprint(row)
        row["fingerprint"] = fingerprint
        row.setdefault("status", "pending")
        row["occurrence_count"] = _occurrence_count(row)
        aliases = _review_id_aliases(row)
        if aliases:
            row["coalesced_review_ids"] = aliases
        else:
            row.pop("coalesced_review_ids", None)
        row.setdefault("created_at", _now_iso())
        row.setdefault("last_seen_at", row["created_at"])
        if row["status"] == "pending" and fingerprint in pending_by_fingerprint:
            target = pending_by_fingerprint[fingerprint]
            target["occurrence_count"] = _occurrence_count(target) + row["occurrence_count"]
            target["last_seen_at"] = max(str(target.get("last_seen_at", "")), str(row.get("last_seen_at", "")))
            _merge_review_ids(target, row, used_ids)
            continue
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or not review_id or review_id in used_ids:
            review_id = _new_review_id()
            while review_id in used_ids:
                review_id = _new_review_id()
            row["review_id"] = review_id
        used_ids.add(review_id)
        aliases = [alias for alias in _review_id_aliases(row) if alias != review_id and alias not in used_ids]
        if aliases:
            row["coalesced_review_ids"] = aliases
            used_ids.update(aliases)
        else:
            row.pop("coalesced_review_ids", None)
        normalized.append(row)
        if row["status"] == "pending":
            pending_by_fingerprint[fingerprint] = row
    changed = sum(1 for before, after in zip(rows, normalized, strict=False) if before != after)
    changed += abs(len(rows) - len(normalized))
    return normalized, changed


def _flat_files(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {name: load_jsonl(data_dir / name) if (data_dir / name).exists() else [] for name in _TRANSACTION_FILE_NAMES}


def _initialize_state_unlocked(data_dir: Path) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    _recover_transaction_unlocked(data_dir)
    try:
        generation, files = _load_snapshot(data_dir)
    except FileNotFoundError:
        files = _flat_files(data_dir)
        files[_queue_path(Path()).name], _ = _normalize_review_rows(files[_queue_path(Path()).name])
        _commit_transaction_unlocked(data_dir, files)
        generation, files = _load_snapshot(data_dir)
    _repair_flat_mirrors_unlocked(data_dir, files)
    return generation, files


def _ensure_review_ids_unlocked(data_dir: Path) -> int:
    _, files = _initialize_state_unlocked(data_dir)
    normalized, changed = _normalize_review_rows(files[_queue_path(Path()).name])
    if changed:
        files[_queue_path(Path()).name] = normalized
        _commit_transaction_unlocked(data_dir, files)
    return changed


def load_review_snapshot(data_dir: Path) -> dict[str, Any]:
    """Read queue, lexicon and audit from exactly one immutable generation."""
    _prepare_state_dir(data_dir)
    needs_locked_recovery = (
        _transaction_path(data_dir).exists()
        or _mirror_dirty_path(data_dir).exists()
        or not _pointer_path(data_dir).exists()
    )
    if needs_locked_recovery:
        with exclusive_file_lock(_lock_path(data_dir), mode=PRIVATE_FILE_MODE):
            generation, files = _initialize_state_unlocked(data_dir)
    else:
        generation, files = _load_snapshot(data_dir)
    return {"generation": generation, "files": files}


def append_review_item(data_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError("review item 必須是 object")
    _prepare_state_dir(data_dir)
    content = _review_content(item)
    fingerprint = _review_fingerprint(content)
    with exclusive_file_lock(_lock_path(data_dir), mode=PRIVATE_FILE_MODE):
        _, files = _initialize_state_unlocked(data_dir)
        queue_rows, _ = _normalize_review_rows(files[_queue_path(Path()).name])
        now = _now_iso()
        for row in queue_rows:
            if row.get("status") != "pending" or row.get("fingerprint") != fingerprint:
                continue
            row["occurrence_count"] = _occurrence_count(row) + 1
            row["last_seen_at"] = now
            files[_queue_path(Path()).name] = queue_rows
            _commit_transaction_unlocked(data_dir, files)
            return copy.deepcopy(row)
        used_ids = {row.get("review_id") for row in queue_rows}
        review_id = _new_review_id()
        while review_id in used_ids:
            review_id = _new_review_id()
        payload = {
            **content,
            "review_id": review_id,
            "fingerprint": fingerprint,
            "created_at": now,
            "last_seen_at": now,
            "occurrence_count": 1,
            "status": "pending",
        }
        queue_rows.append(payload)
        files[_queue_path(Path()).name] = queue_rows
        _commit_transaction_unlocked(data_dir, files)
        return copy.deepcopy(payload)


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
    _prepare_state_dir(data_dir)
    with exclusive_file_lock(_lock_path(data_dir), mode=PRIVATE_FILE_MODE):
        _recover_transaction_unlocked(data_dir)
        return _ensure_review_ids_unlocked(data_dir)


def load_review_queue(data_dir: Path) -> list[dict[str, Any]]:
    """Load the official queue through one atomic generation pointer."""
    snapshot = load_review_snapshot(data_dir)
    return snapshot["files"][_queue_path(Path()).name]


def _pending_review_sort_key(row: dict[str, Any]) -> tuple[float, float, int, str, str]:
    priority = row.get("priority", 0)
    if not isinstance(priority, int | float) or isinstance(priority, bool):
        priority = 0
    evidence = row.get("evidence", {})
    confidence = evidence.get("confidence_score", 1.0) if isinstance(evidence, dict) else 1.0
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        confidence = 1.0
    return (
        -float(priority),
        float(confidence),
        -_occurrence_count(row),
        str(row.get("created_at", "")),
        str(row.get("review_id", "")),
    )


def export_pending_reviews(
    data_dir: Path,
    output_path: Path,
    limit: int = 200,
) -> int:
    rows = load_review_queue(data_dir)
    pending = sorted(
        (row for row in rows if row.get("status", "pending") == "pending"),
        key=_pending_review_sort_key,
    )[:limit]

    atomic_write_jsonl(output_path, pending)
    return len(pending)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不允許非有限數值 {value}")


def _load_decisions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: decision JSON 不合法: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: decision 必須是 object")
            rows.append(row)
    return rows


def _validate_entry_ids(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        entry_id = row.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            continue
        previous = seen.get(entry_id)
        if previous is not None:
            errors.append(f"entry_id={entry_id!r} 同時出現在 lexicon line {previous} 與 {index}")
        else:
            seen[entry_id] = index
    if errors:
        raise ValueError("詞典 entry_id 衝突:\n" + "\n".join(errors[:50]))


def _validate_optional_entry_fields(decision: dict[str, Any]) -> None:
    probe = {
        "entry_id": "lx_review_preflight",
        "src": "字",
        "tgt": "字",
        "level": decision.get("level", "phrase"),
        "tier": decision.get("tier", "manual"),
        "priority": decision.get("priority", 100),
        "context": decision.get("context"),
        "score": decision.get("score", 1.0),
        "status": decision.get("status", "active"),
        "source": "review_queue",
        "trust": "human",
    }
    validate_lexicon_rows([probe])


def _validate_owner_and_reason(decision: dict[str, Any], default_owner: str) -> tuple[str, str]:
    actor = decision.get("owner", default_owner)
    reason = decision.get("reason", "")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("owner 必須是非空字串")
    if not isinstance(reason, str):
        raise ValueError("reason 必須是字串")
    return actor, reason


def _apply_add_override(
    lexicon_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    owner: str,
) -> dict[str, Any]:
    src = decision.get("src")
    tgt = decision.get("tgt")
    if not isinstance(src, str) or not src:
        raise ValueError("add_override 的 src 必須是非空字串")
    if not isinstance(tgt, str):
        raise ValueError("add_override 的 tgt 必須是字串")

    level = decision.get("level", "phrase")
    tier = decision.get("tier", "manual")
    priority = decision.get("priority", 100)
    context = decision.get("context")
    score = decision.get("score", 1.0)
    status = decision.get("status", "active")
    source = decision.get("source", "review_queue")
    if not isinstance(source, str) or not source:
        raise ValueError("source 必須是非空字串")
    explicit_trust = decision.get("trust")
    if explicit_trust is not None and explicit_trust not in VALID_TRUSTS:
        raise ValueError(f"trust={explicit_trust!r} 不合法")
    trust = normalize_trust(
        trust=explicit_trust,
        source=source,
        updated_by=owner,
        tier=tier if isinstance(tier, str) else None,
    )
    updated_at = _now_iso()

    requested_entry_id = decision.get("entry_id")
    if requested_entry_id is not None and (not isinstance(requested_entry_id, str) or not requested_entry_id):
        raise ValueError("entry_id 必須是非空字串")
    entry_id = requested_entry_id or _new_entry_id(src, tgt, str(level), str(tier), source)
    candidate = {
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
    validate_lexicon_rows([candidate])

    matches = [row for row in lexicon_rows if row.get("entry_id") == entry_id]
    if len(matches) > 1:
        raise ValueError(f"entry_id={entry_id!r} 在既有詞典中重複")
    if matches:
        existing = matches[0]
        identity_fields = ("src", "tgt", "level", "tier", "source")
        if any(existing.get(field) != candidate[field] for field in identity_fields):
            raise ValueError(f"entry_id={entry_id!r} 已對應不同詞條，拒絕覆寫")
        existing.update(candidate)
    else:
        lexicon_rows.append(candidate)

    return {"entry_id": entry_id, "op": "add_or_update_override"}


def _apply_disable_base_entry(
    lexicon_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    owner: str,
) -> dict[str, Any]:
    target_entry_id = decision.get("entry_id")
    target_src = decision.get("src")
    target_level = decision.get("level")
    if target_entry_id is not None and (not isinstance(target_entry_id, str) or not target_entry_id):
        raise ValueError("entry_id 必須是非空字串")
    if target_src is not None and (not isinstance(target_src, str) or not target_src):
        raise ValueError("src 必須是非空字串")
    if not target_entry_id and not target_src:
        raise ValueError("disable_base_entry 需要 entry_id 或 src")
    if target_level is not None and target_level not in {"sentence", "phrase", "char"}:
        raise ValueError(f"level={target_level!r} 不合法")

    disabled_count = 0
    for row in lexicon_rows:
        if row.get("tier") != "base" or row.get("status") == "disabled":
            continue
        id_match = bool(target_entry_id and row.get("entry_id") == target_entry_id)
        src_match = bool(target_src and row.get("src") == target_src)
        level_match = (not target_level) or row.get("level") == target_level
        if (id_match or src_match) and level_match:
            row["status"] = "disabled"
            row["updated_by"] = owner
            row["updated_at"] = _now_iso()
            disabled_count += 1

    if disabled_count == 0:
        raise ValueError("disable_base_entry 沒有命中任何 active base 詞條")
    return {"disabled_count": disabled_count, "op": "disable_base_entry"}


def _queue_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        review_ids = [row.get("review_id"), *_review_id_aliases(row)]
        for review_id in review_ids:
            if not isinstance(review_id, str) or not review_id:
                continue
            existing = index.get(review_id)
            if existing is not None and existing is not row:
                raise ValueError(f"review_id={review_id!r} 在 queue 中重複")
            index[review_id] = row
    return index


def _empty_summary(total_decisions: int, errors: list[str]) -> dict[str, Any]:
    return {
        "total_decisions": total_decisions,
        "applied": 0,
        "add_override": 0,
        "disable_base_entry": 0,
        "reject": 0,
        "errors": errors,
    }


def _apply_review_decisions_unlocked(
    data_dir: Path,
    decisions_path: Path,
    *,
    dry_run: bool = False,
    owner: str = DEFAULT_REVIEW_OWNER,
) -> dict[str, Any]:
    _, snapshot_files = _initialize_state_unlocked(data_dir)
    queue_rows = snapshot_files[_queue_path(Path()).name]
    lexicon_rows = snapshot_files[_lexicon_path(Path()).name]
    decisions = _load_decisions(decisions_path)
    staged_queue = copy.deepcopy(queue_rows)
    staged_lexicon = copy.deepcopy(lexicon_rows)

    try:
        validate_lexicon_rows(staged_lexicon)
        _validate_entry_ids(staged_lexicon)
        queue_index = _queue_index(staged_queue)
    except ValueError as exc:
        return _empty_summary(len(decisions), [f"preflight: {exc}"])

    action_counts = {"add_override": 0, "disable_base_entry": 0, "reject": 0}
    errors: list[str] = []
    audit_rows: list[dict[str, Any]] = []

    for idx, decision in enumerate(decisions, start=1):
        review_id = decision.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            errors.append(f"line {idx}: 缺少合法 review_id")
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

        candidate_lexicon = copy.deepcopy(staged_lexicon)
        try:
            actor, reason = _validate_owner_and_reason(decision, owner)
            _validate_optional_entry_fields(decision)
            if action == "add_override":
                info = _apply_add_override(candidate_lexicon, decision, actor)
            elif action == "disable_base_entry":
                info = _apply_disable_base_entry(candidate_lexicon, decision, actor)
            else:
                info = {"op": "reject"}
            validate_lexicon_rows(candidate_lexicon)
            _validate_entry_ids(candidate_lexicon)
        except (TypeError, ValueError) as exc:
            errors.append(f"line {idx}: review_id={review_id} preflight 失敗: {exc}")
            continue

        staged_lexicon = candidate_lexicon
        queue_item["status"] = "resolved"
        queue_item["resolved_at"] = _now_iso()
        queue_item["resolved_by"] = actor
        queue_item["decision"] = action
        queue_item["decision_reason"] = reason
        canonical_review_id = queue_item["review_id"]
        audit_row = {
            "review_id": canonical_review_id,
            "action": action,
            "owner": actor,
            "reason": reason,
            "applied_at": _now_iso(),
            "result": info,
        }
        if review_id != canonical_review_id:
            audit_row["requested_review_id"] = review_id
        audit_rows.append(audit_row)
        action_counts[action] += 1

    if errors:
        return _empty_summary(len(decisions), errors)

    applied = len(decisions)
    summary = {
        "total_decisions": len(decisions),
        "applied": applied,
        "add_override": action_counts["add_override"],
        "disable_base_entry": action_counts["disable_base_entry"],
        "reject": action_counts["reject"],
        "errors": [],
    }
    if dry_run or not decisions:
        return summary

    existing_audit_rows = snapshot_files[_audit_path(Path()).name]
    _commit_transaction_unlocked(
        data_dir,
        {
            _queue_path(data_dir).name: staged_queue,
            _lexicon_path(data_dir).name: staged_lexicon,
            _audit_path(data_dir).name: existing_audit_rows + audit_rows,
        },
    )
    return summary


def apply_review_decisions(
    data_dir: Path,
    decisions_path: Path,
    *,
    dry_run: bool = False,
    owner: str = DEFAULT_REVIEW_OWNER,
) -> dict[str, Any]:
    """Preflight and atomically apply an all-or-nothing decision batch."""

    _prepare_state_dir(data_dir)
    with exclusive_file_lock(_lock_path(data_dir), mode=PRIVATE_FILE_MODE):
        return _apply_review_decisions_unlocked(
            data_dir,
            decisions_path,
            dry_run=dry_run,
            owner=owner,
        )
