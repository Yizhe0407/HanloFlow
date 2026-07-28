from __future__ import annotations

import json
from pathlib import Path

from taigi_converter.artifact_compiler import compile_runtime_artifacts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def valid_entry(
    *,
    entry_id: str = "lx_test00000001",
    src: str = "測試詞",
    tgt: str = "試驗詞",
    level: str = "phrase",
    tier: str = "manual",
    priority: int = 100,
    score: float = 1.0,
    status: str = "active",
    trust: str = "human",
) -> dict:
    return {
        "entry_id": entry_id,
        "src": src,
        "tgt": tgt,
        "level": level,
        "tier": tier,
        "priority": priority,
        "context": None,
        "score": score,
        "status": status,
        "source": "test",
        "trust": trust,
        "updated_by": "test",
        "updated_at": "2026-07-28T00:00:00+08:00",
    }


def make_source_data(
    root: Path,
    *,
    entries: list[dict] | None = None,
    rules: list[dict] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "lexicon_entries.jsonl", entries or [valid_entry()])
    write_jsonl(root / "rule_entries.jsonl", rules or [])
    (root / "core_lexicon.json").write_text("[]\n", encoding="utf-8")
    (root / "char_verified_allowlist.txt").write_text("", encoding="utf-8")
    return root


def build_minimal_runtime(source_dir: Path, output_data_dir: Path) -> dict:
    return compile_runtime_artifacts(source_dir, output_data_dir=output_data_dir)
