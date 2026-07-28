from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .converter import TaigiConverter
from .models import ConversionResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="華語 -> 台語漢字 轉換器")
    parser.add_argument("text", nargs="*", help="要轉換的文字（省略則進入互動模式）")
    parser.add_argument("--trace", action="store_true", help="輸出完整 trace（JSON）")
    parser.add_argument("--explain", action="store_true", help="用易讀模式輸出命中與規則")
    parser.add_argument("--enqueue-review", action="store_true", help="低信心結果寫入 review_queue")
    parser.add_argument("--owner", default="cli", help="review_queue owner 欄位")
    parser.add_argument(
        "--review-data-dir",
        type=Path,
        help="review queue 狀態目錄（預設使用使用者 state 目錄）",
    )
    parser.add_argument(
        "--preserve-spacing",
        action="store_true",
        help="保留原始空白排版（略過 normalization pass）",
    )
    return parser


def _default_review_data_dir() -> Path:
    override = os.environ.get("TAIGI_CONVERTER_STATE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "taigi-converter"
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home).expanduser() / "taigi-converter"
    return Path.home() / ".local" / "state" / "taigi-converter"


def _print_explain(result: ConversionResult) -> None:
    print("\n=== 輸出 ===")
    print(result.output)

    print("\n=== 詞條命中 ===")
    if not result.matches:
        print("(無)")
    else:
        for match in result.matches:
            print(
                f"- [{match.tier}/{match.level}] {match.src} -> {match.tgt} "
                f"({match.start}:{match.end}, id={match.entry_id})"
            )

    print("\n=== 規則命中 ===")
    if not result.rules_applied:
        print("(無)")
    else:
        for rule in result.rules_applied:
            print(
                f"- [{rule.pass_name}] {rule.pattern} -> {rule.replacement} "
                f"(hits={rule.hit_count}, id={rule.rule_id})"
            )

    print("\n=== 警告 ===")
    if not result.warnings:
        print("(無)")
    else:
        for warning in result.warnings:
            print(f"- {warning}")

    print(f"\nlatency_ms: {result.latency_ms:.3f}")


def _run_once(converter, text: str, *, trace: bool, explain: bool, profile: dict | None) -> None:
    wants_trace = trace or explain
    result = converter.convert(text, trace=wants_trace, profile=profile)

    if not wants_trace:
        print("\n=== 輸出 ===")
        print(result)
        return

    assert isinstance(result, ConversionResult)

    if explain:
        _print_explain(result)
    else:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def main() -> None:
    args = _build_parser().parse_args()
    review_data_dir = None
    if args.enqueue_review:
        review_data_dir = args.review_data_dir or _default_review_data_dir()
    converter = TaigiConverter(review_data_dir=review_data_dir)

    profile = None
    if args.enqueue_review:
        profile = {
            "enqueue_review": True,
            "owner": args.owner,
        }
    if args.preserve_spacing:
        profile = profile or {}
        profile["preserve_spacing"] = True

    if args.text:
        text = " ".join(args.text).strip()
        _run_once(
            converter,
            text,
            trace=args.trace,
            explain=args.explain,
            profile=profile,
        )
        return

    print("華語 -> 台語漢字 轉換器")
    print("輸入 exit 離開")

    while True:
        text = input("\n請輸入：").strip()
        if text.lower() in {"exit", "quit"}:
            break
        _run_once(
            converter,
            text,
            trace=args.trace,
            explain=args.explain,
            profile=profile,
        )


if __name__ == "__main__":
    main()
