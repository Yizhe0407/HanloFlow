from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .converter import TaigiConverter
from .models import ConversionResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="華語 -> 台語漢字 轉換器")
    parser.add_argument("text", nargs="*", help="要轉換的文字（省略則讀取 stdin）")
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


def _print_explain(result: ConversionResult, *, stdout: TextIO) -> None:
    print("\n=== 輸出 ===", file=stdout)
    print(result.output, file=stdout)

    print("\n=== 詞條命中 ===", file=stdout)
    if not result.matches:
        print("(無)", file=stdout)
    else:
        for match in result.matches:
            print(
                f"- [{match.tier}/{match.level}] {match.src} -> {match.tgt} "
                f"({match.start}:{match.end}, id={match.entry_id})",
                file=stdout,
            )

    print("\n=== 規則命中 ===", file=stdout)
    if not result.rules_applied:
        print("(無)", file=stdout)
    else:
        for rule in result.rules_applied:
            print(
                f"- [{rule.pass_name}] {rule.pattern} -> {rule.replacement} (hits={rule.hit_count}, id={rule.rule_id})",
                file=stdout,
            )

    print("\n=== 警告 ===", file=stdout)
    if not result.warnings:
        print("(無)", file=stdout)
    else:
        for warning in result.warnings:
            print(f"- {warning}", file=stdout)

    print(f"\nlatency_ms: {result.latency_ms:.3f}", file=stdout)


def _run_once(
    converter: TaigiConverter,
    text: str,
    *,
    trace: bool,
    explain: bool,
    profile: dict | None,
    stdout: TextIO,
    pretty_json: bool,
) -> None:
    wants_trace = trace or explain
    result = converter.convert(text, trace=wants_trace, profile=profile)

    if not wants_trace:
        print(result, file=stdout)
        return

    assert isinstance(result, ConversionResult)

    if explain:
        _print_explain(result, stdout=stdout)
    else:
        indent = 2 if pretty_json else None
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=indent), file=stdout)


def _strip_record_ending(line: str) -> str:
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def _run_batch(
    converter: TaigiConverter,
    stdin: TextIO,
    stdout: TextIO,
    *,
    trace: bool,
    explain: bool,
    profile: dict | None,
) -> None:
    for line in stdin:
        _run_once(
            converter,
            _strip_record_ending(line),
            trace=trace,
            explain=explain,
            profile=profile,
            stdout=stdout,
            pretty_json=False,
        )


def _run_interactive(
    converter: TaigiConverter,
    stdin: TextIO,
    stdout: TextIO,
    *,
    trace: bool,
    explain: bool,
    profile: dict | None,
) -> None:
    print("華語 -> 台語漢字 轉換器", file=stdout)
    print("輸入 exit 離開", file=stdout)

    while True:
        print("\n請輸入：", end="", file=stdout, flush=True)
        line = stdin.readline()
        if line == "":
            # Ctrl-D/Ctrl-Z 是正常互動結束，不應產生 traceback 或非零狀態碼。
            print(file=stdout)
            return

        text = _strip_record_ending(line)
        if text.strip().casefold() in {"exit", "quit"}:
            return
        _run_once(
            converter,
            text,
            trace=trace,
            explain=explain,
            profile=profile,
            stdout=stdout,
            pretty_json=True,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout

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
        # 不在 CLI 層 strip；是否保留外側空白應由 preserve_spacing profile 決定。
        text = " ".join(args.text)
        _run_once(
            converter,
            text,
            trace=args.trace,
            explain=args.explain,
            profile=profile,
            stdout=output_stream,
            pretty_json=True,
        )
        return 0

    if input_stream.isatty():
        _run_interactive(
            converter,
            input_stream,
            output_stream,
            trace=args.trace,
            explain=args.explain,
            profile=profile,
        )
    else:
        _run_batch(
            converter,
            input_stream,
            output_stream,
            trace=args.trace,
            explain=args.explain,
            profile=profile,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
