from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from converter import TaigiConverter


@dataclass(frozen=True)
class RegressionCase:
    category: str
    source: str
    expected: str


WORKPLACE_REGRESSION_CASES: list[RegressionCase] = [
    # meeting
    RegressionCase("meeting", "我今天要開會。", "我今仔日欲開會。"),
    RegressionCase("meeting", "下午要開會。", "下晝欲開會。"),
    RegressionCase("meeting", "會議改到明天。", "會議改做明仔載。"),
    RegressionCase("meeting", "明天早上再討論。", "明仔早起閣討論。"),
    # office location / progressive
    RegressionCase("office_location", "主管在會議室。", "主管佇會議室。"),
    RegressionCase("office_location", "我們在會議室開會。", "咱佇會議室開會。"),
    RegressionCase("office_location", "資料放在桌上。", "資料囥佇桌頂。"),
    RegressionCase("progressive", "同事在等你。", "同事佇咧等你。"),
    RegressionCase("progressive", "主管在看報告。", "主管佇咧看報告。"),
    RegressionCase("progressive", "我正在寫報告。", "我佇咧寫報告。"),
    # workflow
    RegressionCase("workflow", "請你再確認一次。", "請你閣確認一擺。"),
    RegressionCase("workflow", "這份文件要簽名。", "這份文件愛簽名。"),
    RegressionCase("workflow", "報告還沒寫完。", "報告猶未寫完。"),
    RegressionCase("workflow", "請把檔案寄給我。", "請共檔案寄予我。"),
    # leave / availability
    RegressionCase("leave_availability", "我今天請假。", "我今仔日請假。"),
    RegressionCase("leave_availability", "我想請半天假。", "我想欲請半日假。"),
    RegressionCase("leave_availability", "你有空嗎？", "你有閒無？"),
    RegressionCase("leave_availability", "我現在不方便。", "我這馬無方便。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="辦公/會議情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return WORKPLACE_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in WORKPLACE_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in WORKPLACE_REGRESSION_CASES})
    if args.list_categories:
        for category in categories:
            print(category)
        return 0

    cases = _selected_cases(args.category)
    if not cases:
        print("no cases selected")
        return 1

    converter = TaigiConverter()
    latencies_ms: list[float] = []
    category_counts = Counter(case.category for case in cases)

    print({"rounds": args.rounds, "case_count": len(cases), "categories": dict(sorted(category_counts.items()))})

    for round_idx in range(1, args.rounds + 1):
        failures: list[tuple[int, RegressionCase, str]] = []
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            output = converter.convert(case.source)
            latencies_ms.append((time.perf_counter() - started) * 1000)
            if output != case.expected:
                failures.append((index, case, output))
                if args.fail_fast:
                    break
            elif args.show_pass:
                print(f"PASS round={round_idx} idx={index} category={case.category} src={case.source}")

        print({"round": round_idx, "failed": len(failures)})
        if failures:
            for index, case, output in failures[:10]:
                print(f"FAIL idx={index} category={case.category}")
                print(f"  src: {case.source}")
                print(f"  exp: {case.expected}")
                print(f"  out: {output}")
            return 1

    latencies_ms.sort()
    p95_index = max(int(len(latencies_ms) * 0.95) - 1, 0)
    print(
        {
            "status": "ok",
            "rounds": args.rounds,
            "case_count": len(cases),
            "total_checks": len(latencies_ms),
            "mean_ms": round(mean(latencies_ms), 4),
            "p95_ms": round(latencies_ms[p95_index], 4),
            "max_ms": round(max(latencies_ms), 4),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
