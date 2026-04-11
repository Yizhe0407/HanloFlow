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


CONVERSATION_REGRESSION_CASES: list[RegressionCase] = [
    # greetings / farewells
    RegressionCase("greetings", "收到。", "收著。"),
    RegressionCase("greetings", "了解。", "知影。"),
    RegressionCase("greetings", "好久不見。", "好久無見。"),
    RegressionCase("greetings", "掰掰。", "再會。"),
    RegressionCase("greetings", "我回來了。", "我轉來矣。"),
    RegressionCase("greetings", "先走了，再見。", "行先矣，再會。"),
    # status check / location
    RegressionCase("status_check", "到哪了？", "到佗位矣？"),
    RegressionCase("status_check", "剛到家。", "拄到厝。"),
    RegressionCase("status_check", "出發了沒？", "出發矣未？"),
    RegressionCase("status_check", "你到了嗎？", "你到矣無？"),
    RegressionCase("status_check", "下班了沒？", "下班未？"),
    RegressionCase("status_check", "你今天幾點下班？", "你今仔日幾點下班？"),
    # daily chat
    RegressionCase("daily_chat", "你吃飯了嗎？", "你食飯矣無？"),
    RegressionCase("daily_chat", "你吃飽了嗎？", "你食飽未？"),
    RegressionCase("daily_chat", "你最近怎麼樣？", "你最近啥款？"),
    RegressionCase("daily_chat", "好久不見，你最近怎麼樣？", "好久無見，你最近啥款？"),
    RegressionCase("daily_chat", "我現在有點忙。", "我這馬有淡薄仔忙。"),
    RegressionCase("daily_chat", "我有點累了，先去休息一下。", "我有淡薄仔累矣，先去歇睏一下。"),
    # common responses
    RegressionCase("daily_response", "沒事。", "無代誌。"),
    RegressionCase("daily_response", "辛苦了。", "辛苦你矣。"),
    RegressionCase("daily_response", "你辛苦了。", "你辛苦矣。"),
    RegressionCase("daily_response", "麻煩你了。", "麻煩你。"),
    RegressionCase("daily_response", "好，我知道了。", "好，我知影矣。"),
    RegressionCase("daily_response", "沒問題。", "無問題。"),
    # scheduling / plans
    RegressionCase("schedule_plans", "等一下打給你。", "等陣仔閣敲予你。"),
    RegressionCase("schedule_plans", "改天再說。", "改工閣講。"),
    RegressionCase("schedule_plans", "早點休息。", "較早歇睏。"),
    RegressionCase("schedule_plans", "要轉車嗎？", "愛轉車無？"),
    RegressionCase("schedule_plans", "我等你在門口。", "我等你佇門跤口。"),
    RegressionCase("schedule_plans", "我現在在等車。", "我這馬咧等車。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="日常會話情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return CONVERSATION_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in CONVERSATION_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in CONVERSATION_REGRESSION_CASES})
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
