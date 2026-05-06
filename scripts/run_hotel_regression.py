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


HOTEL_REGRESSION_CASES: list[RegressionCase] = [
    # reservation — 訂房
    RegressionCase("reservation", "我要訂房。", "我欲訂房。"),
    RegressionCase("reservation", "請問有空房嗎？", "借問有閒房無？"),
    RegressionCase("reservation", "一晚多少錢？", "一晚偌濟錢？"),
    RegressionCase("reservation", "我要住兩晚。", "我欲住兩晚。"),
    RegressionCase("reservation", "我要住單人房。", "我欲住單人房。"),
    # check_in — 入住
    RegressionCase("check_in", "幾點可以入住？", "幾點會當入住？"),
    RegressionCase("check_in", "可以提早入住嗎？", "會當較早入住無？"),
    RegressionCase("check_in", "行李可以先寄放嗎？", "行李會當先寄放無？"),
    RegressionCase("check_in", "請問有停車場嗎？", "借問有停車場無？"),
    # check_out — 退房
    RegressionCase("check_out", "幾點要退房？", "幾點愛退房？"),
    RegressionCase("check_out", "我要退房。", "我欲退房。"),
    RegressionCase("check_out", "可以延後退房嗎？", "會當較晏退房無？"),
    RegressionCase("check_out", "退房後可以寄放行李嗎？", "退房後會當寄放行李無？"),
    RegressionCase("check_out", "請幫我開收據。", "請共我開收據。"),
    RegressionCase("check_out", "房卡要交回櫃檯嗎？", "房卡愛交轉去櫃檯無？"),
    # amenities — 設施服務
    RegressionCase("amenities", "請問有早餐嗎？", "借問有早頓無？"),
    RegressionCase("amenities", "早餐在哪裡？", "早頓佇佗位？"),
    RegressionCase("amenities", "請問有游泳池嗎？", "借問有泅水池無？"),
    RegressionCase("amenities", "請問有吹風機嗎？", "借問有吹風機無？"),
    RegressionCase("amenities", "請問有洗衣服務嗎？", "借問有洗衫服務無？"),
    # issues — 問題反映
    RegressionCase("issues", "房間太吵了。", "房間太吵矣。"),
    RegressionCase("issues", "冷氣壞了。", "冷氣歹去矣。"),
    RegressionCase("issues", "熱水沒有了。", "熱水無矣。"),
    RegressionCase("issues", "房卡打不開房門。", "房卡開袂開房門。"),
    RegressionCase("issues", "房間的網路不能用。", "房間的網路袂當用。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="飯店住宿情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return HOTEL_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in HOTEL_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in HOTEL_REGRESSION_CASES})
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
