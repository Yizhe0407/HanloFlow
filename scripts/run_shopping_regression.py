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


SHOPPING_REGRESSION_CASES: list[RegressionCase] = [
    # browsing — 選購
    RegressionCase("browsing", "這個多少錢？", "這个偌濟錢？"),
    RegressionCase("browsing", "有沒有其他顏色？", "敢有其他顏色？"),
    RegressionCase("browsing", "可以試穿嗎？", "會當試穿無？"),
    RegressionCase("browsing", "試穿間在哪裡？", "試穿間佇佗位？"),
    # bargaining — 殺價
    RegressionCase("bargaining", "有沒有打折？", "敢有拍折？"),
    RegressionCase("bargaining", "太貴了。", "太貴矣。"),
    RegressionCase("bargaining", "可以便宜一點嗎？", "會當俗淡薄仔無？"),
    RegressionCase("bargaining", "算我便宜一點。", "算我俗淡薄仔。"),
    # purchase — 購買
    RegressionCase("purchase", "我要買這個。", "我欲買這个。"),
    RegressionCase("purchase", "給我一個袋子。", "予我一个袋仔。"),
    RegressionCase("purchase", "謝謝，不用袋子。", "多謝，免袋仔。"),
    # payment — 付款
    RegressionCase("payment", "請問收現金嗎？", "借問收現錢無？"),
    RegressionCase("payment", "可以刷卡嗎？", "會當刷卡無？"),
    RegressionCase("payment", "可以用手機支付嗎？", "會當用手機付錢無？"),
    RegressionCase("payment", "我要找零。", "我欲找錢。"),
    # comparative — 比較詞組
    RegressionCase("comparative", "有沒有大一點的？", "敢有較大的？"),
    RegressionCase("comparative", "有小一點的嗎？", "有較細的無？"),
    RegressionCase("comparative", "可以給我多一點嗎？", "會當予我加一寡無？"),
    # after_sales — 售後
    RegressionCase("after_sales", "我要退貨。", "我欲退貨。"),
    RegressionCase("after_sales", "可以換貨嗎？", "會當換貨無？"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="購物情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return SHOPPING_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in SHOPPING_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in SHOPPING_REGRESSION_CASES})
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
