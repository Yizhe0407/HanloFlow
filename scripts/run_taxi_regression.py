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


TAXI_REGRESSION_CASES: list[RegressionCase] = [
    # hailing — 叫車
    RegressionCase("hailing", "我要叫計程車。", "我欲叫計程車。"),
    RegressionCase("hailing", "請問有沒有叫車服務？", "借問敢有叫車服務？"),
    RegressionCase("hailing", "司機快來了嗎？", "司機緊來矣無？"),
    # destination — 目的地
    RegressionCase("destination", "請問到台北車站多少錢？", "借問到臺北車站偌濟錢？"),
    RegressionCase("destination", "我要去機場。", "我欲去機場。"),
    RegressionCase("destination", "請到這個地址。", "請到這个地址。"),
    # navigation — 行進指引
    RegressionCase("navigation", "請停在前面。", "請停佇頭前。"),
    RegressionCase("navigation", "在前面右轉。", "佇頭前正斡。"),
    RegressionCase("navigation", "在前面左轉。", "佇頭前倒斡。"),
    RegressionCase("navigation", "直走就到了。", "直走就到矣。"),
    RegressionCase("navigation", "就這裡下車。", "就遮落車。"),
    # payment — 付款
    RegressionCase("payment", "多少錢？", "偌濟錢？"),
    RegressionCase("payment", "不用找了。", "免找矣。"),
    RegressionCase("payment", "可以刷卡嗎？", "會當刷卡無？"),
    RegressionCase("payment", "可以開收據嗎？", "會當開收據無？"),
    # misc — 其他
    RegressionCase("misc", "請快一點。", "請較緊咧。"),
    RegressionCase("misc", "等我一下。", "等我一下仔。"),
    RegressionCase("misc", "到了叫我。", "到矣叫我。"),
    RegressionCase("misc", "打開後車廂。", "拍開後行李箱。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="計程車情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return TAXI_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in TAXI_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in TAXI_REGRESSION_CASES})
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
