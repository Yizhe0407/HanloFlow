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


RESTAURANT_REGRESSION_CASES: list[RegressionCase] = [
    # ordering — 點餐
    RegressionCase("ordering", "我要點餐。", "我欲點餐。"),
    RegressionCase("ordering", "請給我菜單。", "請予我菜單。"),
    RegressionCase("ordering", "請問今天有什麼特餐？", "借問今仔日有啥特餐？"),
    RegressionCase("ordering", "請問有什麼推薦的嗎？", "借問有啥推薦的無？"),
    RegressionCase("ordering", "我要點這個。", "我欲點這个。"),
    RegressionCase("ordering", "他在點菜。", "伊佇咧叫菜。"),
    RegressionCase("ordering", "我想加點一份薯條。", "我想欲加點一份薯條。"),
    RegressionCase("ordering", "可以換成套餐嗎？", "會當換做套餐無？"),
    RegressionCase("ordering", "這份可以做外帶嗎？", "這份會當做外帶無？"),
    RegressionCase("ordering", "飲料可以去冰嗎？", "飲料會當去冰無？"),
    RegressionCase("ordering", "可以先上飲料嗎？", "會當先送飲料來無？"),
    RegressionCase("ordering", "我想取消訂位。", "我想欲取消訂位。"),
    RegressionCase("ordering", "我想改訂位時間。", "我想欲改訂位時間。"),
    # spice / dietary — 口味偏好
    RegressionCase("spice_dietary", "不要太辣。", "莫太辣。"),
    RegressionCase("spice_dietary", "我不要加辣。", "我無愛加辣。"),
    RegressionCase("spice_dietary", "辣的還是不辣的？", "辣的猶是無辣的？"),
    RegressionCase("spice_dietary", "有沒有素食？", "敢有素食？"),
    RegressionCase("spice_dietary", "不要香菜。", "莫芫荽。"),
    RegressionCase("spice_dietary", "可以少鹽嗎？", "會當少鹽無？"),
    RegressionCase("spice_dietary", "我對花生過敏。", "我食塗豆會過敏。"),
    RegressionCase("spice_dietary", "我不能吃牛肉。", "我袂當食牛肉。"),
    # seating — 座位
    RegressionCase("seating", "請問還有位子嗎？", "借問閣有位子無？"),
    RegressionCase("seating", "兩位大人一位小孩。", "兩位大人一位囡仔。"),
    RegressionCase("seating", "請問要等多久？", "借問愛等偌久？"),
    RegressionCase("seating", "我有訂位。", "我有訂位。"),
    RegressionCase("seating", "可以幫我安排兒童椅嗎？", "會當幫我安排囡仔椅無？"),
    RegressionCase("seating", "可以坐靠窗的位置嗎？", "會當坐靠窗的位無？"),
    RegressionCase("seating", "可以坐靠窗嗎？", "會當坐靠窗無？"),
    RegressionCase("seating", "需要等位嗎？", "需要等位無？"),
    RegressionCase("seating", "可以併桌嗎？", "會當併桌無？"),
    RegressionCase("seating", "有四個人的位子嗎？", "有四个人的位子無？"),
    RegressionCase("seating", "可以換到裡面的位置嗎？", "會當換到內底的位無？"),
    # payment — 結帳
    RegressionCase("payment", "麻煩結帳。", "麻煩結數。"),
    RegressionCase("payment", "總共多少錢？", "總共偌濟錢？"),
    RegressionCase("payment", "這個多少錢？", "這个偌濟錢？"),
    RegressionCase("payment", "可以刷卡嗎？", "會當刷卡無？"),
    RegressionCase("payment", "可以分開結帳嗎？", "會當分開結數無？"),
    RegressionCase("payment", "我要用現金付。", "我欲付現錢。"),
    RegressionCase("payment", "可以開發票嗎？", "會當開發票無？"),
    RegressionCase("payment", "發票可以用載具嗎？", "發票會當用載具無？"),
    # service — 服務需求
    RegressionCase("service", "可以打包嗎？", "會當打包無？"),
    RegressionCase("service", "我要外帶。", "我欲外帶。"),
    RegressionCase("service", "可以幫我換盤子嗎？", "會當幫我換盤仔無？"),
    RegressionCase("service", "請問廁所在哪裡？", "借問便所佇佗位？"),
    RegressionCase("service", "可以加水嗎？", "會當加水無？"),
    RegressionCase("service", "可以換餐具嗎？", "會當換餐具無？"),
    RegressionCase("service", "我的餐還沒來。", "我的餐猶未來。"),
    RegressionCase("service", "可以加一張椅子嗎？", "會當加一張椅仔無？"),
    RegressionCase("service", "餐點可以快一點嗎？", "餐點會當較緊無？"),
    RegressionCase("service", "可以幫我加飯嗎？", "會當幫我添飯無？"),
    RegressionCase("service", "可以幫我拿餐具嗎？", "會當幫我提餐具來無？"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="餐廳點餐情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return RESTAURANT_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in RESTAURANT_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in RESTAURANT_REGRESSION_CASES})
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
