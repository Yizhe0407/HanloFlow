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
    RegressionCase("browsing", "我想找黑色的。", "我想欲揣烏色的。"),
    RegressionCase("browsing", "有這個尺寸嗎？", "有這个尺寸無？"),
    RegressionCase("browsing", "可以推薦一下嗎？", "會當推薦一下無？"),
    RegressionCase("browsing", "店員可以幫我找尺寸嗎？", "店員會當替我揣尺寸無？"),
    # bargaining — 殺價
    RegressionCase("bargaining", "有沒有打折？", "敢有拍折？"),
    RegressionCase("bargaining", "太貴了。", "太貴矣。"),
    RegressionCase("bargaining", "可以便宜一點嗎？", "會當俗淡薄仔無？"),
    RegressionCase("bargaining", "算我便宜一點。", "算我俗淡薄仔。"),
    RegressionCase("bargaining", "可以再便宜一點嗎？", "會當閣較俗無？"),
    RegressionCase("bargaining", "買兩個可以算便宜一點嗎？", "買兩个會當算俗淡薄仔無？"),
    RegressionCase("bargaining", "有送贈品嗎？", "有送贈品無？"),
    RegressionCase("bargaining", "買多有折扣嗎？", "買較濟有拍折無？"),
    RegressionCase("bargaining", "可以打九折嗎？", "會當拍九折無？"),
    RegressionCase("bargaining", "這個有特價嗎？", "這个有特價無？"),
    # purchase — 購買
    RegressionCase("purchase", "我要買這個。", "我欲買這个。"),
    RegressionCase("purchase", "給我一個袋子。", "予我一个袋仔。"),
    RegressionCase("purchase", "謝謝，不用袋子。", "多謝，免袋仔。"),
    RegressionCase("purchase", "我想試穿這件。", "我想欲試穿這件。"),
    RegressionCase("purchase", "有大一點的尺寸嗎？", "有較大的尺寸無？"),
    RegressionCase("purchase", "這個還有貨嗎？", "這个閣有貨無？"),
    RegressionCase("purchase", "我要結帳。", "我欲結數。"),
    RegressionCase("purchase", "這件可以刷卡嗎？", "這件會當刷卡無？"),
    RegressionCase("purchase", "可以幫我查庫存嗎？", "會當替我查庫存無？"),
    RegressionCase("purchase", "可以幫我換尺寸嗎？", "會當替我換尺寸無？"),
    RegressionCase("purchase", "這件可以幫我留一下嗎？", "這件會當替我留一下無？"),
    RegressionCase("purchase", "可以幫我包起來嗎？", "會當替我包起來無？"),
    RegressionCase("purchase", "可以幫我換顏色嗎？", "會當替我換顏色無？"),
    RegressionCase("purchase", "可以幫我查商品規格嗎？", "會當替我查商品規格無？"),
    RegressionCase("purchase", "可以幫我查門市庫存嗎？", "會當替我查門市庫存無？"),
    RegressionCase("purchase", "可以幫我改商品顏色嗎？", "會當替我改商品顏色無？"),
    RegressionCase("purchase", "可以幫我查商品評價嗎？", "會當替我查商品評價無？"),
    RegressionCase("purchase", "可以幫我查庫存門市嗎？", "會當替我查庫存門市無？"),
    # payment — 付款
    RegressionCase("payment", "請問收現金嗎？", "借問收現錢無？"),
    RegressionCase("payment", "可以刷卡嗎？", "會當刷卡無？"),
    RegressionCase("payment", "可以用手機支付嗎？", "會當用手機付錢無？"),
    RegressionCase("payment", "我要找零。", "我欲找錢。"),
    RegressionCase("payment", "可以用電子支付嗎？", "會當用電子付錢無？"),
    RegressionCase("payment", "可以分期付款嗎？", "會當分期付錢無？"),
    RegressionCase("payment", "有會員折扣嗎？", "有會員拍折無？"),
    RegressionCase("payment", "可以開統一編號嗎？", "會當開統一編號無？"),
    RegressionCase("payment", "發票可以重開嗎？", "發票會當重開無？"),
    RegressionCase("payment", "我想改付款方式。", "我想欲改付款方式。"),
    RegressionCase("payment", "這個可以幫我退刷嗎？", "這个會當替我退刷無？"),
    RegressionCase("payment", "可以幫我查發票號碼嗎？", "會當替我查發票號碼無？"),
    RegressionCase("payment", "我想改發票載具。", "我想欲改發票載具。"),
    RegressionCase("payment", "可以幫我查會員點數嗎？", "會當替我查會員點數無？"),
    RegressionCase("payment", "可以幫我查優惠券嗎？", "會當替我查優惠券無？"),
    RegressionCase("payment", "我想改付款卡片。", "我想欲改付款卡片。"),
    RegressionCase("payment", "可以幫我查折扣碼嗎？", "會當替我查折扣碼無？"),
    RegressionCase("payment", "可以幫我查付款狀態嗎？", "會當替我查付款狀態無？"),
    RegressionCase("payment", "我想改付款日期。", "我想欲改付款日期。"),
    RegressionCase("payment", "我想改刷卡日期。", "我想欲改刷卡日期。"),
    RegressionCase("payment", "我想改發票日期。", "我想欲改發票日期。"),
    # comparative — 比較詞組
    RegressionCase("comparative", "有沒有大一點的？", "敢有較大的？"),
    RegressionCase("comparative", "有小一點的嗎？", "有較細的無？"),
    RegressionCase("comparative", "可以給我多一點嗎？", "會當予我加一寡無？"),
    RegressionCase("comparative", "有更便宜的嗎？", "有閣較俗的無？"),
    RegressionCase("comparative", "這個比較耐用嗎？", "這个較耐用無？"),
    RegressionCase("comparative", "有不同大小嗎？", "敢有大細無仝無？"),
    RegressionCase("comparative", "同款有別的顏色嗎？", "同款有別款顏色無？"),
    RegressionCase("comparative", "哪一個比較划算？", "佗一个較合算？"),
    # after_sales — 售後
    RegressionCase("after_sales", "我要退貨。", "我欲退貨。"),
    RegressionCase("after_sales", "我想退貨。", "我想欲退貨。"),
    RegressionCase("after_sales", "我需要退貨。", "我欲退貨。"),
    RegressionCase("after_sales", "可以換貨嗎？", "會當換貨無？"),
    RegressionCase("after_sales", "我需要取消訂單。", "我欲取消訂單。"),
    RegressionCase("after_sales", "我想取消訂單。", "我想欲取消訂單。"),
    RegressionCase("after_sales", "可以退款嗎？", "會當退錢無？"),
    RegressionCase("after_sales", "可以幫我退費嗎？", "會當幫我退錢無？"),
    RegressionCase("after_sales", "我還沒收到包裹。", "我猶未收著包裹。"),
    RegressionCase("after_sales", "我想查退款進度。", "我想欲查退錢進度。"),
    RegressionCase("after_sales", "可以換同款不同尺寸嗎？", "會當換同款無仝尺寸無？"),
    RegressionCase("after_sales", "這個有保固嗎？", "這个有保固無？"),
    RegressionCase("after_sales", "物流延遲了。", "物流延遲矣。"),
    RegressionCase("after_sales", "拆封了還可以換貨嗎？", "拆封矣猶會當換貨無？"),
    RegressionCase("after_sales", "可以只退一部分嗎？", "會當干焦退一部分無？"),
    RegressionCase("after_sales", "退款要等幾天？", "退錢愛等幾工？"),
    RegressionCase("after_sales", "商品少了一個配件。", "商品欠一个配件。"),
    RegressionCase("after_sales", "我想改送貨時間。", "我想欲改送貨時間。"),
    RegressionCase("after_sales", "物流一直沒有更新。", "物流攏無更新。"),
    RegressionCase("after_sales", "可以幫我查出貨進度嗎？", "會當替我查出貨進度無？"),
    RegressionCase("after_sales", "可以幫我查訂單嗎？", "會當替我查訂單無？"),
    RegressionCase("after_sales", "可以幫我查包裹嗎？", "會當替我查包裹無？"),
    RegressionCase("after_sales", "我想改收件地址。", "我想欲改收件地址。"),
    RegressionCase("after_sales", "可以幫我改送貨地址嗎？", "會當替我改送貨地址無？"),
    RegressionCase("after_sales", "我想改收貨時間。", "我想欲改收貨時間。"),
    RegressionCase("after_sales", "可以幫我查配送狀態嗎？", "會當替我查配送狀態無？"),
    RegressionCase("after_sales", "可以幫我查會員資料嗎？", "會當替我查會員資料無？"),
    RegressionCase("after_sales", "我想改取貨門市。", "我想欲改取貨門市。"),
    RegressionCase("after_sales", "可以幫我取消出貨嗎？", "會當替我取消出貨無？"),
    RegressionCase("after_sales", "可以幫我查保固期限嗎？", "會當替我查保固期限無？"),
    RegressionCase("after_sales", "可以幫我改退貨方式嗎？", "會當替我改退貨方式無？"),
    RegressionCase("after_sales", "我想改配送時間。", "我想欲改配送時間。"),
    RegressionCase("after_sales", "可以幫我申請退貨嗎？", "會當替我申請退貨無？"),
    RegressionCase("after_sales", "我想改取貨時間。", "我想欲改取貨時間。"),
    RegressionCase("after_sales", "可以幫我查退貨進度嗎？", "會當替我查退貨進度無？"),
    RegressionCase("after_sales", "可以幫我改取貨門市嗎？", "會當替我改取貨門市無？"),
    RegressionCase("after_sales", "可以幫我查退款明細嗎？", "會當替我查退錢明細無？"),
    RegressionCase("after_sales", "可以幫我查換貨進度嗎？", "會當替我查換貨進度無？"),
    RegressionCase("after_sales", "我想改退貨日期。", "我想欲改退貨日期。"),
    RegressionCase("after_sales", "可以幫我查取貨狀態嗎？", "會當替我查取貨狀態無？"),
    RegressionCase("after_sales", "可以幫我查退貨狀態嗎？", "會當替我查退貨狀態無？"),
    RegressionCase("after_sales", "可以幫我查訂單狀態嗎？", "會當替我查訂單狀態無？"),
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
