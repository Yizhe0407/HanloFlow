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


BANK_REGRESSION_CASES: list[RegressionCase] = [
    # bank_account — 銀行開戶/帳戶
    RegressionCase("bank_account", "我要開戶。", "我欲開戶。"),
    RegressionCase("bank_account", "帳戶餘額多少？", "口座餘額偌濟？"),
    RegressionCase("bank_account", "我要申請信用卡。", "我欲申請信用卡。"),
    RegressionCase("bank_account", "請協助申請信用卡。", "請鬥相共申請信用卡。"),
    RegressionCase("bank_account", "我要補摺。", "我欲補摺。"),
    RegressionCase("bank_account", "我的卡片到期了。", "我的卡片到期矣。"),
    RegressionCase("bank_account", "開戶需要什麼資料？", "開戶需要啥物資料？"),
    RegressionCase("bank_account", "需要帶印章嗎？", "需要帶印仔無？"),
    RegressionCase("bank_account", "提款卡什麼時候可以拿？", "提款卡啥物時陣會當提？"),
    RegressionCase("bank_account", "可以開通網路銀行嗎？", "會當開通網路銀行無？"),
    RegressionCase("bank_account", "資料不齊可以補件嗎？", "資料無齊會當補件無？"),
    RegressionCase("bank_account", "我想改提款卡密碼。", "我想欲改提款卡密碼。"),
    RegressionCase("bank_account", "提款卡被鎖住了。", "提款卡鎖牢矣。"),
    RegressionCase("bank_account", "我想改通訊地址。", "我想欲改通訊地址。"),
    RegressionCase("bank_account", "可以幫我補辦提款卡嗎？", "會當替我補辦提款卡無？"),
    RegressionCase("bank_account", "我需要補辦提款卡。", "我欲補辦提款卡。"),
    RegressionCase("bank_account", "請協助補辦提款卡。", "請鬥相共補辦提款卡。"),
    RegressionCase("bank_account", "可以協助你們補辦提款卡嗎？", "會當鬥相共恁補辦提款卡無？"),
    RegressionCase("bank_account", "可以協助他們補辦提款卡嗎？", "會當鬥相共怹補辦提款卡無？"),
    RegressionCase("bank_account", "幫我補辦提款卡。", "替我補辦提款卡。"),
    RegressionCase("bank_account", "能不能請您幫我補辦提款卡？", "敢會當請你替我補辦提款卡？"),
    RegressionCase("bank_account", "能麻煩你幫我補辦提款卡嗎？", "會當麻煩你替我補辦提款卡無？"),
    RegressionCase("bank_account", "能否請您幫我補辦提款卡？", "敢會當請你替我補辦提款卡？"),
    RegressionCase("bank_account", "方不方便幫我補辦提款卡？", "敢方便替我補辦提款卡？"),
    RegressionCase("bank_account", "可不可以麻煩您幫我補辦提款卡？", "敢會當麻煩你替我補辦提款卡？"),
    RegressionCase("bank_account", "可否請您幫我補辦提款卡？", "敢會當請你替我補辦提款卡？"),
    RegressionCase("bank_account", "拜託你幫我補辦提款卡。", "拜託你替我補辦提款卡。"),
    RegressionCase("bank_account", "希望您幫我補辦提款卡。", "希望你替我補辦提款卡。"),
    RegressionCase("bank_account", "是否能麻煩您幫我補辦提款卡？", "敢會當麻煩你替我補辦提款卡？"),
    RegressionCase("bank_account", "可以幫我開存款證明嗎？", "會當替我開存款證明無？"),
    RegressionCase("bank_account", "可以幫你們開存款證明嗎？", "會當替恁開存款證明無？"),
    RegressionCase("bank_account", "可以幫他們開存款證明嗎？", "會當替怹開存款證明無？"),
    RegressionCase("bank_account", "可以幫大家開存款證明嗎？", "會當替逐家開存款證明無？"),
    RegressionCase("bank_account", "可以幫各位開存款證明嗎？", "會當替逐家開存款證明無？"),
    RegressionCase("bank_account", "可以幫客人開存款證明嗎？", "會當替人客開存款證明無？"),
    RegressionCase("bank_account", "可以幫病人開存款證明嗎？", "會當替病人開存款證明無？"),
    RegressionCase("bank_account", "可以幫家屬開存款證明嗎？", "會當替家屬開存款證明無？"),
    RegressionCase("bank_account", "我需要開存款證明。", "我欲開存款證明。"),
    RegressionCase("bank_account", "請協助開存款證明。", "請鬥相共開存款證明。"),
    RegressionCase("bank_account", "麻煩你幫我開存款證明。", "麻煩你替我開存款證明。"),
    RegressionCase("bank_account", "麻煩你幫我們開存款證明。", "麻煩你替阮開存款證明。"),
    RegressionCase("bank_account", "可不可以幫我開存款證明？", "敢會當替我開存款證明？"),
    RegressionCase("bank_account", "可不可以幫我們開存款證明？", "敢會當替阮開存款證明？"),
    RegressionCase("bank_account", "能否幫你們開存款證明？", "敢會當替恁開存款證明？"),
    RegressionCase("bank_account", "能否幫他們開存款證明？", "敢會當替怹開存款證明？"),
    RegressionCase("bank_account", "是否可以請您幫我開存款證明？", "敢會當請你替我開存款證明？"),
    RegressionCase("bank_account", "是否可以請您幫我們開存款證明？", "敢會當請你替阮開存款證明？"),
    RegressionCase("bank_account", "方不方便請您幫我開存款證明？", "敢方便請你替我開存款證明？"),
    RegressionCase("bank_account", "方不方便請您幫我們開存款證明？", "敢方便請你替阮開存款證明？"),
    RegressionCase("bank_account", "能否麻煩您幫我開存款證明？", "敢會當麻煩你替我開存款證明？"),
    RegressionCase("bank_account", "拜託您幫我開存款證明。", "拜託你替我開存款證明。"),
    RegressionCase("bank_account", "想請您幫我開存款證明。", "想欲請你替我開存款證明。"),
    RegressionCase("bank_account", "是否可以麻煩您幫我開存款證明？", "敢會當麻煩你替我開存款證明？"),
    RegressionCase("bank_account", "能否協助我開存款證明？", "敢會當鬥相共我開存款證明？"),
    RegressionCase("bank_account", "能否協助我們開存款證明？", "敢會當鬥相共阮開存款證明？"),
    RegressionCase("bank_account", "能否協助大家開存款證明？", "敢會當鬥相共逐家開存款證明？"),
    RegressionCase("bank_account", "能否協助各位開存款證明？", "敢會當鬥相共逐家開存款證明？"),
    RegressionCase("bank_account", "能否協助客人開存款證明？", "敢會當鬥相共人客開存款證明？"),
    RegressionCase("bank_account", "能否協助病人開存款證明？", "敢會當鬥相共病人開存款證明？"),
    RegressionCase("bank_account", "能否協助家屬開存款證明？", "敢會當鬥相共家屬開存款證明？"),
    RegressionCase("bank_account", "可否請您協助我開存款證明？", "敢會當請你鬥相共我開存款證明？"),
    RegressionCase("bank_account", "可否請您協助我們開存款證明？", "敢會當請你鬥相共阮開存款證明？"),
    RegressionCase("bank_account", "可不可以協助我開存款證明？", "敢會當鬥相共我開存款證明？"),
    RegressionCase("bank_account", "可不可以協助我們開存款證明？", "敢會當鬥相共阮開存款證明？"),
    RegressionCase("bank_account", "是否能麻煩您協助我開存款證明？", "敢會當麻煩你鬥相共我開存款證明？"),
    RegressionCase("bank_account", "是否能麻煩您協助我們開存款證明？", "敢會當麻煩你鬥相共阮開存款證明？"),
    RegressionCase("bank_account", "請協助申請存款證明。", "請鬥相共申請存款證明。"),
    RegressionCase("bank_account", "麻煩協助大家申請存款證明。", "麻煩鬥相共逐家申請存款證明。"),
    RegressionCase("bank_account", "麻煩協助各位申請存款證明。", "麻煩鬥相共逐家申請存款證明。"),
    RegressionCase("bank_account", "麻煩協助客人申請存款證明。", "麻煩鬥相共人客申請存款證明。"),
    RegressionCase("bank_account", "麻煩協助病人申請存款證明。", "麻煩鬥相共病人申請存款證明。"),
    RegressionCase("bank_account", "麻煩協助家屬申請存款證明。", "麻煩鬥相共家屬申請存款證明。"),
    RegressionCase("bank_account", "可以幫我改密碼嗎？", "會當替我改密碼無？"),
    RegressionCase("bank_account", "可以幫我重設密碼嗎？", "會當替我重設密碼無？"),
    RegressionCase("bank_account", "我需要重設密碼。", "我欲重設密碼。"),
    RegressionCase("bank_account", "請協助重設密碼。", "請鬥相共重設密碼。"),
    RegressionCase("bank_account", "麻煩幫我重設密碼。", "麻煩替我重設密碼。"),
    RegressionCase("bank_account", "請你幫我重設密碼。", "請你替我重設密碼。"),
    RegressionCase("bank_account", "能請你幫我重設密碼嗎？", "會當請你替我重設密碼無？"),
    RegressionCase("bank_account", "我想改聯絡電話。", "我想欲改聯絡電話。"),
    RegressionCase("bank_account", "可以幫我查帳戶狀態嗎？", "會當替我查口座狀態無？"),
    RegressionCase("bank_account", "可以幫我查金融卡狀態嗎？", "會當替我查金融卡狀態無？"),
    RegressionCase("bank_account", "可以幫我查信用卡額度嗎？", "會當替我查信用卡額度無？"),
    RegressionCase("bank_account", "可以幫我查分行地址嗎？", "會當替我查分行地址無？"),
    RegressionCase("bank_account", "可以幫我查信用卡點數嗎？", "會當替我查信用卡點數無？"),
    # bank_transaction — 存提匯
    RegressionCase("bank_transaction", "我要存款。", "我欲存款。"),
    RegressionCase("bank_transaction", "我要領錢。", "我欲領錢。"),
    RegressionCase("bank_transaction", "我要匯款。", "我欲匯款。"),
    RegressionCase("bank_transaction", "我要換外幣。", "我欲換外票。"),
    RegressionCase("bank_transaction", "我要查詢餘額。", "我欲查詢餘額。"),
    RegressionCase("bank_transaction", "我要轉帳。", "我欲轉帳。"),
    RegressionCase("bank_transaction", "可以跨行轉帳嗎？", "會當跨行轉帳無？"),
    RegressionCase("bank_transaction", "請幫我刷存摺。", "請替我刷存摺。"),
    RegressionCase("bank_transaction", "匯款需要什麼資料？", "匯款需要啥物資料？"),
    RegressionCase("bank_transaction", "提款有限額嗎？", "提款有限額無？"),
    RegressionCase("bank_transaction", "可以幫我查餘額嗎？", "會當替我查餘額無？"),
    RegressionCase("bank_transaction", "我要取消自動扣款。", "我欲取消自動扣款。"),
    RegressionCase("bank_transaction", "我想查帳戶明細。", "我想欲查口座明細。"),
    RegressionCase("bank_transaction", "可以幫我列印明細嗎？", "會當替我列印明細無？"),
    RegressionCase("bank_transaction", "我想查信用卡帳單。", "我想欲查信用卡費用明細。"),
    RegressionCase("bank_transaction", "可以幫我查匯率嗎？", "會當替我查匯率無？"),
    RegressionCase("bank_transaction", "可以幫我補寄帳單嗎？", "會當替我補寄帳單無？"),
    RegressionCase("bank_transaction", "我想改扣款帳戶。", "我想欲改扣款口座。"),
    RegressionCase("bank_transaction", "可以幫我查轉帳紀錄嗎？", "會當替我查轉帳紀錄無？"),
    RegressionCase("bank_transaction", "我想改匯款金額。", "我想欲改匯款金額。"),
    RegressionCase("bank_transaction", "我想改領錢金額。", "我想欲改領錢金額。"),
    RegressionCase("bank_transaction", "可以幫我查扣款紀錄嗎？", "會當替我查扣款紀錄無？"),
    RegressionCase("bank_transaction", "可以幫我查存摺紀錄嗎？", "會當替我查存摺紀錄無？"),
    RegressionCase("bank_transaction", "可以幫我查提款紀錄嗎？", "會當替我查提款紀錄無？"),
    RegressionCase("bank_transaction", "我想改匯款日期。", "我想欲改匯款日期。"),
    RegressionCase("bank_transaction", "可以幫我查帳單狀態嗎？", "會當替我查數單狀態無？"),
    RegressionCase("bank_transaction", "我想改扣款日期。", "我想欲改扣款日期。"),
    RegressionCase("bank_transaction", "可以幫我查信用卡帳單嗎？", "會當替我查信用卡數單無？"),
    # bank_service — 服務詢問
    RegressionCase("bank_service", "請問ATM在哪裡？", "借問ATM佇佗位？"),
    RegressionCase("bank_service", "請問ATM怎麼用？", "借問ATM按怎用？"),
    RegressionCase("bank_service", "請問手續費多少？", "借問手續費偌濟？"),
    RegressionCase("bank_service", "請問利率是多少？", "借問利率是偌濟？"),
    RegressionCase("bank_service", "需要預約嗎？", "需要預約無？"),
    RegressionCase("bank_service", "需要抽號碼牌嗎？", "需要抽號碼牌無？"),
    RegressionCase("bank_service", "還要等多久？", "猶愛等偌久？"),
    RegressionCase("bank_service", "請問開戶櫃檯在哪裡？", "借問開戶櫃檯佇佗位？"),
    RegressionCase("bank_service", "營業時間到幾點？", "營業時間到幾點？"),
    RegressionCase("bank_service", "手續費可以減免嗎？", "手續費會當減免無？"),
    RegressionCase("bank_service", "可以幫我查貸款利率嗎？", "會當替我查貸款利率無？"),
    RegressionCase("bank_service", "我需要申請貸款。", "我欲申請貸款。"),
    RegressionCase("bank_service", "請協助申請貸款。", "請鬥相共申請貸款。"),
    RegressionCase("bank_service", "可以幫我更新資料嗎？", "會當替我更新資料無？"),
    RegressionCase("bank_service", "我需要更新資料。", "我欲更新資料。"),
    RegressionCase("bank_service", "請協助更新資料。", "請鬥相共更新資料。"),
    RegressionCase("bank_service", "可以幫我查貸款進度嗎？", "會當替我查貸款進度無？"),
    RegressionCase("bank_service", "可以幫我查定存利率嗎？", "會當替我查定存利率無？"),
    # postal — 郵局寄件
    RegressionCase("postal", "我要寄掛號信。", "我欲寄掛號信。"),
    RegressionCase("postal", "我要寄包裹。", "我欲寄包裹。"),
    RegressionCase("postal", "這個多重？", "這个偌重？"),
    RegressionCase("postal", "要幾天到？", "要幾工到？"),
    RegressionCase("postal", "請問郵局在哪裡？", "借問郵局佇佗位？"),
    RegressionCase("postal", "我要買郵票。", "我欲買郵票。"),
    RegressionCase("postal", "我要寄國際包裹。", "我欲寄國際包裹。"),
    RegressionCase("postal", "我要寄平信。", "我欲寄平信。"),
    RegressionCase("postal", "郵資多少錢？", "郵資偌濟錢？"),
    RegressionCase("postal", "郵遞區號要寫嗎？", "郵遞區號愛寫無？"),
    RegressionCase("postal", "可以查追蹤號碼嗎？", "會當查追蹤號碼無？"),
    RegressionCase("postal", "這個包裹要多久會到？", "這个包裹偌久會到？"),
    RegressionCase("postal", "可以幫我查郵件狀態嗎？", "會當替我查郵件狀態無？"),
    RegressionCase("postal", "可以幫我寄到國外嗎？", "會當替我寄到國外無？"),
    RegressionCase("postal", "可以幫我查郵局營業時間嗎？", "會當替我查郵局營業時間無？"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="銀行/郵局情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return BANK_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in BANK_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in BANK_REGRESSION_CASES})
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
