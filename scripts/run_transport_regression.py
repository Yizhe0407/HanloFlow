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


TRANSPORT_REGRESSION_CASES: list[RegressionCase] = [
    # shuttle / transfer
    RegressionCase("shuttle_transfer", "請問往火車站的接駁車在哪裡搭？", "借問往火車頭的接駁車佇佗位搭？"),
    RegressionCase("shuttle_transfer", "高鐵接駁車在對面站牌搭。", "高鐵接駁車佇對面站牌搭。"),
    RegressionCase("shuttle_transfer", "這班高鐵接駁車今天改到對面站牌上車。", "這班高鐵接駁車今仔日改去對面站牌上車。"),
    RegressionCase("shuttle_transfer", "這班車有到火車站後站嗎？", "這班車有到火車頭後站無？"),
    RegressionCase("shuttle_transfer", "如果你趕時間，建議你改搭計程車去高鐵站。", "若是你趕時間，建議你改坐計程車去高鐵站。"),
    RegressionCase("shuttle_transfer", "如果你要去北港朝天宮，可以先搭火車到斗六再轉公車。", "若是你欲去北港朝天宮，會當先搭火車到斗六才換公車。"),
    RegressionCase("shuttle_transfer", "這班車不會進站，你要去對面月台。", "這班車袂入去站內，你欲去對面月台。"),
    RegressionCase("shuttle_transfer", "去火車站的車還有五分鐘。", "去火車頭的車閣有五分鐘。"),
    RegressionCase("shuttle_transfer", "這班車等一下會先繞去市場，再回到火車站。", "這班車等咧會先踅去市場，閣轉到火車頭。"),
    # platform / station navigation
    RegressionCase("platform_nav", "這個出口今天封閉，請改走另外一邊。", "這个出口今仔日封閉，請改走另外彼爿。"),
    RegressionCase("platform_nav", "請先看月台公告，再決定要不要上車。", "請先看月台公告，再決定敢欲上車。"),
    RegressionCase("platform_nav", "如果你要去售票口，先走到大廳再右轉。", "若是你欲去賣票口，先行到大廳再正斡。"),
    RegressionCase("platform_nav", "這個月台暫時不能上車，只能下車。", "這个月台暫時袂當上車，干焦會當落車。"),
    RegressionCase("platform_nav", "你先從電扶梯下去，再走到地下月台。", "你先對電扶梯落去，才行到地下月台。"),
    RegressionCase("platform_nav", "你先走到剪票口，再往裡面走。", "你先行到剪票口，再往內底行。"),
    RegressionCase("platform_nav", "火車站前面現在在施工，請照指標走。", "火車站頭前這馬咧施工，請照指標行。"),
    RegressionCase("platform_nav", "往市場的車在這裡排隊。", "欲去市場的車佇遮排線。"),
    RegressionCase("platform_nav", "今天站牌移到巷口那邊。", "今仔日站牌移到巷口彼爿。"),
    RegressionCase("platform_nav", "站牌旁邊那台機器可以查時刻。", "站牌邊仔那台機器會當查時刻。"),
    # train service / announcements
    RegressionCase("rail_service", "這一班火車大概還要等多久？", "這一班火車差不多猶愛等偌久？"),
    RegressionCase("rail_service", "這班列車只停大站，不停靠小站。", "這班列車只停大站，無停細站。"),
    RegressionCase("rail_service", "這班車延誤了，大概要晚二十分鐘。", "這班車延誤矣，差不多要慢分二十分鐘。"),
    RegressionCase("rail_service", "這班列車客滿了，麻煩你等下一班。", "這班列車客滿矣，麻煩你等後一班。"),
    RegressionCase("rail_service", "這一站今天不停靠，請到前一站下車。", "這一站今仔日無停，請到前一站落車。"),
    RegressionCase("rail_service", "這班車不會進醫院急診門口。", "這班車袂進病院急診門跤口。"),
    RegressionCase("rail_service", "你要去門診的話，在病院門口下就可以了。", "若是你欲去門診，在病院門跤口下就會當矣。"),
    RegressionCase("rail_service", "司機休息回來就會發車。", "司機歇睏轉來就會開車。"),
    RegressionCase("rail_service", "這班車到總站就不開了。", "這班車到總站就不開矣。"),
    RegressionCase("rail_service", "司機說下一站之後要換車。", "司機講下一站了後要換車。"),
    # ticketing / payments
    RegressionCase("ticketing", "如果你要補票，先去窗口抽號碼牌。", "若是你要補票，先去窗口提號碼牌。"),
    RegressionCase("ticketing", "這張票刷不過，你去櫃檯問一下。", "這張票鑢袂過，你去櫃檯問一下。"),
    RegressionCase("ticketing", "如果刷卡還是失敗，就先投現金。", "若是刷袂過，就先投現錢。"),
    RegressionCase("ticketing", "愛心卡今天可以正常刷卡。", "愛心卡今仔日會當正常鑢卡。"),
    RegressionCase("ticketing", "老人卡感應不到的話，請你先跟司機說。", "若是老人卡感應袂著，請你先佮司機講。"),
    RegressionCase("ticketing", "你沒有零錢的話，可以去便利商店換。", "若是你無零錢，會當去便利商店換。"),
    RegressionCase("ticketing", "今天的末班車提早十分鐘開車。", "今仔日的尾班車較早十分鐘開車。"),
    RegressionCase("ticketing", "頭班車明天會晚半小時。", "頭班車明仔載會慢分半點鐘。"),
    # service / redirect / lost property
    RegressionCase("service_redirect", "如果你只是問廁所在哪裡，我可以跟你說。", "若是你只是問便所佇佗位，我會當共你講。"),
    RegressionCase("service_redirect", "這個問題不是站務能決定的，你要問服務台。", "這个問題毋是站務會當決定的，你欲問服務台。"),
    RegressionCase("service_redirect", "這個問題跟公車無關，我沒辦法回答。", "這个問題佮公車無關，我無法度回答。"),
    RegressionCase("service_redirect", "如果你要找失物，我可以幫你轉給總站。", "若是你要找失物，我會當替你轉去總站。"),
    RegressionCase("service_redirect", "如果你要找失物，我先幫你記車牌。", "若是你要找失物，我先替你記車牌。"),
    RegressionCase("service_redirect", "失物要送回總站，你下午再打電話確認。", "遺失物愛送轉去總站，你下晝閣敲電話確認。"),
    RegressionCase("service_redirect", "這個不是公車問題，請去問警察。", "這个毋是公車問題，請去問警察。"),
    # crowd / safety / queue
    RegressionCase("crowd_safety", "今天車站人很多，你先到旁邊等一下。", "今仔日車站人誠濟，你先到邊仔等一下。"),
    RegressionCase("crowd_safety", "這裡不能停車接人，請到外圍等。", "遮袂當停車接人，請到外圍等。"),
    RegressionCase("crowd_safety", "今天雨很大，車班可能不穩定。", "今仔日雨真大，車班可能不穩定。"),
    RegressionCase("crowd_safety", "請不要站在黃線外面。", "請莫徛在黃線外口。"),
    RegressionCase("crowd_safety", "這裡現在只下車不上車。", "遮這馬干焦落車，無上車。"),
    # hospital / campus / specific destination
    RegressionCase("destinations", "如果你要去醫院，這班車不會進病院內底。", "若是你欲去病院，這班車袂進病院內底。"),
    RegressionCase("destinations", "這班車今天不停靠學校門口。", "這班車今仔日無停學校門跤口。"),
    RegressionCase("destinations", "如果你要去學校裡面，要在校門口下車再走進去。", "若是你欲去學校內底，要在校門跤口落車才行進去。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="非公車交通情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return TRANSPORT_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in TRANSPORT_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in TRANSPORT_REGRESSION_CASES})
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
