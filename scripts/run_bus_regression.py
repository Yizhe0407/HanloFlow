from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from converter import TaigiConverter


@dataclass(frozen=True)
class RegressionCase:
    category: str
    source: str
    expected: str


BUS_REGRESSION_CASES: list[RegressionCase] = [
    # stop control / stop status
    RegressionCase("stop_control", "這班公車恢復停靠本站了，你不用再走去對面。", "這班公車閣停本站矣，你免閣行去對面。"),
    RegressionCase("stop_control", "這班公車只停外圍站，不會開進廟口前面。", "這班公車干焦停外圍站，袂開進廟埕頭前。"),
    RegressionCase("stop_control", "如果你要去醫院院區裡面，這班車沒有進去。", "若是你欲去病院內底，這班車無進去。"),
    RegressionCase("stop_control", "這個站牌已經恢復正常停靠了。", "這个站牌已經閣正常停矣。"),
    RegressionCase("stop_control", "這個站牌今天恢復停靠，但班次還沒有完全正常。", "這个站牌今仔日閣停，但班次猶未攏正常。"),
    RegressionCase("stop_control", "這個站牌暫時改成下車專用，上車請到對面。", "這个站牌暫時改成干焦落車，上車請到對面。"),
    RegressionCase("stop_control", "這一站今天暫停使用，請改到前面臨時站牌。", "這一站今仔日停用，請改去頭前臨時站牌。"),
    RegressionCase("stop_control", "這班公車今天不停靠學校門口。", "這班公車今仔日無停學校門跤口。"),
    RegressionCase("stop_control", "這班車今天不停靠北港朝天宮。", "這班車今仔日無停北港朝天宮。"),
    RegressionCase("stop_control", "這裡不是下車站，你要到前面那一站下。", "遮毋是落車站，你欲到頭前彼站才落。"),
    RegressionCase("stop_control", "這裡只能下車，不能上車。", "遮干焦會當落車，袂予人上車。"),
    RegressionCase("stop_control", "司機說這一站只下客，不給上車。", "司機講這一站干焦落客，袂予上車。"),
    RegressionCase("stop_control", "這站只讓下車，不讓上車。", "這站干焦落車，無予上車。"),
    RegressionCase("stop_control", "這班公車現在只下客，不載人上車。", "這班公車這馬干焦落客，無予人上車。"),
    RegressionCase("stop_control", "這班公車現在只停外圍站，你要自己走進去老街。", "這班公車這馬干焦停外圍站，你欲家己行入去老街。"),
    RegressionCase("stop_control", "今天因為活動，公車只停外圍，不會開進老街裡面。", "今仔日因為活動，公車干焦停外圍，袂開進老街內底。"),
    RegressionCase("stop_control", "這班車回程不會經過這一站。", "這班車回程袂經過這一站。"),
    RegressionCase("stop_control", "你要去對面搭回程車。", "你愛去對面搭回程車。"),
    RegressionCase("stop_control", "高鐵接駁車在對面站牌搭。", "高鐵接駁車佇對面站牌搭。"),
    RegressionCase("stop_control", "這班車今天只開前門。", "這班車今仔日干焦開前門。"),
    RegressionCase("stop_control", "後門現在不開放，請從前門上車。", "後門這馬無開放，請對頭前門上車。"),
    # delay / eta / detour
    RegressionCase("delay_eta", "如果你趕時間，我不建議你等這班車。", "若是你趕時間，我無建議你等這班車。"),
    RegressionCase("delay_eta", "如果你趕時間，建議你不要等這班車。", "若是你趕時間，建議你莫等這班車。"),
    RegressionCase("delay_eta", "因為前面塞車，現在沒有辦法給你很準的到站時間。", "因為頭前窒車，這馬無法度共你報真準的到站時間。"),
    RegressionCase("delay_eta", "前面有交通事故，公車會慢十五分鐘。", "頭前有交通事故，公車會慢十五分鐘。"),
    RegressionCase("delay_eta", "頭班車明天會晚半小時。", "頭班車明仔載會慢分半點鐘。"),
    RegressionCase("delay_eta", "今天的末班車提早十分鐘開車。", "今仔日的尾班車較早十分鐘開車。"),
    RegressionCase("delay_eta", "這班公車臨時改道，轉車時間也會較慢。", "這班公車臨時改道，轉車時間也會較慢。"),
    RegressionCase("delay_eta", "這班公車改道以後，可能不會經過縣政府。", "這班公車改道了後，可能袂經過縣政府。"),
    RegressionCase("delay_eta", "往縣政府的車今天改道。", "往縣政府的車今仔日改道。"),
    RegressionCase("delay_eta", "這班車現在先停駛，晚一點再公告。", "這班車這馬先停開，較慢再公告。"),
    RegressionCase("delay_eta", "手機顯示不準，現場公告才準。", "手機顯示的無準，現場公告的才準。"),
    RegressionCase("delay_eta", "現在先照站牌公告，不要看手機時間。", "這馬先照站牌公告，莫看手機時間。"),
    RegressionCase("delay_eta", "回程車大概十分鐘後到。", "回程車差不多十分鐘後到。"),
    RegressionCase("delay_eta", "去火車站的車還有五分鐘。", "去火車頭的車閣有五分鐘。"),
    RegressionCase("delay_eta", "司機休息回來就會發車。", "司機歇睏轉來就會開車。"),
    RegressionCase("delay_eta", "司機會在下一站休息五分鐘，你不用下車。", "司機會佇下一站歇睏五分鐘，你免落車。"),
    # payment / cards / ticketing
    RegressionCase("payment_cards", "如果你沒有零錢，可以先投現再去總站補票。", "若是你無零錢，會當先投現錢閣去總站補票。"),
    RegressionCase("payment_cards", "如果你的愛心卡刷不過，可以改用投現。", "若是你的愛心卡鑢袂過，會當改用投現錢。"),
    RegressionCase("payment_cards", "如果刷卡還是不過，我先幫你登記，再請你補票。", "若是鑢卡猶毋過，我先替你登記，再請你補票。"),
    RegressionCase("payment_cards", "如果刷卡還是失敗，就先投現金。", "若是刷袂過，就先投現錢。"),
    RegressionCase("payment_cards", "你先上車，補票到總站再處理。", "你先上車，補票到總站再處理。"),
    RegressionCase("payment_cards", "如果你要去總站補票，先跟司機說一聲。", "若是你欲去總站補票，先佮司機講一聲。"),
    RegressionCase("payment_cards", "這台刷卡機壞了，你到後門那台刷。", "這台鑢卡機歹去矣，你到後門那台刷。"),
    RegressionCase("payment_cards", "愛心卡今天可以正常刷卡。", "愛心卡今仔日會當正常鑢卡。"),
    RegressionCase("payment_cards", "老人卡感應不到的話，請你先跟司機說。", "若是老人卡感應袂著，請你先佮司機講。"),
    RegressionCase("payment_cards", "你沒有零錢的話，可以去便利商店換。", "若是你無零錢，會當去便利商店換。"),
    # accessibility / boarding
    RegressionCase("accessibility", "今天這班低底盤公車壞掉了，換成一般車。", "今仔日這班低底盤公車歹去矣，換成一般車。"),
    RegressionCase("accessibility", "這班車今天改成小車，所以沒有輪椅斜板。", "這班車今仔日改成小車，所以無輪椅斜板。"),
    RegressionCase("accessibility", "輪椅要上車的話，我先幫你放斜板。", "若是輪椅要上車，我先替你共斜板放落來。"),
    RegressionCase("accessibility", "你如果要推輪椅上車，等一下我先請大家讓一下。", "你若是要推輪椅上車，等咧我先請逐家讓一下。"),
    RegressionCase("accessibility", "你先不要排太前面，讓輪椅乘客先上車。", "你先莫排太頭前，讓坐輪椅的乘客先上車。"),
    RegressionCase("accessibility", "嬰兒車也可以上車，但請先收好。", "嬰仔車也會當上車，但請先收予好。"),
    RegressionCase("accessibility", "這班低底盤公車今天沒有來。", "這班低底盤公車今仔日無來。"),
    RegressionCase("accessibility", "這班車今天不載腳踏車。", "這班車今仔日無載跤踏車。"),
    # route / transfer / destinations
    RegressionCase("route_transfer", "這班車今天不會再往前開，你要在火車站轉車。", "這班車今仔日袂閣往前開，你欲在火車頭轉車。"),
    RegressionCase("route_transfer", "這班公車會先到高鐵站，再回到斗六火車站。", "這班公車會先到高鐵站，閣轉到斗六火車站。"),
    RegressionCase("route_transfer", "這班車先到總站，再開去高鐵站。", "這班車先到總站，再開去高鐵站。"),
    RegressionCase("route_transfer", "這班車等一下會先進醫院，再回到車站。", "這班車等咧會先進病院，閣轉到車站。"),
    RegressionCase("route_transfer", "這班車等一下會先繞去市場，再回到火車站。", "這班車等咧會先踅去市場，閣轉到火車頭。"),
    RegressionCase("route_transfer", "如果你要轉火車，這班車可能來不及。", "若是你要轉火車，這班車可能袂赴。"),
    RegressionCase("route_transfer", "這班車到總站就不開了。", "這班車到總站就不開矣。"),
    RegressionCase("route_transfer", "這班車不會進醫院急診門口。", "這班車袂進病院急診門跤口。"),
    RegressionCase("route_transfer", "你要去門診的話，在病院門口下就可以了。", "若是你欲去門診，在病院門跤口下就會當矣。"),
    RegressionCase("route_transfer", "如果你要去學校裡面，要在校門口下車再走進去。", "若是你欲去學校內底，要在校門跤口落車才行進去。"),
    RegressionCase("route_transfer", "要去老街的話，你在外圍下車再走進去。", "若是欲去老街，你佇外圍落車才行進去。"),
    RegressionCase("route_transfer", "要去朝天宮的話，你在外圍下車就可以。", "若是欲去朝天宮，你佇外圍落車就會當。"),
    RegressionCase("route_transfer", "你去對面坐回火車站那班。", "你去對面坐回火車頭彼班。"),
    RegressionCase("route_transfer", "往市場的車在這裡排隊。", "欲去市場的車佇遮排線。"),
    # station service / lost property / redirect
    RegressionCase("service_redirect", "這個問題要問承辦單位，我這邊只能查公車班次。", "這个問題要問承辦單位，我遮干焦會當查公車班次。"),
    RegressionCase("service_redirect", "這個問題跟公車無關，我沒辦法回答。", "這个問題佮公車無關，我無法度回答。"),
    RegressionCase("service_redirect", "這不是我們站務可以決定的，你要問總站。", "這毋是阮站務會當決定的，你欲問總站。"),
    RegressionCase("service_redirect", "這個不是公車問題，請去問警察。", "這个毋是公車問題，請去問警察。"),
    RegressionCase("service_redirect", "如果你只是問廁所在哪裡，我可以跟你說。", "若是你只是問便所佇佗位，我會當共你講。"),
    RegressionCase("service_redirect", "這裡有公車動態可以查。", "遮有公車動態會當查。"),
    RegressionCase("service_redirect", "如果你只是要知道公車到哪裡，我可以幫你查動態。", "若是你只是欲知影公車到佗位，我會當替你查動態。"),
    RegressionCase("service_redirect", "如果你要查公車到哪裡，我可以幫你看。", "若是你要查公車到佗位，我會當替你看。"),
    RegressionCase("service_redirect", "如果你要找失物，我可以幫你轉給總站處理。", "若是你要找失物，我會當替你轉去總站處理。"),
    RegressionCase("service_redirect", "如果你要查失物，我可以先幫你記下車牌和時間。", "若是你要查失物，我會當先幫你記落車牌佮時間。"),
    RegressionCase("service_redirect", "如果你要找失物，我先幫你記車牌。", "若是你要找失物，我先替你記車牌。"),
    RegressionCase("service_redirect", "失物要送回總站，你下午再打電話確認。", "遺失物愛送轉去總站，你下晝閣敲電話確認。"),
    # weather / crowd / queue
    RegressionCase("weather_crowd", "如果車子太滿，站牌這邊就先不要再排。", "若是車子太滿，站牌遮就先莫再排。"),
    RegressionCase("weather_crowd", "等一下如果車子太滿，司機可能不會讓你上車。", "等咧若是車子太滿，司機可能袂予你上車。"),
    RegressionCase("weather_crowd", "站牌這邊現在人很多，你先到旁邊等比較安全。", "站牌遮這馬人誠濟，你先到邊仔等較安全。"),
    RegressionCase("weather_crowd", "連假人很多，你先排旁邊一點。", "連假人誠濟，你先排較邊仔。"),
    RegressionCase("weather_crowd", "今天雨比較大，你先到騎樓下等，車到我再叫你。", "今仔日雨較大，你先到亭仔跤等，車到我才叫你。"),
    RegressionCase("weather_crowd", "你先到亭仔跤等，雨停了再過來。", "你先到亭仔跤等，雨停了再過來。"),
    RegressionCase("weather_crowd", "今天雨很大，車班可能不穩定。", "今仔日雨真大，車班可能不穩定。"),
    RegressionCase("weather_crowd", "廟口前面今天不能停車。", "廟埕頭前今仔日袂當停車。"),
    # misc route / wording
    RegressionCase("misc", "這班車已經客滿了，麻煩你等下一班。", "這班車已經客滿矣，麻煩你等後一班。"),
    RegressionCase("misc", "你先不要上車，讓老人先上。", "你先莫上車，讓老人先上。"),
    RegressionCase("misc", "回程車大概十分鐘後到。", "回程車差不多十分鐘後到。"),
    RegressionCase("misc", "站牌旁邊那台機器可以查時刻。", "站牌邊仔那台機器會當查時刻。"),
    RegressionCase("misc", "今天站牌移到巷口那邊。", "今仔日站牌移到巷口彼爿。"),
    RegressionCase("misc", "行李太大件的話，麻煩你放旁邊。", "若是行李太大件，麻煩你囥邊仔。"),
    RegressionCase("misc", "這班公車今天不停靠北港朝天宮。", "這班公車今仔日無停北港朝天宮。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="公車站務情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="只跑指定 category，可重複傳入",
    )
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return BUS_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in BUS_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in BUS_REGRESSION_CASES})
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

    print(
        {
            "rounds": args.rounds,
            "case_count": len(cases),
            "categories": dict(sorted(category_counts.items())),
        }
    )

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
