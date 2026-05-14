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


MEDICAL_REGRESSION_CASES: list[RegressionCase] = [
    # registration / counters
    RegressionCase("registration", "不好意思，我想改掛號時間。", "歹勢，我想欲改掛號時間。"),
    RegressionCase("registration", "我需要改掛號時間。", "我欲改掛號時間。"),
    RegressionCase("registration", "請協助改掛號時間。", "請鬥相共改掛號時間。"),
    RegressionCase("registration", "請問今天還有現場號碼嗎？", "借問今仔日閣有當場的號碼無？"),
    RegressionCase("registration", "我現在在掛號櫃檯前面。", "我這馬佇掛號櫃檯頭前。"),
    RegressionCase("registration", "我想問批價櫃檯在哪裡。", "我想欲問批價櫃檯佇佗位。"),
    RegressionCase("registration", "這份表單要先填完再去掛號。", "這份表仔愛先填好才去掛號。"),
    RegressionCase("registration", "這裡現在只收文件，不辦批價。", "遮這馬干焦收文件，不辦批價。"),
    RegressionCase("registration", "如果你要取消掛號，先抽號碼牌。", "若是你欲取消掛號，先抽號碼牌。"),
    RegressionCase("registration", "這裡今天人很多，你先到旁邊坐一下。", "遮今仔日人誠濟，你先到邊仔坐咧。"),
    RegressionCase("registration", "這個問題不是我們櫃檯能決定的，要問醫生。", "這个問題毋是阮櫃檯會當決定的，要問醫生。"),
    RegressionCase("registration", "如果你趕時間，我不建議你等下午門診。", "若是你趕時間，我無建議你等下晝門診。"),
    RegressionCase("registration", "我要去診所掛號。", "我欲去醫生館掛號。"),
    RegressionCase("registration", "第一次來診所要帶健保卡。", "頭擺來醫生館愛帶健保卡。"),
    RegressionCase("registration", "請先到櫃台報到。", "請先到櫃檯報到。"),
    RegressionCase("registration", "診所今天看到幾點？", "醫生館今仔日看甲幾點？"),
    RegressionCase("registration", "請把健保卡交給櫃台。", "請共健保卡交予櫃檯。"),
    RegressionCase("registration", "如果過號了，要再去櫃台處理嗎？", "若是過號矣，愛閣去櫃檯處理無？"),
    RegressionCase("registration", "可以幫我掛急診嗎？", "會當替我掛急診無？"),
    RegressionCase("registration", "可以幫我改掛號科別嗎？", "會當替我改掛號科別無？"),
    RegressionCase("registration", "可以幫我改預約時間嗎？", "會當替我改預約時間無？"),
    RegressionCase("registration", "可以幫我查候補名單嗎？", "會當替我查候補名單無？"),
    RegressionCase("registration", "我想改看診科別。", "我想欲改看診科別。"),
    RegressionCase("registration", "可以幫我查掛號費嗎？", "會當替我查掛號費無？"),
    RegressionCase("registration", "可以幫我查掛號狀態嗎？", "會當替我查掛號狀態無？"),
    RegressionCase("registration", "我需要查掛號狀態。", "我欲查掛號狀態。"),
    RegressionCase("registration", "請協助查掛號狀態。", "請鬥相共查掛號狀態。"),
    RegressionCase("registration", "我需要確認掛號狀態。", "我欲確認掛號狀態。"),
    RegressionCase("registration", "我想確認掛號狀態。", "我想欲確認掛號狀態。"),
    RegressionCase("registration", "可以幫我確認掛號狀態嗎？", "會當替我確認掛號狀態無？"),
    RegressionCase("registration", "請幫我確認掛號狀態。", "請替我確認掛號狀態。"),
    RegressionCase("registration", "麻煩你幫我確認掛號狀態。", "麻煩你替我確認掛號狀態。"),
    RegressionCase("registration", "幫我確認掛號狀態。", "替我確認掛號狀態。"),
    RegressionCase("registration", "請協助確認掛號狀態。", "請鬥相共確認掛號狀態。"),
    RegressionCase("registration", "可以幫我查候診號碼嗎？", "會當替我查候診號碼無？"),
    RegressionCase("registration", "可以幫我查病歷號碼嗎？", "會當替我查病歷號碼無？"),
    # tests / labs
    RegressionCase("tests", "請問抽血要先去哪裡報到？", "借問抽血愛先去佗位報到？"),
    RegressionCase("tests", "這個檢查要空腹八小時。", "這个檢查要空腹八點鐘。"),
    RegressionCase("tests", "你先去照X光，再回來找我。", "你先去照X光，才轉來揣我。"),
    RegressionCase("tests", "這位病人等一下先做心電圖，再回來看診。", "這位病人等咧先做心電圖，閣轉來看診。"),
    RegressionCase("tests", "這個檢查室今天暫停使用，請到對面那間。", "這个檢查室今仔日停用，請到對面彼間。"),
    RegressionCase("tests", "抽血室在右手邊，先走到底再左轉。", "抽血室佇正手爿，先行到底才倒手斡。"),
    RegressionCase("tests", "超音波要到二樓，先報到再等候。", "超音波要到二樓，先報到再等候。"),
    RegressionCase("tests", "這個檢查需要家屬陪同。", "這个檢查需要家屬陪同。"),
    RegressionCase("tests", "今天的採樣結果明天才出來。", "今仔日的採樣結果明仔載才出來。"),
    RegressionCase("tests", "請先去抽血，回來再做其他檢查。", "請先去抽血，轉來再做其他檢查。"),
    RegressionCase("tests", "可以幫我安排檢查時間嗎？", "會當替我安排檢查時間無？"),
    RegressionCase("tests", "我需要安排檢查時間。", "我欲安排檢查時間。"),
    RegressionCase("tests", "請協助安排檢查時間。", "請鬥相共安排檢查時間。"),
    RegressionCase("tests", "麻煩幫我安排檢查時間。", "麻煩替我安排檢查時間。"),
    RegressionCase("tests", "請你幫我安排檢查時間。", "請你替我安排檢查時間。"),
    RegressionCase("tests", "能不能幫我安排檢查時間？", "敢會當替我安排檢查時間？"),
    RegressionCase("tests", "能幫我安排檢查時間嗎？", "會當替我安排檢查時間無？"),
    RegressionCase("tests", "能否幫我安排檢查時間？", "敢會當替我安排檢查時間？"),
    RegressionCase("tests", "方便幫我安排檢查時間嗎？", "方便替我安排檢查時間無？"),
    RegressionCase("tests", "可否幫我安排檢查時間？", "敢會當替我安排檢查時間？"),
    RegressionCase("tests", "拜託幫我安排檢查時間。", "拜託替我安排檢查時間。"),
    RegressionCase("tests", "我想請你幫我安排檢查時間。", "我想欲請你替我安排檢查時間。"),
    RegressionCase("tests", "可以幫我查檢查結果嗎？", "會當替我查檢查結果無？"),
    RegressionCase("tests", "我想改抽血時間。", "我想欲改抽血時間。"),
    RegressionCase("tests", "可以幫我查檢驗進度嗎？", "會當替我查檢驗進度無？"),
    RegressionCase("tests", "我想改檢查日期。", "我想欲改檢查日期。"),
    # reports / doctor flow
    RegressionCase("doctor_flow", "我剛剛沒有聽清楚醫生的名字。", "我拄仔無聽清楚醫生的名。"),
    RegressionCase("doctor_flow", "如果有結果我會再通知你。", "若是有結果，我會閣共你講。"),
    RegressionCase("doctor_flow", "如果你要看報告，先去門診櫃檯報到。", "若是你欲看報告，先去門診櫃檯報到。"),
    RegressionCase("doctor_flow", "現在叫號還沒到你，你先坐旁邊等。", "現佇咧叫號猶未到你，你先坐隔壁等。"),
    RegressionCase("doctor_flow", "報告還沒出來，你晚一點再回來拿。", "報告猶未出來，你較慢閣轉來拿。"),
    RegressionCase("doctor_flow", "醫生現在還在看上一位，你再等一下。", "醫生這馬猶在看上一位，你閣等咧。"),
    RegressionCase("doctor_flow", "請問今天看診的醫生是哪一位？", "借問今仔日看診的醫生是佗一位？"),
    RegressionCase("doctor_flow", "醫生說你需要再回診一次。", "醫生講你需要再回診一擺。"),
    RegressionCase("doctor_flow", "這份報告已經轉給主治醫生了。", "這份報告已經轉給主治醫生矣。"),
    RegressionCase("doctor_flow", "這個科今天沒有門診，請你改天再來。", "這个科今仔日無門診，請你改工閣來。"),
    RegressionCase("doctor_flow", "醫生叫我下星期回診。", "醫生叫我下禮拜回診。"),
    RegressionCase("doctor_flow", "今天門診提早結束，請明天再來。", "今仔日門診較早結束，請明仔載閣來。"),
    RegressionCase("doctor_flow", "看完診再去櫃台拿藥單。", "看完診閣去櫃檯提藥單。"),
    RegressionCase("doctor_flow", "醫生還沒來，你先在外面等候。", "醫生猶未來，你先佇外口等候。"),
    RegressionCase("doctor_flow", "醫生叫我下禮拜回診。", "醫生叫我下禮拜回診。"),
    RegressionCase("doctor_flow", "這位醫生今天下午休診。", "這位醫生今仔日下晝無看診。"),
    RegressionCase("doctor_flow", "今天停診，請你改天再來。", "今仔日無看診，請你改工閣來。"),
    RegressionCase("doctor_flow", "如果你要改看診時間，請先打電話。", "若是你欲改看診時間，請先敲電話。"),
    RegressionCase("doctor_flow", "醫生臨時請假，門診改到明天早上。", "醫生臨時請假，門診改做明仔早起。"),
    RegressionCase("doctor_flow", "今天下午的門診改到明天早上。", "今仔日下晝的門診改做明仔早起。"),
    RegressionCase("doctor_flow", "你可以改掛別的醫生。", "你會當改掛別位醫生。"),
    RegressionCase("doctor_flow", "如果你不方便，可以改掛別的醫生。", "若是你無方便，會當改掛別位醫生。"),
    RegressionCase("doctor_flow", "我想查檢查報告。", "我想欲查檢查報告。"),
    RegressionCase("doctor_flow", "可以幫我查報告嗎？", "會當替我查報告無？"),
    RegressionCase("doctor_flow", "我想改回診時間。", "我想欲改回診時間。"),
    RegressionCase("doctor_flow", "我需要改回診時間。", "我欲改回診時間。"),
    RegressionCase("doctor_flow", "請協助改回診時間。", "請鬥相共改回診時間。"),
    RegressionCase("doctor_flow", "可以幫我查門診進度嗎？", "會當替我查門診進度無？"),
    RegressionCase("doctor_flow", "我想取消回診。", "我想欲取消回診。"),
    RegressionCase("doctor_flow", "我想取消掛號。", "我想欲取消掛號。"),
    RegressionCase("doctor_flow", "我需要取消回診。", "我欲取消回診。"),
    RegressionCase("doctor_flow", "我需要取消掛號。", "我欲取消掛號。"),
    RegressionCase("doctor_flow", "可以幫我取消掛號嗎？", "會當替我取消掛號無？"),
    RegressionCase("doctor_flow", "可以幫我取消回診嗎？", "會當替我取消回診無？"),
    RegressionCase("doctor_flow", "請協助取消掛號。", "請鬥相共取消掛號。"),
    RegressionCase("doctor_flow", "請協助取消回診。", "請鬥相共取消回診。"),
    RegressionCase("doctor_flow", "可以幫我查醫生門診時間嗎？", "會當替我查醫生門診時間無？"),
    RegressionCase("doctor_flow", "可以幫我查回診時間嗎？", "會當替我查回診時間無？"),
    RegressionCase("doctor_flow", "我需要查回診時間。", "我欲查回診時間。"),
    RegressionCase("doctor_flow", "請協助查回診時間。", "請鬥相共查回診時間。"),
    RegressionCase("doctor_flow", "我需要確認回診時間。", "我欲確認回診時間。"),
    RegressionCase("doctor_flow", "我想確認回診時間。", "我想欲確認回診時間。"),
    RegressionCase("doctor_flow", "可以幫我確認回診時間嗎？", "會當替我確認回診時間無？"),
    RegressionCase("doctor_flow", "請幫我確認回診時間。", "請替我確認回診時間。"),
    RegressionCase("doctor_flow", "麻煩你幫我確認回診時間。", "麻煩你替我確認回診時間。"),
    RegressionCase("doctor_flow", "幫我確認回診時間。", "替我確認回診時間。"),
    RegressionCase("doctor_flow", "請協助確認回診時間。", "請鬥相共確認回診時間。"),
    RegressionCase("doctor_flow", "可以幫我查健檢報告嗎？", "會當替我查健檢報告無？"),
    RegressionCase("doctor_flow", "我想改回診日期。", "我想欲改回診日期。"),
    # pharmacy / payment / cards
    RegressionCase("pharmacy_payment", "如果你要領藥，先去批價再過來。", "若是欲領藥，先去算錢才過來。"),
    RegressionCase("pharmacy_payment", "你的健保卡刷不過，先去旁邊櫃檯問一下。", "你的健保卡鑢袂過，先去隔壁櫃檯問一下。"),
    RegressionCase("pharmacy_payment", "這張單子你先拿去批價，再回來給我。", "這張單仔你先提去批價，閣轉來予我。"),
    RegressionCase("pharmacy_payment", "這些藥一天吃三次，飯後服用。", "遮的藥一工食三擺，飯後服用。"),
    RegressionCase("pharmacy_payment", "這個藥要冷藏，你回家記得放冰箱。", "這个藥要寒藏，你轉去厝裡記著放冰箱。"),
    RegressionCase("pharmacy_payment", "你的藥袋在這裡，共有三種藥。", "你的藥袋佇遮，共有三種藥。"),
    RegressionCase("pharmacy_payment", "今天藥局到五點，你早一點來拿。", "今仔日藥局到五點，你較早來拿。"),
    RegressionCase("pharmacy_payment", "我要領慢性病藥。", "我欲領慢性病藥。"),
    RegressionCase("pharmacy_payment", "可以幫我量血壓嗎？", "會當替我量血壓無？"),
    RegressionCase("pharmacy_payment", "可以幫我查藥單嗎？", "會當替我查藥單無？"),
    RegressionCase("pharmacy_payment", "我想改取藥時間。", "我想欲改取藥時間。"),
    RegressionCase("pharmacy_payment", "可以幫我改領藥地點嗎？", "會當替我改領藥地點無？"),
    RegressionCase("pharmacy_payment", "可以幫我查藥局位置嗎？", "會當替我查藥局位置無？"),
    RegressionCase("pharmacy_payment", "可以幫我補印收據嗎？", "會當替我補印收據無？"),
    RegressionCase("pharmacy_payment", "可不可以麻煩您幫我補印收據？", "敢會當麻煩你替我補印收據？"),
    RegressionCase("pharmacy_payment", "我需要補印收據。", "我欲補印收據。"),
    RegressionCase("pharmacy_payment", "請協助補印收據。", "請鬥相共補印收據。"),
    RegressionCase("pharmacy_payment", "可以幫我查疫苗紀錄嗎？", "會當替我查疫苗紀錄無？"),
    RegressionCase("pharmacy_payment", "可以幫我查藥品庫存嗎？", "會當替我查藥品庫存無？"),
    RegressionCase("pharmacy_payment", "可以幫我查藥費明細嗎？", "會當替我查藥費明細無？"),
    RegressionCase("pharmacy_payment", "我想改領藥時間。", "我想欲改領藥時間。"),
    RegressionCase("pharmacy_payment", "我想改拿藥日期。", "我想欲改領藥日期。"),
    RegressionCase("pharmacy_payment", "我想改領藥日期。", "我想欲改領藥日期。"),
    RegressionCase("pharmacy_payment", "我想改服藥時間。", "我想欲改服藥時間。"),
    # rooms / inpatient
    RegressionCase("rooms_inpatient", "請問病房在幾樓？", "借問病房佇第幾樓？"),
    RegressionCase("rooms_inpatient", "如果你要住院，先去住院櫃檯辦手續。", "若是欲蹛院，先去住院櫃檯辦手續。"),
    RegressionCase("rooms_inpatient", "請問探視時間是幾點開始？", "借問探視時間是幾點開始？"),
    RegressionCase("rooms_inpatient", "你的床位在三樓，電梯在右手邊。", "你的床位佇三樓，電梯在正手爿。"),
    RegressionCase("rooms_inpatient", "住院需要帶健保卡和身分證。", "蹛院需要帶健保卡和身分證。"),
    RegressionCase("rooms_inpatient", "家屬可以在外面等，不能進去加護病房。", "家屬會當佇外口等，袂當進去加護病房。"),
    RegressionCase("rooms_inpatient", "這間病房今天滿了，要換到四樓。", "這間病房今仔日滿矣，要換到四樓。"),
    RegressionCase("rooms_inpatient", "你先去辦住院手續，再上去找護理站。", "你先去辦蹛院手續，再上去找護理站。"),
    RegressionCase("rooms_inpatient", "陪病家屬要先登記。", "陪病家屬愛先登記。"),
    RegressionCase("rooms_inpatient", "請先到護理站報到。", "請先去護理站報到。"),
    RegressionCase("rooms_inpatient", "這位病人明天要轉到普通病房。", "這位病人明仔載要轉去普通病房。"),
    RegressionCase("rooms_inpatient", "可以幫我查床位嗎？", "會當替我查床位無？"),
    RegressionCase("rooms_inpatient", "我想改病房。", "我想欲改病房。"),
    RegressionCase("rooms_inpatient", "可以幫我改陪病人數嗎？", "會當替我改陪病人數無？"),
    RegressionCase("rooms_inpatient", "可以幫我查住院費用嗎？", "會當替我查蹛院費用無？"),
    RegressionCase("rooms_inpatient", "可以幫我查住院床位嗎？", "會當替我查蹛院床位無？"),
    RegressionCase("rooms_inpatient", "可以幫我查住院手續嗎？", "會當替我查蹛院手續無？"),
    RegressionCase("rooms_inpatient", "我想改住院日期。", "我想欲改蹛院日期。"),
    # redirect / service
    RegressionCase("redirect", "如果你只是要問廁所在哪裡，我可以跟你說。", "若是你只是欲問便所佇佗位，我會當共你講。"),
    RegressionCase("redirect", "如果你要申請病歷，請去一樓服務台。", "若是你欲申請病歷，請去一樓服務台。"),
    RegressionCase("redirect", "可以幫我申請病歷嗎？", "會當替我申請病歷無？"),
    RegressionCase("redirect", "我需要申請病歷。", "我欲申請病歷。"),
    RegressionCase("redirect", "請協助申請病歷。", "請鬥相共申請病歷。"),
    RegressionCase("redirect", "如果你要找失物，我可以幫你轉給總機。", "若是你欲找失物，我會當替你轉去總機。"),
    RegressionCase("redirect", "急診在另外一棟，你先走出去再左轉。", "急診在另外一棟，你先行出去再倒斡。"),
    RegressionCase("redirect", "投訴要去二樓的服務台，不是我們這裡。", "投訴欲去二樓的服務台，毋是阮遮。"),
    RegressionCase("redirect", "掛號要去一樓，不是這裡。", "掛號欲去一樓，毋是遮。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="醫療櫃檯/門診情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return MEDICAL_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in MEDICAL_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in MEDICAL_REGRESSION_CASES})
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
