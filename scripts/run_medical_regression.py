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
    RegressionCase("registration", "請問今天還有現場號碼嗎？", "借問今仔日閣有當場的號碼無？"),
    RegressionCase("registration", "我現在在掛號櫃檯前面。", "我這馬佇掛號櫃檯頭前。"),
    RegressionCase("registration", "我想問批價櫃檯在哪裡。", "我想問批價櫃檯佇佗位。"),
    RegressionCase("registration", "這份表單要先填完再去掛號。", "這份表仔愛先填好才去掛號。"),
    RegressionCase("registration", "這裡現在只收文件，不辦批價。", "遮這馬干焦收文件，不辦批價。"),
    RegressionCase("registration", "如果你要取消掛號，先抽號碼牌。", "若是你要取消掛號，先抽號碼牌。"),
    RegressionCase("registration", "這裡今天人很多，你先到旁邊坐一下。", "遮今仔日人誠濟，你先到邊仔坐咧。"),
    RegressionCase("registration", "這個問題不是我們櫃檯能決定的，要問醫生。", "這个問題毋是阮櫃檯會當決定的，要問醫生。"),
    RegressionCase("registration", "如果你趕時間，我不建議你等下午門診。", "若是你趕時間，我無建議你等下晝門診。"),
    # tests / labs
    RegressionCase("tests", "請問抽血要先去哪裡報到？", "借問抽血愛先去佗位報到？"),
    RegressionCase("tests", "這個檢查要空腹八小時。", "這个檢查要空腹八點鐘。"),
    RegressionCase("tests", "你先去照X光，再回來找我。", "你先去照X光，才轉來揣我。"),
    RegressionCase("tests", "這位病人等一下先做心電圖，再回來看診。", "這位病人等咧先做心電圖，閣轉來看診。"),
    RegressionCase("tests", "這個檢查室今天暫停使用，請到對面那間。", "這个檢查室今仔日停用，請到對面彼間。"),
    RegressionCase("tests", "抽血室在右手邊，先走到底再左轉。", "抽血室佇正手爿，先行到底才倒手斡。"),
    # reports / doctor flow
    RegressionCase("doctor_flow", "我剛剛沒有聽清楚醫生的名字。", "我拄仔無聽清楚醫生的名。"),
    RegressionCase("doctor_flow", "如果有結果我會再通知你。", "若是有結果，我會閣共你講。"),
    RegressionCase("doctor_flow", "如果你要看報告，先去門診櫃檯報到。", "若是你要看報告，先去門診櫃檯報到。"),
    RegressionCase("doctor_flow", "現在叫號還沒到你，你先坐旁邊等。", "現佇咧叫號猶未到你，你先坐隔壁等。"),
    RegressionCase("doctor_flow", "報告還沒出來，你晚一點再回來拿。", "報告猶未出來，你較慢閣轉來拿。"),
    RegressionCase("doctor_flow", "醫生現在還在看上一位，你再等一下。", "醫生這馬猶在看上一位，你閣等咧。"),
    # pharmacy / payment / cards
    RegressionCase("pharmacy_payment", "如果你要領藥，先去批價再過來。", "若是欲領藥，先去算錢才過來。"),
    RegressionCase("pharmacy_payment", "你的健保卡刷不過，先去旁邊櫃檯問一下。", "你的健保卡鑢袂過，先去隔壁櫃檯問一下。"),
    RegressionCase("pharmacy_payment", "這張單子你先拿去批價，再回來給我。", "這張單仔你先提去批價，閣轉來予我。"),
    # rooms / inpatient
    RegressionCase("rooms_inpatient", "請問病房在幾樓？", "借問病房佇第幾樓？"),
    RegressionCase("rooms_inpatient", "如果你要住院，先去住院櫃檯辦手續。", "若是欲蹛院，先去住院櫃檯辦手續。"),
    # redirect / service
    RegressionCase("redirect", "如果你只是要問廁所在哪裡，我可以跟你說。", "若是你只是欲問便所佇佗位，我會當共你講。"),
    RegressionCase("redirect", "如果你要申請病歷，請去一樓服務台。", "若是你要申請病歷，請去一樓服務台。"),
    RegressionCase("redirect", "如果你要找失物，我可以幫你轉給總機。", "若是你要找失物，我會當替你轉去總機。"),
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
