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


WORKPLACE_REGRESSION_CASES: list[RegressionCase] = [
    # meeting
    RegressionCase("meeting", "我今天要開會。", "我今仔日欲開會。"),
    RegressionCase("meeting", "下午要開會。", "下晝欲開會。"),
    RegressionCase("meeting", "會議改到明天。", "會議改做明仔載。"),
    RegressionCase("meeting", "明天早上再討論。", "明仔早起閣討論。"),
    RegressionCase("meeting", "會議取消了。", "會議取消矣。"),
    RegressionCase("meeting", "會議延後到下午。", "會議延後到下晝。"),
    RegressionCase("meeting", "請把會議連結寄給我。", "請共會議連結寄予我。"),
    RegressionCase("meeting", "開會前提醒我。", "開會前共我提醒。"),
    RegressionCase("meeting", "主管說明天要開會。", "主管講明仔載欲開會。"),
    RegressionCase("meeting", "會議要準備資料嗎？", "會議愛準備資料無？"),
    RegressionCase("meeting", "請幫我確認會議時間。", "請替我確認會議時間。"),
    RegressionCase("meeting", "可以改成線上會議嗎？", "會當改做線上會議無？"),
    RegressionCase("meeting", "我想改會議地點。", "我想欲改會議地點。"),
    RegressionCase("meeting", "可以幫我安排會議室嗎？", "會當替我安排會議室無？"),
    RegressionCase("meeting", "可以幫我預約會議室嗎？", "會當替我預約會議室無？"),
    RegressionCase("meeting", "我想改開會時間。", "我想欲改開會時間。"),
    RegressionCase("meeting", "可以幫我更新會議連結嗎？", "會當替我更新會議連結無？"),
    RegressionCase("meeting", "可以幫我寄會議紀錄嗎？", "會當替我寄會議紀錄無？"),
    RegressionCase("meeting", "我想改開會地點。", "我想欲改開會地點。"),
    RegressionCase("meeting", "我想改會議時間。", "我想欲改會議時間。"),
    RegressionCase("meeting", "可以幫我查會議室設備嗎？", "會當替我查會議室設備無？"),
    RegressionCase("meeting", "可以幫我查會議進度嗎？", "會當替我查會議進度無？"),
    # office location / progressive
    RegressionCase("office_location", "主管在會議室。", "主管佇會議室。"),
    RegressionCase("office_location", "我們在會議室開會。", "咱佇會議室開會。"),
    RegressionCase("office_location", "資料放在桌上。", "資料囥佇桌頂。"),
    RegressionCase("office_location", "會議室在哪裡？", "會議室佇佗位？"),
    RegressionCase("office_location", "茶水間在走廊旁邊。", "茶水間佇走廊邊仔。"),
    RegressionCase("office_location", "影印機在櫃檯後面。", "影印機佇櫃檯後壁。"),
    RegressionCase("office_location", "我的座位在窗戶旁邊。", "我的座位佇窗仔門邊仔。"),
    RegressionCase("progressive", "同事在等你。", "同事佇咧等你。"),
    RegressionCase("progressive", "主管在看報告。", "主管佇咧看報告。"),
    RegressionCase("progressive", "我正在寫報告。", "我佇咧寫報告。"),
    RegressionCase("progressive", "我正在整理資料。", "我佇咧整理資料。"),
    RegressionCase("progressive", "他正在確認名單。", "伊佇咧確認名單。"),
    RegressionCase("progressive", "同事正在列印文件。", "同事佇咧列印文件。"),
    RegressionCase("progressive", "我們正在聯絡客戶。", "咱佇咧聯絡客戶。"),
    # workflow
    RegressionCase("workflow", "請你再確認一次。", "請你閣確認一擺。"),
    RegressionCase("workflow", "這份文件要簽名。", "這份文件愛簽名。"),
    RegressionCase("workflow", "報告還沒寫完。", "報告猶未寫完。"),
    RegressionCase("workflow", "請把檔案寄給我。", "請共檔案寄予我。"),
    RegressionCase("workflow", "請幫我上傳檔案。", "請替我上傳檔案。"),
    RegressionCase("workflow", "我等主管回覆。", "我等主管回覆。"),
    RegressionCase("workflow", "資料要改一下。", "資料愛改一下。"),
    RegressionCase("workflow", "請把新版寄給客戶。", "請共新版寄予客戶。"),
    RegressionCase("workflow", "我可以晚一點交報告嗎？", "我會當較晏一點交報告無？"),
    RegressionCase("workflow", "報告要在下班前交。", "報告愛佇下班前交。"),
    RegressionCase("workflow", "請幫我轉給主管。", "請替我轉予主管。"),
    RegressionCase("workflow", "這份資料要更新。", "這份資料愛更新。"),
    RegressionCase("workflow", "資料我晚點補。", "資料我較晏補。"),
    RegressionCase("workflow", "請幫我看一下這份資料。", "請替我看覓這份資料。"),
    RegressionCase("workflow", "請幫我追一下進度。", "請替我追一下進度。"),
    RegressionCase("workflow", "可以幫我列印文件嗎？", "會當替我列印文件無？"),
    RegressionCase("workflow", "我想改報告期限。", "我想欲改報告期限。"),
    RegressionCase("workflow", "可以幫我查簽核進度嗎？", "會當替我查簽核進度無？"),
    RegressionCase("workflow", "可以幫我查報表進度嗎？", "會當替我查報表進度無？"),
    RegressionCase("workflow", "可以幫我查合約進度嗎？", "會當替我查合約進度無？"),
    RegressionCase("workflow", "可以幫我查報銷進度嗎？", "會當替我查報銷進度無？"),
    RegressionCase("workflow", "可以幫我查請款狀態嗎？", "會當替我查請款狀態無？"),
    RegressionCase("workflow", "可以幫我查出差申請嗎？", "會當替我查出張申請無？"),
    RegressionCase("workflow", "可以幫我查專案進度嗎？", "會當替我查專案進度無？"),
    RegressionCase("workflow", "可以幫我查任務進度嗎？", "會當替我查任務進度無？"),
    RegressionCase("workflow", "可以幫我查薪資明細嗎？", "會當替我查月給明細無？"),
    # leave / availability
    RegressionCase("leave_availability", "我今天請假。", "我今仔日請假。"),
    RegressionCase("leave_availability", "我想請半天假。", "我想欲請半日假。"),
    RegressionCase("leave_availability", "你有空嗎？", "你有閒無？"),
    RegressionCase("leave_availability", "我現在不方便。", "我這馬無方便。"),
    RegressionCase("leave_availability", "我今天會晚點到。", "我今仔日會較晏到。"),
    RegressionCase("leave_availability", "我想請病假。", "我想欲告病。"),
    RegressionCase("leave_availability", "我臨時有事。", "我臨時有代誌。"),
    RegressionCase("leave_availability", "我不在座位上。", "我無佇座位頂。"),
    RegressionCase("leave_availability", "我晚點進公司。", "我較晏進公司。"),
    RegressionCase("leave_availability", "可以幫我代班嗎？", "會當替我代班無？"),
    RegressionCase("leave_availability", "可以幫我改時間嗎？", "會當替我改時間無？"),
    RegressionCase("leave_availability", "可以幫我請公假嗎？", "會當替我請公假無？"),
    RegressionCase("leave_availability", "可以幫我登記請假嗎？", "會當替我登記請假無？"),
    RegressionCase("leave_availability", "我想改班表。", "我想欲改班表。"),
    RegressionCase("leave_availability", "我想改上班時間。", "我想欲改上班時間。"),
    RegressionCase("leave_availability", "可以幫我查請假紀錄嗎？", "會當替我查請假紀錄無？"),
    RegressionCase("leave_availability", "可以幫我查加班申請嗎？", "會當替我查加班申請無？"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="辦公/會議情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return WORKPLACE_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in WORKPLACE_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in WORKPLACE_REGRESSION_CASES})
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
