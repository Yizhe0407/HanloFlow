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


SCHOOL_REGRESSION_CASES: list[RegressionCase] = [
    # teacher_class — 教師課堂
    RegressionCase("teacher_class", "老師在上課。", "老師佇咧上課。"),
    RegressionCase("teacher_class", "老師好。", "老師好。"),
    RegressionCase("teacher_class", "請問教室在哪裡？", "借問教室佇佗位？"),
    RegressionCase("teacher_class", "這堂課很有趣。", "這堂課真心適。"),
    RegressionCase("teacher_class", "老師今天會點名嗎？", "老師今仔日會點名無？"),
    RegressionCase("teacher_class", "請幫我聯絡家長。", "請替我聯絡家長。"),
    RegressionCase("teacher_class", "請家長簽聯絡簿。", "請家長簽聯絡簿。"),
    RegressionCase("teacher_class", "今天要考小考嗎？", "今仔日愛考小考無？"),
    RegressionCase("teacher_class", "老師請大家安靜。", "老師請逐家恬恬。"),
    RegressionCase("teacher_class", "請同學打開課本。", "請同學拍開課本。"),
    RegressionCase("teacher_class", "可以幫我影印講義嗎？", "會當替我影印講義無？"),
    RegressionCase("teacher_class", "可以幫我聯絡導師嗎？", "會當替我聯絡導師無？"),
    RegressionCase("teacher_class", "可以幫我通知家長嗎？", "會當替我通知家長無？"),
    RegressionCase("teacher_class", "可以幫我通知同學嗎？", "會當替我通知同學無？"),
    RegressionCase("teacher_class", "可以幫老師換班嗎？", "會當替老師換班無？"),
    RegressionCase("teacher_class", "可不可以幫老師換班？", "敢會當替老師換班？"),
    RegressionCase("teacher_class", "請協助老師換班。", "請鬥相共老師換班。"),
    RegressionCase("teacher_class", "可以幫導師換班嗎？", "會當替導師換班無？"),
    RegressionCase("teacher_class", "可不可以幫導師換班？", "敢會當替導師換班？"),
    RegressionCase("teacher_class", "請協助導師換班。", "請鬥相共導師換班。"),
    RegressionCase("teacher_class", "可以幫導師確認時間嗎？", "會當替導師確認時間無？"),
    RegressionCase("teacher_class", "可不可以幫導師確認時間？", "敢會當替導師確認時間？"),
    RegressionCase("teacher_class", "請協助導師確認時間。", "請鬥相共導師確認時間。"),
    RegressionCase("teacher_class", "可以幫家長確認時間嗎？", "會當替家長確認時間無？"),
    RegressionCase("teacher_class", "可不可以幫家長確認時間？", "敢會當替家長確認時間？"),
    RegressionCase("teacher_class", "請協助家長確認時間。", "請鬥相共家長確認時間。"),
    RegressionCase("teacher_class", "可以幫校長換班嗎？", "會當替校長換班無？"),
    RegressionCase("teacher_class", "可不可以幫校長換班？", "敢會當替校長換班？"),
    RegressionCase("teacher_class", "請協助校長換班。", "請鬥相共校長換班。"),
    RegressionCase("teacher_class", "可以幫助教換班嗎？", "會當替助教換班無？"),
    RegressionCase("teacher_class", "可不可以幫助教換班？", "敢會當替助教換班？"),
    RegressionCase("teacher_class", "請協助助教換班。", "請鬥相共助教換班。"),
    RegressionCase("teacher_class", "可以幫班導換班嗎？", "會當替班導換班無？"),
    RegressionCase("teacher_class", "可不可以幫班導換班？", "敢會當替班導換班？"),
    RegressionCase("teacher_class", "請協助班導換班。", "請鬥相共班導換班。"),
    # student_class — 學生課堂
    RegressionCase("student_class", "我要去學校。", "我欲去學校。"),
    RegressionCase("student_class", "學生在讀書。", "學生佇咧讀冊。"),
    RegressionCase("student_class", "同學在做功課。", "同學佇咧做功課。"),
    RegressionCase("student_class", "大家在學習。", "逐家佇咧學習。"),
    RegressionCase("student_class", "我今天要請假。", "我今仔日欲請假。"),
    RegressionCase("student_class", "學生今天請假。", "學生今仔日請假。"),
    RegressionCase("student_class", "他今天遲到了。", "伊今仔日遲到矣。"),
    RegressionCase("student_class", "明天要交作業。", "明仔載愛交作業。"),
    RegressionCase("student_class", "今天要帶課本嗎？", "今仔日愛帶課本無？"),
    RegressionCase("student_class", "下課後要留下來嗎？", "下課後愛留落來無？"),
    RegressionCase("student_class", "明天要帶聯絡簿嗎？", "明仔載愛帶聯絡簿無？"),
    RegressionCase("student_class", "我想晚一點到學校。", "我想欲較晏一點到學校。"),
    RegressionCase("student_class", "可以幫我請假嗎？", "會當替我請假無？"),
    RegressionCase("student_class", "我想跟老師請假。", "我想欲佮老師請假。"),
    RegressionCase("student_class", "可以幫我請事假嗎？", "會當替我請事假無？"),
    RegressionCase("student_class", "可以幫我查缺課紀錄嗎？", "會當替我查缺課紀錄無？"),
    RegressionCase("student_class", "可以幫我查課表嗎？", "會當替我查課表無？"),
    RegressionCase("student_class", "我想改上課地點。", "我想欲改上課地點。"),
    RegressionCase("student_class", "可以幫我查補課時間嗎？", "會當替我查補課時間無？"),
    RegressionCase("student_class", "可以幫我查校車時間嗎？", "會當替我查校車時間無？"),
    RegressionCase("student_class", "我想改上課時間。", "我想欲改上課時間。"),
    RegressionCase("student_class", "可以幫我查請假狀態嗎？", "會當替我查請假狀態無？"),
    RegressionCase("student_class", "可以幫同學換班嗎？", "會當替同學換班無？"),
    RegressionCase("student_class", "可不可以幫同學換班？", "敢會當替同學換班？"),
    RegressionCase("student_class", "請協助同學換班。", "請鬥相共同學換班。"),
    RegressionCase("student_class", "可以幫學生換班嗎？", "會當替學生換班無？"),
    RegressionCase("student_class", "可不可以幫學生換班？", "敢會當替學生換班？"),
    RegressionCase("student_class", "請協助學生換班。", "請鬥相共學生換班。"),
    RegressionCase("student_class", "請協助取消課程。", "請鬥相共取消課程。"),
    RegressionCase("student_class", "請協助申請獎學金。", "請鬥相共申請獎學金。"),
    # homework — 功課作業
    RegressionCase("homework", "今天有作業嗎？", "今仔日有作業無？"),
    RegressionCase("homework", "功課做完了嗎？", "功課做完矣無？"),
    RegressionCase("homework", "今天有什麼課？", "今仔日有啥課？"),
    RegressionCase("homework", "我忘記帶作業。", "我袂記得帶作業。"),
    RegressionCase("homework", "作業明天要交。", "作業明仔載愛交。"),
    RegressionCase("homework", "可以補交作業嗎？", "會當補交作業無？"),
    RegressionCase("homework", "可以幫我補交作業嗎？", "會當替我補交作業無？"),
    RegressionCase("homework", "我可以晚一點交作業嗎？", "我會當較晏一點交作業無？"),
    RegressionCase("homework", "作業要寫第幾頁？", "作業愛寫第幾頁？"),
    RegressionCase("homework", "作業可以晚一點交嗎？", "作業會當較晏一點交無？"),
    RegressionCase("homework", "我想改作業期限。", "我想欲改作業期限。"),
    RegressionCase("homework", "可以幫我查補交期限嗎？", "會當替我查補交期限無？"),
    RegressionCase("homework", "可以幫我查作業進度嗎？", "會當替我查作業進度無？"),
    RegressionCase("homework", "可以幫我查作業期限嗎？", "會當替我查作業期限無？"),
    RegressionCase("homework", "可以幫我查學費繳費期限嗎？", "會當替我查學費繳費期限無？"),
    # exam — 考試
    RegressionCase("exam", "我要考試了。", "我欲考試矣。"),
    RegressionCase("exam", "考試及格了嗎？", "考試及格矣無？"),
    RegressionCase("exam", "這次考得不好。", "這擺考得毋好。"),
    RegressionCase("exam", "明天要補考嗎？", "明仔載要補考無？"),
    RegressionCase("exam", "考試日期改到下星期。", "考試日期改做下禮拜。"),
    RegressionCase("exam", "我想改考試時間。", "我想欲改考試時間。"),
    RegressionCase("exam", "請問考場在哪裡？", "借問考場佇佗位？"),
    RegressionCase("exam", "老師說明天要小考。", "老師講明仔載愛考小考。"),
    RegressionCase("exam", "考試要帶鉛筆嗎？", "考試愛帶鉛筆無？"),
    RegressionCase("exam", "可以幫我查成績嗎？", "會當替我查成績無？"),
    RegressionCase("exam", "我想改補考日期。", "我想欲改補考日期。"),
    RegressionCase("exam", "可以幫我查成績單嗎？", "會當替我查成績單無？"),
    RegressionCase("exam", "可以幫我查考試地點嗎？", "會當替我查考試地點無？"),
    RegressionCase("exam", "可以幫我查補考地點嗎？", "會當替我查補考地點無？"),
    RegressionCase("exam", "可以幫我查段考時間嗎？", "會當替我查段考時間無？"),
    # campus — 校園設施
    RegressionCase("campus", "我去圖書館借書。", "我去圖冊館借冊。"),
    RegressionCase("campus", "請問廁所在哪裡？", "借問便所佇佗位？"),
    RegressionCase("campus", "這本書借我看看。", "這本冊借我看覓。"),
    RegressionCase("campus", "下課了嗎？", "下課矣無？"),
    RegressionCase("campus", "請問保健室在哪裡？", "借問保健室佇佗位？"),
    RegressionCase("campus", "老師辦公室在二樓。", "老師辦公室佇二樓。"),
    RegressionCase("campus", "操場可以借球嗎？", "操場會當借球無？"),
    RegressionCase("campus", "我要去圖書館還書。", "我欲去圖冊館還書。"),
    RegressionCase("campus", "我想借這本書。", "我想欲借這本冊。"),
    RegressionCase("campus", "可以幫我查教室位置嗎？", "會當替我查教室位置無？"),
    # ability — 能力表達
    RegressionCase("ability", "我不會說台語。", "我袂曉講台語。"),
    RegressionCase("ability", "我不太會游泳。", "我袂啥曉泅水。"),
    RegressionCase("ability", "我不會寫這題。", "我袂曉寫這題。"),
    RegressionCase("ability", "我聽不懂老師說什麼。", "我聽無老師講啥。"),
    RegressionCase("ability", "我看不懂這題。", "我看無這題。"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="學校/教育情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return SCHOOL_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in SCHOOL_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in SCHOOL_REGRESSION_CASES})
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
