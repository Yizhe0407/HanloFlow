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
    # student_class — 學生課堂
    RegressionCase("student_class", "我要去學校。", "我欲去學校。"),
    RegressionCase("student_class", "學生在讀書。", "學生佇咧讀冊。"),
    RegressionCase("student_class", "同學在做功課。", "同學佇咧做功課。"),
    RegressionCase("student_class", "大家在學習。", "逐家佇咧學習。"),
    # homework — 功課作業
    RegressionCase("homework", "今天有作業嗎？", "今仔日有作業無？"),
    RegressionCase("homework", "功課做完了嗎？", "功課做完矣無？"),
    RegressionCase("homework", "今天有什麼課？", "今仔日有啥課？"),
    RegressionCase("homework", "我忘記帶作業。", "我袂記得帶作業。"),
    RegressionCase("homework", "作業明天要交。", "作業明仔載愛交。"),
    RegressionCase("homework", "可以補交作業嗎？", "會當補交作業無？"),
    # exam — 考試
    RegressionCase("exam", "我要考試了。", "我欲考試矣。"),
    RegressionCase("exam", "考試及格了嗎？", "考試及格矣無？"),
    RegressionCase("exam", "這次考得不好。", "這擺考得毋好。"),
    RegressionCase("exam", "明天要補考嗎？", "明仔載要補考無？"),
    RegressionCase("exam", "考試日期改到下星期。", "考試日期改做下禮拜。"),
    RegressionCase("exam", "請問考場在哪裡？", "借問考場佇佗位？"),
    # campus — 校園設施
    RegressionCase("campus", "我去圖書館借書。", "我去圖冊館借冊。"),
    RegressionCase("campus", "請問廁所在哪裡？", "借問便所佇佗位？"),
    RegressionCase("campus", "這本書借我看看。", "這本冊借我看覓。"),
    RegressionCase("campus", "下課了嗎？", "下課矣無？"),
    RegressionCase("campus", "請問保健室在哪裡？", "借問保健室佇佗位？"),
    RegressionCase("campus", "老師辦公室在二樓。", "老師辦公室佇二樓。"),
    RegressionCase("campus", "操場可以借球嗎？", "操場會當借球無？"),
    RegressionCase("campus", "我要去圖書館還書。", "我欲去圖冊館還書。"),
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
