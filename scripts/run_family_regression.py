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


FAMILY_REGRESSION_CASES: list[RegressionCase] = [
    # parent_child — 親子日常
    RegressionCase("parent_child", "媽媽在煮飯。", "阿母佇咧煮飯。"),
    RegressionCase("parent_child", "爸爸去上班了。", "阿爸去上班矣。"),
    RegressionCase("parent_child", "孩子在哭。", "囡仔佇咧哭。"),
    RegressionCase("parent_child", "小孩在玩玩具。", "囡仔佇咧玩𨑨迌物仔。"),
    RegressionCase("parent_child", "孩子該起床了。", "囡仔愛起床矣。"),
    RegressionCase("parent_child", "下午去接孩子。", "下晝去接囡仔。"),
    RegressionCase("parent_child", "幫孩子換衣服。", "幫囡仔換衫。"),
    RegressionCase("parent_child", "不要哭了。", "毋通哭矣。"),
    RegressionCase("parent_child", "我帶小孩去學校。", "我帶囡仔去學校。"),
    RegressionCase("parent_child", "可以幫我接小孩嗎？", "會當替我接囡仔無？"),
    RegressionCase("parent_child", "我晚點帶小孩回家。", "我較晏帶囡仔轉去厝裡。"),
    RegressionCase("parent_child", "我晚點去接小孩。", "我較晏去接囡仔。"),
    # health_care — 照護
    RegressionCase("health_care", "孩子發燒了。", "囡仔發燒矣。"),
    RegressionCase("health_care", "幫孩子洗澡。", "幫囡仔洗身軀。"),
    RegressionCase("health_care", "帶孩子去看醫生。", "帶囡仔去予醫生看。"),
    RegressionCase("health_care", "孩子在睡午覺。", "囡仔佇咧睏晝。"),
    RegressionCase("health_care", "孩子要吃藥。", "囡仔愛食藥仔。"),
    RegressionCase("health_care", "藥要按時吃。", "藥仔愛照時間食。"),
    RegressionCase("health_care", "媽媽要休息。", "阿母欲歇睏。"),
    RegressionCase("health_care", "我陪你去看醫生。", "我陪你去予醫生看。"),
    RegressionCase("health_care", "媽媽說晚上要吃藥。", "阿母講暗時愛食藥仔。"),
    RegressionCase("health_care", "爸爸明天要回診。", "阿爸明仔載愛回診。"),
    RegressionCase("health_care", "今天要帶小孩去看醫生。", "今仔日愛帶囡仔去予醫生看。"),
    RegressionCase("health_care", "小孩肚子痛。", "囡仔腹肚疼。"),
    # siblings — 兄弟姐妹
    RegressionCase("siblings", "姐姐在讀書。", "阿姊佇咧讀冊。"),
    RegressionCase("siblings", "弟弟在做功課。", "阿弟仔佇咧做功課。"),
    RegressionCase("siblings", "妹妹在睡午覺。", "小妹仔佇咧睏晝。"),
    RegressionCase("siblings", "哥哥去上學了。", "阿兄去上課矣。"),
    RegressionCase("siblings", "哥哥妹妹在吵架。", "阿兄小妹仔佇咧冤家。"),
    RegressionCase("siblings", "姐姐幫弟弟拿書包。", "阿姊幫阿弟仔提冊包。"),
    RegressionCase("siblings", "兄弟姐妹要一起分享玩具。", "兄弟姊妹愛鬥陣分享𨑨迌物仔。"),
    RegressionCase("siblings", "妹妹不想寫功課。", "小妹仔毋想欲寫功課。"),
    RegressionCase("siblings", "請幫我照顧弟弟。", "請替我照顧阿弟仔。"),
    # grandparents — 祖父母
    RegressionCase("grandparents", "爺爺在下棋。", "阿公佇咧行棋。"),
    RegressionCase("grandparents", "奶奶在洗碗。", "阿媽佇咧洗碗。"),
    RegressionCase("grandparents", "帶孩子去找爺爺奶奶。", "帶囡仔去找阿公阿媽。"),
    RegressionCase("grandparents", "爺爺在看電視。", "阿公佇咧看電視。"),
    RegressionCase("grandparents", "明天去看爺爺奶奶。", "明仔載去看阿公阿媽。"),
    RegressionCase("grandparents", "阿公身體還好嗎？", "阿公身體敢猶好？"),
    RegressionCase("grandparents", "奶奶要去散步。", "阿媽欲去散步。"),
    RegressionCase("grandparents", "打電話給爺爺。", "拍電話予阿公。"),
    # daily — 家庭日常
    RegressionCase("daily", "老婆在洗衣服。", "某佇咧洗衫。"),
    RegressionCase("daily", "老公在工作。", "翁婿佇咧工作。"),
    RegressionCase("daily", "女兒很乖。", "查某囝真乖。"),
    RegressionCase("daily", "爸爸在喝茶。", "阿爸佇咧啉茶。"),
    RegressionCase("daily", "爸爸要出門了。", "阿爸欲出門矣。"),
    RegressionCase("daily", "晚餐好了。", "暗頓好矣。"),
    RegressionCase("daily", "我們一起吃飯。", "咱鬥陣食飯。"),
    RegressionCase("daily", "記得洗碗。", "記著洗碗。"),
    RegressionCase("daily", "我可以晚一點回家嗎？", "我會當較晏一點轉去厝裡無？"),
    RegressionCase("daily", "請幫我買晚餐。", "請替我買暗頓。"),
    RegressionCase("daily", "晚餐要吃什麼？", "暗頓欲食啥？"),
    RegressionCase("daily", "晚餐吃什麼？", "暗頓食啥？"),
    RegressionCase("daily", "可以幫我買早餐嗎？", "會當替我買早頓無？"),
    RegressionCase("daily", "可以幫我煮晚餐嗎？", "會當替我煮暗頓無？"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="家庭/親子情境 regression runner")
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    return parser


def _selected_cases(categories: list[str]) -> list[RegressionCase]:
    if not categories:
        return FAMILY_REGRESSION_CASES
    wanted = set(categories)
    return [case for case in FAMILY_REGRESSION_CASES if case.category in wanted]


def main() -> int:
    args = _build_parser().parse_args()
    categories = sorted({case.category for case in FAMILY_REGRESSION_CASES})
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
