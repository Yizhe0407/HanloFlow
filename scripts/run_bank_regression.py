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
    RegressionCase("bank_account", "我要補摺。", "我欲補摺。"),
    RegressionCase("bank_account", "我的卡片到期了。", "我的卡片到期矣。"),
    # bank_transaction — 存提匯
    RegressionCase("bank_transaction", "我要存款。", "我欲存款。"),
    RegressionCase("bank_transaction", "我要領錢。", "我欲領錢。"),
    RegressionCase("bank_transaction", "我要匯款。", "我欲匯款。"),
    RegressionCase("bank_transaction", "我要換外幣。", "我欲換外票。"),
    RegressionCase("bank_transaction", "我要查詢餘額。", "我欲查詢餘額。"),
    # bank_service — 服務詢問
    RegressionCase("bank_service", "請問ATM在哪裡？", "借問ATM佇佗位？"),
    RegressionCase("bank_service", "請問ATM怎麼用？", "借問ATM按怎用？"),
    RegressionCase("bank_service", "請問手續費多少？", "借問手續費偌濟？"),
    RegressionCase("bank_service", "請問利率是多少？", "借問利率是偌濟？"),
    RegressionCase("bank_service", "需要預約嗎？", "需要預約無？"),
    # postal — 郵局寄件
    RegressionCase("postal", "我要寄掛號信。", "我欲寄掛號信。"),
    RegressionCase("postal", "我要寄包裹。", "我欲寄包裹。"),
    RegressionCase("postal", "這個多重？", "這个偌重？"),
    RegressionCase("postal", "要幾天到？", "要幾工到？"),
    RegressionCase("postal", "請問郵局在哪裡？", "借問郵局佇佗位？"),
    RegressionCase("postal", "我要買郵票。", "我欲買郵票。"),
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
