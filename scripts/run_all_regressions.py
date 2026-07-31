from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regression_runner import (
    latency_summary,
    load_all_regression_cases,
    run_cases,
)
from taigi_converter import TaigiConverter


def main() -> int:
    parser = argparse.ArgumentParser(description="執行全部 HanloFlow exact-match regressions")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    converter = TaigiConverter()
    total_cases = 0
    all_latencies: list[float] = []
    failed_suites = 0
    suite_cases: dict[str, list] = {}
    for located in load_all_regression_cases(REPO_ROOT / "scripts"):
        suite_cases.setdefault(located.script, []).append(located.case)
    for script_name, cases in suite_cases.items():
        failures, latencies = run_cases(cases, converter=converter, fail_fast=args.fail_fast)
        total_cases += len(cases)
        all_latencies.extend(latencies)
        print({"suite": script_name, "case_count": len(cases), "failed": len(failures)})
        for failure in failures[:10]:
            print(
                {
                    "category": failure.case.category,
                    "source": failure.case.source,
                    "expected": failure.case.expected,
                    "output": failure.output,
                }
            )
        if failures:
            failed_suites += 1
            if args.fail_fast:
                break

    print(
        {
            "status": "ok" if failed_suites == 0 else "failed",
            "suite_count": len(suite_cases),
            "case_count": total_cases,
            "failed_suites": failed_suites,
            **latency_summary(all_latencies),
        }
    )
    return 1 if failed_suites else 0


if __name__ == "__main__":
    raise SystemExit(main())
