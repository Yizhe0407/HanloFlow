from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from taigi_converter import TaigiConverter

REGRESSION_SCRIPT_NAMES = (
    "run_bank_regression.py",
    "run_bus_regression.py",
    "run_conversation_regression.py",
    "run_family_regression.py",
    "run_hotel_regression.py",
    "run_medical_regression.py",
    "run_restaurant_regression.py",
    "run_school_regression.py",
    "run_shopping_regression.py",
    "run_taxi_regression.py",
    "run_transport_regression.py",
    "run_workplace_regression.py",
)


@dataclass(frozen=True)
class RegressionCase:
    category: str
    source: str
    expected: str


@dataclass(frozen=True)
class RegressionFailure:
    index: int
    case: RegressionCase
    output: str


def load_suite_module(script_path: Path) -> ModuleType:
    module_name = f"_hanloflow_regression_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入 regression script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def load_suite_cases(script_path: Path) -> list[RegressionCase]:
    module = load_suite_module(script_path)
    case_lists = [
        value
        for name, value in vars(module).items()
        if name.endswith("_REGRESSION_CASES") and isinstance(value, list)
    ]
    if len(case_lists) != 1:
        raise RuntimeError(
            f"{script_path} 應恰好定義一個 *_REGRESSION_CASES，實際為 {len(case_lists)}"
        )
    cases = case_lists[0]
    if not all(isinstance(case, RegressionCase) for case in cases):
        raise TypeError(f"{script_path} 含非 RegressionCase 項目")
    return cases


def run_cases(
    cases: Sequence[RegressionCase],
    *,
    converter: TaigiConverter | None = None,
    rounds: int = 1,
    show_pass: bool = False,
    fail_fast: bool = False,
) -> tuple[list[RegressionFailure], list[float]]:
    if rounds < 1:
        raise ValueError("rounds 必須至少為 1")
    active_converter = converter or TaigiConverter()
    failures: list[RegressionFailure] = []
    latencies_ms: list[float] = []
    for round_index in range(1, rounds + 1):
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            output = active_converter.convert(case.source)
            latencies_ms.append((time.perf_counter() - started) * 1000)
            if output != case.expected:
                failures.append(RegressionFailure(index, case, output))
                if fail_fast:
                    return failures, latencies_ms
            elif show_pass:
                print(
                    f"PASS round={round_index} idx={index} "
                    f"category={case.category} src={case.source}"
                )
        if failures:
            break
    return failures, latencies_ms


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(latencies_ms)
    p95_index = max(int(len(ordered) * 0.95) - 1, 0)
    return {
        "mean_ms": round(mean(ordered), 4),
        "p95_ms": round(ordered[p95_index], 4),
        "max_ms": round(max(ordered), 4),
    }


def run_regression_cli(
    cases: Sequence[RegressionCase],
    *,
    description: str,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--rounds", type=int, default=1, help="重複跑幾輪 exact-match 回歸")
    parser.add_argument("--category", action="append", default=[], help="只跑指定 category，可重複傳入")
    parser.add_argument("--list-categories", action="store_true", help="列出所有 category")
    parser.add_argument("--show-pass", action="store_true", help="顯示每筆通過案例")
    parser.add_argument("--fail-fast", action="store_true", help="遇到第一個 mismatch 就停止")
    args = parser.parse_args()

    categories = sorted({case.category for case in cases})
    if args.list_categories:
        print("\n".join(categories))
        return 0

    wanted = set(args.category)
    selected = [case for case in cases if not wanted or case.category in wanted]
    if not selected:
        print("no cases selected")
        return 1

    counts = Counter(case.category for case in selected)
    print(
        {
            "rounds": args.rounds,
            "case_count": len(selected),
            "categories": dict(sorted(counts.items())),
        }
    )
    failures, latencies = run_cases(
        selected,
        rounds=args.rounds,
        show_pass=args.show_pass,
        fail_fast=args.fail_fast,
    )
    print({"failed": len(failures)})
    for failure in failures[:10]:
        print(f"FAIL idx={failure.index} category={failure.case.category}")
        print(f"  src: {failure.case.source}")
        print(f"  exp: {failure.case.expected}")
        print(f"  out: {failure.output}")
    if failures:
        return 1

    print(
        {
            "status": "ok",
            "rounds": args.rounds,
            "case_count": len(selected),
            "total_checks": len(latencies),
            **latency_summary(latencies),
        }
    )
    return 0
