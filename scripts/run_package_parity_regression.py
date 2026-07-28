from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from converter import TaigiConverter as RootConverter
from taigi_converter import TaigiConverter as PackageConverter


REGRESSION_SCRIPTS = (
    "run_bus_regression.py",
    "run_medical_regression.py",
    "run_transport_regression.py",
    "run_conversation_regression.py",
    "run_restaurant_regression.py",
    "run_shopping_regression.py",
    "run_hotel_regression.py",
    "run_taxi_regression.py",
    "run_bank_regression.py",
    "run_school_regression.py",
    "run_family_regression.py",
    "run_workplace_regression.py",
)


def _load_cases(script_path: Path) -> list[object]:
    module_name = f"_hanloflow_parity_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入 regression script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    case_lists = [
        value
        for name, value in vars(module).items()
        if name.endswith("_REGRESSION_CASES") and isinstance(value, list)
    ]
    if len(case_lists) != 1:
        raise RuntimeError(
            f"{script_path} 應恰好定義一個 *_REGRESSION_CASES，實際為 {len(case_lists)}"
        )
    return case_lists[0]


def main() -> None:
    root_converter = RootConverter()
    package_converter = PackageConverter()
    failures: list[dict[str, str]] = []
    case_count = 0

    for script_name in REGRESSION_SCRIPTS:
        script_path = REPO_ROOT / "scripts" / script_name
        for case in _load_cases(script_path):
            case_count += 1
            root_output = root_converter.convert(case.source)
            package_output = package_converter.convert(case.source)
            if root_output == case.expected and package_output == case.expected:
                continue
            failures.append(
                {
                    "script": script_name,
                    "category": case.category,
                    "source": case.source,
                    "expected": case.expected,
                    "root_output": root_output,
                    "package_output": package_output,
                }
            )

    print({"case_count": case_count, "failed": len(failures)})
    for failure in failures[:20]:
        print(failure)
    if failures:
        raise SystemExit(1)
    print({"status": "ok", "root_package_parity": True})


if __name__ == "__main__":
    main()
