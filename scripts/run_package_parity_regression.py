from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regression_runner import REGRESSION_SCRIPT_NAMES, load_suite_cases

REQUIRED_ARTIFACTS = {
    "char_map.json",
    "entry_table.json",
    "manifest.json",
    "override_index.json",
    "phrase_trie.json",
    "rule_plan.json",
}
FORBIDDEN_SOURCE_DATA = {
    "char_verified_allowlist.txt",
    "core_lexicon.json",
    "lexicon_entries.jsonl",
    "rule_entries.jsonl",
}


def _default_wheel() -> Path | None:
    wheels = sorted((REPO_ROOT / "dist").glob("taigi_converter-*.whl"))
    return wheels[-1] if wheels else None


def _set_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(0o555)


def _run_extracted_package(package_root: Path, sources: list[str]) -> list[str]:
    child = r'''
import json
import sys
sys.path.insert(0, sys.argv[1])
from taigi_converter import TaigiConverter
converter = TaigiConverter()
print(json.dumps([converter.convert(text) for text in json.load(sys.stdin)], ensure_ascii=False))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", child, str(package_root)],
        input=json.dumps(sources, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
        cwd=package_root.parent,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if completed.returncode:
        raise RuntimeError(
            "wheel isolated smoke 失敗\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="驗證 wheel 內容、唯讀執行與 regression parity")
    parser.add_argument("--wheel", type=Path, default=_default_wheel())
    args = parser.parse_args()
    if args.wheel is None or not args.wheel.exists():
        parser.error("找不到 wheel；請先執行 `python3 -m build --wheel` 或傳入 --wheel")

    cases = [
        case
        for script_name in REGRESSION_SCRIPT_NAMES
        for case in load_suite_cases(REPO_ROOT / "scripts" / script_name)
    ]

    with zipfile.ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
        artifact_names = {
            Path(name).name
            for name in names
            if name.startswith("taigi_converter/data/artifacts/") and not name.endswith("/")
        }
        missing = REQUIRED_ARTIFACTS - artifact_names
        unexpected = artifact_names - REQUIRED_ARTIFACTS
        leaked = {Path(name).name for name in names} & FORBIDDEN_SOURCE_DATA
        if missing:
            raise RuntimeError(f"wheel 缺少 runtime artifacts: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(f"wheel 含未預期 runtime 檔案: {sorted(unexpected)}")
        if leaked:
            raise RuntimeError(f"wheel 不應包含 source data: {sorted(leaked)}")

        with tempfile.TemporaryDirectory(prefix="hanloflow-wheel-") as temp:
            package_root = Path(temp) / "wheel"
            archive.extractall(package_root)
            _set_tree_read_only(package_root)
            outputs = _run_extracted_package(package_root, [case.source for case in cases])

    failures = []
    for case, output in zip(cases, outputs, strict=True):
        if output != case.expected:
            failures.append(
                {
                    "category": case.category,
                    "source": case.source,
                    "expected": case.expected,
                    "wheel_output": output,
                }
            )
    print(
        {
            "wheel": str(args.wheel),
            "case_count": len(cases),
            "failed": len(failures),
            "read_only_runtime": True,
            "source_data_excluded": True,
        }
    )
    for failure in failures[:20]:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
