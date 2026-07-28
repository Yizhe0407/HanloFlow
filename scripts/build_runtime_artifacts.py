from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taigi_converter.artifact_compiler import compile_runtime_artifacts, ensure_runtime_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="編譯台語轉換 runtime artifacts")
    parser.add_argument("--data-dir", default="data", help="原始資料目錄")
    parser.add_argument(
        "--output-data-dir",
        default="taigi_converter/data",
        help="runtime data 輸出目錄",
    )
    parser.add_argument("--fail-on-mask", action="store_true", help="偵測到遮蔽規則時失敗")
    parser.add_argument("--prepare", action="store_true", help="缺檔時先執行 migration / 補齊")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_data_dir = Path(args.output_data_dir)
    if args.prepare:
        manifest = ensure_runtime_ready(
            data_dir=data_dir,
            output_data_dir=output_data_dir,
            fail_on_mask=args.fail_on_mask,
        )
    else:
        manifest = compile_runtime_artifacts(
            data_dir=data_dir,
            output_data_dir=output_data_dir,
            fail_on_mask=args.fail_on_mask,
        )

    print("artifacts built")
    for key in sorted(manifest):
        print(f"{key}: {manifest[key]}")


if __name__ == "__main__":
    main()
