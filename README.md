# taigi-converter（HanloFlow）

確定性、離線的**繁體華語 → 台語漢字**轉換器。核心 runtime 只讀取預先編譯的 JSON artifacts，不在匯入或初始化時修改安裝目錄。

## 安裝

```bash
pip install "taigi-converter @ git+https://github.com/Yizhe0407/HanloFlow.git"
```

核心轉換器沒有第三方 runtime dependency。若要接 Taibun 羅馬字整合，可安裝 optional extra：

```bash
pip install "taigi-converter[taibun] @ git+https://github.com/Yizhe0407/HanloFlow.git"
```

本機開發建議使用 `uv`：

```bash
uv sync --extra dev
```

## Python API

```python
from taigi_converter import TaigiConverter

converter = TaigiConverter()  # 建議重用；後續 instance 會共用已驗證的 runtime cache
print(converter.convert("你在做什麼？"))
# 你咧做啥物？

result = converter.convert("公車到站了", trace=True)
print(result.output)
print(result.matches)
print(result.rules_applied)
```

保留原始空白排版：

```python
converter.convert(text, profile={"preserve_spacing": True})
```

## CLI

安裝後：

```bash
taigi-converter "你在做什麼？"
taigi-converter --trace "你在做什麼？"
taigi-converter --explain "你在做什麼？"
taigi-converter --preserve-spacing "  你   好  "
taigi-converter --enqueue-review "待確認的輸入"
```

`--enqueue-review` 不會寫入 wheel 或 `site-packages`。CLI 預設寫到使用者 state 目錄；可用
`--review-data-dir` 或 `TAIGI_CONVERTER_STATE_DIR` 明確指定。Python API 若要 enqueue，必須傳入獨立的可寫目錄：

```python
converter = TaigiConverter(review_data_dir="var/taigi-review")
converter.convert(text, profile={"enqueue_review": True, "owner": "service"})
```

Repo 內也可執行相容入口：

```bash
python3 app.py "你在做什麼？"
```

## 專案架構與單一來源

```text
taigi_converter/                   # 唯一正式 Python 實作
  data/artifacts/*.json            # wheel/runtime 唯一資料；唯讀、已編譯
data/                              # 唯一原始資料來源，不放進 wheel
  lexicon_entries.jsonl
  rule_entries.jsonl
  core_lexicon.json
  char_verified_allowlist.txt
scripts/
  build_runtime_artifacts.py
  regression_runner.py
  run_all_regressions.py
  run_package_parity_regression.py
```

根目錄的 `converter.py`、`artifact_compiler.py` 等檔案只保留舊 import 的相容 wrapper；新程式一律從 `taigi_converter` 匯入。

## Runtime 與 artifacts

- 一般 `TaigiConverter()` **不會**自動重編資料，也不會寫入 site-packages。
- manifest 以 compiler version、source SHA-256 與每個 artifact 的 SHA-256 驗證完整性。開發模式會重建損毀或缺失的 artifact。
- 建置時先替換資料檔，最後原子替換 manifest；reader 若撞到建置中的混合世代會重試，不會靜默載入半套資料。
- 開發工具需要自動準備時，才明確使用 `auto_prepare=True` 並分開指定 source/output：

```python
TaigiConverter(
    data_dir="build/runtime-data",
    auto_prepare=True,
    source_data_dir="data",
)
```

## 修改詞典或規則

1. 只修改 `data/` 下的 source data。
2. 不手動修改 `taigi_converter/data/artifacts/*.json`。
3. 重建並啟用衝突檢查：

```bash
python3 scripts/build_runtime_artifacts.py --fail-on-mask
```

Compiler 會 fail-fast 檢查 schema、重複 ID、重複 rule pattern、regex 語法，以及同順位卻指向不同 target 的 active 詞條。

## 驗證

```bash
# 語法、lint、unit tests
python3 -m compileall -q taigi_converter scripts tests
uv run ruff check taigi_converter scripts tests
uv run pytest

# 5,160 筆 exact-match regressions
python3 scripts/run_all_regressions.py

# 確認 artifacts 可重現
python3 scripts/build_runtime_artifacts.py --fail-on-mask
git diff --exit-code -- taigi_converter/data/artifacts

# wheel 與唯讀 runtime 驗證
uv run --with hatchling python -m build --no-isolation
python3 scripts/run_package_parity_regression.py --wheel dist/taigi_converter-0.1.0-py3-none-any.whl
```

單一情境 runner 仍可直接執行，並支援 `--category`、`--rounds`、`--fail-fast` 等參數。

## Review queue 一致性

Review queue 使用作業系統管理的跨 process advisory lock；程序異常退出時鎖會自動釋放，不使用有 stale-reclaim 競態的刪檔式鎖。append 採單筆完整寫入與 `fsync`。批次決策會在同一把鎖下更新 queue、lexicon、audit，並透過 durable transaction journal 在程序中斷後自動重播，避免遺失已接受的決策。runtime 與 review state 必須分離，避免嘗試寫入唯讀安裝目錄。

## 維護文件

- `AGENT.md`：工作流程與必跑驗證
- `.claude/rules/curation.md`：詞條策略
- `.claude/rules/validation.md`：驗證規範
- `.claude/rules/progress.md`：目前狀態摘要
