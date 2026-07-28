# HanloFlow 工作指引

HanloFlow 是**繁體華語 → 台語漢字**的確定性轉換器。除非使用者明確要求，回覆一律使用繁體中文；程式碼、識別符與路徑維持原文。

## 單一來源原則

- 正式程式碼：`taigi_converter/`
- 原始資料：`data/`
- 可發布 runtime：`taigi_converter/data/artifacts/`
- 根目錄同名 Python 檔只做 backward-compatible wrapper，不新增實作。
- 不複製 source data 到 package，不手動修改 artifacts。

## 常用指令

```bash
python3 app.py "你在做什麼？"
python3 app.py --trace "你在做什麼？"

python3 scripts/build_runtime_artifacts.py --fail-on-mask
python3 -m compileall -q taigi_converter scripts tests
uv run ruff check taigi_converter scripts tests
uv run pytest
python3 scripts/run_all_regressions.py

uv run --with hatchling python -m build --no-isolation
python3 scripts/run_package_parity_regression.py --wheel dist/taigi_converter-0.1.0-py3-none-any.whl
```

## 資料修改優先順序

1. `data/lexicon_entries.jsonl`
   - 可重用穩定片段優先用 `phrase`
   - 高耦合完整句才用 `sentence`
   - `char` 影響面最大，最後才用
2. `data/rule_entries.jsonl`
   - 只有多句共享同一穩定文法模式時才新增
3. 修改資料後必須重建 artifacts 並跑完整 regressions。

## Runtime 原則

- 正常 runtime 必須唯讀，`TaigiConverter()` 不得隱式編譯或寫入 package。
- 只允許開發/build 工具明確使用 `auto_prepare=True`。
- manifest/compiler/source/artifact checksum 不符必須 fail-fast 或等待一致世代，不能容忍半套資料。
- 重用 converter；runtime cache 不可因轉換操作被 mutation，也不可快取 instance-specific 的 review 路徑。
- Review queue 必須使用獨立可寫 state 目錄，不得回寫 package runtime。

## 完成定義

涉及程式或資料的工作，在回報完成前至少執行：

1. `python3 scripts/build_runtime_artifacts.py --fail-on-mask`
2. `python3 -m compileall -q taigi_converter scripts tests`
3. `uv run ruff check taigi_converter scripts tests`
4. `uv run pytest`
5. `python3 scripts/run_all_regressions.py`
6. 若動到 package/runtime：建 wheel 並執行 package parity/read-only 驗證
7. 重建後確認 tracked artifacts 沒有未解釋的 diff

不得只跑單一 smoke test後宣稱全部完成。
