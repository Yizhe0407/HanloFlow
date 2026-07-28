# 驗證規範

## 必跑順序

```bash
python3 scripts/build_runtime_artifacts.py --fail-on-mask
python3 -m compileall -q taigi_converter scripts tests
uv run ruff check taigi_converter scripts tests
uv run pytest
python3 scripts/run_all_regressions.py
```

修改 package、runtime loader、compiler、artifact 格式或 `pyproject.toml` 時，另跑：

```bash
uv run --with hatchling python -m build --no-isolation
python3 scripts/run_package_parity_regression.py --wheel dist/taigi_converter-0.1.0-py3-none-any.whl
```

## 驗收條件

- compiler：schema、duplicate ID/pattern、regex、同順位 target conflict 全部通過。
- artifacts：重建可重現；manifest source digest 與逐檔 checksum 完整。
- unit tests：唯讀 runtime、cache、trace、spacing、review queue 併發與 journal recovery 全綠。
- regressions：12 suites、4,179 cases，exact match 0 failures。
- wheel：包含六個 runtime JSON，不包含四份約 10 MB source data。
- isolated runtime：repo 外、唯讀 package tree 可初始化並完成全部 parity cases。
- Git diff：只能保留本次有意變更；不得混入 `__pycache__`、build、dist 或暫存 lock/journal。

## 資料變更注意事項

- 不用整份 JSONL reformat；只改必要行，降低 review 噪音。
- 停用詞條而非刪除，保留 `updated_by`、`updated_at` 與原因。
- source data 改完只重建 package artifacts，不再維護第二份 source data。
- regression case 應放在對應 suite；runner 邏輯集中於 `scripts/regression_runner.py`。
