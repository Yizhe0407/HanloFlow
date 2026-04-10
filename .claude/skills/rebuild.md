---
name: rebuild
description: 重編 artifacts、語法健康檢查，可選跑 smoke test
---

執行標準重編流程。

**步驟 1：重編 artifacts**

```bash
python3 scripts/build_runtime_artifacts.py --data-dir data
```

**步驟 2：語法健康檢查**

```bash
python3 -m py_compile artifact_compiler.py converter.py app.py scripts/build_runtime_artifacts.py
```

兩步驟都完成後，彙報：
- 是否成功（有無 error / warning）
- artifacts 的輸出摘要（如有印出詞條數、rule 數等）

**步驟 3（選用）：Smoke test**

若使用者有提供測試句子（`$ARGUMENTS`），額外執行：

```bash
python3 app.py "$ARGUMENTS"
```

並顯示轉換結果。

若過程中任一步驟失敗，立即停止並顯示完整錯誤訊息，不繼續執行後續步驟。
