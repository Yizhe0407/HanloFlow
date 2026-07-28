# Hanloflow 工作指引

這是一個**華語 → 台語漢字**轉換器專案（hanloflow）。

@AGENT.md

@.claude/rules/progress.md

## 常用指令速查

```bash
# 單句轉換
python3 app.py "你在做什麼？"

# 附 trace
python3 app.py --trace "你在做什麼？"

# 重編 artifacts（改完資料後必跑）
python3 scripts/build_runtime_artifacts.py --data-dir data

# 語法健康檢查
python3 -m py_compile artifact_compiler.py converter.py app.py scripts/build_runtime_artifacts.py

# Regression tests
python3 scripts/run_bus_regression.py
python3 scripts/run_medical_regression.py
python3 scripts/run_transport_regression.py
python3 scripts/run_conversation_regression.py
python3 scripts/run_restaurant_regression.py
python3 scripts/run_shopping_regression.py
python3 scripts/run_hotel_regression.py
python3 scripts/run_taxi_regression.py
python3 scripts/run_bank_regression.py
python3 scripts/run_school_regression.py
python3 scripts/run_family_regression.py
python3 scripts/run_workplace_regression.py
python3 scripts/run_package_parity_regression.py
```

## 資料修正優先順序

1. `data/lexicon_entries.jsonl` — 最常改的地方，優先調整詞條
   - 固定句型或常見說法：優先補 `sentence` 或長 `phrase`
   - 可重用穩定片段：補 `phrase`，這是日常優化的主要層級
   - 單字 `char` 最後才補；影響面最大，容易誤傷地名、人名與複合詞
2. `data/rule_entries.jsonl` — 只有多句共享同一個穩定文法模式時才動
3. **不要手動編輯** `data/artifacts/*`（由 build script 生成）
4. 修改根目錄 `data/` 後，同步 `taigi_converter/data/` 並重編兩邊 artifacts

## 回應語言

除非使用者明確要求，一律用**繁體中文**回應。
程式碼、指令、檔案路徑、識別符保持原形。
