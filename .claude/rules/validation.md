# 驗證流程規則

每次修改資料後，**必須**依序執行以下步驟。

## 標準驗證流程

```bash
# 1. 重編 artifacts
python3 scripts/build_runtime_artifacts.py --data-dir data

# 2. 語法健康檢查
python3 -m py_compile artifact_compiler.py converter.py app.py scripts/build_runtime_artifacts.py

# 3. 測試目標句子
python3 app.py "原本失敗的句子"

# 4. Smoke test（至少跑這四種）
python3 app.py "短語本身"
python3 app.py "包含短語的短句"
python3 app.py "包含短語的長句，帶標點。"
python3 app.py "結合兩個近期新增短語的句子"
```

## Regression Tests（重要改動後跑）

```bash
python3 scripts/run_bus_regression.py
python3 scripts/run_medical_regression.py
python3 scripts/run_transport_regression.py
```

## 效能守則

- 暖路徑（warm-path）轉換需 < 0.05 秒
- 避免在 runtime 增加重計算邏輯
- 詞條資料調整比演算法改動更安全

## Debug 工具

```bash
# 取得 trace JSON
python3 app.py --trace "問題句子"

# 人類可讀 explain
python3 app.py --explain "問題句子"
```
