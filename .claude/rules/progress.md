# 現況與下一步方向

> 精簡接手版。只保留目前狀態、最近重要變更與下一步；舊的逐輪細節已封存到 `.claude/rules/archive/progress-2026-05.md`。
> 除非需要追溯某輪詞條/規則的歷史原因，不要預設讀 archive，避免消耗大量 LLM token。

## 目前辭典狀態（2026-05-23）

| 類型 | 數量 |
|------|------|
| 總詞條（active runtime） | 20,309 |
| 人工驗證（trust: human active rows） | 11,857 |
| base seed（trust: seed，低信任） | source 仍有短詞 active；runtime 以 policy 過濾高風險 seed |
| 最新資料 round | round410 |

所有迴歸測試全部通過（共 3205 筆）：
bus 549、medical 220、transport 269、conversation 937、restaurant 98、shopping 188、hotel 171、taxi 83、bank 146、school 110、family 88、workplace 346。

## 最近重要變更

### 2026-05-23：CTS 無菸城市與南港城市治理詞補強（round410）

- 來源：華視台語新聞「無菸城藍綠轟！」，內容涵蓋台北市無菸城市政策、議會質詢、南港城市治理與首都市長戰。
- 補強 `無菸城藍綠轟`、`無菸城市政策`、`社群平台`、`辦公孤島`、`黑鄉變成潮城`、`首都市長戰的攻防之一` 等正式新聞詞。
- 修正正式語境被口語化或誤轉的問題，例如 `他的→伊的`、`先前→原前`、`高價→懸價`、`晚上→暗時`、`從→對`、`攻防→柔道`。
- `scripts/run_conversation_regression.py` 新增 19 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-23：CTS 安平廟埕震天祭詞補強（round409）

- 來源：華視台語新聞「安平廟埕震天祭」明登場，解謎走讀認識信仰。
- 補強 `安平廟埕震天祭`、`安平開台天后宮`、`角頭廟宇`、`實境解謎走讀`、`城隍爺夫人`、`踩街嘉年華` 等活動/信仰詞。
- 修正新聞正式語境被口語化的問題，例如 `這次→這擺`、`我們→咱`、`跟→佮`、`其實→論真`、`這個→這个`、`將在→欲在`。
- `scripts/run_conversation_regression.py` 新增 18 筆 `news_cts_terms` case；12 支 regression 全綠。

### 2026-05-23：converter runtime 效能優化

- `_mask_protected_terms()` 減少重複 runtime phrase overlap 檢查。
- `_apply_rules()` 載入時排除 disabled/空 pattern 規則；literal 和純文字 regex 先用 `in` 快速判斷。
- 300 次混合句 benchmark：平均約 `0.330 ms` 降到 `0.285 ms`，p95 約 `0.478 ms` 降到 `0.421 ms`。
- 12 支 regression 全綠。

### 2026-05-23：README 補資料檔案使用順序

- README 新增 `資料檔案使用順序與功能`。
- 說明 `lexicon_entries.jsonl`、`rule_entries.jsonl`、`core_lexicon.json`、`char_verified_allowlist.txt`、`data/artifacts/*` 的使用順序與功能。

### 2026-05-23：CTS 新聞詞補強 round404-round410

- 補強華視台語新聞相關專有詞、地名、人名、機關名與新聞語境 passthrough。
- 最新 round410：無菸城市與南港城市治理詞補強。
- round409：安平廟埕震天祭與解謎走讀活動詞補強。
- round408：花蓮免費營養午餐採購弊案詞補強。
- round407：彰化縣長提名與藍營選戰詞補強。
- round406：馬英九基金會人事風波與調查聲明詞補強。
- round405：北市公車自撞山壁事故詞補強。
- round404：前鎮百貨系財神殿與參拜詞補強。
- 詳細逐條新增詞條、誤轉修正與來源 URL 請看 archive。

## 維護原則

1. 優先改 `data/lexicon_entries.jsonl`：單詞、片語、句子轉錯時先補詞條。
2. 謹慎改 `data/rule_entries.jsonl`：只有多句共享同一個穩定文法模式時才動規則。
3. 不手動改 `data/artifacts/*`：改來源資料後重編 artifacts。
4. 修改根目錄 `data/` 後，同步 `taigi_converter/data/` 並重編兩邊 artifacts。
5. `progress.md` 只保留接手摘要；大段歷史移到 `.claude/rules/archive/`。

## 常用驗證

```bash
python3 -m py_compile artifact_compiler.py converter.py app.py scripts/build_runtime_artifacts.py
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
```

## 下一步方向

- 若持續做新聞詞補強，維持「來源 URL + 實測誤轉 + regression case」格式，但詳細內容放 archive。
- 若 runtime 資料量繼續變大，優先觀察 cold start、protected masking、regex rule 數量。
- 若 `progress.md` 超過約 150 行，再次整理舊內容進 archive。
