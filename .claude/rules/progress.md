# 現況與下一步方向

> 每次大規模優化後請更新此檔案。這是讓下一個人快速接手的地圖。

## 目前辭典狀態（2026-04-10）

| 類型 | 數量 |
|------|------|
| 總詞條（active） | 18,104 |
| 人工驗證（trust: human） | 5,563 |
| base seed（trust: seed，低信任） | ~10,700 |
| 停用詞條 | 4,919 |
| 最新 round | round111 |

所有迴歸測試全部通過：bus 107、medical 30、transport 52

---

## 近期做了什麼（最新在前）

### 2026-04-07：說->講 大規模修正（e5fbd61）
- 說->講 完整鏈：代名詞/在說/咧說/常見主語/說看
- 位置介系詞：在[台灣主要城市]->佇、住在->蹛佇
- 進行式 verb 清單擴充：等/聽/學/哭
- 停用 8 條嚴重錯誤詞條（你說->外人、不說->不論 等）
- post_cleanup 修正：懸雄->高雄、懸速公路->高速公路

### 2026-04-03～04：日常會話 round107–111
收到、了解、沒事、辛苦了、麻煩你了、掰掰、早點休息、等一下打給你、到哪了、剛到家、出發了沒、要轉車嗎 等 ~50 條日常口語短語

---

## 覆蓋率仍薄的地方

### 迴歸測試內（量少、遇新句型容易破）
- `medical / rooms_inpatient`：只有 2 個案例
- `medical / pharmacy_payment`：3 個
- `medical / redirect`：3 個
- `transport / destinations`：3 個
- `transport / crowd_safety`：5 個

### 尚無迴歸測試的情境（已知會踩雷的領域）
- 一般日常會話（打招呼、確認、表達情緒）← round107–111 正在補，但還缺 regression test
- 餐廳/點餐情境
- 飯店/住宿情境
- 購物/問價情境

---

## 下一步優先工作

1. **繼續日常會話 round112+**
   - 目前 round107–111 已補約 50 條，但缺 regression 保護
   - 建議補一支 `run_daily_speech_regression.py`，收入已知通過的句型

2. **把薄的迴歸類別補厚**
   - `medical / rooms_inpatient`、`pharmacy_payment`、`redirect` 各補到 6–8 條
   - `transport / destinations`、`crowd_safety` 各補到 8 條

3. **評估新領域 regression**
   - 日常會話領域已有足夠詞條，可以開一支 `run_conversation_regression.py`

4. **清查高頻錯誤的 base seed 詞條**
   - 還有大量 `trust: seed` 的舊詞條可能輸出奇怪結果
   - 下次遇到怪輸出，先跑 `/trace` 確認是不是 base seed 在 shadow

---

## 快速定位問題的方法

```bash
# 某句輸出不對，先確認哪個詞條命中
python3 app.py --trace "問題句子"

# 搜尋可能 shadow 的舊詞條
grep -n "關鍵字" data/lexicon_entries.jsonl | grep '"status":"active"'

# 確認目前最新 round
grep -o '"source":"curation:round[0-9]*' data/lexicon_entries.jsonl | grep -o 'round[0-9]*' | sort -t d -k 2 -n | tail -3
```
