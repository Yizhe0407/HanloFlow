# 現況與下一步方向

> 每次大規模優化後請更新此檔案。這是讓下一個人快速接手的地圖。

## 目前辭典狀態（2026-04-11）

| 類型 | 數量 |
|------|------|
| 總詞條（active） | ~18,140 |
| 人工驗證（trust: human） | ~5,600 |
| base seed（trust: seed，低信任） | ~10,700 |
| 停用詞條 | ~4,930 |
| 最新 round | round114 |

所有迴歸測試全部通過：bus 107、medical 51、transport 60、conversation 30

---

## 近期做了什麼（最新在前）

### 2026-04-11：大批 base seed 清查 + 多類別系統修正（round114）
**停用的惡性 base seed（共 5 條）：**
- `不見`→`無去`、`下個月`→`後個月`、`很高`→`足懸的`、`跑走`→`逃走`、`完了`→`完結`

**Allowlist 修正：**
- 移除 `看電視`、`吹冷氣`（不需保護，但遮蓋後阻擋進行式規則）

**Rule 擴充：**
- 進行式動詞 +6：吹/唱/玩/跑/買/忙（`rl_817e6efe2ce1`）
- 新增 fluency rule：`欲說` → `欲講`、`咧說` 已有

**新增詞條（28 條）：**
- `不清楚`/`不明白`/`不記得`/`不確定`/`不太確定` 系列
- `會說`/`欲說`/`想說` → X講
- `睡不著`→`睏袂去`、`上週`/`下週`/`這週`→禮拜系列
- `外面在下雨`/`外面下雨了`、`上個星期`/`下個星期`
- `牙痛`→`牙疼`、`喉嚨痛`→`嚨喉疼`、`喉嚨`→`嚨喉`
- `見到`→`見著`、`找到`→`揣著`、`跟著`→`綴`
- `不願意`→`毋肯`、`不應該`→`毋著`
- `厲害`→`利害`、`好厲害`→`真利害`、`跑走`→`走去`

### 2026-04-11：日常會話 bug 修正 + 薄弱迴歸補強（round113）
- 停用惡性 base seed：`不見`→`無去`（修正「好久不見」→「好久無去」的錯誤）
- 新增 4 條 round113 詞條：`好久不見`→`好久無見`、`你辛苦了`→`你辛苦矣`（修正重複「你」）、`不確定`→`無確定`、`不太確定`→`無啥確定`
- 新增 `run_conversation_regression.py`（30 筆，5 類：greetings/status_check/daily_chat/daily_response/schedule_plans）
- medical 補強：doctor_flow 6→10、tests 6→10（共 43→51）
- transport 補強：crowd_safety 5→8（共 57→60）

### 2026-04-10：base seed 修正 + 薄弱迴歸類別補強（round112）
- 停用 3 條嚴重錯誤的 base seed：`你要注意`→`笑面虎`、`探視`→`探頭`、`住院`→`入院`
- 新增：`右手邊`→`正手爿`、`左手邊`→`倒手爿`、`住院`→`蹛院`、`住院櫃檯` identity 保護、`探視` identity 保護
- 迴歸測試補強：medical 30→43（rooms_inpatient 2→8、pharmacy_payment 3→7、redirect 3→6）
- transport 52→57（destinations 3→8）

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
- `transport / crowd_safety`：5 個（下次補到 8）
- `medical / doctor_flow`：6 個（可補）
- `medical / tests`：6 個（可補）

### 尚無迴歸測試的情境（已知會踩雷的領域）
- 一般日常會話（打招呼、確認、表達情緒）← round107–111 正在補，但還缺 regression test
- 餐廳/點餐情境
- 飯店/住宿情境
- 購物/問價情境

---

## 下一步優先工作

1. **繼續日常會話 round114+**
   - `run_conversation_regression.py` 已建立（30 筆），可繼續擴充
   - 待補情境：餐廳/點餐、飯店/住宿、購物/問價

2. **繼續補薄弱迴歸類別**
   - `medical / doctor_flow`：已 10 條，持續觀察
   - `transport / crowd_safety`：已 8 條，可再補到 10

3. **開新領域 regression**
   - 餐廳點餐：`run_restaurant_regression.py`（尚未建立）
   - 購物情境：`run_shopping_regression.py`（尚未建立）

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
