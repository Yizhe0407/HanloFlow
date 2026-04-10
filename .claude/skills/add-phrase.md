---
name: add-phrase
description: 引導新增 phrase/sentence 詞條到 lexicon_entries.jsonl，含格式產生與 curation checklist
---

使用者提供的資訊：$ARGUMENTS

## 步驟 1：確認 src/tgt

若 $ARGUMENTS 為空，請先詢問：
- 來源詞（src，華語）
- 目標詞（tgt，台語漢字）
- level（phrase 或 sentence，預設 phrase）

## 步驟 2：搜尋現有詞條

```bash
grep -n "<src關鍵字>" data/lexicon_entries.jsonl
```

列出搜尋結果，標示是否有重複或 shadow 風險的舊詞條。

## 步驟 3：Curation checklist

根據結果告知使用者判斷：

- [ ] 是否已有相同 src 的詞條？（active 還是 disabled？）
- [ ] 這個 src 能否在多句話裡重複使用？（是 → phrase）
- [ ] 是否整句都跟官方名詞緊密耦合？（是 → sentence）
- [ ] 是否需要蓋過現有詞條？（是 → priority 調高至 1300+）

## 步驟 4：產生 entry_id 與完整 JSONL

執行以下指令產生 entry_id（round 號用最新的，從現有資料判斷）：

```bash
python3 -c "
import hashlib
src = '<src>'
tgt = '<tgt>'
level = 'phrase'
tier = 'manual'
source = 'curation:round112_<描述>'
raw = f'{src}|{tgt}|{level}|{tier}|{source}'.encode()
print('lx_' + hashlib.sha1(raw).hexdigest()[:12])
"
```

取得 entry_id 後，產生完整 JSONL 行：

```json
{"entry_id":"lx_<產生結果>","src":"<src>","tgt":"<tgt>","level":"phrase","tier":"manual","priority":1200,"context":null,"score":1.0,"status":"active","source":"curation:round112_<描述>","trust":"human","updated_by":"curation_round112_codex","updated_at":"<今天日期>T00:00:00+08:00"}
```

priority 參考：
- `1200` — 標準人工詞條
- `1300+` — 需要蓋過其他 manual 詞條
- `950` / `tier: domain` — 領域詞條（公車、醫療等）

## 步驟 5：確認後寫入

使用者確認後，將詞條 append 到 `data/lexicon_entries.jsonl`。

若有需要停用的舊詞條，先找到它的 entry_id，將 `"status":"active"` 改為 `"status":"disabled"`。

完成後提示執行 `/rebuild` 驗證。
