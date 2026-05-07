# 詞條策略規則

## 新增詞條前的必查清單

1. `grep -n "關鍵字" data/lexicon_entries.jsonl` 確認無重複
2. 確認沒有舊 identity/protected entry shadow（輸出完全不變時先查這個）
3. 決定 tier：`manual` 是新增人工詞條的標準選擇
4. 決定 level：`phrase` 優先，`sentence` 只用於整句高度耦合的場合；`char` 最後才補

## Tier 說明與 Priority 範圍

| tier | priority 範圍 | 用途 |
|------|--------------|------|
| `base` | 40 | Legacy seed 詞條，不要手動新增 |
| `domain` | 900–930 | 領域詞條（公車、醫療等） |
| `manual` | 850–1910 | **手動新增的人工詞條，主要選擇** |
| `manual_hotfix` | 300–1250 | 緊急修正，蓋過 base/domain 但低於 manual |

**新增 manual phrase 的常用 priority：**
- `1200` — 標準人工詞條
- `1300+` — 需要蓋過其他 manual 詞條時
- `950` — 領域詞條（domain tier）

## 完整 JSONL 欄位格式

新增時使用以下格式（`tier: manual`，`trust: human`）：

```json
{"entry_id":"lx_<12hex>","src":"來源詞","tgt":"目標台語漢字","level":"phrase","tier":"manual","priority":1200,"context":null,"score":1.0,"status":"active","source":"curation:round112_<描述>","trust":"human","updated_by":"curation_round112_codex","updated_at":"2026-04-10T00:00:00+08:00"}
```

**entry_id 產生方式**（SHA1 of `src|tgt|level|tier|source` 取前 12 碼）：

```bash
python3 -c "
import hashlib, sys
src, tgt, level, tier, source = sys.argv[1:]
raw = f'{src}|{tgt}|{level}|{tier}|{source}'.encode()
print('lx_' + hashlib.sha1(raw).hexdigest()[:12])
" "來源詞" "目標詞" "phrase" "manual" "curation:round112_描述"
```

## phrase vs sentence 判斷樹

```
這個詞/短語會出現在多句話裡嗎？
├── 是 → level: phrase
│   └── 包含官方固有名詞且整句語氣緊密耦合？
│       ├── 是 → level: sentence
│       └── 否 → 確認 phrase
└── 否 → level: sentence（先確認是否只是一次性 patch）
```

## 優化優先順序

1. 固定句型或常見說法：補 `sentence` 或長 `phrase`
2. 穩定詞組：補 `phrase`
3. 語法型規律：補 `rule_entries.jsonl`
4. 單字：只在對應非常穩定且不會污染複合詞時補 `char`

## Round 編號慣例

- 目前最新：`round210`，下一個新增用 `round211`
- `source` 格式：`curation:round<N>_<英文描述>`，例如 `curation:round112_daily_speech_refresh`
- `updated_by` 格式：`curation_round<N>_codex`
- 每次新增一批相關詞條共用同一個 round 號，不同主題用不同 round

## 危險詞條樣式（避免）

- `src == tgt`（identity entry）除非真的是 passthrough
- 過於通用的 regex rule（例如 `.*走.*`）
- 涵蓋地名、站名、路線名的 global rule
- `tier: base` 新詞條（永遠用 `manual`）

## 詞條停用而非刪除

當舊詞條妨礙更好的 phrase 時，**停用（`"status":"disabled"`）而不刪除**，保留歷史脈絡。
