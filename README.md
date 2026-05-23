# taigi-converter（華語 → 台語漢字）

將繁體中文轉換為台語漢羅（漢字 + 羅馬字）的 Python 套件。

## 安裝

```bash
# 從 GitHub 安裝
pip install git+https://github.com/Yizhe0407/HanloFlow.git

# 或用 uv
uv add "taigi-converter @ git+https://github.com/Yizhe0407/HanloFlow.git"

# 本機開發（editable，改完即時生效）
pip install -e /path/to/HanloFlow
uv add --editable /path/to/HanloFlow
```

## 更新

```bash
# git URL 安裝的用戶：pull 最新 commit
pip install --upgrade git+https://github.com/Yizhe0407/HanloFlow.git

# uv 用戶
uv sync --upgrade-package taigi-converter

# editable 本機安裝的用戶：不需任何指令，改完直接生效
```

## Python API

```python
from taigi_converter import TaigiConverter

c = TaigiConverter()

# 基本轉換（回傳台語漢羅字串）
print(c.convert("你在做什麼？"))
# → 你在做啥物？

print(c.convert("公車到站了"))
# → 公車到站矣
```

## CLI

```bash
# 單句轉換
python3 app.py "你在做什麼？"

# 輸出完整 trace（JSON）
python3 app.py --trace "你在做什麼？"

# 人類可讀模式
python3 app.py --explain "你在做什麼？"

# 互動模式
python3 app.py
```

## 資料更新後重編 artifacts

修改 `data/lexicon_entries.jsonl` 或 `data/rule_entries.jsonl` 後必須重編：

```bash
python3 scripts/build_runtime_artifacts.py --data-dir data
```

## 資料檔案使用順序與功能

### 1. 第一步：先看 `data/lexicon_entries.jsonl`

| 項目 | 說明 |
|---|---|
| 這個檔案是什麼 | 詞條主資料 |
| 功能 | 控制「某個詞、片語、句子」要怎麼轉成台語 |
| 什麼時候用 | 你發現單一句子、單一詞語轉錯時，優先查這裡 |
| 例子 | `一切都是白說的` → `一切攏是白講的` |
| 是否常改 | 最常改 |

### 2. 第二步：再看 `data/rule_entries.jsonl`

| 項目 | 說明 |
|---|---|
| 這個檔案是什麼 | 規則主資料 |
| 功能 | 控制 regex/literal 規則，處理一整類句型或文法 |
| 什麼時候用 | 同一種句型很多句都轉錯時才改 |
| 例子 | `食飽了沒` → `食飽未` |
| 是否常改 | 會改，但要比詞條更小心 |

### 3. 第三步：看 `data/core_lexicon.json`

| 項目 | 說明 |
|---|---|
| 這個檔案是什麼 | 核心詞庫 |
| 功能 | 放高優先級、很基礎、常用的轉換詞 |
| 什麼時候用 | 像代名詞、基本疑問詞、常用位置詞要固定轉法時 |
| 例子 | `他們` → `怹`、`我們` → `咱`、`在哪裡` → `佇佗位` |
| 是否常改 | 偶爾改，不是一般詞條都放這裡 |

### 4. 第四步：看 `data/char_verified_allowlist.txt`

| 項目 | 說明 |
|---|---|
| 這個檔案是什麼 | 白名單/保護詞清單 |
| 功能 | 保護某些詞不要被拆開或被短詞、regex 亂改 |
| 什麼時候用 | 某個詞本身不該被切開轉換時 |
| 例子 | `智慧型手機`、`影印文件`、`下班`、`蘋果` |
| 是否常改 | 需要保護詞時才改 |

### 5. 第五步：不要直接看/改 `data/artifacts/*`

| 項目 | 說明 |
|---|---|
| 這個資料夾是什麼 | 編譯後產物 |
| 功能 | 給程式 runtime 快速查表、套規則 |
| 什麼時候用 | 程式執行時使用，不是人工維護入口 |
| 例子 | `entry_table.json`、`phrase_trie.json`、`rule_plan.json` |
| 是否常改 | 不手動改，改來源資料後重編產生 |

### 6. 第六步：改完來源資料後重編

| 項目 | 說明 |
|---|---|
| 用哪個檔案/指令 | `scripts/build_runtime_artifacts.py` |
| 功能 | 把 `data/` 的來源資料重新編譯成 `data/artifacts/*` |
| 什麼時候用 | 每次改完 `lexicon_entries.jsonl`、`rule_entries.jsonl`、`core_lexicon.json` 後 |
| 指令 | `python3 scripts/build_runtime_artifacts.py --data-dir data` |
| 是否常用 | 改資料後必跑 |

## 專案結構

```
taigi_converter/          ← 可安裝的 Python 套件
├── __init__.py           ← 對外只需 from taigi_converter import TaigiConverter
├── converter.py          ← 轉換核心
├── pipeline.py           ← 函式管線
├── artifact_compiler.py  ← 詞條/規則編譯器
├── models.py             ← 資料模型
├── normalize.py          ← 正規化
├── lexicon_policy.py     ← 詞條信任策略
├── review_queue.py       ← 低信心回填佇列
└── data/
    ├── lexicon_entries.jsonl  ← 詞條主資料
    ├── rule_entries.jsonl     ← 規則主資料
    └── artifacts/             ← 執行期編譯產物
app.py                    ← CLI 入口（不需安裝直接執行）
scripts/                  ← 工具腳本
docs/                     ← 開發文件
```

> 若你是接手的 AI/code agent，先看 `AGENT.md`。
