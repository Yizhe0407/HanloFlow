# taigi-converter（華語 -> 台語漢字）

將繁體中文轉換為台語漢字/漢羅的 Python 套件，內建詞典、規則與 runtime artifacts 編譯流程。

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
# -> 你咧做啥物？

print(c.convert("公車到站了"))
# -> 公車到站矣
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

## 字典與規則維護

先判斷要改的是哪一層：

| 檔案 | 用途 | 適合處理的問題 |
|---|---|---|
| `data/core_lexicon.json` | 核心高優先級常用詞 | 代名詞、疑問詞、位置詞、固定常用說法 |
| `data/lexicon_entries.jsonl` | 主詞典，含人工詞條與停用紀錄 | 單詞、片語、整句翻錯，且需要精準指定轉法 |
| `data/rule_entries.jsonl` | regex/literal 規則 | 同一類句型反覆轉錯，需要批次修正 |
| `data/char_verified_allowlist.txt` | 保護詞與 char 驗證白名單 | 詞不該被拆開、短詞或 regex 不該誤傷 |

建議順序：

1. 單一句子或片語錯誤，先查 `data/lexicon_entries.jsonl`。
2. 高頻基礎詞要固定轉法，再查 `data/core_lexicon.json`。
3. 同一種語法或句型反覆出錯，才改 `data/rule_entries.jsonl`。
4. 詞被拆壞或專有名詞被誤改，再補 `data/char_verified_allowlist.txt`。

不要直接手改 `data/artifacts/*`。那是編譯後產物，來源資料變更後重建即可。

## 重編 artifacts

修改 `data/core_lexicon.json`、`data/lexicon_entries.jsonl`、`data/rule_entries.jsonl` 或 `data/char_verified_allowlist.txt` 後都要重編：

```bash
python3 scripts/build_runtime_artifacts.py --data-dir data
```

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

> 若你是接手的 AI/code agent，先看 `AGENTS.md`。
