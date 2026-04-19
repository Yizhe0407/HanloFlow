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
