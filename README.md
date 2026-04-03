# Taigi Converter（華語 -> 台語漢字）

目前專案已精簡為「可直接執行」的最小版本。

若你是後續接手的 AI/code agent，先看 `AGENT.md`。裡面整理了這個 repo 的資料修正順序、實戰收斂規則、常見踩雷點，特別是 `lexicon`/`rule`/`artifacts` 的操作原則。

## 快速使用

### 1) 單句轉換

```bash
python3 app.py "你在做什麼？"
```

### 2) Trace（JSON）

```bash
python3 app.py --trace "你在做什麼？"
```

### 3) Explain（人類可讀）

```bash
python3 app.py --explain "你在做什麼？"
```

### 4) 互動模式

```bash
python3 app.py
```

## 資料更新後重編 artifacts

```bash
python3 scripts/build_runtime_artifacts.py --data-dir data
```

## 目前保留目錄

- `app.py`：CLI 入口
- `converter.py`：轉換核心流程
- `pipeline.py`：函式管線（華語 -> 台語漢字 -> Taibun 羅馬字數字調）
- `artifact_compiler.py`：詞條/規則編譯器
- `models.py`：資料模型
- `normalize.py`：正規化
- `lexicon_policy.py`：詞條信任與執行期過濾策略
- `review_queue.py`：低信心回填佇列
- `data/lexicon_entries.jsonl`：詞條主資料
- `data/rule_entries.jsonl`：規則主資料
- `data/artifacts/*`：執行期 artifacts
- `scripts/build_runtime_artifacts.py`：重編工具

## 備註

- 執行期讀的是 `data/artifacts/*`。
- 若你改了 `data/lexicon_entries.jsonl` 或 `data/rule_entries.jsonl`，請重跑一次 artifacts 編譯。

## 函式串接（LLM 輸出後）

若你要走「函式」流程，現在只需要呼叫 `run_llm_postprocess`。

先準備 Taibun（clone + 依賴）：

```bash
git clone https://github.com/andreihar/taibun.git /absolute/path/to/taibun
python3 -m pip install msgpack
```

```python
from pipeline import run_llm_postprocess

rom = run_llm_postprocess(
    "我們先去辦公室坐。",                        # LLM 輸出的華語
    taibun_repo_path="/absolute/path/to/taibun",  # clone 的 taibun repo 根目錄
)

print(rom)  # 直接得到 Tailo 數字調
```
```
uv run - <<'PY'
from pipeline import run_llm_postprocess

text = "請往裡面走，不要擠在門口，不然別人的行李會放不下。"
out = run_llm_postprocess(text, taibun_repo_path=".vendor/taibun")
print(out)
PY
```

### Taibun 缺字補丁（內建）

管線內建以下 fallback patch（可被 `taibun_patch_map` 覆蓋）：

- `tone_format=number`：`妳->li2`, `您->lin2`, `她->i1`, `嗎->ma0`, `嘸->bo5`, `呣->m7`
- `tone_format=mark`：`妳->lí`, `您->lín`, `她->i`, `嗎->ma`, `嘸->bô`, `呣->m̄`


## DOM TextNode Integration

If you run this converter in a browser extension, do not rely on cross-node regex over individual TextNodes.

See: docs/dom_textnode_strategy.md


## Production Artifact Notes

- Build now emits compact runtime rule schema (`compact_v2`).
- Only enabled rules are emitted to runtime artifacts.
- Runtime payload strips dev-only fields (for example `note`, masking diagnostics, residual debug lists).
