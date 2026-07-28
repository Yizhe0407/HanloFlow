# HanloFlow — Claude Code 工作指引

本專案是**繁體華語 → 台語漢字**轉換器。主要規範以 `AGENT.md` 為準，詞條與驗證細節見：

- `.claude/rules/curation.md`
- `.claude/rules/validation.md`
- `.claude/rules/progress.md`

## 快速入口

```bash
python3 app.py "你在做什麼？"
python3 app.py --trace "你在做什麼？"
python3 scripts/build_runtime_artifacts.py --fail-on-mask
python3 scripts/run_all_regressions.py
uv run pytest
```

## 重要限制

- 只在 `taigi_converter/` 實作正式程式。
- 只在 `data/` 編輯 source data。
- `taigi_converter/data/artifacts/` 必須由 build script 產生。
- 一般 runtime 不得寫入安裝目錄或自動編譯。
- 修改資料、compiler、converter、封裝設定後，需跑 `AGENT.md` 的完整完成定義。

除非使用者另有要求，一律使用繁體中文回覆。
