# 現況與下一步方向

> 精簡接手版。2026-07-28 以前的逐輪記錄已移到 `.claude/rules/archive/progress-through-2026-07-28.md`。

## 目前狀態（2026-07-28）

- branch：`main`
- 正式程式單一來源：`taigi_converter/`
- source data 單一來源：`data/`
- wheel runtime 單一來源：`taigi_converter/data/artifacts/`
- active runtime entries：20,478
- regression：12 suites、4,179 cases，全部通過
- artifact format / compiler version：3
- runtime 核心第三方 dependency：0

## 2026-07-28 架構治理

- 根目錄同名模組改為 compatibility wrapper，消除 root/package 雙份實作。
- package 不再攜帶約 10 MB 的 source JSONL，只攜帶六個已編譯 artifacts。
- runtime 預設 `auto_prepare=False`，可在唯讀 site-packages 初始化。
- manifest 使用 source digest 與逐 artifact SHA-256；建置以 manifest 作 commit marker，reader 遇混合世代會重試。
- runtime 依 artifact directory + manifest hash cache，後續 converter instance 共用已載入狀態，並清除同路徑舊世代避免累積。
- compiler 新增完整 schema、duplicate ID/pattern、regex 與同順位 target conflict fail-fast。
- 停用 717 筆與官方「臺鐵」字形同順位衝突的 identity entries；既有 regressions 維持全綠。
- review queue 使用 OS-managed process lock、完整 append、`fsync`、atomic replace、獨立可寫 state 目錄與 crash-recovery journal。
- 12 個 regression scripts 共用 `scripts/regression_runner.py`，並新增一次執行全部 suite 的入口。
- 新增 29 個 unit tests、Ruff、Python 3.12/3.13 CI、wheel 內容、安裝後 CLI 與唯讀 parity 驗證。
- `msgpack` 從核心 dependency 移到 `taibun` optional extra；新增正式 console script。

## 維護重點

1. 修改 `data/` 後執行 `python3 scripts/build_runtime_artifacts.py --fail-on-mask`。
2. 不手動編輯 package artifacts，不建立第二份 source data。
3. 每次完成前跑 `AGENT.md` 的完整驗證矩陣。
4. 持續觀察 cold start、RSS、protected masking 與 rule 數量；目前 warm init 約 0.15 ms，第二個 instance 不增加 runtime RSS。
5. 新增詞條前依 `.claude/rules/curation.md` 檢查詞界、tier、priority、identity shadow 與既有衝突。
