# Semantic evaluation

`data/semantic_eval_cases.jsonl` 是與既有 regression suites 零 source overlap 的獨立語意評測集。它用來量測語境詞義、語法、專有名詞與技術詞保護等品質，不取代既有 compatibility regressions。

## Corpus contract

- 300 cases：6 categories，各 50 cases。
- splits：train 150、development 90、holdout 60。
- 所有案例皆使用 `ai_semantic_review`，metadata 明確標示為 AI 語義審查，不宣稱人工翻譯認證。
- source 會先套用與 runtime 第一階段一致的字形、水平空白與外圍空白 canonicalization；raw source 與 canonical source 都不得和既有 regression source 重疊，也不得等於 active phrase/sentence runtime entry。
- `allow_sentence_override` 只可豁免 `sentence_override_entry_ids` 明列的 active sentence entries，且必須同時提供 `sentence_override_reason`；未列出的 sentence entry 與所有 active phrase entries 永遠不得豁免。
- runner 預設只報告 baseline；只有指定 `--fail-on-mismatch` 才會因 exact-match mismatch 回傳非零狀態。

## Commands

```bash
python3 scripts/audit_semantic_eval_leakage.py --fail-on-findings
python3 scripts/run_semantic_evaluation.py
python3 scripts/run_semantic_evaluation.py --split holdout
python3 scripts/run_semantic_evaluation.py --json-output build/semantic-eval.json
python3 scripts/run_semantic_evaluation.py --include-latency
```

## Phase 2 baseline

評測日期：2026-08-03

- corpus SHA-256：`8cb2ed1d7bbe41cb416ae2d4512813436b7cf1f1873f17453a8ac9a5063e8647`
- runtime source digest：`cf451c4035bf4e03a41d5f7a9e108f1090ff47fca24b742c147b47f3eefbcb91`
- passed：37 / 300
- failed：263 / 300
- pass rate：12.3333%
- leakage audit：0 findings

| Category | Passed | Total |
| --- | ---: | ---: |
| conversation | 3 | 50 |
| news | 10 | 50 |
| transport_travel | 3 | 50 |
| medical_public_service | 4 | 50 |
| polysemy_adversarial | 2 | 50 |
| proper_nouns_technical | 15 | 50 |

此數值是 Phase 3 修正前的 reproducible baseline。預設 JSON 與 snapshot 不含 latency，因此相同 corpus/runtime 可產生 deterministic 輸出；需要效能診斷時才使用 `--include-latency`。

## Phase 3 post-remediation result

Phase 2 已量測包含 60 筆 holdout 的完整 baseline；Phase 3 選擇 24 個有語境限制的 reusable phrase 修正目標時，只使用 train/development failures，沒有使用 holdout 選擇或調整 entries。目前以 25 個 contextual phrase entries 實作，其中「最新／最新的」拆為兩筆以完整消耗結構助詞；完成後再重新量測 holdout。

- train + development：30 / 240 → 42 / 240（+12，12.5% → 17.5%）
- holdout：7 / 60 → 7 / 60（維持 11.6667%，沒有用 holdout 調參）
- overall：37 / 300 → 49 / 300（12.3333% → 16.3333%）
- 新增 failure：0
- leakage audit：0 findings

這一階段刻意不以完整句 override 或廣域 char 規則追逐 exact-match；部分案例的目標片段已改善，但因句中仍有其他語義或文體差異，所以尚未計入整句通過。
