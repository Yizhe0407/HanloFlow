# Semantic evaluation

`data/semantic_eval_cases.jsonl` 是與既有 regression suites 零 source overlap 的獨立語意評測集。它用來量測語境詞義、語法、專有名詞與技術詞保護等品質，不取代既有 compatibility regressions。

## Corpus contract

- 300 cases：6 categories，各 50 cases。
- splits：train 150、development 90、holdout 60。
- 所有案例皆使用 `ai_semantic_review`，metadata 明確標示為 AI 語義審查，不宣稱人工翻譯認證。
- source 不得和既有 regression source 重疊，也不得等於 active phrase/sentence runtime entry；例外必須在 case 中明確核准並說明原因。
- runner 預設只報告 baseline；只有指定 `--fail-on-mismatch` 才會因 exact-match mismatch 回傳非零狀態。

## Commands

```bash
python3 scripts/audit_semantic_eval_leakage.py --fail-on-findings
python3 scripts/run_semantic_evaluation.py
python3 scripts/run_semantic_evaluation.py --split holdout
python3 scripts/run_semantic_evaluation.py --json-output build/semantic-eval.json
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

此數值是 Phase 3 修正前的 reproducible baseline。Latency 不納入 snapshot，因為它會受執行環境影響。
