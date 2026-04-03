# AGENT Guide

This file is for future AI/code agents working in this repo.

If a direct user request conflicts with this file, follow the user.

## Repo Facts

- Runtime reads `data/artifacts/*`, not raw source files.
- Source data lives in:
  - `data/lexicon_entries.jsonl`
  - `data/rule_entries.jsonl`
- Core code lives in:
  - `converter.py`
  - `artifact_compiler.py`
  - `lexicon_policy.py`
- After changing lexicon/rules/compiler behavior, rebuild artifacts:
  - `python3 scripts/build_runtime_artifacts.py --data-dir data`

## Response Language

- Unless the user explicitly asks for another language, reply in Traditional Chinese.
- Keep code, commands, file paths, identifiers, and schema keys in their original form when needed.

## Hard Rules

- Do not edit `data/artifacts/*` by hand. They are generated files.
- Prefer fixing translation quality in `data/lexicon_entries.jsonl` before touching `data/rule_entries.jsonl`.
- Avoid broad regex rules unless the pattern is extremely stable across domains.
- Treat official names, route names, stations, attractions, and brand-like transport names as protected terms when needed.
- Be careful with identity entries (`src == tgt`): they can become protected passthroughs and block better translations later.
- If an old identity/protected phrase blocks a better phrase, disable the old entry instead of piling on more overrides.

## Curation Strategy

For this project, the safe order is:

1. Reusable `phrase` abstraction
2. Narrow `sentence` override
3. `rule_entries.jsonl` change only if the pattern is proven stable

Use `phrase` entries when:

- The issue is a reusable local pattern
- The same chunk appears inside many longer traveler replies
- You want better generalization without risking a global rule

Examples of high-value phrase patterns:

- Directional chunks: `往...走`, `走到...`, `先走...`, `走進去`, `走出去`
- Locatives: `在...`, `站在...`
- Traveler guidance chunks: `過馬路`, `過天橋`, `過市場`, `在路邊`, `在巷口`

Use `sentence` overrides only when:

- Proper nouns + service tone + whole-sentence phrasing are tightly coupled
- A phrase abstraction would still be unsafe or too lossy
- Official names must remain exact and the rest of the sentence is highly specific

Avoid using rules for:

- Place names
- Attraction names
- Station names
- Bus route names
- Culture-specific phrases like pilgrimage/event wording
- Anything that would globally rewrite common surface forms with high ambiguity

## What We Learned From Practical Tuning

- The biggest wins usually come from fixing reusable traveler-guidance phrases, not from adding many full sentences.
- Common bad outputs often come from old legacy base phrases, not from the rule engine.
- Before adding new data, check whether a legacy phrase is actively causing the bad output.
- If output looks strangely literal or semantically wrong, search for a low-priority legacy phrase first.
- If output refuses to change even after adding a better phrase, look for an older identity/protected entry shadowing it.

## Domain Preference In This Repo

Recent practical tuning focused heavily on:

- Yunlin bus/travel replies
- Attraction directions
- Station/shuttle guidance
- Short service replies to travelers

Preferred style:

- Natural spoken Taiwanese Hokkien in Han characters
- Keep official proper nouns stable
- Favor reusable directional and locative chunks over one-off sentence patches
- Do not over-normalize everything into rare or literary forms

## Validation Checklist

After edits, always run:

```bash
python3 scripts/build_runtime_artifacts.py --data-dir data
python3 -m py_compile artifact_compiler.py converter.py app.py scripts/build_runtime_artifacts.py
```

Then test both:

- The original failing case
- At least a few unseen longer sentences that should benefit from the same phrase

Useful quick smoke pattern:

- Bare phrase
- Short sentence
- Longer sentence with punctuation
- Sentence combining two recently added phrases

If debugging a stubborn case, use trace:

```bash
python3 app.py --trace "過馬路就會看到公車站。"
```

## Performance Guardrail

- Keep warm-path conversion comfortably under `0.05` seconds.
- Current practical target is far lower than that, so avoid changes that add heavy runtime work for small quality gains.
- Phrase data curation is usually safer than runtime algorithm complexity.

## Working Style

- Keep changes surgical.
- Disable bad legacy entries when they are clearly wrong.
- Prefer one reusable fix over many duplicate sentence fixes.
- Do not churn existing artifacts or rules unless they materially improve quality or unblock the correct phrase from matching.
