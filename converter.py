from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_compiler import ensure_runtime_ready
from lexicon_policy import is_machine_generated_override, is_sentence_manual_override, is_trusted_manual_entry
from models import ConversionResult, LexiconEntry, MatchTrace, RuleEntry, RuleTrace
from normalize import normalize_text
from review_queue import append_review_item


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class Candidate:
    entry: LexiconEntry
    start: int
    end: int
    layer_rank: int


class TaigiConverter:
    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        fail_on_mask: bool = False,
        auto_prepare: bool = True,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.artifact_dir = self.data_dir / "artifacts"

        if auto_prepare:
            ensure_runtime_ready(self.data_dir, fail_on_mask=fail_on_mask)

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        entry_table = self._read_json(self.artifact_dir / "entry_table.json")
        phrase_trie_doc = self._read_json(self.artifact_dir / "phrase_trie.json")
        char_map_doc = self._read_json(self.artifact_dir / "char_map.json")
        rule_plan = self._read_json(self.artifact_dir / "rule_plan.json")
        override_index = self._read_json(self.artifact_dir / "override_index.json")

        self.entries: dict[str, LexiconEntry] = {
            entry_id: LexiconEntry.from_dict(row)
            for entry_id, row in entry_table["entries"].items()
        }
        self.layer_rank_by_entry_id: dict[str, int] = {
            entry_id: self._layer_rank(entry)
            for entry_id, entry in self.entries.items()
        }
        self.phrase_trie = phrase_trie_doc["trie"]
        self.char_map: dict[str, list[str]] = char_map_doc["map"]
        self.has_char_entries: bool = bool(self.char_map)
        self.rule_pass_order: list[str] = rule_plan.get("pass_order", [])
        self.rules_by_pass: dict[str, list[RuleEntry]] = {}
        self.compiled_rules_by_pass: dict[str, list[tuple[RuleEntry, re.Pattern[str] | None]]] = {}
        for pass_name, rows in rule_plan.get("rules", {}).items():
            parsed_rules = [RuleEntry.from_dict(item) for item in rows]
            self.rules_by_pass[pass_name] = parsed_rules
            compiled_rules: list[tuple[RuleEntry, re.Pattern[str] | None]] = []
            for rule in parsed_rules:
                compiled = re.compile(rule.pattern) if rule.type == "regex" and rule.pattern else None
                compiled_rules.append((rule, compiled))
            self.compiled_rules_by_pass[pass_name] = compiled_rules
        self.residual_terms: list[str] = rule_plan.get("residual_terms", [])

        self.sentence_override_map: dict[str, list[str]] = override_index.get("sentence_override_map", {})
        self.contextual_override_ids: list[str] = override_index.get("contextual_override_ids", [])

        self.blocked_sentence_entry_ids = [
            entry_id
            for entry_id, entry in self.entries.items()
            if entry.status == "active" and entry.tier == "blocked" and entry.level == "sentence"
        ]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _layer_rank(self, entry: LexiconEntry) -> int:
        if entry.tier == "blocked":
            return 0
        if is_sentence_manual_override(entry):
            return 1
        if is_trusted_manual_entry(entry):
            return 2
        if entry.tier == "core" and entry.level in {"phrase", "sentence"}:
            return 3
        if entry.tier == "domain" and entry.level in {"phrase", "sentence"}:
            return 4
        if entry.tier == "base" and entry.level in {"phrase", "sentence"}:
            return 5
        if is_machine_generated_override(entry) and entry.level in {"phrase", "sentence"}:
            return 6
        if entry.level == "char":
            return 7
        return 99

    def _candidate_key(self, candidate: Candidate) -> tuple[int, int, int, float, int, str]:
        entry = candidate.entry
        return (
            candidate.layer_rank,
            -entry.priority,
            -(candidate.end - candidate.start),
            -entry.score,
            candidate.start,
            entry.entry_id,
        )

    def _iter_phrase_candidates(self, text: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        root = self.phrase_trie
        entries = self.entries
        layer_rank_by_entry_id = self.layer_rank_by_entry_id

        for start in range(len(text)):
            node = root
            index = start
            while index < len(text):
                ch = text[index]
                child = node["children"].get(ch)
                if child is None:
                    break
                node = child
                index += 1
                for entry_id in node.get("entry_ids", []):
                    entry = entries.get(entry_id)
                    if not entry or entry.status != "active":
                        continue
                    candidates.append(
                        Candidate(
                            entry=entry,
                            start=start,
                            end=index,
                            layer_rank=layer_rank_by_entry_id.get(entry_id, 99),
                        )
                    )
        return candidates

    @staticmethod
    def _context_match(text: str, start: int, end: int, context: dict[str, Any] | None) -> bool:
        if not context:
            return True

        left_text = text[:start]
        right_text = text[end:]

        left_regex = context.get("left_regex")
        if left_regex and not re.search(left_regex, left_text):
            return False

        right_regex = context.get("right_regex")
        if right_regex and not re.search(right_regex, right_text):
            return False

        full_regex = context.get("full_regex")
        if full_regex and not re.search(full_regex, text):
            return False

        left_literal = context.get("left_literal")
        if left_literal and not left_text.endswith(left_literal):
            return False

        right_literal = context.get("right_literal")
        if right_literal and not right_text.startswith(right_literal):
            return False

        return True

    def _iter_contextual_candidates(self, text: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        entries = self.entries
        layer_rank_by_entry_id = self.layer_rank_by_entry_id
        for entry_id in self.contextual_override_ids:
            entry = entries.get(entry_id)
            if not entry or entry.status != "active":
                continue
            if not entry.src:
                continue

            start = 0
            while True:
                found = text.find(entry.src, start)
                if found < 0:
                    break
                end = found + len(entry.src)
                if self._context_match(text, found, end, entry.context):
                    candidates.append(
                        Candidate(
                            entry=entry,
                            start=found,
                            end=end,
                            layer_rank=layer_rank_by_entry_id.get(entry_id, 99),
                        )
                    )
                start = found + 1
        return candidates

    def _iter_char_candidates(self, text: str) -> list[Candidate]:
        if not self.has_char_entries:
            return []
        candidates: list[Candidate] = []
        entries = self.entries
        layer_rank_by_entry_id = self.layer_rank_by_entry_id
        for index, ch in enumerate(text):
            entry_ids = self.char_map.get(ch, [])
            for entry_id in entry_ids:
                entry = entries.get(entry_id)
                if not entry or entry.status != "active":
                    continue
                candidates.append(
                    Candidate(
                        entry=entry,
                        start=index,
                        end=index + 1,
                        layer_rank=layer_rank_by_entry_id.get(entry_id, 99),
                    )
                )
        return candidates

    @staticmethod
    def _span_mask(start: int, end: int) -> int:
        width = end - start
        if width <= 0:
            return 0
        return ((1 << width) - 1) << start

    def _select_non_overlapping(
        self,
        candidates: list[Candidate],
        *,
        reserved: list[Candidate] | None = None,
        text_length: int,
    ) -> list[Candidate]:
        occupied_mask = 0
        for candidate in reserved or []:
            occupied_mask |= self._span_mask(candidate.start, candidate.end)

        selected: list[Candidate] = []
        for candidate in sorted(candidates, key=self._candidate_key):
            if candidate.start >= candidate.end:
                continue
            span_mask = self._span_mask(candidate.start, candidate.end)
            if occupied_mask & span_mask:
                continue
            selected.append(candidate)
            occupied_mask |= span_mask

        return sorted(selected, key=lambda c: c.start)

    def _collect_blocked_candidates(self, text: str, phrase_candidates: list[Candidate]) -> list[Candidate]:
        blocked = [candidate for candidate in phrase_candidates if candidate.entry.tier == "blocked"]

        for entry_id in self.blocked_sentence_entry_ids:
            entry = self.entries[entry_id]
            if text == entry.src:
                blocked.append(
                    Candidate(
                        entry=entry,
                        start=0,
                        end=len(text),
                        layer_rank=0,
                    )
                )

        if self.has_char_entries:
            for idx, ch in enumerate(text):
                for entry_id in self.char_map.get(ch, []):
                    entry = self.entries.get(entry_id)
                    if not entry or entry.status != "active":
                        continue
                    if entry.tier != "blocked":
                        continue
                    blocked.append(
                        Candidate(
                            entry=entry,
                            start=idx,
                            end=idx + 1,
                            layer_rank=0,
                        )
                    )

        return self._select_non_overlapping(blocked, text_length=len(text))

    def _apply_lexicon_layers(self, text: str) -> tuple[str, list[MatchTrace], list[str]]:
        warnings: list[str] = []
        phrase_candidates = self._iter_phrase_candidates(text)
        blocked_candidates = self._collect_blocked_candidates(text, phrase_candidates)

        for blocked in blocked_candidates:
            warnings.append(
                f"blocked:{blocked.entry.entry_id}:{blocked.entry.src}"
            )

        sentence_override_ids = self.sentence_override_map.get(text, [])
        sentence_candidates = [
            Candidate(entry=self.entries[entry_id], start=0, end=len(text), layer_rank=1)
            for entry_id in sentence_override_ids
            if entry_id in self.entries
        ]
        if sentence_candidates:
            sentence_selected = self._select_non_overlapping(
                sentence_candidates,
                reserved=blocked_candidates,
                text_length=len(text),
            )
            if sentence_selected:
                chosen = sentence_selected[0]
                trace = MatchTrace(
                    entry_id=chosen.entry.entry_id,
                    src=chosen.entry.src,
                    tgt=chosen.entry.tgt,
                    level=chosen.entry.level,
                    tier=chosen.entry.tier,
                    start=0,
                    end=len(text),
                    priority=chosen.entry.priority,
                    score=chosen.entry.score,
                )
                return chosen.entry.tgt, [trace], warnings

        contextual_candidates = self._iter_contextual_candidates(text)
        phrase_non_blocked = [
            candidate
            for candidate in phrase_candidates
            if candidate.entry.tier != "blocked"
        ]

        # L2-L5: 先吃句級/詞組級候選；L6 char 僅補未命中區段
        selected_phrase = self._select_non_overlapping(
            contextual_candidates + phrase_non_blocked,
            reserved=blocked_candidates,
            text_length=len(text),
        )

        selected_char: list[Candidate] = []
        if self.has_char_entries:
            char_candidates = self._iter_char_candidates(text)
            if char_candidates:
                selected_char = self._select_non_overlapping(
                    char_candidates,
                    reserved=blocked_candidates + selected_phrase,
                    text_length=len(text),
                )

        selected = sorted(selected_phrase + selected_char, key=lambda c: c.start)

        if not selected:
            return text, [], warnings

        output_parts: list[str] = []
        cursor = 0
        traces: list[MatchTrace] = []
        for candidate in selected:
            output_parts.append(text[cursor : candidate.start])
            output_parts.append(candidate.entry.tgt)
            cursor = candidate.end
            traces.append(
                MatchTrace(
                    entry_id=candidate.entry.entry_id,
                    src=candidate.entry.src,
                    tgt=candidate.entry.tgt,
                    level=candidate.entry.level,
                    tier=candidate.entry.tier,
                    start=candidate.start,
                    end=candidate.end,
                    priority=candidate.entry.priority,
                    score=candidate.entry.score,
                )
            )
        output_parts.append(text[cursor:])
        return "".join(output_parts), traces, warnings

    def _apply_rules(self, text: str, *, collect_trace: bool) -> tuple[str, list[RuleTrace]]:
        traces: list[RuleTrace] = []
        for pass_name in self.rule_pass_order:
            for rule, compiled in self.compiled_rules_by_pass.get(pass_name, []):
                if not rule.enabled:
                    continue
                if not rule.pattern:
                    continue

                if rule.type == "regex":
                    if compiled is None:
                        continue
                    if collect_trace:
                        replaced_text, hit_count = compiled.subn(rule.replacement, text)
                    else:
                        replaced_text = compiled.sub(rule.replacement, text)
                        hit_count = 0
                else:
                    if collect_trace:
                        hit_count = text.count(rule.pattern)
                    else:
                        hit_count = 0
                    replaced_text = text.replace(rule.pattern, rule.replacement)

                if collect_trace and hit_count > 0:
                    traces.append(
                        RuleTrace(
                            rule_id=rule.rule_id,
                            pass_name=rule.pass_name,
                            type=rule.type,
                            pattern=rule.pattern,
                            replacement=rule.replacement,
                            hit_count=hit_count,
                        )
                    )
                text = replaced_text
        return text, traces

    def _post_cleanup(self, text: str) -> tuple[str, list[str]]:
        warnings: list[str] = []

        # L8: 後驗清理
        text = text.replace("這馬咧咧", "這馬咧")
        text = text.replace("真正真", "真")
        text = re.sub(r"([,，。！？!?])\1+", r"\1", text)

        for term in self.residual_terms:
            if term in text:
                warnings.append(f"華語殘留:{term}")

        return text, warnings

    def _enqueue_review_if_needed(
        self,
        *,
        original_text: str,
        output_text: str,
        matches: list[MatchTrace],
        warnings: list[str],
        profile: dict[str, Any] | None,
    ) -> None:
        if not profile or not profile.get("enqueue_review"):
            return

        low_confidence = (not matches) or any(w.startswith("華語殘留") for w in warnings)
        if not low_confidence:
            return

        append_review_item(
            self.data_dir,
            {
                "kind": "online_low_confidence",
                "action": "add_override",
                "owner": profile.get("owner", "runtime"),
                "reason": "auto_enqueued_by_runtime",
                "evidence": {
                    "input": original_text,
                    "output": output_text,
                    "warnings": warnings,
                    "match_count": len(matches),
                    "match_entry_ids": [m.entry_id for m in matches],
                },
            },
        )

    def convert(self, text: str, trace: bool = False, profile: dict[str, Any] | None = None) -> str | ConversionResult:
        started = time.perf_counter()

        normalized = normalize_text(text)
        lexicon_output, matches, lexicon_warnings = self._apply_lexicon_layers(normalized)
        rule_output, rules_applied = self._apply_rules(lexicon_output, collect_trace=trace)
        final_output, cleanup_warnings = self._post_cleanup(rule_output)

        warnings = lexicon_warnings + cleanup_warnings

        self._enqueue_review_if_needed(
            original_text=text,
            output_text=final_output,
            matches=matches,
            warnings=warnings,
            profile=profile,
        )

        latency_ms = (time.perf_counter() - started) * 1000

        if not trace:
            return final_output

        return ConversionResult(
            output=final_output,
            matches=matches,
            rules_applied=rules_applied,
            warnings=warnings,
            latency_ms=latency_ms,
        )
