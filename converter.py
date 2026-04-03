from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_compiler import ensure_runtime_ready
from lexicon_policy import is_machine_generated_override, is_sentence_manual_override, is_trusted_manual_entry
from models import PASS_ORDER, ConversionResult, LexiconEntry, MatchTrace, RuleEntry, RuleTrace
from normalize import normalize_text
from review_queue import append_review_item


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UNICODE_ESCAPE_IN_REPLACEMENT = re.compile(
    r"(?<!\\)(?:\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8}))"
)
ENTRY_ID_PREFIX = "lx_"
RULE_ID_PREFIX = "rl_"
RUNTIME_ID_SUFFIX_LEN = 12
RUNTIME_LEVELS = ("sentence", "phrase", "char")
RUNTIME_TIERS = ("blocked", "manual_hotfix", "manual", "core", "domain", "base")
RUNTIME_TRUSTS = ("human", "machine", "seed")
RUNTIME_CONTEXT_RIGHT_REGEX = "r"
RUNTIME_CONTEXT_LEFT_LITERAL = "l"


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

        self.entries, self.entries_by_index, self.entry_index_by_id = self._load_entry_table(entry_table)
        self.max_phrase_src_len = max(
            (
                len(entry.src)
                for entry in self.entries_by_index
                if entry.level in {"phrase", "sentence"} and entry.status == "active" and not entry.context
            ),
            default=0,
        )
        self.layer_rank_by_index: list[int] = [
            self._layer_rank(entry)
            for entry in self.entries_by_index
        ]
        self.phrase_trie = self._load_phrase_trie(phrase_trie_doc)
        self.char_map = self._load_char_map(char_map_doc)
        self.has_char_entries: bool = bool(self.char_map)
        self.rule_pass_order, self.rules_by_pass = self._load_rule_plan(rule_plan)
        self.compiled_rules_by_pass: dict[str, list[tuple[RuleEntry, re.Pattern[str] | None]]] = {}
        for pass_name, parsed_rules in self.rules_by_pass.items():
            compiled_rules: list[tuple[RuleEntry, re.Pattern[str] | None]] = []
            for rule in parsed_rules:
                if rule.type == "regex":
                    rule.replacement = self._decode_regex_replacement(rule.replacement)
                compiled = re.compile(rule.pattern) if rule.type == "regex" and rule.pattern else None
                compiled_rules.append((rule, compiled))
            self.compiled_rules_by_pass[pass_name] = compiled_rules

        raw_lexicon_stage = str(
            rule_plan.get(
                "x",
                rule_plan.get("pipeline_contract", {}).get("lexicon_stage", "split_char_after_rules"),
            )
        )
        if raw_lexicon_stage not in {"before_rules", "split_char_after_rules"}:
            raise ValueError(f"Unsupported lexicon_stage: {raw_lexicon_stage}")
        self.lexicon_stage: str = raw_lexicon_stage
        residual_terms = rule_plan.get("rt", rule_plan.get("residual_terms", []))
        self.residual_terms: list[str] = (
            [term for term in residual_terms.splitlines() if term]
            if isinstance(residual_terms, str)
            else [term for term in residual_terms if isinstance(term, str) and term]
        )
        self.residual_core_terms: set[str] = {
            term
            for term in rule_plan.get("rc", rule_plan.get("residual_core_terms", []))
            if isinstance(term, str) and term
        }
        protected_cfg = rule_plan.get("protected", {})
        raw_protected_terms = rule_plan.get("pt", rule_plan.get("protected_terms", []))
        if isinstance(raw_protected_terms, str):
            raw_fallback_terms = [term for term in raw_protected_terms.splitlines() if term]
        else:
            raw_fallback_terms = [term for term in raw_protected_terms if isinstance(term, str) and term]
        fallback_terms = sorted(set(raw_fallback_terms), key=lambda item: (-len(item), item))
        self.protected_regex_masks: list[re.Pattern[str]] = []
        if isinstance(protected_cfg, dict):
            trie = protected_cfg.get("trie")
            if isinstance(trie, dict) and isinstance(trie.get("children"), dict) and trie["children"]:
                self.protected_term_trie = trie
                self.protected_terms = sorted(
                    {
                        term
                        for term in protected_cfg.get("terms", fallback_terms)
                        if isinstance(term, str) and term
                    },
                    key=lambda item: (-len(item), item),
                )
            else:
                self.protected_terms = fallback_terms
                self.protected_term_trie = self._build_protected_term_trie(self.protected_terms)

            for pattern in protected_cfg.get("regex_masks", []):
                if not isinstance(pattern, str) or not pattern:
                    continue
                try:
                    self.protected_regex_masks.append(re.compile(pattern))
                except re.error:
                    continue
        else:
            self.protected_terms = fallback_terms
            self.protected_term_trie = self._build_protected_term_trie(self.protected_terms)

        self.sentence_override_map, self.contextual_override_entry_indexes = self._load_override_index(override_index)

        self.blocked_sentence_entry_indexes = [
            entry_index
            for entry_index, entry in enumerate(self.entries_by_index)
            if entry.status == "active" and entry.tier == "blocked" and entry.level == "sentence"
        ]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _decode_regex_replacement(replacement: str) -> str:
        if "\\u" not in replacement and "\\U" not in replacement:
            return replacement

        def _replace_unicode(match: re.Match[str]) -> str:
            codepoint = match.group(1) or match.group(2)
            try:
                return chr(int(codepoint, 16))
            except ValueError:
                return match.group(0)

        return UNICODE_ESCAPE_IN_REPLACEMENT.sub(_replace_unicode, replacement)

    @staticmethod
    def _decode_runtime_context(context: Any) -> dict[str, Any] | None:
        if context is None:
            return None
        if isinstance(context, dict):
            return context
        if (
            isinstance(context, list)
            and len(context) == 2
            and isinstance(context[0], str)
            and isinstance(context[1], str)
        ):
            if context[0] == RUNTIME_CONTEXT_RIGHT_REGEX:
                return {"right_regex": context[1]}
            if context[0] == RUNTIME_CONTEXT_LEFT_LITERAL:
                return {"left_literal": context[1]}
        return None

    @staticmethod
    def _runtime_rule_id(pass_name: str, pattern: str, replacement: str) -> str:
        raw = f"{pass_name}|{pattern}|{replacement}".encode("utf-8")
        return f"{RULE_ID_PREFIX}{hashlib.sha1(raw).hexdigest()[:RUNTIME_ID_SUFFIX_LEN]}"

    def _load_entry_table(
        self,
        entry_table: dict[str, Any],
    ) -> tuple[dict[str, LexiconEntry], list[LexiconEntry], dict[str, int]]:
        if "entries" in entry_table:
            entries = {
                entry_id: LexiconEntry.from_dict(row)
                for entry_id, row in entry_table["entries"].items()
            }
            entries_by_index = list(entries.values())
            entry_index_by_id = {
                entry.entry_id: index
                for index, entry in enumerate(entries_by_index)
            }
            return entries, entries_by_index, entry_index_by_id

        entry_ids: list[str]
        if isinstance(entry_table.get("ih"), str):
            raw_blob = entry_table["ih"]
            if len(raw_blob) % RUNTIME_ID_SUFFIX_LEN != 0:
                raise ValueError("Invalid compact entry id blob")
            entry_ids = [
                f"{ENTRY_ID_PREFIX}{raw_blob[index:index + RUNTIME_ID_SUFFIX_LEN]}"
                for index in range(0, len(raw_blob), RUNTIME_ID_SUFFIX_LEN)
            ]
            if isinstance(entry_table.get("ix"), dict):
                for raw_index, entry_id in entry_table["ix"].items():
                    try:
                        index = int(raw_index)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= index < len(entry_ids) and isinstance(entry_id, str) and entry_id:
                        entry_ids[index] = entry_id
        elif isinstance(entry_table.get("i"), list):
            entry_ids = [
                entry_id if entry_id.startswith(ENTRY_ID_PREFIX) else f"{ENTRY_ID_PREFIX}{entry_id}"
                for entry_id in entry_table["i"]
                if isinstance(entry_id, str) and entry_id
            ]
        else:
            raise ValueError("Unsupported entry_table schema")

        kind_defaults = entry_table.get("k", [])
        raw_rows = entry_table.get("e", [])
        if len(entry_ids) != len(raw_rows):
            raise ValueError("Compact entry_table ids/rows length mismatch")

        entries_by_index: list[LexiconEntry] = []
        entries: dict[str, LexiconEntry] = {}
        for entry_id, raw_row in zip(entry_ids, raw_rows):
            if not isinstance(raw_row, list) or len(raw_row) < 3:
                raise ValueError("Invalid compact entry row")
            src = str(raw_row[0])
            tgt = str(raw_row[1])
            kind_index = int(raw_row[2])
            if kind_index < 0 or kind_index >= len(kind_defaults):
                raise ValueError(f"Invalid compact entry kind: {kind_index}")
            kind_row = kind_defaults[kind_index]
            level = RUNTIME_LEVELS[int(kind_row[0])]
            tier = RUNTIME_TIERS[int(kind_row[1])]
            trust = RUNTIME_TRUSTS[int(kind_row[2])]
            priority = int(kind_row[3])
            score = float(kind_row[4])
            context = None
            if len(raw_row) >= 4:
                priority = int(raw_row[3])
            if len(raw_row) >= 5:
                score = float(raw_row[4])
            if len(raw_row) >= 6:
                context = self._decode_runtime_context(raw_row[5])

            entry = LexiconEntry(
                entry_id=entry_id,
                src=src,
                tgt=tgt,
                level=level,
                tier=tier,
                priority=priority,
                context=context,
                score=score,
                status="active",
                trust=trust,
            )
            entries[entry_id] = entry
            entries_by_index.append(entry)

        entry_index_by_id = {
            entry.entry_id: index
            for index, entry in enumerate(entries_by_index)
        }
        return entries, entries_by_index, entry_index_by_id

    def _normalize_entry_refs(self, raw_refs: Any) -> list[int]:
        if isinstance(raw_refs, int):
            return [raw_refs] if 0 <= raw_refs < len(self.entries_by_index) else []
        if isinstance(raw_refs, str):
            entry_index = self.entry_index_by_id.get(raw_refs)
            return [entry_index] if entry_index is not None else []
        if not isinstance(raw_refs, list):
            return []

        entry_indexes: list[int] = []
        for raw_ref in raw_refs:
            if isinstance(raw_ref, int):
                if 0 <= raw_ref < len(self.entries_by_index):
                    entry_indexes.append(raw_ref)
                continue
            if isinstance(raw_ref, str):
                entry_index = self.entry_index_by_id.get(raw_ref)
                if entry_index is not None:
                    entry_indexes.append(entry_index)
        return entry_indexes

    def _load_phrase_trie(self, phrase_trie_doc: dict[str, Any]) -> dict[str, Any]:
        def decode_legacy(node_doc: dict[str, Any]) -> dict[str, Any]:
            children: dict[str, tuple[str, dict[str, Any]]] = {}
            for ch, child_doc in node_doc.get("children", {}).items():
                if not isinstance(ch, str) or not ch:
                    continue
                children[ch] = ("", decode_legacy(child_doc))
            return {
                "e": self._normalize_entry_refs(node_doc.get("entry_ids", [])),
                "c": children,
            }

        def decode_compact(node_doc: dict[str, Any]) -> dict[str, Any]:
            children: dict[str, tuple[str, dict[str, Any]]] = {}
            for label, child_doc in node_doc.items():
                if not isinstance(label, str) or not label:
                    continue
                if label == "":
                    continue
                child = decode_compact(child_doc)
                children[label[0]] = (label[1:], child)
            return {
                "e": self._normalize_entry_refs(node_doc.get("", [])),
                "c": children,
            }

        if "trie" in phrase_trie_doc:
            return decode_legacy(phrase_trie_doc["trie"])
        if "t" in phrase_trie_doc and isinstance(phrase_trie_doc["t"], dict):
            return decode_compact(phrase_trie_doc["t"])
        return {"e": [], "c": {}}

    def _load_char_map(self, char_map_doc: dict[str, Any]) -> dict[str, list[int]]:
        raw_char_map = char_map_doc.get("m", char_map_doc.get("map", {}))
        if not isinstance(raw_char_map, dict):
            return {}
        return {
            ch: self._normalize_entry_refs(raw_refs)
            for ch, raw_refs in raw_char_map.items()
            if isinstance(ch, str) and ch
        }

    def _load_rule_plan(self, rule_plan: dict[str, Any]) -> tuple[list[str], dict[str, list[RuleEntry]]]:
        if "r" in rule_plan:
            grouped_rows = rule_plan.get("r", [])
            rules_by_pass: dict[str, list[RuleEntry]] = {pass_name: [] for pass_name in PASS_ORDER}
            for pass_name, rows in zip(PASS_ORDER, grouped_rows):
                parsed_rules: list[RuleEntry] = []
                for item in rows:
                    if not isinstance(item, list) or len(item) < 2:
                        continue
                    pattern = str(item[0])
                    replacement = str(item[1])
                    rule_type = "regex" if len(item) >= 3 and int(item[2]) == 1 else "literal"
                    priority = int(item[3]) if len(item) >= 4 else 0
                    parsed_rules.append(
                        RuleEntry(
                            rule_id=self._runtime_rule_id(pass_name, pattern, replacement),
                            pass_name=pass_name,
                            type=rule_type,
                            pattern=pattern,
                            replacement=replacement,
                            priority=priority,
                        )
                    )
                rules_by_pass[pass_name] = parsed_rules
            return list(PASS_ORDER), rules_by_pass

        pass_order = rule_plan.get("pass_order", [])
        rules_by_pass: dict[str, list[RuleEntry]] = {}
        for pass_name, rows in rule_plan.get("rules", {}).items():
            parsed_rules: list[RuleEntry] = []
            for item in rows:
                if isinstance(item, dict) and "id" in item and "p" in item:
                    parsed_rules.append(
                        RuleEntry(
                            rule_id=str(item.get("id", "")),
                            pass_name=pass_name,
                            type="regex" if item.get("k") == "r" else "literal",
                            pattern=str(item.get("p", "")),
                            replacement=str(item.get("r", "")),
                            priority=int(item.get("priority", 0)),
                            enabled=bool(item.get("enabled", True)),
                            note=str(item.get("note", "")),
                        )
                    )
                else:
                    parsed_rules.append(RuleEntry.from_dict(item))
            rules_by_pass[pass_name] = parsed_rules
        return list(pass_order), rules_by_pass

    def _load_override_index(self, override_index: dict[str, Any]) -> tuple[dict[str, list[int]], list[int]]:
        raw_sentence_override_map = override_index.get("s", override_index.get("sentence_override_map", {}))
        sentence_override_map: dict[str, list[int]] = {}
        if isinstance(raw_sentence_override_map, dict):
            for src, raw_entry_refs in raw_sentence_override_map.items():
                if not isinstance(src, str) or not src:
                    continue
                entry_indexes = self._normalize_entry_refs(raw_entry_refs)
                if entry_indexes:
                    sentence_override_map[src] = entry_indexes

        contextual_override_entry_indexes = self._normalize_entry_refs(
            override_index.get("c", override_index.get("contextual_override_ids", []))
        )
        return sentence_override_map, contextual_override_entry_indexes

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

    def _phase_candidate_key(self, candidate: Candidate) -> tuple[int, int, int, float, str]:
        entry = candidate.entry
        return (
            -(candidate.end - candidate.start),
            candidate.layer_rank,
            -entry.priority,
            -entry.score,
            entry.entry_id,
        )

    def _iter_phrase_candidates(self, text: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        root = self.phrase_trie
        entries = self.entries_by_index
        layer_rank_by_index = self.layer_rank_by_index

        for start in range(len(text)):
            node = root
            index = start
            while index < len(text):
                edge = node["c"].get(text[index])
                if edge is None:
                    break
                suffix, child = edge
                if suffix and not text.startswith(suffix, index + 1):
                    break
                node = child
                index += 1 + len(suffix)
                for entry_index in node["e"]:
                    entry = entries[entry_index]
                    if entry.status != "active":
                        continue
                    candidates.append(
                        Candidate(
                            entry=entry,
                            start=start,
                            end=index,
                            layer_rank=layer_rank_by_index[entry_index],
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
        entries = self.entries_by_index
        layer_rank_by_index = self.layer_rank_by_index
        for entry_index in self.contextual_override_entry_indexes:
            entry = entries[entry_index]
            if entry.status != "active":
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
                            layer_rank=layer_rank_by_index[entry_index],
                        )
                    )
                start = found + 1
        return candidates

    def _iter_char_candidates(self, text: str) -> list[Candidate]:
        if not self.has_char_entries:
            return []
        candidates: list[Candidate] = []
        entries = self.entries_by_index
        layer_rank_by_index = self.layer_rank_by_index
        for index, ch in enumerate(text):
            entry_indexes = self.char_map.get(ch, [])
            for entry_index in entry_indexes:
                entry = entries[entry_index]
                if entry.status != "active":
                    continue
                candidates.append(
                    Candidate(
                        entry=entry,
                        start=index,
                        end=index + 1,
                        layer_rank=layer_rank_by_index[entry_index],
                    )
                )
        return candidates

    @staticmethod
    def _span_mask(start: int, end: int) -> int:
        width = end - start
        if width <= 0:
            return 0
        return ((1 << width) - 1) << start

    @staticmethod
    def _length_in_scope(src_len: int, min_src_len: int, max_src_len: int | None) -> bool:
        if src_len < min_src_len:
            return False
        if max_src_len is not None and src_len > max_src_len:
            return False
        return True

    @staticmethod
    def _next_mask_char(used_chars: set[str], text: str) -> str:
        for code_point in range(0xE000, 0xF900):
            token = chr(code_point)
            if token in used_chars or token in text:
                continue
            return token
        raise RuntimeError("No available PUA token for protected-term masking.")

    @staticmethod
    def _build_protected_term_trie(terms: list[str]) -> dict[str, Any]:
        root: dict[str, Any] = {"children": {}}
        for term in terms:
            if len(term) < 2:
                continue
            node = root
            for ch in term:
                node = node["children"].setdefault(ch, {"children": {}})
            node["term"] = term
        return root

    def _has_longer_runtime_phrase(self, text: str, start: int, min_len_exclusive: int) -> bool:
        """Return True when runtime trie has a longer phrase starting at `start`.

        This prevents protected short terms (for example, identity terms like "身體")
        from masking longer, intentional rewrites (for example, "身體不適" -> ...).
        """
        node = self.phrase_trie
        idx = start
        while idx < len(text):
            edge = node["c"].get(text[idx])
            if edge is None:
                break
            suffix, child = edge
            if suffix and not text.startswith(suffix, idx + 1):
                break
            node = child
            idx += 1 + len(suffix)
            if idx - start > min_len_exclusive and node["e"]:
                return True
        return False

    def _is_inside_longer_runtime_phrase(self, text: str, span_start: int, span_end: int) -> bool:
        """Return True if span is covered by any longer runtime phrase match."""
        if span_end <= span_start:
            return False
        if self.max_phrase_src_len <= 0:
            return False

        lookback = self.max_phrase_src_len - 1
        start_min = max(0, span_start - lookback)

        for start in range(start_min, span_start + 1):
            node = self.phrase_trie
            idx = start
            while idx < len(text):
                edge = node["c"].get(text[idx])
                if edge is None:
                    break
                suffix, child = edge
                if suffix and not text.startswith(suffix, idx + 1):
                    break
                node = child
                idx += 1 + len(suffix)
                if idx < span_end:
                    continue
                if node["e"] and (start < span_start or idx > span_end):
                    return True
        return False

    def _overlaps_runtime_phrase(self, text: str, span_start: int, span_end: int) -> bool:
        """Return True if a protected span intersects any multi-char runtime phrase.

        This prevents protected terms like "下班" from masking across phrase
        boundaries inside inputs such as "幫我查一下班次".
        """
        if span_end <= span_start:
            return False
        if self.max_phrase_src_len <= 1:
            return False

        lookback = self.max_phrase_src_len - 1
        start_min = max(0, span_start - lookback)
        start_max = min(len(text) - 1, span_end - 1)

        for start in range(start_min, start_max + 1):
            node = self.phrase_trie
            idx = start
            while idx < len(text):
                edge = node["c"].get(text[idx])
                if edge is None:
                    break
                suffix, child = edge
                if suffix and not text.startswith(suffix, idx + 1):
                    break
                node = child
                idx += 1 + len(suffix)
                if not node["e"]:
                    continue
                if idx <= span_start or start >= span_end:
                    continue
                if start == span_start and idx == span_end:
                    continue
                # Allow masking a protected proper noun even if smaller runtime
                # phrases exist entirely inside it; only reject overlaps that
                # cross the protected span boundary.
                if start >= span_start and idx <= span_end:
                    continue
                return True
        return False

    def _mask_protected_regexes(
        self,
        text: str,
        *,
        used_chars: set[str],
    ) -> tuple[str, dict[str, str]]:
        if not text or not self.protected_regex_masks:
            return text, {}

        token_map: dict[str, str] = {}
        masked = text
        for compiled in self.protected_regex_masks:
            parts: list[str] = []
            cursor = 0
            replaced = False
            for match in compiled.finditer(masked):
                start, end = match.span()
                if end <= start:
                    continue
                replaced = True
                original = masked[start:end]
                mask_char = self._next_mask_char(used_chars, masked)
                used_chars.add(mask_char)
                masked_text = mask_char * (end - start)
                parts.append(masked[cursor:start])
                parts.append(masked_text)
                token_map[masked_text] = original
                cursor = end
            if replaced:
                parts.append(masked[cursor:])
                masked = "".join(parts)
        return masked, token_map

    def _mask_protected_terms(
        self,
        text: str,
        *,
        respect_runtime_phrase_overlap: bool = True,
    ) -> tuple[str, dict[str, str]]:
        has_regex_masks = bool(self.protected_regex_masks)
        has_trie_masks = bool(self.protected_term_trie.get("children"))
        if not text or (not has_regex_masks and not has_trie_masks):
            return text, {}

        used_chars: set[str] = set()
        masked_text, regex_token_map = self._mask_protected_regexes(text, used_chars=used_chars)
        if not has_trie_masks:
            return masked_text, regex_token_map

        parts: list[str] = []
        token_map: dict[str, str] = dict(regex_token_map)
        cursor = 0
        text_len = len(masked_text)
        root = self.protected_term_trie

        while cursor < text_len:
            node = root
            idx = cursor
            longest_end = -1
            while idx < text_len:
                child = node["children"].get(masked_text[idx])
                if child is None:
                    break
                node = child
                idx += 1
                if "term" in node:
                    longest_end = idx

            if longest_end < 0:
                parts.append(masked_text[cursor])
                cursor += 1
                continue

            # Keep raw text when the protected span can participate in a longer
            # runtime phrase, so conversion can win over short protected masking.
            span_covers_full_text = cursor == 0 and longest_end == text_len
            if respect_runtime_phrase_overlap and not span_covers_full_text and (
                self._has_longer_runtime_phrase(masked_text, cursor, longest_end - cursor)
                or self._is_inside_longer_runtime_phrase(masked_text, cursor, longest_end)
                or self._overlaps_runtime_phrase(masked_text, cursor, longest_end)
            ):
                parts.append(masked_text[cursor])
                cursor += 1
                continue

            original = masked_text[cursor:longest_end]
            mask_char = self._next_mask_char(used_chars, masked_text)
            used_chars.add(mask_char)
            mask_token = mask_char * (longest_end - cursor)
            parts.append(mask_token)
            token_map[mask_token] = original
            cursor = longest_end

        return "".join(parts), token_map

    @staticmethod
    def _unmask_protected_terms(text: str, token_map: dict[str, str]) -> str:
        if not token_map:
            return text
        restored = text
        for masked_text, original in sorted(token_map.items(), key=lambda item: -len(item[0])):
            restored = restored.replace(masked_text, original)
        return restored

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

    def _select_leftmost_maximum(
        self,
        candidates: list[Candidate],
        *,
        reserved: list[Candidate] | None = None,
        text_length: int,
    ) -> list[Candidate]:
        occupied_mask = 0
        for candidate in reserved or []:
            occupied_mask |= self._span_mask(candidate.start, candidate.end)

        by_start: dict[int, list[Candidate]] = {}
        for candidate in candidates:
            if candidate.start >= candidate.end:
                continue
            by_start.setdefault(candidate.start, []).append(candidate)

        selected: list[Candidate] = []
        cursor = 0
        while cursor < text_length:
            if occupied_mask & (1 << cursor):
                cursor += 1
                continue

            bucket = by_start.get(cursor, [])
            viable: list[Candidate] = []
            for candidate in bucket:
                span_mask = self._span_mask(candidate.start, candidate.end)
                if occupied_mask & span_mask:
                    continue
                viable.append(candidate)

            if not viable:
                cursor += 1
                continue

            chosen = min(viable, key=self._phase_candidate_key)
            selected.append(chosen)
            occupied_mask |= self._span_mask(chosen.start, chosen.end)
            cursor = chosen.end

        return selected

    def _collect_blocked_candidates(self, text: str, phrase_candidates: list[Candidate]) -> list[Candidate]:
        blocked = [candidate for candidate in phrase_candidates if candidate.entry.tier == "blocked"]

        for entry_index in self.blocked_sentence_entry_indexes:
            entry = self.entries_by_index[entry_index]
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
                for entry_index in self.char_map.get(ch, []):
                    entry = self.entries_by_index[entry_index]
                    if entry.status != "active":
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

        return self._select_leftmost_maximum(blocked, text_length=len(text))

    def _apply_exact_sentence_override(self, text: str) -> tuple[str | None, list[MatchTrace], list[str]]:
        if not text:
            return None, [], []

        sentence_override_ids = self.sentence_override_map.get(text, [])
        if not sentence_override_ids:
            return None, [], []

        all_phrase_candidates = self._iter_phrase_candidates(text)
        blocked_candidates = self._collect_blocked_candidates(text, all_phrase_candidates)
        warnings = [f"blocked:{blocked.entry.entry_id}:{blocked.entry.src}" for blocked in blocked_candidates]

        sentence_candidates = [
            Candidate(entry=self.entries_by_index[entry_index], start=0, end=len(text), layer_rank=1)
            for entry_index in sentence_override_ids
        ]
        if not sentence_candidates:
            return None, [], warnings

        sentence_selected = self._select_leftmost_maximum(
            sentence_candidates,
            reserved=blocked_candidates,
            text_length=len(text),
        )
        if not sentence_selected:
            return None, [], warnings

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

    def _apply_lexicon_layers(
        self,
        text: str,
        *,
        min_src_len: int = 1,
        max_src_len: int | None = None,
        include_char_entries: bool = True,
        allow_sentence_override: bool = True,
    ) -> tuple[str, list[MatchTrace], list[str]]:
        warnings: list[str] = []
        all_phrase_candidates = self._iter_phrase_candidates(text)
        blocked_candidates = self._collect_blocked_candidates(text, all_phrase_candidates)

        for blocked in blocked_candidates:
            warnings.append(f"blocked:{blocked.entry.entry_id}:{blocked.entry.src}")

        if allow_sentence_override and self._length_in_scope(len(text), min_src_len, max_src_len):
            sentence_override_ids = self.sentence_override_map.get(text, [])
            sentence_candidates = [
                Candidate(entry=self.entries_by_index[entry_index], start=0, end=len(text), layer_rank=1)
                for entry_index in sentence_override_ids
            ]
            if sentence_candidates:
                sentence_selected = self._select_leftmost_maximum(
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

        contextual_candidates = [
            candidate
            for candidate in self._iter_contextual_candidates(text)
            if self._length_in_scope(candidate.end - candidate.start, min_src_len, max_src_len)
        ]
        phrase_non_blocked = [
            candidate
            for candidate in all_phrase_candidates
            if candidate.entry.tier != "blocked"
            and self._length_in_scope(candidate.end - candidate.start, min_src_len, max_src_len)
        ]

        all_candidates = contextual_candidates + phrase_non_blocked
        if include_char_entries and self.has_char_entries:
            char_candidates = [
                candidate
                for candidate in self._iter_char_candidates(text)
                if self._length_in_scope(candidate.end - candidate.start, min_src_len, max_src_len)
            ]
            all_candidates.extend(char_candidates)

        selected = self._select_leftmost_maximum(
            all_candidates,
            reserved=blocked_candidates,
            text_length=len(text),
        )

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

    def _apply_rules(
        self,
        text: str,
        *,
        collect_trace: bool,
        skip_passes: set[str] | None = None,
    ) -> tuple[str, list[RuleTrace]]:
        traces: list[RuleTrace] = []
        skip = skip_passes or set()
        for pass_name in self.rule_pass_order:
            if pass_name in skip:
                continue
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
                warning_prefix = "核心漏轉" if term in self.residual_core_terms else "華語殘留"
                warnings.append(f"{warning_prefix}:{term}")

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

        low_confidence = (not matches) or any(
            w.startswith("華語殘留") or w.startswith("核心漏轉")
            for w in warnings
        )
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
        preserve_spacing = bool(profile and profile.get("preserve_spacing"))
        normalized = normalize_text(
            text,
            compress_spaces=not preserve_spacing,
            trim_outer=not preserve_spacing,
        )
        exact_sentence_output, exact_matches, exact_warnings = self._apply_exact_sentence_override(normalized)
        # Protect allowlisted multi-character tokens before lexicon/rule passes.
        masked_input, protected_token_map = self._mask_protected_terms(normalized)
        skip_passes = {"normalization"} if preserve_spacing else set()

        if exact_matches:
            # Exact sentence overrides can still contain protected proper nouns
            # such as official place names, so mask them before char-level passes.
            exact_masked_output, protected_token_map = self._mask_protected_terms(
                exact_sentence_output,
                respect_runtime_phrase_overlap=False,
            )
            rule_output, rules_applied = self._apply_rules(
                exact_masked_output,
                collect_trace=trace,
                skip_passes=skip_passes,
            )
            lexicon_output, post_matches, post_warnings = self._apply_lexicon_layers(
                rule_output,
                min_src_len=1,
                max_src_len=1,
                include_char_entries=True,
                allow_sentence_override=False,
            )
            matches = exact_matches + post_matches
            lexicon_warnings = exact_warnings + post_warnings
        elif self.lexicon_stage == "split_char_after_rules":
            pre_rule_output, pre_matches, pre_warnings = self._apply_lexicon_layers(
                masked_input,
                min_src_len=2,
                include_char_entries=False,
                allow_sentence_override=True,
            )
            rule_output, rules_applied = self._apply_rules(
                pre_rule_output,
                collect_trace=trace,
                skip_passes=skip_passes,
            )
            lexicon_output, post_matches, post_warnings = self._apply_lexicon_layers(
                rule_output,
                min_src_len=1,
                max_src_len=1,
                include_char_entries=True,
                allow_sentence_override=False,
            )
            matches = pre_matches + post_matches
            lexicon_warnings = pre_warnings + post_warnings
        else:
            lexicon_output, matches, lexicon_warnings = self._apply_lexicon_layers(masked_input)
            rule_output, rules_applied = self._apply_rules(
                lexicon_output,
                collect_trace=trace,
                skip_passes=skip_passes,
            )
            lexicon_output = rule_output

        final_output, cleanup_warnings = self._post_cleanup(lexicon_output)
        final_output = self._unmask_protected_terms(final_output, protected_token_map)

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
