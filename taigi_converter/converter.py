from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .artifact_compiler import (
    ARTIFACT_FILES,
    ARTIFACT_SCHEMA_VERSIONS,
    COMPILER_VERSION,
    MANIFEST_VERSION,
    RULE_PLAN_LEXICON_STAGE_KEY,
    RULE_PLAN_STRICT_PROTECTED_TERMS_KEY,
    ensure_runtime_ready,
    validate_artifact_documents,
)
from .context_policy import decode_runtime_context
from .lexicon_policy import is_sentence_manual_override, is_trusted_context_entry, runtime_layer_rank
from .models import (
    PASS_ORDER,
    ConversionResult,
    MatchTrace,
    RuleTrace,
    RuntimeLexiconEntry,
    RuntimeRuleEntry,
)
from .normalize import normalize_cjk_spacing, normalize_text
from .review_queue import append_review_item

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UNICODE_ESCAPE_IN_REPLACEMENT = re.compile(r"(?<!\\)(?:\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8}))")
ENTRY_ID_PREFIX = "lx_"
RULE_ID_PREFIX = "rl_"
RUNTIME_ID_SUFFIX_LEN = 12
RUNTIME_LEVELS = ("sentence", "phrase", "char")
RUNTIME_TIERS = ("blocked", "manual_hotfix", "manual", "core", "domain", "base")
RUNTIME_TRUSTS = ("human", "ai_reviewed", "machine", "seed")
SHADOW_MARKER_CANDIDATES = tuple(chr(codepoint) for codepoint in range(0xFDD0, 0xFDF0)) + tuple(
    chr((plane << 16) | suffix) for plane in range(17) for suffix in (0xFFFE, 0xFFFF)
)
REPEATED_PUNCTUATION_RE = re.compile(r"([,，。！？!?])\1+")
_TIME_AMOUNT_PATTERN = r"[0-9一二兩三四五六七八九十百千零〇半]+"
_TIME_UNIT_PATTERN = r"(?:个點鐘|點半|點鐘|點|分鐘|刻鐘|工|天|日|个月|月|年|冬|禮拜|星期|週|周)"
_TIME_EXPRESSION_PATTERN = rf"{_TIME_AMOUNT_PATTERN}{_TIME_UNIT_PATTERN}"
_NAMED_TIME_PATTERN = (
    r"(?:今仔日|明仔載|明仔|後日|大後日|昨昏|前日|大前日|"
    r"這禮拜|下禮拜|頂禮拜|這个月|後個月|頂個月|"
    r"月頭|月中|月尾|年頭|冬尾)"
)
POST_UNMASK_TIME_SUBSTITUTIONS = (
    (
        re.compile(
            rf"({_TIME_AMOUNT_PATTERN})天(?=(?:以內|以前|之前|以後|之後|了後|內|前|後|到|至))"
        ),
        r"\1工",
    ),
    (re.compile(rf"({_TIME_EXPRESSION_PATTERN})(?:以前|之前)"), r"\1進前"),
    (re.compile(rf"({_TIME_EXPRESSION_PATTERN})(?:以後|之後|了後)"), r"\1後"),
    (re.compile(rf"({_TIME_EXPRESSION_PATTERN})以內"), r"\1內"),
    (re.compile(rf"({_NAMED_TIME_PATTERN})(?:以前|之前)"), r"\1進前"),
    (re.compile(rf"({_NAMED_TIME_PATTERN})(?:以後|之後|了後)"), r"\1後"),
    (re.compile(rf"({_NAMED_TIME_PATTERN})以內"), r"\1內"),
)


@dataclass(frozen=True, slots=True)
class Candidate:
    entry: RuntimeLexiconEntry
    start: int
    end: int
    layer_rank: int


@dataclass(frozen=True, slots=True)
class RuntimeRule:
    rule: RuntimeRuleEntry
    compiled: re.Pattern[str] | None
    required_literal: str | None


@dataclass(frozen=True, slots=True)
class _TextSegment:
    text: str
    protected: bool = False


@dataclass(frozen=True, slots=True)
class _ProtectedText:
    segments: tuple[_TextSegment, ...]

    @classmethod
    def plain(cls, text: str) -> _ProtectedText:
        return cls((_TextSegment(text),)) if text else cls(())

    def render(self) -> str:
        return "".join(segment.text for segment in self.segments)


@dataclass(frozen=True, slots=True)
class _ShadowLayout:
    protected_by_marker: tuple[tuple[str, str], ...]


@dataclass(slots=True)
class _ReviewDiagnostics:
    ambiguous_candidate_count: int = 0


def _linear_identity_ratio(source: str, target: str) -> float:
    """Return a linear-time character-overlap ratio for review heuristics.

    Review diagnostics only need a stable estimate of how much text survived the
    conversion.  A full edit-distance alignment is unnecessary here and can be
    quadratic for repeated user input.
    """

    total_length = len(source) + len(target)
    if total_length == 0:
        return 1.0
    shared_characters = sum((Counter(source) & Counter(target)).values())
    return 2.0 * shared_characters / total_length


def _competing_target_count(chosen: Candidate, candidates: list[Candidate]) -> int:
    """Count distinct outputs competing for exactly the chosen input span."""

    return len(
        {
            candidate.entry.tgt
            for candidate in candidates
            if candidate.start == chosen.start
            and candidate.end == chosen.end
            and candidate.entry.tgt != chosen.entry.tgt
        }
    )


@dataclass(frozen=True, slots=True)
class _RuntimeState:
    entries: Mapping[str, RuntimeLexiconEntry]
    entries_by_index: tuple[RuntimeLexiconEntry, ...]
    entry_index_by_id: Mapping[str, int]
    max_phrase_src_len: int
    layer_rank_by_index: tuple[int, ...]
    phrase_trie: Mapping[str, Any]
    single_char_phrase_map: Mapping[str, tuple[int, ...]]
    char_map: Mapping[str, tuple[int, ...]]
    has_char_entries: bool
    has_blocked_phrase_entries: bool
    has_blocked_char_entries: bool
    rule_pass_order: tuple[str, ...]
    rules_by_pass: Mapping[str, tuple[RuntimeRuleEntry, ...]]
    compiled_rules_by_pass: Mapping[str, tuple[RuntimeRule, ...]]
    lexicon_stage: str
    residual_terms: tuple[str, ...]
    residual_core_terms: frozenset[str]
    protected_regex_masks: tuple[re.Pattern[str], ...]
    protected_terms: tuple[str, ...]
    protected_term_trie: Mapping[str, Any]
    strict_protected_terms: tuple[str, ...]
    strict_protected_term_trie: Mapping[str, Any]
    number_bearing_lexicon_terms: tuple[str, ...]
    number_bearing_lexicon_trie: Mapping[str, Any]
    sentence_override_map: Mapping[str, tuple[int, ...]]
    contextual_override_entry_indexes: tuple[int, ...]
    contextual_entry_indexes_by_first_char: Mapping[str, tuple[int, ...]]
    shadow_forbidden_chars: frozenset[str]


class _ArtifactGenerationMismatch(RuntimeError):
    pass


class TaigiConverter:
    ARTIFACT_LOAD_RETRIES = 60
    ARTIFACT_RETRY_INTERVAL_SECONDS = 0.05
    RUNTIME_CACHE_MAX_ENTRIES = 4
    RUNTIME_LOAD_WAIT_TIMEOUT_SECONDS = 30.0

    _runtime_cache: OrderedDict[tuple[str, str], _RuntimeState] = OrderedDict()
    _runtime_cache_lock = threading.RLock()
    _runtime_load_events: dict[str, threading.Event] = {}

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        fail_on_mask: bool = False,
        auto_prepare: bool = False,
        source_data_dir: Path | str | None = None,
        review_data_dir: Path | str | None = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.artifact_dir = self.data_dir / "artifacts"
        self.review_data_dir = Path(review_data_dir) if review_data_dir is not None else None

        if auto_prepare:
            source_dir = Path(source_data_dir) if source_data_dir else self.data_dir
            ensure_runtime_ready(
                source_dir,
                output_data_dir=self.data_dir,
                fail_on_mask=fail_on_mask,
            )

        artifact_root = str(self.artifact_dir.resolve())
        manifest_path = self.artifact_dir / "manifest.json"
        while True:
            manifest_bytes = self._read_manifest_bytes(manifest_path)
            cache_key = (artifact_root, hashlib.sha256(manifest_bytes).hexdigest())
            with self._runtime_cache_lock:
                cached = self._runtime_cache.get(cache_key)
                if cached is not None:
                    self._runtime_cache.move_to_end(cache_key)
                    self._install_runtime_state(cached)
                    return

                load_event = self._runtime_load_events.get(artifact_root)
                if load_event is None:
                    load_event = threading.Event()
                    self._runtime_load_events[artifact_root] = load_event
                    owns_load = True
                else:
                    owns_load = False

            if not owns_load:
                if not load_event.wait(self.RUNTIME_LOAD_WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError(
                        f"等待 runtime artifacts single-flight 載入逾時；artifact_root={artifact_root!r}"
                    )
                continue

            try:
                state, loaded_cache_key = self._load_consistent_runtime(manifest_path, artifact_root)
                with self._runtime_cache_lock:
                    stale_keys = [key for key in self._runtime_cache if key[0] == artifact_root]
                    for stale_key in stale_keys:
                        self._runtime_cache.pop(stale_key, None)
                    self._runtime_cache[loaded_cache_key] = state
                    self._runtime_cache.move_to_end(loaded_cache_key)
                    while len(self._runtime_cache) > self.RUNTIME_CACHE_MAX_ENTRIES:
                        self._runtime_cache.popitem(last=False)
                self._install_runtime_state(state)
                return
            finally:
                with self._runtime_cache_lock:
                    completed = self._runtime_load_events.pop(artifact_root, None)
                    if completed is not None:
                        completed.set()

    @staticmethod
    def _read_manifest_bytes(manifest_path: Path) -> bytes:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"找不到 runtime artifacts: {manifest_path}。請先執行 scripts/build_runtime_artifacts.py。"
            )
        return manifest_path.read_bytes()

    def _load_consistent_runtime(
        self,
        manifest_path: Path,
        artifact_root: str,
    ) -> tuple[_RuntimeState, tuple[str, str]]:
        last_mismatch: Exception | None = None
        for _ in range(self.ARTIFACT_LOAD_RETRIES):
            manifest_bytes = self._read_manifest_bytes(manifest_path)
            cache_key = (artifact_root, hashlib.sha256(manifest_bytes).hexdigest())
            with self._runtime_cache_lock:
                cached = self._runtime_cache.get(cache_key)
                if cached is not None:
                    self._runtime_cache.move_to_end(cache_key)
                    return cached, cache_key
            try:
                self._load_artifacts(manifest_bytes)
            except _ArtifactGenerationMismatch as exc:
                last_mismatch = exc
                time.sleep(self.ARTIFACT_RETRY_INTERVAL_SECONDS)
                continue
            return self._freeze_runtime_state(), cache_key

        detail = f"：{last_mismatch}" if last_mismatch is not None else ""
        raise RuntimeError("runtime artifacts 正在更新或內容不一致；請等待建置完成後重試" + detail) from last_mismatch

    @classmethod
    def clear_runtime_cache(cls) -> None:
        """Drop completed cached generations without disrupting active loads."""
        with cls._runtime_cache_lock:
            cls._runtime_cache.clear()

    @classmethod
    def runtime_cache_info(cls) -> dict[str, int]:
        with cls._runtime_cache_lock:
            return {
                "size": len(cls._runtime_cache),
                "max_size": cls.RUNTIME_CACHE_MAX_ENTRIES,
                "loads_in_progress": len(cls._runtime_load_events),
            }

    @staticmethod
    def _freeze_tree(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({key: TaigiConverter._freeze_tree(item) for key, item in value.items()})
        if isinstance(value, list | tuple):
            return tuple(TaigiConverter._freeze_tree(item) for item in value)
        if isinstance(value, set | frozenset):
            return frozenset(TaigiConverter._freeze_tree(item) for item in value)
        return value

    def _freeze_runtime_state(self) -> _RuntimeState:
        return _RuntimeState(
            entries=MappingProxyType(dict(self.entries)),
            entries_by_index=tuple(self.entries_by_index),
            entry_index_by_id=MappingProxyType(dict(self.entry_index_by_id)),
            max_phrase_src_len=self.max_phrase_src_len,
            layer_rank_by_index=tuple(self.layer_rank_by_index),
            phrase_trie=self._freeze_tree(self.phrase_trie),
            single_char_phrase_map=MappingProxyType(
                {key: tuple(value) for key, value in self.single_char_phrase_map.items()}
            ),
            char_map=MappingProxyType({key: tuple(value) for key, value in self.char_map.items()}),
            has_char_entries=self.has_char_entries,
            has_blocked_phrase_entries=self.has_blocked_phrase_entries,
            has_blocked_char_entries=self.has_blocked_char_entries,
            rule_pass_order=tuple(self.rule_pass_order),
            rules_by_pass=MappingProxyType({key: tuple(value) for key, value in self.rules_by_pass.items()}),
            compiled_rules_by_pass=MappingProxyType(
                {key: tuple(value) for key, value in self.compiled_rules_by_pass.items()}
            ),
            lexicon_stage=self.lexicon_stage,
            residual_terms=tuple(self.residual_terms),
            residual_core_terms=frozenset(self.residual_core_terms),
            protected_regex_masks=tuple(self.protected_regex_masks),
            protected_terms=tuple(self.protected_terms),
            protected_term_trie=self._freeze_tree(self.protected_term_trie),
            strict_protected_terms=tuple(self.strict_protected_terms),
            strict_protected_term_trie=self._freeze_tree(self.strict_protected_term_trie),
            number_bearing_lexicon_terms=tuple(self.number_bearing_lexicon_terms),
            number_bearing_lexicon_trie=self._freeze_tree(self.number_bearing_lexicon_trie),
            sentence_override_map=MappingProxyType(
                {key: tuple(value) for key, value in self.sentence_override_map.items()}
            ),
            contextual_override_entry_indexes=tuple(self.contextual_override_entry_indexes),
            contextual_entry_indexes_by_first_char=MappingProxyType(
                {key: tuple(value) for key, value in self.contextual_entry_indexes_by_first_char.items()}
            ),
            shadow_forbidden_chars=frozenset(
                char for entry in self.entries_by_index for value in (entry.src, entry.tgt) for char in value
            )
            | frozenset(
                char
                for rules in self.rules_by_pass.values()
                for rule in rules
                for value in (rule.pattern, rule.replacement)
                for char in value
            ),
        )

    def _install_runtime_state(self, state: _RuntimeState) -> None:
        self._runtime = state
        for field in fields(state):
            setattr(self, field.name, getattr(state, field.name))

    def _load_artifacts(self, manifest_bytes: bytes) -> None:
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("runtime manifest 不是有效 JSON") from exc
        manifest_version = manifest.get("v", manifest.get("version"))
        compiler_version = manifest.get("cv", manifest.get("compiler_version"))
        source_digest = manifest.get("sd", manifest.get("source_digest"))
        manifest_stage = manifest.get("ls", manifest.get("lexicon_stage"))
        manifest_schema_versions = manifest.get("sv", manifest.get("artifact_schema_versions"))
        if manifest_version != MANIFEST_VERSION or compiler_version != COMPILER_VERSION:
            raise RuntimeError("runtime artifacts 版本不相容；請重新執行 scripts/build_runtime_artifacts.py")
        if (
            not isinstance(source_digest, str)
            or len(source_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in source_digest)
        ):
            raise RuntimeError("runtime manifest 缺少有效的 source_digest")

        if manifest_schema_versions != ARTIFACT_SCHEMA_VERSIONS:
            raise RuntimeError("runtime manifest 的 artifact schema contract 不相容；請重新建置 runtime artifacts")
        if manifest_stage not in {"before_rules", "split_char_after_rules"}:
            raise RuntimeError("runtime manifest 缺少有效的 lexicon stage")

        artifact_hashes = manifest.get("ah", manifest.get("artifact_hashes"))
        required_artifacts = ARTIFACT_FILES
        if not isinstance(artifact_hashes, dict) or any(
            not isinstance(artifact_hashes.get(name), str) or len(artifact_hashes[name]) != 64
            for name in required_artifacts
        ):
            raise RuntimeError("runtime manifest 缺少 artifact checksums")

        documents: dict[str, dict[str, Any]] = {}
        for name in required_artifacts:
            path = self.artifact_dir / name
            try:
                payload = path.read_bytes()
            except FileNotFoundError as exc:
                raise _ArtifactGenerationMismatch(f"runtime artifact 暫時缺失: {path}") from exc
            if hashlib.sha256(payload).hexdigest() != artifact_hashes[name]:
                raise _ArtifactGenerationMismatch(f"runtime artifact checksum 不符: {path}")
            try:
                document = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"runtime artifact 不是有效 JSON: {path}") from exc
            if not isinstance(document, dict):
                raise RuntimeError(f"runtime artifact 根節點必須是 object: {path}")
            documents[name] = document

        for name, expected_version in ARTIFACT_SCHEMA_VERSIONS.items():
            actual_version = documents[name].get("v")
            if actual_version != expected_version:
                raise RuntimeError(
                    f"runtime artifact schema 不相容: {name} v={actual_version!r}，預期 {expected_version}"
                )
        validate_artifact_documents(documents)

        entry_table = documents["entry_table.json"]
        phrase_trie_doc = documents["phrase_trie.json"]
        char_map_doc = documents["char_map.json"]
        rule_plan = documents["rule_plan.json"]
        override_index = documents["override_index.json"]

        self.entries, self.entries_by_index, self.entry_index_by_id = self._load_entry_table(entry_table)
        self.max_phrase_src_len = max(
            (
                len(entry.src)
                for entry in self.entries_by_index
                if entry.level in {"phrase", "sentence"} and entry.status == "active" and entry.context is None
            ),
            default=0,
        )
        self.layer_rank_by_index: list[int] = [self._layer_rank(entry) for entry in self.entries_by_index]
        self.phrase_trie = self._load_phrase_trie(phrase_trie_doc)
        self.char_map = self._load_char_map(char_map_doc)
        self.has_char_entries: bool = bool(self.char_map)
        self.rule_pass_order, self.rules_by_pass = self._load_rule_plan(rule_plan)
        self.compiled_rules_by_pass: dict[str, list[RuntimeRule]] = {}
        for pass_name, parsed_rules in self.rules_by_pass.items():
            compiled_rules: list[RuntimeRule] = []
            for parsed_rule in parsed_rules:
                if not parsed_rule.enabled or not parsed_rule.pattern:
                    continue
                replacement = (
                    self._decode_regex_replacement(parsed_rule.replacement)
                    if parsed_rule.type == "regex"
                    else parsed_rule.replacement
                )
                rule = RuntimeRuleEntry(
                    rule_id=parsed_rule.rule_id,
                    pass_name=parsed_rule.pass_name,
                    type=parsed_rule.type,
                    pattern=parsed_rule.pattern,
                    replacement=replacement,
                    priority=parsed_rule.priority,
                    enabled=parsed_rule.enabled,
                    note=parsed_rule.note,
                )
                compiled = re.compile(rule.pattern) if rule.type == "regex" and rule.pattern else None
                required_literal = self._regex_required_literal(rule.pattern) if rule.type == "regex" else rule.pattern
                compiled_rules.append(RuntimeRule(rule, compiled, required_literal))
            self.rules_by_pass[pass_name] = [item.rule for item in compiled_rules]
            self.compiled_rules_by_pass[pass_name] = compiled_rules

        raw_lexicon_stage = rule_plan.get(RULE_PLAN_LEXICON_STAGE_KEY)
        if raw_lexicon_stage not in {"before_rules", "split_char_after_rules"}:
            raise RuntimeError(f"rule_plan 缺少有效的 lexicon stage: {RULE_PLAN_LEXICON_STAGE_KEY}")
        if raw_lexicon_stage != manifest_stage:
            raise RuntimeError(
                f"runtime lexicon stage contract 不一致：manifest={manifest_stage!r}, rule_plan={raw_lexicon_stage!r}"
            )
        self.lexicon_stage: str = raw_lexicon_stage
        self.residual_terms = list(rule_plan["rt"])
        if len(set(self.residual_terms)) != len(self.residual_terms):
            raise ValueError("rule_plan residual terms 含重複項目")
        raw_residual_core_terms = rule_plan["rc"]
        if len(set(raw_residual_core_terms)) != len(raw_residual_core_terms):
            raise ValueError("rule_plan residual core terms 含重複項目")
        self.residual_core_terms = set(raw_residual_core_terms)
        if not self.residual_core_terms.issubset(self.residual_terms):
            raise ValueError("rule_plan residual core terms 必須是 residual terms 子集")

        raw_protected_terms = [term for term in rule_plan["pt"].splitlines() if term]
        if len(set(raw_protected_terms)) != len(raw_protected_terms):
            raise ValueError("rule_plan protected terms 含重複項目")
        self.protected_terms = sorted(raw_protected_terms, key=lambda item: (-len(item), item))
        self.protected_term_trie = self._build_protected_term_trie(self.protected_terms)
        raw_strict_protected_terms = [
            term for term in rule_plan[RULE_PLAN_STRICT_PROTECTED_TERMS_KEY].splitlines() if term
        ]
        if len(set(raw_strict_protected_terms)) != len(raw_strict_protected_terms):
            raise ValueError("rule_plan strict protected terms 含重複項目")
        if not set(raw_strict_protected_terms).issubset(self.protected_terms):
            raise ValueError("rule_plan strict protected terms 必須是 protected terms 子集")
        self.strict_protected_terms = sorted(raw_strict_protected_terms, key=lambda item: (-len(item), item))
        self.strict_protected_term_trie = self._build_protected_term_trie(self.strict_protected_terms)
        self.protected_regex_masks = []
        self.number_bearing_lexicon_terms = sorted(
            {
                entry.src
                for entry in self.entries_by_index
                if entry.status == "active"
                and entry.level in {"sentence", "phrase"}
                and any("0" <= char <= "9" for char in entry.src)
            },
            key=lambda item: (-len(item), item),
        )
        self.number_bearing_lexicon_trie = self._build_protected_term_trie(self.number_bearing_lexicon_terms)

        self.sentence_override_map, self.contextual_override_entry_indexes = self._load_override_index(override_index)
        self._validate_runtime_reference_contract()
        self.contextual_entry_indexes_by_first_char: dict[str, list[int]] = {}
        for entry_index in self.contextual_override_entry_indexes:
            first_char = self.entries_by_index[entry_index].src[0]
            self.contextual_entry_indexes_by_first_char.setdefault(first_char, []).append(entry_index)

        self.single_char_phrase_map: dict[str, list[int]] = {}
        self.has_blocked_phrase_entries = False
        self.has_blocked_char_entries = False
        for entry_index, entry in enumerate(self.entries_by_index):
            if entry.status != "active" or entry.context is not None:
                continue
            if entry.level in {"phrase", "sentence"} and not is_sentence_manual_override(entry):
                if len(entry.src) == 1:
                    self.single_char_phrase_map.setdefault(entry.src, []).append(entry_index)
                if entry.tier == "blocked":
                    self.has_blocked_phrase_entries = True
            elif entry.level == "char" and entry.tier == "blocked":
                self.has_blocked_char_entries = True

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
    def _regex_required_literal(pattern: str) -> str | None:
        """Return a literal that every match must contain, when provable.

        Only literal runs at the regex top level are considered. Character
        classes and groups are treated as opaque atoms, and top-level
        alternation disables the optimization entirely. This intentionally
        leaves many optimizable patterns unguarded rather than risking a false
        negative that would change rule semantics.
        """
        if not pattern:
            return None

        global_flags = re.match(r"\(\?([aiLmsux]+)\)", pattern)
        if global_flags and {"i", "x"}.intersection(global_flags.group(1)):
            # A plain case-sensitive substring check cannot model IGNORECASE;
            # VERBOSE changes whether apparent spaces/comments are literals.
            return None

        literal_runs: list[str] = []
        current_run: list[str] = []
        cursor = 0
        pattern_length = len(pattern)

        def flush_run() -> None:
            if current_run:
                literal_runs.append("".join(current_run))
                current_run.clear()

        def skip_character_class(start: int) -> int:
            index = start + 1
            if index < pattern_length and pattern[index] == "^":
                index += 1
            if index < pattern_length and pattern[index] == "]":
                index += 1
            while index < pattern_length:
                if pattern[index] == "\\":
                    index += 2
                elif pattern[index] == "]":
                    return index + 1
                else:
                    index += 1
            return index

        def skip_group(start: int) -> int:
            depth = 1
            index = start + 1
            while index < pattern_length and depth:
                char = pattern[index]
                if char == "\\":
                    index += 2
                elif char == "[":
                    index = skip_character_class(index)
                elif char == "(":
                    depth += 1
                    index += 1
                elif char == ")":
                    depth -= 1
                    index += 1
                else:
                    index += 1
            return index

        while cursor < pattern_length:
            char = pattern[cursor]
            if char == "|":
                return None
            if char == "(":
                flush_run()
                cursor = skip_group(cursor)
                continue
            if char == "[":
                flush_run()
                cursor = skip_character_class(cursor)
                continue
            if char == "\\":
                if cursor + 1 >= pattern_length:
                    flush_run()
                    break
                escape_type = pattern[cursor + 1]
                escape_width = {"u": 4, "U": 8, "x": 2}.get(escape_type)
                if escape_width is not None:
                    escape_end = cursor + 2 + escape_width
                    codepoint = pattern[cursor + 2 : escape_end]
                    if len(codepoint) == escape_width and all(
                        digit in "0123456789abcdefABCDEF" for digit in codepoint
                    ):
                        current_run.append(chr(int(codepoint, 16)))
                        cursor = escape_end
                        continue
                if not escape_type.isalnum():
                    current_run.append(escape_type)
                    cursor += 2
                    continue
                flush_run()
                cursor += 2
                continue
            if char in "?*+{":
                can_match_zero = char in "?*"
                quantifier_end = cursor + 1
                if char == "{":
                    closing = pattern.find("}", cursor + 1)
                    if closing >= 0:
                        minimum = pattern[cursor + 1 : closing].split(",", 1)[0]
                        can_match_zero = minimum == "0"
                        quantifier_end = closing + 1
                if current_run:
                    if can_match_zero:
                        current_run.pop()
                    flush_run()
                cursor = quantifier_end
                continue
            if char in ".^$}":
                flush_run()
                cursor += 1
                continue
            current_run.append(char)
            cursor += 1

        flush_run()
        return max(literal_runs, key=len, default=None)

    @staticmethod
    def _decode_runtime_context(context: Any) -> dict[str, Any] | None:
        return decode_runtime_context(context)

    @staticmethod
    def _runtime_rule_id(pass_name: str, pattern: str, replacement: str) -> str:
        raw = f"{pass_name}|{pattern}|{replacement}".encode()
        return f"{RULE_ID_PREFIX}{hashlib.sha1(raw).hexdigest()[:RUNTIME_ID_SUFFIX_LEN]}"

    def _load_entry_table(
        self,
        entry_table: dict[str, Any],
    ) -> tuple[dict[str, RuntimeLexiconEntry], list[RuntimeLexiconEntry], dict[str, int]]:
        raw_blob = entry_table["ih"]
        raw_rows = entry_table["e"]
        kind_defaults = entry_table["k"]
        if len(raw_blob) % RUNTIME_ID_SUFFIX_LEN != 0:
            raise ValueError("Invalid compact entry id blob")
        entry_ids = [
            f"{ENTRY_ID_PREFIX}{raw_blob[index : index + RUNTIME_ID_SUFFIX_LEN]}"
            for index in range(0, len(raw_blob), RUNTIME_ID_SUFFIX_LEN)
        ]

        id_exceptions = entry_table.get("ix", {})
        for raw_index, entry_id in id_exceptions.items():
            if not isinstance(raw_index, str) or not raw_index.isdecimal():
                raise ValueError(f"Invalid compact entry id exception index: {raw_index!r}")
            index = int(raw_index)
            if raw_index != str(index) or not 0 <= index < len(entry_ids):
                raise ValueError(f"Invalid compact entry id exception index: {raw_index!r}")
            if not isinstance(entry_id, str) or not entry_id.startswith(ENTRY_ID_PREFIX) or not entry_id:
                raise ValueError(f"Invalid compact entry id exception: {entry_id!r}")
            entry_ids[index] = entry_id

        if len(entry_ids) != len(raw_rows):
            raise ValueError("Compact entry_table ids/rows length mismatch")
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("Compact entry_table contains duplicate entry ids")

        parsed_kinds: list[tuple[str, str, str, int, float]] = []
        for kind_row in kind_defaults:
            if not isinstance(kind_row, list) or len(kind_row) != 5:
                raise ValueError("Invalid compact entry kind row")
            level_index, tier_index, trust_index, raw_priority, raw_score = kind_row
            if type(level_index) is not int or not 0 <= level_index < len(RUNTIME_LEVELS):
                raise ValueError(f"Invalid compact entry level index: {level_index!r}")
            if type(tier_index) is not int or not 0 <= tier_index < len(RUNTIME_TIERS):
                raise ValueError(f"Invalid compact entry tier index: {tier_index!r}")
            if type(trust_index) is not int or not 0 <= trust_index < len(RUNTIME_TRUSTS):
                raise ValueError(f"Invalid compact entry trust index: {trust_index!r}")
            if type(raw_priority) is not int:
                raise ValueError(f"Invalid compact entry priority: {raw_priority!r}")
            if isinstance(raw_score, bool) or not isinstance(raw_score, int | float) or not math.isfinite(raw_score):
                raise ValueError(f"Invalid compact entry score: {raw_score!r}")
            parsed_kinds.append(
                (
                    RUNTIME_LEVELS[level_index],
                    RUNTIME_TIERS[tier_index],
                    RUNTIME_TRUSTS[trust_index],
                    raw_priority,
                    float(raw_score),
                )
            )

        entries_by_index: list[RuntimeLexiconEntry] = []
        entries: dict[str, RuntimeLexiconEntry] = {}
        for entry_id, raw_row in zip(entry_ids, raw_rows, strict=True):
            if not isinstance(raw_row, list) or not 3 <= len(raw_row) <= 6:
                raise ValueError("Invalid compact entry row")
            src, tgt, kind_index = raw_row[:3]
            if not isinstance(src, str) or not src or not isinstance(tgt, str) or not tgt:
                raise ValueError("Invalid compact entry src/tgt")
            if type(kind_index) is not int or not 0 <= kind_index < len(parsed_kinds):
                raise ValueError(f"Invalid compact entry kind: {kind_index!r}")
            level, tier, trust, priority, score = parsed_kinds[kind_index]
            context = None
            if len(raw_row) >= 4:
                if type(raw_row[3]) is not int:
                    raise ValueError(f"Invalid compact entry priority: {raw_row[3]!r}")
                priority = raw_row[3]
            if len(raw_row) >= 5:
                raw_score = raw_row[4]
                if (
                    isinstance(raw_score, bool)
                    or not isinstance(raw_score, int | float)
                    or not math.isfinite(raw_score)
                ):
                    raise ValueError(f"Invalid compact entry score: {raw_score!r}")
                score = float(raw_score)
            if len(raw_row) == 6:
                if raw_row[5] is None:
                    raise ValueError("Invalid compact entry row: explicit null context")
                context = self._decode_runtime_context(raw_row[5])

            entry = RuntimeLexiconEntry(
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

        entry_index_by_id = {entry.entry_id: index for index, entry in enumerate(entries_by_index)}
        return entries, entries_by_index, entry_index_by_id

    def _normalize_entry_refs(self, raw_refs: Any) -> list[int]:
        raw_items = raw_refs if isinstance(raw_refs, list) else [raw_refs]
        entry_indexes: list[int] = []
        for raw_ref in raw_items:
            if isinstance(raw_ref, bool):
                raise ValueError(f"Invalid runtime entry reference: {raw_ref!r}")
            if isinstance(raw_ref, int):
                if not 0 <= raw_ref < len(self.entries_by_index):
                    raise ValueError(f"Runtime entry reference out of range: {raw_ref}")
                entry_indexes.append(raw_ref)
                continue
            if isinstance(raw_ref, str):
                entry_index = self.entry_index_by_id.get(raw_ref)
                if entry_index is None:
                    raise ValueError(f"Unknown runtime entry reference: {raw_ref!r}")
                entry_indexes.append(entry_index)
                continue
            raise ValueError(f"Invalid runtime entry reference: {raw_ref!r}")
        return entry_indexes

    def _load_phrase_trie(self, phrase_trie_doc: dict[str, Any]) -> dict[str, Any]:
        def decode_compact(node_doc: Any, *, is_root: bool = False) -> dict[str, Any]:
            if not isinstance(node_doc, dict):
                raise ValueError("Invalid compact phrase trie node")
            entry_indexes: list[int] = []
            if "" in node_doc:
                entry_indexes = self._normalize_entry_refs(node_doc[""])
                if not entry_indexes:
                    raise ValueError("Invalid compact phrase trie terminal")

            children: dict[str, tuple[str, dict[str, Any]]] = {}
            for label, child_doc in node_doc.items():
                if label == "":
                    continue
                if not isinstance(label, str) or not label:
                    raise ValueError(f"Invalid compact phrase trie edge: {label!r}")
                first = label[0]
                if first in children:
                    raise ValueError(f"Compact phrase trie edge collision: {first!r}")
                child = decode_compact(child_doc)
                children[first] = (label[1:], child)

            if not is_root and not entry_indexes and not children:
                raise ValueError("Invalid compact phrase trie dangling node")
            return {"e": entry_indexes, "c": children}

        return decode_compact(phrase_trie_doc["t"], is_root=True)

    def _load_char_map(self, char_map_doc: dict[str, Any]) -> dict[str, list[int]]:
        char_map: dict[str, list[int]] = {}
        for ch, raw_refs in char_map_doc["m"].items():
            if not isinstance(ch, str) or not ch:
                raise ValueError(f"Invalid char map key: {ch!r}")
            entry_indexes = self._normalize_entry_refs(raw_refs)
            if not entry_indexes:
                raise ValueError(f"Invalid empty char map reference: {ch!r}")
            char_map[ch] = entry_indexes
        return char_map

    def _load_rule_plan(self, rule_plan: dict[str, Any]) -> tuple[list[str], dict[str, list[RuntimeRuleEntry]]]:
        rules_by_pass: dict[str, list[RuntimeRuleEntry]] = {pass_name: [] for pass_name in PASS_ORDER}
        seen_rule_ids: set[str] = set()
        for pass_name, rows in zip(PASS_ORDER, rule_plan["r"], strict=True):
            parsed_rules: list[RuntimeRuleEntry] = []
            for item in rows:
                if not isinstance(item, list) or not 2 <= len(item) <= 4:
                    raise ValueError("Invalid compact runtime rule row")
                pattern, replacement = item[:2]
                if not isinstance(pattern, str) or not pattern or not isinstance(replacement, str):
                    raise ValueError("Invalid compact runtime rule pattern/replacement")

                rule_type = "literal"
                priority = 0
                if len(item) >= 3:
                    raw_kind = item[2]
                    if type(raw_kind) is not int or raw_kind not in {0, 1}:
                        raise ValueError(f"Invalid compact runtime rule kind: {raw_kind!r}")
                    if len(item) == 3 and raw_kind != 1:
                        raise ValueError("Non-canonical compact runtime literal rule")
                    rule_type = "regex" if raw_kind == 1 else "literal"
                if len(item) == 4:
                    raw_priority = item[3]
                    if type(raw_priority) is not int or raw_priority == 0:
                        raise ValueError(f"Invalid compact runtime rule priority: {raw_priority!r}")
                    priority = raw_priority

                rule_id = self._runtime_rule_id(pass_name, pattern, replacement)
                if rule_id in seen_rule_ids:
                    raise ValueError(f"Duplicate compact runtime rule: {rule_id}")
                seen_rule_ids.add(rule_id)
                parsed_rules.append(
                    RuntimeRuleEntry(
                        rule_id=rule_id,
                        pass_name=pass_name,
                        type=rule_type,
                        pattern=pattern,
                        replacement=replacement,
                        priority=priority,
                    )
                )
            rules_by_pass[pass_name] = parsed_rules
        return list(PASS_ORDER), rules_by_pass

    def _load_override_index(self, override_index: dict[str, Any]) -> tuple[dict[str, list[int]], list[int]]:
        sentence_override_map: dict[str, list[int]] = {}
        for src, raw_entry_refs in override_index["s"].items():
            if not isinstance(src, str) or not src:
                raise ValueError(f"Invalid sentence override key: {src!r}")
            entry_indexes = self._normalize_entry_refs(raw_entry_refs)
            if not entry_indexes:
                raise ValueError(f"Invalid empty sentence override reference: {src!r}")
            sentence_override_map[src] = entry_indexes

        contextual_override_entry_indexes = self._normalize_entry_refs(override_index["c"])
        return sentence_override_map, contextual_override_entry_indexes

    def _validate_runtime_reference_contract(self) -> None:
        errors: list[str] = []

        def phrase_references(node: dict[str, Any], prefix: str = "") -> list[tuple[str, int]]:
            references = [(prefix, entry_index) for entry_index in node.get("e", ())]
            for first, (suffix, child) in node.get("c", {}).items():
                references.extend(phrase_references(child, prefix + first + suffix))
            return references

        phrase_refs = phrase_references(self.phrase_trie)
        actual_phrase_indexes = [entry_index for _, entry_index in phrase_refs]
        if len(set(actual_phrase_indexes)) != len(actual_phrase_indexes):
            errors.append("phrase trie 含重複引用")
        for src, entry_index in phrase_refs:
            entry = self.entries_by_index[entry_index]
            if entry.level not in {"phrase", "sentence"}:
                errors.append(f"phrase trie 引用非 phrase/sentence 詞條: {entry.entry_id}")
            if entry.src != src:
                errors.append(f"phrase trie 路徑與詞條來源不一致: {entry.entry_id}")
            if entry.context is not None:
                errors.append(f"phrase trie 不可引用 context 詞條: {entry.entry_id}")
            if is_sentence_manual_override(entry):
                errors.append(f"phrase trie 不可引用 sentence manual override: {entry.entry_id}")

        expected_phrase_indexes = {
            index
            for index, entry in enumerate(self.entries_by_index)
            if entry.status == "active"
            and entry.level in {"phrase", "sentence"}
            and entry.context is None
            and not is_sentence_manual_override(entry)
        }
        if set(actual_phrase_indexes) != expected_phrase_indexes:
            errors.append("phrase trie 與 runtime entries 不一致")

        actual_char_indexes: list[int] = []
        for src, entry_indexes in self.char_map.items():
            for entry_index in entry_indexes:
                actual_char_indexes.append(entry_index)
                entry = self.entries_by_index[entry_index]
                if entry.level != "char" or entry.src != src:
                    errors.append(f"char map 引用不相容詞條: {entry.entry_id}")
                if entry.context is not None:
                    errors.append(f"char map 不可引用 context 詞條: {entry.entry_id}")
        if len(set(actual_char_indexes)) != len(actual_char_indexes):
            errors.append("char map 含重複引用")
        expected_char_indexes = {
            index
            for index, entry in enumerate(self.entries_by_index)
            if entry.status == "active" and entry.level == "char" and entry.context is None
        }
        if set(actual_char_indexes) != expected_char_indexes:
            errors.append("char map 與 runtime entries 不一致")

        actual_sentence_refs = [
            entry_index for entry_indexes in self.sentence_override_map.values() for entry_index in entry_indexes
        ]
        if len(set(actual_sentence_refs)) != len(actual_sentence_refs):
            errors.append("sentence override index 含重複引用")
        actual_sentence_indexes = set(actual_sentence_refs)
        for src, entry_indexes in self.sentence_override_map.items():
            for entry_index in entry_indexes:
                entry = self.entries_by_index[entry_index]
                if entry.src != src or not is_sentence_manual_override(entry):
                    errors.append(f"sentence override 引用不相容詞條: {entry.entry_id}")

        expected_sentence_indexes = {
            index for index, entry in enumerate(self.entries_by_index) if is_sentence_manual_override(entry)
        }
        if actual_sentence_indexes != expected_sentence_indexes:
            errors.append("sentence override index 與 runtime entries 不一致")

        actual_context_indexes = set(self.contextual_override_entry_indexes)
        if len(actual_context_indexes) != len(self.contextual_override_entry_indexes):
            errors.append("contextual override index 含重複引用")
        for entry_index in actual_context_indexes:
            entry = self.entries_by_index[entry_index]
            if entry.status != "active" or not is_trusted_context_entry(entry):
                errors.append(f"contextual override 引用不相容詞條: {entry.entry_id}")

        expected_context_indexes = {
            index
            for index, entry in enumerate(self.entries_by_index)
            if entry.status == "active" and is_trusted_context_entry(entry)
        }
        if actual_context_indexes != expected_context_indexes:
            errors.append("contextual override index 與 runtime entries 不一致")

        if errors:
            raise ValueError("runtime artifact reference contract 驗證失敗:\n" + "\n".join(errors))

    @staticmethod
    def _layer_rank(entry: RuntimeLexiconEntry) -> int:
        return runtime_layer_rank(entry)

    def _phase_candidate_key(self, candidate: Candidate) -> tuple[int, int, int, int, float, str]:
        entry = candidate.entry
        if entry.context is not None:
            # A matched context is more specific than a global mapping. Among
            # contextual candidates, explicit governance priority chooses the
            # narrower override before provenance rank; all runtime contexts
            # have already passed the curated-trust policy gate.
            precedence = (-entry.priority, candidate.layer_rank)
            context_rank = 0
        else:
            precedence = (candidate.layer_rank, -entry.priority)
            context_rank = 1
        return (
            -(candidate.end - candidate.start),
            context_rank,
            *precedence,
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
                    if entry.context is not None:
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

    def _iter_single_char_phrase_candidates(self, text: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        entries = self.entries_by_index
        layer_rank_by_index = self.layer_rank_by_index
        phrase_map = self.single_char_phrase_map
        for start, char in enumerate(text):
            for entry_index in phrase_map.get(char, ()):
                candidates.append(
                    Candidate(
                        entry=entries[entry_index],
                        start=start,
                        end=start + 1,
                        layer_rank=layer_rank_by_index[entry_index],
                    )
                )
        return candidates

    @staticmethod
    def _context_match(text: str, start: int, end: int, context: dict[str, Any] | None) -> bool:
        if context is None:
            return False

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

    def _iter_contextual_candidates(
        self,
        text: str,
        *,
        context_text: str | None = None,
    ) -> list[Candidate]:
        """Return contextual matches without losing protected-neighbour context.

        ``text`` may be the same-length shadow string used to keep protected
        spans out of lexicon matching. Context predicates, however, must see
        the original rendered text; otherwise a protected number, identifier,
        or neighbouring protected term becomes an opaque marker and valid
        left/right context silently stops matching.

        Candidate discovery intentionally still runs against ``text``. This
        prevents a contextual entry whose source itself overlaps a protected
        span from rewriting protected content, while ``context_text`` is used
        only to evaluate the surrounding predicate. Shadow encoding preserves
        code-point length, so candidate offsets are valid in both strings.
        """
        context_source = text if context_text is None else context_text
        if len(context_source) != len(text):
            raise ValueError("context_text 必須與 lexicon matching text 等長")

        candidates: list[Candidate] = []
        entries = self.entries_by_index
        layer_rank_by_index = self.layer_rank_by_index
        entry_indexes_by_first_char = self.contextual_entry_indexes_by_first_char
        for start, char in enumerate(text):
            for entry_index in entry_indexes_by_first_char.get(char, ()):
                entry = entries[entry_index]
                end = start + len(entry.src)
                if not text.startswith(entry.src, start):
                    continue
                if self._context_match(context_source, start, end, entry.context):
                    candidates.append(
                        Candidate(
                            entry=entry,
                            start=start,
                            end=end,
                            layer_rank=layer_rank_by_index[entry_index],
                        )
                    )
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
                if entry.context is not None:
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

    @staticmethod
    def _merge_segments(segments: list[_TextSegment]) -> _ProtectedText:
        merged: list[_TextSegment] = []
        pending_parts: list[str] = []
        pending_protected = False

        def flush() -> None:
            if pending_parts:
                merged.append(
                    _TextSegment(
                        "".join(pending_parts),
                        protected=pending_protected,
                    )
                )
                pending_parts.clear()

        for segment in segments:
            if not segment.text:
                continue
            if pending_parts and pending_protected != segment.protected:
                flush()
            if not pending_parts:
                pending_protected = segment.protected
            pending_parts.append(segment.text)
        flush()
        return _ProtectedText(tuple(merged))

    def _protect_text(
        self,
        text: str,
        *,
        respect_runtime_phrase_overlap: bool = True,
    ) -> _ProtectedText:
        """Split text into protected and transformable spans without sentinels.

        The span representation has no token namespace, occurrence ceiling, or
        unmask pass. Protected text is never exposed to lexicon/rule transforms.
        """
        has_regex_masks = bool(self.protected_regex_masks)
        has_trie_masks = bool(self.protected_term_trie.get("children"))
        if not text or (not has_regex_masks and not has_trie_masks):
            return _ProtectedText.plain(text)

        text_len = len(text)
        occupied = bytearray(text_len)
        protected_spans: list[tuple[int, int]] = []

        for compiled in self.protected_regex_masks:
            for match in compiled.finditer(text):
                start, end = match.span()
                if end <= start or occupied.find(b"\x01", start, end) >= 0:
                    continue
                occupied[start:end] = b"\x01" * (end - start)
                protected_spans.append((start, end))

        for root, enforce_strict in (
            (self.strict_protected_term_trie, True),
            (self.protected_term_trie, False),
        ):
            if not root["children"]:
                continue
            cursor = 0
            while cursor < text_len:
                if occupied[cursor]:
                    cursor += 1
                    continue

                node = root
                idx = cursor
                longest_end = -1
                while idx < text_len and not occupied[idx]:
                    child = node["children"].get(text[idx])
                    if child is None:
                        break
                    node = child
                    idx += 1
                    if "term" in node:
                        longest_end = idx

                if longest_end < 0:
                    cursor += 1
                    continue

                span_covers_full_text = cursor == 0 and longest_end == text_len
                if (
                    not enforce_strict
                    and respect_runtime_phrase_overlap
                    and not span_covers_full_text
                    and self._overlaps_runtime_phrase(text, cursor, longest_end)
                ):
                    cursor += 1
                    continue

                occupied[cursor:longest_end] = b"\x01" * (longest_end - cursor)
                protected_spans.append((cursor, longest_end))
                cursor = longest_end

        if not protected_spans:
            return _ProtectedText.plain(text)

        segments: list[_TextSegment] = []
        cursor = 0
        for start, end in sorted(protected_spans):
            if cursor < start:
                segments.append(_TextSegment(text[cursor:start]))
            segments.append(_TextSegment(text[start:end], protected=True))
            cursor = end
        if cursor < text_len:
            segments.append(_TextSegment(text[cursor:]))
        return self._merge_segments(segments)

    def _protect_number_bearing_lexicon_terms(self, text: str) -> _ProtectedText:
        """Protect lexicon terms containing digits during number normalization.

        Number conversion runs before phrase matching. Without this dedicated
        pre-lexicon span pass, a governed term such as ``台北101`` becomes
        ``台北一百空一`` and can no longer reach its lexicon entry. This trie is
        intentionally separate from identity passthrough protection: it delays
        number conversion only, then lets the normal phrase pipeline apply the
        entry's canonical target.
        """

        root = self.number_bearing_lexicon_trie
        if not text or not root.get("children"):
            return _ProtectedText.plain(text)

        text_len = len(text)
        spans: list[tuple[int, int]] = []
        cursor = 0
        while cursor < text_len:
            node = root
            idx = cursor
            longest_end = -1
            while idx < text_len:
                child = node["children"].get(text[idx])
                if child is None:
                    break
                node = child
                idx += 1
                if "term" in node:
                    longest_end = idx

            if longest_end < 0:
                cursor += 1
                continue

            spans.append((cursor, longest_end))
            cursor = longest_end

        if not spans:
            return _ProtectedText.plain(text)

        segments: list[_TextSegment] = []
        cursor = 0
        for start, end in spans:
            if cursor < start:
                segments.append(_TextSegment(text[cursor:start]))
            segments.append(_TextSegment(text[start:end], protected=True))
            cursor = end
        if cursor < text_len:
            segments.append(_TextSegment(text[cursor:]))
        return self._merge_segments(segments)

    def _protect_unprotected_segments(self, text: _ProtectedText) -> _ProtectedText:
        segments: list[_TextSegment] = []
        for segment in text.segments:
            if segment.protected:
                segments.append(segment)
                continue
            protected = self._protect_text(
                segment.text,
                respect_runtime_phrase_overlap=False,
            )
            segments.extend(protected.segments)
        return self._merge_segments(segments)

    @staticmethod
    def _shift_matches(matches: list[MatchTrace], offset: int) -> list[MatchTrace]:
        if offset == 0:
            return matches
        return [
            MatchTrace(
                entry_id=match.entry_id,
                src=match.src,
                tgt=match.tgt,
                level=match.level,
                tier=match.tier,
                start=match.start + offset,
                end=match.end + offset,
                priority=match.priority,
                score=match.score,
            )
            for match in matches
        ]

    @staticmethod
    def _shadow_marker_candidates() -> tuple[str, ...]:
        # Unicode noncharacters are valid Python scalar values but cannot be
        # assigned as text characters. The immutable candidate pool is shared
        # across conversions instead of being rebuilt for every protected span.
        return SHADOW_MARKER_CANDIDATES

    def _encode_protected(self, text: _ProtectedText) -> tuple[str, _ShadowLayout] | None:
        protected_values = tuple(segment.text for segment in text.segments if segment.protected)
        if not protected_values:
            return None

        used_chars = {char for segment in text.segments for char in segment.text}
        candidates = (
            candidate
            for candidate in self._shadow_marker_candidates()
            if candidate not in used_chars and candidate not in self.shadow_forbidden_chars
        )
        marker_by_value: dict[str, str] = {}
        protected_by_marker: list[tuple[str, str]] = []
        shadow_parts: list[str] = []
        for segment in text.segments:
            if not segment.protected:
                shadow_parts.append(segment.text)
                continue
            marker = marker_by_value.get(segment.text)
            if marker is None:
                marker = next(candidates, None)
                if marker is None:
                    return None
                marker_by_value[segment.text] = marker
                protected_by_marker.append((marker, segment.text))
            shadow_parts.append(marker * len(segment.text))

        return "".join(shadow_parts), _ShadowLayout(tuple(protected_by_marker))

    @staticmethod
    def _decode_protected(shadow: str, layout: _ShadowLayout) -> _ProtectedText | None:
        protected_by_marker = dict(layout.protected_by_marker)
        if not protected_by_marker:
            return _ProtectedText.plain(shadow)

        segments: list[_TextSegment] = []
        cursor = 0
        plain_start = 0
        while cursor < len(shadow):
            marker = shadow[cursor]
            value = protected_by_marker.get(marker)
            if value is None:
                cursor += 1
                continue
            if plain_start < cursor:
                segments.append(_TextSegment(shadow[plain_start:cursor]))

            run_end = cursor + 1
            while run_end < len(shadow) and shadow[run_end] == marker:
                run_end += 1
            run_length = run_end - cursor
            value_length = len(value)
            if value_length == 0 or run_length % value_length:
                return None
            segments.extend(_TextSegment(value, protected=True) for _ in range(run_length // value_length))
            cursor = run_end
            plain_start = cursor

        if plain_start < len(shadow):
            segments.append(_TextSegment(shadow[plain_start:]))
        return TaigiConverter._merge_segments(segments)

    def _apply_lexicon_by_segment(
        self,
        text: _ProtectedText,
        *,
        min_src_len: int,
        max_src_len: int | None,
        include_char_entries: bool,
        allow_sentence_override: bool,
        collect_matches: bool,
        collect_warnings: bool,
        review_diagnostics: _ReviewDiagnostics | None,
    ) -> tuple[_ProtectedText, list[MatchTrace], list[str]]:
        segments: list[_TextSegment] = []
        matches: list[MatchTrace] = []
        warnings: list[str] = []
        input_offset = 0
        for segment in text.segments:
            if segment.protected:
                segments.append(segment)
            else:
                output, segment_matches, segment_warnings = self._apply_lexicon_layers(
                    segment.text,
                    min_src_len=min_src_len,
                    max_src_len=max_src_len,
                    include_char_entries=include_char_entries,
                    allow_sentence_override=allow_sentence_override,
                    collect_matches=collect_matches,
                    collect_warnings=collect_warnings,
                    review_diagnostics=review_diagnostics,
                )
                segments.append(_TextSegment(output))
                if collect_matches:
                    matches.extend(self._shift_matches(segment_matches, input_offset))
                warnings.extend(segment_warnings)
            input_offset += len(segment.text)
        return self._merge_segments(segments), matches, warnings

    def _apply_lexicon_to_protected(
        self,
        text: _ProtectedText,
        *,
        min_src_len: int = 1,
        max_src_len: int | None = None,
        include_char_entries: bool = True,
        allow_sentence_override: bool = True,
        collect_matches: bool,
        collect_warnings: bool,
        review_diagnostics: _ReviewDiagnostics | None = None,
    ) -> tuple[_ProtectedText, list[MatchTrace], list[str]]:
        encoded = self._encode_protected(text)
        if encoded is not None:
            shadow, layout = encoded
            output, matches, warnings = self._apply_lexicon_layers(
                shadow,
                context_text=text.render(),
                min_src_len=min_src_len,
                max_src_len=max_src_len,
                include_char_entries=include_char_entries,
                allow_sentence_override=allow_sentence_override,
                collect_matches=collect_matches,
                collect_warnings=collect_warnings,
                review_diagnostics=review_diagnostics,
            )
            decoded = self._decode_protected(output, layout)
            if decoded is not None:
                return decoded, matches, warnings
        elif not any(segment.protected for segment in text.segments):
            output, matches, warnings = self._apply_lexicon_layers(
                text.render(),
                min_src_len=min_src_len,
                max_src_len=max_src_len,
                include_char_entries=include_char_entries,
                allow_sentence_override=allow_sentence_override,
                collect_matches=collect_matches,
                collect_warnings=collect_warnings,
                review_diagnostics=review_diagnostics,
            )
            return _ProtectedText.plain(output), matches, warnings

        return self._apply_lexicon_by_segment(
            text,
            min_src_len=min_src_len,
            max_src_len=max_src_len,
            include_char_entries=include_char_entries,
            allow_sentence_override=allow_sentence_override,
            collect_matches=collect_matches,
            collect_warnings=collect_warnings,
            review_diagnostics=review_diagnostics,
        )

    def _apply_rules_by_segment(
        self,
        text: _ProtectedText,
        *,
        collect_trace: bool,
        skip_passes: set[str] | None,
    ) -> tuple[_ProtectedText, list[RuleTrace]]:
        segments: list[_TextSegment] = []
        trace_order: list[tuple[str, str, str, str, str]] = []
        trace_by_key: dict[tuple[str, str, str, str, str], RuleTrace] = {}
        for segment in text.segments:
            if segment.protected:
                segments.append(segment)
                continue
            output, segment_traces = self._apply_rules(
                segment.text,
                collect_trace=collect_trace,
                skip_passes=skip_passes,
            )
            segments.append(_TextSegment(output))
            for trace in segment_traces:
                key = (
                    trace.rule_id,
                    trace.pass_name,
                    trace.type,
                    trace.pattern,
                    trace.replacement,
                )
                existing = trace_by_key.get(key)
                if existing is None:
                    trace_order.append(key)
                    trace_by_key[key] = trace
                else:
                    existing.hit_count += trace.hit_count
                    existing.matched_chars += trace.matched_chars
        return self._merge_segments(segments), [trace_by_key[key] for key in trace_order]

    def _apply_rules_to_protected(
        self,
        text: _ProtectedText,
        *,
        collect_trace: bool,
        skip_passes: set[str] | None = None,
    ) -> tuple[_ProtectedText, list[RuleTrace]]:
        encoded = self._encode_protected(text)
        if encoded is not None:
            shadow, layout = encoded
            output, traces = self._apply_rules(
                shadow,
                collect_trace=collect_trace,
                skip_passes=skip_passes,
            )
            decoded = self._decode_protected(output, layout)
            if decoded is not None:
                return decoded, traces
        elif not any(segment.protected for segment in text.segments):
            output, traces = self._apply_rules(
                text.render(),
                collect_trace=collect_trace,
                skip_passes=skip_passes,
            )
            return _ProtectedText.plain(output), traces

        return self._apply_rules_by_segment(
            text,
            collect_trace=collect_trace,
            skip_passes=skip_passes,
        )

    def _cleanup_by_segment(
        self, text: _ProtectedText, *, collect_warnings: bool
    ) -> tuple[_ProtectedText, list[str]]:
        segments: list[_TextSegment] = []
        warnings: list[str] = []
        for segment in text.segments:
            if segment.protected:
                segments.append(segment)
                continue
            output, segment_warnings = self._post_cleanup(
                segment.text, collect_warnings=collect_warnings
            )
            segments.append(_TextSegment(output))
            warnings.extend(segment_warnings)
        return self._merge_segments(segments), warnings

    def _cleanup_protected(
        self, text: _ProtectedText, *, collect_warnings: bool
    ) -> tuple[_ProtectedText, list[str]]:
        encoded = self._encode_protected(text)
        if encoded is not None:
            shadow, layout = encoded
            output, warnings = self._post_cleanup(shadow, collect_warnings=collect_warnings)
            decoded = self._decode_protected(output, layout)
            if decoded is not None:
                return decoded, warnings
        elif not any(segment.protected for segment in text.segments):
            output, warnings = self._post_cleanup(
                text.render(), collect_warnings=collect_warnings
            )
            return _ProtectedText.plain(output), warnings

        return self._cleanup_by_segment(text, collect_warnings=collect_warnings)

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
        if not (self.has_blocked_phrase_entries or self.has_blocked_char_entries):
            return []

        blocked = [candidate for candidate in phrase_candidates if candidate.entry.tier == "blocked"]

        if self.has_blocked_char_entries:
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

    def _apply_exact_sentence_override(
        self,
        text: str,
        *,
        collect_matches: bool,
        collect_warnings: bool,
        review_diagnostics: _ReviewDiagnostics | None = None,
    ) -> tuple[str | None, list[MatchTrace], list[str]]:
        if not text:
            return None, [], []

        sentence_override_ids = self.sentence_override_map.get(text, [])
        if not sentence_override_ids:
            return None, [], []

        blocked_phrase_candidates = (
            self._iter_phrase_candidates(text) if self.has_blocked_phrase_entries else []
        )
        blocked_candidates = self._collect_blocked_candidates(text, blocked_phrase_candidates)
        warnings = (
            [f"blocked:{blocked.entry.entry_id}:{blocked.entry.src}" for blocked in blocked_candidates]
            if collect_warnings
            else []
        )

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
        if review_diagnostics is not None:
            review_diagnostics.ambiguous_candidate_count += _competing_target_count(
                chosen, sentence_candidates
            )
        if not collect_matches:
            return chosen.entry.tgt, [], warnings

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
        context_text: str | None = None,
        min_src_len: int = 1,
        max_src_len: int | None = None,
        include_char_entries: bool = True,
        allow_sentence_override: bool = True,
        collect_matches: bool,
        collect_warnings: bool,
        review_diagnostics: _ReviewDiagnostics | None = None,
    ) -> tuple[str, list[MatchTrace], list[str]]:
        warnings: list[str] = []
        all_phrase_candidates = (
            self._iter_single_char_phrase_candidates(text)
            if max_src_len == 1 and not self.has_blocked_phrase_entries
            else self._iter_phrase_candidates(text)
        )
        blocked_candidates = self._collect_blocked_candidates(text, all_phrase_candidates)

        if collect_warnings:
            warnings.extend(
                f"blocked:{blocked.entry.entry_id}:{blocked.entry.src}"
                for blocked in blocked_candidates
            )

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
                    if review_diagnostics is not None:
                        review_diagnostics.ambiguous_candidate_count += (
                            _competing_target_count(chosen, sentence_candidates)
                        )
                    if not collect_matches:
                        return chosen.entry.tgt, [], warnings
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
            for candidate in self._iter_contextual_candidates(text, context_text=context_text)
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

        if review_diagnostics is not None:
            for chosen in selected:
                review_diagnostics.ambiguous_candidate_count += _competing_target_count(
                    chosen, all_candidates
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
            if collect_matches:
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
            for runtime_rule in self.compiled_rules_by_pass.get(pass_name, []):
                rule = runtime_rule.rule
                compiled = runtime_rule.compiled
                if runtime_rule.required_literal and runtime_rule.required_literal not in text:
                    continue

                matched_chars = 0
                if rule.type == "regex":
                    if compiled is None:
                        continue
                    if collect_trace:
                        replacement = rule.replacement

                        def replace_with_trace(
                            match: re.Match[str], replacement: str = replacement
                        ) -> str:
                            nonlocal matched_chars
                            matched_chars += match.end() - match.start()
                            return match.expand(replacement)

                        replaced_text, hit_count = compiled.subn(replace_with_trace, text)
                    else:
                        replaced_text = compiled.sub(rule.replacement, text)
                        hit_count = 0
                else:
                    if collect_trace:
                        hit_count = text.count(rule.pattern)
                        matched_chars = hit_count * len(rule.pattern)
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
                            matched_chars=matched_chars,
                        )
                    )
                text = replaced_text
        return text, traces

    def _post_cleanup(
        self, text: str, *, collect_warnings: bool
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []

        # L8: 後驗清理
        text = text.replace("這馬咧咧", "這馬咧")
        text = text.replace("真正真", "真")
        text = REPEATED_PUNCTUATION_RE.sub(r"\1", text)
        # 高雄、高速公路等地名/複合詞被 char 誤轉「高->懸」之後復原
        text = text.replace("懸雄", "高雄")
        text = text.replace("懸速公路", "高速公路")

        if collect_warnings:
            for term in self.residual_terms:
                if term in text:
                    warning_prefix = "核心漏轉" if term in self.residual_core_terms else "華語殘留"
                    warnings.append(f"{warning_prefix}:{term}")

        return text, warnings

    def _post_unmask_time_cleanup(self, text: str) -> str:
        for pattern, replacement in POST_UNMASK_TIME_SUBSTITUTIONS:
            text = pattern.sub(replacement, text)
        text = text.replace("進前要報到", "進前愛報到")
        text = text.replace("進前要到", "進前愛到")
        return text

    def _enqueue_review_if_needed(
        self,
        *,
        original_text: str,
        normalized_text: str,
        output_text: str,
        protected_input: _ProtectedText,
        matches: list[MatchTrace],
        rules_applied: list[RuleTrace],
        warnings: list[str],
        review_diagnostics: _ReviewDiagnostics | None,
        profile: dict[str, Any] | None,
    ) -> None:
        if not profile or not profile.get("enqueue_review"):
            return

        input_length = len(normalized_text)
        matched_chars = min(sum(max(match.end - match.start, 0) for match in matches), input_length)
        protected_segments = [segment.text for segment in protected_input.segments if segment.protected]
        protected_values = list(dict.fromkeys(protected_segments))
        protected_chars = min(sum(len(value) for value in protected_segments), input_length)
        rule_chars = min(sum(rule.matched_chars for rule in rules_applied), input_length)
        matched_span_ratio = matched_chars / input_length if input_length else 0.0
        protected_span_ratio = protected_chars / input_length if input_length else 0.0
        rule_span_ratio = rule_chars / input_length if input_length else 0.0
        evidence_span_ratio = min(
            matched_span_ratio + protected_span_ratio + rule_span_ratio,
            1.0,
        )
        identity_ratio = _linear_identity_ratio(normalized_text, output_text)
        residual_terms = list(
            dict.fromkeys(
                warning.partition(":")[2]
                for warning in warnings
                if warning.startswith(("華語殘留:", "核心漏轉:"))
            )
        )
        blocked_candidates = [warning for warning in warnings if warning.startswith("blocked:")]
        ambiguous_candidate_count = (
            review_diagnostics.ambiguous_candidate_count if review_diagnostics is not None else 0
        )

        low_confidence_reasons: list[str] = []
        if residual_terms:
            low_confidence_reasons.append("residual_terms")
        if blocked_candidates:
            low_confidence_reasons.append("blocked_candidates")
        if ambiguous_candidate_count:
            low_confidence_reasons.append("ambiguous_candidates")
        if not matches and not rules_applied and not protected_values and identity_ratio >= 0.98:
            low_confidence_reasons.append("no_transform_evidence")
        rule_only_evidence = bool(rules_applied) and not matches
        if (
            input_length >= 4
            and evidence_span_ratio < 0.35
            and identity_ratio >= 0.70
            and not rule_only_evidence
        ):
            low_confidence_reasons.append("sparse_conversion_coverage")
        if not low_confidence_reasons:
            return

        confidence_score = 0.15
        confidence_score += 0.55 * evidence_span_ratio
        confidence_score += 0.15 * min(len(rules_applied), 1)
        confidence_score += 0.15 * (1.0 - identity_ratio)
        confidence_score -= 0.20 * min(len(residual_terms), 2)
        confidence_score -= 0.05 * min(ambiguous_candidate_count, 2)
        confidence_score = round(min(max(confidence_score, 0.0), 1.0), 4)
        review_priority = min(
            100,
            max(
                1,
                round(
                    (1.0 - confidence_score) * 80
                    + 10 * bool(residual_terms)
                    + 5 * bool(blocked_candidates)
                    + min(ambiguous_candidate_count * 3, 10)
                ),
            ),
        )

        if self.review_data_dir is None:
            raise RuntimeError("enqueue_review 需要明確設定 review_data_dir；不得寫入唯讀 runtime 目錄")

        append_review_item(
            self.review_data_dir,
            {
                "kind": "online_low_confidence",
                "action": "add_override",
                "owner": profile.get("owner", "runtime"),
                "reason": "auto_enqueued_by_runtime",
                "priority": review_priority,
                "evidence": {
                    "input": original_text,
                    "normalized_input": normalized_text,
                    "output": output_text,
                    "warnings": warnings,
                    "low_confidence_reasons": low_confidence_reasons,
                    "confidence_score": confidence_score,
                    "review_priority": review_priority,
                    "matched_span_ratio": round(matched_span_ratio, 4),
                    "identity_ratio": round(identity_ratio, 4),
                    "protected_span_ratio": round(protected_span_ratio, 4),
                    "rule_span_ratio": round(rule_span_ratio, 4),
                    "evidence_span_ratio": round(evidence_span_ratio, 4),
                    "residual_terms": residual_terms,
                    "protected_terms": protected_values,
                    "blocked_candidates": blocked_candidates,
                    "ambiguous_candidate_count": ambiguous_candidate_count,
                    "match_count": len(matches),
                    "match_entry_ids": [match.entry_id for match in matches],
                    "matches": [match.to_dict() for match in matches],
                    "rule_count": len(rules_applied),
                    "rule_ids": [rule.rule_id for rule in rules_applied],
                    "rules_applied": [rule.to_dict() for rule in rules_applied],
                },
            },
        )

    def _normalize_input(self, text: str, *, preserve_spacing: bool) -> str:
        """Normalize numbers only outside lexicon-governed protected terms.

        The first pass canonicalizes glyphs and spacing without changing numeric
        content, so protected names such as ``臺北101`` can match their canonical
        ``台北101`` entry. The second pass converts prose numbers only in
        transformable spans. This keeps number-bearing proper nouns under the same
        auditable data policy as every other protected term.
        """

        canonical = normalize_text(
            text,
            compress_spaces=not preserve_spacing,
            trim_outer=not preserve_spacing,
            convert_numbers=False,
        )
        protected = self._protect_number_bearing_lexicon_terms(canonical)
        if not any(segment.protected for segment in protected.segments):
            return normalize_text(
                canonical,
                compress_spaces=not preserve_spacing,
                trim_outer=False,
            )

        return "".join(
            segment.text
            if segment.protected
            else normalize_text(
                segment.text,
                compress_spaces=not preserve_spacing,
                trim_outer=False,
            )
            for segment in protected.segments
        )

    def convert(
        self,
        text: str,
        trace: bool = False,
        profile: dict[str, Any] | None = None,
    ) -> str | ConversionResult:
        started = time.perf_counter() if trace else 0.0
        preserve_spacing = bool(profile and profile.get("preserve_spacing"))
        enqueue_review = bool(profile and profile.get("enqueue_review"))
        collect_diagnostics = trace or enqueue_review
        collect_matches = collect_diagnostics
        collect_warnings = collect_diagnostics
        collect_rule_trace = trace or enqueue_review
        review_diagnostics = _ReviewDiagnostics() if enqueue_review else None
        normalized = self._normalize_input(text, preserve_spacing=preserve_spacing)
        exact_sentence_output, exact_matches, exact_warnings = self._apply_exact_sentence_override(
            normalized,
            collect_matches=collect_matches,
            collect_warnings=collect_warnings,
            review_diagnostics=review_diagnostics,
        )
        protected_input = self._protect_text(normalized)
        skip_passes = {"normalization"} if preserve_spacing else set()

        if exact_sentence_output is not None:
            exact_protected_output = self._protect_text(
                exact_sentence_output,
                respect_runtime_phrase_overlap=False,
            )
            rule_output, rules_applied = self._apply_rules_to_protected(
                exact_protected_output,
                collect_trace=collect_rule_trace,
                skip_passes=skip_passes,
            )
            lexicon_output, post_matches, post_warnings = self._apply_lexicon_to_protected(
                rule_output,
                min_src_len=1,
                max_src_len=1,
                include_char_entries=True,
                allow_sentence_override=False,
                collect_matches=collect_matches,
                collect_warnings=collect_warnings,
                review_diagnostics=review_diagnostics,
            )
            matches = exact_matches + post_matches
            lexicon_warnings = exact_warnings + post_warnings
        elif self.lexicon_stage == "split_char_after_rules":
            pre_rule_output, pre_matches, pre_warnings = self._apply_lexicon_to_protected(
                protected_input,
                min_src_len=2,
                include_char_entries=False,
                allow_sentence_override=True,
                collect_matches=collect_matches,
                collect_warnings=collect_warnings,
                review_diagnostics=review_diagnostics,
            )
            rule_output, rules_applied = self._apply_rules_to_protected(
                pre_rule_output,
                collect_trace=collect_rule_trace,
                skip_passes=skip_passes,
            )
            rule_output = self._protect_unprotected_segments(rule_output)
            lexicon_output, post_matches, post_warnings = self._apply_lexicon_to_protected(
                rule_output,
                min_src_len=1,
                max_src_len=1,
                include_char_entries=True,
                allow_sentence_override=False,
                collect_matches=collect_matches,
                collect_warnings=collect_warnings,
                review_diagnostics=review_diagnostics,
            )
            matches = pre_matches + post_matches
            lexicon_warnings = pre_warnings + post_warnings
        else:
            lexicon_output, matches, lexicon_warnings = self._apply_lexicon_to_protected(
                protected_input,
                collect_matches=collect_matches,
                collect_warnings=collect_warnings,
                review_diagnostics=review_diagnostics,
            )
            lexicon_output, rules_applied = self._apply_rules_to_protected(
                lexicon_output,
                collect_trace=collect_rule_trace,
                skip_passes=skip_passes,
            )

        final_protected, cleanup_warnings = self._cleanup_protected(
            lexicon_output, collect_warnings=collect_warnings
        )
        final_output = self._post_unmask_time_cleanup(final_protected.render())
        if not preserve_spacing:
            final_output = normalize_cjk_spacing(final_output)

        warnings = lexicon_warnings + cleanup_warnings

        self._enqueue_review_if_needed(
            original_text=text,
            normalized_text=normalized,
            output_text=final_output,
            protected_input=protected_input,
            matches=matches,
            rules_applied=rules_applied,
            warnings=warnings,
            review_diagnostics=review_diagnostics,
            profile=profile,
        )

        if not trace:
            return final_output

        latency_ms = (time.perf_counter() - started) * 1000
        return ConversionResult(
            output=final_output,
            matches=matches,
            rules_applied=rules_applied,
            warnings=warnings,
            latency_ms=latency_ms,
        )
