from __future__ import annotations

import re
from typing import Any

TRUST_HUMAN = "human"
TRUST_MACHINE = "machine"
TRUST_SEED = "seed"
VALID_TRUSTS = {TRUST_HUMAN, TRUST_MACHINE, TRUST_SEED}

MACHINE_UPDATERS = {"itaigi_full", "itaigi_bot", "codex"}
MACHINE_SOURCES = {"review_queue"}
LOW_TRUST_LONG_PHRASE_MIN_LEN = 8

RUNTIME_FILTER_NOOP_MANUAL_HOTFIX = "noop_manual_hotfix"
RUNTIME_FILTER_SINGLE_CHAR_MACHINE = "single_char_machine_override"
RUNTIME_FILTER_SINGLE_CHAR_LOW_TRUST_PHRASE = "single_char_low_trust_phrase"
RUNTIME_FILTER_LOW_TRUST_LONG_PHRASE = "low_trust_long_phrase"
RUNTIME_FILTER_DEFINITION_LIKE_LOW_TRUST = "definition_like_low_trust_phrase"
RUNTIME_FILTER_NON_HANJI_TARGET = "non_hanji_target"


_HANJI_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002FA1F]")
_DEFINITION_STRONG_PREFIXES = (
    "泛指",
    "比喻",
    "形容",
    "表示",
    "指",
    "古書上",
    "傳說中",
)
_DEFINITION_SOFT_PREFIXES = (
    "一種",
    "一個",
    "一位",
    "一條",
    "一片",
    "一粒",
    "一隻",
    "一頭",
    "從事",
    "用來",
    "使人",
    "使在",
    "使勁",
    "使水",
)


def infer_trust_from_metadata(
    *,
    source: str | None,
    updated_by: str | None,
    tier: str | None,
) -> str:
    source_v = (source or "").strip()
    updated_by_v = (updated_by or "").strip()
    tier_v = (tier or "").strip()

    if source_v in MACHINE_SOURCES and updated_by_v in MACHINE_UPDATERS:
        return TRUST_MACHINE
    if source_v.startswith("user:"):
        return TRUST_HUMAN
    if tier_v in {"manual", "manual_hotfix"} and source_v == "review_queue":
        return TRUST_HUMAN
    return TRUST_SEED


def normalize_trust(
    *,
    trust: str | None,
    source: str | None,
    updated_by: str | None,
    tier: str | None,
) -> str:
    trust_v = (trust or "").strip()
    if trust_v in VALID_TRUSTS:
        return trust_v
    return infer_trust_from_metadata(source=source, updated_by=updated_by, tier=tier)


def is_trusted_manual_entry(entry: Any) -> bool:
    return getattr(entry, "tier", None) in {"manual", "manual_hotfix"} and getattr(entry, "trust", None) == TRUST_HUMAN


def is_machine_generated_override(entry: Any) -> bool:
    return getattr(entry, "tier", None) == "manual_hotfix" and getattr(entry, "trust", None) == TRUST_MACHINE


def is_sentence_manual_override(entry: Any) -> bool:
    return (
        getattr(entry, "level", None) == "sentence"
        and is_trusted_manual_entry(entry)
        and not getattr(entry, "context", None)
        and getattr(entry, "status", None) == "active"
    )


def is_noop_manual_hotfix(entry: Any) -> bool:
    return (
        getattr(entry, "tier", None) == "manual_hotfix"
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and getattr(entry, "trust", None) == TRUST_MACHINE
        and getattr(entry, "src", "") == getattr(entry, "tgt", "")
    )


def is_single_char_machine_override(entry: Any) -> bool:
    return (
        getattr(entry, "trust", None) == TRUST_MACHINE
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and len(getattr(entry, "src", "")) == 1
    )


def is_single_char_low_trust_phrase(entry: Any) -> bool:
    return (
        getattr(entry, "tier", None) in {"base", "domain"}
        and getattr(entry, "trust", None) in {TRUST_SEED, TRUST_MACHINE}
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and len(getattr(entry, "src", "")) == 1
    )


def is_low_trust_long_phrase(entry: Any, min_len: int = LOW_TRUST_LONG_PHRASE_MIN_LEN) -> bool:
    return (
        getattr(entry, "tier", None) in {"base", "domain"}
        and getattr(entry, "trust", None) in {TRUST_SEED, TRUST_MACHINE}
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and len(getattr(entry, "src", "")) >= min_len
    )


def is_definition_like_low_trust_phrase(entry: Any) -> bool:
    if (
        getattr(entry, "tier", None) not in {"base", "domain"}
        or getattr(entry, "trust", None) not in {TRUST_SEED, TRUST_MACHINE}
        or getattr(entry, "level", None) not in {"phrase", "sentence"}
    ):
        return False

    src = getattr(entry, "src", "")
    if len(src) < 4:
        return False

    if src.startswith(_DEFINITION_STRONG_PREFIXES):
        return True

    if src.startswith(_DEFINITION_SOFT_PREFIXES):
        return ("的" in src) or (len(src) >= 8)

    return False


def is_non_hanji_target(entry: Any) -> bool:
    if (
        getattr(entry, "tier", None) not in {"base", "domain", "manual_hotfix"}
        or getattr(entry, "trust", None) not in {TRUST_SEED, TRUST_MACHINE}
        or getattr(entry, "level", None) not in {"phrase", "sentence", "char"}
    ):
        return False

    tgt = getattr(entry, "tgt", "")
    if not tgt:
        return True
    return _HANJI_RE.search(tgt) is None


def runtime_exclusion_reason(entry: Any) -> str | None:
    if is_noop_manual_hotfix(entry):
        return RUNTIME_FILTER_NOOP_MANUAL_HOTFIX
    if is_single_char_machine_override(entry):
        return RUNTIME_FILTER_SINGLE_CHAR_MACHINE
    if is_single_char_low_trust_phrase(entry):
        return RUNTIME_FILTER_SINGLE_CHAR_LOW_TRUST_PHRASE
    if is_low_trust_long_phrase(entry):
        return RUNTIME_FILTER_LOW_TRUST_LONG_PHRASE
    if is_definition_like_low_trust_phrase(entry):
        return RUNTIME_FILTER_DEFINITION_LIKE_LOW_TRUST
    if is_non_hanji_target(entry):
        return RUNTIME_FILTER_NON_HANJI_TARGET
    return None
