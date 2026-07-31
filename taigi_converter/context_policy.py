from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

VALID_CONTEXT_FIELDS = frozenset(
    {
        "left_regex",
        "right_regex",
        "full_regex",
        "left_literal",
        "right_literal",
    }
)
CONTEXT_REGEX_FIELDS = frozenset({"left_regex", "right_regex", "full_regex"})
RUNTIME_CONTEXT_RIGHT_REGEX = "r"
RUNTIME_CONTEXT_LEFT_LITERAL = "l"


def context_validation_errors(context: Any) -> list[str]:
    """Return deterministic schema errors for a source or runtime context payload."""

    if context is None:
        return []
    if not isinstance(context, Mapping):
        return ["context 必須是 object 或 null"]
    if not context:
        return ["context 若有提供，至少必須包含一個條件欄位"]

    errors: list[str] = []
    unknown = set(context) - VALID_CONTEXT_FIELDS
    if unknown:
        errors.append(f"context 含未知欄位 {sorted(str(field) for field in unknown)}")

    for field in sorted(set(context) & VALID_CONTEXT_FIELDS):
        value = context[field]
        if not isinstance(value, str) or not value:
            errors.append(f"context.{field} 必須是非空字串")
            continue
        if field in CONTEXT_REGEX_FIELDS:
            try:
                re.compile(value)
            except re.error as exc:
                errors.append(f"context.{field} regex 無法編譯: {exc}")
    return errors


def validated_context(context: Any, *, error_prefix: str = "Invalid runtime context") -> dict[str, str] | None:
    """Return a detached valid context mapping, or fail closed."""

    errors = context_validation_errors(context)
    if errors:
        raise ValueError(f"{error_prefix}: {'; '.join(errors)}")
    if context is None:
        return None
    return {str(field): str(value) for field, value in context.items()}


def decode_runtime_context(context: Any) -> dict[str, str] | None:
    """Decode compact runtime context metadata and validate the full shared schema."""

    decoded = context
    if isinstance(context, list):
        if len(context) != 2 or not isinstance(context[0], str) or not isinstance(context[1], str):
            raise ValueError("Invalid runtime context encoding")
        if context[0] == RUNTIME_CONTEXT_RIGHT_REGEX:
            decoded = {"right_regex": context[1]}
        elif context[0] == RUNTIME_CONTEXT_LEFT_LITERAL:
            decoded = {"left_literal": context[1]}
        else:
            raise ValueError("Invalid runtime context encoding")
    return validated_context(decoded)
