from __future__ import annotations

import re
import unicodedata

# Unicode 17.0 Han ideograph coverage. Keep this as the single source of truth
# for runtime lexical eligibility and CJK whitespace normalization so newly
# standardized Extension G/H/I/J characters are not mistaken for non-Han text.
HAN_IDEOGRAPH_CLASS = (
    r"\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
    r"\u4e00-\u9fff"  # CJK Unified Ideographs
    r"\uf900-\ufaff"  # CJK Compatibility Ideographs
    r"\U00020000-\U0002ee5f"  # Extensions B-I
    r"\U0002f800-\U0002fa1f"  # CJK Compatibility Ideographs Supplement
    r"\U00030000-\U0003347f"  # Extensions G, H, and J
)
HAN_IDEOGRAPH_RE = re.compile(f"[{HAN_IDEOGRAPH_CLASS}]")


def contains_han_ideograph(text: str) -> bool:
    return HAN_IDEOGRAPH_RE.search(text) is not None


def private_use_code_points(text: str) -> tuple[int, ...]:
    """Return Unicode private-use code points in source order.

    Runtime lexicon targets must use standardized Unicode characters. Legacy
    font-specific PUA glyphs are not portable and therefore fail closed during
    artifact compilation while disabled historical rows remain auditable.
    """

    return tuple(ord(character) for character in text if unicodedata.category(character) == "Co")
