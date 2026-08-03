from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

from .context_policy import validated_context
from .lexicon_policy import normalize_trust

TIER_ORDER = ["blocked", "manual_hotfix", "manual", "core", "domain", "base"]
PASS_ORDER = ["normalization", "grammar", "fluency", "punctuation"]


@dataclass
class LexiconEntry:
    entry_id: str
    src: str
    tgt: str
    level: str
    tier: str
    priority: int = 0
    context: dict[str, Any] | None = None
    score: float = 0.0
    status: str = "active"
    source: str = ""
    trust: str = "seed"
    updated_by: str = "system"
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LexiconEntry:
        return cls(
            entry_id=data["entry_id"],
            src=data["src"],
            tgt=data.get("tgt", ""),
            level=data["level"],
            tier=data["tier"],
            priority=int(data.get("priority", 0)),
            context=data.get("context"),
            score=float(data.get("score", 0.0)),
            status=data.get("status", "active"),
            source=data.get("source", ""),
            trust=normalize_trust(
                trust=data.get("trust"),
                source=data.get("source"),
                updated_by=data.get("updated_by"),
                tier=data.get("tier"),
            ),
            updated_by=data.get("updated_by", "system"),
            updated_at=data.get("updated_at", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleEntry:
    rule_id: str
    pass_name: str
    type: str
    pattern: str
    replacement: str
    priority: int = 0
    enabled: bool = True
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleEntry:
        return cls(
            rule_id=data["rule_id"],
            pass_name=data["pass_name"],
            type=data.get("type", "literal"),
            pattern=data["pattern"],
            replacement=data.get("replacement", ""),
            priority=int(data.get("priority", 0)),
            enabled=bool(data.get("enabled", True)),
            note=data.get("note", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _freeze_runtime_value(value: Any) -> Any:
    """Recursively detach and freeze JSON-like runtime metadata."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_runtime_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_runtime_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_runtime_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class RuntimeLexiconEntry:
    """Immutable, cache-safe lexicon representation used by the converter."""

    entry_id: str
    src: str
    tgt: str
    level: str
    tier: str
    priority: int = 0
    context: Mapping[str, Any] | None = None
    score: float = 0.0
    status: str = "active"
    source: str = ""
    trust: str = "seed"
    updated_by: str = "system"
    updated_at: str = ""

    def __post_init__(self) -> None:
        checked_context = validated_context(self.context)
        if checked_context is not None:
            object.__setattr__(self, "context", _freeze_runtime_value(checked_context))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeLexiconEntry:
        return cls(
            entry_id=str(data["entry_id"]),
            src=str(data["src"]),
            tgt=str(data.get("tgt", "")),
            level=str(data["level"]),
            tier=str(data["tier"]),
            priority=int(data.get("priority", 0)),
            context=data.get("context"),
            score=float(data.get("score", 0.0)),
            status=str(data.get("status", "active")),
            source=str(data.get("source", "")),
            trust=normalize_trust(
                trust=data.get("trust"),
                source=data.get("source"),
                updated_by=data.get("updated_by"),
                tier=data.get("tier"),
            ),
            updated_by=str(data.get("updated_by", "system")),
            updated_at=str(data.get("updated_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "src": self.src,
            "tgt": self.tgt,
            "level": self.level,
            "tier": self.tier,
            "priority": self.priority,
            "context": dict(self.context) if self.context is not None else None,
            "score": self.score,
            "status": self.status,
            "source": self.source,
            "trust": self.trust,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class RuntimeRuleEntry:
    """Immutable, cache-safe rule representation used by the converter."""

    rule_id: str
    pass_name: str
    type: str
    pattern: str
    replacement: str
    priority: int = 0
    enabled: bool = True
    note: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeRuleEntry:
        return cls(
            rule_id=str(data["rule_id"]),
            pass_name=str(data["pass_name"]),
            type=str(data.get("type", "literal")),
            pattern=str(data["pattern"]),
            replacement=str(data.get("replacement", "")),
            priority=int(data.get("priority", 0)),
            enabled=bool(data.get("enabled", True)),
            note=str(data.get("note", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "pass_name": self.pass_name,
            "type": self.type,
            "pattern": self.pattern,
            "replacement": self.replacement,
            "priority": self.priority,
            "enabled": self.enabled,
            "note": self.note,
        }


@dataclass
class MatchTrace:
    entry_id: str
    src: str
    tgt: str
    level: str
    tier: str
    start: int
    end: int
    priority: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleTrace:
    rule_id: str
    pass_name: str
    type: str
    pattern: str
    replacement: str
    hit_count: int
    matched_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversionResult:
    output: str
    matches: list[MatchTrace] = field(default_factory=list)
    rules_applied: list[RuleTrace] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "matches": [m.to_dict() for m in self.matches],
            "rules_applied": [r.to_dict() for r in self.rules_applied],
            "warnings": self.warnings,
            "latency_ms": self.latency_ms,
        }
