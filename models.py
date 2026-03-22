from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from lexicon_policy import normalize_trust

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
    def from_dict(cls, data: dict[str, Any]) -> "LexiconEntry":
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
    def from_dict(cls, data: dict[str, Any]) -> "RuleEntry":
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
