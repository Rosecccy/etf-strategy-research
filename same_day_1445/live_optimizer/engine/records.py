from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

LINES = ("C", "S", "D", "R")


@dataclass(frozen=True)
class EvaluationMetrics:
    trades: int
    opportunities: int
    trigger_retention: float
    win_rate: float
    mean_return: float
    max_drawdown: float
    final_equity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerState:
    mode: str
    active_formal: dict[str, str]
    previous_formal: dict[str, str | None]
    shadow_leader: dict[str, str | None]
    last_promotion_at: dict[str, str | None]

    @classmethod
    def initial(cls) -> "OptimizerState":
        return cls(
            mode="SHADOW_ONLY",
            active_formal={line: "release_v2" for line in LINES},
            previous_formal={line: None for line in LINES},
            shadow_leader={line: None for line in LINES},
            last_promotion_at={line: None for line in LINES},
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OptimizerState":
        base = cls.initial()
        return cls(
            mode=str(value.get("mode", base.mode)),
            active_formal={**base.active_formal, **value.get("active_formal", {})},
            previous_formal={**base.previous_formal, **value.get("previous_formal", {})},
            shadow_leader={**base.shadow_leader, **value.get("shadow_leader", {})},
            last_promotion_at={**base.last_promotion_at, **value.get("last_promotion_at", {})},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
