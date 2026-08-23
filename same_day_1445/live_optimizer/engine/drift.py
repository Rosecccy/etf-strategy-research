from __future__ import annotations

from statistics import mean
from typing import Any


def _bool(v: Any) -> bool:
    return v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "y"}


def _stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"trigger_rate": 0.0, "win_rate": 0.0, "mean_return": 0.0}
    trigger = [_bool(r.get("triggered", True)) for r in rows]
    triggered_rows = [r for r, t in zip(rows, trigger) if t]
    wins = [(_bool(r.get("won")) if "won" in r else float(r.get("ret", 0) or 0) > 0) for r in triggered_rows]
    returns = [float(r.get("ret", 0) or 0) for r in triggered_rows]
    return {
        "trigger_rate": sum(trigger) / len(trigger),
        "win_rate": sum(wins) / len(wins) if wins else 0.0,
        "mean_return": mean(returns) if returns else 0.0,
    }


def detect_drift(recent_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    recent = _stats(recent_rows)
    ref = _stats(reference_rows)
    dc = config.get("drift", {})
    limits = {
        "trigger_rate": float(dc.get("trigger_rate_abs_max", 0.30)),
        "win_rate": float(dc.get("win_rate_abs_max", 0.30)),
        "mean_return": float(dc.get("mean_return_abs_max", 0.10)),
    }
    deltas = {key: recent[key] - ref[key] for key in limits}
    breaches = [key for key, limit in limits.items() if abs(deltas[key]) > limit]
    return {"severe": bool(breaches), "breaches": breaches, "deltas": deltas, "recent": recent, "reference": ref}
