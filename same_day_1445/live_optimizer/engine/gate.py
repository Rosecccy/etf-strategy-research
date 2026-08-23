from __future__ import annotations

from datetime import date
from typing import Any

from .records import GateResult, OptimizerState


def evaluate_gate(line: str, candidate_eval: dict[str, Any], baseline_eval: dict[str, Any], new_evidence: list[dict[str, Any]], state: OptimizerState, config: dict[str, Any], as_of: str) -> GateResult:
    gates = config["gates"]
    c = candidate_eval["overall"]
    b = baseline_eval["overall"]
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    trigger_min = float(gates.get("trigger_retention_min", 0.90))
    win_delta = c.win_rate - b.win_rate
    mean_delta = c.mean_return - b.mean_return
    dd_deterioration = max(0.0, b.max_drawdown - c.max_drawdown)
    evidence.update({"trigger_retention": c.trigger_retention, "win_rate_delta": win_delta, "mean_return_delta": mean_delta, "drawdown_deterioration": dd_deterioration})
    if not candidate_eval.get("causal", True):
        reasons.append("CAUSALITY")
    if c.trigger_retention < trigger_min:
        reasons.append("TRIGGER_RETENTION")
    if win_delta < float(gates.get("win_rate_delta_min", 0.005)):
        reasons.append("WIN_RATE_DELTA")
    if mean_delta < float(gates.get("mean_return_delta_min", 0.0)):
        reasons.append("MEAN_RETURN_DELTA")
    if dd_deterioration > float(gates.get("max_drawdown_deterioration_max", 0.01)):
        reasons.append("MAX_DRAWDOWN")
    windows = candidate_eval.get("windows", [])
    if windows:
        improved = sum(bool(w.get("improved")) for w in windows) / len(windows)
        if improved < float(gates.get("recent_windows_improve_fraction_min", 2 / 3)):
            reasons.append("WINDOW_STABILITY")
        recent = windows[-1]
        if (not recent.get("improved")) or float(recent.get("win_rate_delta", 0.0)) < -float(gates.get("recent_window_win_rate_regression_max", 0.0)):
            reasons.append("RECENT_WINDOW_REGRESSION")
        evidence["improved_window_fraction"] = improved
    neighbor = float(candidate_eval.get("neighbor_pass_rate", 0.0))
    if neighbor < float(gates.get("neighbor_pass_rate_min", 0.5)):
        reasons.append("NEIGHBOR_STABILITY")
    line_cfg = config["lines"][line]
    if len(new_evidence) < int(line_cfg.get("minimum_new_samples", 0)):
        reasons.append("MIN_NEW_SAMPLES")
    regimes = {str(row.get("regime", "")) for row in new_evidence if str(row.get("regime", ""))}
    if len(regimes) < int(line_cfg.get("minimum_regimes", 1)):
        reasons.append("MIN_REGIMES")
    last = state.last_promotion_at.get(line)
    if last:
        elapsed = (date.fromisoformat(str(as_of)[:10]) - date.fromisoformat(str(last)[:10])).days
        if elapsed < int(config.get("cooldown_days", 0)):
            reasons.append("COOLDOWN")
        evidence["cooldown_elapsed_days"] = elapsed
    evidence["new_samples"] = len(new_evidence)
    evidence["regimes"] = sorted(regimes)
    evidence["neighbor_pass_rate"] = neighbor
    reasons = list(dict.fromkeys(reasons))
    return GateResult(passed=not reasons, reasons=reasons, evidence=evidence)
