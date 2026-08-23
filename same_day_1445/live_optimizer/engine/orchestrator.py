from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_optimizer_config, stable_json_hash
from .drift import detect_drift
from .gate import evaluate_gate
from .ledger import AppendOnlyCsvLedger
from .records import OptimizerState
from .rolling_evaluator import compute_metrics, evaluate_candidate_windows


def _state(root: Path, cfg: dict[str, Any]) -> OptimizerState:
    path = root / "state" / "optimizer_state.json"
    if path.exists():
        return OptimizerState.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
    state = OptimizerState.initial()
    state.mode = cfg["mode"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return state


def _save_state(root: Path, state: OptimizerState) -> None:
    path = root / "state" / "optimizer_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _score(eval_result: dict[str, Any]) -> tuple[float, float, float]:
    m = eval_result["overall"]
    return (m.win_rate, m.mean_return, m.final_equity)


def run_optimizer(root: Path, as_of: str) -> dict[str, Any]:
    root = Path(root)
    cfg = load_optimizer_config(root / "config" / "optimizer.json")
    state = _state(root, cfg)
    state.mode = cfg["mode"] if state.mode not in {"DATA_HOLD", "ROLLBACK"} else state.mode
    ledger = AppendOnlyCsvLedger(root / "ledger" / "closed_trades.csv", ("sample_id", "candidate_id"))
    rows = ledger.rows()
    lines_out: dict[str, Any] = {}

    for line in ("C", "S", "D", "R"):
        line_rows = [r for r in rows if r.get("line") == line]
        formal = state.active_formal[line]
        baseline_rows = [r for r in line_rows if r.get("candidate_id") == formal]
        candidate_ids = sorted({r.get("candidate_id") for r in line_rows if r.get("candidate_id") and r.get("candidate_id") != formal})
        accounting = cfg["lines"][line].get("accounting", "research_sum" if line == "D" else "compound")
        baseline_eval = {"overall": compute_metrics(baseline_rows, accounting)}

        ordered_baseline = sorted(baseline_rows, key=lambda r: (str(r.get("decision_at", "")), str(r.get("sample_id", ""))))
        recent_count = int(cfg.get("drift", {}).get("recent_count", 20))
        if recent_count > 0 and len(ordered_baseline) >= recent_count * 2:
            drift = detect_drift(ordered_baseline[-recent_count:], ordered_baseline[:-recent_count], cfg)
        else:
            drift = {"severe": False, "breaches": [], "deltas": {}, "recent": {}, "reference": {}}
        if drift.get("severe") and state.mode == "NORMAL":
            state.mode = "SHADOW_ONLY"

        candidates = []
        for cid in candidate_ids:
            c_rows = [r for r in line_rows if r.get("candidate_id") == cid]
            ev = evaluate_candidate_windows(c_rows, baseline_rows, cfg.get("evaluation_windows", []), accounting)
            gate = evaluate_gate(line, ev, baseline_eval, c_rows, state, cfg, as_of)
            candidates.append({"candidate_id": cid, "evaluation": ev, "gate": gate})

        eligible = [item for item in candidates if item["gate"].passed]
        if eligible:
            leader = max(eligible, key=lambda item: _score(item["evaluation"]))["candidate_id"]
            state.shadow_leader[line] = leader

        lines_out[line] = {
            "formal": formal,
            "shadow_leader": state.shadow_leader[line],
            "drift": drift,
            "candidates": [
                {
                    "candidate_id": item["candidate_id"],
                    "gate": item["gate"].to_dict(),
                    "overall": item["evaluation"]["overall"].to_dict(),
                }
                for item in candidates
            ],
        }

    _save_state(root, state)
    decision_core = {"as_of": as_of, "mode": state.mode, "lines": lines_out}
    decision_hash = stable_json_hash(decision_core)
    result = {**decision_core, "decision_hash": decision_hash}
    runs = AppendOnlyCsvLedger(root / "ledger" / "optimizer_runs.csv", ("decision_hash",))
    runs.append({
        "decision_hash": decision_hash,
        "as_of": as_of,
        "mode": state.mode,
        "state_hash": stable_json_hash(state.to_dict()),
        "line_summary_hash": stable_json_hash(lines_out),
    })
    return result
