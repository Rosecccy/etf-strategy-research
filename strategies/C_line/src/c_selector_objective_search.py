from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core
import walkforward_overlay_validation as wf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "c_selector_objective"


@dataclass(frozen=True)
class Objective:
    min_triggers: int
    min_ratio: float
    win_weight: float
    dd_weight: float
    trigger_penalty: float

    @property
    def key(self) -> str:
        return (
            f"objective:n{self.min_triggers}:r{self.min_ratio:.3f}:"
            f"w{self.win_weight:.2f}:d{self.dd_weight:.2f}:p{self.trigger_penalty:.3f}"
        )


def objectives() -> list[Objective]:
    return [
        Objective(n, ratio, win, dd, penalty)
        for n in (2, 3, 4)
        for ratio in (1.0, 1.005, 1.01)
        for win in (0.30, 0.60, 1.0)
        for dd in (0.10, 0.25, 0.50)
        for penalty in (0.0, 0.002, 0.004)
    ]


def choose(
    profiles: dict[str, dict[int, dict]], year: int, objective: Objective
) -> tuple[str, dict]:
    base = profiles["baseline"][year]
    rows = []
    for rule, values in profiles.items():
        if rule == "baseline":
            continue
        current = values[year]
        ratio = current["final_value"] / base["final_value"]
        win_delta = current["win_rate"] - base["win_rate"]
        dd_delta = current["max_drawdown"] - base["max_drawdown"]
        eligible = bool(
            current["trades"] >= wf.MIN_TRAIN_TRADES
            and current["trigger_count"] >= objective.min_triggers
            and ratio >= objective.min_ratio
            and win_delta >= -1e-12
            and dd_delta >= -0.005
        )
        score = (
            np.log(max(ratio, 1e-12))
            + objective.win_weight * win_delta
            + objective.dd_weight * min(dd_delta, 0.20)
            - objective.trigger_penalty * max(current["trigger_count"] - 12, 0)
        )
        rows.append(
            {
                "rule": rule,
                "eligible": eligible,
                "score": score,
                "ratio": ratio,
                "win_delta": win_delta,
                "dd_delta": dd_delta,
                "triggers": current["trigger_count"],
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return "baseline", {"year": year, "rule": "baseline", "objective": objective.key}
    winner = max(eligible, key=lambda row: row["score"])
    return str(winner["rule"]), {"year": year, "objective": objective.key, **winner}


def build(
    frames: dict[str, pd.DataFrame],
    profiles: dict[str, dict[int, dict]],
    years: list[int],
    objective: Objective,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    rows = []
    for year in years:
        rule, audit = choose(profiles, int(year), objective)
        frame = frames[rule]
        local = frame[pd.to_datetime(frame["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_objective"] = objective.key
        local["walkforward_rule"] = rule
        parts.append(local)
        rows.append(audit)
    return pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]), pd.DataFrame(rows)


def metrics(frame: pd.DataFrame) -> dict:
    return {period: search.subset_stats(frame, period) for period in ("full", "dev", "holdout")}


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = next(spec for spec in core.SPECS if spec.key == "C")
    trades = core.load_trades(spec)
    prices = core.load_prices(spec.project, set(trades["symbol"]))
    frames = wf.candidate_frames(spec, trades, prices)
    years = sorted(pd.to_datetime(trades["entry_date"]).dt.year.unique())
    profiles = {rule: wf.training_profile(frame, years) for rule, frame in frames.items()}

    current = pd.read_csv(ROOT / "fit" / "rolling_upgrades" / "c_trades.csv", encoding="utf-8-sig", dtype={"symbol": str})
    current["symbol"] = current["symbol"].str.zfill(6)
    current_stats = metrics(current)
    generated = {}
    rows = []
    for objective in objectives():
        frame, selections = build(frames, profiles, years, objective)
        generated[objective.key] = (frame, selections)
        stats = metrics(frame)
        rows.append(
            {
                "objective": objective.key,
                "min_triggers": objective.min_triggers,
                "min_ratio": objective.min_ratio,
                "win_weight": objective.win_weight,
                "dd_weight": objective.dd_weight,
                "trigger_penalty": objective.trigger_penalty,
                "full_final": stats["full"]["final_value"],
                "full_win": stats["full"]["win_rate"],
                "full_annual": stats["full"]["avg_annual_return"],
                "full_dd": stats["full"]["max_drawdown"],
                "dev_final": stats["dev"]["final_value"],
                "dev_win": stats["dev"]["win_rate"],
                "dev_dd": stats["dev"]["max_drawdown"],
                "holdout_final": stats["holdout"]["final_value"],
                "holdout_win": stats["holdout"]["win_rate"],
                "selected_years": int((selections["rule"] != "baseline").sum()),
            }
        )
    table = pd.DataFrame(rows)
    base_dev = current_stats["dev"]
    eligible = table[
        table["dev_final"].gt(base_dev["final_value"])
        & table["dev_win"].ge(base_dev["win_rate"] - 1e-12)
        & table["dev_dd"].ge(base_dev["max_drawdown"] - 0.005)
    ].copy()
    if eligible.empty:
        winner_key = "current"
        winner_frame = current
        selections = pd.DataFrame()
    else:
        eligible["dev_score"] = (
            np.log(eligible["dev_final"] / base_dev["final_value"])
            + 0.6 * (eligible["dev_win"] - base_dev["win_rate"])
            + 0.2 * np.minimum(eligible["dev_dd"] - base_dev["max_drawdown"], 0.2)
        )
        selected = eligible.sort_values(["dev_score", "selected_years"], ascending=[False, True]).iloc[0]
        winner_key = str(selected["objective"])
        winner_frame, selections = generated[winner_key]
    winner_stats = metrics(winner_frame)
    nearby_count = 0
    if winner_key != "current":
        winner_row = table[table["objective"].eq(winner_key)].iloc[0]
        nearby = table[
            table["min_triggers"].sub(winner_row["min_triggers"]).abs().le(1)
            & table["min_ratio"].sub(winner_row["min_ratio"]).abs().le(0.006)
            & table["win_weight"].sub(winner_row["win_weight"]).abs().le(0.31)
            & table["dd_weight"].sub(winner_row["dd_weight"]).abs().le(0.16)
            & table["trigger_penalty"].sub(winner_row["trigger_penalty"]).abs().le(0.0021)
        ]
        nearby_count = int(
            (
                nearby["dev_final"].gt(base_dev["final_value"])
                & nearby["holdout_final"].ge(current_stats["holdout"]["final_value"])
                & nearby["holdout_win"].ge(current_stats["holdout"]["win_rate"] - 1e-12)
            ).sum()
        )
    payload = {
        "method": "objective selected on rolling development outcomes; 2024-2026 validation untouched by ranking",
        "candidate_objectives": len(table),
        "winner_objective": winner_key,
        "nearby_passing": nearby_count,
        "current": current_stats,
        "winner": winner_stats,
        "full_ratio": winner_stats["full"]["final_value"] / current_stats["full"]["final_value"],
        "holdout_ratio": winner_stats["holdout"]["final_value"] / current_stats["holdout"]["final_value"],
        "holdout_pass": bool(
            winner_stats["holdout"]["final_value"] >= current_stats["holdout"]["final_value"]
            and winner_stats["holdout"]["win_rate"] >= current_stats["holdout"]["win_rate"] - 1e-12
        ),
    }
    table.to_csv(OUT / "objectives.csv", index=False, encoding="utf-8-sig")
    winner_frame.to_csv(OUT / "winner_trades.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(OUT / "winner_selections.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
