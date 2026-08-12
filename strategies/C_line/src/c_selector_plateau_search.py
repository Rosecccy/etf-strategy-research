from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core
import walkforward_overlay_validation as wf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "c_selector_plateau"


@dataclass(frozen=True)
class Policy:
    min_triggers: int
    min_ratio: float
    band: float
    choice: str

    @property
    def key(self) -> str:
        return f"plateau:n{self.min_triggers}:r{self.min_ratio:.3f}:b{self.band:.3f}:{self.choice}"


def policies() -> list[Policy]:
    return [
        Policy(min_triggers, min_ratio, band, choice)
        for min_triggers in (2, 3, 4, 5)
        for min_ratio in (1.0, 1.005, 1.01, 1.02)
        for band in (0.005, 0.01, 0.02, 0.03, 0.05)
        for choice in ("median", "lower_complexity")
    ]


def parameters(rule: str) -> tuple[float, float, int]:
    matches = re.findall(r"trailing:a\+([0-9.]+):b\+([0-9.]+):h([0-9]+)", rule)
    if not matches:
        return 0.0, 0.0, 0
    a, b, hold = matches[-1]
    return float(a), float(b), int(hold)


def candidate_rows(
    profiles: dict[str, dict[int, dict]],
    year: int,
    policy: Policy,
) -> list[dict]:
    base = profiles["baseline"][year]
    rows = []
    for name, values in profiles.items():
        if name == "baseline":
            continue
        current = values[year]
        ratio = current["final_value"] / base["final_value"]
        win_delta = current["win_rate"] - base["win_rate"]
        dd_delta = current["max_drawdown"] - base["max_drawdown"]
        eligible = bool(
            current["trades"] >= wf.MIN_TRAIN_TRADES
            and current["trigger_count"] >= policy.min_triggers
            and ratio >= policy.min_ratio
            and win_delta >= -1e-12
            and dd_delta >= -0.005
        )
        score = (
            np.log(max(ratio, 1e-12))
            + 0.60 * win_delta
            + 0.25 * min(dd_delta, 0.20)
            - 0.002 * max(current["trigger_count"] - 12, 0)
        )
        a, b, hold = parameters(name)
        rows.append(
            {
                "rule": name,
                "eligible": eligible,
                "score": score,
                "ratio": ratio,
                "win_delta": win_delta,
                "dd_delta": dd_delta,
                "triggers": current["trigger_count"],
                "a": a,
                "b": b,
                "hold": hold,
            }
        )
    return rows


def choose(profiles: dict[str, dict[int, dict]], year: int, policy: Policy) -> tuple[str, dict]:
    rows = candidate_rows(profiles, year, policy)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return "baseline", {"year": year, "rule": "baseline", "policy": policy.key}
    best_score = max(row["score"] for row in eligible)
    plateau = [row for row in eligible if row["score"] >= best_score - policy.band]
    if len(plateau) < 3:
        return "baseline", {
            "year": year,
            "rule": "baseline",
            "policy": policy.key,
            "plateau_size": len(plateau),
        }
    if policy.choice == "median":
        center = np.median([[row["a"], row["b"], row["hold"] / 100.0] for row in plateau], axis=0)
        winner = min(
            plateau,
            key=lambda row: (
                (row["a"] - center[0]) ** 2
                + (row["b"] - center[1]) ** 2
                + (row["hold"] / 100.0 - center[2]) ** 2,
                -row["score"],
            ),
        )
    else:
        # Prefer fewer historical interventions when performance is near-equal.
        winner = max(plateau, key=lambda row: (row["score"] - 0.003 * row["triggers"], row["hold"]))
    return str(winner["rule"]), {
        "year": year,
        "rule": winner["rule"],
        "policy": policy.key,
        "plateau_size": len(plateau),
        **{key: winner[key] for key in ("score", "ratio", "win_delta", "dd_delta", "triggers", "a", "b", "hold")},
    }


def build_policy(
    frames: dict[str, pd.DataFrame],
    profiles: dict[str, dict[int, dict]],
    years: list[int],
    policy: Policy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    selections = []
    for year in years:
        rule, audit = choose(profiles, int(year), policy)
        source = frames[rule]
        local = source[pd.to_datetime(source["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_plateau_policy"] = policy.key
        local["walkforward_rule"] = rule
        parts.append(local)
        selections.append(audit)
    return (
        pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]),
        pd.DataFrame(selections),
    )


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
    profiles = {name: wf.training_profile(frame, years) for name, frame in frames.items()}

    current = pd.read_csv(ROOT / "fit" / "rolling_upgrades" / "c_trades.csv", encoding="utf-8-sig", dtype={"symbol": str})
    current["symbol"] = current["symbol"].str.zfill(6)
    current_stats = metrics(current)
    rows = []
    generated: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for policy in policies():
        frame, selections = build_policy(frames, profiles, years, policy)
        generated[policy.key] = (frame, selections)
        stats = metrics(frame)
        rows.append(
            {
                "policy": policy.key,
                "min_triggers": policy.min_triggers,
                "min_ratio": policy.min_ratio,
                "band": policy.band,
                "choice": policy.choice,
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
        winner_selections = pd.DataFrame()
    else:
        eligible["dev_score"] = (
            np.log(eligible["dev_final"] / base_dev["final_value"])
            + 0.6 * (eligible["dev_win"] - base_dev["win_rate"])
            + 0.2 * np.minimum(eligible["dev_dd"] - base_dev["max_drawdown"], 0.2)
        )
        selected = eligible.sort_values(["dev_score", "selected_years"], ascending=[False, True]).iloc[0]
        winner_key = str(selected["policy"])
        winner_frame, winner_selections = generated[winner_key]
    winner_stats = metrics(winner_frame)
    nearby_count = 0
    if winner_key != "current":
        winner_row = table[table["policy"].eq(winner_key)].iloc[0]
        nearby = table[
            table["min_triggers"].sub(winner_row["min_triggers"]).abs().le(1)
            & table["min_ratio"].sub(winner_row["min_ratio"]).abs().le(0.006)
            & table["band"].sub(winner_row["band"]).abs().le(0.011)
            & table["choice"].eq(winner_row["choice"])
        ]
        nearby_count = int(
            (
                nearby["dev_final"].gt(base_dev["final_value"])
                & nearby["holdout_final"].ge(current_stats["holdout"]["final_value"])
            ).sum()
        )
    payload = {
        "method": "development-selected annual rolling plateau selector; 2024-2026 retained as validation",
        "candidate_policies": len(table),
        "winner_policy": winner_key,
        "nearby_passing": nearby_count,
        "current": current_stats,
        "winner": winner_stats,
        "full_ratio": winner_stats["full"]["final_value"] / current_stats["full"]["final_value"],
        "holdout_ratio": winner_stats["holdout"]["final_value"] / current_stats["holdout"]["final_value"],
    }
    table.to_csv(OUT / "policies.csv", index=False, encoding="utf-8-sig")
    winner_frame.to_csv(OUT / "winner_trades.csv", index=False, encoding="utf-8-sig")
    winner_selections.to_csv(OUT / "winner_selections.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
