from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as control
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "c_profit_lock_rolling"
SOURCE = ROOT / "fit" / "rolling_upgrades" / "c_trades.csv"


@dataclass(frozen=True)
class Rule:
    arm: float
    floor: float
    hold: int

    @property
    def key(self) -> str:
        return f"profit_lock:a{self.arm:.3f}:f{self.floor:.3f}:h{self.hold}"


def rules() -> list[Rule]:
    return [
        Rule(arm, floor, hold)
        for arm in (0.08, 0.10, 0.12, 0.15)
        for floor in (0.04, 0.06, 0.08, 0.10)
        for hold in (20, 30, 40)
        if floor < arm
    ]


def load() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in ("entry_date", "exit_date_test"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def apply_rule(frame: pd.DataFrame, prices: dict[str, pd.DataFrame], rule: Rule) -> pd.DataFrame:
    mapped = control.Rule("profit_floor", rule.arm, rule.floor, rule.hold)
    result = control.apply_rule(frame, prices, mapped)
    result["profit_lock_rule"] = rule.key
    result["profit_lock_triggered"] = result["control_triggered"]
    return result


def training_stats(frame: pd.DataFrame, year: int) -> dict:
    local = frame[pd.to_datetime(frame["entry_date"]).dt.year.lt(year)].copy()
    return core.simulate_account(local)[0]


def choose(
    base: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    rule_map: dict[str, Rule],
    year: int,
) -> tuple[str, dict]:
    train_count = int(pd.to_datetime(base["entry_date"]).dt.year.lt(year).sum())
    if train_count < 12:
        return "baseline", {"test_year": year, "rule": "baseline", "reason": "insufficient_history"}
    baseline = training_stats(base, year)
    rows = []
    for key, candidate in candidates.items():
        current = training_stats(candidate, year)
        triggered = int(
            candidate[
                pd.to_datetime(candidate["entry_date"]).dt.year.lt(year)
            ]["profit_lock_triggered"].sum()
        )
        rows.append(
            {
                "rule": key,
                "arm": rule_map[key].arm,
                "floor": rule_map[key].floor,
                "hold": rule_map[key].hold,
                "ratio": current["final_value"] / baseline["final_value"],
                "win_delta": current["win_rate"] - baseline["win_rate"],
                "dd_delta": current["max_drawdown"] - baseline["max_drawdown"],
                "triggered": triggered,
            }
        )
    table = pd.DataFrame(rows)
    passing = []
    for _, row in table.iterrows():
        neighbor = table[
            table["hold"].eq(row["hold"])
            & table["arm"].sub(row["arm"]).abs().le(0.021)
            & table["floor"].sub(row["floor"]).abs().le(0.021)
        ]
        stable = bool(
            row["triggered"] >= 2
            and row["ratio"] > 1.002
            and row["win_delta"] >= -1e-12
            and row["dd_delta"] >= -0.005
            and len(neighbor) >= 3
            and (neighbor["ratio"] > 1.0).mean() >= 2 / 3
            and neighbor["win_delta"].median() >= -1e-12
        )
        if stable:
            score = (
                np.log(float(neighbor["ratio"].median()))
                + 0.7 * float(neighbor["win_delta"].median())
                + 0.15 * min(float(neighbor["dd_delta"].median()), 0.10)
            )
            passing.append({**row.to_dict(), "plateau_score": score})
    if not passing:
        return "baseline", {"test_year": year, "rule": "baseline", "reason": "no_stable_past_plateau"}
    winner = max(passing, key=lambda item: item["plateau_score"])
    return str(winner["rule"]), {"test_year": year, "reason": "past_only_plateau", **winner}


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    base = load()
    prices = core.load_prices(ROOT, set(base["symbol"]))
    rule_list = rules()
    rule_map = {rule.key: rule for rule in rule_list}
    candidates = {rule.key: apply_rule(base, prices, rule) for rule in rule_list}
    years = sorted(pd.to_datetime(base["entry_date"]).dt.year.unique())
    parts = []
    selections = []
    for year in years:
        key, selection = choose(base, candidates, rule_map, int(year))
        source = base if key == "baseline" else candidates[key]
        local = source[pd.to_datetime(source["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_profit_lock_rule"] = key
        parts.append(local)
        selections.append(selection)
    winner = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    base_stats = {period: control.subset_stats(base, period) for period in ("full", "holdout")}
    winner_stats = {period: control.subset_stats(winner, period) for period in ("full", "holdout")}
    payload = {
        "method": "annual past-only C profit-lock plateau selector",
        "candidate_count": len(rule_list),
        "baseline": base_stats,
        "winner": winner_stats,
        "triggers": len(winner),
        "changed_exits": int(winner.get("profit_lock_triggered", pd.Series(dtype=bool)).fillna(False).sum()),
        "full_ratio": winner_stats["full"]["final_value"] / base_stats["full"]["final_value"],
        "holdout_ratio": winner_stats["holdout"]["final_value"] / base_stats["holdout"]["final_value"],
    }
    winner.to_csv(OUT / "c_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
