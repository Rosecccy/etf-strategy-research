from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as stats_mod
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "loss_recovery_grace"
SOURCES = {
    "C": ROOT / "fit" / "rolling_upgrades" / "c_trades.csv",
    "S": ROOT / "fit" / "trend_extension" / "s_trades.csv",
    "D": ROOT / "fit" / "novel_path_exits" / "d_trades.csv",
}
PROJECTS = {"C": ROOT, "S": ROOT.parent / "S", "D": ROOT}


@dataclass(frozen=True)
class Rule:
    momentum_days: int
    momentum_min: float
    ma_days: int
    grace_days: int
    target: float

    @property
    def key(self) -> str:
        return (
            f"recovery:m{self.momentum_days}:{self.momentum_min:+.3f}:"
            f"ma{self.ma_days}:g{self.grace_days}:t{self.target:.3f}"
        )


def rules() -> list[Rule]:
    return [
        Rule(momentum_days, momentum_min, ma_days, grace_days, target)
        for momentum_days in (5, 10)
        for momentum_min in (0.00, 0.02)
        for ma_days in (5, 10)
        for grace_days in (5, 10, 15)
        for target in (0.005, 0.01)
    ]


def load(line: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCES[line], encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in ("entry_date", "exit_date_test"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def apply_rule(
    frame: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    rule: Rule,
) -> pd.DataFrame:
    ordered = frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    next_entries = list(pd.to_datetime(ordered["entry_date"]).shift(-1))
    rows = []
    for index, trade in ordered.iterrows():
        item = trade.to_dict()
        item["recovery_rule"] = rule.key
        item["recovery_triggered"] = False
        item["recovery_target_hit"] = False
        if bool(trade["open_mark_test"]):
            rows.append(item)
            continue
        entry_price = float(trade["entry_close"])
        exit_price = float(trade["exit_close_test"])
        if exit_price >= entry_price:
            rows.append(item)
            continue
        data = prices[str(trade["symbol"])]
        dates = pd.DatetimeIndex(data["date"])
        base_pos = int(dates.searchsorted(pd.Timestamp(trade["exit_date_test"]), side="left"))
        if base_pos <= max(rule.momentum_days, rule.ma_days) or base_pos >= len(data):
            rows.append(item)
            continue
        close = data["close"].to_numpy(dtype=float)
        momentum = close[base_pos] / close[base_pos - rule.momentum_days] - 1.0
        ma_value = float(np.mean(close[base_pos - rule.ma_days + 1:base_pos + 1]))
        if momentum < rule.momentum_min or close[base_pos] < ma_value:
            rows.append(item)
            continue
        latest = min(base_pos + rule.grace_days, len(data) - 1)
        next_entry = next_entries[index]
        if pd.notna(next_entry):
            next_pos = int(dates.searchsorted(pd.Timestamp(next_entry), side="left"))
            latest = min(latest, next_pos)
        if latest <= base_pos:
            rows.append(item)
            continue
        target = entry_price * (1.0 + rule.target)
        chosen = latest
        chosen_price = float(data.iloc[chosen]["close"])
        hit_target = False
        for pos in range(base_pos + 1, latest + 1):
            row = data.iloc[pos]
            if float(row["high"]) >= target:
                chosen = pos
                chosen_price = max(target, float(row["open"]))
                hit_target = True
                break
        item["exit_date_test"] = pd.Timestamp(data.iloc[chosen]["date"])
        item["exit_close_test"] = chosen_price
        item["exit_reason_test"] = rule.key + ("|target" if hit_target else "|timeout")
        item["gross_return_test"] = chosen_price / entry_price - 1.0
        item["recovery_triggered"] = True
        item["recovery_target_hit"] = hit_target
        rows.append(item)
    return pd.DataFrame(rows)


def profile(frame: pd.DataFrame, years: list[int]) -> dict[int, dict]:
    result = {}
    entry_year = pd.to_datetime(frame["entry_date"]).dt.year
    for year in years:
        local = frame[entry_year.lt(year)].copy()
        if local.empty:
            result[year] = {
                "trades": 0,
                "closed_trades": 0,
                "final_value": core.INITIAL_CAPITAL,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
            }
        else:
            result[year] = core.simulate_account(local)[0]
        result[year]["triggered"] = int(local.get("recovery_triggered", pd.Series(dtype=bool)).fillna(False).sum())
    return result


def select(
    base_profile: dict[int, dict],
    profiles: dict[str, dict[int, dict]],
    rule_map: dict[str, Rule],
    year: int,
) -> tuple[str, dict]:
    base = base_profile[year]
    if base["trades"] < 12:
        return "baseline", {"test_year": year, "rule": "baseline", "reason": "insufficient_history"}
    rows = []
    for key, candidate_profile in profiles.items():
        current = candidate_profile[year]
        rows.append(
            {
                "rule": key,
                "momentum_days": rule_map[key].momentum_days,
                "momentum_min": rule_map[key].momentum_min,
                "ma_days": rule_map[key].ma_days,
                "grace_days": rule_map[key].grace_days,
                "target": rule_map[key].target,
                "ratio": current["final_value"] / base["final_value"],
                "win_delta": current["win_rate"] - base["win_rate"],
                "dd_delta": current["max_drawdown"] - base["max_drawdown"],
                "triggered": current["triggered"],
            }
        )
    table = pd.DataFrame(rows)
    passing = []
    for _, row in table.iterrows():
        neighbors = table[
            table["momentum_days"].eq(row["momentum_days"])
            & table["ma_days"].eq(row["ma_days"])
            & table["grace_days"].sub(row["grace_days"]).abs().le(5)
        ]
        stable = bool(
            row["triggered"] >= 2
            and row["ratio"] > 1.002
            and row["win_delta"] >= -1e-12
            and row["dd_delta"] >= -0.005
            and len(neighbors) >= 4
            and (neighbors["ratio"] > 1.0).mean() >= 0.60
            and neighbors["win_delta"].median() >= -1e-12
        )
        if stable:
            score = (
                np.log(float(neighbors["ratio"].median()))
                + 0.8 * float(neighbors["win_delta"].median())
                + 0.15 * min(float(neighbors["dd_delta"].median()), 0.1)
            )
            passing.append({**row.to_dict(), "plateau_score": score})
    if not passing:
        return "baseline", {"test_year": year, "rule": "baseline", "reason": "no_stable_past_plateau"}
    winner = max(passing, key=lambda value: value["plateau_score"])
    return str(winner["rule"]), {"test_year": year, "reason": "past_only_plateau", **winner}


def evaluate(line: str) -> dict:
    base = load(line)
    prices = core.load_prices(PROJECTS[line], set(base["symbol"]))
    rule_list = rules()
    rule_map = {rule.key: rule for rule in rule_list}
    candidates = {rule.key: apply_rule(base, prices, rule) for rule in rule_list}
    years = sorted(pd.to_datetime(base["entry_date"]).dt.year.unique())
    base_profile = profile(base, years)
    profiles = {key: profile(frame, years) for key, frame in candidates.items()}
    parts = []
    selections = []
    for year in years:
        key, selection = select(base_profile, profiles, rule_map, int(year))
        source = base if key == "baseline" else candidates[key]
        local = source[pd.to_datetime(source["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_recovery_rule"] = key
        parts.append(local)
        selections.append(selection)
    winner = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    baseline = {period: stats_mod.subset_stats(base, period) for period in ("full", "holdout")}
    current = {period: stats_mod.subset_stats(winner, period) for period in ("full", "holdout")}
    winner.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    return {
        "line": line,
        "baseline": baseline,
        "winner": current,
        "triggers": len(winner),
        "recovery_trades": int(winner.get("recovery_triggered", pd.Series(dtype=bool)).fillna(False).sum()),
        "full_ratio": current["full"]["final_value"] / baseline["full"]["final_value"],
        "holdout_ratio": current["holdout"]["final_value"] / baseline["holdout"]["final_value"],
    }


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "annual past-only loss-recovery grace selector",
        "constraint": "all entries and trade count retained; only losing exits may be delayed",
        "results": [evaluate(line) for line in ("C", "S", "D")],
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
