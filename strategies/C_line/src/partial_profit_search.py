from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "partial_profit"
MIN_TRAIN_TRADES = 12


@dataclass(frozen=True)
class PartialRule:
    target: float
    fraction: float

    @property
    def key(self) -> str:
        return f"partial:t{self.target:.3f}:f{self.fraction:.2f}"


def commission(notional: float) -> float:
    return max(core.MIN_COMMISSION, abs(notional) * core.COMMISSION_RATE)


def apply_rule(
    frame: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    rule: PartialRule | None,
) -> pd.DataFrame:
    result = frame.copy()
    result["partial_rule"] = "baseline" if rule is None else rule.key
    result["partial_triggered"] = False
    result["partial_date"] = pd.NaT
    result["partial_price"] = np.nan
    result["partial_fraction"] = 0.0
    if rule is None:
        return result
    for index, trade in result.iterrows():
        data = prices[str(trade["symbol"])]
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date_test"])
        target = float(trade["entry_close"]) * (1.0 + rule.target)
        path = data[data["date"].gt(entry_date) & data["date"].lt(exit_date)]
        hits = path[path["high"].ge(target)]
        if hits.empty:
            continue
        hit = hits.iloc[0]
        result.at[index, "partial_triggered"] = True
        result.at[index, "partial_date"] = pd.Timestamp(hit["date"])
        result.at[index, "partial_price"] = max(target, float(hit["open"]))
        result.at[index, "partial_fraction"] = rule.fraction
    return result


def simulate(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    cash = core.INITIAL_CAPITAL
    rows = []
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        before = cash
        entry = float(trade["entry_close"])
        final_price = float(trade["exit_close_test"])
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + commission(quantity * entry) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy_value = quantity * entry
        cash -= buy_value + commission(buy_value)
        partial_quantity = 0
        if bool(trade.get("partial_triggered", False)):
            partial_quantity = math.floor(quantity * float(trade["partial_fraction"]) / 100) * 100
            partial_quantity = min(partial_quantity, quantity - 100) if quantity >= 200 else 0
            if partial_quantity >= 100:
                partial_value = partial_quantity * float(trade["partial_price"])
                cash += partial_value - commission(partial_value)
        remaining = quantity - partial_quantity
        final_value = remaining * final_price
        if bool(trade["open_mark_test"]):
            after = cash + final_value - commission(final_value)
        else:
            cash += final_value - commission(final_value)
            after = cash
        item = trade.to_dict()
        item.update(
            {
                "quantity_partial_test": int(quantity),
                "partial_quantity_test": int(partial_quantity),
                "cash_before_partial_test": float(before),
                "cash_after_partial_test": float(after),
                "account_return_partial_test": float(after / before - 1.0),
            }
        )
        rows.append(item)
    detail = pd.DataFrame(rows)
    final_value = float(detail.iloc[-1]["cash_after_partial_test"]) if len(detail) else core.INITIAL_CAPITAL
    returns = detail["account_return_partial_test"].astype(float)
    equity = core.INITIAL_CAPITAL * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    detail["year_partial_test"] = pd.to_datetime(detail["entry_date"]).dt.year
    annual = (
        detail.groupby("year_partial_test")["account_return_partial_test"]
        .apply(lambda values: float(np.prod(1.0 + values.astype(float)) - 1.0))
        .reindex(range(int(detail["year_partial_test"].min()), 2027), fill_value=0.0)
    )
    closed = detail[~detail["open_mark_test"].astype(bool)]
    stats = {
        "trades": int(len(detail)),
        "closed_trades": int(len(closed)),
        "final_value": final_value,
        "win_rate": float((closed["account_return_partial_test"] > 0).mean()) if len(closed) else 0.0,
        "avg_annual_return": float(annual.mean()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "partial_trades": int(detail["partial_quantity_test"].gt(0).sum()),
    }
    return stats, detail


def subset_stats(frame: pd.DataFrame, period: str) -> dict:
    years = pd.to_datetime(frame["entry_date"]).dt.year
    if period == "dev":
        local = frame[years.le(2023)].copy()
    elif period == "holdout":
        local = frame[years.ge(2024)].copy()
    else:
        local = frame.copy()
    return simulate(local)[0]


def training_profile(frame: pd.DataFrame, years: list[int]) -> dict[int, dict]:
    _, detail = simulate(frame)
    detail_year = pd.to_datetime(detail["entry_date"]).dt.year
    result = {}
    for year in years:
        local = detail[detail_year.lt(year)]
        if local.empty:
            result[year] = {
                "trades": 0,
                "final_value": core.INITIAL_CAPITAL,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "partial_trades": 0,
            }
            continue
        returns = local["account_return_partial_test"].astype(float)
        equity = core.INITIAL_CAPITAL * (1.0 + returns).cumprod()
        closed = local[~local["open_mark_test"].astype(bool)]
        result[year] = {
            "trades": int(len(local)),
            "final_value": float(local.iloc[-1]["cash_after_partial_test"]),
            "win_rate": float((closed["account_return_partial_test"] > 0).mean()),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "partial_trades": int(local["partial_quantity_test"].gt(0).sum()),
        }
    return result


def choose(profiles: dict[str, dict[int, dict]], rules_by_key: dict[str, PartialRule], year: int) -> tuple[str, dict]:
    base = profiles["baseline"][year]
    rows = []
    for name, profile in profiles.items():
        if name == "baseline":
            continue
        current = profile[year]
        ratio = current["final_value"] / base["final_value"]
        win_delta = current["win_rate"] - base["win_rate"]
        dd_delta = current["max_drawdown"] - base["max_drawdown"]
        passes = bool(
            current["trades"] >= MIN_TRAIN_TRADES
            and current["partial_trades"] >= 2
            and ratio > 1.0
            and win_delta >= 0
            and dd_delta >= -0.005
        )
        rows.append(
            {
                "year": year,
                "rule": name,
                "target": rules_by_key[name].target,
                "fraction": rules_by_key[name].fraction,
                "ratio": ratio,
                "win_delta": win_delta,
                "passes": passes,
                "score": np.log(max(ratio, 1e-12)) + 0.8 * win_delta + 0.2 * min(dd_delta, 0.2),
            }
        )
    passing = [row for row in rows if row["passes"]]
    if len(passing) < 2:
        return "baseline", {"year": year, "rule": "baseline"}
    winner = max(passing, key=lambda row: row["score"])
    return str(winner["rule"]), winner


def validate_line(spec: core.LineSpec, path: Path, candidate_rules: list[PartialRule]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in ("entry_date", "exit_date_test", "control_signal_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    prices = core.load_prices(spec.project, set(frame["symbol"]))
    frames = {"baseline": apply_rule(frame, prices, None)}
    rules_by_key = {}
    for rule in candidate_rules:
        frames[rule.key] = apply_rule(frame, prices, rule)
        rules_by_key[rule.key] = rule
    years = sorted(pd.to_datetime(frame["entry_date"]).dt.year.unique())
    profiles = {name: training_profile(candidate, years) for name, candidate in frames.items()}
    parts = []
    selections = []
    for year in years:
        rule, selection = choose(profiles, rules_by_key, int(year))
        local = frames[rule][pd.to_datetime(frames[rule]["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_partial_rule"] = rule
        parts.append(local)
        selections.append(selection)
    combined = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    baseline = {period: subset_stats(frames["baseline"], period) for period in ("full", "dev", "holdout")}
    winner = {period: subset_stats(combined, period) for period in ("full", "dev", "holdout")}
    return {
        "line": spec.key,
        "baseline": baseline,
        "winner": winner,
        "full_ratio": winner["full"]["final_value"] / baseline["full"]["final_value"],
        "full_win_delta": winner["full"]["win_rate"] - baseline["full"]["win_rate"],
        "holdout_ratio": winner["holdout"]["final_value"] / baseline["holdout"]["final_value"],
        "holdout_win_delta": winner["holdout"]["win_rate"] - baseline["holdout"]["win_rate"],
        "selected_years": int(sum(row["rule"] != "baseline" for row in selections)),
    }, combined, pd.DataFrame(selections)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        line: ROOT / "fit" / "rolling_upgrades" / f"{line.lower()}_trades.csv"
        for line in ("C", "S", "D")
    }
    specs = {spec.key: spec for spec in core.SPECS}
    candidate_rules = [
        PartialRule(target, fraction)
        for target in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)
        for fraction in (0.25, 0.50, 0.75)
    ]
    results = []
    for line in ("C", "S", "D"):
        payload, trades, selections = validate_line(specs[line], paths[line], candidate_rules)
        results.append(payload)
        trades.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        selections.to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window partial-profit insurance",
        "fees": "0.03% per execution, minimum CNY 5, 100-share lots",
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
