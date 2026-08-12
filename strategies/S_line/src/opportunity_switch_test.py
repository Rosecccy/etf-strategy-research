from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shadow_model import ShadowResearchEngine


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
BASE = WORKSPACE / "C" / "fit" / "execution_timing_audit" / "s_corrected_trades.csv"
OUT = ROOT / "fit" / "opportunity_switch_test"
DEV = set(range(2016, 2024))
HOLDOUT = {2024, 2025, 2026}
INITIAL = 10_000.0
FEE_RATE = 0.0003
MIN_FEE = 5.0


@dataclass(frozen=True)
class Rule:
    check_every: int
    min_hold: int
    gap: float
    switch_ineligible: bool
    max_switches: int

    @property
    def rule_id(self) -> str:
        return (
            f"e{self.check_every}_h{self.min_hold}_g{self.gap:.2f}_"
            f"i{int(self.switch_ineligible)}_n{self.max_switches}"
        )


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_trades() -> pd.DataFrame:
    frame = pd.read_csv(BASE, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def row_param(row: pd.Series) -> dict:
    return {
        "scope": str(row["scope"]),
        "ma": int(float(row["ma"])),
        "mom": int(float(row["mom"])),
        "model": str(row["model"]),
        "alpha": float(row["alpha"]),
        "min_history": 250,
        "min_amount20": 2_000_000.0,
        "vol_cap": float(row["vol_cap"]),
        "accel_cap": float(row["accel_cap"]),
    }


def close_at(
    engine: ShadowResearchEngine,
    symbol: str,
    date: pd.Timestamp,
) -> float:
    frame = engine.panel[symbol]
    pos = engine.pos_before(frame, date)
    if pos is None:
        raise KeyError((symbol, date))
    return float(frame.iloc[pos]["close"])


def switch_trade(
    engine: ShadowResearchEngine,
    trade: pd.Series,
    rule: Rule,
) -> tuple[dict, list[dict]]:
    item = trade.to_dict()
    if str(trade["source"]) == "主策略":
        item["switches"] = 0
        item["switch_rule"] = rule.rule_id
        return item, []
    param = row_param(trade)
    entry_date = pd.Timestamp(trade["entry_date"])
    exit_date = pd.Timestamp(trade["exit_date"])
    calendar = engine.calendar[
        (engine.calendar >= entry_date) & (engine.calendar <= exit_date)
    ].reset_index(drop=True)
    if len(calendar) < 2:
        item["switches"] = 0
        item["switch_rule"] = rule.rule_id
        return item, []

    current = str(trade["symbol"]).zfill(6)
    current_entry = entry_date
    current_entry_price = close_at(engine, current, current_entry)
    factor = 1.0
    legs: list[dict] = []
    switches = 0
    held = 0
    for index in range(len(calendar) - 1):
        decision_date = pd.Timestamp(calendar.iloc[index])
        if decision_date < current_entry:
            continue
        held += 1
        if held < rule.min_hold or held % rule.check_every != 0:
            continue
        candidates = engine.candidate_list(
            decision_date.value,
            param["scope"],
            param["ma"],
            param["mom"],
            param["model"],
            param["alpha"],
            param["min_history"],
            param["min_amount20"],
            param["vol_cap"],
            param["accel_cap"],
        )
        if not candidates:
            continue
        score_map = {symbol: score for score, symbol, _ in candidates}
        top_score, top_symbol, _ = candidates[0]
        if top_symbol == current:
            continue
        current_score = score_map.get(current)
        should_switch = (
            current_score is not None
            and float(top_score) - float(current_score) >= rule.gap
        )
        if current_score is None and rule.switch_ineligible:
            should_switch = True
        if not should_switch or switches >= rule.max_switches:
            continue
        execute_date = pd.Timestamp(calendar.iloc[index + 1])
        old_exit_price = close_at(engine, current, execute_date)
        new_entry_price = close_at(engine, top_symbol, execute_date)
        leg_return = old_exit_price / current_entry_price - 1.0
        factor *= 1.0 + leg_return
        legs.append(
            {
                "original_entry": entry_date,
                "original_exit": exit_date,
                "decision_date": decision_date,
                "execute_date": execute_date,
                "old_symbol": current,
                "new_symbol": top_symbol,
                "old_score": current_score,
                "new_score": top_score,
                "score_gap": (
                    float(top_score) - float(current_score)
                    if current_score is not None
                    else np.nan
                ),
                "old_leg_return": leg_return,
                "rule_id": rule.rule_id,
            }
        )
        current = top_symbol
        current_entry = execute_date
        current_entry_price = new_entry_price
        switches += 1
        held = 0

    final_price = close_at(engine, current, exit_date)
    factor *= final_price / current_entry_price
    item["ret"] = factor - 1.0
    item["exit_symbol"] = current
    item["switches"] = switches
    item["switch_rule"] = rule.rule_id
    item["entry_close"] = close_at(
        engine, str(trade["symbol"]).zfill(6), entry_date
    )
    item["exit_close"] = final_price
    return item, legs


def apply_rule(
    engine: ShadowResearchEngine,
    frame: pd.DataFrame,
    rule: Rule,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    legs = []
    for _, trade in frame.iterrows():
        item, trade_legs = switch_trade(engine, trade, rule)
        rows.append(item)
        legs.extend(trade_legs)
    return pd.DataFrame(rows), pd.DataFrame(legs)


def metrics(frame: pd.DataFrame, years: set[int]) -> dict:
    local = frame[
        pd.to_datetime(frame["entry_date"]).dt.year.isin(years)
    ].copy()
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual = (
        local.assign(year=local["entry_date"].dt.year)
        .groupby("year")["ret"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    )
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]) if len(equity) else 1000.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_return": float(returns.mean()) if len(returns) else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "losing_years": int((annual < 0).sum()),
        "switches": int(
            pd.to_numeric(local.get("switches", 0), errors="coerce")
            .fillna(0)
            .sum()
        ),
    }


def score(stat: dict) -> float:
    return (
        math.log(max(float(stat["final_1000"]), 1.0) / 1000.0)
        + 0.30 * float(np.nan_to_num(stat["win_rate"], nan=0.0))
        + 0.70 * float(stat["max_drawdown"])
        - 0.05 * int(stat["losing_years"])
        - 0.001 * int(stat["switches"])
    )


def fee(value: float) -> float:
    return max(MIN_FEE, abs(value) * FEE_RATE)


def approximate_net(frame: pd.DataFrame) -> dict:
    cash = INITIAL
    executed = 0
    for _, row in frame.sort_values(["entry_date", "symbol"]).iterrows():
        gross_ret = float(row["ret"])
        switches = int(row.get("switches", 0))
        fee_drag = 2 * (1 + switches) * max(FEE_RATE, MIN_FEE / cash)
        cash *= max(0.0, 1.0 + gross_ret - fee_drag)
        executed += 1 + switches
    return {
        "initial_cash": INITIAL,
        "final_cash": float(cash),
        "net_profit": float(cash - INITIAL),
        "executed_legs": executed,
    }


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    engine = ShadowResearchEngine()
    baseline = load_trades()
    baseline["switches"] = 0
    rules = [
        Rule(check_every, minimum, gap, ineligible, max_switches)
        for check_every in (1, 3, 5, 10)
        for minimum in (3, 5, 10, 15, 20)
        for gap in (0.05, 0.10, 0.20, 0.30, 0.40)
        for ineligible in (False, True)
        for max_switches in (1, 2, 3)
    ]
    rows = []
    accounts: dict[str, pd.DataFrame] = {"baseline": baseline}
    legs_map: dict[str, pd.DataFrame] = {}
    base_dev = metrics(baseline, DEV)
    base_holdout = metrics(baseline, HOLDOUT)
    rows.append(
        {
            "rule_id": "baseline",
            "dev_score": score(base_dev),
            **{f"dev_{key}": value for key, value in base_dev.items()},
            **{
                f"holdout_{key}": value
                for key, value in base_holdout.items()
            },
        }
    )
    for index, rule in enumerate(rules, start=1):
        account, legs = apply_rule(engine, baseline, rule)
        dev = metrics(account, DEV)
        holdout = metrics(account, HOLDOUT)
        rows.append(
            {
                "rule_id": rule.rule_id,
                "check_every": rule.check_every,
                "min_hold": rule.min_hold,
                "gap": rule.gap,
                "switch_ineligible": rule.switch_ineligible,
                "max_switches": rule.max_switches,
                "dev_score": score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{
                    f"holdout_{key}": value
                    for key, value in holdout.items()
                },
            }
        )
        accounts[rule.rule_id] = account
        legs_map[rule.rule_id] = legs
        if index % 100 == 0:
            print(f"[switch {index}/{len(rules)}]")

    table = pd.DataFrame(rows).sort_values(
        ["dev_score", "dev_final_1000", "dev_win_rate"],
        ascending=False,
    )
    winner = table.iloc[0]
    winner_id = str(winner["rule_id"])
    base = table[table["rule_id"].eq("baseline")].iloc[0]
    nearby = table[
        table["dev_final_1000"].ge(float(winner["dev_final_1000"]) * 0.95)
    ]
    passing = nearby[
        nearby["holdout_final_1000"].gt(base["holdout_final_1000"] * 1.03)
        & nearby["holdout_win_rate"].ge(base["holdout_win_rate"])
        & nearby["holdout_max_drawdown"].ge(base["holdout_max_drawdown"])
    ]
    promoted = bool(
        winner_id != "baseline"
        and winner["holdout_final_1000"]
        > base["holdout_final_1000"] * 1.03
        and winner["holdout_win_rate"] >= base["holdout_win_rate"]
        and winner["holdout_max_drawdown"] >= base["holdout_max_drawdown"]
        and len(passing) >= 3
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "S fixed-entry opportunity-cost switching",
        "candidate_rules": int(len(table)),
        "baseline": base.to_dict(),
        "development_winner": winner.to_dict(),
        "nearby_development_rules": int(len(nearby)),
        "passing_neighbors": int(len(passing)),
        "baseline_approx_net": approximate_net(baseline),
        "winner_approx_net": approximate_net(accounts[winner_id]),
        "promoted": promoted,
    }
    table.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    accounts[winner_id].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    legs_map.get(winner_id, pd.DataFrame()).to_csv(
        OUT / "winner_switches.csv", index=False, encoding="utf-8-sig"
    )
    passing.to_csv(
        OUT / "passing_neighbors.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
