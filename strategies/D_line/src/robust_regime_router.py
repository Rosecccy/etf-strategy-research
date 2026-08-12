from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "D" / "out" / "line_specific_gates" / "rolling_line_gates_trades.csv"
PHASE = ROOT / "D" / "out" / "phase_judge" / "daily_phase_scores.csv"
OUT = ROOT / "D" / "out" / "robust_regime_router"
LINES = ("C", "S", "D")


@dataclass(frozen=True)
class Rule:
    key: str
    trend: float
    rebound: float
    panic: float
    order_rebound: str
    order_trend: str
    order_panic: str
    order_neutral: str
    recent_gate: bool
    cash_if_weak: bool


def read_phase() -> pd.DataFrame:
    phase = pd.read_csv(PHASE, encoding="utf-8-sig")
    phase["date"] = pd.to_datetime(phase["date"], errors="coerce")
    return phase[["date", "trend_score", "rebound_score", "panic_score"]].dropna()


def read_panel() -> pd.DataFrame:
    df = pd.read_csv(INPUT, dtype={"symbol": str}, encoding="utf-8-sig")
    for col in ("signal_date", "entry_date", "exit_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("ret", "rank_score", "trend_score", "rebound_score", "panic_score"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["active"] = df["active"].astype(bool)
    df["completed"] = df["completed"].astype(bool)
    df = df[(df["active"]) & (df["completed"]) & df["ret"].notna()].copy()
    df = df.sort_values(["signal_date", "line", "rank_score"], ascending=[True, True, False])
    df = df.drop_duplicates(["signal_date", "line"], keep="first")
    phase = read_phase().rename(columns={"date": "signal_date"})
    cols = ["signal_date", "line", "symbol", "name", "ret", "rank_score",
            "trend_score", "rebound_score", "panic_score"]
    df = df[cols].merge(phase, on="signal_date", how="left", suffixes=("", "_phase"))
    for col in ("trend_score", "rebound_score", "panic_score"):
        df[col] = df[col].fillna(df[f"{col}_phase"])
    df = df.drop(columns=[c for c in df.columns if c.endswith("_phase")])
    return df.dropna(subset=["signal_date", "trend_score", "rebound_score", "panic_score"])


def rules() -> list[Rule]:
    out = []
    order_sets = [
        ("CDS", "SDC", "DCS", "DCS"),
        ("CSD", "SCD", "DCS", "DCS"),
        ("CDS", "SDC", "DCS", "CDS"),
        ("CSD", "SCD", "DCS", "CDS"),
    ]
    for trend, rebound, panic in product((0.50, 0.60), (0.50, 0.60), (0.65, 0.75)):
        for order_r, order_t, order_p, order_n in order_sets:
            # The platform intentionally tests a small set of interpretable
            # priority maps instead of mining every permutation.
            for recent_gate, cash_if_weak in product((False, True), (False, True)):
                key = (
                    f"t{trend:.2f}_r{rebound:.2f}_p{panic:.2f}_"
                    f"r{''.join(order_r)}_t{''.join(order_t)}_p{''.join(order_p)}_"
                    f"g{int(recent_gate)}_cash{int(cash_if_weak)}"
                )
                out.append(Rule(key, trend, rebound, panic,
                                order_r, order_t, order_p, order_n,
                                recent_gate, cash_if_weak))
    return out


def line_map(df: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    return {
        day: dict(zip(g["line"], g["ret"]))
        for day, g in df.groupby("signal_date", sort=True)
    }


def phase_map(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("signal_date", as_index=False).agg(
        trend_score=("trend_score", "first"),
        rebound_score=("rebound_score", "first"),
        panic_score=("panic_score", "first"),
    ).set_index("signal_date")


def order_for(row: pd.Series, rule: Rule) -> str:
    if (
        rule.cash_if_weak
        and row["trend_score"] < rule.trend
        and row["rebound_score"] < rule.rebound
        and row["panic_score"] < rule.panic
    ):
        return ""
    if row["rebound_score"] >= rule.rebound and row["trend_score"] <= rule.trend:
        return rule.order_rebound
    if row["panic_score"] >= rule.panic:
        return rule.order_panic
    if row["trend_score"] >= rule.trend and row["rebound_score"] <= rule.rebound:
        return rule.order_trend
    return rule.order_neutral


def choose_line(
    day_lines: dict[str, float],
    order: str,
    recent_by_line: dict[str, list[float]],
    rule: Rule,
) -> tuple[str, float] | None:
    for line in order:
        if line not in day_lines:
            continue
        if rule.recent_gate:
            past = pd.Series(recent_by_line.get(line, [])[-30:], dtype=float)
            if len(past) < 8:
                continue
            # Recent protection is intentionally simple and causal: reject a line
            # with both negative recent average and sub-1 profit factor.
            wins = past[past > 0].sum()
            losses = -past[past < 0].sum()
            pf = wins / losses if losses > 0 else 99.0
            if float(past.mean()) < 0 and pf < 1.0:
                continue
        return line, float(day_lines[line])
    return None


def apply_rule(df: pd.DataFrame, rule: Rule, train_end: pd.Timestamp | None = None) -> pd.DataFrame:
    dates = sorted(df["signal_date"].unique())
    states = phase_map(df)
    wide = df.pivot_table(index="signal_date", columns="line", values="ret", aggfunc="first")
    rows = []
    recent_by_line = {line: [] for line in LINES}
    for day in dates:
        day = pd.Timestamp(day)
        row = states.loc[day]
        day_series = wide.loc[day] if day in wide.index else pd.Series(dtype=float)
        day_lines = {line: float(day_series[line]) for line in LINES if line in day_series and pd.notna(day_series[line])}
        chosen = choose_line(day_lines, order_for(row, rule), recent_by_line, rule)
        if chosen is None:
            line, ret = "CASH", 0.0
        else:
            line, ret = chosen
        rows.append({
            "date": day,
            "rule": rule.key,
            "line": line,
            "ret": ret,
            "trend_score": row["trend_score"],
            "rebound_score": row["rebound_score"],
            "panic_score": row["panic_score"],
        })
        for line, value in day_lines.items():
            recent_by_line[line].append(float(value))
    return pd.DataFrame(rows)


def stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"days": 0, "active_days": 0, "win_rate": np.nan, "sum_ret": 0.0,
                "compound_ret": 0.0, "max_drawdown": np.nan, "worst_year": np.nan}
    ret = frame["ret"].astype(float).fillna(0.0)
    equity = (1 + ret).cumprod()
    dd = equity / equity.cummax() - 1
    active = frame[frame["line"] != "CASH"]
    annual = frame.groupby(frame["date"].dt.year)["ret"].sum()
    return {
        "days": int(len(frame)),
        "active_days": int(len(active)),
        "active_rate": float(len(active) / len(frame)),
        "win_rate": float((active["ret"] > 0).mean()) if len(active) else np.nan,
        "avg_active_ret": float(active["ret"].mean()) if len(active) else np.nan,
        "sum_ret": float(ret.sum()),
        "compound_ret": float(equity.iloc[-1] - 1),
        "max_drawdown": float(dd.min()),
        "worst_year": float(annual.min()),
        "positive_year_rate": float((annual > 0).mean()) if len(annual) else np.nan,
    }


def selector_score(stat: dict) -> float:
    # Return first, then recent protection, with explicit penalties for drawdown
    # and inactivity. This score is used only inside past-only training windows.
    return (
        stat["sum_ret"]
        + 0.80 * np.nan_to_num(stat["win_rate"], nan=0.0)
        + 0.30 * np.nan_to_num(stat["positive_year_rate"], nan=0.0)
        + 0.50 * np.nan_to_num(stat["max_drawdown"], nan=-1.0)
        + 0.10 * stat["active_rate"]
    )


def rolling(df: pd.DataFrame, candidate_rules: list[Rule], min_years: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(df["signal_date"].dt.year.unique())
    outputs, choices, ranks = [], [], []
    baseline = next(r for r in candidate_rules if r.key.startswith("t0.50_r0.50_p0.65") and not r.recent_gate and not r.cash_if_weak)
    rule_results = {rule.key: apply_rule(df, rule) for rule in candidate_rules}
    for year in years:
        train_year_count = len([y for y in years if y < year])
        if train_year_count < min_years:
            picked = baseline
            picked_frame = rule_results[picked.key]
            train_stat = stats(picked_frame[picked_frame["date"].dt.year < year])
        else:
            rows = []
            for rule in candidate_rules:
                result = rule_results[rule.key]
                result = result[result["date"].dt.year < year]
                st = stats(result)
                st.update({"rule": rule.key, "score": selector_score(st)})
                rows.append(st)
            rank = pd.DataFrame(rows).sort_values(["score", "sum_ret", "win_rate"], ascending=False)
            ranks.append(rank.assign(test_year=year))
            base_row = rank[rank["rule"] == baseline.key].iloc[0]
            eligible = rank[
                (rank["active_rate"] >= max(0.15, float(base_row["active_rate"]) * 0.45))
                & (rank["max_drawdown"] >= float(base_row["max_drawdown"]) - 0.04)
                & (rank["win_rate"] >= float(base_row["win_rate"]) - 0.04)
                & (rank["sum_ret"] >= float(base_row["sum_ret"]) * 0.90)
            ]
            picked_key = str(eligible.iloc[0]["rule"]) if len(eligible) else baseline.key
            picked = next(r for r in candidate_rules if r.key == picked_key)
            train_stat = dict(rank[rank["rule"] == picked_key].iloc[0])
        result = rule_results[picked.key]
        result = result[result["date"].dt.year == year].copy()
        result["test_year"] = year
        outputs.append(result)
        choices.append({
            "test_year": year,
            "rule": picked.key,
            "train_years": int(train_year_count),
            "train_score": train_stat.get("score"),
            "train_sum_ret": train_stat.get("sum_ret"),
            "train_win_rate": train_stat.get("win_rate"),
            "train_max_drawdown": train_stat.get("max_drawdown"),
        })
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(choices), pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_panel()
    df.to_csv(OUT / "active_clean_panel.csv", index=False, encoding="utf-8-sig")
    candidate_rules = rules()
    result, choices, ranks = rolling(df, candidate_rules)
    result.to_csv(OUT / "rolling_router_daily.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "rolling_router_choices.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(OUT / "rolling_router_training_rank.csv", index=False, encoding="utf-8-sig")

    annual = result.groupby("test_year").agg(
        days=("ret", "size"), active_days=("line", lambda s: int((s != "CASH").sum())),
        win_rate=("ret", lambda s: float((s[s != 0] > 0).mean()) if (s != 0).any() else np.nan),
        avg_ret=("ret", "mean"), sum_ret=("ret", "sum"),
    ).reset_index()
    annual.to_csv(OUT / "rolling_router_annual.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(INPUT),
        "data_range": [str(df["signal_date"].min().date()), str(df["signal_date"].max().date())],
        "strict_rolling": "Each test year is selected using completed earlier years only; no future idle or future state data.",
        "candidate_count": len(candidate_rules),
        "overall": stats(result),
        "annual": annual.to_dict("records"),
        "choices": choices.to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
