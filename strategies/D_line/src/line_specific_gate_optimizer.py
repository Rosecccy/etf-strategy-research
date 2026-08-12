from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[2]
TRADES = ROOT / "D" / "out" / "forced_daily_top1" / "forced_top1_trades_2019_2026.csv"
PHASE = ROOT / "D" / "out" / "phase_judge" / "daily_phase_scores.csv"
OUT = ROOT / "D" / "out" / "line_specific_gates"
LINES = ("C", "S", "D")
LOOKBACKS = (252, 504, 99999)


def read_inputs() -> pd.DataFrame:
    trades = pd.read_csv(TRADES, dtype={"symbol": str}, encoding="utf-8-sig")
    for col in ("signal_date", "entry_date", "exit_date"):
        trades[col] = pd.to_datetime(trades[col], errors="coerce")
    trades["ret"] = pd.to_numeric(trades["ret"], errors="coerce")
    trades["completed"] = trades["completed"].astype(bool)
    trades["year"] = trades["signal_date"].dt.year

    phase = pd.read_csv(PHASE, encoding="utf-8-sig")
    phase["date"] = pd.to_datetime(phase["date"], errors="coerce")
    keep = [
        "date",
        "trend_score",
        "rebound_score",
        "panic_score",
        "median_ret20",
        "median_ret60",
        "low_pos_rate",
        "deep_drop_rate",
        "median_dd60",
    ]
    data = trades.merge(phase[keep], left_on="signal_date", right_on="date", how="left")
    data = data.dropna(subset=["trend_score", "rebound_score", "panic_score"]).copy()
    for col in keep[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce").ffill().fillna(0.0)
    return data.sort_values(["line", "signal_date"]).reset_index(drop=True)


def prior_stats(data: pd.DataFrame) -> pd.DataFrame:
    completed = data[data["completed"] & data["ret"].notna() & data["exit_date"].notna()].copy()
    completed["ret"] = completed["ret"].astype(float)
    rows = []
    for _, row in data.iterrows():
        item = {"_row_id": int(row.name)}
        date = pd.Timestamp(row["signal_date"])
        for lb in LOOKBACKS:
            start = date - pd.Timedelta(days=lb) if lb < 90000 else pd.Timestamp("1900-01-01")
            known = completed[
                (completed["exit_date"] < date)
                & (completed["exit_date"] >= start)
                & (completed["line"].eq(row["line"]))
            ]
            sym = known[known["symbol"].eq(row["symbol"])]
            for prefix, part in ((f"sym_{lb}", sym), (f"line_{lb}", known)):
                vals = part["ret"].astype(float)
                item[f"{prefix}_n"] = int(len(vals))
                item[f"{prefix}_avg"] = float(vals.mean()) if len(vals) else np.nan
                item[f"{prefix}_win"] = float((vals > 0).mean()) if len(vals) else np.nan
                wins = vals[vals > 0]
                losses = vals[vals < 0]
                item[f"{prefix}_pf"] = float(wins.sum() / abs(losses.sum())) if len(wins) and len(losses) else np.nan
        rows.append(item)
    stats = pd.DataFrame(rows).set_index("_row_id")
    return pd.concat([data, stats], axis=1)


def metrics(frame: pd.DataFrame) -> dict:
    active = frame[frame["active"]].copy()
    if active.empty:
        return {
            "signals": int(len(frame)),
            "active": 0,
            "active_rate": 0.0,
            "win": np.nan,
            "avg": np.nan,
            "sum": 0.0,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "payoff": np.nan,
            "pf": np.nan,
        }
    ret = active["ret"].astype(float)
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    return {
        "signals": int(len(frame)),
        "active": int(len(active)),
        "active_rate": float(len(active) / len(frame)) if len(frame) else 0.0,
        "win": float((ret > 0).mean()),
        "avg": float(ret.mean()),
        "sum": float(ret.sum()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan,
        "pf": float(wins.sum() / abs(losses.sum())) if len(wins) and len(losses) else np.nan,
    }


def score(stat: dict) -> float:
    # Keep trigger rate in the objective, but do not let low-quality high-frequency signals dominate.
    return float(
        stat["sum"]
        + 0.75 * np.nan_to_num(stat["win"], nan=0.0)
        + 0.25 * np.nan_to_num(stat["pf"], nan=0.0)
        + 0.10 * stat["active_rate"]
    )


def line_grid(line: str) -> list[dict]:
    if line == "C":
        return [
            {
                "trend_min": 0.0,
                "trend_max": trend_max,
                "rebound_min": rebound_min,
                "rebound_max": 1.0,
                "panic_min": panic_min,
                "panic_max": 1.0,
                "m20_min": -9.0,
                "m20_max": m20_max,
                "lookback": lookback,
                "min_n": min_n,
                "min_avg": min_avg,
                "min_win": min_win,
                "min_pf": min_pf,
            }
            for trend_max, rebound_min, panic_min, m20_max, lookback, min_n, min_avg, min_win, min_pf in product(
                (0.35, 0.50, 0.65, 0.80, 1.0),
                (0.0, 0.30, 0.45, 0.55),
                (0.0, 0.40, 0.50, 0.60),
                (9.0, 0.0),
                LOOKBACKS,
                (0, 5, 15),
                (-0.05, -0.01, 0.0),
                (0.0, 0.45, 0.52),
                (0.0, 1.0, 1.3),
            )
        ]
    if line == "S":
        return [
            {
                "trend_min": trend_min,
                "trend_max": 1.0,
                "rebound_min": 0.0,
                "rebound_max": rebound_max,
                "panic_min": 0.0,
                "panic_max": panic_max,
                "m20_min": m20_min,
                "m20_max": 9.0,
                "lookback": lookback,
                "min_n": min_n,
                "min_avg": min_avg,
                "min_win": min_win,
                "min_pf": min_pf,
            }
            for trend_min, rebound_max, panic_max, m20_min, lookback, min_n, min_avg, min_win, min_pf in product(
                (0.0, 0.35, 0.50, 0.65, 0.75),
                (1.0, 0.65, 0.50, 0.40),
                (1.0, 0.70, 0.55),
                (-9.0, 0.0),
                LOOKBACKS,
                (0, 5, 15),
                (-0.05, -0.01, 0.0),
                (0.0, 0.45, 0.52),
                (0.0, 1.0, 1.3),
            )
        ]
    return [
        {
            "trend_min": 0.0,
            "trend_max": trend_max,
            "rebound_min": rebound_min,
            "rebound_max": 1.0,
            "panic_min": panic_min,
            "panic_max": 1.0,
            "m20_min": -9.0,
            "m20_max": m20_max,
            "lookback": lookback,
            "min_n": min_n,
            "min_avg": min_avg,
            "min_win": min_win,
            "min_pf": min_pf,
        }
        for trend_max, rebound_min, panic_min, m20_max, lookback, min_n, min_avg, min_win, min_pf in product(
            (0.30, 0.45, 0.60, 0.75, 1.0),
            (0.0, 0.30, 0.45, 0.55),
            (0.0, 0.35, 0.45, 0.55, 0.65),
            (9.0, 0.0, -0.01),
            LOOKBACKS,
            (0, 5, 15),
            (-0.05, -0.01, 0.0),
            (0.0, 0.45, 0.52),
            (0.0, 1.0, 1.3),
        )
    ]


def apply_params(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    lb = int(params["lookback"])
    n_col = f"sym_{lb}_n"
    avg_col = f"sym_{lb}_avg"
    win_col = f"sym_{lb}_win"
    pf_col = f"sym_{lb}_pf"
    market = (
        data["trend_score"].between(params["trend_min"], params["trend_max"], inclusive="both")
        & data["rebound_score"].between(params["rebound_min"], params["rebound_max"], inclusive="both")
        & data["panic_score"].between(params["panic_min"], params["panic_max"], inclusive="both")
        & data["median_ret20"].between(params["m20_min"], params["m20_max"], inclusive="both")
    )
    enough = data[n_col].fillna(0) >= int(params["min_n"])
    history_ok = (
        (int(params["min_n"]) == 0)
        | (
            enough
            & (data[avg_col].fillna(-99) >= float(params["min_avg"]))
            & (data[win_col].fillna(-99) >= float(params["min_win"]))
            & (data[pf_col].fillna(-99) >= float(params["min_pf"]))
        )
    )
    out = data.copy()
    out["active"] = market & history_ok & out["completed"].astype(bool) & out["ret"].notna()
    return out


def rank_line(data: pd.DataFrame, line: str, train_mask: pd.Series | None = None) -> pd.DataFrame:
    local = data[data["line"].eq(line)].copy()
    if train_mask is not None:
        local = local[train_mask.loc[local.index]].copy()
    rows = []
    for i, params in enumerate(line_grid(line)):
        result = apply_params(local, params)
        stat = metrics(result)
        if stat["active"] < 30:
            continue
        rows.append({"line": line, "param_id": f"{line}{i:05d}", "score": score(stat), **params, **stat})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "sum", "pf", "win"], ascending=False)


def rolling_select(
    data: pd.DataFrame,
    full_ranks: dict[str, pd.DataFrame],
    candidate_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = []
    choices = []
    years = sorted(data["year"].dropna().astype(int).unique())
    for line in LINES:
        local = data[data["line"].eq(line)].copy()
        for year in years:
            test = local[local["year"].eq(year)].copy()
            if test.empty:
                continue
            train = local[local["signal_date"] < pd.Timestamp(year=year, month=1, day=1)].copy()
            candidates = full_ranks[line].head(candidate_limit).copy()
            scored = []
            for _, candidate in candidates.iterrows():
                params = {k: candidate[k] for k in line_grid(line)[0].keys()}
                result = apply_params(train, params)
                stat = metrics(result)
                if stat["active"] < 10:
                    continue
                scored.append({"param_id": candidate["param_id"], "score": score(stat), **params, **stat})
            ranked = pd.DataFrame(scored).sort_values(["score", "sum", "pf", "win"], ascending=False) if scored else pd.DataFrame()
            if ranked.empty:
                # First years may not have enough known trades; use best full-grid shape as a cold-start candidate.
                selected = full_ranks[line].iloc[0].to_dict()
                cold_start = True
            else:
                selected = ranked.iloc[0].to_dict()
                cold_start = False
            params = {k: selected[k] for k in line_grid(line)[0].keys()}
            result = apply_params(test, params)
            result["selected_param_id"] = selected["param_id"]
            result["cold_start"] = cold_start
            outputs.append(result)
            choices.append(
                {
                    "line": line,
                    "year": year,
                    "selected_param_id": selected["param_id"],
                    "cold_start": cold_start,
                    "train_score": selected.get("score"),
                    "train_sum": selected.get("sum"),
                    "train_win": selected.get("win"),
                    "train_pf": selected.get("pf"),
                    "candidate_source": f"full_rank_top_{candidate_limit}",
                    **params,
                }
            )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(choices)


def summarize_active(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in data.groupby(group_cols, dropna=False):
        stat = metrics(group)
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(group_cols, key)), **stat})
    return pd.DataFrame(rows)


def daily_combo(active: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in active.groupby("signal_date"):
        picked = group[group["active"]].copy()
        if picked.empty:
            rows.append({"date": date, "lines": "CASH", "ret": 0.0})
            continue
        rows.append({"date": date, "lines": "+".join(sorted(picked["line"].unique())), "ret": float(picked["ret"].mean())})
    return pd.DataFrame(rows).sort_values("date")


def main() -> None:
    configure_stdout()
    parser = ArgumentParser()
    parser.add_argument("--candidate-limit", type=int, default=800)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "candidate_with_prior_stats.csv"
    if cache.exists() and not args.refresh:
        data = pd.read_csv(cache, dtype={"symbol": str}, encoding="utf-8-sig")
        for col in ("signal_date", "entry_date", "exit_date", "date"):
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors="coerce")
        data["completed"] = data["completed"].astype(bool)
        data["ret"] = pd.to_numeric(data["ret"], errors="coerce")
    else:
        data = prior_stats(read_inputs())
        data.to_csv(cache, index=False, encoding="utf-8-sig")

    full_ranks = {}
    for line in LINES:
        rank_path = OUT / f"{line}_gate_rank_full.csv"
        if rank_path.exists() and not args.refresh:
            ranked = pd.read_csv(rank_path, encoding="utf-8-sig")
        else:
            ranked = rank_line(data, line)
            ranked.to_csv(rank_path, index=False, encoding="utf-8-sig")
        full_ranks[line] = ranked

    rolling, choices = rolling_select(data, full_ranks, args.candidate_limit)
    rolling.to_csv(OUT / "rolling_line_gates_trades.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "rolling_line_gates_choices.csv", index=False, encoding="utf-8-sig")
    by_line = summarize_active(rolling, ["line"])
    by_line_year = summarize_active(rolling, ["line", "year"])
    by_line.to_csv(OUT / "rolling_line_summary.csv", index=False, encoding="utf-8-sig")
    by_line_year.to_csv(OUT / "rolling_line_year_summary.csv", index=False, encoding="utf-8-sig")

    combo = daily_combo(rolling)
    combo["active"] = combo["lines"].ne("CASH")
    combo_stat = metrics(combo.rename(columns={"date": "signal_date"}))
    combo.to_csv(OUT / "rolling_combo_daily.csv", index=False, encoding="utf-8-sig")

    summary = {
        "full_best": {line: full_ranks[line].head(10).to_dict("records") for line in LINES},
        "rolling_by_line": by_line.to_dict("records"),
        "rolling_by_line_year": by_line_year.to_dict("records"),
        "rolling_combo": combo_stat,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
