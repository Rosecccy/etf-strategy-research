from __future__ import annotations

import json
import sys
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
OUT = ROOT / "D" / "out" / "phase_judge" / "expanded_tests"
TRADES = ROOT / "D" / "out" / "forced_daily_top1" / "forced_top1_trades_2019_2026.csv"
PHASE = ROOT / "D" / "out" / "phase_judge" / "daily_phase_scores.csv"
LINES = ("C", "S", "D")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(TRADES, dtype={"symbol": str}, encoding="utf-8-sig")
    for col in ("signal_date", "entry_date", "exit_date"):
        trades[col] = pd.to_datetime(trades[col], errors="coerce")
    trades["completed"] = trades["completed"].astype(bool)
    trades["ret"] = pd.to_numeric(trades["ret"], errors="coerce")
    trades["ret0"] = np.where(trades["completed"] & trades["ret"].notna(), trades["ret"], 0.0)

    daily = trades.pivot_table(index="signal_date", columns="line", values="ret0", aggfunc="sum").reset_index()
    for line in LINES:
        if line not in daily.columns:
            daily[line] = 0.0
    daily = daily.rename(columns={"signal_date": "date"})[["date", *LINES]].sort_values("date")

    phase = pd.read_csv(PHASE, encoding="utf-8-sig")
    phase["date"] = pd.to_datetime(phase["date"], errors="coerce")
    for col in ("trend_score", "rebound_score", "panic_score"):
        phase[col] = pd.to_numeric(phase[col], errors="coerce").ffill().fillna(0.0)
    daily = daily.merge(phase[["date", "trend_score", "rebound_score", "panic_score"]], on="date", how="left")
    daily = daily.dropna(subset=["trend_score", "rebound_score", "panic_score"]).reset_index(drop=True)
    return trades, daily


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}
    ret = pd.to_numeric(frame["ret"], errors="coerce").fillna(0.0)
    active = frame[frame["lines"].ne("CASH")]
    years = max((frame["date"].max() - frame["date"].min()).days / 365.25, 1e-9)
    equity = (1.0 + ret).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "days": int(len(frame)),
        "active_days": int(len(active)),
        "active_rate": float(len(active) / len(frame)),
        "active_win_rate": float((active["ret"] > 0).mean()) if len(active) else np.nan,
        "avg_active_ret": float(active["ret"].mean()) if len(active) else np.nan,
        "sum_ret": float(ret.sum()),
        "annualized_sum_ret": float(ret.sum() / years),
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
    }


def phase_order(row: pd.Series) -> list[str]:
    scores = {
        "C": float(row["rebound_score"]),
        "S": float(row["trend_score"]),
        "D": float(row["panic_score"]),
    }
    return sorted(scores, key=scores.get, reverse=True)


def build_stats_cache(trades: pd.DataFrame, daily: pd.DataFrame, lookbacks: tuple[int, ...]) -> dict[int, pd.DataFrame]:
    completed = trades[trades["completed"] & trades["ret"].notna() & trades["exit_date"].notna()].copy()
    completed["ret"] = completed["ret"].astype(float)
    dates = pd.to_datetime(daily["date"]).drop_duplicates().sort_values().tolist()
    cache: dict[int, pd.DataFrame] = {}
    for lookback in lookbacks:
        rows = []
        for date in dates:
            start = date - pd.Timedelta(days=lookback)
            known = completed[(completed["exit_date"] < date) & (completed["exit_date"] >= start)]
            row = {"date": date}
            for line in LINES:
                vals = known.loc[known["line"].eq(line), "ret"]
                row[f"{line}_n"] = int(len(vals))
                row[f"{line}_avg"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{line}_win"] = float((vals > 0).mean()) if len(vals) else np.nan
            rows.append(row)
        cache[lookback] = pd.DataFrame(rows).set_index("date")
    return cache


def stats_from_cache(cache: dict[int, pd.DataFrame], date: pd.Timestamp, lookback: int) -> dict[str, dict]:
    row = cache[lookback].loc[date]
    return {
        line: {
            "n": int(row[f"{line}_n"]),
            "avg": float(row[f"{line}_avg"]) if pd.notna(row[f"{line}_avg"]) else np.nan,
            "win": float(row[f"{line}_win"]) if pd.notna(row[f"{line}_win"]) else np.nan,
        }
        for line in LINES
    }


def choose_line(
    row: pd.Series,
    stats: dict[str, dict],
    mode: str,
    min_n: int,
    min_avg: float,
    min_win: float,
) -> tuple[str, str]:
    if mode == "recent_best":
        ranked = sorted(
            LINES,
            key=lambda line: (
                stats[line]["n"] >= min_n,
                -999 if np.isnan(stats[line]["avg"]) else stats[line]["avg"],
                -999 if np.isnan(stats[line]["win"]) else stats[line]["win"],
            ),
            reverse=True,
        )
    elif mode == "phase_then_guard":
        ranked = phase_order(row)
    elif mode == "blend_phase_recent":
        phase_rank = {line: rank for rank, line in enumerate(phase_order(row))}
        ranked = sorted(
            LINES,
            key=lambda line: (
                (0 if np.isnan(stats[line]["avg"]) else stats[line]["avg"])
                + 0.25 * (0 if np.isnan(stats[line]["win"]) else stats[line]["win"])
                - 0.03 * phase_rank[line]
            ),
            reverse=True,
        )
    else:
        raise ValueError(mode)

    for line in ranked:
        stat = stats[line]
        if stat["n"] < min_n:
            continue
        if np.isnan(stat["avg"]) or stat["avg"] < min_avg:
            continue
        if np.isnan(stat["win"]) or stat["win"] < min_win:
            continue
        return line, f"{mode};n={stat['n']};avg={stat['avg']:.4f};win={stat['win']:.3f}"
    return "CASH", f"{mode};no_line_passed"


def run_policy(daily: pd.DataFrame, stats_cache: dict[int, pd.DataFrame], params: dict) -> pd.DataFrame:
    rows = []
    for _, row in daily.iterrows():
        date = pd.Timestamp(row["date"])
        if date.year < int(params["start_year"]):
            line = "CASH"
            detail = "before_start_year"
        else:
            stats = stats_from_cache(stats_cache, date, int(params["lookback"]))
            line, detail = choose_line(
                row,
                stats,
                str(params["mode"]),
                int(params["min_n"]),
                float(params["min_avg"]),
                float(params["min_win"]),
            )
        ret = 0.0 if line == "CASH" else float(row[line])
        rows.append({"date": date, "lines": line, "ret": ret, "detail": detail, **params})
    return pd.DataFrame(rows)


def grid_search(trades: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    outputs = []
    grid = [
        {
            "mode": mode,
            "lookback": lookback,
            "min_n": min_n,
            "min_avg": min_avg,
            "min_win": min_win,
            "start_year": 2020,
        }
        for mode, lookback, min_n, min_avg, min_win in product(
            ("recent_best", "phase_then_guard", "blend_phase_recent"),
            (60, 120, 180, 252, 504),
            (3, 5, 10, 20),
            (-0.03, -0.01, 0.0, 0.01),
            (0.0, 0.45, 0.50, 0.55),
        )
    ]
    stats_cache = build_stats_cache(trades, daily, (60, 120, 180, 252, 504))
    for index, params in enumerate(grid):
        result = run_policy(daily, stats_cache, params)
        stat = metrics(result)
        score = stat["sum_ret"] + 0.5 * np.nan_to_num(stat["active_win_rate"], nan=0.0) + 0.1 * stat["active_rate"]
        rows.append({"policy_id": f"P{index:04d}", "score": score, **params, **stat})
        if index % 250 == 0:
            outputs.append(result.assign(policy_id=f"P{index:04d}"))
    table = pd.DataFrame(rows).sort_values(["score", "sum_ret", "active_win_rate"], ascending=False)
    best_id = str(table.iloc[0]["policy_id"])
    best_params = table.iloc[0][["mode", "lookback", "min_n", "min_avg", "min_win", "start_year"]].to_dict()
    best = run_policy(daily, stats_cache, best_params).assign(policy_id=best_id)
    return table, best


def score_frame(frame: pd.DataFrame) -> float:
    stat = metrics(frame)
    if not stat:
        return -1e9
    return float(
        stat.get("sum_ret", 0.0)
        + 0.5 * np.nan_to_num(stat.get("active_win_rate", 0.0), nan=0.0)
        + 0.1 * stat.get("active_rate", 0.0)
    )


def safe_metrics(frame: pd.DataFrame) -> dict:
    stat = metrics(frame)
    if stat:
        return stat
    return {
        "days": 0,
        "active_days": 0,
        "active_rate": 0.0,
        "active_win_rate": np.nan,
        "avg_active_ret": np.nan,
        "sum_ret": 0.0,
        "annualized_sum_ret": np.nan,
        "max_drawdown": np.nan,
    }


def rolling_grid_select(trades: pd.DataFrame, daily: pd.DataFrame, rank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_cache = build_stats_cache(trades, daily, (60, 120, 180, 252, 504))
    # Keep the search broad but bounded: use the top half of the discovery grid plus several distinct modes.
    candidates = rank.head(240).copy()
    for mode in ("recent_best", "blend_phase_recent"):
        extra = rank[rank["mode"].eq(mode)].head(60)
        candidates = pd.concat([candidates, extra], ignore_index=True)
    candidates = candidates.drop_duplicates("policy_id").reset_index(drop=True)

    policy_outputs: dict[str, pd.DataFrame] = {}
    for _, row in candidates.iterrows():
        params = row[["mode", "lookback", "min_n", "min_avg", "min_win", "start_year"]].to_dict()
        params["lookback"] = int(params["lookback"])
        params["min_n"] = int(params["min_n"])
        params["start_year"] = int(params["start_year"])
        params["min_avg"] = float(params["min_avg"])
        params["min_win"] = float(params["min_win"])
        policy_outputs[str(row["policy_id"])] = run_policy(daily, stats_cache, params)

    parts = []
    choices = []
    years = sorted(daily["date"].dt.year.unique())
    for year in years:
        train_mask = daily["date"] < pd.Timestamp(year=year, month=1, day=1)
        test_mask = daily["date"].dt.year.eq(year)
        if not test_mask.any():
            continue
        scored = []
        for policy_id, frame in policy_outputs.items():
            train = frame.loc[train_mask].copy()
            scored.append({"policy_id": policy_id, "train_score": score_frame(train), **safe_metrics(train)})
        table = pd.DataFrame(scored).sort_values(
            ["train_score", "sum_ret", "active_win_rate", "active_rate"],
            ascending=False,
        )
        selected_id = str(table.iloc[0]["policy_id"])
        selected = policy_outputs[selected_id].loc[test_mask].copy()
        selected["test_year"] = year
        selected["selected_policy_id"] = selected_id
        parts.append(selected)
        spec = candidates[candidates["policy_id"].eq(selected_id)].iloc[0].to_dict()
        choices.append({"year": int(year), **spec, **table.iloc[0].add_prefix("train_").to_dict()})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(choices)


def baseline(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in LINES:
        frame = daily[["date", line]].rename(columns={line: "ret"}).copy()
        frame["lines"] = line
        rows.append({"policy": f"always_{line}", **metrics(frame)})
    for combo in (("C", "S"), ("C", "D"), ("S", "D"), ("C", "S", "D")):
        frame = daily[["date", *combo]].copy()
        frame["ret"] = frame[list(combo)].mean(axis=1)
        frame["lines"] = "+".join(combo)
        rows.append({"policy": "+".join(combo), **metrics(frame[["date", "lines", "ret"]])})
    return pd.DataFrame(rows).sort_values("sum_ret", ascending=False)


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    trades, daily = load_inputs()
    base = baseline(daily)
    rank, best = grid_search(trades, daily)
    rolling, rolling_choices = rolling_grid_select(trades, daily, rank)
    best["year"] = best["date"].dt.year
    rolling["year"] = rolling["date"].dt.year
    by_year = best.groupby("year").agg(
        days=("ret", "size"),
        active=("lines", lambda x: int((x != "CASH").sum())),
        win=("ret", lambda x: float((x > 0).mean())),
        avg=("ret", "mean"),
        sum=("ret", "sum"),
    ).reset_index()
    rolling_by_year = rolling.groupby("year").agg(
        days=("ret", "size"),
        active=("lines", lambda x: int((x != "CASH").sum())),
        win=("ret", lambda x: float((x > 0).mean())),
        avg=("ret", "mean"),
        sum=("ret", "sum"),
    ).reset_index()

    base.to_csv(OUT / "baseline.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(OUT / "guard_policy_rank.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT / "best_guard_daily_returns.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(OUT / "best_guard_by_year.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(OUT / "rolling_guard_daily_returns.csv", index=False, encoding="utf-8-sig")
    rolling_choices.to_csv(OUT / "rolling_guard_choices.csv", index=False, encoding="utf-8-sig")
    rolling_by_year.to_csv(OUT / "rolling_guard_by_year.csv", index=False, encoding="utf-8-sig")
    summary = {
        "baseline": base.head(10).to_dict("records"),
        "best_guard": rank.head(10).to_dict("records"),
        "best_guard_by_year": by_year.to_dict("records"),
        "rolling_guard": metrics(rolling),
        "rolling_guard_by_year": rolling_by_year.to_dict("records"),
        "rolling_guard_choices": rolling_choices.to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
