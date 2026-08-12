from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
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
PHASE_BASE = ROOT / "D" / "out" / "phase_judge"
OUT = PHASE_BASE
FORCED_TRADES = ROOT / "D" / "out" / "forced_daily_top1" / "forced_top1_trades_2y.csv"
SUMMARY = OUT / "summary.json"
PHASE_CACHE = OUT / "daily_phase_scores.csv"
LINES = ("C", "S", "D")


def add_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


add_path(ROOT / "D" / "src")
import fear_greed_oos as fg  # noqa: E402
import fear_greed_weight_sweep as sweep  # noqa: E402


@dataclass(frozen=True)
class Policy:
    key: str
    trend_threshold: float
    rebound_threshold: float
    panic_threshold: float
    mode: str


def sigmoid_scale(x: pd.Series, scale: float) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce") / scale
    vals = vals.clip(-8, 8)
    return 1.0 / (1.0 + np.exp(-vals))


def load_phase_scores() -> pd.DataFrame:
    if PHASE_CACHE.exists():
        cached = pd.read_csv(PHASE_CACHE, encoding="utf-8-sig")
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
        if "panic_score" in cached.columns:
            cached["panic_score"] = pd.to_numeric(cached["panic_score"], errors="coerce").ffill().fillna(0.0)
        return cached
    raw, _pool = fg.load_clean_raw()
    pieces = []
    for symbol, group in raw.groupby("symbol", sort=False):
        df = group.sort_values("date").copy()
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change()
        df["ret5"] = close.pct_change(5)
        df["ret20"] = close.pct_change(20)
        df["ret60"] = close.pct_change(60)
        df["ma60"] = close.rolling(60, min_periods=30).mean()
        df["above_ma60"] = close > df["ma60"]
        low120 = close.rolling(120, min_periods=60).min()
        high120 = close.rolling(120, min_periods=60).max()
        df["pos120"] = (close - low120) / (high120 - low120).replace(0, np.nan)
        df["dd60"] = close / close.rolling(60, min_periods=30).max() - 1.0
        df["vol20"] = ret.rolling(20, min_periods=15).std() * np.sqrt(252)
        pieces.append(df)
    panel = pd.concat(pieces, ignore_index=True, sort=False)
    daily = (
        panel.groupby("date")
        .agg(
            etfs=("symbol", "nunique"),
            above_ma60=("above_ma60", "mean"),
            median_ret5=("ret5", "median"),
            median_ret20=("ret20", "median"),
            median_ret60=("ret60", "median"),
            low_pos_rate=("pos120", lambda values: float((pd.to_numeric(values, errors="coerce") <= 0.25).mean())),
            deep_drop_rate=("ret20", lambda values: float((pd.to_numeric(values, errors="coerce") <= -0.08).mean())),
            median_dd60=("dd60", "median"),
            median_vol20=("vol20", "median"),
        )
        .reset_index()
        .sort_values("date")
    )

    sentiment_path = ROOT / "D" / "out" / "fear_greed" / "sentiment_daily.csv"
    if sentiment_path.exists():
        sentiment = pd.read_csv(sentiment_path, encoding="utf-8-sig")
        sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
        sentiment = sentiment[["date", "fear_core", "fear_qvix"]].copy()
    else:
        sentiment = fg.build_sentiment(raw)
        sentiment = sentiment[["date", "fear_core", "fear_qvix"]].copy()
    sentiment["panic_score"] = sweep.composite(sentiment["fear_core"], sentiment["fear_qvix"], 0.2) / 100.0
    daily = daily.merge(sentiment[["date", "panic_score"]], on="date", how="left")

    trend_components = [
        daily["above_ma60"].clip(0, 1),
        sigmoid_scale(daily["median_ret20"], 0.04),
        sigmoid_scale(daily["median_ret60"], 0.08),
    ]
    daily["trend_score"] = pd.concat(trend_components, axis=1).mean(axis=1)

    # Rebound is intentionally allowed to overlap with trend and panic:
    # low-position breadth + prior drawdown + first signs of 5-day repair.
    rebound_components = [
        daily["low_pos_rate"].clip(0, 1),
        daily["deep_drop_rate"].clip(0, 1),
        (-daily["median_dd60"] / 0.12).clip(0, 1),
        sigmoid_scale(daily["median_ret5"], 0.03),
    ]
    daily["rebound_score"] = pd.concat(rebound_components, axis=1).mean(axis=1)
    daily["panic_score"] = daily["panic_score"].clip(0, 1).ffill().fillna(0.0)
    daily.to_csv(PHASE_CACHE, index=False, encoding="utf-8-sig")
    return daily


def load_daily_returns(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    for col in ("signal_date", "entry_date", "exit_date"):
        trades[col] = pd.to_datetime(trades[col], errors="coerce")
    trades["completed"] = trades["completed"].astype(bool)
    trades["ret0"] = np.where(trades["completed"] & trades["ret"].notna(), trades["ret"].astype(float), 0.0)
    pivot = trades.pivot_table(index="signal_date", columns="line", values="ret0", aggfunc="sum").reset_index()
    for line in LINES:
        if line not in pivot.columns:
            pivot[line] = 0.0
    return pivot.rename(columns={"signal_date": "date"})[["date", *LINES]]


def selected_lines(row: pd.Series, policy: Policy) -> tuple[str, ...]:
    if policy.mode == "single_argmax":
        scores = {
            "C": float(row["rebound_score"]),
            "S": float(row["trend_score"]),
            "D": float(row["panic_score"]),
        }
        return (max(scores, key=scores.get),)

    active = []
    if float(row["rebound_score"]) >= policy.rebound_threshold:
        active.append("C")
    if float(row["trend_score"]) >= policy.trend_threshold:
        active.append("S")
    if float(row["panic_score"]) >= policy.panic_threshold:
        active.append("D")
    if active:
        return tuple(active)
    if policy.mode == "cash_if_none":
        return tuple()
    if policy.mode == "fallback_argmax":
        return selected_lines(row, Policy("single_argmax", 0, 0, 0, "single_argmax"))
    raise ValueError(f"unknown policy mode: {policy.mode}")


def apply_policy(data: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        lines = selected_lines(row, policy)
        if lines:
            ret = float(np.mean([row[line] for line in lines]))
        else:
            ret = 0.0
        rows.append(
            {
                "date": row["date"],
                "policy": policy.key,
                "lines": "+".join(lines) if lines else "CASH",
                "ret": ret,
                "trend_score": row["trend_score"],
                "rebound_score": row["rebound_score"],
                "panic_score": row["panic_score"],
            }
        )
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "days": 0,
            "active_days": 0,
            "win_rate": np.nan,
            "avg_day_ret": np.nan,
            "sum_ret": 0.0,
            "annualized_sum_ret": np.nan,
            "max_drawdown": np.nan,
        }
    ret = pd.to_numeric(frame["ret"], errors="coerce").fillna(0.0)
    equity = (1.0 + ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    active = frame[frame["lines"].ne("CASH")]
    years = max((pd.Timestamp(frame["date"].max()) - pd.Timestamp(frame["date"].min())).days / 365.25, 1e-9)
    return {
        "days": int(len(frame)),
        "active_days": int(len(active)),
        "active_rate": float(len(active) / len(frame)) if len(frame) else np.nan,
        "win_rate": float((ret > 0).mean()) if len(frame) else np.nan,
        "active_win_rate": float((active["ret"] > 0).mean()) if len(active) else np.nan,
        "avg_day_ret": float(ret.mean()),
        "sum_ret": float(ret.sum()),
        "annualized_sum_ret": float(ret.sum() / years),
        "compound_ret": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def score(stat: dict) -> float:
    active_rate = float(np.nan_to_num(stat.get("active_rate"), nan=0.0))
    win_rate = float(np.nan_to_num(stat.get("active_win_rate"), nan=0.0))
    dd = float(np.nan_to_num(stat.get("max_drawdown"), nan=-1.0))
    return float(stat["sum_ret"] + 0.75 * win_rate + 0.15 * active_rate + 0.35 * dd)


def policy_grid() -> list[Policy]:
    policies = [
        Policy("always_C", 2.0, 0.0, 2.0, "cash_if_none"),
        Policy("always_S", 0.0, 2.0, 2.0, "cash_if_none"),
        Policy("always_D", 2.0, 2.0, 0.0, "cash_if_none"),
        Policy("single_argmax", 0.0, 0.0, 0.0, "single_argmax"),
        Policy("always_CSD_equal", 0.0, 0.0, 0.0, "cash_if_none"),
    ]
    thresholds = (0.35, 0.45, 0.55, 0.65, 0.75)
    for trend, rebound, panic, mode in product(thresholds, thresholds, thresholds, ("cash_if_none", "fallback_argmax")):
        policies.append(
            Policy(
                f"multi_t{trend:.2f}_r{rebound:.2f}_p{panic:.2f}_{mode}",
                trend,
                rebound,
                panic,
                mode,
            )
        )
    return policies


def evaluate_all(data: pd.DataFrame, policies: list[Policy], label: str) -> pd.DataFrame:
    rows = []
    for policy in policies:
        result = apply_policy(data, policy)
        stat = metrics(result)
        rows.append({"window": label, "policy": policy.key, "score": score(stat), **stat})
    return pd.DataFrame(rows).sort_values(["score", "sum_ret", "active_win_rate"], ascending=False)


def rolling_select(data: pd.DataFrame, policies: list[Policy]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    choices = []
    years = sorted({pd.Timestamp(x).year for x in data["date"].dropna().unique()})
    for year in years:
        train = data[data["date"] < pd.Timestamp(year=year, month=1, day=1)].copy()
        test = data[data["date"].dt.year.eq(year)].copy()
        if test.empty:
            continue
        if len(train) < 80:
            selected = Policy("single_argmax", 0.0, 0.0, 0.0, "single_argmax")
            audit = {"fallback": True}
        else:
            ranked = evaluate_all(train, policies, f"train_to_{year - 1}")
            selected_key = str(ranked.iloc[0]["policy"])
            selected = next(policy for policy in policies if policy.key == selected_key)
            audit = ranked.iloc[0].to_dict()
            audit["fallback"] = False
        current = apply_policy(test, selected)
        current["test_year"] = year
        parts.append(current)
        choices.append(
            {
                "year": year,
                "policy": selected.key,
                "train_days": int(len(train)),
                "fallback": bool(audit.get("fallback", False)),
                "train_score": audit.get("score"),
                "train_sum_ret": audit.get("sum_ret"),
                "train_active_win_rate": audit.get("active_win_rate"),
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(choices)


def phase_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    weights = np.array([1.0, 1.0, 1.15], dtype=float)
    return np.sqrt(((b - a) ** 2 * weights).sum(axis=1))


def choose_by_similarity(
    history: pd.DataFrame,
    row: pd.Series,
    k: int,
    min_edge: float,
    top_n: int,
) -> tuple[tuple[str, ...], dict]:
    if len(history) < max(30, k):
        return selected_lines(row, Policy("single_argmax", 0, 0, 0, "single_argmax")), {"fallback": True}
    point = row[["trend_score", "rebound_score", "panic_score"]].to_numpy(dtype=float)
    hist_points = history[["trend_score", "rebound_score", "panic_score"]].to_numpy(dtype=float)
    dist = phase_distance(point, hist_points)
    order = np.argsort(dist)[: min(k, len(history))]
    near = history.iloc[order]
    line_edges = []
    for line in LINES:
        vals = pd.to_numeric(near[line], errors="coerce").fillna(0.0)
        # Closer days get slightly larger weight, but no one day can dominate.
        local_dist = dist[order]
        weights = 1.0 / (local_dist + 0.08)
        edge = float(np.average(vals, weights=weights))
        win = float((vals > 0).mean())
        line_edges.append((edge, win, line))
    ranked = sorted(line_edges, key=lambda item: (item[0], item[1]), reverse=True)
    selected = [line for edge, _win, line in ranked if edge > min_edge][:top_n]
    if not selected:
        return tuple(), {"fallback": False, "edges": ranked}
    return tuple(selected), {"fallback": False, "edges": ranked}


def apply_similarity_policy(
    test: pd.DataFrame,
    history: pd.DataFrame,
    k: int,
    min_edge: float,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        lines, audit = choose_by_similarity(history, row, k, min_edge, top_n)
        ret = float(np.mean([row[line] for line in lines])) if lines else 0.0
        rows.append(
            {
                "date": row["date"],
                "policy": f"similar_k{k}_edge{min_edge:.3f}_top{top_n}",
                "lines": "+".join(lines) if lines else "CASH",
                "ret": ret,
                "trend_score": row["trend_score"],
                "rebound_score": row["rebound_score"],
                "panic_score": row["panic_score"],
                "similarity_fallback": bool(audit.get("fallback", False)),
            }
        )
    return pd.DataFrame(rows)


def expanding_similarity_score(train: pd.DataFrame, k: int, min_edge: float, top_n: int) -> dict:
    parts = []
    dates = list(train["date"])
    for index in range(len(train)):
        row = train.iloc[index]
        history = train.iloc[:index].copy()
        if len(history) < max(40, k):
            continue
        parts.append(apply_similarity_policy(pd.DataFrame([row]), history, k, min_edge, top_n))
    if not parts:
        return {"score": -np.inf, "days": 0}
    result = pd.concat(parts, ignore_index=True)
    stat = metrics(result)
    return {"score": score(stat), **stat}


def rolling_similarity_select(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = [
        {"k": k, "min_edge": edge, "top_n": top_n}
        for k in (20, 40, 80, 120)
        for edge in (-0.002, 0.0, 0.002, 0.005)
        for top_n in (1, 2, 3)
    ]
    parts = []
    choices = []
    grid_rows = []
    years = sorted({pd.Timestamp(x).year for x in data["date"].dropna().unique()})
    for year in years:
        train = data[data["date"] < pd.Timestamp(year=year, month=1, day=1)].copy()
        test = data[data["date"].dt.year.eq(year)].copy()
        if test.empty:
            continue
        if len(train) < 90:
            selected = {"k": 40, "min_edge": 0.0, "top_n": 1}
            current = apply_policy(test, Policy("single_argmax", 0, 0, 0, "single_argmax"))
            audit = {"score": np.nan, "days": 0, "fallback": True}
        else:
            scored = []
            for params in grid:
                stat = expanding_similarity_score(train, **params)
                scored.append({**params, **stat})
            table = pd.DataFrame(scored).sort_values(
                ["score", "sum_ret", "active_win_rate", "active_rate"],
                ascending=False,
            )
            table["test_year"] = year
            grid_rows.append(table)
            selected = table.iloc[0][["k", "min_edge", "top_n"]].to_dict()
            selected["k"] = int(selected["k"])
            selected["top_n"] = int(selected["top_n"])
            selected["min_edge"] = float(selected["min_edge"])
            current = apply_similarity_policy(test, train, **selected)
            audit = table.iloc[0].to_dict()
            audit["fallback"] = False
        current["test_year"] = year
        parts.append(current)
        choices.append(
            {
                "year": year,
                **selected,
                "train_days": int(len(train)),
                "fallback": bool(audit.get("fallback", False)),
                "train_score": audit.get("score"),
                "train_sum_ret": audit.get("sum_ret"),
                "train_active_win_rate": audit.get("active_win_rate"),
            }
        )
    grid_table = pd.concat(grid_rows, ignore_index=True) if grid_rows else pd.DataFrame()
    return pd.concat(parts, ignore_index=True), pd.DataFrame(choices), grid_table


def main() -> None:
    global OUT, SUMMARY
    configure_stdout()
    parser = ArgumentParser()
    parser.add_argument("--trades", default=str(FORCED_TRADES), help="forced daily top1 trades csv")
    parser.add_argument("--tag", default="2y", help="output subfolder tag")
    parser.add_argument("--skip-sim", action="store_true", help="skip slow similarity rolling search")
    args = parser.parse_args()
    OUT = PHASE_BASE if args.tag == "2y" else PHASE_BASE / args.tag
    SUMMARY = OUT / "summary.json"
    OUT.mkdir(parents=True, exist_ok=True)
    phases = load_phase_scores()
    trades_path = Path(args.trades)
    returns = load_daily_returns(trades_path)
    data = returns.merge(phases, on="date", how="left").dropna(
        subset=["trend_score", "rebound_score", "panic_score"]
    )
    policies = policy_grid()
    full_rank = evaluate_all(data, policies, "available_2y")
    rolling, choices = rolling_select(data, policies)
    rolling_stat = metrics(rolling)
    if args.skip_sim:
        sim_rolling = pd.DataFrame()
        sim_choices = pd.DataFrame()
        sim_grid = pd.DataFrame()
        sim_stat = {}
    else:
        sim_rolling, sim_choices, sim_grid = rolling_similarity_select(data)
        sim_stat = metrics(sim_rolling)

    full_rank.to_csv(OUT / "policy_rank_available.csv", index=False, encoding="utf-8-sig")
    data.to_csv(OUT / "daily_phase_scores_and_line_returns.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(OUT / "rolling_judge_daily_returns.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "rolling_judge_choices.csv", index=False, encoding="utf-8-sig")
    if not args.skip_sim:
        sim_rolling.to_csv(OUT / "similarity_judge_daily_returns.csv", index=False, encoding="utf-8-sig")
        sim_choices.to_csv(OUT / "similarity_judge_choices.csv", index=False, encoding="utf-8-sig")
        sim_grid.to_csv(OUT / "similarity_param_grid_by_year.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forced_trades": str(trades_path),
        "available_range": [str(data["date"].min().date()), str(data["date"].max().date())],
        "definition": {
            "trend_score": "breadth above MA60 + median 20/60-day returns",
            "rebound_score": "low-position breadth + 20-day deep-drop breadth + 60-day drawdown + 5-day repair",
            "panic_score": "internal fear score + QVIX weighted by D-line sentiment model",
            "overlap": "scores are independent; multiple lines may be selected on the same day",
        },
        "best_full_sample_policies": full_rank.head(10).to_dict("records"),
        "rolling_judge": rolling_stat,
        "rolling_choices": choices.to_dict("records"),
        "similarity_judge": sim_stat,
        "similarity_choices": sim_choices.to_dict("records") if not sim_choices.empty else [],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(
        f"\nfiles:\n{OUT / 'policy_rank_available.csv'}"
        f"\n{OUT / 'rolling_judge_daily_returns.csv'}"
        f"\n{OUT / 'rolling_judge_choices.csv'}"
        f"\n{OUT / 'similarity_judge_daily_returns.csv'}"
        f"\n{OUT / 'similarity_judge_choices.csv'}"
        f"\n{SUMMARY}"
    )


if __name__ == "__main__":
    main()
