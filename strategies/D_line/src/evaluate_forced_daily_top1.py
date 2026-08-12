from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from functools import lru_cache
from itertools import combinations
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
OUT = ROOT / "D" / "out" / "forced_daily_top1"
TODAY = pd.Timestamp("2026-08-07")
LINES = ("C", "S", "D")


def add_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


add_path(ROOT / "C" / "src")
from c2_pipeline import load_raw as c_load_raw  # noqa: E402
from live_signal import (  # noqa: E402
    BEST_TRADES,
    RAW as C_RAW,
    SELECTED_TRADES,
    SELECTOR,
    factor_name,
    freq_name,
    parse_combo,
    parse_threshold,
    pick_active_strategies,
    signal_panel,
)

add_path(ROOT / "S" / "src")
from shadow_model import ShadowResearchEngine  # noqa: E402

add_path(ROOT / "D" / "src")
import fear_greed_oos as fg  # noqa: E402
import fear_greed_weight_sweep as sweep  # noqa: E402
from daily_panic_stale_exit_test import adjust_exits  # noqa: E402
from daily_panic_live import selected_rule  # noqa: E402


def next_trade_row(df: pd.DataFrame, dt: pd.Timestamp) -> pd.Series | None:
    hit = df[df["date"] > dt]
    if hit.empty:
        return None
    return hit.iloc[0]


def row_at_or_after(df: pd.DataFrame, dt: pd.Timestamp) -> tuple[int | None, pd.Series | None]:
    arr = df["date"].to_numpy()
    pos = int(np.searchsorted(arr, np.datetime64(dt), side="left"))
    if pos >= len(df):
        return None, None
    return pos, df.iloc[pos]


def build_strategy_catalog() -> pd.DataFrame:
    frames = []
    for path in [SELECTED_TRADES, BEST_TRADES]:
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"symbol": str}))
    if not frames:
        return pd.DataFrame()
    cols = [
        "strategy_id",
        "buy_low",
        "sell_high",
        "buy_target",
        "buy_frequency",
        "buy_combo",
        "buy_threshold",
        "sell_frequency",
        "sell_combo",
        "sell_threshold",
        "max_hold",
        "exit_rule",
    ]
    data = pd.concat(frames, ignore_index=True)
    return data[[c for c in cols if c in data.columns]].drop_duplicates("strategy_id")


@lru_cache(maxsize=None)
def c_factor_panel(symbol: str, strategy_id: str, signal_type: str) -> pd.DataFrame:
    raw = c_load_raw(symbol)
    catalog = build_strategy_catalog()
    rule = catalog[catalog["strategy_id"].astype(str).eq(str(strategy_id))]
    if rule.empty:
        raise KeyError(f"missing C strategy rule: {strategy_id}")
    strategy = rule.iloc[0]
    if signal_type == "buy":
        freq = freq_name(strategy["buy_frequency"])
        combo = parse_combo(strategy["buy_combo"])
        side = "low"
        value = float(strategy["buy_low"]) if "buy_low" in strategy.index else 0.25
    else:
        freq = freq_name(strategy["sell_frequency"])
        combo = parse_combo(strategy["sell_combo"])
        side = "high"
        value = float(strategy["sell_high"]) if "sell_high" in strategy.index else 0.90
    panel = signal_panel(raw, freq, combo, side, value)
    factors = [factor_name(x) for x in combo]
    return panel[["date", *factors]].copy()


def c_hit_on(symbol: str, strategy: pd.Series, signal_type: str, dt: pd.Timestamp) -> tuple[int, int, bool]:
    panel = c_factor_panel(symbol, str(strategy["strategy_id"]), signal_type)
    if signal_type == "buy":
        combo = parse_combo(strategy["buy_combo"])
        threshold = parse_threshold(strategy["buy_threshold"], len(combo))
    else:
        combo = parse_combo(strategy["sell_combo"])
        threshold = parse_threshold(strategy["sell_threshold"], len(combo))
    factors = [factor_name(x) for x in combo]
    hit = panel[panel["date"].eq(dt)]
    if hit.empty:
        return 0, threshold, False
    count = int(sum(bool(hit.iloc[0][factor]) for factor in factors))
    return count, threshold, count >= threshold


def c_exit(symbol: str, strategy: pd.Series, entry_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    raw = c_load_raw(symbol)
    entry_pos, entry = row_at_or_after(raw, entry_date)
    if entry_pos is None or entry is None:
        return None, None
    max_hold = int(strategy.get("max_hold", 40) or 40)
    exit_pos = None
    for pos in range(entry_pos + 1, min(len(raw), entry_pos + max_hold + 1)):
        dt = pd.Timestamp(raw.iloc[pos]["date"])
        try:
            _, _, ok = c_hit_on(symbol, strategy, "sell", dt)
        except Exception:
            ok = False
        if ok:
            exit_pos = pos
            break
    if exit_pos is None:
        exit_pos = entry_pos + max_hold
    if exit_pos >= len(raw):
        return None, None
    exit_row = raw.iloc[exit_pos]
    return pd.Timestamp(exit_row["date"]), float(exit_row["close"] / entry["close"] - 1.0)


def generate_c_forced(start: pd.Timestamp) -> pd.DataFrame:
    pool = pd.read_csv(C_RAW / "pool.csv", dtype={"symbol": str})
    pool = pool[pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    selection = pd.read_csv(SELECTOR)
    catalog = build_strategy_catalog()
    calendar = c_load_raw("510880")["date"]
    dates = [pd.Timestamp(x) for x in calendar[(calendar >= start) & (calendar <= TODAY)]]
    rows = []
    for dt in dates:
        active = pick_active_strategies(selection, int(dt.year))
        active = active.merge(catalog, on="strategy_id", how="left")
        active_by_category = {row["category"]: row for _, row in active.iterrows()}
        candidates = []
        for _, etf in pool.iterrows():
            category = etf["category"]
            strategy = active_by_category.get(category)
            if strategy is None or pd.isna(strategy.get("buy_frequency", np.nan)):
                continue
            symbol = str(etf["symbol"]).zfill(6)
            try:
                hit_count, threshold, is_signal = c_hit_on(symbol, strategy, "buy", dt)
            except Exception:
                hit_count, threshold, is_signal = 0, 1, False
            hit_ratio = hit_count / max(threshold, 1)
            selected_score = float(strategy.get("selected_score", 0) or 0)
            candidates.append(
                (
                    int(is_signal),
                    float(hit_ratio),
                    selected_score,
                    symbol,
                    etf,
                    strategy,
                    hit_count,
                    threshold,
                )
            )
        if not candidates:
            continue
        _, hit_ratio, selected_score, symbol, etf, strategy, hit_count, threshold = sorted(
            candidates,
            key=lambda x: (x[0], x[1], x[2], x[3]),
            reverse=True,
        )[0]
        raw = c_load_raw(symbol)
        entry = next_trade_row(raw, dt)
        if entry is None:
            entry_date, exit_date, ret, completed = pd.NaT, pd.NaT, np.nan, False
        else:
            entry_date = pd.Timestamp(entry["date"])
            exit_date, ret = c_exit(symbol, strategy, entry_date)
            completed = bool(exit_date is not None and exit_date <= TODAY)
        rows.append(
            {
                "signal_date": dt,
                "line": "C",
                "symbol": symbol,
                "name": etf["display_name"],
                "rank_score": hit_ratio,
                "score_detail": f"hit={hit_count}/{threshold};selected_score={selected_score}",
                "entry_date": entry_date,
                "exit_date": exit_date if exit_date is not None else pd.NaT,
                "completed": completed,
                "ret": ret if completed else np.nan,
            }
        )
    return pd.DataFrame(rows)


def generate_s_forced(start: pd.Timestamp) -> pd.DataFrame:
    engine = ShadowResearchEngine()
    selection = pd.read_csv(ROOT / "S" / "fit" / "selector" / "selected_by_year.csv")
    dates = [pd.Timestamp(x) for x in engine.calendar[(engine.calendar >= start) & (engine.calendar <= TODAY)]]
    rows = []
    for dt in dates:
        hit = selection[selection["year"].astype(int).le(int(dt.year))].tail(1)
        if hit.empty or str(hit.iloc[0].get("param_id", "")).upper() == "CASH":
            continue
        row = hit.iloc[0]
        candidates = engine.candidate_list(
            dt.value,
            str(row["scope"]),
            int(row["ma"]),
            int(row["mom"]),
            str(row["model"]),
            float(row["alpha"]),
            250,
            2_000_000,
            float(row["vol_cap"]),
            float(row["accel_cap"]),
        )
        if not candidates:
            continue
        score, symbol, extra = candidates[0]
        df = engine.panel[symbol]
        entry_pos = engine.pos_after(df, dt + pd.Timedelta(days=1))
        if entry_pos is None:
            entry_date, exit_date, ret, completed = pd.NaT, pd.NaT, np.nan, False
        else:
            entry = df.iloc[entry_pos]
            exit_pos = entry_pos + int(row["hold"])
            if exit_pos >= len(df):
                entry_date, exit_date, ret, completed = pd.Timestamp(entry["date"]), pd.NaT, np.nan, False
            else:
                exit_row = df.iloc[exit_pos]
                entry_date = pd.Timestamp(entry["date"])
                exit_date = pd.Timestamp(exit_row["date"])
                ret = float(exit_row["close"] / entry["close"] - 1.0)
                completed = bool(exit_date <= TODAY)
        rows.append(
            {
                "signal_date": dt,
                "line": "S",
                "symbol": symbol,
                "name": engine.meta.get(symbol, {}).get("display_name", symbol),
                "rank_score": float(score),
                "score_detail": json.dumps(extra, ensure_ascii=False, default=str),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "completed": completed,
                "ret": ret if completed else np.nan,
            }
        )
    return pd.DataFrame(rows)


def generate_d_forced(start: pd.Timestamp) -> pd.DataFrame:
    config = json.loads((ROOT / "D" / "cfg" / "daily_panic.json").read_text(encoding="utf-8"))
    raw, pool = fg.load_clean_raw()
    sentiment = fg.build_sentiment(raw)
    symbol_fear = fg.build_symbol_fear(raw)
    market = sentiment[["date", "fear_core", "fear_qvix"]].copy()
    market["fear_weighted"] = sweep.composite(market["fear_core"], market["fear_qvix"], 0.2)
    market["fear_enhanced"] = market["fear_weighted"]
    market["greed_enhanced"] = 100.0 - market["fear_enhanced"]

    panel = raw.merge(symbol_fear, on=["date", "symbol"], how="left")
    panel = panel.merge(market[["date", "fear_core", "fear_qvix", "fear_weighted", "greed_enhanced"]], on="date", how="left")
    panel = panel.sort_values(["symbol", "date"])
    panel["entry_date"] = panel.groupby("symbol")["date"].shift(-1)
    panel["entry_close"] = panel.groupby("symbol")["close"].shift(-1)
    panel = panel[(panel["date"] >= start) & (panel["date"] <= TODAY) & panel["entry_date"].notna()].copy()
    name_map = pool.set_index("symbol")["name"].to_dict()
    panel["signal_date"] = panel["date"]
    panel["name"] = panel["symbol"].map(name_map).fillna("")
    panel["test_year"] = panel["entry_date"].dt.year
    panel["exit_date"] = pd.NaT
    panel["exit_close"] = np.nan
    panel["status"] = "open"
    panel["return_rate"] = np.nan
    panel["mark_close"] = panel.groupby("symbol")["close"].transform("last")
    panel["unrealized_return"] = panel["mark_close"] / panel["entry_close"] - 1.0
    panel["buy_quality"] = 0.5 * panel["fear_weighted"] + 0.5 * panel["symbol_fear"]
    panel = panel.dropna(subset=["buy_quality"]).sort_values(
        ["signal_date", "buy_quality", "amount", "symbol"],
        ascending=[True, False, False, True],
    )
    top = panel.groupby("signal_date", as_index=False).head(1).copy()
    top = top[
        [
            "test_year",
            "symbol",
            "name",
            "signal_date",
            "entry_date",
            "entry_close",
            "exit_date",
            "exit_close",
            "status",
            "return_rate",
            "mark_close",
            "unrealized_return",
            "fear_core",
            "fear_qvix",
            "fear_weighted",
            "greed_enhanced",
            "symbol_fear",
            "buy_quality",
        ]
    ].reset_index(drop=True)
    adjusted = fg.apply_exit_rule(
        top,
        fg.ExitRule("forced_daily_top1_h90_g70", 90, 70.0),
        raw,
        market,
    )
    parts = []
    for year in sorted(adjusted["test_year"].dropna().astype(int).unique()):
        local = adjusted[adjusted["test_year"].astype(int).eq(year)].copy()
        rule = selected_rule(config, year)
        if rule is not None:
            local = adjust_exits(local, raw, rule)
        parts.append(local)
    out = pd.concat(parts, ignore_index=True, sort=False) if parts else adjusted
    out["completed"] = out["exit_date"].notna() & (pd.to_datetime(out["exit_date"]) <= TODAY)
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(out["signal_date"]),
            "line": "D",
            "symbol": out["symbol"].astype(str).str.zfill(6),
            "name": out["name"],
            "rank_score": out["buy_quality"].astype(float),
            "score_detail": "fear_weighted="
            + out["fear_weighted"].astype(float).round(4).astype(str)
            + ";symbol_fear="
            + out["symbol_fear"].astype(float).round(4).astype(str),
            "entry_date": pd.to_datetime(out["entry_date"], errors="coerce"),
            "exit_date": pd.to_datetime(out["exit_date"], errors="coerce"),
            "completed": out["completed"].astype(bool),
            "ret": np.where(out["completed"].astype(bool), out["return_rate"].astype(float), np.nan),
        }
    )


def combo_stats(trades: pd.DataFrame, start: pd.Timestamp, label: str) -> pd.DataFrame:
    data = trades.copy()
    for col in ["signal_date", "entry_date", "exit_date"]:
        data[col] = pd.to_datetime(data[col], errors="coerce")
    data = data[(data["signal_date"] >= start) & (data["signal_date"] <= TODAY)].copy()
    rows = []
    for r in range(1, 4):
        for combo in combinations(LINES, r):
            sub = data[data["line"].isin(combo)].copy()
            comp = sub[sub["completed"].astype(bool) & sub["ret"].notna()].copy()
            comp["win"] = comp["ret"] > 0
            day_source = sub[["signal_date", "line", "completed", "ret"]].copy()
            day_source["ret0"] = np.where(day_source["completed"].astype(bool) & day_source["ret"].notna(), day_source["ret"], 0.0)
            daily = day_source.groupby("signal_date", as_index=False)["ret0"].sum()
            daily["basket_ret"] = daily["ret0"] / len(combo)
            rows.append(
                {
                    "window": label,
                    "combo": "+".join(combo),
                    "signal_trades": int(len(sub)),
                    "completed_trades": int(len(comp)),
                    "unfinished_trades": int(len(sub) - len(comp)),
                    "single_trade_win_rate": float(comp["win"].mean()) if len(comp) else np.nan,
                    "avg_trade_ret": float(comp["ret"].mean()) if len(comp) else np.nan,
                    "sum_trade_ret": float(comp["ret"].sum()) if len(comp) else 0.0,
                    "basket_days": int(len(daily)),
                    "basket_win_rate": float((daily["basket_ret"] > 0).mean()) if len(daily) else np.nan,
                    "avg_basket_day_ret": float(daily["basket_ret"].mean()) if len(daily) else np.nan,
                    "simple_basket_sum_ret": float(daily["basket_ret"].sum()) if len(daily) else 0.0,
                    "profit_per_1000_cny": float(daily["basket_ret"].sum() * 1000.0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    configure_stdout()
    parser = ArgumentParser()
    parser.add_argument("--start", default=None, help="YYYY-MM-DD; defaults to two years before TODAY")
    parser.add_argument("--tag", default="2y", help="output tag")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start_date = pd.Timestamp(args.start) if args.start else TODAY - pd.DateOffset(years=2)
    one_year_start = TODAY - pd.DateOffset(years=1)
    trades = pd.concat(
        [
            generate_c_forced(start_date),
            generate_s_forced(start_date),
            generate_d_forced(start_date),
        ],
        ignore_index=True,
        sort=False,
    )
    for col in ["signal_date", "entry_date", "exit_date"]:
        trades[col] = pd.to_datetime(trades[col], errors="coerce")
    trades_path = OUT / f"forced_top1_trades_{args.tag}.csv"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    frames = []
    if args.start is None:
        frames.append(combo_stats(trades, one_year_start, "past_1y"))
        frames.append(combo_stats(trades, start_date, "past_2y"))
    else:
        frames.append(combo_stats(trades, start_date, f"from_{start_date.date()}"))
    summary = pd.concat(frames, ignore_index=True, sort=False)
    summary_path = OUT / f"forced_top1_combo_summary_{args.tag}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nfiles:\n{trades_path}\n{summary_path}")


if __name__ == "__main__":
    main()
