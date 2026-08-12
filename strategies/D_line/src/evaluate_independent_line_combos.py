from __future__ import annotations

import json
import sys


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "D" / "out" / "independent_line_combos"


def add_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


add_path(ROOT / "C" / "src")
from c2_pipeline import load_raw as c_load_raw  # noqa: E402
from live_model import choose_fallback_symbol  # noqa: E402
from live_signal import (  # noqa: E402
    BEST_TRADES,
    RAW as C_RAW,
    SELECTED_TRADES,
    SELECTOR,
    enrich_indicators,
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
from daily_panic_live import build_live_candidates  # noqa: E402
from daily_panic_rolling_test import prepare  # noqa: E402


TODAY = pd.Timestamp("2026-08-07")
WINDOWS = {
    "过去一年": TODAY - pd.DateOffset(years=1),
    "过去两年": TODAY - pd.DateOffset(years=2),
}
LINES = ("C", "S", "D")


def next_trade_row(df: pd.DataFrame, dt: pd.Timestamp) -> pd.Series | None:
    hit = df[df["date"] > dt]
    if hit.empty:
        return None
    return hit.iloc[0]


def row_at_or_after(df: pd.DataFrame, dt: pd.Timestamp) -> tuple[int, pd.Series] | tuple[None, None]:
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
def c_signal_series(symbol: str, strategy_id: str, signal_type: str) -> pd.DataFrame:
    raw = c_load_raw(symbol)
    selection = pd.read_csv(SELECTOR)
    catalog = build_strategy_catalog()
    rule = catalog[catalog["strategy_id"].astype(str).eq(str(strategy_id))]
    if rule.empty:
        raise KeyError(f"missing C strategy rule: {strategy_id}")
    strategy = rule.iloc[0]
    if signal_type == "buy":
        freq = freq_name(strategy["buy_frequency"])
        combo = parse_combo(strategy["buy_combo"])
        threshold = parse_threshold(strategy["buy_threshold"], len(combo))
        side = "low"
        value = float(strategy["buy_low"]) if "buy_low" in strategy.index else 0.25
    else:
        freq = freq_name(strategy["sell_frequency"])
        combo = parse_combo(strategy["sell_combo"])
        threshold = parse_threshold(strategy["sell_threshold"], len(combo))
        side = "high"
        value = float(strategy["sell_high"]) if "sell_high" in strategy.index else 0.90
    panel = signal_panel(raw, freq, combo, side, value)
    factors = [factor_name(x) for x in combo]
    panel = panel[["date", *factors]].copy()
    panel["combined"] = panel[factors].astype(bool).sum(axis=1).ge(threshold)
    return panel[["date", "combined"]]


def c_signal_on(symbol: str, strategy_id: str, signal_type: str, dt: pd.Timestamp) -> bool:
    panel = c_signal_series(symbol, str(strategy_id), signal_type)
    hit = panel[panel["date"].eq(dt)]
    return bool(not hit.empty and bool(hit.iloc[0]["combined"]))


def c_exit_main(symbol: str, strategy_id: str, entry_date: pd.Timestamp, max_hold: int) -> tuple[pd.Timestamp | None, float | None]:
    raw = c_load_raw(symbol)
    entry_pos, entry = row_at_or_after(raw, entry_date)
    if entry_pos is None:
        return None, None
    exit_pos = None
    for pos in range(entry_pos + 1, min(len(raw), entry_pos + int(max_hold) + 1)):
        dt = pd.Timestamp(raw.iloc[pos]["date"])
        if c_signal_on(symbol, strategy_id, "sell", dt):
            exit_pos = pos
            break
    if exit_pos is None:
        exit_pos = entry_pos + int(max_hold)
    if exit_pos >= len(raw):
        return None, None
    exit_ = raw.iloc[exit_pos]
    ret = float(exit_["close"] / entry["close"] - 1.0)
    return pd.Timestamp(exit_["date"]), ret


def c_fallback_for_day(dt: pd.Timestamp, selected: pd.DataFrame) -> dict | None:
    hit = selected[selected["year"].astype(int).le(int(dt.year))].tail(1)
    if hit.empty:
        return None
    row = hit.iloc[0]
    if str(row.get("param_id", "")).upper() == "CASH" or pd.isna(row.get("mode")):
        return None
    symbol, _ = choose_fallback_symbol(dt, str(row["mode"]), int(row["ma"]), int(row["mom"]))
    if symbol is None:
        return None
    raw = c_load_raw(symbol)
    entry = next_trade_row(raw, dt)
    if entry is None:
        return {
            "signal_date": dt,
            "line": "C",
            "kind": "fallback",
            "symbol": symbol,
            "name": symbol,
            "entry_date": pd.NaT,
            "exit_date": pd.NaT,
            "completed": False,
            "ret": np.nan,
        }
    entry_pos = int(raw.index[raw["date"].eq(entry["date"])][0])
    exit_pos = entry_pos + int(row["hold"])
    if exit_pos >= len(raw):
        exit_date, ret, completed = pd.NaT, np.nan, False
    else:
        exit_ = raw.iloc[exit_pos]
        exit_date = pd.Timestamp(exit_["date"])
        ret = float(exit_["close"] / entry["close"] - 1.0)
        completed = bool(exit_date <= TODAY)
    return {
        "signal_date": dt,
        "line": "C",
        "kind": "fallback",
        "symbol": symbol,
        "name": "红利ETF" if symbol == "510880" else "深成ETF" if symbol == "159902" else symbol,
        "entry_date": pd.Timestamp(entry["date"]),
        "exit_date": exit_date,
        "completed": completed,
        "ret": ret if completed else np.nan,
    }


def generate_c_trades(start: pd.Timestamp) -> pd.DataFrame:
    pool = pd.read_csv(C_RAW / "pool.csv", dtype={"symbol": str})
    pool = pool[pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    selection = pd.read_csv(SELECTOR)
    catalog = build_strategy_catalog()
    fallback_selected = pd.read_csv(ROOT / "C" / "fit" / "fallback" / "selected_by_year.csv")
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
                ok = c_signal_on(symbol, str(strategy["strategy_id"]), "buy", dt)
            except Exception:
                ok = False
            if ok:
                candidates.append((float(strategy.get("selected_score", 0) or 0), symbol, etf, strategy))
        if candidates:
            _, symbol, etf, strategy = sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)[0]
            raw = c_load_raw(symbol)
            entry = next_trade_row(raw, dt)
            if entry is None:
                exit_date, ret, completed = pd.NaT, np.nan, False
                entry_date = pd.NaT
            else:
                entry_date = pd.Timestamp(entry["date"])
                exit_date, ret = c_exit_main(symbol, str(strategy["strategy_id"]), entry_date, int(strategy.get("max_hold", 40) or 40))
                completed = bool(exit_date is not None and exit_date <= TODAY)
            rows.append(
                {
                    "signal_date": dt,
                    "line": "C",
                    "kind": "main",
                    "symbol": symbol,
                    "name": etf["display_name"],
                    "entry_date": entry_date,
                    "exit_date": exit_date if exit_date is not None else pd.NaT,
                    "completed": completed,
                    "ret": ret if completed else np.nan,
                }
            )
        else:
            fb = c_fallback_for_day(dt, fallback_selected)
            if fb is not None:
                rows.append(fb)
    return pd.DataFrame(rows)


def generate_s_trades(start: pd.Timestamp) -> pd.DataFrame:
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
        _, symbol, _extra = candidates[0]
        df = engine.panel[symbol]
        entry_pos = engine.pos_after(df, dt + pd.Timedelta(days=1))
        if entry_pos is None:
            entry_date, exit_date, ret, completed = pd.NaT, pd.NaT, np.nan, False
        else:
            exit_pos = entry_pos + int(row["hold"])
            entry = df.iloc[entry_pos]
            if exit_pos >= len(df):
                entry_date, exit_date, ret, completed = pd.Timestamp(entry["date"]), pd.NaT, np.nan, False
            else:
                exit_ = df.iloc[exit_pos]
                exit_date = pd.Timestamp(exit_["date"])
                ret = float(exit_["close"] / entry["close"] - 1.0)
                completed = bool(exit_date <= TODAY)
                entry_date = pd.Timestamp(entry["date"])
        rows.append(
            {
                "signal_date": dt,
                "line": "S",
                "kind": "s",
                "symbol": symbol,
                "name": engine.meta.get(symbol, {}).get("display_name", symbol),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "completed": completed,
                "ret": ret if completed else np.nan,
            }
        )
    return pd.DataFrame(rows)


def generate_d_trades(start: pd.Timestamp) -> pd.DataFrame:
    config = json.loads((ROOT / "D" / "cfg" / "daily_panic.json").read_text(encoding="utf-8"))
    adjusted, raw, _pool = prepare()
    formal = build_live_candidates(adjusted, raw, config)
    formal = formal[
        (formal["signal_date"] >= start)
        & (formal["signal_date"] <= TODAY)
        & (formal["fear_weighted"] >= float(config["market_fear_threshold"]))
        & (formal["symbol_fear"] >= float(config["symbol_fear_threshold"]))
    ].copy()
    if formal.empty:
        return pd.DataFrame()
    formal = formal.sort_values(["signal_date", "buy_quality", "entry_close", "symbol"], ascending=[True, False, False, True])
    top = formal.groupby("signal_date", as_index=False).head(1).copy()
    top["completed"] = top["exit_date"].notna() & (top["exit_date"] <= TODAY)
    return pd.DataFrame(
        {
            "signal_date": top["signal_date"],
            "line": "D",
            "kind": "d",
            "symbol": top["symbol"].astype(str).str.zfill(6),
            "name": top["name"],
            "entry_date": top["entry_date"],
            "exit_date": top["exit_date"],
            "completed": top["completed"],
            "ret": np.where(top["completed"], top["return_rate"], np.nan),
        }
    )


def combo_stats(trades: pd.DataFrame, start: pd.Timestamp, label: str) -> pd.DataFrame:
    trades = trades.copy()
    for col in ["signal_date", "entry_date", "exit_date"]:
        trades[col] = pd.to_datetime(trades[col], errors="coerce")
    trades = trades[(trades["signal_date"] >= start) & (trades["signal_date"] <= TODAY)].copy()
    trades["completed"] = trades["completed"].astype(bool)
    out = []
    for r in range(1, 4):
        for combo in combinations(LINES, r):
            sub = trades[trades["line"].isin(combo)].copy()
            comp = sub[sub["completed"] & sub["ret"].notna()].copy()
            comp["win"] = comp["ret"] > 0
            day_source = sub[["signal_date", "line", "completed", "ret"]].copy()
            day_source["ret0"] = np.where(day_source["completed"] & day_source["ret"].notna(), day_source["ret"], 0.0)
            daily = day_source.groupby("signal_date", as_index=False)["ret0"].sum()
            daily["basket_ret"] = daily["ret0"] / len(combo)
            out.append(
                {
                    "window": label,
                    "combo": "+".join(combo),
                    "trigger_trades": int(len(sub)),
                    "completed_trades": int(len(comp)),
                    "unfinished_trades": int(len(sub) - len(comp)),
                    "single_trade_win_rate": float(comp["win"].mean()) if len(comp) else np.nan,
                    "avg_trade_ret": float(comp["ret"].mean()) if len(comp) else np.nan,
                    "sum_trade_ret": float(comp["ret"].sum()) if len(comp) else 0.0,
                    "basket_signal_days": int(len(daily)),
                    "basket_win_rate": float((daily["basket_ret"] > 0).mean()) if len(daily) else np.nan,
                    "avg_basket_day_ret": float(daily["basket_ret"].mean()) if len(daily) else np.nan,
                    "simple_basket_sum_ret": float(daily["basket_ret"].sum()) if len(daily) else 0.0,
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    two_year_start = WINDOWS["过去两年"]
    generated = pd.concat(
        [
            generate_c_trades(two_year_start),
            generate_s_trades(two_year_start),
            generate_d_trades(two_year_start),
        ],
        ignore_index=True,
        sort=False,
    )
    for col in ["signal_date", "entry_date", "exit_date"]:
        generated[col] = pd.to_datetime(generated[col], errors="coerce")
    generated.to_csv(OUT / "trades_2y_generated.csv", index=False, encoding="utf-8-sig")

    existing_1y = ROOT / "D" / "out" / "one_year_three_line_eval" / "trades.csv"
    one_year_trades = pd.read_csv(existing_1y, dtype={"symbol": str})
    summary = pd.concat(
        [
            combo_stats(one_year_trades, WINDOWS["过去一年"], "过去一年"),
            combo_stats(generated, WINDOWS["过去两年"], "过去两年"),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(OUT / "combo_summary_1y_2y.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nfiles:\n{OUT / 'trades_2y_generated.csv'}\n{OUT / 'combo_summary_1y_2y.csv'}")


if __name__ == "__main__":
    main()
