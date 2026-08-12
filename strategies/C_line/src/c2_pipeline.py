from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "cfg" / "model.json"
RAW = ROOT / "raw"
QUALITY = ROOT / "pool"
PARAMS = ROOT / "out" / "params"
COMBOS = ROOT / "out" / "combos"
TRADES = ROOT / "out" / "trades"
DATA_DIR = RAW / "etf"
POOL_PATH = RAW / "pool.csv"


def tdx_sma(s: pd.Series, n: int, m: int = 1) -> pd.Series:
    values = s.astype(float).to_numpy()
    out = np.full(len(values), np.nan)
    last = np.nan
    for i, x in enumerate(values):
        if np.isnan(x):
            out[i] = last
            continue
        last = x if np.isnan(last) else (m * x + (n - m) * last) / n
        out[i] = last
    return pd.Series(out, index=s.index)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.astype(float).ewm(span=span, adjust=False).mean()


def load_raw(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["close"].notna() & (df["close"] > 0)].copy()
    return df


def clean_report(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol in pool["symbol"].astype(str):
        df = load_raw(symbol)
        ret = df["close"].pct_change()
        rows.append(
            {
                "symbol": symbol,
                "rows": len(df),
                "start": df["date"].min().date(),
                "end": df["date"].max().date(),
                "zero_volume_days": int((df["volume"].fillna(0) <= 0).sum()),
                "max_daily_return": float(ret.max(skipna=True)),
                "min_daily_return": float(ret.min(skipna=True)),
                "jump_over_12pct_days": int((ret.abs() > 0.12).sum()),
            }
        )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    d = df.copy()
    if freq == "D":
        return d.reset_index(drop=True)
    if freq == "W":
        d["_g"] = d["date"].dt.to_period("W-FRI").astype(str)
    elif freq == "M":
        d["_g"] = d["date"].dt.to_period("M").astype(str)
    elif freq == "B15":
        d["_g"] = np.arange(len(d)) // 15
    else:
        raise ValueError(f"unknown freq: {freq}")
    grouped = d.groupby("_g", sort=True)
    out = grouped.agg(
        date=("date", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    return out.reset_index(drop=True)


def add_indicators(df: pd.DataFrame, caliber: dict) -> pd.DataFrame:
    d = df.copy()
    low9 = d["low"].rolling(9, min_periods=1).min()
    high9 = d["high"].rolling(9, min_periods=1).max()
    rsv = (d["close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    if caliber["kdj"] == "tdx":
        k = tdx_sma(rsv, 3, 1)
        kd = tdx_sma(k, 3, 1)
    else:
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        kd = k.ewm(alpha=1 / 3, adjust=False).mean()
    d["KDJ_K"] = k
    d["KDJ_D"] = kd
    d["KDJ_J"] = 3 * k - 2 * kd
    d["KD"] = k - kd

    change = d["close"].diff()
    up = change.clip(lower=0)
    abs_change = change.abs()
    if caliber["rsi"] == "tdx":
        d["RSI"] = tdx_sma(up, 6, 1) / tdx_sma(abs_change, 6, 1).replace(0, np.nan) * 100
    else:
        d["RSI"] = up.rolling(6, min_periods=1).mean() / abs_change.rolling(6, min_periods=1).mean().replace(0, np.nan) * 100

    d["DIF"] = ema(d["close"], 12) - ema(d["close"], 26)

    hl_range = (d["high"] - d["low"]).replace(0, np.nan)
    if caliber["emv"] == "tdx":
        vol_ratio = d["volume"].rolling(14, min_periods=1).mean() / d["volume"].replace(0, np.nan)
        mid = 100 * (d["high"] + d["low"] - (d["high"] + d["low"]).shift(1)) / (d["high"] + d["low"]).replace(0, np.nan)
        d["EMV"] = (mid * vol_ratio * hl_range / hl_range.rolling(14, min_periods=1).mean()).rolling(14, min_periods=1).mean()
    else:
        mid_move = ((d["high"] + d["low"]) / 2).diff()
        box_ratio = d["volume"].replace(0, np.nan) / hl_range
        d["EMV"] = (mid_move / box_ratio * 1e8).rolling(14, min_periods=1).mean()

    pct = d["close"].pct_change() * 100
    if caliber["lon"] == "tdx":
        d["LON"] = tdx_sma(pct, 10, 1) - tdx_sma(pct, 20, 1)
    else:
        d["LON"] = ema(pct.fillna(0), 10) - ema(pct.fillna(0), 20)

    d["VOL"] = np.log1p(d["volume"].clip(lower=0))
    return d


def add_low_signals(df: pd.DataFrame, factors: list[str], window: int, minp: int, low_q: float) -> pd.DataFrame:
    out = df[["date"]].copy()
    for f in factors:
        x = df[f].astype(float)
        lo = x.rolling(window, min_periods=minp).min()
        hi = x.rolling(window, min_periods=minp).max()
        pos = (x - lo) / (hi - lo).replace(0, np.nan)
        out[f] = (pos <= low_q).fillna(False)
    return out


def carry_to_daily(freq_signals: pd.DataFrame, daily_dates: pd.Series, freq: str) -> pd.DataFrame:
    factors = [c for c in freq_signals.columns if c != "date"]
    daily = pd.DataFrame({"date": daily_dates})
    if freq == "D":
        return daily.merge(freq_signals, on="date", how="left").fillna(False)
    shifted = freq_signals.copy()
    next_dates = []
    daily_values = daily_dates.to_list()
    for dt in shifted["date"]:
        pos = np.searchsorted(daily_values, dt, side="right")
        next_dates.append(daily_values[pos] if pos < len(daily_values) else pd.NaT)
    shifted["date"] = next_dates
    shifted = shifted.dropna(subset=["date"])
    merged = daily.merge(shifted, on="date", how="left").sort_values("date")
    merged[factors] = merged[factors].ffill().fillna(False)
    return merged


def make_daily_state(df: pd.DataFrame, freq: str, caliber: dict, factors: list[str], window: int, minp: int, low_q: float) -> pd.DataFrame:
    bars = add_indicators(aggregate(df, freq), caliber)
    sig = add_low_signals(bars, factors, window, minp, low_q)
    return carry_to_daily(sig, df["date"], freq)


def trades_for_combo(
    df: pd.DataFrame,
    state: pd.DataFrame,
    combo: tuple[str, ...],
    hold_days: int,
    meta: dict,
) -> list[dict]:
    merged = df[["date", "close"]].merge(state[["date", *combo]], on="date", how="left").fillna(False)
    close = merged["close"].to_numpy()
    dates = merged["date"].to_list()
    hits = merged[list(combo)].all(axis=1).to_numpy()
    rows = []
    for i, ok in enumerate(hits):
        if not ok:
            continue
        entry_i = i + 1
        exit_i = entry_i + hold_days
        if exit_i >= len(merged):
            continue
        entry = close[entry_i]
        exit_ = close[exit_i]
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_):
            continue
        ret = exit_ / entry - 1
        rows.append(
            {
                **meta,
                "signal_date": dates[i].date(),
                "entry_date": dates[entry_i].date(),
                "exit_date": dates[exit_i].date(),
                "entry_close": float(entry),
                "exit_close": float(exit_),
                "ret": float(ret),
                "win": bool(ret > 0),
            }
        )
    return rows


def collect_returns_for_state(
    df: pd.DataFrame,
    state: pd.DataFrame,
    combos: list[tuple[str, ...]],
    hold_days_list: list[int],
    meta_base: dict,
    stats: dict[tuple, list[float]],
) -> None:
    factors = [c for c in state.columns if c != "date"]
    merged = df[["date", "close"]].merge(state, on="date", how="left")
    merged[factors] = merged[factors].fillna(False)
    close = merged["close"].astype(float).to_numpy()
    state_bool = {f: merged[f].astype(bool).to_numpy() for f in factors}
    n = len(merged)
    for hold_days in hold_days_list:
        ret = np.full(n, np.nan)
        entry_idx = np.arange(n) + 1
        exit_idx = entry_idx + hold_days
        ok = exit_idx < n
        valid_entry = entry_idx[ok]
        valid_exit = exit_idx[ok]
        entry = close[valid_entry]
        exit_ = close[valid_exit]
        good = np.isfinite(entry) & (entry > 0) & np.isfinite(exit_)
        ret[np.where(ok)[0][good]] = exit_[good] / entry[good] - 1
        valid_ret = np.isfinite(ret)
        for combo in combos:
            hit = np.ones(n, dtype=bool)
            for f in combo:
                hit &= state_bool[f]
            vals = ret[hit & valid_ret]
            if len(vals) == 0:
                continue
            key = (
                meta_base["layer_id"],
                meta_base["caliber"],
                meta_base["freq"],
                "+".join(combo),
                len(combo),
                hold_days,
            )
            stats.setdefault(key, []).extend(vals.astype(float).tolist())


def summarize_stats(stats: dict[tuple, list[float]], key_cols: list[str]) -> pd.DataFrame:
    if not stats:
        return pd.DataFrame()
    rows = []
    for key, vals in stats.items():
        rets = pd.Series(vals, dtype=float)
        rows.append(
            {
                **dict(zip(key_cols, key)),
                "trades": int(len(rets)),
                "wins": int((rets > 0).sum()),
                "win_rate": float((rets > 0).mean()),
                "avg_ret": float(rets.mean()),
                "median_ret": float(rets.median()),
                "compound": float(np.prod(1 + rets)),
                "total_profit_per_100": float((rets * 100).sum()),
                "avg_profit_per_100": float(rets.mean() * 100),
            }
        )
    out = pd.DataFrame(rows)
    out["score_profit"] = out["compound"]
    out["score_balanced"] = out["win_rate"] * 100 + out["avg_ret"] * 1000 + np.log1p(out["trades"])
    return out.sort_values(["score_balanced", "avg_ret", "win_rate", "trades"], ascending=[False, False, False, False])


def rebuild_best_trades(pool: pd.DataFrame, cfg: dict, best: dict) -> pd.DataFrame:
    if not best:
        return pd.DataFrame()
    layer = pd.read_csv(PARAMS / "layers.csv")
    layer_row = layer[layer["layer_id"] == best["layer_id"]].iloc[0].to_dict()
    caliber = {
        "caliber": layer_row["caliber"],
        "kdj": layer_row["kdj"],
        "rsi": layer_row["rsi"],
        "emv": layer_row["emv"],
        "lon": layer_row["lon"],
    }
    combo = tuple(str(best["combo"]).split("+"))
    rows = []
    for symbol in pool["symbol"].astype(str):
        raw = load_raw(symbol)
        name = pool.loc[pool["symbol"].astype(str) == symbol, "display_name"].iloc[0] if "display_name" in pool.columns else symbol
        state = make_daily_state(raw, str(best["freq"]), caliber, cfg["indicators"], int(layer_row["window"]), int(layer_row["minp"]), float(layer_row["low_q"]))
        meta = {
            "layer_id": best["layer_id"],
            "caliber": best["caliber"],
            "symbol": symbol,
            "name": name,
            "freq": best["freq"],
            "combo": best["combo"],
            "combo_size": int(best["combo_size"]),
            "hold_days": int(best["hold_days"]),
            "entry_rule": cfg["strategy_grid"]["entry"],
        }
        rows.extend(trades_for_combo(raw, state, combo, int(best["hold_days"]), meta))
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"])


def simulate_single_position(trades: pd.DataFrame, initial_cash: float = 1000.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    t = t.sort_values(["entry_date", "symbol"])
    cash = initial_cash
    busy_until = pd.Timestamp.min
    rows = []
    for _, r in t.iterrows():
        if r["entry_date"] <= busy_until:
            continue
        before = cash
        cash = cash * (1 + float(r["ret"]))
        busy_until = r["exit_date"]
        rows.append(
            {
                "entry_date": r["entry_date"].date(),
                "exit_date": r["exit_date"].date(),
                "symbol": r["symbol"],
                "name": r.get("name", ""),
                "ret": float(r["ret"]),
                "cash_before": before,
                "cash_after": cash,
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    for path in [QUALITY, PARAMS, COMBOS, TRADES]:
        path.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    pool = pd.read_csv(POOL_PATH, dtype={"symbol": str})
    if "enabled" in pool.columns:
        pool = pool[pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])]
    symbols = pool["symbol"].astype(str).tolist()
    factors = cfg["indicators"]
    combos = []
    for size in cfg["strategy_grid"]["combo_sizes"]:
        combos.extend(itertools.combinations(factors, size))

    clean = clean_report(pool)
    clean.to_csv(QUALITY / "quality.csv", index=False, encoding="utf-8-sig")

    stats: dict[tuple, list[float]] = {}
    layer_rows = []
    for caliber in cfg["search"]["calibers"]:
        for low_q, high_q in zip(cfg["signal"]["low_threshold"], cfg["signal"]["high_threshold"]):
            for minp in cfg["signal"]["min_periods"]:
                layer_id = f"{caliber['caliber']}_low{int(low_q*100)}_min{minp}"
                layer_rows.append(
                    {
                        "layer_id": layer_id,
                        "caliber": caliber["caliber"],
                        "kdj": caliber["kdj"],
                        "rsi": caliber["rsi"],
                        "emv": caliber["emv"],
                        "lon": caliber["lon"],
                        "rank_mode": "range",
                        "window": cfg["signal"]["rolling_window"],
                        "low_q": low_q,
                        "high_q": high_q,
                        "minp": minp,
                    }
                )
                for symbol in symbols:
                    raw = load_raw(symbol)
                    name = pool.loc[pool["symbol"].astype(str) == symbol, "display_name"].iloc[0] if "display_name" in pool.columns else symbol
                    for freq in cfg["frequencies"]:
                        state = make_daily_state(raw, freq, caliber, factors, cfg["signal"]["rolling_window"], minp, low_q)
                        meta = {
                            "layer_id": layer_id,
                            "caliber": caliber["caliber"],
                            "symbol": symbol,
                            "name": name,
                            "freq": freq,
                        }
                        collect_returns_for_state(raw, state, combos, cfg["strategy_grid"]["hold_days"], meta, stats)

    pd.DataFrame(layer_rows).to_csv(PARAMS / "layers.csv", index=False, encoding="utf-8-sig")

    key = ["layer_id", "caliber", "freq", "combo", "combo_size", "hold_days"]
    summary = summarize_stats(stats, key)
    summary.to_csv(COMBOS / "all.csv", index=False, encoding="utf-8-sig")
    min_trades = int(cfg["strategy_grid"]["min_trades_for_ranking"])
    ranked = summary[summary["trades"] >= min_trades].copy()
    ranked = ranked.sort_values(["score_balanced", "avg_ret", "win_rate", "trades"], ascending=[False, False, False, False])
    ranked.to_csv(COMBOS / "ranked.csv", index=False, encoding="utf-8-sig")

    ranked.sort_values(["avg_ret", "win_rate", "trades"], ascending=[False, False, False]).head(100).to_csv(
        COMBOS / "top_avg.csv", index=False, encoding="utf-8-sig"
    )
    ranked.sort_values(["win_rate", "avg_ret", "trades"], ascending=[False, False, False]).head(100).to_csv(
        COMBOS / "top_win.csv", index=False, encoding="utf-8-sig"
    )

    by_hold = ranked.sort_values(["hold_days", "score_balanced"], ascending=[True, False]).groupby("hold_days").head(20)
    by_hold.to_csv(COMBOS / "top_by_hold.csv", index=False, encoding="utf-8-sig")

    by_layer = ranked.groupby("layer_id", as_index=False).head(1)
    by_layer.to_csv(COMBOS / "best_by_layer.csv", index=False, encoding="utf-8-sig")

    best = ranked.iloc[0].to_dict() if not ranked.empty else {}
    if best:
        best_trades = rebuild_best_trades(pool, cfg, best)
        best_trades.to_csv(TRADES / "best_trades.csv", index=False, encoding="utf-8-sig")
        simulate_single_position(best_trades).to_csv(TRADES / "account.csv", index=False, encoding="utf-8-sig")
    (COMBOS / "summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    run()
