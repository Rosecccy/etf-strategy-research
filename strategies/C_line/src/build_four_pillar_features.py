from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "etf"
S_RAW = ROOT.parent / "S" / "raw" / "etf"
S_POOL = ROOT.parent / "S" / "raw" / "pool.csv"
POOL = ROOT / "raw" / "pool.csv"
FACTOR_ROOT = ROOT / "factors"
TURNOVER = FACTOR_ROOT / "raw" / "turnover"
VALUATION = FACTOR_ROOT / "raw" / "valuation"
OUT = FACTOR_ROOT / "panel.parquet"


def rolling_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def read_valuation() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for pe_path in sorted(VALUATION.glob("*_pe.csv")):
        name = pe_path.stem.removesuffix("_pe")
        pb_path = VALUATION / f"{name}_pb.csv"
        if not pb_path.exists():
            continue
        pe = pd.read_csv(pe_path)
        pb = pd.read_csv(pb_path)
        pe = pe.rename(columns={"日期": "date", "滚动市盈率": "pe"})[["date", "pe"]]
        pb = pb.rename(columns={"日期": "date", "市净率": "pb"})[["date", "pb"]]
        data = pe.merge(pb, on="date", how="outer")
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        for column in ("pe", "pb"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
            data[f"{column}_rank"] = rolling_rank(data[column], 1260, 252)
        data["index_name"] = name
        pieces.append(data)
    all_values = pd.concat(pieces, ignore_index=True)
    daily = (
        all_values.groupby("date")
        .agg(
            market_pe_rank=("pe_rank", "median"),
            market_pb_rank=("pb_rank", "median"),
            valuation_index_count=("index_name", "nunique"),
        )
        .sort_index()
        .reset_index()
    )
    daily["valuation_cheap"] = 1.0 - daily[["market_pe_rank", "market_pb_rank"]].mean(axis=1)
    return daily


def chip_features(close: np.ndarray, typical: np.ndarray, volume: np.ndarray, turnover: np.ndarray) -> dict[str, np.ndarray]:
    n = len(close)
    result = {
        "chip_profit60": np.full(n, np.nan),
        "chip_cost_gap60": np.full(n, np.nan),
        "chip_overhead5_60": np.full(n, np.nan),
        "chip_support5_60": np.full(n, np.nan),
        "chip_concentration60": np.full(n, np.nan),
    }
    for end in range(n):
        start = max(0, end - 119)
        prices = typical[start : end + 1]
        vols = volume[start : end + 1]
        ages = np.arange(end - start, -1, -1, dtype=float)
        decay = np.exp(-ages / 35.0)
        local_turnover = np.nan_to_num(turnover[start : end + 1], nan=0.0) / 100.0
        survival = np.ones_like(decay)
        if len(local_turnover) > 1:
            future_survival = np.cumprod(np.clip(1.0 - local_turnover[::-1], 0.0, 1.0))[::-1]
            survival[:-1] = future_survival[1:]
        weights = np.nan_to_num(vols, nan=0.0) * decay * survival
        valid = np.isfinite(prices) & (weights > 0)
        if valid.sum() < 20 or not np.isfinite(close[end]):
            continue
        prices = prices[valid]
        weights = weights[valid]
        total = weights.sum()
        current = close[end]
        average_cost = float(np.average(prices, weights=weights))
        result["chip_profit60"][end] = float(weights[prices <= current].sum() / total)
        result["chip_cost_gap60"][end] = current / average_cost - 1.0
        result["chip_overhead5_60"][end] = float(weights[(prices > current) & (prices <= current * 1.05)].sum() / total)
        result["chip_support5_60"][end] = float(weights[(prices < current) & (prices >= current * 0.95)].sum() / total)
        variance = float(np.average((prices / average_cost - 1.0) ** 2, weights=weights))
        result["chip_concentration60"][end] = float(np.sqrt(max(variance, 0.0)))
    return result


def build_symbol(symbol: str, category: str) -> pd.DataFrame:
    source_path = RAW / f"{symbol}.csv"
    if not source_path.exists():
        source_path = S_RAW / f"{symbol}.csv"
    data = pd.read_csv(source_path, dtype={"symbol": str})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "close", "volume"]).sort_values("date").drop_duplicates("date")
    turnover_path = TURNOVER / f"{symbol}.csv"
    if turnover_path.exists():
        turnover = pd.read_csv(turnover_path)
        turnover["date"] = pd.to_datetime(turnover["date"], errors="coerce")
        data = data.merge(turnover[["date", "turnover_rate"]], on="date", how="left", suffixes=("", "_download"))
        if "turnover_rate_download" in data:
            data["turnover_rate"] = pd.to_numeric(data["turnover_rate_download"], errors="coerce").combine_first(
                pd.to_numeric(data.get("turnover_rate"), errors="coerce")
            )
            data = data.drop(columns=["turnover_rate_download"])
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    amount = pd.to_numeric(data["amount"], errors="coerce")
    turn = pd.to_numeric(data["turnover_rate"], errors="coerce")
    ret1 = close.pct_change()
    log_volume = np.log1p(volume)
    volume_median20 = volume.rolling(20, min_periods=10).median()
    amount_median20 = amount.rolling(20, min_periods=10).median()
    data["ret1"] = ret1
    data["ret5"] = close.pct_change(5)
    data["ret20"] = close.pct_change(20)
    data["volume_ratio20"] = volume / volume_median20.replace(0, np.nan)
    data["amount_ratio20"] = amount / amount_median20.replace(0, np.nan)
    # A fixed-scale, causal fallback keeps LOF histories usable when the source omits
    # free-float turnover. It never uses future shares or future turnover observations.
    turnover_proxy = (2.0 * data["volume_ratio20"]).clip(0.05, 20.0)
    turn_model = turn.combine_first(turnover_proxy)
    data["turnover_rate_model"] = turn_model
    data["turnover_is_proxy"] = turn.isna() & turn_model.notna()
    data["price_volume_corr20"] = ret1.rolling(20, min_periods=15).corr(log_volume.diff())
    data["signed_volume20"] = (np.sign(ret1.fillna(0)) * data["volume_ratio20"]).rolling(20, min_periods=15).mean()
    data["sell_pressure"] = (-ret1).clip(lower=0) * np.sqrt(data["volume_ratio20"].clip(0, 10))
    data["buy_pressure"] = ret1.clip(lower=0) * np.sqrt(data["volume_ratio20"].clip(0, 10))
    data["obv_balance20"] = (np.sign(ret1.fillna(0)) * volume).rolling(20, min_periods=15).sum() / volume.rolling(20, min_periods=15).sum()
    data["turnover_rank120"] = rolling_rank(turn_model, 120, 40)
    data["turnover_ratio20"] = turn_model / turn_model.rolling(20, min_periods=10).median().replace(0, np.nan)
    data["turnover_change20"] = turn_model / turn_model.rolling(60, min_periods=20).median().replace(0, np.nan) - 1.0

    typical = ((high + low + close) / 3.0).to_numpy(dtype=float)
    chips = chip_features(
        close.to_numpy(dtype=float),
        typical,
        volume.to_numpy(dtype=float),
        turn_model.to_numpy(dtype=float),
    )
    for name, values in chips.items():
        data[name] = values
    data["chip_trapped60"] = 1.0 - data["chip_profit60"]
    data["symbol"] = symbol
    data["category"] = category
    return data


def main() -> None:
    pool = pd.read_csv(POOL, dtype={"symbol": str})
    pool = pool.query("enabled == True and clean_pool == True").copy()
    pool["symbol"] = pool["symbol"].str.zfill(6)
    s_trades = pd.read_csv(ROOT / "fit" / "trigger_safe_upgrades" / "s_trades.csv", dtype={"symbol": str})
    needed = set(s_trades["symbol"].astype(str).str.zfill(6)) - set(pool["symbol"])
    if needed:
        s_pool = pd.read_csv(S_POOL, dtype={"symbol": str})
        s_pool["symbol"] = s_pool["symbol"].str.zfill(6)
        additions = s_pool[s_pool["symbol"].isin(needed)][["symbol", "category"]].copy()
        additions["enabled"] = True
        additions["clean_pool"] = True
        pool = pd.concat([pool, additions], ignore_index=True)
    frames = [build_symbol(row.symbol, row.category) for row in pool.itertuples(index=False)]
    panel = pd.concat(frames, ignore_index=True)
    valuation = read_valuation()
    panel = pd.merge_asof(
        panel.sort_values("date"),
        valuation.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    rank_columns = [
        "volume_ratio20",
        "amount_ratio20",
        "signed_volume20",
        "sell_pressure",
        "buy_pressure",
        "obv_balance20",
        "turnover_rank120",
        "turnover_ratio20",
        "chip_profit60",
        "chip_cost_gap60",
        "chip_overhead5_60",
        "chip_support5_60",
    ]
    for column in rank_columns:
        panel[f"{column}_market_rank"] = panel.groupby("date")[column].rank(pct=True)
        panel[f"{column}_category_rank"] = panel.groupby(["date", "category"])[column].rank(pct=True)

    panel["pillar_volume_reversal"] = panel[
        ["sell_pressure_market_rank", "volume_ratio20_category_rank"]
    ].mean(axis=1)
    panel["pillar_volume_momentum"] = panel[
        ["buy_pressure_market_rank", "signed_volume20_market_rank", "obv_balance20_market_rank"]
    ].mean(axis=1)
    panel["pillar_turnover"] = panel[
        ["turnover_rank120", "turnover_ratio20_market_rank", "turnover_ratio20_category_rank"]
    ].mean(axis=1)
    panel["pillar_valuation"] = panel["valuation_cheap"]
    panel["pillar_chip_reversal"] = panel[
        ["chip_trapped60", "chip_support5_60_market_rank"]
    ].mean(axis=1)
    panel["pillar_chip_momentum"] = panel[
        ["chip_profit60", "chip_cost_gap60_market_rank", "chip_support5_60_market_rank"]
    ].mean(axis=1) - 0.25 * panel["chip_overhead5_60_market_rank"]
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    feature_columns = [column for column in panel.columns if column.startswith("pillar_")]
    summary = {
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "start": str(panel["date"].min().date()),
        "end": str(panel["date"].max().date()),
        "turnover_coverage": float(panel["turnover_rate"].notna().mean()),
        "valuation_coverage": float(panel["valuation_cheap"].notna().mean()),
        "chip_coverage": float(panel["chip_profit60"].notna().mean()),
        "features": feature_columns,
    }
    (FACTOR_ROOT / "panel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
