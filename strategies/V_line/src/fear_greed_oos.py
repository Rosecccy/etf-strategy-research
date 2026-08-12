from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
C_ROOT = ROOT.parent / "C"
C_POOL = C_ROOT / "raw" / "pool.csv"
CLEAN_RAW = ROOT / "out" / "research_30_extrema_cleaned_v2" / "raw"
LIVE_RAW = C_ROOT / "raw" / "etf"
OFFICIAL_RAW = (
    ROOT
    / "out"
    / "research_30_extrema_cleaned_v2"
    / "market_sentiment"
    / "data"
    / "official_raw"
)
BASE_TRADES = ROOT / "out" / "ma120_trade_log.csv"
OUT = ROOT / "out" / "fear_greed"

COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
INITIAL_CAPITAL = 10_000.0
MIN_HISTORY_DAYS = 60
RANK_WINDOW = 756
MIN_TRAIN_TRADES = 30


@dataclass(frozen=True)
class Overlay:
    key: str
    family: str
    threshold: float
    recovery_change: float | None = None


@dataclass(frozen=True)
class ExitRule:
    key: str
    max_hold: int
    greed_threshold: float | None


def prior_rank(series: pd.Series, window: int = RANK_WINDOW, min_periods: int = MIN_HISTORY_DAYS) -> pd.Series:
    """Rank today's value against prior observations only."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for index, current in enumerate(values):
        if not np.isfinite(current):
            continue
        history = values[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if len(history) < min_periods:
            continue
        result[index] = float((history <= current).mean())
    return pd.Series(result, index=series.index, dtype=float)


def load_clean_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pd.read_csv(C_POOL, dtype={"symbol": str}, encoding="utf-8-sig")
    pool["symbol"] = pool["symbol"].astype(str).str.zfill(6)
    if "enabled" in pool:
        pool = pool[pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    quality_path = C_ROOT / "raw" / "quality.json"
    if not quality_path.exists():
        raise FileNotFoundError("Canonical ETF quality manifest is missing.")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if not bool(quality.get("passed")):
        raise RuntimeError("Canonical ETF quality gate failed; D-line execution is blocked.")
    approved = {
        str(symbol).zfill(6)
        for symbol in quality.get("approved_symbols", [])
    }
    pool = pool[pool["symbol"].isin(approved)].copy()
    frames: list[pd.DataFrame] = []
    for symbol in pool["symbol"]:
        path = LIVE_RAW / f"{symbol}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
        frame["symbol"] = symbol
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount"):
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
        frame = (
            frame.dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )
        frame = frame[frame["close"] > 0].copy()
        frames.append(frame[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]])
    if not frames:
        raise FileNotFoundError(f"No clean ETF files found under {CLEAN_RAW}")
    return pd.concat(frames, ignore_index=True), pool


def make_internal_fear(raw: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for symbol, group in raw.groupby("symbol", sort=False):
        data = group.sort_values("date").copy()
        close = data["close"]
        previous_close = close.shift(1)
        ret1 = close.pct_change()
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        data["ret1"] = ret1
        data["vol20"] = ret1.rolling(20, min_periods=20).std() * np.sqrt(252)
        data["range20"] = (true_range / previous_close).rolling(5, min_periods=5).mean()
        data["volume_ratio20"] = data["volume"] / data["volume"].rolling(20, min_periods=20).median()
        data["sell_pressure"] = (-ret1).clip(lower=0) * np.sqrt(data["volume_ratio20"].clip(0, 10))
        data["below_ma60"] = close < close.rolling(60, min_periods=60).mean()
        data["drawdown60"] = 1 - close / close.rolling(60, min_periods=60).max()
        data["new_low60"] = close <= close.rolling(60, min_periods=60).min()
        data["new_high60"] = close >= close.rolling(60, min_periods=60).max()
        pieces.append(data)
    panel = pd.concat(pieces, ignore_index=True)

    daily = (
        panel.groupby("date")
        .agg(
            symbols=("symbol", "nunique"),
            realized_vol=("vol20", "median"),
            range_stress=("range20", "median"),
            down_breadth=("ret1", lambda values: float((values < 0).mean())),
            below_ma60=("below_ma60", "mean"),
            drawdown60=("drawdown60", "median"),
            sell_pressure=("sell_pressure", "median"),
            new_low60=("new_low60", "mean"),
            new_high60=("new_high60", "mean"),
        )
        .sort_index()
    )
    daily["down_breadth5"] = daily["down_breadth"].rolling(5, min_periods=3).mean()
    daily["new_low_pressure"] = daily["new_low60"] - daily["new_high60"]

    source_columns = {
        "fear_volatility": "realized_vol",
        "fear_range": "range_stress",
        "fear_breadth": "down_breadth5",
        "fear_trend": "below_ma60",
        "fear_drawdown": "drawdown60",
        "fear_flow": "sell_pressure",
        "fear_new_low": "new_low_pressure",
    }
    for output, source in source_columns.items():
        daily[output] = prior_rank(daily[source]) * 100
    components = list(source_columns)
    daily["fear_core"] = daily[components].mean(axis=1, skipna=True)
    daily["core_components"] = daily[components].notna().sum(axis=1)
    daily.loc[daily["core_components"] < 4, "fear_core"] = np.nan
    return daily.reset_index()


def load_official() -> pd.DataFrame:
    qvix_path = OFFICIAL_RAW / "qvix_50etf.csv"
    north_path = OFFICIAL_RAW / "northbound_hist.csv"
    official = pd.DataFrame(columns=["date"])
    if qvix_path.exists():
        qvix = pd.read_csv(qvix_path, encoding="utf-8-sig")
        qvix["date"] = pd.to_datetime(qvix["date"], errors="coerce")
        qvix["qvix_close"] = pd.to_numeric(qvix["close"], errors="coerce")
        qvix = qvix[["date", "qvix_close"]].dropna().drop_duplicates("date").sort_values("date")
        qvix["fear_qvix"] = prior_rank(qvix["qvix_close"], min_periods=40) * 100
        official = qvix
    if north_path.exists():
        north = pd.read_csv(north_path, encoding="utf-8-sig")
        north["date"] = pd.to_datetime(north["日期"], errors="coerce")
        north["north_net"] = pd.to_numeric(north["当日成交净买额"], errors="coerce")
        north = north[["date", "north_net"]].dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        # The exchange publishes completed margin/flow information after the session.
        # Shift one row so the score never uses a value unavailable at decision time.
        north["north_net_available"] = north["north_net"].shift(1)
        north["fear_north"] = prior_rank(-north["north_net_available"], min_periods=40) * 100
        official = north if official.empty else official.merge(north, on="date", how="outer")
    return official.sort_values("date")


def build_sentiment(raw: pd.DataFrame) -> pd.DataFrame:
    sentiment = make_internal_fear(raw)
    official = load_official()
    if not official.empty:
        sentiment = sentiment.merge(official, on="date", how="left")
    for column in ("fear_qvix", "fear_north"):
        if column not in sentiment:
            sentiment[column] = np.nan
    # QVIX is the only official component included in the formal score.
    # Northbound is retained for audit because its disclosure regime changed.
    sentiment["fear_enhanced"] = np.where(
        sentiment["fear_qvix"].notna(),
        0.70 * sentiment["fear_core"] + 0.30 * sentiment["fear_qvix"],
        sentiment["fear_core"],
    )
    sentiment["fear_change3"] = sentiment["fear_enhanced"].diff(3)
    sentiment["fear_peak5"] = sentiment["fear_enhanced"].rolling(5, min_periods=1).max()
    sentiment["greed_enhanced"] = 100 - sentiment["fear_enhanced"]
    sentiment["regime"] = pd.cut(
        sentiment["fear_enhanced"],
        bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=["极度贪婪", "贪婪", "中性", "恐惧", "极度恐惧"],
    ).astype("string")
    return sentiment.sort_values("date").reset_index(drop=True)


def build_symbol_fear(raw: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol, group in raw.groupby("symbol", sort=False):
        data = group.sort_values("date").copy()
        close = data["close"]
        ret1 = close.pct_change()
        drawdown = 1 - close / close.rolling(60, min_periods=60).max()
        position = (close - close.rolling(60, min_periods=60).min()) / (
            close.rolling(60, min_periods=60).max() - close.rolling(60, min_periods=60).min()
        ).replace(0, np.nan)
        volume_ratio = data["volume"] / data["volume"].rolling(20, min_periods=20).median()
        sell_pressure = (-ret1).clip(lower=0) * np.sqrt(volume_ratio.clip(0, 10))
        volatility = ret1.rolling(20, min_periods=20).std()
        data["symbol_fear_drawdown"] = prior_rank(drawdown, min_periods=40) * 100
        data["symbol_fear_position"] = prior_rank(1 - position, min_periods=40) * 100
        data["symbol_fear_pressure"] = prior_rank(sell_pressure, min_periods=40) * 100
        data["symbol_fear_volatility"] = prior_rank(volatility, min_periods=40) * 100
        data["symbol_fear"] = data[
            [
                "symbol_fear_drawdown",
                "symbol_fear_position",
                "symbol_fear_pressure",
                "symbol_fear_volatility",
            ]
        ].mean(axis=1, skipna=True)
        parts.append(data[["date", "symbol", "symbol_fear"]])
    return pd.concat(parts, ignore_index=True)


def rebuild_trade_prices(trades: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    result = trades.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    for column in ("entry_date", "exit_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    price = raw.set_index(["symbol", "date"])["close"]
    latest = raw.sort_values("date").groupby("symbol", sort=False).tail(1).set_index("symbol")
    result["entry_close"] = [
        price.get((symbol, date), old)
        for symbol, date, old in zip(result["symbol"], result["entry_date"], result["entry_close"])
    ]
    result["exit_close"] = [
        price.get((symbol, date), old) if pd.notna(date) else np.nan
        for symbol, date, old in zip(result["symbol"], result["exit_date"], result["exit_close"])
    ]
    result["mark_close"] = result["symbol"].map(latest["close"])
    closed = result["status"].eq("closed") & result["exit_close"].notna()
    result.loc[closed, "return_rate"] = (
        result.loc[closed, "exit_close"] / result.loc[closed, "entry_close"] - 1
    )
    result.loc[closed, "pnl_cny"] = result.loc[closed, "return_rate"] * 100
    result["unrealized_return"] = np.where(
        result["status"].eq("open"),
        result["mark_close"] / result["entry_close"] - 1,
        np.nan,
    )
    return result


def attach_sentiment(
    trades: pd.DataFrame,
    sentiment: pd.DataFrame,
    symbol_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    result = trades.copy()
    calendar = pd.DatetimeIndex(sentiment["date"].dropna().sort_values().unique())
    signal_dates = []
    for entry_date in result["entry_date"]:
        position = int(calendar.searchsorted(entry_date, side="left")) - 1
        signal_dates.append(calendar[position] if position >= 0 else pd.NaT)
    result["signal_date"] = signal_dates
    columns = [
        "date",
        "fear_core",
        "fear_qvix",
        "fear_north",
        "fear_enhanced",
        "fear_change3",
        "fear_peak5",
        "greed_enhanced",
        "regime",
    ]
    result = result.merge(
        sentiment[columns].rename(columns={"date": "signal_date"}),
        on="signal_date",
        how="left",
    )
    result = result.merge(
        symbol_sentiment.rename(columns={"date": "signal_date"}),
        on=["signal_date", "symbol"],
        how="left",
    )
    result["buy_quality"] = 0.5 * result["fear_enhanced"] + 0.5 * result["symbol_fear"]
    return result


def overlay_grid() -> list[Overlay]:
    grid = [Overlay("baseline", "baseline", 0)]
    for threshold in range(35, 76, 5):
        grid.append(Overlay(f"core_ge_{threshold}", "core", float(threshold)))
        grid.append(Overlay(f"enhanced_ge_{threshold}", "enhanced", float(threshold)))
    for threshold in range(40, 81, 10):
        grid.append(Overlay(f"qvix_ge_{threshold}", "qvix", float(threshold)))
    for threshold in range(50, 81, 5):
        for change in (-2.5, 0.0, 2.5):
            suffix = str(change).replace("-", "m").replace(".", "p")
            grid.append(
                Overlay(
                    f"recovery_peak_{threshold}_change_le_{suffix}",
                    "recovery",
                    float(threshold),
                    float(change),
                )
            )
    return grid


def exit_grid() -> list[ExitRule]:
    return [
        ExitRule(
            key=f"hold_{hold}_greed_{'off' if greed is None else int(greed)}",
            max_hold=hold,
            greed_threshold=greed,
        )
        for hold in (20, 40, 60, 90, 120)
        for greed in (None, 60.0, 70.0, 80.0)
    ]


def overlay_mask(trades: pd.DataFrame, overlay: Overlay) -> pd.Series:
    if overlay.family == "baseline":
        return pd.Series(True, index=trades.index)
    if overlay.family == "core":
        return trades["fear_core"] >= overlay.threshold
    if overlay.family == "enhanced":
        return trades["fear_enhanced"] >= overlay.threshold
    if overlay.family == "qvix":
        return trades["fear_qvix"] >= overlay.threshold
    if overlay.family == "recovery":
        return (trades["fear_peak5"] >= overlay.threshold) & (
            trades["fear_change3"] <= float(overlay.recovery_change)
        )
    raise ValueError(overlay.family)


def trade_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    closed = frame[frame["status"].eq("closed") & frame["return_rate"].notna()]
    open_lots = frame[frame["status"].eq("open") & frame["unrealized_return"].notna()]
    realized = float(closed["return_rate"].sum() * 100)
    unrealized = float(open_lots["unrealized_return"].sum() * 100)
    year_stats = closed.groupby("test_year")["return_rate"].agg(
        trades="size",
        win_rate=lambda values: float((values > 0).mean()),
        avg_return="mean",
    )
    return {
        "signals": int(len(frame)),
        "closed": int(len(closed)),
        "open": int(len(open_lots)),
        "wins": int((closed["return_rate"] > 0).sum()),
        "win_rate": float((closed["return_rate"] > 0).mean()) if len(closed) else np.nan,
        "avg_return": float(closed["return_rate"].mean()) if len(closed) else np.nan,
        "median_return": float(closed["return_rate"].median()) if len(closed) else np.nan,
        "realized_pnl_per_100": realized,
        "unrealized_pnl_per_100": unrealized,
        "mark_to_market_pnl_per_100": realized + unrealized,
        "covered_years": int(len(year_stats)),
        "worst_year_win_rate": float(year_stats["win_rate"].min()) if len(year_stats) else np.nan,
        "year_win_rate_std": float(year_stats["win_rate"].std(ddof=0)) if len(year_stats) else np.nan,
    }


def apply_exit_rule(
    trades: pd.DataFrame,
    rule: ExitRule,
    raw: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    result = trades.copy()
    result["entry_date"] = pd.to_datetime(result["entry_date"], errors="coerce")
    result["exit_date"] = pd.to_datetime(result["exit_date"], errors="coerce")
    price_by_symbol = {
        symbol: group.sort_values("date").set_index("date")["close"]
        for symbol, group in raw.groupby("symbol", sort=False)
    }
    greed = sentiment.set_index("date")["greed_enhanced"].sort_index()
    latest_date = pd.Timestamp(raw["date"].max())
    rows: list[dict] = []
    for _, trade in result.iterrows():
        row = trade.to_dict()
        symbol = str(trade["symbol"]).zfill(6)
        prices = price_by_symbol.get(symbol)
        if prices is None or pd.isna(trade["entry_date"]):
            continue
        dates = pd.DatetimeIndex(prices.index)
        entry_position = int(dates.searchsorted(pd.Timestamp(trade["entry_date"]), side="left"))
        if entry_position >= len(dates):
            continue
        candidates: list[tuple[int, str]] = []
        original_exit = pd.to_datetime(trade["exit_date"], errors="coerce")
        if pd.notna(original_exit):
            original_position = int(dates.searchsorted(original_exit, side="left"))
            if original_position < len(dates):
                candidates.append((original_position, "technical_sell"))
        max_position = entry_position + rule.max_hold
        if max_position < len(dates):
            candidates.append((max_position, "max_hold"))
        if rule.greed_threshold is not None:
            start_signal = entry_position + 7
            for signal_position in range(start_signal, len(dates) - 1):
                signal_date = dates[signal_position]
                score = greed.get(signal_date, np.nan)
                if pd.notna(score) and float(score) >= rule.greed_threshold:
                    candidates.append((signal_position + 1, "greed_exit"))
                    break
        if candidates:
            exit_position, exit_reason = min(candidates, key=lambda item: item[0])
            exit_date = dates[exit_position]
            exit_close = float(prices.iloc[exit_position])
            row.update(
                {
                    "exit_date": exit_date,
                    "exit_close": exit_close,
                    "status": "closed",
                    "exit_reason": exit_reason,
                    "return_rate": exit_close / float(row["entry_close"]) - 1,
                    "unrealized_return": np.nan,
                }
            )
            row["pnl_cny"] = row["return_rate"] * 100
        else:
            mark_close = float(prices.iloc[-1])
            row.update(
                {
                    "exit_date": pd.NaT,
                    "exit_close": np.nan,
                    "status": "open",
                    "exit_reason": "open",
                    "mark_close": mark_close,
                    "unrealized_return": mark_close / float(row["entry_close"]) - 1,
                }
            )
        row["exit_key"] = rule.key
        row["max_hold"] = rule.max_hold
        row["greed_threshold"] = rule.greed_threshold
        row["mark_date"] = latest_date
        rows.append(row)
    return pd.DataFrame(rows)


def practical_grid_metrics(
    trades: pd.DataFrame,
    overlays: list[Overlay],
    exits: list[ExitRule],
    adjusted_by_exit: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for exit_rule in exits:
        adjusted = adjusted_by_exit[exit_rule.key]
        for overlay in overlays:
            selected = adjusted[overlay_mask(adjusted, overlay).fillna(False)].copy()
            rows.append(
                {
                    "overlay": overlay.key,
                    "family": overlay.family,
                    "threshold": overlay.threshold,
                    "recovery_change": overlay.recovery_change,
                    "exit_key": exit_rule.key,
                    "max_hold": exit_rule.max_hold,
                    "greed_threshold": exit_rule.greed_threshold,
                    **trade_metrics(selected),
                }
            )
    result = pd.DataFrame(rows)
    eligible = result["closed"] >= MIN_TRAIN_TRADES
    result["robust_score"] = np.where(
        eligible,
        0.50 * result["win_rate"].fillna(0)
        + 0.30 * np.clip(result["avg_return"].fillna(0), -0.20, 0.40)
        + 0.10 * result["worst_year_win_rate"].fillna(0)
        + 0.10 * np.minimum(result["closed"] / 100, 1.0)
        - 0.08 * result["year_win_rate_std"].fillna(1),
        -np.inf,
    )
    return result.sort_values(
        ["robust_score", "win_rate", "avg_return", "closed"],
        ascending=[False, False, False, False],
    )


def grid_metrics(trades: pd.DataFrame, overlays: list[Overlay]) -> pd.DataFrame:
    rows = []
    for overlay in overlays:
        selected = trades[overlay_mask(trades, overlay).fillna(False)].copy()
        rows.append(
            {
                "overlay": overlay.key,
                "family": overlay.family,
                "threshold": overlay.threshold,
                "recovery_change": overlay.recovery_change,
                **trade_metrics(selected),
            }
        )
    result = pd.DataFrame(rows)
    eligible = result["closed"] >= MIN_TRAIN_TRADES
    result["robust_score"] = np.where(
        eligible,
        0.55 * result["win_rate"].fillna(0)
        + 0.25 * np.clip(result["avg_return"].fillna(0), -0.20, 0.40)
        + 0.10 * result["worst_year_win_rate"].fillna(0)
        + 0.10 * np.minimum(result["closed"] / 100, 1.0)
        - 0.08 * result["year_win_rate_std"].fillna(1),
        -np.inf,
    )
    return result.sort_values(
        ["robust_score", "win_rate", "avg_return", "closed"],
        ascending=[False, False, False, False],
    )


def choose_by_prior_years(history: pd.DataFrame, overlays: list[Overlay]) -> Overlay:
    if history["test_year"].nunique() < 2:
        return overlays[0]
    metrics = grid_metrics(history, overlays)
    baseline = metrics[metrics["overlay"].eq("baseline")].iloc[0]
    eligible = metrics[
        metrics["family"].isin(["baseline", "core"])
        &
        (metrics["closed"] >= MIN_TRAIN_TRADES)
        & (metrics["covered_years"] >= 2)
        & (metrics["closed"] >= baseline["closed"] * 0.60)
        & (metrics["win_rate"] >= baseline["win_rate"] - 0.02)
        & (metrics["avg_return"] >= baseline["avg_return"] - 0.01)
    ]
    if eligible.empty:
        return overlays[0]
    core = eligible[eligible["family"].eq("core")].sort_values("threshold")
    stable_keys: list[str] = []
    for _, row in core.iterrows():
        threshold = float(row["threshold"])
        neighbors = core[core["threshold"].isin([threshold - 5, threshold, threshold + 5])]
        if len(neighbors) < 3:
            continue
        if (
            neighbors["win_rate"].min() >= baseline["win_rate"] - 0.02
            and neighbors["avg_return"].min() >= baseline["avg_return"] - 0.01
        ):
            stable_keys.append(str(row["overlay"]))
    stable = eligible[eligible["overlay"].isin(stable_keys)]
    winner = stable.iloc[0] if not stable.empty else baseline
    key = str(winner["overlay"])
    return next(item for item in overlays if item.key == key)


def rolling_overlay(trades: pd.DataFrame, overlays: list[Overlay]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts: list[pd.DataFrame] = []
    rows: list[dict] = []
    years = sorted(int(year) for year in trades["test_year"].dropna().unique())
    for year in years:
        history = trades[trades["test_year"] < year].copy()
        chosen = choose_by_prior_years(history, overlays)
        current = trades[trades["test_year"].eq(year)].copy()
        selected = current[overlay_mask(current, chosen).fillna(False)].copy()
        selected["overlay"] = chosen.key
        selected_parts.append(selected)
        rows.append(
            {
                "test_year": year,
                "overlay": chosen.key,
                "family": chosen.family,
                "threshold": chosen.threshold,
                "prior_years": int(history["test_year"].nunique()),
                "prior_closed": int((history["status"] == "closed").sum()),
                **trade_metrics(selected),
            }
        )
    result = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    return result, pd.DataFrame(rows)


def rolling_practical(
    trades: pd.DataFrame,
    overlays: list[Overlay],
    exits: list[ExitRule],
    adjusted_by_exit: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts: list[pd.DataFrame] = []
    rows: list[dict] = []
    years = sorted(int(year) for year in trades["test_year"].dropna().unique())
    for year in years:
        history_years = [item for item in years if item < year]
        chosen_exit = next(item for item in exits if item.key == "hold_40_greed_off")
        if len(history_years) < 2:
            chosen_overlay = overlays[0]
        else:
            history = adjusted_by_exit[chosen_exit.key]
            history = history[history["test_year"].isin(history_years)].copy()
            chosen_overlay = choose_by_prior_years(history, overlays)
        current = adjusted_by_exit[chosen_exit.key]
        current = current[current["test_year"].eq(year)].copy()
        selected = current[overlay_mask(current, chosen_overlay).fillna(False)].copy()
        selected["overlay"] = chosen_overlay.key
        selected["exit_key"] = chosen_exit.key
        selected_parts.append(selected)
        rows.append(
            {
                "test_year": year,
                "overlay": chosen_overlay.key,
                "exit_key": chosen_exit.key,
                "max_hold": chosen_exit.max_hold,
                "greed_threshold": chosen_exit.greed_threshold,
                "prior_years": len(history_years),
                **trade_metrics(selected),
            }
        )
    result = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    return result, pd.DataFrame(rows)


def commission(notional: float) -> float:
    return max(MIN_COMMISSION, notional * COMMISSION_RATE)


def apply_max_hold(trades: pd.DataFrame, raw: pd.DataFrame, max_hold: int | None) -> pd.DataFrame:
    if max_hold is None:
        return trades.copy()
    result = trades.copy()
    calendars = {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in raw.groupby("symbol", sort=False)
    }
    exit_dates: list[pd.Timestamp | pd.NaT] = []
    exit_prices: list[float] = []
    statuses: list[str] = []
    returns: list[float] = []
    for _, trade in result.iterrows():
        data = calendars[str(trade["symbol"])]
        dates = pd.DatetimeIndex(data["date"])
        entry_date = pd.Timestamp(trade["entry_date"])
        entry_position = int(dates.searchsorted(entry_date, side="left"))
        forced_position = entry_position + max_hold
        forced_date = dates[forced_position] if forced_position < len(dates) else pd.NaT
        technical_date = pd.to_datetime(trade["exit_date"], errors="coerce")
        if pd.notna(technical_date) and (pd.isna(forced_date) or technical_date <= forced_date):
            exit_date = pd.Timestamp(technical_date)
        else:
            exit_date = pd.Timestamp(forced_date) if pd.notna(forced_date) else pd.NaT
        if pd.notna(exit_date):
            position = int(dates.searchsorted(exit_date, side="left"))
            exit_price = float(data.iloc[position]["close"]) if position < len(data) else np.nan
            status = "closed"
            return_rate = exit_price / float(trade["entry_close"]) - 1
        else:
            exit_price = np.nan
            status = "open"
            return_rate = np.nan
        exit_dates.append(exit_date)
        exit_prices.append(exit_price)
        statuses.append(status)
        returns.append(return_rate)
    result["exit_date"] = exit_dates
    result["exit_close"] = exit_prices
    result["status"] = statuses
    result["return_rate"] = returns
    result["pnl_cny"] = result["return_rate"] * 100
    result["unrealized_return"] = np.where(
        result["status"].eq("open"),
        result["mark_close"] / result["entry_close"] - 1,
        np.nan,
    )
    return result


def portfolio_backtest(
    candidates: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    max_positions: int,
    ranking: str,
) -> tuple[pd.DataFrame, dict]:
    candidates = candidates.copy()
    candidates["entry_date"] = pd.to_datetime(candidates["entry_date"])
    candidates["exit_date"] = pd.to_datetime(candidates["exit_date"], errors="coerce")
    liquidity = pool.set_index("symbol")["amount20"].apply(pd.to_numeric, errors="coerce").fillna(0).to_dict()
    candidates["liquidity"] = candidates["symbol"].map(liquidity).fillna(0)
    if ranking == "fear":
        candidates = candidates.sort_values(
            ["entry_date", "buy_quality", "liquidity", "symbol"],
            ascending=[True, False, False, True],
        )
    else:
        candidates = candidates.sort_values(
            ["entry_date", "liquidity", "symbol"],
            ascending=[True, False, True],
        )

    price_map = raw.set_index(["symbol", "date"])["close"]
    latest_date = pd.Timestamp(raw["date"].max())
    latest_price = raw.sort_values("date").groupby("symbol", sort=False).tail(1).set_index("symbol")["close"]
    dates = sorted(set(candidates["entry_date"].dropna()) | set(candidates["exit_date"].dropna()))
    by_entry = {date: group for date, group in candidates.groupby("entry_date", sort=False)}
    cash = INITIAL_CAPITAL
    positions: list[dict] = []
    records: list[dict] = []

    for date in dates:
        for position in list(positions):
            if pd.notna(position["exit_date"]) and position["exit_date"] == date:
                price = float(price_map.get((position["symbol"], date), position["exit_close"]))
                proceeds = position["quantity"] * price
                fee = commission(proceeds)
                cash += proceeds - fee
                ret = (proceeds - fee - position["cost_total"]) / position["cost_total"]
                record = dict(position)
                record.update(
                    {
                        "actual_exit_date": date.date().isoformat(),
                        "actual_exit_close": price,
                        "sell_fee": fee,
                        "net_return": ret,
                        "net_pnl": proceeds - fee - position["cost_total"],
                        "status_portfolio": "closed",
                    }
                )
                records.append(record)
                positions.remove(position)

        if date not in by_entry:
            continue
        available = max_positions - len(positions)
        if available <= 0:
            continue
        for _, trade in by_entry[date].head(available).iterrows():
            price = float(trade["entry_close"])
            if not np.isfinite(price) or price <= 0:
                continue
            target = cash / max(1, available)
            quantity = math.floor((target - MIN_COMMISSION) / (price * 100)) * 100
            if quantity < 100:
                continue
            notional = quantity * price
            fee = commission(notional)
            if notional + fee > cash:
                continue
            cash -= notional + fee
            positions.append(
                {
                    "symbol": trade["symbol"],
                    "name": trade.get("name", ""),
                    "overlay": trade.get("overlay", "baseline"),
                    "entry_date": date.date().isoformat(),
                    "entry_close": price,
                    "exit_date": trade["exit_date"],
                    "exit_close": trade.get("exit_close", np.nan),
                    "quantity": int(quantity),
                    "buy_fee": fee,
                    "cost_total": notional + fee,
                }
            )
            available -= 1
            if available <= 0:
                break

    open_value = 0.0
    for position in positions:
        price = float(latest_price.get(position["symbol"], position["entry_close"]))
        liquidation_fee = commission(position["quantity"] * price)
        value = position["quantity"] * price - liquidation_fee
        open_value += value
        record = dict(position)
        record.update(
            {
                "actual_exit_date": "",
                "actual_exit_close": price,
                "sell_fee": liquidation_fee,
                "net_return": (value - position["cost_total"]) / position["cost_total"],
                "net_pnl": value - position["cost_total"],
                "status_portfolio": "open_marked",
            }
        )
        records.append(record)
    log = pd.DataFrame(records)
    closed = log[log["status_portfolio"].eq("closed")] if not log.empty else log
    final_value = cash + open_value
    years = max((latest_date - pd.Timestamp(candidates["entry_date"].min())).days / 365.25, 1e-9)
    summary = {
        "max_positions": max_positions,
        "ranking": ranking,
        "initial_capital": INITIAL_CAPITAL,
        "final_value": float(final_value),
        "net_profit": float(final_value - INITIAL_CAPITAL),
        "total_return": float(final_value / INITIAL_CAPITAL - 1),
        "cagr": float((final_value / INITIAL_CAPITAL) ** (1 / years) - 1),
        "accepted_trades": int(len(log)),
        "closed_trades": int(len(closed)),
        "win_rate": float((closed["net_return"] > 0).mean()) if len(closed) else np.nan,
        "avg_net_return": float(closed["net_return"].mean()) if len(closed) else np.nan,
        "open_positions": int(len(positions)),
        "cash": float(cash),
        "open_mark_value": float(open_value),
        "as_of": latest_date.date().isoformat(),
        "fee_rule": "万三，单边最低5元；100股整数手",
    }
    return log, summary


def latest_decision(sentiment: pd.DataFrame, selected_by_year: pd.DataFrame) -> dict:
    latest = sentiment.dropna(subset=["fear_enhanced"]).iloc[-1]
    current_year = int(pd.Timestamp(latest["date"]).year)
    selection = selected_by_year[selected_by_year["test_year"].eq(current_year)]
    overlay = str(selection.iloc[-1]["overlay"]) if not selection.empty else "baseline"
    return {
        "date": pd.Timestamp(latest["date"]).date().isoformat(),
        "fear_core": float(latest["fear_core"]),
        "fear_qvix": float(latest["fear_qvix"]) if pd.notna(latest["fear_qvix"]) else None,
        "fear_enhanced": float(latest["fear_enhanced"]),
        "greed_enhanced": float(latest["greed_enhanced"]),
        "regime": str(latest["regime"]),
        "active_overlay": overlay,
        "note": "情绪层只决定是否放行技术买点；没有技术买点时不会单独买入。",
    }


def write_report(
    fixed: pd.DataFrame,
    selected_years: pd.DataFrame,
    rolling_metrics: dict,
    portfolio: list[dict],
    latest: dict,
) -> None:
    top = fixed.head(12).copy()
    for column in ("win_rate", "avg_return", "worst_year_win_rate"):
        top[column] = top[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    years = selected_years.copy()
    for column in ("win_rate", "avg_return"):
        years[column] = years[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    lines = [
        "# FG1 恐惧贪婪实盘候选",
        "",
        "## 结论口径",
        "",
        "- 技术母策略：D 线年度滚动选择 + ETF 自身收盘价不低于 MA120。",
        "- 情绪层：30 ETF 市场广度、波动、回撤、量价压力与新低扩散；2015 年后用 QVIX 占 30% 增强。",
        "- 退出层：最长持有 20/40/60/90/120 日，并测试贪婪分 60/70/80 的 T+1 退出。",
        "- 北向资金按下一交易日可得处理，但因披露口径变化只保留审计，不进入正式分数。",
        "- 每个测试年只用更早年份选择情绪阈值；信号收盘确认，T+1 收盘执行。",
        "- 账户回测采用 100 股整数手、万三且单边最低 5 元。",
        "",
        "## 滚动样本外结果",
        "",
        f"- 候选信号 {rolling_metrics['signals']}，已完成 {rolling_metrics['closed']}，未完成 {rolling_metrics['open']}。",
        f"- 已完成胜率 {rolling_metrics['win_rate']:.2%}，平均每笔 {rolling_metrics['avg_return']:.2%}。",
        f"- 每个 100 元独立批次已实现/浮动盈亏：{rolling_metrics['realized_pnl_per_100']:.2f} / {rolling_metrics['unrealized_pnl_per_100']:.2f} 元。",
        "",
        "## 账户模拟",
        "",
    ]
    for item in portfolio:
        lines.append(
            f"- {item['max_positions']} 仓：1 万元 -> {item['final_value']:.2f} 元，"
            f"总收益 {item['total_return']:.2%}，已完成胜率 {item['win_rate']:.2%}，"
            f"已完成 {item['closed_trades']} 笔。"
        )
    lines.extend(
        [
            "",
            "## 当前情绪",
            "",
            f"- 日期：{latest['date']}；恐惧分 {latest['fear_enhanced']:.2f}；"
            f"贪婪分 {latest['greed_enhanced']:.2f}；状态：{latest['regime']}。",
            f"- 当前放行规则：`{latest['active_overlay']}`。",
            "",
            "## 固定阈值研究排名",
            "",
            top[
                [
                    "overlay",
                    "exit_key",
                    "closed",
                    "win_rate",
                    "avg_return",
                    "realized_pnl_per_100",
                    "worst_year_win_rate",
                    "robust_score",
                ]
            ].to_markdown(index=False),
            "",
            "## 每年只用历史选择",
            "",
            years[
                [
                    "test_year",
                    "overlay",
                    "prior_years",
                    "closed",
                    "win_rate",
                    "avg_return",
                    "realized_pnl_per_100",
                ]
            ].to_markdown(index=False),
            "",
            "## 限制",
            "",
            "- 这是情绪覆盖层的严格滚动样本外结果，不把阶段高低点标签用于交易。",
            "- 2021 年没有更早 OOS 记录，按预注册基线执行；以后年度才由历史表现选择阈值。",
            "- QVIX 由公开行情接口下载，不等同于任何商业软件的私有“恐慌因子”。",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    raw, pool = load_clean_raw()
    sentiment = build_sentiment(raw)
    sentiment.to_csv(OUT / "sentiment_daily.csv", index=False, encoding="utf-8-sig")
    symbol_sentiment = build_symbol_fear(raw)
    symbol_sentiment.to_csv(
        OUT / "symbol_fear_daily.csv", index=False, encoding="utf-8-sig"
    )

    trades = pd.read_csv(BASE_TRADES, dtype={"symbol": str}, encoding="utf-8-sig")
    trades = rebuild_trade_prices(trades, raw)
    trades = attach_sentiment(trades, sentiment, symbol_sentiment)
    trades.to_csv(OUT / "base_candidates.csv", index=False, encoding="utf-8-sig")

    overlays = overlay_grid()
    exits = exit_grid()
    adjusted_by_exit = {
        rule.key: apply_exit_rule(trades, rule, raw, sentiment) for rule in exits
    }
    fixed = practical_grid_metrics(trades, overlays, exits, adjusted_by_exit)
    fixed.to_csv(OUT / "fixed_grid.csv", index=False, encoding="utf-8-sig")

    rolling_trades, selected_years = rolling_practical(
        trades, overlays, exits, adjusted_by_exit
    )
    rolling_trades.to_csv(OUT / "rolling_trades.csv", index=False, encoding="utf-8-sig")
    selected_years.to_csv(OUT / "selected_by_year.csv", index=False, encoding="utf-8-sig")
    rolling_metrics = trade_metrics(rolling_trades)

    portfolio_summaries = []
    for slots in (1, 3):
        log, summary = portfolio_backtest(
            rolling_trades, raw, pool, slots, ranking="fear"
        )
        log.to_csv(OUT / f"portfolio_{slots}slot.csv", index=False, encoding="utf-8-sig")
        portfolio_summaries.append(summary)

    baseline_trades = adjusted_by_exit["hold_40_greed_off"].copy()
    baseline_trades["overlay"] = "baseline"
    baseline_summaries = []
    for slots in (1, 3):
        log, summary = portfolio_backtest(
            baseline_trades, raw, pool, slots, ranking="liquidity"
        )
        log.to_csv(OUT / f"baseline_portfolio_{slots}slot.csv", index=False, encoding="utf-8-sig")
        baseline_summaries.append(summary)

    latest = latest_decision(sentiment, selected_years)
    (OUT / "today.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_end": str(raw["date"].max().date()),
        "symbols": int(raw["symbol"].nunique()),
        "method": "Strict prior-only fear/greed overlay on D nested MA120 technical candidate trades.",
        "rolling_lot_metrics": rolling_metrics,
        "portfolio": portfolio_summaries,
        "baseline_portfolio": baseline_summaries,
        "latest": latest,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(fixed, selected_years, rolling_metrics, portfolio_summaries, latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
