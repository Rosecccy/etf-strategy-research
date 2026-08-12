from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "live"
STATE_FILE = LIVE_DIR / "extrema_exit_state.csv"
LIVE_SIGNALS = LIVE_DIR / "extrema_high_signals.csv"
HISTORICAL_SIGNALS = ROOT / "fit" / "extrema" / "historical_signals.csv"

STATE_COLUMNS = [
    "symbol",
    "entry_date",
    "trigger_date",
    "original_reason",
    "recent_signal_days",
    "delay_trade_days",
    "active",
    "last_checked_date",
    "last_action",
    "closed_date",
]


def load_state() -> pd.DataFrame:
    if not STATE_FILE.exists():
        return pd.DataFrame(columns=STATE_COLUMNS)
    state = pd.read_csv(STATE_FILE, dtype={"symbol": str})
    for column in STATE_COLUMNS:
        if column not in state.columns:
            state[column] = ""
    state["symbol"] = state["symbol"].astype(str).str.zfill(6)
    return state[STATE_COLUMNS]


def save_state(state: pd.DataFrame) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    state[STATE_COLUMNS].to_csv(STATE_FILE, index=False, encoding="utf-8-sig")


def high_signal_dates(symbol: str) -> pd.DatetimeIndex:
    parts = []
    if HISTORICAL_SIGNALS.exists():
        data = pd.read_csv(HISTORICAL_SIGNALS, encoding="utf-8-sig", dtype={"symbol": str})
        data = data[data["target"].eq("高点") & data["predicted"].astype(str).str.lower().eq("true")]
        parts.append(data[["symbol", "signal_date"]])
    if LIVE_SIGNALS.exists():
        data = pd.read_csv(LIVE_SIGNALS, encoding="utf-8-sig", dtype={"symbol": str})
        parts.append(data[["symbol", "signal_date"]])
    if not parts:
        return pd.DatetimeIndex([])
    signals = pd.concat(parts, ignore_index=True)
    signals["symbol"] = signals["symbol"].astype(str).str.zfill(6)
    dates = pd.to_datetime(signals.loc[signals["symbol"].eq(str(symbol).zfill(6)), "signal_date"], errors="coerce")
    return pd.DatetimeIndex(dates.dropna().drop_duplicates().sort_values())


def recent_high_count(raw: pd.DataFrame, symbol: str, asof_date: pd.Timestamp, sessions: int) -> int:
    calendar = pd.DatetimeIndex(pd.to_datetime(raw["date"]).drop_duplicates().sort_values())
    eligible = calendar[calendar <= pd.Timestamp(asof_date)]
    if len(eligible) < sessions:
        return 0
    start = pd.Timestamp(eligible[-sessions])
    dates = high_signal_dates(symbol)
    return int(((dates >= start) & (dates <= pd.Timestamp(asof_date))).sum())


def elapsed_sessions(raw: pd.DataFrame, start_date: str, asof_date: pd.Timestamp) -> int:
    calendar = pd.DatetimeIndex(pd.to_datetime(raw["date"]).drop_duplicates().sort_values())
    start = pd.Timestamp(start_date)
    return int(((calendar > start) & (calendar <= pd.Timestamp(asof_date))).sum())


def close_stale_states(state: pd.DataFrame, positions: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    if state.empty:
        return state
    active_keys = {
        (str(row["symbol"]).zfill(6), str(pd.Timestamp(row["entry_date"]).date()))
        for _, row in positions.dropna(subset=["symbol", "entry_date"]).iterrows()
    }
    for index, row in state.iterrows():
        if str(row["active"]).lower() not in {"true", "1", "yes"}:
            continue
        key = (str(row["symbol"]).zfill(6), str(pd.Timestamp(row["entry_date"]).date()))
        if key not in active_keys:
            state.at[index, "active"] = False
            state.at[index, "closed_date"] = str(pd.Timestamp(asof_date).date())
            state.at[index, "last_action"] = "position_closed"
    return state


def evaluate_exit(
    *,
    state: pd.DataFrame,
    raw: pd.DataFrame,
    symbol: str,
    entry_date: str,
    asof_date: pd.Timestamp,
    baseline_triggered: bool,
    baseline_reason: str,
    recent_signal_days: int,
    delay_trade_days: int,
    switch_candidate: dict | None,
) -> tuple[dict, pd.DataFrame]:
    symbol = str(symbol).zfill(6)
    entry_key = str(pd.Timestamp(entry_date).date())
    active_mask = (
        state["symbol"].astype(str).str.zfill(6).eq(symbol)
        & state["entry_date"].astype(str).map(lambda value: str(pd.Timestamp(value).date()) if value else "").eq(entry_key)
        & state["active"].astype(str).str.lower().isin(["true", "1", "yes"])
    ) if not state.empty else pd.Series(False, index=state.index)

    if active_mask.any():
        index = active_mask[active_mask].index[-1]
        row = state.loc[index]
        elapsed = elapsed_sessions(raw, str(row["trigger_date"]), asof_date)
        switch_start = pd.to_datetime(
            switch_candidate.get("signal_start_date"), errors="coerce"
        ) if switch_candidate else pd.NaT
        is_new_switch = bool(
            switch_candidate
            and pd.notna(switch_start)
            and pd.Timestamp(switch_start) > pd.Timestamp(row["trigger_date"])
        )
        if is_new_switch:
            action = "next_trade_close_switch_to_main"
            reason = "extrema_extension_rotates_to_new_main_signal"
        elif elapsed >= int(row["delay_trade_days"]):
            action = "next_trade_close_sell"
            reason = "extrema_extension_complete"
        else:
            action = "hold"
            reason = "extrema_extension_wait"
        state.at[index, "last_checked_date"] = str(pd.Timestamp(asof_date).date())
        state.at[index, "last_action"] = action
        return {
            "next_action": action,
            "reason": reason,
            "extrema_exit_active": True,
            "extrema_trigger_date": row["trigger_date"],
            "extrema_recent_high_count": 0,
            "extrema_elapsed_days": elapsed,
            "extrema_delay_days": int(row["delay_trade_days"]),
            "switch_symbol": switch_candidate.get("symbol", "") if is_new_switch else "",
            "switch_display_name": switch_candidate.get("display_name", "") if is_new_switch else "",
            "switch_strategy_id": switch_candidate.get("strategy_id", "") if is_new_switch else "",
        }, state

    if not baseline_triggered:
        return {
            "next_action": "hold",
            "reason": "no_exit",
            "extrema_exit_active": False,
            "extrema_trigger_date": "",
            "extrema_recent_high_count": 0,
            "extrema_elapsed_days": 0,
            "extrema_delay_days": delay_trade_days,
            "switch_symbol": "",
            "switch_display_name": "",
            "switch_strategy_id": "",
        }, state

    count = recent_high_count(raw, symbol, asof_date, recent_signal_days)
    if count > 0:
        return {
            "next_action": "next_trade_close_sell",
            "reason": baseline_reason,
            "extrema_exit_active": False,
            "extrema_trigger_date": "",
            "extrema_recent_high_count": count,
            "extrema_elapsed_days": 0,
            "extrema_delay_days": delay_trade_days,
            "switch_symbol": "",
            "switch_display_name": "",
            "switch_strategy_id": "",
        }, state

    new_row = {
        "symbol": symbol,
        "entry_date": entry_key,
        "trigger_date": str(pd.Timestamp(asof_date).date()),
        "original_reason": baseline_reason,
        "recent_signal_days": recent_signal_days,
        "delay_trade_days": delay_trade_days,
        "active": True,
        "last_checked_date": str(pd.Timestamp(asof_date).date()),
        "last_action": "hold",
        "closed_date": "",
    }
    state = pd.concat([state, pd.DataFrame([new_row])], ignore_index=True)
    return {
        "next_action": "hold",
        "reason": "extrema_no_recent_high_delay_started",
        "extrema_exit_active": True,
        "extrema_trigger_date": new_row["trigger_date"],
        "extrema_recent_high_count": 0,
        "extrema_elapsed_days": 0,
        "extrema_delay_days": delay_trade_days,
        "switch_symbol": "",
        "switch_display_name": "",
        "switch_strategy_id": "",
    }, state
