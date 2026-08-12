from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from c2_pipeline import add_indicators, aggregate, carry_to_daily, load_raw
from c2lib import ACCOUNT_DIR
from extrema_exit import close_stale_states, evaluate_exit, load_state, save_state
from live_model import build_fallback_candidate, load_live_config, parse_fallback_hold_days


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
LIVE = ROOT / "live"
SELECTOR = ROOT / "fit" / "selector" / "best_selection.csv"
SELECTED_TRADES = ROOT / "fit" / "selector" / "best_selected_trades.csv"
BEST_TRADES = ROOT / "fit" / "selector" / "best_trades.csv"
ACCOUNT = ACCOUNT_DIR / "account.csv"
CFG = ROOT / "cfg" / "selector.json"
LIVE_CFG = ROOT / "cfg" / "live_model.json"


FACTOR_ALIASES = {
    "RSI6": "RSI",
}

FREQ_ALIASES = {
    "D": "D",
    "W": "W",
    "B15": "B15",
    "M": "M",
    "日K": "D",
    "周K": "W",
    "15日K": "B15",
    "月K": "M",
}


def factor_name(name: str) -> str:
    return FACTOR_ALIASES.get(name, name)


def freq_name(name: str) -> str:
    text = str(name).strip()
    if text in FREQ_ALIASES:
        return FREQ_ALIASES[text]
    if "15" in text:
        return "B15"
    if "日" in text:
        return "D"
    if "周" in text:
        return "W"
    if "月" in text:
        return "M"
    raise ValueError(f"unknown frequency label: {name}")


def parse_combo(combo: str) -> list[str]:
    return [x.strip() for x in str(combo).split("+") if x.strip()]


def parse_threshold(threshold: str, total: int) -> int:
    text = str(threshold)
    if "/" in text:
        return int(text.split("/", 1)[0])
    return total


def ensure_account_template() -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    ACCOUNT_DIR.mkdir(parents=True, exist_ok=True)
    if ACCOUNT.exists():
        return
    pd.DataFrame(
        columns=[
            "symbol",
            "display_name",
            "category",
            "strategy_id",
            "entry_date",
            "entry_close",
            "shares",
            "note",
        ]
    ).to_csv(ACCOUNT, index=False, encoding="utf-8-sig")


def enrich_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    caliber = {"caliber": "profit_best_ema25_style", "kdj": "ema", "rsi": "rolling", "emv": "rolling", "lon": "ewm"}
    out = add_indicators(bars, caliber)
    out["AMOUNT"] = np.log1p(out["amount"].clip(lower=0))
    return out


def add_exit_regime_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["ma20"] = out["close"].rolling(20, min_periods=10).mean()
    out["ma60"] = out["close"].rolling(60, min_periods=20).mean()
    out["ret20"] = out["close"].pct_change(20)
    return out


def main_strategy_runtime(selector_cfg: dict, live_cfg: dict) -> dict:
    live_main = live_cfg.get("main_strategy", {})
    base_max_hold = int(
        live_main.get(
            "max_hold_days",
            selector_cfg.get("base_signal", {}).get("max_hold_days", 40),
        )
    )
    adaptive_cfg = live_main.get("adaptive_exit", {})
    extrema_cfg = live_main.get("extrema_exit", {})
    return {
        "base_max_hold_days": base_max_hold,
        "adaptive_exit": {
            "enabled": bool(adaptive_cfg.get("enabled", False)),
            "profile_id": str(adaptive_cfg.get("profile_id", "") or ""),
            "strong_bonus_days": int(adaptive_cfg.get("strong_bonus_days", 0) or 0),
            "weak_cut_days": int(adaptive_cfg.get("weak_cut_days", 0) or 0),
            "min_hold_days": int(adaptive_cfg.get("min_hold_days", 5) or 5),
            "trend_break_enabled": bool(adaptive_cfg.get("trend_break_enabled", False)),
            "strong_ret20_min": float(adaptive_cfg.get("strong_ret20_min", 0.03)),
            "weak_ret20_max": float(adaptive_cfg.get("weak_ret20_max", -0.03)),
            "trend_break_ret20_below": float(adaptive_cfg.get("trend_break_ret20_below", 0.0)),
        },
        "extrema_exit": {
            "enabled": bool(extrema_cfg.get("enabled", False)),
            "profile_id": str(extrema_cfg.get("profile_id", "") or ""),
            "recent_signal_days": int(extrema_cfg.get("recent_signal_days", 15) or 15),
            "delay_trade_days": int(extrema_cfg.get("delay_trade_days", 8) or 8),
        },
    }


def entry_regime(raw: pd.DataFrame, entry_date: str, adaptive_cfg: dict) -> str:
    hit = raw.loc[raw["date"] == pd.to_datetime(entry_date)]
    if hit.empty:
        return "unknown"
    row = hit.iloc[0]
    close = float(row.get("close", np.nan))
    ma60 = float(row.get("ma60", np.nan))
    ret20 = float(row.get("ret20", np.nan))
    if np.isfinite(close) and np.isfinite(ma60) and np.isfinite(ret20):
        if close >= ma60 and ret20 >= float(adaptive_cfg["strong_ret20_min"]):
            return "strong"
        if close < ma60 and ret20 <= float(adaptive_cfg["weak_ret20_max"]):
            return "weak"
        return "neutral"
    return "unknown"


def adaptive_exit_status(raw: pd.DataFrame, entry_date: str, hold_days: int | None, runtime_cfg: dict) -> dict:
    adaptive_cfg = runtime_cfg["adaptive_exit"]
    base_max_hold = int(runtime_cfg["base_max_hold_days"])
    status = {
        "entry_regime": "disabled" if not adaptive_cfg["enabled"] else "unknown",
        "effective_max_hold_days": base_max_hold,
        "trend_break_triggered": False,
        "trend_break_reason": "",
    }
    if not adaptive_cfg["enabled"] or hold_days is None:
        return status

    regime = entry_regime(raw, entry_date, adaptive_cfg)
    effective_max_hold = base_max_hold
    if regime == "strong":
        effective_max_hold += int(adaptive_cfg["strong_bonus_days"])
    elif regime == "weak":
        effective_max_hold = max(
            int(adaptive_cfg["min_hold_days"]),
            base_max_hold - int(adaptive_cfg["weak_cut_days"]),
        )
    status["entry_regime"] = regime
    status["effective_max_hold_days"] = effective_max_hold

    if not adaptive_cfg["trend_break_enabled"]:
        return status
    if regime != "strong":
        return status
    if hold_days < base_max_hold or hold_days >= effective_max_hold:
        return status
    latest = raw.iloc[-1]
    close = float(latest.get("close", np.nan))
    ma20 = float(latest.get("ma20", np.nan))
    ret20 = float(latest.get("ret20", np.nan))
    if np.isfinite(close) and np.isfinite(ma20) and close < ma20:
        status["trend_break_triggered"] = True
        status["trend_break_reason"] = "close_below_ma20_during_extension"
        return status
    if np.isfinite(ret20) and ret20 < float(adaptive_cfg["trend_break_ret20_below"]):
        status["trend_break_triggered"] = True
        status["trend_break_reason"] = "ret20_below_break_threshold_during_extension"
    return status


def signal_panel(raw: pd.DataFrame, freq: str, combo: list[str], side: str, threshold_value: float) -> pd.DataFrame:
    factors = [factor_name(x) for x in combo]
    bars = enrich_indicators(aggregate(raw, freq))
    sig = bars[["date"]].copy()
    for factor in factors:
        if factor not in bars.columns:
            raise ValueError(f"missing factor {factor} in indicators")
        x = bars[factor].astype(float)
        lo = x.rolling(120, min_periods=20).min()
        hi = x.rolling(120, min_periods=20).max()
        pos = (x - lo) / (hi - lo).replace(0, np.nan)
        if side == "low":
            sig[factor] = (pos <= threshold_value).fillna(False)
        elif side == "high":
            sig[factor] = (pos >= threshold_value).fillna(False)
        else:
            raise ValueError(f"unknown side {side}")
    return carry_to_daily(sig, raw["date"], freq)


def latest_signal(raw: pd.DataFrame, strategy: pd.Series, signal_type: str) -> tuple[bool, int, int, str, str]:
    if signal_type == "buy":
        freq = freq_name(strategy["buy_frequency"])
        combo = parse_combo(strategy["buy_combo"])
        threshold = parse_threshold(strategy["buy_threshold"], len(combo))
        side = "low"
        value = float(strategy["threshold_low"]) if "threshold_low" in strategy.index else 0.25
    else:
        freq = freq_name(strategy["sell_frequency"])
        combo = parse_combo(strategy["sell_combo"])
        threshold = parse_threshold(strategy["sell_threshold"], len(combo))
        side = "high"
        value = float(strategy["threshold_high"]) if "threshold_high" in strategy.index else 0.90

    panel = signal_panel(raw, freq, combo, side, value)
    factors = [factor_name(x) for x in combo]
    latest = panel.iloc[-1]
    hit_count = int(sum(bool(latest[f]) for f in factors))
    combined = panel[factors].astype(bool).sum(axis=1).ge(threshold)
    signal_start_date = ""
    if bool(combined.iloc[-1]):
        false_positions = np.flatnonzero(~combined.to_numpy())
        start_position = int(false_positions[-1] + 1) if len(false_positions) else 0
        signal_start_date = pd.Timestamp(panel.iloc[start_position]["date"]).strftime("%Y-%m-%d")
    return hit_count >= threshold, hit_count, threshold, "+".join(combo), signal_start_date


def trading_days_since(raw: pd.DataFrame, entry_date: str, asof_date: pd.Timestamp) -> int | None:
    dates = list(raw["date"])
    entry = pd.to_datetime(entry_date)
    entry_pos = np.searchsorted(dates, entry)
    asof_pos = np.searchsorted(dates, asof_date)
    if entry_pos >= len(dates) or dates[entry_pos] != entry:
        return None
    return int(asof_pos - entry_pos)


def pick_active_strategies(selection: pd.DataFrame, year: int) -> pd.DataFrame:
    available = selection[selection["test_year"].astype(int) <= year].copy()
    idx = available.groupby("category")["test_year"].idxmax()
    return available.loc[idx].sort_values("category").reset_index(drop=True)


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
    catalog = data[[c for c in cols if c in data.columns]].drop_duplicates("strategy_id").sort_values("strategy_id")
    catalog.to_csv(LIVE / "strategy_catalog.csv", index=False, encoding="utf-8-sig")
    return catalog


def main() -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    ensure_account_template()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    live_cfg = load_live_config()
    runtime_cfg = main_strategy_runtime(cfg, live_cfg)
    max_hold = int(runtime_cfg["base_max_hold_days"])

    pool = pd.read_csv(RAW / "pool.csv", dtype={"symbol": str})
    pool = pool[pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    selection = pd.read_csv(SELECTOR)
    catalog = build_strategy_catalog()

    latest_dates = []
    raw_cache: dict[str, pd.DataFrame] = {}
    for symbol in pool["symbol"].astype(str):
        raw = load_raw(symbol)
        raw_cache[symbol] = raw
        latest_dates.append(raw["date"].max())
    min_data_date = min(latest_dates)
    max_data_date = max(latest_dates)
    stale_count = sum(dt < max_data_date for dt in latest_dates)
    data_date = min_data_date
    active = pick_active_strategies(selection, int(max_data_date.year))
    if not catalog.empty:
        active = active.merge(catalog, on="strategy_id", how="left")
    active.to_csv(LIVE / "active_strategies.csv", index=False, encoding="utf-8-sig")
    missing_rules = active[active.get("buy_frequency", pd.Series(index=active.index, dtype=object)).isna()].copy()
    missing_rules.to_csv(LIVE / "missing_rules.csv", index=False, encoding="utf-8-sig")

    active_by_category = {row["category"]: row for _, row in active.iterrows()}
    buy_rows = []
    for _, etf in pool.iterrows():
        category = etf["category"]
        if category not in active_by_category:
            continue
        strategy = active_by_category[category]
        symbol = str(etf["symbol"])
        raw = raw_cache[symbol]
        if pd.isna(strategy.get("buy_frequency", np.nan)):
            buy_rows.append(
                {
                    "data_date": raw["date"].max().date(),
                    "symbol": symbol,
                    "display_name": etf["display_name"],
                    "category": category,
                    "strategy_id": strategy["strategy_id"],
                    "is_buy_signal": False,
                    "next_action": "missing_strategy_rule",
                    "selected_score": strategy.get("selected_score", ""),
                    "score_func": strategy.get("score_func", ""),
                }
            )
            continue
        ok, hit_count, threshold, combo, signal_start_date = latest_signal(raw, strategy, "buy")
        buy_rows.append(
            {
                "data_date": raw["date"].max().date(),
                "symbol": symbol,
                "display_name": etf["display_name"],
                "category": category,
                "strategy_id": strategy["strategy_id"],
                "buy_frequency": strategy["buy_frequency"],
                "buy_combo": combo,
                "buy_threshold": f"{threshold}/{len(parse_combo(strategy['buy_combo']))}",
                "hit_count": hit_count,
                "is_buy_signal": bool(ok),
                "signal_start_date": signal_start_date,
                "next_action": "next_trade_close_buy" if ok else "wait",
                "selected_score": strategy.get("selected_score", ""),
                "score_func": strategy.get("score_func", ""),
            }
        )
    buy_df = pd.DataFrame(buy_rows).sort_values(["is_buy_signal", "selected_score"], ascending=[False, False])
    buy_df.to_csv(LIVE / "today_buy_signals.csv", index=False, encoding="utf-8-sig")

    account = pd.read_csv(ACCOUNT, dtype={"symbol": str})
    holding_rows = account.dropna(subset=["symbol", "entry_date"]).copy()
    exit_state = close_stale_states(load_state(), holding_rows, max_data_date)
    fallback_candidate = build_fallback_candidate(max_data_date, live_cfg)
    fallback_df = pd.DataFrame([fallback_candidate]) if fallback_candidate else pd.DataFrame()
    fallback_df.to_csv(LIVE / "today_fallback_signal.csv", index=False, encoding="utf-8-sig")
    sell_rows = []
    best_main_signal = buy_df[buy_df["is_buy_signal"] == True].head(1)
    for _, pos in holding_rows.iterrows():
        symbol = str(pos["symbol"])
        strategy_id_text = str(pos.get("strategy_id", "") or "")
        if strategy_id_text.startswith("FB_"):
            hold_days = trading_days_since(raw_cache[symbol], str(pos["entry_date"]), raw_cache[symbol]["date"].max()) if symbol in raw_cache else None
            hold_target = parse_fallback_hold_days(strategy_id_text)
            has_main_signal = not buy_df[buy_df["is_buy_signal"] == True].empty
            best_main = buy_df[buy_df["is_buy_signal"] == True].head(1)
            action = "hold"
            reason = "fallback_continue"
            switch_symbol = ""
            switch_name = ""
            switch_strategy = ""
            if has_main_signal and not best_main.empty:
                action = "next_trade_close_switch_to_main"
                reason = "main_signal_has_priority"
                switch_symbol = str(best_main.iloc[0]["symbol"])
                switch_name = str(best_main.iloc[0]["display_name"])
                switch_strategy = str(best_main.iloc[0]["strategy_id"])
            elif hold_target is not None and hold_days is not None and hold_days >= hold_target:
                action = "next_trade_close_sell"
                reason = "fallback_max_hold"
            sell_rows.append(
                {
                    "data_date": raw_cache[symbol]["date"].max().date() if symbol in raw_cache else max_data_date.date(),
                    "symbol": symbol,
                    "display_name": pos.get("display_name", ""),
                    "category": pos.get("category", ""),
                    "strategy_id": strategy_id_text,
                    "entry_date": pos["entry_date"],
                    "hold_trade_days": hold_days,
                    "sell_frequency": "",
                    "sell_combo": "",
                    "sell_threshold": "",
                    "hit_count": "",
                    "is_sell_signal": False,
                    "is_max_hold": bool(hold_target is not None and hold_days is not None and hold_days >= hold_target),
                    "next_action": action,
                    "reason": reason,
                    "switch_symbol": switch_symbol,
                    "switch_display_name": switch_name,
                    "switch_strategy_id": switch_strategy,
                }
            )
            continue
        if symbol not in raw_cache:
            sell_rows.append({"symbol": symbol, "next_action": "unknown_symbol"})
            continue
        raw = add_exit_regime_columns(raw_cache[symbol])
        raw_cache[symbol] = raw
        category = pos["category"] if "category" in pos.index and pd.notna(pos["category"]) else pool.loc[pool["symbol"] == symbol, "category"].iloc[0]
        strategy = active_by_category.get(category)
        if strategy is None:
            sell_rows.append({"symbol": symbol, "category": category, "next_action": "no_active_strategy"})
            continue
        if pd.isna(strategy.get("sell_frequency", np.nan)):
            sell_rows.append({"symbol": symbol, "category": category, "strategy_id": strategy["strategy_id"], "next_action": "missing_strategy_rule"})
            continue
        sell_ok, hit_count, threshold, combo, _ = latest_signal(raw, strategy, "sell")
        hold_days = trading_days_since(raw, str(pos["entry_date"]), raw["date"].max())
        exit_status = adaptive_exit_status(raw, str(pos["entry_date"]), hold_days, runtime_cfg)
        max_hold_ok = hold_days is not None and hold_days >= int(exit_status["effective_max_hold_days"])
        trend_break_ok = bool(exit_status["trend_break_triggered"])
        baseline_triggered = bool(sell_ok or max_hold_ok or trend_break_ok)
        action = "next_trade_close_sell" if baseline_triggered else "hold"
        if sell_ok:
            reason = "sell_signal"
        elif trend_break_ok:
            reason = str(exit_status["trend_break_reason"])
        elif max_hold_ok:
            reason = "max_hold"
        else:
            reason = "no_exit"
        extrema_result = {
            "next_action": action,
            "reason": reason,
            "extrema_exit_active": False,
            "extrema_trigger_date": "",
            "extrema_recent_high_count": 0,
            "extrema_elapsed_days": 0,
            "extrema_delay_days": 0,
            "switch_symbol": "",
            "switch_display_name": "",
            "switch_strategy_id": "",
        }
        extrema_cfg = runtime_cfg["extrema_exit"]
        if extrema_cfg["enabled"]:
            switch_candidate = None
            if not best_main_signal.empty:
                switch_candidate = best_main_signal.iloc[0].to_dict()
            extrema_result, exit_state = evaluate_exit(
                state=exit_state,
                raw=raw,
                symbol=symbol,
                entry_date=str(pos["entry_date"]),
                asof_date=pd.Timestamp(raw["date"].max()),
                baseline_triggered=baseline_triggered,
                baseline_reason=reason,
                recent_signal_days=int(extrema_cfg["recent_signal_days"]),
                delay_trade_days=int(extrema_cfg["delay_trade_days"]),
                switch_candidate=switch_candidate,
            )
        sell_rows.append(
            {
                "data_date": raw["date"].max().date(),
                "symbol": symbol,
                "display_name": pos.get("display_name", ""),
                "category": category,
                "strategy_id": strategy["strategy_id"],
                "entry_date": pos["entry_date"],
                "hold_trade_days": hold_days,
                "sell_frequency": strategy["sell_frequency"],
                "sell_combo": combo,
                "sell_threshold": f"{threshold}/{len(parse_combo(strategy['sell_combo']))}",
                "hit_count": hit_count,
                "is_sell_signal": bool(sell_ok),
                "is_max_hold": bool(max_hold_ok),
                "entry_regime": exit_status["entry_regime"],
                "effective_max_hold_days": int(exit_status["effective_max_hold_days"]),
                "is_trend_break": trend_break_ok,
                "next_action": extrema_result["next_action"],
                "reason": extrema_result["reason"],
                "extrema_exit_active": extrema_result["extrema_exit_active"],
                "extrema_trigger_date": extrema_result["extrema_trigger_date"],
                "extrema_recent_high_count": extrema_result["extrema_recent_high_count"],
                "extrema_elapsed_days": extrema_result["extrema_elapsed_days"],
                "extrema_delay_days": extrema_result["extrema_delay_days"],
                "switch_symbol": extrema_result["switch_symbol"],
                "switch_display_name": extrema_result["switch_display_name"],
                "switch_strategy_id": extrema_result["switch_strategy_id"],
            }
        )
    save_state(exit_state)
    sell_columns = [
        "data_date",
        "symbol",
        "display_name",
        "category",
        "strategy_id",
        "entry_date",
        "hold_trade_days",
        "sell_frequency",
        "sell_combo",
        "sell_threshold",
        "hit_count",
        "is_sell_signal",
        "is_max_hold",
        "entry_regime",
        "effective_max_hold_days",
        "is_trend_break",
        "extrema_exit_active",
        "extrema_trigger_date",
        "extrema_recent_high_count",
        "extrema_elapsed_days",
        "extrema_delay_days",
        "next_action",
        "reason",
        "switch_symbol",
        "switch_display_name",
        "switch_strategy_id",
    ]
    sell_df = pd.DataFrame(sell_rows, columns=sell_columns)
    sell_df.to_csv(LIVE / "today_sell_checks.csv", index=False, encoding="utf-8-sig")

    holding = len(holding_rows) > 0
    if holding:
        decision = sell_df.copy()
        if decision.empty:
            decision = pd.DataFrame([{"data_date": data_date.date(), "latest_data_date": max_data_date.date(), "next_action": "hold", "reason": "holding_but_no_position_rows"}])
    else:
        candidates = buy_df[buy_df["is_buy_signal"] == True].copy()
        if candidates.empty:
            if fallback_candidate and bool(fallback_candidate.get("enabled", False)):
                decision = pd.DataFrame(
                    [
                        {
                            "data_date": data_date.date(),
                            "latest_data_date": max_data_date.date(),
                            "symbol": fallback_candidate["symbol"],
                            "display_name": fallback_candidate["display_name"],
                            "category": "宽基/红利 fallback",
                            "strategy_id": f"FB_{fallback_candidate['param_id']}",
                            "decision_type": "fallback_buy",
                            "next_action": "next_trade_close_buy",
                            "reason": fallback_candidate["reason"],
                            "planned_hold_days": fallback_candidate["hold_days"],
                            "idle_trade_days": fallback_candidate["idle_trade_days"],
                            "required_idle_days": fallback_candidate["required_idle_days"],
                            "profile": fallback_candidate["profile"],
                            "entry_date": fallback_candidate["entry_date"],
                            "ret20": fallback_candidate["ret20"],
                            "ytd": fallback_candidate["ytd"],
                            "above_ma60": fallback_candidate["above_ma60"],
                        }
                    ]
                )
            else:
                decision = pd.DataFrame(
                    [
                        {
                            "data_date": data_date.date(),
                            "latest_data_date": max_data_date.date(),
                            "next_action": "wait",
                            "reason": fallback_candidate["reason"] if fallback_candidate else "no_buy_signal",
                            "decision_type": "wait",
                            "profile": live_cfg.get("profile", ""),
                        }
                    ]
                )
        else:
            decision = candidates.head(1).copy()
            decision["decision_type"] = "main_buy"
            decision["reason"] = "best_buy_signal_when_empty"
    decision.to_csv(LIVE / "today_decision.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data_date": str(data_date.date()),
        "latest_data_date": str(max_data_date.date()),
        "run_date": str(pd.Timestamp.today().date()),
        "data_lag_days": int((pd.Timestamp.today().normalize() - data_date.normalize()).days),
        "latest_data_lag_days": int((pd.Timestamp.today().normalize() - max_data_date.normalize()).days),
        "stale_symbol_count": int(stale_count),
        "active_strategy_categories": int(len(active)),
        "missing_rule_categories": int(len(missing_rules)),
        "buy_signal_count": int(buy_df["is_buy_signal"].sum()) if not buy_df.empty else 0,
        "holding_rows": int(len(holding_rows)),
        "fallback_enabled": bool(fallback_candidate and fallback_candidate.get("enabled", False)),
        "fallback_reason": fallback_candidate.get("reason", "") if fallback_candidate else "",
        "live_profile": str(live_cfg.get("profile", "")),
        "extrema_exit_profile": str(runtime_cfg["extrema_exit"].get("profile_id", "")),
        "decision_file": "07_live/today_decision.csv",
    }
    (LIVE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    live_status = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile": live_cfg.get("profile", ""),
        "fallback": fallback_candidate,
        "schedule": live_cfg.get("schedule", {}),
    }
    (LIVE / "live_model_status.json").write_text(json.dumps(live_status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
