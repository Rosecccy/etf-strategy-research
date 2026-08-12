from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from c2_pipeline import load_raw
from c2lib import ACCOUNT_DIR, load_pool


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "cfg" / "live_model.json"
IDLE_DIR = ROOT / "out" / "idle"
MODEL_STATE = ACCOUNT_DIR / "model_state.json"

BROAD_NAMES = {
    "159902": "深成ETF",
    "510880": "红利ETF",
}


def load_live_config() -> dict:
    env_path = os.environ.get("C2_LIVE_CONFIG", "").strip()
    path = Path(env_path) if env_path else CFG
    if not path.is_absolute():
        path = ROOT / path
    # Windows editors may add a UTF-8 BOM; accept both UTF-8 variants for live config.
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    data["_config_path"] = str(path)
    return data


def active_guard_rule(cfg: dict) -> dict:
    profile = str(cfg.get("profile", "") or "")
    rules = cfg.get("fallback", {}).get("profiles", {})
    if profile not in rules:
        raise KeyError(f"unknown live profile: {profile}")
    return rules[profile]


def guard_enabled(cfg: dict) -> bool:
    return bool(cfg.get("fallback", {}).get("guard_enabled", True))


def annual_rollover_policy(cfg: dict) -> str:
    return str(cfg.get("fallback", {}).get("annual_rollover_policy", "allow") or "allow")


def selection_file_path(cfg: dict) -> Path | None:
    rel = str(cfg.get("fallback", {}).get("selection_file", "") or "").strip()
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    return path


def apply_variant_gate(row: pd.Series, variant: str) -> bool:
    mode = str(row.get("mode", ""))
    hold = int(row.get("hold", 0) or 0)
    if variant == "raw":
        return True
    if variant == "profit_max":
        return not (mode == "rotate_ma_mom" and hold <= 20)
    if variant == "stable_max":
        return mode != "rotate_ma_mom"
    raise ValueError(f"unknown fallback variant: {variant}")


def frozen_selection(asof_year: int, cfg: dict) -> dict | None:
    path = selection_file_path(cfg)
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "year" not in df.columns:
        return None
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    hit = df[df["year"] == asof_year]
    if hit.empty:
        hit = df[df["year"] <= asof_year].sort_values("year").tail(1)
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def current_fallback_selection(asof_year: int, cfg: dict) -> dict | None:
    frozen = frozen_selection(asof_year, cfg)
    if frozen is not None:
        return frozen
    raise FileNotFoundError(
        "The frozen rolling-OOS fallback selection is missing. "
        "Live mode will not rebuild or re-optimize a strategy implicitly."
    )


def last_trade_date() -> pd.Timestamp | None:
    path = ACCOUNT_DIR / "trades.csv"
    if path.exists():
        trades = pd.read_csv(path, dtype={"symbol": str})
        if not trades.empty and "date" in trades.columns:
            trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
            trades = trades.dropna(subset=["date"])
            if not trades.empty:
                return pd.Timestamp(trades["date"].max())
    if MODEL_STATE.exists():
        state = json.loads(MODEL_STATE.read_text(encoding="utf-8"))
        dt = pd.to_datetime(state.get("last_activity_date"), errors="coerce")
        if pd.notna(dt):
            return pd.Timestamp(dt)
    return None


def idle_trade_days(asof_date: pd.Timestamp, calendar: pd.Series) -> int:
    last_dt = last_trade_date()
    if last_dt is None:
        return 0
    return int(((calendar > last_dt) & (calendar <= asof_date)).sum())


def latest_row_on_or_before(df: pd.DataFrame, asof_date: pd.Timestamp) -> pd.Series | None:
    hit = df[df["date"] <= asof_date]
    if hit.empty:
        return None
    return hit.iloc[-1]


def next_trade_date(df: pd.DataFrame, asof_date: pd.Timestamp) -> pd.Timestamp | None:
    hit = df[df["date"] > asof_date]
    if hit.empty:
        return None
    return pd.Timestamp(hit.iloc[0]["date"])


def projected_next_trade_date(df: pd.DataFrame, asof_date: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    exact = next_trade_date(df, asof_date)
    if exact is not None:
        return exact, "observed_trading_calendar"
    return pd.Timestamp(asof_date) + pd.offsets.BDay(1), "business_day_projection"


def projected_fallback_exit_date(symbol: str, entry_date: pd.Timestamp, hold_days: int) -> tuple[pd.Timestamp, str]:
    raw = load_raw(symbol)[["date"]].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    dates = pd.DatetimeIndex(raw["date"].dropna().drop_duplicates().sort_values())
    entry = pd.Timestamp(entry_date).normalize()
    pos = int(dates.searchsorted(entry, side="left"))
    exit_pos = pos + int(hold_days)
    if pos < len(dates) and dates[pos].normalize() == entry and exit_pos < len(dates):
        return pd.Timestamp(dates[exit_pos]), "observed_trading_calendar"
    return entry + pd.offsets.BDay(int(hold_days)), "business_day_projection"


def fallback_entry_crosses_year(symbol: str, asof_date: pd.Timestamp, hold_days: int, cfg: dict) -> dict:
    policy = annual_rollover_policy(cfg)
    raw = load_raw(symbol)
    entry_date, entry_basis = projected_next_trade_date(raw, asof_date)
    exit_date, exit_basis = projected_fallback_exit_date(symbol, entry_date, hold_days)
    crosses = bool(exit_date.year > entry_date.year)
    blocked = bool(policy == "skip_entry_if_expected_exit_crosses_year" and crosses)
    return {
        "annual_rollover_policy": policy,
        "entry_date": entry_date,
        "projected_exit_date": exit_date,
        "entry_date_basis": entry_basis,
        "exit_date_basis": exit_basis,
        "crosses_calendar_year": crosses,
        "blocked": blocked,
    }


def choose_fallback_symbol(asof_date: pd.Timestamp, mode: str, ma: int, mom: int) -> tuple[str, pd.Series] | tuple[None, None]:
    choices: list[tuple[float, str, pd.Series]] = []
    for symbol in BROAD_NAMES:
        df = load_raw(symbol)[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        if ma > 0:
            df[f"ma{ma}"] = df["close"].rolling(ma, min_periods=max(5, ma // 3)).mean()
        df[f"mom{mom}"] = df["close"].pct_change(mom)
        row = latest_row_on_or_before(df, asof_date)
        if row is None:
            continue
        close = float(row["close"])
        ma_val = float(row.get(f"ma{ma}", np.nan)) if ma > 0 else np.nan
        mom_val = float(row.get(f"mom{mom}", np.nan))
        if mode == "red_ma":
            if symbol != "510880" or not np.isfinite(ma_val) or close <= ma_val:
                continue
            score = mom_val if np.isfinite(mom_val) else 0.0
        elif mode == "rotate_ma_mom":
            if not np.isfinite(ma_val) or close <= ma_val or not np.isfinite(mom_val) or mom_val <= 0:
                continue
            score = mom_val
        elif mode == "rotate_mom":
            score = mom_val if np.isfinite(mom_val) else -np.inf
        else:
            continue
        choices.append((score, symbol, row))
    if not choices:
        return None, None
    _, symbol, row = sorted(choices, reverse=True)[0]
    return symbol, row


def symbol_features(symbol: str, asof_date: pd.Timestamp) -> dict:
    df = load_raw(symbol)[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ret20"] = df["close"].pct_change(20)
    df["ma60"] = df["close"].rolling(60, min_periods=20).mean()
    df["above_ma60"] = df["close"] > df["ma60"]
    first_by_year = df.groupby(df["date"].dt.year)["close"].transform("first")
    df["ytd"] = df["close"] / first_by_year - 1
    row = latest_row_on_or_before(df, asof_date)
    if row is None:
        return {"ret20": np.nan, "ytd": np.nan, "above_ma60": False, "close": np.nan}
    return {
        "ret20": float(row["ret20"]) if pd.notna(row["ret20"]) else np.nan,
        "ytd": float(row["ytd"]) if pd.notna(row["ytd"]) else np.nan,
        "above_ma60": bool(row["above_ma60"]) if pd.notna(row["above_ma60"]) else False,
        "close": float(row["close"]),
    }


def guard_pass(mode: str, feats: dict, rule: dict) -> tuple[bool, str]:
    ret20 = feats["ret20"]
    ytd = feats["ytd"]
    above_ma60 = feats["above_ma60"]
    if mode == "red_ma":
        if np.isfinite(ret20) and ret20 < float(rule["red_ma_ret20_min"]):
            return False, "红利20日动量未达门槛"
        if np.isfinite(ytd) and ytd < float(rule["red_ma_ytd_min"]):
            return False, "红利年内涨幅未达门槛"
        return True, "红利保护通过"
    if mode == "rotate_mom":
        if np.isfinite(ret20) and ret20 < float(rule["rotate_mom_ret20_min"]):
            return False, "动量20日收益未达门槛"
        if np.isfinite(ytd) and ytd < float(rule["rotate_mom_ytd_min"]):
            return False, "动量年内涨幅未达门槛"
        if bool(rule["rotate_mom_require_ma60"]) and not above_ma60:
            return False, "动量未站上60日均线"
        return True, "动量保护通过"
    return False, "当前 fallback 模式不在 live 开放范围"


def build_fallback_candidate(asof_date: pd.Timestamp, cfg: dict) -> dict | None:
    if not bool(cfg.get("fallback", {}).get("enabled", False)):
        return None
    selected = current_fallback_selection(asof_date.year, cfg)
    if selected is None:
        return None
    calendar = load_raw("510880")["date"]
    idle_days_now = idle_trade_days(asof_date, calendar)
    if str(selected.get("param_id", "")).upper() == "CASH":
        return {
            "enabled": False,
            "reason": "本年度滚动选择为空仓，不启用空仓补偿。",
            "idle_trade_days": idle_days_now,
            "required_idle_days": 0,
            "param_id": "CASH",
            "mode": "cash",
        }
    required_idle = int(selected["idle_days"])
    if last_trade_date() is None:
        return {
            "enabled": False,
            "reason": "账户状态尚未初始化，禁止生成空仓替代买入",
            "idle_trade_days": 0,
            "required_idle_days": required_idle,
            "param_id": selected["param_id"],
            "mode": selected["mode"],
        }
    if idle_days_now < required_idle:
        return {
            "enabled": False,
            "reason": f"当前空仓仅 {idle_days_now} 个交易日，未达到 {required_idle}",
            "idle_trade_days": idle_days_now,
            "required_idle_days": required_idle,
            "param_id": selected["param_id"],
            "mode": selected["mode"],
        }
    symbol, _ = choose_fallback_symbol(
        asof_date=asof_date,
        mode=str(selected["mode"]),
        ma=int(selected["ma"]),
        mom=int(selected["mom"]),
    )
    if symbol is None:
        return {
            "enabled": False,
            "reason": "当前宽基/红利不满足 fallback 选股条件",
            "idle_trade_days": idle_days_now,
            "required_idle_days": required_idle,
            "param_id": selected["param_id"],
            "mode": selected["mode"],
        }
    feats = symbol_features(symbol, asof_date)
    if guard_enabled(cfg):
        rule = active_guard_rule(cfg)
        ok, reason = guard_pass(str(selected["mode"]), feats, rule)
    else:
        ok, reason = True, "未启用额外 guard，按冻结年度选择执行"

    pool = load_pool(enabled_only=False)
    name_hit = pool[pool["symbol"].astype(str).str.zfill(6) == symbol]
    display_name = name_hit["display_name"].iloc[0] if not name_hit.empty else BROAD_NAMES.get(symbol, symbol)

    rollover = fallback_entry_crosses_year(
        symbol=symbol,
        asof_date=asof_date,
        hold_days=int(selected["hold"]),
        cfg=cfg,
    )
    if rollover["blocked"]:
        ok = False
        reason = (
            f"预计持有至 {pd.Timestamp(rollover['projected_exit_date']).strftime('%Y-%m-%d')}，"
            "将跨越年度模型切换，按 C2.2 规则跳过本次 fallback 开仓"
        )
    next_dt = pd.Timestamp(rollover["entry_date"])
    return {
        "enabled": bool(ok),
        "reason": reason,
        "profile": str(cfg.get("profile", "")),
        "param_id": selected["param_id"],
        "mode": selected["mode"],
        "symbol": symbol,
        "display_name": display_name,
        "idle_trade_days": idle_days_now,
        "required_idle_days": required_idle,
        "hold_days": int(selected["hold"]),
        "ma": int(selected["ma"]),
        "mom": int(selected["mom"]),
        "entry_date": next_dt.strftime("%Y-%m-%d"),
        "projected_exit_date": pd.Timestamp(rollover["projected_exit_date"]).strftime("%Y-%m-%d"),
        "annual_rollover_policy": rollover["annual_rollover_policy"],
        "crosses_calendar_year": bool(rollover["crosses_calendar_year"]),
        "date_projection_basis": f"{rollover['entry_date_basis']}|{rollover['exit_date_basis']}",
        "ret20": feats["ret20"],
        "ytd": feats["ytd"],
        "above_ma60": feats["above_ma60"],
        "close": feats["close"],
        "selection_source": str(selection_file_path(cfg)) if selection_file_path(cfg) else "rolling_select",
    }


def parse_fallback_hold_days(strategy_id: str) -> int | None:
    text = str(strategy_id or "")
    m = re.search(r"hold(\d+)", text)
    if not m:
        return None
    return int(m.group(1))
