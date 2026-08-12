from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from s1lib import configure_stdout_utf8
from shadow_model import LIVE, ShadowResearchEngine


CFG = Path(__file__).resolve().parents[1] / "cfg" / "s1_model.json"
STATE = Path(__file__).resolve().parents[1] / "live" / "model_state.json"
SELECTION = Path(__file__).resolve().parents[1] / "fit" / "selector" / "selected_by_year.csv"


def json_ready(obj):
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def main() -> None:
    configure_stdout_utf8()
    engine = ShadowResearchEngine()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    asof_date = pd.Timestamp(engine.calendar.iloc[-1])
    selection = pd.read_csv(SELECTION)
    selection["year"] = pd.to_numeric(selection["year"], errors="coerce")
    year_rows = selection[selection["year"] == asof_date.year]
    row = year_rows.iloc[-1].to_dict() if not year_rows.empty else {
        "year": asof_date.year,
        "param_id": "CASH",
        "reason": "missing_frozen_annual_selection",
    }
    annual_cash = str(row.get("param_id", "")).upper() == "CASH"
    fresh_symbols = {
        symbol
        for symbol, df in engine.panel.items()
        if not df.empty and pd.Timestamp(df["date"].max()) >= asof_date
    }
    stale_rows = []
    for symbol, df in engine.panel.items():
        latest = pd.Timestamp(df["date"].max()) if not df.empty else pd.NaT
        if symbol not in fresh_symbols:
            stale_rows.append(
                {
                    "asof_date": asof_date.date(),
                    "symbol": symbol,
                    "display_name": engine.meta.get(symbol, {}).get("display_name", symbol),
                    "latest_date": latest.date() if pd.notna(latest) else "",
                }
            )
    pd.DataFrame(stale_rows).to_csv(LIVE / "stale_symbols.csv", index=False, encoding="utf-8-sig")
    picks = () if annual_cash else engine.candidate_list(
        asof_date.value,
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
    picks = tuple((score, symbol, extra) for score, symbol, extra in picks if symbol in fresh_symbols)
    rows = []
    for score, symbol, extra in picks[:10]:
        rows.append(
            {
                "asof_date": asof_date.date(),
                "symbol": symbol,
                "display_name": engine.meta[symbol]["display_name"],
                "category": engine.meta[symbol]["category"],
                "score": score,
                **extra,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(LIVE / "today_shadow_candidates.csv", index=False, encoding="utf-8-sig")
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    last_activity = pd.to_datetime(state.get("last_activity_date"), errors="coerce")
    idle_elapsed = (
        int(((engine.calendar > last_activity) & (engine.calendar <= asof_date)).sum())
        if pd.notna(last_activity)
        else 0
    )
    idle_required = 0 if annual_cash else int(row["idle"])
    if cfg.get("status") not in {"live_active", "live_active_clean"}:
        action = "research_hold"
        reason = str(cfg.get("disabled_reason", "策略尚未通过严格实盘验收"))
    elif annual_cash:
        action = "wait_cash_rule"
        reason = "本年度滚动选择为空仓，不启用空仓补偿。"
    elif not state.get("initialized"):
        action = "account_state_required"
        reason = "账户状态尚未初始化"
    elif state.get("position_status") == "holding":
        action = "hold"
        reason = "当前已有持仓"
    elif idle_elapsed < idle_required:
        action = "wait_idle"
        reason = f"空仓 {idle_elapsed}/{idle_required} 个交易日"
    elif out.empty:
        action = "wait_no_candidate"
        reason = "没有合格候选"
    else:
        action = "next_trade_close_buy"
        reason = "空仓门槛已满足，买入排名第一候选"
    decision = {
        "asof_date": str(asof_date.date()),
        "action": action,
        "reason": reason,
        "symbol": str(out.iloc[0]["symbol"]) if len(out) else "",
        "display_name": str(out.iloc[0]["display_name"]) if len(out) else "",
        "score": float(out.iloc[0]["score"]) if len(out) else None,
        "idle_trade_days": idle_elapsed,
        "required_idle_days": idle_required,
        "planned_hold_days": 0 if annual_cash else int(row["hold"]),
        "state_mode": state.get("mode", "unknown"),
        "actual_account_confirmed": bool(state.get("actual_account_confirmed", False)),
    }
    pd.DataFrame([decision]).to_csv(LIVE / "today_decision.csv", index=False, encoding="utf-8-sig")
    summary = {
        "asof_date": str(asof_date.date()),
        "selector_score_kind": cfg["selector"]["score_kind"],
        "selector_scheme": cfg["selector"]["scheme"],
        "current_param": row,
        "candidate_count": int(len(out)),
        "top_symbol": out.iloc[0]["symbol"] if len(out) else "",
        "top_name": out.iloc[0]["display_name"] if len(out) else "",
        "fresh_symbol_count": int(len(fresh_symbols)),
        "stale_symbol_count": int(len(stale_rows)),
        "stale_symbols_file": str((LIVE / "stale_symbols.csv").relative_to(LIVE.parent)),
        "decision": decision,
    }
    summary_clean = json_ready(summary)
    (LIVE / "today_shadow_summary.json").write_text(json.dumps(summary_clean, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary_clean, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
