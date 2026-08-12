from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CFG = ROOT / "cfg" / "hybrid.json"
C_DECISION = ROOT / "live" / "today_decision.csv"
S_DECISION = WORKSPACE / "S" / "live" / "today_decision.csv"
C_HISTORY = ROOT / "fit" / "execution_timing_audit" / "c_corrected_trades.csv"
S_HISTORY = ROOT / "fit" / "execution_timing_audit" / "s_corrected_trades.csv"
FORMAL = ROOT / "fit" / "hybrid_formal" / "summary.json"
DATA_QUALITY = ROOT / "raw" / "quality.json"
OUT_CSV = ROOT / "live" / "hybrid_today_decision.csv"
OUT_JSON = ROOT / "live" / "hybrid_today_decision.json"
ACCOUNT = ROOT / "live" / "hybrid_account.json"


def configure_stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_decision(path: Path) -> dict:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    symbol = str(row.get("symbol", "")).strip()
    row["symbol"] = symbol.zfill(6) if symbol and symbol.lower() != "nan" else ""
    return row


def load_history(path: Path, family: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["family"] = family
    s_quality_path = WORKSPACE / "S" / "raw" / "quality.json"
    if not s_quality_path.exists():
        raise FileNotFoundError("S-line ETF quality manifest is missing.")
    quality = json.loads(s_quality_path.read_text(encoding="utf-8"))
    if not bool(quality.get("passed")):
        raise RuntimeError("S-line ETF quality gate failed; hybrid history is blocked.")
    approved = {
        str(symbol).zfill(6)
        for symbol in quality.get("approved_symbols", [])
    }
    frame = frame[frame["symbol"].isin(approved)].copy()
    source = frame["source"].astype(str)
    return frame[~source.eq("主策略")].copy()


def fallback_score(
    history: pd.DataFrame,
    family: str,
    entry_date: pd.Timestamp,
) -> dict:
    past = history[
        history["family"].eq(family)
        & history["exit_date"].lt(entry_date)
    ].copy()
    returns = past["ret"].dropna()
    if len(returns) < 3:
        return {
            "family": family,
            "score": 0.0,
            "closed_trades": int(len(returns)),
            "mean_return": None,
            "win_rate": None,
            "eligible": True,
        }
    mean = float(returns.mean())
    win = float((returns > 0).mean())
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
    score = mean + 0.10 * (win - 0.5) - 0.25 * downside
    return {
        "family": family,
        "score": score,
        "closed_trades": int(len(returns)),
        "mean_return": mean,
        "win_rate": win,
        "downside_rms": downside,
        "eligible": bool(mean > 0),
    }


def is_buy(row: dict, family: str) -> bool:
    if family == "C":
        return (
            str(row.get("next_action", "")) == "next_trade_close_buy"
            and str(row.get("decision_type", "")) in {"main_buy", "fallback_buy"}
        )
    return str(row.get("action", "")) == "next_trade_close_buy"


def asof_date(c_row: dict, s_row: dict) -> pd.Timestamp:
    values = [
        pd.to_datetime(c_row.get("data_date"), errors="coerce"),
        pd.to_datetime(s_row.get("asof_date"), errors="coerce"),
    ]
    valid = [value for value in values if pd.notna(value)]
    if not valid:
        raise RuntimeError("C/S 每日决策均缺少有效日期")
    if len({value.normalize() for value in valid}) != 1:
        raise RuntimeError(f"C/S 决策日期不一致: {valid}")
    return max(valid).normalize()


def initialize_account(asof: pd.Timestamp) -> dict:
    if ACCOUNT.exists():
        return json.loads(ACCOUNT.read_text(encoding="utf-8"))
    trades = pd.read_csv(
        ROOT / "fit" / "hybrid_formal" / "final_trades.csv",
        dtype={"symbol": str},
        encoding="utf-8-sig",
    )
    s_quality = json.loads(
        (WORKSPACE / "S" / "raw" / "quality.json").read_text(encoding="utf-8")
    )
    approved = {
        str(symbol).zfill(6)
        for symbol in s_quality.get("approved_symbols", [])
    }
    trades["symbol"] = trades["symbol"].astype(str).str.zfill(6)
    trades = trades[trades["symbol"].isin(approved)].copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    completed = trades[trades["exit_date"].le(asof)]
    last_exit = completed["exit_date"].max() if not completed.empty else pd.NaT
    state = {
        "mode": "paper_model",
        "actual_account_confirmed": False,
        "initialized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof.strftime("%Y-%m-%d"),
        "position_status": "cash",
        "position": None,
        "last_activity_date": (
            last_exit.strftime("%Y-%m-%d") if pd.notna(last_exit) else None
        ),
        "pending_order": None,
        "note": "如有真实成交，必须先同步账户状态，系统不会自行假定成交。",
    }
    ACCOUNT.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return state


def candidate_from_c(row: dict) -> dict:
    return {
        "family": "C",
        "source": str(row.get("decision_type", "")),
        "symbol": row.get("symbol", ""),
        "display_name": str(row.get("display_name", "")),
        "planned_hold_days": row.get("planned_hold_days", ""),
        "planned_entry_date": str(row.get("entry_date", "")),
        "reason": str(row.get("reason", "")),
    }


def candidate_from_s(row: dict) -> dict:
    return {
        "family": "S",
        "source": "s_fallback",
        "symbol": row.get("symbol", ""),
        "display_name": str(row.get("display_name", "")),
        "planned_hold_days": row.get("planned_hold_days", ""),
        "planned_entry_date": "",
        "reason": str(row.get("reason", "")),
    }


def main() -> None:
    configure_stdout_utf8()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    quality = json.loads(DATA_QUALITY.read_text(encoding="utf-8"))
    if not bool(quality.get("passed")):
        raise RuntimeError("Canonical ETF data quality gate failed; hybrid decision is blocked.")
    historical_clean = bool(
        quality.get("historical_model_artifacts", {}).get("passed")
    )
    if not formal.get("passed"):
        raise RuntimeError("混合模型正式审计未通过，拒绝生成实盘建议")
    c_row = load_decision(C_DECISION)
    s_row = load_decision(S_DECISION)
    asof = asof_date(c_row, s_row)
    account = initialize_account(asof)
    history = pd.concat(
        [load_history(C_HISTORY, "C"), load_history(S_HISTORY, "S")],
        ignore_index=True,
    )

    c_buy = is_buy(c_row, "C")
    s_buy = is_buy(s_row, "S")
    c_main = c_buy and str(c_row.get("decision_type", "")) == "main_buy"
    candidates: list[dict] = []
    score_rows: list[dict] = []
    action = "wait"
    reason = "当前没有符合条件的主策略或补偿信号"
    chosen: dict = {}

    if str(account.get("position_status", "cash")) == "holding":
        action = "hold_or_follow_position_exit"
        reason = "模型账户显示已有持仓，必须先按持仓来源执行退出规则"
        chosen = dict(account.get("position") or {})
    elif c_main:
        chosen = candidate_from_c(c_row)
        action = "next_trade_close_buy"
        reason = "C 主策略优先级最高"
    else:
        if c_buy:
            candidates.append(candidate_from_c(c_row))
        if s_buy:
            candidates.append(candidate_from_s(s_row))
        entry_date = pd.to_datetime(
            c_row.get("entry_date"), errors="coerce"
        )
        if pd.isna(entry_date):
            entry_date = asof + pd.offsets.BDay(1)
        for candidate in candidates:
            score = fallback_score(history, candidate["family"], entry_date)
            score_rows.append(score)
            candidate.update(score)
        eligible = [
            item for item in candidates if bool(item.get("eligible", False))
        ]
        if eligible:
            chosen = max(
                eligible,
                key=lambda item: (
                    float(item["score"]),
                    item["family"] == "C",
                    item["symbol"],
                ),
            )
            action = "next_trade_close_buy"
            reason = (
                f"补偿候选按冻结历史分数择优，选择 {chosen['family']} 引擎"
            )

    decision = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": cfg["name"],
        "version": cfg["version"],
        "asof_date": asof.strftime("%Y-%m-%d"),
        "action": action,
        "symbol": chosen.get("symbol", ""),
        "display_name": chosen.get("display_name", ""),
        "source": chosen.get("source", ""),
        "planned_entry_date": chosen.get("planned_entry_date", ""),
        "planned_hold_days": chosen.get("planned_hold_days", ""),
        "selection_score": chosen.get("score", ""),
        "reason": reason,
        "account_mode": account.get("mode", "unknown"),
        "actual_account_confirmed": bool(
            account.get("actual_account_confirmed", False)
        ),
        "account_warning": (
            ""
            if account.get("actual_account_confirmed")
            else "当前为模型账户；真实持仓与成交需人工确认后才能作为账户指令。"
        ),
        "runtime_data_quality": "passed",
        "historical_performance_clean_recheck": (
            "passed" if historical_clean else "pending"
        ),
    }
    pd.DataFrame([decision]).to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )
    payload = {
        "decision": decision,
        "fallback_scores": score_rows,
        "c_engine": c_row,
        "s_engine": s_row,
        "formal_audit_passed": bool(formal.get("passed") and historical_clean),
        "legacy_formal_audit_passed": bool(formal.get("passed")),
        "data_quality_warning": (
            ""
            if historical_clean
            else "当前信号只读取质量白名单数据；清洗前历史收益文件仍需重新训练复核，不作为干净数据业绩证据。"
        ),
    }
    payload = json_ready(payload)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
