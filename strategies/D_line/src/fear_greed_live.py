from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import factor_dca_scan as dca
import fear_greed_oos as fg
import gate_nested_oos as gated


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "live"
SELECTIONS = ROOT / "out" / "ma120_selected_rules.csv"
C_RAW_ETF = ROOT.parent / "C" / "raw" / "etf"
QVIX_WEIGHT = 0.20
ENTRY_THRESHOLD = 45.0
GREED_EXIT = 70.0
MAX_HOLD = 90


def mask_from_id(identifier: str, conditions: dict[str, np.ndarray]) -> np.ndarray:
    factors = [item.strip() for item in str(identifier).split(" + ")]
    missing = [item for item in factors if item not in conditions]
    if missing:
        raise KeyError(f"Missing condition(s): {missing}")
    result = np.ones(len(next(iter(conditions.values()))), dtype=bool)
    for factor in factors:
        result &= conditions[factor]
    return result


def next_weekday(date: pd.Timestamp) -> str:
    return str((date + BDay(1)).date())


def sync_clean_raw_from_c() -> dict[str, object]:
    """Append new C-line rows without replacing historically repaired prices."""
    updated_symbols: list[str] = []
    appended_rows = 0
    latest_dates: list[pd.Timestamp] = []

    for clean_path in sorted(fg.CLEAN_RAW.glob("*.csv")):
        source_path = C_RAW_ETF / clean_path.name
        if not source_path.exists():
            continue

        clean = pd.read_csv(clean_path, dtype={"symbol": str}, encoding="utf-8-sig")
        source = pd.read_csv(source_path, dtype={"symbol": str}, encoding="utf-8-sig")
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
        source["date"] = pd.to_datetime(source["date"], errors="coerce")
        clean = clean.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
        source = source.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
        if clean.empty or source.empty:
            continue

        additions = source[source["date"] > clean["date"].max()].copy()
        if not additions.empty:
            for column in clean.columns:
                if column not in additions:
                    additions[column] = np.nan
            merged = pd.concat(
                [clean, additions[clean.columns]],
                ignore_index=True,
            ).sort_values("date")
            merged = merged.drop_duplicates("date", keep="first").reset_index(drop=True)
            if "pct_change" in merged:
                merged["pct_change"] = (
                    pd.to_numeric(merged["close"], errors="coerce").pct_change() * 100
                )
            merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
            merged.to_csv(clean_path, index=False, encoding="utf-8-sig")
            updated_symbols.append(clean_path.stem)
            appended_rows += len(additions)
            latest_dates.append(pd.to_datetime(merged["date"]).max())
        else:
            latest_dates.append(clean["date"].max())

    return {
        "updated_symbols": len(updated_symbols),
        "appended_rows": appended_rows,
        "latest_date": (
            str(max(latest_dates).date()) if latest_dates else None
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    LIVE.mkdir(parents=True, exist_ok=True)

    sync_summary = sync_clean_raw_from_c()
    raw, pool = fg.load_clean_raw()
    sentiment = fg.build_sentiment(raw)
    symbol_fear = fg.build_symbol_fear(raw)
    latest_sentiment = sentiment.dropna(subset=["fear_core"]).iloc[-1]
    fear_score = (
        (1 - QVIX_WEIGHT) * float(latest_sentiment["fear_core"])
        + QVIX_WEIGHT * float(latest_sentiment["fear_qvix"])
        if pd.notna(latest_sentiment["fear_qvix"])
        else float(latest_sentiment["fear_core"])
    )
    greed_score = 100 - fear_score

    panel, _ = dca.load_panel(non_daily_signal_on_period_close=True)
    buy_conditions, _, _ = dca.build_conditions(panel, "buy")
    sell_conditions, _, _ = dca.build_conditions(panel, "sell")
    selection = pd.read_csv(SELECTIONS, encoding="utf-8-sig")
    current_year = int(pd.Timestamp(panel["date"].max()).year)
    chosen = selection[selection["test_year"].eq(current_year)].iloc[-1]

    buy_mask = mask_from_id(str(chosen["buy_id"]), buy_conditions)
    buy_mask &= gated.gates(panel)["self_ma120"]
    sell_mask = mask_from_id(str(chosen["sell_id"]), sell_conditions)
    latest_indices = panel.groupby("symbol", sort=False).tail(1).index
    latest_rows = panel.loc[
        latest_indices, ["date", "symbol", "name", "close", "amount"]
    ].copy()
    latest_rows["technical_buy"] = buy_mask[latest_indices]
    latest_rows["technical_sell"] = sell_mask[latest_indices]

    latest_symbol_fear = (
        symbol_fear.sort_values("date")
        .groupby("symbol", sort=False)
        .tail(1)
        .set_index("symbol")["symbol_fear"]
    )
    latest_rows["symbol_fear"] = latest_rows["symbol"].map(latest_symbol_fear)
    latest_rows["buy_quality"] = (
        0.5 * fear_score + 0.5 * latest_rows["symbol_fear"]
    )
    latest_rows["amount"] = pd.to_numeric(
        latest_rows["amount"], errors="coerce"
    ).fillna(0)
    latest_rows = latest_rows.sort_values(
        ["buy_quality", "amount", "symbol"],
        ascending=[False, False, True],
    )

    technical_buys = latest_rows[latest_rows["technical_buy"]].copy()
    allowed_buys = (
        technical_buys if fear_score >= ENTRY_THRESHOLD else technical_buys.iloc[0:0]
    )
    top_buy = allowed_buys.head(1)
    data_date = pd.Timestamp(latest_rows["date"].max())
    execution_date = next_weekday(data_date)

    if not top_buy.empty:
        row = top_buy.iloc[0]
        action = "BUY"
        symbol = str(row["symbol"]).zfill(6)
        name = str(row["name"])
        reason = (
            f"技术买点成立，恐惧分{fear_score:.2f}>={ENTRY_THRESHOLD:.0f}，"
            "单仓按恐惧质量排序第一。"
        )
    else:
        action = "WAIT"
        symbol = ""
        name = ""
        reason = (
            "恐惧过滤已通过，但今日没有技术买点。"
            if fear_score >= ENTRY_THRESHOLD
            else f"今日恐惧分{fear_score:.2f}低于{ENTRY_THRESHOLD:.0f}，不放行买点。"
        )

    decision = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_sync": sync_summary,
        "data_date": str(data_date.date()),
        "expected_execution_date": execution_date,
        "action": action,
        "symbol": symbol,
        "name": name,
        "execution_price": "下一交易日收盘价",
        "positions": 1,
        "ranking": "0.5*市场恐惧分 + 0.5*ETF自身恐惧分",
        "fear_score": fear_score,
        "greed_score": greed_score,
        "entry_threshold": ENTRY_THRESHOLD,
        "greed_exit": GREED_EXIT,
        "max_hold_days": MAX_HOLD,
        "technical_buy_count": int(len(technical_buys)),
        "allowed_buy_count": int(len(allowed_buys)),
        "technical_sell_count": int(latest_rows["technical_sell"].sum()),
        "buy_rule_id": str(chosen["buy_id"]),
        "sell_rule_id": str(chosen["sell_id"]),
        "reason": reason,
        "note": "卖出只对真实持仓检查：技术卖点、持有满7日后的贪婪70退出、或90日上限，均在下一交易日收盘执行。",
    }
    (LIVE / "today.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([decision]).to_csv(
        LIVE / "today.csv", index=False, encoding="utf-8-sig"
    )
    latest_rows.to_csv(
        LIVE / "market_scan.csv", index=False, encoding="utf-8-sig"
    )
    strategy = {
        "name": "FG1恐惧贪婪单仓影子策略",
        "status": "shadow_live",
        "market_fear": "80%内部广度/波动/回撤 + 20% 50ETF Q-VIX历史分位",
        "entry": "D线年度滚动技术买点 AND 恐惧分>=45",
        "selection": "同日只选买入质量最高的一只ETF",
        "exit": "技术卖点优先；持有至少7日后贪婪分>=70；最长90交易日",
        "execution": "收盘确认，下一交易日收盘执行",
        "capital": "单仓，100股整数手，手续费万三且单边最低5元",
    }
    (LIVE / "strategy.json").write_text(
        json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
