from __future__ import annotations

import json
import time
from pathlib import Path

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "raw" / "pool.csv"
OUT = ROOT / "factors" / "raw"
TURNOVER_DIR = OUT / "turnover"
VALUATION_DIR = OUT / "valuation"


def retry(call, attempts: int = 4):
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # External endpoints occasionally reset connections.
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(error))


def download_turnover(symbol: str) -> dict:
    frame = retry(
        lambda: ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date="20000101",
            end_date="20500101",
            adjust="",
        )
    )
    rename = {
        "日期": "date",
        "成交量": "source_volume",
        "成交额": "source_amount",
        "换手率": "turnover_rate",
    }
    frame = frame.rename(columns=rename)
    required = ["date", "turnover_rate"]
    if any(column not in frame.columns for column in required):
        raise ValueError(f"{symbol}: turnover columns missing")
    keep = [column for column in ["date", "turnover_rate", "source_volume", "source_amount"] if column in frame.columns]
    frame = frame[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    frame.to_csv(TURNOVER_DIR / f"{symbol}.csv", index=False, encoding="utf-8-sig")
    return {
        "symbol": symbol,
        "rows": int(len(frame)),
        "start": str(frame["date"].min().date()) if len(frame) else "",
        "end": str(frame["date"].max().date()) if len(frame) else "",
        "coverage": float(frame["turnover_rate"].notna().mean()) if len(frame) else 0.0,
    }


def download_valuation(name: str) -> dict:
    pe = retry(lambda: ak.stock_index_pe_lg(symbol=name))
    pb = retry(lambda: ak.stock_index_pb_lg(symbol=name))
    pe.to_csv(VALUATION_DIR / f"{name}_pe.csv", index=False, encoding="utf-8-sig")
    pb.to_csv(VALUATION_DIR / f"{name}_pb.csv", index=False, encoding="utf-8-sig")
    return {"name": name, "pe_rows": int(len(pe)), "pb_rows": int(len(pb))}


def main() -> None:
    TURNOVER_DIR.mkdir(parents=True, exist_ok=True)
    VALUATION_DIR.mkdir(parents=True, exist_ok=True)
    symbols = (
        pd.read_csv(POOL, dtype={"symbol": str})
        .query("enabled == True and clean_pool == True")["symbol"]
        .astype(str)
        .str.zfill(6)
        .tolist()
    )
    turnover = []
    failures = []
    for symbol in symbols:
        try:
            turnover.append(download_turnover(symbol))
        except Exception as exc:
            failures.append({"kind": "turnover", "item": symbol, "error": str(exc)})

    valuations = []
    for name in ["上证50", "沪深300", "中证500", "中证1000", "上证红利", "创业板50"]:
        try:
            valuations.append(download_valuation(name))
        except Exception as exc:
            failures.append({"kind": "valuation", "item": name, "error": str(exc)})

    summary = {
        "symbols_requested": len(symbols),
        "turnover_downloaded": len(turnover),
        "valuation_downloaded": len(valuations),
        "turnover": turnover,
        "valuation": valuations,
        "failures": failures,
    }
    (OUT / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Keep Windows terminals with legacy code pages from failing after files are saved.
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
