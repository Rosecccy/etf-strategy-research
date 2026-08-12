from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "out"
TEMPLATE = Path(__file__).with_name("ma120_trade_template.html")
VIS_DIR = Path(r"C:\Users\10619\.codex\visualizations\2026\06\21\019ee960-4ad4-7931-9173-60478085fa8b")


def clean_number(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return round(float(number), 6) if pd.notna(number) else None


def trade_record(row: pd.Series) -> dict:
    return {
        "entry_date": str(row["entry_date"]),
        "exit_date": str(row["exit_date"]) if pd.notna(row["exit_date"]) and str(row["exit_date"]) else "",
        "entry_close": clean_number(row["entry_close"]),
        "exit_close": clean_number(row["exit_close"]),
        "return_rate": clean_number(row["return_rate"]),
        "status": str(row["status"]),
    }


def main() -> None:
    trades = pd.read_csv(OUT / "ma120_trade_log.csv", encoding="utf-8-sig", dtype={"symbol": str})
    trades["symbol"] = trades["symbol"].astype(str).str.zfill(6)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    summary = json.loads((OUT / "ma120_audit_summary.json").read_text(encoding="utf-8"))

    year_symbols: dict[int, set[str]] = {}
    for _, row in trades.iterrows():
        year_symbols.setdefault(int(row["entry_date"].year), set()).add(row["symbol"])
        if pd.notna(row["exit_date"]):
            year_symbols.setdefault(int(row["exit_date"].year), set()).add(row["symbol"])

    by_year: dict[str, dict] = {}
    raw_dir = PROJECT / "C" / "raw" / "etf"
    for year, symbols in sorted(year_symbols.items()):
        by_year[str(year)] = {}
        for symbol in sorted(symbols):
            related = trades[
                (trades["symbol"].eq(symbol))
                & ((trades["entry_date"].dt.year.eq(year)) | (trades["exit_date"].dt.year.eq(year)))
            ].copy()
            raw = pd.read_csv(raw_dir / f"{symbol}.csv", encoding="utf-8-sig")
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
            raw = raw[(raw["date"].dt.year.eq(year)) & raw["close"].notna()].sort_values("date")
            if raw.empty:
                continue
            buy_rows = related[related["entry_date"].dt.year.eq(year)]
            sell_rows = related[related["exit_date"].dt.year.eq(year)]
            by_year[str(year)][symbol] = {
                "symbol": symbol,
                "name": str(related.iloc[0]["name"]),
                "prices": [
                    {"date": item.date.date().isoformat(), "close": round(float(item.close), 6)}
                    for item in raw[["date", "close"]].itertuples(index=False)
                ],
                "buys": [
                    {"date": item.entry_date.date().isoformat(), "price": clean_number(item.entry_close)}
                    for item in buy_rows.itertuples(index=False)
                ],
                "sells": [
                    {"date": item.exit_date.date().isoformat(), "price": clean_number(item.exit_close)}
                    for item in sell_rows.itertuples(index=False)
                ],
                "trades": [trade_record(row) for _, row in related.iterrows()],
            }

    data = {
        "summary": {
            "win_rate": summary["win_rate"],
            "total_pnl": summary["total_mark_to_market_pnl_cny"],
            "max_notional": summary["max_concurrent_notional_cny_conservative"],
        },
        "years": list(by_year),
        "byYear": by_year,
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    if template.count("__DATA__") != 1:
        raise RuntimeError("Visualization template must contain exactly one data marker.")
    fragment = template.replace("__DATA__", payload)
    if '\\"' in fragment or "\\n" in fragment:
        raise RuntimeError("Visualization fragment contains escaped markup.")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    (VIS_DIR / "ma120-trade-points.html").write_text(fragment, encoding="utf-8")
    (OUT / "ma120-trade-points-fragment.html").write_text(fragment, encoding="utf-8")
    print(json.dumps({"years": list(by_year), "bytes": len(fragment.encode("utf-8"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
