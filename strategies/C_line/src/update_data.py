from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from c2lib import RAW, ensure_dirs, etf_path, load_pool, normalize_local_ohlcv, read_etf, write_etf


def fetch_eastmoney_direct(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    symbol = str(symbol).zfill(6)
    secid = f"1.{symbol}" if symbol.startswith(("5", "6")) else f"0.{symbol}"
    fqt = {"": "0", "qfq": "1", "hfq": "2"}[adjust]
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": fqt,
        "beg": start,
        "end": end,
        "secid": secid,
    }
    session = requests.Session()
    session.trust_env = False
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    resp = session.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    data = (payload.get("data") or {}).get("klines") or []
    rows = []
    for line in data:
        parts = line.split(",")
        if len(parts) < 11:
            continue
        rows.append(
            {
                "日期": parts[0],
                "开盘": parts[1],
                "收盘": parts[2],
                "最高": parts[3],
                "最低": parts[4],
                "成交量": parts[5],
                "成交额": parts[6],
                "振幅": parts[7],
                "涨跌幅": parts[8],
                "涨跌额": parts[9],
                "换手率": parts[10],
            }
        )
    return pd.DataFrame(rows)


def fetch_tencent_etf(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    symbol = str(symbol).zfill(6)
    prefix = "sh" if symbol.startswith(("5", "6")) else "sz"
    adjust_flag = "qfq" if adjust == "qfq" else ("hfq" if adjust == "hfq" else "")
    # Tencent rejects very large day counts for some ETFs. Incremental daily
    # updates only need recent bars; 1000 trading days leaves a wide overlap.
    count = 1000
    param = f"{prefix}{symbol},day,,,{count},{adjust_flag}" if adjust_flag else f"{prefix}{symbol},day,,,{count}"
    session = requests.Session()
    session.trust_env = True
    resp = session.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": param},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    item = (payload.get("data") or {}).get(f"{prefix}{symbol}") or {}
    if isinstance(item, dict):
        rows_raw = item.get("day") or item.get("qfqday") or item.get("hfqday") or []
    elif isinstance(item, list):
        rows_raw = item
    else:
        rows_raw = []
    rows = []
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    for r in rows_raw:
        if len(r) < 6:
            continue
        dt = pd.to_datetime(r[0])
        if dt < start_dt or dt > end_dt:
            continue
        open_, close, high, low = map(float, [r[1], r[2], r[3], r[4]])
        volume_shares = float(r[5]) * 100.0
        avg_price = (open_ + close + high + low) / 4
        rows.append(
            {
                "日期": r[0],
                "开盘": open_,
                "收盘": close,
                "最高": high,
                "最低": low,
                "成交量": volume_shares,
                "成交额": volume_shares * avg_price,
                "涨跌幅": pd.NA,
                "换手率": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def fetch_akshare_etf(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    import akshare as ak

    return ak.fund_etf_hist_em(
        symbol=str(symbol).zfill(6),
        period="daily",
        start_date=start,
        end_date=end,
        adjust=adjust,
    )


def fetch_etf(symbol: str, start: str, end: str, adjust: str, source: str) -> pd.DataFrame:
    if source == "tencent":
        return fetch_tencent_etf(symbol, start, end, adjust)
    if source == "eastmoney_direct":
        return fetch_eastmoney_direct(symbol, start, end, adjust)
    if source == "akshare":
        return fetch_akshare_etf(symbol, start, end, adjust)
    try:
        return fetch_tencent_etf(symbol, start, end, adjust)
    except Exception:
        try:
            return fetch_eastmoney_direct(symbol, start, end, adjust)
        except Exception:
            return fetch_akshare_etf(symbol, start, end, adjust)


def start_from_existing(symbol: str, fallback: str, refresh_days: int) -> str:
    path = etf_path(symbol)
    if not path.exists():
        return fallback
    old = read_etf(symbol)
    if old.empty:
        return fallback
    latest = old["date"].max().date()
    start = latest - timedelta(days=refresh_days)
    return start.strftime("%Y%m%d")


def update_one(
    symbol: str,
    name: str,
    start: str,
    end: str,
    adjust: str,
    force_full: bool,
    refresh_days: int,
    retries: int,
    source: str,
) -> dict:
    symbol = str(symbol).zfill(6)
    old_path = etf_path(symbol)
    old_rows = 0
    old_latest = ""
    if old_path.exists():
        old = read_etf(symbol)
        old_rows = len(old)
        old_latest = old["date"].max().strftime("%Y-%m-%d") if len(old) else ""
    else:
        old = pd.DataFrame()

    fetch_start = start if force_full else start_from_existing(symbol, start, refresh_days)
    last_error = None
    for attempt in range(retries + 1):
        try:
            raw_new = fetch_etf(symbol, fetch_start, end, adjust, source)
            break
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0 + attempt)
            else:
                raw_new = None
    try:
        if raw_new is None:
            raise last_error if last_error else RuntimeError("download failed")
        new = normalize_local_ohlcv(raw_new, symbol, name)
        if new.empty:
            return {
                "symbol": symbol,
                "name": name,
                "status": "empty",
                "old_rows": old_rows,
                "new_rows": 0,
                "final_rows": old_rows,
                "old_latest": old_latest,
                "final_latest": old_latest,
                "fetch_start": fetch_start,
            }
        if old.empty or force_full:
            merged = new
        else:
            old_copy = old.copy()
            old_copy["date"] = old_copy["date"].dt.strftime("%Y-%m-%d")
            merged = pd.concat([old_copy, new], ignore_index=True)
            merged = merged.sort_values("date").drop_duplicates("date", keep="last")
        write_etf(symbol, merged)
        final_latest = pd.to_datetime(merged["date"]).max().strftime("%Y-%m-%d")
        return {
            "symbol": symbol,
            "name": name,
            "status": "ok",
            "old_rows": old_rows,
            "new_rows": len(new),
            "final_rows": len(merged),
            "old_latest": old_latest,
            "final_latest": final_latest,
            "fetch_start": fetch_start,
        }
    except Exception as exc:
        fallback_status = "error"
        try:
            if old_latest and pd.to_datetime(old_latest) >= pd.to_datetime(end):
                fallback_status = "reuse_current"
        except Exception:
            fallback_status = "error"
        return {
            "symbol": symbol,
            "name": name,
            "status": fallback_status,
            "error": repr(exc),
            "old_rows": old_rows,
            "new_rows": 0,
            "final_rows": old_rows,
            "old_latest": old_latest,
            "final_latest": old_latest,
            "fetch_start": fetch_start,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Update C-line ETF daily data into raw/etf.")
    parser.add_argument("--symbols", nargs="*", help="ETF symbols. Default: all enabled pool symbols.")
    parser.add_argument("--start", default="20100101", help="Full-download start date, YYYYMMDD.")
    parser.add_argument("--end", default=datetime.today().strftime("%Y%m%d"), help="End date, YYYYMMDD.")
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="Price adjustment. qfq=front adjusted.")
    parser.add_argument("--source", default="auto", choices=["tencent", "eastmoney_direct", "akshare", "auto"], help="Market data source.")
    parser.add_argument("--force-full", action="store_true", help="Overwrite with full download.")
    parser.add_argument("--refresh-days", type=int, default=14, help="Incremental refresh overlap days.")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay seconds between symbols.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for each symbol.")
    args = parser.parse_args()

    ensure_dirs()
    # Raw data must remain refreshable even if yesterday's quality gate failed.
    # Strategy readers still enforce the gate before producing any decision.
    pool = load_pool(enabled_only=True, enforce_quality=False)
    if args.symbols:
        wanted = {str(x).zfill(6) for x in args.symbols}
        pool = pool[pool["symbol"].astype(str).str.zfill(6).isin(wanted)]
    rows = []
    for _, row in pool.iterrows():
        rows.append(
            update_one(
                symbol=str(row["symbol"]).zfill(6),
                name=str(row.get("display_name", row.get("name", ""))),
                start=args.start,
                end=args.end,
                adjust=args.adjust,
                force_full=args.force_full,
                refresh_days=args.refresh_days,
                retries=args.retries,
                source=args.source,
            )
        )
        if args.delay:
            time.sleep(args.delay)
    log = pd.DataFrame(rows)
    log_path = RAW / "update_log.csv"
    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    final_latest = pd.to_datetime(log["final_latest"], errors="coerce") if len(log) else pd.Series(dtype="datetime64[ns]")
    max_latest = final_latest.max().strftime("%Y-%m-%d") if len(final_latest.dropna()) else ""
    min_latest = final_latest.min().strftime("%Y-%m-%d") if len(final_latest.dropna()) else ""
    stale_count = int((final_latest < final_latest.max()).sum()) if len(final_latest.dropna()) else 0
    summary = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": int(len(log)),
        "ok": int((log["status"] == "ok").sum()) if len(log) else 0,
        "empty": int((log["status"] == "empty").sum()) if len(log) else 0,
        "reuse_current": int((log["status"] == "reuse_current").sum()) if len(log) else 0,
        "error": int((log["status"] == "error").sum()) if len(log) else 0,
        "min_latest": min_latest,
        "max_latest": max_latest,
        "stale_count": stale_count,
        "log": str(log_path.relative_to(RAW.parent)),
    }
    (RAW / "update_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
