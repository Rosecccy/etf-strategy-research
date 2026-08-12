from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
import sys

import pandas as pd
import requests

from s1lib import RAW, ensure_dirs, etf_path, load_pool, normalize_local_ohlcv, read_etf, write_etf


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass


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
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    session = requests.Session()
    session.trust_env = False
    resp = session.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    data = (resp.json().get("data") or {}).get("klines") or []
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


def fetch_akshare_fund(symbol: str, start: str, end: str, adjust: str, mode: str) -> pd.DataFrame:
    import akshare as ak

    if mode == "etf":
        return ak.fund_etf_hist_em(
            symbol=str(symbol).zfill(6),
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
    return ak.fund_lof_hist_em(
        symbol=str(symbol).zfill(6),
        period="daily",
        start_date=start,
        end_date=end,
        adjust=adjust,
    )


def _has_rows(df: pd.DataFrame | None) -> bool:
    return df is not None and not df.empty and len(df) > 0


def prefixed_symbol(symbol: str) -> str:
    symbol = str(symbol).zfill(6)
    return f"sh{symbol}" if symbol.startswith(("5", "6")) else f"sz{symbol}"


def fetch_sina_etf(symbol: str) -> pd.DataFrame:
    import akshare as ak

    return ak.fund_etf_hist_sina(symbol=prefixed_symbol(symbol))


def fetch_tx_stock(symbol: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_zh_a_hist_tx(symbol=prefixed_symbol(symbol))
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if "amount" in df.columns and "volume" not in df.columns:
        # Tencent's `amount` here is turnover volume in "hand"; convert to
        # shares and approximate turnover amount in CNY.
        hand = pd.to_numeric(df["amount"], errors="coerce")
        avg_price = (
            pd.to_numeric(df["open"], errors="coerce")
            + pd.to_numeric(df["close"], errors="coerce")
            + pd.to_numeric(df["high"], errors="coerce")
            + pd.to_numeric(df["low"], errors="coerce")
        ) / 4
        df["volume"] = hand * 100.0
        df["amount"] = hand * 100.0 * avg_price
    return df


def fetch_tencent_direct(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    symbol = str(symbol).zfill(6)
    prefix = "sh" if symbol.startswith(("5", "6")) else "sz"
    adjust_flag = "qfq" if adjust == "qfq" else ("hfq" if adjust == "hfq" else "")
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    # Tencent may return an empty payload when the requested history is too
    # large. Fetch a wide recent overlap and merge it into the local history.
    count = 1000
    param = f"{prefix}{symbol},day,,,{count},{adjust_flag}" if adjust_flag else f"{prefix}{symbol},day,,,{count}"
    session = requests.Session()
    session.trust_env = True
    resp = session.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get", params={"param": param}, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    item = (payload.get("data") or {}).get(f"{prefix}{symbol}") or {}
    rows_raw = []
    if isinstance(item, dict):
        rows_raw = item.get("qfqday") or item.get("day") or item.get("hfqday") or []
    rows = []
    for r in rows_raw:
        if len(r) < 6:
            continue
        dt = pd.to_datetime(r[0], errors="coerce")
        if pd.isna(dt) or dt < start_dt or dt > end_dt:
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
            }
        )
    return pd.DataFrame(rows)


def fetch_with_fallback(symbol: str, start: str, end: str, adjust: str, trade_class: str, source: str) -> pd.DataFrame:
    if source == "eastmoney":
        return fetch_eastmoney_direct(symbol, start, end, adjust)
    if source == "tencent":
        return fetch_tencent_direct(symbol, start, end, adjust)
    if source == "sina_etf":
        return fetch_sina_etf(symbol)
    if source == "tx_stock":
        return fetch_tx_stock(symbol)
    if source == "akshare":
        preferred = "lof" if "LOF" in str(trade_class).upper() else "etf"
        return fetch_akshare_fund(symbol, start, end, adjust, preferred)

    errors = []
    trade_class_upper = str(trade_class).upper()
    is_lof_like = "LOF" in trade_class_upper

    if not is_lof_like:
        try:
            df = fetch_sina_etf(symbol)
            if _has_rows(df):
                return df
        except Exception as exc:
            errors.append(f"sina_etf={exc!r}")

    try:
        df = fetch_tx_stock(symbol)
        if _has_rows(df):
            return df
    except Exception as exc:
        errors.append(f"tx_stock={exc!r}")

    try:
        df = fetch_eastmoney_direct(symbol, start, end, adjust)
        if _has_rows(df):
            return df
    except Exception as exc:
        errors.append(f"eastmoney={exc!r}")
    try:
        df = fetch_tencent_direct(symbol, start, end, adjust)
        if _has_rows(df):
            return df
    except Exception as exc:
        errors.append(f"tencent={exc!r}")
    preferred = "lof" if is_lof_like else "etf"
    try:
        df = fetch_akshare_fund(symbol, start, end, adjust, preferred)
        if _has_rows(df):
            return df
    except Exception as exc:
        errors.append(f"akshare_{preferred}={exc!r}")
    try:
        alt = "etf" if preferred == "lof" else "lof"
        df = fetch_akshare_fund(symbol, start, end, adjust, alt)
        if _has_rows(df):
            return df
    except Exception as exc:
        errors.append(f"akshare_alt={exc!r}")
    raise RuntimeError("all_sources_failed_or_empty: " + " | ".join(errors))


def source_order_for(trade_class: str) -> list[str]:
    """Prefer sources that usually publish same-day ETF bars earlier."""
    is_lof_like = "LOF" in str(trade_class).upper()
    if is_lof_like:
        return ["tencent", "eastmoney", "tx_stock", "akshare"]
    return ["tencent", "eastmoney", "tx_stock", "sina_etf", "akshare"]


def fetch_clean_best(
    symbol: str,
    name: str,
    trade_class: str,
    start: str,
    end: str,
    adjust: str,
    source: str,
    retries: int,
) -> tuple[pd.DataFrame, str, list[str]]:
    errors: list[str] = []
    if source != "auto":
        raw = fetch_with_fallback(symbol, start, end, adjust, trade_class, source)
        clean = normalize_local_ohlcv(raw, symbol, name)
        return clean, source, errors

    best_clean = pd.DataFrame()
    best_source = ""
    best_latest = pd.Timestamp.min
    best_rows = -1
    target_end = pd.to_datetime(end, format="%Y%m%d", errors="coerce")

    for candidate_source in source_order_for(trade_class):
        last_exc = None
        raw = None
        for attempt in range(retries + 1):
            try:
                raw = fetch_with_fallback(symbol, start, end, adjust, trade_class, candidate_source)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(1.0 + attempt)
        if raw is None:
            errors.append(f"{candidate_source}={last_exc!r}")
            continue
        clean = normalize_local_ohlcv(raw, symbol, name)
        if clean.empty:
            errors.append(f"{candidate_source}=empty_after_normalize")
            continue
        latest = pd.to_datetime(clean["date"], errors="coerce").max()
        rows = len(clean)
        if latest > best_latest or (latest == best_latest and rows > best_rows):
            best_clean = clean
            best_source = candidate_source
            best_latest = latest
            best_rows = rows
        if pd.notna(target_end) and latest >= target_end:
            break

    if best_clean.empty:
        raise RuntimeError("all_sources_failed_or_empty: " + " | ".join(errors))
    return best_clean, best_source, errors


def update_one(symbol: str, name: str, trade_class: str, start: str, end: str, adjust: str, source: str, retries: int) -> dict:
    symbol = str(symbol).zfill(6)
    try:
        clean, source_used, source_errors = fetch_clean_best(symbol, name, trade_class, start, end, adjust, source, retries)
        if clean.empty:
            return {"symbol": symbol, "name": name, "trade_class": trade_class, "status": "empty", "rows": 0}
        path = etf_path(symbol)
        if path.exists():
            old = read_etf(symbol, enforce_quality=False)
            old["date"] = old["date"].dt.strftime("%Y-%m-%d")
            clean = (
                pd.concat([old, clean], ignore_index=True)
                .sort_values("date")
                .drop_duplicates("date", keep="last")
                .reset_index(drop=True)
            )
        write_etf(symbol, clean)
        latest = str(clean["date"].iloc[-1])
        target_end = pd.to_datetime(end, format="%Y%m%d", errors="coerce")
        stale = bool(pd.notna(target_end) and pd.to_datetime(latest) < target_end)
        return {
            "symbol": symbol,
            "name": name,
            "trade_class": trade_class,
            "status": "ok",
            "source_used": source_used,
            "rows": int(len(clean)),
            "start": str(clean["date"].iloc[0]),
            "end": latest,
            "stale": stale,
            "source_errors": " | ".join(source_errors[-3:]),
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "name": name,
            "trade_class": trade_class,
            "status": "error",
            "rows": 0,
            "error": repr(exc),
        }


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Download S1 shadow ETF/LOF universe data.")
    parser.add_argument("--pool", default="pool.csv", help="Pool file under raw.")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols to download.")
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default=datetime.today().strftime("%Y%m%d"))
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--source", default="auto", choices=["auto", "eastmoney", "tencent", "sina_etf", "tx_stock", "akshare"])
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true", help="Skip symbols whose CSV already exists.")
    args = parser.parse_args()

    ensure_dirs()
    # The updater may repair a previously blocked symbol. Live strategy code
    # applies the quality allowlist after the fresh download is audited.
    pool = load_pool(args.pool, enabled_only=True, enforce_quality=False)
    if args.symbols:
        wanted = {str(x).zfill(6) for x in args.symbols}
        pool = pool[pool["symbol"].isin(wanted)].copy()
    rows = []
    total = len(pool)
    for i, (_, row) in enumerate(pool.iterrows(), start=1):
        symbol = row["symbol"]
        if args.skip_existing and etf_path(symbol).exists():
            rows.append(
                {
                    "symbol": symbol,
                    "name": str(row.get("display_name", row.get("name", symbol))),
                    "trade_class": str(row.get("trade_class", "ETF")),
                    "status": "skip_existing",
                    "rows": 0,
                }
            )
            print(f"[{i}/{total}] skip {symbol}", flush=True)
            continue
        print(f"[{i}/{total}] downloading {symbol} {row.get('display_name', '')}", flush=True)
        rows.append(
            update_one(
                symbol=symbol,
                name=str(row.get("display_name", row.get("name", row["symbol"]))),
                trade_class=str(row.get("trade_class", "ETF")),
                start=args.start,
                end=args.end,
                adjust=args.adjust,
                source=args.source,
                retries=args.retries,
            )
        )
        if args.delay:
            time.sleep(args.delay)

    log = pd.DataFrame(rows)
    log_path = RAW / "download_log.csv"
    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    summary = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pool_file": args.pool,
        "symbols": int(len(log)),
        "ok": int((log["status"] == "ok").sum()) if len(log) else 0,
        "empty": int((log["status"] == "empty").sum()) if len(log) else 0,
        "error": int((log["status"] == "error").sum()) if len(log) else 0,
        "skip_existing": int((log["status"] == "skip_existing").sum()) if len(log) else 0,
        "min_latest": str(log.loc[log["status"] == "ok", "end"].min()) if "end" in log.columns and (log["status"] == "ok").any() else "",
        "max_latest": str(log.loc[log["status"] == "ok", "end"].max()) if "end" in log.columns and (log["status"] == "ok").any() else "",
        "stale_count": int(log.get("stale", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(log) else 0,
        "log": str(log_path.relative_to(RAW.parent)),
    }
    (RAW / "download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
