from __future__ import annotations

from pathlib import Path

import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
ETF_DIR = RAW / "etf"
QUALITY = ROOT / "pool"
LIVE = ROOT / "live"
ACCOUNT_DIR = ROOT / "account"


def ensure_dirs() -> None:
    for path in [RAW, ETF_DIR, QUALITY, LIVE, ACCOUNT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def pool_path() -> Path:
    return RAW / "pool.csv"


def load_pool(enabled_only: bool = True, enforce_quality: bool = True) -> pd.DataFrame:
    pool = pd.read_csv(pool_path(), dtype={"symbol": str})
    pool["symbol"] = pool["symbol"].astype(str).str.zfill(6)
    if enabled_only and "enabled" in pool.columns:
        pool = pool[pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])]
    quality_path = RAW / "quality.json"
    if enabled_only and enforce_quality and quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if not bool(quality.get("passed")):
            raise RuntimeError("ETF data quality gate did not pass; strategy execution is blocked.")
        approved = {
            str(symbol).zfill(6)
            for symbol in quality.get("approved_symbols", [])
        }
        pool = pool[pool["symbol"].isin(approved)].copy()
    return pool.copy()


def etf_path(symbol: str) -> Path:
    return ETF_DIR / f"{str(symbol).zfill(6)}.csv"


def normalize_local_ohlcv(df: pd.DataFrame, symbol: str, name: str = "") -> pd.DataFrame:
    rename = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_change",
        "换手率": "turnover_rate",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["symbol"] = str(symbol).zfill(6)
    out["name"] = name
    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0].copy()
    keep = [
        "date",
        "symbol",
        "name",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_change",
        "turnover_rate",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[keep].sort_values("date").drop_duplicates("date", keep="last")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def read_etf(symbol: str) -> pd.DataFrame:
    path = etf_path(symbol)
    df = pd.read_csv(path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def write_etf(symbol: str, df: pd.DataFrame) -> None:
    path = etf_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8-sig")
