from __future__ import annotations

from pathlib import Path

import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
ETF_DIR = RAW / "etf"
QUALITY = ROOT / "pool"
C_ROOT = ROOT.parent / "C"
C_ETF_DIR = C_ROOT / "raw" / "etf"


def configure_stdout_utf8() -> None:
    try:
        import sys

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ensure_dirs() -> None:
    for path in [RAW, ETF_DIR, QUALITY]:
        path.mkdir(parents=True, exist_ok=True)


def pool_path(name: str = "pool.csv") -> Path:
    return RAW / name


def load_pool(
    name: str = "pool.csv",
    enabled_only: bool = True,
    enforce_quality: bool = True,
) -> pd.DataFrame:
    pool = pd.read_csv(pool_path(name), dtype={"symbol": str}, encoding="utf-8-sig")
    pool["symbol"] = pool["symbol"].astype(str).str.zfill(6)
    if enabled_only and "enabled" in pool.columns:
        pool = pool[pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])]
    quality_path = RAW / "quality.json"
    if enabled_only and enforce_quality and quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if not bool(quality.get("passed")):
            raise RuntimeError("S-line ETF data quality gate failed; strategy execution is blocked.")
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
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover_rate",
        "?交?": "date",
        "撘??": "open",
        "?擃?": "high",
        "?雿?": "low",
        "?嗥?": "close",
        "?漱??": "volume",
        "?漱憸?": "amount",
        "瘨刻?撟?": "pct_change",
        "?Ｘ???": "turnover_rate",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover_rate", "amplitude", "change"]:
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
        "amplitude",
        "change",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[keep].sort_values("date").drop_duplicates("date", keep="last")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def read_etf(symbol: str, enforce_quality: bool = True) -> pd.DataFrame:
    symbol = str(symbol).zfill(6)
    path = etf_path(symbol)
    s_quality_path = RAW / "quality.json"
    if enforce_quality and s_quality_path.exists():
        s_quality = json.loads(s_quality_path.read_text(encoding="utf-8"))
        if not bool(s_quality.get("passed")):
            raise RuntimeError("S-line ETF data quality gate failed; data read is blocked.")
        approved = {
            str(item).zfill(6)
            for item in s_quality.get("approved_symbols", [])
        }
        if symbol not in approved:
            raise RuntimeError(f"ETF {symbol} is excluded by the data quality gate.")
    c_quality_path = C_ROOT / "raw" / "quality.json"
    if c_quality_path.exists():
        quality = json.loads(c_quality_path.read_text(encoding="utf-8"))
        approved = {
            str(item).zfill(6)
            for item in quality.get("approved_symbols", [])
        }
        canonical = C_ETF_DIR / f"{symbol}.csv"
        if bool(quality.get("passed")) and symbol in approved and canonical.exists():
            path = canonical
    df = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def write_etf(symbol: str, df: pd.DataFrame) -> None:
    path = etf_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8-sig")
