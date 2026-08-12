from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from s1lib import ETF_DIR, ROOT


PRICE_COLS = ["open", "high", "low", "close"]
FACTOR_GRID = np.array([0.2, 0.25, 1 / 3, 0.4, 0.5, 2 / 3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
RET_LIMIT = 0.25
INTRADAY_LIMIT = 0.15
NEXTDAY_LIMIT = 0.15
FACTOR_TOL = 0.12


def is_split_like(df: pd.DataFrame, idx: int) -> tuple[bool, float | None]:
    if idx <= 0 or idx >= len(df):
        return False, None
    prev_close = float(df.loc[idx - 1, "close"])
    curr_close = float(df.loc[idx, "close"])
    if prev_close <= 0 or curr_close <= 0:
        return False, None
    ratio = curr_close / prev_close
    if abs(ratio - 1.0) <= RET_LIMIT:
        return False, None
    nearest = FACTOR_GRID[np.argmin(np.abs(FACTOR_GRID - ratio))]
    rel_err = abs(ratio - nearest) / nearest
    if rel_err > FACTOR_TOL:
        return False, None
    intraday = abs(float(df.loc[idx, "high"]) - float(df.loc[idx, "low"])) / curr_close if curr_close else 0.0
    if intraday > INTRADAY_LIMIT:
        return False, None
    if idx + 1 < len(df):
        next_ret = abs(float(df.loc[idx + 1, "close"]) / curr_close - 1.0)
        if next_ret > NEXTDAY_LIMIT:
            return False, None
    return True, ratio


def repair_one(path: Path) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in PRICE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    actions: list[dict] = []
    changed = False

    idx = 1
    while idx < len(df):
        prev_close = float(df.loc[idx - 1, "close"]) if pd.notna(df.loc[idx - 1, "close"]) else np.nan
        curr_close = float(df.loc[idx, "close"]) if pd.notna(df.loc[idx, "close"]) else np.nan
        if prev_close > 0 and curr_close > 0:
            ret = curr_close / prev_close - 1.0
            if abs(ret) > RET_LIMIT:
                ok, ratio = is_split_like(df, idx)
                if ok and ratio is not None:
                    df.loc[: idx - 1, PRICE_COLS] = df.loc[: idx - 1, PRICE_COLS] * ratio
                    actions.append(
                        {
                            "symbol": path.stem,
                            "date": str(df.loc[idx, "date"]),
                            "ratio": ratio,
                            "ret_before": ret,
                        }
                    )
                    changed = True
        idx += 1

    if changed:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return actions


def main() -> None:
    reports: list[dict] = []
    for path in sorted(ETF_DIR.glob("*.csv")):
        reports.extend(repair_one(path))
    out_dir = ROOT / "pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "split_repairs.csv"
    pd.DataFrame(reports).to_csv(report_path, index=False, encoding="utf-8-sig")
    summary = {
        "repair_count": int(len(reports)),
        "symbols": sorted({r["symbol"] for r in reports}),
    }
    (out_dir / "split_repairs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
