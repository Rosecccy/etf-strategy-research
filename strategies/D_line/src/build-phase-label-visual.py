from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd

from factor_dca_scan import clustered_extrema_labels


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
TEMPLATE = Path(__file__).with_name("phase-label-template.html")
VIS_DIR = Path(r"C:\Users\10619\.codex\visualizations\2026\06\21\019ee960-4ad4-7931-9173-60478085fa8b")
SYMBOL = "510880"
NAME = "红利ETF"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--fragment", default="phase-labels.html")
    args = parser.parse_args()
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    lows, highs = clustered_extrema_labels(raw["close"], window=args.window, tolerance=0.005, max_gap=5)
    raw["low"] = lows
    raw["high"] = highs
    by_year: dict[str, list[dict]] = {}
    for year, frame in raw.groupby(raw["date"].dt.year, sort=True):
        by_year[str(year)] = [
            {
                "date": row.date.date().isoformat(),
                "close": round(float(row.close), 6),
                "low": bool(row.low),
                "high": bool(row.high),
            }
            for row in frame[["date", "close", "low", "high"]].itertuples(index=False)
        ]
    data = {
        "symbol": SYMBOL,
        "name": NAME,
        "window": args.window,
        "years": list(by_year),
        "defaultYear": "2012",
        "byYear": by_year,
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise RuntimeError("Expected one data marker in phase-label template.")
    fragment = template.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    if '\\"' in fragment or "\\n" in fragment:
        raise RuntimeError("Visualization fragment contains escaped markup.")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    fragment_name = Path(args.fragment).name
    (VIS_DIR / fragment_name).write_text(fragment, encoding="utf-8")
    (ROOT / "out" / f"{Path(fragment_name).stem}-fragment.html").write_text(fragment, encoding="utf-8")
    print(json.dumps({"symbol": SYMBOL, "window": args.window, "labels": int(raw["low"].sum() + raw["high"].sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
