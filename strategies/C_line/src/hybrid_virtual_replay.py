from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
S_SRC = WORKSPACE / "S" / "src"
if str(S_SRC) not in sys.path:
    sys.path.insert(0, str(S_SRC))

from shadow_model import Param, ShadowResearchEngine  # noqa: E402


OUT = ROOT / "fit" / "hybrid_virtual_replay"
SELECTION = WORKSPACE / "S" / "fit" / "selector" / "selected_by_year.csv"
FORMAL = WORKSPACE / "S" / "fit" / "selector" / "final_trades.csv"


def load_selection() -> pd.DataFrame:
    frame = pd.read_csv(SELECTION, encoding="utf-8-sig")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    return frame


def to_param(row: pd.Series) -> Param:
    return Param(
        scope=str(row["scope"]),
        idle=int(float(row["idle"])),
        hold=int(float(row["hold"])),
        ma=int(float(row["ma"])),
        mom=int(float(row["mom"])),
        model=str(row["model"]),
        alpha=float(row["alpha"]),
        vol_cap=float(row["vol_cap"]),
        accel_cap=float(row["accel_cap"]),
        min_history=250,
        min_amount20=2_000_000,
    )


def replay() -> pd.DataFrame:
    engine = ShadowResearchEngine()
    selection = load_selection()
    parts = [engine.main.copy()]
    for _, row in selection.iterrows():
        if str(row["param_id"]).upper() == "CASH":
            continue
        year = int(row["year"])
        param = to_param(row)
        trades = engine.generate_shadow_trades(param, causal_idle=True)
        if trades.empty:
            continue
        entry_year = pd.to_datetime(
            trades["entry_date"], errors="coerce"
        ).dt.year
        parts.append(trades[entry_year.eq(year)].copy())
    return engine.account_from_trades(
        pd.concat(parts, ignore_index=True, sort=False)
    )


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    local["symbol"] = local["symbol"].astype(str).str.zfill(6)
    local["entry_date"] = pd.to_datetime(
        local["entry_date"], errors="coerce"
    )
    local["exit_date"] = pd.to_datetime(
        local["exit_date"], errors="coerce"
    )
    local["ret"] = pd.to_numeric(local["ret"], errors="coerce")
    return local.sort_values(
        ["entry_date", "exit_date", "symbol"]
    ).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generated = normalize(replay())
    formal = normalize(
        pd.read_csv(FORMAL, dtype={"symbol": str}, encoding="utf-8-sig")
    )
    key = ["symbol", "entry_date", "exit_date"]
    compare = generated.merge(
        formal[key + ["ret"]].rename(columns={"ret": "formal_ret"}),
        on=key,
        how="outer",
        indicator=True,
    )
    compare["return_error"] = (
        pd.to_numeric(compare["ret"], errors="coerce")
        - pd.to_numeric(compare["formal_ret"], errors="coerce")
    ).abs()
    exact = compare["_merge"].eq("both") & (
        compare["return_error"].fillna(0).le(1e-12)
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "raw ETF prices + frozen annual S parameters",
        "generated_trades": int(len(generated)),
        "formal_trades": int(len(formal)),
        "matched_keys": int(compare["_merge"].eq("both").sum()),
        "exact_rows": int(exact.sum()),
        "missing_generated": int(compare["_merge"].eq("right_only").sum()),
        "extra_generated": int(compare["_merge"].eq("left_only").sum()),
        "max_return_error": float(
            np.nan_to_num(compare["return_error"].max(), nan=0.0)
        ),
    }
    summary["passed"] = bool(
        summary["generated_trades"] == summary["formal_trades"]
        and summary["exact_rows"] == summary["formal_trades"]
        and summary["missing_generated"] == 0
        and summary["extra_generated"] == 0
    )
    generated.to_csv(
        OUT / "replayed_s_trades.csv", index=False, encoding="utf-8-sig"
    )
    compare.to_csv(
        OUT / "trade_compare.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
