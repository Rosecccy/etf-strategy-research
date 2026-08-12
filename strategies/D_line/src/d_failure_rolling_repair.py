from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "D" / "out" / "d_failure_diagnosis" / "d_fixed_hold_returns.csv"
OUT_DIR = ROOT / "D" / "out" / "d_failure_diagnosis"


def pf(ret: pd.Series) -> float:
    win = ret[ret > 0].sum()
    loss = -ret[ret < 0].sum()
    if loss == 0:
        return 99.0 if win > 0 else 0.0
    return float(win / loss)


def danger(df: pd.DataFrame, t: float, r: float, m: float) -> pd.Series:
    return (df["trend_score"] > t) & (df["rebound_score"] < r) & (df["median_ret20"] > m)


def weak(df: pd.DataFrame, t: float, m: float) -> pd.Series:
    return (df["trend_score"] <= t) & (df["median_ret20"] < m)


def score(ret: pd.Series) -> float:
    ret = ret.dropna()
    if ret.empty:
        return -1e9
    y = ret.groupby(ret.index).sum() if False else ret
    # 稳定分：收益为主，但惩罚最差年份，避免只靠某一年撑起。
    return float(ret.sum() + (ret > 0).mean() * 8.0 + min(pf(ret), 8.0))


def summarize(df: pd.DataFrame, col: str) -> dict[str, float]:
    x = df[df[col].notna()].copy()
    r = x[col]
    annual = x.groupby("year")[col].sum()
    return {
        "n": int(len(x)),
        "win": float((r > 0).mean()) if len(x) else np.nan,
        "avg": float(r.mean()) if len(x) else np.nan,
        "sum": float(r.sum()) if len(x) else 0.0,
        "pf": pf(r) if len(x) else np.nan,
        "annual_avg_sum": float(annual.mean()) if len(annual) else np.nan,
        "worst_year_sum": float(annual.min()) if len(annual) else np.nan,
    }


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out: list[pd.DataFrame] = []

    def add(name: str, ret: pd.Series, active: pd.Series | None = None) -> None:
        active = pd.Series(True, index=df.index) if active is None else active
        tmp = df.loc[active].copy()
        tmp["policy"] = name
        tmp["policy_ret"] = ret.loc[active]
        out.append(tmp[["signal_date", "year", "symbol", "name", "policy", "policy_ret"]])

    add("base_original_exit", df["ret"])
    for h in [30, 40, 60, 90]:
        add(f"fixed_h{h}", df[f"ret_h{h}"])

    for short_h in [30, 40]:
        for long_h in [60, 90]:
            if short_h >= long_h:
                continue
            for t in [0.55, 0.60]:
                for r in [0.35, 0.40]:
                    for m in [-0.04, -0.02]:
                        dm = danger(df, t, r, m)
                        add(
                            f"dyn_pseudo_short_t{t:.2f}_r{r:.2f}_m{m:.2f}_h{short_h}_{long_h}",
                            df[f"ret_h{long_h}"].where(~dm, df[f"ret_h{short_h}"]),
                        )

    for dt in [0.50, 0.55]:
        for dr in [0.35, 0.40]:
            for dm in [-0.04, -0.02]:
                active = ~danger(df, dt, dr, dm)
                for wt in [0.55, 0.65, 0.75]:
                    for wm in [-0.04, 0.00]:
                        for short_h, long_h in [(30, 90), (30, 60)]:
                            wk = weak(df, wt, wm)
                            add(
                                (
                                    f"skip_pseudo_weak_short_dt{dt:.2f}_dr{dr:.2f}_dm{dm:.2f}"
                                    f"_wt{wt:.2f}_wm{wm:.2f}_h{short_h}_{long_h}"
                                ),
                                df[f"ret_h{long_h}"].where(~wk, df[f"ret_h{short_h}"]),
                                active,
                            )
    return pd.concat(out, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(IN_PATH, parse_dates=["signal_date"])
    raw = raw[(raw["line"] == "D") & (raw["completed"] == True)].copy()
    raw["year"] = raw["signal_date"].dt.year

    panel = build_candidates(raw)
    panel = panel[panel["policy_ret"].notna()].copy()
    panel.to_csv(OUT_DIR / "rolling_repair_candidate_panel.csv", index=False, encoding="utf-8-sig")

    choices = []
    selected = []
    for year in sorted(panel["year"].unique()):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if train.empty:
            chosen = "base_original_exit"
        else:
            rows = []
            for p, g in train.groupby("policy"):
                yr = g.groupby("year")["policy_ret"].sum()
                s = g["policy_ret"].sum() + (g["policy_ret"] > 0).mean() * 8 + min(pf(g["policy_ret"]), 8)
                s += min(float(yr.min()), 0.0) * 2.0
                rows.append({"policy": p, "train_score": s, "train_sum": g["policy_ret"].sum(), "train_worst_year": yr.min()})
            chosen = pd.DataFrame(rows).sort_values(["train_score", "train_sum"], ascending=False).iloc[0]["policy"]
        picked = test[test["policy"] == chosen].copy()
        choices.append({"year": int(year), "chosen_policy": chosen, "n": int(len(picked)), "sum": float(picked["policy_ret"].sum()), "win": float((picked["policy_ret"] > 0).mean()) if len(picked) else np.nan})
        selected.append(picked)

    selected_df = pd.concat(selected, ignore_index=True)
    choices_df = pd.DataFrame(choices)
    selected_df.to_csv(OUT_DIR / "rolling_repair_selected_trades.csv", index=False, encoding="utf-8-sig")
    choices_df.to_csv(OUT_DIR / "rolling_repair_choices.csv", index=False, encoding="utf-8-sig")

    candidate_rank = []
    for p, g in panel.groupby("policy"):
        row = {"policy": p}
        row.update(summarize(g, "policy_ret"))
        candidate_rank.append(row)
    rank = pd.DataFrame(candidate_rank).sort_values(["sum", "win"], ascending=False)
    rank.to_csv(OUT_DIR / "rolling_repair_candidate_full_rank.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"model": "rolling_repair", **summarize(selected_df, "policy_ret")},
            {"model": "base_original_exit", **summarize(panel[panel["policy"] == "base_original_exit"], "policy_ret")},
        ]
    )
    summary.to_csv(OUT_DIR / "rolling_repair_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(choices_df.to_string(index=False))


if __name__ == "__main__":
    main()
