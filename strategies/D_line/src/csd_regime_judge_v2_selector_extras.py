from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "D" / "out" / "csd_regime_judge_v2_strict" / "policy_candidate_panel.csv"
OUT_DIR = ROOT / "D" / "out" / "csd_regime_judge_v2_strict"


def pf(s: pd.Series) -> float:
    w = s[s > 0].sum()
    l = -s[s < 0].sum()
    if l == 0:
        return 99.0 if w > 0 else 0.0
    return float(w / l)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    r = df["ret"].dropna()
    annual = df.groupby("year")["ret"].sum()
    return {
        "n": int(len(r)),
        "win": float((r > 0).mean()) if len(r) else np.nan,
        "avg": float(r.mean()) if len(r) else np.nan,
        "sum": float(r.sum()) if len(r) else 0.0,
        "pf": pf(r) if len(r) else np.nan,
        "annual_avg_sum": float(annual.mean()) if len(annual) else np.nan,
        "worst_year_sum": float(annual.min()) if len(annual) else np.nan,
    }


def score_policy(g: pd.DataFrame, mode: str, train_years: list[int]) -> float:
    r = g["ret"]
    annual = g.groupby("year")["ret"].sum()
    if len(g) < max(80, len(train_years) * 35):
        return -1e9
    if mode == "simple_sum":
        return float(r.sum() + (r > 0).mean() * 8 + min(pf(r), 8))
    if mode == "recent2":
        last = train_years[-2:] if len(train_years) >= 2 else train_years
        recent = g[g["year"].isin(last)]["ret"]
        return float(r.sum() * 0.45 + recent.sum() * 0.75 + (r > 0).mean() * 8 + min(pf(r), 8) + min(annual.min(), 0) * 1.0)
    if mode == "minimax":
        return float(annual.mean() * 2.0 + min(annual.min(), 0) * 3.0 + r.sum() * 0.55 + (r > 0).mean() * 8 + min(pf(r), 8))
    if mode == "priority_bias":
        bonus = 2.5 if str(g["policy"].iloc[0]).startswith(("priority_CSD", "priority_SCD")) else 0.0
        return float(r.sum() + annual.mean() + (r > 0).mean() * 8 + min(pf(r), 8) + min(annual.min(), 0) * 1.5 + bonus)
    raise ValueError(mode)


def select(panel: pd.DataFrame, min_train: int, mode: str, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if subset == "priority_only":
        panel = panel[panel["policy"].str.startswith("priority_")].copy()
    elif subset == "no_quality":
        panel = panel[~panel["policy"].str.startswith("quality_")].copy()
    elif subset == "all":
        panel = panel.copy()
    else:
        raise ValueError(subset)

    years = sorted(panel["year"].unique())
    selected = []
    choices = []
    for year in years:
        train_years = [y for y in years if y < year]
        test = panel[panel["year"] == year]
        if len(train_years) < min_train:
            chosen = "priority_DCS_raw_noskip"
            rank = pd.DataFrame()
        else:
            train = panel[panel["year"] < year]
            rows = []
            for p, g in train.groupby("policy"):
                s = summarize(g)
                s["policy"] = p
                s["score"] = score_policy(g, mode, train_years)
                rows.append(s)
            rank = pd.DataFrame(rows).sort_values(["score", "sum"], ascending=False)
            chosen = str(rank.iloc[0]["policy"])
        picked = test[test["policy"] == chosen].copy()
        picked["selector"] = f"{subset}_{mode}_min{min_train}"
        selected.append(picked)
        choices.append(
            {
                "selector": f"{subset}_{mode}_min{min_train}",
                "year": int(year),
                "chosen_policy": chosen,
                "test_n": int(len(picked)),
                "test_win": float((picked["ret"] > 0).mean()) if len(picked) else np.nan,
                "test_sum": float(picked["ret"].sum()) if len(picked) else 0.0,
                "top_train_policy": "" if rank.empty else str(rank.iloc[0]["policy"]),
            }
        )
    return pd.concat(selected, ignore_index=True), pd.DataFrame(choices)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(IN_PATH, parse_dates=["signal_date"])
    all_sel = []
    all_choices = []
    for subset in ["all", "no_quality", "priority_only"]:
        for mode in ["simple_sum", "recent2", "minimax", "priority_bias"]:
            for min_train in [2, 3, 5]:
                s, c = select(panel, min_train, mode, subset)
                all_sel.append(s)
                all_choices.append(c)
    sel = pd.concat(all_sel, ignore_index=True)
    choices = pd.concat(all_choices, ignore_index=True)
    sel.to_csv(OUT_DIR / "extra_selector_selected_trades.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT_DIR / "extra_selector_choices.csv", index=False, encoding="utf-8-sig")

    rows = []
    for name, g in sel.groupby("selector"):
        rows.append({"selector": name, **summarize(g)})
    out = pd.DataFrame(rows).sort_values(["sum", "win"], ascending=False)
    out.to_csv(OUT_DIR / "extra_selector_summary.csv", index=False, encoding="utf-8-sig")
    print(out.head(30).to_string(index=False))
    best = out.iloc[0]["selector"]
    print("\nBEST CHOICES")
    print(choices[choices["selector"] == best].to_string(index=False))


if __name__ == "__main__":
    main()
