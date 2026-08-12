from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRADES_PATH = ROOT / "D" / "out" / "line_specific_gates" / "rolling_line_gates_trades.csv"
D_HOLD_PATH = ROOT / "D" / "out" / "d_failure_diagnosis" / "d_fixed_hold_returns.csv"
OUT_DIR = ROOT / "D" / "out" / "csd_regime_judge_v2_strict"


def pf(s: pd.Series) -> float:
    win = s[s > 0].sum()
    loss = -s[s < 0].sum()
    if loss == 0:
        return 99.0 if win > 0 else 0.0
    return float(win / loss)


def summarize(df: pd.DataFrame, ret_col: str = "ret") -> dict[str, float]:
    x = df[df[ret_col].notna()].copy()
    r = x[ret_col]
    annual = x.groupby("year")[ret_col].sum() if len(x) else pd.Series(dtype=float)
    return {
        "n": int(len(x)),
        "win": float((r > 0).mean()) if len(x) else np.nan,
        "avg": float(r.mean()) if len(x) else np.nan,
        "sum": float(r.sum()) if len(x) else 0.0,
        "pf": pf(r) if len(x) else np.nan,
        "annual_avg_sum": float(annual.mean()) if len(annual) else np.nan,
        "worst_year_sum": float(annual.min()) if len(annual) else np.nan,
    }


def robust_score(g: pd.DataFrame) -> float:
    """Training-only score: pursue return, but penalize a bad historical year."""
    r = g["ret"].dropna()
    if len(r) < 80:
        return -1e9
    annual = g.groupby("year")["ret"].sum()
    return float(
        r.sum()
        + (r > 0).mean() * 10.0
        + min(pf(r), 10.0)
        + min(float(annual.min()), 0.0) * 2.2
        + float(annual.mean()) * 0.8
    )


def prepare_active() -> pd.DataFrame:
    df = pd.read_csv(TRADES_PATH, parse_dates=["signal_date", "entry_date", "exit_date"])
    df = df[(df["active"] == True) & (df["completed"] == True)].copy()
    df["year"] = df["signal_date"].dt.year

    d_hold = pd.read_csv(D_HOLD_PATH, parse_dates=["signal_date", "exit_h20", "exit_h30", "exit_h40", "exit_h60"])
    d_hold = d_hold[(d_hold["line"] == "D") & (d_hold["completed"] == True)].copy()
    keep = ["signal_date", "line", "symbol", "ret_h20", "ret_h30", "ret_h40", "ret_h60", "exit_h20", "exit_h30", "exit_h40", "exit_h60"]
    df = df.merge(d_hold[keep], on=["signal_date", "line", "symbol"], how="left")

    # Same day + same line: keep highest ranked candidate.
    df = df.sort_values(["signal_date", "line", "rank_score"], ascending=[True, True, False])
    df = df.drop_duplicates(["signal_date", "line"], keep="first")
    return df


def d_guard_ret(row: pd.Series, cap_h: int, trend_t: float, rebound_t: float) -> float:
    if row["line"] != "D":
        return float(row["ret"])
    ret_col = f"ret_h{cap_h}"
    exit_col = f"exit_h{cap_h}"
    if pd.isna(row.get(ret_col)) or pd.isna(row.get(exit_col)):
        return float(row["ret"])
    guard = (
        float(row["trend_score"]) > trend_t
        and float(row["rebound_score"]) < rebound_t
        and pd.Timestamp(row["exit_date"]) > pd.Timestamp(row[exit_col])
    )
    return float(row[ret_col]) if guard else float(row["ret"])


def pseudo_danger(row: pd.Series, trend_t: float, rebound_t: float, m20_t: float) -> bool:
    return (
        row["line"] == "D"
        and float(row["trend_score"]) > trend_t
        and float(row["rebound_score"]) < rebound_t
        and float(row["median_ret20"]) > m20_t
    )


def choose(day: pd.DataFrame, order: tuple[str, ...]) -> pd.Series:
    for line in order:
        hit = day[day["line"] == line]
        if len(hit):
            return hit.iloc[0]
    return day.iloc[0]


def add_variant(
    rows: list[dict],
    policy: str,
    row: pd.Series | None,
    guard: tuple[int, float, float] | None = None,
    skip_danger: tuple[float, float, float] | None = None,
) -> None:
    if row is None:
        return
    if skip_danger and pseudo_danger(row, *skip_danger):
        return
    data = row.to_dict()
    if guard:
        data["ret"] = d_guard_ret(row, *guard)
    data["policy"] = policy
    rows.append(data)


def build_panel(active: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    guard_variants: list[tuple[str, tuple[int, float, float] | None]] = [
        ("raw", None),
        ("g40_t055_r030", (40, 0.55, 0.30)),
        ("g40_t060_r040", (40, 0.60, 0.40)),
        ("g30_t055_r030", (30, 0.55, 0.30)),
        ("g60_t055_r030", (60, 0.55, 0.30)),
    ]
    skip_variants: list[tuple[str, tuple[float, float, float] | None]] = [
        ("noskip", None),
        ("skip_t055_r030_m000", (0.55, 0.30, 0.00)),
        ("skip_t060_r040_m000", (0.60, 0.40, 0.00)),
        ("skip_t060_r040_m-002", (0.60, 0.40, -0.02)),
    ]

    for _, day in active.groupby("signal_date", sort=True):
        lines = set(day["line"])
        st = day.iloc[0]

        for order in permutations(("C", "S", "D")):
            base = choose(day, order)
            for g_name, guard in guard_variants:
                for s_name, skip in skip_variants:
                    add_variant(rows, f"priority_{''.join(order)}_{g_name}_{s_name}", base, guard, skip)

        # Rule-based judge: C gets first right in rebound-panic, S gets trend-low-panic, D default.
        for c_trend in [0.45, 0.55]:
            for c_rebound in [0.55, 0.65]:
                for s_trend in [0.45, 0.60]:
                    for s_rebound in [0.30, 0.45]:
                        c_ok = ("C" in lines) and float(st["trend_score"]) <= c_trend and float(st["rebound_score"]) >= c_rebound
                        s_ok = ("S" in lines) and float(st["trend_score"]) >= s_trend and float(st["rebound_score"]) <= s_rebound
                        if c_ok:
                            base = day[day["line"] == "C"].iloc[0]
                        elif s_ok:
                            base = day[day["line"] == "S"].iloc[0]
                        else:
                            base = choose(day, ("D", "C", "S"))
                        for g_name, guard in guard_variants:
                            for s_name, skip in skip_variants:
                                add_variant(
                                    rows,
                                    f"state_Ct{c_trend:.2f}_Cr{c_rebound:.2f}_St{s_trend:.2f}_Sr{s_rebound:.2f}_{g_name}_{s_name}",
                                    base,
                                    guard,
                                    skip,
                                )

        # Quality judge: use only rolling historical quality already available on signal day.
        for lb in [252, 504, 99999]:
            for scope in ["line", "sym"]:
                avg_col = f"{scope}_{lb}_avg"
                win_col = f"{scope}_{lb}_win"
                pf_col = f"{scope}_{lb}_pf"
                if avg_col not in day.columns:
                    continue
                tmp = day.copy()
                tmp["_q"] = tmp[avg_col].fillna(-9) * 100 + tmp[win_col].fillna(0) * 6 + np.log1p(tmp[pf_col].clip(lower=0).fillna(0))
                base = tmp.sort_values("_q", ascending=False).iloc[0]
                for g_name, guard in guard_variants:
                    for s_name, skip in skip_variants:
                        add_variant(rows, f"quality_{scope}_lb{lb}_{g_name}_{s_name}", base, guard, skip)

    panel = pd.DataFrame(rows)
    if "date" in panel.columns:
        panel = panel.drop(columns=["date"])
    panel["year"] = pd.to_datetime(panel["signal_date"]).dt.year
    return panel


def select_rolling(panel: pd.DataFrame, min_train_years: int, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(panel["year"].unique())
    selected = []
    choices = []
    for year in years:
        train_years = [y for y in years if y < year]
        test = panel[panel["year"] == year]
        if len(train_years) < min_train_years:
            chosen = "priority_DCS_raw_noskip"
            rank = pd.DataFrame()
        else:
            train = panel[panel["year"] < year]
            rows = []
            for p, g in train.groupby("policy"):
                # Practical trigger-rate guard: avoid selecting rules that vanish.
                if len(g) < max(80, len(train_years) * 35):
                    continue
                s = summarize(g)
                s["policy"] = p
                s["score"] = robust_score(g)
                rows.append(s)
            rank = pd.DataFrame(rows).sort_values(["score", "sum"], ascending=False)
            base = rank[rank["policy"] == "priority_DCS_raw_noskip"].iloc[0]
            if mode == "argmax":
                chosen = str(rank.iloc[0]["policy"])
            elif mode == "beat_base":
                eligible = rank[
                    (rank["sum"] > float(base["sum"]) + 0.5)
                    & (rank["pf"] >= float(base["pf"]) * 0.98)
                    & (rank["worst_year_sum"] >= float(base["worst_year_sum"]) - 0.6)
                ]
                chosen = str(eligible.iloc[0]["policy"]) if len(eligible) else "priority_DCS_raw_noskip"
            elif mode == "stable":
                eligible = rank[
                    (rank["sum"] > float(base["sum"]) + 1.0)
                    & (rank["win"] >= float(base["win"]))
                    & (rank["worst_year_sum"] >= float(base["worst_year_sum"]))
                ]
                chosen = str(eligible.iloc[0]["policy"]) if len(eligible) else "priority_DCS_raw_noskip"
            else:
                raise ValueError(mode)
        picked = test[test["policy"] == chosen].copy()
        picked["selector_mode"] = mode
        picked["min_train_years"] = min_train_years
        selected.append(picked)
        choices.append(
            {
                "year": int(year),
                "mode": mode,
                "min_train_years": min_train_years,
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
    active = prepare_active()
    active.to_csv(OUT_DIR / "active_line_candidates.csv", index=False, encoding="utf-8-sig")

    panel = build_panel(active)
    panel.to_csv(OUT_DIR / "policy_candidate_panel.csv", index=False, encoding="utf-8-sig")

    full_rows = []
    for p, g in panel.groupby("policy"):
        s = summarize(g)
        s["policy"] = p
        full_rows.append(s)
    full_rank = pd.DataFrame(full_rows).sort_values(["sum", "win"], ascending=False)
    full_rank.to_csv(OUT_DIR / "full_sample_policy_rank.csv", index=False, encoding="utf-8-sig")

    selected_all = []
    choices_all = []
    for min_train in [2, 3, 5]:
        for mode in ["argmax", "beat_base", "stable"]:
            selected, choices = select_rolling(panel, min_train, mode)
            selected_all.append(selected)
            choices_all.append(choices)
    selected_df = pd.concat(selected_all, ignore_index=True)
    choices_df = pd.concat(choices_all, ignore_index=True)
    selected_df.to_csv(OUT_DIR / "strict_rolling_selected_trades.csv", index=False, encoding="utf-8-sig")
    choices_df.to_csv(OUT_DIR / "strict_rolling_choices.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    fixed_models = [
        "priority_DCS_raw_noskip",
        "priority_CSD_raw_noskip",
        "priority_CSD_g40_t055_r030_noskip",
        "priority_CSD_g40_t055_r030_skip_t055_r030_m000",
        "quality_line_lb252_g40_t055_r030_noskip",
        "quality_sym_lb252_g40_t055_r030_noskip",
    ]
    for p in fixed_models:
        if p in set(panel["policy"]):
            summary_rows.append({"model": f"fixed_{p}", **summarize(panel[panel["policy"] == p])})
    for (mode, mt), g in selected_df.groupby(["selector_mode", "min_train_years"]):
        summary_rows.append({"model": f"rolling_{mode}_mintrain{mt}", **summarize(g)})
    summary = pd.DataFrame(summary_rows).sort_values(["sum", "win"], ascending=False)
    summary.to_csv(OUT_DIR / "strict_rolling_summary.csv", index=False, encoding="utf-8-sig")

    core = []
    for name, g in {
        "base_DCS": panel[panel["policy"] == "priority_DCS_raw_noskip"],
        "fixed_CSD_guard": panel[panel["policy"] == "priority_CSD_g40_t055_r030_noskip"],
    }.items():
        y = g.groupby("year").agg(n=("ret", "size"), win=("ret", lambda s: (s > 0).mean()), avg=("ret", "mean"), sum=("ret", "sum")).reset_index()
        y.insert(0, "model", name)
        core.append(y)
    best = summary.iloc[0]["model"]
    if str(best).startswith("rolling_"):
        parts = str(best).replace("rolling_", "").split("_mintrain")
        mode = parts[0]
        mt = int(parts[1])
        g = selected_df[(selected_df["selector_mode"] == mode) & (selected_df["min_train_years"] == mt)]
        y = g.groupby("year").agg(n=("ret", "size"), win=("ret", lambda s: (s > 0).mean()), avg=("ret", "mean"), sum=("ret", "sum")).reset_index()
        y.insert(0, "model", str(best))
        core.append(y)
    annual = pd.concat(core, ignore_index=True)
    annual.to_csv(OUT_DIR / "annual_compare_core.csv", index=False, encoding="utf-8-sig")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nTOP FULL")
    print(full_rank.head(25).to_string(index=False))
    print("\nCHOICES")
    print(choices_df.to_string(index=False))


if __name__ == "__main__":
    main()
