from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "D" / "out" / "d_failure_diagnosis" / "d_fixed_hold_returns.csv"
OUT_DIR = ROOT / "D" / "out" / "d_failure_diagnosis" / "strict_rolling"


def profit_factor(s: pd.Series) -> float:
    win = s[s > 0].sum()
    loss = -s[s < 0].sum()
    if loss == 0:
        return 99.0 if win > 0 else 0.0
    return float(win / loss)


def summary(df: pd.DataFrame, ret_col: str = "ret") -> dict[str, float]:
    x = df[df[ret_col].notna()].copy()
    r = x[ret_col]
    annual = x.groupby("year")[ret_col].sum()
    return {
        "n": int(len(x)),
        "win": float((r > 0).mean()) if len(x) else np.nan,
        "avg": float(r.mean()) if len(x) else np.nan,
        "sum": float(r.sum()) if len(x) else 0.0,
        "pf": profit_factor(r) if len(x) else np.nan,
        "annual_avg_sum": float(annual.mean()) if len(annual) else np.nan,
        "worst_year_sum": float(annual.min()) if len(annual) else np.nan,
    }


def policy_score(g: pd.DataFrame) -> float:
    r = g["policy_ret"].dropna()
    if len(r) < 60:
        return -1e9
    annual = g.groupby("year")["policy_ret"].sum()
    # 收益为主，胜率和收益因子辅助，明显惩罚历史最差年。
    return float(r.sum() + (r > 0).mean() * 8.0 + min(profit_factor(r), 8.0) + min(annual.min(), 0.0) * 2.0)


def cap_policy_ret(df: pd.DataFrame, trend_t: float, rebound_t: float, cap_h: int) -> pd.Series:
    exit_col = f"exit_h{cap_h}"
    ret_col = f"ret_h{cap_h}"
    condition = (
        (df["trend_score"] > trend_t)
        & (df["rebound_score"] < rebound_t)
        & df[exit_col].notna()
        & (df["exit_date"] > df[exit_col])
        & df[ret_col].notna()
    )
    return df["ret"].where(~condition, df[ret_col])


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(name: str, ret: pd.Series) -> None:
        tmp = raw[["signal_date", "year", "symbol", "name", "ret"]].copy()
        tmp["policy"] = name
        tmp["policy_ret"] = ret
        rows.append(tmp)

    add("base_original_exit", raw["ret"])
    for cap_h in [20, 30, 40, 60]:
        add(f"cap_all_h{cap_h}", cap_policy_ret(raw, -1.0, 99.0, cap_h))
        for trend_t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            for rebound_t in [0.20, 0.25, 0.30, 0.35, 0.40]:
                add(
                    f"guard_t{trend_t:.2f}_r{rebound_t:.2f}_h{cap_h}",
                    cap_policy_ret(raw, trend_t, rebound_t, cap_h),
                )
    return pd.concat(rows, ignore_index=True)


def select_policy(train: pd.DataFrame, mode: str) -> tuple[str, pd.DataFrame]:
    stats = []
    for p, g in train.groupby("policy"):
        s = summary(g.assign(ret_eval=g["policy_ret"]), "ret_eval")
        s["policy"] = p
        s["score"] = policy_score(g)
        stats.append(s)
    rank = pd.DataFrame(stats).sort_values(["score", "sum"], ascending=False).reset_index(drop=True)
    base = rank[rank["policy"] == "base_original_exit"].iloc[0]
    best = rank.iloc[0]

    if mode == "argmax":
        return str(best["policy"]), rank

    if mode == "beat_base":
        # 只有过去数据中“收益更高、收益因子不低、最差年不明显更差”才允许替换。
        eligible = rank[
            (rank["sum"] > float(base["sum"]))
            & (rank["pf"] >= float(base["pf"]) * 0.98)
            & (rank["worst_year_sum"] >= float(base["worst_year_sum"]) - 0.5)
        ]
        if len(eligible):
            return str(eligible.iloc[0]["policy"]), rank
        return "base_original_exit", rank

    if mode == "conservative":
        eligible = rank[
            (rank["sum"] > float(base["sum"]) + 1.0)
            & (rank["win"] >= float(base["win"]))
            & (rank["worst_year_sum"] >= float(base["worst_year_sum"]))
        ]
        if len(eligible):
            return str(eligible.iloc[0]["policy"]), rank
        return "base_original_exit", rank

    raise ValueError(mode)


def run_rolling(panel: pd.DataFrame, min_train_years: int, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = []
    selected = []
    years = sorted(panel["year"].unique())
    for year in years:
        train_years = [y for y in years if y < year]
        test = panel[panel["year"] == year]
        if len(train_years) < min_train_years:
            chosen = "base_original_exit"
            rank = pd.DataFrame()
        else:
            train = panel[panel["year"] < year]
            chosen, rank = select_policy(train, mode)
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
                "train_years": ",".join(map(str, train_years[-min_train_years:])) if len(train_years) >= min_train_years else "",
                "test_n": int(len(picked)),
                "test_win": float((picked["policy_ret"] > 0).mean()) if len(picked) else np.nan,
                "test_sum": float(picked["policy_ret"].sum()) if len(picked) else 0.0,
                "top_policy_in_train": "" if rank.empty else str(rank.iloc[0]["policy"]),
                "base_rank_sum": np.nan if rank.empty else float(rank[rank["policy"] == "base_original_exit"].iloc[0]["sum"]),
            }
        )
    return pd.concat(selected, ignore_index=True), pd.DataFrame(choices)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(IN_PATH, parse_dates=["signal_date", "entry_date", "exit_date", "exit_h20", "exit_h30", "exit_h40", "exit_h60"])
    raw = raw[(raw["line"] == "D") & (raw["completed"] == True)].copy()
    raw["year"] = raw["signal_date"].dt.year

    panel = build_panel(raw)
    panel.to_csv(OUT_DIR / "candidate_panel.csv", index=False, encoding="utf-8-sig")

    full_rank = []
    for p, g in panel.groupby("policy"):
        s = summary(g.assign(ret_eval=g["policy_ret"]), "ret_eval")
        s["policy"] = p
        full_rank.append(s)
    full_rank_df = pd.DataFrame(full_rank).sort_values(["sum", "win"], ascending=False)
    full_rank_df.to_csv(OUT_DIR / "full_sample_candidate_rank.csv", index=False, encoding="utf-8-sig")

    all_selected = []
    all_choices = []
    for min_train in [1, 2, 3, 5]:
        for mode in ["argmax", "beat_base", "conservative"]:
            selected, choices = run_rolling(panel, min_train, mode)
            all_selected.append(selected)
            all_choices.append(choices)

    selected_df = pd.concat(all_selected, ignore_index=True)
    choices_df = pd.concat(all_choices, ignore_index=True)
    selected_df.to_csv(OUT_DIR / "strict_rolling_selected_trades.csv", index=False, encoding="utf-8-sig")
    choices_df.to_csv(OUT_DIR / "strict_rolling_choices.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    base = panel[panel["policy"] == "base_original_exit"].copy().assign(ret_eval=lambda x: x["policy_ret"])
    summary_rows.append({"model": "base_original_exit_all_years", **summary(base, "ret_eval")})
    v1 = panel[panel["policy"] == "guard_t0.55_r0.30_h40"].copy().assign(ret_eval=lambda x: x["policy_ret"])
    summary_rows.append({"model": "fixed_D_fail_guard_v1_all_years", **summary(v1, "ret_eval")})

    for (mode, min_train), g in selected_df.groupby(["selector_mode", "min_train_years"]):
        tmp = g.assign(ret_eval=g["policy_ret"])
        summary_rows.append({"model": f"rolling_{mode}_mintrain{min_train}", **summary(tmp, "ret_eval")})
    summary_df = pd.DataFrame(summary_rows).sort_values(["sum", "win"], ascending=False)
    summary_df.to_csv(OUT_DIR / "strict_rolling_summary.csv", index=False, encoding="utf-8-sig")

    # 同步输出 2021-2026，避免 2019/2020 暖启动年份干扰判断。
    sub_rows = []
    for model, group in [("base_original_exit_2021_2026", base[base["year"] >= 2021]), ("fixed_D_fail_guard_v1_2021_2026", v1[v1["year"] >= 2021])]:
        sub_rows.append({"model": model, **summary(group, "ret_eval")})
    for (mode, min_train), g in selected_df[selected_df["year"] >= 2021].groupby(["selector_mode", "min_train_years"]):
        sub_rows.append({"model": f"rolling_{mode}_mintrain{min_train}_2021_2026", **summary(g.assign(ret_eval=g["policy_ret"]), "ret_eval")})
    sub_df = pd.DataFrame(sub_rows).sort_values(["sum", "win"], ascending=False)
    sub_df.to_csv(OUT_DIR / "strict_rolling_summary_2021_2026.csv", index=False, encoding="utf-8-sig")

    print("ALL YEARS")
    print(summary_df.to_string(index=False))
    print("\n2021-2026")
    print(sub_df.to_string(index=False))
    print("\nCHOICES")
    print(choices_df.to_string(index=False))


if __name__ == "__main__":
    main()
