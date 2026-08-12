from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "D" / "out" / "d_failure_diagnosis" / "d_fixed_hold_returns.csv"
OUT_DIR = ROOT / "D" / "out" / "d_failure_diagnosis"


def profit_factor(ret: pd.Series) -> float:
    wins = ret[ret > 0].sum()
    losses = -ret[ret < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def summarize(df: pd.DataFrame, ret_col: str, prefix: str) -> dict[str, float]:
    x = df[df[ret_col].notna()].copy()
    r = x[ret_col]
    out: dict[str, float] = {
        f"{prefix}_n": int(len(x)),
        f"{prefix}_win": float((r > 0).mean()) if len(x) else np.nan,
        f"{prefix}_avg": float(r.mean()) if len(x) else np.nan,
        f"{prefix}_sum": float(r.sum()) if len(x) else 0.0,
        f"{prefix}_pf": float(profit_factor(r)) if len(x) else np.nan,
    }
    if len(x):
        yearly = x.groupby("year")[ret_col].sum()
        out[f"{prefix}_annual_avg_sum"] = float(yearly.mean())
        out[f"{prefix}_worst_year_sum"] = float(yearly.min())
    else:
        out[f"{prefix}_annual_avg_sum"] = np.nan
        out[f"{prefix}_worst_year_sum"] = np.nan
    return out


def make_ret(df: pd.DataFrame, hold: int) -> pd.Series:
    col = f"ret_h{hold}"
    if col not in df.columns:
        raise ValueError(f"missing {col}")
    return df[col]


def danger_mask(df: pd.DataFrame, trend: float, rebound: float, m20: float) -> pd.Series:
    # 伪恐慌：市场仍偏强，反弹并不充分，20日中位收益也未明显转弱。
    return (
        (df["trend_score"] > trend)
        & (df["rebound_score"] < rebound)
        & (df["median_ret20"] > m20)
    )


def weak_mask(df: pd.DataFrame, trend: float, m20: float) -> pd.Series:
    return (df["trend_score"] <= trend) & (df["median_ret20"] < m20)


def evaluate_policy(
    src: pd.DataFrame,
    name: str,
    ret: pd.Series,
    active: pd.Series | None = None,
) -> dict[str, float | str]:
    df = src.copy()
    df["policy_ret"] = ret
    if active is not None:
        df = df[active.fillna(False)].copy()
    df = df[df["policy_ret"].notna()].copy()
    y26 = df[df["year"] == 2026]
    pre26 = df[df["year"] < 2026]
    out: dict[str, float | str] = {"policy": name}
    out.update(summarize(df, "policy_ret", "all"))
    out.update(summarize(pre26, "policy_ret", "pre26"))
    out.update(summarize(y26, "policy_ret", "y26"))
    out["active_rate"] = float(len(df) / len(src)) if len(src) else np.nan
    # 诊断分：重罚 2026 亏损，同时保留总收益、胜率、收益因子。
    y26_sum = float(out["y26_sum"])
    out["balanced_score"] = (
        float(out["all_sum"])
        + min(y26_sum, 0.0) * 6.0
        + float(out["all_win"]) * 10.0
        + min(float(out["all_pf"]), 10.0)
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH, parse_dates=["signal_date", "entry_date", "exit_date"])
    df = df[(df["line"] == "D") & (df["completed"] == True)].copy()
    df["year"] = df["signal_date"].dt.year

    policies: list[dict[str, float | str]] = []

    # Baselines: 原始 D 和固定持有。
    policies.append(evaluate_policy(df, "base_original_exit", df["ret"]))
    for h in [10, 15, 20, 30, 40, 60, 90]:
        policies.append(evaluate_policy(df, f"fixed_h{h}", make_ret(df, h)))

    # 只过滤买入，不改变持有。
    for t in np.arange(0.35, 0.86, 0.05):
        active = df["trend_score"] <= round(float(t), 2)
        policies.append(evaluate_policy(df, f"skip_trend_gt_{t:.2f}_orig", df["ret"], active))
        for h in [20, 30, 40, 60, 90]:
            policies.append(
                evaluate_policy(df, f"skip_trend_gt_{t:.2f}_h{h}", make_ret(df, h), active)
            )

    # 伪恐慌过滤：危险形态直接跳过。
    for t in [0.45, 0.50, 0.55, 0.60, 0.65]:
        for r in [0.25, 0.30, 0.35, 0.40]:
            for m20 in [-0.04, -0.02, 0.00, 0.02]:
                active = ~danger_mask(df, t, r, m20)
                policies.append(
                    evaluate_policy(df, f"skip_pseudo_t{t:.2f}_r{r:.2f}_m{m20:.2f}_orig", df["ret"], active)
                )
                for h in [20, 30, 40, 60, 90]:
                    policies.append(
                        evaluate_policy(
                            df,
                            f"skip_pseudo_t{t:.2f}_r{r:.2f}_m{m20:.2f}_h{h}",
                            make_ret(df, h),
                            active,
                        )
                    )

    # 动态持有：正常长持有，危险或弱势缩短。
    for short_h in [10, 15, 20, 30, 40]:
        for long_h in [40, 60, 90]:
            if short_h >= long_h:
                continue
            for t in [0.35, 0.45, 0.55, 0.65]:
                ret = make_ret(df, long_h).where(df["trend_score"] > t, make_ret(df, short_h))
                policies.append(evaluate_policy(df, f"dyn_trend_t{t:.2f}_h{short_h}_{long_h}", ret))

            for t in [0.45, 0.50, 0.55, 0.60]:
                for r in [0.25, 0.30, 0.35, 0.40]:
                    for m20 in [-0.04, -0.02, 0.00]:
                        danger = danger_mask(df, t, r, m20)
                        ret = make_ret(df, long_h).where(~danger, make_ret(df, short_h))
                        policies.append(
                            evaluate_policy(
                                df,
                                f"dyn_pseudo_short_t{t:.2f}_r{r:.2f}_m{m20:.2f}_h{short_h}_{long_h}",
                                ret,
                            )
                        )
                        active = ~danger
                        policies.append(
                            evaluate_policy(
                                df,
                                f"skip_pseudo_dyn_t{t:.2f}_r{r:.2f}_m{m20:.2f}_h{short_h}_{long_h}",
                                make_ret(df, long_h),
                                active,
                            )
                        )

            for t in [0.35, 0.45, 0.55, 0.65, 0.75]:
                for m20 in [-0.06, -0.04, -0.02, 0.00]:
                    weak = weak_mask(df, t, m20)
                    ret = make_ret(df, long_h).where(~weak, make_ret(df, short_h))
                    policies.append(
                        evaluate_policy(
                            df,
                            f"dyn_weak_short_t{t:.2f}_m{m20:.2f}_h{short_h}_{long_h}",
                            ret,
                        )
                    )

    # 组合型落地候选：伪恐慌跳过，弱势缩短，正常长持有。
    for short_h in [10, 15, 20, 30, 40]:
        for long_h in [40, 60, 90]:
            if short_h >= long_h:
                continue
            for dt in [0.45, 0.50, 0.55, 0.60]:
                for dr in [0.25, 0.30, 0.35, 0.40]:
                    for dm20 in [-0.04, -0.02, 0.00]:
                        danger = danger_mask(df, dt, dr, dm20)
                        for wt in [0.35, 0.45, 0.55, 0.65, 0.75]:
                            for wm20 in [-0.06, -0.04, -0.02, 0.00]:
                                weak = weak_mask(df, wt, wm20)
                                ret = make_ret(df, long_h).where(~weak, make_ret(df, short_h))
                                policies.append(
                                    evaluate_policy(
                                        df,
                                        (
                                            f"skip_pseudo_weak_short_dt{dt:.2f}_dr{dr:.2f}_dm{dm20:.2f}"
                                            f"_wt{wt:.2f}_wm{wm20:.2f}_h{short_h}_{long_h}"
                                        ),
                                        ret,
                                        ~danger,
                                    )
                                )

    # 组合型保守候选：伪恐慌和强趋势追高都跳过，弱势短持有。
    for short_h in [15, 20, 30]:
        for long_h in [40, 60, 90]:
            for strong_t in [0.60, 0.65, 0.70, 0.75]:
                for dt in [0.50, 0.55, 0.60]:
                    for dr in [0.30, 0.35, 0.40]:
                        for dm20 in [-0.04, -0.02, 0.00]:
                            danger = danger_mask(df, dt, dr, dm20) | (
                                (df["trend_score"] > strong_t) & (df["median_ret20"] > 0.00)
                            )
                            for wt in [0.45, 0.55, 0.65]:
                                weak = weak_mask(df, wt, -0.02)
                                ret = make_ret(df, long_h).where(~weak, make_ret(df, short_h))
                                policies.append(
                                    evaluate_policy(
                                        df,
                                        (
                                            f"skip_pseudo_strong_weak_short_st{strong_t:.2f}"
                                            f"_dt{dt:.2f}_dr{dr:.2f}_dm{dm20:.2f}"
                                            f"_wt{wt:.2f}_h{short_h}_{long_h}"
                                        ),
                                        ret,
                                        ~danger,
                                    )
                                )

    rank = pd.DataFrame(policies)
    rank = rank.replace([np.inf, -np.inf], np.nan)
    rank = rank.sort_values(["balanced_score", "all_sum"], ascending=False)
    rank.to_csv(OUT_DIR / "dynamic_hold_policy_rank.csv", index=False, encoding="utf-8-sig")

    best = {
        "base_original_exit": rank[rank["policy"] == "base_original_exit"].iloc[0].to_dict(),
        "best_balanced": rank.iloc[0].to_dict(),
        "best_y26_nonnegative": (
            rank[rank["y26_sum"] >= 0].sort_values("all_sum", ascending=False).head(1).to_dict("records")
        ),
        "best_y26_loss_under_100pct": (
            rank[rank["y26_sum"] >= -1.0].sort_values("all_sum", ascending=False).head(1).to_dict("records")
        ),
        "best_all_sum": rank.sort_values("all_sum", ascending=False).head(1).to_dict("records"),
    }
    with open(OUT_DIR / "dynamic_hold_best_summary.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("saved", OUT_DIR / "dynamic_hold_policy_rank.csv")
    print(rank.head(20)[["policy", "all_n", "all_win", "all_avg", "all_sum", "all_pf", "y26_n", "y26_win", "y26_sum", "balanced_score"]].to_string(index=False))
    print("best_y26_nonnegative")
    print(pd.DataFrame(best["best_y26_nonnegative"])[["policy", "all_n", "all_win", "all_sum", "all_pf", "y26_sum"]].to_string(index=False))


if __name__ == "__main__":
    main()
