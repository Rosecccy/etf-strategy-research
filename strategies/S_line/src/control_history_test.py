from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from control_hierarchy_test import (
    DEV_YEARS,
    FORMAL_TRADES_PATH,
    HOLDOUT_YEARS,
    ROOT,
    SELECTED_PATH,
    approximate_net,
    metrics,
    parse_param,
    rolling_mode,
    score,
)
from shadow_model import Param, ShadowResearchEngine


OUT = ROOT / "fit" / "history_test"


@dataclass(frozen=True)
class HistoryPolicy:
    key: str
    family: str
    minimum: int
    reliable_at: int


def policies() -> list[HistoryPolicy]:
    return [
        HistoryPolicy("flat", "hard", 250, 250),
        HistoryPolicy("hard_120", "hard", 120, 120),
        HistoryPolicy("hard_180", "hard", 180, 180),
        HistoryPolicy("hard_360", "hard", 360, 360),
        HistoryPolicy("hard_500", "hard", 500, 500),
        HistoryPolicy("shrink_120_250", "shrink", 120, 250),
        HistoryPolicy("shrink_120_360", "shrink", 120, 360),
        HistoryPolicy("shrink_120_500", "shrink", 120, 500),
        HistoryPolicy("shrink_180_360", "shrink", 180, 360),
        HistoryPolicy("shrink_180_500", "shrink", 180, 500),
    ]


def with_history(param: Param, minimum: int) -> Param:
    return Param(
        param.scope,
        param.idle,
        param.hold,
        param.ma,
        param.mom,
        param.model,
        param.alpha,
        param.vol_cap,
        param.accel_cap,
        minimum,
        param.min_amount20,
    )


def choose(
    candidates: tuple[tuple[float, str, dict], ...],
    policy: HistoryPolicy,
) -> tuple[float, str, dict] | None:
    if not candidates:
        return None
    if policy.family == "hard":
        return candidates[0]
    adjusted: list[tuple[float, str, dict]] = []
    for score_value, symbol, extra in candidates:
        history = int(extra.get("history_days", 0))
        reliability = min(1.0, history / max(policy.reliable_at, 1))
        shrunk_score = 0.5 + reliability * (float(score_value) - 0.5)
        copied = dict(extra)
        copied["raw_score"] = float(score_value)
        copied["history_reliability"] = reliability
        copied["score"] = shrunk_score
        adjusted.append((shrunk_score, symbol, copied))
    return max(adjusted, key=lambda item: (item[0], item[1]))


def generate_trades(
    engine: ShadowResearchEngine,
    original_param_id: str,
    param: Param,
    policy: HistoryPolicy,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for start, end, idle_days in engine.idle_windows:
        if idle_days < param.idle:
            continue
        idle_calendar = engine.calendar[
            (engine.calendar >= start) & (engine.calendar <= end)
        ].reset_index(drop=True)
        if len(idle_calendar) < param.idle:
            continue
        current = pd.Timestamp(idle_calendar.iloc[param.idle - 1])
        while current <= end:
            candidates = engine.candidate_list(
                current.value,
                param.scope,
                param.ma,
                param.mom,
                param.model,
                param.alpha,
                policy.minimum,
                param.min_amount20,
                param.vol_cap,
                param.accel_cap,
            )
            chosen = choose(candidates, policy)
            if chosen is None:
                next_date = engine.next_date(current + pd.Timedelta(days=1))
                if next_date is None:
                    break
                current = next_date
                continue
            _, symbol, extra = chosen
            frame = engine.panel[symbol]
            entry_pos = engine.pos_after(frame, current + pd.Timedelta(days=1))
            end_pos = engine.pos_before(frame, end)
            if entry_pos is None or end_pos is None or entry_pos > end_pos:
                break
            exit_pos = min(entry_pos + param.hold, end_pos)
            if exit_pos <= entry_pos:
                break
            entry = frame.iloc[entry_pos]
            exit_row = frame.iloc[exit_pos]
            rows.append(
                {
                    "source": "空仓补偿_S1",
                    "source_type": "空仓补偿",
                    "category": engine.meta[symbol]["category"],
                    "symbol": symbol,
                    "display_name": engine.meta[symbol]["display_name"],
                    "decision_date": current,
                    "entry_date": pd.Timestamp(entry["date"]),
                    "exit_date": pd.Timestamp(exit_row["date"]),
                    "entry_close": float(entry["close"]),
                    "exit_close": float(exit_row["close"]),
                    "ret": float(exit_row["close"] / entry["close"] - 1.0),
                    "priority": 100.0,
                    "history_policy": policy.key,
                    "original_param_id": original_param_id,
                    "param_id": param.param_id,
                    "idle_days": param.idle,
                    "hold": param.hold,
                    "ma": param.ma,
                    "mom": param.mom,
                    "model": param.model,
                    "alpha": param.alpha,
                    **extra,
                }
            )
            next_date = engine.next_date(
                pd.Timestamp(exit_row["date"]) + pd.Timedelta(days=1)
            )
            if next_date is None:
                break
            current = next_date
    return pd.DataFrame(rows)


def build_account(
    engine: ShadowResearchEngine,
    annual: pd.DataFrame,
    generated: dict[tuple[str, str], pd.DataFrame],
    policy: HistoryPolicy,
) -> pd.DataFrame:
    pieces = [engine.main.copy()]
    for _, choice_row in annual.iterrows():
        param_id = str(choice_row["param_id"])
        if param_id == "CASH":
            continue
        year = int(choice_row["year"])
        frame = generated[(policy.key, param_id)].copy()
        years = pd.to_datetime(frame["entry_date"]).dt.year
        pieces.append(frame[years.eq(year)].copy())
    return engine.account_from_trades(
        pd.concat(pieces, ignore_index=True, sort=False)
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = ShadowResearchEngine()
    annual = pd.read_csv(
        SELECTED_PATH, dtype={"param_id": str}, encoding="utf-8-sig"
    )
    base_params = {
        param_id: parse_param(param_id)
        for param_id in annual["param_id"].dropna().unique()
        if param_id != "CASH"
    }
    candidates = policies()
    generated: dict[tuple[str, str], pd.DataFrame] = {}
    total = len(candidates) * len(base_params)
    counter = 0
    for policy in candidates:
        for param_id, base in base_params.items():
            counter += 1
            param = with_history(base, policy.minimum)
            generated[(policy.key, param_id)] = generate_trades(
                engine, param_id, param, policy
            )
            print(f"[S history {counter:02d}/{total}] {policy.key} {param_id}")

    accounts = {
        policy.key: build_account(engine, annual, generated, policy)
        for policy in candidates
    }
    policy_lookup = {policy.key: policy for policy in candidates}
    fixed_rows: list[dict[str, object]] = []
    for key, account in accounts.items():
        dev = metrics(account, DEV_YEARS)
        holdout = metrics(account, HOLDOUT_YEARS)
        policy = policy_lookup[key]
        fixed_rows.append(
            {
                "history_policy": key,
                "family": policy.family,
                "minimum_history": policy.minimum,
                "fully_reliable_at": policy.reliable_at,
                "dev_score": score(dev),
                **{f"dev_{name}": value for name, value in dev.items()},
                **{f"holdout_{name}": value for name, value in holdout.items()},
                "approx_net_final_10000": approximate_net(account),
            }
        )
    fixed = pd.DataFrame(fixed_rows).sort_values(
        ["dev_score", "dev_final_1000", "dev_win_rate"], ascending=False
    )

    method_rows: list[dict[str, object]] = []
    choices_map: dict[str, pd.DataFrame] = {}
    logs_map: dict[str, pd.DataFrame] = {}
    for window in (3, 5, 7, None):
        method = f"w{window if window is not None else 'all'}"
        choices, log = rolling_mode(accounts, window, engine)
        choices_map[method] = choices
        logs_map[method] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        method_rows.append(
            {
                "method": method,
                "dev_score": score(dev),
                **{f"dev_{name}": value for name, value in dev.items()},
                **{f"holdout_{name}": value for name, value in holdout.items()},
                "approx_net_final_10000": approximate_net(log),
            }
        )
    methods = pd.DataFrame(method_rows).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )
    winner = str(methods.iloc[0]["method"])

    formal = pd.read_csv(
        FORMAL_TRADES_PATH, dtype={"symbol": str}, encoding="utf-8-sig"
    )
    formal["entry_date"] = pd.to_datetime(formal["entry_date"])
    baseline_holdout = metrics(formal, HOLDOUT_YEARS)
    reproduced = metrics(accounts["flat"], DEV_YEARS | HOLDOUT_YEARS)
    winner_holdout = metrics(logs_map[winner], HOLDOUT_YEARS)
    neighbor_passes = int(
        (
            (methods["holdout_final_1000"] >= baseline_holdout["final_1000"])
            & (
                methods["holdout_win_rate"]
                >= float(baseline_holdout["win_rate"]) - 0.02
            )
            & (
                methods["holdout_max_drawdown"]
                >= float(baseline_holdout["max_drawdown"]) - 0.02
            )
        ).sum()
    )
    promoted = bool(
        winner_holdout["final_1000"] > baseline_holdout["final_1000"]
        and winner_holdout["win_rate"] >= baseline_holdout["win_rate"]
        and winner_holdout["max_drawdown"] >= baseline_holdout["max_drawdown"]
        and neighbor_passes >= 2
    )

    fixed.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    methods.to_csv(OUT / "rolling_methods.csv", index=False, encoding="utf-8-sig")
    choices_map[winner].to_csv(
        OUT / "winner_choices.csv", index=False, encoding="utf-8-sig"
    )
    logs_map[winner].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "etf_history_credibility_only",
        "formal_baseline_holdout": baseline_holdout,
        "baseline_reproduction_all": reproduced,
        "development_selected_method": winner,
        "winner_holdout": winner_holdout,
        "winner_approx_net_final_10000": approximate_net(logs_map[winner]),
        "neighbor_methods_passing_holdout_floor": neighbor_passes,
        "promoted": promoted,
        "formal_config_changed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report = (
        "# S 线 ETF 历史可信度单变量测试\n\n"
        f"- 正式基线留出期终值：{baseline_holdout['final_1000']:.2f}\n"
        f"- 开发期选中的滚动窗口：{winner}\n"
        f"- 候选留出期终值：{winner_holdout['final_1000']:.2f}\n"
        f"- 候选留出期胜率：{winner_holdout['win_rate']:.2%}\n"
        f"- 候选留出期最大回撤：{winner_holdout['max_drawdown']:.2%}\n"
        f"- 邻域通过数量：{neighbor_passes}\n"
        f"- 是否升级：{'是' if promoted else '否'}\n\n"
        "本轮只改变 ETF 最短历史长度，以及短历史 ETF 排名向中性分数收缩的程度。"
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
