from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from control_exit_test import (
    DEV_YEARS,
    ETF_DIR,
    HOLDOUT_YEARS,
    ROOT,
    TRADES_PATH,
    load_prices,
    metrics,
    net_account,
    position_on_or_after,
    rolling_choices,
    selection_score,
)


OUT = ROOT / "fit" / "regime_test"
EXTEND_DAYS = 3


@dataclass(frozen=True)
class RegimeRule:
    key: str
    family: str
    ret20_min: float = -np.inf
    breadth_min: float = -np.inf


def rules() -> list[RegimeRule]:
    values = [RegimeRule("baseline", "baseline")]
    values.extend(
        RegimeRule(f"market_ret20_{int(value * 100):02d}", "ret20", value)
        for value in (0.02, 0.03, 0.04, 0.05)
    )
    values.extend(
        RegimeRule(f"breadth_{int(value * 100):02d}", "breadth", breadth_min=value)
        for value in (0.50, 0.55, 0.60, 0.65)
    )
    for ret20 in (0.02, 0.03, 0.04, 0.05):
        for breadth in (0.50, 0.55, 0.60, 0.65):
            values.append(
                RegimeRule(
                    f"combo_r{int(ret20 * 100):02d}_b{int(breadth * 100):02d}",
                    "combined",
                    ret20,
                    breadth,
                )
            )
    return values


def load_market_state() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(ETF_DIR.glob("*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        frame = frame.drop_duplicates("date", keep="last")
        frame["ret20"] = frame["close"].pct_change(20)
        frame["ma60"] = frame["close"].rolling(60, min_periods=60).mean()
        frame["above_ma60"] = np.where(
            frame["ma60"].notna(),
            frame["close"].ge(frame["ma60"]).astype(float),
            np.nan,
        )
        pieces.append(frame[["date", "ret20", "above_ma60"]])
    if not pieces:
        raise FileNotFoundError(f"No ETF CSV files found in {ETF_DIR}")
    panel = pd.concat(pieces, ignore_index=True)
    return (
        panel.groupby("date", as_index=False)
        .agg(
            market_ret20=("ret20", "median"),
            breadth_ma60=("above_ma60", "mean"),
            available_etfs=("above_ma60", "count"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def state_on_or_before(state: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    pos = int(state["date"].searchsorted(date, side="right")) - 1
    return state.iloc[pos] if pos >= 0 else None


def is_strong(row: pd.Series | None, rule: RegimeRule) -> bool:
    if row is None or rule.family == "baseline":
        return False
    ret20 = float(row["market_ret20"])
    breadth = float(row["breadth_ma60"])
    if rule.family == "ret20":
        return np.isfinite(ret20) and ret20 >= rule.ret20_min
    if rule.family == "breadth":
        return np.isfinite(breadth) and breadth >= rule.breadth_min
    return (
        np.isfinite(ret20)
        and np.isfinite(breadth)
        and ret20 >= rule.ret20_min
        and breadth >= rule.breadth_min
    )


def build_log(
    source: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    state: pd.DataFrame,
    rule: RegimeRule,
    extend_days: int = EXTEND_DAYS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    busy_until = pd.Timestamp.min
    for _, trade in source.sort_values(["entry_date", "symbol"]).iterrows():
        entry_date = pd.Timestamp(trade["entry_date"])
        if entry_date < busy_until:
            continue
        symbol = str(trade["symbol"]).zfill(6)
        frame = price_map[symbol]
        entry_pos = position_on_or_after(frame, entry_date)
        planned_pos = position_on_or_after(frame, pd.Timestamp(trade["exit_date"]))
        if entry_pos is None or planned_pos is None or planned_pos <= entry_pos:
            continue

        exit_pos = planned_pos
        decision_pos = max(entry_pos, planned_pos - 1)
        decision_date = pd.Timestamp(frame.iloc[decision_pos]["date"])
        market_row = state_on_or_before(state, decision_date)
        strong = str(trade["source"]).strip() == "主策略" and is_strong(
            market_row, rule
        )
        if strong:
            exit_pos = min(planned_pos + extend_days, len(frame) - 1)

        entry_close = float(frame.iloc[entry_pos]["close"])
        exit_close = float(frame.iloc[exit_pos]["close"])
        changed = exit_pos != planned_pos
        item = trade.to_dict()
        item.update(
            {
                "rule": rule.key,
                "entry_date": entry_date,
                "original_exit_date": pd.Timestamp(trade["exit_date"]),
                "exit_date": pd.Timestamp(frame.iloc[exit_pos]["date"]),
                "entry_close_test": entry_close,
                "exit_close_test": exit_close,
                "ret_test": (
                    exit_close / entry_close - 1.0
                    if changed
                    else float(trade["ret"])
                ),
                "exit_shift_days": int(exit_pos - planned_pos),
                "market_decision_date": decision_date,
                "market_ret20": (
                    float(market_row["market_ret20"])
                    if market_row is not None
                    else np.nan
                ),
                "market_breadth_ma60": (
                    float(market_row["breadth_ma60"])
                    if market_row is not None
                    else np.nan
                ),
                "market_strong": strong,
            }
        )
        rows.append(item)
        busy_until = pd.Timestamp(item["exit_date"])
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        TRADES_PATH, dtype={"symbol": str}, encoding="utf-8-sig"
    )
    source["symbol"] = source["symbol"].astype(str).str.zfill(6)
    source["entry_date"] = pd.to_datetime(source["entry_date"])
    source["exit_date"] = pd.to_datetime(source["exit_date"])
    source["ret"] = pd.to_numeric(source["ret"], errors="coerce")
    price_map = {
        symbol: load_prices(symbol)
        for symbol in sorted(source["symbol"].unique())
    }
    state = load_market_state()
    state.to_csv(OUT / "market_state.csv", index=False, encoding="utf-8-sig")

    logs: dict[str, pd.DataFrame] = {}
    fixed_rows: list[dict[str, object]] = []
    candidates = rules()
    for index, rule in enumerate(candidates, start=1):
        log = build_log(source, price_map, state, rule)
        logs[rule.key] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        fixed_rows.append(
            {
                "rule": rule.key,
                "family": rule.family,
                "ret20_min": rule.ret20_min,
                "breadth_min": rule.breadth_min,
                "extend_days": EXTEND_DAYS,
                "dev_score": selection_score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{f"holdout_{key}": value for key, value in holdout.items()},
            }
        )
        print(f"[C regime {index:02d}/{len(candidates)}] {rule.key}")
    fixed = pd.DataFrame(fixed_rows).sort_values(
        ["dev_score", "dev_final_1000", "dev_win_rate"], ascending=False
    )

    method_rows: list[dict[str, object]] = []
    choices_map: dict[str, pd.DataFrame] = {}
    logs_map: dict[str, pd.DataFrame] = {}
    for window in (3, 5, 7, None):
        method = f"w{window if window is not None else 'all'}"
        choices, log = rolling_choices(logs, window)
        choices_map[method] = choices
        logs_map[method] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        method_rows.append(
            {
                "method": method,
                "dev_score": selection_score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{f"holdout_{key}": value for key, value in holdout.items()},
            }
        )
    methods = pd.DataFrame(method_rows).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )
    winner = str(methods.iloc[0]["method"])
    baseline_holdout = metrics(logs["baseline"], HOLDOUT_YEARS)
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
    winner_net_log, winner_net = net_account(logs_map[winner])
    _, baseline_net = net_account(logs["baseline"])

    fixed.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    methods.to_csv(OUT / "rolling_methods.csv", index=False, encoding="utf-8-sig")
    choices_map[winner].to_csv(
        OUT / "winner_choices.csv", index=False, encoding="utf-8-sig"
    )
    logs_map[winner].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    winner_net_log.to_csv(
        OUT / "winner_net_trades.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "market_regime_definition_only",
        "fixed_exit_extension_days": EXTEND_DAYS,
        "baseline_holdout": baseline_holdout,
        "baseline_net": baseline_net,
        "development_selected_method": winner,
        "winner_holdout": winner_holdout,
        "winner_net": winner_net,
        "neighbor_methods_passing_holdout_floor": neighbor_passes,
        "promoted": promoted,
        "formal_config_changed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report = (
        "# C 线市场状态单变量测试\n\n"
        f"- 正式基线留出期终值：{baseline_holdout['final_1000']:.2f}\n"
        f"- 开发期选中的滚动窗口：{winner}\n"
        f"- 候选留出期终值：{winner_holdout['final_1000']:.2f}\n"
        f"- 候选留出期胜率：{winner_holdout['win_rate']:.2%}\n"
        f"- 候选留出期最大回撤：{winner_holdout['max_drawdown']:.2%}\n"
        f"- 邻域通过数量：{neighbor_passes}\n"
        f"- 是否升级：{'是' if promoted else '否'}\n\n"
        "本轮只改变全市场强势状态定义；强势时原卖点统一延后 3 个交易日。"
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
