from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import factor_dca_scan as dca


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIRST_TEST_YEAR = 2015
LAST_TEST_YEAR = 2026
TRAIN_MIN_CLOSED = 40
TRAIN_MIN_CLOSED_RATE = 0.60
ROLLING_BUY_LIMIT = 25
ROLLING_SELL_LIMIT = 25


def group_slices(panel: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    slices: list[tuple[int, int]] = []
    for _, group in panel.groupby("symbol", sort=False):
        positions = group.index.to_numpy(dtype=int)
        if not len(positions):
            continue
        if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            raise RuntimeError("Panel rows must remain contiguous by ETF.")
        slices.append((int(positions[0]), int(positions[-1] + 1)))
    return {"all": slices}


def training_panel(full: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    indices = np.flatnonzero(pd.to_datetime(full["date"]).to_numpy() < cutoff)
    train = full.iloc[indices].copy().reset_index(drop=True)
    for _, group in train.groupby("symbol", sort=False):
        close = group["close"].astype(float)
        for horizon in (5, 20, 40, 60):
            entry = close.shift(-1)
            exit_price = close.shift(-(horizon + 1))
            train.loc[group.index, f"ret_{horizon}"] = exit_price / entry - 1
            train.loc[group.index, f"win_{horizon}"] = train.loc[group.index, f"ret_{horizon}"] > 0
        low, high = dca.original_paired_labels(close)
        train.loc[group.index, "label_low"] = low
        train.loc[group.index, "label_high"] = high
        train.loc[group.index, "near_low"] = dca.nearby_label(low)
        train.loc[group.index, "near_high"] = dca.nearby_label(high)
    return train, indices


def expand_candidate(candidate: dca.Candidate, full_conditions: dict[str, np.ndarray]) -> dca.Candidate:
    masks = [full_conditions[factor] for factor in candidate.factors]
    return dca.Candidate(candidate.identifier, candidate.label, candidate.factors, np.logical_and.reduce(masks))


def choose_for_year(
    train: pd.DataFrame,
    full_conditions_buy: dict[str, np.ndarray],
    full_conditions_sell: dict[str, np.ndarray],
    train_indices: np.ndarray,
    buy_labels: dict[str, str],
    buy_cores: dict[str, str],
    sell_labels: dict[str, str],
    sell_cores: dict[str, str],
) -> tuple[dca.Candidate, dca.Candidate, dict] | None:
    train_buy_conditions = {key: value[train_indices] for key, value in full_conditions_buy.items()}
    train_sell_conditions = {key: value[train_indices] for key, value in full_conditions_sell.items()}
    buys, _, _ = dca.candidates_for_side(
        train_buy_conditions, buy_labels, buy_cores, train, "buy", max_factor_count=3
    )
    sells, _, _ = dca.candidates_for_side(
        train_sell_conditions, sell_labels, sell_cores, train, "sell", max_factor_count=2
    )
    train_groups = group_slices(train)
    ranked: list[tuple[dca.Candidate, dca.Candidate, dict]] = []
    for buy in buys[:ROLLING_BUY_LIMIT]:
        for sell in sells[:ROLLING_SELL_LIMIT]:
            metrics = dca.pair_result(buy, sell, train, train_groups)
            if metrics["closed_batches"] < TRAIN_MIN_CLOSED or metrics["closed_rate"] < TRAIN_MIN_CLOSED_RATE:
                continue
            ranked.append((buy, sell, metrics))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[2]["win_rate"], item[2]["avg_return"], item[2]["closed_batches"]), reverse=True)
    train_buy, train_sell, metrics = ranked[0]
    return expand_candidate(train_buy, full_conditions_buy), expand_candidate(train_sell, full_conditions_sell), metrics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, full_groups = dca.load_panel()
    buy_conditions, buy_labels, buy_cores = dca.build_conditions(full, "buy")
    sell_conditions, sell_labels, sell_cores = dca.build_conditions(full, "sell")
    max_year = min(LAST_TEST_YEAR, int(pd.to_datetime(full["date"]).dt.year.max()))
    rows: list[dict] = []

    for year in range(FIRST_TEST_YEAR, max_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train, train_indices = training_panel(full, cutoff)
        if train.empty or train["symbol"].nunique() < 5:
            continue
        chosen = choose_for_year(
            train,
            buy_conditions,
            sell_conditions,
            train_indices,
            buy_labels,
            buy_cores,
            sell_labels,
            sell_cores,
        )
        if chosen is None:
            rows.append({"test_year": year, "status": "no_train_candidate"})
            continue
        buy, sell, train_metrics = chosen
        test_metrics = dca.pair_result(
            buy,
            sell,
            full,
            full_groups,
            entry_start=cutoff,
            entry_end=pd.Timestamp(year=year + 1, month=1, day=1),
        )
        rows.append(
            {
                "test_year": year,
                "status": "selected",
                "buy_combo": buy.label,
                "sell_combo": sell.label,
                "train_closed_batches": train_metrics["closed_batches"],
                "train_closed_rate": train_metrics["closed_rate"],
                "train_win_rate": train_metrics["win_rate"],
                "train_avg_return": train_metrics["avg_return"],
                "test_buy_batches": test_metrics["buy_batches"],
                "test_closed_batches": test_metrics["closed_batches"],
                "test_closed_rate": test_metrics["closed_rate"],
                "test_win_batches": test_metrics["win_batches"],
                "test_win_rate": test_metrics["win_rate"],
                "test_avg_return": test_metrics["avg_return"],
                "test_return_sum": test_metrics["return_sum"],
            }
        )
        print(f"{year}: {buy.label} -> {sell.label}; test win={test_metrics['win_rate']:.2%} ({test_metrics['closed_batches']} closed)")

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "rolling_oos.csv", index=False, encoding="utf-8-sig")
    selected = detail[detail["status"].eq("selected")].copy()
    closed = selected["test_closed_batches"].fillna(0).sum()
    wins = selected["test_win_batches"].fillna(0).sum()
    total_return = selected["test_return_sum"].fillna(0).sum()
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Annual rolling selection. Each test year selects a buy/sell pair using only prior data, then evaluates that year’s newly opened DCA lots with the selected exit rule.",
        "test_years": f"{FIRST_TEST_YEAR}-{max_year}",
        "selected_years": int(len(selected)),
        "closed_batches": int(closed),
        "win_batches": int(wins),
        "oos_win_rate": float(wins / closed) if closed else None,
        "average_return_per_closed_batch": float(total_return / closed) if closed else None,
        "total_pnl_per_100_cny_batch": float(total_return * 100),
        "minimum_hold_days": dca.MIN_HOLD_DAYS,
        "entry_execution": "T+1 close",
    }
    (OUT / "rolling_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# D 线年度滚动样本外验证",
        "",
        "每个测试年都只使用该年以前的数据重新选择低位买入组合与高位卖出组合。每笔低位信号在下一交易日收盘买入 100 元；卖点在下一交易日收盘卖出已持有满 7 个交易日的批次。",
        "",
        "## 汇总",
        f"- 测试区间：{summary['test_years']}。",
        f"- 已完成批次：{summary['closed_batches']}；盈利批次：{summary['win_batches']}；逐笔胜率：{summary['oos_win_rate']:.2%}。",
        f"- 已完成批次平均收益：{summary['average_return_per_closed_batch']:.2%}。",
        f"- 每次 100 元、仅统计已完成批次的收益合计：{summary['total_pnl_per_100_cny_batch']:.2f} 元。",
        "",
        "## 各年度滚动选择结果",
        selected.to_markdown(index=False),
        "",
        "注意：2025 年新开批次在数据截止日尚未触发可执行卖点，不能据此计算胜率；该部分不应被当作亏损或盈利处理。",
    ]
    (OUT / "rolling_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
