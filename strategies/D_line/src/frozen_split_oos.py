from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

import factor_dca_scan as dca
import rolling_dca_validate as rolling


OUT = rolling.OUT
DISCOVERY_CUTOFF = pd.Timestamp("2017-01-01")
MIN_DISCOVERY_CLOSED = 60
MIN_DISCOVERY_CLOSED_RATE = 0.60
PAIR_LIMIT = 50


def select_frozen_pair(
    train: pd.DataFrame,
    full_buy_conditions: dict[str, np.ndarray],
    full_sell_conditions: dict[str, np.ndarray],
    indices: np.ndarray,
    buy_labels: dict[str, str],
    buy_cores: dict[str, str],
    sell_labels: dict[str, str],
    sell_cores: dict[str, str],
) -> tuple[dca.Candidate, dca.Candidate, dict, pd.DataFrame]:
    train_buy_conditions = {key: value[indices] for key, value in full_buy_conditions.items()}
    train_sell_conditions = {key: value[indices] for key, value in full_sell_conditions.items()}
    buys, _, _ = dca.candidates_for_side(
        train_buy_conditions, buy_labels, buy_cores, train, "buy", max_factor_count=3
    )
    sells, _, _ = dca.candidates_for_side(
        train_sell_conditions, sell_labels, sell_cores, train, "sell", max_factor_count=2
    )
    groups = rolling.group_slices(train)
    rows: list[dict] = []
    lookup: dict[tuple[str, str], tuple[dca.Candidate, dca.Candidate, dict]] = {}
    for buy in buys[:PAIR_LIMIT]:
        for sell in sells[:PAIR_LIMIT]:
            metrics = dca.pair_result(buy, sell, train, groups, with_years=True)
            if (
                metrics["closed_batches"] < MIN_DISCOVERY_CLOSED
                or metrics["closed_rate"] < MIN_DISCOVERY_CLOSED_RATE
            ):
                continue
            rows.append(metrics)
            lookup[(buy.identifier, sell.identifier)] = (buy, sell, metrics)
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise RuntimeError("No frozen candidate meets the pre-2017 evidence threshold.")
    ranking = ranking.sort_values(
        ["win_rate", "avg_return", "closed_batches"], ascending=[False, False, False]
    ).reset_index(drop=True)
    winner = ranking.iloc[0]
    buy, sell, metrics = lookup[(winner["buy_id"], winner["sell_id"])]
    return (
        rolling.expand_candidate(buy, full_buy_conditions),
        rolling.expand_candidate(sell, full_sell_conditions),
        metrics,
        ranking,
    )


def annual_test_rows(buy: dca.Candidate, sell: dca.Candidate, full: pd.DataFrame, groups: dict) -> pd.DataFrame:
    max_year = int(pd.to_datetime(full["date"]).dt.year.max())
    rows = []
    for year in range(DISCOVERY_CUTOFF.year, max_year + 1):
        result = dca.pair_result(
            buy,
            sell,
            full,
            groups,
            entry_start=pd.Timestamp(year=year, month=1, day=1),
            entry_end=pd.Timestamp(year=year + 1, month=1, day=1),
        )
        rows.append(
            {
                "test_year": year,
                "buy_batches": result["buy_batches"],
                "closed_batches": result["closed_batches"],
                "win_batches": result["win_batches"],
                "win_rate": result["win_rate"],
                "avg_return": result["avg_return"],
                "return_sum": result["return_sum"],
                "closed_rate": result["closed_rate"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, groups = dca.load_panel()
    buy_conditions, buy_labels, buy_cores = dca.build_conditions(full, "buy")
    sell_conditions, sell_labels, sell_cores = dca.build_conditions(full, "sell")
    train, indices = rolling.training_panel(full, DISCOVERY_CUTOFF)
    buy, sell, discovery, discovery_ranking = select_frozen_pair(
        train,
        buy_conditions,
        sell_conditions,
        indices,
        buy_labels,
        buy_cores,
        sell_labels,
        sell_cores,
    )
    post = dca.pair_result(buy, sell, full, groups, entry_start=DISCOVERY_CUTOFF)
    annual = annual_test_rows(buy, sell, full, groups)
    discovery_ranking.head(100).to_csv(OUT / "frozen_discovery_ranking.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "frozen_oos_years.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Frozen split OOS. Formula is selected once from pre-2017 data and never changed afterward.",
        "discovery_period": f"{pd.to_datetime(train['date']).min().date()} to 2016-12-31",
        "test_period": f"2017-01-01 to {pd.to_datetime(full['date']).max().date()}",
        "discovery_min_closed_batches": MIN_DISCOVERY_CLOSED,
        "discovery_min_closed_rate": MIN_DISCOVERY_CLOSED_RATE,
        "buy_combo": buy.label,
        "sell_combo": sell.label,
        "discovery_closed_batches": discovery["closed_batches"],
        "discovery_win_rate": discovery["win_rate"],
        "discovery_avg_return": discovery["avg_return"],
        "test_buy_batches": post["buy_batches"],
        "test_closed_batches": post["closed_batches"],
        "test_win_batches": post["win_batches"],
        "test_win_rate": post["win_rate"],
        "test_avg_return": post["avg_return"],
        "test_pnl_100_cny": post["total_pnl_100_per_batch"],
        "minimum_hold_days": dca.MIN_HOLD_DAYS,
        "execution": "T+1 close; each independent buy is 100 CNY; no fee; no capital cap.",
    }
    (OUT / "frozen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    display = annual.copy()
    for column in ("win_rate", "avg_return", "closed_rate"):
        display[column] = display[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    report = [
        "# D 线：固定公式分割样本外验证",
        "",
        "公式只在 2017 年以前的数据上选择一次；2017 年起不再改因子、阈值、买卖组合或选择器。后续上市 ETF 在自身累积足够指标历史后可使用同一条冻结公式。",
        "",
        "## 冻结公式",
        f"- 买入：{buy.label}",
        f"- 卖出：{sell.label}",
        f"- 发现期：{summary['discovery_period']}；已完成批次 {summary['discovery_closed_batches']}，胜率 {summary['discovery_win_rate']:.2%}，平均收益 {summary['discovery_avg_return']:.2%}。",
        "",
        "## 2017 年后固定公式结果",
        f"- 买入批次：{summary['test_buy_batches']}；已完成批次：{summary['test_closed_batches']}；盈利批次：{summary['test_win_batches']}。",
        f"- 胜率：{summary['test_win_rate']:.2%}；平均单批收益：{summary['test_avg_return']:.2%}。",
        f"- 每批 100 元、仅已完成批次的收益合计：{summary['test_pnl_100_cny']:.2f} 元。",
        "",
        "## 年度明细",
        display.to_markdown(index=False),
        "",
        "注意：此测试避免了 2017 年之后用未来数据挑公式，但发现期仍是从大量候选中筛出的历史最佳，结果应作为固定策略候选，而非直接视为实盘承诺。",
    ]
    (OUT / "frozen_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
