from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "D" / "site" / "data" / "dashboard.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def annual_rows(candidate_path: Path, baseline_path: Path, value_key: str) -> list[dict[str, Any]]:
    candidate = {int(row["year"]): finite(row.get(value_key)) for row in read_csv(candidate_path)}
    baseline = {int(row["year"]): finite(row.get(value_key)) for row in read_csv(baseline_path)}
    return [
        {
            "year": year,
            "candidate": candidate.get(year, 0.0),
            "baseline": baseline.get(year, 0.0),
            "delta": candidate.get(year, 0.0) - baseline.get(year, 0.0),
        }
        for year in sorted(candidate.keys() | baseline.keys())
    ]


def comparison_index() -> dict[str, dict[str, str]]:
    rows = read_csv(ROOT / "C" / "fit" / "clean_upgrades" / "comparison.csv")
    return {row["line"]: row for row in rows}


def metric_block(row: dict[str, str], prefix: str) -> dict[str, float]:
    return {
        "final_value": finite(row.get(f"{prefix}_final_cny")),
        "win_rate": finite(row.get(f"{prefix}_win_rate")),
        "avg_annual_return": finite(row.get(f"{prefix}_avg_annual_return")),
        "max_drawdown": finite(row.get(f"{prefix}_max_drawdown")),
    }


def clean_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text and "�" not in text else fallback


def normalize_trades(line: str, path: Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(read_csv(path), start=1):
        entry_date = clean_text(row.get("entry_date"))
        if line in {"C", "S"}:
            exit_date = clean_text(row.get("new_exit_date")) or clean_text(row.get("exit_date"))
            trade_return = finite(row.get("new_ret"), finite(row.get("ret")))
            name = clean_text(row.get("display_name")) or clean_text(row.get("name"))
            source = clean_text(row.get("source_type")) or clean_text(row.get("source"), "主策略")
            status = "closed" if exit_date else "open"
        else:
            exit_date = clean_text(row.get("actual_exit_date"))
            trade_return = finite(row.get("net_return"))
            name = clean_text(row.get("name"))
            source = "D线策略"
            status = "closed" if exit_date else "open"
        symbol = clean_text(row.get("symbol"))
        if not entry_date or not symbol:
            continue
        normalized.append(
            {
                "id": f"{line}{index:03d}",
                "symbol": symbol,
                "name": name or symbol,
                "source": source,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "return": trade_return,
                "status": status,
            }
        )
    return normalized


def price_path(symbol: str, raw_dirs: list[Path]) -> Path | None:
    for directory in raw_dirs:
        candidate = directory / f"{symbol}.csv"
        if candidate.exists():
            return candidate
    return None


def build_trade_views(trades: list[dict[str, Any]], raw_dirs: list[Path]) -> dict[str, Any]:
    event_map: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    names: dict[str, str] = {}
    for trade in trades:
        symbol = trade["symbol"]
        names[symbol] = trade["name"]
        entry_year = int(trade["entry_date"][:4])
        event_map[(entry_year, symbol)].append(
            {
                "type": "buy",
                "date": trade["entry_date"],
                "trade_id": trade["id"],
                "source": trade["source"],
                "entry_date": trade["entry_date"],
                "exit_date": trade["exit_date"],
                "return": trade["return"],
                "status": trade["status"],
            }
        )
        if trade["exit_date"]:
            exit_year = int(trade["exit_date"][:4])
            event_map[(exit_year, symbol)].append(
                {
                    "type": "sell",
                    "date": trade["exit_date"],
                    "trade_id": trade["id"],
                    "source": trade["source"],
                    "entry_date": trade["entry_date"],
                    "exit_date": trade["exit_date"],
                    "return": trade["return"],
                    "status": trade["status"],
                }
            )

    price_cache: dict[str, list[dict[str, str]]] = {}
    views: dict[str, dict[str, Any]] = defaultdict(dict)
    missing: list[str] = []
    for (year, symbol), events in sorted(event_map.items()):
        if symbol not in price_cache:
            path = price_path(symbol, raw_dirs)
            if path is None:
                missing.append(symbol)
                price_cache[symbol] = []
            else:
                price_cache[symbol] = read_csv(path)
        year_prices = []
        close_by_date: dict[str, float] = {}
        for row in price_cache[symbol]:
            date = clean_text(row.get("date"))
            if not date.startswith(str(year)):
                continue
            close = finite(row.get("close"), math.nan)
            if not math.isfinite(close):
                continue
            year_prices.append([date, close])
            close_by_date[date] = close
            row_name = clean_text(row.get("name"))
            if row_name:
                names[symbol] = row_name
        if not year_prices:
            continue
        for event in events:
            event["price"] = close_by_date.get(event["date"])
        views[str(year)][symbol] = {
            "symbol": symbol,
            "name": names.get(symbol, symbol),
            "prices": year_prices,
            "events": events,
            "buy_count": sum(event["type"] == "buy" for event in events),
            "sell_count": sum(event["type"] == "sell" for event in events),
        }
    return {
        "years": sorted((int(year) for year in views), reverse=True),
        "by_year": dict(views),
        "missing_price_symbols": sorted(set(missing)),
    }


def build_payload() -> dict[str, Any]:
    comparisons = comparison_index()
    cs_summary = read_json(ROOT / "C" / "fit" / "clean_upgrades" / "summary.json")
    cs_results = {item["line"]: item for item in cs_summary.get("results", [])}
    d_summary = read_json(
        ROOT / "D" / "out" / "clean_control_sweep" / "summary_strong_exit_d3_pos.json"
    )

    common_execution = "收盘后确认状态，下一交易日收盘执行；佣金万三、单边最低5元；100股整数手。"
    common_validation = (
        "每个测试年度只使用此前已完成年度的数据选择参数。2024-2026属于滚动样本外，"
        "但经过多轮研究后已不再是未接触的盲测区间。"
    )

    specs = {
        "C": {
            "title": "C线稳健升级候选",
            "subtitle": "主策略与空仓补偿的单仓滚动执行线",
            "period": "2014-2026",
            "as_of": "2026-07-31",
            "color": "#16836b",
            "quality": "30/30只ETF通过数据清洁门槛",
            "method": "历史证据门槛 + 强趋势延后卖出",
            "parameter": "历史证据下限32；从2023年起按年度滚动启用",
            "stability": "相邻5个证据门槛均改善",
            "annual_candidate": ROOT / "C" / "fit" / "clean_upgrades" / "c_annual_net.csv",
            "annual_baseline": ROOT / "C" / "fit" / "clean_upgrades" / "c_baseline_annual_net.csv",
            "annual_key": "net_account_return",
            "trades": int(cs_results.get("C", {}).get("winner_full", {}).get("trades", 0)),
            "open_positions": 0,
            "positive_year_rate": finite(
                cs_results.get("C", {}).get("winner_full", {}).get("positive_year_rate")
            ),
            "trade_path": ROOT / "C" / "fit" / "clean_upgrades" / "c_trades.csv",
            "raw_dirs": [ROOT / "C" / "raw" / "etf"],
        },
        "S": {
            "title": "S线扩展池升级候选",
            "subtitle": "扩展ETF池的影子研究与跨资产轮动线",
            "period": "2014-2026",
            "as_of": "2026-07-31",
            "color": "#2d6cdf",
            "quality": "61/62只ETF通过数据清洁门槛，异常标的已剔除",
            "method": "历史证据门槛 + 强趋势延后卖出",
            "parameter": "历史证据下限5；从2022年起按年度滚动启用",
            "stability": "相邻7个证据门槛均改善",
            "annual_candidate": ROOT / "C" / "fit" / "clean_upgrades" / "s_annual_net.csv",
            "annual_baseline": ROOT / "C" / "fit" / "clean_upgrades" / "s_baseline_annual_net.csv",
            "annual_key": "net_account_return",
            "trades": int(cs_results.get("S", {}).get("winner_full", {}).get("trades", 0)),
            "open_positions": 0,
            "positive_year_rate": finite(
                cs_results.get("S", {}).get("winner_full", {}).get("positive_year_rate")
            ),
            "trade_path": ROOT / "C" / "fit" / "clean_upgrades" / "s_trades.csv",
            "raw_dirs": [ROOT / "S" / "raw" / "etf", ROOT / "C" / "raw" / "etf"],
        },
        "D": {
            "title": "D线高胜率升级候选",
            "subtitle": "恐惧强度排序与连续信号的单仓验证线",
            "period": "2019-2026",
            "as_of": str(d_summary.get("winner_full", {}).get("as_of", "2026-07-31")),
            "color": "#d48816",
            "quality": "沿用C线30只清洁ETF池",
            "method": "强趋势延后卖出 + 年度滚动选择",
            "parameter": "强趋势近20日涨幅至少2%；卖出最多延后3个交易日",
            "stability": "相邻3个延后窗口均改善",
            "annual_candidate": ROOT / "D" / "out" / "clean_control_sweep" / "strong_exit_d3_pos_annual.csv",
            "annual_baseline": ROOT / "D" / "out" / "formal_clean" / "annual.csv",
            "annual_key": "compounded_net_return",
            "trades": int(d_summary.get("winner_full", {}).get("closed_trades", 0)),
            "open_positions": int(d_summary.get("winner_full", {}).get("open_positions", 0)),
            "positive_year_rate": finite(d_summary.get("winner_full", {}).get("positive_year_rate")),
            "trade_path": ROOT / "D" / "out" / "clean_control_sweep" / "strong_exit_d3_pos_trades.csv",
            "raw_dirs": [ROOT / "C" / "raw" / "etf"],
        },
    }

    versions: dict[str, Any] = {}
    for key, spec in specs.items():
        row = comparisons.get(key, {})
        candidate = metric_block(row, "winner")
        baseline = metric_block(row, "baseline")
        trades = normalize_trades(key, spec["trade_path"])
        versions[key] = {
            "key": key,
            "title": spec["title"],
            "subtitle": spec["subtitle"],
            "status": "影子候选",
            "period": spec["period"],
            "as_of": spec["as_of"],
            "color": spec["color"],
            "quality": spec["quality"],
            "method": spec["method"],
            "parameter": spec["parameter"],
            "stability": spec["stability"],
            "execution": common_execution,
            "validation": common_validation,
            "candidate": {
                **candidate,
                "trades": spec["trades"],
                "open_positions": spec["open_positions"],
                "positive_year_rate": spec["positive_year_rate"],
            },
            "baseline": baseline,
            "recent": {
                "period": "2024-2026",
                "baseline_return": finite(row.get("holdout_baseline_return")),
                "candidate_return": finite(row.get("holdout_winner_return")),
                "baseline_win_rate": finite(row.get("holdout_baseline_win_rate")),
                "candidate_win_rate": finite(row.get("holdout_winner_win_rate")),
            },
            "annual": annual_rows(
                spec["annual_candidate"], spec["annual_baseline"], spec["annual_key"]
            ),
            "trade_views": build_trade_views(trades, spec["raw_dirs"]),
            "audits": [
                "仅使用通过清洁检查的ETF",
                "成交价格可回查至原始日线",
                "无同一账户重叠持仓",
                "含真实佣金与100股整数手约束",
            ],
        }

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": "C / S / D 三线策略候选看板",
        "notice": (
            "三条升级策略均通过当前历史数据审计，但仍属于影子候选，尚未自动替换正式实盘配置。"
        ),
        "versions": versions,
    }


def main() -> None:
    payload = build_payload()
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATA_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Dashboard data written: {DATA_FILE}")


if __name__ == "__main__":
    main()
