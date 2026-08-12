from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
S_SRC = WORKSPACE / "S" / "src"
if str(S_SRC) not in sys.path:
    sys.path.insert(0, str(S_SRC))

from shadow_model import ShadowResearchEngine  # noqa: E402


OUT = ROOT / "fit" / "pullback_recovery_test"
BASELINE_PATH = ROOT / "fit" / "fallback" / "final_trades.csv"
DEV_YEARS = set(range(2016, 2024))
HOLDOUT_YEARS = {2024, 2025, 2026}
TEST_YEARS = range(2016, 2027)


@dataclass(frozen=True)
class Param:
    idle: int
    hold: int
    ma: int
    mom_min: float
    lookback: int
    pullback_min: float
    pullback_max: float
    reversal: str
    vol_cap: float
    model: str

    @property
    def condition_key(self) -> str:
        return (
            f"i{self.idle}_ma{self.ma}_m{int(self.mom_min * 100)}_"
            f"lb{self.lookback}_pb{int(self.pullback_min * 100)}-"
            f"{int(self.pullback_max * 100)}_{self.reversal}_"
            f"v{int(self.vol_cap * 100)}_{self.model}"
        )

    @property
    def key(self) -> str:
        return f"{self.condition_key}_h{self.hold}"


def load_baseline() -> pd.DataFrame:
    frame = pd.read_csv(
        BASELINE_PATH,
        dtype={"symbol": str},
        encoding="utf-8-sig",
    )
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    for column in ("decision_date", "entry_date", "exit_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["priority"] = 1000.0
    frame["family"] = "BASE"
    frame["param_id"] = "BASE"
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def prepare_engine(baseline: pd.DataFrame) -> ShadowResearchEngine:
    engine = ShadowResearchEngine()
    engine.main = baseline.copy()
    engine.idle_windows = engine._build_idle_windows()
    return engine


def build_panel(engine: ShadowResearchEngine) -> pd.DataFrame:
    rows = []
    for symbol in engine.scope_map["all_clean"]:
        frame = engine.panel[symbol].copy()
        close = pd.to_numeric(frame["close"], errors="coerce")
        ret = close.pct_change()
        frame["ret1"] = ret
        frame["ret3"] = close.pct_change(3)
        frame["ma5"] = close.rolling(5, min_periods=5).mean()
        frame["high20"] = close.rolling(20, min_periods=15).max()
        frame["high40"] = close.rolling(40, min_periods=25).max()
        frame["dd20"] = close / frame["high20"] - 1.0
        frame["dd40"] = close / frame["high40"] - 1.0
        frame["symbol"] = symbol
        frame["display_name"] = engine.meta[symbol].get(
            "display_name", symbol
        )
        frame["category"] = engine.meta[symbol].get("category", "")
        rows.append(
            frame[
                [
                    "date",
                    "symbol",
                    "display_name",
                    "category",
                    "close",
                    "ma5",
                    "ma60",
                    "ma120",
                    "mom20",
                    "mom60",
                    "mom120",
                    "ret1",
                    "ret3",
                    "dd20",
                    "dd40",
                    "vol20",
                    "amount20",
                    "history",
                ]
            ]
        )
    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    numeric = [
        column
        for column in panel.columns
        if column not in {"date", "symbol", "display_name", "category"}
    ]
    panel[numeric] = panel[numeric].apply(pd.to_numeric, errors="coerce")
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel["date_code"], _ = pd.factorize(panel["date"], sort=True)
    trend = panel["mom60"].clip(-0.25, 0.75)
    long_trend = panel["mom120"].clip(-0.40, 1.20)
    recovery = panel["ret3"].clip(-0.10, 0.15)
    risk = panel["vol20"].clip(0.05, 1.20)
    for lookback in (20, 40):
        depth = (-panel[f"dd{lookback}"]).clip(0, 0.30)
        panel[f"score_trend_recovery_{lookback}"] = (
            1.00 * trend
            + 0.50 * long_trend
            + 1.50 * recovery
            + 0.35 * depth
            - 0.15 * risk
        )
        panel[f"score_deep_quality_{lookback}"] = (
            0.60 * trend
            + 0.80 * long_trend
            + 0.80 * recovery
            + 1.20 * depth
            - 0.30 * risk
        )
    return panel


def best_daily(panel: pd.DataFrame, param: Param) -> pd.DataFrame:
    ma = panel[f"ma{param.ma}"]
    drawdown = panel[f"dd{param.lookback}"]
    mask = (
        panel["close"].gt(ma)
        & panel["mom60"].ge(param.mom_min)
        & drawdown.le(-param.pullback_min)
        & drawdown.ge(-param.pullback_max)
        & panel["vol20"].le(param.vol_cap)
        & panel["amount20"].ge(2_000_000)
        & panel["history"].ge(250)
    )
    if param.reversal == "ret1":
        mask &= panel["ret1"].gt(0)
    elif param.reversal == "ret3":
        mask &= panel["ret3"].gt(0)
    else:
        raise ValueError(param.reversal)
    score_column = f"score_{param.model}_{param.lookback}"
    scores = panel[score_column].to_numpy(dtype=float, copy=False)
    codes = panel["date_code"].to_numpy(dtype=int, copy=False)
    valid = mask.to_numpy(dtype=bool, copy=False) & np.isfinite(scores)
    if not valid.any():
        return panel.iloc[0:0].copy()
    maxima = np.full(int(codes.max()) + 1, -np.inf)
    np.maximum.at(maxima, codes, np.where(valid, scores, -np.inf))
    winners = np.flatnonzero(
        valid & np.isclose(scores, maxima[codes], rtol=0.0, atol=1e-14)
    )
    winner_codes = codes[winners]
    first = winners[
        np.r_[True, winner_codes[1:] != winner_codes[:-1]]
    ]
    result = panel.iloc[first].copy()
    result["rank_score"] = result[score_column]
    return result.reset_index(drop=True)


def materialize(
    engine: ShadowResearchEngine,
    best: pd.DataFrame,
    param: Param,
) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    best_indexed = best.set_index("date", drop=False)
    by_date = best_indexed.to_dict("index")
    rows = []
    for start, end, _ in engine.idle_windows:
        calendar = engine.calendar[
            engine.calendar.between(start, end)
        ].reset_index(drop=True)
        calendar_values = calendar.to_numpy()
        if len(calendar) <= param.idle:
            continue
        pos = param.idle - 1
        while pos < len(calendar) - 1:
            decision = pd.Timestamp(calendar_values[pos])
            candidate = by_date.get(decision)
            if candidate is None:
                pos += 1
                continue
            symbol = str(candidate["symbol"])
            prices = engine.panel[symbol]
            entry_pos = engine.pos_after(
                prices, decision + pd.Timedelta(days=1)
            )
            end_pos = engine.pos_before(prices, end)
            if (
                entry_pos is None
                or end_pos is None
                or entry_pos >= end_pos
            ):
                pos += 1
                continue
            exit_pos = min(entry_pos + param.hold, end_pos)
            if exit_pos <= entry_pos:
                pos += 1
                continue
            entry = prices.iloc[entry_pos]
            exit_ = prices.iloc[exit_pos]
            rows.append(
                {
                    "source": "强趋势回撤修复",
                    "source_type": "空仓补偿",
                    "family": "PULLBACK_RECOVERY",
                    "symbol": symbol,
                    "display_name": candidate["display_name"],
                    "category": candidate["category"],
                    "decision_date": decision,
                    "entry_date": pd.Timestamp(entry["date"]),
                    "exit_date": pd.Timestamp(exit_["date"]),
                    "entry_close": float(entry["close"]),
                    "exit_close": float(exit_["close"]),
                    "ret": float(exit_["close"] / entry["close"] - 1.0),
                    "priority": 200.0,
                    "param_id": param.key,
                    "rank_score": float(candidate["rank_score"]),
                    "mom60": float(candidate["mom60"]),
                    "mom120": float(candidate["mom120"]),
                    "pullback": float(candidate[f"dd{param.lookback}"]),
                    "ret3": float(candidate["ret3"]),
                    "vol20": float(candidate["vol20"]),
                }
            )
            next_date = engine.next_date(
                pd.Timestamp(exit_["date"]) + pd.Timedelta(days=1)
            )
            if next_date is None or next_date > end:
                break
            next_pos = int(
                np.searchsorted(
                    calendar_values,
                    np.datetime64(next_date),
                    side="left",
                )
            )
            if next_pos >= len(calendar_values):
                break
            pos = next_pos
    return pd.DataFrame(rows)


def account(baseline: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [baseline, extras], ignore_index=True, sort=False
    ).sort_values(
        ["entry_date", "priority", "symbol"],
        ascending=[True, False, True],
    )
    rows = []
    busy = pd.Timestamp.min
    for _, trade in combined.iterrows():
        entry = pd.Timestamp(trade["entry_date"])
        if entry <= busy:
            continue
        rows.append(trade)
        busy = pd.Timestamp(trade["exit_date"])
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"])


def metrics(frame: pd.DataFrame, years: set[int]) -> dict[str, float | int]:
    local = frame[
        pd.to_datetime(frame["entry_date"]).dt.year.isin(years)
    ].copy()
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    if returns.empty:
        return {
            "trades": 0,
            "final_1000": 1000.0,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
            "losing_years": 0,
        }
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual = (
        local.assign(
            year=pd.to_datetime(local["entry_date"]).dt.year,
            ret_num=pd.to_numeric(local["ret"], errors="coerce"),
        )
        .groupby("year")["ret_num"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    )
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]),
        "win_rate": float((returns > 0).mean()),
        "max_drawdown": float(drawdown.min()),
        "losing_years": int((annual < 0).sum()),
    }


def score(stat: dict[str, float | int], kind: str) -> float:
    profit = math.log(max(float(stat["final_1000"]), 1.0) / 1000.0)
    win = float(np.nan_to_num(stat["win_rate"], nan=0.0))
    drawdown = float(stat["max_drawdown"])
    losing = int(stat["losing_years"])
    if kind == "profit":
        return profit
    if kind == "balanced":
        return profit + 0.25 * win + 0.70 * drawdown - 0.03 * losing
    return profit + 0.40 * win + 1.20 * drawdown - 0.06 * losing


def annual_summary(frame: pd.DataFrame, param_id: str) -> pd.DataFrame:
    rows = []
    local = frame.assign(year=pd.to_datetime(frame["entry_date"]).dt.year)
    for year, group in local.groupby("year"):
        returns = pd.to_numeric(group["ret"], errors="coerce").dropna()
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rows.append(
            {
                "param_id": param_id,
                "year": int(year),
                "trades": int(len(returns)),
                "wins": int((returns > 0).sum()),
                "annual_return": float(equity.iloc[-1] - 1.0),
                "max_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def rolling_method(
    logs: dict[str, pd.DataFrame],
    annual: pd.DataFrame,
    window: int | None,
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = []
    pieces = []
    for year in TEST_YEARS:
        start = 2014 if window is None else max(2014, year - window)
        train = annual[annual["year"].between(start, year - 1)].copy()
        grouped = (
            train.groupby("param_id")
            .agg(
                covered_years=("year", "nunique"),
                trades=("trades", "sum"),
                wins=("wins", "sum"),
                log_growth=(
                    "annual_return",
                    lambda values: float(np.log1p(values).sum()),
                ),
                max_drawdown=("max_drawdown", "min"),
                losing_years=(
                    "annual_return",
                    lambda values: int((values < 0).sum()),
                ),
            )
            .reset_index()
        )
        required = year - start
        grouped = grouped[
            (grouped["covered_years"] >= required)
            & (grouped["trades"] >= max(3, required))
        ].copy()
        if grouped.empty:
            selected = "BASE"
            selected_score = np.nan
        else:
            grouped["win_rate"] = grouped["wins"] / grouped["trades"]
            if kind == "profit":
                grouped["selector_score"] = grouped["log_growth"]
            elif kind == "balanced":
                grouped["selector_score"] = (
                    grouped["log_growth"]
                    + 0.25 * grouped["win_rate"]
                    + 0.70 * grouped["max_drawdown"]
                    - 0.03 * grouped["losing_years"]
                )
            else:
                grouped["selector_score"] = (
                    grouped["log_growth"]
                    + 0.40 * grouped["win_rate"]
                    + 1.20 * grouped["max_drawdown"]
                    - 0.06 * grouped["losing_years"]
                )
            picked = grouped.sort_values(
                ["selector_score", "log_growth", "win_rate", "param_id"],
                ascending=[False, False, False, True],
            ).iloc[0]
            selected = str(picked["param_id"])
            selected_score = float(picked["selector_score"])
        chosen = logs[selected]
        pieces.append(
            chosen[
                pd.to_datetime(chosen["entry_date"]).dt.year.eq(year)
            ].copy()
        )
        choices.append(
            {
                "test_year": year,
                "train_start": start,
                "train_end": year - 1,
                "window": "all" if window is None else window,
                "score_kind": kind,
                "param_id": selected,
                "train_score": selected_score,
            }
        )
    pre = logs["BASE"]
    result = pd.concat(
        [
            pre[pd.to_datetime(pre["entry_date"]).dt.year.lt(2016)],
            *pieces,
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["entry_date", "symbol"])
    return pd.DataFrame(choices), result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline()
    engine = prepare_engine(baseline)
    panel = build_panel(engine)
    logs: dict[str, pd.DataFrame] = {"BASE": baseline}
    annual_parts = [annual_summary(baseline, "BASE")]
    fixed_rows = []
    conditions = list(
        product(
            (5, 10, 20),
            (60, 120),
            (0.0, 0.05),
            (20, 40),
            (0.02, 0.04),
            (0.10, 0.15),
            ("ret1", "ret3"),
            (0.50, 0.70),
            ("trend_recovery", "deep_quality"),
        )
    )
    total = len(conditions) * 4
    done = 0
    for condition in conditions:
        template = Param(
            idle=condition[0],
            hold=5,
            ma=condition[1],
            mom_min=condition[2],
            lookback=condition[3],
            pullback_min=condition[4],
            pullback_max=condition[5],
            reversal=condition[6],
            vol_cap=condition[7],
            model=condition[8],
        )
        best = best_daily(panel, template)
        for hold in (5, 10, 20, 40):
            param = Param(
                idle=template.idle,
                hold=hold,
                ma=template.ma,
                mom_min=template.mom_min,
                lookback=template.lookback,
                pullback_min=template.pullback_min,
                pullback_max=template.pullback_max,
                reversal=template.reversal,
                vol_cap=template.vol_cap,
                model=template.model,
            )
            extras = materialize(engine, best, param)
            log = account(baseline, extras)
            logs[param.key] = log
            annual_parts.append(annual_summary(log, param.key))
            dev = metrics(log, DEV_YEARS)
            holdout = metrics(log, HOLDOUT_YEARS)
            fixed_rows.append(
                {
                    "param_id": param.key,
                    **param.__dict__,
                    **{f"dev_{key}": value for key, value in dev.items()},
                    **{
                        f"holdout_{key}": value
                        for key, value in holdout.items()
                    },
                }
            )
            done += 1
            if done % 400 == 0:
                print(f"[pullback {done}/{total}]", flush=True)

    annual = pd.concat(annual_parts, ignore_index=True)
    methods = []
    choices_map: dict[str, pd.DataFrame] = {}
    log_map: dict[str, pd.DataFrame] = {}
    for window in (3, 5, 7, None):
        for kind in ("profit", "balanced", "stable"):
            method = f"w{window if window is not None else 'all'}_{kind}"
            choices, log = rolling_method(logs, annual, window, kind)
            choices_map[method] = choices
            log_map[method] = log
            dev = metrics(log, DEV_YEARS)
            holdout = metrics(log, HOLDOUT_YEARS)
            methods.append(
                {
                    "method": method,
                    "dev_score": score(dev, kind),
                    **{f"dev_{key}": value for key, value in dev.items()},
                    **{
                        f"holdout_{key}": value
                        for key, value in holdout.items()
                    },
                }
            )
    method_table = pd.DataFrame(methods).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )
    winner = method_table.iloc[0]
    winner_id = str(winner["method"])
    baseline_holdout = metrics(baseline, HOLDOUT_YEARS)
    passing = (
        (
            method_table["holdout_final_1000"]
            >= float(baseline_holdout["final_1000"]) * 1.05
        )
        & (
            method_table["holdout_win_rate"]
            >= float(baseline_holdout["win_rate"]) - 0.02
        )
        & (
            method_table["holdout_max_drawdown"]
            >= float(baseline_holdout["max_drawdown"]) - 0.02
        )
        & (method_table["holdout_trades"] >= 10)
    )
    neighbors = int(passing.sum())
    promoted = bool(passing.loc[winner.name] and neighbors >= 2)

    pd.DataFrame(fixed_rows).to_csv(
        OUT / "fixed_grid.csv", index=False, encoding="utf-8-sig"
    )
    method_table.to_csv(
        OUT / "rolling_methods.csv", index=False, encoding="utf-8-sig"
    )
    choices_map[winner_id].to_csv(
        OUT / "winner_choices.csv", index=False, encoding="utf-8-sig"
    )
    log_map[winner_id].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "single-account strong-trend pullback recovery",
        "candidate_params": total,
        "method_variants": len(method_table),
        "baseline_holdout": baseline_holdout,
        "development_selected_method": winner_id,
        "winner": winner.to_dict(),
        "holdout_neighbors_passing": neighbors,
        "promoted": promoted,
        "formal_config_changed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
