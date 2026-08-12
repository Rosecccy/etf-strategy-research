from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "novel_entry_gates"
MIN_TRAIN_TRADES = 12
MIN_KEEP_TRADES = 8
QUANTILES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
FEATURES = (
    "ret1",
    "ret3",
    "ret5",
    "ret10",
    "ret20",
    "ret60",
    "vol20",
    "range20",
    "drawdown20",
    "drawdown60",
    "dist_ma20",
    "dist_ma60",
    "ma_gap",
    "volume_ratio",
    "down_streak",
)


@dataclass(frozen=True)
class Gate:
    feature: str
    direction: str
    quantile: float

    @property
    def key(self) -> str:
        return f"{self.feature}:{self.direction}:q{self.quantile:.2f}"


def price_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    close = frame["close"].astype(float)
    returns = close.pct_change()
    for days in (1, 3, 5, 10, 20, 60):
        frame[f"ret{days}"] = close.pct_change(days)
    frame["vol20"] = returns.rolling(20, min_periods=15).std()
    frame["range20"] = (frame["high"] / frame["low"] - 1.0).rolling(20, min_periods=15).mean()
    frame["drawdown20"] = close / close.rolling(20, min_periods=15).max() - 1.0
    frame["drawdown60"] = close / close.rolling(60, min_periods=40).max() - 1.0
    ma20 = close.rolling(20, min_periods=15).mean()
    ma60 = close.rolling(60, min_periods=40).mean()
    frame["dist_ma20"] = close / ma20 - 1.0
    frame["dist_ma60"] = close / ma60 - 1.0
    frame["ma_gap"] = ma20 / ma60 - 1.0
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    frame["volume_ratio"] = volume.rolling(5, min_periods=3).mean() / volume.rolling(20, min_periods=15).mean()
    streak = []
    count = 0
    for value in returns.fillna(0.0):
        count = count + 1 if value < 0 else 0
        streak.append(count)
    frame["down_streak"] = streak
    return frame


def enrich(spec: core.LineSpec, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    prices = core.load_prices(spec.project, set(result["symbol"]))
    feature_rows = []
    for _, trade in result.iterrows():
        data = price_features(prices[str(trade["symbol"])])
        entry = pd.Timestamp(trade["entry_date"])
        prior = data[data["date"].lt(entry)]
        values = {feature: np.nan for feature in FEATURES}
        if len(prior):
            last = prior.iloc[-1]
            values.update({feature: float(last[feature]) for feature in FEATURES})
        feature_rows.append(values)
    features = pd.DataFrame(feature_rows, index=result.index)
    for feature in FEATURES:
        result[f"entry_{feature}"] = features[feature]
    return result


def apply_gate(frame: pd.DataFrame, gate: Gate, threshold: float) -> pd.DataFrame:
    values = pd.to_numeric(frame[f"entry_{gate.feature}"], errors="coerce")
    if gate.direction == "keep_high":
        keep = values.ge(threshold)
    else:
        keep = values.le(threshold)
    return frame[keep.fillna(False)].copy()


def candidate_rows(train: pd.DataFrame) -> list[dict]:
    base_stats, _, _ = core.simulate_account(train)
    rows = []
    for feature in FEATURES:
        values = pd.to_numeric(train[f"entry_{feature}"], errors="coerce").dropna()
        if len(values) < MIN_TRAIN_TRADES:
            continue
        for direction in ("keep_high", "keep_low"):
            for quantile in QUANTILES:
                threshold_quantile = quantile if direction == "keep_high" else 1.0 - quantile
                threshold = float(values.quantile(threshold_quantile))
                gate = Gate(feature, direction, quantile)
                kept = apply_gate(train, gate, threshold)
                if len(kept) < MIN_KEEP_TRADES or len(kept) >= len(train):
                    continue
                current, _, _ = core.simulate_account(kept)
                ratio = current["final_value"] / base_stats["final_value"]
                win_delta = current["win_rate"] - base_stats["win_rate"]
                dd_delta = current["max_drawdown"] - base_stats["max_drawdown"]
                passes = bool(ratio > 1.0 and win_delta > 0 and dd_delta >= -0.005)
                rows.append(
                    {
                        "gate": gate,
                        "rule": gate.key,
                        "feature": feature,
                        "direction": direction,
                        "quantile": quantile,
                        "threshold": threshold,
                        "kept": len(kept),
                        "ratio": ratio,
                        "win_delta": win_delta,
                        "dd_delta": dd_delta,
                        "passes": passes,
                        "score": np.log(max(ratio, 1e-12)) + 1.25 * win_delta + 0.20 * min(dd_delta, 0.20),
                    }
                )
    passed_keys = {
        (row["feature"], row["direction"], row["quantile"])
        for row in rows
        if row["passes"]
    }
    for row in rows:
        neighbors = (
            (row["feature"], row["direction"], round(row["quantile"] - 0.05, 2)),
            (row["feature"], row["direction"], round(row["quantile"] + 0.05, 2)),
        )
        row["stable"] = bool(row["passes"] and any(key in passed_keys for key in neighbors))
    return rows


def select_for_year(frame: pd.DataFrame, year: int) -> tuple[Gate | None, float | None, dict]:
    years = pd.to_datetime(frame["entry_date"]).dt.year
    train = frame[years.lt(year)].copy()
    if len(train) < MIN_TRAIN_TRADES:
        return None, None, {"year": year, "rule": "baseline", "train_trades": len(train)}
    rows = candidate_rows(train)
    stable = [row for row in rows if row["stable"]]
    if not stable:
        return None, None, {"year": year, "rule": "baseline", "train_trades": len(train)}
    winner = max(stable, key=lambda row: row["score"])
    return winner["gate"], float(winner["threshold"]), {
        "year": year,
        "rule": winner["rule"],
        "threshold": winner["threshold"],
        "train_trades": len(train),
        "train_kept": winner["kept"],
        "train_ratio": winner["ratio"],
        "train_win_delta": winner["win_delta"],
        "stable": True,
    }


def validate_line(spec: core.LineSpec, path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in ("entry_date", "exit_date_test", "control_signal_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = enrich(spec, frame)
    years = sorted(pd.to_datetime(frame["entry_date"]).dt.year.unique())
    parts = []
    selections = []
    for year in years:
        gate, threshold, selection = select_for_year(frame, int(year))
        local = frame[pd.to_datetime(frame["entry_date"]).dt.year.eq(year)].copy()
        if gate is not None and threshold is not None:
            local = apply_gate(local, gate, threshold)
        local["entry_gate_rule"] = selection["rule"]
        local["entry_gate_threshold"] = threshold
        parts.append(local)
        selections.append(selection)
    combined = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    baseline = {
        period: search.subset_stats(frame, period)
        for period in ("full", "dev", "holdout")
    }
    winner = {
        period: search.subset_stats(combined, period)
        for period in ("full", "dev", "holdout")
    }
    return {
        "line": spec.key,
        "baseline": baseline,
        "winner": winner,
        "full_ratio": winner["full"]["final_value"] / baseline["full"]["final_value"],
        "full_win_delta": winner["full"]["win_rate"] - baseline["full"]["win_rate"],
        "holdout_ratio": winner["holdout"]["final_value"] / baseline["holdout"]["final_value"],
        "holdout_win_delta": winner["holdout"]["win_rate"] - baseline["holdout"]["win_rate"],
        "selected_years": int(sum(item["rule"] != "baseline" for item in selections)),
    }, combined, pd.DataFrame(selections)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "C": ROOT / "fit" / "rolling_upgrades" / "c_trades.csv",
        "S": ROOT / "fit" / "rolling_upgrades" / "s_trades.csv",
        "D": ROOT / "fit" / "rolling_upgrades" / "d_trades.csv",
    }
    results = []
    specs = {spec.key: spec for spec in core.SPECS}
    for line in ("C", "S", "D"):
        payload, trades, selections = validate_line(specs[line], paths[line])
        results.append(payload)
        trades.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        selections.to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window stable quantile entry gate",
        "features_use": "last completed trading day before entry",
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
