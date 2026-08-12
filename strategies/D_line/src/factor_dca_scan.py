from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
C_RAW = PROJECT / "C" / "raw"
CANONICAL_RAW = C_RAW / "etf"
QUALITY_MANIFEST = C_RAW / "quality.json"
OUT = ROOT / "out"
DATA = ROOT / "data"

FREQUENCIES = {"D": "日K", "W": "周K", "B15": "15日K", "M": "月K"}
RANGE_WINDOWS = (60, 90, 120, 180)
LOW_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25)
MIN_SIGNAL_SAMPLES = 100
MIN_HOLD_DAYS = 7
SEED_LIMIT = 80
FINAL_CANDIDATE_LIMIT = 70
MIN_CLOSED_RATE = 0.60

NUMERIC_FACTORS = {
    "KDJ_K": "KDJ-K",
    "KDJ_D": "KDJ-D",
    "KDJ_J": "KDJ-J",
    "KD": "KD线(K-D)",
    "RSI6": "RSI6",
    "DIF": "DIF",
    "EMV": "EMV",
    "LON": "LON",
    "VOL": "成交量",
}
STRUCTURAL_FACTOR = "BOLL"


@dataclass(frozen=True)
class Candidate:
    identifier: str
    label: str
    factors: tuple[str, ...]
    mask: np.ndarray


def tdx_sma(series: pd.Series, period: int, weight: int = 1) -> pd.Series:
    values = series.astype(float).to_numpy()
    result = np.full(len(values), np.nan)
    previous = np.nan
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            result[idx] = previous
            continue
        previous = value if not np.isfinite(previous) else (weight * value + (period - weight) * previous) / period
        result[idx] = previous
    return pd.Series(result, index=series.index)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def aggregate(raw: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "D":
        return raw.copy().reset_index(drop=True)
    data = raw.copy()
    if frequency == "W":
        data["_bucket"] = data["date"].dt.to_period("W-FRI").astype(str)
    elif frequency == "M":
        data["_bucket"] = data["date"].dt.to_period("M").astype(str)
    elif frequency == "B15":
        data["_bucket"] = np.arange(len(data)) // 15
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    grouped = data.groupby("_bucket", sort=True)
    return grouped.agg(
        date=("date", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).reset_index(drop=True)


def indicators(bars: pd.DataFrame) -> pd.DataFrame:
    data = bars.copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float).replace(0, np.nan)

    low9 = low.rolling(9, min_periods=1).min()
    high9 = high.rolling(9, min_periods=1).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k = tdx_sma(rsv, 3, 1)
    d = tdx_sma(k, 3, 1)
    data["KDJ_K"] = k
    data["KDJ_D"] = d
    data["KDJ_J"] = 3 * k - 2 * d
    data["KD"] = k - d

    change = close.diff()
    data["RSI6"] = tdx_sma(change.clip(lower=0), 6, 1) / tdx_sma(change.abs(), 6, 1).replace(0, np.nan) * 100
    data["DIF"] = ema(close, 12) - ema(close, 26)

    spread = (high - low).replace(0, np.nan)
    volume_ratio = volume.rolling(14, min_periods=1).mean() / volume
    mid = 100 * (high + low - (high + low).shift(1)) / (high + low).replace(0, np.nan)
    data["EMV"] = (mid * volume_ratio * spread / spread.rolling(14, min_periods=1).mean()).rolling(14, min_periods=1).mean()
    percent_change = close.pct_change() * 100
    data["LON"] = tdx_sma(percent_change, 10, 1) - tdx_sma(percent_change, 20, 1)
    data["VOL"] = np.log1p(volume.clip(lower=0))

    mid20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    data["BOLL_LOW"] = close <= (mid20 - 2 * std20)
    data["BOLL_HIGH"] = close >= (mid20 + 2 * std20)
    return data


def to_daily_state(
    raw: pd.DataFrame,
    frequency: str,
    activate_on_next_daily: bool = True,
) -> pd.DataFrame:
    bars = indicators(aggregate(raw, frequency))
    daily_dates = raw["date"].reset_index(drop=True)
    state = bars[["date", *NUMERIC_FACTORS.keys(), "BOLL_LOW", "BOLL_HIGH"]].copy()
    if frequency != "D" and activate_on_next_daily:
        # Legacy alignment: expose a completed non-daily bar on the next daily row.
        # The corrected T+1 study instead keeps the period-end date as the signal date,
        # so pair_result executes it on the first following trading day.
        daily_values = daily_dates.to_numpy()
        positions = np.searchsorted(daily_values, state["date"].to_numpy(), side="right")
        state["date"] = [daily_values[pos] if pos < len(daily_values) else pd.NaT for pos in positions]
        state = state.dropna(subset=["date"])
    merged = pd.merge_asof(
        pd.DataFrame({"date": daily_dates}),
        state.sort_values("date"),
        on="date",
        direction="backward",
    )
    return merged


def original_paired_labels(close: pd.Series, window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    values = close.to_numpy(dtype=float)
    lows: list[int] = []
    highs: list[int] = []
    for idx in range(window, len(values) - window):
        before = values[idx - window : idx]
        after = values[idx + 1 : idx + window + 1]
        if np.all(before > values[idx]) and np.all(after > values[idx]):
            lows.append(idx)
        if np.all(before < values[idx]) and np.all(after < values[idx]):
            highs.append(idx)

    events = sorted([(idx, "low") for idx in lows] + [(idx, "high") for idx in highs])
    paired_low: set[int] = set()
    paired_high: set[int] = set()
    active_low: int | None = None
    best_high: int | None = None
    for idx, kind in events:
        if kind == "low":
            if active_low is not None and best_high is not None:
                paired_low.add(active_low)
                paired_high.add(best_high)
            active_low = idx
            best_high = None
        elif active_low is not None:
            if best_high is None or values[idx] > values[best_high]:
                best_high = idx
    if active_low is not None and best_high is not None:
        paired_low.add(active_low)
        paired_high.add(best_high)

    low_label = np.zeros(len(values), dtype=bool)
    high_label = np.zeros(len(values), dtype=bool)
    low_label[list(paired_low)] = True
    high_label[list(paired_high)] = True
    return low_label, high_label


def clustered_paired_labels(
    close: pd.Series,
    window: int = 10,
    tolerance: float = 0.005,
    max_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one label per near-equal double-top or double-bottom cluster."""
    values = close.to_numpy(dtype=float)
    high_candidates: list[int] = []
    low_candidates: list[int] = []
    for idx in range(window, len(values) - window):
        neighbors = np.concatenate((values[idx - window : idx], values[idx + 1 : idx + window + 1]))
        if not np.isfinite(values[idx]) or not np.all(np.isfinite(neighbors)):
            continue
        if values[idx] >= np.max(neighbors) * (1 - tolerance):
            high_candidates.append(idx)
        if values[idx] <= np.min(neighbors) * (1 + tolerance):
            low_candidates.append(idx)

    def merge(indices: list[int], high: bool) -> list[int]:
        clusters: list[list[int]] = []
        for idx in indices:
            if not clusters:
                clusters.append([idx])
                continue
            previous = clusters[-1]
            extreme = max(values[item] for item in previous) if high else min(values[item] for item in previous)
            near_extreme = abs(values[idx] - extreme) / max(abs(extreme), 1e-12) <= tolerance
            if idx - previous[-1] <= max_gap and near_extreme:
                previous.append(idx)
            else:
                clusters.append([idx])
        representatives: list[int] = []
        for cluster in clusters:
            extreme = max(values[item] for item in cluster) if high else min(values[item] for item in cluster)
            representatives.append(next(item for item in cluster if values[item] == extreme))
        return representatives

    lows = merge(low_candidates, high=False)
    highs = merge(high_candidates, high=True)
    events = sorted([(idx, "low") for idx in lows] + [(idx, "high") for idx in highs])
    paired_low: set[int] = set()
    paired_high: set[int] = set()
    active_low: int | None = None
    best_high: int | None = None
    for idx, kind in events:
        if kind == "low":
            if active_low is not None and best_high is not None:
                paired_low.add(active_low)
                paired_high.add(best_high)
            active_low = idx
            best_high = None
        elif active_low is not None:
            if best_high is None or values[idx] > values[best_high]:
                best_high = idx
    if active_low is not None and best_high is not None:
        paired_low.add(active_low)
        paired_high.add(best_high)

    low_label = np.zeros(len(values), dtype=bool)
    high_label = np.zeros(len(values), dtype=bool)
    low_label[list(paired_low)] = True
    high_label[list(paired_high)] = True
    return low_label, high_label


def clustered_extrema_labels(
    close: pd.Series,
    window: int = 10,
    tolerance: float = 0.005,
    max_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Label all merged local bottoms and tops without enforcing trade pairs."""
    values = close.to_numpy(dtype=float)
    high_candidates: list[int] = []
    low_candidates: list[int] = []
    for idx in range(window, len(values) - window):
        neighbors = np.concatenate((values[idx - window : idx], values[idx + 1 : idx + window + 1]))
        if not np.isfinite(values[idx]) or not np.all(np.isfinite(neighbors)):
            continue
        if values[idx] >= np.max(neighbors) * (1 - tolerance):
            high_candidates.append(idx)
        if values[idx] <= np.min(neighbors) * (1 + tolerance):
            low_candidates.append(idx)

    def merge(indices: list[int], high: bool) -> list[int]:
        clusters: list[list[int]] = []
        for idx in indices:
            if not clusters:
                clusters.append([idx])
                continue
            previous = clusters[-1]
            extreme = max(values[item] for item in previous) if high else min(values[item] for item in previous)
            near_extreme = abs(values[idx] - extreme) / max(abs(extreme), 1e-12) <= tolerance
            if idx - previous[-1] <= max_gap and near_extreme:
                previous.append(idx)
            else:
                clusters.append([idx])
        representatives: list[int] = []
        for cluster in clusters:
            extreme = max(values[item] for item in cluster) if high else min(values[item] for item in cluster)
            representatives.append(next(item for item in cluster if values[item] == extreme))
        return representatives

    low_label = np.zeros(len(values), dtype=bool)
    high_label = np.zeros(len(values), dtype=bool)
    low_label[merge(low_candidates, high=False)] = True
    high_label[merge(high_candidates, high=True)] = True
    return low_label, high_label


def nearby_label(label: np.ndarray, radius: int = 3) -> np.ndarray:
    result = np.zeros(len(label), dtype=bool)
    for idx in np.flatnonzero(label):
        result[max(0, idx - radius) : min(len(label), idx + radius + 1)] = True
    return result


def load_panel(non_daily_signal_on_period_close: bool = False) -> tuple[pd.DataFrame, dict[str, list[tuple[int, int]]]]:
    pool = pd.read_csv(C_RAW / "pool.csv", dtype={"symbol": str}, encoding="utf-8-sig")
    pool["symbol"] = pool["symbol"].astype(str).str.zfill(6)
    pool = pool[pool.get("enabled", True).astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if not QUALITY_MANIFEST.exists():
        raise FileNotFoundError("Canonical ETF quality manifest is missing.")
    quality = json.loads(QUALITY_MANIFEST.read_text(encoding="utf-8"))
    if not bool(quality.get("passed")):
        raise RuntimeError("Canonical ETF quality gate failed; D-line training is blocked.")
    approved = {
        str(symbol).zfill(6) for symbol in quality.get("approved_symbols", [])
    }
    pool = pool[pool["symbol"].isin(approved)].copy()
    DATA.mkdir(parents=True, exist_ok=True)
    pool.to_csv(DATA / "pool_snapshot.csv", index=False, encoding="utf-8-sig")

    parts: list[pd.DataFrame] = []
    groups: dict[str, list[tuple[int, int]]] = {"all": []}
    offset = 0
    for _, row in pool.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        path = CANONICAL_RAW / f"{symbol}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(raw.columns):
            continue
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column not in raw:
                raw[column] = np.nan
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        raw = raw[raw["close"] > 0].copy().reset_index(drop=True)
        if len(raw) < 240:
            continue
        raw["symbol"] = symbol
        raw["name"] = str(row.get("display_name", row.get("name", symbol)))
        raw["category"] = str(row.get("category", ""))
        low_label, high_label = original_paired_labels(raw["close"])
        raw["label_low"] = low_label
        raw["label_high"] = high_label
        raw["near_low"] = nearby_label(low_label)
        raw["near_high"] = nearby_label(high_label)
        raw["year"] = raw["date"].dt.year
        for horizon in (5, 20, 40, 60):
            entry = raw["close"].shift(-1)
            exit_price = raw["close"].shift(-(horizon + 1))
            raw[f"ret_{horizon}"] = exit_price / entry - 1
            raw[f"win_{horizon}"] = raw[f"ret_{horizon}"] > 0

        states = []
        for frequency, freq_name in FREQUENCIES.items():
            state = to_daily_state(
                raw,
                frequency,
                activate_on_next_daily=not non_daily_signal_on_period_close,
            )
            state = state.drop(columns=["date"])
            state.columns = [f"{frequency}::{column}" for column in state.columns]
            states.append(state.reset_index(drop=True))
        part = pd.concat([raw.reset_index(drop=True), *states], axis=1)
        parts.append(part)
        groups["all"].append((offset, offset + len(part)))
        offset += len(part)
    if not parts:
        raise RuntimeError("No valid ETF data found in C/raw/etf")
    panel = pd.concat(parts, ignore_index=True)
    source = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": str(CANONICAL_RAW),
        "quality_manifest": str(QUALITY_MANIFEST),
        "quality_gate_passed": True,
        "pool_source": "C/raw/pool.csv",
        "symbols": int(panel["symbol"].nunique()),
        "rows": int(len(panel)),
        "date_start": str(panel["date"].min().date()),
        "date_end": str(panel["date"].max().date()),
        "label_rule": "A paired low is lower than each of the prior and next 10 closes; the paired high is the highest local high before the next low.",
        "execution": "Signal close is known after T; buy and sell execute at T+1 close. Each buy adds 100 CNY. Sell only lots held at least 7 trading days.",
        "non_daily_alignment": (
            "Signal is dated on the completed period-end close; T+1 is the first following trading day."
            if non_daily_signal_on_period_close
            else "Legacy: completed non-daily state first appears on the following daily row, then the generic T+1 execution adds another day."
        ),
    }
    (DATA / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    return panel, groups


def build_conditions(panel: pd.DataFrame, side: str) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    conditions: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    cores: dict[str, str] = {}
    for frequency, freq_name in FREQUENCIES.items():
        for factor, factor_name in NUMERIC_FACTORS.items():
            value = pd.to_numeric(panel[f"{frequency}::{factor}"], errors="coerce")
            for window in RANGE_WINDOWS:
                # Every ETF owns its own history; never let a rolling range cross symbols.
                low = value.groupby(panel["symbol"], sort=False).transform(
                    lambda series: series.rolling(window, min_periods=20).min()
                )
                high = value.groupby(panel["symbol"], sort=False).transform(
                    lambda series: series.rolling(window, min_periods=20).max()
                )
                pos = (value - low) / (high - low).replace(0, np.nan)
                for level in LOW_LEVELS:
                    threshold = level if side == "buy" else 1 - level
                    mask = (pos <= threshold) if side == "buy" else (pos >= threshold)
                    identifier = f"{frequency}|{factor}|{window}|{int(level * 100)}"
                    direction = "低位" if side == "buy" else "高位"
                    conditions[identifier] = mask.fillna(False).to_numpy(dtype=bool)
                    labels[identifier] = f"{freq_name}{factor_name}{direction}({window}日{int(level * 100)}%)"
                    cores[identifier] = f"{frequency}|{factor}"
        boll_column = "BOLL_LOW" if side == "buy" else "BOLL_HIGH"
        identifier = f"{frequency}|{STRUCTURAL_FACTOR}|touch"
        conditions[identifier] = panel[f"{frequency}::{boll_column}"].fillna(False).to_numpy(dtype=bool)
        labels[identifier] = f"{freq_name}布林{'下' if side == 'buy' else '上'}轨触及"
        cores[identifier] = f"{frequency}|{STRUCTURAL_FACTOR}"
    return conditions, labels, cores


def quality(mask: np.ndarray, panel: pd.DataFrame, side: str) -> dict[str, float | int]:
    target = panel["near_low"].to_numpy(dtype=bool) if side == "buy" else panel["near_high"].to_numpy(dtype=bool)
    signal_count = int(mask.sum())
    outcome = panel["win_20"].to_numpy(dtype=bool) if side == "buy" else ~(panel["win_20"].to_numpy(dtype=bool))
    valid = mask & panel["ret_20"].notna().to_numpy()
    sample = int(valid.sum())
    hit_rate = float(target[mask].mean()) if signal_count else np.nan
    outcome_rate = float(outcome[valid].mean()) if sample else np.nan
    return {"signals": signal_count, "samples_20": sample, "label_hit_3d": hit_rate, "future20_quality": outcome_rate}


def candidates_for_side(
    conditions: dict[str, np.ndarray],
    labels: dict[str, str],
    cores: dict[str, str],
    panel: pd.DataFrame,
    side: str,
    max_factor_count: int,
) -> tuple[list[Candidate], pd.DataFrame]:
    rows = []
    for identifier, mask in conditions.items():
        metrics = quality(mask, panel, side)
        rows.append({"candidate_id": identifier, "factors": identifier, "label": labels[identifier], "factor_count": 1, **metrics})
    singles = pd.DataFrame(rows)
    metric = "future20_quality"
    eligible = singles[(singles["samples_20"] >= MIN_SIGNAL_SAMPLES) & singles[metric].notna()].copy()
    eligible = eligible.sort_values([metric, "label_hit_3d", "samples_20"], ascending=[False, False, False])
    diversified = eligible.assign(core=eligible["candidate_id"].map(cores)).groupby("core", as_index=False, group_keys=False).head(3)
    seed_ids = diversified.head(SEED_LIMIT)["candidate_id"].tolist()

    generated: list[Candidate] = []
    for identifier in seed_ids:
        generated.append(Candidate(identifier, labels[identifier], (identifier,), conditions[identifier]))
    pair_candidates: list[Candidate] = []
    for left, right in itertools.combinations(seed_ids, 2):
        if cores[left] == cores[right]:
            continue
        factors = tuple(sorted((left, right)))
        mask = conditions[left] & conditions[right]
        identifier = " + ".join(factors)
        pair_candidates.append(Candidate(identifier, " + ".join(labels[item] for item in factors), factors, mask))
    pair_rows = []
    for candidate in pair_candidates:
        metrics = quality(candidate.mask, panel, side)
        pair_rows.append({"candidate_id": candidate.identifier, "factors": " + ".join(candidate.factors), "label": candidate.label, "factor_count": 2, **metrics})
    pairs = pd.DataFrame(pair_rows)
    pair_keep = pairs[pairs["samples_20"] >= MIN_SIGNAL_SAMPLES].sort_values([metric, "label_hit_3d", "samples_20"], ascending=[False, False, False]).head(SEED_LIMIT)
    pair_lookup = {candidate.identifier: candidate for candidate in pair_candidates}

    triple_candidates: list[Candidate] = []
    if max_factor_count >= 3:
        triple_seed = seed_ids[:50]
        for pair_id in pair_keep["candidate_id"].tolist():
            base = pair_lookup[pair_id]
            used = {cores[item] for item in base.factors}
            for extra in triple_seed:
                if cores[extra] in used:
                    continue
                factors = tuple(sorted((*base.factors, extra)))
                identifier = " + ".join(factors)
                mask = base.mask & conditions[extra]
                triple_candidates.append(Candidate(identifier, " + ".join(labels[item] for item in factors), factors, mask))
    triples_by_id = {candidate.identifier: candidate for candidate in triple_candidates}
    triple_rows = []
    for candidate in triples_by_id.values():
        metrics = quality(candidate.mask, panel, side)
        triple_rows.append({"candidate_id": candidate.identifier, "factors": " + ".join(candidate.factors), "label": candidate.label, "factor_count": 3, **metrics})
    triples = pd.DataFrame(triple_rows)

    catalog = pd.concat([singles, pairs, triples], ignore_index=True)
    catalog = catalog.drop_duplicates("candidate_id")
    eligible_catalog = catalog[
        (catalog["samples_20"] >= MIN_SIGNAL_SAMPLES) & (catalog["factor_count"] <= max_factor_count)
    ].sort_values([metric, "label_hit_3d", "samples_20"], ascending=[False, False, False])
    quotas = {1: 25, 2: 25, 3: 20} if max_factor_count >= 3 else {1: 35, 2: 35}
    selected = pd.concat(
        [eligible_catalog[eligible_catalog["factor_count"].eq(size)].head(quota) for size, quota in quotas.items()],
        ignore_index=True,
    ).drop_duplicates("candidate_id").head(FINAL_CANDIDATE_LIMIT)
    candidate_lookup = {candidate.identifier: candidate for candidate in generated}
    candidate_lookup.update(pair_lookup)
    candidate_lookup.update(triples_by_id)
    chosen = [candidate_lookup[item] for item in selected["candidate_id"] if item in candidate_lookup]
    return chosen, catalog.sort_values([metric, "label_hit_3d", "samples_20"], ascending=[False, False, False]), selected


def pair_result(
    buy: Candidate,
    sell: Candidate,
    panel: pd.DataFrame,
    groups: dict[str, list[tuple[int, int]]],
    with_years: bool = False,
    entry_start: pd.Timestamp | None = None,
    entry_end: pd.Timestamp | None = None,
    exit_before: pd.Timestamp | None = None,
) -> dict:
    closes = panel["close"].to_numpy(dtype=float)
    dates = pd.to_datetime(panel["date"]).to_numpy()
    returns: list[np.ndarray] = []
    years: list[np.ndarray] = []
    purchased = 0
    for start, end in groups["all"]:
        buy_signal = buy.mask[start:end]
        sell_signal = sell.mask[start:end]
        local_close = closes[start:end]
        local_dates = dates[start:end]
        entry = np.flatnonzero(buy_signal[:-1]) + 1
        if len(entry) and (entry_start is not None or entry_end is not None):
            entry_dates = pd.DatetimeIndex(local_dates[entry])
            keep = np.ones(len(entry), dtype=bool)
            if entry_start is not None:
                keep &= entry_dates >= pd.Timestamp(entry_start)
            if entry_end is not None:
                keep &= entry_dates < pd.Timestamp(entry_end)
            entry = entry[keep]
        purchased += len(entry)
        sell_execution = np.flatnonzero(sell_signal[:-1]) + 1
        if exit_before is not None:
            sell_execution = sell_execution[pd.DatetimeIndex(local_dates[sell_execution]) < pd.Timestamp(exit_before)]
        if not len(entry) or not len(sell_execution):
            continue
        position = np.searchsorted(sell_execution, entry + MIN_HOLD_DAYS, side="left")
        valid = position < len(sell_execution)
        entry = entry[valid]
        exit_index = sell_execution[position[valid]]
        if not len(entry):
            continue
        result = local_close[exit_index] / local_close[entry] - 1
        returns.append(result)
        if with_years:
            years.append(pd.DatetimeIndex(local_dates[entry]).year.to_numpy())
    result_all = np.concatenate(returns) if returns else np.array([], dtype=float)
    closed = int(len(result_all))
    record = {
        "buy_id": buy.identifier,
        "buy_combo": buy.label,
        "buy_factor_count": len(buy.factors),
        "sell_id": sell.identifier,
        "sell_combo": sell.label,
        "sell_factor_count": len(sell.factors),
        "buy_batches": int(purchased),
        "closed_batches": closed,
        "closed_rate": float(closed / purchased) if purchased else np.nan,
        "win_batches": int((result_all > 0).sum()) if closed else 0,
        "win_rate": float((result_all > 0).mean()) if closed else np.nan,
        "avg_return": float(result_all.mean()) if closed else np.nan,
        "return_sum": float(result_all.sum()) if closed else np.nan,
        "median_return": float(np.median(result_all)) if closed else np.nan,
        "worst_return": float(result_all.min()) if closed else np.nan,
        "best_return": float(result_all.max()) if closed else np.nan,
        "total_pnl_100_per_batch": float((result_all * 100).sum()) if closed else np.nan,
    }
    if with_years and years:
        year_values = np.concatenate(years)
        annual = pd.DataFrame({"year": year_values, "ret": result_all}).groupby("year")["ret"].agg(["count", lambda x: (x > 0).mean()])
        annual.columns = ["year_batches", "year_win_rate"]
        annual = annual[annual["year_batches"] >= 3]
        record["covered_years"] = int(len(annual))
        record["worst_year_win_rate"] = float(annual["year_win_rate"].min()) if not annual.empty else np.nan
        record["mean_year_win_rate"] = float(annual["year_win_rate"].mean()) if not annual.empty else np.nan
    return record


def write_report(summary: dict, ranking: pd.DataFrame, buy_catalog: pd.DataFrame, sell_catalog: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame, columns: list[str], count: int = 15) -> str:
        if frame.empty:
            return "无符合最低样本条件的结果。"
        view = frame.head(count)[columns].copy()
        for column in ("win_rate", "avg_return", "median_return", "label_hit_3d", "future20_quality", "closed_rate"):
            if column in view:
                view[column] = view[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    lines = [
        "# D 线：指标交叉定投首轮扫描",
        "",
        "本轮使用 C 线相同的 30 只清洗 ETF 日线。每次低位买点均独立买入 100 元；高位卖点只卖出已持有满 7 个交易日的批次；同一 ETF 可连续加仓，多 ETF 可并行持有；不计手续费和总资金限制。",
        "",
        "**重要**：这里的“原始高低点”只用于研究阶段的事后质量标签，最终排名的主指标是按 T+1 收盘实际进出、逐笔计算的胜率，并非事后低买高卖收益。结果仍是全样本发现，尚未通过滚动样本外验证。",
        "",
        "## 数据与扫描范围",
        f"- 数据：{summary['symbols']} 只 ETF，{summary['date_start']} 至 {summary['date_end']}，共 {summary['rows']} 行。",
        "- 因子：KDJ-K、KDJ-D、KDJ-J、KD、RSI6、DIF、EMV、LON、成交量、布林线；周期：日K、周K、15日K、月K。",
        "- 分位参数：自身 60/90/120/180 期区间；低位 5%–25%，高位 75%–95%；组合最多 3 个条件。",
        "- 主排名门槛：至少 100 笔已完成卖出批次，且至少 60% 买入批次已经有卖点退出；完成率会与胜率同时展示。",
        "",
        "## 胜率最高的买卖组合",
        table(ranking, ["win_rate", "closed_batches", "closed_rate", "avg_return", "worst_year_win_rate", "buy_combo", "sell_combo"], 20),
        "",
        "## 买点候选：未来 20 日正收益率",
        table(buy_catalog, ["future20_quality", "label_hit_3d", "samples_20", "label"], 15),
        "",
        "## 卖点候选：未来 20 日回落率",
        table(sell_catalog, ["future20_quality", "label_hit_3d", "samples_20", "label"], 15),
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, groups = load_panel()
    source = json.loads((DATA / "source.json").read_text(encoding="utf-8"))
    buy_conditions, buy_labels, buy_cores = build_conditions(panel, "buy")
    sell_conditions, sell_labels, sell_cores = build_conditions(panel, "sell")
    buys, buy_catalog, buy_selected = candidates_for_side(buy_conditions, buy_labels, buy_cores, panel, "buy", max_factor_count=3)
    sells, sell_catalog, sell_selected = candidates_for_side(sell_conditions, sell_labels, sell_cores, panel, "sell", max_factor_count=2)
    buy_catalog.to_csv(OUT / "buy_catalog.csv", index=False, encoding="utf-8-sig")
    sell_catalog.to_csv(OUT / "sell_catalog.csv", index=False, encoding="utf-8-sig")
    buy_selected.to_csv(OUT / "buy_selected.csv", index=False, encoding="utf-8-sig")
    sell_selected.to_csv(OUT / "sell_selected.csv", index=False, encoding="utf-8-sig")

    results = []
    for buy, sell in itertools.product(buys, sells):
        result = pair_result(buy, sell, panel, groups)
        if result["closed_batches"] >= MIN_SIGNAL_SAMPLES and result["closed_rate"] >= MIN_CLOSED_RATE:
            results.append(result)
    ranking = pd.DataFrame(results)
    if not ranking.empty:
        ranking = ranking.sort_values(["win_rate", "avg_return", "closed_batches"], ascending=[False, False, False]).head(300).copy()
        stable_rows = []
        buy_map = {candidate.identifier: candidate for candidate in buys}
        sell_map = {candidate.identifier: candidate for candidate in sells}
        for _, row in ranking.iterrows():
            stable_rows.append(pair_result(buy_map[row["buy_id"]], sell_map[row["sell_id"]], panel, groups, with_years=True))
        ranking = pd.DataFrame(stable_rows)
        ranking = ranking[(ranking["covered_years"] >= 3) & ranking["worst_year_win_rate"].notna()]
        ranking = ranking.sort_values(["win_rate", "avg_return", "closed_batches"], ascending=[False, False, False])
    ranking.to_csv(OUT / "pair_ranking.csv", index=False, encoding="utf-8-sig")

    summary = {
        **source,
        "buy_conditions": len(buy_conditions),
        "sell_conditions": len(sell_conditions),
        "buy_candidates": len(buys),
        "sell_candidates": len(sells),
        "tested_pairs": int(len(results)),
        "ranked_pairs": int(len(ranking)),
        "minimum_completed_batches": MIN_SIGNAL_SAMPLES,
        "minimum_closed_rate": MIN_CLOSED_RATE,
        "minimum_hold_days": MIN_HOLD_DAYS,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, ranking, buy_selected, sell_selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
