from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from v1_ml_confirmation import confirmation_dataset
from v1_ml_extrema import ModelConfig, RANDOM_SEEDS, add_features, make_model


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw" / "etf"
MODEL_DIR = ROOT / "live" / "models"
LIVE_DIR = ROOT / "live"
HISTORICAL_PANEL = ROOT / "fit" / "extrema" / "feature_panel.parquet"
MODEL_SUMMARY = ROOT / "fit" / "extrema" / "summary.json"
YEARLY_THRESHOLDS = ROOT / "fit" / "extrema" / "strict_yearly.csv"
DELAY_DAYS = 5


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def build_live_panel(canonical_panel: pd.DataFrame | None = None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        raw = pd.read_csv(path, dtype={"symbol": str})
        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        if not required.issubset(raw.columns):
            continue
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
        if "name" not in raw.columns:
            raw["name"] = ""
        else:
            raw["name"] = raw["name"].fillna("")
        enriched = add_features(raw.reset_index(drop=True))
        # Live scoring never consumes the future-defined extrema labels.
        enriched["is_low"] = np.int8(0)
        enriched["is_high"] = np.int8(0)
        enriched["year"] = enriched["date"].dt.year
        enriched["bar_index"] = np.arange(len(enriched), dtype=int)
        parts.append(enriched)
    if not parts:
        raise RuntimeError(f"No valid ETF files found under {RAW_DIR}")

    panel = pd.concat(parts, ignore_index=True, sort=False)
    if canonical_panel is None:
        canonical_panel = pd.read_parquet(HISTORICAL_PANEL, columns=["date", "symbol"])
    calendar = canonical_panel[["date", "symbol"]].copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["symbol"] = calendar["symbol"].astype(str).str.zfill(6)
    calendar["_canonical_date"] = True
    last_dates = calendar.groupby("symbol", as_index=False)["date"].max().rename(columns={"date": "_canonical_last"})
    panel = panel.merge(calendar, on=["date", "symbol"], how="left", validate="many_to_one")
    panel = panel.merge(last_dates, on="symbol", how="left", validate="many_to_one")
    # Preserve the exact research calendar, then append only genuinely new bars.
    # This prevents an upstream provider adding old dates from changing what
    # "five trading days later" meant in the audited model.
    panel = panel[
        panel["_canonical_date"].fillna(False)
        | panel["_canonical_last"].isna()
        | panel["date"].gt(panel["_canonical_last"])
    ].drop(columns=["_canonical_date", "_canonical_last"])
    market_inputs = ["ret_1", "ret_5", "ret_20", "ret_60", "ma_gap_20", "ma_gap_60", "range_pos_20"]
    for feature in market_inputs:
        panel[f"market_mean_{feature}"] = panel.groupby("date")[feature].transform("mean")
        panel[f"market_rank_{feature}"] = panel.groupby("date")[feature].rank(pct=True)
    panel["market_breadth_ma20"] = panel.groupby("date")["ma_gap_20"].transform(lambda x: (x > 0).mean())
    panel["market_breadth_ma60"] = panel.groupby("date")["ma_gap_60"].transform(lambda x: (x > 0).mean())
    return panel.replace([np.inf, -np.inf], np.nan).sort_values(["date", "symbol"]).reset_index(drop=True)


def selected_high_configs() -> list[ModelConfig]:
    summary = json.loads(MODEL_SUMMARY.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(ModelConfig)}
    return [
        ModelConfig(**{key: value for key, value in row.items() if key in allowed})
        for row in summary["selected_configs"]["is_high"]
    ]


def model_signature(year: int, configs: list[ModelConfig], features: list[str]) -> str:
    payload = {
        "year": year,
        "delay_days": DELAY_DAYS,
        "configs": [config.__dict__ for config in configs],
        "features": features,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def load_or_train_models(
    historical_candidate: pd.DataFrame,
    features: list[str],
    year: int,
    configs: list[ModelConfig],
) -> tuple[list, str, int, int]:
    cutoff = pd.Timestamp(year, 1, 1)
    train_mask = historical_candidate["label_known_date"].lt(cutoff)
    train = historical_candidate.loc[train_mask].copy()
    y = train["is_high"].to_numpy(dtype=np.int8)
    if len(train) < 300 or int(y.sum()) < 50:
        raise RuntimeError(f"Insufficient past-only training data for {year}: rows={len(train)}, highs={int(y.sum())}")

    signature = model_signature(year, configs, features)
    target_dir = MODEL_DIR / f"high_{year}_{signature}"
    target_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for index, config in enumerate(configs):
        path = target_dir / f"model_{index + 1}.joblib"
        if path.exists():
            model = joblib.load(path)
        else:
            model = make_model(config, y.mean(), RANDOM_SEEDS[index] + year)
            model.fit(train[features], y)
            joblib.dump(model, path)
        models.append(model)
    metadata = {
        "year": year,
        "signature": signature,
        "train_cutoff_exclusive": cutoff.strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "train_highs": int(y.sum()),
        "features": int(len(features)),
        "models": int(len(models)),
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return models, signature, int(len(train)), int(y.sum())


def current_threshold(year: int) -> float:
    table = pd.read_csv(YEARLY_THRESHOLDS)
    rows = table[
        table["target"].eq("is_high")
        & table["window"].astype(str).eq("5")
        & pd.to_numeric(table["year"], errors="coerce").le(year)
    ].sort_values("year")
    if rows.empty:
        raise RuntimeError(f"No past-only high threshold available for {year}")
    return float(rows.iloc[-1]["threshold"])


def main() -> None:
    configure_stdout()
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    historical_panel = pd.read_parquet(HISTORICAL_PANEL)
    historical_candidate, features = confirmation_dataset(historical_panel, "is_high", DELAY_DAYS)

    live_panel = build_live_panel(historical_panel)
    latest_date = pd.Timestamp(live_panel["date"].max())
    year = int(latest_date.year)
    live_candidate, _ = confirmation_dataset(live_panel, "is_high", DELAY_DAYS)
    live_candidate = live_candidate[live_candidate["signal_year"].eq(year)].copy()
    for column in features:
        if column not in live_candidate.columns:
            live_candidate[column] = np.nan

    configs = selected_high_configs()
    models, signature, train_rows, train_highs = load_or_train_models(
        historical_candidate, features, year, configs
    )
    probabilities = [model.predict_proba(live_candidate[features])[:, 1] for model in models]
    live_candidate["score"] = np.mean(np.vstack(probabilities), axis=0)
    threshold = current_threshold(year)
    live_candidate["threshold"] = threshold
    live_candidate["predicted"] = live_candidate["score"].ge(threshold)
    live_candidate["candidate_date"] = pd.to_datetime(live_candidate["date"])

    columns = [
        "symbol",
        "name",
        "candidate_date",
        "signal_date",
        "signal_year",
        "close",
        "score",
        "threshold",
        "predicted",
    ]
    candidates = live_candidate[columns].sort_values(["signal_date", "symbol"])
    signals = candidates[candidates["predicted"]].copy()
    candidates.to_csv(LIVE_DIR / "extrema_high_candidates.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(LIVE_DIR / "extrema_high_signals.csv", index=False, encoding="utf-8-sig")

    audit_error = None
    reference_path = ROOT / "fit" / "extrema" / "predictions.csv"
    if reference_path.exists():
        reference = pd.read_csv(reference_path, dtype={"symbol": str}, usecols=["date", "symbol", "high_score"])
        reference["date"] = pd.to_datetime(reference["date"])
        reference["symbol"] = reference["symbol"].astype(str).str.zfill(6)
        overlap = candidates.merge(
            reference.rename(columns={"date": "candidate_date", "high_score": "reference_score"}),
            on=["candidate_date", "symbol"],
            how="inner",
        ).dropna(subset=["reference_score"])
        if not overlap.empty:
            audit_error = float((overlap["score"] - overlap["reference_score"]).abs().max())

    status = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_data_date": latest_date.strftime("%Y-%m-%d"),
        "model_year": year,
        "train_cutoff_exclusive": f"{year}-01-01",
        "train_rows": train_rows,
        "train_highs": train_highs,
        "confirmation_delay_days": DELAY_DAYS,
        "threshold_window_years": 5,
        "threshold": threshold,
        "model_signature": signature,
        "candidate_count": int(len(candidates)),
        "signal_count": int(len(signals)),
        "latest_signal_date": (
            pd.Timestamp(signals["signal_date"].max()).strftime("%Y-%m-%d") if not signals.empty else None
        ),
        "reference_score_max_abs_error": audit_error,
        "causal_guarantee": "Only labels known before the model year are used for fitting; live labels are never read.",
    }
    (LIVE_DIR / "extrema_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
