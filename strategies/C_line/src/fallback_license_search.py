from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "fallback_license"


@dataclass(frozen=True)
class LicenseRule:
    lookback: int
    min_count: int
    min_win: float
    min_mean: float
    min_compound: float
    level: str

    @property
    def key(self) -> str:
        return (
            f"license:{self.level}:y{self.lookback}:n{self.min_count}:"
            f"w{self.min_win:.2f}:m{self.min_mean:+.3f}:c{self.min_compound:+.3f}"
        )


def rules() -> list[LicenseRule]:
    result: list[LicenseRule] = []
    for level in ("all_fallback", "source"):
        for lookback in (3, 5, 99):
            for min_count in (2, 4):
                for min_win in (0.40, 0.50, 0.60):
                    for min_mean in (0.0, 0.005):
                        for min_compound in (0.0, 0.02):
                            result.append(
                                LicenseRule(
                                    lookback,
                                    min_count,
                                    min_win,
                                    min_mean,
                                    min_compound,
                                    level,
                                )
                            )
    return result


def is_main(frame: pd.DataFrame) -> pd.Series:
    if "source_type" in frame:
        return ~frame["source_type"].fillna("").astype(str).str.contains("空仓补偿")
    if "source" in frame:
        return frame["source"].fillna("").astype(str).eq("主策略")
    return pd.Series(True, index=frame.index)


def apply_rule(frame: pd.DataFrame, rule: LicenseRule) -> pd.DataFrame:
    data = frame.copy()
    data["entry_date"] = pd.to_datetime(data["entry_date"])
    data["license_kept"] = True
    data["license_rule"] = rule.key
    data["license_history_count"] = 0
    data["license_history_win"] = np.nan
    data["license_history_mean"] = np.nan
    data["license_history_compound"] = np.nan
    main_mask = is_main(data)
    fallback = data.loc[~main_mask].copy()
    returns = pd.to_numeric(fallback["gross_return_test"], errors="coerce")
    fallback["_return"] = returns
    fallback["_year"] = fallback["entry_date"].dt.year

    for idx, row in fallback.iterrows():
        year = int(row["_year"])
        start_year = -10_000 if rule.lookback >= 99 else year - rule.lookback
        history = fallback[
            fallback["_year"].between(start_year, year - 1, inclusive="both")
        ]
        if rule.level == "source":
            history = history[history["source"].astype(str).eq(str(row["source"]))]
        history_return = history["_return"].dropna().astype(float)
        count = len(history_return)
        win = float((history_return > 0).mean()) if count else np.nan
        mean = float(history_return.mean()) if count else np.nan
        compound = float(np.prod(1.0 + history_return) - 1.0) if count else np.nan
        keep = bool(
            count < rule.min_count
            or (
                win >= rule.min_win
                and mean >= rule.min_mean
                and compound >= rule.min_compound
            )
        )
        data.loc[idx, "license_kept"] = keep
        data.loc[idx, "license_history_count"] = count
        data.loc[idx, "license_history_win"] = win
        data.loc[idx, "license_history_mean"] = mean
        data.loc[idx, "license_history_compound"] = compound
    return data[data["license_kept"].astype(bool)].copy()


def profile(frame: pd.DataFrame, years: list[int]) -> dict[int, dict]:
    _, detail, _ = core.simulate_account(frame)
    detail_year = pd.to_datetime(detail["entry_date"]).dt.year
    source_year = pd.to_datetime(frame["entry_date"]).dt.year
    result: dict[int, dict] = {}
    for year in years:
        local = detail[detail_year.lt(year)].copy()
        source = frame[source_year.lt(year)]
        if local.empty:
            result[year] = {
                "trades": 0,
                "final_value": core.INITIAL_CAPITAL,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "removed": 0,
            }
            continue
        returns = local["account_return_test"].astype(float)
        equity = core.INITIAL_CAPITAL * (1.0 + returns).cumprod()
        closed = local[~local["open_mark_test"].astype(bool)]
        result[year] = {
            "trades": int(len(local)),
            "final_value": float(local.iloc[-1]["cash_after_test"]),
            "win_rate": float((closed["account_return_test"] > 0).mean()),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "removed": int((~source.get("license_kept", pd.Series(True, index=source.index)).astype(bool)).sum()),
        }
    return result


def choose(profiles: dict[str, dict[int, dict]], rule_map: dict[str, LicenseRule], year: int) -> tuple[str, dict]:
    base = profiles["baseline"][year]
    rows = []
    for name, values in profiles.items():
        if name == "baseline":
            continue
        current = values[year]
        ratio = current["final_value"] / base["final_value"]
        win_delta = current["win_rate"] - base["win_rate"]
        dd_delta = current["max_drawdown"] - base["max_drawdown"]
        rule = rule_map[name]
        passes = bool(
            current["trades"] >= 12
            and current["removed"] >= 1
            and ratio > 1.0
            and win_delta >= 0.0
            and dd_delta >= -0.005
        )
        rows.append(
            {
                "year": year,
                "rule": name,
                "level": rule.level,
                "lookback": rule.lookback,
                "min_count": rule.min_count,
                "min_win": rule.min_win,
                "min_mean": rule.min_mean,
                "min_compound": rule.min_compound,
                "passes": passes,
                "ratio": ratio,
                "win_delta": win_delta,
                "dd_delta": dd_delta,
                "removed": current["removed"],
                "score": np.log(max(ratio, 1e-12)) + 0.8 * win_delta + 0.2 * min(dd_delta, 0.2),
            }
        )
    passing = [row for row in rows if row["passes"]]
    # A broad plateau is required: the same source level and lookback must work
    # under several nearby evidence thresholds.
    family_counts = pd.Series(
        [(row["level"], row["lookback"]) for row in passing]
    ).value_counts().to_dict()
    stable = [
        row
        for row in passing
        if family_counts.get((row["level"], row["lookback"]), 0) >= 6
    ]
    if not stable:
        return "baseline", {"year": year, "rule": "baseline", "level": "baseline"}
    winner = max(stable, key=lambda row: row["score"])
    return str(winner["rule"]), winner


def validate_line(spec: core.LineSpec, path: Path, candidates: list[LicenseRule]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frames = {"baseline": frame}
    rule_map = {}
    for rule in candidates:
        frames[rule.key] = apply_rule(frame, rule)
        rule_map[rule.key] = rule
    years = sorted(frame["entry_date"].dt.year.unique())
    profiles = {name: profile(candidate, years) for name, candidate in frames.items()}
    selections = []
    parts = []
    for year in years:
        name, audit = choose(profiles, rule_map, int(year))
        candidate = frames[name]
        local = candidate[candidate["entry_date"].dt.year.eq(year)].copy()
        local["walkforward_license_rule"] = name
        parts.append(local)
        selections.append(audit)
    winner_frame = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    baseline = {period: search.subset_stats(frame, period) for period in ("full", "dev", "holdout")}
    winner = {period: search.subset_stats(winner_frame, period) for period in ("full", "dev", "holdout")}
    return {
        "line": spec.key,
        "candidate_rules": len(candidates),
        "baseline": baseline,
        "winner": winner,
        "full_ratio": winner["full"]["final_value"] / baseline["full"]["final_value"],
        "full_win_delta": winner["full"]["win_rate"] - baseline["full"]["win_rate"],
        "holdout_ratio": winner["holdout"]["final_value"] / baseline["holdout"]["final_value"],
        "holdout_win_delta": winner["holdout"]["win_rate"] - baseline["holdout"]["win_rate"],
        "selected_years": int(sum(row["rule"] != "baseline" for row in selections)),
        "trades_removed": int(len(frame) - len(winner_frame)),
    }, winner_frame, pd.DataFrame(selections)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {spec.key: spec for spec in core.SPECS}
    candidates = rules()
    results = []
    for line in ("C", "S"):
        path = ROOT / "fit" / "rolling_upgrades" / f"{line.lower()}_trades.csv"
        payload, trades, selections = validate_line(specs[line], path, candidates)
        results.append(payload)
        trades.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        selections.to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window fallback license selector",
        "execution": "each test year uses only completed prior-year fallback evidence",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
