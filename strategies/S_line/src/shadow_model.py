from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from s1lib import QUALITY, ROOT, read_etf


BASE = ROOT / "fit" / "base"
RESEARCH = ROOT / "out"
LIVE = ROOT / "live"


@dataclass(frozen=True)
class Param:
    scope: str
    idle: int
    hold: int
    ma: int
    mom: int
    model: str
    alpha: float
    vol_cap: float
    accel_cap: float
    min_history: int
    min_amount20: float

    @property
    def param_id(self) -> str:
        alpha_txt = str(int(round(self.alpha * 100)))
        vol_txt = str(int(round(self.vol_cap * 100)))
        accel_txt = str(int(round(self.accel_cap * 100)))
        return (
            f"{self.scope}_i{self.idle}_h{self.hold}_ma{self.ma}_m{self.mom}_"
            f"{self.model}_a{alpha_txt}_v{vol_txt}_ac{accel_txt}_"
            f"hist{self.min_history}_amt{int(self.min_amount20)}"
        )


def _rank(values: np.ndarray, smaller_is_better: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return out
    vals = arr[valid]
    order = np.argsort(-vals if smaller_is_better else vals)
    ranked = np.linspace(1.0 / len(vals), 1.0, len(vals))
    fill = np.empty(len(vals))
    fill[order] = ranked
    out[np.where(valid)[0]] = fill
    return out


def _safe_ratio(a: float, b: float, floor: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return a / max(b, floor)


def _model_rank(name: str, preferred: list[str]) -> int:
    try:
        return preferred.index(str(name))
    except ValueError:
        return len(preferred)


class ShadowResearchEngine:
    def __init__(self) -> None:
        RESEARCH.mkdir(parents=True, exist_ok=True)
        LIVE.mkdir(parents=True, exist_ok=True)
        self.pool = pd.read_csv(QUALITY / "pool_clean.csv", dtype={"symbol": str}, encoding="utf-8-sig")
        self.pool["symbol"] = self.pool["symbol"].astype(str).str.zfill(6)
        quality_path = ROOT / "raw" / "quality.json"
        if not quality_path.exists():
            raise FileNotFoundError("S-line ETF quality manifest is missing.")
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if not bool(quality.get("passed")):
            raise RuntimeError("S-line ETF quality gate failed; research engine is blocked.")
        self.approved_symbols = {
            str(symbol).zfill(6)
            for symbol in quality.get("approved_symbols", [])
        }
        self.pool = self.pool[self.pool["symbol"].isin(self.approved_symbols)].copy()
        self.meta = self.pool.set_index("symbol").to_dict("index")
        self.seed_trades = self._load_seed_trades()
        self.seed_yearly = self._load_seed_yearly()
        self.main = self.seed_trades[self.seed_trades["source_type"] == "主策略"].copy()
        self.main = self.main[["source", "category", "symbol", "display_name", "entry_date", "exit_date", "ret", "source_type"]]
        self.main["priority"] = 1000.0
        self.pre_account = self.seed_trades[self.seed_trades["entry_year"] < 2024].copy()
        self.broad = dict(zip(self.seed_yearly["year"].astype(int), self.seed_yearly["broad_return"].astype(float)))
        self.baseline_account = self._load_baseline_account()
        self.baseline_yearly = self._account_yearly(self.baseline_account)
        self.scope_map = self._build_scope_map()
        self.panel = self._load_panel()
        self.calendar = self.panel["510880"]["date"].reset_index(drop=True)
        self.cal_arr = self.calendar.to_numpy()
        self.idle_windows = self._build_idle_windows()

    def _load_seed_trades(self) -> pd.DataFrame:
        df = pd.read_csv(BASE / "s1_seed_trades.csv", dtype={"symbol": str}, encoding="utf-8-sig")
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        df = df[df["symbol"].isin(self.approved_symbols)].copy()
        df = self._reprice_trades(df)
        df["entry_year"] = df["entry_date"].dt.year
        return df.sort_values(["entry_date", "exit_date", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _reprice_trades(frame: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for symbol, group in frame.groupby("symbol", sort=False):
            prices = read_etf(symbol)[["date", "close"]].copy()
            prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
            price_map = prices.drop_duplicates("date").set_index("date")["close"]
            local = group.copy()
            local["entry_close_clean"] = local["entry_date"].map(price_map)
            local["exit_close_clean"] = local["exit_date"].map(price_map)
            local = local.dropna(subset=["entry_close_clean", "exit_close_clean"])
            local["ret"] = (
                local["exit_close_clean"].astype(float)
                / local["entry_close_clean"].astype(float)
                - 1.0
            )
            parts.append(local)
        return pd.concat(parts, ignore_index=True, sort=False) if parts else frame.iloc[0:0].copy()

    def _load_seed_yearly(self) -> pd.DataFrame:
        df = pd.read_csv(BASE / "s1_seed_yearly.csv", encoding="utf-8-sig")
        rename = {
            "年份": "year",
            "交易数": "trades",
            "主策略笔数": "main_trades",
            "补偿笔数": "shadow_trades",
            "胜率": "win_rate",
            "开仓年收益": "entry_year_return",
            "自然年收益": "calendar_return",
            "宽基收益": "broad_return",
            "自然年超额": "calendar_excess",
            "年末净值": "ending_equity",
        }
        return df.rename(columns=rename)

    def _load_baseline_account(self) -> pd.DataFrame:
        df = pd.read_csv(BASE / "domestic_baseline_account.csv", dtype={"symbol": str}, encoding="utf-8-sig")
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        df = df[df["symbol"].isin(self.approved_symbols)].copy()
        df = self._reprice_trades(df)
        return df.sort_values(["entry_date", "exit_date", "symbol"]).reset_index(drop=True)

    def _build_scope_map(self) -> dict[str, tuple[str, ...]]:
        domestic = pd.read_csv(BASE / "domestic_pool_s1.csv", dtype={"symbol": str}, encoding="utf-8-sig")
        domestic["symbol"] = domestic["symbol"].astype(str).str.zfill(6)
        domestic = domestic[domestic["symbol"].isin(self.approved_symbols)].copy()
        domestic_symbols = tuple(sorted(domestic["symbol"].unique().tolist()))

        pool = self.pool.copy()
        precious = tuple(
            sorted(
                pool.loc[pool["sub_category"].isin(["黄金", "白银", "贵金属篮子"]), "symbol"].astype(str).str.zfill(6).unique().tolist()
            )
        )
        us_core = tuple(
            sorted(
                pool.loc[pool["sub_category"].isin(["纳斯达克100", "标普500"]), "symbol"].astype(str).str.zfill(6).unique().tolist()
            )
        )
        no_newenergy = tuple(
            sorted(
                domestic.loc[domestic["category"] != "新能源/高端制造", "symbol"].astype(str).str.zfill(6).unique().tolist()
            )
        )
        all_clean = tuple(sorted(pool["symbol"].astype(str).str.zfill(6).unique().tolist()))

        def merged(*groups: tuple[str, ...]) -> tuple[str, ...]:
            vals: set[str] = set()
            for group in groups:
                vals.update(group)
            return tuple(sorted(vals))

        return {
            "domestic_core": domestic_symbols,
            "domestic_no_newenergy": no_newenergy,
            "domestic_plus_precious": merged(domestic_symbols, precious),
            "domestic_plus_us": merged(domestic_symbols, us_core),
            "domestic_plus_precious_us": merged(domestic_symbols, precious, us_core),
            "all_clean": all_clean,
        }

    def _load_panel(self) -> dict[str, pd.DataFrame]:
        panel: dict[str, pd.DataFrame] = {}
        for symbol in self.pool["symbol"]:
            df = read_etf(symbol).copy()
            df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            close = pd.to_numeric(df["close"], errors="coerce")
            ret = close.pct_change()
            for n in [10, 20, 30, 40, 60, 120]:
                df[f"mom{n}"] = close.pct_change(n)
            for n in [40, 60, 90, 120]:
                df[f"ma{n}"] = close.rolling(n, min_periods=max(20, n // 2)).mean()
            df["vol20"] = ret.rolling(20, min_periods=15).std() * np.sqrt(252)
            df["down20"] = ret.where(ret < 0, 0).rolling(20, min_periods=15).std() * np.sqrt(252)
            df["amount20"] = pd.to_numeric(df["amount"], errors="coerce").rolling(20, min_periods=10).median()
            df["history"] = np.arange(1, len(df) + 1)
            df["dist_ma120"] = (close / df["ma120"] - 1).abs()
            panel[symbol] = df
        return panel

    def _build_idle_windows(self) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
        wins: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
        start = pd.Timestamp("2014-01-01")
        for _, row in self.main.sort_values("entry_date").iterrows():
            st = self.next_date(start)
            ep = int(np.searchsorted(self.cal_arr, np.datetime64(row["entry_date"] - pd.Timedelta(days=1)), side="right") - 1)
            if st is not None and ep >= 0:
                en = pd.Timestamp(self.calendar.iloc[ep])
                n = int(((self.calendar >= st) & (self.calendar <= en)).sum())
                if n > 0:
                    wins.append((st, en, n))
            start = pd.Timestamp(row["exit_date"]) + pd.Timedelta(days=1)
        st = self.next_date(start)
        if st is not None:
            en = pd.Timestamp(self.calendar.iloc[-1])
            n = int(((self.calendar >= st) & (self.calendar <= en)).sum())
            wins.append((st, en, n))
        return wins

    def next_date(self, dt: pd.Timestamp) -> pd.Timestamp | None:
        pos = int(np.searchsorted(self.cal_arr, np.datetime64(dt), side="left"))
        if pos >= len(self.cal_arr):
            return None
        return pd.Timestamp(self.calendar.iloc[pos])

    @staticmethod
    def pos_before(df: pd.DataFrame, dt: pd.Timestamp) -> int | None:
        arr = df["date"].to_numpy()
        pos = int(np.searchsorted(arr, np.datetime64(dt), side="right") - 1)
        return pos if pos >= 0 else None

    @staticmethod
    def pos_after(df: pd.DataFrame, dt: pd.Timestamp) -> int | None:
        arr = df["date"].to_numpy()
        pos = int(np.searchsorted(arr, np.datetime64(dt), side="left"))
        return pos if pos < len(arr) else None

    @lru_cache(maxsize=None)
    def candidate_list(
        self,
        decision_ns: int,
        scope: str,
        ma: int,
        mom: int,
        model: str,
        alpha: float,
        min_history: int,
        min_amount20: float,
        vol_cap: float,
        accel_cap: float,
    ) -> tuple[tuple[float, str, dict], ...]:
        dt = pd.Timestamp(decision_ns)
        records = []
        for symbol in self.scope_map[scope]:
            df = self.panel[symbol]
            pos = self.pos_before(df, dt)
            if pos is None:
                continue
            row = df.iloc[pos]
            values = {
                "close": float(row["close"]) if pd.notna(row["close"]) else np.nan,
                "ma": float(row[f"ma{ma}"]) if pd.notna(row[f"ma{ma}"]) else np.nan,
                "mom20": float(row["mom20"]) if pd.notna(row["mom20"]) else np.nan,
                "mom40": float(row["mom40"]) if pd.notna(row["mom40"]) else np.nan,
                "mom60": float(row["mom60"]) if pd.notna(row["mom60"]) else np.nan,
                "mom120": float(row["mom120"]) if pd.notna(row["mom120"]) else np.nan,
                "base_mom": float(row[f"mom{mom}"]) if pd.notna(row[f"mom{mom}"]) else np.nan,
                "vol20": float(row["vol20"]) if pd.notna(row["vol20"]) else np.nan,
                "down20": float(row["down20"]) if pd.notna(row["down20"]) else np.nan,
                "amount20": float(row["amount20"]) if pd.notna(row["amount20"]) else np.nan,
                "dist_ma120": float(row["dist_ma120"]) if pd.notna(row["dist_ma120"]) else np.nan,
                "history": int(row["history"]) if pd.notna(row["history"]) else 0,
            }
            if not np.isfinite(values["ma"]) or values["close"] <= values["ma"]:
                continue
            if values["history"] < min_history:
                continue
            if not np.isfinite(values["amount20"]) or values["amount20"] < min_amount20:
                continue
            if not np.isfinite(values["vol20"]) or values["vol20"] > vol_cap:
                continue
            accel = _safe_ratio(values["mom20"], values["base_mom"], 1e-9)
            if np.isfinite(accel) and accel > accel_cap:
                continue
            if not np.isfinite(values["base_mom"]) or values["base_mom"] <= 0:
                continue
            values["accel"] = accel
            records.append((symbol, values))

        if not records:
            return tuple()

        symbols = [s for s, _ in records]
        base_vals = np.array([v["base_mom"] for _, v in records], dtype=float)
        rk_base = _rank(base_vals)
        target_score = rk_base.copy()

        if model != "raw":
            mom20 = np.array([v["mom20"] for _, v in records], dtype=float)
            mom60 = np.array([v["mom60"] for _, v in records], dtype=float)
            mom120 = np.array([v["mom120"] for _, v in records], dtype=float)
            vol20 = np.array([v["vol20"] for _, v in records], dtype=float)
            down20 = np.array([v["down20"] for _, v in records], dtype=float)
            dist120 = np.array([v["dist_ma120"] for _, v in records], dtype=float)
            rk_m60 = _rank(mom60)
            rk_m120 = _rank(mom120)
            rk_lowvol = _rank(vol20, smaller_is_better=True)
            rk_near = _rank(dist120, smaller_is_better=True)
            rk_risk60 = _rank(np.array([_safe_ratio(x, y, 0.05) for x, y in zip(mom60, vol20)], dtype=float))
            rk_risk120 = _rank(np.array([_safe_ratio(x, y, 0.05) for x, y in zip(mom120, vol20)], dtype=float))
            rk_down = _rank(np.array([_safe_ratio(x, y, 0.03) for x, y in zip(base_vals, down20)], dtype=float))
            consistency_raw = np.array([min(a, b, c) + 0.25 * (a + b + c) for a, b, c in zip(mom20, mom60, mom120)], dtype=float)
            rk_consistency = _rank(consistency_raw)

            if model == "defensive":
                target_score = 0.30 * rk_m60 + 0.35 * rk_m120 + 0.20 * rk_risk60 + 0.10 * rk_lowvol + 0.05 * rk_near
            elif model == "stable":
                target_score = 0.40 * rk_m60 + 0.35 * rk_m120 + 0.15 * rk_lowvol + 0.10 * rk_near
            elif model == "risk60":
                target_score = rk_risk60
            elif model == "risk120":
                target_score = rk_risk120
            elif model == "downside":
                target_score = rk_down
            elif model == "consistent":
                target_score = rk_consistency
            else:
                raise ValueError(f"unknown model: {model}")

        final_score = rk_base if model == "raw" else (1.0 - alpha) * rk_base + alpha * target_score
        ordered = np.lexsort((np.array(symbols), -final_score))
        out = []
        for idx in ordered:
            symbol = symbols[idx]
            vals = records[idx][1]
            out.append(
                (
                    float(final_score[idx]),
                    symbol,
                    {
                        "score": float(final_score[idx]),
                        "momentum": float(vals["base_mom"]),
                        "mom20": float(vals["mom20"]),
                        "mom60": float(vals["mom60"]),
                        "mom120": float(vals["mom120"]),
                        "vol20": float(vals["vol20"]),
                        "amount20": float(vals["amount20"]),
                        "history_days": int(vals["history"]),
                        "accel": float(vals["accel"]) if np.isfinite(vals["accel"]) else np.nan,
                    },
                )
            )
        return tuple(out)

    def generate_shadow_trades(self, param: Param, *, causal_idle: bool = False) -> pd.DataFrame:
        rows: list[dict] = []
        for start, end, idle_days in self.idle_windows:
            if idle_days < param.idle:
                continue
            cur = start
            if causal_idle:
                idle_calendar = self.calendar[(self.calendar >= start) & (self.calendar <= end)].reset_index(drop=True)
                if len(idle_calendar) < param.idle:
                    continue
                # Do not use the future total length of the idle window to trade early.
                cur = pd.Timestamp(idle_calendar.iloc[param.idle - 1])
            while cur <= end:
                candidates = self.candidate_list(
                    cur.value,
                    param.scope,
                    param.ma,
                    param.mom,
                    param.model,
                    param.alpha,
                    param.min_history,
                    param.min_amount20,
                    param.vol_cap,
                    param.accel_cap,
                )
                if not candidates:
                    nxt = self.next_date(cur + pd.Timedelta(days=1))
                    if nxt is None:
                        break
                    cur = nxt
                    continue
                _, symbol, extra = candidates[0]
                df = self.panel[symbol]
                entry_pos = self.pos_after(df, cur + pd.Timedelta(days=1))
                end_pos = self.pos_before(df, end)
                if entry_pos is None or end_pos is None or entry_pos > end_pos:
                    break
                exit_pos = min(entry_pos + param.hold, end_pos)
                if exit_pos <= entry_pos:
                    break
                entry = df.iloc[entry_pos]
                exit_ = df.iloc[exit_pos]
                ret = float(exit_["close"] / entry["close"] - 1)
                rows.append(
                    {
                        "source": "空仓补偿_S1",
                        "source_type": "空仓补偿",
                        "category": self.meta[symbol]["category"],
                        "symbol": symbol,
                        "display_name": self.meta[symbol]["display_name"],
                        "decision_date": cur,
                        "entry_date": pd.Timestamp(entry["date"]),
                        "exit_date": pd.Timestamp(exit_["date"]),
                        "ret": ret,
                        "priority": 100.0,
                        "scope": param.scope,
                        "param_id": param.param_id,
                        "idle_days": param.idle,
                        "hold": param.hold,
                        "ma": param.ma,
                        "mom": param.mom,
                        "model": param.model,
                        "alpha": param.alpha,
                        "vol_cap": param.vol_cap,
                        "accel_cap": param.accel_cap,
                        **extra,
                    }
                )
                nxt = self.next_date(pd.Timestamp(exit_["date"]) + pd.Timedelta(days=1))
                if nxt is None:
                    break
                cur = nxt
        return pd.DataFrame(rows)

    def account_from_trades(self, trades: pd.DataFrame, initial_cash: float = 1000.0) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()
        df = trades.copy()
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        if "priority" not in df.columns:
            df["priority"] = 0.0
        df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(0.0)
        df = df.sort_values(["entry_date", "priority", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
        cash = initial_cash
        busy = pd.Timestamp.min
        rows = []
        for _, row in df.iterrows():
            if row["entry_date"] <= busy:
                continue
            before = cash
            cash *= 1.0 + float(row["ret"])
            busy = row["exit_date"]
            item = row.to_dict()
            item["cash_before"] = before
            item["cash_after"] = cash
            rows.append(item)
        return pd.DataFrame(rows)

    def _account_yearly(self, account: pd.DataFrame) -> pd.DataFrame:
        if account.empty:
            return pd.DataFrame()
        df = account.copy()
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["entry_year"] = df["entry_date"].dt.year
        rows = []
        for year in range(2014, 2027):
            grp = df[df["entry_year"] == year]
            strat_ret = float(np.prod(1.0 + grp["ret"].astype(float)) - 1.0) if len(grp) else 0.0
            rows.append(
                {
                    "year": year,
                    "trade_count": int(len(grp)),
                    "main_trades": int((grp.get("source_type", "") == "主策略").sum()) if len(grp) else 0,
                    "shadow_trades": int((grp.get("source_type", "") == "空仓补偿").sum()) if len(grp) else 0,
                    "win_rate": float((grp["ret"] > 0).mean()) if len(grp) else np.nan,
                    "strategy_return": strat_ret,
                    "broad_return": float(self.broad.get(year, np.nan)),
                    "excess_return": strat_ret - float(self.broad.get(year, np.nan)),
                    "underperform": bool(strat_ret < float(self.broad.get(year, np.nan))) if year in self.broad else False,
                    "ending_equity": float(grp["cash_after"].iloc[-1]) if len(grp) else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def param_yearly(self, trades: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if trades.empty:
            return pd.DataFrame()
        for param_id, grp in trades.groupby("param_id"):
            merged = pd.concat([self.main.copy(), grp.copy()], ignore_index=True, sort=False)
            account = self.account_from_trades(merged)
            yearly = self._account_yearly(account)
            yearly["param_id"] = param_id
            yearly["scope"] = grp["scope"].iloc[0]
            yearly["idle"] = int(grp["idle_days"].iloc[0])
            yearly["hold"] = int(grp["hold"].iloc[0])
            yearly["ma"] = int(grp["ma"].iloc[0])
            yearly["mom"] = int(grp["mom"].iloc[0])
            yearly["model"] = grp["model"].iloc[0]
            yearly["alpha"] = float(grp["alpha"].iloc[0])
            yearly["vol_cap"] = float(grp["vol_cap"].iloc[0])
            yearly["accel_cap"] = float(grp["accel_cap"].iloc[0])
            rows.append(yearly)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def training_score(self, yearly_slice: pd.DataFrame, trades_slice: pd.DataFrame, score_kind: str) -> tuple[float, dict]:
        if yearly_slice.empty:
            return -np.inf, {}
        excess_mean = float(yearly_slice["excess_return"].mean())
        under_ratio = float(yearly_slice["underperform"].mean())
        excess_std = float(yearly_slice["excess_return"].std(ddof=0)) if len(yearly_slice) > 1 else 0.0
        max_win = float(trades_slice["ret"].max()) if len(trades_slice) else 0.0
        max_loss = float(trades_slice["ret"].min()) if len(trades_slice) else 0.0
        if score_kind == "stable":
            score = excess_mean - 0.20 * under_ratio
        elif score_kind == "balanced":
            score = excess_mean - 0.20 * under_ratio - 0.10 * excess_std
        elif score_kind == "anti_conc":
            score = excess_mean - 0.20 * under_ratio - 0.20 * max(max_win, 0.0)
        elif score_kind == "robust":
            score = excess_mean - 0.20 * under_ratio - 0.10 * excess_std - 0.20 * max(max_win, 0.0)
        else:
            raise ValueError(f"unknown score kind: {score_kind}")
        meta = {
            "excess_mean": excess_mean,
            "under_ratio": under_ratio,
            "excess_std": excess_std,
            "max_win": max_win,
            "max_loss": max_loss,
        }
        return float(score), meta

    def training_candidates(
        self,
        yearly: pd.DataFrame,
        trades: pd.DataFrame,
        test_year: int,
        window_years: list[int],
        score_kind: str,
    ) -> pd.DataFrame:
        rows = []
        trades_local = trades.copy()
        trades_local["entry_year"] = pd.to_datetime(trades_local["entry_date"], errors="coerce").dt.year
        trade_groups = {param_id: grp for param_id, grp in trades_local.groupby("param_id", sort=False)}
        for param_id, grp in yearly.groupby("param_id"):
            window_scores = []
            activity_ratios = []
            avg_trade_counts = []
            recent_excesses = []
            recent_shadow_counts = []
            metas: list[dict] = []
            ok = True
            for years in window_years:
                train = grp[(grp["year"] >= test_year - years) & (grp["year"] < test_year)].sort_values("year")
                if train["year"].nunique() < years:
                    ok = False
                    break
                param_trades = trade_groups.get(param_id)
                train_trades = (
                    param_trades[param_trades["entry_year"].isin(train["year"])]
                    if param_trades is not None
                    else trades_local.iloc[0:0]
                )
                score, meta = self.training_score(train, train_trades, score_kind)
                window_scores.append(score)
                counts = train.groupby("year")["shadow_trades"].sum().astype(float)
                activity_ratios.append(float((counts > 0).mean()) if len(counts) else 0.0)
                avg_trade_counts.append(float(counts.mean()) if len(counts) else 0.0)
                recent_excesses.append(float(train.iloc[-1]["excess_return"]))
                recent_shadow_counts.append(float(train.iloc[-1]["shadow_trades"]))
                metas.append(meta)
            if not ok:
                continue
            base_row = grp.iloc[0]
            latest_meta = metas[-1]
            rows.append(
                {
                    "test_year": test_year,
                    "param_id": param_id,
                    "score": float(min(window_scores)),
                    "window_scores": json.dumps(window_scores, ensure_ascii=False),
                    "scope": base_row["scope"],
                    "idle": int(base_row["idle"]),
                    "hold": int(base_row["hold"]),
                    "ma": int(base_row["ma"]),
                    "mom": int(base_row["mom"]),
                    "model": base_row["model"],
                    "alpha": float(base_row["alpha"]),
                    "vol_cap": float(base_row["vol_cap"]),
                    "accel_cap": float(base_row["accel_cap"]),
                    "activity_ratio": float(np.mean(activity_ratios)) if activity_ratios else 0.0,
                    "avg_shadow_trades": float(np.mean(avg_trade_counts)) if avg_trade_counts else 0.0,
                    "recent_excess": float(recent_excesses[-1]) if recent_excesses else 0.0,
                    "recent_shadow_trades": float(recent_shadow_counts[-1]) if recent_shadow_counts else 0.0,
                    **latest_meta,
                }
            )
        return pd.DataFrame(rows)

    def select_params(
        self,
        yearly: pd.DataFrame,
        trades: pd.DataFrame,
        window_years: list[int],
        score_kind: str,
        test_years: list[int],
    ) -> pd.DataFrame:
        rows = []
        for test_year in test_years:
            cand = self.training_candidates(
                yearly=yearly,
                trades=trades,
                test_year=test_year,
                window_years=window_years,
                score_kind=score_kind,
            )
            if cand.empty:
                continue
            cand = cand.copy()
            cand["selector_score"] = (
                cand["score"]
                + 0.01 * cand["activity_ratio"]
                + 0.003 * cand["avg_shadow_trades"]
                - 0.0015 * cand["idle"]
            )
            ordered = cand.sort_values(
                ["selector_score", "score", "activity_ratio", "avg_shadow_trades", "under_ratio", "max_win", "excess_std", "hold", "ma", "mom", "idle", "param_id"],
                ascending=[False, False, False, False, True, True, True, False, False, False, True, True],
            )
            rows.append(ordered.iloc[0])
        return pd.DataFrame(rows)

    def select_params_formal(
        self,
        yearly: pd.DataFrame,
        trades: pd.DataFrame,
        test_years: list[int],
        formal_cfg: dict,
    ) -> pd.DataFrame:
        rows = []
        window_years = list(formal_cfg["window_years"])
        score_kind = str(formal_cfg["score_kind"])
        for test_year in test_years:
            cand = self.training_candidates(
                yearly=yearly,
                trades=trades,
                test_year=test_year,
                window_years=window_years,
                score_kind=score_kind,
            )
            if cand.empty:
                continue

            rows.append(self.pick_formal_year(cand=cand, formal_cfg=formal_cfg))
        return pd.DataFrame(rows)

    def pick_formal_year(self, cand: pd.DataFrame, formal_cfg: dict) -> dict:
        stable_scope = set(formal_cfg["stable_anchor_scopes"])
        stable_models = set(formal_cfg["stable_anchor_models"])
        stable_anchor = cand[
            cand["scope"].isin(stable_scope)
            & cand["model"].isin(stable_models)
            & (cand["hold"] == int(formal_cfg["hold"]))
            & (cand["ma"] == int(formal_cfg["ma"]))
            & (cand["mom"] == int(formal_cfg["mom"]))
            & (cand["idle"] <= int(formal_cfg["stable_max_idle"]))
            & (cand["vol_cap"] >= float(formal_cfg["stable_min_vol_cap"]))
        ].copy()
        if stable_anchor.empty:
            stable_anchor = cand.copy()

        stable_pref = list(formal_cfg.get("stable_model_preference", []))
        stable_anchor["model_rank"] = stable_anchor["model"].map(lambda x: _model_rank(x, stable_pref))
        anchor_recent_weight = float(formal_cfg.get("anchor_recent_weight", 0.0))
        cool_recent_cut = float(formal_cfg.get("anchor_cool_recent_excess", np.inf))
        low_vol_weight = float(formal_cfg.get("anchor_low_vol_weight", 0.0))
        high_vol_weight = float(formal_cfg.get("anchor_high_vol_weight", formal_cfg.get("anchor_vol_weight", 0.0)))
        stable_anchor["anchor_vol_term"] = np.where(
            stable_anchor["recent_excess"] <= cool_recent_cut,
            -stable_anchor["vol_cap"],
            stable_anchor["vol_cap"],
        )
        stable_anchor["anchor_score"] = (
            stable_anchor["score"]
            + float(formal_cfg["anchor_activity_weight"]) * stable_anchor["activity_ratio"]
            + float(formal_cfg["anchor_trade_weight"]) * stable_anchor["avg_shadow_trades"]
            - float(formal_cfg["anchor_idle_penalty"]) * stable_anchor["idle"]
            + anchor_recent_weight * stable_anchor["recent_excess"]
            + np.where(
                stable_anchor["recent_excess"] <= cool_recent_cut,
                low_vol_weight * stable_anchor["anchor_vol_term"],
                high_vol_weight * stable_anchor["anchor_vol_term"],
            )
        )
        anchor = stable_anchor.sort_values(
            ["anchor_score", "score", "alpha", "model_rank", "activity_ratio", "avg_shadow_trades", "vol_cap", "idle", "param_id"],
            ascending=[False, False, False, True, False, False, False, True, True],
        ).iloc[0]

        risk_scope = set(formal_cfg["risk_scopes"])
        risk_models = set(formal_cfg["risk_models"])
        risk_family = cand[
            cand["scope"].isin(risk_scope)
            & cand["model"].isin(risk_models)
            & (cand["hold"] == int(formal_cfg["hold"]))
            & (cand["ma"] == int(formal_cfg["ma"]))
            & (cand["mom"] == int(formal_cfg["mom"]))
            & (cand["idle"] <= int(formal_cfg["risk_max_idle"]))
            & (cand["vol_cap"] >= float(formal_cfg["risk_min_vol_cap"]))
        ].copy()

        chosen = anchor
        switch_reason = "anchor"
        if not risk_family.empty:
            risk_pref = list(formal_cfg.get("risk_model_preference", []))
            risk_family["model_rank"] = risk_family["model"].map(lambda x: _model_rank(x, risk_pref))
            risk_family["risk_score"] = (
                risk_family["score"]
                + float(formal_cfg["risk_activity_weight"]) * risk_family["activity_ratio"]
                + float(formal_cfg["risk_trade_weight"]) * risk_family["avg_shadow_trades"]
                - float(formal_cfg["risk_idle_penalty"]) * risk_family["idle"]
                + float(formal_cfg.get("risk_recent_weight", 0.0)) * risk_family["recent_excess"]
            )
            risk = risk_family.sort_values(
                ["risk_score", "score", "model_rank", "recent_excess", "activity_ratio", "avg_shadow_trades", "idle", "param_id"],
                ascending=[False, False, True, False, False, False, True, True],
            ).iloc[0]
            score_gap = float(anchor["score"] - risk["score"])
            should_switch = (
                float(risk["activity_ratio"]) >= float(formal_cfg["switch_min_activity"])
                and float(risk["recent_excess"]) >= float(formal_cfg["switch_min_risk_recent_excess"])
                and float(anchor["recent_excess"]) <= float(formal_cfg["switch_max_anchor_recent_excess"])
                and score_gap <= float(formal_cfg["switch_max_score_gap"])
                and float(risk["under_ratio"]) <= float(anchor["under_ratio"])
            )
            if should_switch:
                chosen = risk
                switch_reason = "risk_switch"

        picked = chosen.to_dict()
        picked["switch_reason"] = switch_reason
        picked["selector_family"] = "formal_staged"
        return picked

    def selected_account(self, selected: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        pieces = [self.pre_account.copy()]
        for _, row in selected.iterrows():
            year = int(row["test_year"])
            pieces.append(self.main[self.main["entry_date"].dt.year == year].copy())
            shadow = trades[(trades["param_id"] == row["param_id"]) & (pd.to_datetime(trades["entry_date"]).dt.year == year)].copy()
            pieces.append(shadow)
        return self.account_from_trades(pd.concat(pieces, ignore_index=True, sort=False))

    def summarize_account(self, account: pd.DataFrame) -> dict:
        if account.empty:
            return {"trades": 0, "final_cash": 1000.0}
        ret = account["ret"].astype(float)
        equity = account["cash_after"].astype(float)
        drawdown = equity / equity.cummax() - 1.0
        yearly = self._account_yearly(account)
        losses = ret[ret <= 0]
        wins = ret[ret > 0]
        return {
            "trades": int(len(account)),
            "final_cash": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1] / 1000.0 - 1.0),
            "win_rate": float((ret > 0).mean()),
            "avg_return": float(ret.mean()),
            "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan,
            "max_drawdown": float(drawdown.min()),
            "underperform_years": int(yearly["underperform"].fillna(False).sum()),
            "underperform_list": ",".join(str(y) for y in yearly.loc[yearly["underperform"], "year"].tolist()),
            "worst_excess": float(yearly["excess_return"].min()),
        }

    def top_trade_impact(self, account: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        if account.empty:
            return pd.DataFrame()
        full = float(account["cash_after"].iloc[-1])
        rows = []
        for idx, row in account.iterrows():
            if float(row["ret"]) <= 0:
                continue
            stripped = account.drop(index=idx).reset_index(drop=True)
            again = self.account_from_trades(stripped)
            ending = float(again["cash_after"].iloc[-1]) if not again.empty else 1000.0
            rows.append(
                {
                    "entry_date": pd.Timestamp(row["entry_date"]).date(),
                    "symbol": row["symbol"],
                    "display_name": row["display_name"],
                    "ret": float(row["ret"]),
                    "ending_without_trade": ending,
                    "drop_vs_full": full - ending,
                    "drop_ratio": (full - ending) / full if full > 0 else np.nan,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("drop_vs_full", ascending=False).head(top_n).reset_index(drop=True)

    def year_compare(self, account: pd.DataFrame, reference_yearly: pd.DataFrame, years: list[int]) -> pd.DataFrame:
        yearly = self._account_yearly(account)
        ref = reference_yearly[reference_yearly["year"].isin(years)][["year", "strategy_return"]].rename(columns={"strategy_return": "reference_return"})
        cur = yearly[yearly["year"].isin(years)][["year", "strategy_return"]].rename(columns={"strategy_return": "s1_return"})
        out = cur.merge(ref, on="year", how="left")
        out["difference"] = out["s1_return"] - out["reference_return"]
        return out.sort_values("year").reset_index(drop=True)

    def delay_stress(self, selected: pd.DataFrame, trades: pd.DataFrame, delay_days: list[int]) -> pd.DataFrame:
        rows = []
        base_account = self.selected_account(selected, trades)
        rows.append({"extra_delay_days": 0, **self.summarize_account(base_account)})
        for delay in delay_days:
            pieces = [self.pre_account.copy()]
            for _, sel in selected.iterrows():
                year = int(sel["test_year"])
                pieces.append(self.main[self.main["entry_date"].dt.year == year].copy())
                part = trades[(trades["param_id"] == sel["param_id"]) & (pd.to_datetime(trades["entry_date"]).dt.year == year)].copy()
                delayed_rows = []
                for _, tr in part.iterrows():
                    df = self.panel[tr["symbol"]]
                    entry_pos = self.pos_after(df, pd.Timestamp(tr["entry_date"]))
                    exit_pos = self.pos_after(df, pd.Timestamp(tr["exit_date"]))
                    if entry_pos is None or exit_pos is None:
                        continue
                    if entry_pos + delay >= exit_pos:
                        continue
                    new_entry = df.iloc[entry_pos + delay]
                    new_exit = df.iloc[exit_pos]
                    item = tr.copy()
                    item["entry_date"] = pd.Timestamp(new_entry["date"])
                    item["ret"] = float(new_exit["close"] / new_entry["close"] - 1.0)
                    delayed_rows.append(item)
                if delayed_rows:
                    pieces.append(pd.DataFrame(delayed_rows))
            account = self.account_from_trades(pd.concat(pieces, ignore_index=True, sort=False))
            rows.append({"extra_delay_days": delay, **self.summarize_account(account)})
        return pd.DataFrame(rows)

    def cost_stress(self, account: pd.DataFrame, extra_roundtrip_costs: list[float]) -> pd.DataFrame:
        rows = []
        for cost in extra_roundtrip_costs:
            cash = 1000.0
            series = []
            for ret in account["ret"].astype(float):
                cash *= (1.0 + ret) * (1.0 - cost)
                series.append(cash)
            rows.append({"extra_roundtrip_cost": cost, "final_cash": float(series[-1]) if series else 1000.0})
        return pd.DataFrame(rows)
