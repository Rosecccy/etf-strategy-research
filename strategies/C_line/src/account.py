from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from c2lib import ACCOUNT_DIR, ensure_dirs, read_etf


ACCOUNT_COLUMNS = [
    "symbol",
    "display_name",
    "category",
    "strategy_id",
    "entry_date",
    "entry_close",
    "shares",
    "cost",
    "note",
]
TRADE_COLUMNS = [
    "date",
    "action",
    "symbol",
    "display_name",
    "category",
    "strategy_id",
    "price",
    "shares",
    "fee",
    "amount",
    "note",
]


def account_path():
    return ACCOUNT_DIR / "account.csv"


def trades_path():
    return ACCOUNT_DIR / "trades.csv"


def init_files(force: bool = False) -> None:
    ensure_dirs()
    if force or not account_path().exists():
        pd.DataFrame(columns=ACCOUNT_COLUMNS).to_csv(account_path(), index=False, encoding="utf-8-sig")
    if force or not trades_path().exists():
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(trades_path(), index=False, encoding="utf-8-sig")


def load_account() -> pd.DataFrame:
    init_files()
    return pd.read_csv(account_path(), dtype={"symbol": str})


def load_trades() -> pd.DataFrame:
    init_files()
    return pd.read_csv(trades_path(), dtype={"symbol": str})


def latest_close(symbol: str) -> tuple[str, float]:
    df = read_etf(symbol)
    row = df.iloc[-1]
    return row["date"].strftime("%Y-%m-%d"), float(row["close"])


def add_trade(args) -> None:
    init_files()
    account = load_account()
    trades = load_trades()
    symbol = str(args.symbol).zfill(6)
    amount = float(args.price) * float(args.shares)
    record = {
        "date": args.date,
        "action": args.action,
        "symbol": symbol,
        "display_name": args.name or "",
        "category": args.category or "",
        "strategy_id": args.strategy_id or "",
        "price": float(args.price),
        "shares": float(args.shares),
        "fee": float(args.fee),
        "amount": amount,
        "note": args.note or "",
    }
    trades = pd.concat([trades, pd.DataFrame([record])], ignore_index=True)
    trades.to_csv(trades_path(), index=False, encoding="utf-8-sig")

    if args.action == "buy":
        position = {
            "symbol": symbol,
            "display_name": args.name or "",
            "category": args.category or "",
            "strategy_id": args.strategy_id or "",
            "entry_date": args.date,
            "entry_close": float(args.price),
            "shares": float(args.shares),
            "cost": amount + float(args.fee),
            "note": args.note or "",
        }
        account = account[account["symbol"].astype(str).str.zfill(6) != symbol]
        account = pd.concat([account, pd.DataFrame([position])], ignore_index=True)
    elif args.action == "sell":
        account = account[account["symbol"].astype(str).str.zfill(6) != symbol]
    else:
        raise ValueError(f"unknown action: {args.action}")
    account.to_csv(account_path(), index=False, encoding="utf-8-sig")


def reconcile(cash: float | None = None) -> dict:
    account = load_account()
    rows = []
    total_market_value = 0.0
    for _, row in account.dropna(subset=["symbol"]).iterrows():
        symbol = str(row["symbol"]).zfill(6)
        date, price = latest_close(symbol)
        shares = float(row.get("shares", 0) or 0)
        cost = float(row.get("cost", 0) or 0)
        market_value = shares * price
        total_market_value += market_value
        rows.append(
            {
                "symbol": symbol,
                "display_name": row.get("display_name", ""),
                "category": row.get("category", ""),
                "entry_date": row.get("entry_date", ""),
                "entry_close": row.get("entry_close", ""),
                "shares": shares,
                "latest_date": date,
                "latest_close": price,
                "cost": cost,
                "market_value": market_value,
                "unrealized_pnl": market_value - cost,
                "unrealized_return": (market_value / cost - 1) if cost else None,
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(ACCOUNT_DIR / "reconcile_positions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cash": cash,
        "position_count": int(len(detail)),
        "market_value": float(total_market_value),
        "total_equity": float(total_market_value + cash) if cash is not None else float(total_market_value),
    }
    (ACCOUNT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="C2 account state tools.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")

    add = sub.add_parser("add", help="Add a manual buy/sell trade.")
    add.add_argument("--date", required=True)
    add.add_argument("--action", required=True, choices=["buy", "sell"])
    add.add_argument("--symbol", required=True)
    add.add_argument("--price", required=True, type=float)
    add.add_argument("--shares", required=True, type=float)
    add.add_argument("--fee", default=0.0, type=float)
    add.add_argument("--name", default="")
    add.add_argument("--category", default="")
    add.add_argument("--strategy-id", default="")
    add.add_argument("--note", default="")

    rec = sub.add_parser("reconcile")
    rec.add_argument("--cash", type=float, default=None)

    args = parser.parse_args()
    if args.cmd == "init":
        init_files()
        print(json.dumps({"account": str(account_path()), "trades": str(trades_path())}, ensure_ascii=False, indent=2))
    elif args.cmd == "add":
        add_trade(args)
        print(json.dumps({"status": "ok", "account": str(account_path()), "trades": str(trades_path())}, ensure_ascii=False, indent=2))
    elif args.cmd == "reconcile":
        print(json.dumps(reconcile(args.cash), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
