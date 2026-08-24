from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from .base import MinuteBar


class FileMinuteProvider:
    name = 'file'

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, symbol: str, trade_date: date) -> Path:
        dated = self.root / symbol / f'{trade_date.isoformat()}.csv'
        if dated.exists():
            return dated
        return self.root / f'{symbol}.csv'

    def fetch(self, symbol: str, trade_date: date) -> list[MinuteBar]:
        path = self._path(symbol, trade_date)
        if not path.exists():
            return []
        rows: list[MinuteBar] = []
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                ts = datetime.fromisoformat(str(row['timestamp']).replace(' ', 'T'))
                if ts.date() != trade_date:
                    continue
                rows.append(MinuteBar(
                    timestamp=ts,
                    open=float(row['open']), high=float(row['high']), low=float(row['low']), close=float(row['close']),
                    volume=float(row.get('volume') or 0.0), amount=float(row.get('amount') or 0.0),
                ))
        return sorted(rows, key=lambda x: x.timestamp)
