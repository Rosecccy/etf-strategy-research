from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True)
class MinuteBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


class MinuteProvider(Protocol):
    name: str

    def fetch(self, symbol: str, trade_date: date) -> list[MinuteBar]: ...
