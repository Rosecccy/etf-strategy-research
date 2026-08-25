from __future__ import annotations

from datetime import date

from .eastmoney import EastmoneyMinuteProvider
from .tencent import TencentMinuteProvider


class AutoMinuteProvider:
    """Prefer Tencent current-day minute history and fall back to Eastmoney."""

    name = "auto_1m"

    def __init__(self, tencent=None, eastmoney=None):
        self.tencent = tencent or TencentMinuteProvider()
        self.eastmoney = eastmoney or EastmoneyMinuteProvider()

    def fetch(self, symbol: str, trade_date: date):
        last_exc: Exception | None = None
        for provider in (self.tencent, self.eastmoney):
            try:
                bars = provider.fetch(symbol, trade_date)
                if bars:
                    return bars
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return []
