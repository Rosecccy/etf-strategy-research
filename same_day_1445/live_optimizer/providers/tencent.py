from __future__ import annotations

import json
from datetime import date, datetime, time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .base import MinuteBar


def _code(symbol: str) -> str:
    symbol = str(symbol).zfill(6)
    return f"sh{symbol}" if symbol.startswith(("5", "6")) else f"sz{symbol}"


def _request_json(urls: str | list[str]) -> dict:
    candidates = [urls] if isinstance(urls, str) else list(urls)
    last_exc: Exception | None = None
    for url in candidates:
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://gu.qq.com/",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            raw = urlopen(req, timeout=15).read().decode("utf-8", "ignore").strip()
            if raw and not raw.startswith("{") and "=" in raw:
                raw = raw.split("=", 1)[1].strip().rstrip(";")
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover - exercised through provider fallback
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no Tencent URL attempted")


def _parse_m1_timestamp(value: object, trade_date: date) -> datetime:
    text = str(value).strip()
    for fmt in ("%Y%m%d%H%M", "%Y%m%d %H%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    if len(text) == 4 and text.isdigit():
        return datetime.combine(trade_date, time(int(text[:2]), int(text[2:])))
    raise ValueError(f"unrecognized Tencent minute timestamp: {value!r}")


class TencentMinuteProvider:
    """Current-day Tencent 1-minute provider.

    Tencent reports ETF minute volume in lots. The optimizer's runtime contract is
    shares, so volume is normalized by multiplying by 100. Turnover amount comes
    from Tencent's cumulative minute-query series and is converted to per-minute
    deltas before MinuteBar objects are returned.
    """

    name = "tencent_1m"

    def fetch(self, symbol: str, trade_date: date) -> list[MinuteBar]:
        code = _code(symbol)
        mk_param = f"{code},m1,,320"
        mk_urls = [
            "https://web.ifzq.gtimg.cn/appstock/app/kline/mkline?" + urlencode({"param": mk_param}),
            "https://ifzq.gtimg.cn/appstock/app/kline/mkline?" + urlencode({"param": mk_param}),
        ]
        mk = _request_json(mk_urls)
        node = (mk.get("data") or {}).get(code) or {}
        raw_bars = node.get("m1") or []

        q_url = "https://web.ifzq.gtimg.cn/appstock/app/minute/query?" + urlencode({"code": code})
        q = _request_json(q_url)
        qnode = (q.get("data") or {}).get(code) or {}
        qdate = str(qnode.get("date") or "")
        if qdate and qdate != trade_date.strftime("%Y%m%d"):
            return []
        qitems = ((qnode.get("data") or {}).get("data") or [])

        cumulative: dict[datetime, tuple[float, float]] = {}
        for item in qitems:
            parts = str(item).split()
            if len(parts) < 3:
                continue
            hhmm = parts[0]
            if len(hhmm) != 4 or not hhmm.isdigit():
                continue
            ts = datetime.combine(trade_date, time(int(hhmm[:2]), int(hhmm[2:])))
            cum_volume_lots = float(parts[2] or 0.0)
            cum_amount = float(parts[3] or 0.0) if len(parts) >= 4 and parts[3] else 0.0
            cumulative[ts] = (cum_volume_lots, cum_amount)

        parsed: list[tuple[datetime, float, float, float, float, float]] = []
        for item in raw_bars:
            if len(item) < 6:
                continue
            ts = _parse_m1_timestamp(item[0], trade_date)
            if ts.date() != trade_date:
                continue
            parsed.append((ts, float(item[1]), float(item[3]), float(item[4]), float(item[2]), float(item[5])))
        parsed.sort(key=lambda x: x[0])

        out: list[MinuteBar] = []
        prev_cum_amount = 0.0
        for ts, open_, high, low, close, volume_lots in parsed:
            amount = 0.0
            if ts in cumulative:
                _, cum_amount = cumulative[ts]
                # Use cumulative deltas when available. The m1 volume is retained
                # as the source of truth for minute volume because its timestamps
                # line up exactly with the OHLC bars.
                amount = max(0.0, cum_amount - prev_cum_amount)
                prev_cum_amount = cum_amount
            else:
                amount = volume_lots * 100.0 * close
            out.append(MinuteBar(ts, open_, high, low, close, volume_lots * 100.0, amount))
        return out
