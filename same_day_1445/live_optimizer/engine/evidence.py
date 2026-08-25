from __future__ import annotations

from datetime import date
from typing import Any


def _row_date(row: dict[str, Any]) -> date | None:
    value = str(row.get("entry_date") or row.get("decision_date") or row.get("decision_at") or "")[:10]
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def filter_evidence_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    start = str(config.get("evidence_start_date") or "").strip()[:10]
    if not start:
        return list(rows)
    cutoff = date.fromisoformat(start)
    return [row for row in rows if (d := _row_date(row)) is not None and d >= cutoff]
