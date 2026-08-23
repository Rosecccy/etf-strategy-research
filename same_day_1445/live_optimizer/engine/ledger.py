from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


class LedgerConflictError(RuntimeError):
    pass


def _canon(row: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if isinstance(value, bool):
            out[str(key)] = "True" if value else "False"
        elif value is None:
            out[str(key)] = ""
        else:
            out[str(key)] = str(value)
    return out


class AppendOnlyCsvLedger:
    def __init__(self, path: Path, key_fields: Iterable[str]):
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        if not self.key_fields:
            raise ValueError("key_fields must not be empty")

    def rows(self) -> list[dict[str, str]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(field, "")) for field in self.key_fields)

    def append(self, row: dict[str, Any]) -> bool:
        canonical = _canon(row)
        if any(field not in canonical for field in self.key_fields):
            raise ValueError(f"missing ledger key fields: {self.key_fields}")
        existing = self.rows()
        key = self._key(canonical)
        for old in existing:
            if self._key(old) == key:
                if old == canonical:
                    return False
                raise LedgerConflictError(f"append-only conflict for key {key}")
        fieldnames = list(existing[0].keys()) if existing else list(canonical.keys())
        if set(canonical) != set(fieldnames):
            if existing:
                raise LedgerConflictError("ledger schema change is not allowed")
            fieldnames = list(canonical.keys())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        all_rows = existing + [canonical]
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return True
