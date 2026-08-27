from pathlib import Path
import pytest

from same_day_1445.live_optimizer.engine.ledger import AppendOnlyCsvLedger, LedgerConflictError
from same_day_1445.live_optimizer.engine.audit import audit_causal_rows, hash_file


def test_append_only_ledger_is_idempotent_but_rejects_rewrite(tmp_path: Path):
    ledger = AppendOnlyCsvLedger(tmp_path/'signals.csv', key_fields=('sample_id',))
    row = {'sample_id':'x1','line':'C','action':'BUY'}
    assert ledger.append(row) is True
    assert ledger.append(dict(row)) is False
    with pytest.raises(LedgerConflictError):
        ledger.append({'sample_id':'x1','line':'C','action':'SELL'})
    assert ledger.rows() == [row]


def test_causal_audit_rejects_future_observable_evidence():
    rows = [
        {'sample_id':'ok','decision_at':'2026-08-23T14:45:00+08:00','observable_at':'2026-08-23T14:44:00+08:00'},
        {'sample_id':'bad','decision_at':'2026-08-23T14:45:00+08:00','observable_at':'2026-08-23T15:00:00+08:00'},
    ]
    result = audit_causal_rows(rows, 'decision_at', 'observable_at')
    assert result['ok'] is False
    assert [v['sample_id'] for v in result['violations']] == ['bad']


def test_hash_file_is_deterministic(tmp_path: Path):
    path = tmp_path/'a.txt'; path.write_text('abc', encoding='utf-8')
    assert hash_file(path) == hash_file(path)
