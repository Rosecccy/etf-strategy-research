from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest


def test_eastmoney_volume_is_normalized_from_lots_to_shares(monkeypatch):
    import same_day_1445.live_optimizer.providers.eastmoney as mod

    payload = {
        'data': {
            'klines': [
                '2026-08-25 14:45,3.40,3.41,3.42,3.39,1234,420000.0,0'
            ]
        }
    }

    class Resp:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(payload).encode('utf-8')

    monkeypatch.setattr(mod, 'urlopen', lambda *a, **k: Resp())
    bars = mod.EastmoneyMinuteProvider().fetch('510880', date(2026, 8, 25))
    assert len(bars) == 1
    assert bars[0].volume == 123400.0
    assert bars[0].amount == 420000.0


def test_tencent_provider_parses_current_day_m1_and_normalizes_volume(monkeypatch):
    import same_day_1445.live_optimizer.providers.tencent as mod

    mk_payload = {
        'data': {
            'sh510880': {
                'm1': [
                    ['202608251444', '3.40', '3.41', '3.42', '3.39', '100'],
                    ['202608251445', '3.41', '3.42', '3.43', '3.40', '120'],
                ]
            }
        }
    }
    minute_payload = {
        'data': {
            'sh510880': {
                'date': '20260825',
                'data': {'data': ['1444 3.41 100 34100', '1445 3.42 220 75140']},
            }
        }
    }

    def fake_json(urls):
        url = urls[0] if isinstance(urls, list) else urls
        return minute_payload if 'minute/query' in url else mk_payload

    monkeypatch.setattr(mod, '_request_json', fake_json)
    bars = mod.TencentMinuteProvider().fetch('510880', date(2026, 8, 25))
    assert [b.timestamp.isoformat() for b in bars] == [
        '2026-08-25T14:44:00', '2026-08-25T14:45:00'
    ]
    assert [b.volume for b in bars] == [10000.0, 12000.0]
    assert [b.amount for b in bars] == [34100.0, 41040.0]


def test_auto_provider_prefers_tencent_and_falls_back_to_eastmoney():
    from same_day_1445.live_optimizer.providers.auto import AutoMinuteProvider
    from same_day_1445.live_optimizer.providers.base import MinuteBar

    bar = MinuteBar(datetime(2026, 8, 25, 14, 45), 1, 1, 1, 1, 100, 100)

    class Provider:
        def __init__(self, name, result=None, exc=None):
            self.name = name
            self.result = result
            self.exc = exc
            self.calls = 0
        def fetch(self, symbol, trade_date):
            self.calls += 1
            if self.exc:
                raise self.exc
            return list(self.result or [])

    tencent = Provider('tencent', [bar])
    eastmoney = Provider('eastmoney', [bar])
    auto = AutoMinuteProvider(tencent=tencent, eastmoney=eastmoney)
    assert auto.fetch('510880', date(2026, 8, 25)) == [bar]
    assert tencent.calls == 1 and eastmoney.calls == 0

    tencent = Provider('tencent', exc=RuntimeError('down'))
    eastmoney = Provider('eastmoney', [bar])
    auto = AutoMinuteProvider(tencent=tencent, eastmoney=eastmoney)
    assert auto.fetch('510880', date(2026, 8, 25)) == [bar]
    assert tencent.calls == 1 and eastmoney.calls == 1


def test_provider_factory_supports_auto_tencent_eastmoney_and_file(tmp_path: Path):
    from same_day_1445.live_optimizer.providers.factory import build_provider
    from same_day_1445.live_optimizer.providers.auto import AutoMinuteProvider
    from same_day_1445.live_optimizer.providers.tencent import TencentMinuteProvider
    from same_day_1445.live_optimizer.providers.eastmoney import EastmoneyMinuteProvider
    from same_day_1445.live_optimizer.providers.file_provider import FileMinuteProvider

    assert isinstance(build_provider({'kind': 'auto'}), AutoMinuteProvider)
    assert isinstance(build_provider({'kind': 'tencent'}), TencentMinuteProvider)
    assert isinstance(build_provider({'kind': 'eastmoney'}), EastmoneyMinuteProvider)
    assert isinstance(build_provider({'kind': 'file', 'file_root': str(tmp_path)}), FileMinuteProvider)
    with pytest.raises(ValueError, match='provider'):
        build_provider({'kind': 'unknown'})


def test_bootstrap_defaults_to_auto_and_requires_lightgbm(tmp_path: Path):
    from same_day_1445.live_optimizer.ops.bootstrap import bootstrap_runtime

    source = tmp_path / 'source'
    runtime = tmp_path / 'runtime'
    opt = tmp_path / 'opt'
    source.mkdir(); opt.mkdir()
    for rel in ['C/src/run_daily.py','S/src/run_s1_live.py','D/src/daily_panic_live.py','R/formal/r_single_yearly_choice.csv']:
        p = source / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel, encoding='utf-8')
    bootstrap_runtime(source, runtime, opt)
    cfg = json.loads((opt/'runtime/workspace.json').read_text(encoding='utf-8'))
    assert cfg['provider']['kind'] == 'auto'
    assert 'lightgbm' in cfg['required_python_modules']


def test_evidence_start_date_excludes_debug_rows():
    from same_day_1445.live_optimizer.engine.evidence import filter_evidence_rows

    rows = [
        {'entry_date': '2026-08-24', 'sample_id': 'debug1'},
        {'entry_date': '2026-08-25', 'sample_id': 'debug2'},
        {'entry_date': '2026-08-26', 'sample_id': 'clean1'},
    ]
    kept = filter_evidence_rows(rows, {'evidence_start_date': '2026-08-26'})
    assert [r['sample_id'] for r in kept] == ['clean1']


def test_daily_drift_honors_evidence_start_date(tmp_path: Path):
    from same_day_1445.live_optimizer.engine.daily_monitor import detect_daily_signal_drift

    root = tmp_path
    (root/'ledger').mkdir()
    (root/'ledger/formal_signals.csv').write_text(
        'signal_id,line,decision_date,action,tradable,status,observed_at\n'
        'a,C,2026-08-24,BUY,1,OK,2026-08-24T14:45:00+08:00\n'
        'b,C,2026-08-26,HOLD,1,OK,2026-08-26T14:45:00+08:00\n',
        encoding='utf-8',
    )
    cfg = {
        'evidence_start_date': '2026-08-26',
        'drift': {'daily_recent_days': 1, 'daily_min_reference_days': 1},
    }
    result = detect_daily_signal_drift(root, 'C', cfg)
    assert result['eligible'] is False
    assert result['recent']['days'] == 1.0


def test_gitignore_excludes_runtime_recovery_and_pytest_tmp():
    text = (Path(__file__).resolve().parents[1]/'.gitignore').read_text(encoding='utf-8')
    assert '.pytest_tmp/' in text
    assert '/same_day_1445/live_optimizer/recovery/' in text


def test_requirements_include_lightgbm():
    lines = {
        x.strip().lower()
        for x in (Path(__file__).resolve().parents[1]/'requirements.txt').read_text(encoding='utf-8').splitlines()
        if x.strip()
    }
    assert 'lightgbm' in lines


def test_all_optimizer_text_writes_declare_encoding():
    import ast
    root = Path(__file__).resolve().parents[1]/'same_day_1445/live_optimizer'
    offenders = []
    for path in root.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != 'write_text':
                continue
            if not any(k.arg == 'encoding' for k in node.keywords):
                offenders.append(f'{path.relative_to(root)}:{node.lineno}')
    assert offenders == []


def test_closed_trade_ledger_filters_debug_epoch_without_dropping_rows_on_append(tmp_path: Path):
    from same_day_1445.live_optimizer.engine.ledger import AppendOnlyCsvLedger

    root = tmp_path / 'opt'
    (root / 'config').mkdir(parents=True)
    (root / 'config' / 'optimizer.json').write_text(
        json.dumps({'evidence_start_date': '2026-08-26'}), encoding='utf-8'
    )
    ledger = AppendOnlyCsvLedger(root / 'ledger' / 'closed_trades.csv', ('sample_id', 'candidate_id'))
    ledger.append({'sample_id': 'old', 'candidate_id': 'release_v2', 'entry_date': '2026-08-25', 'line': 'C'})
    ledger.append({'sample_id': 'new', 'candidate_id': 'release_v2', 'entry_date': '2026-08-26', 'line': 'C'})
    assert [r['sample_id'] for r in ledger.rows()] == ['new']
    raw = (root / 'ledger' / 'closed_trades.csv').read_text(encoding='utf-8-sig')
    assert 'old' in raw and 'new' in raw
