from pathlib import Path
import json

from same_day_1445.live_optimizer.engine.orchestrator import run_optimizer
from same_day_1445.live_optimizer.engine.ledger import AppendOnlyCsvLedger


def build_root(tmp_path: Path) -> Path:
    root = tmp_path / 'live_optimizer'
    (root / 'config').mkdir(parents=True)
    (root / 'state').mkdir()
    (root / 'ledger').mkdir()
    (root / 'config' / 'optimizer.json').write_text(json.dumps({
        'mode': 'SHADOW_ONLY',
        'gates': {'trigger_retention_min': 0.90, 'win_rate_delta_min': 0.005, 'mean_return_delta_min': 0.0, 'max_drawdown_deterioration_max': 0.01, 'recent_windows_improve_fraction_min': 0.6666666667, 'recent_window_win_rate_regression_max': 0.0, 'neighbor_pass_rate_min': 0.5},
        'lines': {'C': {'minimum_new_samples': 2, 'minimum_regimes': 1, 'accounting': 'compound'}, 'S': {'minimum_new_samples': 2, 'minimum_regimes': 1, 'accounting': 'compound'}, 'D': {'minimum_new_samples': 2, 'minimum_regimes': 1, 'accounting': 'research_sum'}, 'R': {'minimum_new_samples': 2, 'minimum_regimes': 1, 'accounting': 'compound'}},
        'cooldown_days': 0,
        'candidate_spaces': {'C': {'delay_days': [0, 1]}},
        'drift': {'trigger_rate_abs_max': 0.30, 'win_rate_abs_max': 0.30, 'mean_return_abs_max': 0.10, 'recent_count': 2},
        'evaluation_windows': [{'name': 'w1', 'start': '2026-01-01', 'end': '2026-12-31'}]
    }), encoding='utf-8')
    (root / 'state' / 'optimizer_state.json').write_text(json.dumps({'mode': 'SHADOW_ONLY', 'active_formal': {'C': 'release_v2', 'S': 'release_v2', 'D': 'release_v2', 'R': 'release_v2'}, 'previous_formal': {'C': None, 'S': None, 'D': None, 'R': None}, 'shadow_leader': {'C': None, 'S': None, 'D': None, 'R': None}, 'last_promotion_at': {'C': None, 'S': None, 'D': None, 'R': None}}), encoding='utf-8')
    ledger = AppendOnlyCsvLedger(root / 'ledger' / 'closed_trades.csv', ('sample_id', 'candidate_id'))
    for sid, ret in [('b1', 0.01), ('b2', -0.01), ('b3', 0.01), ('b4', -0.01)]:
        ledger.append({'sample_id': sid, 'line': 'C', 'candidate_id': 'release_v2', 'decision_at': '2026-02-01T14:45:00+08:00', 'observable_at': '2026-02-01T14:44:00+08:00', 'entry_date': '2026-02-01', 'exit_date': '2026-02-02', 'ret': ret, 'triggered': True, 'regime': 'bull'})
    for sid, ret in [('c1', 0.02), ('c2', 0.01), ('c3', 0.02), ('c4', -0.005)]:
        ledger.append({'sample_id': sid, 'line': 'C', 'candidate_id': 'C_DELAY1', 'decision_at': '2026-02-01T14:45:00+08:00', 'observable_at': '2026-02-01T14:44:00+08:00', 'entry_date': '2026-02-01', 'exit_date': '2026-02-02', 'ret': ret, 'triggered': True, 'regime': 'bull'})
    return root


def test_daily_optimizer_can_update_shadow_without_formal_promotion(tmp_path: Path):
    root = build_root(tmp_path)
    result = run_optimizer(root, '2026-08-23T16:00:00+08:00')
    state = json.loads((root / 'state' / 'optimizer_state.json').read_text(encoding='utf-8'))
    assert state['active_formal']['C'] == 'release_v2'
    assert state['shadow_leader']['C'] == 'C_DELAY1'
    assert result['lines']['C']['shadow_leader'] == 'C_DELAY1'


def test_repeated_run_is_deterministic(tmp_path: Path):
    root = build_root(tmp_path)
    a = run_optimizer(root, '2026-08-23T16:00:00+08:00')
    b = run_optimizer(root, '2026-08-23T16:00:00+08:00')
    assert a['decision_hash'] == b['decision_hash']


def test_severe_live_drift_forces_shadow_only(tmp_path: Path):
    root = build_root(tmp_path)
    cfg = json.loads((root / 'config' / 'optimizer.json').read_text(encoding='utf-8'))
    cfg['mode'] = 'NORMAL'
    (root / 'config' / 'optimizer.json').write_text(json.dumps(cfg), encoding='utf-8')
    state = json.loads((root / 'state' / 'optimizer_state.json').read_text(encoding='utf-8'))
    state['mode'] = 'NORMAL'
    (root / 'state' / 'optimizer_state.json').write_text(json.dumps(state), encoding='utf-8')
    (root / 'ledger' / 'closed_trades.csv').unlink()
    ledger = AppendOnlyCsvLedger(root / 'ledger' / 'closed_trades.csv', ('sample_id', 'candidate_id'))
    values = [('r1', True, 0.10), ('r2', True, 0.10), ('r3', False, -0.10), ('r4', False, -0.10)]
    for sid, triggered, ret in values:
        ledger.append({'sample_id': sid, 'line': 'C', 'candidate_id': 'release_v2', 'decision_at': '2026-02-01T14:45:00+08:00', 'observable_at': '2026-02-01T14:44:00+08:00', 'entry_date': '2026-02-01', 'exit_date': '2026-02-02', 'ret': ret, 'triggered': triggered, 'regime': 'bull'})
    result = run_optimizer(root, '2026-08-23T16:00:00+08:00')
    state = json.loads((root / 'state' / 'optimizer_state.json').read_text(encoding='utf-8'))
    assert state['mode'] == 'SHADOW_ONLY'
    assert result['lines']['C']['drift']['severe'] is True
