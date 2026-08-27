import json
from pathlib import Path

from same_day_1445.live_optimizer.engine.ledger import AppendOnlyCsvLedger
from same_day_1445.live_optimizer.engine.orchestrator import run_optimizer


def test_promoted_release_rolls_back_after_sufficient_new_live_deterioration(tmp_path: Path):
    root = tmp_path / 'opt'; (root / 'config').mkdir(parents=True); (root / 'state').mkdir(); (root / 'ledger').mkdir()
    candidate_id = 'C_DELAY1'; release_id = 'auto_C_20260820T160000_C_DELAY1'
    (root / f'candidates/{candidate_id}').mkdir(parents=True)
    (root / f'candidates/{candidate_id}/manifest.json').write_text(json.dumps({'candidate_id': candidate_id, 'line': 'C', 'module': 'C_DELAY1', 'params': {'delay_days': 1}}), encoding='utf-8')
    (root / f'releases/{release_id}').mkdir(parents=True)
    (root / f'releases/{release_id}/manifest.json').write_text(json.dumps({'release_id': release_id, 'line': 'C', 'candidate_id': candidate_id, 'parent_release': 'release_v2'}), encoding='utf-8')
    cfg = {
        'mode': 'NORMAL', 'cooldown_days': 0,
        'gates': {'trigger_retention_min': 0.0, 'win_rate_delta_min': 0.0, 'mean_return_delta_min': 0.0, 'max_drawdown_deterioration_max': 1.0, 'recent_windows_improve_fraction_min': 0.0, 'recent_window_win_rate_regression_max': 1.0, 'neighbor_pass_rate_min': 0.0, 'minimum_evaluation_windows': 0},
        'rollback': {'enabled': True, 'minimum_new_samples': {'C': 3, 'S': 3, 'D': 3, 'R': 3}, 'trigger_retention_min': 0.5, 'win_rate_delta_min': -0.20, 'mean_return_delta_min': -0.01, 'max_drawdown_deterioration_max': 0.20, 'breach_count_min': 2},
        'drift': {'recent_count': 20},
        'lines': {x: {'minimum_new_samples': 999, 'minimum_regimes': 0, 'accounting': 'research_sum' if x == 'D' else 'compound'} for x in 'CSDR'},
        'candidate_spaces': {}, 'evaluation_windows': []
    }
    (root / 'config/optimizer.json').write_text(json.dumps(cfg), encoding='utf-8')
    (root / 'state/optimizer_state.json').write_text(json.dumps({
        'mode': 'NORMAL',
        'active_formal': {'C': release_id, 'S': 'release_v2', 'D': 'release_v2', 'R': 'release_v2'},
        'previous_formal': {'C': 'release_v2', 'S': None, 'D': None, 'R': None},
        'shadow_leader': {'C': candidate_id, 'S': None, 'D': None, 'R': None},
        'last_promotion_at': {'C': '2026-08-20T16:00:00+08:00', 'S': None, 'D': None, 'R': None},
        'pending_promotion': {x: None for x in 'CSDR'},
    }), encoding='utf-8')
    led = AppendOnlyCsvLedger(root / 'ledger/closed_trades.csv', ('sample_id', 'candidate_id'))
    for i in range(3):
        obs = f'2026-08-{21+i:02d}T15:00:00+08:00'
        common = {'opportunity_id': f'o{i}', 'line': 'C', 'decision_at': obs, 'observable_at': obs, 'entry_date': f'2026-08-{21+i:02d}', 'exit_date': f'2026-08-{21+i:02d}', 'triggered': 1, 'regime': 'neutral'}
        led.append({'sample_id': f'v{i}', 'candidate_id': 'release_v2', 'ret': 0.02, **common})
        led.append({'sample_id': f'a{i}', 'candidate_id': candidate_id, 'ret': -0.03, **common})
    result = run_optimizer(root, '2026-08-24T16:30:00+08:00')
    state = json.loads((root / 'state/optimizer_state.json').read_text(encoding='utf-8'))
    assert state['active_formal']['C'] == 'release_v2'
    assert result['lines']['C']['rollback']['status'] == 'ROLLED_BACK'
    assert 'WIN_RATE' in result['lines']['C']['rollback']['breaches']
    assert 'MEAN_RETURN' in result['lines']['C']['rollback']['breaches']
    text = (root / 'ledger/promotion.csv').read_text(encoding='utf-8')
    assert 'ROLLBACK' in text
