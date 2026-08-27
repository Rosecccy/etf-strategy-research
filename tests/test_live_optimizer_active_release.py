import csv
import json
from datetime import date, time
from pathlib import Path

from same_day_1445.live_optimizer.engine.active_policy import resolve_active_policy
from same_day_1445.live_optimizer.engine.signal_adapter import NormalizedSignal
from same_day_1445.live_optimizer.jobs.run_1445 import run_1445_cycle
from same_day_1445.live_optimizer.providers.file_provider import FileMinuteProvider


def _raw(path: Path, symbol: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ['date,symbol,name,open,high,low,close,volume,amount,pct_change,turnover_rate,source_file']
    for i in range(1, 23):
        rows.append(f'2026-07-{i:02d},{symbol},x,1,1,1,1,1,1,,,old')
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8-sig')


def _minute(root: Path, symbol: str, day: str):
    root.mkdir(parents=True, exist_ok=True)
    (root / f'{symbol}.csv').write_text(
        'timestamp,open,high,low,close,volume,amount\n'
        f'{day} 14:45:00,1,1,1,1,100,100\n',
        encoding='utf-8',
    )


def _runtime(tmp_path: Path):
    runtime = tmp_path / 'runtime'; feed = tmp_path / 'feed'; opt = tmp_path / 'opt'
    for symbol in ('510880', '510500'):
        _raw(runtime / 'C/raw/etf' / f'{symbol}.csv', symbol)
        _raw(runtime / 'S/raw/etf' / f'{symbol}.csv', symbol)
        _minute(feed, symbol, '2026-08-24')
    (runtime / 'R/formal').mkdir(parents=True, exist_ok=True)
    (runtime / 'R/formal/r_single_yearly_choice.csv').write_text(
        'year,policy_id,policy\n2026,base_CS,"{""policy_id"":""base_CS""}"\n',
        encoding='utf-8-sig',
    )
    return runtime, feed, opt


def _promoted_c_delay1(opt: Path) -> str:
    candidate = opt / 'candidates/C_DELAY1/manifest.json'
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps({
        'candidate_id': 'C_DELAY1', 'line': 'C', 'module': 'C_DELAY1', 'params': {'delay_days': 1}
    }), encoding='utf-8')
    release_id = 'auto_C_20260824T080000_C_DELAY1'
    release = opt / 'releases' / release_id / 'manifest.json'
    release.parent.mkdir(parents=True, exist_ok=True)
    release.write_text(json.dumps({
        'release_id': release_id, 'line': 'C', 'candidate_id': 'C_DELAY1', 'parent_release': 'release_v2'
    }), encoding='utf-8')
    (opt / 'state').mkdir(parents=True, exist_ok=True)
    (opt / 'state/optimizer_state.json').write_text(json.dumps({
        'mode': 'NORMAL',
        'active_formal': {'C': release_id, 'S': 'release_v2', 'D': 'release_v2', 'R': 'release_v2'},
        'previous_formal': {'C': 'release_v2', 'S': None, 'D': None, 'R': None},
        'shadow_leader': {'C': 'C_DELAY1', 'S': None, 'D': None, 'R': None},
        'last_promotion_at': {'C': '2026-08-23', 'S': None, 'D': None, 'R': None},
        'pending_promotion': {'C': None, 'S': None, 'D': None, 'R': None},
    }), encoding='utf-8')
    return release_id


def test_resolve_active_release_to_executable_candidate(tmp_path: Path):
    root = tmp_path / 'opt'; root.mkdir()
    release_id = _promoted_c_delay1(root)
    resolved = resolve_active_policy(root, 'C', release_id)
    assert resolved.release_id == release_id
    assert resolved.evidence_id == 'C_DELAY1'
    assert resolved.module == 'C_DELAY1'
    assert resolved.params == {'delay_days': 1}


def test_promoted_c_delay1_is_formal_signal_while_v2_anchor_still_runs(tmp_path: Path):
    runtime, feed, opt = _runtime(tmp_path)
    release_id = _promoted_c_delay1(opt)

    def runner(root, target, dpos):
        return {
            'C': NormalizedSignal('C', target, 'BUY', '510880', 'x', True, 'OK', 'same_day_preclose_buy', '', None, {}),
            'S': NormalizedSignal('S', target, 'HOLD', '', '', True, 'OK', 'wait_idle', '', None, {}),
            'D': NormalizedSignal('D', target, 'HOLD', '', '', True, 'OK', '空仓等待', '', None, {}),
        }, []

    result = run_1445_cycle(
        opt, runtime, FileMinuteProvider(feed), date(2026, 8, 24), time(14, 45), 1,
        adapter_runner=runner,
        shadow_config={'C_DELAY1': {'enabled': True}, 'D_STRICT': {'enabled': False}, 'D_GRID': {'enabled': False}},
    )
    assert result['signals']['C']['action'] == 'HOLD'
    assert result['signals']['C']['metadata']['active_release'] == release_id
    assert result['anchor_signals']['C']['action'] == 'BUY'

    with (opt / 'ledger/formal_signals.csv').open('r', encoding='utf-8-sig', newline='') as h:
        formal = list(csv.DictReader(h))
    c_formal = [row for row in formal if row['line'] == 'C'][-1]
    assert c_formal['candidate_id'] == release_id
    assert c_formal['action'] == 'HOLD'

    positions = json.loads((opt / 'state/forward_positions.json').read_text(encoding='utf-8'))
    assert 'C:release_v2' in positions
    assert 'C:C_DELAY1' not in positions


def test_optimizer_uses_promoted_candidate_as_current_formal_evidence(tmp_path: Path):
    from same_day_1445.live_optimizer.engine.ledger import AppendOnlyCsvLedger
    from same_day_1445.live_optimizer.engine.orchestrator import run_optimizer

    root = tmp_path / 'opt'; (root / 'config').mkdir(parents=True); (root / 'state').mkdir(); (root / 'ledger').mkdir()
    release_id = _promoted_c_delay1(root)
    cfg = {
        'mode': 'SHADOW_ONLY', 'cooldown_days': 0,
        'gates': {'trigger_retention_min': 0.0, 'win_rate_delta_min': -1.0, 'mean_return_delta_min': -1.0, 'max_drawdown_deterioration_max': 1.0, 'recent_windows_improve_fraction_min': 0.0, 'recent_window_win_rate_regression_max': 1.0, 'neighbor_pass_rate_min': 0.0, 'minimum_evaluation_windows': 0},
        'drift': {'recent_count': 20},
        'lines': {x: {'minimum_new_samples': 0, 'minimum_regimes': 0, 'accounting': 'research_sum' if x == 'D' else 'compound'} for x in 'CSDR'},
        'candidate_spaces': {}, 'evaluation_windows': []
    }
    (root / 'config/optimizer.json').write_text(json.dumps(cfg), encoding='utf-8')
    ledger = AppendOnlyCsvLedger(root / 'ledger/closed_trades.csv', ('sample_id', 'candidate_id'))
    for cid, ret in [('release_v2', 0.01), ('C_DELAY1', 0.02), ('C_CHALLENGER', 0.03)]:
        ledger.append({'sample_id': cid, 'opportunity_id': 'o1', 'line': 'C', 'candidate_id': cid, 'decision_at': '2026-08-24T15:00:00+08:00', 'observable_at': '2026-08-24T15:00:00+08:00', 'entry_date': '2026-08-01', 'exit_date': '2026-08-02', 'ret': ret, 'triggered': 1, 'regime': 'neutral'})
    result = run_optimizer(root, '2026-08-25T16:00:00+08:00')
    line = result['lines']['C']
    assert line['formal'] == release_id
    assert line['formal_evidence_id'] == 'C_DELAY1'
    ids = {item['candidate_id'] for item in line['candidates']}
    assert ids == {'C_CHALLENGER'}


def test_challenger_must_also_beat_frozen_v2_anchor(tmp_path: Path):
    from same_day_1445.live_optimizer.engine.ledger import AppendOnlyCsvLedger
    from same_day_1445.live_optimizer.engine.orchestrator import run_optimizer

    root = tmp_path / 'opt'; (root / 'config').mkdir(parents=True); (root / 'state').mkdir(); (root / 'ledger').mkdir()
    _promoted_c_delay1(root)
    cfg = {
        'mode': 'SHADOW_ONLY', 'cooldown_days': 0,
        'gates': {'trigger_retention_min': 0.0, 'win_rate_delta_min': 0.0, 'mean_return_delta_min': 0.0, 'max_drawdown_deterioration_max': 1.0, 'recent_windows_improve_fraction_min': 0.0, 'recent_window_win_rate_regression_max': 1.0, 'neighbor_pass_rate_min': 0.0, 'minimum_evaluation_windows': 0},
        'drift': {'recent_count': 20},
        'lines': {x: {'minimum_new_samples': 0, 'minimum_regimes': 0, 'accounting': 'research_sum' if x == 'D' else 'compound'} for x in 'CSDR'},
        'candidate_spaces': {}, 'evaluation_windows': []
    }
    (root / 'config/optimizer.json').write_text(json.dumps(cfg), encoding='utf-8')
    ledger = AppendOnlyCsvLedger(root / 'ledger/closed_trades.csv', ('sample_id', 'candidate_id'))
    for cid, ret in [('release_v2', 0.02), ('C_DELAY1', 0.00), ('C_CHALLENGER', 0.01)]:
        ledger.append({'sample_id': cid, 'opportunity_id': 'o1', 'line': 'C', 'candidate_id': cid, 'decision_at': '2026-08-24T15:00:00+08:00', 'observable_at': '2026-08-24T15:00:00+08:00', 'entry_date': '2026-08-01', 'exit_date': '2026-08-02', 'ret': ret, 'triggered': 1, 'regime': 'neutral'})
    result = run_optimizer(root, '2026-08-25T16:00:00+08:00')
    gate = result['lines']['C']['candidates'][0]['gate']
    assert 'V2_ANCHOR_MEAN_RETURN_DELTA' in gate['reasons']
    assert gate['evidence']['v2_anchor']['matched_opportunities'] == 1
