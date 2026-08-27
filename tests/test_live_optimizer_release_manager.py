from pathlib import Path
import pytest

from same_day_1445.live_optimizer.engine.records import GateResult, OptimizerState
from same_day_1445.live_optimizer.engine.release_manager import write_candidate_release, promote_candidate, rollback, ReleaseError


def passing_gate():
    return GateResult(passed=True, reasons=[], evidence={})


def failing_gate():
    return GateResult(passed=False, reasons=['X'], evidence={})


def test_candidate_release_is_immutable(tmp_path: Path):
    root = tmp_path / 'live_optimizer'
    root.mkdir()
    path = write_candidate_release(root, 'cand1', {'line': 'C', 'params': {'x': 1}})
    assert path.exists()
    with pytest.raises(ReleaseError, match='exists'):
        write_candidate_release(root, 'cand1', {'line': 'C', 'params': {'x': 2}})


def test_shadow_only_cannot_formal_promote_and_v2_is_never_written(tmp_path: Path):
    root = tmp_path / 'live_optimizer'
    root.mkdir()
    state = OptimizerState.initial()
    with pytest.raises(ReleaseError, match='SHADOW_ONLY'):
        promote_candidate(root, state, 'C', 'cand1', passing_gate(), '2026-08-23T16:00:00+08:00')
    assert not (tmp_path / 'release_v2').exists()


def test_promotion_requires_gate_and_rollback_restores_parent(tmp_path: Path):
    root = tmp_path / 'live_optimizer'
    root.mkdir()
    state = OptimizerState.initial()
    state.mode = 'NORMAL'
    with pytest.raises(ReleaseError, match='gate'):
        promote_candidate(root, state, 'D', 'cand1', failing_gate(), '2026-08-23T16:00:00+08:00')
    write_candidate_release(root, 'cand1', {'line': 'D', 'params': {'x': 1}})
    promoted = promote_candidate(root, state, 'D', 'cand1', passing_gate(), '2026-08-23T16:00:00+08:00')
    assert promoted.active_formal['D'].startswith('auto_D_')
    assert promoted.previous_formal['D'] == 'release_v2'
    rolled = rollback(root, promoted, 'D', 'test', '2026-08-24T16:00:00+08:00')
    assert rolled.active_formal['D'] == 'release_v2'
    ledger = (root / 'ledger' / 'promotion.csv').read_text(encoding='utf-8')
    assert 'PROMOTE' in ledger and 'ROLLBACK' in ledger
