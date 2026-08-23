from pathlib import Path
import json
import pytest

from same_day_1445.live_optimizer.engine.config import load_optimizer_config, stable_json_hash
from same_day_1445.live_optimizer.engine.records import OptimizerState


def test_load_optimizer_config_has_safe_defaults(tmp_path: Path):
    path = tmp_path / 'optimizer.json'
    path.write_text(json.dumps({
        'mode': 'SHADOW_ONLY',
        'lines': {
            'C': {'minimum_new_samples': 15, 'minimum_regimes': 2},
            'S': {'minimum_new_samples': 15, 'minimum_regimes': 2},
            'D': {'minimum_new_samples': 100, 'minimum_regimes': 1},
            'R': {'minimum_new_samples': 15, 'minimum_regimes': 2},
        },
        'gates': {'trigger_retention_min': 0.90, 'win_rate_delta_min': 0.005},
        'cooldown_days': 20,
        'candidate_spaces': {},
    }), encoding='utf-8')
    cfg = load_optimizer_config(path)
    assert cfg['mode'] == 'SHADOW_ONLY'
    assert set(cfg['lines']) == {'C','S','D','R'}
    assert cfg['lines']['D']['minimum_new_samples'] == 100


def test_stable_json_hash_ignores_dict_key_order():
    assert stable_json_hash({'b': 2, 'a': 1}) == stable_json_hash({'a': 1, 'b': 2})


def test_invalid_mode_is_rejected(tmp_path: Path):
    path = tmp_path / 'optimizer.json'
    path.write_text(json.dumps({'mode': 'AUTO_MAGIC', 'lines': {'C':{},'S':{},'D':{},'R':{}}, 'gates':{}, 'candidate_spaces':{}}), encoding='utf-8')
    with pytest.raises(ValueError, match='mode'):
        load_optimizer_config(path)


def test_optimizer_state_defaults_to_shadow_only():
    state = OptimizerState.initial()
    assert state.mode == 'SHADOW_ONLY'
    assert state.active_formal['C'] == 'release_v2'
    assert state.shadow_leader == {'C': None, 'S': None, 'D': None, 'R': None}
