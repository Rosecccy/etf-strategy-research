import json
from pathlib import Path
import pytest
from same_day_1445.live_optimizer.ops.deploy import DeploymentError, set_deployment_mode
from same_day_1445.live_optimizer.engine.orchestrator import run_optimizer

def _root(tmp_path: Path):
    root=tmp_path/'opt';(root/'config').mkdir(parents=True);(root/'state').mkdir();(root/'ledger').mkdir();cfg={'mode':'SHADOW_ONLY','cooldown_days':0,'gates':{'trigger_retention_min':0.9,'win_rate_delta_min':0.005,'mean_return_delta_min':0.0,'max_drawdown_deterioration_max':0.01,'recent_windows_improve_fraction_min':0.0,'recent_window_win_rate_regression_max':1.0,'neighbor_pass_rate_min':0.0,'minimum_evaluation_windows':0},'drift':{},'lines':{x:{'minimum_new_samples':0,'minimum_regimes':0,'accounting':'research_sum' if x=='D' else 'compound'} for x in 'CSDR'},'candidate_spaces':{},'evaluation_windows':[]};(root/'config/optimizer.json').write_text(json.dumps(cfg),encoding='utf-8');state={'mode':'SHADOW_ONLY','active_formal':{x:'release_v2' for x in 'CSDR'},'previous_formal':{x:None for x in 'CSDR'},'shadow_leader':{x:None for x in 'CSDR'},'last_promotion_at':{x:None for x in 'CSDR'},'pending_promotion':{x:None for x in 'CSDR'}};(root/'state/optimizer_state.json').write_text(json.dumps(state),encoding='utf-8');return root

def test_normal_mode_requires_v2_verification_marker(tmp_path: Path):
    root=_root(tmp_path)
    with pytest.raises(DeploymentError,match='V2'):set_deployment_mode(root,'NORMAL','2026-08-24T16:00:00+08:00',health={'status':'OK','codes':[]})
def test_normal_mode_override_is_persistent_for_optimizer(tmp_path: Path):
    root=_root(tmp_path);(root/'state/v2_baseline_verified.json').write_text(json.dumps({'ok':True,'canonical_baseline':'same_day_1445/release_v2','verified_at':'2026-08-24T15:59:00+08:00'}),encoding='utf-8');result=set_deployment_mode(root,'NORMAL','2026-08-24T16:00:00+08:00',health={'status':'OK','codes':[]});assert result['mode']=='NORMAL';run=run_optimizer(root,'2026-08-24T16:01:00+08:00');assert run['mode']=='NORMAL';assert 'NORMAL' in (root/'ledger/mode_changes.csv').read_text(encoding='utf-8')
def test_shadow_only_is_always_allowed_as_safety_action(tmp_path: Path):
    root=_root(tmp_path);result=set_deployment_mode(root,'SHADOW_ONLY','2026-08-24T16:00:00+08:00',health={'status':'BLOCKED','codes':['X']});assert result['mode']=='SHADOW_ONLY'
def test_normal_mode_is_blocked_by_current_line_data_hold(tmp_path: Path):
    root=_root(tmp_path);(root/'state/v2_baseline_verified.json').write_text(json.dumps({'ok':True,'canonical_baseline':'same_day_1445/release_v2','verified_at':'2026-08-24T15:59:00+08:00'}),encoding='utf-8')
    with pytest.raises(DeploymentError,match='DATA_HOLD'):set_deployment_mode(root,'NORMAL','2026-08-24T16:00:00+08:00',health={'status':'WARN','codes':['LINE_DATA_HOLD'],'data_hold_lines':['C']})
