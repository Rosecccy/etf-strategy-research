import json
from pathlib import Path
from same_day_1445.live_optimizer.engine.ledger import AppendOnlyCsvLedger
from same_day_1445.live_optimizer.engine.orchestrator import run_optimizer


def root_with_candidate(tmp_path: Path, mismatched_positions=False):
    root=tmp_path/'opt';(root/'config').mkdir(parents=True);(root/'state').mkdir();(root/'ledger').mkdir();(root/'candidates/C_DELAY1').mkdir(parents=True)
    cfg={'mode':'NORMAL','cooldown_days':0,'gates':{'trigger_retention_min':0.9,'win_rate_delta_min':0.0,'mean_return_delta_min':-1.0,'max_drawdown_deterioration_max':1.0,'recent_windows_improve_fraction_min':0.0,'recent_window_win_rate_regression_max':1.0,'neighbor_pass_rate_min':0.0,'minimum_evaluation_windows':0},'drift':{'recent_count':20},'lines':{x:{'minimum_new_samples':0,'minimum_regimes':0,'accounting':'research_sum' if x=='D' else 'compound'} for x in 'CSDR'},'candidate_spaces':{},'evaluation_windows':[]}
    (root/'config/optimizer.json').write_text(json.dumps(cfg),encoding='utf-8')
    (root/'candidates/C_DELAY1/manifest.json').write_text(json.dumps({'candidate_id':'C_DELAY1','line':'C','module':'C_DELAY1','params':{'delay_days':1}}),encoding='utf-8')
    led=AppendOnlyCsvLedger(root/'ledger/closed_trades.csv',('sample_id','candidate_id'))
    for i in range(2):
        common={'opportunity_id':f'o{i}','line':'C','decision_at':'2026-08-24T15:00:00+08:00','observable_at':'2026-08-24T15:00:00+08:00','entry_date':f'2026-07-0{i+1}','exit_date':f'2026-07-1{i+1}','triggered':1,'regime':'neutral'}
        led.append({'sample_id':f'b{i}','candidate_id':'release_v2','ret':0.01,**common})
        led.append({'sample_id':f'c{i}','candidate_id':'C_DELAY1','ret':0.02,**common})
    if mismatched_positions:
        (root/'state/forward_positions.json').write_text(json.dumps({'C:release_v2':{'symbol':'510880','entry_date':'2026-08-20','entry_price':1.0},'C:C_DELAY1':{'symbol':'510880','entry_date':'2026-08-21','entry_price':1.1}}),encoding='utf-8')
    return root


def test_normal_mode_promotes_gate_passed_candidate_when_transition_safe(tmp_path: Path):
    root=root_with_candidate(tmp_path)
    result=run_optimizer(root,'2026-08-24T16:00:00+08:00')
    state=json.loads((root/'state/optimizer_state.json').read_text())
    assert state['active_formal']['C'].startswith('auto_C_')
    assert result['lines']['C']['promotion']['status']=='PROMOTED'
    assert (root/'ledger/promotion.csv').exists()


def test_promotion_waits_when_formal_and_candidate_positions_are_not_compatible(tmp_path: Path):
    root=root_with_candidate(tmp_path,mismatched_positions=True)
    result=run_optimizer(root,'2026-08-24T16:00:00+08:00')
    state=json.loads((root/'state/optimizer_state.json').read_text())
    assert state['active_formal']['C']=='release_v2'
    assert result['lines']['C']['promotion']['status']=='PENDING_TRANSITION'
    assert state['pending_promotion']['C']=='C_DELAY1'
