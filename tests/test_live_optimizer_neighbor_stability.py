import json
from pathlib import Path
from same_day_1445.live_optimizer.engine.candidate_generator import generate_candidates
from same_day_1445.live_optimizer.engine.ledger import AppendOnlyCsvLedger
from same_day_1445.live_optimizer.engine.orchestrator import run_optimizer

def test_isolated_d_parameter_peak_is_rejected_by_real_neighbor_rate(tmp_path: Path):
    root=tmp_path/'opt';(root/'config').mkdir(parents=True);(root/'state').mkdir();(root/'ledger').mkdir();cfg={'mode':'SHADOW_ONLY','cooldown_days':0,'gates':{'trigger_retention_min':0.9,'win_rate_delta_min':0.0,'mean_return_delta_min':0.0,'max_drawdown_deterioration_max':1.0,'recent_windows_improve_fraction_min':0.0,'recent_window_win_rate_regression_max':1.0,'neighbor_pass_rate_min':0.5,'minimum_evaluation_windows':0},'drift':{'recent_count':20},'lines':{x:{'minimum_new_samples':0,'minimum_regimes':0,'accounting':'research_sum' if x=='D' else 'compound'} for x in 'CSDR'},'candidate_spaces':{'D':{'weak_market_return':[-0.05,0.0,0.05],'overheat_ma20':[0.08],'profit_arm':[0.2],'giveback':[0.02],'minimum_hold_days':[10]}},'evaluation_windows':[]};(root/'config/optimizer.json').write_text(json.dumps(cfg),encoding='utf-8');specs=generate_candidates(cfg,'D');led=AppendOnlyCsvLedger(root/'ledger/closed_trades.csv',('sample_id','candidate_id'))
    for i in range(3):
        common={'opportunity_id':f'o{i}','line':'D','decision_at':'2026-08-24T15:00:00+08:00','observable_at':'2026-08-24T15:00:00+08:00','entry_date':f'2026-07-0{i+1}','exit_date':f'2026-07-1{i+1}','triggered':1,'regime':'weak'};led.append({'sample_id':f'b{i}','candidate_id':'release_v2','ret':0.01,**common})
        for spec in specs:
            weak=spec['params']['weak_market_return'];ret=0.02 if weak==0.0 else -0.01;led.append({'sample_id':f"{spec['candidate_id']}-{i}",'candidate_id':spec['candidate_id'],'ret':ret,**common})
    result=run_optimizer(root,'2026-08-24T16:00:00+08:00');middle=next(x for x in result['lines']['D']['candidates'] if x['candidate_id']==next(s['candidate_id'] for s in specs if s['params']['weak_market_return']==0.0));assert middle['gate']['evidence']['neighbor_pass_rate']==0.0;assert 'NEIGHBOR_STABILITY' in middle['gate']['reasons']
