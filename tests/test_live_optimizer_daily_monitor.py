import json
from pathlib import Path
from same_day_1445.live_optimizer.engine.forward_ledger import append_csv_record
from same_day_1445.live_optimizer.engine.orchestrator import run_optimizer

def test_daily_signal_drift_can_force_shadow_only_without_closed_trades(tmp_path: Path):
    root=tmp_path/'opt';(root/'config').mkdir(parents=True);(root/'state').mkdir();(root/'ledger').mkdir();cfg={'mode':'NORMAL','cooldown_days':0,'gates':{'trigger_retention_min':0.9,'win_rate_delta_min':0.0,'mean_return_delta_min':0.0,'max_drawdown_deterioration_max':1.0,'recent_windows_improve_fraction_min':0.0,'recent_window_win_rate_regression_max':1.0,'neighbor_pass_rate_min':0.0,'minimum_evaluation_windows':0},'drift':{'recent_count':20,'daily_recent_days':10,'daily_min_reference_days':10,'daily_actionable_rate_abs_max':0.5,'daily_data_hold_rate_max':0.5},'lines':{x:{'minimum_new_samples':0,'minimum_regimes':0,'accounting':'research_sum' if x=='D' else 'compound'} for x in 'CSDR'},'candidate_spaces':{},'evaluation_windows':[]};(root/'config/optimizer.json').write_text(json.dumps(cfg),encoding='utf-8')
    for i in range(20):
        recent=i>=10;row={'signal_id':f's{i}','line':'C','candidate_id':'release_v2','decision_date':f'2026-07-{i+1:02d}','action':'BUY' if recent else 'HOLD','symbol':'510880' if recent else '','tradable':1,'status':'OK','observed_at':f'2026-07-{i+1:02d}T14:45:00'};append_csv_record(root/'ledger/formal_signals.csv',row,('signal_id',))
    result=run_optimizer(root,'2026-08-24T16:00:00+08:00');assert result['mode']=='SHADOW_ONLY';daily=result['lines']['C']['daily_drift'];assert daily['severe'] is True;assert 'ACTIONABLE_RATE_SHIFT' in daily['breaches']
