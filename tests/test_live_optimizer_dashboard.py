import json
from pathlib import Path

from same_day_1445.live_optimizer.engine.forward_ledger import append_csv_record
from same_day_1445.live_optimizer.engine.config import stable_json_hash
from same_day_1445.live_optimizer.ops.dashboard import build_dashboard_data, write_dashboard


def test_dashboard_is_read_only_projection_of_state_and_ledgers(tmp_path: Path):
    root=tmp_path/'opt';(root/'state').mkdir(parents=True);(root/'ledger').mkdir();(root/'site/data').mkdir(parents=True)
    state={'mode':'SHADOW_ONLY','active_formal':{x:'release_v2' for x in 'CSDR'},'previous_formal':{x:None for x in 'CSDR'},'shadow_leader':{'C':'C_DELAY1','S':None,'D':'D_STRICT','R':None},'last_promotion_at':{x:None for x in 'CSDR'},'pending_promotion':{x:None for x in 'CSDR'}}
    state_path=root/'state/optimizer_state.json';state_path.write_text(json.dumps(state),encoding='utf-8')
    for ledger,cid,action in [('formal_signals.csv','release_v2','BUY'),('anchor_signals.csv','release_v2','BUY'),('shadow_signals.csv','C_DELAY1','HOLD')]:
        append_csv_record(root/'ledger'/ledger,{'signal_id':ledger,'line':'C','candidate_id':cid,'decision_date':'2026-08-24','action':action,'symbol':'510880','tradable':1,'status':'OK','observed_at':'2026-08-24T14:45:00+08:00'},('signal_id',))
    append_csv_record(root/'ledger/closed_trades.csv',{'sample_id':'c1','line':'C','candidate_id':'release_v2','opportunity_id':'o1','entry_date':'2026-08-20','exit_date':'2026-08-24','ret':0.03,'triggered':1,'regime':'bull','observable_at':'2026-08-24T15:00:00+08:00','decision_at':'2026-08-24T15:00:00+08:00'},('sample_id','candidate_id'))
    append_csv_record(root/'ledger/execution_reference.csv',{'reference_id':'r1','trade_date':'2026-08-24','symbol':'510880','price_1445':1.02,'close_price':1.03,'close_vs_1445':0.0098,'observable_at':'2026-08-24T15:00:00+08:00','source_hash':'x'},('reference_id',))
    decision={'as_of':'2026-08-24T15:20:00+08:00','mode':'SHADOW_ONLY','lines':{'C':{'formal':'release_v2','formal_evidence_id':'release_v2','shadow_leader':'C_DELAY1','promotion':{'status':'SHADOW_ONLY'},'rollback':{'status':'NONE'},'drift':{'severe':False},'daily_drift':{'severe':False},'candidates':[{'candidate_id':'C_DELAY1','gate':{'passed':False,'reasons':['MIN_NEW_SAMPLES'],'evidence':{'new_samples':1}},'overall':{'trades':1,'win_rate':1.0}}]}}}
    (root/'state/latest_optimizer_decision.json').write_text(json.dumps(decision),encoding='utf-8')
    before=stable_json_hash(json.loads(state_path.read_text(encoding='utf-8')))
    data=build_dashboard_data(root)
    assert data['mode']=='SHADOW_ONLY'
    assert data['lines']['C']['latest']['formal']['action']=='BUY'
    assert data['lines']['C']['latest']['shadow']['action']=='HOLD'
    assert data['lines']['C']['shadow_leader']=='C_DELAY1'
    assert data['lines']['C']['candidates'][0]['gate']['reasons']==['MIN_NEW_SAMPLES']
    assert data['lines']['C']['price_curve']['symbol']=='510880'
    assert data['lines']['C']['price_curve']['points'][0]['price_1445']==1.02
    path=write_dashboard(root);assert path.exists()
    after=stable_json_hash(json.loads(state_path.read_text(encoding='utf-8')));assert before==after
