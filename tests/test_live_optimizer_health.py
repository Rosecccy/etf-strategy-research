import json
from pathlib import Path
from same_day_1445.live_optimizer.ops.health import build_health_report

def _workspace(root: Path,runtime: Path):
    (root/'runtime').mkdir(parents=True);(root/'runtime/workspace.json').write_text(json.dumps({'runtime_root':str(runtime)}),encoding='utf-8');(root/'config').mkdir(parents=True,exist_ok=True);(root/'config/optimizer.json').write_text(json.dumps({'mode':'SHADOW_ONLY','lines':{x:{} for x in 'CSDR'},'gates':{},'candidate_spaces':{}}),encoding='utf-8');(root/'state').mkdir(exist_ok=True);(root/'state/optimizer_state.json').write_text(json.dumps({'mode':'SHADOW_ONLY','active_formal':{x:'release_v2' for x in 'CSDR'},'previous_formal':{x:None for x in 'CSDR'},'shadow_leader':{x:None for x in 'CSDR'},'last_promotion_at':{x:None for x in 'CSDR'},'pending_promotion':{x:None for x in 'CSDR'}}),encoding='utf-8')
def test_health_is_blocked_when_runtime_is_missing(tmp_path: Path):
    root=tmp_path/'opt';root.mkdir();(root/'runtime').mkdir();(root/'runtime/workspace.json').write_text(json.dumps({'runtime_root':str(tmp_path/'missing')}),encoding='utf-8');r=build_health_report(root,now='2026-08-24T09:00:00+08:00');assert r['status']=='BLOCKED';assert 'RUNTIME_ROOT_MISSING' in r['codes']
def test_health_reports_fresh_pipeline_and_state(tmp_path: Path):
    root=tmp_path/'opt';root.mkdir();runtime=tmp_path/'runtime';runtime.mkdir();_workspace(root,runtime)
    for rel in ['C/src/run_daily.py','S/src/run_s1_live.py','D/src/daily_panic_live.py','R/formal/r_single_yearly_choice.csv']:
        p=runtime/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('x',encoding='utf-8')
    (root/'ledger').mkdir();(root/'ledger/run_status.json').write_text(json.dumps({'preclose':'2026-08-24T14:46:00+08:00','close':'2026-08-24T15:10:00+08:00','optimizer':'2026-08-24T15:20:00+08:00'}),encoding='utf-8');r=build_health_report(root,now='2026-08-24T16:00:00+08:00');assert r['status']=='OK';assert r['state']['mode']=='SHADOW_ONLY';assert r['state']['active_formal']['D']=='release_v2'
def test_health_reports_latest_line_data_hold(tmp_path: Path):
    from same_day_1445.live_optimizer.engine.forward_ledger import append_csv_record
    root=tmp_path/'opt';root.mkdir();runtime=tmp_path/'runtime';runtime.mkdir();_workspace(root,runtime)
    for rel in ['C/src/run_daily.py','S/src/run_s1_live.py','D/src/daily_panic_live.py','R/formal/r_single_yearly_choice.csv']:
        p=runtime/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('x',encoding='utf-8')
    (root/'ledger').mkdir(exist_ok=True);append_csv_record(root/'ledger/formal_signals.csv',{'signal_id':'c1','line':'C','candidate_id':'release_v2','decision_date':'2026-08-24','action':'HOLD','symbol':'','name':'','tradable':0,'status':'DATA_HOLD','raw_action':'','reason':'adapter failed','score':'','observed_at':'2026-08-24T14:45:30+08:00','metadata_json':'{}'},('signal_id',));r=build_health_report(root,now='2026-08-24T16:00:00+08:00');assert r['status']=='WARN';assert r['data_hold_lines']==['C'];assert 'LINE_DATA_HOLD' in r['codes']
def test_health_blocks_when_declared_python_dependency_is_missing(tmp_path: Path):
    root=tmp_path/'opt';root.mkdir();runtime=tmp_path/'runtime';runtime.mkdir();_workspace(root,runtime);cfg=json.loads((root/'runtime/workspace.json').read_text());cfg['required_python_modules']=['module_that_should_never_exist_9f12'];(root/'runtime/workspace.json').write_text(json.dumps(cfg),encoding='utf-8')
    for rel in ['C/src/run_daily.py','S/src/run_s1_live.py','D/src/daily_panic_live.py','R/formal/r_single_yearly_choice.csv']:
        p=runtime/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('x',encoding='utf-8')
    r=build_health_report(root,now='2026-08-24T10:00:00+08:00');assert r['status']=='BLOCKED';assert 'PYTHON_DEPENDENCY_MISSING' in r['codes'];assert r['details']['missing_python_modules']==['module_that_should_never_exist_9f12']
def test_health_reports_last_promotion_or_rollback_transition(tmp_path: Path):
    from same_day_1445.live_optimizer.engine.forward_ledger import append_csv_record
    root=tmp_path/'opt';root.mkdir();runtime=tmp_path/'runtime';runtime.mkdir();_workspace(root,runtime)
    for rel in ['C/src/run_daily.py','S/src/run_s1_live.py','D/src/daily_panic_live.py','R/formal/r_single_yearly_choice.csv']:
        p=runtime/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('x',encoding='utf-8')
    (root/'ledger').mkdir(exist_ok=True);append_csv_record(root/'ledger/promotion.csv',{'event_id':'e1','timestamp':'2026-08-23T15:20:00+08:00','line':'D','event':'PROMOTE','from_release':'release_v2','to_release':'auto_D_x','reason':'GATE_PASS'},('event_id',));append_csv_record(root/'ledger/promotion.csv',{'event_id':'e2','timestamp':'2026-08-24T09:50:00+08:00','line':'D','event':'ROLLBACK','from_release':'auto_D_x','to_release':'release_v2','reason':'LIVE_REGRESSION'},('event_id',));r=build_health_report(root,now='2026-08-24T10:00:00+08:00');assert r['details']['last_transition']['event']=='ROLLBACK';assert r['details']['last_transition']['line']=='D';assert r['details']['last_transition']['to_release']=='release_v2'
