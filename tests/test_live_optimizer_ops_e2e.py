import json
from datetime import date,time
from pathlib import Path
from same_day_1445.live_optimizer.engine.records import OptimizerState
from same_day_1445.live_optimizer.engine.signal_adapter import NormalizedSignal
from same_day_1445.live_optimizer.jobs.run_1445 import run_1445_cycle
from same_day_1445.live_optimizer.jobs.run_close import run_close_cycle
from same_day_1445.live_optimizer.jobs.run_pipeline import run_pipeline
from same_day_1445.live_optimizer.providers.file_provider import FileMinuteProvider

def _raw(path:Path,symbol:str):
    path.parent.mkdir(parents=True,exist_ok=True);rows=['date,symbol,name,open,high,low,close,volume,amount,pct_change,turnover_rate,source_file'];[rows.append(f'2026-07-{i:02d},{symbol},x,1,1,1,1,1,1,,,old') for i in range(1,23)];path.write_text('\n'.join(rows)+'\n',encoding='utf-8-sig')
def _minute(root,symbol,day):
    root.mkdir(parents=True,exist_ok=True);(root/f'{symbol}.csv').write_text('timestamp,open,high,low,close,volume,amount\n'+f'{day} 14:45:00,1.00,1.02,0.99,1.01,100,101\n'+f'{day} 15:00:00,1.01,1.03,1.00,1.02,100,102\n',encoding='utf-8')
def _environment(tmp_path):
    runtime=tmp_path/'runtime';root=tmp_path/'optimizer';feed=tmp_path/'feed'
    for symbol in ('510880','510500'):_raw(runtime/'C/raw/etf'/f'{symbol}.csv',symbol);_raw(runtime/'S/raw/etf'/f'{symbol}.csv',symbol);_minute(feed,symbol,'2026-08-24')
    (runtime/'R/formal').mkdir(parents=True,exist_ok=True);(runtime/'R/formal/r_single_yearly_choice.csv').write_text('year,policy_id,policy\n2026,base_CS,"{""policy_id"":""base_CS""}"\n',encoding='utf-8-sig');(root/'runtime').mkdir(parents=True);workspace=root/'runtime/workspace.json';workspace.write_text(json.dumps({'runtime_root':str(runtime),'provider':{'kind':'file','file_root':str(feed)},'cutoff':'14:45','close_cutoff':'15:00','max_staleness_minutes':2,'close_staleness_minutes':5,'market_proxy':'510500','d_account_mode':'model','shadow':{'C_DELAY1':{'enabled':True},'D_STRICT':{'enabled':False},'D_GRID':{'enabled':False}}}),encoding='utf-8');(root/'config').mkdir(parents=True);source_cfg=Path(__file__).resolve().parents[1]/'same_day_1445/live_optimizer/config/optimizer.json';(root/'config/optimizer.json').write_text(source_cfg.read_text(encoding='utf-8'),encoding='utf-8');(root/'state').mkdir(parents=True);state=OptimizerState.initial();(root/'state/optimizer_state.json').write_text(json.dumps(state.to_dict()),encoding='utf-8');return root,runtime,feed,workspace
def test_offline_preclose_close_optimizer_dashboard_is_deterministic(tmp_path:Path):
    root,runtime,feed,workspace=_environment(tmp_path);provider=FileMinuteProvider(feed)
    def adapter_runner(_root,target,_dpos,enabled_lines=None):
        enabled=set(enabled_lines or {'C','S','D'});out={}
        if 'C' in enabled:out['C']=NormalizedSignal('C',target,'BUY','510880','x',True,'OK','same_day_preclose_buy','',None,{})
        if 'S' in enabled:out['S']=NormalizedSignal('S',target,'HOLD','','',True,'OK','wait_idle','',None,{})
        if 'D' in enabled:out['D']=NormalizedSignal('D',target,'HOLD','','',True,'OK','空仓等待','',None,{})
        return out,[]
    def preclose(**_kwargs):return run_1445_cycle(root,runtime,provider,date(2026,8,24),time(14,45),2,adapter_runner=adapter_runner,shadow_config={'C_DELAY1':{'enabled':True},'D_STRICT':{'enabled':False},'D_GRID':{'enabled':False}},market_proxy='510500')
    def close(**_kwargs):return run_close_cycle(root,runtime,provider,date(2026,8,24),time(15,0),5)
    fixed={'preclose':'2026-08-24T14:45:30+08:00','close':'2026-08-24T15:10:00+08:00','optimizer':'2026-08-24T15:20:00+08:00'};handlers={'preclose':preclose,'close':close};run_pipeline('preclose',root,workspace,'2026-08-24',fixed['preclose'],handlers=handlers);run_pipeline('close',root,workspace,'2026-08-24',fixed['close'],handlers=handlers);a=run_pipeline('optimizer',root,workspace,'2026-08-24',fixed['optimizer']);first=(root/'site/data/dashboard.json').read_bytes();run_pipeline('preclose',root,workspace,'2026-08-24',fixed['preclose'],handlers=handlers);run_pipeline('close',root,workspace,'2026-08-24',fixed['close'],handlers=handlers);b=run_pipeline('optimizer',root,workspace,'2026-08-24',fixed['optimizer']);second=(root/'site/data/dashboard.json').read_bytes();assert a['decision_hash']==b['decision_hash'];assert first==second
