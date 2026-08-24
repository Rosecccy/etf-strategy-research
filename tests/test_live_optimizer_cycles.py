import csv, json
from datetime import date, time
from pathlib import Path

from same_day_1445.live_optimizer.providers.file_provider import FileMinuteProvider
from same_day_1445.live_optimizer.engine.signal_adapter import NormalizedSignal
from same_day_1445.live_optimizer.jobs.run_1445 import run_1445_cycle
from same_day_1445.live_optimizer.jobs.run_close import run_close_cycle


def raw(path: Path, symbol: str):
    path.parent.mkdir(parents=True,exist_ok=True)
    rows=['date,symbol,name,open,high,low,close,volume,amount,pct_change,turnover_rate,source_file']
    for i in range(1, 23): rows.append(f'2026-07-{i:02d},{symbol},x,1,1,1,1,1,1,,,old')
    path.write_text('\n'.join(rows)+'\n',encoding='utf-8-sig')

def minute(root: Path, symbol: str, day: str):
    root.mkdir(parents=True,exist_ok=True)
    (root/f'{symbol}.csv').write_text('timestamp,open,high,low,close,volume,amount\n'+f'{day} 14:45:00,1.00,1.02,0.99,1.01,100,101\n'+f'{day} 15:00:00,1.01,1.03,1.00,1.02,100,102\n',encoding='utf-8')

def test_1445_and_close_cycle_records_forward_evidence(tmp_path: Path):
    runtime=tmp_path/'runtime'; opt=tmp_path/'optimizer'; feed=tmp_path/'feed'
    for symbol in ('510880','510500'):
        raw(runtime/'C/raw/etf'/f'{symbol}.csv',symbol);raw(runtime/'S/raw/etf'/f'{symbol}.csv',symbol);minute(feed,symbol,'2026-08-24')
    (runtime/'R/formal').mkdir(parents=True,exist_ok=True);(runtime/'R/formal/r_single_yearly_choice.csv').write_text('year,policy_id,policy\n2026,base_CS,"{\"\"policy_id\"\":\"\"base_CS\"\"}"\n',encoding='utf-8-sig')
    calls=[]
    def runner(root,target,dpos):
        calls.append((target,dpos));return {'C':NormalizedSignal('C',target,'BUY','510880','x',True,'OK','same_day_preclose_buy','',None,{}),'S':NormalizedSignal('S',target,'HOLD','', '',True,'OK','wait_idle','',None,{}),'D':NormalizedSignal('D',target,'HOLD','', '',True,'OK','空仓等待','',None,{})}, []
    result=run_1445_cycle(opt,runtime,FileMinuteProvider(feed),date(2026,8,24),time(14,45),1,adapter_runner=runner,shadow_config={'C_DELAY1':{'enabled':True},'D_STRICT':{'enabled':False}});assert result['status']=='OK';assert result['signals']['R']['action']=='BUY';assert (opt/'ledger/formal_signals.csv').exists()
    with (runtime/'C/raw/etf/510880.csv').open('r',encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
    assert float(rows[-1]['close'])==1.01
    close=run_close_cycle(opt,runtime,FileMinuteProvider(feed),date(2026,8,24),time(15,0),1,optimizer_callback=lambda root,asof:{'called':asof});assert close['status']=='OK' and close['optimizer']['called'].endswith('15:00:00')
    with (runtime/'C/raw/etf/510880.csv').open('r',encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
    assert float(rows[-1]['close'])==1.02

def _dated_minute(root: Path, symbol: str, day: str, p1445: float, p1500: float):
    d=root/symbol; d.mkdir(parents=True,exist_ok=True);(d/f'{day}.csv').write_text('timestamp,open,high,low,close,volume,amount\n'+f'{day} 14:45:00,{p1445},{p1445},{p1445},{p1445},100,100\n'+f'{day} 15:00:00,{p1500},{p1500},{p1500},{p1500},100,100\n',encoding='utf-8')

def _runtime_two_days(tmp_path: Path):
    runtime=tmp_path/'runtime'; opt=tmp_path/'optimizer'; feed=tmp_path/'feed'
    for symbol in ('510880','510500'):
        raw(runtime/'C/raw/etf'/f'{symbol}.csv',symbol); raw(runtime/'S/raw/etf'/f'{symbol}.csv',symbol);_dated_minute(feed,symbol,'2026-08-24',1.20 if symbol=='510880' else 1.00,1.21 if symbol=='510880' else 1.00);_dated_minute(feed,symbol,'2026-08-25',1.10 if symbol=='510880' else 1.00,1.11 if symbol=='510880' else 1.00)
    (runtime/'R/formal').mkdir(parents=True,exist_ok=True);(runtime/'R/formal/r_single_yearly_choice.csv').write_text('year,policy_id,policy\n2026,base_CS,"{""policy_id"":""base_CS""}"\n',encoding='utf-8-sig');return runtime,opt,feed

def test_formal_forward_trade_uses_active_release_and_causal_maturity(tmp_path: Path):
    runtime,opt,feed=_runtime_two_days(tmp_path)
    def runner(root,target,dpos):
        action='BUY' if target=='2026-08-24' else 'SELL';raw_action='same_day_preclose_buy' if action=='BUY' else 'same_day_preclose_sell';return {'C':NormalizedSignal('C',target,action,'510880','x',True,'OK',raw_action,'',None,{}),'S':NormalizedSignal('S',target,'HOLD','', '',True,'OK','wait_idle','',None,{}),'D':NormalizedSignal('D',target,'HOLD','', '',True,'OK','空仓等待','',None,{})}, []
    provider=FileMinuteProvider(feed);run_1445_cycle(opt,runtime,provider,date(2026,8,24),time(14,45),1,adapter_runner=runner,shadow_config={'C_DELAY1':{'enabled':False},'D_STRICT':{'enabled':False}});run_close_cycle(opt,runtime,provider,date(2026,8,24),time(15,0),1);run_1445_cycle(opt,runtime,provider,date(2026,8,25),time(14,45),1,adapter_runner=runner,shadow_config={'C_DELAY1':{'enabled':False},'D_STRICT':{'enabled':False}});run_close_cycle(opt,runtime,provider,date(2026,8,25),time(15,0),1)
    with (opt/'ledger/closed_trades.csv').open('r',encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
    c_rows=[r for r in rows if r['line']=='C'];assert len(c_rows)==1;row=c_rows[0];assert row['candidate_id']=='release_v2';assert row['opportunity_id'];assert row['decision_at']==row['observable_at']=='2026-08-25T15:00:00'

def test_d_strict_filtered_buy_emits_nontriggered_matched_opportunity(tmp_path: Path):
    runtime,opt,feed=_runtime_two_days(tmp_path)
    def runner(root,target,dpos):
        d_action='BUY' if target=='2026-08-24' else 'SELL';return {'C':NormalizedSignal('C',target,'HOLD','', '',True,'OK','wait_idle','',None,{}),'S':NormalizedSignal('S',target,'HOLD','', '',True,'OK','wait_idle','',None,{}),'D':NormalizedSignal('D',target,d_action,'510880','x',True,'OK','买入' if d_action=='BUY' else '卖出','',None,{})}, []
    provider=FileMinuteProvider(feed);shadow={'C_DELAY1':{'enabled':False},'D_STRICT':{'enabled':True,'params':{'market_ret120_max':0.0,'overheat':0.08,'profit_arm':0.2,'giveback':0.02,'min_hold_days':10}}};run_1445_cycle(opt,runtime,provider,date(2026,8,24),time(14,45),1,adapter_runner=runner,shadow_config=shadow);run_close_cycle(opt,runtime,provider,date(2026,8,24),time(15,0),1);run_1445_cycle(opt,runtime,provider,date(2026,8,25),time(14,45),1,adapter_runner=runner,shadow_config=shadow);run_close_cycle(opt,runtime,provider,date(2026,8,25),time(15,0),1)
    with (opt/'ledger/closed_trades.csv').open('r',encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
    baseline=[r for r in rows if r['candidate_id']=='release_v2' and r['line']=='D'];shadow_rows=[r for r in rows if r['candidate_id']=='D_STRICT' and r['line']=='D'];assert len(baseline)==1 and len(shadow_rows)==1;assert baseline[0]['opportunity_id']==shadow_rows[0]['opportunity_id'];assert shadow_rows[0]['triggered']=='0'

def test_missing_s_only_symbol_does_not_block_c_and_d(tmp_path: Path):
    runtime=tmp_path/'runtime'; opt=tmp_path/'optimizer'; feed=tmp_path/'feed';raw(runtime/'C/raw/etf/510880.csv','510880');raw(runtime/'C/raw/etf/510500.csv','510500');raw(runtime/'S/raw/etf/510880.csv','510880');raw(runtime/'S/raw/etf/159999.csv','159999');minute(feed,'510880','2026-08-24'); minute(feed,'510500','2026-08-24');(runtime/'R/formal').mkdir(parents=True,exist_ok=True);(runtime/'R/formal/r_single_yearly_choice.csv').write_text('year,policy_id,policy\n2026,base_CS,"{""policy_id"":""base_CS""}"\n',encoding='utf-8-sig');called=[]
    def runner(root,target,dpos,enabled_lines=None):
        called.append(set(enabled_lines or []));out={}
        if 'C' in enabled_lines: out['C']=NormalizedSignal('C',target,'BUY','510880','x',True,'OK','same_day_preclose_buy','',None,{})
        if 'D' in enabled_lines: out['D']=NormalizedSignal('D',target,'HOLD','', '',True,'OK','空仓等待','',None,{})
        if 'S' in enabled_lines: out['S']=NormalizedSignal('S',target,'HOLD','', '',True,'OK','wait_idle','',None,{})
        return out,[]
    result=run_1445_cycle(opt,runtime,FileMinuteProvider(feed),date(2026,8,24),time(14,45),1,adapter_runner=runner,shadow_config={'C_DELAY1':{'enabled':False},'D_STRICT':{'enabled':False}});assert result['status']=='PARTIAL';assert result['signals']['C']['tradable'] is True;assert result['signals']['D']['tradable'] is True;assert result['signals']['S']['status']=='DATA_HOLD';assert result['signals']['R']['status']=='DATA_HOLD';assert called==[{'C','D'}]

def test_d_grid_generates_only_declared_shadow_candidates(tmp_path: Path):
    runtime,opt,feed=_runtime_two_days(tmp_path);(opt/'config').mkdir(parents=True,exist_ok=True);cfg={'mode':'SHADOW_ONLY','gates':{},'lines':{x:{} for x in 'CSDR'},'candidate_spaces':{'D':{'weak_market_return':[0.0],'overheat_ma20':[0.08,0.10],'profit_arm':[0.2],'giveback':[0.02],'minimum_hold_days':[10]}}};(opt/'config/optimizer.json').write_text(json.dumps(cfg),encoding='utf-8')
    def runner(root,target,dpos):return {'C':NormalizedSignal('C',target,'HOLD','', '',True,'OK','wait_idle','',None,{}),'S':NormalizedSignal('S',target,'HOLD','', '',True,'OK','wait_idle','',None,{}),'D':NormalizedSignal('D',target,'BUY','510880','x',True,'OK','买入','',None,{})}, []
    run_1445_cycle(opt,runtime,FileMinuteProvider(feed),date(2026,8,24),time(14,45),1,adapter_runner=runner,shadow_config={'C_DELAY1':{'enabled':False},'D_STRICT':{'enabled':False},'D_GRID':{'enabled':True}})
    with (opt/'ledger/shadow_signals.csv').open('r',encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
    drows=[r for r in rows if r['line']=='D'];assert len(drows)==2;assert all(r['candidate_id'].startswith('D_') for r in drows);assert len(list((opt/'candidates').glob('D_*/manifest.json')))==2

def test_partial_close_still_matures_known_1445_trade(tmp_path: Path):
    from same_day_1445.live_optimizer.engine.forward_ledger import append_pending_trade
    runtime=tmp_path/'runtime'; opt=tmp_path/'optimizer'; feed=tmp_path/'feed';raw(runtime/'C/raw/etf/510880.csv','510880');raw(runtime/'S/raw/etf/159999.csv','159999');_dated_minute(feed,'510880','2026-08-24',1.00,1.01);append_pending_trade(opt/'ledger/pending_closed_trades.jsonl',{'line':'C','candidate_id':'release_v2','opportunity_id':'o1','symbol':'510880','entry_date':'2026-08-20','exit_date':'2026-08-24','entry_price':0.90,'exit_price':1.00,'ret':1.00/0.90-1.0,'triggered':1,'regime':'neutral','entry_observed_at':'2026-08-20T14:45:00+08:00','source_line':'C'});called=[];result=run_close_cycle(opt,runtime,FileMinuteProvider(feed),date(2026,8,24),time(15,0),1,optimizer_callback=lambda root,asof: called.append(asof) or {'ok':True});assert result['status']=='PARTIAL';assert result['matured_trades']==1;assert called
    with (opt/'ledger/closed_trades.csv').open('r',encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
    assert len(rows)==1 and rows[0]['opportunity_id']=='o1';assert any(err['symbol']=='159999' for err in result['errors'])

def test_d_grid_reuses_feature_snapshot_across_neighbor_candidates(tmp_path: Path, monkeypatch):
    import same_day_1445.live_optimizer.jobs.run_1445 as job
    runtime,opt,feed=_runtime_two_days(tmp_path);(opt/'config').mkdir(parents=True,exist_ok=True);cfg={'mode':'SHADOW_ONLY','gates':{},'lines':{x:{} for x in 'CSDR'},'candidate_spaces':{'D':{'weak_market_return':[0.0],'overheat_ma20':[0.08,0.10,0.12],'profit_arm':[0.2],'giveback':[0.02],'minimum_hold_days':[10]}}};(opt/'config/optimizer.json').write_text(json.dumps(cfg),encoding='utf-8');calls=[]
    def feature_stub(runtime_root,symbol,target,current_price,market_proxy):calls.append((symbol,target,current_price,market_proxy));return {'price':current_price,'ma20':1.0,'market_ret120':0.0}
    monkeypatch.setattr(job,'d_strict_features',feature_stub)
    def runner(root,target,dpos):return {'C':NormalizedSignal('C',target,'HOLD','', '',True,'OK','wait_idle','',None,{}),'S':NormalizedSignal('S',target,'HOLD','', '',True,'OK','wait_idle','',None,{}),'D':NormalizedSignal('D',target,'BUY','510880','x',True,'OK','买入','',None,{})}, []
    job.run_1445_cycle(opt,runtime,FileMinuteProvider(feed),date(2026,8,24),time(14,45),1,adapter_runner=runner,shadow_config={'C_DELAY1':{'enabled':False},'D_STRICT':{'enabled':False},'D_GRID':{'enabled':True}});assert len(calls)==1

def test_adapter_data_hold_marks_cycle_partial(tmp_path: Path):
    runtime,opt,feed=_runtime_two_days(tmp_path)
    def runner(root,target,dpos):return {'C':NormalizedSignal('C',target,'HOLD','','',False,'DATA_HOLD','','internal failure',None,{}),'S':NormalizedSignal('S',target,'HOLD','','',True,'OK','wait_idle','',None,{}),'D':NormalizedSignal('D',target,'HOLD','','',True,'OK','空仓等待','',None,{})}, []
    result=run_1445_cycle(opt,runtime,FileMinuteProvider(feed),date(2026,8,24),time(14,45),1,adapter_runner=runner,shadow_config={'C_DELAY1':{'enabled':False},'D_STRICT':{'enabled':False}});assert result['status']=='PARTIAL';assert result['signals']['C']['status']=='DATA_HOLD';assert result['signals']['R']['status']=='DATA_HOLD'
