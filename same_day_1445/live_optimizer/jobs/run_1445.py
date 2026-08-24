from __future__ import annotations
import argparse,hashlib,inspect,json
from datetime import date,datetime,time
from pathlib import Path
from typing import Callable
from ..providers.eastmoney import EastmoneyMinuteProvider
from ..providers.file_provider import FileMinuteProvider
from ..engine.candidate_generator import generate_candidates
from ..engine.active_policy import resolve_active_policy,ActivePolicyError
from ..engine.forward_ledger import ForwardPositionBook,SkippedOpportunityBook,append_csv_record,append_pending_trade
from ..engine.r_router import load_r_single_policy,route_r_single
from ..engine.reconciliation import reconcile_unreconciled_close
from ..engine.shadow_policy import apply_c_delay1,apply_d_strict
from ..engine.signal_adapter import NormalizedSignal,run_csd_adapters
from ..engine.snapshot import build_snapshot
from ..engine.workspace import apply_snapshot_to_runtime,d_strict_features,market_regime,required_symbol_groups,required_symbols

def _stable_hash(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def _write_once(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:raise RuntimeError(f'immutable snapshot conflict: {path}')
        return
    path.write_text(text,encoding='utf-8')
def _active_formal_ids(root):
    p=root/'state/optimizer_state.json';d={l:'release_v2' for l in 'CSDR'}
    if not p.exists() or not p.read_text(encoding='utf-8').strip():return d
    try:s=json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception:return d
    return {**d,**s.get('active_formal',{})}
def _write_module_manifest(root,line,candidate_id,module,params):
    p=root/'candidates'/candidate_id/'manifest.json';payload={'line':line,'candidate_id':candidate_id,'params':params,'module':module}
    if p.exists():
        try:e=json.loads(p.read_text(encoding='utf-8-sig'))
        except json.JSONDecodeError as exc:raise RuntimeError(f'invalid immutable candidate manifest: {p}') from exc
        if e!=payload:raise RuntimeError(f'immutable candidate manifest conflict: {p}')
        return
    _write_once(p,payload)
def _write_candidate_manifest(root,candidate):_write_module_manifest(root,candidate['line'],candidate['candidate_id'],'D_STRICT_GRID',candidate['params'])
def _d_params_from_candidate(p):return {'market_ret120_max':float(p['weak_market_return']),'overheat':float(p['overheat_ma20']),'profit_arm':float(p['profit_arm']),'giveback':float(p['giveback']),'min_hold_days':int(p['minimum_hold_days'])}
def _signal_row(signal,candidate_id,observed_at):
    core={'line':signal.line,'candidate_id':candidate_id,'decision_date':signal.decision_date,'action':signal.action,'symbol':signal.symbol,'name':signal.name,'tradable':int(signal.tradable),'status':signal.status,'raw_action':signal.raw_action,'reason':signal.reason,'score':'' if signal.score is None else signal.score,'observed_at':observed_at,'metadata_json':json.dumps(signal.metadata,ensure_ascii=False,sort_keys=True)};core['signal_id']=_stable_hash(core)[:24];return core
def _price_for(signal,price_map,position):
    symbol=signal.symbol or (str(position.get('symbol')) if position else '');return price_map.get(str(symbol).zfill(6)) if symbol else None
def _data_hold(line,target,reason):return NormalizedSignal(line,target,'HOLD','','',False,'DATA_HOLD','',reason,None,{})
def _formal_opportunity_id(line,formal_id,signal):return hashlib.sha256(f'{line}|{formal_id}|{signal.decision_date}|{signal.symbol}|BUY'.encode()).hexdigest()[:24]
def _call_adapters(runner,runtime,target,dpos,enabled):
    try:params=inspect.signature(runner).parameters
    except (TypeError,ValueError):params={}
    if 'enabled_lines' in params:return runner(runtime,target,dpos,enabled_lines=enabled)
    if enabled=={'C','S','D'}:return runner(runtime,target,dpos)
    return {},[{'line':'*','returncode':-1,'stderr':'adapter_runner lacks enabled_lines under partial snapshot'}]
def _group_health(snapshot,groups):
    bad={str(x.get('symbol','')).zfill(6) for x in snapshot.get('errors',[])};return {'C':not(set(groups['C'])&bad) and bool(groups['C']),'S':not(set(groups['S'])&bad) and bool(groups['S'])}
def _preclose_receipt_path(root,target):return Path(root)/'ledger/runs'/f'{target}.preclose.json'
def _load_successful_preclose_receipt(root,target):
    p=_preclose_receipt_path(root,target)
    if not p.exists() or not p.read_text(encoding='utf-8').strip():return None
    try:v=json.loads(p.read_text(encoding='utf-8-sig'))
    except (json.JSONDecodeError,OSError):return None
    return v if isinstance(v,dict) and v.get('trade_date')==target and v.get('status')=='OK' else None
def _write_preclose_receipt(root,target,result):
    if result.get('status')!='OK':return
    p=_preclose_receipt_path(root,target);p.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n'
    if p.exists():
        if p.read_text(encoding='utf-8')==payload:return
        raise RuntimeError(f'preclose receipt conflict for {target}')
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(payload,encoding='utf-8');t.replace(p)
def run_1445_cycle(optimizer_root,runtime_root,provider,trade_date,cutoff=time(14,45),max_staleness_minutes=2,adapter_runner=run_csd_adapters,shadow_config=None,market_proxy='510500',d_account_mode='model',snapshot_workers=8):
    optimizer_root=Path(optimizer_root);runtime_root=Path(runtime_root);target=trade_date.isoformat();cached=_load_successful_preclose_receipt(optimizer_root,target)
    if cached is not None:return cached
    observed_at=datetime.combine(trade_date,cutoff).isoformat();groups=required_symbol_groups(runtime_root);prior=reconcile_unreconciled_close(optimizer_root,runtime_root,provider,5,snapshot_workers);snapshot=build_snapshot(provider,required_symbols(runtime_root),trade_date,cutoff,max_staleness_minutes,max_workers=snapshot_workers);unresolved=set(prior.get('symbols',[])) if prior.get('status')=='DATA_HOLD' else set()
    if unresolved:snapshot={**snapshot,'errors':[ *snapshot.get('errors',[]), *[{'symbol':s,'reason':'prior_close_unreconciled','prior_trade_date':str(prior.get('trade_date',''))} for s in sorted(unresolved)]]}
    health=_group_health(snapshot,groups);status='OK' if all(health.values()) else 'PARTIAL' if any(health.values()) else 'DATA_HOLD';snapshot={**snapshot,'status':status,'line_health':health};_write_once(optimizer_root/'ledger/snapshots'/f"{target}.{'1445' if status!='DATA_HOLD' else 'data_hold'}.json",snapshot)
    if status=='DATA_HOLD':return {'trade_date':target,'status':status,'errors':snapshot['errors'],'line_health':health}
    writable=[r for r in snapshot['rows'] if str(r['symbol']).zfill(6) not in unresolved];apply_snapshot_to_runtime(runtime_root,writable,'live_optimizer:1445');prices={str(r['symbol']).zfill(6):float(r['close']) for r in writable};book=ForwardPositionBook(optimizer_root/'state/forward_positions.json');skipped=SkippedOpportunityBook(optimizer_root/'state/skipped_opportunities.json');formal_ids=_active_formal_ids(optimizer_root);active_policies={};active_errors={}
    for line in 'CSDR':
        try:active_policies[line]=resolve_active_policy(optimizer_root,line,formal_ids[line])
        except ActivePolicyError as exc:active_errors[line]=str(exc)
    enabled=set();
    if health['C']:enabled.update({'C','D'})
    if health['S']:enabled.add('S')
    signals,command_runs=_call_adapters(adapter_runner,runtime_root,target,d_account_mode,enabled)
    if not health['C']:signals['C']=_data_hold('C',target,'c_snapshot_incomplete');signals['D']=_data_hold('D',target,'c_snapshot_incomplete')
    else:signals.setdefault('C',_data_hold('C',target,'missing_c_adapter_output'));signals.setdefault('D',_data_hold('D',target,'missing_d_adapter_output'))
    if not health['S']:signals['S']=_data_hold('S',target,'s_snapshot_incomplete')
    else:signals.setdefault('S',_data_hold('S',target,'missing_s_adapter_output'))
    if signals['C'].tradable and signals['S'].tradable:
        try:r=route_r_single(load_r_single_policy(runtime_root,target),signals['C'],signals['S'],book.get('R','release_v2'))
        except Exception as exc:r=_data_hold('R',target,f'r_router_error:{type(exc).__name__}')
    else:r=_data_hold('R',target,'c_or_s_unavailable_for_r')
    signals['R']=r;regime,mret=market_regime(runtime_root,target,market_proxy)
    for sig in signals.values():sig.metadata.setdefault('market_regime',regime);sig.metadata.setdefault('regime',regime);sig.metadata.setdefault('market_ret120',mret) if mret is not None else None
    for line in 'CSDR':
        sig=signals[line]
        if sig.tradable and sig.action=='BUY' and book.get(line,'release_v2') is None:sig.metadata['opportunity_id']=_formal_opportunity_id(line,'release_v2',sig)
    anchor_trades=[]
    for line in 'CSDR':
        sig=signals[line];append_csv_record(optimizer_root/'ledger/anchor_signals.csv',_signal_row(sig,'release_v2',observed_at),('signal_id',));pos=book.get(line,'release_v2');price=_price_for(sig,prices,pos)
        if price is not None:
            trade=book.apply(line,'release_v2',sig,price,observed_at)
            if trade:
                append_pending_trade(optimizer_root/'ledger/pending_closed_trades.jsonl',trade);anchor_trades.append(trade)
                for sr in skipped.finalize(trade,observed_at):append_pending_trade(optimizer_root/'ledger/pending_closed_trades.jsonl',sr)
    shadow_config=shadow_config or {};shadow=[];c_active=active_policies.get('C') is not None and active_policies['C'].module=='C_DELAY1'
    if (shadow_config.get('C_DELAY1',{}).get('enabled',True) or c_active) and signals['C'].tradable:
        _write_module_manifest(optimizer_root,'C','C_DELAY1','C_DELAY1',{'delay_days':1});pp=optimizer_root/'state/c_delay1_pending.json';pending=json.loads(pp.read_text()) if pp.exists() and pp.read_text().strip() else None;c,p2=apply_c_delay1(signals['C'],pending,target);pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text(json.dumps(p2,ensure_ascii=False,sort_keys=True) if p2 else '');shadow.append(('C','C_DELAY1',c))
    cache={}
    def features(symbol,current_price):
        key=(str(symbol).zfill(6),float(current_price))
        if key not in cache:cache[key]=d_strict_features(runtime_root,key[0],target,current_price,market_proxy)
        return cache[key]
    dcfg=shadow_config.get('D_STRICT',{});d_active=active_policies.get('D') is not None and active_policies['D'].module=='D_STRICT'
    if (dcfg.get('enabled',True) or d_active) and signals['D'].tradable:
        pos=book.get('D','D_STRICT') or {};symbol=signals['D'].symbol or str(pos.get('symbol') or '');cp=prices.get(str(symbol).zfill(6),0.0) if symbol else 0.0;ft=features(symbol,cp) if symbol and cp else {'price':cp,'ma20':0.0,'market_ret120':0.0};params=(active_policies['D'].params if d_active else dcfg.get('params')) or {'market_ret120_max':0.0,'overheat':.08,'profit_arm':.2,'giveback':.02,'min_hold_days':10};_write_module_manifest(optimizer_root,'D','D_STRICT','D_STRICT',params);ds=apply_d_strict(signals['D'],ft,pos,params);ds.metadata['features']=ft
        if signals['D'].action=='BUY' and ds.action!='BUY' and signals['D'].metadata.get('opportunity_id'):skipped.register('D','D_STRICT',signals['D'],observed_at)
        shadow.append(('D','D_STRICT',ds))
    grid=shadow_config.get('D_GRID',{});cfgp=optimizer_root/'config/optimizer.json';d_active_grid=active_policies.get('D') is not None and active_policies['D'].module=='D_STRICT_GRID'
    if (grid.get('enabled',False) or d_active_grid) and signals['D'].tradable and cfgp.exists():
        ocfg=json.loads(cfgp.read_text(encoding='utf-8-sig'));cands=generate_candidates(ocfg,'D') if grid.get('enabled',False) else []
        if d_active_grid and all(x['candidate_id']!=active_policies['D'].evidence_id for x in cands):cands.append({'line':'D','candidate_id':active_policies['D'].evidence_id,'params':active_policies['D'].params})
        for c in cands:
            cid=c['candidate_id'];_write_candidate_manifest(optimizer_root,c);pos=book.get('D',cid) or {};symbol=signals['D'].symbol or str(pos.get('symbol') or '');cp=prices.get(str(symbol).zfill(6),0.0) if symbol else 0.0;ft=features(symbol,cp) if symbol and cp else {'price':cp,'ma20':0.0,'market_ret120':0.0};cs=apply_d_strict(signals['D'],ft,pos,_d_params_from_candidate(c['params']));cs.metadata['features']=ft;cs.metadata['candidate_params']=c['params']
            if signals['D'].action=='BUY' and cs.action!='BUY' and signals['D'].metadata.get('opportunity_id'):skipped.register('D',cid,signals['D'],observed_at)
            shadow.append(('D',cid,cs))
    smap={}
    for line,cid,sig in shadow:
        smap[(line,cid)]=sig;append_csv_record(optimizer_root/'ledger/shadow_signals.csv',_signal_row(sig,cid,observed_at),('signal_id',));pos=book.get(line,cid);price=_price_for(sig,prices,pos)
        if price is not None:
            trade=book.apply(line,cid,sig,price,observed_at)
            if trade:append_pending_trade(optimizer_root/'ledger/pending_closed_trades.jsonl',trade)
    active={}
    for line in 'CSDR':
        rid=formal_ids[line];policy=active_policies.get(line)
        if policy is None:sig=_data_hold(line,target,f"active_release_error:{active_errors.get(line,'unknown')}")
        elif policy.module=='BASE_V2':sig=signals[line]
        else:sig=smap.get((line,policy.evidence_id)) or _data_hold(line,target,f'active_candidate_not_executed:{policy.evidence_id}')
        sig.metadata=dict(sig.metadata);sig.metadata['active_release']=rid;sig.metadata['active_evidence_id']=policy.evidence_id if policy else '';active[line]=sig;append_csv_record(optimizer_root/'ledger/formal_signals.csv',_signal_row(sig,rid,observed_at),('signal_id',))
    holds=sorted(l for l,s in active.items() if not s.tradable)
    if holds:status='DATA_HOLD' if len(holds)==4 else 'PARTIAL'
    result={'trade_date':target,'status':status,'adapter_hold_lines':holds,'line_health':health,'prior_reconciliation':prior,'snapshot_hash':_stable_hash(snapshot),'signals':{k:v.__dict__ for k,v in active.items()},'anchor_signals':{k:v.__dict__ for k,v in signals.items()},'command_runs':command_runs,'shadow_count':len(shadow),'closed_now':len(anchor_trades)};_write_preclose_receipt(optimizer_root,target,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--optimizer-root',required=True);p.add_argument('--workspace-config',required=True);p.add_argument('--date',default=date.today().isoformat());a=p.parse_args();cfg=json.loads(Path(a.workspace_config).read_text());pcfg=cfg.get('provider',{});provider=FileMinuteProvider(Path(pcfg['file_root'])) if pcfg.get('kind')=='file' else EastmoneyMinuteProvider();r=run_1445_cycle(Path(a.optimizer_root),Path(cfg['runtime_root']),provider,date.fromisoformat(a.date),time.fromisoformat(cfg.get('cutoff','14:45')),int(cfg.get('max_staleness_minutes',2)),shadow_config=cfg.get('shadow',{}),market_proxy=cfg.get('market_proxy','510500'),d_account_mode=cfg.get('d_account_mode','model'),snapshot_workers=int(cfg.get('snapshot_workers',8)));print(json.dumps(r,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
