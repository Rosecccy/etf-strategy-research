from __future__ import annotations
import csv,json
from pathlib import Path
from typing import Any
from .signal_adapter import NormalizedSignal

def _r_hold(date,tradable=True,status='OK',reason=''):return NormalizedSignal('R',date,'HOLD','','',tradable,status,'HOLD',reason,None,{})
def route_r_single(policy,c_signal,s_signal,position):
    date=c_signal.decision_date or s_signal.decision_date
    if str(policy.get('policy_id'))!='base_CS':return _r_hold(date,False,'DATA_HOLD','unsupported_r_single_policy')
    if position:
        sl=str(position.get('source_line') or '');src=c_signal if sl=='C' else s_signal if sl=='S' else None
        if src is None or not src.tradable:return _r_hold(date,False,'DATA_HOLD','r_source_line_unavailable')
        if src.action=='SELL' and (not src.symbol or src.symbol==str(position.get('symbol') or '').zfill(6)):return NormalizedSignal('R',date,'SELL',str(position.get('symbol') or '').zfill(6),src.name,True,'OK',src.raw_action,'source_line_exit',src.score,{'source_line':sl})
        return NormalizedSignal('R',date,'HOLD',str(position.get('symbol') or '').zfill(6),src.name,True,'OK',src.raw_action,'source_line_hold',src.score,{'source_line':sl})
    if not c_signal.tradable or not s_signal.tradable:return _r_hold(date,False,'DATA_HOLD','c_or_s_adapter_unavailable')
    buys=[x for x in (c_signal,s_signal) if x.action=='BUY']
    if not buys:return _r_hold(date)
    if len(buys)==1:
        src=buys[0];return NormalizedSignal('R',date,'BUY',src.symbol,src.name,True,'OK',src.raw_action,'single_actionable_source',src.score,{'source_line':src.line})
    scored=[]
    for src in buys:
        value=src.metadata.get('r_alloc_score')
        if value is None:return _r_hold(date,False,'DATA_HOLD','dual_entry_missing_exact_r_score')
        scored.append((float(value),src))
    scored.sort(key=lambda x:(-x[0],x[1].line,x[1].symbol));src=scored[0][1];return NormalizedSignal('R',date,'BUY',src.symbol,src.name,True,'OK',src.raw_action,'exact_r_score_route',scored[0][0],{'source_line':src.line,'r_alloc_score':scored[0][0]})
def load_r_single_policy(runtime_root,target_date):
    path=Path(runtime_root)/'R/formal/r_single_yearly_choice.csv'
    if not path.exists():return {'policy_id':'MISSING','error':'missing_r_single_yearly_choice'}
    year=int(str(target_date)[:4])
    with path.open('r',encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
    selected=[r for r in rows if int(float(r.get('year',-1)))==year]
    if not selected:return {'policy_id':'MISSING','error':'missing_r_policy_year'}
    row=selected[-1];policy={'policy_id':str(row.get('policy_id') or '')}
    if row.get('policy'):
        try:policy.update(json.loads(row['policy']))
        except Exception:policy['policy_parse_error']=True
    return policy
