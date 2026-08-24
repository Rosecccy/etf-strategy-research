from __future__ import annotations
import csv,json
from pathlib import Path
from typing import Any
from ..engine.active_policy import resolve_active_policy,ActivePolicyError
from ..engine.rolling_evaluator import compute_metrics
LINES=('C','S','D','R')
def _read_json(path,default=None):
    if not path.exists() or not path.read_text(encoding='utf-8').strip():return default
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError:return default
def _read_csv(path):
    if not path.exists() or path.stat().st_size==0:return []
    with path.open('r',encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def _latest(rows,line,candidate_id=None):
    s=[r for r in rows if str(r.get('line',''))==line]
    if candidate_id is not None:s=[r for r in s if str(r.get('candidate_id',''))==str(candidate_id)]
    return max(s,key=lambda r:(str(r.get('decision_date','')),str(r.get('observed_at','')),str(r.get('signal_id','')))) if s else None
def _new_samples(rows,line,eid,cutoff):
    s=[r for r in rows if r.get('line')==line and r.get('candidate_id')==eid]
    return len(s) if not cutoff else sum(str(r.get('observable_at',''))>str(cutoff) for r in s)
def _price_curve(execution_rows,formal_rows,shadow_rows,line,symbol,limit=60):
    symbol=str(symbol or '').zfill(6) if symbol else ''
    if not symbol:return {'symbol':'','points':[],'markers':[]}
    points=[]
    for r in execution_rows:
        if str(r.get('symbol','')).zfill(6)!=symbol:continue
        try:p=float(r.get('price_1445',''))
        except (TypeError,ValueError):continue
        try:c=float(r.get('close_price',''))
        except (TypeError,ValueError):c=p
        points.append({'date':str(r.get('trade_date',''))[:10],'price_1445':p,'close':c})
    points=sorted(points,key=lambda x:x['date'])[-limit:];dates={x['date'] for x in points};markers=[]
    for source,rows in (('formal',formal_rows),('shadow',shadow_rows)):
        for r in rows:
            if str(r.get('line',''))!=line or str(r.get('symbol','')).zfill(6)!=symbol:continue
            action=str(r.get('action','')).upper();day=str(r.get('decision_date',''))[:10]
            if action in {'BUY','SELL'} and (not dates or day in dates):markers.append({'date':day,'action':action,'source':source,'candidate_id':str(r.get('candidate_id',''))})
    return {'symbol':symbol,'points':points,'markers':markers}
def build_dashboard_data(root):
    root=Path(root);state=_read_json(root/'state/optimizer_state.json',{}) or {};latest=_read_json(root/'state/latest_optimizer_decision.json',{}) or {};formal=_read_csv(root/'ledger/formal_signals.csv');anchor=_read_csv(root/'ledger/anchor_signals.csv');shadow=_read_csv(root/'ledger/shadow_signals.csv');closed=_read_csv(root/'ledger/closed_trades.csv');promos=_read_csv(root/'ledger/promotion.csv');refs=_read_csv(root/'ledger/execution_reference.csv');unrec=_read_json(root/'state/unreconciled_close.json',{}) or {};lines={}
    for line in LINES:
        release=str(state.get('active_formal',{}).get(line,'release_v2'))
        try:eid=resolve_active_policy(root,line,release).evidence_id
        except ActivePolicyError:eid=release
        sid=state.get('shadow_leader',{}).get(line);acct='research_sum' if line=='D' else 'compound';fr=[r for r in closed if r.get('line')==line and r.get('candidate_id')==eid];ar=[r for r in closed if r.get('line')==line and r.get('candidate_id')=='release_v2'];dl=latest.get('lines',{}).get(line,{}) if isinstance(latest,dict) else {};lp=[r for r in promos if r.get('line')==line];lf=_latest(formal,line);la=_latest(anchor,line,'release_v2');ls=_latest(shadow,line,sid) if sid else None;sym=str((lf or la or ls or {}).get('symbol',''))
        lines[line]={'formal_release':release,'formal_evidence_id':eid,'shadow_leader':sid,'pending_promotion':state.get('pending_promotion',{}).get(line),'last_promotion_at':state.get('last_promotion_at',{}).get(line),'latest':{'formal':lf,'anchor':la,'shadow':ls},'price_curve':_price_curve(refs,formal,shadow,line,sym),'forward_metrics':{'formal':compute_metrics(fr,acct).to_dict(),'v2_anchor':compute_metrics(ar,acct).to_dict()},'new_matured_samples':_new_samples(closed,line,eid,state.get('last_promotion_at',{}).get(line)),'promotion':dl.get('promotion',{'status':'UNKNOWN'}),'rollback':dl.get('rollback',{'status':'NONE'}),'performance_drift':dl.get('drift',{'severe':False}),'daily_drift':dl.get('daily_drift',{'severe':False}),'candidates':dl.get('candidates',[]),'last_transition':lp[-1] if lp else None}
    times=[str(r.get('observed_at','')) for r in [*formal,*anchor,*shadow] if str(r.get('observed_at',''))];generated=(latest.get('as_of') if isinstance(latest,dict) else None) or (max(times) if times else None);return {'generated_at':generated,'mode':state.get('mode','UNKNOWN'),'unreconciled_close':unrec,'optimizer_as_of':latest.get('as_of') if isinstance(latest,dict) else None,'lines':lines}
def write_dashboard(root):
    root=Path(root);p=root/'site/data/dashboard.json';p.parent.mkdir(parents=True,exist_ok=True);payload=build_dashboard_data(root);t=p.with_suffix('.json.tmp');t.write_text(json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n',encoding='utf-8');t.replace(p);return p
