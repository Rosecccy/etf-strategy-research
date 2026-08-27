from __future__ import annotations
import csv,importlib.util,json
from datetime import datetime
from pathlib import Path
from typing import Any
CRITICAL_RUNTIME_PATHS=('C/src/run_daily.py','S/src/run_s1_live.py','D/src/daily_panic_live.py','R/formal/r_single_yearly_choice.csv')
def _dt(v):return datetime.fromisoformat(str(v).replace('Z','+00:00'))
def _read_json(p):
    if not p.exists() or not p.read_text(encoding='utf-8').strip():return None
    try:v=json.loads(p.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError:return None
    return v if isinstance(v,dict) else None
def _read_csv(p):
    if not p.exists() or p.stat().st_size==0:return []
    try:
        with p.open('r',encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
    except OSError:return []
def _latest_data_hold_lines(root):
    rows=_read_csv(root/'ledger/formal_signals.csv');latest={}
    for r in rows:
        line=str(r.get('line',''))
        if line not in {'C','S','D','R'}:continue
        key=(str(r.get('decision_date','')),str(r.get('observed_at','')),str(r.get('signal_id','')));old=latest.get(line);ok=(str(old.get('decision_date','')),str(old.get('observed_at','')),str(old.get('signal_id',''))) if old else None
        if old is None or key>ok:latest[line]=r
    return sorted(line for line,r in latest.items() if str(r.get('status','')).upper()=='DATA_HOLD' or str(r.get('tradable','1')).strip().lower() in {'0','false','no'})
def build_health_report(root,now=None):
    root=Path(root);current=_dt(now) if now else datetime.now().astimezone();codes=[];details={};wc=_read_json(root/'runtime/workspace.json')
    if wc is None:codes.append('WORKSPACE_CONFIG_MISSING');runtime=None
    else:
        req=[str(x).strip() for x in wc.get('required_python_modules',[]) if str(x).strip()];missing_mod=[x for x in req if importlib.util.find_spec(x) is None]
        if missing_mod:codes.append('PYTHON_DEPENDENCY_MISSING');details['missing_python_modules']=missing_mod
        rv=str(wc.get('runtime_root','')).strip();runtime=Path(rv) if rv else None
        if runtime is None or not runtime.exists():codes.append('RUNTIME_ROOT_MISSING')
        else:
            missing=[r for r in CRITICAL_RUNTIME_PATHS if not (runtime/r).exists()]
            if missing:codes.append('RUNTIME_FILES_MISSING');details['missing_runtime_files']=missing
    if _read_json(root/'config/optimizer.json') is None:codes.append('OPTIMIZER_CONFIG_INVALID')
    state=_read_json(root/'state/optimizer_state.json')
    if state is None:codes.append('OPTIMIZER_STATE_INVALID');state={}
    unresolved=_read_json(root/'state/unreconciled_close.json');unresolved_symbols=[]
    if unresolved:
        unresolved_symbols=[str(x) for x in unresolved.get('symbols',[])]
        if unresolved_symbols:codes.append('UNRECONCILED_CLOSE')
    rs=_read_json(root/'ledger/run_status.json') or {};ages={}
    for name in ('preclose','close','optimizer'):
        v=str(rs.get(name,'')).strip()
        if v:
            try:ages[name]=max(0.0,(current-_dt(v)).total_seconds()/60.0)
            except (ValueError,TypeError):codes.append(f'{name.upper()}_TIMESTAMP_INVALID')
    if rs and len(ages)<3:codes.append('PIPELINE_RUN_INCOMPLETE')
    holds=_latest_data_hold_lines(root)
    if holds:codes.append('LINE_DATA_HOLD');details['data_hold_lines']=holds
    transitions=_read_csv(root/'ledger/promotion.csv')
    if transitions:details['last_transition']=max(transitions,key=lambda r:(str(r.get('timestamp','')),str(r.get('event_id',''))))
    mode=str(state.get('mode',''))
    if mode=='DATA_HOLD':codes.append('STATE_DATA_HOLD')
    elif mode=='ROLLBACK':codes.append('STATE_ROLLBACK')
    blocked={'WORKSPACE_CONFIG_MISSING','RUNTIME_ROOT_MISSING','RUNTIME_FILES_MISSING','PYTHON_DEPENDENCY_MISSING','OPTIMIZER_CONFIG_INVALID','OPTIMIZER_STATE_INVALID'};status='BLOCKED' if any(c in blocked for c in codes) else 'WARN' if codes else 'OK';return {'status':status,'codes':list(dict.fromkeys(codes)),'checked_at':current.isoformat(),'runtime_root':str(runtime) if runtime is not None else '','run_ages_minutes':ages,'unreconciled_symbols':unresolved_symbols,'data_hold_lines':holds,'state':state,'details':details}
