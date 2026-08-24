from __future__ import annotations
import csv,json,subprocess,sys
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any
@dataclass
class NormalizedSignal:
    line:str;decision_date:str;action:str;symbol:str;name:str;tradable:bool;status:str;raw_action:str;reason:str;score:float|None;metadata:dict[str,Any]=field(default_factory=dict)
def _hold(line,date,raw='',reason='',tradable=True,status='OK',symbol='',name='',metadata=None):return NormalizedSignal(line,date,'HOLD',str(symbol).zfill(6) if symbol else '',str(name),tradable,status,str(raw),str(reason),None,metadata or {})
def _action(v):
    x=str(v or '').strip().lower()
    if 'buy' in x or '买入' in x:return 'BUY'
    if 'sell' in x or '卖出' in x:return 'SELL'
    return 'HOLD'
def parse_c_row(row,target):
    d=str(row.get('data_date') or row.get('date') or '')[:10]
    if d!=target:return _hold('C',target,tradable=False,status='DATA_HOLD',reason='stale_c_output')
    raw=str(row.get('next_action') or row.get('action') or '');a=_action(raw);return NormalizedSignal('C',target,a,str(row.get('symbol') or '').zfill(6) if row.get('symbol') else '',str(row.get('display_name') or row.get('name') or ''),True,'OK',raw,str(row.get('reason') or ''),None,{})
def parse_s_row(row,target):
    d=str(row.get('asof_date') or row.get('data_date') or row.get('date') or '')[:10]
    if d!=target:return _hold('S',target,tradable=False,status='DATA_HOLD',reason='stale_s_output')
    raw=str(row.get('action') or row.get('next_action') or '');a=_action(raw);return NormalizedSignal('S',target,a,str(row.get('symbol') or '').zfill(6) if row.get('symbol') else '',str(row.get('display_name') or row.get('name') or ''),True,'OK',raw,str(row.get('reason') or ''),None,{})
def parse_d_payload(payload,target,position_status='model'):
    if str(payload.get('data_date') or '')[:10]!=target:return _hold('D',target,tradable=False,status='DATA_HOLD',reason='stale_d_output')
    node=payload.get('if_real_account_is_cash',{}) if position_status=='cash' else payload.get('model_account',{}) if position_status in {'holding','model'} else {};raw=node.get('action') or payload.get('cash_action' if position_status=='cash' else 'model_action','');symbol=node.get('symbol') or payload.get('cash_symbol' if position_status=='cash' else 'model_symbol','');name=node.get('name','');meta={k:v for k,v in (node.get('position_state') or {}).items()};a=_action(raw);return NormalizedSignal('D',target,a,str(symbol or '').zfill(6) if symbol else '',str(name),True,'OK',str(raw),str(node.get('reason') or ''),None,meta)
def run_command(root,args):
    p=subprocess.run([sys.executable,*[str(root/x) if i==0 else str(x) for i,x in enumerate(args)]],cwd=root,text=True,encoding='utf-8',capture_output=True);return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'cmd':args}
def _read_first(path):
    with path.open('r',encoding='utf-8-sig',newline='') as h:return next(csv.DictReader(h),{})
def _c_internal_steps_ok(root):
    p=Path(root)/'C/live/daily_run.json'
    if not p.exists():return True
    try:v=json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception:return False
    for step in v.get('steps',[]):
        if step.get('skipped'):continue
        try:rc=int(step.get('returncode',0))
        except Exception:rc=1
        if rc!=0:return False
    return True
def run_csd_adapters(runtime_root,target,d_position_status='model',enabled_lines=None):
    root=Path(runtime_root);enabled=set(enabled_lines or {'C','S','D'});out={};runs=[];cmds={'C':['C/src/run_daily.py','--skip-update'],'S':['S/src/run_s1_live.py'],'D':['D/src/daily_panic_live.py']}
    for line in ('C','S','D'):
        if line not in enabled:continue
        run=run_command(root,cmds[line]);runs.append(run)
        if run['returncode']!=0:out[line]=_hold(line,target,tradable=False,status='DATA_HOLD',reason='adapter_command_failed');continue
        try:
            if line=='C':
                if not _c_internal_steps_ok(root):out[line]=_hold('C',target,tradable=False,status='DATA_HOLD',reason='c_internal_step_failed');continue
                out[line]=parse_c_row(_read_first(root/'C/live/today_decision.csv'),target)
            elif line=='S':out[line]=parse_s_row(_read_first(root/'S/live/today_decision.csv'),target)
            else:out[line]=parse_d_payload(json.loads((root/'D/live/daily_panic_today.json').read_text(encoding='utf-8-sig')),target,d_position_status)
        except Exception as exc:out[line]=_hold(line,target,tradable=False,status='DATA_HOLD',reason=f'adapter_output_error:{type(exc).__name__}')
    return out,runs
