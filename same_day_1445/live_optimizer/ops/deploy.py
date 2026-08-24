from __future__ import annotations
import json,hashlib,subprocess,sys
from datetime import datetime,timedelta
from pathlib import Path
from typing import Any
from ..engine.active_policy import resolve_active_policy,ActivePolicyError
from ..engine.config import stable_json_hash
from ..engine.forward_ledger import append_csv_record
from ..engine.records import OptimizerState
from .health import build_health_report
class DeploymentError(RuntimeError):pass
def _dt(v):return datetime.fromisoformat(str(v).replace('Z','+00:00'))
def _read_json(p):
    if not p.exists() or not p.read_text(encoding='utf-8').strip():return None
    try:v=json.loads(p.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError:return None
    return v if isinstance(v,dict) else None
def _atomic_json(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n');t.replace(p)
def _verified_v2(root,now,max_age_hours=24.0):
    m=_read_json(root/'state/v2_baseline_verified.json')
    if not m or m.get('ok') is not True or m.get('canonical_baseline')!='same_day_1445/release_v2':raise DeploymentError('V2 baseline verification marker is missing or invalid')
    va=str(m.get('verified_at',''))
    if not va:raise DeploymentError('V2 baseline verification marker has no timestamp')
    if _dt(now)-_dt(va)>timedelta(hours=max_age_hours):raise DeploymentError('V2 baseline verification marker is stale')
    return m
def verify_v2_baseline(repo_root,optimizer_root,timestamp,python_exe=None):
    repo_root=Path(repo_root);optimizer_root=Path(optimizer_root);script=repo_root/'scripts/verify_current_baseline.py'
    if not script.exists():raise DeploymentError(f'V2 verifier is missing: {script}')
    r=subprocess.run([python_exe or sys.executable,str(script)],cwd=repo_root,text=True,encoding='utf-8',capture_output=True)
    if r.returncode!=0:raise DeploymentError(f'V2 baseline verification failed with exit {r.returncode}: {r.stdout}{r.stderr}')
    output=(r.stdout or '')+(r.stderr or '');marker={'ok':True,'canonical_baseline':'same_day_1445/release_v2','verified_at':timestamp,'verifier':str(script.relative_to(repo_root)).replace('\\','/'),'output_hash':hashlib.sha256(output.encode()).hexdigest()};_atomic_json(optimizer_root/'state/v2_baseline_verified.json',marker);return marker
def set_deployment_mode(root,mode,timestamp,health=None):
    root=Path(root);target=str(mode).upper()
    if target not in {'NORMAL','SHADOW_ONLY'}:raise DeploymentError(f'unsupported deployment mode: {mode}')
    if target=='NORMAL':
        report=health if health is not None else build_health_report(root,timestamp)
        if report.get('status')=='BLOCKED':raise DeploymentError(f"health is BLOCKED: {report.get('codes',[])}")
        dh=[str(x) for x in report.get('data_hold_lines',[])]
        if dh:raise DeploymentError(f'DATA_HOLD lines block NORMAL mode: {dh}')
        u=_read_json(root/'state/unreconciled_close.json') or {}
        if u.get('symbols'):raise DeploymentError('unreconciled close blocks NORMAL mode')
        _verified_v2(root,timestamp)
    sp=root/'state/optimizer_state.json';state=OptimizerState.from_dict(_read_json(sp) or OptimizerState.initial().to_dict())
    if target=='NORMAL':
        for line,rid in state.active_formal.items():
            try:resolve_active_policy(root,line,rid)
            except ActivePolicyError as exc:raise DeploymentError(f'active formal is not executable for {line}: {exc}') from exc
    prev=state.mode;state.mode=target;_atomic_json(sp,state.to_dict());_atomic_json(root/'state/deployment_mode.json',{'mode':target,'changed_at':timestamp});core={'timestamp':timestamp,'from_mode':prev,'to_mode':target};append_csv_record(root/'ledger/mode_changes.csv',{'event_id':stable_json_hash(core)[:24],**core},('event_id',));return {'mode':target,'previous_mode':prev,'timestamp':timestamp}
