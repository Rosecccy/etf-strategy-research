from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..engine.records import OptimizerState

REQUIRED=("C/src/run_daily.py","S/src/run_s1_live.py","D/src/daily_panic_live.py","R/formal/r_single_yearly_choice.csv")
class BootstrapError(RuntimeError):pass
def _same_path(a,b):
    try:return a.resolve()==b.resolve()
    except OSError:return False
def _copy_line(source,runtime,line):
    src=source/line
    if not src.exists():raise BootstrapError(f'missing source strategy directory: {line}')
    shutil.copytree(src,runtime/line,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.log','*.tmp','*.bak','*.zip'))
def bootstrap_runtime(source_root,runtime_root,optimizer_root,provider_kind='eastmoney',file_root=''):
    source=Path(source_root).resolve();runtime=Path(runtime_root).resolve();optimizer=Path(optimizer_root)
    if _same_path(source,runtime):raise BootstrapError('source_root and runtime_root must be different')
    if not source.exists():raise BootstrapError(f'source_root does not exist: {source}')
    if runtime.exists() and any(runtime.iterdir()):raise BootstrapError(f'runtime_root is not empty: {runtime}')
    runtime.mkdir(parents=True,exist_ok=True)
    for line in 'CSDR':_copy_line(source,runtime,line)
    missing=[rel for rel in REQUIRED if not (runtime/rel).exists()]
    if missing:raise BootstrapError(f'runtime validation failed: {missing}')
    workspace={'runtime_root':str(runtime),'provider':{'kind':provider_kind,'file_root':file_root},'cutoff':'14:45','close_cutoff':'15:00','max_staleness_minutes':2,'close_staleness_minutes':5,'market_proxy':'510500','shadow':{'C_DELAY1':{'enabled':True},'D_STRICT':{'enabled':False,'params':{'market_ret120_max':0.0,'overheat':0.08,'profit_arm':0.2,'giveback':0.02,'min_hold_days':10}},'D_GRID':{'enabled':True}},'d_account_mode':'model','snapshot_workers':8,'close_workers':8,'required_python_modules':['pandas','numpy','sklearn','pyarrow']}
    sp=optimizer/'state/optimizer_state.json'
    if not sp.exists():
        mode='SHADOW_ONLY';op=optimizer/'config/optimizer.json'
        if op.exists():
            try:mode=str(json.loads(op.read_text(encoding='utf-8-sig')).get('mode',mode))
            except (json.JSONDecodeError,OSError):mode='SHADOW_ONLY'
        state=OptimizerState.initial();state.mode=mode;sp.parent.mkdir(parents=True,exist_ok=True);sp.write_text(json.dumps(state.to_dict(),ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    cp=optimizer/'runtime/workspace.json';cp.parent.mkdir(parents=True,exist_ok=True);cp.write_text(json.dumps(workspace,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8');meta={'status':'OK','source_root':str(source),'runtime_root':str(runtime),'created_at':datetime.now().astimezone().isoformat(),'workspace_config':str(cp)};(optimizer/'runtime/bootstrap.json').write_text(json.dumps(meta,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8');return meta
