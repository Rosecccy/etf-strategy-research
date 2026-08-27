from __future__ import annotations
import json
from datetime import date,time
from pathlib import Path
from .snapshot import build_snapshot
from .workspace import apply_snapshot_to_runtime

def reconcile_unreconciled_close(optimizer_root,runtime_root,provider,max_staleness_minutes=5,max_workers=8):
    optimizer_root=Path(optimizer_root);runtime_root=Path(runtime_root);path=optimizer_root/'state/unreconciled_close.json'
    if not path.exists() or not path.read_text(encoding='utf-8').strip():return {'status':'NONE','errors':[],'symbols':[]}
    state=json.loads(path.read_text(encoding='utf-8'));trade_date=date.fromisoformat(str(state['trade_date'])[:10]);symbols=[str(x).zfill(6) for x in state.get('symbols',[])]
    if not symbols:path.unlink(missing_ok=True);return {'status':'OK','errors':[],'symbols':[]}
    snap=build_snapshot(provider,symbols,trade_date,time(15,0),max_staleness_minutes,max_workers=max_workers)
    if snap.get('rows'):apply_snapshot_to_runtime(runtime_root,snap['rows'],'live_optimizer:close_recovery')
    errors=snap.get('errors',[]);remaining=sorted({str(x.get('symbol','')).zfill(6) for x in errors if x.get('symbol')})
    if remaining:path.write_text(json.dumps({'trade_date':trade_date.isoformat(),'symbols':remaining,'errors':errors},ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8');return {'status':'DATA_HOLD','errors':errors,'symbols':remaining}
    path.unlink(missing_ok=True);return {'status':'OK','errors':[],'symbols':[]}
