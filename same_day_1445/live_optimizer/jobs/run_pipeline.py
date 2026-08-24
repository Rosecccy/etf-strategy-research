from __future__ import annotations
import argparse,json
from datetime import date,datetime,time
from pathlib import Path
from typing import Any,Callable
from ..engine.orchestrator import run_optimizer
from ..ops.health import build_health_report
from ..ops.dashboard import write_dashboard
from ..providers.eastmoney import EastmoneyMinuteProvider
from ..providers.file_provider import FileMinuteProvider
from .run_1445 import run_1445_cycle
from .run_close import run_close_cycle

def _load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def _provider(cfg):
 pc=cfg.get('provider',{});return FileMinuteProvider(Path(pc['file_root'])) if pc.get('kind')=='file' else EastmoneyMinuteProvider()
def _write_run_status(root,command,observed_at):
 p=Path(root)/'ledger/run_status.json';p.parent.mkdir(parents=True,exist_ok=True);v=_load(p) if p.exists() and p.read_text(encoding='utf-8').strip() else {};v[str(command)]=observed_at;t=p.with_suffix('.json.tmp');t.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n');t.replace(p)
def _has_unreconciled_close(root):
 p=Path(root)/'state/unreconciled_close.json'
 if not p.exists() or not p.read_text(encoding='utf-8').strip():return False
 try:v=_load(p)
 except (json.JSONDecodeError,OSError):return True
 return bool(v.get('symbols'))
def run_pipeline(command,optimizer_root,workspace_config,trade_date,observed_at=None,handlers=None):
 optimizer_root=Path(optimizer_root);cfg=_load(Path(workspace_config));runtime=Path(cfg['runtime_root']);day=date.fromisoformat(str(trade_date)[:10]);now=observed_at or datetime.now().astimezone().isoformat();custom=handlers or {}
 if command=='optimizer' and _has_unreconciled_close(optimizer_root):return {'status':'BLOCKED','reason':'UNRECONCILED_CLOSE','trade_date':day.isoformat()}
 if command=='preclose':
  h=custom.get('preclose');kw={'optimizer_root':optimizer_root,'runtime_root':runtime,'provider':_provider(cfg) if h is None else None,'trade_date':day,'cutoff':time.fromisoformat(cfg.get('cutoff','14:45')),'max_staleness_minutes':int(cfg.get('max_staleness_minutes',2)),'shadow_config':cfg.get('shadow',{}),'market_proxy':cfg.get('market_proxy','510500'),'d_account_mode':cfg.get('d_account_mode','model'),'snapshot_workers':int(cfg.get('snapshot_workers',8))};result=h(**kw) if h else run_1445_cycle(**kw)
 elif command=='close':
  h=custom.get('close');kw={'optimizer_root':optimizer_root,'runtime_root':runtime,'provider':_provider(cfg) if h is None else None,'trade_date':day,'cutoff':time.fromisoformat(cfg.get('close_cutoff','15:00')),'max_staleness_minutes':int(cfg.get('close_staleness_minutes',5)),'optimizer_callback':None,'snapshot_workers':int(cfg.get('close_workers',8))};result=h(**kw) if h else run_close_cycle(**kw)
 elif command=='optimizer':
  h=custom.get('optimizer');result=h(root=optimizer_root,as_of=now) if h else run_optimizer(optimizer_root,now);result={'status':'OK'} if result is None else result;write_dashboard(optimizer_root)
 elif command=='health':
  h=custom.get('health');result=h(root=optimizer_root,now=now) if h else build_health_report(optimizer_root,now)
 else:raise ValueError(f'unknown pipeline command: {command}')
 if not isinstance(result,dict):result={'status':'OK','result':result}
 result.setdefault('status','OK')
 if result.get('status')!='BLOCKED':_write_run_status(optimizer_root,command,now)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('preclose','close','optimizer','health'));p.add_argument('--optimizer-root',type=Path,required=True);p.add_argument('--workspace-config',type=Path,required=True);p.add_argument('--date',default=date.today().isoformat());p.add_argument('--observed-at');a=p.parse_args();r=run_pipeline(a.command,a.optimizer_root,a.workspace_config,a.date,a.observed_at);print(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2,default=str));raise SystemExit(2 if r.get('status')=='BLOCKED' else 0)
if __name__=='__main__':main()
