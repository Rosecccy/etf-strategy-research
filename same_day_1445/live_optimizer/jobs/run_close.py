from __future__ import annotations
import argparse,json
from datetime import date,datetime,time
from pathlib import Path
from typing import Callable
from ..providers.eastmoney import EastmoneyMinuteProvider
from ..providers.file_provider import FileMinuteProvider
from ..engine.forward_ledger import append_csv_record,drain_pending_trades
from ..engine.snapshot import build_snapshot
from ..engine.workspace import apply_snapshot_to_runtime,required_symbols

def _stable_id(parts):
 import hashlib
 return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:24]
def _write_snapshot(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n'
 if path.exists():
  if path.read_text(encoding='utf-8')!=text:raise RuntimeError(f'immutable close snapshot conflict: {path}')
  return
 path.write_text(text,encoding='utf-8')
def _write_unreconciled(root,target,errors):
 p=Path(root)/'state/unreconciled_close.json';symbols=sorted({str(e.get('symbol','')).zfill(6) for e in errors if e.get('symbol')});p.parent.mkdir(parents=True,exist_ok=True)
 if symbols:p.write_text(json.dumps({'trade_date':target,'symbols':symbols,'errors':errors},ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
 else:p.unlink(missing_ok=True)
def _append_execution_references(root,snapshot,observable_at):
 pre=Path(root)/'ledger/snapshots'/f"{snapshot['trade_date']}.1445.json";pmap={}
 if pre.exists():
  try:pmap={str(r['symbol']).zfill(6):float(r['close']) for r in json.loads(pre.read_text(encoding='utf-8'))['rows']}
  except Exception:pmap={}
 for r in snapshot.get('rows',[]):
  s=str(r['symbol']).zfill(6);p1445=pmap.get(s)
  if p1445 is None:continue
  close=float(r['close']);append_csv_record(Path(root)/'ledger/execution_reference.csv',{'reference_id':_stable_id((snapshot['trade_date'],s)),'trade_date':snapshot['trade_date'],'symbol':s,'price_1445':p1445,'close_price':close,'close_vs_1445':close/p1445-1 if p1445 else '', 'observable_at':observable_at,'source_hash':r.get('source_hash','')},('reference_id',))
def run_close_cycle(optimizer_root,runtime_root,provider,trade_date,cutoff=time(15,0),max_staleness_minutes=5,optimizer_callback:Callable|None=None,snapshot_workers=8):
 optimizer_root=Path(optimizer_root);runtime_root=Path(runtime_root);target=trade_date.isoformat();snap=build_snapshot(provider,required_symbols(runtime_root),trade_date,cutoff,max_staleness_minutes,max_workers=snapshot_workers);status='PARTIAL' if snap.get('errors') and snap.get('rows') else 'DATA_HOLD' if snap.get('errors') else 'OK';snap={**snap,'status':status};_write_snapshot(optimizer_root/'ledger/snapshots'/f'{target}.close.json',snap)
 if snap.get('rows'):apply_snapshot_to_runtime(runtime_root,snap['rows'],'live_optimizer:close')
 _write_unreconciled(optimizer_root,target,snap.get('errors',[]));observable_at=datetime.combine(trade_date,cutoff).isoformat();_append_execution_references(optimizer_root,snap,observable_at);m=drain_pending_trades(optimizer_root/'ledger/pending_closed_trades.jsonl',optimizer_root/'ledger/closed_trades.csv',observable_at);opt=optimizer_callback(optimizer_root,observable_at) if optimizer_callback else None;return {'trade_date':target,'status':status,'errors':snap.get('errors',[]),'matured_trades':len(m),'optimizer':opt}
def main():
 p=argparse.ArgumentParser();p.add_argument('--optimizer-root',required=True);p.add_argument('--workspace-config',required=True);p.add_argument('--date',default=date.today().isoformat());a=p.parse_args();cfg=json.loads(Path(a.workspace_config).read_text());pcfg=cfg.get('provider',{});provider=FileMinuteProvider(Path(pcfg['file_root'])) if pcfg.get('kind')=='file' else EastmoneyMinuteProvider();r=run_close_cycle(Path(a.optimizer_root),Path(cfg['runtime_root']),provider,date.fromisoformat(a.date),time.fromisoformat(cfg.get('close_cutoff','15:00')),int(cfg.get('close_staleness_minutes',5)),None,int(cfg.get('close_workers',8)));print(json.dumps(r,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
