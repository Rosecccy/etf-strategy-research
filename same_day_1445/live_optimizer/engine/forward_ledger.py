from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .signal_adapter import NormalizedSignal

class ForwardPositionBook:
    def __init__(self,path):
        self.path=Path(path); self.state=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() and self.path.read_text(encoding='utf-8').strip() else {}
    def _save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True); self.path.write_text(json.dumps(self.state,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    def get(self,line,variant): return self.state.get(f'{line}:{variant}')
    def apply(self,line,variant,signal,price,observed_at):
        if not signal.tradable:return None
        key=f'{line}:{variant}';pos=self.state.get(key)
        if signal.action=='BUY' and pos is None:
            self.state[key]={'symbol':signal.symbol,'entry_date':signal.decision_date,'entry_price':float(price),'entry_observed_at':observed_at,'name':signal.name,'source_line':signal.metadata.get('source_line',line),'peak_price':float(price),'hold_days':0,'opportunity_id':signal.metadata.get('opportunity_id',''),'regime':signal.metadata.get('regime','')};self._save();return None
        if pos is not None:
            pos['peak_price']=max(float(pos.get('peak_price',pos['entry_price'])),float(price));pos['hold_days']=int(pos.get('hold_days',0))+1
            if signal.action=='SELL' and (not signal.symbol or signal.symbol==pos['symbol']):
                entry=float(pos['entry_price']);exit_price=float(price);trade={'line':line,'candidate_id':variant,'opportunity_id':pos.get('opportunity_id',''),'symbol':pos['symbol'],'entry_date':pos['entry_date'],'exit_date':signal.decision_date,'entry_price':entry,'exit_price':exit_price,'ret':round(exit_price/entry-1.0,12),'triggered':1,'regime':pos.get('regime') or signal.metadata.get('regime',''),'entry_observed_at':pos.get('entry_observed_at',''),'observable_at':observed_at,'source_line':pos.get('source_line',line)};del self.state[key];self._save();return trade
            self.state[key]=pos;self._save()
        return None

class SkippedOpportunityBook:
    def __init__(self,path):
        self.path=Path(path); self.state=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() and self.path.read_text(encoding='utf-8').strip() else {}
    def _save(self): self.path.parent.mkdir(parents=True,exist_ok=True);self.path.write_text(json.dumps(self.state,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    def register(self,line,candidate_id,signal,observed_at):
        oid=str(signal.metadata.get('opportunity_id',''))
        if not oid: raise ValueError('skipped opportunity requires opportunity_id')
        key=f'{line}:{candidate_id}:{oid}';record={'line':line,'candidate_id':candidate_id,'opportunity_id':oid,'symbol':signal.symbol,'entry_date':signal.decision_date,'entry_observed_at':observed_at,'regime':signal.metadata.get('regime',''),'source_line':signal.metadata.get('source_line',line)};old=self.state.get(key)
        if old is not None and old!=record: raise RuntimeError(f'skipped opportunity conflict: {key}')
        self.state[key]=record;self._save()
    def finalize(self,formal_trade,observed_at):
        oid=str(formal_trade.get('opportunity_id',''));line=str(formal_trade.get('line',''));out=[]
        for key,record in list(self.state.items()):
            if record.get('line')!=line or record.get('opportunity_id')!=oid:continue
            out.append({'line':line,'candidate_id':record['candidate_id'],'opportunity_id':oid,'symbol':formal_trade.get('symbol',record.get('symbol','')),'entry_date':formal_trade.get('entry_date',record.get('entry_date','')),'exit_date':formal_trade.get('exit_date',''),'entry_price':'','exit_price':'','ret':0.0,'triggered':0,'regime':record.get('regime') or formal_trade.get('regime',''),'entry_observed_at':record.get('entry_observed_at',''),'observable_at':observed_at,'source_line':record.get('source_line',line)});del self.state[key]
        if out:self._save()
        return out

def _canonical_row(row): return json.dumps({k:row[k] for k in sorted(row)},ensure_ascii=False,sort_keys=True,separators=(',',':'))
def append_csv_record(path,row,key_fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);rows=[];fields=list(row.keys())
    if path.exists():
        with path.open('r',encoding='utf-8-sig',newline='') as h: reader=csv.DictReader(h);rows=list(reader);fields=list(reader.fieldnames or fields)
        key=tuple(str(row.get(f,'')) for f in key_fields)
        for old in rows:
            if tuple(str(old.get(f,'')) for f in key_fields)==key:
                comparable={f:str(row.get(f,'')) for f in fields}
                if all(str(old.get(f,''))==comparable[f] for f in fields):return False
                raise RuntimeError(f'append-only ledger conflict: {key}')
    for k in row:
        if k not in fields:fields.append(k)
    normalized=[{f:r.get(f,'') for f in fields} for r in rows]+[{f:row.get(f,'') for f in fields}]
    fd,tmp_name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=str(path.parent));os.close(fd);tmp=Path(tmp_name)
    try:
        with tmp.open('w',encoding='utf-8-sig',newline='') as h: w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(normalized)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
    return True
def append_pending_trade(path,trade):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as h:h.write(_canonical_row(trade)+'\n')
def drain_pending_trades(pending_path,closed_csv,observable_at):
    pending_path=Path(pending_path)
    if not pending_path.exists():return []
    rows=[json.loads(line) for line in pending_path.read_text(encoding='utf-8').splitlines() if line.strip()];matured=[]
    for row in rows:
        row=dict(row); row['observable_at']=observable_at; row['decision_at']=observable_at; oid=str(row.get('opportunity_id','')); core={'line':row.get('line'),'candidate_id':row.get('candidate_id'),'opportunity_id':oid,'entry_date':row.get('entry_date'),'exit_date':row.get('exit_date')};row['sample_id']=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24];append_csv_record(closed_csv,row,('sample_id','candidate_id'));matured.append(row)
    pending_path.unlink(missing_ok=True);return matured
