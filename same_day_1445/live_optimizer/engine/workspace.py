from __future__ import annotations
import csv,hashlib,os,statistics,tempfile
from pathlib import Path
class WorkspaceMutationError(RuntimeError):pass
def _hash_rows(rows,fields):return hashlib.sha256('\n'.join('|'.join(str(r.get(f,'')) for f in fields) for r in rows).encode()).hexdigest()
def upsert_daily_bar(path,bar,source_file='live_optimizer'):
    path=Path(path)
    if not path.exists():raise WorkspaceMutationError(f'missing daily file: {path}')
    with path.open('r',encoding='utf-8-sig',newline='') as h:reader=csv.DictReader(h);fields=list(reader.fieldnames or []);rows=list(reader)
    if 'date' not in fields:raise WorkspaceMutationError('daily file has no date column')
    td=str(bar['date']);dates=[str(r.get('date','')) for r in rows]
    if dates and td<max(dates):raise WorkspaceMutationError('refuse to rewrite a prior date')
    prior=[r for r in rows if str(r.get('date',''))<td];same=[r for r in rows if str(r.get('date',''))==td]
    if len(same)>1:raise WorkspaceMutationError('duplicate current-date rows')
    ph=_hash_rows(prior,fields);template=dict(same[0] if same else (rows[-1] if rows else {}));current={f:template.get(f,'') for f in fields}
    for k in ('date','symbol','open','high','low','close','volume','amount'):
        if k in fields and k in bar:current[k]=str(bar[k])
    if 'name' in fields and bar.get('name'):current['name']=str(bar['name'])
    if 'source_file' in fields:current['source_file']=source_file
    new=prior+[current];fd,tmp_name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=str(path.parent));os.close(fd);tmp=Path(tmp_name)
    try:
        with tmp.open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(new)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
    return {'path':str(path),'date':td,'prior_rows_hash':ph,'current_row_hash':_hash_rows([current],fields)}
def required_symbols(runtime_root):
    root=Path(runtime_root);symbols=set()
    for rel in ('C/raw/etf','S/raw/etf'):
        d=root/rel
        if d.exists():symbols.update(p.stem.zfill(6) for p in d.glob('*.csv'))
    return sorted(symbols)
def apply_snapshot_to_runtime(runtime_root,snapshot_rows,source_file):
    root=Path(runtime_root);audits=[]
    for row in snapshot_rows:
        symbol=str(row['symbol']).zfill(6)
        for line in ('C','S'):
            path=root/line/'raw/etf'/f'{symbol}.csv'
            if path.exists():audits.append({'line':line,'symbol':symbol,**upsert_daily_bar(path,row,source_file)})
    return audits
def _raw_path(root,symbol):
    for line in ('C','S'):
        p=root/line/'raw/etf'/f'{symbol}.csv'
        if p.exists():return p
def prior_closes(runtime_root,symbol,trade_date):
    p=_raw_path(Path(runtime_root),str(symbol).zfill(6))
    if p is None:return []
    with p.open('r',encoding='utf-8-sig',newline='') as h:rows=[r for r in csv.DictReader(h) if str(r.get('date',''))<trade_date]
    rows.sort(key=lambda r:str(r.get('date','')));out=[]
    for r in rows:
        try:out.append(float(r['close']))
        except Exception:pass
    return out
def d_strict_features(runtime_root,symbol,trade_date,current_price,market_proxy='510500'):
    own=prior_closes(runtime_root,symbol,trade_date);market=prior_closes(runtime_root,market_proxy,trade_date);ma20=statistics.fmean(own[-20:]) if len(own)>=20 else 0.0;mret=market[-1]/market[-121]-1 if len(market)>=121 and market[-121] else 0.0;return {'price':float(current_price),'ma20':ma20,'market_ret120':mret}
def required_symbol_groups(runtime_root):
    root=Path(runtime_root);groups={}
    for line in ('C','S'):
        d=root/line/'raw/etf';groups[line]=sorted(p.stem.zfill(6) for p in d.glob('*.csv')) if d.exists() else []
    groups['D']=list(groups['C']);groups['R']=sorted(set(groups['C'])|set(groups['S']));return groups
def market_regime(runtime_root,trade_date,market_proxy='510500'):
    market=prior_closes(runtime_root,market_proxy,trade_date)
    if len(market)<121 or not market[-121]:return 'unknown',None
    r=market[-1]/market[-121]-1
    return ('weak' if r<=-.05 else 'strong' if r>=.05 else 'neutral'),r
