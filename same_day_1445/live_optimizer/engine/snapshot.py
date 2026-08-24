from __future__ import annotations
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
from datetime import date,datetime,time,timedelta
from typing import Any
from ..providers.base import MinuteProvider

def _hash_bars(bars):return hashlib.sha256(json.dumps([[b.timestamp.isoformat(),b.open,b.high,b.low,b.close,b.volume,b.amount] for b in bars],separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def _snapshot_symbol(provider,symbol,trade_date,cutoff_dt,max_staleness_minutes):
    try:fetched=provider.fetch(symbol,trade_date)
    except Exception as exc:return None,{'symbol':symbol,'reason':'provider_error','error_type':type(exc).__name__}
    bars=[b for b in fetched if b.timestamp<=cutoff_dt]
    if not bars:return None,{'symbol':symbol,'reason':'missing_minute_data'}
    latest=bars[-1].timestamp
    if cutoff_dt-latest>timedelta(minutes=max_staleness_minutes):return None,{'symbol':symbol,'reason':'stale_minute_data','latest_observed_at':latest.isoformat()}
    return {'date':trade_date.isoformat(),'symbol':str(symbol).zfill(6),'open':bars[0].open,'high':max(b.high for b in bars),'low':min(b.low for b in bars),'close':bars[-1].close,'volume':float(sum(b.volume for b in bars)),'amount':float(sum(b.amount for b in bars)),'requested_cutoff':cutoff_dt.isoformat(),'latest_observed_at':latest.isoformat(),'provider':provider.name,'source_hash':_hash_bars(bars),'bar_count':len(bars)},None
def build_snapshot(provider,symbols,trade_date,cutoff,max_staleness_minutes=2,max_workers=1):
    cutoff_dt=datetime.combine(trade_date,cutoff);ordered=[str(s).zfill(6) for s in symbols]
    def collect(s):return _snapshot_symbol(provider,s,trade_date,cutoff_dt,max_staleness_minutes)
    workers=max(1,int(max_workers));results=[collect(s) for s in ordered] if workers==1 or len(ordered)<=1 else list(ThreadPoolExecutor(max_workers=min(workers,len(ordered))).map(collect,ordered));rows=[];errors=[]
    for row,error in results:
        if row is not None:rows.append(row)
        if error is not None:errors.append(error)
    return {'status':'DATA_HOLD' if errors else 'OK','trade_date':trade_date.isoformat(),'cutoff':cutoff_dt.isoformat(),'rows':rows,'errors':errors}
