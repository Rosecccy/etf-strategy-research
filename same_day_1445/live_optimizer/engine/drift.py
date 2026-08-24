from __future__ import annotations
from statistics import mean
from typing import Any
def _bool(v): return v if isinstance(v,bool) else str(v).lower() in {'1','true','yes','y'}
def _stats(rows):
    if not rows:return {'trigger_rate':0.0,'win_rate':0.0,'mean_return':0.0}
    trig=[_bool(r.get('triggered',True)) for r in rows]; tr=[r for r,t in zip(rows,trig) if t]
    wins=[(_bool(r.get('won')) if 'won' in r else float(r.get('ret',0) or 0)>0) for r in tr]; rets=[float(r.get('ret',0) or 0) for r in tr]
    return {'trigger_rate':sum(trig)/len(trig),'win_rate':sum(wins)/len(wins) if wins else 0.0,'mean_return':mean(rets) if rets else 0.0}
def detect_drift(recent_rows,reference_rows,config):
    recent=_stats(recent_rows);ref=_stats(reference_rows);dc=config.get('drift',{});limits={'trigger_rate':float(dc.get('trigger_rate_abs_max',.3)),'win_rate':float(dc.get('win_rate_abs_max',.3)),'mean_return':float(dc.get('mean_return_abs_max',.1))};d={k:recent[k]-ref[k] for k in limits};b=[k for k,l in limits.items() if abs(d[k])>l];return {'severe':bool(b),'breaches':b,'deltas':d,'recent':recent,'reference':ref}
