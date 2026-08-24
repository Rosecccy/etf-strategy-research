from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

def _bool(value:Any,default:bool=True)->bool:
    if value is None or str(value).strip()=="": return default
    if isinstance(value,bool): return value
    return str(value).strip().lower() in {"1","true","yes","y"}
def _read_rows(path:Path)->list[dict[str,str]]:
    if not path.exists() or path.stat().st_size==0:return []
    with path.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def _daily_rows(rows:list[dict[str,Any]],line:str)->list[dict[str,Any]]:
    latest={}
    for row in rows:
        if str(row.get("line",""))!=str(line):continue
        day=str(row.get("decision_date",""))[:10]
        if not day:continue
        old=latest.get(day)
        if old is None or str(row.get("observed_at",""))>=str(old.get("observed_at","")):latest[day]=row
    return [latest[d] for d in sorted(latest)]
def summarize_daily_signals(rows:list[dict[str,Any]])->dict[str,float]:
    if not rows:return {"days":0.0,"actionable_rate":0.0,"data_hold_rate":0.0,"buy_rate":0.0,"sell_rate":0.0,"hold_rate":0.0}
    actions=[str(r.get("action","HOLD")).strip().upper() for r in rows];trad=[_bool(r.get("tradable"),True) for r in rows];status=[str(r.get("status","OK")).strip().upper() for r in rows];n=len(rows)
    return {"days":float(n),"actionable_rate":sum(ok and a in {"BUY","SELL"} for ok,a in zip(trad,actions))/n,"data_hold_rate":sum((not ok) or st=="DATA_HOLD" for ok,st in zip(trad,status))/n,"buy_rate":sum(a=="BUY" and ok for a,ok in zip(actions,trad))/n,"sell_rate":sum(a=="SELL" and ok for a,ok in zip(actions,trad))/n,"hold_rate":sum(a=="HOLD" and ok for a,ok in zip(actions,trad))/n}
def detect_daily_signal_drift(root:Path,line:str,config:dict[str,Any])->dict[str,Any]:
    dc=config.get("drift",{});recent_days=int(dc.get("daily_recent_days",20) or 0);min_ref=int(dc.get("daily_min_reference_days",20) or 0)
    if recent_days<=0 or min_ref<=0:return {"severe":False,"breaches":[],"deltas":{},"recent":{},"reference":{},"eligible":False}
    rows=_daily_rows(_read_rows(Path(root)/"ledger"/"formal_signals.csv"),line)
    if len(rows)<recent_days+min_ref:return {"severe":False,"breaches":[],"deltas":{},"recent":summarize_daily_signals(rows[-recent_days:] if recent_days else []),"reference":summarize_daily_signals(rows[:-recent_days] if recent_days else rows),"eligible":False}
    recent_rows=rows[-recent_days:];ref_rows=rows[:-recent_days];recent=summarize_daily_signals(recent_rows);ref=summarize_daily_signals(ref_rows);ad=recent["actionable_rate"]-ref["actionable_rate"]
    deltas={"actionable_rate":ad,"hold_rate":recent["hold_rate"]-ref["hold_rate"],"data_hold_rate":recent["data_hold_rate"]-ref["data_hold_rate"]};breaches=[]
    if abs(ad)>float(dc.get("daily_actionable_rate_abs_max",0.5)):breaches.append("ACTIONABLE_RATE_SHIFT")
    if recent["data_hold_rate"]>float(dc.get("daily_data_hold_rate_max",0.5)):breaches.append("DATA_HOLD_RATE")
    return {"severe":bool(breaches),"breaches":breaches,"deltas":deltas,"recent":recent,"reference":ref,"eligible":True,"recent_start":str(recent_rows[0].get("decision_date",""))[:10],"recent_end":str(recent_rows[-1].get("decision_date",""))[:10]}
