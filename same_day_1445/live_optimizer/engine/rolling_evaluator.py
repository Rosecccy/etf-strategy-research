from __future__ import annotations
from datetime import date,datetime
from statistics import mean
from typing import Any
from .audit import audit_causal_rows
from .records import EvaluationMetrics

def _bool(v):return v if isinstance(v,bool) else str(v).strip().lower() in {'1','true','yes','y'}
def _ret(r):
    try:return float(r.get('ret',0.0) or 0.0)
    except (TypeError,ValueError):return 0.0
def compute_metrics(rows,accounting):
    opportunities=len(rows);tr=[r for r in rows if _bool(r.get('triggered',True))];rets=[_ret(r) for r in tr];n=len(rets);wr=sum(r>0 for r in rets)/n if n else 0.0;mr=mean(rets) if rets else 0.0
    if accounting not in {'compound','research_sum'}:raise ValueError(f'unknown accounting: {accounting}')
    eq=10000.0;peak=eq;dd=0.0
    for r in rets:eq=eq*(1+r) if accounting=='compound' else eq+10000*r;peak=max(peak,eq);dd=min(dd,eq/peak-1.0 if peak else 0.0)
    return EvaluationMetrics(n,opportunities,n/opportunities if opportunities else 0.0,wr,mr,dd,eq)
def _row_date(row):return datetime.fromisoformat(str(row.get('entry_date') or row.get('decision_at') or '')[:10]).date()
def _rows_for_window(rows,w):
    if w.get('opportunity_ids'):
        ids={str(x) for x in w['opportunity_ids']};return [r for r in rows if str(r.get('opportunity_id','')) in ids]
    start=date.fromisoformat(str(w['start'])[:10]);end=date.fromisoformat(str(w['end'])[:10]);return [r for r in rows if start<=_row_date(r)<=end]
def evaluate_candidate_windows(candidate_rows,baseline_rows,windows,accounting):
    cc=audit_causal_rows(candidate_rows,'decision_at','observable_at') if candidate_rows else {'ok':True,'violations':[]};bc=audit_causal_rows(baseline_rows,'decision_at','observable_at') if baseline_rows else {'ok':True,'violations':[]};violations=cc['violations']+bc['violations'];overall=compute_metrics(candidate_rows,accounting);bo=compute_metrics(baseline_rows,accounting);wr=[]
    for w in windows:
        cr=_rows_for_window(candidate_rows,w);br=_rows_for_window(baseline_rows,w);cm=compute_metrics(cr,accounting);bm=compute_metrics(br,accounting);wd=cm.win_rate-bm.win_rate;md=cm.mean_return-bm.mean_return;fd=cm.final_equity-bm.final_equity;wr.append({'name':str(w['name']),'candidate':cm.to_dict(),'baseline':bm.to_dict(),'win_rate_delta':wd,'mean_return_delta':md,'final_equity_delta':fd,'improved':wd>=0 and md>=0 and fd>=0})
    return {'overall':overall,'baseline_overall':bo,'windows':wr,'causal':not violations,'violations':violations,'neighbor_pass_rate':0.0}
def matched_opportunity_rows(candidate_rows,baseline_rows):
    c={str(r.get('opportunity_id','')):r for r in candidate_rows if str(r.get('opportunity_id',''))};b={str(r.get('opportunity_id','')):r for r in baseline_rows if str(r.get('opportunity_id',''))};ids=sorted(set(c)&set(b));return [c[i] for i in ids],[b[i] for i in ids]
