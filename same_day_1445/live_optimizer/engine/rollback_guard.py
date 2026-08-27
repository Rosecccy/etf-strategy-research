from __future__ import annotations
from typing import Any
from .rolling_evaluator import compute_metrics

def evaluate_rollback_guard(line,active_rows,parent_rows,accounting,config):
    cfg=config.get('rollback',{})
    if not cfg.get('enabled',False):return {'triggered':False,'status':'DISABLED','breaches':[],'new_samples':len(active_rows)}
    mins=cfg.get('minimum_new_samples',{});minimum=int(mins.get(line,0) if isinstance(mins,dict) else mins or 0)
    if len(active_rows)<minimum:return {'triggered':False,'status':'MONITORING','breaches':[],'new_samples':len(active_rows),'minimum_new_samples':minimum}
    active=compute_metrics(active_rows,accounting);parent=compute_metrics(parent_rows,accounting);wd=active.win_rate-parent.win_rate;md=active.mean_return-parent.mean_return;dd=max(0.0,parent.max_drawdown-active.max_drawdown);b=[]
    if active.trigger_retention<float(cfg.get('trigger_retention_min',.5)):b.append('TRIGGER_RETENTION')
    if wd<float(cfg.get('win_rate_delta_min',-.2)):b.append('WIN_RATE')
    if md<float(cfg.get('mean_return_delta_min',-.01)):b.append('MEAN_RETURN')
    if dd>float(cfg.get('max_drawdown_deterioration_max',.2)):b.append('MAX_DRAWDOWN')
    n=int(cfg.get('breach_count_min',2));return {'triggered':len(b)>=n,'status':'ROLLBACK' if len(b)>=n else 'OK','breaches':b,'new_samples':len(active_rows),'minimum_new_samples':minimum,'metrics':{'active':active.to_dict(),'parent':parent.to_dict(),'win_rate_delta':wd,'mean_return_delta':md,'drawdown_deterioration':dd}}
evaluate_live_rollback=evaluate_rollback_guard
