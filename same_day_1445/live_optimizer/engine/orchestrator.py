from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .active_policy import resolve_active_policy,ActivePolicyError,runtime_executable_candidate
from .candidate_generator import generate_candidates
from .config import load_optimizer_config,stable_json_hash
from .daily_monitor import detect_daily_signal_drift
from .drift import detect_drift
from .gate import evaluate_gate
from .ledger import AppendOnlyCsvLedger
from .records import OptimizerState
from .release_manager import promote_candidate,ReleaseError
from .rollback_guard import evaluate_live_rollback
from .rolling_evaluator import compute_metrics,evaluate_candidate_windows,matched_opportunity_rows

def _read_state(root,cfg):
    p=root/'state/optimizer_state.json'
    if p.exists():return OptimizerState.from_dict(json.loads(p.read_text(encoding='utf-8-sig')))
    s=OptimizerState.initial();s.mode=cfg['mode'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(s.to_dict(),sort_keys=True,indent=2)+'\n');return s
def _save_state(root,state):
    p=root/'state/optimizer_state.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(state.to_dict(),ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def _score(e):m=e['overall'];return (m.win_rate,m.mean_return,m.final_equity)
def _read_positions(root):
    p=root/'state/forward_positions.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() and p.read_text(encoding='utf-8').strip() else {}
def _transition_safe(root,line,formal,candidate):
    pos=_read_positions(root);a=pos.get(f'{line}:{formal}');b=pos.get(f'{line}:{candidate}')
    if a is None and b is None:return True
    if a is None or b is None:return False
    return str(a.get('symbol',''))==str(b.get('symbol','')) and str(a.get('entry_date',''))==str(b.get('entry_date',''))
def _window_dates(rows,count):
    dates=sorted({str(r.get('decision_at') or r.get('entry_date') or '')[:10] for r in rows if str(r.get('decision_at') or r.get('entry_date') or '')[:10]})
    if not dates:return []
    n=max(1,int(count));parts=[]
    for i in range(n):
        lo=(len(dates)*i)//n;hi=(len(dates)*(i+1))//n
        if hi>lo:parts.append({'name':f'forward_{i+1}','start':dates[lo],'end':dates[hi-1]})
    return parts

def _write_decision_artifacts(root,result):
    p=root/'state/latest_optimizer_decision.json';p.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n';tmp=p.with_suffix('.json.tmp');tmp.write_text(payload,encoding='utf-8');tmp.replace(p)

def run_optimizer(root:Path,as_of:str)->dict[str,Any]:
    root=Path(root);cfg=load_optimizer_config(root/'config/optimizer.json');state=_read_state(root,cfg);state.mode=cfg['mode'] if state.mode not in {'DATA_HOLD','ROLLBACK'} else state.mode;ledger=AppendOnlyCsvLedger(root/'ledger/closed_trades.csv',('sample_id','candidate_id'));rows=ledger.rows();lines_out={}
    for line in ('C','S','D','R'):
        formal=state.active_formal[line]
        try:formal_policy=resolve_active_policy(root,line,formal);formal_evidence_id=formal_policy.evidence_id
        except ActivePolicyError:formal_evidence_id=formal
        line_rows=[r for r in rows if r.get('line')==line];baseline_all=[r for r in line_rows if r.get('candidate_id')==formal_evidence_id];anchor_all=[r for r in line_rows if r.get('candidate_id')=='release_v2'];candidate_ids=sorted({r.get('candidate_id') for r in line_rows if r.get('candidate_id') and r.get('candidate_id') not in {formal_evidence_id,'release_v2'}});accounting=cfg['lines'][line].get('accounting','research_sum' if line=='D' else 'compound');baseline_eval={'overall':compute_metrics(baseline_all,accounting)};anchor_eval={'overall':compute_metrics(anchor_all,accounting)}
        last=state.last_promotion_at.get(line);new_baseline=[r for r in baseline_all if not last or str(r.get('observable_at',''))>str(last)];ordered=sorted(new_baseline,key=lambda r:(str(r.get('decision_at','')),str(r.get('sample_id',''))));recent_count=int(cfg.get('drift',{}).get('recent_count',20));drift=detect_drift(ordered[-recent_count:],ordered[:-recent_count],cfg) if recent_count>0 and len(ordered)>=recent_count*2 else {'severe':False,'breaches':[],'deltas':{},'recent':{},'reference':{}};daily_drift=detect_daily_signal_drift(root,line,cfg)
        if (drift.get('severe') or daily_drift.get('severe')) and state.mode=='NORMAL':state.mode='SHADOW_ONLY'
        rollback_info={'status':'NONE','breaches':[]}
        if formal!='release_v2' and state.previous_formal.get(line):
            parent_release=state.previous_formal[line]
            try:parent_policy=resolve_active_policy(root,line,parent_release);parent_id=parent_policy.evidence_id
            except ActivePolicyError:parent_id=parent_release
            current_since=[r for r in line_rows if r.get('candidate_id')==formal_evidence_id and (not last or str(r.get('observable_at',''))>str(last))];parent_since=[r for r in line_rows if r.get('candidate_id')==parent_id and (not last or str(r.get('observable_at',''))>str(last))];rollback_info=evaluate_live_rollback(line,current_since,parent_since,accounting,cfg)
            if rollback_info.get('rollback'):
                from .release_manager import rollback
                state=rollback(root,state,line,'LIVE_REGRESSION',as_of);rollback_info['status']='ROLLED_BACK';formal=state.active_formal[line];formal_evidence_id=parent_id;baseline_all=parent_since;baseline_eval={'overall':compute_metrics(baseline_all,accounting)};candidate_ids=[]
        candidates=[];eval_map={}
        for cid in candidate_ids:
            c_rows=[r for r in line_rows if r.get('candidate_id')==cid];matched_c,matched_b=matched_opportunity_rows(c_rows,baseline_all);windows=_window_dates(matched_b,int(cfg.get('forward_evaluation',{}).get('window_count',3)));ev=evaluate_candidate_windows(matched_c,matched_b,windows,accounting);eval_map[cid]=(ev,matched_c,matched_b)
        preliminary={}
        for cid,(ev,mc,mb) in eval_map.items():preliminary[cid]=evaluate_gate(line,ev,{'overall':compute_metrics(mb,accounting)},mc,state,cfg,as_of,ignore_neighbor=True).passed
        generated={x['candidate_id']:x['params'] for x in generate_candidates(cfg,line)};space=cfg.get('candidate_spaces',{}).get(line,{})
        for cid,(ev,mc,mb) in eval_map.items():
            params=generated.get(cid);neighbors=[]
            if params:
                for key,values in space.items():
                    vals=list(values);idx=vals.index(params[key])
                    for j in (idx-1,idx+1):
                        if 0<=j<len(vals):
                            p=dict(params);p[key]=vals[j];sig={'line':line,'params':p};nid=f"{line}_{stable_json_hash(sig)[:12]}";neighbors.append(nid)
            ev['neighbor_pass_rate']=sum(preliminary.get(n,False) for n in set(neighbors))/len(set(neighbors)) if neighbors else 1.0;gate=evaluate_gate(line,ev,{'overall':compute_metrics(mb,accounting)},mc,state,cfg,as_of);anchor_gate=evaluate_gate(line,ev,anchor_eval,mc,state,cfg,as_of) if anchor_all else gate
            if not anchor_gate.passed and gate.passed:gate=type(gate)(False,list(dict.fromkeys([*gate.reasons,'V2_ANCHOR'])),{**gate.evidence,'v2_anchor_reasons':anchor_gate.reasons})
            candidates.append({'candidate_id':cid,'evaluation':ev,'gate':gate})
        eligible=[x for x in candidates if x['gate'].passed];promotion={'status':'NONE','candidate_id':None}
        if eligible:
            winner=max(eligible,key=lambda x:_score(x['evaluation']));leader=winner['candidate_id'];state.shadow_leader[line]=leader
            if state.mode=='NORMAL' and runtime_executable_candidate(root,line,leader):
                if _transition_safe(root,line,formal,leader):
                    try:state=promote_candidate(root,state,line,leader,winner['gate'],as_of);state.pending_promotion[line]=None;promotion={'status':'PROMOTED','candidate_id':leader,'release_id':state.active_formal[line]}
                    except ReleaseError as exc:promotion={'status':'BLOCKED','candidate_id':leader,'reason':str(exc)}
                else:state.pending_promotion[line]=leader;promotion={'status':'PENDING_TRANSITION','candidate_id':leader}
        else:
            state.pending_promotion[line]=None
            if state.mode=='SHADOW_ONLY':promotion={'status':'SHADOW_ONLY','candidate_id':state.shadow_leader[line]}
        lines_out[line]={'formal':state.active_formal[line],'formal_evidence_id':formal_evidence_id,'shadow_leader':state.shadow_leader[line],'promotion':promotion,'rollback':rollback_info,'drift':drift,'daily_drift':daily_drift,'candidates':[{'candidate_id':x['candidate_id'],'gate':x['gate'].to_dict(),'overall':x['evaluation']['overall'].to_dict()} for x in candidates]}
    _save_state(root,state);core={'as_of':as_of,'mode':state.mode,'lines':lines_out};dh=stable_json_hash(core);result={**core,'decision_hash':dh,'status':'OK'};_write_decision_artifacts(root,result);AppendOnlyCsvLedger(root/'ledger/optimizer_runs.csv',('decision_hash',)).append({'decision_hash':dh,'as_of':as_of,'mode':state.mode,'state_hash':stable_json_hash(state.to_dict()),'line_summary_hash':stable_json_hash(lines_out)});return result
