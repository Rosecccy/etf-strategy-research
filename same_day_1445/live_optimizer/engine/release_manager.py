from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import stable_json_hash
from .ledger import AppendOnlyCsvLedger
from .records import GateResult, OptimizerState

class ReleaseError(RuntimeError): pass

def _write_json_atomic(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def _save_state(root:Path,state:OptimizerState)->None:_write_json_atomic(root/'state/optimizer_state.json',state.to_dict())
def _promotion_ledger(root:Path)->AppendOnlyCsvLedger:return AppendOnlyCsvLedger(root/'ledger/promotion.csv',('event_id',))
def write_candidate_release(root:Path,candidate_id:str,manifest:dict[str,Any])->Path:
    path=Path(root)/'candidates'/candidate_id/'manifest.json'
    if path.exists():raise ReleaseError(f'candidate release exists: {candidate_id}')
    _write_json_atomic(path,{'candidate_id':candidate_id,**manifest});return path
def promote_candidate(root:Path,state:OptimizerState,line:str,candidate_id:str,gate:GateResult,timestamp:str)->OptimizerState:
    root=Path(root)
    if state.mode!='NORMAL':raise ReleaseError(f'formal promotion blocked in {state.mode} mode')
    if not gate.passed:raise ReleaseError('promotion gate did not pass')
    candidate_path=root/'candidates'/candidate_id/'manifest.json'
    if not candidate_path.exists():raise ReleaseError(f'candidate does not exist: {candidate_id}')
    stamp=datetime.fromisoformat(timestamp.replace('Z','+00:00')).strftime('%Y%m%dT%H%M%S');release_id=f'auto_{line}_{stamp}_{candidate_id[:12]}';release_path=root/'releases'/release_id/'manifest.json'
    if release_path.exists():raise ReleaseError(f'release exists: {release_id}')
    parent=state.active_formal[line];_write_json_atomic(release_path,{'release_id':release_id,'line':line,'candidate_id':candidate_id,'parent_release':parent,'promoted_at':timestamp,'gate':gate.to_dict()});new=deepcopy(state);new.previous_formal[line]=parent;new.active_formal[line]=release_id;new.last_promotion_at[line]=timestamp;new.pending_promotion[line]=None;_save_state(root,new);_promotion_ledger(root).append({'event_id':stable_json_hash({'event':'PROMOTE','line':line,'release':release_id,'timestamp':timestamp})[:24],'timestamp':timestamp,'line':line,'event':'PROMOTE','from_release':parent,'to_release':release_id,'reason':'GATE_PASS'});return new
def rollback(root:Path,state:OptimizerState,line:str,reason:str,timestamp:str)->OptimizerState:
    root=Path(root);previous=state.previous_formal.get(line)
    if not previous:raise ReleaseError(f'no rollback release for {line}')
    current=state.active_formal[line];new=deepcopy(state);new.active_formal[line]=previous;new.previous_formal[line]=None;new.last_promotion_at[line]=timestamp;new.pending_promotion[line]=None;new.mode='ROLLBACK';_save_state(root,new);_promotion_ledger(root).append({'event_id':stable_json_hash({'event':'ROLLBACK','line':line,'from':current,'to':previous,'timestamp':timestamp})[:24],'timestamp':timestamp,'line':line,'event':'ROLLBACK','from_release':current,'to_release':previous,'reason':reason});return new
