from __future__ import annotations
import itertools
from typing import Any
from .config import stable_json_hash

def generate_candidates(config: dict[str, Any], line: str) -> list[dict[str, Any]]:
    space=config.get('candidate_spaces',{}).get(line,{})
    if not space:return []
    keys=sorted(space);values=[list(space[k]) for k in keys];out=[]
    for combo in itertools.product(*values):
        params=dict(zip(keys,combo));sig={'line':line,'params':params};cid=f"{line}_{stable_json_hash(sig)[:12]}";out.append({'line':line,'candidate_id':cid,'params':params})
    return out
