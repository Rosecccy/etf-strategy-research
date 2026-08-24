from __future__ import annotations
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
def _dt(value:Any)->datetime: return datetime.fromisoformat(str(value).replace('Z','+00:00'))
def audit_causal_rows(rows:list[dict[str,Any]],decision_field:str,observable_field:str)->dict[str,Any]:
    violations=[]
    for row in rows:
        if _dt(row[observable_field])>_dt(row[decision_field]): violations.append(dict(row))
    return {'ok':not violations,'violations':violations}
def hash_file(path:Path)->str:
    d=hashlib.sha256()
    with Path(path).open('rb') as h:
        for c in iter(lambda:h.read(1024*1024),b''): d.update(c)
    return d.hexdigest()
