from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
ALLOWED_MODES={"NORMAL","SHADOW_ONLY","DATA_HOLD","ROLLBACK"}; LINES={"C","S","D","R"}
def stable_json_hash(value: object)->str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def load_optimizer_config(path: Path)->dict[str,Any]:
    cfg=json.loads(path.read_text(encoding='utf-8-sig')); mode=str(cfg.get('mode',''))
    if mode not in ALLOWED_MODES: raise ValueError(f'invalid optimizer mode: {mode}')
    lines=cfg.get('lines',{})
    if set(lines)!=LINES: raise ValueError(f'lines must be exactly {sorted(LINES)}')
    if not isinstance(cfg.get('gates',{}),dict): raise ValueError('gates must be an object')
    if not isinstance(cfg.get('candidate_spaces',{}),dict): raise ValueError('candidate_spaces must be an object')
    return cfg
