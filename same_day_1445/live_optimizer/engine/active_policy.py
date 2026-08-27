from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

class ActivePolicyError(RuntimeError): pass

@dataclass(frozen=True)
class ActivePolicy:
    line: str
    release_id: str
    evidence_id: str
    module: str
    params: dict[str, Any]

SUPPORTED_RUNTIME_MODULES = {"BASE_V2", "C_DELAY1", "D_STRICT", "D_STRICT_GRID"}

def _read_json(path: Path) -> dict[str, Any]:
    try: return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc: raise ActivePolicyError(f"missing active policy artifact: {path}") from exc
    except json.JSONDecodeError as exc: raise ActivePolicyError(f"invalid active policy artifact: {path}") from exc

def resolve_active_policy(root: Path, line: str, release_id: str) -> ActivePolicy:
    root=Path(root)
    if release_id=="release_v2": return ActivePolicy(line,release_id,"release_v2","BASE_V2",{})
    release=_read_json(root/"releases"/release_id/"manifest.json")
    if str(release.get("release_id",""))!=release_id: raise ActivePolicyError(f"release id mismatch: {release_id}")
    if str(release.get("line",""))!=line: raise ActivePolicyError(f"release line mismatch: {release_id}")
    candidate_id=str(release.get("candidate_id",""))
    if not candidate_id: raise ActivePolicyError(f"release has no candidate id: {release_id}")
    candidate=_read_json(root/"candidates"/candidate_id/"manifest.json")
    if str(candidate.get("candidate_id",""))!=candidate_id: raise ActivePolicyError(f"candidate id mismatch: {candidate_id}")
    if str(candidate.get("line",""))!=line: raise ActivePolicyError(f"candidate line mismatch: {candidate_id}")
    module=str(candidate.get("module",""))
    if module not in SUPPORTED_RUNTIME_MODULES: raise ActivePolicyError(f"unsupported runtime module: {module or '<missing>'}")
    params=candidate.get("params",{})
    if not isinstance(params,dict): raise ActivePolicyError(f"candidate params must be an object: {candidate_id}")
    return ActivePolicy(line,release_id,candidate_id,module,dict(params))

def runtime_executable_candidate(root: Path,line: str,candidate_id: str)->bool:
    root=Path(root); path=root/"candidates"/candidate_id/"manifest.json"
    if not path.exists(): return False
    try: candidate=_read_json(path)
    except ActivePolicyError: return False
    return str(candidate.get("candidate_id",""))==candidate_id and str(candidate.get("line",""))==line and str(candidate.get("module","")) in SUPPORTED_RUNTIME_MODULES-{"BASE_V2"} and isinstance(candidate.get("params",{}),dict)
