from __future__ import annotations

import itertools
from typing import Any

from .config import stable_json_hash


def generate_candidates(config: dict[str, Any], line: str) -> list[dict[str, Any]]:
    space = config.get("candidate_spaces", {}).get(line, {})
    if not space:
        return []
    keys = sorted(space)
    values = [list(space[key]) for key in keys]
    result = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        signature = {"line": line, "params": params}
        cid = f"{line}_{stable_json_hash(signature)[:12]}"
        result.append({"line": line, "candidate_id": cid, "params": params})
    return result
