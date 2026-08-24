import json,sys
from pathlib import Path
import pytest
from same_day_1445.live_optimizer.ops.deploy import DeploymentError,verify_v2_baseline

def test_verify_v2_baseline_runs_repo_verifier_and_writes_marker(tmp_path:Path):
    repo=tmp_path/'repo';opt=repo/'same_day_1445/live_optimizer';(repo/'scripts').mkdir(parents=True);opt.mkdir(parents=True);(repo/'scripts/verify_current_baseline.py').write_text("print('CURRENT BASELINE OK')\n",encoding='utf-8');marker=verify_v2_baseline(repo,opt,'2026-08-24T16:00:00+08:00',python_exe=sys.executable);saved=json.loads((opt/'state/v2_baseline_verified.json').read_text(encoding='utf-8'));assert marker['ok'] is True and saved['ok'] is True;assert saved['canonical_baseline']=='same_day_1445/release_v2';assert saved['output_hash']
def test_verify_v2_baseline_refuses_failed_verifier(tmp_path:Path):
    repo=tmp_path/'repo';opt=repo/'same_day_1445/live_optimizer';(repo/'scripts').mkdir(parents=True);opt.mkdir(parents=True);(repo/'scripts/verify_current_baseline.py').write_text('raise SystemExit(3)\n',encoding='utf-8')
    with pytest.raises(DeploymentError,match='failed'):verify_v2_baseline(repo,opt,'2026-08-24T16:00:00+08:00',python_exe=sys.executable)
