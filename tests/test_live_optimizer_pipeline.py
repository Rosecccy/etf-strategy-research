import json
from pathlib import Path
from same_day_1445.live_optimizer.jobs.run_pipeline import run_pipeline

def _cfg(root: Path,runtime: Path):
    (root/'runtime').mkdir(parents=True);path=root/'runtime/workspace.json';path.write_text(json.dumps({'runtime_root':str(runtime),'cutoff':'14:45','close_cutoff':'15:00'}),encoding='utf-8');return path

def test_preclose_uses_fixed_cutoff_even_when_started_later(tmp_path: Path):
    root=tmp_path/'opt';root.mkdir();runtime=tmp_path/'runtime';runtime.mkdir();cfg=_cfg(root,runtime);seen={}
    def preclose(**kwargs):seen.update(kwargs);return {'status':'OK'}
    result=run_pipeline('preclose',root,cfg,'2026-08-24','2026-08-24T14:46:00+08:00',handlers={'preclose':preclose});assert result['status']=='OK';assert str(seen['cutoff'])=='14:45:00';status=json.loads((root/'ledger/run_status.json').read_text(encoding='utf-8'));assert status['preclose']=='2026-08-24T14:46:00+08:00'

def test_optimizer_is_blocked_while_prior_close_is_unreconciled(tmp_path: Path):
    root=tmp_path/'opt';root.mkdir();runtime=tmp_path/'runtime';runtime.mkdir();cfg=_cfg(root,runtime);(root/'state').mkdir();(root/'state/unreconciled_close.json').write_text(json.dumps({'trade_date':'2026-08-24','symbols':['510880']}),encoding='utf-8');called=[];result=run_pipeline('optimizer',root,cfg,'2026-08-24','2026-08-24T15:20:00+08:00',handlers={'optimizer':lambda **kwargs: called.append(1)});assert result['status']=='BLOCKED';assert result['reason']=='UNRECONCILED_CLOSE';assert called==[]
