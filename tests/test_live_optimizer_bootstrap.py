import json
from pathlib import Path
import pytest

from same_day_1445.live_optimizer.ops.bootstrap import BootstrapError, bootstrap_runtime


def _source(root: Path):
    for rel in ['C/src/run_daily.py','S/src/run_s1_live.py','D/src/daily_panic_live.py','R/formal/r_single_yearly_choice.csv']:
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(rel,encoding='utf-8')
    (root/'C/raw/etf').mkdir(parents=True);(root/'C/raw/etf/510880.csv').write_text('x',encoding='utf-8')


def test_bootstrap_copies_runtime_and_writes_workspace_config(tmp_path: Path):
    source=tmp_path/'source';runtime=tmp_path/'runtime';opt=tmp_path/'opt';source.mkdir();opt.mkdir();_source(source)
    result=bootstrap_runtime(source,runtime,opt,provider_kind='file',file_root=str(tmp_path/'feed'))
    assert (runtime/'C/src/run_daily.py').exists()
    cfg=json.loads((opt/'runtime/workspace.json').read_text(encoding='utf-8'))
    assert cfg['runtime_root']==str(runtime.resolve())
    assert cfg['provider']['kind']=='file'
    assert result['status']=='OK'


def test_bootstrap_refuses_source_equal_runtime(tmp_path: Path):
    source=tmp_path/'source';source.mkdir();_source(source)
    with pytest.raises(BootstrapError,match='different'):
        bootstrap_runtime(source,source,tmp_path/'opt')


def test_bootstrap_initializes_optimizer_state_for_health(tmp_path: Path):
    from same_day_1445.live_optimizer.ops.health import build_health_report
    source=tmp_path/'source';runtime=tmp_path/'runtime';opt=tmp_path/'opt';source.mkdir();opt.mkdir();_source(source)
    (opt/'config').mkdir(parents=True)
    (opt/'config/optimizer.json').write_text(json.dumps({'mode':'SHADOW_ONLY','gates':{},'lines':{'C':{},'S':{},'D':{},'R':{}},'candidate_spaces':{}}),encoding='utf-8')
    bootstrap_runtime(source,runtime,opt,provider_kind='file',file_root=str(tmp_path/'feed'))
    state_path=opt/'state/optimizer_state.json';assert state_path.exists();state=json.loads(state_path.read_text(encoding='utf-8'))
    assert state['mode']=='SHADOW_ONLY';assert state['active_formal']=={'C':'release_v2','S':'release_v2','D':'release_v2','R':'release_v2'}
    report=build_health_report(opt,'2026-08-24T10:20:00+08:00');assert 'OPTIMIZER_STATE_INVALID' not in report['codes']
