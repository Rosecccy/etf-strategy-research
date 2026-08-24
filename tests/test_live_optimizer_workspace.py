from pathlib import Path
import pytest
from same_day_1445.live_optimizer.engine.workspace import WorkspaceMutationError,upsert_daily_bar

def _csv(path:Path):path.write_text('date,symbol,name,open,high,low,close,volume,amount,pct_change,turnover_rate,source_file\n2026-08-21,510880,红利ETF,1.0,1.1,0.9,1.05,100,105,,,old\n',encoding='utf-8-sig')
def test_workspace_appends_current_day_and_replaces_only_same_date(tmp_path:Path):
    path=tmp_path/'510880.csv';_csv(path);bar={'date':'2026-08-24','symbol':'510880','open':1.06,'high':1.12,'low':1.02,'close':1.10,'volume':120.0,'amount':130.0};first=upsert_daily_bar(path,bar,source_file='live_optimizer:1445');before=path.read_text(encoding='utf-8-sig').splitlines()[1];assert before.startswith('2026-08-21,510880');bar2={**bar,'close':1.11};second=upsert_daily_bar(path,bar2,source_file='live_optimizer:close');lines=path.read_text(encoding='utf-8-sig').splitlines();assert len(lines)==3;assert lines[1]==before;assert ',1.11,' in lines[2];assert first['prior_rows_hash']==second['prior_rows_hash']
def test_workspace_refuses_rewriting_prior_date(tmp_path:Path):
    path=tmp_path/'510880.csv';_csv(path)
    with pytest.raises(WorkspaceMutationError):upsert_daily_bar(path,{'date':'2026-08-20','symbol':'510880','open':1,'high':1,'low':1,'close':1,'volume':1,'amount':1})
