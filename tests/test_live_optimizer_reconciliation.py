import csv,json
from pathlib import Path
from same_day_1445.live_optimizer.engine.reconciliation import reconcile_unreconciled_close
from same_day_1445.live_optimizer.providers.file_provider import FileMinuteProvider

def _raw(path:Path,symbol:str):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('date,symbol,name,open,high,low,close,volume,amount,pct_change,turnover_rate,source_file\n'+f'2026-08-24,{symbol},x,1,1,1,1.00,1,1,,,live_optimizer:1445\n',encoding='utf-8-sig')
def test_unreconciled_close_is_backfilled_before_next_day(tmp_path: Path):
    runtime=tmp_path/'runtime';opt=tmp_path/'opt';feed=tmp_path/'feed';_raw(runtime/'S/raw/etf/159999.csv','159999');(opt/'state').mkdir(parents=True);(opt/'state/unreconciled_close.json').write_text(json.dumps({'trade_date':'2026-08-24','symbols':['159999'],'errors':[]}),encoding='utf-8');(feed/'159999').mkdir(parents=True);(feed/'159999/2026-08-24.csv').write_text('timestamp,open,high,low,close,volume,amount\n2026-08-24 15:00:00,1.00,1.03,0.99,1.02,100,102\n',encoding='utf-8');result=reconcile_unreconciled_close(opt,runtime,FileMinuteProvider(feed),max_workers=2);assert result['status']=='OK';assert not (opt/'state/unreconciled_close.json').exists()
    with (runtime/'S/raw/etf/159999.csv').open('r',encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
    assert float(rows[-1]['close'])==1.02;assert rows[-1]['source_file']=='live_optimizer:close_recovery'
