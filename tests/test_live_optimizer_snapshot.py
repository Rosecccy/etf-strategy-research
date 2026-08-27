from datetime import date,datetime,time
from pathlib import Path
from same_day_1445.live_optimizer.providers.base import MinuteBar
from same_day_1445.live_optimizer.providers.file_provider import FileMinuteProvider
from same_day_1445.live_optimizer.engine.snapshot import build_snapshot

def test_snapshot_ignores_bars_after_1445_and_aggregates_exactly(tmp_path:Path):
    p=tmp_path/'510880.csv';p.write_text('timestamp,open,high,low,close,volume,amount\n2026-08-24 14:43:00,1.00,1.02,0.99,1.01,100,101\n2026-08-24 14:44:00,1.01,1.04,1.00,1.03,200,205\n2026-08-24 14:45:00,1.03,1.05,1.02,1.04,300,312\n2026-08-24 14:46:00,1.04,9.99,0.01,9.00,999,9999\n',encoding='utf-8');r=build_snapshot(FileMinuteProvider(tmp_path),['510880'],date(2026,8,24),time(14,45),max_staleness_minutes=1);assert r['status']=='OK';row=r['rows'][0];assert row['open']==1.0;assert row['high']==1.05;assert row['low']==0.99;assert row['close']==1.04;assert row['volume']==600.0;assert row['amount']==618.0;assert row['latest_observed_at']=='2026-08-24T14:45:00'
def test_snapshot_fails_closed_when_latest_bar_is_stale(tmp_path:Path):
    (tmp_path/'510880.csv').write_text('timestamp,open,high,low,close,volume,amount\n2026-08-24 14:40:00,1,1,1,1,1,1\n',encoding='utf-8');r=build_snapshot(FileMinuteProvider(tmp_path),['510880'],date(2026,8,24),time(14,45),max_staleness_minutes=2);assert r['status']=='DATA_HOLD';assert r['errors'][0]['reason']=='stale_minute_data'
def test_snapshot_contains_provider_error_per_symbol_instead_of_crashing():
    class BrokenProvider:
        name='broken'
        def fetch(self,symbol,trade_date):
            if symbol=='159999':raise TimeoutError('network')
            return [MinuteBar(datetime(2026,8,24,14,45),1,1,1,1,1,1)]
    r=build_snapshot(BrokenProvider(),['510880','159999'],date(2026,8,24),time(14,45),max_staleness_minutes=1,max_workers=2);assert len(r['rows'])==1;assert r['rows'][0]['symbol']=='510880';bad=next(item for item in r['errors'] if item['symbol']=='159999');assert bad['reason']=='provider_error';assert bad['error_type']=='TimeoutError'
