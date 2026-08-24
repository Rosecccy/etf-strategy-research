from same_day_1445.live_optimizer.engine.signal_adapter import parse_c_row,parse_s_row,parse_d_payload

def test_parse_realistic_c_buy_and_s_wait():
    c=parse_c_row({'data_date':'2026-08-20','symbol':'510880','display_name':'红利ETF','next_action':'same_day_preclose_buy','reason':'x'},'2026-08-20');assert (c.action,c.tradable,c.symbol)==('BUY',True,'510880');s=parse_s_row({'asof_date':'2026-08-20','action':'wait_idle','symbol':'159992','display_name':'创新药ETF','reason':'x'},'2026-08-20');assert (s.action,s.tradable)==('HOLD',True)
def test_adapter_rejects_stale_date():
    c=parse_c_row({'data_date':'2026-08-19','symbol':'510880','next_action':'same_day_preclose_buy'},'2026-08-20');assert c.tradable is False;assert c.status=='DATA_HOLD'
def test_parse_d_uses_account_state():
    payload={'data_date':'2026-08-20','model_action':'继续持有','model_symbol':'159930','cash_action':'下一交易日收盘买入','cash_symbol':'159813'};cash=parse_d_payload(payload,'2026-08-20',position_status='cash');hold=parse_d_payload(payload,'2026-08-20',position_status='holding');assert (cash.action,cash.symbol)==('BUY','159813');assert (hold.action,hold.symbol)==('HOLD','159930')
def test_parse_real_nested_d_payload_cash_and_holding():
    payload={'data_date':'2026-08-20','market_fear':55.8768,'model_account':{'action':'继续持有','symbol':'159930','name':'能源ETF','reason':'尚未触发卖点','position_state':{'held_trading_days':45,'peak_return':0.0865,'current_return':0.0809}},'if_real_account_is_cash':{'action':'下一交易日收盘买入','symbol':'159813','name':'半导体ETF鹏华','reason':'恐慌分'}};cash=parse_d_payload(payload,'2026-08-20','cash');holding=parse_d_payload(payload,'2026-08-20','holding');assert cash.action=='BUY' and cash.symbol=='159813';assert holding.action=='HOLD' and holding.symbol=='159930';assert holding.metadata['held_trading_days']==45
def test_c_wrapper_internal_step_failure_blocks_signal(tmp_path,monkeypatch):
    import json
    from same_day_1445.live_optimizer.engine import signal_adapter as sa
    runtime=tmp_path;(runtime/'C/live').mkdir(parents=True);(runtime/'C/live/today_decision.csv').write_text('data_date,next_action,symbol,display_name\n2026-08-20,same_day_preclose_buy,510880,红利ETF\n',encoding='utf-8-sig');(runtime/'C/live/daily_run.json').write_text(json.dumps({'steps':[{'cmd':'update','skipped':True,'returncode':0},{'cmd':'extrema','returncode':1,'stderr':'missing pyarrow'},{'cmd':'signal','returncode':0},{'cmd':'account','returncode':0}]}),encoding='utf-8');monkeypatch.setattr(sa,'run_command',lambda *args,**kwargs:{'returncode':0,'stdout':'{}','stderr':'','cmd':args[1]});signals,runs=sa.run_csd_adapters(runtime,'2026-08-20','model',enabled_lines={'C'});assert signals['C'].tradable is False;assert signals['C'].status=='DATA_HOLD';assert signals['C'].reason=='c_internal_step_failed'
