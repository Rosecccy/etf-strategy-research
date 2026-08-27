from same_day_1445.live_optimizer.engine.shadow_policy import apply_c_delay1,apply_d_strict
from same_day_1445.live_optimizer.engine.signal_adapter import NormalizedSignal
def sig(line,action,date='2026-08-24',symbol='510880'):return NormalizedSignal(line=line,decision_date=date,action=action,symbol=symbol,name=symbol,tradable=True,status='OK',raw_action=action,reason='',score=None,metadata={})
def test_c_delay1_queues_then_executes_next_observed_day():
    out1,pending=apply_c_delay1(sig('C','BUY'),None,'2026-08-24');assert out1.action=='HOLD' and pending['action']=='BUY';out2,pending2=apply_c_delay1(sig('C','HOLD','2026-08-25'),pending,'2026-08-25');assert out2.action=='BUY' and out2.symbol=='510880';assert pending2 is None
def test_d_strict_filters_weak_market_overheat_buy():
    result=apply_d_strict(sig('D','BUY',symbol='159813'),{'price':1.12,'ma20':1.00,'market_ret120':-0.02},{},{'overheat':0.08,'market_ret120_max':0.0,'profit_arm':0.20,'giveback':0.02,'min_hold_days':10});assert result.action=='HOLD';assert result.metadata['shadow_reason']=='weak_market_overheat_filter'
