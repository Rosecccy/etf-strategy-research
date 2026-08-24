from same_day_1445.live_optimizer.engine.signal_adapter import NormalizedSignal
from same_day_1445.live_optimizer.engine.r_router import route_r_single

def sig(line,action,symbol='',score=None,tradable=True):return NormalizedSignal(line=line,decision_date='2026-08-24',action=action,symbol=symbol,name=symbol,tradable=tradable,status='OK' if tradable else 'DATA_HOLD',raw_action=action,reason='',score=score,metadata={})
def test_r_single_routes_only_unambiguous_entry():
    result=route_r_single({'policy_id':'base_CS'},sig('C','BUY','510880'),sig('S','HOLD'),position=None);assert (result.action,result.symbol,result.tradable)==('BUY','510880',True)
def test_r_single_dual_entry_without_exact_r_scores_fails_closed():
    result=route_r_single({'policy_id':'base_CS'},sig('C','BUY','510880'),sig('S','BUY','159992'),position=None);assert result.status=='DATA_HOLD';assert result.tradable is False
def test_r_single_dual_entry_with_exact_scores_selects_higher():
    c=sig('C','BUY','510880');c.metadata['r_alloc_score']=300.0;s=sig('S','BUY','159992');s.metadata['r_alloc_score']=320.0;result=route_r_single({'policy_id':'base_CS'},c,s,position=None);assert result.symbol=='159992'
def test_r_holding_follows_source_sell_only():
    pos={'symbol':'510880','source_line':'C'};result=route_r_single({'policy_id':'base_CS'},sig('C','SELL','510880'),sig('S','BUY','159992'),position=pos);assert result.action=='SELL'
