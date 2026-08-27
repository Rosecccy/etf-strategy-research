from __future__ import annotations
from dataclasses import replace
from typing import Any
from .signal_adapter import NormalizedSignal

def apply_c_delay1(formal,pending,today):
    if pending and str(pending.get('queued_date',''))<today:
        out=replace(formal,action=str(pending['action']),symbol=str(pending.get('symbol','')),name=str(pending.get('name','')),raw_action='C_DELAY1',reason='C_DELAY1 execute queued',metadata={**formal.metadata,'shadow_id':'C_DELAY1','source_signal_date':pending.get('queued_date')});new=None
    else:out=replace(formal,action='HOLD',symbol=formal.symbol,name=formal.name,raw_action=formal.raw_action,reason='C_DELAY1 waiting',metadata={**formal.metadata,'shadow_id':'C_DELAY1'});new=pending
    if formal.action in {'BUY','SELL'} and not pending:new={'queued_date':today,'action':formal.action,'symbol':formal.symbol,'name':formal.name}
    return out,new
def apply_d_strict(formal,features,position,params):
    out=replace(formal,metadata=dict(formal.metadata));out.metadata['shadow_id']='D_STRICT';price=float(features.get('price',0) or 0);ma20=float(features.get('ma20',0) or 0);mret=float(features.get('market_ret120',0) or 0);weak=float(params.get('market_ret120_max',params.get('weak_market_return',0.0)));over=float(params.get('overheat',params.get('overheat_ma20',0.08)));arm=float(params.get('profit_arm',.2));give=float(params.get('giveback',.02));hold=int(params.get('min_hold_days',params.get('minimum_hold_days',10)))
    if formal.action=='BUY' and ma20>0 and price/ma20-1>=over and mret<=weak:return replace(out,action='HOLD',reason='weak_market_overheat_filter',metadata={**out.metadata,'shadow_reason':'weak_market_overheat_filter'})
    if position and formal.action!='SELL':
        entry=float(position.get('entry_price',0) or 0);peak=float(position.get('peak_price',entry) or entry);days=int(position.get('hold_days',0));peak_ret=peak/entry-1 if entry else 0;giveback=price/peak-1 if peak else 0
        if peak_ret>=arm and days>=hold and giveback<=-give:return replace(out,action='SELL',symbol=str(position.get('symbol',formal.symbol)),reason='armed_profit_giveback',metadata={**out.metadata,'shadow_reason':'armed_profit_giveback'})
    return out
