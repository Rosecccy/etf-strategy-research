from pathlib import Path
from same_day_1445.live_optimizer.engine.forward_ledger import ForwardPositionBook
from same_day_1445.live_optimizer.engine.signal_adapter import NormalizedSignal
def sig(action,date,symbol='510880'):return NormalizedSignal(line='C',decision_date=date,action=action,symbol=symbol,name='x',tradable=True,status='OK',raw_action=action,reason='',score=None,metadata={})
def test_buy_hold_sell_matures_exactly_one_trade(tmp_path: Path):
    book=ForwardPositionBook(tmp_path/'positions.json');assert book.apply('C','formal',sig('BUY','2026-08-24'),1.00,'2026-08-24T14:45:00') is None;assert book.apply('C','formal',sig('HOLD','2026-08-25'),1.02,'2026-08-25T14:45:00') is None;trade=book.apply('C','formal',sig('SELL','2026-08-26'),1.10,'2026-08-26T14:45:00');assert trade is not None;assert trade['ret']==0.10;assert book.apply('C','formal',sig('HOLD','2026-08-27'),1.08,'2026-08-27T14:45:00') is None
