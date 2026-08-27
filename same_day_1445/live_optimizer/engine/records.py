from __future__ import annotations
from dataclasses import asdict,dataclass
from typing import Any
LINES=("C","S","D","R")
@dataclass(frozen=True)
class EvaluationMetrics:
    trades:int;opportunities:int;trigger_retention:float;win_rate:float;mean_return:float;max_drawdown:float;final_equity:float
    def to_dict(self)->dict[str,Any]:return asdict(self)
@dataclass(frozen=True)
class GateResult:
    passed:bool;reasons:list[str];evidence:dict[str,Any]
    def to_dict(self)->dict[str,Any]:return asdict(self)
@dataclass
class OptimizerState:
    mode:str;active_formal:dict[str,str];previous_formal:dict[str,str|None];shadow_leader:dict[str,str|None];last_promotion_at:dict[str,str|None];pending_promotion:dict[str,str|None]
    @classmethod
    def initial(cls):return cls('SHADOW_ONLY',{l:'release_v2' for l in LINES},{l:None for l in LINES},{l:None for l in LINES},{l:None for l in LINES},{l:None for l in LINES})
    @classmethod
    def from_dict(cls,value):
        b=cls.initial();return cls(str(value.get('mode',b.mode)),{**b.active_formal,**value.get('active_formal',{})},{**b.previous_formal,**value.get('previous_formal',{})},{**b.shadow_leader,**value.get('shadow_leader',{})},{**b.last_promotion_at,**value.get('last_promotion_at',{})},{**b.pending_promotion,**value.get('pending_promotion',{})})
    def to_dict(self):return asdict(self)
