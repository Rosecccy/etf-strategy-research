from __future__ import annotations
from datetime import date
from .records import GateResult, OptimizerState


def evaluate_gate(line, candidate_eval, baseline_eval, new_evidence, state, config, as_of):
    gates = config['gates']
    c = candidate_eval['overall']
    b = baseline_eval['overall']
    reasons = []
    evidence = {}
    trigger_min = float(gates.get('trigger_retention_min', 0.9))
    wd = c.win_rate - b.win_rate
    md = c.mean_return - b.mean_return
    dd = max(0.0, b.max_drawdown - c.max_drawdown)
    evidence.update({'trigger_retention': c.trigger_retention, 'win_rate_delta': wd, 'mean_return_delta': md, 'drawdown_deterioration': dd})
    if not candidate_eval.get('causal', True):
        reasons.append('CAUSALITY')
    if c.trigger_retention < trigger_min:
        reasons.append('TRIGGER_RETENTION')
    if wd < float(gates.get('win_rate_delta_min', 0.005)):
        reasons.append('WIN_RATE_DELTA')
    if md < float(gates.get('mean_return_delta_min', 0)):
        reasons.append('MEAN_RETURN_DELTA')
    if dd > float(gates.get('max_drawdown_deterioration_max', 0.01)):
        reasons.append('MAX_DRAWDOWN')
    windows = [w for w in candidate_eval.get('windows', []) if w.get('eligible', True)]
    minimum_windows = int(gates.get('minimum_evaluation_windows', 0) or 0)
    evidence['eligible_evaluation_windows'] = len(windows)
    if len(windows) < minimum_windows:
        reasons.append('MIN_EVAL_WINDOWS')
    if windows:
        imp = sum((bool(w.get('improved')) for w in windows)) / len(windows)
        if imp < float(gates.get('recent_windows_improve_fraction_min', 2 / 3)):
            reasons.append('WINDOW_STABILITY')
        recent = windows[-1]
        if not recent.get('improved') or float(recent.get('win_rate_delta', 0)) < -float(gates.get('recent_window_win_rate_regression_max', 0)):
            reasons.append('RECENT_WINDOW_REGRESSION')
        evidence['improved_window_fraction'] = imp
    neighbor = float(candidate_eval.get('neighbor_pass_rate', 0))
    if neighbor < float(gates.get('neighbor_pass_rate_min', 0.5)):
        reasons.append('NEIGHBOR_STABILITY')
    lc = config['lines'][line]
    if len(new_evidence) < int(lc.get('minimum_new_samples', 0)):
        reasons.append('MIN_NEW_SAMPLES')
    regimes = {str(r.get('regime', '')) for r in new_evidence if str(r.get('regime', ''))}
    if len(regimes) < int(lc.get('minimum_regimes', 1)):
        reasons.append('MIN_REGIMES')
    last = state.last_promotion_at.get(line)
    if last:
        elapsed = (date.fromisoformat(str(as_of)[:10]) - date.fromisoformat(str(last)[:10])).days
        if elapsed < int(config.get('cooldown_days', 0)):
            reasons.append('COOLDOWN')
    evidence.update({'new_samples': len(new_evidence), 'regimes': sorted(regimes), 'neighbor_pass_rate': neighbor})
    reasons = list(dict.fromkeys(reasons))
    return GateResult(not reasons, reasons, evidence)
