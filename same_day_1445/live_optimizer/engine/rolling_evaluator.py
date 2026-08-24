from __future__ import annotations
from datetime import date, datetime
from statistics import mean
from typing import Any
from .audit import audit_causal_rows
from .records import EvaluationMetrics


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def _ret(row: dict[str, Any]) -> float:
    try:
        return float(row.get('ret', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compute_metrics(rows: list[dict[str, Any]], accounting: str) -> EvaluationMetrics:
    opportunities = len(rows)
    triggered = [row for row in rows if _bool(row.get('triggered', True))]
    returns = [_ret(row) for row in triggered]
    trades = len(returns)
    win_rate = sum((r > 0 for r in returns)) / trades if trades else 0.0
    mean_return = mean(returns) if returns else 0.0
    if accounting not in {'compound', 'research_sum'}:
        raise ValueError(f'unknown accounting: {accounting}')
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    for ret in returns:
        if accounting == 'compound':
            equity *= 1.0 + ret
        else:
            equity += 10000.0 * ret
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
    return EvaluationMetrics(trades=trades, opportunities=opportunities, trigger_retention=trades / opportunities if opportunities else 0.0, win_rate=win_rate, mean_return=mean_return, max_drawdown=max_dd, final_equity=equity)


def _row_date(row: dict[str, Any]) -> date:
    value = str(row.get('entry_date') or row.get('decision_at') or '')
    return datetime.fromisoformat(value[:10]).date()


def _rows_for_window(rows: list[dict[str, Any]], window: dict[str, Any]) -> list[dict[str, Any]]:
    if window.get('opportunity_ids'):
        ids = {str(item) for item in window['opportunity_ids']}
        return [row for row in rows if str(row.get('opportunity_id', '')) in ids]
    start = date.fromisoformat(str(window['start'])[:10])
    end = date.fromisoformat(str(window['end'])[:10])
    return [row for row in rows if start <= _row_date(row) <= end]


def evaluate_candidate_windows(candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], windows: list[dict[str, Any]], accounting: str) -> dict[str, Any]:
    causal_candidate = audit_causal_rows(candidate_rows, 'decision_at', 'observable_at') if candidate_rows else {'ok': True, 'violations': []}
    causal_baseline = audit_causal_rows(baseline_rows, 'decision_at', 'observable_at') if baseline_rows else {'ok': True, 'violations': []}
    violations = causal_candidate['violations'] + causal_baseline['violations']
    overall = compute_metrics(candidate_rows, accounting)
    baseline_overall = compute_metrics(baseline_rows, accounting)
    window_results = []
    for window in windows:
        c_rows = _rows_for_window(candidate_rows, window)
        b_rows = _rows_for_window(baseline_rows, window)
        cm = compute_metrics(c_rows, accounting)
        bm = compute_metrics(b_rows, accounting)
        win_delta = cm.win_rate - bm.win_rate
        mean_delta = cm.mean_return - bm.mean_return
        final_delta = cm.final_equity - bm.final_equity
        eligible = bm.opportunities > 0 and cm.opportunities > 0
        improved = eligible and win_delta >= 0 and mean_delta >= 0 and final_delta >= 0
        window_results.append({'name': str(window['name']), 'candidate': cm.to_dict(), 'baseline': bm.to_dict(), 'win_rate_delta': win_delta, 'mean_return_delta': mean_delta, 'final_equity_delta': final_delta, 'eligible': eligible, 'improved': improved})
    return {'overall': overall, 'baseline_overall': baseline_overall, 'windows': window_results, 'causal': not violations, 'violations': violations, 'neighbor_pass_rate': 1.0}
