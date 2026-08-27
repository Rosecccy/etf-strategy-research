import math

from same_day_1445.live_optimizer.engine.candidate_generator import generate_candidates
from same_day_1445.live_optimizer.engine.rolling_evaluator import compute_metrics, evaluate_candidate_windows


def test_generate_candidates_is_bounded_and_deterministic():
    cfg = {'candidate_spaces': {'D': {'weak_ret': [-0.05, 0.0], 'profit_arm': [0.15, 0.20]}}}
    a = generate_candidates(cfg, 'D')
    b = generate_candidates(cfg, 'D')
    assert a == b
    assert len(a) == 4
    assert len({x['candidate_id'] for x in a}) == 4
    assert all(set(x['params']) == {'weak_ret', 'profit_arm'} for x in a)


def test_compute_metrics_compound_and_research_sum():
    rows = [
        {'ret': 0.10, 'triggered': True},
        {'ret': -0.05, 'triggered': True},
        {'ret': 0.00, 'triggered': False},
    ]
    compound = compute_metrics(rows, 'compound')
    research = compute_metrics(rows, 'research_sum')
    assert compound.trades == 2
    assert compound.trigger_retention == 2 / 3
    assert compound.win_rate == 0.5
    assert math.isclose(compound.mean_return, 0.025)
    assert math.isclose(compound.final_equity, 10450.0)
    assert math.isclose(research.final_equity, 10500.0)
    assert compound.max_drawdown <= 0


def test_window_evaluation_rejects_future_observable_rows():
    baseline = [
        {'ret': 0.01, 'triggered': True, 'decision_at': '2024-01-10T14:45:00+08:00', 'observable_at': '2024-01-10T14:44:00+08:00'},
    ]
    candidate = baseline + [
        {'ret': 0.02, 'triggered': True, 'decision_at': '2024-01-11T14:45:00+08:00', 'observable_at': '2024-01-12T09:30:00+08:00'},
    ]
    result = evaluate_candidate_windows(candidate, baseline, [{'name': 'w1', 'start': '2024-01-01', 'end': '2024-01-31'}], 'compound')
    assert result['causal'] is False
    assert result['violations'][0]['decision_at'].startswith('2024-01-11')
