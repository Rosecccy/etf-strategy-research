from same_day_1445.live_optimizer.engine.gate import evaluate_gate
from same_day_1445.live_optimizer.engine.records import EvaluationMetrics, OptimizerState
from same_day_1445.live_optimizer.engine.drift import detect_drift


def cfg():
    return {
        'gates': {
            'trigger_retention_min': 0.90,
            'win_rate_delta_min': 0.005,
            'mean_return_delta_min': 0.0,
            'max_drawdown_deterioration_max': 0.01,
            'recent_windows_improve_fraction_min': 2 / 3,
            'recent_window_win_rate_regression_max': 0.0,
            'neighbor_pass_rate_min': 0.5,
        },
        'lines': {
            'C': {'minimum_new_samples': 15, 'minimum_regimes': 2},
            'S': {'minimum_new_samples': 15, 'minimum_regimes': 2},
            'D': {'minimum_new_samples': 100, 'minimum_regimes': 1},
            'R': {'minimum_new_samples': 15, 'minimum_regimes': 2},
        },
        'cooldown_days': 20,
        'drift': {'trigger_rate_abs_max': 0.30, 'win_rate_abs_max': 0.30, 'mean_return_abs_max': 0.10},
    }


def ev(win=0.70, mean=0.05, dd=-0.10, trigger=0.95, neighbor=0.8):
    return {
        'overall': EvaluationMetrics(trades=100, opportunities=105, trigger_retention=trigger, win_rate=win, mean_return=mean, max_drawdown=dd, final_equity=20000),
        'windows': [
            {'name': 'a', 'win_rate_delta': 0.01, 'mean_return_delta': 0.001, 'improved': True},
            {'name': 'b', 'win_rate_delta': 0.01, 'mean_return_delta': 0.001, 'improved': True},
            {'name': 'c', 'win_rate_delta': 0.01, 'mean_return_delta': 0.001, 'improved': True},
        ],
        'neighbor_pass_rate': neighbor,
        'causal': True,
    }


def baseline():
    return {'overall': EvaluationMetrics(trades=100, opportunities=100, trigger_retention=1.0, win_rate=0.69, mean_return=0.049, max_drawdown=-0.10, final_equity=19000)}


def state():
    s = OptimizerState.initial()
    s.last_promotion_at['D'] = '2026-01-01'
    return s


def test_gate_accepts_candidate_that_passes_all_hard_limits():
    evidence = [{'regime': 'bull'} for _ in range(100)]
    result = evaluate_gate('D', ev(), baseline(), evidence, state(), cfg(), as_of='2026-08-23')
    assert result.passed is True
    assert result.reasons == []


def test_gate_rejects_trigger_win_mean_drawdown_and_neighbor_boundaries():
    evidence = [{'regime': 'bull'} for _ in range(100)]
    cases = [
        (ev(trigger=0.899), 'TRIGGER_RETENTION'),
        (ev(win=0.694), 'WIN_RATE_DELTA'),
        (ev(mean=0.048), 'MEAN_RETURN_DELTA'),
        (ev(dd=-0.111), 'MAX_DRAWDOWN'),
        (ev(neighbor=0.49), 'NEIGHBOR_STABILITY'),
    ]
    for candidate, code in cases:
        result = evaluate_gate('D', candidate, baseline(), evidence, state(), cfg(), as_of='2026-08-23')
        assert code in result.reasons


def test_gate_requires_two_of_three_windows_and_recent_window():
    candidate = ev()
    candidate['windows'][1]['improved'] = False
    candidate['windows'][2]['improved'] = False
    result = evaluate_gate('D', candidate, baseline(), [{'regime': 'bull'} for _ in range(100)], state(), cfg(), as_of='2026-08-23')
    assert 'WINDOW_STABILITY' in result.reasons
    assert 'RECENT_WINDOW_REGRESSION' in result.reasons


def test_gate_enforces_minimum_new_samples_regimes_and_cooldown():
    c_state = OptimizerState.initial()
    c_state.last_promotion_at['C'] = '2026-08-15'
    evidence = [{'regime': 'bull'} for _ in range(14)]
    result = evaluate_gate('C', ev(), baseline(), evidence, c_state, cfg(), as_of='2026-08-23')
    assert {'MIN_NEW_SAMPLES', 'MIN_REGIMES', 'COOLDOWN'} <= set(result.reasons)


def test_detect_drift_can_force_shadow_only():
    reference = [{'triggered': True, 'ret': 0.10, 'won': True} for _ in range(20)]
    recent = [{'triggered': False, 'ret': -0.10, 'won': False} for _ in range(20)]
    result = detect_drift(recent, reference, cfg())
    assert result['severe'] is True
    assert 'trigger_rate' in result['breaches']
