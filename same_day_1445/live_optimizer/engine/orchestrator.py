import json
from datetime import datetime, date
from pathlib import Path
from typing import Any
from .config import load_optimizer_config, stable_json_hash
from .candidate_generator import generate_candidates
from .active_policy import resolve_active_policy, runtime_executable_candidate, ActivePolicyError
from .drift import detect_drift
from .daily_monitor import detect_daily_signal_drift
from .gate import evaluate_gate
from .ledger import AppendOnlyCsvLedger
from .records import GateResult, OptimizerState
from .release_manager import promote_candidate, rollback as rollback_release, ReleaseError
from .rollback_guard import evaluate_rollback_guard
from .rolling_evaluator import compute_metrics, evaluate_candidate_windows


def _state(root, cfg):
    path = root / 'state' / 'optimizer_state.json'
    if path.exists():
        return OptimizerState.from_dict(json.loads(path.read_text(encoding='utf-8-sig')))
    state = OptimizerState.initial()
    state.mode = cfg['mode']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return state


def _save_state(root, state):
    path = root / 'state' / 'optimizer_state.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def _score(eval_result):
    m = eval_result['overall']
    return (m.win_rate, m.mean_return, m.final_equity)


def _matched_pair(candidate_rows, baseline_rows):
    candidate_ids = {str(row.get('opportunity_id', '')) for row in candidate_rows if str(row.get('opportunity_id', ''))}
    baseline_ids = {str(row.get('opportunity_id', '')) for row in baseline_rows if str(row.get('opportunity_id', ''))}
    if candidate_ids and baseline_ids:
        matched_ids = candidate_ids & baseline_ids
        candidate = [row for row in candidate_rows if str(row.get('opportunity_id', '')) in matched_ids]
        baseline = [row for row in baseline_rows if str(row.get('opportunity_id', '')) in matched_ids]
        return (candidate, baseline)
    return (candidate_rows, baseline_rows)


def _dynamic_windows(matched_baseline, line, cfg):
    forward = cfg.get('forward_evaluation', {})
    count = int(forward.get('window_count', 0) or 0)
    per_cfg = forward.get('minimum_opportunities_per_window', {})
    per = int(per_cfg.get(line, 0) or 0) if isinstance(per_cfg, dict) else int(per_cfg or 0)
    if count <= 0 or per <= 0:
        return cfg.get('evaluation_windows', [])
    rows = [row for row in matched_baseline if str(row.get('opportunity_id', ''))]
    rows = sorted(rows, key=lambda row: (str(row.get('entry_date', '')), str(row.get('opportunity_id', ''))))
    needed = count * per
    if len(rows) < needed:
        return []
    rows = rows[-needed:]
    windows = []
    for index in range(count):
        chunk = rows[index * per:(index + 1) * per]
        windows.append({'name': f'forward_{index + 1}', 'opportunity_ids': [str(row['opportunity_id']) for row in chunk]})
    return windows


def _new_evidence_rows(rows, last_promotion_at):
    if not last_promotion_at:
        return rows
    text = str(last_promotion_at)
    if 'T' in text:
        cutoff = datetime.fromisoformat(text.replace('Z', '+00:00'))
        out = []
        for row in rows:
            value = str(row.get('observable_at', ''))
            if not value:
                continue
            observed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if observed > cutoff:
                out.append(row)
        return out
    cutoff_date = date.fromisoformat(text[:10])
    return [row for row in rows if str(row.get('observable_at', ''))[:10] and date.fromisoformat(str(row['observable_at'])[:10]) > cutoff_date]


def _positions(root):
    path = root / 'state' / 'forward_positions.json'
    if not path.exists() or not path.read_text(encoding='utf-8').strip():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _transition_safe(root, line, formal_id, candidate_id):
    positions = _positions(root)
    try:
        formal_evidence_id = resolve_active_policy(root, line, formal_id).evidence_id
    except ActivePolicyError:
        return False
    formal = positions.get(f'{line}:{formal_evidence_id}')
    candidate = positions.get(f'{line}:{candidate_id}')
    if formal is None and candidate is None:
        return True
    if formal is None or candidate is None:
        return False
    keys = ('symbol', 'entry_date', 'opportunity_id')
    if any((str(formal.get(key, '')) != str(candidate.get(key, '')) for key in keys)):
        return False
    try:
        return abs(float(formal.get('entry_price', 0.0)) - float(candidate.get('entry_price', 0.0))) <= 1e-12
    except (TypeError, ValueError):
        return False


def _declared_neighbor_ids(config, line, candidate_id):
    specs = generate_candidates(config, line)
    by_id = {str(item['candidate_id']): item for item in specs}
    target = by_id.get(str(candidate_id))
    if target is None:
        return []
    space = config.get('candidate_spaces', {}).get(line, {})
    params = target['params']
    neighbors: list[str] = []
    for other in specs:
        other_id = str(other['candidate_id'])
        if other_id == str(candidate_id):
            continue
        differing = [key for key in space if other['params'].get(key) != params.get(key)]
        if len(differing) != 1:
            continue
        key = differing[0]
        values = list(space[key])
        try:
            left = values.index(params[key])
            right = values.index(other['params'][key])
        except ValueError:
            continue
        if abs(left - right) == 1:
            neighbors.append(other_id)
    return sorted(neighbors)


def _apply_neighbor_stability(line, candidates, baseline_rows, accounting, state, config, as_of):
    provisional_pass = {str(item['candidate_id']): bool(item['gate'].passed) for item in candidates}
    declared_ids = {str(item['candidate_id']) for item in generate_candidates(config, line)}
    for item in candidates:
        candidate_id = str(item['candidate_id'])
        if candidate_id not in declared_ids:
            continue
        neighbors = _declared_neighbor_ids(config, line, candidate_id)
        rate = sum((provisional_pass.get(neighbor, False) for neighbor in neighbors)) / len(neighbors) if neighbors else 1.0
        item['evaluation']['neighbor_pass_rate'] = rate
        all_candidate_rows = item['all_candidate_rows']
        candidate_rows, comparator = _matched_pair(all_candidate_rows, baseline_rows)
        baseline_eval = {'overall': compute_metrics(comparator, accounting)}
        new_evidence = _new_evidence_rows(candidate_rows, state.last_promotion_at.get(line))
        gate = evaluate_gate(line, item['evaluation'], baseline_eval, new_evidence, state, config, as_of)
        evidence = dict(gate.evidence)
        evidence['matched_baseline_opportunities'] = len(comparator)
        evidence['evaluation_windows'] = len([w for w in item['evaluation'].get('windows', []) if w.get('eligible')])
        evidence['neighbor_count'] = len(neighbors)
        gate = GateResult(gate.passed, gate.reasons, evidence)
        item['gate'] = gate


def _apply_v2_anchor_guard(line, formal_evidence_id, candidates, v2_rows, accounting, state, config, as_of):
    if formal_evidence_id == 'release_v2':
        return
    for item in candidates:
        candidate_rows, comparator = _matched_pair(item['all_candidate_rows'], v2_rows)
        evidence = dict(item['gate'].evidence)
        if not comparator or not candidate_rows:
            reasons = list(dict.fromkeys([*item['gate'].reasons, 'V2_ANCHOR_NO_MATCHED_EVIDENCE']))
            evidence['v2_anchor'] = {'matched_opportunities': 0, 'gate_passed': False}
            item['gate'] = GateResult(False, reasons, evidence)
            continue
        windows = _dynamic_windows(comparator, line, config)
        anchor_eval = evaluate_candidate_windows(candidate_rows, comparator, windows, accounting)
        anchor_eval['neighbor_pass_rate'] = float(item['evaluation'].get('neighbor_pass_rate', 1.0))
        baseline_eval = {'overall': compute_metrics(comparator, accounting)}
        new_evidence = _new_evidence_rows(candidate_rows, state.last_promotion_at.get(line))
        anchor_gate = evaluate_gate(line, anchor_eval, baseline_eval, new_evidence, state, config, as_of)
        prefixed = [f'V2_ANCHOR_{reason}' for reason in anchor_gate.reasons]
        reasons = list(dict.fromkeys([*item['gate'].reasons, *prefixed]))
        evidence['v2_anchor'] = {'matched_opportunities': len(comparator), 'gate_passed': anchor_gate.passed, 'reasons': anchor_gate.reasons, 'candidate': anchor_eval['overall'].to_dict(), 'baseline': baseline_eval['overall'].to_dict()}
        item['gate'] = GateResult(item['gate'].passed and anchor_gate.passed, reasons, evidence)


def _write_decision_artifacts(root, result):
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str) + '\n'
    immutable = root / 'ledger' / 'optimizer_decisions' / f"{result['decision_hash']}.json"
    immutable.parent.mkdir(parents=True, exist_ok=True)
    if immutable.exists():
        if immutable.read_text(encoding='utf-8') != text:
            raise RuntimeError(f'optimizer decision artifact conflict: {immutable}')
    else:
        immutable.write_text(text, encoding='utf-8')
    latest = root / 'state' / 'latest_optimizer_decision.json'
    latest.parent.mkdir(parents=True, exist_ok=True)
    temp = latest.with_suffix('.json.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(latest)


def _configured_runtime_mode(root, cfg):
    path = root / 'state' / 'deployment_mode.json'
    if path.exists() and path.read_text(encoding='utf-8').strip():
        try:
            value = json.loads(path.read_text(encoding='utf-8-sig'))
            mode = str(value.get('mode', ''))
            if mode in {'NORMAL', 'SHADOW_ONLY'}:
                return mode
        except json.JSONDecodeError:
            pass
    return str(cfg['mode'])


def run_optimizer(root, as_of):
    root = Path(root)
    cfg = load_optimizer_config(root / 'config' / 'optimizer.json')
    state = _state(root, cfg)
    runtime_mode = _configured_runtime_mode(root, cfg)
    if state.mode == 'ROLLBACK':
        state.mode = 'SHADOW_ONLY'
    elif state.mode != 'DATA_HOLD':
        state.mode = runtime_mode
    ledger = AppendOnlyCsvLedger(root / 'ledger' / 'closed_trades.csv', ('sample_id', 'candidate_id'))
    rows = ledger.rows()
    lines_out: dict[str, Any] = {}
    for line in ('C', 'S', 'D', 'R'):
        line_rows = [row for row in rows if row.get('line') == line]
        formal = state.active_formal[line]
        try:
            active_policy = resolve_active_policy(root, line, formal)
            formal_evidence_id = active_policy.evidence_id
        except ActivePolicyError:
            formal_evidence_id = formal
            state.mode = 'SHADOW_ONLY'
        baseline_rows = [row for row in line_rows if row.get('candidate_id') == formal_evidence_id]
        excluded = {formal_evidence_id, 'release_v2'}
        candidate_ids = sorted({row.get('candidate_id') for row in line_rows if row.get('candidate_id') and row.get('candidate_id') not in excluded})
        accounting = cfg['lines'][line].get('accounting', 'research_sum' if line == 'D' else 'compound')
        rollback_info: dict[str, Any] = {'status': 'NONE', 'breaches': []}
        previous_release = state.previous_formal.get(line)
        if formal_evidence_id != 'release_v2' and previous_release:
            try:
                parent_policy = resolve_active_policy(root, line, previous_release)
                parent_rows_all = [row for row in line_rows if row.get('candidate_id') == parent_policy.evidence_id]
                active_since = _new_evidence_rows(baseline_rows, state.last_promotion_at.get(line))
                active_new, parent_new = _matched_pair(active_since, parent_rows_all)
                rollback_info = evaluate_rollback_guard(line, active_new, parent_new, accounting, cfg)
                if rollback_info.get('triggered'):
                    state = rollback_release(root, state, line, ','.join(rollback_info.get('breaches', [])) or 'LIVE_GUARD', as_of)
                    state.last_promotion_at[line] = as_of
                    rollback_info = {**rollback_info, 'status': 'ROLLED_BACK', 'to_release': state.active_formal[line]}
                    formal = state.active_formal[line]
                    active_policy = resolve_active_policy(root, line, formal)
                    formal_evidence_id = active_policy.evidence_id
                    baseline_rows = [row for row in line_rows if row.get('candidate_id') == formal_evidence_id]
                    excluded = {formal_evidence_id, 'release_v2'}
                    candidate_ids = sorted({row.get('candidate_id') for row in line_rows if row.get('candidate_id') and row.get('candidate_id') not in excluded})
            except ActivePolicyError as exc:
                rollback_info = {'status': 'BLOCKED', 'breaches': [], 'reason': str(exc)}
                state.mode = 'SHADOW_ONLY'
        ordered_baseline = sorted(baseline_rows, key=lambda row: (str(row.get('observable_at', '')), str(row.get('sample_id', ''))))
        recent_count = int(cfg.get('drift', {}).get('recent_count', 20))
        if recent_count > 0 and len(ordered_baseline) >= recent_count * 2:
            drift = detect_drift(ordered_baseline[-recent_count:], ordered_baseline[:-recent_count], cfg)
        else:
            drift = {'severe': False, 'breaches': [], 'deltas': {}, 'recent': {}, 'reference': {}}
        if drift.get('severe') and state.mode == 'NORMAL':
            state.mode = 'SHADOW_ONLY'
        daily_drift = detect_daily_signal_drift(root, line, cfg)
        if daily_drift.get('severe') and state.mode == 'NORMAL':
            state.mode = 'SHADOW_ONLY'
        candidates = []
        for candidate_id in candidate_ids:
            all_candidate_rows = [row for row in line_rows if row.get('candidate_id') == candidate_id]
            candidate_rows, comparator = _matched_pair(all_candidate_rows, baseline_rows)
            windows = _dynamic_windows(comparator, line, cfg)
            baseline_eval = {'overall': compute_metrics(comparator, accounting)}
            evaluation = evaluate_candidate_windows(candidate_rows, comparator, windows, accounting)
            new_evidence = _new_evidence_rows(candidate_rows, state.last_promotion_at.get(line))
            gate = evaluate_gate(line, evaluation, baseline_eval, new_evidence, state, cfg, as_of)
            evidence = dict(gate.evidence)
            evidence['matched_baseline_opportunities'] = len(comparator)
            evidence['evaluation_windows'] = len([w for w in evaluation.get('windows', []) if w.get('eligible')])
            gate = GateResult(gate.passed, gate.reasons, evidence)
            candidates.append({'candidate_id': candidate_id, 'evaluation': evaluation, 'gate': gate, 'all_candidate_rows': all_candidate_rows})
        _apply_neighbor_stability(line, candidates, baseline_rows, accounting, state, cfg, as_of)
        v2_rows = [row for row in line_rows if row.get('candidate_id') == 'release_v2']
        _apply_v2_anchor_guard(line, formal_evidence_id, candidates, v2_rows, accounting, state, cfg, as_of)
        eligible = [item for item in candidates if item['gate'].passed]
        promotion = {'status': 'NONE', 'candidate_id': None}
        if eligible:
            winner = max(eligible, key=lambda item: _score(item['evaluation']))
            leader = winner['candidate_id']
            state.shadow_leader[line] = leader
            if state.mode == 'NORMAL' and leader != formal_evidence_id:
                if not runtime_executable_candidate(root, line, leader):
                    promotion = {'status': 'BLOCKED_NONEXECUTABLE', 'candidate_id': leader}
                elif _transition_safe(root, line, formal, leader):
                    try:
                        state = promote_candidate(root, state, line, leader, winner['gate'], as_of)
                        state.pending_promotion[line] = None
                        promotion = {'status': 'PROMOTED', 'candidate_id': leader, 'release_id': state.active_formal[line]}
                    except ReleaseError as exc:
                        promotion = {'status': 'BLOCKED', 'candidate_id': leader, 'reason': str(exc)}
                else:
                    state.pending_promotion[line] = leader
                    promotion = {'status': 'PENDING_TRANSITION', 'candidate_id': leader}
        else:
            state.pending_promotion[line] = None
            if state.mode == 'SHADOW_ONLY':
                promotion = {'status': 'SHADOW_ONLY', 'candidate_id': state.shadow_leader[line]}
        lines_out[line] = {'formal': state.active_formal[line], 'formal_evidence_id': formal_evidence_id, 'shadow_leader': state.shadow_leader[line], 'promotion': promotion, 'rollback': rollback_info, 'drift': drift, 'daily_drift': daily_drift, 'candidates': [{'candidate_id': item['candidate_id'], 'gate': item['gate'].to_dict(), 'overall': item['evaluation']['overall'].to_dict()} for item in candidates]}
    _save_state(root, state)
    decision_core = {'as_of': as_of, 'mode': state.mode, 'lines': lines_out}
    decision_hash = stable_json_hash(decision_core)
    result = {**decision_core, 'decision_hash': decision_hash}
    _write_decision_artifacts(root, result)
    runs = AppendOnlyCsvLedger(root / 'ledger' / 'optimizer_runs.csv', ('decision_hash',))
    runs.append({'decision_hash': decision_hash, 'as_of': as_of, 'mode': state.mode, 'state_hash': stable_json_hash(state.to_dict()), 'line_summary_hash': stable_json_hash(lines_out)})
    return result
