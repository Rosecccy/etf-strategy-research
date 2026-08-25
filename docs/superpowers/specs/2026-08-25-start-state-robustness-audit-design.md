# Start-State Robustness Audit Design

## 1. Goal

Audit whether the frozen C/S/D/R strategy family is robust to the date on which a user starts running the strategy and to the account state inherited at that date.

The motivating failure mode is path dependence: a strategy that started earlier may already hold a position and therefore reject a later BUY signal, while the same strategy started later from cash may accept that BUY signal. The accepted trade sequence can then diverge for months or years even though the market data, model, and strategy parameters are otherwise identical.

This audit is diagnostic only. It must not modify `same_day_1445/release_v2`, the live optimizer state, the forward ledgers, the Windows scheduler, or any active formal strategy.

## 2. Core distinction

The audit separates two concepts that must never be mixed:

- **information start**: the earliest historical market/model information allowed to calculate indicators, rolling selectors, and causal parameters;
- **account start**: the date on which performance measurement and account-state initialization begin.

A later account start must still be allowed to use all causal historical information that would genuinely have been available on that date. The test changes account state, not historical feature availability.

## 3. Primary paths

For each eligible start date `t`, run two paths from the same causal strategy state.

### 3.1 Warm / inherited path

Run the strategy from its canonical historical origin through `t`. At `t`, inherit the position that the canonical path actually holds, if any. Performance is then re-based to 1.0 at `t` and measured forward.

### 3.2 Cold / flat path

Use the same causal historical indicators, model parameters, rolling policy choices, and market data as the warm path, but force the account to cash at `t`. From that date onward, process signals normally under the strategy's position constraints.

The only intentional difference between Warm and Cold is the account state at `t`.

## 4. Strategy-line treatment

### 4.1 C / S / R

C, S, and R use the single-position compound-account convention. These lines receive the full Warm-vs-Cold account-state audit.

The simulator must preserve each line's frozen V2 signal/evaluation semantics, including single-position exclusivity, same-day sell-then-switch behavior where already allowed, and the selected V2 execution overlay.

### 4.2 D

D retains the existing research-sum convention and cannot be treated as an identical single-position account without changing its accounting definition.

For D, the audit therefore measures start-origin sensitivity of the executable/eligible trade set, filter decisions, annual policy selection, and resulting research-sum return stream. D results are reported separately and must not be compared directly with C/S/R compound-equity values.

## 5. Start-date grid

The primary audit uses every eligible trading day as an account start, subject to sufficient future horizon availability.

For computational diagnostics, results must also be aggregatable by:

- calendar year;
- calendar month;
- whether the warm path is already in a position at the start;
- market regime if a causal regime label is already available;
- distance to the nearest canonical BUY signal.

No start dates may be removed because their results are poor.

## 6. Buy-signal perturbation audit

For every canonical BUY signal, define nearby account-start offsets:

- `-20`, `-10`, `-5`, `-1`, `0`, `+1`, `+5`, `+10`, `+20` trading days.

Deduplicate repeated dates and rerun Warm and Cold paths.

This audit specifically tests the failure mode where missing one earlier entry exposes a later BUY that would have been blocked under the inherited-position path.

## 7. Fixed evaluation horizons

For each eligible account start, report fixed forward horizons where sufficient data exist:

- 1 year / 252 trading days;
- 2 years / 504 trading days;
- 3 years / 756 trading days;
- 5 years / 1260 trading days;
- full remaining history as a secondary descriptive view.

Fixed horizons are primary because comparing only terminal 2026 values gives early starts an unfairly longer compounding period.

## 8. Per-start metrics

For every `(line, start_date, path, horizon)` record at minimum:

- start date;
- warm start-position state and symbol;
- first accepted trade after start;
- first Warm/Cold divergence date;
- number of BUY signals blocked by an inherited position;
- number of additional BUY signals accepted by the Cold path;
- accepted-trade count;
- trade-sequence similarity;
- cumulative net return;
- annualized return where horizon permits;
- win rate;
- mean net trade return;
- maximum drawdown;
- time to first path re-synchronization;
- whether paths ever re-synchronize before horizon end.

A re-synchronization occurs only when Warm and Cold have the same account position state after processing the same day's signal. A temporary coincidence in equity is not a re-synchronization.

## 9. Aggregate robustness metrics

For each line and horizon, report distributions rather than a single best backtest:

- median return;
- 10th percentile return;
- 25th percentile return;
- worst 5% average return;
- profitable-start ratio;
- median and 90th percentile maximum drawdown;
- median absolute Warm-vs-Cold return gap;
- 90th percentile absolute Warm-vs-Cold return gap;
- median re-synchronization time;
- 90th percentile re-synchronization time;
- percentage of starts that never re-synchronize inside the horizon;
- start-date return dispersion.

The output must expose the full start-date series so isolated favorable start dates cannot dominate the interpretation.

## 10. Pre-registered interpretation

The first audit is descriptive and does not change a strategy automatically. To avoid choosing criteria after seeing the results, use the following pre-registered robustness labels for C/S/R on the 3-year horizon when enough observations exist.

### Robust

All of the following:

- profitable-start ratio >= 75%;
- 10th-percentile cumulative return > 0;
- median absolute Warm-vs-Cold cumulative-return gap <= 5 percentage points;
- 90th-percentile absolute Warm-vs-Cold gap <= 15 percentage points;
- median re-synchronization <= 63 trading days;
- 90th-percentile re-synchronization <= 252 trading days.

### Moderately path-dependent

The strategy fails at least one Robust criterion but still has:

- profitable-start ratio >= 60%;
- median 3-year return > 0;
- no more than 25% of starts remaining permanently unsynchronized within the 3-year horizon.

### Fragile

Any of the following:

- profitable-start ratio < 60%;
- median 3-year return <= 0;
- 10th-percentile 3-year return is materially negative while the canonical path is strongly positive;
- more than 25% of starts never re-synchronize within 3 years;
- Warm-vs-Cold 90th-percentile return gap > 30 percentage points.

The 1-, 2-, and 5-year distributions are supporting evidence and must be shown even if the 3-year label is favorable.

D receives no C/S/R robustness label until an equivalent, pre-registered research-sum interpretation is defined from its own distribution.

## 11. Divergence attribution

For the worst and most persistent start-date divergences, emit an attribution trace containing:

1. account state immediately before the first divergent signal;
2. the signal accepted by one path and rejected by the other;
3. the position constraint causing the difference;
4. the resulting entry/exit sequence on both paths;
5. the first date, if any, on which the two paths re-synchronize;
6. the return and drawdown contribution attributable to that branch.

This trace is required before proposing a fix. The audit must distinguish genuine strategy-state fragility from a simulator/accounting bug.

## 12. Outputs

Create a research-only artifact tree under a new audit directory, separate from `same_day_1445/live_optimizer/ledger` and `state`.

Expected outputs:

- per-start C/S/R results;
- D start-origin sensitivity results;
- buy-signal perturbation results;
- aggregate summary JSON/CSV;
- worst-divergence attribution table;
- plots of start date vs fixed-horizon return, Warm-vs-Cold gap, drawdown, and re-synchronization time;
- a Markdown report explaining which lines are robust, moderately path-dependent, or fragile under the pre-registered rules.

No output from this audit is promotion evidence for the live optimizer.

## 13. Validation requirements

Before trusting the scan:

1. A Warm run started on the canonical first eligible date must reproduce the frozen V2 C/S/R trade sequence within the data available to the audit.
2. A Cold run started on a date when Warm is already flat must initially match Warm until a genuine later state divergence occurs.
3. Re-running the same start date must be deterministic.
4. No future data may affect parameters, signals, or account decisions before its timestamp.
5. The frozen V2 verifier must still pass after the audit code is added.
6. The audit must write only to its research output directory.

If any invariant fails, stop the robustness interpretation and treat it as an implementation/debugging failure.

## 14. Non-goals

This phase does not:

- change V2 parameters;
- change formal C/S/D/R strategy code;
- change live optimizer promotion thresholds;
- rewrite historical forward ledgers;
- invent a start-state synchronization fix before the actual divergence mechanisms are measured;
- merge the experimental branch into `main`.

## 15. Follow-up decision

Only after the audit is complete should a second design decide whether a mitigation is needed. Candidate mitigations may include inherited-state reconstruction, a startup synchronization period, first-trade eligibility rules, or a state-independent signal architecture, but none is selected in advance.
