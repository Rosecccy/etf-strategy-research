# Live Self-Optimizer Design

## 1. Goal

Build a programmatic, GPT-independent optimization loop for the 14:45 ETF strategy system. Every trading day adds new market and execution evidence, updates shadow candidates, re-evaluates them with causal rolling tests, detects drift, and promotes only candidates that satisfy hard statistical and execution guardrails.

The frozen canonical baseline remains `same_day_1445/release_v2`. The optimizer must never overwrite that directory. Any promoted strategy is emitted as a new immutable release.

## 2. Operating principles

1. No GPT or manual interpretation is required at runtime.
2. A new trading day may update evidence and shadow candidates, but it may not directly rewrite the active formal strategy merely because one day looked better.
3. All training, parameter selection, thresholds, and routing for date T may use only data available before the decision being evaluated.
4. Every candidate comparison is made against the currently active formal release and against the frozen V2 anchor.
5. C/S/R retain single-position compound accounting. D retains its research-sum accounting convention.
6. The optimizer stores append-only evidence and reproducible decision records so every promotion can be replayed.

## 3. Directory layout

Create `same_day_1445/live_optimizer/` with the following units:

- `config/optimizer.json`: immutable-by-run policy for windows, guardrails, minimum sample sizes, cooldowns, search spaces, and active data providers.
- `engine/snapshot.py`: freezes the 14:45 observable state and records source timestamps.
- `engine/feature_builder.py`: builds only causal features from the frozen snapshot and historical records.
- `engine/signal_adapter.py`: runs the currently active C/S/D/R formal strategies and registered shadow candidates on the same snapshot.
- `engine/labeler.py`: closes labels only when the required future outcome becomes observable; labels are never available to earlier decisions.
- `engine/candidate_generator.py`: creates bounded candidate variants from declared search spaces. It cannot invent arbitrary code.
- `engine/rolling_evaluator.py`: performs expanding/rolling walk-forward evaluation and reports per-window metrics.
- `engine/gate.py`: applies promotion/rejection hard constraints.
- `engine/drift.py`: detects feature, trigger-rate, return, and error-profile drift relative to recent history.
- `engine/release_manager.py`: writes immutable candidate/promotion artifacts and updates active pointers only after gate approval.
- `engine/audit.py`: verifies append-only evidence, causal timestamps, hashes, accounting consistency, and replayability.
- `jobs/run_1445.py`: trading-day 14:45 job.
- `jobs/run_close.py`: end-of-day reconciliation job.
- `jobs/run_daily_optimizer.py`: post-close optimization and gate job.
- `jobs/run_replay.py`: deterministic replay for a historical date range.
- `state/`: active pointers and machine state only; no hand-edited research results.
- `ledger/`: append-only signal, market, execution, label, evaluation, and promotion ledgers.
- `candidates/`: generated shadow candidate specifications and frozen evaluation artifacts.
- `releases/`: newly promoted immutable releases; V2 itself remains outside this subtree.
- `site/`: local dashboard showing formal vs shadow decisions, metrics, drift, and promotion state.

## 4. Daily data flow

### 4.1 14:45 job

At approximately 14:45 on a trading day:

1. Fetch and timestamp the current market snapshot.
2. Persist the exact observable inputs used by all strategies.
3. Run formal C/S/D/R and all enabled shadow candidates on the same frozen snapshot.
4. Record BUY/SELL/HOLD, symbol, confidence/score if available, rule identifiers, feature hash, data timestamp, strategy version, and intended execution mode.
5. Do not train on the current day after seeing its later close.

The snapshot record becomes immutable once written.

### 4.2 Close job

After the market closes:

1. Append official close and available execution-reference values.
2. Reconcile actual/estimated execution prices against the 14:45 snapshot.
3. Update open-position state.
4. Close only labels whose evaluation horizon is now complete.
5. Mark data-quality failures rather than silently imputing them into promotion evidence.

### 4.3 Daily optimizer job

After reconciliation:

1. Add newly matured labeled samples to the training/evaluation pool.
2. Recompute drift statistics.
3. Generate only candidates permitted by `optimizer.json`.
4. Evaluate each candidate using causal walk-forward windows.
5. Compare candidate vs current formal release and frozen V2 anchor.
6. Apply hard promotion gates.
7. Update the shadow leader if warranted.
8. Promote a formal release only when all automatic gates, minimum-new-sample requirements, and cooldown rules pass.
9. Write a complete promotion or rejection record.

## 5. Candidate model

The first version is intentionally bounded. Candidate search may change only declared strategy parameters or declared execution-policy modules. It may not mutate arbitrary source code.

Supported candidate dimensions initially:

- Existing C/S/D/R strategy thresholds already represented as numeric/configurable parameters.
- D strict-rolling protections already identified in research, including weak-market overheat filters and profit-giveback protection, expressed as bounded threshold grids.
- C Delay1 as a shadow-only execution module initially.
- Same-day / delay-1 execution alternatives only when explicitly enabled for a line.
- Window length, minimum-history length, and promotion score weights within predeclared ranges.

New feature families or arbitrary model classes require a code review and a new optimizer version; they are not self-generated by the running program.

## 6. Causal evaluation

For each test window W, all candidate selection must be trained on observations strictly before W. The evaluator stores exact training dates and row hashes.

Default hierarchy:

- Fast daily diagnostic windows for drift and shadow ranking.
- Medium rolling windows for candidate stability.
- Formal promotion windows using multiple non-overlapping recent evaluation segments.

No candidate may be promoted based on a single aggregate backtest number.

## 7. Promotion gate

A candidate must pass every hard gate. Defaults:

- Trigger retention >= 90% against the active formal release, unless a line-specific approved policy is stricter.
- Rolling win rate improvement >= 0.5 percentage points.
- Mean net trade return must not decline.
- Maximum drawdown may not worsen by more than 1 percentage point.
- At least 2/3 recent formal evaluation windows must improve on the composite criteria.
- The most recent window must not materially regress.
- No single trade may account for an excessive share of the measured improvement; concentration limits are recorded in config.
- Neighboring parameter settings must show a stable performance plateau rather than an isolated optimum.
- Candidate must satisfy minimum new, previously unseen evidence since its defining research cutoff.
- Candidate must survive a cooldown period before formal promotion.

Initial minimum new-evidence defaults:

- D: 100 newly closed formal-comparable trades.
- C: 15 newly closed trades and at least two market-regime buckets.
- S: 15 newly closed trades and at least two market-regime buckets.
- R: 15 newly closed trades and at least two market-regime buckets.

These thresholds are configuration values and changes to them are themselves audited.

## 8. Shadow leader and formal promotion

There are three states per line:

- `formal`: currently used release.
- `shadow_leader`: best candidate that has passed ranking criteria but not all promotion gates.
- `challengers`: other eligible candidates.

Daily optimization may replace the shadow leader. Formal promotion is rarer.

A formal promotion produces:

- immutable release directory;
- exact parent release;
- parameter/config snapshot;
- training/evaluation windows;
- candidate metrics and baseline metrics;
- data hashes;
- gate results;
- promotion timestamp;
- rollback pointer.

Formal promotion never deletes the parent release.

## 9. Rollback and kill switches

Automatic rollback is triggered when a promoted release violates configured live guardrails after sufficient evidence, including abnormal trigger collapse, severe drawdown deterioration, broken data integrity, or strategy runtime failure.

Hard safety states:

- `NORMAL`: formal + shadows run normally.
- `SHADOW_ONLY`: formal remains frozen; optimizer can learn but cannot promote.
- `DATA_HOLD`: data-quality failure; no new optimization evidence accepted.
- `ROLLBACK`: active pointer returns to the last valid formal release.

No state silently modifies historical ledgers.

## 10. Drift detection

Track at minimum:

- feature distribution shift;
- trigger-rate shift;
- ETF mix shift;
- win-rate and mean-return shift;
- MAE/MFE and profit-giveback shift;
- execution slippage / 14:45-to-close proxy drift;
- missing-data and stale-data rates.

Drift does not automatically mean 'optimize harder'. High drift first tightens promotion permissions and can force `SHADOW_ONLY`.

## 11. Dashboard

The local HTML dashboard should show, per C/S/D/R:

- formal release and shadow leader;
- latest 14:45 decision;
- formal vs shadow execution point on the real price curve;
- cumulative forward-only comparison;
- trigger retention;
- rolling win rate, mean return, drawdown;
- number of newly matured samples;
- current promotion-gate checklist;
- drift status;
- last promotion / rollback event.

The dashboard reads generated JSON/CSV artifacts. It contains no optimization logic.

## 12. Persistence and audit rules

Use CSV/JSON for small append-only ledgers and configuration. Large market snapshots may use Parquet when available, with a CSV fallback for environments without a Parquet engine.

Each job writes a run record containing:

- run id;
- code version / git commit if available;
- config hash;
- input file hashes;
- output hashes;
- start/end timestamps;
- status and error summary.

A replay must reproduce the same decisions from the same inputs and versioned configuration.

## 13. Failure handling

- Missing or stale 14:45 data: emit no new tradable shadow evidence and enter `DATA_HOLD` for the affected line/date.
- Incomplete close data: postpone labeling; do not fabricate a result.
- Candidate evaluation error: candidate is rejected for that run, formal release remains unchanged.
- Audit mismatch: block promotion globally until resolved.
- Optimizer crash: formal strategy remains usable because formal execution does not depend on successful optimizer completion.

## 14. Testing strategy

Implementation must include tests for:

1. Future-data leakage: current/future rows cannot enter earlier feature/training windows.
2. Append-only ledgers: duplicate or rewritten sample ids are rejected.
3. Trigger retention calculations.
4. Promotion gate boundaries.
5. Cooldown and minimum-new-sample enforcement.
6. Parameter-neighborhood stability checks.
7. Formal release immutability.
8. Automatic rollback state transitions.
9. Data-hold behavior under stale/missing market data.
10. Replay determinism.
11. Accounting conventions for C/S/R and D.
12. Baseline regression: existing `release_v2` verification must still pass unchanged.

## 15. Initial rollout

Phase 1 runs the optimizer in `SHADOW_ONLY` mode. It records fresh evidence and ranks candidates but cannot promote.

Phase 2 enables automatic formal promotion only after enough truly new evidence has accumulated and the promotion gate can be exercised on forward-only samples.

Initial shadow set:

- C formal V2 + C Delay1 challenger.
- S formal V2 + bounded parameter challengers only.
- D formal V2 + D Strict Rolling challenger and bounded neighboring variants.
- R formal V2 + bounded parameter challengers only.

The system may discover a different shadow leader over time, but only from the declared candidate space.

## 16. Non-goals

- No language-model-generated strategy code.
- No daily unrestricted hyperparameter search over the full historical dataset.
- No rewriting the frozen V2 release.
- No promotion based only on cumulative return.
- No claim that forward profitability is guaranteed.
