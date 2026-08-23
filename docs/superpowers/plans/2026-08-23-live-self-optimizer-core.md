# Live Self-Optimizer Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GPT-independent core that ingests newly matured forward samples, maintains append-only evidence, evaluates bounded shadow candidates causally, detects drift, and promotes or rejects candidates through hard automatic gates without modifying the frozen V2 baseline.

**Architecture:** Phase 1 is a deterministic Python package under `same_day_1445/live_optimizer/`. It is intentionally provider-agnostic: daily market/signal acquisition is connected in Phase 2, while this phase accepts normalized sample/evaluation records and proves that the optimizer, gate, release manager, audit, and replay loop work end-to-end. Formal V2 remains immutable; Phase 1 starts in `SHADOW_ONLY` and can create immutable candidate release artifacts but cannot overwrite `same_day_1445/release_v2`.

**Tech Stack:** Python 3.11+, standard library (`csv`, `json`, `hashlib`, `dataclasses`, `statistics`, `pathlib`, `datetime`), pytest. No GPT/runtime LLM dependency.

**Spec:** `docs/superpowers/specs/2026-08-23-live-self-optimizer-design.md`

## Global Constraints

- No GPT or manual interpretation is required at runtime.
- `same_day_1445/release_v2` is immutable and must never be overwritten.
- Training/evaluation for date T may use only evidence whose `observable_at < T`.
- Ledgers are append-only; duplicate sample ids or changed historical rows are rejected.
- Candidate search is bounded to declared JSON parameter/execution-module spaces; no generated strategy source code.
- Initial mode is `SHADOW_ONLY`; automatic formal promotion is disabled until forward-evidence minimums are met and mode is explicitly `NORMAL`.
- Default hard gates: trigger retention >= 90%, rolling win-rate delta >= +0.005, mean net trade return delta >= 0, max-drawdown deterioration <= 0.01, at least 2/3 recent windows improve, recent window not materially worse, parameter-neighborhood stability required, minimum new samples required, cooldown required.
- Initial minimum new evidence: D=100 closed trades; C/S/R=15 closed trades plus >=2 regime buckets.
- C/S/R accounting remains single-position compound; D remains research-sum.
- Existing `tests/test_v2_audit_consistency.py` and `scripts/verify_current_baseline.py` must remain green.

---

### Task 1: Core configuration, records, and machine state

**Files:**
- Create: `same_day_1445/live_optimizer/__init__.py`
- Create: `same_day_1445/live_optimizer/config/optimizer.json`
- Create: `same_day_1445/live_optimizer/engine/__init__.py`
- Create: `same_day_1445/live_optimizer/engine/config.py`
- Create: `same_day_1445/live_optimizer/engine/records.py`
- Create: `same_day_1445/live_optimizer/state/optimizer_state.json`
- Test: `tests/test_live_optimizer_config.py`

**Interfaces:**
- Produces: `load_optimizer_config(path: Path) -> dict`, `stable_json_hash(value: object) -> str`, `EvaluationMetrics`, `GateResult`, `OptimizerState` dataclasses.

- [ ] **Step 1: Write failing tests** for loading defaults, stable config hashes, required line configuration, and invalid mode rejection.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_live_optimizer_config.py` and confirm failure because package/modules do not exist.
- [ ] **Step 3: Implement** strict JSON loading and dataclasses. `optimizer.json` must define `mode="SHADOW_ONLY"`, line-specific minimum samples, rolling windows, gate thresholds, cooldown days, initial candidates (`C_DELAY1`, `D_STRICT_ROLLING`), and bounded parameter spaces.
- [ ] **Step 4: Run** the test and confirm all assertions pass.
- [ ] **Step 5: Commit** with `feat: add live optimizer core configuration`.

### Task 2: Append-only ledgers and causal evidence audit

**Files:**
- Create: `same_day_1445/live_optimizer/engine/ledger.py`
- Create: `same_day_1445/live_optimizer/engine/audit.py`
- Create: `same_day_1445/live_optimizer/ledger/README.md`
- Test: `tests/test_live_optimizer_ledger.py`

**Interfaces:**
- Produces: `AppendOnlyCsvLedger(path, key_fields)`, `append(row)`, `rows()`, `audit_causal_rows(rows, decision_field, observable_field)`, `hash_file(path)`.

- [ ] **Step 1: Write failing tests** that prove a first append succeeds, an identical duplicate is idempotent, a duplicate key with changed content raises `LedgerConflictError`, future-observable evidence is rejected, and hashes are deterministic.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_live_optimizer_ledger.py` and confirm expected failure.
- [ ] **Step 3: Implement** CSV ledgers using temp-file + atomic replace for writes. Preserve column order, UTF-8, and reject mutation of existing keyed rows.
- [ ] **Step 4: Run** the ledger tests and confirm pass.
- [ ] **Step 5: Commit** with `feat: add append-only optimizer ledgers`.

### Task 3: Bounded candidate generation and causal rolling evaluation

**Files:**
- Create: `same_day_1445/live_optimizer/engine/candidate_generator.py`
- Create: `same_day_1445/live_optimizer/engine/rolling_evaluator.py`
- Test: `tests/test_live_optimizer_evaluator.py`

**Interfaces:**
- Consumes: optimizer JSON candidate spaces and normalized closed-trade rows with `line`, `candidate_id`, `entry_date`, `exit_date`, `ret`, `triggered`, `regime`, `observable_at`.
- Produces: `generate_candidates(config, line) -> list[dict]`, `compute_metrics(rows, accounting) -> EvaluationMetrics`, `evaluate_candidate_windows(candidate_rows, baseline_rows, windows, accounting) -> dict`.

- [ ] **Step 1: Write failing tests** for bounded-grid generation, deterministic candidate ids, strict-before-window training evidence, trigger retention, win rate, mean return, max drawdown, and C/S/R vs D accounting conventions.
- [ ] **Step 2: Run** evaluator tests and confirm expected failure.
- [ ] **Step 3: Implement** only declared parameter grids. Candidate IDs are SHA-256-derived from canonical JSON. Evaluation windows must contain no rows whose observation timestamp is after the window decision boundary.
- [ ] **Step 4: Run** evaluator tests and confirm pass.
- [ ] **Step 5: Commit** with `feat: add bounded rolling candidate evaluation`.

### Task 4: Drift detection and hard promotion gate

**Files:**
- Create: `same_day_1445/live_optimizer/engine/drift.py`
- Create: `same_day_1445/live_optimizer/engine/gate.py`
- Test: `tests/test_live_optimizer_gate.py`

**Interfaces:**
- Produces: `detect_drift(recent_rows, reference_rows, config) -> dict`, `evaluate_gate(line, candidate_eval, baseline_eval, new_evidence, state, config) -> GateResult`.

- [ ] **Step 1: Write failing tests** for each gate boundary: 89.9% trigger retention rejects, +0.49pp win-rate rejects, mean-return decline rejects, >1pp drawdown deterioration rejects, 1/3 improving windows rejects, minimum-new-sample rejects, regime-count rejects for C/S/R, cooldown rejects, isolated parameter peak rejects, and severe drift forces `SHADOW_ONLY`.
- [ ] **Step 2: Run** gate tests and confirm expected failure.
- [ ] **Step 3: Implement** gate reasons as machine-readable codes plus numeric evidence. Drift initially uses robust relative changes in trigger rate, win rate, mean return, and feature summary statistics supplied by normalized rows; no third-party ML dependency.
- [ ] **Step 4: Run** gate tests and confirm pass.
- [ ] **Step 5: Commit** with `feat: add optimizer drift and promotion gates`.

### Task 5: Immutable release manager and rollback state machine

**Files:**
- Create: `same_day_1445/live_optimizer/engine/release_manager.py`
- Test: `tests/test_live_optimizer_release_manager.py`

**Interfaces:**
- Produces: `write_candidate_release(...) -> Path`, `promote_candidate(...) -> OptimizerState`, `rollback(...) -> OptimizerState`.

- [ ] **Step 1: Write failing tests** proving V2 cannot be modified, release directories are immutable once present, promotion requires `GateResult.passed`, `SHADOW_ONLY` cannot formal-promote, rollback restores previous active pointer, and every state transition emits an append-only promotion ledger row.
- [ ] **Step 2: Run** release-manager tests and confirm expected failure.
- [ ] **Step 3: Implement** immutable JSON release manifests under `live_optimizer/releases/<release_id>/`, candidate artifacts under `candidates/<candidate_id>/`, and active pointers only in `state/optimizer_state.json`.
- [ ] **Step 4: Run** release-manager tests and confirm pass.
- [ ] **Step 5: Commit** with `feat: add immutable optimizer release management`.

### Task 6: Daily optimizer orchestration and deterministic replay

**Files:**
- Create: `same_day_1445/live_optimizer/jobs/__init__.py`
- Create: `same_day_1445/live_optimizer/jobs/run_daily_optimizer.py`
- Create: `same_day_1445/live_optimizer/jobs/run_replay.py`
- Create: `same_day_1445/live_optimizer/engine/orchestrator.py`
- Test: `tests/test_live_optimizer_orchestrator.py`

**Interfaces:**
- Produces: `run_optimizer(root: Path, as_of: datetime) -> dict`, CLI `python -m same_day_1445.live_optimizer.jobs.run_daily_optimizer --root ... --as-of ...`, replay CLI over a ledger copy.

- [ ] **Step 1: Write failing end-to-end tests** with synthetic C and D evidence: new mature samples are ingested, candidates are ranked, shadow leader changes when metrics justify it, formal release remains unchanged in `SHADOW_ONLY`, audit log records the decision, and replay yields byte-identical decision JSON.
- [ ] **Step 2: Run** orchestrator tests and confirm expected failure.
- [ ] **Step 3: Implement** deterministic orchestration: load config/state → audit ledgers → build candidates → evaluate rolling windows → compute drift → apply gate → update shadow leader → optionally promote only in NORMAL mode → write run record with hashes.
- [ ] **Step 4: Run** orchestrator tests and confirm pass.
- [ ] **Step 5: Commit** with `feat: add daily self-optimizer orchestration`.

### Task 7: Phase-1 regression, README, and operational boundary

**Files:**
- Create: `same_day_1445/live_optimizer/README.md`
- Modify: `docs/OPTIMIZATION_PROTOCOL.md`
- Modify: `docs/REPOSITORY_SCOPE.md`
- Test: existing suite plus all new live-optimizer tests.

**Interfaces:**
- Documents exact Phase-1 input schema, commands, SHADOW_ONLY safety, Phase-2 provider boundary, and promotion semantics.

- [ ] **Step 1: Run full suite before doc changes**: `python -m pytest -q` plus `python scripts/verify_current_baseline.py` and record results.
- [ ] **Step 2: Update docs** to state that the optimizer core exists on the experimental branch, V2 remains canonical, live data/signal adapters are Phase 2, and no formal auto-promotion is enabled until forward evidence is accumulated.
- [ ] **Step 3: Run full suite again** and require zero failures.
- [ ] **Step 4: Run a synthetic CLI replay twice** and compare output hashes; require equality.
- [ ] **Step 5: Commit** with `docs: document live optimizer core and boundaries`.

## Phase-1 Acceptance Criteria

- `python -m pytest -q` passes existing V2 tests and all new optimizer tests.
- `python scripts/verify_current_baseline.py` still reports the frozen V2 baseline valid.
- Append-only conflict, future-data leakage, gate-boundary, cooldown, release immutability, rollback, and replay determinism all have regression tests.
- A synthetic daily run can change a shadow leader from new evidence without changing any formal V2 file.
- No runtime import or API call to GPT/LLM exists anywhere under `same_day_1445/live_optimizer/`.
- The core exposes stable interfaces required by Phase 2: normalized 14:45 snapshot records, normalized formal/shadow signal records, close/execution reconciliation rows, and matured closed-trade evidence.

## Follow-up Plans

After Phase 1 is green:

1. `live-self-optimizer-data-adapters`: real 14:45 minute snapshot acquisition, close reconciliation, C/S/D/R formal signal adapters, data-staleness checks, and local scheduling hooks.
2. `live-self-optimizer-dashboard-ops`: formal-vs-shadow real-price dashboard, promotion/drift views, Windows scheduler setup, runtime health checks, and forward-shadow operating instructions.
