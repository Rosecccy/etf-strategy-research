# Live Self-Optimizer Dashboard and Ops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use the already-approved live-self-optimizer design and execute test-first on `opt/live-self-optimizer-v1`.

**Goal:** Make the forward optimizer operable without GPT: install deterministic Windows schedules, expose runtime health checks, generate a local formal-vs-shadow dashboard, and provide a safe deployment switch from SHADOW_ONLY to automatic NORMAL mode without modifying frozen V2.

**Architecture:** The runtime remains three jobs: 14:45 evidence/signals, post-close reconciliation, and post-close optimizer. Scheduling invokes Python CLIs only. A new ops layer reads append-only ledgers/state and produces machine-readable health/dashboard JSON plus a static HTML view; it never contains trading or optimization logic.

**Tech Stack:** Python standard library, PowerShell/Windows Task Scheduler, HTML/CSS/vanilla JavaScript, pytest. No GPT/LLM runtime dependency.

**Spec:** `docs/superpowers/specs/2026-08-23-live-self-optimizer-design.md`

## Global constraints

- 14:45 decision cutoff remains exactly 14:45:00 local market time even if the scheduled process starts seconds later so the 14:45 minute bar is published.
- `same_day_1445/release_v2` remains immutable and is always retained as an anchor stream.
- Automatic promotion is enabled only by an explicit deployment config/state switch to `NORMAL`; all statistical/sample/transition gates remain mandatory.
- Dashboard/health code is read-only with respect to formal strategy state.
- Scheduler install scripts must be idempotent and must not embed machine-specific absolute paths in Git.

---

### Task 1: Runtime health model and preflight

**Create:**
- `same_day_1445/live_optimizer/ops/health.py`
- `same_day_1445/live_optimizer/jobs/run_health.py`
- `tests/test_live_optimizer_health.py`

**Behavior/tests:**
- report workspace config present, runtime root exists, required C/S/D formal scripts/outputs are discoverable, optimizer config/state parse, last 14:45/close/optimizer run ages, unresolved close count, DATA_HOLD lines, current formal/shadow release ids, and last promotion/rollback;
- return `OK`, `WARN`, or `BLOCKED` with machine-readable codes;
- a stale or missing critical runtime dependency must be `BLOCKED`, not silently accepted.

### Task 2: Deterministic daily pipeline wrapper

**Create:**
- `same_day_1445/live_optimizer/jobs/run_pipeline.py`
- `tests/test_live_optimizer_pipeline.py`

**Behavior/tests:**
- subcommands `preclose`, `close`, `optimizer`, and `health` delegate to existing jobs;
- `preclose` always passes a 14:45 cutoff even when launched at 14:45:30/14:46;
- optimizer never runs if close audit reports unresolved critical evidence for the affected line;
- repeated invocation for the same date remains append-only/idempotent.

### Task 3: Windows scheduler installer

**Create:**
- `same_day_1445/live_optimizer/ops/install_windows_tasks.ps1`
- `same_day_1445/live_optimizer/ops/remove_windows_tasks.ps1`
- `same_day_1445/live_optimizer/runtime/scheduler.example.json`
- `tests/test_live_optimizer_scheduler_files.py`

**Behavior/tests:**
- create three named tasks with configurable Python executable/repo/runtime paths;
- recommended defaults: preclose launch 14:45:30 with cutoff fixed at 14:45, close reconciliation 15:10, optimizer 15:20;
- task actions call `python -m same_day_1445.live_optimizer.jobs.run_pipeline ...`;
- scripts are idempotent and support removal;
- no user-specific path is committed.

### Task 4: Dashboard data builder

**Create:**
- `same_day_1445/live_optimizer/ops/dashboard.py`
- `same_day_1445/live_optimizer/jobs/build_dashboard.py`
- `tests/test_live_optimizer_dashboard.py`

**Behavior/tests:**
- read state, formal/anchor/shadow signal ledgers, closed trades, optimizer runs, promotion ledger, execution references and daily drift;
- produce `site/data/dashboard.json` atomically;
- per C/S/D/R show current formal release/evidence id, shadow leader, latest action, V2 anchor action, new matured samples since last promotion, trigger retention, rolling win/mean-return/drawdown metrics, gate status, daily/performance drift, pending promotion, rollback status and unresolved data issues;
- builder never mutates trading state.

### Task 5: Static local dashboard

**Create:**
- `same_day_1445/live_optimizer/site/index.html`
- `same_day_1445/live_optimizer/site/app.js`
- `same_day_1445/live_optimizer/site/styles.css`
- `same_day_1445/live_optimizer/jobs/serve_dashboard.py`
- `tests/test_live_optimizer_site_contract.py`

**Behavior/tests:**
- light research dashboard, four C/S/D/R cards, latest formal vs V2-anchor vs shadow actions, forward-only metric table, promotion-gate checklist, drift/data-quality banner, promotion/rollback timeline;
- no optimization code in JavaScript;
- serve only local files with Python HTTP server.

### Task 6: Deployment-mode guard

**Create:**
- `same_day_1445/live_optimizer/ops/deploy.py`
- `same_day_1445/live_optimizer/jobs/set_mode.py`
- `tests/test_live_optimizer_deploy.py`

**Behavior/tests:**
- `SHADOW_ONLY -> NORMAL` is allowed only if health is not BLOCKED, V2 baseline verification marker is present/current, runtime candidate modules are executable, and there is no unresolved DATA_HOLD/close reconciliation state;
- switching mode writes an append-only mode-change ledger and updates only optimizer state/config deployment override, never release_v2;
- `NORMAL -> SHADOW_ONLY` is always permitted as a safety action;
- mode change is programmatic CLI, not GPT-mediated.

### Task 7: End-to-end ops regression

**Modify:**
- `same_day_1445/live_optimizer/README.md`
- `docs/OPTIMIZATION_PROTOCOL.md`
- tests

**Verification:**
- full pytest suite;
- compileall under `same_day_1445/live_optimizer`;
- scan Python runtime for GPT/LLM imports/calls;
- deterministic offline fixture: preclose -> close -> optimizer -> dashboard twice with identical outputs;
- confirm `same_day_1445/release_v2` tree/hash is unchanged versus main.

## Acceptance criteria

- A Windows machine can configure a dedicated runtime, install tasks, and run the full pipeline without ChatGPT.
- Every trading day contributes immediate signal/data-quality drift evidence; closed trades contribute outcome labels only after maturity.
- Automatic promotion/rollback works only through programmatic hard gates and account-transition safety.
- The dashboard is a read-only projection of ledgers/state and can explain why a candidate is shadow, pending, promoted, rejected, rolled back, or blocked.
- Frozen V2 remains immutable and continuously available as the anchor comparator.
