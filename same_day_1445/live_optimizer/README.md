# 14:45 Live Self-Optimizer

This package is the GPT-independent forward optimization core for the C/S/D/R 14:45 ETF research system.

## Current status

Phase 1 is implemented as a provider-agnostic core and must run in `SHADOW_ONLY` by default. The canonical formal baseline remains `same_day_1445/release_v2`; this package never rewrites that directory.

The core already provides:

- append-only evidence ledgers with mutation conflict detection;
- causal timestamp auditing;
- bounded candidate-grid generation from `config/optimizer.json`;
- C/S/R compound and D research-sum evaluation;
- rolling-window comparison against the active formal release;
- hard promotion gates for trigger retention, win rate, mean return, drawdown, recent-window stability, minimum new evidence, regime coverage, cooldown and parameter-neighborhood stability;
- drift detection that can force `SHADOW_ONLY`;
- immutable candidate/release manifests and rollback pointers;
- deterministic daily optimizer and replay entry points.

Phase 2 will connect real 14:45 minute snapshots, close reconciliation and the existing C/S/D/R live signal entry points. Until that adapter phase is merged, `closed_trades.csv` is a normalized input ledger produced by tests or an external adapter, not a claim that GitHub itself is already collecting live market data.

## Runtime layout

- `config/optimizer.json`: declared candidate spaces and hard gates.
- `state/optimizer_state.example.json`: tracked initial template. Runtime creates ignored `state/optimizer_state.json` with active formal pointers and shadow leaders.
- `ledger/closed_trades.csv`: matured formal/shadow evidence; append-only.
- `ledger/optimizer_runs.csv`: deterministic optimizer run records.
- `ledger/promotion.csv`: promotion/rollback events.
- `candidates/<candidate_id>/manifest.json`: immutable candidate manifests.
- `releases/<release_id>/manifest.json`: immutable promoted release manifests.

Runtime ledgers are intentionally not pre-populated in Git. Production/forward evidence should be stored in controlled runtime storage and backed up independently.

## Normalized closed-trade schema

The Phase-1 evaluator consumes rows with at least:

```text
sample_id,line,candidate_id,decision_at,observable_at,entry_date,exit_date,ret,triggered,regime
```

Rules:

- `decision_at` is the decision timestamp being evaluated.
- `observable_at` is the latest timestamp needed to construct that row's decision evidence and must be `<= decision_at`.
- `candidate_id=release_v2` is the initial formal comparator.
- `triggered` records whether an opportunity became an executed trade under that candidate.
- `ret` is the closed-trade return used only after the label has matured.
- `regime` is a bounded programmatic bucket used for minimum-regime coverage, not an LLM-generated label.

## Daily optimizer

With a populated runtime root:

```bash
python -m same_day_1445.live_optimizer.jobs.run_daily_optimizer \
  --root same_day_1445/live_optimizer \
  --as-of 2026-08-23T16:00:00+08:00
```

Replay the same evidence/config at the same timestamp:

```bash
python -m same_day_1445.live_optimizer.jobs.run_replay \
  --root same_day_1445/live_optimizer \
  --as-of 2026-08-23T16:00:00+08:00
```

The replay prints the deterministic `decision_hash`.

## Promotion behavior

Three concepts are distinct:

1. `formal`: currently active release.
2. `shadow_leader`: best candidate that passed the configured gate using available matured evidence.
3. `challengers`: other evaluated candidates.

In `SHADOW_ONLY`, the program may update `shadow_leader` but cannot formal-promote. `release_manager.promote_candidate()` also enforces this at code level. In future `NORMAL` mode, promotion additionally requires a passed `GateResult`, sufficient forward evidence and a frozen candidate manifest.

Default minimum new evidence is D=100 closed trades and C/S/R=15 closed trades with at least two regime buckets. These are config values, not hidden heuristics.

## Safety

- Severe live drift forces `SHADOW_ONLY`.
- Duplicate ledger keys with changed content raise `LedgerConflictError`.
- Candidate/release directories are immutable once created.
- Rollback points to the previous formal release; history is never deleted.
- No file under this package imports or calls GPT/LLM services.
