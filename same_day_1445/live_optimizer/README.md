# 14:45 Live Self-Optimizer

This package is the GPT-independent forward optimization runtime for the C/S/D/R 14:45 ETF research system. It collects new trading-day evidence, runs the frozen formal strategy and executable shadow candidates in parallel, evaluates only matured forward outcomes, detects drift, and can programmatically promote or roll back a candidate through hard gates.

The canonical anchor remains `same_day_1445/release_v2`. The optimizer never rewrites that directory.

## What is implemented

The experimental branch currently contains the complete programmatic loop:

- true 14:45 minute snapshot acquisition with an Eastmoney provider and deterministic file-provider fixtures;
- causal provisional daily-bar overlay into a dedicated C/S/D/R runtime copy;
- adapters for the existing C/S/D formal engines;
- safe current `R-single` routing from `R/formal/r_single_yearly_choice.csv`, with ambiguous cases failing closed;
- V2 anchor signals plus executable shadow streams;
- C `C_DELAY1` shadow execution;
- bounded D strict/grid candidates for weak-market overheat and profit-giveback protection;
- append-only signal, execution, closed-trade, optimizer, promotion and mode-change evidence;
- same-opportunity formal/candidate matching so filtered trades remain part of trigger-retention accounting;
- causal rolling evaluation, forward window checks and parameter-neighborhood stability;
- frozen V2 anchor guard even after a newer formal release is active;
- minimum-new-sample accounting that resets after promotion;
- automatic promotion only when the account transition is safe;
- automatic rollback after sufficient post-promotion live deterioration;
- daily signal/data-quality drift plus closed-trade performance drift;
- dependency/runtime health checks;
- Windows Task Scheduler install/remove scripts;
- read-only local HTML dashboard;
- deterministic replay and immutable optimizer decision artifacts.

No runtime Python module in this package imports or calls GPT/LLM services.

## Safety model

Runtime modes are:

- `SHADOW_ONLY`: formal strategy continues, candidates learn/rank, no automatic formal promotion.
- `NORMAL`: automatic promotion is allowed only if every configured gate passes.
- `DATA_HOLD`: incomplete/stale data blocks affected evidence.
- `ROLLBACK`: a promoted release has been reverted to its parent; the optimizer returns to protected shadow operation.

Default promotion gates include:

- trigger retention `>= 90%`;
- rolling win-rate improvement `>= +0.5` percentage points;
- mean closed-trade return must not decline;
- maximum-drawdown deterioration `<= 1` percentage point;
- sufficient eligible forward windows and recent-window stability;
- stable neighboring parameter settings rather than an isolated optimum;
- minimum new evidence since the last promotion;
- cooldown and regime-coverage requirements;
- continued comparison with the immutable V2 anchor.

Default new-evidence minimums are D=100 closed trades and C/S/R=15 closed trades with at least two regime buckets. A newly observed trading day can affect data-quality/trigger drift immediately, but it cannot be scored as a win or loss until the trade outcome is actually observable.

## Required environments

Two directories have different roles:

1. This Git repository contains the optimizer code and frozen V2 audit surface.
2. A dedicated local runtime copy contains the full private C/S/D/R source package and mutable market/runtime files.

Do not point `runtime_root` at the archived source package itself. Bootstrap creates a disposable working copy so provisional 14:45 bars never rewrite the archive.

Install the repository dependencies first:

```bash
python -m pip install -r requirements.txt
```

The live health check explicitly verifies the critical runtime modules used by the existing strategies, including `pandas`, `numpy`, `sklearn` and `pyarrow`. Missing dependencies are reported as `BLOCKED` rather than being treated as valid strategy evidence.

## Bootstrap a dedicated runtime

From the repository root, with an empty destination runtime directory:

```bash
python -m same_day_1445.live_optimizer.jobs.bootstrap_runtime \
  --source-root "PATH_TO_FULL_CSDR_PACKAGE" \
  --runtime-root "PATH_TO_DEDICATED_LIVE_COPY" \
  --optimizer-root "same_day_1445/live_optimizer" \
  --provider eastmoney
```

For deterministic offline testing, use `--provider file --file-root PATH_TO_MINUTE_FIXTURES`.

Bootstrap writes the ignored runtime configuration to:

```text
same_day_1445/live_optimizer/runtime/workspace.json
```

and initializes the optimizer in `SHADOW_ONLY` unless an explicit deployment state already exists.

## Health check

Before scheduling or enabling automatic promotion:

```bash
python -m same_day_1445.live_optimizer.jobs.run_health \
  --root same_day_1445/live_optimizer
```

A critical dependency, runtime-script, state, reconciliation or data-quality failure returns `BLOCKED` and exit code 2.

## Daily pipeline

The scheduler calls the same deterministic CLI manually available here:

```bash
python -m same_day_1445.live_optimizer.jobs.run_pipeline preclose \
  --optimizer-root same_day_1445/live_optimizer \
  --workspace-config same_day_1445/live_optimizer/runtime/workspace.json

python -m same_day_1445.live_optimizer.jobs.run_pipeline close \
  --optimizer-root same_day_1445/live_optimizer \
  --workspace-config same_day_1445/live_optimizer/runtime/workspace.json

python -m same_day_1445.live_optimizer.jobs.run_pipeline optimizer \
  --optimizer-root same_day_1445/live_optimizer \
  --workspace-config same_day_1445/live_optimizer/runtime/workspace.json
```

The pre-close job is launched after the 14:45 minute observation is available, but its decision cutoff remains exactly `14:45:00`; bars from 14:46 through 15:00 cannot enter the signal snapshot. The close job reconciles the official close/reference values and matures only completed trades. The optimizer then evaluates the newly matured evidence and updates shadow/promotion/rollback state.

## Windows scheduling

Copy and edit:

```text
same_day_1445/live_optimizer/runtime/scheduler.example.json
```

Recommended defaults are:

- preclose: 14:45:30, with the internal cutoff fixed at 14:45;
- close reconciliation: 15:10;
- optimizer: 15:20.

Install the tasks from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\same_day_1445\live_optimizer\ops\install_windows_tasks.ps1 `
  -ConfigPath .\same_day_1445\live_optimizer\runtime\scheduler.json
```

Remove them with `remove_windows_tasks.ps1` using the same config. The scripts are idempotent and contain no repository-specific absolute user path.

## Enabling automatic promotion

Keep the system in `SHADOW_ONLY` while collecting genuinely new forward evidence. To enable `NORMAL`, first create a fresh V2 verification marker and pass the deployment health guard:

```bash
python -m same_day_1445.live_optimizer.jobs.set_mode NORMAL \
  --root same_day_1445/live_optimizer \
  --repo-root . \
  --verify-v2
```

`NORMAL` does not mean a candidate is automatically accepted. It only permits promotion when the statistical, trigger, neighborhood, forward-window, minimum-sample, cooldown, V2-anchor and account-transition gates all pass. `NORMAL -> SHADOW_ONLY` is always permitted as a safety action.

## Dashboard

Build the read-only dashboard data:

```bash
python -m same_day_1445.live_optimizer.jobs.build_dashboard \
  --root same_day_1445/live_optimizer
```

Serve it locally:

```bash
python -m same_day_1445.live_optimizer.jobs.serve_dashboard \
  --root same_day_1445/live_optimizer \
  --host 127.0.0.1 \
  --port 8765
```

Then open `http://127.0.0.1:8765/`.

The dashboard shows C/S/D/R formal release, V2 anchor, shadow leader, latest actions, forward-only metrics, new evidence count, gate status, drift/data quality, and promotion/rollback history. JavaScript is presentation-only and contains no trading or optimization logic.

## Evidence and replay

Important generated artifacts include:

- `ledger/anchor_signals.csv`: frozen V2 anchor decisions;
- `ledger/formal_signals.csv`: currently active formal decisions;
- `ledger/shadow_signals.csv`: executable candidate decisions;
- `ledger/closed_trades.csv`: matured comparable outcome evidence;
- `ledger/optimizer_runs.csv`: optimizer-run hashes;
- `ledger/optimizer_decisions/<hash>.json`: immutable decision artifacts;
- `ledger/promotion.csv`: promotion/rollback history;
- `state/optimizer_state.json`: active formal/shadow pointers;
- `state/deployment_mode.json`: requested deployment mode;
- `state/unreconciled_close.json`: unresolved close recovery state;
- `state/latest_optimizer_decision.json`: latest read-only decision projection.

Runtime evidence/state is ignored by Git and should be backed up separately.

Replay the same mature evidence/config at the same timestamp with:

```bash
python -m same_day_1445.live_optimizer.jobs.run_replay \
  --root same_day_1445/live_optimizer \
  --as-of 2026-08-24T15:20:00+08:00
```

The same inputs and configuration produce the same `decision_hash`.

## Verification

The experimental branch CI runs on Python 3.11 and requires all of the following:

```text
compile live_optimizer runtime
pytest tests/
verify_current_baseline.py
reject GPT/LLM runtime dependencies
```

The frozen `same_day_1445/release_v2` audit remains an independent required check. New forward results do not rewrite historical V2 files.
