# Live Self-Optimizer Data Adapters Implementation Plan

> **Execution note:** implement test-first on `opt/live-self-optimizer-v1`. The canonical V2 release remains untouched.

**Goal:** Connect the Phase-1 optimizer core to real trading-day evidence: capture an immutable 14:45 minute snapshot, build a provisional daily bar using only data observed by 14:45, run the existing C/S/D signal engines in a dedicated runtime workspace, safely route current formal R-single when it is reproducible without guessing, reconcile the final close, and turn completed formal/shadow positions into append-only optimizer evidence.

**Critical execution convention:** existing formal C/S/D engines were historically built around a next-close execution ledger. The audited V2 overlay moves an already-confirmed signal to the same signal day. The adapter therefore runs the existing signal engines on the provisional 14:45 daily bar and maps confirmed next-close-style actions to `same_day_1445`. It must never use the 15:00 final close to create the 14:45 signal.

## Runtime boundary

The full C/S/D/R source package is a local runtime dependency and is not copied into the public GitHub repo. Its path is configured in `runtime/workspace.json`. A dedicated live copy is required; the archived audit package remains immutable.

The existing full-package orchestration entrypoint is `C/src/run_all_daily.py --execution-mode preclose`; Phase 2 reuses its formal C/S/D logic rather than rewriting those strategies.

## Data source

Primary forward provider: Eastmoney one-minute K-line endpoint using `klt=1`, no adjustment for current intraday aggregation. A file provider is mandatory for deterministic offline tests. Stale or missing required 14:45 observations fail closed into `DATA_HOLD`. Historical research continues to use the existing daily-close proxy; only newly collected forward evidence is labeled as true intraday evidence.

### Task A: Snapshot/provider layer

**Create:**
- `same_day_1445/live_optimizer/providers/base.py`
- `same_day_1445/live_optimizer/providers/eastmoney.py`
- `same_day_1445/live_optimizer/providers/file_provider.py`
- `same_day_1445/live_optimizer/engine/snapshot.py`
- `tests/test_live_optimizer_snapshot.py`

**Behavior:** fetch minute bars, discard observations later than the requested decision time, aggregate open/high/low/close/volume/amount through 14:45, record source hashes/timestamps, and fail closed on stale/missing symbols.

### Task B: Provisional daily-bar overlay

**Create:**
- `same_day_1445/live_optimizer/engine/workspace.py`
- `tests/test_live_optimizer_workspace.py`

**Behavior:** patch only the dedicated runtime workspace. C raw receives its 14:45 provisional bar and remains the D source; S raw receives its own provisional bar. Prior dates are immutable. The current date may be replaced during the session and reconciled once after close, with before/after hashes recorded.

### Task C: C/S/D formal signal adapters

**Create:**
- `same_day_1445/live_optimizer/engine/signal_adapter.py`
- `same_day_1445/live_optimizer/config/signal_adapters.json`
- `tests/test_live_optimizer_signal_adapter.py`

**Behavior:**
- C: run `C/src/run_daily.py --skip-update`, parse `C/live/today_decision.csv`;
- S: run `S/src/run_s1_live.py`, parse `S/live/today_decision.csv`;
- D: run `D/src/daily_panic_live.py`, parse `D/live/daily_panic_today.json` plus scan evidence;
- map only unambiguous actions to normalized BUY/SELL/HOLD;
- require engine data date to equal the snapshot trade date;
- command failure, stale date, missing output, or ambiguous action becomes non-tradable adapter evidence and blocks promotion evidence for that line/date.

### Task D: Current formal R-single live router

**Create:**
- `same_day_1445/live_optimizer/engine/r_router.py`
- `tests/test_live_optimizer_r_router.py`

**Correct formal source:** `R/formal/r_single_yearly_choice.csv`, not historical `r_plus` policies. Current frozen yearly policy is `base_CS`, using only C/S.

**Behavior:**
- read the current-year `r_single_yearly_choice.csv` policy and refuse unknown/unsupported policy ids;
- preserve R single-position semantics and same-day rotation rule;
- if only one of C/S has an actionable formal entry, route that candidate without reinterpretation;
- if neither is actionable, HOLD/no-entry;
- if both are actionable, select only when the exact formal scoring inputs needed by the frozen policy are present in normalized adapter evidence; otherwise return R `DATA_HOLD` for that date instead of guessing;
- if R already holds a source-line position, exit only when the recorded source-line exit state can be reproduced from current formal evidence;
- never use future return, final close, or completed-trade outcomes for routing.

### Task E: 14:45 and close jobs

**Create:**
- `same_day_1445/live_optimizer/jobs/run_1445.py`
- `same_day_1445/live_optimizer/jobs/run_close.py`
- `same_day_1445/live_optimizer/engine/forward_ledger.py`
- `tests/test_live_optimizer_forward_jobs.py`

**14:45 job:** load workspace/provider config, freeze true 14:45 evidence, patch provisional daily bars, run C/S/D formal adapters and the safe R-single router, append formal and implemented shadow signals, and update pending-position state only from recorded decisions.

**Close job:** fetch/reconcile final close reference, append execution rows, mature a closed-trade label only when a recorded SELL closes a recorded BUY, append `closed_trades.csv`, then invoke the Phase-1 daily optimizer after successful audit.

### Task F: Initial programmatic shadow modules

**Create:**
- `same_day_1445/live_optimizer/engine/shadow_policy.py`
- `tests/test_live_optimizer_shadow_policy.py`

**Initial shadows:**
- C `C_DELAY1`: queue a formal BUY/SELL signal and execute it on the next observed trading day without changing direction;
- D strict bounded protection family: weak-market overheat entry filters plus armed-profit giveback exits from declared JSON grids;
- S/R challengers are recorded only when an executable declared module exists. A config value alone must never be counted as a shadow strategy.

### Task G: Data-hold, audit, replay and runtime config

**Create/modify:**
- `same_day_1445/live_optimizer/runtime/workspace.example.json`
- `same_day_1445/live_optimizer/engine/orchestrator.py`
- `same_day_1445/live_optimizer/engine/audit.py`
- `same_day_1445/live_optimizer/config/optimizer.json`
- tests

**Behavior:** stale snapshots or adapter failures block affected line/date evidence; incomplete trades are never scored; replay from stored snapshots and normalized signal outputs is network-free and deterministic; every forward row retains source/config/code hashes.

## Phase-2 acceptance criteria

- A fixture containing 14:46–15:00 bars proves they cannot affect the 14:45 snapshot.
- Provisional daily aggregation is exact and prior dates cannot be rewritten.
- Representative real C/S/D output rows parse to the expected normalized actions.
- Current formal R-single reads `r_single_yearly_choice.csv`; unsupported or ambiguous dual-entry cases fail closed rather than using an obsolete R policy.
- A synthetic BUY → HOLD → SELL sequence produces exactly one mature closed-trade evidence row and never matures earlier.
- C Delay1 produces a separate candidate ledger without altering formal C.
- Adapter/data failures cannot reach promotion evidence.
- Phase-1 tests remain green and `same_day_1445/release_v2` remains byte-for-byte untouched.
