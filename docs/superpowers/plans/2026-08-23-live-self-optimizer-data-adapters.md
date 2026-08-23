# Live Self-Optimizer Data Adapters Implementation Plan

> **Execution note:** implement test-first on `opt/live-self-optimizer-v1`. The canonical V2 release remains untouched.

**Goal:** Connect the Phase-1 optimizer core to real trading-day evidence: capture an immutable 14:45 minute snapshot, build a provisional daily bar using only data observed by 14:45, run the existing C/S/D signal engines in a dedicated runtime workspace, route R from the same-day C/S/D signal set, reconcile the final close, and turn completed formal/shadow positions into append-only optimizer evidence.

**Critical execution convention:** existing formal C/S/D engines were historically built around a next-close execution ledger. The audited V2 overlay moves the already-confirmed signal to the same signal day. Therefore the adapter does not reinterpret strategy logic: it runs the existing signal engine on the provisional 14:45 daily bar and maps a confirmed `next_trade_close_buy`/sell-style signal to a V2 `BUY`/`SELL` with `execution_mode=same_day_1445`. It must never use the 15:00 final close to create the 14:45 signal.

## Runtime boundary

The full C/S/D/R source package is a local runtime dependency and is not copied into the public GitHub repo. Configure its path with `runtime/workspace.json`. A dedicated live copy is required; the archived audit package itself must remain immutable.

## Data source

Primary provider: Eastmoney one-minute K-line endpoint using `klt=1`, no adjustment for current intraday aggregation. The provider must expose a file/mock implementation for deterministic offline tests and fail closed on stale/missing 14:45 data. Historical research still uses the existing daily-close proxy; only new forward evidence uses true intraday snapshots.

### Task A: Snapshot/provider layer

**Create:**
- `same_day_1445/live_optimizer/providers/base.py`
- `same_day_1445/live_optimizer/providers/eastmoney.py`
- `same_day_1445/live_optimizer/providers/file_provider.py`
- `same_day_1445/live_optimizer/engine/snapshot.py`
- `tests/test_live_optimizer_snapshot.py`

**Behavior:**
- fetch minute bars for each configured symbol;
- reject bars after target time from feature construction;
- aggregate `open/high/low/close/volume/amount` through 14:45;
- record requested time, latest observed minute, provider, and source hash;
- stale/missing required symbols cause `DATA_HOLD` evidence instead of synthetic prices.

### Task B: Provisional daily-bar overlay

**Create:**
- `same_day_1445/live_optimizer/engine/workspace.py`
- `tests/test_live_optimizer_workspace.py`

**Behavior:**
- patch only the dedicated runtime workspace;
- C raw receives the 14:45 provisional bar and feeds D;
- S raw receives its own 14:45 provisional bar;
- preserve all prior daily rows unchanged;
- current date is replaceable during the day but a prior date may never be rewritten by the adapter;
- final-close reconciliation replaces only the current provisional row after 15:00 and records the before/after hashes.

### Task C: C/S/D formal signal adapters

**Create:**
- `same_day_1445/live_optimizer/engine/signal_adapter.py`
- `same_day_1445/live_optimizer/config/signal_adapters.json`
- `tests/test_live_optimizer_signal_adapter.py`

**Behavior:**
- command adapter runs declared existing scripts in the configured full runtime workspace;
- C: `C/src/run_daily.py --skip-update`, parse `C/live/today_decision.csv`;
- S: `S/src/run_s1_live.py`, parse `S/live/today_decision.csv`;
- D: `D/src/daily_panic_live.py`, parse `D/live/daily_panic_today.json` and `daily_panic_scan.csv`;
- map confirmed next-close formal signals to audited V2 same-day 14:45 execution semantics;
- validate the engine data date equals the snapshot trade date;
- any non-zero command exit, stale data date, missing output, or ambiguous action returns a non-tradable adapter error and blocks optimizer evidence for that line/date.

### Task D: R live router adapter

**Create:**
- `same_day_1445/live_optimizer/engine/r_router.py`
- `tests/test_live_optimizer_r_router.py`

**Behavior:**
- read the current-year policy JSON from local `R/formal/r_plus_yearly_policy.csv`;
- use the same causal priority as formal `search_router_plus.py`: eligible C is anchor, otherwise evaluate declared filler order;
- S eligibility follows configured market-state set; D must satisfy current policy rank/panic/position/RSI thresholds;
- if R already has a position, its formal exit follows its selected source line signal/exit state and single-position rule;
- no future return or completed close may enter routing.

### Task E: 14:45 and close jobs

**Create:**
- `same_day_1445/live_optimizer/jobs/run_1445.py`
- `same_day_1445/live_optimizer/jobs/run_close.py`
- `same_day_1445/live_optimizer/engine/forward_ledger.py`
- `tests/test_live_optimizer_forward_jobs.py`

**14:45 job:**
1. load workspace/provider configuration;
2. obtain required C/S symbol universes;
3. freeze true 14:45 minute evidence;
4. update provisional daily bars in the dedicated runtime workspace;
5. run C/S/D adapters and R router;
6. append formal signal rows and enabled shadow signal rows;
7. update pending-position state only from already-recorded decisions.

**Close job:**
1. fetch final close reference after market close;
2. reconcile 14:45 proxy vs official close/VWAP reference;
3. update execution/position ledgers;
4. mature closed trade labels only when an actual recorded SELL closes a recorded BUY;
5. append normalized `closed_trades.csv` rows consumed by Phase 1;
6. invoke `run_daily_optimizer` after successful audit.

### Task F: Initial programmatic shadow modules

**Create:**
- `same_day_1445/live_optimizer/engine/shadow_policy.py`
- `tests/test_live_optimizer_shadow_policy.py`

**Initial shadows:**
- C `C_DELAY1`: queue formal BUY/SELL signal and execute on the next observed trading day without changing signal direction.
- D strict bounded protection family: weak-market overheat entry filters plus armed-profit giveback exits. Parameters are declared in JSON and include the latest researched neighborhood; no arbitrary runtime code generation.
- S/R bounded challengers may be recorded only when a declared module exists; the system must not pretend an unapplied config changed a strategy.

The self-optimizer ranks only evidence actually produced by a shadow module.

### Task G: Data-hold, audit, and replay integration

**Modify:**
- `engine/orchestrator.py`
- `engine/audit.py`
- `config/optimizer.json`
- tests

**Behavior:**
- stale minute snapshot or adapter failure writes a run record and blocks that line/date from promotion evidence;
- no incomplete current-day trade is scored as a loss/win;
- replay from saved snapshots and signal outputs must reproduce the same formal/shadow decisions without network access;
- true forward records retain data/source hashes and code/config hashes.

## Phase-2 acceptance criteria

- Offline fixture at 14:45 proves no 14:46–15:00 bar can influence a signal snapshot.
- Current-day provisional daily bar aggregation is exact and prior dates are immutable.
- C/S/D output parsers map representative real package outputs to normalized BUY/SELL/HOLD records.
- R router reproduces `plus_00014` priority/filters on fixture candidates.
- A synthetic sequence BUY → HOLD → SELL produces exactly one mature closed-trade evidence row, never earlier.
- C Delay1 produces a separate candidate ledger without altering formal C.
- Adapter/data failures force non-tradable evidence and cannot reach the promotion gate.
- Phase-1 tests remain green and V2 tree remains untouched.
