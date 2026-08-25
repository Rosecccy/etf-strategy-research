# C/S/R Start-State Robustness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-only audit that measures how C/S/R results change when the same causal strategy is started on different trading dates from either inherited account state (Warm) or forced cash (Cold), without modifying frozen V2 or the live optimizer.

**Architecture:** The audit is isolated under `research/start_state_robustness/`. It first discovers and validates source-package opportunity ledgers against the frozen V2 C/S/R trade sequence; it refuses to scan if canonical replay cannot be reproduced. A deterministic single-position simulator then runs Warm/Cold paths over every eligible start date and fixed 1/2/3/5-year horizons, followed by buy-date perturbation, aggregation, divergence attribution, and a research report.

**Tech Stack:** Python 3.11+ standard library, pandas, numpy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-start-state-robustness-audit-design.md`

## Global Constraints

- Never modify `same_day_1445/release_v2`.
- Never modify `same_day_1445/live_optimizer/state`, `ledger`, `runtime`, `recovery`, Windows scheduler files, or active formal pointers.
- C/S/R retain the frozen V2 single-position compound-account convention.
- Same-day sell-then-switch is allowed; a candidate entry is blocked only when `entry_date < current_exit_date`.
- Information history is preserved; only account state changes at the tested start date.
- Warm inherited positions are re-based using the actual start-date close, not the original entry price.
- No robustness interpretation is allowed unless source-opportunity replay reproduces frozen V2 trade keys for that line.
- All generated artifacts go under `research/start_state_robustness/out/`.
- This phase does not implement D; D receives a separate research-sum plan after C/S/R is validated.

---

### Task 1: Source-package opportunity inventory and canonical replay gate

**Files:**
- Create: `research/start_state_robustness/__init__.py`
- Create: `research/start_state_robustness/source_inventory.py`
- Create: `tests/test_start_state_source_inventory.py`

**Interfaces:**
- Consumes: local full source package root containing `C/`, `S/`, and `R/`; frozen reference `same_day_1445/release_v2/selected_trades_csr.csv`.
- Produces: `inventory_source_tables(source_root: Path, repo_root: Path, line: str) -> list[dict]` and CLI output `research/start_state_robustness/out/source_inventory.csv`.

- [ ] **Step 1: Write failing inventory tests**

Create synthetic `C/fit/*.csv` candidates and a tiny frozen ledger. Test that only CSVs containing a symbol column, an entry-date column, an exit-date column, and enough rows are eligible; test aliases `entry`/`entry_date`, `exit`/`exit_date`/`exit_date_test`/`actual_exit_date`, `entry_close`/`entry_price`, and `exit_close`/`exit_price`/`exit_close_test`/`actual_exit_close`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest -q tests/test_start_state_source_inventory.py`

Expected: import/function failure because the research package does not exist yet.

- [ ] **Step 3: Implement header-only discovery and candidate scoring**

Recursively scan only `<source_root>/<line>/fit/**/*.csv`, `<source_root>/<line>/results/**/*.csv`, and `<source_root>/<line>/out/**/*.csv`; never scan raw ETF price files as opportunity tables. Preserve source row order as `_source_row`. For each candidate report row count, mapped columns, canonical frozen-key coverage on `(symbol, entry, exit)`, extra opportunity count, and path.

- [ ] **Step 4: Add canonical replay scoring**

For each structurally valid candidate, normalize rows and replay the V2 single-position acceptance rule in source-row order within each entry date. Compare resulting `(symbol, entry, exit)` keys with frozen V2 keys for the requested line. Record `exact_replay`, `missing_frozen`, `unexpected_accepted`, and `opportunity_rows`. Prefer exact replay with `opportunity_rows > frozen_rows`; if multiple exact candidates exist, do not guess: rank them and require an explicit path override.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_start_state_source_inventory.py`

Expected: PASS.

Commit message: `feat: add start-state source inventory gate`.

---

### Task 2: Deterministic C/S/R Warm-Cold simulator

**Files:**
- Create: `research/start_state_robustness/csr_engine.py`
- Create: `tests/test_start_state_csr_engine.py`

**Interfaces:**
- Consumes: normalized opportunity DataFrame with `symbol`, `entry`, `exit`, `entry_close`, `exit_close`, optional `policy_id`, `_source_row`; trading calendar and raw-close lookup.
- Produces: `simulate_path(...)`, `warm_state_at(...)`, `compare_paths(...)` and per-start metric dictionaries.

- [ ] **Step 1: Write failing state-path tests**

Use a synthetic opportunity stream where trade A is open across a later B entry. Verify: Warm rejects B while Cold accepts B; same-day `entry == current_exit` is allowed; starting on a flat date produces identical paths; Warm inherited return is re-based from start-date mark to inherited exit; deterministic reruns are identical.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_start_state_csr_engine.py`

- [ ] **Step 3: Implement position-state model**

Represent a position with immutable future-relevant state: symbol, source opportunity id, original entry date, scheduled exit date, scheduled exit close, policy id. Cold initializes flat. Warm reconstructs the canonical accepted path from origin and carries the active position across the account start if `entry <= start < exit`.

- [ ] **Step 4: Implement horizon mark-to-market and metrics**

For horizons 252/504/756/1260 trading days, compound completed trade returns. If a position remains open at the horizon endpoint, mark it at that day's raw close. Report cumulative return, annualized return where defined, completed win rate, mean completed trade return, max drawdown, accepted count, blocked-buy count, extra-Cold count, first divergence date, first re-synchronization date, re-synchronization trading days, and trade-sequence Jaccard similarity.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_start_state_csr_engine.py`

Expected: PASS.

Commit message: `feat: add deterministic warm cold account simulator`.

---

### Task 3: Full start-date scan and buy-signal perturbation

**Files:**
- Create: `research/start_state_robustness/scan.py`
- Create: `tests/test_start_state_scan.py`

**Interfaces:**
- Consumes: validated opportunity source per line, raw price directory, frozen reference ledger.
- Produces: `scan_line(...) -> (per_start_df, perturbation_df, divergence_df)`.

- [ ] **Step 1: Write failing scan tests**

Create a 40-day synthetic calendar with several overlapping opportunities. Verify every eligible trading date is tested for a short fixture horizon, insufficient-future dates are excluded, and BUY-offset starts use exactly `-20,-10,-5,-1,0,1,5,10,20` trading-day offsets with deduplication.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_start_state_scan.py`

- [ ] **Step 3: Implement calendar and raw-close adapter**

Use `<source_root>/C/raw/etf/510500.csv` as the preferred common trading calendar, falling back to the line's first opportunity symbol raw file when unavailable. Raw-close lookup must reject missing start/horizon marks rather than impute them.

- [ ] **Step 4: Implement all-start scan**

For each start and each available fixed horizon, run Warm and Cold using identical opportunity rows and causal policy data. Persist one paired result row containing both path metrics plus return/drawdown gaps and start-state metadata.

- [ ] **Step 5: Implement divergence attribution**

For each start with divergent accepted sequences, save the first signal where one path accepts and the other rejects, both pre-signal position states, the blocking reason, subsequent accepted opportunity ids until re-synchronization, and the return/drawdown gap at re-synchronization or horizon end.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest -q tests/test_start_state_scan.py`

Expected: PASS.

Commit message: `feat: scan start-date and buy-offset path dependence`.

---

### Task 4: Aggregate robustness labels and research report

**Files:**
- Create: `research/start_state_robustness/report.py`
- Create: `tests/test_start_state_report.py`

**Interfaces:**
- Consumes: paired per-start results from Task 3.
- Produces: `summary.csv`, `summary.json`, `worst_divergences.csv`, PNG plots, and `REPORT.md` under `research/start_state_robustness/out/`.

- [ ] **Step 1: Write failing label tests**

Use synthetic 3-year distributions to verify the pre-registered labels from the approved spec: Robust, Moderately path-dependent, and Fragile. Assert boundary cases for profitable-start ratio, P10 return, median/P90 Warm-Cold gap, and median/P90 re-synchronization days.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_start_state_report.py`

- [ ] **Step 3: Implement aggregate statistics**

For each line/horizon calculate median, P10, P25, worst-5%-mean return, profitable-start ratio, median/P90 max drawdown, median/P90 absolute Warm-Cold return gap, median/P90 re-synchronization days, never-resynchronized ratio, and return dispersion.

- [ ] **Step 4: Implement report and plots**

Use matplotlib with one plot per figure and default colors: start-date vs 1/2/3/5-year return, start-date vs Warm-Cold return gap, start-date vs max drawdown, and start-date vs re-synchronization time. The report must state the exact validated opportunity source path and canonical replay result for each line before giving any robustness label.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_start_state_report.py`

Expected: PASS.

Commit message: `feat: report start-state robustness distributions`.

---

### Task 5: CLI, safety guards, full regression, and local source-package execution

**Files:**
- Create: `research/start_state_robustness/run_audit.py`
- Create: `tests/test_start_state_cli_safety.py`
- Modify: `.gitignore`

**Interfaces:**
- CLI: `python -m research.start_state_robustness.run_audit --source-root <path> [--c-opportunities <csv>] [--s-opportunities <csv>] [--r-opportunities <csv>]`.

- [ ] **Step 1: Write failing safety tests**

Verify the CLI refuses to run the interpretation phase when any line lacks exact canonical replay, refuses output roots inside `same_day_1445/release_v2` or `same_day_1445/live_optimizer`, and never writes outside `research/start_state_robustness/out/` by default.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_start_state_cli_safety.py`

- [ ] **Step 3: Implement two-stage CLI**

Stage `inventory` always works and writes ranked source candidates. Stage `scan` runs only for lines whose source file passes exact replay. If auto-discovery is ambiguous, print the top candidates and require the corresponding `--*-opportunities` argument; do not silently choose.

- [ ] **Step 4: Run complete regression suite**

Run: `python -m pytest -q tests --basetemp=.pytest_tmp`

Expected: all existing tests plus the new start-state tests pass.

- [ ] **Step 5: Verify frozen baseline remains unchanged**

Run: `python scripts/verify_current_baseline.py`

Expected: `CURRENT BASELINE OK`.

- [ ] **Step 6: Execute local source inventory before the full scan**

On the user's machine, run:

`python -m research.start_state_robustness.run_audit inventory --source-root "%USERPROFILE%\Desktop\gsqh"`

Inspect `research/start_state_robustness/out/source_inventory.csv`. Do not run the full interpretation if C/S/R canonical replay is not exact.

- [ ] **Step 7: Execute the validated C/S/R scan**

Once source paths are exact or explicitly selected, run:

`python -m research.start_state_robustness.run_audit scan --source-root "%USERPROFILE%\Desktop\gsqh"`

Review `REPORT.md`, `summary.csv`, and worst-divergence traces. Only after these results are understood should any mitigation be designed.

- [ ] **Step 8: Commit**

Commit message: `test: add isolated C S R start-state robustness audit`.

---

## Acceptance Criteria

- The research branch and local patch do not alter frozen V2 or live optimizer state/runtime/scheduler files.
- A validated opportunity source reproduces the frozen C/S/R canonical trade sequence before any robustness result is trusted.
- Warm and Cold differ only in account state at the account start; historical information availability is identical.
- Every eligible start date and pre-registered BUY offset is represented; no poor start is dropped.
- 1/2/3/5-year fixed-horizon distributions and Warm-Cold gaps are reported.
- Robustness labels use the thresholds fixed in the approved design, not thresholds chosen after seeing results.
- Full test suite and `verify_current_baseline.py` remain green.
