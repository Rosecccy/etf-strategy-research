# C/S/D v2 exit-layer test

Only the exit layer is changed. Entry signals, C/S/D judge, ETF universe, and v2 selected trades are fixed.

All stop/take-profit/trailing rules are close-price based. `same_close` is a near-close execution approximation; `next_close` is the stricter next-trading-day close version.

## Baseline v2
- n: 1513
- win: 60.08%
- avg trade: 3.43%
- return sum: 5194.67%
- profit factor: 2.84
- annual avg sum: 649.33%
- worst year sum: -23.38%

## Full-sample diagnostic best
- rule: line_tp_C25%_S12%_D15%_next_close
- win: 61.73%
- avg trade: 4.27%
- return sum: 6454.21%
- profit factor: 3.36
- annual avg sum: 806.78%
- worst year sum: 134.05%

## Strict rolling best
- selector: conservative_tp_center15_profit_first
- win: 61.60%
- avg trade: 3.73%
- return sum: 5646.43%
- profit factor: 3.05
- annual avg sum: 705.80%
- worst year sum: 93.76%