# Strict rolling upgrade report

Generated: 2026-08-11

## Result

The useful upgrade is not the small 2026 S-line overheat filter. The stronger and stricter solution is the existing C/S/D v2 exit guard branch:

`conservative_tp_center15_profit_first`

This strategy is a strict rolling exit selector:

- The base trade candidates come from `all_priority_bias_min3`.
- For each test year, the main policy is selected using earlier years only.
- The exit rule is also selected using earlier years only.
- The selected exit rule is then applied to that test year.

## Best rule path

The best selector mainly converges to:

`tp15%_all_next_close`

Plain meaning:

- Keep the original C/S/D buy signals.
- Do not reduce signal frequency.
- If a trade reaches about +15%, sell on the next close.
- If it does not reach the take-profit level, keep the original exit.

## Core comparison

| Version | Trades | Win Rate | Avg Trade | Sum Return | Profit Factor | Worst Year |
|---|---:|---:|---:|---:|---:|---:|
| v2 original repriced | 1513 | 60.08% | 3.43% | 51.95 | 2.84 | -0.23 |
| strict rolling exit guard | 1513 | 61.60% | 3.73% | 56.46 | 3.05 | +0.94 |

## Year-by-year comparison

| Year | Original Sum | Exit Guard Sum | Original Win | Exit Guard Win |
|---:|---:|---:|---:|---:|
| 2019 | 6.24 | 6.24 | 78.09% | 78.09% |
| 2020 | 8.72 | 8.02 | 78.75% | 85.00% |
| 2021 | 7.93 | 9.30 | 61.78% | 63.56% |
| 2022 | 3.27 | 6.30 | 51.54% | 55.07% |
| 2023 | 3.41 | 3.71 | 52.89% | 53.31% |
| 2024 | 14.83 | 15.66 | 61.98% | 61.98% |
| 2025 | 7.77 | 6.31 | 57.61% | 58.02% |
| 2026 | -0.23 | 0.94 | 43.42% | 48.68% |

## Diagnosis

The old C/S/D issue is not mainly bad buying. Many buy points are usable, but fixed exits often hold through a good rebound and give back profit. The strict rolling exit guard improves the system by selling part of the rebound earlier after a large enough move.

This directly fixes the problem the charts exposed:

- It avoids waiting too long after a rebound.
- It does not cut signal frequency.
- It improves the weakest year from negative to positive.
- It is selected by earlier years, not by peeking at 2026.

## Failed add-on

The S-line overheat filter is not recommended on top of this exit guard.

| Version | All Sum | 2026 Sum | Recent 3M Sum | Active Rate |
|---|---:|---:|---:|---:|
| exit guard | 56.46 | 0.94 | 0.69 | 99.60% |
| exit guard + S-overheat filter | 55.12 | 0.83 | 0.69 | 94.84% |

The filter removes some bad 2026 S trades, but it also removes profitable metal-sector trades. After exit optimization, it is no longer helpful.

## Recommendation

This is currently the best strict rolling C/S/D candidate:

`all_priority_bias_min3 + conservative_tp_center15_profit_first`

It is strong enough to enter formal candidate status. The next engineering step is wiring daily execution and visualization to this exit-guard version, while keeping the older CSD+D40 result only as a comparison baseline.

