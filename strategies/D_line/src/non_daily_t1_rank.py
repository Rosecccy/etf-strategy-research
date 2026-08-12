from __future__ import annotations

"""Rank D-line trend gates after fixing non-daily signal execution timing.

Daily signals are unchanged.  For week / 15-day / month signals, the completed
period is dated on its final trading-day close and pair_result buys or sells at
the following trading-day close.  This is a T+1 execution without look-ahead.
"""

import json
from datetime import datetime

import pandas as pd

import factor_dca_scan as dca
import gate_nested_oos as gate_study


OUT = gate_study.OUT


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, groups = dca.load_panel(non_daily_signal_on_period_close=True)
    buy_conditions, buy_labels, buy_cores = dca.build_conditions(full, "buy")
    sell_conditions, sell_labels, sell_cores = dca.build_conditions(full, "sell")
    all_gates = gate_study.gates(full)

    summaries: list[dict] = []
    details: list[dict] = []
    for gate_key, gate in all_gates.items():
        print(f"Running corrected T+1 gate: {gate_key}")
        summary, rows = gate_study.run_gate(
            gate_key,
            gate,
            full,
            groups,
            buy_conditions,
            buy_labels,
            buy_cores,
            sell_conditions,
            sell_labels,
            sell_cores,
        )
        summaries.append(summary)
        details.extend(rows)
        print(json.dumps(summary, ensure_ascii=False))

    ranking = pd.DataFrame(summaries).sort_values(
        ["oos_win_rate", "oos_avg_return", "closed_batches"],
        ascending=[False, False, False],
    )
    yearly = pd.DataFrame(details)
    ranking.to_csv(OUT / "non_daily_t1_gate_ranking.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "non_daily_t1_gate_years.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Nested annual rolling selection. Only non-daily signal timing changes.",
        "execution": "A week/15-day/month bar is known after its final close; its T+1 order executes at the next trading-day close. Daily signals remain T+1 close.",
        "no_lookahead": True,
        "comparison": "Each gate keeps the same strict nested rolling selector. Ranking is by OOS win rate, then mean return, then closed lots.",
    }
    (OUT / "non_daily_t1_gate_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nCorrected non-daily T+1 ranking")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
