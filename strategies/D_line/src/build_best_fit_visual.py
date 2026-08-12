from __future__ import annotations

"""Build the interactive strict-OOS extrema-fit chart for the conversation."""

import json
from pathlib import Path

import pandas as pd

from filter_extrema5_audit import HIGH_CANDIDATE, LOW_CANDIDATE, all_filters
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features


VIS = Path(r"C:\Users\10619\.codex\visualizations\2026\06\21\019ee960-4ad4-7931-9173-60478085fa8b")
TARGET = VIS / "best-extrema-fit.html"


def prediction_with_features(target: str, candidate: str, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(OUT / "online_holdout_signals_2020_2025.csv", encoding="utf-8-sig")
    frame = predictions.loc[(predictions["target"] == target) & (predictions["candidate"] == candidate)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    lookup = features.copy()
    lookup["date"] = pd.to_datetime(labels["date"]).to_numpy()
    return frame.merge(lookup, on="date", how="left", validate="one_to_one")


def main() -> None:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    low = prediction_with_features("low", LOW_CANDIDATE, features, labels)
    high = prediction_with_features("high", HIGH_CANDIDATE, features, labels)
    low["model"] = (low["rank"] >= 0.85) & (low["W_RSI6"] <= 45)
    filter_map = dict(all_filters("high"))
    decisions = pd.read_csv(OUT / "adaptive_filter_holdout_yearly.csv", encoding="utf-8-sig")
    decisions = decisions.loc[decisions["target"] == "high", ["year", "selected_filter"]]
    selected = dict(zip(decisions["year"], decisions["selected_filter"]))
    high["model"] = False
    for year, filter_name in selected.items():
        part = high["year"] == int(year)
        high.loc[part, "model"] = (high.loc[part, "rank"] >= 0.85) & filter_map[filter_name](high.loc[part]).fillna(False)
    joined = raw[["date", "close"]].copy()
    joined = joined.merge(labels, on="date", how="left")
    joined = joined.merge(low[["date", "model"]].rename(columns={"model": "low_model"}), on="date", how="left")
    joined = joined.merge(high[["date", "model"]].rename(columns={"model": "high_model"}), on="date", how="left")
    joined[["low", "high", "low_model", "high_model"]] = joined[["low", "high", "low_model", "high_model"]].fillna(False).astype(bool)
    years: dict[str, list[dict[str, object]]] = {}
    for year, frame in joined.loc[joined["date"].dt.year.between(2020, 2025)].groupby(joined["date"].dt.year):
        years[str(year)] = [
            {
                "d": row.date.strftime("%Y-%m-%d"),
                "p": round(float(row.close), 3),
                "ml": bool(row.low_model), "mh": bool(row.high_model),
                "al": bool(row.low), "ah": bool(row.high),
            }
            for row in frame.itertuples(index=False)
        ]
    payload = json.dumps(years, ensure_ascii=False, separators=(",", ":"))
    fragment = f'''<div id="best-extrema-fit" aria-label="红利ETF五日阶段高低点严格滚动拟合图">
  <div class="viz-grid">
    <section class="card viz-stat" aria-label="低点精确命中">
      <div class="text-muted">低点精确命中</div>
      <div class="viz-stat-value">33.3%</div>
      <div class="text-muted text-small">2020-2025 保留期 · 27 / 81</div>
    </section>
    <section class="card viz-stat" aria-label="高点精确命中">
      <div class="text-muted">高点精确命中</div>
      <div class="viz-stat-value">29.9%</div>
      <div class="text-muted text-small">2020-2025 保留期 · 38 / 127</div>
    </section>
  </div>
  <div class="viz-controls">
    <label class="form-label" for="fit-year">查看年份
      <select id="fit-year" class="form-select" aria-label="选择验证年份"></select>
    </label>
    <span id="fit-count" class="viz-badge" aria-live="polite"></span>
  </div>
  <div class="fit-chart-wrap">
    <canvas id="fit-chart" role="img" aria-label="红利ETF收盘价与模型阶段高低点候选图"></canvas>
  </div>
  <div class="viz-row text-small" aria-label="图例">
    <span><i class="legend-line"></i>真实收盘价</span>
    <span><i class="legend-dot legend-buy"></i>模型低点候选：买入</span>
    <span><i class="legend-square legend-sell"></i>模型高点候选：卖出</span>
    <span><i class="legend-hollow"></i>真实阶段点：空心标记</span>
  </div>
</div>
<style>
  #best-extrema-fit .viz-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom: 1rem; }}
  #best-extrema-fit .fit-chart-wrap {{ position: relative; min-height: 22rem; }}
  #best-extrema-fit canvas {{ max-width: 100%; }}
  #best-extrema-fit .legend-line, #best-extrema-fit .legend-dot, #best-extrema-fit .legend-square, #best-extrema-fit .legend-hollow {{ display: inline-block; vertical-align: middle; margin-right: .35rem; }}
  #best-extrema-fit .legend-line {{ width: 1.25rem; border-top: 2px solid var(--viz-series-1); }}
  #best-extrema-fit .legend-dot {{ width: .65rem; height: .65rem; border-radius: 50%; background: var(--viz-series-2); }}
  #best-extrema-fit .legend-square {{ width: .65rem; height: .65rem; background: var(--viz-series-4); }}
  #best-extrema-fit .legend-hollow {{ width: .58rem; height: .58rem; border: 1px solid var(--muted-foreground); border-radius: 50%; }}
  #best-extrema-fit .viz-row {{ justify-content: flex-start; gap: 1rem; }}
  @media (max-width: 420px) {{ #best-extrema-fit .viz-grid {{ grid-template-columns: 1fr; }} #best-extrema-fit .fit-chart-wrap {{ min-height: 18rem; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('best-extrema-fit');
  const series = {payload};
  const years = Object.keys(series).sort();
  const select = root.querySelector('#fit-year');
  const count = root.querySelector('#fit-count');
  years.forEach(year => {{ const option = document.createElement('option'); option.value = year; option.textContent = year + ' 年'; select.append(option); }});
  select.value = years.includes('2025') ? '2025' : years.at(-1);
  const style = getComputedStyle(root);
  const color = token => style.getPropertyValue(token).trim();
  const palette = {{ line: color('--viz-series-1'), buy: color('--viz-series-2'), sell: color('--viz-series-4'), actual: color('--muted-foreground'), grid: color('--border'), text: color('--muted-foreground') }};
  const ctx = root.querySelector('#fit-chart');
  const chart = new Chart(ctx, {{
    type: 'line', data: {{ labels: [], datasets: [] }},
    options: {{ responsive: true, maintainAspectRatio: false, animation: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
        title(items) {{ return items.length ? items[0].label : ''; }},
        label(item) {{ return item.dataset.label + '：' + (item.raw == null ? '' : Number(item.raw).toFixed(3)); }}
      }}, filter(item) {{ return item.raw != null; }} }} }},
      scales: {{ x: {{ ticks: {{ color: palette.text, maxTicksLimit: 7 }}, grid: {{ color: palette.grid }} }}, y: {{ ticks: {{ color: palette.text }}, grid: {{ color: palette.grid }}, title: {{ display: true, text: '收盘价', color: palette.text }} }} }}
    }}
  }});
  function points(rows, key) {{ return rows.map(row => row[key] ? row.p : null); }}
  function render() {{
    const rows = series[select.value];
    const lows = rows.filter(row => row.ml).length;
    const highs = rows.filter(row => row.mh).length;
    count.textContent = lows + ' 个买入候选 · ' + highs + ' 个卖出候选';
    chart.data.labels = rows.map(row => row.d.slice(5));
    chart.data.datasets = [
      {{ label: '真实收盘价', data: rows.map(row => row.p), borderColor: palette.line, backgroundColor: palette.line, borderWidth: 2, pointRadius: 0, tension: .12 }},
      {{ label: '模型低点候选（买入）', data: points(rows, 'ml'), showLine: false, pointRadius: 5, pointHoverRadius: 7, pointStyle: 'circle', pointBackgroundColor: palette.buy, pointBorderColor: palette.buy }},
      {{ label: '模型高点候选（卖出）', data: points(rows, 'mh'), showLine: false, pointRadius: 5, pointHoverRadius: 7, pointStyle: 'rect', pointBackgroundColor: palette.sell, pointBorderColor: palette.sell }},
      {{ label: '真实阶段低点', data: points(rows, 'al'), showLine: false, pointRadius: 4, pointHoverRadius: 5, pointStyle: 'circle', pointBackgroundColor: color('--background'), pointBorderColor: palette.actual, pointBorderWidth: 1.5 }},
      {{ label: '真实阶段高点', data: points(rows, 'ah'), showLine: false, pointRadius: 4, pointHoverRadius: 5, pointStyle: 'rect', pointBackgroundColor: color('--background'), pointBorderColor: palette.actual, pointBorderWidth: 1.5 }}
    ];
    chart.update();
  }}
  select.addEventListener('change', () => {{ chart.options.animation = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? false : {{ duration: 220 }}; render(); }});
  render();
}})();
</script>'''
    VIS.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(fragment, encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
