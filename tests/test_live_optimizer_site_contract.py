from pathlib import Path

def test_static_site_reads_dashboard_json_and_has_four_line_cards():
    root=Path(__file__).resolve().parents[1]/'same_day_1445/live_optimizer/site';html=(root/'index.html').read_text(encoding='utf-8');js=(root/'app.js').read_text(encoding='utf-8');css=(root/'styles.css').read_text(encoding='utf-8');assert 'C / S / D / R' in html;assert 'data/dashboard.json' in js;assert 'promotion' in js and 'rollback' in js and 'daily_drift' in js;assert 'candidate_generator' not in js and 'promote_candidate' not in js;assert '--bg' in css
def test_static_site_renders_forward_price_curve_and_trade_markers():
    root=Path(__file__).resolve().parents[1]/'same_day_1445/live_optimizer/site';js=(root/'app.js').read_text(encoding='utf-8');assert 'price_curve' in js;assert 'renderCurve' in js;assert '<svg' in js;assert 'markers' in js
