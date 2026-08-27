const fmtPct=v=>Number.isFinite(Number(v))?`${(Number(v)*100).toFixed(2)}%`:'—';
const fmtNum=v=>Number.isFinite(Number(v))?Number(v).toLocaleString(undefined,{maximumFractionDigits:2}):'—';
const signal=(name,row)=>`<div class="signal"><div class="label">${name}</div><div class="action ${row?.action||'HOLD'}">${row?.action||'—'}</div><small>${row?.symbol||''} ${row?.status&&row.status!=='OK'?`· ${row.status}`:''}</small></div>`;
const pill=(text,cls='')=>`<span class="pill ${cls}">${text}</span>`;

function renderCurve(price_curve){
  const points=price_curve?.points||[], markers=price_curve?.markers||[];
  if(points.length<2) return '<div class="curve empty">Forward price curve appears after at least two reconciled sessions.</div>';
  const width=520,height=150,pad=18;
  const values=points.flatMap(p=>[Number(p.price_1445),Number(p.close)]).filter(Number.isFinite);
  const min=Math.min(...values),max=Math.max(...values),span=Math.max(max-min,Math.abs(max||1)*0.005,1e-9);
  const x=i=>pad+(width-pad*2)*(i/(points.length-1));
  const y=v=>height-pad-(height-pad*2)*((Number(v)-min)/span);
  const path=points.map((p,i)=>`${i?'L':'M'} ${x(i).toFixed(1)} ${y(p.price_1445).toFixed(1)}`).join(' ');
  const closePath=points.map((p,i)=>`${i?'L':'M'} ${x(i).toFixed(1)} ${y(p.close).toFixed(1)}`).join(' ');
  const indexByDate=new Map(points.map((p,i)=>[p.date,i]));
  const markerSvg=markers.map(m=>{const i=indexByDate.get(m.date);if(i===undefined)return '';const px=x(i),py=y(points[i].price_1445);const glyph=m.action==='BUY'?'▲':'▼';return `<text x="${px.toFixed(1)}" y="${(py+(m.action==='BUY'?-7:15)).toFixed(1)}" text-anchor="middle" class="trade-marker ${m.source} ${m.action}">${glyph}</text>`;}).join('');
  return `<div class="curve-wrap"><div class="curve-title">${price_curve.symbol||''} · 14:45 vs close · markers are formal/shadow decisions</div><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="forward price curve"><path class="price-line close" d="${closePath}"/><path class="price-line preclose" d="${path}"/>${markerSvg}</svg><div class="curve-legend"><span>14:45</span><span>Close</span><span>▲ BUY / ▼ SELL</span></div></div>`;
}
function renderLine(line,d){
  const fm=d.forward_metrics?.formal||{}, anchor=d.forward_metrics?.v2_anchor||{};
  const daily=d.daily_drift||{}, perf=d.performance_drift||{}, promotion=d.promotion||{}, rollback=d.rollback||{};
  const candidates=(d.candidates||[]).slice(0,5).map(c=>`<div class="candidate"><div><b>${c.candidate_id}</b><br><small>${(c.gate?.reasons||[]).join(', ')||'gate pass'}</small></div><div>${c.gate?.passed?pill('PASS','ok'):pill('WAIT','warn')}</div></div>`).join('');
  return `<article class="card">
    <div class="card-head"><div class="line-name">${line}</div><div class="release">${d.formal_release}</div></div>
    <div class="signal-row">${signal('FORMAL',d.latest?.formal)}${signal('V2 ANCHOR',d.latest?.anchor)}${signal('SHADOW',d.latest?.shadow)}</div>
    <div class="metrics"><div class="metric"><b>${fmtPct(fm.win_rate)}</b><span>Forward win</span></div><div class="metric"><b>${fmtPct(fm.mean_return)}</b><span>Mean trade</span></div><div class="metric"><b>${fmtPct(fm.max_drawdown)}</b><span>Max DD</span></div><div class="metric"><b>${d.new_matured_samples??0}</b><span>New samples</span></div></div>
    <div class="status-row">${pill(`shadow: ${d.shadow_leader||'none'}`,'shadow')}${pill(`promotion: ${promotion.status||'UNKNOWN'}`,promotion.status==='PROMOTED'?'ok':'')}${pill(`rollback: ${rollback.status||'NONE'}`,rollback.status==='ROLLED_BACK'?'bad':'')}${pill(`daily drift: ${daily.severe?'ALERT':'OK'}`,daily.severe?'bad':'ok')}${pill(`perf drift: ${perf.severe?'ALERT':'OK'}`,perf.severe?'bad':'ok')}${d.pending_promotion?pill(`pending: ${d.pending_promotion}`,'warn'):''}</div>
    ${renderCurve(d.price_curve)}
    <div class="candidates">${candidates||'<small>No evaluated challenger yet.</small>'}</div>
  </article>`;
}
async function load(){
  const res=await fetch('data/dashboard.json',{cache:'no-store'}); if(!res.ok) throw new Error(`dashboard ${res.status}`);
  const data=await res.json(); document.getElementById('modeBadge').textContent=data.mode||'UNKNOWN';
  document.getElementById('updatedAt').textContent=data.generated_at||'';
  const unresolved=data.unreconciled_close?.symbols||[]; if(unresolved.length){const b=document.getElementById('globalBanner');b.classList.remove('hidden');b.textContent=`Unreconciled close: ${unresolved.join(', ')}`;}
  document.getElementById('lineGrid').innerHTML=['C','S','D','R'].map(line=>renderLine(line,data.lines?.[line]||{})).join('');
  const events=['C','S','D','R'].map(line=>({line,event:data.lines?.[line]?.last_transition})).filter(x=>x.event).map(x=>`<div class="event"><b>${x.line} · ${x.event.event}</b> ${x.event.from_release||''} → ${x.event.to_release||''}<br><small>${x.event.timestamp||''} · ${x.event.reason||''}</small></div>`).join('');
  document.getElementById('timeline').innerHTML=events||'<small>No promotion or rollback event yet.</small>';
}
load().catch(err=>{const b=document.getElementById('globalBanner');b.classList.remove('hidden');b.textContent=`Dashboard load failed: ${err.message}`;});
