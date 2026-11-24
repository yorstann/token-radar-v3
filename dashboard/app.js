// Token Launch Radar v3 — the dashboard's behaviour. Everything on this page is
// rendered from window.RADAR (data.js, written by the entry's program beside this
// page): the board, the fit report and the diagnostics are the program's own
// numbers, recomputed on every run. No fetch, no network.
'use strict';

const R = window.RADAR;
const F = R.fit;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function money(v) {
  if (v === null || v === undefined) return '—';
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  return `$${Math.round(v)}`;
}
function pct1(v) { return v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(1)}%`; }
function signed(v, d = 2) { return `${v > 0 ? '+' : ''}${v.toFixed(d)}`; }

// ------------------------------------------------------------ header + KPIs
$('m-version').textContent = R.model_version;
$('m-asof').textContent = R.as_of;
$('m-source').textContent = R.source;

const kpis = [
  { v: `${F.fit_n}`, plain: true, k: 'protocols fitted', s: `of ${F.priced_n} priced in the snapshot` },
  { v: F.cv.r2_log.toFixed(3), k: 'held-out R² (log)', s: '5-fold cross-validation' },
  { v: `${F.cv.median_ape.toFixed(1)}%`, k: 'held-out median error', s: '5-fold cross-validation' },
  { v: `${F.time.median_ape.toFixed(1)}%`, k: 'time-split median error', s: `train before ${F.time.split_date}` },
  { v: `${R.stats.board_size}`, plain: true, k: 'pre-token protocols ranked', s: `${R.stats.shadow_n} of them shadow-priced` },
  { v: `${R.stats.watchlist_covered}/${R.stats.watchlist_total}`, plain: true, k: 'house watchlist covered', s: `floor $${(R.floor_usd / 1e6).toFixed(0)}M TVL` },
];
$('kpis').innerHTML = kpis.map((k) => `<div class="kpi"><div class="v${k.plain ? ' plain' : ''}">${k.v}</div><div class="k">${k.k}</div><div class="s">${k.s}</div></div>`).join('');

// ------------------------------------------------------------ methodology
const keptLabels = F.coefficients.map((c) => c.label).join(', ') || 'none';
$('model-story').innerHTML =
  `The snapshot holds <b>${F.priced_n} protocols that already trade</b> with a positive market cap; ` +
  `<b>${F.fit_n}</b> of them also record a listing date and form the fitting set (the other ${F.dateless_n} are ` +
  `checked separately below). For each, the model explains <span class="mono">log(market cap)</span> with a ` +
  `<b>category offset</b> — the category's mean log(cap/TVL), shrunk toward the global mean with a pseudo-count of ` +
  `${F.shrinkage_k} — plus an intercept and whichever continuous terms earn their place. Four candidate terms were ` +
  `offered (log TVL, log age, log chain count, damped 7-day momentum); a term was <b>kept only if it lowered the ` +
  `held-out median error in both</b> the 5-fold cross-validation and the time split. Kept: <b>${esc(keptLabels)}</b>. ` +
  `Every validation refits offsets and coefficients on its training part alone, so no held-out protocol ever prices ` +
  `itself. The fit prices the typical protocol; the largest exchange-scale caps sit far above the fitted line, ` +
  `which is exactly why each board row carries the P10–P90 band the held-out residuals imply.`;

$('coef-table').innerHTML =
  `<tr><th>Term</th><th class="n">Coefficient</th></tr>` +
  `<tr><td>Category offset (per category, shrunk)</td><td class="n">see multiples</td></tr>` +
  `<tr><td>Intercept</td><td class="n">${F.intercept.toFixed(4)}</td></tr>` +
  F.coefficients.map((c) => `<tr><td>${esc(c.label)}</td><td class="n">${c.coef.toFixed(4)}</td></tr>`).join('') +
  `<tr><td>Global mean multiple (all priced)</td><td class="n">${F.global_multiple.toFixed(3)}× TVL</td></tr>`;

$('cat-table').innerHTML =
  `<tr><th>Category</th><th class="n">Fitted n</th><th class="n">Multiple</th></tr>` +
  F.categories.slice(0, 10).map((c) => `<tr><td>${esc(c.category)}</td><td class="n">${c.n}</td><td class="n">${c.multiple.toFixed(3)}×</td></tr>`).join('');
$('cat-note').textContent =
  `${F.categories.length} categories carry a fitted offset. ${F.board_fallback_categories} board categories have no priced peer in the snapshot and fall back to the global multiple — the model says so rather than inventing a peer set.`;

const apePair = (b, a) => {
  const d = a - b;
  return `${b.toFixed(1)}% → ${a.toFixed(1)}% <span class="${d < 0 ? 'delta-down' : 'delta-up'}">(${signed(d, 1)})</span>`;
};
$('term-table').innerHTML =
  `<tr><th>Candidate term</th><th>Verdict</th><th>CV median error</th><th>Time-split median error</th></tr>` +
  F.verdicts.map((v) => `<tr><td>${esc(v.label)}</td><td><span class="chip ${v.status}">${v.status.toUpperCase()}</span></td>` +
    `<td>${apePair(v.cv_ape_before, v.cv_ape_after)}</td><td>${apePair(v.time_ape_before, v.time_ape_after)}</td></tr>`).join('');

const valRow = (name, s, extra) => `<tr><td>${name}</td><td class="n">${s.n}</td><td class="n">${s.r2_log.toFixed(3)}</td><td class="n">${s.r2_usd.toFixed(3)}</td><td class="n">${s.median_ape.toFixed(1)}%</td>${extra || ''}</tr>`;
$('val-table').innerHTML =
  `<tr><th>Validation</th><th class="n">n</th><th class="n">R² log</th><th class="n">R² $</th><th class="n">Median error</th></tr>` +
  valRow('5-fold cross-validation (held out)', F.cv) +
  valRow(`Time split at ${F.time.split_date} (held out)`, F.time) +
  valRow('Date-less priced protocols (extra check)', F.dateless) +
  `<tr><td>In-sample (for reference only)</td><td class="n">${F.fit_n}</td><td class="n">${F.insample.r2_log.toFixed(3)}</td><td class="n">—</td><td class="n">${F.insample.median_ape.toFixed(1)}%</td></tr>`;

$('pva-table').innerHTML =
  `<tr><th>Protocol</th><th class="n">Actual cap</th><th class="n">Fitted</th><th class="n">Fitted / actual</th></tr>` +
  F.largest.map((r) => `<tr><td>${esc(r.name)}</td><td class="n">${money(r.actual)}</td><td class="n">${money(r.fitted)}</td><td class="n">${r.ratio < 0.1 ? r.ratio.toFixed(3) : r.ratio.toFixed(2)}×</td></tr>`).join('');
$('interval-note').textContent =
  `Each board row's band is exp(prediction + q) for q the 10th and 90th percentiles of the out-of-fold residuals (${F.interval.q10.toFixed(2)} … +${F.interval.q90.toFixed(2)} in log space). On the held-out folds that band held the actual cap ${(F.interval.coverage * 100).toFixed(0)}% of the time, against a nominal 80%.`;

// ------------------------------------------------------------ the board
$('board-sub').textContent = `${R.stats.board_size} pre-token protocols · implied value is the fitted estimate, band is its held-out error`;

const catSel = $('f-cat');
[...new Set(R.rows.map((r) => r.category || 'Uncategorized'))].sort().forEach((c) => {
  const o = document.createElement('option'); o.value = c; o.textContent = c; catSel.appendChild(o);
});
const chainSel = $('f-chain');
[...new Set(R.rows.flatMap((r) => r.chains))].sort().forEach((c) => {
  const o = document.createElement('option'); o.value = c; o.textContent = c; chainSel.appendChild(o);
});

const state = { shown: 60, detail: null };
function visibleRows() {
  const q = $('q').value.trim().toLowerCase();
  const cat = catSel.value, chain = chainSel.value;
  const watchOnly = $('f-watch').checked, noShadow = $('f-noshadow').checked;
  let rows = R.rows.filter((r) =>
    (q === '' || r.name.toLowerCase().includes(q) || r.slug.includes(q)) &&
    (cat === '' || (r.category || 'Uncategorized') === cat) &&
    (chain === '' || r.chains.includes(chain)) &&
    (!watchOnly || r.watchlist) && (!noShadow || !r.shadow));
  const sort = $('f-sort').value;
  const by = {
    rank: (a, b) => a.rank - b.rank,
    implied: (a, b) => b.implied_value_usd - a.implied_value_usd,
    tvl: (a, b) => b.tvl_usd - a.tvl_usd,
    change: (a, b) => (b.change_7d ?? -1e9) - (a.change_7d ?? -1e9),
    age: (a, b) => (b.age_days ?? -1) - (a.age_days ?? -1),
    signal: (a, b) => b.signal - a.signal,
  }[sort];
  return rows.sort(by);
}

function rowHtml(r) {
  const chips = `${r.watchlist ? ' <span class="chip watch">WATCHLIST</span>' : ''}${r.shadow ? ' <span class="chip shadow">SHADOW ×0.35</span>' : ''}`;
  const chg = r.change_7d === null ? '—' : `<span class="${r.change_7d >= 0 ? 'up' : 'down'}">${pct1(r.change_7d)}</span>`;
  return `<tr data-slug="${esc(r.slug)}" class="${state.detail === r.slug ? 'sel' : ''}">
    <td class="rank">${r.rank}</td>
    <td><span class="nm">${esc(r.name)}</span>${chips}<span class="slug">${esc(r.slug)}</span></td>
    <td>${esc(r.category || '—')}</td>
    <td><span class="chains">${esc(r.chains.slice(0, 3).join(', '))}${r.chains.length > 3 ? ` +${r.chains.length - 3}` : ''}</span></td>
    <td class="money">${money(r.tvl_usd)}</td>
    <td class="money">${chg}</td>
    <td class="money"><b>${money(r.implied_value_usd)}</b></td>
    <td class="band">${money(r.implied_low_usd)} – ${money(r.implied_high_usd)}</td>
    <td><div class="sig"><div class="bar"><div class="fill" style="width:${r.signal}%"></div></div><span class="val">${r.signal}</span></div></td>
  </tr>`;
}

function renderBoard() {
  const rows = visibleRows();
  $('rows').innerHTML = rows.slice(0, state.shown).map(rowHtml).join('');
  $('more').style.display = rows.length > state.shown ? '' : 'none';
  $('more').textContent = `Show more rows (${rows.length - state.shown} more)`;
  document.querySelectorAll('#rows tr').forEach((tr) => tr.addEventListener('click', () => openDetail(tr.dataset.slug, true)));
}

// ------------------------------------------------------------ detail view
const TERM_SHORT = { log_tvl: 'TVL scale', log_age: 'Age', log_chains: 'Chain breadth', momentum: 'Momentum' };
function factorBars(r) {
  const parts = r.model.parts;
  const items = [{ label: 'Category offset', v: parts.category }];
  for (const [t, v] of Object.entries(parts.terms)) items.push({ label: TERM_SHORT[t] || t, v });
  if (parts.shadow !== 0) items.push({ label: 'Shadow discount', v: parts.shadow });
  const scale = Math.max(0.5, ...items.map((i) => Math.abs(i.v)));
  return items.map((i) => {
    const w = Math.min(50, (Math.abs(i.v) / scale) * 50);
    const style = i.v >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
    return `<div class="fbar"><span class="fl">${esc(i.label)}</span><span class="ft"><span class="fv ${i.v >= 0 ? 'pos' : 'neg'}" style="${style}"></span></span><span class="fn">${signed(i.v)} log</span></div>`;
  }).join('');
}

function openDetail(slug, scroll) {
  const r = R.rows.find((x) => x.slug === slug);
  if (!r) return;
  state.detail = slug;
  const d = $('detail');
  d.hidden = false;
  d.innerHTML = `
    <h3>${esc(r.name)} ${r.watchlist ? '<span class="chip watch">WATCHLIST</span>' : ''} ${r.shadow ? '<span class="chip shadow">SHADOW ×0.35</span>' : ''}</h3>
    <div class="grid">
      <div>
        <div class="big">${money(r.implied_value_usd)}</div>
        <p class="kv">fitted implied value · band <b>${money(r.implied_low_usd)} – ${money(r.implied_high_usd)}</b> (P10–P90 of held-out residuals)<br>
        rank <b>#${r.rank}</b> of ${R.stats.board_size} · signal <b>${r.signal}</b> (its implied-value percentile on this board; context only, it never moves a rank)</p>
        <p class="kv">Category <b>${esc(r.category || '—')}</b> · fitted category multiple <b>${r.model.category_multiple.toFixed(3)}×</b> TVL<br>
        TVL <b>${money(r.tvl_usd)}</b> · 7d <b>${pct1(r.change_7d)}</b> · chains <b>${esc(r.chains.join(', ') || '—')}</b><br>
        Listed <b>${r.listed_at || 'not recorded'}</b> · age <b>${r.age_days === null ? 'unknown' : `${r.age_days} days`}</b></p>
        ${r.shadow ? `<p class="kv"><b>Shadow-priced:</b> ${esc(r.shadow_reason || '')}. The fitted value is discounted to 0.35× and the row is flagged, here and on the board.</p>` : ''}
        <h3 style="margin-top:14px">Where the price comes from (log space)</h3>
        ${factorBars(r)}
        <p class="note">Bars are each part's distance from the typical fitted protocol; the shared base (global ratio + intercept + terms at their medians) is ${signed(r.model.parts.base)} log. Parts sum to the row's predicted log cap, before the shadow discount where one applies.</p>
      </div>
      <div>
        <h3 style="margin-top:0">Farm plan</h3>
        <ol class="farm">${r.farm_steps.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>
      </div>
    </div>`;
  renderBoard();
  if (scroll) d.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ------------------------------------------------------------ live tokens
$('live-rows').innerHTML = R.live.slice(0, 12).map((r) => `<tr data-slug="">
  <td><span class="nm">${esc(r.name)}</span><span class="slug">${esc(r.slug)}</span></td>
  <td>${esc(r.category || '—')}</td>
  <td class="money">${money(r.mcap_usd)}</td>
  <td class="money">${money(r.tvl_usd)}</td>
  <td class="money">${r.change_7d === null ? '—' : `<span class="${r.change_7d >= 0 ? 'up' : 'down'}">${pct1(r.change_7d)}</span>`}</td></tr>`).join('');

// ------------------------------------------------------------ wiring + hash views
for (const id of ['q', 'f-cat', 'f-chain', 'f-watch', 'f-noshadow', 'f-sort']) {
  $(id).addEventListener('input', () => { state.shown = 60; renderBoard(); });
}
$('more').addEventListener('click', () => { state.shown += 120; renderBoard(); });

const hash = new URLSearchParams(location.hash.replace(/^#/, ''));
if (hash.get('watchlist') === '1') $('f-watch').checked = true;
renderBoard();
// Hash views (used for the entry's screenshots): jump after a beat, so the page
// has painted once before it moves.
const jump = (el) => setTimeout(() => el.scrollIntoView(), 350);
if (hash.get('detail')) { openDetail(hash.get('detail'), false); jump($('detail')); }
else if (hash.get('view') === 'board') jump($('board-panel'));
