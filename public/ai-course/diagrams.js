/* =========================================================================
   DIAGRAMS
   Drawing grammar, applied to every figure:
     · ink hairlines, no drop shadows, no gradients
     · every box carries a concrete number or shape where one exists
     · numbered reading order  (1)(2)(3)  so the eye knows where to start
     · grey side-notes explain WHY, not just WHAT
   The target: you can read the concept off the figure without the prose.
   ========================================================================= */

const INK = '#14161C', BLUE = '#1B4DB1', MARK = '#FFE58A',
      GREY = '#5E6678', SOFT = '#F4F6FA', RED = '#B4441F', GREEN = '#1B6B4F';

function defs() {
  return `<defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,1 L9,5 L0,9 z" fill="${INK}"/></marker>
    <marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,1 L9,5 L0,9 z" fill="${BLUE}"/></marker>
    <marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,1 L9,5 L0,9 z" fill="${GREY}"/></marker>
  </defs>`;
}
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

/* a labelled box; label lines separated by | */
function box(x, y, w, h, label, o = {}) {
  const fill = o.fill || '#fff', stroke = o.stroke || INK, sw = o.sw || 1.5;
  const lines = String(label).split('|');
  let fs = o.fs || 13;
  // auto-fit: a label must never be wider than the box that holds it
  const widest = Math.max(...lines.map((l, i) => l.length * (i === 0 ? fs : fs - 1.5) * 0.602));
  if (widest > w - 14) fs = Math.max(8.5, fs * (w - 14) / widest);
  const startY = y + h/2 - (lines.length - 1) * (fs + 3) / 2 + fs/3;
  let t = lines.map((ln, i) =>
    `<text x="${x + w/2}" y="${startY + i*(fs+3)}" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="${i===0?fs:fs-1.5}" fill="${i===0?INK:GREY}">${esc(ln)}</text>`
  ).join('');
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="2" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>${t}`;
}
function arrow(x1, y1, x2, y2, o = {}) {
  const c = o.color || INK, m = o.color === BLUE ? 'ahb' : (o.color === GREY ? 'ahg' : 'ah');
  const dash = o.dash ? ` stroke-dasharray="${o.dash}"` : '';
  const path = o.curve
    ? `M${x1},${y1} C${x1 + o.curve},${y1} ${x2 - o.curve},${y2} ${x2},${y2}`
    : `M${x1},${y1} L${x2},${y2}`;
  let s = `<path d="${path}" fill="none" stroke="${c}" stroke-width="${o.sw||1.5}"${dash} marker-end="url(#${m})"/>`;
  if (o.label) {
    const mx = (x1+x2)/2, my = (y1+y2)/2 + (o.ly || -7);
    s += `<text x="${mx}" y="${my}" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11.5" fill="${c}">${esc(o.label)}</text>`;
  }
  return s;
}
function txt(x, y, s, o = {}) {
  return `<text x="${x}" y="${y}" text-anchor="${o.a||'start'}" font-family="${o.mono===false?"'Source Serif 4',serif":"'IBM Plex Mono',monospace"}" font-size="${o.fs||12}" fill="${o.fill||INK}" ${o.style?`font-style="${o.style}"`:''} ${o.w?`font-weight="${o.w}"`:''}>${esc(s)}</text>`;
}
/* grey side-note with a leader line */
function note(x, y, lines, o = {}) {
  const a = o.a || 'start';
  let s = lines.map((l, i) => txt(x, y + i*15, l, {a, fs: 11.5, fill: GREY})).join('');
  if (o.to) s += `<path d="${`M${o.from[0]},${o.from[1]} L${o.to[0]},${o.to[1]}`}" stroke="${GREY}" stroke-width="1" stroke-dasharray="2 3" fill="none"/>`;
  return s;
}
function step(x, y, n) {
  return `<circle cx="${x}" cy="${y}" r="10.5" fill="${BLUE}"/><text x="${x}" y="${y+4}" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="12" font-weight="600" fill="#fff">${n}</text>`;
}
function hl(x, y, w, h) { return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${MARK}" opacity="0.55"/>`; }
function wrap(h, inner) {
  return `<svg viewBox="0 0 900 ${h}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">${defs()}<rect width="900" height="${h}" fill="#fff"/>${inner}</svg>`;
}

const FIG = {

/* ------------------------------------------------------------ funnel */
funnel() {
  let s = '';
  const stages = [
    { x: 60,  w: 178, t: 'RETRIEVAL', sub: 'cheap model|dot product + ANN', n: '10⁹ → 10³', ms: '~5 ms' },
    { x: 306, w: 178, t: 'RANKING',   sub: 'expensive model|full cross features', n: '10³ → 50', ms: '~40 ms' },
    { x: 552, w: 178, t: 'RE-RANKING',sub: 'slate logic|dedup, diversity, rules', n: '50 → 10', ms: '~10 ms' }
  ];
  s += txt(60, 32, 'ONE REQUEST · TOTAL BUDGET ≈ 100 ms', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

  stages.forEach((st, i) => {
    s += box(st.x, 90, st.w, 78, `${st.t}|${st.sub}`, { fill: i === 0 ? SOFT : '#fff' });
    s += step(st.x + 12, 82, i + 1);
    s += txt(st.x + st.w/2, 186, st.n, { a: 'middle', fs: 13, w: 600 });
    s += txt(st.x + st.w/2, 203, st.ms, { a: 'middle', fs: 11.5, fill: GREY });
    if (i < 2) s += arrow(st.x + st.w + 8, 129, stages[i+1].x - 8, 129);
  });
  s += box(786, 90, 64, 78, 'USER|sees 10', { fill: MARK });
  s += arrow(738, 129, 778, 129);

  // the constraint, stated plainly
  s += `<line x1="60" y1="240" x2="840" y2="240" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += txt(60, 266, 'cost_per_item  ×  n_items   ≤   latency_budget', { fs: 15, w: 600, fill: BLUE });
  s += txt(60, 286, 'each stage may be slower only because n is smaller. n is the only variable you control.', { fs: 12, fill: GREY });

  // failure annotation
  s += hl(60, 314, 208, 20);
  s += txt(64, 329, 'WHERE LAUNCHES DIE', { fs: 12, w: 600 });
  s += txt(60, 354, 'stage 2 can only reorder what stage 1 hands it.', { fs: 12.5 });
  s += txt(60, 374, "if the ranker's favourites are never candidates, an offline gain has nowhere to land.", { fs: 12, fill: GREY });
  return wrap(396, s);
},

/* ---------------------------------------------------------- features */
features() {
  let s = '';
  s += txt(60, 30, 'CLASSICAL DEEP RECOMMENDER (DLRM-family)', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  // sparse tables
  const tx = 60;
  s += txt(tx, 70, 'sparse categorical  →  embedding lookup', { fs: 12, fill: GREY });
  for (let i = 0; i < 5; i++) {
    s += box(tx + i*72, 82, 62, 46, `id ${i+1}|table`, { fill: SOFT, fs: 11.5 });
  }
  s += step(48, 82, 1);
  s += hl(tx, 136, 350, 18);
  s += txt(tx + 4, 150, '≈ 95% of all parameters live here', { fs: 12, w: 600 });

  // dense
  s += txt(500, 70, 'dense numeric', { fs: 12, fill: GREY });
  s += box(500, 82, 130, 46, 'counts, price|age, ctr', { fill: '#fff', fs: 11.5 });

  // interaction
  s += box(250, 200, 240, 44, 'EXPLICIT INTERACTION  (dot / cross / FM)', { fill: '#fff', fs: 12 });
  s += step(238, 196, 2);
  for (let i = 0; i < 5; i++) s += arrow(tx + i*72 + 31, 160, 340, 196, { color: GREY, sw: 1 });
  s += arrow(565, 128, 420, 196, { color: GREY, sw: 1 });

  // mlp
  s += box(300, 286, 140, 40, 'MLP  2–4 layers', { fill: '#fff', fs: 12 });
  s += step(288, 282, 3);
  s += arrow(370, 244, 370, 282, { color: BLUE });
  s += box(320, 366, 100, 34, 'p(click)', { fill: MARK, fs: 12.5 });
  s += arrow(370, 326, 370, 362, { color: BLUE });

  s += note(560, 214, [
    'THE SHAPE THAT MATTERS:',
    'huge in memory  (lookup tables)',
    'tiny in compute (a few layers)',
    '',
    '→ adding GPUs runs the same shallow',
    '   computation faster, and buys',
    '   almost no extra quality.',
    '',
    'Language models have the opposite',
    'shape, which is why they kept scaling.'
  ], { from: [556, 218], to: [494, 222] });
  return wrap(420, s);
},

/* ------------------------------------------------------------- embed */
embed() {
  let s = '';
  s += txt(60, 30, 'TWO-TOWER RETRIEVAL', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  s += box(60, 80, 150, 60, 'USER TOWER|history, context', { fill: '#fff' });
  s += box(60, 160, 150, 34, 'u ∈ ℝ²⁵⁶', { fill: SOFT, fs: 12 });
  s += arrow(135, 140, 135, 156, { color: BLUE });
  s += step(48, 74, 1);

  s += box(690, 80, 150, 60, 'ITEM TOWER|title, category, price', { fill: '#fff' });
  s += box(690, 160, 150, 34, 'v ∈ ℝ²⁵⁶', { fill: SOFT, fs: 12 });
  s += arrow(765, 140, 765, 156, { color: BLUE });
  s += step(678, 74, 2);

  // shared space
  const cx = 450, cy = 210, R = 78;
  s += `<circle cx="${cx}" cy="${cy}" r="${R}" fill="#fff" stroke="${INK}" stroke-width="1.5"/>`;
  const pts = [[-52,-30],[-30,18],[-8,-46],[12,32],[38,-14],[54,26],[-46,44],[24,-58],[62,-40],[-16,60]];
  pts.forEach((p, i) => {
    s += `<circle cx="${cx+p[0]}" cy="${cy+p[1]}" r="3.4" fill="${i<3?BLUE:GREY}"/>`;
  });
  s += `<circle cx="${cx-20}" cy="${cy-6}" r="5.5" fill="${RED}"/>`;
  s += txt(cx - 12, cy - 12, 'q', { fs: 12, fill: RED, w: 600 });
  [[-52,-30],[-30,18],[-8,-46]].forEach(p => {
    s += `<line x1="${cx-20}" y1="${cy-6}" x2="${cx+p[0]}" y2="${cy+p[1]}" stroke="${BLUE}" stroke-width="1"/>`;
  });
  s += txt(cx, cy + R + 22, 'shared space · score = u · v', { a: 'middle', fs: 12.5, w: 600 });

  s += arrow(212, 177, 372, 205, { color: GREY, sw: 1.2, curve: 60 });
  s += arrow(688, 177, 528, 205, { color: GREY, sw: 1.2, curve: -60 });

  s += hl(60, 330, 578, 20);
  s += txt(64, 345, 'the towers never meet before the dot product — that is the whole point', { fs: 12.5, w: 600 });
  s += note(60, 378, [
    'BECAUSE the item tower cannot see the user:',
    '   · every item vector is computed once, offline',
    '   · they go into an ANN index (next figure)',
    '   · query time = 1 user forward pass + 1 index lookup',
    '',
    'Any user–item interaction inside the encoder would force',
    'scoring the entire corpus per request — the thing retrieval exists to avoid.'
  ]);
  return wrap(486, s);
},

/* --------------------------------------------------------------- ann */
ann() {
  let s = '';
  s += txt(60, 30, 'HNSW · GRAPH SEARCH INSTEAD OF SCANNING', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  const layers = [
    { y: 90,  label: 'layer 2 · sparse, long edges', pts: [[120,0],[330,10],[560,-6],[760,6]] },
    { y: 190, label: 'layer 1 · medium',             pts: [[120,0],[230,12],[330,-8],[450,10],[560,-4],[660,8],[760,0]] },
    { y: 300, label: 'layer 0 · every point, short edges', pts: [[110,0],[170,10],[230,-8],[290,6],[350,12],[410,-6],[470,8],[530,-10],[590,4],[650,10],[710,-6],[770,6]] }
  ];
  layers.forEach((L, li) => {
    s += txt(60, L.y - 24, L.label, { fs: 11.5, fill: GREY });
    for (let i = 0; i < L.pts.length - 1; i++) {
      s += `<line x1="${L.pts[i][0]}" y1="${L.y+L.pts[i][1]}" x2="${L.pts[i+1][0]}" y2="${L.y+L.pts[i+1][1]}" stroke="${GREY}" stroke-width="1"/>`;
    }
    L.pts.forEach(p => { s += `<circle cx="${p[0]}" cy="${L.y+p[1]}" r="3.6" fill="#fff" stroke="${INK}" stroke-width="1.3"/>`; });
  });

  // the search path
  const path = [[120,90],[330,200],[350,312],[410,294],[470,308]];
  for (let i = 0; i < path.length - 1; i++) {
    s += arrow(path[i][0], path[i][1], path[i+1][0], path[i+1][1], { color: BLUE, sw: 2 });
  }
  path.forEach((p, i) => { if (i < path.length-1) s += `<circle cx="${p[0]}" cy="${p[1]}" r="5" fill="${BLUE}"/>`; });
  s += `<circle cx="470" cy="308" r="7" fill="${MARK}" stroke="${INK}" stroke-width="1.5"/>`;
  s += txt(486, 334, 'nearest neighbour found', { fs: 12, w: 600 });
  s += step(96, 84, 1);
  s += txt(60, 356, 'enter at the top, hop greedily toward the query, descend a layer, repeat', { fs: 12, fill: GREY });

  s += `<line x1="60" y1="386" x2="840" y2="386" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  const dials = [
    ['M', 'edges per node', 'memory + graph quality', 'build'],
    ['efConstruction', 'build thoroughness', 'index quality', 'build'],
    ['efSearch', 'frontier width', 'recall ↔ latency', 'RUNTIME DIAL']
  ];
  dials.forEach((d, i) => {
    const x = 60 + i * 265;
    s += box(x, 408, 240, 62, `${d[0]}|${d[1]} → ${d[2]}`, { fill: i === 2 ? MARK : '#fff', fs: 12.5 });
    if (i === 2) s += txt(x + 120, 488, d[3], { a: 'middle', fs: 11.5, w: 600, fill: BLUE });
  });
  s += txt(60, 512, 'exact search = 10⁹ distance computations.  graph search ≈ 50 hops.  cost: occasionally you miss the true nearest.', { fs: 12, fill: GREY });
  return wrap(534, s);
},

/* --------------------------------------------------------- attention */
attention() {
  let s = '';
  s += txt(60, 30, 'CAUSAL SELF-ATTENTION · WHY GENERATION IS POSSIBLE', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  const toks = ['the', 'cat', 'sat', 'on', 'the', '?'];
  const bw = 88, x0 = 90, y = 260;
  toks.forEach((t, i) => {
    s += box(x0 + i*bw, y, bw - 14, 40, t, { fill: i === 5 ? MARK : '#fff', fs: 14 });
    s += txt(x0 + i*bw + (bw-14)/2, y + 58, `t${i+1}`, { a: 'middle', fs: 11, fill: GREY });
  });
  // attention arcs into position 6
  const tgt = x0 + 5*bw + (bw-14)/2;
  for (let j = 0; j < 5; j++) {
    const src = x0 + j*bw + (bw-14)/2;
    const h = 55 + (5-j)*24;
    s += `<path d="M${src},${y} C${src},${y-h} ${tgt},${y-h} ${tgt},${y}" fill="none" stroke="${BLUE}" stroke-width="${1.6 - j*0.15}" opacity="${0.35 + j*0.13}"/>`;
  }
  s += step(x0 - 22, y + 20, 1);
  s += txt(60, 72, 'position 6 reads positions 1–6.  thickness ≈ attention weight.', { fs: 12, fill: GREY });

  // the mask
  const mx = 646, my = 110, c = 26;
  s += txt(mx, my - 14, 'the mask', { fs: 12, w: 600 });
  for (let r = 0; r < 6; r++) for (let k = 0; k < 6; k++) {
    const allowed = k <= r;
    s += `<rect x="${mx + k*c}" y="${my + r*c}" width="${c-2}" height="${c-2}" fill="${allowed ? BLUE : '#fff'}" opacity="${allowed ? 0.18 + 0.1*(1-(r-k)/6) : 1}" stroke="${GREY}" stroke-width="0.8"/>`;
  }
  s += txt(mx + 6*c + 10, my + 30, 'allowed', { fs: 11.5, fill: BLUE });
  s += txt(mx + 6*c + 10, my + 100, 'blocked', { fs: 11.5, fill: GREY });
  s += txt(mx - 8, my + 3*c, 'i', { a: 'end', fs: 12, fill: GREY });
  s += txt(mx + 3*c, my - 26, 'j', { a: 'middle', fs: 12, fill: GREY });

  s += note(60, 360, [
    'softmax( Q Kᵀ / √d ) V',
    '',
    'Q, K, V are linear projections of the',
    'same sequence — hence "self".',
    '',
    'Two consequences to carry forward:',
    '  · softmax normalizes → relative only,',
    '    magnitude is discarded  (see 3.2)',
    '  · cost grows with length²  → the whole',
    '    long-context research field'
  ]);

  s += hl(60, 516, 570, 20);
  s += txt(64, 531, 'without the mask, predicting t6 is trivial: just read t6 from the input.', { fs: 12.5, w: 600 });
  s += txt(60, 558, 'The mask also matches inference, where future tokens genuinely do not exist yet.', { fs: 12, fill: GREY });
  return wrap(580, s);
},

/* ------------------------------------------------------------ seqrec */
seqrec() {
  let s = '';
  s += txt(60, 30, 'BAG OF FEATURES  →  SEQUENCE OF ACTIONS', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  // before
  s += txt(60, 70, 'BEFORE · order thrown away', { fs: 12, fill: GREY });
  ['likes comedy', 'age bracket', 'watches at night', 'device: phone'].forEach((t, i) => {
    s += box(60 + i*100, 84, 92, 36, t, { fill: '#fff', fs: 10.5 });
  });
  s += box(470, 84, 90, 36, 'MLP', { fill: SOFT, fs: 12 });
  s += arrow(456, 102, 466, 102, { color: GREY, sw: 1 });
  s += txt(576, 106, 'no notion of "what happened just now"', { fs: 11.5, fill: GREY });

  // after
  s += txt(60, 172, 'AFTER · one interleaved causal stream', { fs: 12, fill: GREY });
  const seq = [['item','A'],['act','watch 92%'],['item','B'],['act','skip'],['item','C'],['act','buy'],['?','next']];
  const bw = 108, y = 240;
  seq.forEach((t, i) => {
    const isQ = t[0] === '?';
    s += box(60 + i*bw, y, bw - 12, 46, `${t[0]}|${t[1]}`, { fill: isQ ? MARK : (t[0] === 'act' ? SOFT : '#fff'), fs: 11.5 });
  });
  s += step(48, y + 23, 1);
  const tgt = 60 + 6*bw + (bw-12)/2;
  for (let j = 0; j < 6; j++) {
    const src = 60 + j*bw + (bw-12)/2;
    s += `<path d="M${src},${y} C${src},${y-40-(6-j)*6} ${tgt},${y-40-(6-j)*6} ${tgt},${y}" fill="none" stroke="${BLUE}" stroke-width="1.2" opacity="${0.3+j*0.1}"/>`;
  }

  s += hl(60, 322, 560, 20);
  s += txt(64, 337, 'same machinery as a language model — the tokens are behaviour, not words', { fs: 12.5, w: 600 });

  // scaling panel
  s += `<line x1="60" y1="370" x2="840" y2="370" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  const gx = 60, gy = 524, gw = 300, gh = 120;
  s += `<line x1="${gx}" y1="${gy-gh}" x2="${gx}" y2="${gy}" stroke="${INK}" stroke-width="1.2"/>`;
  s += `<line x1="${gx}" y1="${gy}" x2="${gx+gw}" y2="${gy}" stroke="${INK}" stroke-width="1.2"/>`;
  let dl = '', gl = '';
  for (let i = 0; i <= 40; i++) {
    const px = gx + gw*i/40;
    dl += `${i?'L':'M'}${px},${gy - 26 - Math.log(1+i/9)*9} `;
    gl += `${i?'L':'M'}${px},${gy - 12 - Math.pow(i/40, 0.72)*(gh-22)} `;
  }
  s += `<path d="${dl}" fill="none" stroke="${GREY}" stroke-width="1.6" stroke-dasharray="4 3"/>`;
  s += `<path d="${gl}" fill="none" stroke="${BLUE}" stroke-width="2.2"/>`;
  s += txt(gx + gw + 8, gy - 118, 'generative', { fs: 11.5, fill: BLUE });
  s += txt(gx + gw + 8, gy - 46, 'DLRM', { fs: 11.5, fill: GREY });
  s += txt(gx + gw/2, gy + 20, 'training compute (log)', { a: 'middle', fs: 11.5, fill: GREY });
  s += txt(gx - 8, gy - gh - 6, 'quality', { fs: 11.5, fill: GREY });

  s += note(420, 426, [
    'HSTU  (Actions Speak Louder than Words, ICML 2024)',
    '',
    '· pointwise aggregated attention, not softmax',
    '  — engagement intensity is signal; softmax throws it away',
    '· relative attention bias over time and position',
    '· heterogeneous features collapsed into one token stream',
    '',
    'reported:  +65.8% NDCG over baselines',
    '           5.3–15.2× faster at 8192-length sequences',
    '           +12.4% online A/B  ·  1.5T parameters',
    '',
    'The scaling curve on the left is the real claim:',
    'compute becomes a reliable lever, as it already was for LLMs.'
  ]);
  return wrap(632, s);
},

/* --------------------------------------------------------------- sid */
sid() {
  let s = '';
  s += txt(60, 30, 'RESIDUAL QUANTIZATION  →  SEMANTIC ID', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  s += box(60, 96, 130, 52, 'item text|"red running shoe"', { fill: '#fff', fs: 11.5 });
  s += arrow(190, 122, 224, 122, { color: BLUE });
  s += box(224, 96, 110, 52, 'encoder|frozen', { fill: SOFT, fs: 11.5 });
  s += arrow(334, 122, 368, 122, { color: BLUE });
  s += box(368, 106, 92, 32, 'x ∈ ℝ⁷⁶⁸', { fill: '#fff', fs: 12 });
  s += step(48, 90, 1);

  const lv = [
    { x: 120, code: 12, res: '‖r₁‖ = 0.42‖x‖', label: 'codebook 1 · 256 codes', mean: 'broad category' },
    { x: 380, code: 7,  res: '‖r₂‖ = 0.17‖x‖', label: 'codebook 2 · 256 codes', mean: 'sub-type' },
    { x: 640, code: 3,  res: '‖r₃‖ = 0.06‖x‖', label: 'codebook 3 · 256 codes', mean: 'fine detail' }
  ];
  const ly = 208;
  lv.forEach((L, i) => {
    s += box(L.x, ly, 180, 96, '', { fill: '#fff' });
    s += txt(L.x + 172, ly + 14, L.label, { fs: 11.5, fill: GREY, a: 'end' });
    for (let k = 0; k < 4; k++) {
      const sel = k === [1, 3, 2][i];
      s += `<circle cx="${L.x + 26}" cy="${ly + 22 + k*20}" r="${sel?6:3.4}" fill="${sel?BLUE:'#DDE1EA'}"/>`;
      if (sel) s += txt(L.x + 42, ly + 26 + k*20, `code ${L.code}`, { fs: 12, w: 600, fill: BLUE });
    }
    s += txt(L.x + 10, ly + 116, `input: ${i === 0 ? 'x' : 'r' + i}   →   residual ${'r' + (i+1)}`, { fs: 11.5 });
    s += txt(L.x + 10, ly + 132, L.res, { fs: 11.5, fill: GREY });
    s += txt(L.x + 10, ly + 152, L.mean, { fs: 11.5, fill: BLUE, w: 600 });
    s += step(L.x - 12, ly - 4, i + 2);
    if (i < 2) s += arrow(L.x + 180, ly + 48, lv[i+1].x - 4, ly + 48, { color: INK, label: 'subtract' , ly: -8});
  });
  s += arrow(400, 140, 150, 202, { color: BLUE, sw: 1.2 });

  s += box(330, 404, 240, 40, 'semantic ID  =  (12, 7, 3)', { fill: MARK, fs: 14 });

  s += hl(60, 476, 710, 20);
  s += txt(64, 491, 'level k encodes only what levels 1..k−1 missed — that is why a shared prefix means something', { fs: 12.5, w: 600 });
  s += note(60, 520, [
    'CONSEQUENCE:  items (12,7,*) are all running shoes — the * varies only the fine detail.',
    'A brand-new item lands beside its neighbours and inherits their statistical strength,',
    'instead of starting from a random vector with no history.',
    '',
    'WATCH FOR:  codebook collapse (few codes absorbing everything — track utilization),',
    'and encoder drift silently invalidating every ID when the content model is retrained.'
  ]);
  return wrap(632, s);
},

/* --------------------------------------------------------------- rag */
rag() {
  let s = '';
  s += txt(60, 30, 'RETRIEVAL-AUGMENTED GENERATION', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  s += box(60, 96, 120, 48, 'question', { fill: '#fff' });
  s += step(48, 90, 1);
  s += arrow(180, 120, 206, 120, { color: BLUE });
  s += box(210, 96, 130, 48, 'retriever', { fill: SOFT, fs: 12.5 });
  s += step(198, 90, 2);
  s += box(210, 176, 130, 60, 'document store|updated, versioned', { fill: '#fff', fs: 11.5 });
  s += arrow(275, 172, 275, 148, { color: GREY, sw: 1.2 });

  s += arrow(340, 120, 446, 120, { color: BLUE, label: 'top-k passages', ly: -12 });
  s += box(450, 88, 150, 64, 'context window|question + passages', { fill: '#fff', fs: 11.5 });
  s += step(438, 82, 3);
  s += arrow(600, 120, 636, 120, { color: BLUE });
  s += box(640, 96, 110, 48, 'model', { fill: SOFT, fs: 12.5 });
  s += arrow(750, 120, 782, 120, { color: BLUE });
  s += box(786, 88, 64, 64, 'answer|+ cite', { fill: MARK, fs: 11.5 });
  s += step(774, 82, 4);

  s += `<line x1="60" y1="278" x2="840" y2="278" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += txt(60, 306, 'THE DECISION RULE', { fs: 12, w: 600, fill: BLUE });
  s += box(60, 322, 370, 52, 'FINE-TUNING  changes behaviour, tone, format', { fill: '#fff', fs: 12.5 });
  s += box(460, 322, 380, 52, 'RETRIEVAL  changes what the model can see right now', { fill: MARK, fs: 12.5 });
  s += txt(60, 400, 'Needs to reflect a document updated this morning?  Retrieval.', { fs: 12, fill: GREY });
  s += txt(60, 418, 'Needs to cite its source?  Retrieval — a fine-tuned fact has no address.', { fs: 12, fill: GREY });
  s += txt(60, 436, 'Needs to always answer in the same JSON shape?  Fine-tuning.', { fs: 12, fill: GREY });

  s += note(60, 470, [
    'WHY "just don\'t make things up" fails: the model has no internal signal separating',
    '"I read this" from "this pattern fits". An instruction to verify has nothing to verify against.',
    'RAG supplies the thing to check against — and the citation that lets a human check it too.'
  ]);
  return wrap(536, s);
},

/* ------------------------------------------------------------ hybrid */
hybrid() {
  let s = '';
  s += txt(60, 30, 'HYBRID RETRIEVAL + RECIPROCAL RANK FUSION', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  s += box(60, 150, 120, 46, 'query', { fill: '#fff' });
  s += step(48, 144, 1);
  s += arrow(180, 160, 236, 100, { color: BLUE });
  s += arrow(180, 186, 236, 244, { color: BLUE });
  s += box(236, 76, 150, 48, 'BM25  lexical', { fill: SOFT, fs: 12.5 });
  s += box(236, 220, 150, 48, 'dense  ANN', { fill: SOFT, fs: 12.5 });
  s += txt(236, 66, 'exact rare tokens', { fs: 11, fill: GREY });
  s += txt(236, 288, 'paraphrase, meaning', { fs: 11, fill: GREY });

  // ranked lists with incompatible scores
  const listA = [['doc-91', '18.4'], ['doc-12', '11.2'], ['doc-55', '9.7']];
  const listB = [['doc-12', '0.83'], ['doc-77', '0.81'], ['doc-91', '0.78']];
  listA.forEach((d, i) => { s += box(400, 76 + i*30, 130, 26, `${i+1}. ${d[0]}   ${d[1]}`, { fill: '#fff', fs: 11.5 }); });
  listB.forEach((d, i) => { s += box(400, 220 + i*30, 130, 26, `${i+1}. ${d[0]}   ${d[1]}`, { fill: '#fff', fs: 11.5 }); });
  s += arrow(386, 100, 396, 100, { color: GREY, sw: 1 });
  s += arrow(386, 244, 396, 244, { color: GREY, sw: 1 });

  s += hl(236, 172, 300, 20);
  s += txt(240, 187, 'scores 18.4 vs 0.83 — incomparable scales', { fs: 12, w: 600 });

  s += arrow(534, 106, 592, 160, { color: BLUE });
  s += arrow(534, 250, 592, 190, { color: BLUE });
  s += box(592, 148, 150, 52, 'RRF', { fill: MARK, fs: 13 });
  s += txt(592, 218, 'Σ 1/(k + rankᵢ),  k=60', { fs: 11.5, fill: BLUE });
  s += step(580, 142, 2);
  s += arrow(742, 174, 786, 174, { color: BLUE });
  s += box(786, 148, 64, 52, 'fused|list', { fill: '#fff', fs: 11.5 });

  s += `<line x1="60" y1="326" x2="840" y2="326" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += txt(60, 354, 'doc-91:  1/(60+1) + 1/(60+3)  =  0.0323', { fs: 12.5 });
  s += txt(60, 374, 'doc-12:  1/(60+2) + 1/(60+1)  =  0.0325   ← wins, and no score scale was ever compared', { fs: 12.5, fill: BLUE });

  s += note(60, 412, [
    'THEN RERANK.  A cross-encoder reads query and document together and is far more accurate —',
    'but it has no independent document representation, so it cannot be indexed. Cost is linear in candidates.',
    'Hence the same funnel as part 1:  retrieve 50–200 cheaply  →  fuse  →  cross-encode  →  keep 5–15.'
  ]);
  return wrap(478, s);
},

/* ------------------------------------------------------------- chunk */
chunk() {
  let s = '';
  s += txt(60, 30, 'THE SAME PASSAGE, STORED TWO WAYS', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  s += txt(60, 76, 'A · naive split every 40 words', { fs: 12, fill: GREY });
  s += box(60, 90, 360, 96, '', { fill: '#fff' });
  s += txt(76, 118, '"This applies only after two full', { fs: 12.5, mono: false });
  s += txt(76, 140, 'years of service."', { fs: 12.5, mono: false });
  s += txt(60, 210, 'query: "can I carry over unused leave?"', { fs: 11.5, fill: GREY });
  s += box(60, 224, 360, 34, 'NOT RETRIEVED', { fill: '#fff', stroke: RED, fs: 12.5 });
  s += txt(60, 282, 'nothing in the chunk says what "This" is —', { fs: 11.5, fill: GREY });
  s += txt(60, 298, 'the leave rule it belongs to fell on the other', { fs: 11.5, fill: GREY });
  s += txt(60, 314, 'side of the cut. Intact words, unfindable fact.', { fs: 11.5, fill: GREY });

  s += `<line x1="450" y1="76" x2="450" y2="330" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;

  s += txt(480, 76, 'B · semantic unit + situating line + metadata', { fs: 12, fill: GREY });
  s += box(480, 90, 360, 96, '', { fill: '#fff' });
  s += hl(492, 100, 336, 34);
  s += txt(496, 114, 'Staff handbook, annual leave:', { fs: 11.5, mono: false, w: 600 });
  s += txt(496, 130, 'carry-over eligibility rules.', { fs: 11.5, mono: false, w: 600 });
  s += txt(496, 152, '"This applies only after two full', { fs: 12.5, mono: false });
  s += txt(496, 168, 'years of service."', { fs: 12.5, mono: false });
  s += txt(496, 182, 'source: staff-handbook.md §Leave ¶2', { fs: 10.5, fill: BLUE });
  s += box(480, 224, 360, 34, 'RETRIEVED  ·  AND CITABLE', { fill: MARK, stroke: GREEN, fs: 12.5 });
  s += txt(480, 282, 'the added line is generated in bulk by a small', { fs: 11.5, fill: GREY });
  s += txt(480, 298, 'model at ingest time. It is usually the single', { fs: 11.5, fill: GREY });
  s += txt(480, 314, 'largest retrieval-quality gain available.', { fs: 11.5, fill: GREY });

  s += `<line x1="60" y1="352" x2="840" y2="352" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += txt(60, 380, 'RULE: chunk on semantic units — a section, a function, a clause.', { fs: 12.5, w: 600 });
  s += txt(60, 400, 'A chunk should be a complete thought with a citable address.', { fs: 12.5, w: 600 });
  s += txt(60, 424, 'Almost every "the LLM is bad" complaint in a RAG system turns out', { fs: 12, fill: GREY });
  s += txt(60, 440, 'to be a retrieval bug wearing a costume.', { fs: 12, fill: GREY });
  return wrap(462, s);
},

/* ------------------------------------------------------------- agent */
agent() {
  let s = '';
  s += txt(60, 30, 'THE AGENT LOOP, WITH ITS RAILS', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  const cx = 252, cy = 226, R = 104;
  const P = (deg, r) => [cx + Math.cos(deg*Math.PI/180)*r, cy + Math.sin(deg*Math.PI/180)*r];
  // arcs first, so the boxes sit on top of them
  [[-90, 30], [30, 150], [150, 270]].forEach(([a1, a2]) => {
    const [x1, y1] = P(a1 + 34, R), [x2, y2] = P(a2 - 34, R);
    s += `<path d="M${x1},${y1} A${R},${R} 0 0 1 ${x2},${y2}" fill="none" stroke="${BLUE}" stroke-width="1.8" marker-end="url(#ahb)"/>`;
  });
  const nodes = [
    { a: -90, t: 'THINK',   sub: 'what next?' },
    { a: 30,  t: 'ACT',     sub: 'call a tool' },
    { a: 150, t: 'OBSERVE', sub: 'read the result' }
  ];
  nodes.forEach((n, i) => {
    const [x, y] = P(n.a, R);
    s += box(x - 68, y - 25, 136, 50, `${n.t}|${n.sub}`, { fill: i === 0 ? SOFT : '#fff', fs: 12.5 });
  });
  s += txt(cx, cy + 4, 'repeat until done', { a: 'middle', fs: 11.5, fill: GREY });
  s += txt(cx, cy + 20, 'or until a cap fires', { a: 'middle', fs: 11.5, fill: GREY });
  s += step(cx - 96, cy - 116, 1);

  // rails
  const rails = [
    ['max_turns = 12', 'the loop always terminates'],
    ['token + cost budget', 'enforced per run, outside the model'],
    ['tool timeout + retry cap', 'a hung tool cannot hang the agent'],
    ['errors → observations', 'so it adapts instead of repeating'],
    ['full trajectory logged', 'agent bugs hide from the final output'],
    ['kill switch', 'no deploy required']
  ];
  s += txt(470, 78, 'PRODUCTION RAILS  ·  none of these are optional', { fs: 12, w: 600, fill: BLUE });
  rails.forEach((r, i) => {
    s += box(470, 92 + i*54, 370, 46, `${r[0]}|${r[1]}`, { fill: i < 2 ? MARK : '#fff', fs: 12 });
  });

  s += hl(60, 386, 406, 20);
  s += txt(64, 401, 'anything with a right answer is a tool, not a token', { fs: 12.5, w: 600 });
  s += note(60, 432, [
    'A prompt is advisory. A cap is enforcement.',
    'The model may respect "stop if you get stuck" 95% of the time —',
    'the other 5% is what runs overnight and empties the budget.'
  ]);
  return wrap(496, s);
},

/* ------------------------------------------------------------ verify */
verify() {
  let s = '';
  s += txt(60, 30, 'VERIFICATION IS A CLAIM-LEVEL OPERATION', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  s += box(60, 88, 150, 52, 'draft answer', { fill: '#fff' });
  s += step(48, 82, 1);
  s += arrow(210, 114, 284, 114, { color: BLUE, label: 'decompose', ly: -10 });

  const claims = [
    ['12,000 panels', 'report §2 confirms it', 'SUPPORTED', GREEN],
    ['output dipped in winter', 'report §5 says opposite', 'CONTRADICTED', RED],
    ['expanding to Spain', '—  no source anywhere', 'UNSUPPORTED', RED]
  ];
  claims.forEach((c, i) => {
    const y = 80 + i*56;
    s += box(288, y, 150, 44, c[0], { fill: '#fff', fs: 12.5 });
    s += arrow(438, y + 22, 486, y + 22, { color: GREY, sw: 1.2 });
    s += box(486, y, 200, 44, c[1], { fill: SOFT, fs: 11.5 });
    s += txt(700, y + 27, c[2], { fs: 12, w: 600, fill: c[3] });
  });
  s += step(276, 74, 2);

  s += `<line x1="60" y1="330" x2="840" y2="330" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += box(60, 352, 250, 46, 'supported  →  kept, cited', { fill: MARK, fs: 12.5 });
  s += box(340, 352, 250, 46, 'other two  →  deleted', { fill: '#fff', stroke: RED, fs: 12.5 });
  s += txt(614, 380, 'not hedged. deleted.', { fs: 12.5, w: 600, fill: RED });

  s += hl(60, 424, 730, 20);
  s += txt(64, 439, '"it may be the case that C" still puts C in front of the reader, wearing the costume of caution', { fs: 12.5, w: 600 });

  s += note(60, 472, [
    'REPORT ALL FOUR OR NONE:',
    '   outcome  ·  trajectory (right tools, sane order, recovered from errors)',
    '   faithfulness (every claim traceable)  ·  cost + latency',
    '',
    'And validate your LLM judge against human labels before trusting it.',
    'Judges have documented verbosity, position and self-preference biases —',
    'an unvalidated judge is a confident random number generator.'
  ]);
  return wrap(590, s);
},


/* --------------------------------------------------- capstone: genrec */
capstone_genrec() {
  let s = '';
  s += txt(60, 30, 'CAPSTONE · GENERATIVE RECOMMENDER · END TO END', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  // ---- offline column
  s += txt(60, 70, 'OFFLINE  ·  runs once, or on a retrain schedule', { fs: 11.5, fill: GREY });
  const off = [
    ['raw dataset', 'Amazon 2023 / Yambda'],
    ['5-core filter|chronological split', 'frozen + hashed'],
    ['item text → encoder', 'frozen text encoder'],
    ['RQ-VAE', 'levels, codebooks'],
    ['item → SID table', '(c₁, c₂, c₃)']
  ];
  off.forEach((o, i) => {
    const y = 88 + i * 74;
    s += box(60, y, 210, 52, o[0], { fill: i === 3 ? MARK : '#fff', fs: 12 });
    s += txt(278, y + 30, o[1], { fs: 11, fill: GREY });
    if (i) s += arrow(165, y - 22, 165, y - 4, { color: BLUE });
    s += step(48, y + 8, i + 1);
  });

  // ---- divider
  s += `<line x1="430" y1="62" x2="430" y2="452" stroke="${GREY}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  s += txt(438, 70, 'ONLINE  ·  runs per request', { fs: 11.5, fill: GREY });

  // ---- online column
  const on = [
    ['user history', 'last N items as SIDs'],
    ['decoder', 'causal, SID vocabulary'],
    ['constrained beam search', 'valid prefixes only'],
    ['candidate items', 'top-k, all real'],
    ['ranker (optional)', 'or serve directly']
  ];
  on.forEach((o, i) => {
    const y = 88 + i * 74;
    s += box(470, y, 230, 52, o[0], { fill: i === 2 ? MARK : '#fff', fs: 12 });
    s += txt(710, y + 30, o[1], { fs: 11, fill: GREY });
    if (i) s += arrow(585, y - 22, 585, y - 4, { color: BLUE });
    s += step(458, y + 8, i + 6);
  });
  s += `<path d="M165,436 L165,464 L442,464 L442,136 L466,136" fill="none" stroke="${GREY}" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#ahg)"/>`;
  s += txt(292, 486, 'the SID table is the join between the two halves', { fs: 11, fill: GREY });

  // ---- gates
  s += `<line x1="60" y1="524" x2="840" y2="524" stroke="${INK}" stroke-width="1"/>`;
  s += txt(60, 550, 'GATES · a step is not done until its check passes', { fs: 12, w: 600, fill: BLUE });
  const gates = [
    ['split', 'no future leakage'],
    ['tokenizer', 'utilization > 50%'],
    ['decoder', 'tuple → real item'],
    ['eval', 'no sampled negatives'],
    ['serving', 'p99 measured']
  ];
  gates.forEach((g, i) => {
    const x = 60 + i * 157;
    s += box(x, 566, 148, 56, `${g[0]}|${g[1]}`, { fill: '#fff', fs: 11.5 });
  });
  s += hl(60, 646, 470, 20);
  s += txt(64, 661, 'the baseline (SASRec) is built in week 2, before any of this', { fs: 12.5, w: 600 });
  s += txt(60, 690, 'A generative model that has not beaten a properly tuned baseline on a frozen protocol', { fs: 12, fill: GREY });
  s += txt(60, 706, 'has not been shown to do anything.', { fs: 12, fill: GREY });
  return wrap(730, s);
},

/* ---------------------------------------------------------------- rl */
rl() {
  let s = '';
  s += txt(60, 30, 'VALUE PROPAGATES BACKWARDS FROM THE REWARD', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  const G = 5, cell = 56, ox = 60, oy = 76;
  for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
    const d = Math.hypot(G-1-c, G-1-r);
    const v = Math.max(0, 1 - d/(G*1.15));
    s += `<rect x="${ox+c*cell}" y="${oy+r*cell}" width="${cell-3}" height="${cell-3}" fill="${BLUE}" opacity="${0.06+v*0.62}" stroke="#D8DCE6" stroke-width="1"/>`;
    s += txt(ox+c*cell+6, oy+r*cell+16, v.toFixed(2), { fs: 10, fill: v > 0.45 ? '#fff' : GREY });
    if (!(r === G-1 && c === G-1)) {
      const right = c < G-1 ? 1 - Math.hypot(G-1-(c+1), G-1-r)/(G*1.15) : -1;
      const down  = r < G-1 ? 1 - Math.hypot(G-1-c, G-1-(r+1))/(G*1.15) : -1;
      const mx = ox+c*cell+(cell-3)/2, my = oy+r*cell+(cell-3)/2 + 6;
      s += right >= down
        ? `<path d="M${mx-11},${my} L${mx+9},${my} M${mx+4},${my-5} L${mx+9},${my} L${mx+4},${my+5}" stroke="${INK}" stroke-width="1.4" fill="none"/>`
        : `<path d="M${mx},${my-11} L${mx},${my+9} M${mx-5},${my+4} L${mx},${my+9} L${mx+5},${my+4}" stroke="${INK}" stroke-width="1.4" fill="none"/>`;
    }
  }
  s += txt(ox + (G-1)*cell + 4, oy + (G-1)*cell + 46, 'R = +1', { fs: 12, w: 600, fill: '#fff' });
  s += step(ox - 12, oy - 6, 1);
  s += txt(60, oy + G*cell + 26, 'numbers = value estimate', { fs: 12, fill: GREY });
  s += txt(60, oy + G*cell + 44, 'arrows = always step toward the highest value', { fs: 12, fill: GREY });

  s += note(400, 84, [
    'THE TWO DIFFICULTIES',
    '',
    'no answer key —  nobody can demonstrate the',
    '   correct action, only score the outcome',
    '',
    'delayed credit —  you crashed at t=12,',
    '   the mistake was at t=3',
    '',
    '────────────────────────────────',
    '',
    'POLICY GRADIENT     ∇J = E[ ∇log π(a|s) · A ]',
    '',
    '   A = G − b(s)     the advantage: b(s) depends only',
    '   on state, so it cancels in expectation —',
    '   unbiased, and much lower variance.',
    '',
    '────────────────────────────────',
    '',
    'PRICING A POLICY FROM LOGS  (no deployment)',
    '',
    '   logged:  (x, a, p, r)  — p = μ(a|x), recorded',
    '',
    '   V̂(π) = mean of  [ π(a|x) / p ] · r',
    '',
    '   events the new policy would repeat count for',
    '   more; events it would avoid count for less.',
    '   no recorded p, no estimate — at any volume.'
  ]);

  s += hl(60, 488, 420, 20);
  s += txt(64, 503, 'explore in proportion to uncertainty, not at random', { fs: 12.5, w: 600 });
  s += txt(60, 530, 'Always serving the current best guess guarantees you never', { fs: 12, fill: GREY });
  s += txt(60, 546, 'discover a better one. Random exploration wastes traffic.', { fs: 12, fill: GREY });
  s += txt(60, 568, 'Thompson sampling: try each option as often as you believe', { fs: 12, fill: GREY });
  s += txt(60, 584, 'it might be the best.', { fs: 12, fill: GREY });
  return wrap(612, s);
},

/* ---------------------------------------------------------- course map */
course_map() {
  let s = '';
  s += txt(60, 32, 'HOW THE SIX PARTS FEED EACH OTHER', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

  s += step(48, 76, 1);
  s += box(60, 66, 210, 64, 'CHOOSING WHAT TO SHOW|the funnel · features|the ad auction');
  s += step(338, 76, 2);
  s += box(350, 66, 210, 64, 'MEANING AS GEOMETRY|embeddings|searching a billion');
  s += step(628, 76, 3);
  s += box(640, 66, 210, 64, 'SEQUENCES|next-word · next-item|IDs with meaning');
  s += arrow(270, 98, 350, 98, { label: 'needs' });
  s += arrow(560, 98, 640, 98, { label: 'became' });

  s += txt(60, 168, 'part 1 poses the problem: pick a few from a billion, fast. part 2 is the tool that makes the', { fs: 11.5, fill: GREY });
  s += txt(60, 184, 'first cheap stage possible. part 3 is what happened when prediction machinery met sequences.', { fs: 11.5, fill: GREY });

  s += step(48, 240, 4);
  s += box(60, 230, 210, 64, 'LOOKING THINGS UP|retrieval for models|hybrid · chunking');
  s += step(338, 240, 5);
  s += box(350, 230, 210, 64, 'ACTING|the agent loop|the second opinion');
  s += step(628, 240, 6);
  s += box(640, 230, 210, 64, 'CONSEQUENCES|bandits · logs|learning from preference', { fill: MARK });
  s += arrow(270, 262, 350, 262, { label: 'grounds' });
  s += arrow(560, 262, 640, 262, { label: 'improves' });

  s += txt(60, 332, 'part 4 gives a model real documents so it can show its work. part 5 lets it act and be checked.', { fs: 11.5, fill: GREY });
  s += txt(60, 348, 'part 6 closes the loop: every system above improves by learning from what happened.', { fs: 11.5, fill: GREY });

  s += hl(60, 372, 620, 20);
  s += txt(64, 387, 'read in order the first time — every part leans on the ones before it', { fs: 12.5, w: 600 });
  return wrap(410, s);
},

/* ---------------------------------------------------------------- ads */
ads() {
  let s = '';
  s += txt(60, 30, 'ONE AD REQUEST · WHO WINS, WHAT THEY PAY, WHETHER THEY MAY COMPETE', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  // ---- top row: the request path -------------------------------------
  s += box(60, 70, 118, 62, 'AD REQUEST|user + slot|+ context');
  s += step(72, 62, 1);
  s += arrow(186, 101, 214, 101);

  s += box(222, 70, 150, 62, 'ELIGIBILITY|targeting · brand safety|freq cap ≤ 3/user/day');
  s += step(234, 62, 2);
  s += arrow(380, 101, 408, 101);

  s += box(416, 70, 150, 62, 'PACING GATE|compete with|probability p', { fill: SOFT });
  s += step(428, 62, 3);
  s += arrow(574, 101, 602, 101);

  s += box(610, 70, 150, 62, 'AUCTION|rank by bid × pCTR');
  s += step(622, 62, 4);
  s += arrow(768, 101, 796, 101);

  s += box(804, 70, 56, 62, 'AD|shown', { fill: MARK });

  // ---- auction detail: the ranking and the price ---------------------
  s += txt(432, 172, 'inside the auction — expected value per impression:', { fs: 11.5, fill: GREY });
  s += txt(432, 195, 'A  bid $4.00/click × pCTR 1.2% = 4.8¢', { fs: 12.5 });
  s += txt(730, 195, '← wins', { fs: 12.5, w: 600, fill: GREEN });
  s += txt(432, 215, 'B  bid $6.00/click × pCTR 0.6% = 3.6¢', { fs: 12.5, fill: GREY });
  s += txt(432, 235, 'C  bid $2.50/click × pCTR 1.0% = 2.5¢', { fs: 12.5, fill: GREY });

  s += hl(428, 246, 412, 20);
  s += step(408, 256, 6);
  s += txt(434, 261, 'A pays 3.6/1.2 = $3.00 — the least that still beats B', { fs: 12.5, w: 600 });
  s += note(432, 290, [
    'second price: your bid decides IF you win,',
    'never WHAT you pay — so bid your value.',
    'first price (display since 2019): you pay',
    'your own bid — so bidders shade below value.'
  ]);

  // ---- pacing loop: budget -> controller -> gate ---------------------
  s += box(60, 180, 180, 70, 'PACING CONTROLLER|target spend curve|vs actual · PID');
  s += step(72, 172, 5);
  s += box(60, 290, 180, 50, 'DAILY BUDGET|$650 · resets 00:00', { fill: SOFT });
  s += arrow(150, 286, 150, 254);

  s += arrow(240, 196, 448, 136, { color: BLUE, label: 'raise / lower p', ly: -14 });
  s += note(246, 296, [
    'spend too fast → lower p',
    'too slow → raise p —',
    'the bid is never touched'
  ]);

  // ---- spend feedback: serve -> controller ---------------------------
  s += `<path d="M832,136 L832,152 L868,152 L868,372 L424,372 L424,214 L248,214" fill="none" stroke="${GREY}" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ahg)"/>`;
  s += txt(430, 366, 'spend feedback — what keeps the budget alive all day', { fs: 11.5, fill: GREY });

  return wrap(396, s);
},

/* --------------------------------------------------------------- rlhf */
rlhf() {
  let s = '';
  s += txt(60, 32, 'THE PREFERENCE PIPELINE — AND WHAT EACH ERA DELETED', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

  s += step(42, 96, 1);
  s += box(60, 70, 150, 56, 'PAIRS|two answers,|one human pick');
  s += box(280, 70, 170, 56, 'REWARD MODEL|Bradley–Terry fit|learned taste');
  s += box(520, 70, 180, 56, 'RL  (PPO / GRPO)|maximise r̂ − β·KL');
  s += box(760, 70, 90, 56, 'POLICY|tuned', { fill: MARK });
  s += arrow(210, 98, 280, 98);
  s += arrow(450, 98, 520, 98);
  s += arrow(700, 98, 760, 98);
  s += txt(478, 152, 'β·KL — the leash: how far the proxy may be trusted', { fs: 11.5, fill: GREY });

  s += step(42, 226, 2);
  s += txt(60, 204, 'SHORTCUT A · DPO — delete the middle', { fs: 12, w: 600 });
  s += box(60, 216, 150, 44, 'PAIRS', { fill: SOFT });
  s += box(520, 216, 180, 44, 'classification loss|on log-prob margins', { fill: SOFT });
  s += arrow(210, 238, 520, 238, { color: BLUE, label: 'closed form skips the reward model' });
  s += arrow(700, 238, 790, 160, { color: BLUE });

  s += step(42, 330, 3);
  s += txt(60, 308, 'SHORTCUT B · VERIFIABLE REWARDS (RLVR) — replace the judge with a checker', { fs: 12, w: 600 });
  s += box(60, 320, 150, 44, 'TASK|code · maths', { fill: SOFT });
  s += box(280, 320, 170, 44, 'CHECKER|tests pass? r ∈ {0,1}', { fill: SOFT });
  s += box(520, 320, 180, 44, 'GRPO|siblings as baseline', { fill: SOFT });
  s += arrow(210, 342, 280, 342);
  s += arrow(450, 342, 520, 342);
  s += arrow(700, 342, 800, 160, { color: BLUE });
  s += note(730, 320, ['a checker cannot', 'be flattered'], {});

  s += `<line x1="60" y1="398" x2="840" y2="398" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += hl(60, 412, 340, 20);
  s += txt(64, 427, 'THE CONSTANT ACROSS ALL THREE', { fs: 12, w: 600 });
  s += txt(60, 452, 'the training signal is a proxy for what you want. optimise it too hard and', { fs: 12.5 });
  s += txt(60, 470, 'measured reward keeps rising after true quality has turned down.', { fs: 12.5 });
  return wrap(496, s);
},

/* ----------------------------------------------------- capstone: rlab */
capstone_rlab() {
  let s = '';
  s += txt(60, 40, 'LANE 1 · SIMULATOR — where truth is known', { fs: 12, fill: BLUE, w: 600 });
  s += step(42, 78, 1);
  s += box(60, 60, 170, 56, 'five-arm world|true means known|numpy, an afternoon');
  s += box(290, 60, 190, 56, 'bandit algorithms|eps-greedy · UCB|Thompson sampling');
  s += box(540, 60, 170, 56, 'regret curves|vs fixed A/B split');
  s += arrow(230, 88, 290, 88);
  s += arrow(480, 88, 540, 88);
  s += note(730, 74, ['the only place an', 'estimator can be', 'checked against', 'the answer'], {});

  s += txt(60, 172, 'LANE 2 · LOGGED DATA — where the world is real', { fs: 12, fill: BLUE, w: 600 });
  s += step(42, 210, 2);
  s += box(60, 192, 170, 62, 'Open Bandit Dataset|26M impressions|context, action,|propensity, click');
  s += box(290, 192, 190, 62, 'sanity first|propensity histogram|effective sample size');
  s += box(540, 192, 170, 62, 'OPE estimators|IPS · SNIPS · DR|bootstrap CIs');
  s += arrow(230, 223, 290, 223);
  s += arrow(480, 223, 540, 223);
  s += note(725, 206, ['propensities are', 'the rare part —', 'they are why any', 'estimation works'], {});

  s += step(42, 330, 3);
  s += `<path d="M400,116 L400,152 L510,152 L510,300" fill="none" stroke="${GREY}" stroke-width="1.5" stroke-dasharray="3 4" marker-end="url(#ahg)"/>`;
  s += arrow(620, 254, 620, 300, {});
  s += box(240, 306, 420, 52, 'THE DECISION TABLE|policy × estimator · value ± CI · which one ships', { fill: MARK, sw: 2 });
  s += note(60, 324, ['algorithms proven in', 'lane 1 are the rows;', 'lane 2 prices them'], {});
  s += note(690, 324, ['no deployment', 'happened. that is', 'the entire point.'], {});
  s += txt(240, 396, 'optional last rung: run the winner on a surface you own and compare outcome to prediction —', { fs: 11.5, fill: GREY });
  s += txt(240, 412, 'the calibration gap is the finding, whichever way it goes.', { fs: 11.5, fill: GREY });
  return wrap(430, s);
},

/* ------------------------------------------- worked examples, inline */
/* ---- funnel: item #482,119,204, 0.71 → 0.93 → tile #2, ~55 ms ---- */
funnel_ex: function () {
  let s = '';
  s += txt(60, 30, 'ITEM #482,119,204 — YOUR BLUE TRAINERS, ONE REQUEST', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  s += box(60, 62, 220, 66, 'RETRIEVAL|scores them 0.71|above the bar', { fill: SOFT, fs: 13.5 });
  s += step(72, 56, 1);
  s += box(340, 62, 220, 66, 'RANKING|scores them 0.93|3rd best of 1,000', { fs: 13.5 });
  s += step(352, 56, 2);
  s += box(620, 62, 220, 66, 'RE-RANKING|near-duplicate above|is dropped', { fs: 13.5 });
  s += step(632, 56, 3);
  s += arrow(284, 95, 336, 95);
  s += arrow(564, 95, 616, 95);

  s += txt(170, 150, '~5 ms sweep', { a: 'middle', fs: 12, fill: GREY });
  s += txt(450, 150, '~40 ms judging', { a: 'middle', fs: 12, fill: GREY });
  s += txt(730, 150, '~10 ms tidy-up', { a: 'middle', fs: 12, fill: GREY });

  s += arrow(730, 158, 730, 184);
  s += hl(340, 186, 530, 26);
  s += txt(350, 204, 'RESULT: second tile on your screen, ~55 ms after enter', { fs: 14, w: 600 });

  s += txt(60, 250, 'had the cheap test said 0.12 instead of 0.71, the story ends at stage 1 —', { fs: 12, fill: GREY });
  s += txt(60, 268, 'the careful judge never even sees what retrieval drops.', { fs: 12, fill: GREY });
  return wrap(296, s);
},

/* ---- features: £9 scorecard contradiction, then cards → mix → 0.87 ---- */
features_ex: function () {
  let s = '';
  s += txt(60, 28, 'TWO £9 LISTINGS MEET THE ADD-THE-POINTS SCORECARD', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="60" y1="38" x2="840" y2="38" stroke="${BLUE}" stroke-width="1"/>`;

  s += step(44, 62, 1);
  s += txt(62, 66, 'headphones: cheap (>10 pts) + five stars = honest bargain', { fs: 12.5 });
  s += hl(62, 76, 660, 20);
  s += txt(66, 90, 'luxury watch: cheap (>10) + luxury brand (>10) = over 20 — top score, worst listing', { fs: 12.5, w: 600 });

  s += step(44, 124, 2);
  s += txt(62, 128, 'the fix — a feature cross, one new line on the card:', { fs: 12.5 });
  s += txt(62, 148, '"cheap × luxury brand" = −30  →  the fake sinks, both honest listings stay', { fs: 12.5 });

  s += step(44, 186, 3);
  s += txt(62, 190, 'now score the £9 headphones for you:', { fs: 12.5 });
  s += box(62, 204, 92, 48, 'your|card', { fill: SOFT, fs: 13.5 });
  s += box(164, 204, 92, 48, 'product|card', { fill: SOFT, fs: 13.5 });
  s += box(266, 204, 92, 48, 'brand|card', { fill: SOFT, fs: 13.5 });
  s += box(368, 204, 108, 48, 'price £9|rating ★★★★★', { fs: 13.5 });
  s += arrow(482, 228, 530, 228);
  s += box(534, 204, 160, 48, 'SMALL MIXER|a few layers deep', { fs: 13.5 });
  s += arrow(700, 228, 746, 228);
  s += box(750, 204, 108, 48, '0.87|click chance', { fs: 13.5 });

  s += txt(62, 286, 'the mixing is where cheap-meets-luxury gets learned: microseconds of arithmetic,', { fs: 12, fill: GREY });
  s += txt(62, 304, 'while the warehouse of cards behind it holds tens of billions of numbers.', { fs: 12, fill: GREY });
  return wrap(330, s);
},

/* ---- ads: Anna $4.00 / Ben $2.50 / Carla $1.00 — winner pays $2.50 ---- */
ads_ex: function () {
  let s = '';
  s += txt(60, 30, 'ONE SLOT, THREE BIDDERS — SECOND-PRICE RULE', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

  s += step(48, 62, 1);
  s += box(60, 70, 170, 42, 'ANNA bids $4.00', { fill: SOFT, fs: 13 });
  s += box(60, 122, 170, 42, 'BEN bids $2.50', { fs: 13 });
  s += box(60, 174, 170, 42, 'CARLA bids $1.00', { fs: 13 });

  s += arrow(234, 91, 296, 91);
  s += box(300, 70, 176, 42, 'ANNA WINS|highest bid', { fs: 13.5 });
  s += step(288, 62, 2);
  s += hl(300, 124, 246, 24);
  s += txt(306, 141, 'her bill: $2.50 — Ben\'s bid', { fs: 14, w: 600 });
  s += txt(306, 164, 'not her own $4.00', { fs: 12, fill: GREY });

  s += step(548, 62, 3);
  s += txt(562, 66, 'YOUR TURN — a click is worth $3.00 to you', { fs: 12.5, w: 600 });
  s += txt(562, 94, 'bid $3.00 → win, pay $2.50 → $0.50 profit', { fs: 12.5 });
  s += txt(562, 118, 'bid $3.60 → extra wins all cost over $3.00', { fs: 12.5 });
  s += txt(562, 142, 'bid $2.40 → lose auctions worth winning', { fs: 12.5 });

  s += `<line x1="60" y1="236" x2="840" y2="236" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += txt(60, 262, 'your bid decides whether you win, never what you pay —', { fs: 12, fill: GREY });
  s += txt(60, 280, 'so bidding your true $3.00 beats every clever alternative.', { fs: 12, fill: GREY });
  return wrap(306, s);
},

/* ---------------------------------------------------------- embed_ex
   The pinboard with real coordinates: query (8,2), A (7,3) ≈1.4,
   B (1,9) ≈9.9 — the sneakers win by a factor of seven. */
embed_ex: function() {
  let s = '';
  const H = 320;
  s += txt(60, 28, 'THE PINBOARD, WITH REAL COORDINATES', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="60" y1="38" x2="840" y2="38" stroke="${BLUE}" stroke-width="1"/>`;

  // plot: board coords (0..10, 0..10) -> screen
  const X = c => 70 + c * 36;          // 0..10 -> 70..430
  const Y = c => 285 - c * 23;         // 0..10 -> 285..55
  // soft grid, pinboard feel
  for (let g = 2; g <= 8; g += 2) {
    s += `<line x1="${X(g)}" y1="${Y(0)}" x2="${X(g)}" y2="${Y(10)}" stroke="${SOFT}" stroke-width="1"/>`;
    s += `<line x1="${X(0)}" y1="${Y(g)}" x2="${X(10)}" y2="${Y(g)}" stroke="${SOFT}" stroke-width="1"/>`;
  }
  // axes
  s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(10)}" y2="${Y(0)}" stroke="${INK}" stroke-width="1.5"/>`;
  s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(0)}" y2="${Y(10)}" stroke="${INK}" stroke-width="1.5"/>`;
  s += txt(X(0), 302, '0', { a: 'middle', fs: 12, fill: GREY });
  s += txt(X(10), 302, '10', { a: 'middle', fs: 12, fill: GREY });
  s += txt(X(0) - 12, Y(10) + 4, '10', { a: 'end', fs: 12, fill: GREY });

  const qx = X(8), qy = Y(2);          // query (8,2)
  const ax = X(7), ay = Y(3);          // A (7,3)
  const bx = X(1), by = Y(9);          // B (1,9)

  // distances first, so pins sit on top
  s += `<line x1="${qx}" y1="${qy}" x2="${ax}" y2="${ay}" stroke="${GREEN}" stroke-width="2"/>`;
  s += `<line x1="${qx}" y1="${qy}" x2="${bx}" y2="${by}" stroke="${RED}" stroke-width="1.5" stroke-dasharray="4 4"/>`;
  s += txt(374, 226, '≈ 1.4', { fs: 12.5, w: 600, fill: GREEN });
  s += txt(238, 150, '≈ 9.9', { fs: 12.5, w: 600, fill: RED });

  // the three pins
  s += `<circle cx="${qx}" cy="${qy}" r="6" fill="${BLUE}"/>`;
  s += `<circle cx="${ax}" cy="${ay}" r="6" fill="${INK}"/>`;
  s += `<circle cx="${bx}" cy="${by}" r="6" fill="${INK}"/>`;
  s += txt(qx, qy + 24, 'query (8, 2)', { a: 'middle', fs: 12.5, w: 600, fill: BLUE });
  s += txt(qx, qy + 40, '"white trainers for the gym"', { a: 'middle', fs: 12, fill: GREY });
  s += txt(ax, ay - 36, '"Court white leather sneakers"', { a: 'middle', fs: 12, fill: GREY });
  s += txt(ax, ay - 20, 'A (7, 3)', { a: 'middle', fs: 12.5, w: 600 });
  s += txt(bx + 14, by - 4, 'B (1, 9)', { fs: 12.5, w: 600 });
  s += txt(bx + 14, by + 12, '"white gloss paint"', { fs: 12, fill: GREY });

  // right panel: the argument in reading order
  s += step(492, 64, 1);
  s += txt(510, 60, 'word overlap: A and B each share', { fs: 12.5 });
  s += txt(510, 76, 'exactly one word, "white" — a tie', { fs: 12.5, fill: GREY });

  s += step(492, 114, 2);
  s += txt(510, 110, 'measure distances on the board:', { fs: 12.5 });
  s += txt(510, 128, 'to A: 1 across, 1 down  ≈ 1.4', { fs: 12.5, w: 600, fill: GREEN });
  s += txt(510, 146, 'to B: 7 across, 7 up    ≈ 9.9', { fs: 12.5, w: 600, fill: RED });

  s += step(492, 186, 3);
  s += hl(506, 174, 296, 20);
  s += txt(510, 188, 'the sneakers win by a factor of seven', { fs: 12.5, w: 600 });

  s += note(510, 218, [
    'the word-matcher had no way to prefer',
    'A over B; the pinboard makes it obvious.',
    'position came from behaviour — the same',
    'runners buy trainers and sneakers.'
  ]);
  return wrap(H, s);
},

/* ------------------------------------------------------------ ann_ex
   12 points, 3 neighbourhoods; B's best is 0.31 in 7 checks, the true
   nearest 0.28 sits across C's border; widen to 2 -> 11 checks. */
ann_ex: function() {
  let s = '';
  const H = 340;
  s += txt(60, 26, '12 POINTS, 3 NEIGHBOURHOODS, ONE BORDERLINE MISS', { fs: 13, w: 600, fill: BLUE });

  // neighbourhood circles (dashed) + centres (squares) + members (dots)
  const hood = (cx, cy, r) =>
    `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${GREY}" stroke-width="1" stroke-dasharray="4 4"/>`;
  const centre = (cx, cy) =>
    `<rect x="${cx - 5}" y="${cy - 5}" width="10" height="10" fill="#fff" stroke="${INK}" stroke-width="1.8"/>`;
  const dot = (cx, cy) => `<circle cx="${cx}" cy="${cy}" r="4.5" fill="${INK}"/>`;

  // A — far away, never visited
  s += hood(120, 265, 55) + centre(120, 265);
  [[85, 235], [90, 295], [155, 297], [160, 238]].forEach(p => s += dot(p[0], p[1]));
  s += txt(120, 202, 'A', { a: 'middle', fs: 15, w: 600 });

  // C — holds the true nearest at 0.28, just inside its border
  s += hood(235, 125, 88) + centre(235, 125);
  [[190, 90], [285, 85], [180, 160]].forEach(p => s += dot(p[0], p[1]));
  s += `<circle cx="312" cy="158" r="5" fill="${RED}"/>`;   // the 0.28 point
  s += txt(160, 66, 'C', { a: 'middle', fs: 15, w: 600 });

  // B — nearest centre, searched first
  s += hood(425, 225, 72) + centre(425, 225);
  [[388, 197], [458, 275], [487, 200], [405, 282]].forEach(p => s += dot(p[0], p[1]));
  s += txt(430, 314, 'B', { a: 'middle', fs: 15, w: 600 });

  // query + the two candidate distances
  const qx = 348, qy = 178;
  s += `<line x1="${qx}" y1="${qy}" x2="388" y2="197" stroke="${GREY}" stroke-width="1.5"/>`;
  s += `<line x1="${qx}" y1="${qy}" x2="312" y2="158" stroke="${RED}" stroke-width="1.8" stroke-dasharray="4 3"/>`;
  s += `<circle cx="${qx}" cy="${qy}" r="6" fill="${BLUE}"/>`;
  s += txt(qx + 12, qy - 8, 'query', { fs: 13, w: 600, fill: BLUE });
  s += txt(384, 218, '0.31', { fs: 12.5, w: 600, fill: GREY });
  s += txt(296, 146, '0.28', { fs: 12.5, w: 600, fill: RED });
  s += note(240, 196, ['true nearest — just', "across C's border"],
            { from: [300, 190], to: [310, 164] });

  // right panel: the search, in order
  s += step(590, 64, 1);
  s += txt(608, 60, 'compare the 3 centres only —', { fs: 12.5 });
  s += txt(608, 76, "B's is nearest · 3 checks", { fs: 12.5, fill: GREY });

  s += step(590, 114, 2);
  s += txt(608, 110, "walk B's 4 members —", { fs: 12.5 });
  s += txt(608, 126, 'best found: 0.31 · 7 checks total', { fs: 12.5, fill: GREY });
  s += txt(608, 152, 'but the true nearest, 0.28, lives', { fs: 12.5, fill: RED });
  s += txt(608, 168, 'in C — and we never entered C', { fs: 12.5, fill: RED });

  s += step(590, 204, 3);
  s += hl(604, 192, 258, 20);
  s += txt(608, 206, 'widen to 2 neighbourhoods → 0.28', { fs: 12.5, w: 600 });
  s += txt(608, 226, '11 checks: 3 centres + 8 members', { fs: 12.5, fill: GREY });

  s += note(608, 258, [
    "that widening dial is HNSW's efSearch:",
    'wider = more reliable and slower;',
    'narrower = faster and riskier.'
  ]);
  return wrap(H, s);
},

/* ------------------------------------------------------ attention_ex
   "The bees stung the boy, so he began to" -> shares of notice
   0.40 / 0.25 / 0.15 / 0.10 (+0.10 shared) -> "cry". */
attention_ex: function() {
  let s = '';
  const H = 300;
  s += txt(60, 26, 'ONE PREDICTION, TRACED', { fs: 13, w: 600, fill: BLUE });

  const slotX = 610, baseY = 212;
  // words with their x positions; big four carry named weights
  const words = [
    { t: 'The',   x: 55,  wgt: 0,    peak: 62  },
    { t: 'bees',  x: 118, wgt: 0.10, peak: 76  },
    { t: 'stung', x: 196, wgt: 0.40, peak: 90  },
    { t: 'the',   x: 268, wgt: 0,    peak: 104 },
    { t: 'boy,',  x: 330, wgt: 0.25, peak: 118 },
    { t: 'so',    x: 388, wgt: 0,    peak: 132 },
    { t: 'he',    x: 442, wgt: 0.15, peak: 146 },
    { t: 'began', x: 510, wgt: 0,    peak: 160 },
    { t: 'to',    x: 566, wgt: 0,    peak: 174 }
  ];
  // attention arcs: thickness = share of notice
  words.forEach(w => {
    const major = w.wgt > 0;
    const sw = major ? Math.max(1.4, w.wgt * 13) : 0.9;
    const col = major ? BLUE : GREY;
    s += `<path d="M${w.x},${baseY} Q${(w.x + slotX) / 2},${w.peak} ${slotX},${baseY}" fill="none" stroke="${col}" stroke-width="${sw}" opacity="${major ? 0.9 : 0.55}"/>`;
  });
  // the sentence
  words.forEach(w => {
    s += txt(w.x, 232, w.t, { a: 'middle', fs: 14.5 });
    if (w.wgt > 0) s += txt(w.x, 252, w.wgt.toFixed(2), { a: 'middle', fs: 12.5, w: 600, fill: BLUE });
  });
  s += txt(slotX, 232, '___', { a: 'middle', fs: 14.5, w: 600, fill: BLUE });
  s += txt(305, 276, 'the small connecting words share the last 0.10', { a: 'middle', fs: 12, fill: GREY });

  // right panel: interview -> shares -> blend -> word
  s += step(652, 64, 1);
  s += txt(670, 60, 'the position after "to" interviews', { fs: 12.5 });
  s += txt(670, 76, 'every word already written', { fs: 12.5, fill: GREY });

  s += step(652, 110, 2);
  s += txt(670, 106, 'scores become shares of notice:', { fs: 12.5 });
  s += txt(670, 122, '.40+.25+.15+.10+.10 = 1.00', { fs: 12.5, w: 600 });

  s += step(652, 156, 3);
  s += txt(670, 152, 'the blend that comes back says', { fs: 12.5 });
  s += txt(670, 168, '"something painful just', { fs: 12.5, fill: GREY });
  s += txt(670, 184, 'happened to a child"', { fs: 12.5, fill: GREY });

  s += hl(664, 204, 174, 22);
  s += txt(670, 220, '⇒ next word: "cry"', { fs: 14, w: 600 });

  s += note(664, 248, [
    'then "cry" is appended and the whole',
    'procedure runs again from scratch —',
    'fresh interviews, fresh scores.'
  ]);
  return wrap(H, s);
},

/* ------------------------------------------------ seqrec worked example
   kettle → teapot → mugs, read as a sentence; similarity says "another
   kettle", the sequence model says "tin of loose-leaf tea". */
seqrec_ex: function() {
  let s = '';
  s += txt(20, 28, 'ONE HISTORY, TWO WAYS TO FINISH THE SENTENCE', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="20" y1="38" x2="880" y2="38" stroke="${BLUE}" stroke-width="1"/>`;

  // (1) the history, read in time order like a sentence
  s += step(30, 100, 1);
  s += box(52, 72, 140, 56, 'KETTLE|March', { fs: 13.5 });
  s += arrow(196, 100, 216, 100);
  s += box(220, 72, 140, 56, 'TEAPOT|April', { fs: 13.5 });
  s += arrow(364, 100, 384, 100);
  s += box(388, 72, 140, 56, 'MUGS|this week', { fs: 13.5 });
  s += arrow(532, 100, 552, 100);
  s += box(556, 72, 130, 56, 'NEXT ?|the empty slot', { fs: 13.5, fill: SOFT });
  s += txt(706, 92, 'read in order,', { fs: 12, fill: GREY });
  s += txt(706, 108, 'like a sentence', { fs: 12, fill: GREY });
  s += txt(52, 152, '"kettle, teapot, mugs, …"', { fs: 12.5, fill: GREY });

  // (2) similarity system: keys on the latest purchase only
  s += step(30, 180, 2);
  s += txt(50, 185, 'SIMILARITY: keys on the latest purchase', { fs: 12.5, w: 600 });
  s += arrow(430, 132, 200, 212, { color: GREY, dash: '3 3' });
  s += box(52, 218, 230, 56, 'ANOTHER KETTLE|slightly fancier', { fs: 13.5, stroke: RED });
  s += txt(52, 296, 'more of what you already own', { fs: 12, fill: GREY });

  // (3) sequence model: reads the story's direction
  s += step(490, 180, 3);
  s += txt(510, 185, "SEQUENCE: reads the story's direction", { fs: 12.5, w: 600 });
  s += txt(510, 204, 'kitting out a tea corner: hardware → the drink', { fs: 12, fill: GREY });
  s += arrow(620, 132, 660, 212, { color: BLUE });
  s += box(512, 218, 300, 56, 'TIN OF LOOSE-LEAF TEA|or a strainer', { fs: 13.5, stroke: GREEN });

  // punchline
  s += hl(20, 306, 640, 20);
  s += txt(26, 321, 'the guess follows the direction of the story, not the look of the last item', { fs: 13, w: 600 });
  return wrap(336, s);
},

/* --------------------------------------------------- sid worked example
   The one-hour-old trail-running shoes get code (12, 7, 3) and inherit
   day-one knowledge from their (12, 7, …) prefix family. */
sid_ex: function() {
  let s = '';
  s += txt(20, 28, 'ONE HOUR OLD, ALREADY ON THE RIGHT SHELF', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="20" y1="38" x2="880" y2="38" stroke="${BLUE}" stroke-width="1"/>`;

  // (1) new item → code, from content alone
  s += step(30, 82, 1);
  s += box(52, 52, 230, 60, 'NEW TRAIL-RUNNING SHOES|1 hour old · zero clicks', { fs: 13.5 });
  s += arrow(286, 82, 340, 82);
  s += txt(286, 128, 'from title, photos,', { fs: 12, fill: GREY });
  s += txt(286, 144, 'description alone', { fs: 12, fill: GREY });
  s += box(344, 52, 130, 60, 'CODE|(12, 7, 3)', { fs: 13.5 });

  // (2) the prefix family it joins
  s += step(510, 45, 2);
  s += `<rect x="500" y="55" width="380" height="150" rx="2" fill="${SOFT}" stroke="${GREY}" stroke-width="1.5"/>`;
  s += txt(520, 80, 'THE (12, 7, …) SHELF', { fs: 13, w: 600 });
  s += txt(520, 98, 'every item here: some kind of running shoe', { fs: 12, fill: GREY });
  s += box(520, 110, 100, 44, '(12, 7, 2)', { fs: 13 });
  s += box(632, 110, 100, 44, '(12, 7, 5)', { fs: 13 });
  s += box(744, 110, 118, 44, '(12, 7, 3)|newcomer', { fs: 13.5, fill: MARK });
  s += txt(520, 188, 'years of clicks pinned to these shelf-mates', { fs: 12, fill: GREY });
  s += arrow(474, 82, 500, 82, { color: BLUE });

  // contrast: what a serial number would have given it
  s += txt(52, 170, 'compare: serial #482,119,204', { fs: 12.5, fill: RED });
  s += txt(52, 188, 'records only arrival order — a locked', { fs: 12, fill: GREY });
  s += txt(52, 204, 'door with nothing pinned to it', { fs: 12, fill: GREY });

  // (3) what transfers on day one
  s += step(30, 262, 3);
  s += arrow(690, 205, 480, 262, { color: BLUE });
  s += txt(600, 250, 'inherits', { fs: 12, fill: BLUE });
  s += box(52, 236, 420, 56, 'DAY-ONE KNOWLEDGE|inherited before its first click', { fs: 13.5, stroke: GREEN });

  // punchline
  s += hl(20, 304, 720, 20);
  s += txt(26, 319, 'everything learned from clicks on (12, 7, …) items transfers to the newcomer', { fs: 13, w: 600 });
  return wrap(336, s);
},

/* --------------------------------------------------- rag worked example
   The parental-leave question: three fetched passages, the answer quoting
   passage 2, and the failure lane when the right passage isn't fetched. */
rag_ex: function() {
  let s = '';
  s += txt(20, 28, 'ONE QUESTION, TWO RUNS OF THE PIPELINE', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="20" y1="38" x2="880" y2="38" stroke="${BLUE}" stroke-width="1"/>`;

  // (1) the question and the three fetched passages
  s += box(20, 58, 190, 64, 'YOUR QUESTION|parental leave at|Meridian Foods?', { fs: 13.5 });
  s += step(232, 60, 1);
  s += arrow(214, 100, 258, 100);
  s += txt(222, 130, 'fetch 3', { fs: 12, fill: GREY });
  s += box(262, 50, 254, 38, 'passage 1 · holiday allowance', { fs: 13 });
  s += box(262, 94, 254, 38, 'passage 2 · parental: 18 weeks', { fs: 13, fill: MARK });
  s += box(262, 138, 254, 38, 'passage 3 · sick-day rules', { fs: 13 });
  s += txt(262, 196, '"eighteen weeks … effective 1 March"', { fs: 12, fill: GREY });

  // (2) the model answers from the passages, and cites
  s += step(542, 60, 2);
  s += arrow(520, 113, 560, 113);
  s += box(564, 80, 316, 66, 'ANSWER|"Eighteen weeks of paid leave,|per [passage 2]"', { fs: 13.5, stroke: GREEN });
  s += hl(564, 154, 288, 18);
  s += txt(570, 168, 'you can open passage 2 and check', { fs: 12.5, w: 600 });

  // (3) same question, but the search misses
  s += `<line x1="20" y1="212" x2="880" y2="212" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += step(30, 240, 3);
  s += txt(50, 245, 'SAME QUESTION, SEARCH MISSES', { fs: 12.5, w: 600, fill: RED });
  s += txt(50, 268, 'no parental-leave', { fs: 12, fill: RED });
  s += txt(50, 284, 'passage fetched', { fs: 12, fill: RED });
  s += box(262, 226, 254, 32, 'passage · holidays only', { fs: 13 });
  s += box(262, 264, 254, 32, 'passage · sick days only', { fs: 13 });
  s += arrow(520, 262, 560, 262);
  s += box(564, 232, 316, 60, '"Twelve weeks" — a fluent guess|screen looks identical', { fs: 13.5, stroke: RED });
  s += txt(50, 322, 'the answer can never beat the fetched passages — measure the fetch on its own, yes or no', { fs: 12, fill: GREY });
  return wrap(336, s);
},

/* ------------------------------------------------ hybrid: the dishwasher query,
   two ranked lists, and 1/2 + 1/1 = 1.5 beating 1/1 = 1.0 */
hybrid_ex: function() {
  let s = '';
  s += txt(450, 28, 'QUERY: "my dishwasher shows error code E-4471 and won\'t start"', { a: 'middle', fs: 13.5, w: 600 });
  s += arrow(330, 38, 210, 58, { color: GREY });
  s += arrow(570, 38, 690, 58, { color: GREY });

  // left column — keyword librarian
  s += step(75, 68, 1);
  s += txt(93, 72, 'KEYWORD LIBRARIAN · BM25', { fs: 12.5, w: 600, fill: BLUE });
  s += box(60, 84, 300, 32, '#1  E-4471 fault code reference');
  s += box(60, 122, 300, 32, "#2  dishwasher won't start guide", { fill: SOFT });
  s += box(60, 160, 300, 32, '#3  old parts catalogue');

  // right column — semantic librarian
  s += step(555, 68, 2);
  s += txt(573, 72, 'SEMANTIC LIBRARIAN', { fs: 12.5, w: 600, fill: BLUE });
  s += box(540, 84, 300, 32, "#1  dishwasher won't start guide", { fill: SOFT });
  s += box(540, 122, 300, 32, '#2  fail-to-begin-a-cycle article');
  s += box(540, 160, 300, 32, '#3  maintenance checklist');

  // why no arrow ever averages the raw scores
  s += note(450, 104, ['raw scores:', '12.7 vs 0.83', '— different', 'units, never', 'averaged'], { a: 'middle' });

  // fuse by position
  s += arrow(210, 198, 300, 230, { color: GREY });
  s += arrow(690, 198, 560, 230, { color: GREY });
  s += step(40, 250, 3);
  s += txt(58, 254, 'FUSE: each list gives 1/position, then add', { fs: 13, w: 600, fill: BLUE });
  s += hl(56, 266, 470, 21);
  s += txt(60, 281, "dishwasher won't start guide  1/2 + 1/1 = 1.5  ← wins", { fs: 13.5, w: 600 });
  s += txt(60, 304, 'E-4471 fault code reference   1/1  +  —  = 1.0', { fs: 13 });
  s += txt(60, 325, 'fail-to-begin-a-cycle article  —  + 1/2 = 0.5', { fs: 13 });
  s += note(560, 281, ['in both lists → its two shares add:', 'agreement beats one lone favourite.', 'real systems add ~60 to each rank.']);
  return wrap(340, s);
},

/* ------------------------------------------------ chunk: the 49-word passage as a
   strip, the two knife positions, and what each cut produces */
chunk_ex: function() {
  let s = '';
  s += txt(60, 26, 'THE 49-WORD HANDBOOK SECTION — TWO PLACES THE KNIFE CAN FALL', { fs: 12.5, w: 600, fill: BLUE });
  s += `<line x1="60" y1="34" x2="840" y2="34" stroke="${BLUE}" stroke-width="1"/>`;

  // the passage as a strip, to scale: 49 words across 780px
  s += step(38, 72, 1);
  s += box(60, 50, 541, 44, 'booking paragraph|words 1–34: portal, manager, busy periods', { fill: SOFT });
  s += box(601, 50, 95, 44, 'allowed|"may carry|over leave"');
  s += box(696, 50, 144, 44, 'restriction|"two full years"|words 41–49');

  // the two knife positions
  s += `<line x1="601" y1="42" x2="601" y2="100" stroke="${GREEN}" stroke-width="2" stroke-dasharray="4 3"/>`;
  s += `<line x1="696" y1="42" x2="696" y2="100" stroke="${RED}" stroke-width="2" stroke-dasharray="4 3"/>`;
  s += txt(593, 110, 'CUT B — at the ¶ break', { a: 'end', fs: 12.5, w: 600, fill: GREEN });
  s += txt(704, 110, 'CUT A — after word 40', { fs: 12.5, w: 600, fill: RED });
  s += arrow(692, 116, 250, 142, { color: RED });
  s += arrow(605, 116, 655, 142, { color: GREEN });

  // what each cut produces
  s += step(38, 148, 2);
  s += txt(56, 152, 'CUT A PRODUCES', { fs: 12.5, w: 600 });
  s += box(60, 162, 360, 38, "A1 · words 1–40|booking + 'may carry over unused leave'", { stroke: RED });
  s += box(60, 208, 360, 38, "A2 · words 41–49|'This applies only after two full years'", { stroke: RED });
  s += step(478, 148, 3);
  s += txt(496, 152, 'CUT B PRODUCES', { fs: 12.5, w: 600 });
  s += box(500, 162, 340, 38, 'B1 · words 1–34|the booking paragraph', { stroke: GREEN });
  s += box(500, 208, 340, 38, 'B2 · words 35–49|both carry-over sentences together', { stroke: GREEN });

  // same question to both
  s += txt(450, 270, 'ASK BOTH: "Can I carry over leave in my first year?"', { a: 'middle', fs: 13.5, w: 600 });
  s += txt(60, 296, 'retrieval returns A1 → answer "yes" — WRONG', { fs: 12.5, w: 600, fill: RED });
  s += hl(496, 282, 360, 20);
  s += txt(500, 296, 'returns B2 → "not yet — two full years" — RIGHT', { fs: 12.5, w: 600, fill: GREEN });
  s += note(60, 318, ['A2 shares no word with the query — never fetched']);
  s += txt(500, 318, 'only the cut moved; the answer flipped', { fs: 12, fill: GREY });
  return wrap(332, s);
},

/* ------------------------------------------------ agent: the three-flights task as
   think / act / observe turns, with the failed action and the corrected retry */
agent_ex: function() {
  let s = '';
  s += txt(60, 26, 'TASK: "find the cheapest of these three flights and hold a seat"', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="60" y1="34" x2="854" y2="34" stroke="${BLUE}" stroke-width="1"/>`;

  const cols = [100, 256, 412, 568, 724], W = 130;
  for (let i = 0; i < 5; i++) s += txt(cols[i] + W / 2, 54, 'turn ' + (i + 1), { a: 'middle', fs: 12, fill: GREY });

  // the three lanes of the loop
  s += step(30, 84, 1);  s += txt(46, 88, 'THINK', { fs: 12, w: 600 });
  s += step(30, 150, 2); s += txt(46, 154, 'ACT', { fs: 12, w: 600 });
  s += step(30, 216, 3); s += txt(46, 220, 'OBSERVE', { fs: 12, w: 600 });

  // turn 1
  s += box(100, 64, W, 48, 'need 3 prices|before comparing');
  s += box(100, 130, W, 48, 'fare tool|airline A');
  s += box(100, 196, W, 48, '£212', { fill: SOFT });
  s += arrow(165, 112, 165, 128); s += arrow(165, 178, 165, 194);
  // turn 2
  s += box(256, 130, W, 48, 'fare tool|airline B');
  s += box(256, 196, W, 48, '£185|new cheapest', { fill: SOFT });
  s += arrow(321, 178, 321, 194);
  // turn 3 — the failure
  s += box(412, 130, W, 48, 'fare tool|airline C');
  s += box(412, 196, W, 48, 'ERROR|unknown|route code', { stroke: RED, sw: 2 });
  s += arrow(477, 178, 477, 194);
  // turn 4 — the corrected retry
  s += box(568, 64, W, 48, 'failed ≠ none|wrong route code');
  s += box(568, 130, W, 48, 'look up C code|retry');
  s += box(568, 196, W, 48, '£169', { fill: SOFT });
  s += arrow(633, 112, 633, 128); s += arrow(633, 178, 633, 194);
  // turn 5 — done
  s += box(724, 64, W, 48, 'C wins:|£169 < £185');
  s += box(724, 130, W, 48, 'hold-seat tool');
  s += box(724, 196, W, 48, 'confirmation #|done', { stroke: GREEN, sw: 2 });
  s += arrow(789, 112, 789, 128); s += arrow(789, 178, 789, 194);

  // each observation feeds the next thought
  s += arrow(232, 216, 254, 158, { color: GREY });
  s += arrow(388, 216, 410, 158, { color: GREY });
  s += arrow(544, 216, 566, 90, { color: RED });
  s += arrow(700, 216, 722, 90, { color: GREY });

  s += hl(100, 258, 644, 20);
  s += txt(104, 272, 'the error came back as just another observation — the next thought steered around it', { fs: 12.5, w: 600 });
  s += note(100, 296, ['a script would have crashed at turn 3;',
                       'the loop reads its own results — including the bad ones — and corrects course.']);
  return wrap(324, s);
},

/* ------------------------------------------------ verify: judge bias demo */
verify_ex: function() {
  let s = '';
  s += txt(20, 26, 'JUDGE BIAS — SAME TWO ANSWERS, TWO SEATINGS', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="20" y1="36" x2="880" y2="36" stroke="${BLUE}" stroke-width="1"/>`;

  // round 1: A first, B is the long one
  s += step(30, 75, 1);
  s += box(48, 52, 168, 46, 'ANSWER A|short, correct', { fs: 14 });
  s += box(232, 52, 190, 46, 'ANSWER B|long, adds nothing', { fs: 14 });

  // round 2: positions swapped
  s += step(30, 153, 2);
  s += box(48, 130, 168, 46, 'ANSWER B|now read first', { fs: 14 });
  s += box(232, 130, 190, 46, 'ANSWER A|now read second', { fs: 14 });

  // the judge, sitting between both rounds
  s += box(460, 64, 170, 84, 'CLAUDE|the judge|likes long + first', { fs: 14, fill: SOFT });

  // verdicts
  s += box(680, 52, 200, 46, 'VERDICT 1|"B — more thorough"', { fs: 14 });
  s += box(680, 130, 200, 46, 'VERDICT 2|"A — more focused"', { fs: 14 });
  s += arrow(422, 75, 458, 95);
  s += arrow(422, 153, 458, 117);
  s += arrow(630, 95, 678, 75);
  s += arrow(630, 117, 678, 153);

  // punchline
  s += hl(20, 196, 640, 20);
  s += txt(24, 211, 'same answers, opposite verdicts — it graded length and seating, not truth', { fs: 12.5, w: 600 });

  // the two honest fixes
  s += step(30, 252, 3);
  s += box(48, 230, 296, 50, 'FIX 1 — SWAP TEST|trust only verdicts that survive', { fs: 14 });
  s += box(364, 230, 296, 50, 'FIX 2 — DIFFERENT MODEL|writer never grades its own work', { fs: 14 });
  s += note(680, 244, ['a judge that shares the', "writer's blind spots nods at", 'its own inventions']);
  return wrap(300, s);
},

/* --------------------------------- rl: two headlines + the diary row, side by side */
rl_ex: function() {
  let s = '';
  s += txt(20, 26, 'PICK A HEADLINE — EXPLORE VS EXPLOIT', { fs: 13, w: 600, fill: BLUE });
  s += txt(470, 26, 'PRICE A NEW POLICY FROM THE DIARY', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="20" y1="36" x2="430" y2="36" stroke="${BLUE}" stroke-width="1"/>`;
  s += `<line x1="470" y1="36" x2="880" y2="36" stroke="${BLUE}" stroke-width="1"/>`;
  s += `<line x1="450" y1="46" x2="450" y2="272" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;

  // panel 1: the two-headlines table
  s += step(30, 80, 1);
  s += box(48, 56, 372, 46, 'HEADLINE A — 2%|shown 1,000 · 20 clicks · firm', { fs: 14 });
  s += box(48, 112, 372, 46, 'HEADLINE B — 3%?|shown 100 · 3 clicks · luck?', { fs: 14, fill: SOFT });
  s += txt(48, 180, 'trust A. but show only A and B is buried forever —', { fs: 12, fill: GREY });
  s += txt(48, 196, 'if B really is 3%, that costs a click per 100 visitors.', { fs: 12, fill: GREY });
  s += step(30, 226, 2);
  s += txt(48, 230, 'explore in proportion to uncertainty', { fs: 12.5, w: 600, fill: BLUE });

  // panel 2: one diary row, re-weighted
  s += step(480, 80, 3);
  s += box(498, 58, 190, 60, 'DIARY ROW|old policy: prob 0.1|→ visitor clicked', { fs: 14 });
  s += box(498, 138, 190, 46, 'NEW POLICY|would show A: 0.5', { fs: 14 });
  s += box(730, 84, 140, 66, 'WEIGHT|0.5 / 0.1|= 5', { fs: 14, fill: SOFT });
  s += arrow(688, 88, 728, 108);
  s += arrow(688, 161, 728, 130);
  s += hl(700, 168, 180, 20);
  s += txt(704, 183, 'the click counts ×5', { fs: 12.5, w: 600 });
  s += txt(498, 216, 'average such rows → the new policy, priced,', { fs: 12, fill: GREY });
  s += txt(498, 232, 'without one visitor served under it', { fs: 12, fill: GREY });
  s += note(498, 258, ['the recorded 0.1 — the propensity — is the', 'denominator; never written down = never recovered']);
  return wrap(290, s);
},

/* ----------------------- rlhf: sourdough pair → taste-predictor → gamed 0.91 */
rlhf_ex: function() {
  let s = '';
  s += txt(20, 26, 'SOURDOUGH PAIR → TASTE-PREDICTOR → THE GAMED 0.91', { fs: 13, w: 600, fill: BLUE });
  s += `<line x1="20" y1="36" x2="880" y2="36" stroke="${BLUE}" stroke-width="1"/>`;

  // the pair the reader judged instantly
  s += step(30, 70, 1);
  s += box(48, 48, 235, 46, 'REPLY A|"a hunger signal — feed it"', { fs: 14 });
  s += txt(292, 66, '✓ picked at once', { fs: 12, fill: GREEN });
  s += box(48, 104, 235, 46, 'REPLY B|"many causes… try a forum"', { fs: 14 });
  s += txt(292, 122, '✗ passed over', { fs: 12, fill: GREY });
  s += txt(300, 90, 'A beats B', { fs: 12, fill: GREY });
  s += arrow(288, 98, 418, 93);

  // the taste-predictor trained on the picks
  s += step(410, 50, 2);
  s += box(420, 56, 210, 74, 'TASTE-PREDICTOR|(the reward model) trained|on thousands of picks', { fs: 14, fill: SOFT });
  s += arrow(525, 130, 525, 156);
  s += txt(538, 150, 'grades every practice draft, instantly', { fs: 12, fill: GREY });

  // practice — and the gaming moment
  s += step(30, 192, 3);
  s += box(48, 170, 160, 44, 'draft 1|scores 0.58', { fs: 14 });
  s += arrow(208, 192, 232, 192);
  s += box(234, 170, 206, 44, 'draft 2|direct reassurance — 0.66', { fs: 14 });
  s += arrow(440, 192, 464, 192);
  s += box(466, 166, 300, 52, '"Great news! 1) 2) 3)"|says nothing — scores 0.91', { fs: 14 });
  s += hl(466, 226, 300, 20);
  s += txt(470, 241, 'studying the critic, not the cooking', { fs: 12.5, w: 600 });
  s += note(48, 262, ['the counter is the KL leash: the judge is trustworthy only near', 'answers it was trained on — this far out, a rising score means little']);
  return wrap(300, s);
},

};
