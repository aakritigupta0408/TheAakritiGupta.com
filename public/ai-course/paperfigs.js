/* =========================================================================
   PAPER FIGURES
   One drawn summary per paper cited in a lesson's research tab.
   Same drawing grammar as diagrams.js (box / arrow / txt / note / step /
   hl / wrap are already in scope — this file loads after diagrams.js).

   Two animation helpers on top of that grammar:
     aflow(...)  — an arrow whose dashes march, for data/gradient flow
     apulse(svg) — wraps any fragment in a slow attention pulse
   Both are disabled automatically under prefers-reduced-motion.

   Schema per lesson:  PAPERFIGS[lessonId] = [
     { id: 'short-slug', paper: 'Name — Authors, venue year',
       claim: 'the one-sentence result the figure shows',
       svg() { return wrap(H, inner) } }, ...]
   tools/check.py verifies every entry renders, carries enough labels,
   and that no cited paper is left undrawn.
   ========================================================================= */

function aflow(x1, y1, x2, y2, o = {}) {
  return arrow(x1, y1, x2, y2, o).replace('<path ', '<path class="af" ');
}
function apulse(inner) { return `<g class="ap">${inner}</g>`; }

/* --- features locals --- */
/* Paper figures for lesson `features` — one drawn summary per cited paper.
   Uses the diagrams.js grammar plus aflow/apulse from paperfigs.js. */

/* --- embed locals --- */
/* Paper figures — lesson `embed`. One drawn summary per cited paper.
   Same grammar as diagrams.js; aflow/apulse from paperfigs.js.
   Local palette mirror (PFC_embed) so this file resolves the shared ink colors
   without re-declaring the diagrams.js constants. */

const PFC_embed = { ink: '#14161C', blue: '#1B4DB1', mark: '#FFE58A',
              grey: '#5E6678', soft: '#F4F6FA', red: '#B4441F', green: '#1B6B4F' };

/* --- ann locals --- */
/* Paper figures — lesson `ann`. One drawn summary per cited paper.
   Uses the diagrams.js grammar (box/arrow/txt/note/step/hl/wrap) plus
   aflow/apulse from paperfigs.js. All numbers from depth.js `ann` text. */

/* --- attention locals --- */
/* Paper figures — lesson: attention.
   One annotated SVG per cited paper in depth.js research.papers.
   All numbers come from that verified text. */

/* --- seqrec locals --- */
/* Paper figures — lesson: seqrec
   One annotated SVG per paper cited in depth.js → seqrec.research.papers.
   Uses the drawing grammar from diagrams.js and aflow/apulse from paperfigs.js. */

/* --- sid locals --- */
/* Paper figures for lesson `sid` — one drawn summary per cited paper. */

/* --- rag locals --- */
/* Paper figures for lesson: rag
   One annotated SVG per cited paper in depth.js research.papers.
   Uses the diagrams.js grammar (box/arrow/txt/note/step/hl/wrap) plus
   aflow/apulse from paperfigs.js. */

const PF_INK_rag = '#14161C', PF_BLUE = '#1B4DB1', PF_MARK = '#FFE58A',
      PF_GREY = '#5E6678', PF_SOFT = '#F4F6FA', PF_RED = '#B4441F',
      PF_GREEN = '#1B6B4F';

function pfRule(y) {
  return `<line x1="60" y1="${y}" x2="840" y2="${y}" stroke="${PF_BLUE}" stroke-width="1"/>`;
}

/* --- hybrid locals --- */
/* Paper figures for lesson `hybrid` — one drawn summary per cited paper. */

/* --- chunk locals --- */
/* Paper figures for lesson `chunk` — one drawn summary per cited paper. */

/* --- agent locals --- */
/* Paper figures for lesson `agent` — one per <li> in depth.js research.papers */

/* --- verify locals --- */
/* Paper figures for lesson `verify` — one annotated SVG per cited paper.
   Uses the drawing grammar from diagrams.js and aflow/apulse from paperfigs.js. */

/* --- rl locals --- */
/* Paper figures for lesson `rl` — one annotated SVG per cited paper.
   Uses the diagrams.js grammar (box/arrow/txt/note/step/hl/wrap) plus
   aflow/apulse from paperfigs.js. */

/* palette aliases (same values as diagrams.js) */
var PFC_rl = { INK: '#14161C', BLUE: '#1B4DB1', MARK: '#FFE58A',
            GREY: '#5E6678', SOFT: '#F4F6FA', RED: '#B4441F', GREEN: '#1B6B4F' };

const PAPERFIGS = {

funnel: [

  /* ------------------------------------------------ cascade ranking, 2011 */
  { id: 'cascade-ranking-2011',
    paper: 'A Cascade Ranking Model — Wang, Lin & Metzler, SIGIR 2011',
    claim: 'Jointly training successively richer stages against an effectiveness-plus-feature-cost objective improved both quality and cost at once, not merely traded one for the other.',
    svg() {
      let s = '';
      s += txt(60, 32, 'CASCADE RANKING · THE FORMAL ANCESTOR OF THE FUNNEL', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      // three stages: richer features over smaller candidate sets
      s += box(60, 90, 200, 78, 'STAGE 1|cheap features|whole corpus', { fill: SOFT });
      s += box(340, 90, 200, 78, 'STAGE 2|richer features|survivors only');
      s += box(620, 90, 200, 78, 'STAGE 3|richest features|final few');
      s += step(72, 82, 1);
      s += step(352, 82, 2);
      s += step(632, 82, 3);
      s += aflow(268, 129, 332, 129, { label: 'prune' });
      s += aflow(548, 129, 612, 129, { label: 'prune' });
      s += txt(160, 188, 'many docs', { a: 'middle', fs: 11.5, fill: GREY });
      s += txt(440, 188, 'fewer', { a: 'middle', fs: 11.5, fill: GREY });
      s += txt(720, 188, 'fewest', { a: 'middle', fs: 11.5, fill: GREY });

      // the single joint objective wired to every stage
      s += apulse(box(280, 226, 340, 52, 'ONE JOINT OBJECTIVE|effectiveness balanced against feature cost', { fill: MARK }));
      s += arrow(340, 226, 160, 172, { color: GREY, dash: '3 3' });
      s += arrow(470, 226, 462, 172, { color: GREY, dash: '3 3' });
      s += arrow(560, 226, 720, 172, { color: GREY, dash: '3 3' });
      s += txt(450, 296, 'all stages trained together, not independently', { a: 'middle', fs: 11.5, fill: GREY });

      // headline result
      s += hl(60, 318, 476, 20);
      s += txt(64, 333, 'RESULT: the cascade improved BOTH effectiveness AND cost at once', { fs: 12.5, w: 600 });
      s += note(60, 362, ['why it matters: independently trained stages drift apart, and stage 1 starts',
                          'pruning exactly what stage 3 wanted — the joint objective is what closes the seam.']);
      return wrap(400, s);
    } },

  /* ------------------------------------------------ youtube dnn, 2016 */
  { id: 'youtube-two-stage-2016',
    paper: 'Deep Neural Networks for YouTube Recommendations — Covington et al., RecSys 2016',
    claim: 'A candidate-generation network narrows millions of videos to hundreds via ANN lookup, and a separate ranking network orders them — retrieval optimises recall, ranking optimises precision.',
    svg() {
      let s = '';
      s += txt(60, 32, 'YOUTUBE 2016 · THE TWO-STAGE SPLIT THAT BECAME THE INDUSTRY DEFAULT', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      s += box(60, 90, 150, 78, 'CORPUS|millions of|videos', { fill: SOFT });
      s += box(280, 90, 210, 78, 'CANDIDATE GENERATION|network + ANN lookup|at serving time');
      s += box(560, 90, 190, 78, 'RANKING NETWORK|full features per|(user, video) pair');
      s += box(796, 90, 54, 78, 'USER|slate', { fill: MARK });
      s += step(72, 82, 1);
      s += step(292, 82, 2);
      s += step(572, 82, 3);
      s += aflow(218, 129, 272, 129);
      s += aflow(498, 129, 552, 129, { label: 'hundreds', ly: -9 });
      s += arrow(758, 129, 788, 129);

      // the framing from Section 3, in two lines
      s += txt(385, 192, 'optimises RECALL', { a: 'middle', fs: 12, w: 600 });
      s += txt(385, 208, 'do not lose the right video', { a: 'middle', fs: 11, fill: GREY });
      s += txt(655, 192, 'optimises PRECISION', { a: 'middle', fs: 12, w: 600 });
      s += txt(655, 208, 'order the survivors well', { a: 'middle', fs: 11, fill: GREY });

      // headline numbers
      s += hl(60, 236, 330, 20);
      s += txt(64, 251, 'millions  →  hundreds  →  a ranked slate', { fs: 12.5, w: 600 });
      s += note(60, 282, ['why it matters: one model cannot serve both objectives at corpus scale —',
                          'splitting them let each stage specialise, and every feed since copied the shape.']);
      return wrap(330, s);
    } },

  /* ------------------------------------------------ COLD, 2020 */
  { id: 'cold-preranking-2020',
    paper: 'COLD — Wang et al., Alibaba 2020',
    claim: 'Pre-ranking is treated as algorithm-system co-design: the model and the computing power it costs are optimised jointly, over a candidate set tens to hundreds of times larger than ranking sees.',
    svg() {
      let s = '';
      s += txt(60, 32, 'COLD · PRE-RANKING AS ALGORITHM-SYSTEM CO-DESIGN', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      s += box(60, 90, 170, 78, 'MATCHING|candidate|sources', { fill: SOFT });
      s += box(340, 90, 220, 78, 'PRE-RANKING (COLD)|flexible network, not|a fixed architecture');
      s += box(660, 90, 180, 78, 'RANKING|heavy model|full features');
      s += step(72, 82, 1);
      s += step(352, 82, 2);
      s += step(672, 82, 3);
      s += aflow(238, 129, 332, 129, { label: 'hundreds of', ly: -20 });
      s += txt(285, 122, 'thousands', { a: 'middle', fs: 11.5 });
      s += aflow(568, 129, 652, 129, { label: 'top', ly: -20 });
      s += txt(610, 122, 'thousands', { a: 'middle', fs: 11.5 });

      // the co-design loop under the pre-ranker
      s += box(210, 226, 200, 52, 'MODEL SIDE|accuracy of the|pre-rank scores');
      s += box(500, 226, 200, 52, 'SYSTEM SIDE|computing power|the model costs');
      s += arrow(414, 244, 496, 244, { color: GREY });
      s += arrow(496, 262, 414, 262, { color: GREY });
      s += txt(455, 232, 'optimised jointly', { a: 'middle', fs: 11, fill: GREY });
      s += arrow(450, 222, 450, 176, { color: GREY, dash: '3 3' });

      // headline number
      s += hl(60, 302, 480, 20);
      s += txt(64, 317, 'pre-ranking candidate set: 10s-100s × larger than ranking sees', { fs: 12.5, w: 600 });
      s += note(60, 346, ['why it matters: fixing an architecture first bakes in a compute budget;',
                          'searching model and cost together buys accuracy a fixed design cannot afford.']);
      return wrap(384, s);
    } },

  /* ------------------------------------------------ OneRec, 2025 */
  { id: 'onerec-generative-2025',
    paper: 'OneRec — Kuaishou, 2025',
    claim: 'One generative model decoding semantic IDs replaces the retrieval-coarse-fine cascade, serving roughly 25% of queries with a +1.68% total watch-time gain — at the price of the cascade cost structure.',
    svg() {
      let s = '';
      s += txt(60, 32, 'ONEREC · THE COUNTER-THESIS: DELETE THE FUNNEL', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      // left: the cascade being replaced, with its seams
      s += txt(90, 74, 'BEFORE: the cascade', { fs: 12, fill: GREY });
      s += box(60, 86, 130, 44, 'RETRIEVAL', { stroke: GREY });
      s += box(60, 154, 130, 44, 'COARSE RANK', { stroke: GREY });
      s += box(60, 222, 130, 44, 'FINE RANK', { stroke: GREY });
      s += arrow(125, 130, 125, 150, { color: GREY });
      s += arrow(125, 198, 125, 218, { color: GREY });
      s += txt(200, 144, 'seam: stages can disagree', { fs: 11, fill: RED });
      s += txt(200, 212, 'seam: stages can disagree', { fs: 11, fill: RED });
      s += step(72, 80, 1);

      // right: one generative model
      s += txt(430, 74, 'AFTER: one model', { fs: 12, fill: GREY });
      s += box(430, 108, 240, 96, 'ONE GENERATIVE MODEL|decodes semantic IDs|retrieval + ranking unified');
      s += step(442, 102, 2);
      s += aflow(674, 156, 760, 156, { label: 'slate' });
      s += box(764, 130, 76, 52, 'USER|feed', { fill: MARK });
      s += txt(550, 226, 'no seams left to disagree across', { a: 'middle', fs: 11.5, fill: GREY });

      // headline numbers from the technical report
      s += step(72, 292, 3);
      s += txt(92, 296, 'reported in the technical report:', { fs: 12, fill: GREY });
      s += hl(60, 308, 380, 20);
      s += txt(64, 323, '~25% of queries served on the main app + Lite', { fs: 12.5, w: 600 });
      s += apulse(hl(470, 308, 250, 20) + txt(474, 323, '+1.68% total watch time', { fs: 12.5, w: 600 }));
      s += note(60, 352, ['why it matters: the price is the cost structure that justified the cascade — and whether',
                          'the win is removed seam losses or simply concentrated compute is genuinely unresolved.']);
      return wrap(384, s);
    } }
],

features: [

/* ------------------------------------------------- 1 · Wide & Deep */
{ id: 'wide-and-deep',
  paper: 'Wide & Deep — Cheng et al., 2016',
  claim: 'Joint-training a memorising linear branch over hand-crossed features with a generalising MLP significantly increased app acquisitions on Google Play.',
  svg() {
    let s = '';
    s += txt(60, 30, 'WIDE & DEEP — MEMORISATION + GENERALISATION, TRAINED JOINTLY', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // inputs
    s += step(64, 68, 1);
    s += txt(82, 72, 'raw sparse features', { fs: 11.5, fill: GREY });
    s += box(60, 84, 140, 46, 'user features|installed apps', { fill: SOFT });
    s += box(60, 144, 140, 46, 'context|impression app', { fill: SOFT });

    // two branches
    s += step(258, 68, 2);
    s += box(250, 84, 205, 50, 'WIDE — linear|memorises crosses x1 × x2');
    s += box(250, 152, 205, 50, 'DEEP — MLP|generalises via embeddings');
    s += arrow(200, 107, 250, 109);
    s += arrow(200, 167, 250, 177);
    s += arrow(200, 117, 250, 170, { color: GREY, dash: '2 3' });

    // joint head
    s += step(534, 108, 3);
    s += box(520, 116, 155, 52, 'JOINT TRAIN|sum of log-odds');
    s += aflow(455, 109, 520, 136);
    s += aflow(455, 177, 520, 148);
    s += box(730, 116, 120, 52, 'sigmoid|P(install)');
    s += arrow(675, 142, 730, 142);

    // headline result
    s += hl(60, 226, 560, 20);
    s += txt(66, 241, 'RESULT: significantly increased app acquisitions on Google Play', { fs: 12.5, w: 600 });

    // why the wide branch exists
    s += note(60, 280, [
      'why a wide branch at all: a linear model cannot fit "positive only when cheap',
      'AND five-stars" — no w1, w2, b satisfy all four cases. Add one product feature',
      'x1·x2 and it is trivially separable. The wide branch memorises exactly such crosses.'
    ]);
    s += note(700, 200, ['the paper that made', '"memorisation vs', 'generalisation" the', 'standard framing'], {});
    return wrap(360, s);
  } },

/* ------------------------------------------------- 2 · DCN / DCN-v2 */
{ id: 'dcn-v2-cross-layers',
  paper: 'DCN & DCN-v2 — Wang et al., 2017 / WWW 2021',
  claim: 'Each cross layer raises interaction degree by one with parameters linear in input width; the low-rank v2 was deployed across Google web-scale ranking systems with significant gains.',
  svg() {
    let s = '';
    s += txt(60, 30, 'DEEP & CROSS NETWORK — LEARN THE CROSSES INSTEAD OF HAND-WRITING THEM', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // stack of cross layers
    s += step(64, 78, 1);
    s += box(60, 92, 135, 52, 'x0|embeds + dense', { fill: SOFT });
    s += box(255, 92, 170, 52, 'CROSS LAYER 1|degree-2 crosses');
    s += box(475, 92, 170, 52, 'CROSS LAYER 2|degree-3 crosses');
    s += box(695, 92, 160, 52, 'LAYER l|degree-(l+1)');
    s += aflow(195, 118, 255, 118);
    s += aflow(425, 118, 475, 118);
    s += arrow(645, 118, 695, 118);
    // x0 re-injected into every layer
    s += arrow(127, 144, 340, 178, { color: GREY, dash: '2 3', curve: 40 });
    s += arrow(127, 144, 560, 186, { color: GREY, dash: '2 3', curve: 60 });
    s += txt(330, 200, 'x0 re-injected at every layer', { fs: 11.5, fill: GREY });

    // the formula
    s += step(64, 232, 2);
    s += hl(82, 220, 380, 24);
    s += txt(90, 237, 'x_{l+1} = x0 ⊙ (W·x_l + b) + x_l', { fs: 14.5, w: 600 });
    s += txt(480, 237, 'parameters linear in input width', { fs: 11.5, fill: GREY });

    // v2 delta
    s += box(60, 268, 265, 48, 'DCN-v2|low-rank mixture cross: W ≈ U·Vᵀ');
    s += note(345, 286, ['low-rank because serving budgets,', 'not benchmarks, set the design'], {});

    // headline deployment
    s += step(64, 344, 3);
    s += hl(82, 332, 690, 20);
    s += txt(88, 347, 'DEPLOYED across Google web-scale ranking systems — significant offline + online gains', { fs: 12.5, w: 600 });
    return wrap(380, s);
  } },

/* ------------------------------------------------- 3 · DLRM */
{ id: 'dlrm-reference-design',
  paper: 'DLRM — Naumov et al., 2019',
  claim: 'Hard-wired pairwise dot-product interaction over embeddings became the open reference workload: MLPerf benchmark, terabyte tables, and DLRMs up to 12 trillion parameters on ZionEX.',
  svg() {
    let s = '';
    s += txt(60, 30, 'DLRM — LESS AN ARCHITECTURE PAPER THAN A SYSTEMS STATEMENT', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // sparse side
    s += step(56, 74, 1);
    s += txt(76, 78, 'sparse IDs → embedding tables (dim d)', { fs: 11.5, fill: GREY });
    for (let i = 0; i < 4; i++) s += box(60 + i * 82, 88, 72, 44, `id ${i + 1}|table`, { fill: SOFT, fs: 11.5 });
    s += txt(60, 156, 'e1 … e_F', { fs: 12, fill: GREY });
    // dense side
    s += box(470, 88, 150, 44, 'dense feats|bottom MLP', { fill: SOFT });

    // interaction
    s += step(296, 174, 2);
    s += box(310, 182, 300, 54, 'PAIRWISE DOT PRODUCTS|all F(F−1)/2 scalars');
    s += aflow(230, 132, 340, 182);
    s += aflow(545, 132, 545, 182);
    s += hl(310, 244, 300, 18);
    s += txt(316, 258, 'second-order interaction hard-wired', { fs: 12 });

    // head
    s += box(670, 182, 115, 54, 'concat|top MLP');
    s += arrow(610, 209, 670, 209);
    s += step(816, 174, 3);
    s += box(805, 188, 55, 42, 'CTR');
    s += arrow(785, 209, 805, 209);

    // scale facts
    s += hl(60, 296, 560, 20);
    s += txt(66, 311, 'ZionEX: DLRMs up to 12 TRILLION parameters · 40× time-to-solution speedup', { fs: 12.5, w: 600 });
    s += txt(60, 336, 'embedding tables at terabyte scale — nearly all parameters live there', { fs: 12, fill: GREY });
    s += txt(60, 356, 'became the MLPerf recommendation benchmark', { fs: 12 });
    s += note(660, 300, ['open-sourced so hardware and', 'systems people had a real', 'workload; TorchRec descends', 'from this stack'], {});
    return wrap(390, s);
  } },

/* ------------------------------------------------- 4 · Feature hashing */
{ id: 'hashing-trick-birthday',
  paper: 'Feature hashing — Weinberger et al., ICML 2009',
  claim: 'Hashing N IDs into M buckets collides at 1 − e^(−N/M): a table as big as the vocabulary still leaves ~63% of IDs sharing a vector.',
  svg() {
    let s = '';
    s += txt(60, 30, 'THE HASHING TRICK — STATELESS, FIXED-SIZE, AND NOT COLLISION-FREE', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // IDs into buckets
    s += step(64, 72, 1);
    s += box(60, 86, 165, 50, 'N = 100,000|distinct sparse IDs', { fill: SOFT });
    s += aflow(225, 111, 380, 111, { label: 'hash(id) mod M' });
    for (let i = 0; i < 7; i++) s += box(385 + i * 58, 86, 50, 50, `b${i + 1}`, { fill: i === 2 ? MARK : '#fff', fs: 11.5 });
    s += apulse(txt(520, 160, '↑ two IDs, one shared vector', { fs: 11.5, fill: RED }));
    s += txt(385, 74, 'M buckets, one embedding each', { fs: 11.5, fill: GREY });

    // the bound
    s += step(64, 200, 2);
    s += txt(84, 205, 'P(collide) = 1 − (1 − 1/M)^(N−1) ≈ 1 − e^(−N/M)', { fs: 14, w: 600 });
    s += hl(84, 220, 590, 20);
    s += txt(90, 235, 'at M = N: 1 − 1/e ≈ 63% of IDs share their vector with another ID', { fs: 12.5, w: 600 });
    s += txt(84, 262, 'expected colliding pairs N(N−1)/2M — the birthday bound bites long before the table looks full', { fs: 11.5, fill: GREY });

    // consequence
    s += step(64, 292, 3);
    s += txt(84, 297, 'ceiling of a hashed model: one vector per bucket → kept signal ≈ 63% at M = N', { fs: 12.5 });
    s += note(84, 322, [
      'why it still works: on sparse data collisions hurt less than intuition says —',
      'the trick Vowpal Wabbit built its whole design around. Run the collision',
      'arithmetic before picking a table size, not after offline metrics look odd.'
    ]);
    return wrap(370, s);
  } },

/* ------------------------------------------------- 5 · Monolith */
{ id: 'monolith-freshness',
  paper: 'Monolith — Liu et al., 2022 (ByteDance)',
  claim: 'Collisionless cuckoo-hashed tables plus minute-level online sync of sparse parameters measurably beat batch-nightly training on freshness-sensitive counter features.',
  svg() {
    let s = '';
    s += txt(60, 30, 'MONOLITH — FEATURE FRESHNESS BEATS BIGGER MODELS', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // losing path
    s += txt(60, 66, 'BATCH-NIGHTLY (loses)', { fs: 11.5, w: 600, fill: GREY });
    s += box(60, 76, 130, 44, 'serving logs', { fill: SOFT });
    s += arrow(190, 98, 255, 98, { color: GREY, dash: '3 3' });
    s += box(255, 76, 150, 44, 'nightly train|hours stale', { stroke: GREY });
    s += arrow(405, 98, 470, 98, { color: GREY, dash: '3 3' });
    s += box(470, 76, 160, 44, 'next-day push|params stale', { stroke: GREY });

    // winning path
    s += txt(60, 158, 'ONLINE TRAINING (wins)', { fs: 11.5, w: 600, fill: BLUE });
    s += step(52, 190, 1);
    s += box(60, 168, 130, 44, 'serving logs', { fill: SOFT });
    s += aflow(190, 190, 255, 190);
    s += step(248, 158, 2);
    s += box(255, 168, 160, 44, 'online training|streaming');
    s += aflow(415, 190, 470, 190, { label: 'minute-level sync', ly: 40 });
    s += step(462, 158, 3);
    s += box(470, 168, 160, 44, 'serving params|sparse part live');

    // the table itself
    s += box(700, 84, 150, 130, 'EMBEDDING TABLE|collisionless|cuckoo hash|expiry +|frequency filter');
    s += note(690, 236, ['without expiry + filtering,', 'new users and items arrive', 'forever and the table', 'eats the cluster'], {});

    // headline
    s += hl(60, 254, 620, 20);
    s += txt(66, 269, 'batch-nightly measurably LOSES to online training on exactly these counter features', { fs: 12.5, w: 600 });
    s += note(60, 300, [
      'behavioural counters (clicks-last-hour, CTR-so-far) decay in value within',
      'hours — the win is freshness of the sparse parameters, not model size.'
    ]);
    return wrap(350, s);
  } },

/* ------------------------------------------------- 6 · HSTU */
{ id: 'hstu-actions-speak-louder',
  paper: 'HSTU / "Actions Speak Louder than Words" — Zhai et al., ICML 2024',
  claim: 'A 1.5-trillion-parameter generative recommender over raw action sequences reported +12.4% in online A/B tests on surfaces with billions of users.',
  svg() {
    let s = '';
    s += txt(60, 30, 'HSTU — THE RAW SEQUENCE SUBSUMES THE ENGINEERED FEATURES', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // what gets replaced
    s += step(56, 68, 1);
    s += txt(76, 72, 'engineered features = lossy summaries', { fs: 11.5, fill: GREY });
    s += box(60, 84, 175, 40, 'counters|clicks-last-hour', { stroke: GREY });
    s += box(60, 134, 175, 40, 'hand crosses|user × item', { stroke: GREY });
    s += box(60, 184, 175, 40, 'CTR-so-far|point-in-time', { stroke: GREY });
    s += arrow(235, 154, 300, 154, { color: GREY, dash: '2 3' });
    s += txt(272, 150, 'derived from', { a: 'middle', fs: 10, fill: GREY });

    // the raw sequence
    s += step(352, 68, 2);
    s += txt(372, 72, 'raw user action sequence', { fs: 11.5, fill: GREY });
    for (let i = 0; i < 6; i++) s += box(310 + i * 54, 130, 46, 44, `a${i + 1}`, { fill: SOFT, fs: 12 });
    s += aflow(634, 152, 690, 152);

    // the model
    s += box(690, 118, 160, 70, 'HSTU|generative|recommender');
    s += txt(690, 208, '1.5 TRILLION parameters', { fs: 12.5, w: 600 });

    // headline
    s += step(56, 264, 3);
    s += apulse(hl(76, 252, 330, 22));
    s += txt(82, 268, '+12.4% in online A/B tests', { fs: 14, w: 600 });
    s += txt(420, 268, 'on surfaces with billions of users', { fs: 12, fill: GREY });

    s += note(60, 302, [
      'motivation: DLRM-style feature interaction had stopped scaling with compute —',
      'sequentialise the raw actions and much of this chapter’s machinery is subsumed.'
    ]);
    return wrap(340, s);
  } }

],

embed: [

/* ---------------------------------------------------- 1 · word2vec */
{ id: 'word2vec-sgns',
  paper: 'word2vec — Mikolov et al., 2013',
  claim: 'Replacing the full softmax over the vocabulary with k ≈ 5–20 noise samples makes training on billions of tokens feasible on a CPU.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SKIP-GRAM WITH NEGATIVE SAMPLING (SGNS)', { fs: 12.5, w: 600, fill: PFC_embed.blue });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_embed.blue}" stroke-width="1"/>`;

    // (1) center word vector
    s += step(56, 74, 1);
    s += box(60, 84, 160, 56, `center word|u_w`, { fill: PFC_embed.soft });

    // (2) one real context vs k sampled fakes
    s += step(326, 62, 2);
    s += box(340, 68, 180, 46, `real context|v_c  ·  label = 1`);
    s += box(340, 158, 180, 40, `noise n_1|label = 0`);
    s += box(340, 204, 180, 40, `noise n_2|label = 0`);
    s += txt(430, 262, 'k noise samples, k ≈ 5–20', { a: 'middle', fs: 11.5, fill: PFC_embed.grey });
    s += txt(430, 278, 'drawn from P_n ∝ unigram^0.75', { a: 'middle', fs: 11.5, fill: PFC_embed.grey });

    // flows: main flow marches
    s += aflow(224, 100, 336, 91, { color: PFC_embed.blue, label: 'σ(u_w·v_c) → 1' });
    s += arrow(224, 122, 336, 180, { color: PFC_embed.grey, label: 'σ(−u_w·v_n) → 0', ly: 14 });

    // (3) the objective
    s += step(586, 62, 3);
    s += box(600, 68, 240, 76, `L = log σ(u_w·v_c)|+ k · E log σ(−u_w·v_n)`);
    s += arrow(524, 91, 596, 100);

    // headline: what the trick buys
    s += apulse(hl(600, 168, 240, 20) +
      txt(604, 183, 'full softmax → k samples', { fs: 12.5, w: 600 }));
    s += txt(600, 212, 'billions of tokens, trained on a CPU', { fs: 12.5, fill: PFC_embed.red });

    s += note(600, 246, ['classify real co-occurrences against', 'sampled fakes — no normalisation over', 'the whole vocabulary ever computed.'],
      { from: [700, 200], to: [720, 152] });

    s += `<line x1="60" y1="306" x2="840" y2="306" stroke="${PFC_embed.grey}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 330, 'the efficiency argument: k noise samples stand in for the entire vocabulary.', { fs: 12, fill: PFC_embed.grey });
    return wrap(352, s);
  } },

/* ------------------------------------- 2 · implicit matrix factorization */
{ id: 'sgns-implicit-pmi',
  paper: 'Neural Word Embedding as Implicit Matrix Factorization — Levy & Goldberg, NeurIPS 2014',
  claim: 'At its optimum SGNS satisfies u_w·v_c = PMI(w,c) − log k: word2vec is implicit low-rank factorization of the shifted PMI matrix.',
  svg() {
    let s = '';
    s += txt(60, 30, 'WHAT WORD2VEC IS ACTUALLY DOING', { fs: 12.5, w: 600, fill: PFC_embed.blue });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_embed.blue}" stroke-width="1"/>`;

    // (1) the training view
    s += step(56, 74, 1);
    s += box(60, 84, 190, 66, `SGNS training|streams (w, c) pairs|k negatives each`, { fill: PFC_embed.soft });

    // (2) the two learned matrices
    s += step(306, 74, 2);
    s += box(320, 84, 120, 66, `W|word vectors`);
    s += txt(452, 122, '·', { fs: 20, w: 600 });
    s += box(474, 84, 120, 66, `Cᵀ|context vectors`);
    s += aflow(254, 117, 316, 117, { color: PFC_embed.blue, label: 'SGD' });

    // (3) the matrix they factorize
    s += step(646, 74, 3);
    s += box(660, 84, 180, 66, `M|shifted PMI matrix`);
    s += arrow(598, 117, 656, 117);
    s += txt(627, 166, '= at optimum', { a: 'middle', fs: 11, fill: PFC_embed.grey });

    // headline equation
    s += apulse(hl(300, 190, 330, 26) +
      txt(465, 209, 'u_w · v_c = PMI(w, c) − log k', { a: 'middle', fs: 14.5, w: 600 }));

    s += note(660, 190, ['a low-rank log co-occurrence', 'model — "meaning as coordinates"', 'is not a metaphor.'],
      { from: [656, 200], to: [634, 203] });

    s += `<line x1="60" y1="248" x2="840" y2="248" stroke="${PFC_embed.grey}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 272, 'the famous analogies are a side effect of this log-linear structure,', { fs: 12.5 });
    s += txt(60, 290, 'not a designed feature. short paper, worth reading in full.', { fs: 12, fill: PFC_embed.grey });
    return wrap(316, s);
  } },

/* ---------------------------------------------------- 3 · Sentence-BERT */
{ id: 'sbert-tower-split',
  paper: 'Sentence-BERT — Reimers & Gurevych, EMNLP 2019',
  claim: 'Cross-encoding all pairs of 10,000 sentences takes ≈65 hours; a siamese encoder plus cosine takes ≈5 seconds — four orders of magnitude.',
  svg() {
    let s = '';
    s += txt(60, 30, 'ALL-PAIRS SIMILARITY OVER 10,000 SENTENCES — TWO WAYS', { fs: 12.5, w: 600, fill: PFC_embed.blue });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_embed.blue}" stroke-width="1"/>`;

    // shared input
    s += step(56, 74, 1);
    s += box(60, 84, 150, 144, `10,000|sentences`, { fill: PFC_embed.soft });

    // path A: cross-encoder, pair by pair
    s += step(266, 74, 2);
    s += box(280, 84, 220, 56, `cross-encoder|every PAIR through BERT`);
    s += arrow(214, 112, 276, 112, { color: PFC_embed.grey, label: 'all pairs' });
    s += box(560, 84, 130, 56, `≈65 hours`, { stroke: PFC_embed.red });
    s += arrow(504, 112, 556, 112, { color: PFC_embed.grey });

    // path B: siamese towers, sentence by sentence
    s += step(266, 162, 3);
    s += box(280, 172, 220, 56, `siamese encoder|each sentence ONCE`);
    s += aflow(214, 200, 276, 200, { color: PFC_embed.blue });
    s += box(560, 172, 130, 56, `≈5 seconds`, { stroke: PFC_embed.green });
    s += arrow(504, 200, 556, 200, { color: PFC_embed.blue, label: 'cosine' });
    s += txt(390, 248, '10,000 vectors, compared by cosine similarity', { a: 'middle', fs: 11.5, fill: PFC_embed.grey });

    // headline gap
    s += apulse(hl(714, 130, 150, 26) +
      txt(722, 149, '10⁴× faster', { fs: 15, w: 600 }));
    s += note(714, 176, ['four orders of magnitude —', 'the entire case for the', 'two-tower split.'],
      { from: [710, 180], to: [694, 143] });

    s += `<line x1="60" y1="272" x2="840" y2="272" stroke="${PFC_embed.grey}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 296, 'encode once, precompute vectors, compare cheaply —', { fs: 12.5 });
    s += txt(60, 314, 'a precomputation strategy wearing an architecture costume.', { fs: 12, fill: PFC_embed.grey });
    return wrap(340, s);
  } },

/* ------------------------------------- 4 · DPR + sampling-bias-corrected */
{ id: 'dual-encoder-logq',
  paper: 'Dense Passage Retrieval — Karpukhin et al., 2020 · Sampling-Bias-Corrected Neural Modeling — Yi et al., RecSys 2019',
  claim: 'QA retrieval and YouTube-scale recommendation independently converged on the same dual encoder; Yi et al. contribute the logQ popularity correction.',
  svg() {
    let s = '';
    s += txt(60, 30, 'ONE ARCHITECTURE, DISCOVERED TWICE', { fs: 12.5, w: 600, fill: PFC_embed.blue });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_embed.blue}" stroke-width="1"/>`;

    // (1) two fields, independently
    s += step(56, 74, 1);
    s += box(60, 84, 200, 66, `DPR · QA, 2020|question tower|passage tower`, { fill: PFC_embed.soft });
    s += box(60, 168, 200, 66, `YouTube · RecSys 2019|user tower|item tower`, { fill: PFC_embed.soft });

    // (2) the shared shape
    s += step(356, 74, 2);
    s += box(370, 110, 210, 88, `same dual encoder|query tower: per request|item tower: precomputed|into an ANN index`);
    s += aflow(264, 117, 366, 140, { color: PFC_embed.blue });
    s += arrow(264, 201, 366, 172, { color: PFC_embed.blue });

    // (3) the training correction
    s += step(646, 74, 3);
    s += box(660, 110, 180, 60, `train with|in-batch negatives`);
    s += arrow(584, 154, 656, 140);
    s += apulse(hl(660, 190, 180, 22) +
      txt(750, 206, 's(x, y) − log Q(y)', { a: 'middle', fs: 13.5, w: 600 }));
    s += txt(750, 230, 'the logQ correction (Yi et al.)', { a: 'middle', fs: 11.5, fill: PFC_embed.grey });

    s += note(370, 226, ['popular items appear in batches too', 'often; subtracting log sampling', 'probability stops the model from', 'learning to punish popularity.'],
      { from: [640, 236], to: [656, 210] });

    s += `<line x1="60" y1="290" x2="840" y2="290" stroke="${PFC_embed.grey}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 314, 'nearly every large feed and search stack since is a variation on this template.', { fs: 12.5 });
    s += txt(60, 332, 'the negatives pipeline is where the recall lives; the architecture barely matters by comparison.', { fs: 12, fill: PFC_embed.grey });
    return wrap(358, s);
  } },

/* ---------------------------------------------------- 5 · E5 / GTE recipe */
{ id: 'e5-gte-recipe',
  paper: 'E5 — Wang et al., 2022 · GTE — Li et al., 2023',
  claim: 'Weakly supervised contrastive pretraining on web-mined pairs, then supervised fine-tuning, became the standard recipe for open text encoders.',
  svg() {
    let s = '';
    s += txt(60, 30, 'THE NOW-STANDARD ENCODER RECIPE', { fs: 12.5, w: 600, fill: PFC_embed.blue });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_embed.blue}" stroke-width="1"/>`;

    // (1) weak pairs from the web
    s += step(56, 74, 1);
    s += box(60, 84, 200, 70, `web-mined text pairs|weak supervision|no human labels`, { fill: PFC_embed.soft });

    // (2) contrastive pretraining
    s += step(316, 74, 2);
    s += box(330, 84, 230, 70, `contrastive pretraining|InfoNCE objective|in-batch negatives`);
    s += aflow(264, 119, 326, 119, { color: PFC_embed.blue, label: 'stage 1' });

    // (3) supervised fine-tune
    s += step(616, 74, 3);
    s += box(630, 84, 210, 70, `supervised fine-tuning|labelled pairs|E5 2022 · GTE 2023`);
    s += arrow(564, 119, 626, 119, { label: 'stage 2' });

    s += apulse(hl(330, 180, 230, 22) +
      txt(445, 196, 'weak pairs first, labels last', { a: 'middle', fs: 13, w: 600 }));

    s += note(630, 184, ['scale comes from cheap web pairs;', 'precision comes from the small', 'supervised pass on top.'],
      { from: [626, 194], to: [564, 191] });

    s += `<line x1="60" y1="238" x2="840" y2="238" stroke="${PFC_embed.grey}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 262, 'the workhorse open baselines — but measure recall on your own labelled queries', { fs: 12.5 });
    s += txt(60, 280, 'before believing any leaderboard: benchmark overlap is endemic.', { fs: 12, fill: PFC_embed.grey });
    return wrap(306, s);
  } },

/* ---------------------------------------------------- 6 · Matryoshka */
{ id: 'matryoshka-nesting',
  paper: 'Matryoshka Representation Learning — Kusupati et al., NeurIPS 2022',
  claim: 'Training losses at nested prefix lengths lets one vector serve every budget — up to 14× smaller embeddings at matched ImageNet-1k accuracy.',
  svg() {
    let s = '';
    s += txt(60, 30, 'ONE VECTOR, EVERY BUDGET', { fs: 12.5, w: 600, fill: PFC_embed.blue });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_embed.blue}" stroke-width="1"/>`;

    // (1) the nested vector: prefixes are themselves embeddings
    s += step(56, 74, 1);
    s += box(60, 88, 780, 40, ` `, { fill: '#fff' });
    s += box(60, 88, 195, 40, `dims 1–256`, { fill: PFC_embed.mark });
    s += box(255, 88, 585, 40, `dims 257–1024`, { fill: PFC_embed.soft });
    s += txt(600, 146, 'loss applied at nested prefix lengths —', { fs: 11.5, fill: PFC_embed.grey });
    s += txt(600, 161, 'every prefix is a working embedding', { fs: 11.5, fill: PFC_embed.grey });

    // (2) two serving paths from the same trained vector
    s += step(56, 186, 2);
    s += box(70, 196, 240, 56, `serve the 256-dim prefix|cheap ANN retrieval`);
    s += aflow(157, 128, 157, 192, { color: PFC_embed.blue });
    s += box(370, 196, 240, 56, `keep all 1024 dims|for rescoring`);
    s += arrow(490, 128, 490, 192, { color: PFC_embed.grey });

    // (3) headline result
    s += step(666, 186, 3);
    s += apulse(hl(680, 196, 160, 26) +
      txt(760, 215, '14× smaller', { a: 'middle', fs: 15, w: 600 }));
    s += txt(760, 240, 'at matched ImageNet-1k', { a: 'middle', fs: 11.5, fill: PFC_embed.grey });
    s += txt(760, 256, 'accuracy (original paper)', { a: 'middle', fs: 11.5, fill: PFC_embed.grey });

    s += note(70, 284, ['truncation degrades gracefully because the loss made every prefix count —', 'vector cost becomes a product decision, not a footnote.'],
      { from: [64, 290], to: [96, 258] });
    return wrap(330, s);
  } },

],

ann: [

/* ------------------------------------------------- 1 · HNSW layered graph */
{ id: 'hnsw-layered-graph',
  paper: 'HNSW — Malkov & Yashunin, arXiv 1603.09320, 2016',
  claim: 'Hierarchical navigable small-world graphs give ~O(log N) greedy search: enter a sparse top layer, hop toward the query, drop a layer, beam-search layer 0.',
  svg() {
    const node = (x, y, f = '#fff') =>
      `<circle cx="${x}" cy="${y}" r="6" fill="${f}" stroke="${INK}" stroke-width="1.5"/>`;
    const edge = (x1, y1, x2, y2) =>
      `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${GREY}" stroke-width="1" opacity="0.55"/>`;
    let s = '';
    s += txt(60, 30, 'HNSW — HIERARCHY OF NAVIGABLE SMALL-WORLD GRAPHS', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // three layer bands
    [['LAYER 2 · sparse, long-range edges', 60],
     ['LAYER 1 · denser', 140],
     ['LAYER 0 · every point lives here', 220]].forEach(([lab, y]) => {
      s += `<rect x="60" y="${y}" width="560" height="64" fill="${SOFT}" stroke="${GREY}" stroke-width="0.75"/>`;
      s += txt(66, y + 14, lab, { fs: 10.5, fill: GREY });
    });

    // layer-2 nodes + edges
    const L2 = [140, 320, 500], y2 = 98;
    s += edge(L2[0], y2, L2[1], y2) + edge(L2[1], y2, L2[2], y2);
    L2.forEach(x => s += node(x, y2));
    // layer-1 nodes + edges
    const L1 = [100, 190, 280, 370, 460, 550], y1 = 178;
    for (let i = 0; i < L1.length - 1; i++) s += edge(L1[i], y1, L1[i + 1], y1);
    s += edge(190, y1, 370, y1 - 1);
    L1.forEach(x => s += node(x, y1));
    // layer-0 nodes + edges
    const L0 = [80, 125, 170, 215, 260, 305, 350, 395, 440, 485, 530, 575], y0 = 258;
    for (let i = 0; i < L0.length - 1; i++) s += edge(L0[i], y0, L0[i + 1], y0);
    L0.forEach(x => s += node(x, y0));

    // search path: enter top, greedy hop, descend, hop, descend, beam
    s += step(122, 74, 1);
    s += aflow(148, y2, 311, y2, { color: BLUE, label: 'greedy hop', sw: 2 });
    s += arrow(324, y2 + 8, 366, y1 - 8, { color: BLUE, dash: '4 3' });
    s += step(348, 132, 2);
    s += arrow(378, y1, 451, y1, { color: BLUE, sw: 2 });
    s += arrow(464, y1 + 8, 526, y0 - 8, { color: BLUE, dash: '4 3', label: 'drop a layer', ly: 2 });
    s += aflow(521, y0, 450, y0, { color: BLUE, label: 'efSearch beam', ly: 38, sw: 2 });
    s += step(590, y0 - 24, 3);
    s += apulse(node(440, y0, MARK));
    s += txt(440, y0 + 24, 'nearest neighbour', { a: 'middle', fs: 10.5, fill: GREY });

    // right column: why + tuning
    s += txt(650, 72, 'WHY ~O(log N)', { fs: 12, w: 600 });
    s += note(650, 92, ['each point’s top layer is drawn', 'from a geometric distribution', '→ O(log N) layers, bounded', 'hops per layer (empirical', 'regularity, not a worst-case', 'theorem)']);
    s += box(650, 196, 200, 58, 'M = 16–32|tune once at build — almost|always right', { fill: '#fff' });
    s += box(650, 264, 200, 58, 'efSearch = runtime dial|measure recall vs brute-force|ground truth, don’t guess', { fill: SOFT });

    // headline
    s += hl(60, 312, 330, 22);
    s += txt(66, 328, 'SEARCH SCALES ≈ O(log N) IN CORPUS SIZE', { fs: 13, w: 600 });
    s += txt(60, 352, 'best recall/QPS tradeoff on CPU when vectors fit in RAM — the default index in nearly every vector database.', { fs: 11.5, fill: GREY });
    return wrap(374, s);
  } },

/* -------------------------------------- 2 · Product quantization (IVF-PQ) */
{ id: 'pq-memory-arithmetic',
  paper: 'Product Quantization — Jégou, Douze & Schmid, TPAMI 2011',
  claim: 'Splitting a 128-dim float32 vector (512 B) into m=8 subvectors, each coded against 256 centroids, stores it in 8 bytes — 64× smaller — with distances estimated by table lookups, never decompressing.',
  svg() {
    let s = '';
    s += txt(60, 30, 'PRODUCT QUANTIZATION — THE MEMORY ARITHMETIC', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) split
    s += box(60, 72, 150, 62, 'x ∈ R¹²⁸|float32|512 bytes', { fill: SOFT });
    s += step(66, 66, 1);
    s += aflow(218, 103, 268, 103, { label: 'split m=8', ly: -10 });
    const subs = ['x₁','x₂','x₃','x₄','x₅','x₆','x₇','x₈'];
    subs.forEach((t, i) => s += box(276 + i * 46, 84, 40, 38, t, { fs: 12 }));

    // (2) quantize each subvector
    s += step(282, 150, 2);
    s += arrow(460, 126, 460, 152, { color: GREY });
    s += box(300, 156, 210, 56, 'codebook Cⱼ per subvector|K = 256 centroids each', { fill: '#fff' });
    s += arrow(514, 184, 560, 184, { label: 'argminₖ' });
    s += box(564, 156, 130, 56, 'code cⱼ|1 byte each', { fill: SOFT });
    s += arrow(698, 184, 726, 184);
    s += hl(728, 154, 128, 60);
    s += box(730, 156, 124, 56, '(c₁ … c₈)|8 BYTES|64× smaller', { sw: 2 });

    // (3) asymmetric distance computation
    s += step(66, 240, 3);
    s += txt(84, 245, 'SEARCH WITHOUT DECOMPRESSING — ADC', { fs: 12, w: 600 });
    s += box(60, 258, 130, 52, 'query q|arrives', { fill: '#fff' });
    s += arrow(194, 284, 236, 284);
    s += box(240, 258, 220, 52, 'build m·K lookup tables|once per query', { fill: SOFT });
    s += arrow(464, 284, 506, 284);
    s += box(510, 258, 250, 52, 'd(q,x)² ≈ Σⱼ tableⱼ[cⱼ]|m lookups per database point', { fill: '#fff' });

    // why it works
    s += note(60, 336, ['effective codebook: K^m = 256⁸ ≈ 1.8×10¹⁹ centroids, but only m·K·(D/m) floats stored — the cross product is free.']);
    s += note(60, 356, ['128-dim SIFT → 8–16 bytes with usable recall; IVFADC (coarse k-means', 'partition + PQ) is the IVF-PQ pattern everyone still ships.']);
    return wrap(382, s);
  } },

/* --------------------------------- 3 · Billion-scale GPUs + Faiss library */
{ id: 'faiss-gpu-scale',
  paper: 'Billion-scale similarity search with GPUs — Johnson, Douze & Jégou 2017 · The Faiss library — Douze et al., arXiv 2401.08281, 2024',
  claim: 'The Faiss line made quantization-based IVF-PQ search run at GPU speed on 10⁹ vectors, and its engineering docs describe a 1.5-trillion-vector index sharded by ID and by inverted list across machines.',
  svg() {
    let s = '';
    s += txt(60, 30, 'FAISS — THE ENGINEERING CANON FOR QUANTIZATION-BASED INDEXES', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // pipeline
    s += box(60, 72, 110, 58, 'query q', { fill: '#fff' });
    s += step(66, 66, 1);
    s += arrow(174, 101, 214, 101);
    s += box(218, 72, 170, 58, 'coarse quantizer|k-means partition (IVF)', { fill: SOFT });
    s += step(224, 66, 2);
    s += aflow(392, 101, 432, 101, { label: 'nprobe lists', ly: 38 });
    s += box(436, 72, 170, 58, 'scan inverted lists|PQ codes + ADC tables', { fill: SOFT });
    s += step(442, 66, 3);
    s += arrow(610, 101, 650, 101);
    s += box(654, 72, 196, 58, 'GPU k-selection|top-k at memory speed', { fill: '#fff' });
    s += note(218, 158, ['the 2017 paper put this whole pipeline on GPUs at billion scale;', 'the 2024 library paper is the written-down engineering canon.']);

    // scale axis
    s += `<line x1="80" y1="240" x2="820" y2="240" stroke="${INK}" stroke-width="1.5" marker-end="url(#ah)"/>`;
    s += txt(826, 244, 'scale', { fs: 11, fill: GREY });
    s += `<line x1="220" y1="233" x2="220" y2="247" stroke="${INK}" stroke-width="1.5"/>`;
    s += txt(220, 264, '10⁹ vectors', { a: 'middle', fs: 12.5, w: 600 });
    s += txt(220, 280, 'GPUs, 2017 paper', { a: 'middle', fs: 11, fill: GREY });
    s += `<line x1="640" y1="233" x2="640" y2="247" stroke="${INK}" stroke-width="1.5"/>`;
    s += hl(548, 250, 186, 20);
    s += apulse(txt(640, 264, '1.5 TRILLION vectors', { a: 'middle', fs: 13, w: 600 }));
    s += txt(640, 280, 'Faiss engineering docs (Meta):', { a: 'middle', fs: 11, fill: GREY });
    s += txt(640, 294, 'sharded by ID and by inverted list across machines', { a: 'middle', fs: 11, fill: GREY });
    s += note(80, 322, ['at that size the index IS a distributed system — and quantization is not optional.']);
    return wrap(344, s);
  } },

/* --------------------------------------------- 4 · DiskANN billion-on-box */
{ id: 'diskann-billion-one-box',
  paper: 'DiskANN — Subramanya et al., NeurIPS 2019',
  claim: 'One billion points served from a 64 GB workstation with an NVMe SSD: 5000+ QPS at under 3 ms mean latency with 95%+ 1-recall@1, where equal-memory IVF baselines plateaued near 50%.',
  svg() {
    let s = '';
    s += txt(60, 30, 'DISKANN — ONE BILLION POINTS ON ONE 64 GB WORKSTATION', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // the machine
    s += `<rect x="60" y="64" width="490" height="226" rx="2" fill="none" stroke="${INK}" stroke-width="2"/>`;
    s += txt(72, 84, 'ONE COMMODITY WORKSTATION · 64 GB RAM · NVMe SSD', { fs: 11.5, w: 600 });
    s += box(80, 100, 200, 78, 'RAM (64 GB)|compressed (PQ) vectors|steer the traversal', { fill: SOFT });
    s += box(320, 100, 210, 110, 'NVMe SSD|graph (billion nodes)|+ full-precision vectors', { fill: '#fff' });
    s += step(86, 96, 1);
    s += step(326, 96, 2);
    s += aflow(284, 126, 316, 126, { color: BLUE, label: 'walk', ly: -8 });
    s += arrow(316, 156, 284, 156, { color: GREY, label: 'fetch', ly: 14 });
    s += txt(80, 200, 'traverse graph greedily using cheap', { fs: 10.5, fill: GREY });
    s += txt(80, 214, 'compressed distances held in RAM;', { fs: 10.5, fill: GREY });
    s += txt(80, 228, 'read each node’s neighbourhood off SSD', { fs: 10.5, fill: GREY });
    s += step(86, 252, 3);
    s += box(104, 238, 300, 40, 're-rank candidates with full vectors from SSD', { fill: SOFT });
    s += arrow(404, 258, 440, 258, { label: 'top-k', ly: -8 });

    // headline numbers
    s += hl(576, 66, 268, 22);
    s += txt(582, 82, '5000+ QPS · < 3 ms MEAN LATENCY', { fs: 13.5, w: 600 });

    // recall bars at equal memory
    s += txt(600, 122, '1-RECALL@1 AT EQUAL MEMORY', { fs: 11.5, w: 600 });
    s += `<line x1="600" y1="270" x2="844" y2="270" stroke="${INK}" stroke-width="1.5"/>`;
    s += apulse(`<rect x="622" y="128" width="76" height="142" fill="${MARK}" stroke="${INK}" stroke-width="1.5"/>`);
    s += txt(660, 150, '95%+', { a: 'middle', fs: 14, w: 600 });
    s += txt(660, 286, 'DiskANN', { a: 'middle', fs: 11.5 });
    s += `<rect x="740" y="195" width="76" height="75" fill="${SOFT}" stroke="${GREY}" stroke-width="1.5"/>`;
    s += txt(778, 216, '≈50%', { a: 'middle', fs: 13, fill: GREY });
    s += txt(778, 286, 'IVF baselines', { a: 'middle', fs: 11.5, fill: GREY });
    s += note(600, 312, ['still the reference point for', 'cost-per-vector arguments; ships as the', 'vector index in Azure database products.']);
    return wrap(348, s);
  } },

/* ------------------------------------------ 5 · ScaNN anisotropic quantization */
{ id: 'scann-anisotropic',
  paper: 'ScaNN — Guo et al., ICML 2020',
  claim: 'Anisotropic quantization penalizes error parallel to the vector direction more than orthogonal error — the parallel part is what corrupts inner products — and took the top of ann-benchmarks glove-1.2M on publication.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SCANN — ANISOTROPIC VECTOR QUANTIZATION', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // geometry: x, quantized x~, error split
    const ox = 110, oy = 268;
    s += `<line x1="${ox - 30}" y1="${oy}" x2="${ox + 330}" y2="${oy}" stroke="${GREY}" stroke-width="0.75"/>`;
    s += `<line x1="${ox}" y1="${oy + 20}" x2="${ox}" y2="${oy - 190}" stroke="${GREY}" stroke-width="0.75"/>`;
    s += arrow(ox, oy, 350, 108, { sw: 2.5 });                       // x
    s += txt(358, 102, 'x  (database vector)', { fs: 12, w: 600 });
    s += `<circle cx="292" cy="168" r="5" fill="${BLUE}"/>`;         // x~
    s += txt(300, 186, 'x̃ = quantized x', { fs: 11.5, fill: BLUE });
    s += step(70, 84, 1);
    s += txt(88, 89, 'quantize x → x̃', { fs: 11.5 });
    // error r and its decomposition
    s += arrow(292, 168, 336, 118, { color: GREY, dash: '3 3' });
    s += txt(322, 152, 'r = x − x̃', { fs: 11, fill: GREY });
    s += aflow(292, 168, 318, 137, { color: RED, sw: 2 });           // parallel comp
    s += txt(216, 128, 'r∥ — along x', { fs: 11.5, fill: RED, w: 600 });
    s += arrow(292, 168, 316, 190, { color: GREEN, sw: 2 });
    s += txt(322, 208, 'r⊥ — orthogonal', { fs: 11.5, fill: GREEN });
    s += step(70, 130, 2);
    s += txt(88, 135, 'split the error', { fs: 11.5 });

    // the reweighted loss
    s += step(506, 84, 3);
    s += box(524, 66, 320, 62, 'codebook loss = w∥·‖r∥‖² + w⊥·‖r⊥‖²|with w∥ > w⊥ — parallel error costs more', { fill: SOFT });
    s += note(524, 152, ['WHY: parallel error is what corrupts the', 'inner product ⟨q, x⟩ — the quantity MIPS', 'retrieval actually ranks by. Orthogonal', 'error mostly cancels against the query.'], {});
    s += `<path d="M520,110 L500,110" stroke="${GREY}" stroke-width="1" stroke-dasharray="2 3"/>`;

    // headline
    s += hl(524, 232, 320, 22);
    s += txt(530, 248, 'TOP OF ANN-BENCHMARKS GLOVE-1.2M', { fs: 13, w: 600 });
    s += txt(524, 270, 'on publication (ICML 2020); the same lineage now runs', { fs: 11, fill: GREY });
    s += txt(524, 284, 'under Vertex AI Vector Search at Google.', { fs: 11, fill: GREY });
    return wrap(316, s);
  } },

/* ------------------------------------------------------- 6 · RaBitQ */
{ id: 'rabitq-rotate-binarize',
  paper: 'RaBitQ — Gao & Long, SIGMOD 2024',
  claim: 'Randomly rotate the vector, then binarize to one bit per coordinate: the estimator comes with a sharp theoretical error bound, and the paper reports beating PQ variants on the accuracy–efficiency tradeoff.',
  svg() {
    let s = '';
    s += txt(60, 30, 'RABITQ — ROTATE, THEN BINARIZE, WITH A GUARANTEE', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // pipeline
    s += box(60, 76, 130, 58, 'x ∈ Rᴰ|float vector', { fill: '#fff' });
    s += step(66, 70, 1);
    s += aflow(194, 105, 236, 105, { label: 'P·x', ly: -9 });
    s += box(240, 76, 160, 58, 'random rotation P|scrambles coordinates', { fill: SOFT });
    s += step(246, 70, 2);
    s += arrow(404, 105, 446, 105, { label: 'sign(·)' });
    s += box(450, 76, 160, 58, 'binarize|1 bit per coordinate', { fill: SOFT });
    s += step(456, 70, 3);
    s += arrow(614, 105, 656, 105);
    s += hl(658, 74, 192, 62);
    s += box(660, 76, 188, 58, 'distance estimator|with a SHARP theoretical|error bound', { sw: 2 });
    s += note(240, 158, ['WHY rotate first: randomizing the coordinates is what', 'makes the per-query error bound provable — the bound,', 'not the bits, is the paper’s contribution.']);

    // accuracy–efficiency mini-plot
    const px = 90, py = 316, pw = 300, ph = 96;
    s += `<line x1="${px}" y1="${py}" x2="${px + pw}" y2="${py}" stroke="${INK}" stroke-width="1.25" marker-end="url(#ah)"/>`;
    s += `<line x1="${px}" y1="${py}" x2="${px}" y2="${py - ph}" stroke="${INK}" stroke-width="1.25" marker-end="url(#ah)"/>`;
    s += txt(px + pw + 8, py + 4, 'efficiency', { fs: 10.5, fill: GREY });
    s += txt(px - 4, py - ph - 8, 'accuracy', { fs: 10.5, fill: GREY });
    s += `<path d="M${px + 16},${py - 78} C ${px + 120},${py - 74} ${px + 200},${py - 56} ${px + 280},${py - 26}" fill="none" stroke="${BLUE}" stroke-width="2"/>`;
    s += txt(px + 150, py - 80, 'RaBitQ', { fs: 11.5, fill: BLUE, w: 600 });
    s += `<path d="M${px + 16},${py - 58} C ${px + 120},${py - 52} ${px + 200},${py - 34} ${px + 280},${py - 8}" fill="none" stroke="${GREY}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    s += txt(px + 150, py - 12, 'PQ variants', { fs: 11, fill: GREY });

    // headline + context
    s += hl(470, 240, 330, 22);
    s += apulse(txt(476, 256, 'BEATS PQ VARIANTS ON ACCURACY–EFFICIENCY', { fs: 12.5, w: 600 }));
    s += txt(470, 280, 'as reported in the paper — and it is driving the current', { fs: 11, fill: GREY });
    s += txt(470, 294, 'quantization-first turn in vector search.', { fs: 11, fill: GREY });
    return wrap(348, s);
  } },

],

attention: [

/* ---------------------------------------------- 1 · Vaswani et al. 2017 */
{ id: 'vaswani-qkv-flow',
  paper: 'Attention Is All You Need — Vaswani et al., 2017',
  claim: 'Dropping recurrence for scaled dot-product attention hit 28.4 BLEU on WMT14 EN→DE with the big model trained in only 3.5 days on 8 GPUs.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SCALED DOT-PRODUCT ATTENTION · Attention(Q,K,V) = softmax(QKᵀ/√d_k)V', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) input and projections
    s += box(60, 96, 130, 70, 'INPUT X|n tokens × d_model', { fill: SOFT });
    s += step(64, 90, 1);
    s += box(250, 62, 104, 40, 'Q = XW_Q', { fs: 12 });
    s += box(250, 112, 104, 40, 'K = XW_K', { fs: 12 });
    s += box(250, 162, 104, 40, 'V = XW_V', { fs: 12 });
    s += arrow(190, 112, 242, 82);
    s += arrow(190, 131, 242, 132);
    s += arrow(190, 150, 242, 182);

    // (2) scores → softmax
    s += box(414, 72, 150, 90, 'QKᵀ / √d_k|n×n scores', { fill: SOFT });
    s += step(418, 66, 2);
    s += aflow(354, 82, 406, 100);
    s += arrow(354, 132, 406, 122);
    s += box(620, 72, 122, 90, 'softmax|rows sum to 1');
    s += aflow(564, 117, 612, 117);

    // (3) weighted sum of V
    s += box(560, 208, 170, 54, 'weights · V|context per token');
    s += step(564, 202, 3);
    s += arrow(681, 162, 660, 200);
    s += arrow(354, 182, 552, 230, { label: 'V', ly: -8 });
    s += box(790, 208, 74, 54, 'OUT|n × d_v');
    s += arrow(730, 235, 782, 235);

    // why √d_k — the one-line fix
    s += note(60, 210, [
      'why √d_k: var(q·k) grows like d_k, so',
      'unscaled logits blow up with head size —',
      'softmax saturates one-hot, gradients vanish.',
      'Dividing by √d_k pins logit variance at 1.'
    ], { from: [340, 220], to: [420, 170] });

    s += txt(60, 300, 'no recurrence: every position attends to every other in one matmul — the whole sequence trains in parallel.', { fs: 12, fill: GREY });
    s += hl(60, 316, 610, 20);
    s += txt(64, 331, '28.4 BLEU on WMT14 EN→DE · big model trained in 3.5 days on 8 GPUs', { fs: 12.5, w: 600 });
    s += txt(60, 356, 'the efficiency claim aged even better than the quality claim.', { fs: 11.5, fill: GREY });
    return wrap(380, s);
  } },

/* ---------------------------------------------- 2 · FlashAttention 2022 */
{ id: 'flash-attention-tiling',
  paper: 'FlashAttention — Dao et al., NeurIPS 2022',
  claim: 'IO-aware tiling computes exact attention blockwise in on-chip SRAM, never materialising the n×n matrix in HBM — memory linear in n, up to ~3× training speedup.',
  svg() {
    let s = '';
    s += txt(60, 30, 'IO-AWARE EXACT ATTENTION · zero FLOPs changed, zero maths changed', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // left: standard attention writes n×n to HBM
    s += txt(60, 66, 'STANDARD ATTENTION', { fs: 12, w: 600 });
    s += box(60, 78, 170, 120, '', { stroke: RED });
    s += txt(145, 96, 'GPU HBM', { a: 'middle', fs: 12.5, w: 600 });
    s += txt(145, 111, 'large, slow', { a: 'middle', fs: 10.5, fill: GREY });
    s += box(84, 118, 122, 62, 'n×n matrix|materialised', { stroke: RED, fill: '#fff', fs: 11.5 });
    s += txt(60, 224, 'memory O(n²) · reads and re-writes', { fs: 11.5, fill: RED });
    s += txt(60, 240, 'the full score matrix every pass', { fs: 11.5, fill: RED });

    // right: flash tiles in SRAM
    s += txt(352, 66, 'FLASHATTENTION', { fs: 12, w: 600 });
    s += box(330, 78, 190, 120, 'GPU HBM|holds only Q, K, V, O', { fill: SOFT });
    s += box(600, 78, 190, 120, '', { fill: SOFT });
    s += txt(695, 96, 'SRAM', { a: 'middle', fs: 12.5, w: 600 });
    s += txt(695, 111, 'small, fast, on-chip', { a: 'middle', fs: 10.5, fill: GREY });
    for (let i = 0; i < 3; i++)
      s += box(620 + i * 52, 130, 44, 40, 'tile', { fs: 10.5, fill: '#fff' });
    s += step(334, 72, 1);
    s += aflow(520, 118, 592, 118, { label: 'load tile', color: BLUE });
    s += step(604, 72, 2);
    s += txt(695, 182, 'blockwise softmax,', { a: 'middle', fs: 10.5, fill: GREY });
    s += txt(695, 194, 'rescaled on the fly', { a: 'middle', fs: 10.5, fill: GREY });
    s += step(334, 214, 3);
    s += arrow(592, 178, 520, 178, { label: 'write O only', color: BLUE, ly: 14 });
    s += txt(330, 232, 'the n×n matrix is never written to HBM —', { fs: 11.5, fill: GREY });
    s += txt(330, 248, 'memory linear in sequence length', { fs: 11.5, fill: GREY });

    s += note(660, 232, ['exact attention:', 'same output bits,', 'fewer memory trips'], {});

    s += hl(60, 282, 620, 20);
    s += txt(64, 297, 'exact attention · memory linear in n · up to ~3× end-to-end training speedup', { fs: 12.5, w: 600 });
    s += txt(60, 324, 'the lesson: the bottleneck was memory traffic, not arithmetic.', { fs: 12, fill: GREY });
    return wrap(348, s);
  } },

/* ------------------------------------------------- 3 · Chinchilla 2022 */
{ id: 'chinchilla-20-tokens',
  paper: 'Chinchilla scaling laws — Hoffmann et al., 2022',
  claim: 'Compute-optimal training wants roughly 20 tokens per parameter — a 70B model on 1.4T tokens beat much larger under-trained peers.',
  svg() {
    let s = '';
    s += txt(60, 30, 'COMPUTE-OPTIMAL ALLOCATION · one budget C, two dials: params N and tokens D', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    s += box(60, 90, 170, 66, 'FIXED COMPUTE C|loss follows power laws', { fill: SOFT });
    s += step(64, 84, 1);
    s += txt(60, 188, 'Kaplan et al. (2020) fit the power', { fs: 11.5, fill: GREY });
    s += txt(60, 204, 'laws; Chinchilla corrected the split.', { fs: 11.5, fill: GREY });

    // two allocations
    s += box(330, 66, 250, 60, 'BIGGER MODEL, FEWER TOKENS|under-trained — wasted capacity', { stroke: RED });
    s += box(330, 156, 250, 60, 'SMALLER MODEL, MORE TOKENS|compute-optimal split', { stroke: GREEN, fill: SOFT });
    s += arrow(230, 108, 322, 96, { color: GREY, dash: '3 3' });
    s += aflow(230, 130, 322, 186, { color: BLUE });
    s += step(334, 150, 2);

    // the ratio, concrete
    s += apulse(box(650, 140, 200, 76, '≈ 20 TOKENS|per parameter', { stroke: BLUE, sw: 2, fill: MARK }));
    s += aflow(580, 186, 642, 180, { color: BLUE });
    s += step(654, 134, 3);

    s += hl(60, 250, 620, 20);
    s += txt(64, 265, 'Chinchilla 70B × 1.4T tokens beat much larger under-trained peers', { fs: 12.5, w: 600 });

    s += note(60, 296, [
      'why production overshoots the ratio: both papers ignored inference cost,',
      'which favours smaller models trained far past 20 tokens per parameter.'
    ], {});
    return wrap(330, s);
  } },

/* ------------------------------------- 4 · Speculative decoding 2023 */
{ id: 'speculative-decoding',
  paper: 'Speculative decoding — Leviathan et al., 2023',
  claim: 'A small draft model proposes tokens and the big model verifies them in one parallel pass; rejection sampling keeps the output distribution exactly unchanged, for a 2–3× decode speedup.',
  svg() {
    let s = '';
    s += txt(60, 30, 'DRAFT, THEN VERIFY IN PARALLEL · decode is bandwidth-bound, so verification rides almost free', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) draft proposes
    s += box(60, 84, 150, 66, 'DRAFT MODEL|small, fast, serial', { fill: SOFT });
    s += step(64, 78, 1);
    const toks = ['t₁', 't₂', 't₃', 't₄', 't₅'];
    toks.forEach((t, i) => {
      s += box(280 + i * 62, 96, 52, 42, t, { fs: 12 });
    });
    s += aflow(210, 117, 272, 117, { label: 'proposes k tokens', ly: -37 });

    // (2) target verifies all at once
    s += box(680, 84, 170, 66, 'TARGET MODEL|one parallel pass', { stroke: BLUE, sw: 2 });
    s += step(684, 78, 2);
    s += aflow(600, 117, 672, 117, { color: BLUE });

    // (3) accept / reject
    s += step(64, 190, 3);
    s += txt(84, 195, 'rejection sampling per token — output distribution exactly the target model’s:', { fs: 12 });
    toks.forEach((t, i) => {
      const ok = i < 3;
      s += box(280 + i * 62, 210, 52, 42, t, { fs: 12, stroke: ok ? GREEN : RED, fill: ok ? SOFT : '#fff' });
      s += txt(306 + i * 62, 270, ok ? 'keep' : (i === 3 ? 'reject' : 'drop'), { a: 'middle', fs: 10.5, fill: ok ? GREEN : RED });
    });
    s += txt(660, 236, '→ resample from target at t₄', { fs: 11.5, fill: RED });

    s += note(660, 176, ['big model reads its weights', 'once to score k tokens,', 'not once per token'], {});

    s += hl(60, 296, 520, 20);
    s += txt(64, 311, 'reported 2–3× decode speedup · output distribution exactly unchanged', { fs: 12.5, w: 600 });
    return wrap(336, s);
  } },

/* ---------------------------------------------- 5 · StreamingLLM 2023 */
{ id: 'streamingllm-attention-sinks',
  paper: 'StreamingLLM — Xiao et al., 2023',
  claim: 'Keeping just ~4 initial "attention sink" tokens plus a sliding window preserves quality on unbounded streams — softmax parks its mass on the first tokens.',
  svg() {
    let s = '';
    s += txt(60, 30, 'ATTENTION SINKS + SLIDING WINDOW · a KV cache that never grows', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // token strip
    s += txt(84, 76, 'KV cache over an unbounded stream:', { fs: 12 });
    // (1) sinks
    for (let i = 0; i < 4; i++)
      s += box(60 + i * 48, 90, 40, 40, `t${i}`, { fs: 11, fill: MARK });
    s += apulse(hl(58, 88, 196, 44));
    s += step(64, 84, 1);
    s += txt(60, 152, '~4 “sink” tokens, kept forever', { fs: 11, fill: GREY });

    // (2) evicted middle
    s += `<rect x="280" y="90" width="260" height="40" fill="none" stroke="${GREY}" stroke-width="1.2" stroke-dasharray="4 4" rx="2"/>`;
    s += txt(410, 114, 'middle tokens evicted', { a: 'middle', fs: 11.5, fill: GREY });
    s += step(284, 84, 2);

    // (3) sliding window
    const wins = ['tₙ₋₄', 'tₙ₋₃', 'tₙ₋₂', 'tₙ₋₁', 'tₙ'];
    wins.forEach((t, i) => { s += box(570 + i * 48, 90, 40, 40, t, { fs: 10, fill: SOFT }); });
    s += step(574, 84, 3);
    s += txt(570, 152, 'sliding window of recent tokens', { fs: 11, fill: GREY });
    s += aflow(818, 110, 856, 110, { label: '', color: BLUE });
    s += txt(826, 148, 'stream →', { fs: 10.5, fill: BLUE });

    // why it works
    s += note(60, 190, [
      'why sinks: softmax must put its attention mass somewhere;',
      'the model learns to park it on the first tokens. Evict them',
      'and quality collapses — keep 4 and the stream runs forever.'
    ]);

    s += hl(60, 258, 640, 20);
    s += txt(64, 273, '~4 sink tokens + sliding window preserves quality on unbounded streams', { fs: 12.5, w: 600 });
    s += txt(60, 300, 'cache stays constant-size while context length is effectively infinite.', { fs: 11.5, fill: GREY });
    return wrap(324, s);
  } },

],

seqrec: [

/* ---------------------------------------------------------------- SASRec */
{ id: 'sasrec-causal-attention',
  paper: 'SASRec — Kang & McAuley, ICDM 2018',
  claim: 'Causal self-attention over the raw item sequence, trained with next-item cross-entropy, is still the baseline every sequential recommender is measured against.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SASREC · CAUSAL SELF-ATTENTION OVER ITEM IDS', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) the ordered history, oldest → newest
    const ix = [60, 154, 248, 342, 436];
    const ilab = ['i₁|oldest', 'i₂', 'i₃', 'i₄', 'i₅|newest'];
    ix.forEach((x, k) => {
      s += box(x, 300, 82, 44, ilab[k], { fill: SOFT, fs: 12.5 });
      s += arrow(x + 41, 292, x + 41, 254, { color: GREY, sw: 1.2 });
    });
    s += step(48, 292, 1);
    s += txt(60, 370, 'user history = ordered item IDs + position embedding', { fs: 11.5, fill: GREY });

    // (2) causal attention row — each state sees only its past
    ix.forEach((x, k) => {
      s += box(x, 206, 82, 44, `h${['₁','₂','₃','₄','₅'][k]}`, { fs: 12.5 });
    });
    s += step(48, 198, 2);
    // causal edges into h₅ from every earlier state
    s += arrow(101, 206, 462, 176, { color: GREY, sw: 1.1, curve: 60 });
    s += arrow(289, 206, 468, 182, { color: GREY, sw: 1.1, curve: 40 });
    s += arrow(383, 206, 472, 190, { color: GREY, sw: 1.1, curve: 24 });
    s += txt(300, 168, 'causal mask: hₜ attends only to i₁ … iₜ', { fs: 11.5, fill: GREY });

    // (3) prediction head — dot with every item embedding, softmax
    s += aflow(522, 228, 596, 228, { label: 'hₜ' });
    s += box(600, 200, 118, 56, 'hₜ · E|score every item', { fs: 12.5 });
    s += arrow(722, 228, 756, 228);
    s += box(760, 200, 92, 56, 'softmax|→ p(i₆)', { fill: MARK, fs: 12.5 });
    s += step(590, 192, 3);

    // training signal
    s += hl(600, 286, 252, 20);
    s += txt(604, 301, 'L = −Σₜ log p(i_{t+1} | i₁…iₜ)', { fs: 12.5, w: 600 });
    s += txt(600, 326, 'every prefix predicts its next item:', { fs: 11.5, fill: GREY });
    s += txt(600, 341, 'one pass = T training signals', { fs: 11.5, fill: GREY });

    // why it persists
    s += note(60, 100, ['WHY IT REFUSES TO DIE: one model, one objective —',
                        'serving is nearest-neighbour retrieval over the final',
                        'hidden state hₜ. Everything since is a delta against it.']);
    return wrap(400, s);
  } },

/* ------------------------------------------------- BERT4Rec replicability */
{ id: 'bert4rec-30x-repro',
  paper: 'BERT4Rec — Sun et al., CIKM 2019 · replicability: Petrov & Macdonald, RecSys 2022',
  claim: 'BERT4Rec did not reproduce with its default configuration — matching the published numbers took up to 30× the default training time.',
  svg() {
    let s = '';
    s += txt(60, 30, 'BERT4REC · MASKED-ITEM TRAINING, AND THE REPRODUCIBILITY BILL', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) the mechanism: cloze objective, bidirectional context
    const bx = [60, 138, 216, 294, 372];
    const blab = ['i₁', '[MASK]', 'i₃', 'i₄', '[MASK]'];
    bx.forEach((x, k) => {
      s += box(x, 92, 66, 42, blab[k], { fill: blab[k] === '[MASK]' ? MARK : SOFT, fs: 12 });
    });
    s += step(48, 84, 1);
    // bidirectional arrows into the first mask — left AND right context
    s += arrow(126, 158, 165, 142, { color: GREY, sw: 1.2 });
    s += arrow(288, 158, 213, 142, { color: GREY, sw: 1.2 });
    s += txt(60, 176, 'bidirectional: a mask is predicted from BOTH sides (no causal mask)', { fs: 11.5, fill: GREY });
    s += step(48, 208, 2);
    s += txt(66, 213, 'claim in the 2019 paper: beats SASRec on the standard benchmarks', { fs: 12 });

    // (2) the rerun — training-time bars
    s += step(48, 256, 3);
    s += txt(66, 261, 'PETROV & MACDONALD RERUN THE BASELINE', { fs: 12, w: 600 });
    // dashed target line = published number
    s += `<line x1="620" y1="278" x2="620" y2="366" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(628, 290, 'published result', { fs: 11, fill: GREY });
    // bar 1: default config falls short
    s += `<rect x="66" y="282" width="300" height="24" fill="${SOFT}" stroke="${INK}" stroke-width="1.2"/>`;
    s += txt(74, 299, 'default config · default training time', { fs: 11.5 });
    s += txt(376, 299, 'falls short', { fs: 11.5, fill: RED });
    // bar 2: 30× training reaches it
    s += `<rect x="66" y="316" width="554" height="24" fill="#fff" stroke="${INK}" stroke-width="1.2"/>`;
    s += hl(70, 320, 190, 16);
    s += txt(74, 333, 'up to 30× the default training time', { fs: 11.5, w: 600 });
    s += txt(376, 333, 'matches the paper', { fs: 11.5, fill: GREEN });
    s += aflow(366, 294, 560, 294, { color: BLUE, label: 'train longer' });

    // the consequence
    s += note(650, 310, ['CONSEQUENCE: half the literature',
                         'comparing BERT4Rec to SASRec',
                         'compared an undertrained model.',
                         'Rerun baselines yourself.']);
    s += apulse(txt(66, 366, 'reproducibility is a training-budget question, not a code question', { fs: 12, fill: BLUE, w: 600 }));
    return wrap(392, s);
  } },

/* ------------------------------------------------------- HSTU / M-FALCON */
{ id: 'hstu-generative-recsys',
  paper: 'Actions Speak Louder than Words (HSTU) — Zhai et al., ICML 2024',
  claim: 'A 1.5T-parameter generative recommender served with M-FALCON micro-batching delivered +12.4% in online A/B tests, with quality scaling as a power law of training compute.',
  svg() {
    let s = '';
    s += txt(60, 30, 'HSTU · THE RANKER AS A SEQUENCE MODEL, SERVED VIA M-FALCON', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) interleaved token stream
    s += box(60, 78, 190, 62, 'TOKEN STREAM|items ⊕ actions|interleaved', { fill: SOFT });
    s += step(52, 70, 1);
    s += aflow(254, 109, 318, 109);

    // (2) HSTU block — no softmax, time-aware bias, gate
    s += box(322, 70, 226, 78, 'HSTU BLOCK ×L|φ(QKᵀ + rabᵖ·ᵗ)·V  — no softmax|Norm(AV) ⊙ U(X) gate', {});
    s += step(314, 62, 2);
    s += txt(322, 168, 'φ = SiLU, pointwise: attention keeps its', { fs: 11.5, fill: GREY });
    s += txt(322, 183, 'MAGNITUDE — repeat-intensity is signal,', { fs: 11.5, fill: GREY });
    s += txt(322, 198, 'softmax would renormalize it away', { fs: 11.5, fill: GREY });
    s += arrow(552, 109, 616, 109);

    // (3) M-FALCON serving
    s += box(620, 70, 230, 78, 'M-FALCON|user-history pass computed ONCE|+ m candidates micro-batched', {});
    s += step(612, 62, 3);
    s += txt(620, 168, 'scoring m candidates ≈ one', { fs: 11.5, fill: GREY });
    s += txt(620, 183, 'sequence pass + m cheap extensions', { fs: 11.5, fill: GREY });
    s += txt(620, 198, '(1024–16384 candidates)', { fs: 11.5, fill: GREY });

    // headline results band
    s += `<line x1="60" y1="212" x2="840" y2="212" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += apulse(hl(60, 228, 402, 24) +
      txt(66, 245, '+12.4% online A/B · 1.5T params · billions of users', { fs: 13.5, w: 600 }));
    s += txt(60, 282, 'offline: up to 65.8% better NDCG than baselines', { fs: 12 });
    s += txt(60, 302, 'kernel: 5.3–15.2× faster than FlashAttention2 transformers at length 8192', { fs: 12 });
    s += txt(60, 322, 'serving: 285× the FLOPs of the DLRM it replaced, yet 1.50–2.48× HIGHER throughput', { fs: 12 });

    s += note(660, 258, ['THE IMPORTANT CLAIM: quality',
                         'scales as a power law of',
                         'training compute across three',
                         'orders of magnitude — compute',
                         'becomes a purchasable lever',
                         'for recsys, which it never',
                         'reliably was.']);
    return wrap(360, s);
  } },

/* ------------------------------------------------------- sampled metrics */
{ id: 'sampled-metrics-reversal',
  paper: 'On Sampled Metrics for Item Recommendation — Krichene & Rendle, KDD 2020',
  claim: 'Ranking the true item against 100 sampled negatives instead of the full catalogue can reverse which model looks better — offline leaderboards can be fiction.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SAMPLED METRICS · WHY YOUR OFFLINE NUMBERS MAY BE FICTION', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) full evaluation
    s += step(52, 72, 1);
    s += txt(66, 77, 'FULL EVALUATION', { fs: 12, w: 600 });
    s += box(60, 90, 240, 56, 'rank true next item|against the ENTIRE catalogue|(all N−1 negatives)', { fill: SOFT, fs: 12 });
    s += txt(60, 168, 'verdict:  model A  >  model B', { fs: 12.5, w: 600, fill: GREEN });

    // (2) sampled evaluation
    s += step(52, 210, 2);
    s += txt(66, 215, 'SAMPLED EVALUATION (the common shortcut)', { fs: 12, w: 600 });
    s += box(60, 228, 240, 56, 'rank true next item|against 100 random|sampled negatives', { fs: 12 });
    s += txt(60, 306, 'verdict:  model B  >  model A', { fs: 12.5, w: 600, fill: RED });

    // the reversal, drawn as crossing flows
    s += aflow(310, 118, 470, 118, { color: BLUE, label: 'same models' });
    s += aflow(310, 256, 470, 256, { color: BLUE, label: 'same models' });
    s += `<path d="M480,130 L560,244" stroke="${RED}" stroke-width="1.5" fill="none" marker-end="url(#ah)"/>`;
    s += `<path d="M480,244 L560,130" stroke="${RED}" stroke-width="1.5" fill="none" marker-end="url(#ah)"/>`;
    s += apulse(hl(474, 176, 200, 22) + txt(480, 192, 'ORDERING CAN REVERSE', { fs: 13, w: 600 }));

    // (3) the fix
    s += step(52, 344, 3);
    s += txt(66, 349, 'FIX: evaluate on the full catalogue, or apply the paper’s corrections — and say which you did', { fs: 12 });

    s += note(620, 90, ['WHY: sampled rank is not a noisy',
                        'version of the true rank. As the',
                        'sample shrinks, every metric',
                        '(HR@k, NDCG@k, MRR…) collapses',
                        'toward AUC — fine-grained top-k',
                        'differences are erased.']);
    s += note(620, 290, ['same trap at training time: sampled',
                         'softmax needs the logQ correction',
                         'or popular items get suppressed.']);
    return wrap(376, s);
  } },

/* ---------------------------------------------------------- TransAct V2 */
{ id: 'transact-v2-lifelong',
  paper: 'TransAct V2 — Pinterest, 2025 (arXiv 2506.02267)',
  claim: 'Lifelong action sequences plus a next-action auxiliary loss inside an existing pointwise CTR ranker lifted engagement volume and diversity online — no generative re-platform needed.',
  svg() {
    let s = '';
    s += txt(60, 30, 'TRANSACT V2 · SEQUENCE AS A FEATURE, NOT A RE-PLATFORM', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) lifelong action sequence
    s += box(60, 80, 200, 64, 'LIFELONG ACTION|SEQUENCE|clicks · saves · hides…', { fill: SOFT });
    s += step(52, 72, 1);
    s += txt(60, 164, 'V1 (KDD 2023): realtime actions only —', { fs: 11.5, fill: GREY });
    s += txt(60, 179, 'V2 extends to the user’s lifelong history', { fs: 11.5, fill: GREY });
    s += aflow(264, 112, 336, 112);

    // (2) modest transformer encoder + auxiliary loss
    s += box(340, 80, 196, 64, 'SEQUENCE|TRANSFORMER|pooled user state', {});
    s += step(332, 72, 2);
    s += arrow(438, 148, 438, 186, { color: BLUE, sw: 1.4 });
    s += box(340, 190, 196, 46, 'AUX HEAD|next-action loss', { fill: MARK, fs: 12 });
    s += txt(340, 258, 'auxiliary objective teaches the', { fs: 11.5, fill: GREY });
    s += txt(340, 273, 'encoder sequence dynamics the', { fs: 11.5, fill: GREY });
    s += txt(340, 288, 'CTR label alone won’t', { fs: 11.5, fill: GREY });
    s += aflow(540, 112, 612, 112);

    // (3) existing pointwise CTR ranker
    s += box(616, 66, 234, 92, 'EXISTING POINTWISE|CTR RANKER|sequence state = one input|among many features', {});
    s += step(608, 58, 3);
    s += box(616, 190, 110, 42, 'other|features', { fill: SOFT, fs: 11.5 });
    s += arrow(671, 190, 671, 162, { color: GREY, sw: 1.2 });
    s += txt(744, 216, '→ p(click)', { fs: 12.5, w: 600 });

    // headline
    s += hl(60, 300, 470, 22);
    s += txt(66, 316, 'online A/B: gains in engagement VOLUME and DIVERSITY (Homefeed)', { fs: 12.5, w: 600 });

    s += note(616, 258, ['NOTE WHAT THEY DID NOT DO:',
                         'replace the ranker with a',
                         'generative model. The pragmatic',
                         'counterpoint to Meta’s HSTU',
                         'reframing — ship this first.']);
    return wrap(360, s);
  } },

],

sid: [

/* ------------------------------------------------- TIGER, NeurIPS 2023 */
{ id: 'tiger-generative-retrieval',
  paper: 'TIGER — Rajput et al., NeurIPS 2023',
  claim: 'Decoding RQ-VAE semantic IDs autoregressively replaces the ANN index and lifts Recall@5 by roughly 17% over the strongest baselines.',
  svg() {
    let s = '';
    s += txt(60, 30, 'TIGER · RQ-VAE SEMANTIC IDS + GENERATIVE RETRIEVAL', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) index time: tokenize every item once
    s += step(48, 76, 1);
    s += txt(64, 68, 'index time · tokenize every item once', { fs: 11.5, fill: GREY });
    s += box(60, 88, 150, 50, 'item content|title, description', { fill: '#fff', fs: 11.5 });
    s += arrow(210, 113, 240, 113);
    s += box(240, 88, 130, 50, 'Sentence-T5|frozen encoder', { fill: SOFT, fs: 11.5 });
    s += arrow(370, 113, 400, 113);
    s += box(400, 88, 180, 50, 'RQ-VAE|hierarchical codebooks', { fill: SOFT, fs: 11.5 });
    s += arrow(580, 113, 610, 113);
    s += box(610, 88, 230, 50, 'semantic ID (c1, c2, c3)|shared prefix = related items', { fill: '#fff', fs: 12 });

    // (2) serving: retrieval is decoding
    s += step(48, 176, 2);
    s += txt(64, 168, 'serving · retrieval is decoding — no ANN index at all', { fs: 11.5, fill: GREY });
    s += box(60, 188, 180, 50, 'user history|as SID token sequence', { fill: '#fff', fs: 11.5 });
    s += aflow(240, 213, 270, 213, { color: BLUE });
    s += box(270, 188, 180, 50, 'transformer|autoregressive decode', { fill: SOFT, fs: 11.5 });
    s += aflow(450, 213, 480, 213, { color: BLUE });
    s += box(480, 188, 210, 50, 'constrained beam search|over the code tree', { fill: SOFT, fs: 11.5 });
    s += arrow(690, 213, 720, 213, { color: BLUE });
    s += box(720, 188, 120, 50, 'next item|= its SID', { fill: '#fff', fs: 11.5 });

    // (3) headline result
    s += step(48, 286, 3);
    s += apulse(hl(64, 272, 344, 24) + txt(70, 289, '≈ +17% Recall@5 · +29% NDCG@5', { fs: 14.5, w: 600 }));
    s += txt(416, 289, 'over the strongest baselines', { fs: 11.5, fill: GREY });
    s += note(64, 318, [
      'WHY cold start improves: a brand-new item still shares a SID prefix with trained neighbours,',
      'so partial prefix matching at inference surfaces it before it has any interaction history.'
    ]);
    return wrap(360, s);
  } },

/* ---------------------------------------- Singh et al., 2023 (features) */
{ id: 'sid-ranker-features',
  paper: 'Better Generalization with Semantic IDs — Singh et al., 2023',
  claim: 'Feeding SID code tokens to an existing ranker as sparse features captures the generalization win under production latency, with no generative decoding.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SEMANTIC IDS AS RANKER FEATURES — THE PRODUCTION PATH', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // pipeline
    s += step(48, 76, 1);
    s += box(60, 88, 150, 50, 'item|content embedding', { fill: '#fff', fs: 11.5 });
    s += arrow(210, 113, 240, 113);
    s += box(240, 88, 170, 50, 'RQ-VAE tokenizer|trained once, frozen', { fill: SOFT, fs: 11.5 });
    s += aflow(410, 113, 440, 113, { color: BLUE });
    s += step(432, 80, 2);
    s += box(440, 88, 180, 50, 'SID tokens (c1,c2,c3)|categorical features', { fill: '#fff', fs: 11.5 });
    s += aflow(620, 113, 650, 113, { color: BLUE });
    s += step(642, 80, 3);
    s += box(650, 88, 190, 50, 'existing ranker|extra sparse features', { fill: SOFT, fs: 11.5 });

    // what a shared embedding row means
    s += txt(60, 176, 'the crux is hashing: what an embedding-row collision means', { fs: 11.5, fill: GREY });
    s += box(60, 190, 360, 62, 'random hashed item ID|collision = two arbitrary items|forced onto one row', { fill: '#fff', stroke: RED, fs: 12 });
    s += box(480, 190, 360, 62, 'semantic ID tokens|collision = hierarchy siblings|that share statistical strength', { fill: '#fff', stroke: GREEN, fs: 12 });
    s += txt(240, 270, 'baseline', { a: 'middle', fs: 11.5, fill: RED });
    s += txt(660, 270, 'this paper', { a: 'middle', fs: 11.5, fill: GREEN });

    // headline
    s += apulse(hl(60, 290, 512, 22) + txt(66, 306, 'better generalization to new & tail items, inside the latency budget', { fs: 13, w: 600 }));
    s += note(60, 334, [
      'WHY this is the de-risking move: no re-platforming and no decoder — the practical crux is',
      'hashing + adaptation through per-level embeddings, and it is measurable in one experiment.'
    ]);
    return wrap(372, s);
  } },

/* -------------------------------------------------- LETTER + ETEGRec */
{ id: 'letter-etegrec-tokenizer',
  paper: 'LETTER & ETEGRec — tokenizer improvements',
  claim: 'LETTER regularizes code assignment with collaborative signal and diversity, while ETEGRec alternates tokenizer and generator optimization — two direct attacks on collapse and the frozen tokenizer.',
  svg() {
    let s = '';
    s += txt(60, 30, 'FIXING THE SID TOKENIZER · LETTER & ETEGRec', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) baseline pipeline
    s += step(48, 76, 1);
    s += txt(64, 68, 'baseline generative recommender (TIGER-style)', { fs: 11.5, fill: GREY });
    s += box(60, 88, 190, 50, 'content embedding|items only, no behaviour', { fill: '#fff', fs: 11.5 });
    s += aflow(250, 113, 286, 113, { color: BLUE });
    s += box(286, 88, 180, 50, 'tokenizer|RQ-VAE, frozen', { fill: SOFT, fs: 11.5 });
    s += aflow(466, 113, 502, 113, { color: BLUE });
    s += box(502, 88, 150, 50, 'semantic IDs', { fill: '#fff', fs: 12 });
    s += arrow(652, 113, 688, 113, { color: BLUE });
    s += box(688, 88, 152, 50, 'generator|autoregressive', { fill: SOFT, fs: 11.5 });

    // failure mode: codebook collapse (utilization bars)
    s += txt(286, 172, 'failure mode: codebook collapse — code utilization per level', { fs: 11.5, fill: RED });
    const bars = [34, 6, 3, 28, 4, 3, 3, 20, 3, 3, 4, 3];
    bars.forEach((h, i) => {
      s += `<rect x="${286 + i * 14}" y="${216 - h}" width="10" height="${h}" fill="${h > 10 ? RED : '#DDE1EA'}"/>`;
    });
    s += txt(340, 238, 'a few codes absorb everything', { fs: 11, fill: GREY });

    // (2) LETTER: fix the assignment
    s += step(48, 262, 2);
    s += box(60, 274, 360, 62, 'LETTER|+ collaborative signal in the tokenizer|+ diversity regularization on code assignment', { fill: '#fff', stroke: GREEN, fs: 12 });
    s += arrow(240, 274, 340, 220, { color: GREY, dash: '3 3', sw: 1.2 });

    // (3) ETEGRec: fix the freeze
    s += step(478, 262, 3);
    s += box(490, 274, 350, 62, 'ETEGRec|alternate optimization: tokenizer ⇄ generator|instead of freezing the tokenizer', { fill: '#fff', stroke: GREEN, fs: 12 });
    s += arrow(600, 274, 420, 140, { color: GREY, dash: '3 3', sw: 1.2 });
    s += arrow(700, 274, 740, 142, { color: GREY, dash: '3 3', sw: 1.2 });

    // headline
    s += apulse(hl(60, 356, 486, 22) + txt(66, 372, 'collapse addressed directly — the frozen-tokenizer loop is opened', { fs: 13, w: 600 }));
    s += note(60, 392, [
      'WHY: a frozen tokenizer never hears what the generator needs to distinguish; both papers',
      'open that loop from different ends — LETTER at assignment, ETEGRec at optimization.'
    ]);
    return wrap(420, s);
  } },

/* ------------------------------------------------- OneRec / RQ-Kmeans */
{ id: 'onerec-rqkmeans-tokenizer',
  paper: 'OneRec — Kuaishou (RQ-Kmeans tokenizer)',
  claim: 'OneRec swaps RQ-VAE for RQ-Kmeans over a collaborative-aware multimodal tokenizer that fuses title, tags, audio and images with user behaviour.',
  svg() {
    let s = '';
    s += txt(60, 30, 'ONEREC TOKENIZER · RQ-KMEANS OVER MULTIMODAL FUSION', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) fuse content with behaviour
    s += step(48, 70, 1);
    s += txt(64, 74, 'fuse every modality with behaviour', { fs: 11.5, fill: GREY });
    const mods = ['title', 'tags', 'audio', 'images'];
    mods.forEach((m, i) => {
      s += box(60, 86 + i * 36, 110, 28, m, { fill: '#fff', fs: 11.5 });
      s += arrow(170, 100 + i * 36, 240, 140 + i * 14, { color: GREY, sw: 1.2 });
    });
    s += box(60, 236, 150, 44, 'user behaviour|collaborative signal', { fill: SOFT, fs: 11.5 });
    s += arrow(210, 258, 258, 212, { sw: 1.5 });
    s += box(240, 122, 200, 90, 'collaborative-aware|multimodal tokenizer|(fusion)', { fill: SOFT, fs: 12 });

    // (2) quantize residuals with k-means
    s += step(468, 132, 2);
    s += aflow(440, 167, 476, 167, { color: BLUE });
    s += apulse(box(476, 142, 170, 50, 'RQ-Kmeans|replaces RQ-VAE', { fill: MARK, fs: 12.5 }));
    s += arrow(646, 167, 682, 167);
    s += box(682, 142, 158, 50, 'semantic IDs|behaviour-aware codes', { fill: '#fff', fs: 11.5 });

    // (3) feed the generator
    s += step(592, 236, 3);
    s += aflow(761, 192, 728, 242, { color: BLUE });
    s += box(604, 242, 236, 44, 'OneRec|generative recommender', { fill: SOFT, fs: 12 });

    s += note(60, 306, [
      'WHY k-means on residuals: no encoder/decoder training loop to fight, and k-means fitting',
      'is the classic guard against codebook collapse; fusing behaviour means codes reflect what',
      'users consume together, not content similarity alone.'
    ]);
    return wrap(352, s);
  } }

],

rag: [

/* ---------------------------------------------------- Lewis et al. 2020 */
{ id: 'rag-marginalization',
  paper: 'Retrieval-Augmented Generation — Lewis et al., NeurIPS 2020',
  claim: 'Treating the retrieved document as a latent variable and marginalising over the top-k gives RAG-Sequence 44.5 exact match on Natural Questions — state of the art at publication.',
  svg() {
    let s = '';
    s += txt(60, 30, 'RAG — MARGINALISE OVER RETRIEVED DOCUMENTS', { fs: 12.5, w: 600, fill: PF_BLUE });
    s += pfRule(40);

    // (1) query -> retriever
    s += box(60, 88, 120, 52, 'QUERY x|knowledge-heavy', { fill: PF_SOFT });
    s += step(64, 80, 1);
    s += aflow(180, 114, 238, 114, { label: 'q(x)' });
    s += box(238, 88, 168, 52, 'RETRIEVER p_η(z∣x)|dense dot product');
    s += txt(242, 156, 'z is a latent variable', { fs: 11, fill: PF_GREY });

    // (2) top-k docs, each with a weight
    s += step(474, 34, 2);
    s += box(470, 42, 120, 40, 'doc z₁', { fill: '#fff' });
    s += box(470, 94, 120, 40, 'doc z₂', { fill: '#fff' });
    s += box(470, 146, 120, 40, 'doc z₃', { fill: '#fff' });
    s += arrow(406, 104, 466, 62,  { color: PF_GREY, sw: 1, label: 'p=.5', ly: -4 });
    s += arrow(406, 114, 466, 114, { color: PF_GREY, sw: 1, label: 'p=.3', ly: -4 });
    s += arrow(406, 124, 466, 166, { color: PF_GREY, sw: 1, label: 'p=.2', ly: 12 });

    // (3) generator per doc, weighted mixture
    s += step(644, 80, 3);
    s += box(640, 88, 122, 52, 'Σ  MIXTURE|p_η(z∣x)·p_θ(y∣x,z)');
    s += arrow(590, 62, 636, 104, { sw: 1 });
    s += arrow(590, 114, 636, 114, { sw: 1 });
    s += arrow(590, 166, 636, 124, { sw: 1 });
    s += arrow(762, 114, 792, 114);
    s += box(792, 90, 68, 48, 'ANSWER|y', { fill: PF_MARK });

    // the formula, stated plainly
    s += txt(60, 224, 'p(y|x)  ≈  Σ_{z ∈ top-k}  p_η(z|x) · p_θ(y|x,z)', { fs: 15, w: 600, fill: PF_BLUE });
    s += txt(60, 244, 'the answer distribution is a mixture weighted by retrieval', { fs: 11.5, fill: PF_GREY });

    // headline number
    s += apulse(
      hl(60, 264, 486, 22) +
      txt(66, 280, 'RAG-Sequence: 44.5 exact match on Natural Questions — SOTA 2020', { fs: 12.5, w: 600 })
    );

    s += note(560, 268, [
      'RAG-Token re-marginalises per token;',
      'modern systems concat top-k and generate',
      'once — the mixture reading survives.'
    ], { from: [556, 264], to: [700, 145] });
    return wrap(330, s);
  } },

/* ------------------------------------------------ Karpukhin et al. 2020 */
{ id: 'dpr-dual-encoder',
  paper: 'Dense Passage Retrieval — Karpukhin et al., EMNLP 2020',
  claim: 'Two BERT encoders trained on question–passage pairs map both into one vector space where a single dot product d(z)ᵀq(x) ranks the whole corpus.',
  svg() {
    let s = '';
    s += txt(60, 30, 'DPR — DUAL ENCODERS, ONE DOT PRODUCT', { fs: 12.5, w: 600, fill: PF_BLUE });
    s += pfRule(40);

    // (1) question side, online
    s += box(60, 66, 140, 46, 'QUESTION x', { fill: PF_SOFT });
    s += step(64, 58, 1);
    s += aflow(200, 89, 258, 89);
    s += box(258, 60, 190, 58, 'QUESTION ENCODER|BERT → vector q(x)');

    // (2) passage side, offline
    s += box(60, 168, 140, 46, 'PASSAGE z|corpus docs', { fill: PF_SOFT });
    s += step(64, 160, 2);
    s += arrow(200, 191, 258, 191);
    s += box(258, 162, 190, 58, 'PASSAGE ENCODER|BERT → vector d(z)');
    s += txt(258, 238, 'run offline over the whole corpus', { fs: 11, fill: PF_GREY });

    // (3) similarity = dot product, into ANN index
    s += apulse(box(492, 112, 96, 56, 'd(z)ᵀ q(x)|dot product', { fill: PF_SOFT }));
    s += step(496, 104, 3);
    s += arrow(448, 89, 490, 118, { sw: 1 });
    s += arrow(448, 191, 490, 162, { sw: 1 });
    s += arrow(588, 140, 648, 140, { label: 'score' });
    s += box(648, 112, 182, 56, 'ANN INDEX|top-k passages out');

    // headline: how it is trained
    s += hl(60, 262, 268, 20);
    s += txt(64, 277, 'trained on question–passage pairs', { fs: 12.5, w: 600 });
    s += note(470, 252, [
      'one shared vector space — the retriever',
      'half of RAG:  p_η(z|x) ∝ exp(d(z)ᵀq(x))'
    ], { from: [466, 248], to: [545, 172] });
    return wrap(310, s);
  } },

/* ---------------------------------------------------- Asai et al. 2023 */
{ id: 'self-rag-reflection',
  paper: 'Self-RAG — Asai et al., ICLR 2024 (arXiv 2023)',
  claim: 'Reflection tokens trained into the vocabulary let the model decide when to retrieve and critique whether each generated segment is supported by evidence.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SELF-RAG — REFLECTION TOKENS', { fs: 12.5, w: 600, fill: PF_BLUE });
    s += pfRule(40);

    // (1) the model decides whether to retrieve
    s += box(60, 70, 120, 50, 'PROMPT x', { fill: PF_SOFT });
    s += aflow(180, 95, 240, 95);
    s += apulse(box(240, 64, 170, 62, 'LM emits|[Retrieve?] token'));
    s += step(244, 56, 1);

    s += arrow(410, 82, 478, 70, { label: 'yes', ly: -5 });
    s += box(480, 46, 160, 50, 'RETRIEVE|top passages');
    s += arrow(410, 112, 478, 150, { color: PF_GREY, dash: '4 3', label: 'no', ly: 12 });
    s += box(480, 132, 160, 44, 'answer from weights|no retrieval step', { fill: '#fff' });

    // (2) generate a segment per passage
    s += arrow(640, 71, 688, 71);
    s += box(688, 46, 152, 54, 'GENERATE|segment y_t');
    s += step(692, 38, 2);

    // (3) self-critique with trained tokens
    s += `<line x1="764" y1="100" x2="764" y2="195" stroke="${PF_INK_rag}" stroke-width="1"/>`;
    s += arrow(764, 195, 464, 216, { sw: 1 });
    s += box(240, 210, 220, 66, 'CRITIQUE TOKENS|[IsSup] grounded in passage?|[IsUse] useful for x?');
    s += step(244, 202, 3);
    s += arrow(460, 243, 528, 243);
    s += box(528, 218, 170, 50, 'OUTPUT|supported segments', { fill: PF_MARK });

    // headline
    s += hl(60, 300, 528, 20);
    s += txt(64, 315, 'the model learns WHEN to retrieve and whether its output is supported', { fs: 12.5, w: 600 });
    s += note(60, 344, [
      'reflection tokens are trained into the vocabulary — retrieval becomes a per-step decision,',
      'not a fixed always-on pipeline stage bolted in front of the generator.'
    ]);
    return wrap(380, s);
  } },

/* ------------------------------------------------------- Es et al. 2023 */
{ id: 'ragas-metric-split',
  paper: 'RAGAS — Es et al., arXiv 2023',
  claim: 'Reference-free evaluation splits RAG quality into retrieval scores (context precision, context recall) and generation scores (faithfulness, answer relevance) so failures can be localised.',
  svg() {
    let s = '';
    s += txt(60, 30, 'RAGAS — SCORE RETRIEVAL AND GENERATION SEPARATELY', { fs: 12.5, w: 600, fill: PF_BLUE });
    s += pfRule(40);

    // (1) retrieval metrics point at the context
    s += step(164, 42, 1);
    s += box(160, 50, 175, 54, 'CONTEXT PRECISION|is retrieved relevant?', { fill: PF_SOFT });
    s += box(355, 50, 175, 54, 'CONTEXT RECALL|is the evidence there?', { fill: PF_SOFT });
    s += txt(60, 72, 'SCORE THE', { fs: 11, fill: PF_GREY });
    s += txt(60, 86, 'RETRIEVAL', { fs: 11, fill: PF_GREY });
    s += arrow(247, 104, 305, 138, { color: PF_GREY, sw: 1 });
    s += arrow(442, 104, 385, 138, { color: PF_GREY, sw: 1 });

    // the pipeline under test
    s += box(60, 140, 130, 50, 'QUERY', { fill: '#fff' });
    s += aflow(190, 165, 250, 165);
    s += box(250, 140, 180, 50, 'RETRIEVED CONTEXT|top-k chunks');
    s += aflow(430, 165, 490, 165);
    s += box(490, 140, 160, 50, 'ANSWER|generated');

    // (2) generation metrics point at the answer
    s += step(434, 234, 2);
    s += box(430, 242, 170, 54, 'FAITHFULNESS|grounded in context?', { fill: PF_SOFT });
    s += box(620, 242, 175, 54, 'ANSWER RELEVANCE|addresses the query?', { fill: PF_SOFT });
    s += txt(735, 222, 'SCORE THE GENERATION', { fs: 11, fill: PF_GREY, a: 'end' });
    s += arrow(515, 242, 552, 194, { color: PF_GREY, sw: 1 });
    s += arrow(700, 242, 608, 194, { color: PF_GREY, sw: 1 });

    // (3) the point of the decomposition
    s += note(60, 244, [
      'an LLM judge scores each metric from',
      'query, context and answer alone —',
      'the decomposition matters more',
      'than the specific scorers.'
    ], { from: [56, 240], to: [125, 192] });
    s += step(68, 316, 3);
    s += apulse(
      hl(88, 306, 512, 20) +
      txt(92, 321, 'four scores, two halves: localise the failure before fixing it', { fs: 12.5, w: 600 })
    );
    return wrap(350, s);
  } },

/* -------------------------------------------------------- Liu et al. */
{ id: 'lost-in-the-middle',
  paper: 'Lost in the Middle — Liu et al., TACL 2024',
  claim: 'Answer accuracy is highest when the gold passage sits at the start or end of the context and degrades in the middle, so adding more chunks is not monotonically better.',
  svg() {
    let s = '';
    s += txt(60, 30, 'LOST IN THE MIDDLE — POSITION BIAS IN LONG CONTEXT', { fs: 12.5, w: 600, fill: PF_BLUE });
    s += pfRule(40);

    // (1) vary where the gold passage sits in the context
    s += txt(60, 66, 'context window, k docs', { fs: 11, fill: PF_GREY });
    s += step(218, 60, 1);
    s += box(60, 74, 140, 30, 'doc 1 — start', { fill: '#fff', fs: 11.5 });
    s += box(60, 108, 140, 30, 'doc 2', { fill: '#fff', fs: 11.5 });
    s += box(60, 142, 140, 30, 'gold — middle', { fill: PF_MARK, fs: 11.5 });
    s += box(60, 176, 140, 30, 'doc k−1', { fill: '#fff', fs: 11.5 });
    s += box(60, 210, 140, 30, 'doc k — end', { fill: '#fff', fs: 11.5 });
    s += aflow(200, 157, 296, 157, { label: 'vary position', ly: -9 });

    // (2) accuracy traces a U over gold-passage position
    s += step(304, 62, 2);
    s += `<line x1="300" y1="240" x2="830" y2="240" stroke="${PF_INK_rag}" stroke-width="1.5"/>`;
    s += `<line x1="300" y1="72" x2="300" y2="240" stroke="${PF_INK_rag}" stroke-width="1.5"/>`;
    s += txt(330, 68, 'answer accuracy', { fs: 11.5, fill: PF_GREY });
    s += txt(565, 262, 'gold passage position in the context', { a: 'middle', fs: 11.5, fill: PF_GREY });
    s += `<path d="M320,100 C450,108 480,210 565,210 C650,210 700,108 810,100" fill="none" stroke="${PF_BLUE}" stroke-width="2.5"/>`;
    s += `<circle cx="320" cy="100" r="4.5" fill="${PF_GREEN}"/>`;
    s += `<circle cx="810" cy="100" r="4.5" fill="${PF_GREEN}"/>`;
    s += txt(330, 92, 'start: high', { fs: 11.5, fill: PF_GREEN });
    s += txt(800, 92, 'end: high', { a: 'end', fs: 11.5, fill: PF_GREEN });
    s += apulse(`<circle cx="565" cy="210" r="5.5" fill="${PF_RED}"/>` +
      txt(565, 196, 'middle: worst', { a: 'middle', fs: 11.5, fill: PF_RED, w: 600 }));

    s += note(720, 254, [
      'stuffing more chunks',
      'pushes evidence toward',
      'the middle — retrieval',
      'precision beats volume.'
    ]);

    // (3) the headline
    s += hl(60, 284, 580, 20);
    s += txt(64, 299, 'accuracy is U-shaped in position — more chunks is not monotonically better', { fs: 12.5, w: 600 });
    s += step(658, 294, 3);
    return wrap(330, s);
  } }
],

hybrid: [

/* ------------------------------------------------- 1 · RRF, SIGIR 2009 */
{ id: 'rrf-rank-fusion-k60',
  paper: 'Reciprocal Rank Fusion — Cormack, Clarke & Buettcher, SIGIR 2009',
  claim: 'Fusing by rank position with RRF(d)=Σ 1/(60+rank) beat Condorcet fusion and individual learned rankers on TREC — in three pages, with no tuning.',
  svg() {
    let s = '';
    s += txt(60, 30, 'RECIPROCAL RANK FUSION · FUSE BY POSITION, NOT BY SCORE', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) two ranked lists with incompatible score units
    s += step(60, 66, 1);
    s += txt(78, 70, 'two retrievers, two incompatible score scales', { fs: 12, fill: GREY });
    s += txt(60, 96, 'BM25 (unbounded +)', { fs: 11.5 });
    s += box(60, 104, 130, 30, 'r1  doc A', { fill: SOFT, fs: 11.5 });
    s += box(60, 138, 130, 30, 'r2  doc C', { fill: '#fff', fs: 11.5 });
    s += box(60, 172, 130, 30, 'r3  doc B', { fill: '#fff', fs: 11.5 });
    s += txt(230, 142, 'dense cosine [−1, 1]', { fs: 11.5 });
    s += box(230, 150, 130, 30, 'r1  doc D', { fill: SOFT, fs: 11.5 });
    s += box(230, 184, 130, 30, 'r2  doc C', { fill: '#fff', fs: 11.5 });
    s += box(230, 218, 130, 30, 'r3  doc E', { fill: '#fff', fs: 11.5 });

    // (2) the formula — ranks only
    s += step(432, 66, 2);
    s += box(420, 118, 210, 70, 'RRF(d) = Σᵣ 1 / (k + rankᵣ(d))|k = 60 · no tuning', { fill: '#fff' });
    s += hl(447, 158, 156, 16);
    s += aflow(196, 119, 412, 136, {});
    s += arrow(364, 165, 416, 158, {});

    // (3) fused list — agreement wins
    s += step(672, 66, 3);
    s += aflow(636, 153, 688, 153, {});
    s += apulse(box(696, 104, 144, 34, 'doc C  0.0323', { fill: MARK, fs: 11.5 }));
    s += box(696, 142, 144, 30, 'doc A  0.0164', { fill: '#fff', fs: 11.5 });
    s += box(696, 176, 144, 30, 'doc D  0.0164', { fill: '#fff', fs: 11.5 });
    s += txt(696, 96, 'fused ranking', { fs: 11.5 });
    s += txt(696, 226, 'doc C: rank 2 in BOTH lists → 1/62 + 1/62', { fs: 11.5, fill: GREY });
    s += txt(696, 242, 'beats either list’s confident #1 (1/61)', { fs: 11.5, fill: GREY });

    // why it works
    s += `<line x1="60" y1="262" x2="840" y2="262" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 286, 'k = 60 damps the top: 1/61 vs 1/62 is a tiny gap, so broad agreement outvotes one loud first place.', { fs: 12.5 });
    s += note(60, 308, [
      'why ranks, not scores: whatever units each retriever prints, "second place" means the same thing in both lists.',
      'result: beat Condorcet fusion and individual learned rankers on TREC. Three pages — probably the highest impact-per-page in IR.'
    ]);
    return wrap(340, s);
  } },

/* --------------------------------------------- 2 · BEIR, NeurIPS 2021 */
{ id: 'beir-8-of-18',
  paper: 'BEIR benchmark — Thakur et al., NeurIPS 2021',
  claim: 'Zero-shot across 18 datasets, the best dense retriever evaluated (TAS-B) beat BM25 on only 8 of 18 — in-domain wins on MS MARCO do not transfer.',
  svg() {
    let s = '';
    s += txt(60, 30, 'BEIR · WHAT HAPPENS TO DENSE RETRIEVAL OUT OF DOMAIN', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) train in-domain
    s += step(60, 68, 1);
    s += txt(78, 72, 'train + evaluate in-domain', { fs: 12, fill: GREY });
    s += box(60, 86, 200, 62, 'MS MARCO|dense (TAS-B) trained here', { fill: SOFT });
    s += txt(60, 170, 'in-domain: dense beats BM25', { fs: 12 });
    s += txt(60, 187, 'comfortably', { fs: 12 });

    // (2) same model, zero-shot on 18 datasets
    s += step(320, 68, 2);
    s += txt(338, 72, 'same model, zero-shot on 18 datasets', { fs: 12, fill: GREY });
    s += aflow(268, 117, 312, 117, { label: 'as-is' });
    const gx = 320, gy = 92;
    for (let i = 0; i < 18; i++) {
      const col = i % 9, row = Math.floor(i / 9);
      const win = i < 8; // 8 dense wins
      s += `<rect x="${gx + col * 46}" y="${gy + row * 46}" width="38" height="38" rx="2" fill="${win ? BLUE : '#fff'}" stroke="${INK}" stroke-width="1.2"/>`;
      s += `<text x="${gx + col * 46 + 19}" y="${gy + row * 46 + 24}" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="${win ? '#fff' : INK}">${i + 1}</text>`;
    }
    s += `<rect x="${gx}" y="${gy + 96}" width="12" height="12" fill="${BLUE}"/>`;
    s += txt(gx + 18, gy + 106, 'dense (TAS-B) wins · 8', { fs: 11.5 });
    s += `<rect x="${gx + 210}" y="${gy + 96}" width="12" height="12" fill="#fff" stroke="${INK}"/>`;
    s += txt(gx + 228, gy + 106, 'BM25 wins · 10', { fs: 11.5 });

    // (3) the headline
    s += step(770, 68, 3);
    s += hl(748, 92, 116, 40);
    s += apulse(txt(806, 112, 'dense wins', { a: 'middle', fs: 13, w: 600 }) +
                txt(806, 128, 'only 8 / 18', { a: 'middle', fs: 13, w: 600 }));
    s += txt(806, 152, 'zero-shot', { a: 'middle', fs: 11.5, fill: GREY });

    // the lesson
    s += `<line x1="60" y1="222" x2="840" y2="222" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 246, 'in-domain wins do not transfer; lexical robustness does.', { fs: 13.5, w: 600 });
    s += note(60, 268, [
      'this is the load-bearing evidence for hybrid: BM25 stays respectable everywhere,',
      'dense is brilliant near its training data and unreliable off it — so run both and fuse.'
    ]);
    return wrap(310, s);
  } },

/* ---------------------------------------------- 3 · SPLADE, 2021 */
{ id: 'splade-learned-sparse',
  paper: 'SPLADE / SPLADE v2 — Formal et al., 2021',
  claim: 'A masked-language-model head learns sparse term expansions — a document about jams also indexes "stuck" — giving learned semantics served from an ordinary inverted index, with strong zero-shot BEIR results.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SPLADE · LEARNED SPARSITY — SEMANTICS ON AN INVERTED INDEX', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) document in
    s += step(60, 68, 1);
    s += txt(78, 72, 'document text', { fs: 12, fill: GREY });
    s += box(60, 86, 170, 60, '"the door jams|when it rains"', { fill: SOFT });

    // (2) MLM head expands over the vocabulary
    s += step(300, 68, 2);
    s += txt(318, 72, 'MLM head scores every vocab term', { fs: 12, fill: GREY });
    s += aflow(230, 116, 292, 116, {});
    s += box(300, 86, 180, 60, 'BERT + MLM head|sparse weights over vocab', { fill: '#fff' });

    // expansion terms — bars, no invented decimals
    const terms = [ ['jams', 150, true], ['door', 118, true], ['stuck', 96, false], ['sticks', 66, false] ];
    terms.forEach((t, i) => {
      const y = 170 + i * 26;
      s += txt(300, y + 11, t[0], { fs: 12 });
      s += `<rect x="366" y="${y}" width="${t[1]}" height="14" fill="${t[2] ? INK : BLUE}"/>`;
    });
    s += txt(366, 284, 'black: terms in the document · blue: learned expansion', { fs: 11.5, fill: GREY });
    s += hl(296, 218, 226, 20);
    s += apulse(txt(540, 233, '← "stuck" never appears in the text', { fs: 12, w: 600 }));

    // (3) served from an ordinary inverted index
    s += step(620, 68, 3);
    s += txt(638, 72, 'index + serve like BM25', { fs: 12, fill: GREY });
    s += aflow(480, 116, 612, 116, { label: 'sparse vector' });
    s += box(620, 86, 220, 60, 'ORDINARY INVERTED INDEX|same machinery as BM25', { fill: SOFT });
    s += txt(620, 168, 'zero-shot BEIR: strong —', { fs: 12 });
    s += txt(620, 185, 'learned semantics, lexical serving', { fs: 12 });

    // why
    s += `<line x1="60" y1="304" x2="840" y2="304" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += note(60, 326, [
      'why it matters: the hybrid stack exists because dense needs an ANN index and lexical needs an inverted index.',
      'SPLADE claims to collapse that to one index — semantics without leaving the inverted-index world.'
    ]);
    return wrap(360, s);
  } },

/* ------------------------------- 4 · ColBERT 2020 + PLAID, CIKM 2022 */
{ id: 'colbert-late-interaction-plaid',
  paper: 'ColBERT — Khattab & Zaharia, 2020 · PLAID — Santhanam et al., CIKM 2022',
  claim: 'Per-token embeddings with MaxSim keep the document index precomputable; PLAID serves it 3.7× faster on GPU and 22× on CPU than vanilla ColBERTv2 with no quality loss — tens of milliseconds at 140M passages.',
  svg() {
    let s = '';
    s += txt(60, 30, 'ColBERT · LATE INTERACTION — MATCH TOKENS, PRECOMPUTE THE INDEX', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) offline: doc token embeddings
    s += step(60, 68, 1);
    s += txt(78, 72, 'offline: embed every document token once', { fs: 12, fill: GREY });
    for (let j = 0; j < 4; j++) s += box(60 + j * 78, 86, 68, 36, `d${j + 1}`, { fill: SOFT, fs: 11.5 });
    s += txt(60, 142, 'stored per-token vectors — the index is precomputable', { fs: 11.5, fill: GREY });

    // (2) online: query token embeddings
    s += step(60, 180, 2);
    s += txt(78, 184, 'query time: embed query tokens', { fs: 12, fill: GREY });
    for (let i = 0; i < 3; i++) s += box(60 + i * 78, 198, 68, 36, `q${i + 1}`, { fill: '#fff', fs: 11.5 });

    // (3) MaxSim per query token, then sum
    s += step(452, 68, 3);
    s += txt(470, 72, 'late interaction: MaxSim, then sum', { fs: 12, fill: GREY });
    s += aflow(372, 104, 444, 140, { curve: 30 });
    s += aflow(296, 216, 444, 178, { curve: 30 });
    s += box(452, 130, 240, 66, 'score = Σᵢ maxⱼ (qᵢ · dⱼ)|each query token picks its best|document token', { fill: '#fff' });
    s += txt(452, 216, 'query and document meet only here —', { fs: 11.5, fill: GREY });
    s += txt(452, 232, 'after all document work is done', { fs: 11.5, fill: GREY });

    // PLAID headline
    s += arrow(692, 163, 730, 163, {});
    s += hl(734, 122, 132, 20);
    s += box(734, 130, 132, 92, 'PLAID (2022)|3.7× faster GPU|22× faster CPU|no quality loss', { fill: MARK });
    s += apulse(txt(786, 244, 'tens of ms @ 140M passages', { a: 'middle', fs: 12, w: 600 }));

    // why
    s += `<line x1="60" y1="268" x2="840" y2="268" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += note(60, 290, [
      'why "late" matters: a cross-encoder must re-run the model per query-document pair; single-vector dense collapses',
      'the document to one point. ColBERT keeps token-level matching AND a precomputed index — PLAID made it cheap.'
    ]);
    return wrap(330, s);
  } },

],

chunk: [

/* ------------------------------------------------- 1 · Chroma report */
{ id: 'chroma-chunking-eval',
  paper: 'Evaluating Chunking Strategies for Retrieval — Smith & Troynikov, Chroma technical report 2024',
  claim: 'Scored at the token level over five corpora, chunking strategies spread by up to 9 points of recall — the LLM chunker wins recall at 91.9% while the popular 800/400 default is worst-in-precision.',
  svg() {
    let s = '';
    s += txt(60, 30, 'FIVE CORPORA · SAME QUERIES · SCORED AT THE TOKEN LEVEL, NOT THE DOCUMENT LEVEL', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // mechanism: chunkers -> token-level scorer -> spread
    s += box(60, 62, 190, 66, 'CHUNKERS|token windows · recursive|3 semantic chunkers', { fill: SOFT });
    s += step(72, 56, 1);
    s += aflow(258, 95, 322, 95);
    s += box(330, 62, 220, 66, 'TOKEN-LEVEL SCORING|recall · precision · IoU|per answer span', {});
    s += step(342, 56, 2);
    s += arrow(558, 95, 622, 95);
    s += box(630, 62, 210, 66, 'RESULT|up to 9 pts of recall|between strategies', {});
    s += step(642, 56, 3);
    s += note(330, 148, ['"did the right document come back" hides chunking failures —', 'the unit of credit here is the token span, not the doc ID.']);

    // recall bars (lengths qualitative except the stated 91.9%)
    const bx = 330;
    s += txt(60, 208, 'recall by strategy', { fs: 11.5, fill: GREY });
    s += txt(60, 236, 'LLM-driven chunker', { fs: 12 });
    s += `<rect x="${bx}" y="224" width="460" height="15" fill="${SOFT}" stroke="${INK}" stroke-width="1.2"/>`;
    s += hl(794, 222, 52, 19);
    s += txt(798, 236, '91.9%', { fs: 12.5, w: 600 });
    s += txt(60, 268, 'ClusterSemantic @200', { fs: 12 });
    s += `<rect x="${bx}" y="256" width="438" height="15" fill="#fff" stroke="${INK}" stroke-width="1.2"/>`;
    s += txt(762, 268, 'best precision + IoU', { a: 'end', fs: 11, fill: GREEN });
    s += txt(60, 300, 'tuned recursive splitter', { fs: 12 });
    s += `<rect x="${bx}" y="288" width="430" height="15" fill="#fff" stroke="${INK}" stroke-width="1.2"/>`;
    s += txt(754, 300, 'close behind, tiny cost', { a: 'end', fs: 11, fill: GREY });
    s += txt(60, 332, 'default: 800 tok / 400 overlap', { fs: 12 });
    s += `<rect x="${bx}" y="320" width="378" height="15" fill="#fff" stroke="${RED}" stroke-width="1.4"/>`;
    s += txt(702, 332, 'below-avg recall · WORST precision', { a: 'end', fs: 11, fill: RED });

    // the 9-pt spread bracket
    s += `<line x1="708" y1="316" x2="708" y2="345" stroke="${GREY}" stroke-width="1" stroke-dasharray="2 3"/>`;
    s += `<line x1="790" y1="218" x2="790" y2="345" stroke="${GREY}" stroke-width="1" stroke-dasharray="2 3"/>`;
    s += arrow(714, 352, 784, 352, { color: GREY, sw: 1 });
    s += txt(749, 370, '≤ 9 pts', { a: 'middle', fs: 11.5, fill: GREY });

    s += note(60, 388, ['semantic (embedding-discontinuity) chunkers were not reliably better than a well-tuned',
                        'recursive splitter — and the default nobody chose lost everywhere. measure, never inherit.']);
    return wrap(418, s);
  } },

/* --------------------------------------- 2 · Anthropic contextual retrieval */
{ id: 'anthropic-contextual-retrieval',
  paper: 'Introducing Contextual Retrieval — Anthropic, 2024',
  claim: 'Prepending a Claude-written situating sentence to each chunk before indexing cuts top-20 retrieval failures 35%, 49% with contextual BM25, and 67% with reranking — 5.7% down to 1.9%.',
  svg() {
    let s = '';
    s += txt(60, 30, 'PREPEND WHAT THE CHUNK IS ABOUT — THEN INDEX CONTEXT + CHUNK TOGETHER', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    s += box(60, 66, 160, 60, 'FULL DOCUMENT|sits in prompt cache', { fill: SOFT });
    s += box(60, 168, 160, 54, 'CHUNK i|no idea what "it" is', {});
    s += step(72, 60, 1);
    s += arrow(220, 96, 288, 128, { curve: 30 });
    s += arrow(220, 195, 288, 162, { curve: 30 });
    s += box(296, 118, 190, 68, 'CLAUDE|writes one situating|sentence per chunk', {});
    s += step(308, 112, 2);
    s += aflow(486, 152, 552, 152);
    s += box(560, 118, 180, 68, 'CONTEXT + CHUNK|embedded and|BM25-indexed', { fill: SOFT });
    s += step(572, 112, 3);
    s += note(296, 212, ['one-time cost: $1.02 per million document tokens —', 'prompt caching keeps the full doc resident while', 'Claude annotates every chunk against it.'], {});

    // failure-rate ladder
    const lx = 60, ly = 286;
    s += txt(lx, ly - 10, 'top-20 retrieval failure rate', { fs: 11.5, fill: GREY });
    s += `<rect x="${lx}" y="${ly}" width="330" height="15" fill="#fff" stroke="${INK}" stroke-width="1.2"/>`;
    s += txt(lx + 340, ly + 12, '5.7%  baseline chunks', { fs: 12 });
    s += `<rect x="${lx}" y="${ly + 24}" width="214" height="15" fill="#fff" stroke="${INK}" stroke-width="1.2"/>`;
    s += txt(lx + 340, ly + 36, '-35%  contextual embeddings', { fs: 12 });
    s += `<rect x="${lx}" y="${ly + 48}" width="168" height="15" fill="#fff" stroke="${INK}" stroke-width="1.2"/>`;
    s += txt(lx + 340, ly + 60, '-49%  + contextual BM25', { fs: 12 });
    s += `<rect x="${lx}" y="${ly + 72}" width="109" height="15" fill="${SOFT}" stroke="${GREEN}" stroke-width="1.4"/>`;
    s += apulse(hl(lx + 336, ly + 62, 190, 19) + txt(lx + 340, ly + 84, '-67%  + reranking -> 1.9%', { fs: 12.5, w: 600 }));

    s += note(660, 300, ['the rare technique whose', 'numbers survived independent', 'replication attempts.'], {});
    return wrap(392, s);
  } },

/* ----------------------------------------------------- 3 · Late chunking */
{ id: 'late-chunking',
  paper: 'Late Chunking — Günther et al., Jina AI, arXiv 2409.04701',
  claim: 'Embed the whole document first and pool token vectors per chunk afterwards: every chunk vector has already attended to its neighbours, so pronouns keep their referents with no text generation and no training.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SWAP THE ORDER: EMBED THE WHOLE DOCUMENT, THEN CUT', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // naive lane
    s += txt(60, 68, 'NAIVE: cut -> embed each chunk alone', { fs: 12, w: 600, fill: RED });
    s += box(60, 80, 150, 46, 'chunk 1|"Berlin grew..."', { fill: '#fff' });
    s += box(222, 80, 150, 46, 'chunk 2|"It cut costs..."', { fill: '#fff' });
    s += `<line x1="216" y1="74" x2="216" y2="132" stroke="${RED}" stroke-width="2"/>`;
    s += txt(216, 148, 'the cut severs "It" from its referent', { a: 'middle', fs: 11, fill: RED });
    s += arrow(384, 103, 448, 103);
    s += box(456, 80, 170, 46, 'embedder x2|each call sees|one chunk only', {});
    s += arrow(638, 103, 702, 103);
    s += box(710, 80, 130, 46, 'vectors|context-blind', { fill: '#fff', stroke: RED });

    // late lane
    s += txt(60, 188, 'LATE: embed -> cut -> pool', { fs: 12, w: 600, fill: GREEN });
    s += box(60, 208, 190, 52, 'WHOLE DOCUMENT|one long-context|embedder pass', { fill: SOFT });
    s += step(72, 202, 1);
    s += aflow(258, 234, 322, 234);
    // token-vector strip with chunk boundaries drawn on it
    s += `<rect x="330" y="216" width="300" height="36" fill="#fff" stroke="${INK}" stroke-width="1.5"/>`;
    for (let i = 1; i < 12; i++) s += `<line x1="${330 + i * 25}" y1="216" x2="${330 + i * 25}" y2="252" stroke="${GREY}" stroke-width="0.7"/>`;
    s += `<line x1="430" y1="210" x2="430" y2="258" stroke="${BLUE}" stroke-width="2" stroke-dasharray="4 3"/>`;
    s += `<line x1="530" y1="210" x2="530" y2="258" stroke="${BLUE}" stroke-width="2" stroke-dasharray="4 3"/>`;
    s += txt(480, 208, 'token vectors', { a: 'middle', fs: 11, fill: GREY });
    s += txt(480, 272, 'chunk boundaries applied AFTER attention', { a: 'middle', fs: 11, fill: BLUE });
    s += step(342, 202, 2);
    s += arrow(638, 234, 702, 234);
    s += box(710, 208, 130, 52, 'mean-pool|per chunk', { fill: SOFT });
    s += step(722, 202, 3);

    s += hl(60, 296, 520, 20);
    s += txt(64, 311, 'every chunk vector has already attended to its neighbours', { fs: 12.5, w: 600 });
    s += note(60, 340, ['the pronoun problem dissolves without generating any text:', 'no training needed — a dedicated fine-tune helps further.'], {});
    s += note(620, 340, ['same index size, same', 'retrieval pipeline downstream.'], {});
    return wrap(378, s);
  } },

/* ----------------------------------------------------------- 4 · RAPTOR */
{ id: 'raptor-summary-tree',
  paper: 'RAPTOR — Sarthi et al., arXiv 2401.18059',
  claim: 'Recursive clustering and summarization builds a tree over the corpus, so retrieval can answer at paragraph, section, or whole-document abstraction — questions no single chunk can answer.',
  svg() {
    let s = '';
    s += txt(60, 30, 'CLUSTER -> SUMMARIZE -> RECURSE: A TREE OVER THE CHUNKS', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // leaves
    const lw = 96, ly = 306;
    for (let i = 0; i < 5; i++) s += box(60 + i * 110, ly, lw, 44, `leaf ${i + 1}|chunk`, { fill: '#fff', fs: 11.5 });
    s += step(48, 300, 1);
    s += txt(60, 372, 'leaf chunks: the corpus as usually chunked', { fs: 11.5, fill: GREY });

    // mid summaries
    s += box(116, 208, 150, 48, 'summary A|clusters leaves 1-3', { fill: SOFT });
    s += box(390, 208, 150, 48, 'summary B|clusters leaves 4-5', { fill: SOFT });
    s += step(104, 202, 2);
    s += arrow(108, ly - 4, 160, 260, { color: GREY, sw: 1.2 });
    s += arrow(218, ly - 4, 191, 260, { color: GREY, sw: 1.2 });
    s += arrow(328, ly - 4, 232, 260, { color: GREY, sw: 1.2 });
    s += arrow(438, ly - 4, 445, 260, { color: GREY, sw: 1.2 });
    s += arrow(548, ly - 4, 490, 260, { color: GREY, sw: 1.2 });

    // root
    s += box(220, 108, 200, 50, 'ROOT SUMMARY|whole-document gist', { fill: SOFT });
    s += step(208, 102, 3);
    s += aflow(191, 204, 280, 162, { curve: 20 });
    s += arrow(465, 204, 380, 162, { curve: 20 });

    // retrieval hits any level
    s += `<line x1="660" y1="80" x2="660" y2="352" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(668, 92, 'retrieval searches ALL levels', { fs: 12, w: 600 });
    s += txt(680, 130, '"what is the theme?"', { fs: 11.5 });
    s += arrow(676, 134, 428, 134, { color: BLUE, sw: 1.2 });
    s += txt(680, 230, '"what happens in part B?"', { fs: 11.5 });
    s += arrow(676, 234, 548, 234, { color: BLUE, sw: 1.2 });
    s += txt(680, 322, '"what exact figure...?"', { fs: 11.5 });
    s += arrow(676, 326, 610, 326, { color: BLUE, sw: 1.2 });

    s += hl(680, 340, 160, 20);
    s += txt(684, 355, 'one index, every altitude', { fs: 12, w: 600 });
    s += note(60, 70, ['the honest response to questions', 'no single chunk can answer.'], {});
    return wrap(396, s);
  } },
],

agent: [

/* ------------------------------------------------------ 1 · ReAct */
{ id: 'react-interleave',
  paper: 'ReAct — Yao et al., 2022 (arXiv 2210.03629)',
  claim: 'Interleaving reasoning traces with actions beats either alone: +34% absolute success on ALFWorld and +10% on WebShop over imitation and RL baselines.',
  svg() {
    let s = '';
    s += txt(60, 30, 'REACT · REASONING TRACES INTERLEAVED WITH ACTIONS', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // left: act-only baseline
    s += txt(60, 70, 'ACT-ONLY (imitation / RL baselines)', { fs: 12, fill: GREY });
    s += box(70, 84, 130, 44, 'act', { fill: '#fff' });
    s += arrow(135, 128, 135, 152);
    s += box(70, 152, 130, 44, 'observe', { fill: SOFT });
    s += arrow(135, 196, 135, 220);
    s += box(70, 220, 130, 44, 'act', { fill: '#fff' });
    s += note(220, 108, ['no thought between steps —', 'the policy reacts, never', 'plans or corrects course'], {});
    s += txt(70, 292, 'errors compound silently', { fs: 12, fill: RED });

    // divider
    s += `<line x1="430" y1="56" x2="430" y2="296" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;

    // right: the ReAct loop
    s += txt(480, 70, 'REACT LOOP', { fs: 12, w: 600 });
    s += box(560, 84, 200, 52, 'THINK|"I need the pan first"', { fill: MARK });
    s += step(548, 84, 1);
    s += box(660, 190, 190, 52, 'ACT|goto(counter 1)', { fill: '#fff' });
    s += step(648, 190, 2);
    s += box(470, 190, 160, 52, 'OBSERVE|env feedback', { fill: SOFT });
    s += step(458, 190, 3);
    s += aflow(725, 136, 755, 184, { color: BLUE, label: 'act', ly: -2 });
    s += arrow(660, 226, 636, 226, { label: '' });
    s += aflow(545, 184, 590, 142, { color: BLUE, label: 'reason', ly: 6 });
    s += note(480, 270, ['the trace is written into context, so the', 'next action conditions on an explicit plan'], {});

    // results band
    s += `<line x1="60" y1="308" x2="840" y2="308" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 336, 'ALFWorld success:', { fs: 13 });
    s += hl(228, 322, 132, 20);
    s += txt(232, 336, '+34% absolute', { fs: 13, w: 600 });
    s += txt(430, 336, 'WebShop: +10% absolute', { fs: 13 });
    s += txt(60, 362, 'thinking-only cannot ground itself; acting-only cannot recover.', { fs: 11.5, fill: GREY });
    s += txt(60, 377, 'every agent framework since is this loop with better plumbing.', { fs: 11.5, fill: GREY });
    return wrap(392, s);
  } },

/* --------------------------------------------------- 2 · SWE-bench */
{ id: 'swe-bench-arc',
  paper: 'SWE-bench — Jimenez et al., ICLR 2024 (arXiv 2310.06770)',
  claim: 'On 2,294 real GitHub issues the best assisted model resolved 1.96%; two years later Claude Opus 4.5 reports 80.9% on the human-validated SWE-bench Verified subset.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SWE-BENCH · REAL GITHUB ISSUES AS THE EXAM', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // task pipeline
    s += box(60, 64, 170, 54, 'GITHUB ISSUE|2,294 real bug reports', { fill: SOFT });
    s += step(48, 64, 1);
    s += aflow(238, 91, 292, 91, { color: BLUE });
    s += box(300, 64, 180, 54, 'AGENT + REPO|navigate, edit files', { fill: '#fff' });
    s += step(288, 64, 2);
    s += arrow(488, 91, 542, 91);
    s += box(550, 64, 130, 54, 'PATCH|a real diff', { fill: '#fff' });
    s += arrow(688, 91, 742, 91);
    s += box(750, 64, 110, 54, 'TESTS|fail → pass', { fill: MARK });
    s += step(738, 64, 3);

    // the arc, as bars (score × 1.55 px, baseline y = 330)
    const base = 330;
    s += txt(60, 158, '% ISSUES RESOLVED', { fs: 11.5, fill: GREY });
    s += `<line x1="60" y1="${base}" x2="840" y2="${base}" stroke="${INK}" stroke-width="1.5"/>`;

    s += `<rect x="120" y="${base - 4}" width="90" height="4" fill="${INK}"/>`;
    s += txt(165, base - 14, '1.96%', { a: 'middle', fs: 13, w: 600 });
    s += txt(165, base + 20, 'best assisted model', { a: 'middle', fs: 11.5, fill: GREY });
    s += txt(165, base + 36, 'original paper, 2023', { a: 'middle', fs: 11.5, fill: GREY });

    s += `<rect x="400" y="${base - 76}" width="90" height="76" fill="${SOFT}" stroke="${INK}" stroke-width="1.5"/>`;
    s += txt(445, base - 86, '49.0%', { a: 'middle', fs: 13, w: 600 });
    s += txt(445, base + 20, 'Claude 3.5 Sonnet', { a: 'middle', fs: 11.5, fill: GREY });
    s += txt(445, base + 36, 'Oct 2024 · Verified', { a: 'middle', fs: 11.5, fill: GREY });

    s += apulse(`<rect x="680" y="${base - 125}" width="90" height="125" fill="${MARK}" stroke="${INK}" stroke-width="1.5"/>`);
    s += hl(650, base - 152, 150, 20);
    s += txt(725, base - 137, '80.9%', { a: 'middle', fs: 15, w: 600 });
    s += txt(725, base + 20, 'Claude Opus 4.5', { a: 'middle', fs: 11.5, fill: GREY });
    s += txt(725, base + 36, 'Nov 2025 · Verified', { a: 'middle', fs: 11.5, fill: GREY });

    s += arrow(215, base - 20, 395, base - 60, { color: GREY, dash: '3 3' });
    s += arrow(495, base - 90, 675, base - 118, { color: GREY, dash: '3 3' });

    s += note(60, 200, ['SWE-bench Verified (Aug 2024):', '500-problem human-validated subset;', 'fixed unsolvable and under-specified', 'tasks. it is the number everyone quotes.'], {});
    s += txt(60, 384, 'two years from research curiosity (1.96%) to a shipped product category —', { fs: 11.5, fill: GREY });
    s += txt(60, 399, 'tests are the free verifier that made coding the beachhead.', { fs: 11.5, fill: GREY });
    return wrap(410, s);
  } },

/* --------------------------------------------------- 3 · SWE-agent */
{ id: 'swe-agent-aci',
  paper: 'SWE-agent — Yang et al., 2024 (arXiv 2405.15793)',
  claim: 'The agent–computer interface is a first-class design object: the identical model with a better-shaped file viewer and edit commands solves far more issues.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SWE-AGENT · THE AGENT–COMPUTER INTERFACE IS A DESIGN OBJECT', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // one model, two interfaces
    s += apulse(box(60, 130, 160, 70, 'SAME MODEL|weights unchanged|prompt unchanged', { fill: MARK }));
    s += step(48, 130, 1);

    // path A: raw shell
    s += arrow(220, 148, 296, 108, { color: GREY });
    s += box(304, 76, 250, 70, 'RAW SHELL|bash + cat: full-file dumps|no feedback on bad edits', { fill: '#fff', stroke: GREY });
    s += arrow(554, 111, 630, 111, { color: GREY });
    s += box(638, 84, 200, 54, 'FEWER ISSUES|solved', { fill: '#fff', stroke: GREY });

    // path B: designed ACI
    s += aflow(220, 182, 296, 222, { color: BLUE });
    s += box(304, 196, 250, 84, 'DESIGNED ACI|windowed file viewer|edit command + lint guardrail|compact search results', { fill: SOFT });
    s += step(292, 196, 2);
    s += aflow(554, 238, 630, 238, { color: BLUE });
    s += hl(634, 210, 208, 62);
    s += box(638, 210, 200, 62, 'FAR MORE ISSUES|solved — same model,|better interface', { fill: '#fff' });
    s += step(626, 210, 3);

    s += note(304, 306, ['every observation is shaped for the context window: short windows instead of', 'full files, actionable errors instead of silent failures'], {});
    s += txt(60, 356, 'tool design is a research result, not a taste question — the interface, not the model, was the bottleneck.', { fs: 12, fill: GREY });
    return wrap(380, s);
  } },

/* ------------------------------------- 4 · Building effective agents */
{ id: 'effective-agents-taxonomy',
  paper: 'Building effective agents — Anthropic engineering, December 2024',
  claim: 'The field\'s most-cited taxonomy: most production wins are workflows on predefined code paths; graduate to an agent only when the simpler pattern demonstrably fails.',
  svg() {
    let s = '';
    s += txt(60, 30, 'BUILDING EFFECTIVE AGENTS · WORKFLOWS FIRST, AGENTS LAST', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // the decision
    s += box(60, 70, 220, 64, 'CAN YOU ENUMERATE|the steps in advance?', { fill: MARK });
    s += step(48, 70, 1);

    // yes → workflows
    s += arrow(280, 88, 356, 88, { label: 'yes', ly: -6 });
    s += box(364, 64, 200, 48, 'WORKFLOW|LLM calls on fixed paths', { fill: SOFT });
    s += step(352, 64, 2);
    const wf = ['prompt chaining', 'routing', 'parallelization', 'orchestrator–workers', 'evaluator–optimizer'];
    wf.forEach((w, i) => {
      s += box(364 + (i % 2) * 210, 126 + Math.floor(i / 2) * 44, 200, 36, w, { fill: '#fff', fs: 12 });
    });
    s += arrow(564, 88, 620, 88, { color: GREY });
    s += txt(628, 92, 'five named patterns', { fs: 11.5, fill: GREY });

    // no → agent
    s += arrow(170, 134, 170, 210);
    s += txt(180, 164, 'no — simpler pattern', { fs: 11.5, fill: INK });
    s += txt(180, 180, 'demonstrably fails', { fs: 11.5, fill: INK });
    s += box(60, 218, 220, 64, 'AGENT|model directs its own|process via tools', { fill: '#fff' });
    s += step(48, 218, 3);
    s += aflow(280, 250, 356, 250, { color: BLUE });
    s += txt(318, 296, 'think → act → observe', { a: 'middle', fs: 11.5, fill: BLUE });
    s += box(364, 226, 200, 48, 'LOOP UNTIL DONE|or a hard cap', { fill: SOFT });

    // the advice
    s += `<line x1="60" y1="308" x2="840" y2="308" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += hl(60, 320, 340, 20);
    s += txt(64, 335, 'USE THE SIMPLEST THING THAT WORKS', { fs: 12.5, w: 600 });
    s += txt(60, 360, 'if you can draw the flowchart, write the flowchart — most production wins come from workflows, not agents.', { fs: 12, fill: GREY });
    s += note(620, 335, ['the advice nobody', 'wants to hear'], {});
    return wrap(384, s);
  } },
],

verify: [

/* ---------------------------------------------------- MT-Bench */
{ id: 'mtbench-judge-agreement',
  paper: 'Judging LLM-as-a-Judge with MT-Bench & Chatbot Arena — Zheng et al., NeurIPS 2023',
  claim: 'Strong LLM judges reach over 80% agreement with human preferences — the human–human level — but carry position, verbosity and self-enhancement biases.',
  svg() {
    let s = '';
    s += txt(60, 30, 'LLM-AS-A-JUDGE · MT-BENCH: 80 questions, 3K expert votes, 30K human conversations', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) two answers to the same question
    s += step(56, 72, 1);
    s += box(60, 84, 150, 50, 'ANSWER A|from model 1', { fill: SOFT });
    s += box(60, 148, 150, 50, 'ANSWER B|from model 2', { fill: SOFT });

    // (2) judge, both orders
    s += step(296, 72, 2);
    s += box(300, 96, 180, 90, 'STRONG LLM JUDGE|pairwise: which is|better, A or B?');
    s += aflow(214, 109, 296, 125, { label: 'order (A,B)' });
    s += arrow(214, 173, 296, 157, { label: 'order (B,A)', ly: 16 });
    s += txt(300, 206, 'run BOTH slot orders; if the two verdicts', { fs: 11.5, fill: GREY });
    s += txt(300, 221, 'disagree, record a tie', { fs: 11.5, fill: GREY });

    // (3) compare with humans
    s += arrow(484, 141, 556, 141, { label: 'verdict' });
    s += step(552, 72, 3);
    s += box(560, 84, 130, 50, 'JUDGE VOTES|A / B / tie');
    s += box(560, 148, 130, 50, 'HUMAN VOTES|3K experts');
    s += arrow(694, 109, 730, 132, { color: GREY });
    s += arrow(694, 173, 730, 150, { color: GREY });
    s += apulse(hl(724, 122, 116, 38) + txt(782, 140, '>80%', { a: 'middle', fs: 17, w: 600 }) + txt(782, 155, 'agreement', { a: 'middle', fs: 11 }));
    s += note(700, 196, ['same level as', 'human-human agreement:', 'the judge is as reliable', 'as another person'], {});

    // documented biases
    s += `<line x1="60" y1="252" x2="840" y2="252" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 274, 'THE THREE BIASES THE PAPER NAMES', { fs: 12, w: 600, fill: RED });
    s += box(60, 286, 190, 52, 'POSITION|first slot favoured', { stroke: RED });
    s += box(270, 286, 190, 52, 'VERBOSITY|longer scores higher', { stroke: RED });
    s += box(480, 286, 190, 52, 'SELF-ENHANCEMENT|prefers its own style', { stroke: RED });
    s += note(690, 300, ['swap-and-aggregate cures', 'position bias only; the', 'other two need a rubric.'], {});
    return wrap(370, s);
  } },

/* ----------------------------------------------- Self-Consistency */
{ id: 'self-consistency-voting',
  paper: 'Self-Consistency Improves Chain of Thought — Wang et al., ICLR 2023',
  claim: 'Sampling several reasoning chains and majority-voting the answers gains +17.9% on GSM8K over greedy chain-of-thought — verification with no judge at all.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SELF-CONSISTENCY · sample many reasoning paths, keep the answer they agree on', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) one question
    s += step(56, 74, 1);
    s += box(60, 86, 150, 78, 'QUESTION|one math word|problem (GSM8K)', { fill: SOFT });

    // (2) sampled chains
    s += step(266, 74, 2);
    s += txt(282, 78, 'sample with temperature — paths diverge', { fs: 11.5, fill: GREY });
    s += box(270, 86, 250, 40, 'CHAIN 1  ...half of 36... = 18');
    s += box(270, 134, 250, 40, 'CHAIN 2  ...adds 8 twice... = 26');
    s += box(270, 182, 250, 40, 'CHAIN 3  ...36 - 18... = 18');
    s += aflow(214, 125, 266, 106);
    s += arrow(214, 125, 266, 154);
    s += arrow(214, 125, 266, 202, { curve: 20 });

    // (3) majority vote → answer
    s += step(576, 74, 3);
    s += box(580, 110, 140, 66, 'MAJORITY VOTE|18: 2 votes|26: 1 vote');
    s += arrow(524, 106, 576, 130);
    s += arrow(524, 154, 576, 143);
    s += arrow(524, 202, 576, 156);
    s += aflow(724, 143, 764, 143);
    s += box(768, 116, 72, 54, '18', { fill: MARK });
    s += txt(804, 186, 'answer', { a: 'middle', fs: 11, fill: GREY });

    // headline numbers
    s += `<line x1="60" y1="244" x2="840" y2="244" stroke="${GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += apulse(hl(60, 258, 220, 22) + txt(66, 275, 'GSM8K  +17.9%', { fs: 15, w: 600 }));
    s += txt(300, 275, 'SVAMP +11.0%   ·   AQuA +12.2%   (over greedy chain-of-thought)', { fs: 12.5 });
    s += note(60, 302, ['why it works: sampled reasoning paths diverge, so their errors decorrelate —',
                        'agreement itself is evidence of correctness. no judge model anywhere in the loop.'], {});
    return wrap(340, s);
  } },

/* ------------------------------------------ Process reward models */
{ id: 'process-reward-model',
  paper: "Let's Verify Step by Step — Lightman et al., 2023",
  claim: 'Grading every reasoning step instead of only the final answer lets a process-supervised reward model solve 78% of a MATH test subset, beating outcome supervision.',
  svg() {
    let s = '';
    s += txt(60, 30, 'PROCESS vs OUTCOME SUPERVISION · where does the grade attach?', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) a stepped solution
    s += step(56, 72, 1);
    s += txt(76, 76, 'one sampled solution, as steps', { fs: 11.5, fill: GREY });
    const steps = ['STEP 1', 'STEP 2', 'STEP 3', 'ANSWER'];
    steps.forEach((t, i) => {
      s += box(60 + i * 130, 88, 112, 40, t, { fill: i === 3 ? SOFT : '#fff' });
      if (i < 3) s += arrow(172 + i * 130, 108, 190 + i * 130, 108);
    });

    // (2) two graders
    s += step(56, 168, 2);
    s += box(60, 180, 250, 58, 'OUTCOME RM|grades ANSWER only|(right / wrong)', { stroke: GREY });
    s += arrow(450, 132, 240, 176, { color: GREY, dash: '3 3' });
    s += box(340, 180, 250, 58, 'PROCESS RM (PRM)|grades EVERY step', { stroke: GREEN });
    s += aflow(190, 132, 430, 176);
    s += txt(76, 144, 'error hidden in', { fs: 11.5, fill: RED });
    s += txt(76, 158, 'step 3: ORM cannot', { fs: 11.5, fill: RED });
    s += txt(76, 172, 'see it, PRM flags it', { fs: 11.5, fill: RED });
    s += txt(340, 254, 'trained on PRM800K: 800K step-level human labels', { fs: 11.5, fill: GREY });

    // (3) PRM as search signal
    s += step(56, 292, 3);
    s += box(60, 304, 230, 50, 'SAMPLE N SOLUTIONS|rank all by PRM score', { fill: SOFT });
    s += aflow(294, 329, 344, 329, { label: 'pick top' });
    s += box(348, 304, 170, 50, 'BEST-OF-N|by process score');
    s += arrow(522, 329, 562, 329);
    s += apulse(hl(566, 306, 180, 46) + txt(656, 328, '78% solved', { a: 'middle', fs: 16, w: 600 }) + txt(656, 344, 'MATH test subset', { a: 'middle', fs: 11 }));
    s += note(700, 200, ['beats outcome supervision', 'on the same sampler —', 'step-level credit turned', 'verification into a', 'search signal'], {});
    return wrap(384, s);
  } },

/* ------------------------------------------------ Constitutional AI */
{ id: 'constitutional-ai-rlaif',
  paper: 'Constitutional AI — Bai et al., 2022',
  claim: 'The model critiques and revises its own outputs against written principles, and an AI preference model replaces most human harmlessness labels.',
  svg() {
    let s = '';
    s += txt(60, 30, 'CONSTITUTIONAL AI · verification pushed into training time', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) critique-revise loop
    s += step(56, 72, 1);
    s += txt(76, 76, 'supervised phase: critique -> revise, guided by a written constitution', { fs: 11.5, fill: GREY });
    s += box(60, 88, 160, 54, 'DRAFT RESPONSE|may be harmful');
    s += aflow(224, 115, 274, 115, { label: 'critique' });
    s += box(278, 88, 180, 54, 'SELF-CRITIQUE|vs a named principle');
    s += aflow(462, 115, 512, 115, { label: 'revise' });
    s += box(516, 88, 160, 54, 'REVISION|harm removed');
    s += arrow(596, 146, 596, 176);
    s += txt(606, 166, 'finetune on revisions', { fs: 11.5 });
    s += box(278, 160, 180, 40, 'CONSTITUTION|written principles', { fill: MARK });
    s += arrow(368, 156, 368, 146, { color: GREY });

    // (2) AI preference labels
    s += step(56, 232, 2);
    s += box(60, 244, 160, 54, 'RESPONSE PAIR|A vs B');
    s += arrow(224, 271, 274, 271);
    s += box(278, 244, 200, 54, 'AI PREFERENCE MODEL|judges with constitution');
    s += arrow(482, 271, 532, 271, { label: 'labels' });

    // (3) RLAIF
    s += step(536, 232, 3);
    s += box(536, 244, 140, 54, 'RLAIF|RL from AI|feedback');
    s += apulse(hl(690, 244, 150, 54) + txt(765, 266, 'replaces most', { a: 'middle', fs: 12, w: 600 }) + txt(765, 282, 'human harm labels', { a: 'middle', fs: 11 }));
    s += note(60, 326, ['why: a rubric applied at scale, before deployment rather than after —',
                        'the judge sits inside the training pipeline, and the published principles are the rubric.'], {});
    return wrap(366, s);
  } },

/* --------------------------------------------------- Self-preference */
{ id: 'self-preference-recognition',
  paper: 'LLM Evaluators Recognize and Favor Their Own Generations — Panickssery et al., NeurIPS 2024',
  claim: 'Evaluator models score their own outputs higher, and the inflation grows linearly with how well the model recognizes its own text.',
  svg() {
    let s = '';
    s += txt(60, 30, 'SELF-PREFERENCE · why the judge must not be the generator', { fs: 12.5, w: 600, fill: BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${BLUE}" stroke-width="1"/>`;

    // (1) same model writes and judges
    s += step(56, 72, 1);
    s += box(60, 84, 170, 50, 'MODEL M WRITES|its own summary', { fill: SOFT });
    s += box(60, 148, 170, 50, "OTHER MODEL'S|summary", { fill: SOFT });
    s += aflow(234, 109, 294, 130, { label: 'judge = M' });
    s += arrow(234, 173, 294, 152);

    // (2) inflated verdict
    s += step(290, 72, 2);
    s += box(298, 108, 170, 64, 'MODEL M JUDGES|scores its own|output higher');
    s += arrow(472, 140, 516, 140, { color: RED });
    s += txt(520, 132, 'inflated score for', { fs: 12, fill: RED });
    s += txt(520, 148, 'its own text', { fs: 12, fill: RED });

    // (3) the measured link, as a small plot
    s += step(56, 226, 3);
    const px = 90, py = 328, pw = 300, ph = 84;
    s += `<line x1="${px}" y1="${py}" x2="${px + pw}" y2="${py}" stroke="${INK}" stroke-width="1.2"/>`;
    s += `<line x1="${px}" y1="${py}" x2="${px}" y2="${py - ph}" stroke="${INK}" stroke-width="1.2"/>`;
    s += txt(px + pw / 2, py + 18, 'self-recognition accuracy ->', { a: 'middle', fs: 11, fill: GREY });
    s += txt(px - 14, py - ph - 8, 'self-preference', { fs: 11, fill: GREY });
    [[30, 16], [90, 30], [150, 42], [210, 58], [270, 72]].forEach(([dx, dy]) => {
      s += `<circle cx="${px + dx}" cy="${py - dy}" r="3.5" fill="${BLUE}"/>`;
    });
    s += `<line x1="${px + 16}" y1="${py - 10}" x2="${px + 286}" y2="${py - 78}" stroke="${RED}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    s += apulse(hl(410, 268, 250, 22) + txt(416, 285, 'LINEAR LINK, measured properly', { fs: 13, w: 600 }));
    s += txt(410, 310, 'the better a model recognizes its', { fs: 12 });
    s += txt(410, 326, 'own text, the more it favours', { fs: 12 });
    s += txt(410, 342, 'that text as a judge', { fs: 12 });

    // the fix
    s += box(690, 268, 150, 58, 'FIX|judge != generator|different model', { stroke: GREEN });
    s += note(690, 348, ['or at minimum a different', 'prompt persona'], {});
    return wrap(420, s);
  } },
],

rl: [
{ id: 'linucb-yahoo',
  paper: 'Contextual-Bandit News Recommendation — Li et al., WWW 2010',
  claim: 'LinUCB on the Yahoo! front page lifted clicks 12.5% over a context-free bandit on 33M+ events, and the gap widened as data got scarcer.',
  svg() {
    let s = '';
    s += txt(60, 30, 'LINUCB · ONE NEWS SLOT ON THE YAHOO! FRONT PAGE', { fs: 12.5, w: 600, fill: PFC_rl.BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_rl.BLUE}" stroke-width="1"/>`;

    // (1) context
    s += step(56, 78, 1);
    s += box(60, 90, 195, 72, 'CONTEXT xₜ|user features ×|article features', { fill: PFC_rl.SOFT });
    s += aflow(255, 126, 295, 126);
    s += txt(270, 174, 'per visit', { a: 'middle', fs: 11.5 });

    // (2) per-arm linear UCB scores
    s += step(301, 78, 2);
    s += box(305, 62, 280, 42, 'article a₁ · x̄₁ + α·√(xᵀA₁⁻¹x)', { fs: 12 });
    s += apulse(box(305, 110, 280, 42, 'article a₂ · x̄₂ + α·√(xᵀA₂⁻¹x)  ← max', { fill: PFC_rl.MARK, fs: 12 }));
    s += box(305, 158, 280, 42, 'article a₃ · x̄₃ + α·√(xᵀA₃⁻¹x)', { fs: 12 });
    s += txt(445, 220, 'mean estimate + uncertainty bonus, per arm', { a: 'middle', fs: 11.5, fill: PFC_rl.GREY });

    // (3) serve, observe, update
    s += step(631, 78, 3);
    s += aflow(585, 131, 630, 131);
    s += box(635, 104, 205, 56, 'SERVE argmax|observe click ∈ {0,1}');
    s += arrow(737, 160, 737, 196, { color: PFC_rl.GREY });
    s += box(635, 196, 205, 48, 'update served arm only|Aₐ, bₐ ridge update', { fill: PFC_rl.SOFT, fs: 12 });
    s += arrow(635, 220, 445, 220 + 36, { color: PFC_rl.GREY, dash: '3 3', curve: -60 });

    // headline
    s += hl(60, 288, 520, 22);
    s += txt(66, 304, '+12.5% click lift vs context-free bandit · 33M+ logged events', { fs: 13.5, w: 600 });
    s += note(60, 336, [
      'why the bonus pays: the gap over context-free widened as data got scarcer —',
      'uncertainty-directed exploration matters most exactly where means are unreliable.'
    ]);
    return wrap(380, s);
  } },

{ id: 'topk-offpolicy-dr',
  paper: 'Top-K Off-Policy REINFORCE — Chen et al., 2019 · Doubly Robust OPE — Dudík et al., 2011',
  claim: 'RL made deployable in a production recommender by training on logged traffic instead of exploring online, with doubly robust estimation making the logged evaluation trustworthy.',
  svg() {
    let s = '';
    s += txt(60, 30, 'THE RANKING ↔ RL BRIDGE · LEARN AND EVALUATE FROM LOGS', { fs: 12.5, w: 600, fill: PFC_rl.BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_rl.BLUE}" stroke-width="1"/>`;

    // (1) logs
    s += step(56, 82, 1);
    s += box(60, 94, 220, 66, 'LOGGED TRAFFIC|(context, action,|propensity, reward)', { fill: PFC_rl.SOFT });
    s += aflow(280, 127, 328, 127);

    // (2) off-policy REINFORCE with corrections
    s += step(324, 82, 2);
    s += box(332, 94, 230, 66, 'REINFORCE|clipped importance|weights π_new/π_logged');
    s += arrow(447, 160, 447, 190, { color: PFC_rl.GREY });
    s += box(332, 190, 230, 44, 'TOP-K CORRECTION|slate of K, not one item', { fs: 12 });
    s += aflow(562, 127, 610, 127);

    // (3) deployed policy, no online exploration
    s += step(606, 82, 3);
    s += box(614, 94, 226, 66, 'NEW POLICY|shipped in a production|video recommender');
    s += hl(614, 172, 226, 36);
    s += txt(727, 194, 'zero online exploration', { a: 'middle', fs: 12.5, w: 600 });

    // bottom strip: doubly robust
    s += `<line x1="60" y1="256" x2="840" y2="256" stroke="${PFC_rl.GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += txt(60, 280, 'DUDÍK ET AL. 2011 — CAN YOU TRUST A NUMBER COMPUTED FROM LOGS?', { fs: 12, w: 600, fill: PFC_rl.BLUE });
    s += box(60, 294, 210, 48, 'model estimate|biased, low variance', { fs: 12 });
    s += txt(288, 322, '+', { fs: 16, w: 600 });
    s += box(310, 294, 230, 48, 'IPS correction|unbiased, high variance', { fs: 12 });
    s += arrow(540, 318, 588, 318);
    s += box(592, 294, 248, 48, 'DOUBLY ROBUST|trustworthy logged evaluation', { fill: PFC_rl.MARK, fs: 12 });
    s += note(60, 366, [
      'why it matters: propensity logging plus these estimators lets a policy be trained and judged before it is ever served.'
    ]);
    return wrap(390, s);
  } }
],

rlhf: [
{ id: 'christiano-preferences',
  paper: 'Deep RL from Human Preferences — Christiano et al., NeurIPS 2017',
  claim: 'Complex behaviours were learned from human feedback on less than 1% of the agent’s interactions, by fitting a reward model to clip comparisons.',
  svg() {
    let s = '';
    s += txt(60, 30, 'RLHF, THE ORIGIN LOOP · HUMANS JUDGE, A MODEL REWARDS', { fs: 12.5, w: 600, fill: PFC_rl.BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_rl.BLUE}" stroke-width="1"/>`;

    // (1) agent acts
    s += step(56, 82, 1);
    s += box(60, 94, 180, 66, 'AGENT|acts in the|environment', { fill: PFC_rl.SOFT });
    s += aflow(240, 127, 292, 127);

    // (2) trajectory pairs to a human
    s += step(288, 82, 2);
    s += box(296, 94, 210, 66, 'TRAJECTORY PAIRS|two short clips|of behaviour');
    s += aflow(506, 127, 558, 127, { label: 'which is better?', ly: -45 });
    s += apulse(box(562, 94, 200, 66, 'HUMAN|picks the preferred|clip of the two', { fill: PFC_rl.MARK }));

    // (3) reward model, then RL against it
    s += step(558, 196, 3);
    s += arrow(662, 160, 662, 200, { color: PFC_rl.GREY });
    s += txt(672, 186, 'comparisons', { fs: 11, fill: PFC_rl.GREY });
    s += box(562, 200, 200, 58, 'REWARD MODEL r̂|fit to preferences');
    s += arrow(562, 229, 300, 229);
    s += box(96, 200, 204, 58, 'RL UPDATE|maximise r̂, not|a true reward');
    s += arrow(150, 200, 150, 160, { color: PFC_rl.GREY, dash: '3 3' });
    s += txt(160, 184, 'loop', { fs: 11, fill: PFC_rl.GREY });

    // headline
    s += hl(60, 288, 560, 22);
    s += txt(66, 304, 'complex behaviours from human feedback on less than 1% of interactions', { fs: 13.5, w: 600 });
    s += note(60, 336, [
      'why: judging which of two clips is better is far cheaper than demonstrating',
      'or hand-scoring the behaviour — so a little human time trains a lot of reward.'
    ]);
    return wrap(380, s);
  } },

{ id: 'ouyang-rlhf-pipeline',
  paper: 'InstructGPT paper (the RLHF pipeline) — Ouyang et al., 2022',
  claim: 'The SFT → reward model → PPO-with-KL pipeline made labellers prefer a 1.3B RLHF-tuned model over a 100×-larger untuned one.',
  svg() {
    let s = '';
    s += txt(60, 30, 'THE PIPELINE EVERYONE COPIED · SFT → RM → PPO', { fs: 12.5, w: 600, fill: PFC_rl.BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_rl.BLUE}" stroke-width="1"/>`;

    // three stages
    s += step(56, 82, 1);
    s += box(60, 94, 220, 84, 'SFT|supervised fine-tune|on human|demonstrations', { fill: PFC_rl.SOFT });
    s += aflow(280, 136, 330, 136);

    s += step(326, 82, 2);
    s += box(330, 94, 230, 84, 'REWARD MODEL|trained on pairwise|comparisons of|model outputs');
    s += aflow(560, 136, 610, 136);

    s += step(606, 82, 3);
    s += box(610, 94, 230, 84, 'PPO|maximise reward|− β · KL to the|reference model');
    s += arrow(725, 178, 725, 208, { color: PFC_rl.GREY });
    s += box(610, 208, 230, 40, 'KL leash: stay near SFT model', { fill: PFC_rl.SOFT, fs: 11.5 });

    // headline
    s += hl(60, 276, 640, 22);
    s += txt(66, 292, 'labellers preferred the 1.3B RLHF model over a 100×-larger untuned model', { fs: 13.5, w: 600 });
    s += note(60, 322, [
      'the sentence that launched alignment-as-post-training: post-training beat two',
      'orders of magnitude of scale on human preference — compute went to the recipe.'
    ]);
    return wrap(360, s);
  } },

{ id: 'dpo-grpo-simplifications',
  paper: 'DPO — Rafailov et al., 2023 · GRPO (DeepSeekMath) — Shao et al., 2024',
  claim: 'DPO deletes the reward model, sampling loop and critic; GRPO deletes the value network by using group statistics as the baseline.',
  svg() {
    let s = '';
    s += txt(60, 30, 'TWO SIMPLIFICATIONS OF THE PPO-RLHF STACK', { fs: 12.5, w: 600, fill: PFC_rl.BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_rl.BLUE}" stroke-width="1"/>`;

    // the stack being simplified
    s += step(56, 66, 1);
    s += box(60, 78, 780, 40, 'PPO-RLHF · policy + reference + reward model + value network + sampling loop', { fill: PFC_rl.SOFT, fs: 12.5 });

    // left: DPO
    s += step(56, 148, 2);
    s += txt(80, 152, 'DPO — no RL loop at all', { fs: 12.5, w: 600 });
    s += box(60, 164, 360, 48, 'paired data · y_w chosen, y_l rejected', { fs: 12 });
    s += aflow(240, 212, 240, 244);
    s += box(60, 244, 360, 54, 'L = −log σ( β·Δ log π/π_ref )|Z(x) cancels between the pair', { fs: 12 });
    s += txt(60, 322, 'deleted: reward model · sampling loop · critic', { fs: 12, fill: PFC_rl.RED });
    s += txt(60, 342, 'runs like supervised fine-tuning', { fs: 11.5, fill: PFC_rl.GREY });

    // right: GRPO
    s += step(496, 148, 3);
    s += txt(520, 152, 'GRPO — keep RL, delete the critic', { fs: 12.5, w: 600 });
    s += box(500, 164, 340, 48, 'sample G completions per prompt, score all', { fs: 12 });
    s += aflow(670, 212, 670, 244);
    s += apulse(box(500, 244, 340, 54, 'Âᵢ = (rᵢ − mean(r₁..r_G)) / std(r₁..r_G)|siblings from the same prompt are the critic', { fill: PFC_rl.MARK, fs: 12 }));
    s += txt(500, 322, 'deleted: value network — a policy-sized model', { fs: 12, fill: PFC_rl.RED });
    s += txt(500, 342, 'less memory, one less thing to tune', { fs: 11.5, fill: PFC_rl.GREY });

    s += `<line x1="460" y1="150" x2="460" y2="345" stroke="${PFC_rl.GREY}" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += note(60, 376, [
      'why they won: both remove whole models from the loop —',
      'value-free and reward-model-free methods won by being cheaper, not cleverer.'
    ]);
    return wrap(400, s);
  } },

{ id: 'tulu3-rlvr',
  paper: 'Tülu 3 — Lambert et al., 2024',
  claim: 'RLVR replaces the learned reward model with a deterministic checker, so the training signal cannot be flattered or drifted against.',
  svg() {
    let s = '';
    s += txt(60, 30, 'RLVR · REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS', { fs: 12.5, w: 600, fill: PFC_rl.BLUE });
    s += `<line x1="60" y1="40" x2="840" y2="40" stroke="${PFC_rl.BLUE}" stroke-width="1"/>`;

    // (1) prompt → policy
    s += step(56, 82, 1);
    s += box(60, 94, 180, 64, 'PROMPT|math or code task', { fill: PFC_rl.SOFT });
    s += aflow(240, 126, 288, 126);
    s += box(292, 94, 190, 64, 'POLICY|samples an answer');
    s += aflow(482, 126, 530, 126);

    // (2) the swap: checker instead of reward model
    s += step(526, 82, 2);
    s += apulse(box(534, 94, 200, 64, 'CHECKER|unit tests, answer key|deterministic', { fill: PFC_rl.MARK }));
    s += box(534, 186, 200, 46, 'learned reward model|hackable, drifts', { fs: 11.5 });
    s += `<line x1="534" y1="186" x2="734" y2="232" stroke="${PFC_rl.RED}" stroke-width="1.5"/>`;
    s += `<line x1="534" y1="232" x2="734" y2="186" stroke="${PFC_rl.RED}" stroke-width="1.5"/>`;
    s += txt(744, 212, 'replaced', { fs: 11.5, fill: PFC_rl.RED });

    // (3) binary reward → RL update
    s += step(770, 82, 3);
    s += arrow(734, 126, 774, 126);
    s += box(778, 94, 62, 64, 'r ∈|{0,1}');
    s += arrow(560, 158, 420, 246, { color: PFC_rl.GREY, dash: '3 3', curve: -40 });
    s += box(240, 250, 250, 46, 'RL UPDATE|GRPO-style, on r', { fill: PFC_rl.SOFT, fs: 12 });

    // headline + note
    s += hl(60, 316, 470, 22);
    s += txt(66, 332, 'the reward model becomes a deterministic checker', { fs: 13.5, w: 600 });
    s += note(556, 320, [
      'why: a checker cannot be flattered —',
      'reward hacking needs a learnable judge.'
    ]);
    return wrap(370, s);
  } }
],

ads: [

  /* ------------------------------------------------ GSP / EOS, AER 2007 */
  { id: 'gsp-not-vickrey-2007',
    paper: 'Internet Advertising and the Generalized Second-Price Auction — Edelman, Ostrovsky & Schwarz, AER 2007',
    claim: 'With more than one slot, GSP is not Vickrey: a bidder can profit by shading below value to hold a cheaper slot, yet envy-free equilibria exist and revenue-dominate VCG’s truthful one.',
    svg() {
      let s = '';
      s += txt(60, 32, 'GSP · TWO SLOTS, AND WHY TRUTH-TELLING IS NOT AN EQUILIBRIUM', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      // the market: two slots, three bidders
      s += box(60, 66, 150, 44, 'SLOT 1|q₁ = 200 clicks/day', { fill: SOFT });
      s += box(60, 122, 150, 44, 'SLOT 2|q₂ = 100 clicks/day', { fill: SOFT });
      s += txt(60, 194, 'X values a click at $5.00', { fs: 12 });
      s += txt(60, 212, 'rivals bid: Y $4.00 · Z $2.00', { fs: 12, fill: GREY });
      s += txt(60, 248, 'price for slot i:  pᵢ = bᵢ₊₁ — the next bid down', { fs: 12 });
      s += txt(60, 265, 'equal ad quality here → pay the next bid down', { fs: 11.5, fill: GREY });

      // case 1: truthful
      s += box(250, 66, 280, 30, 'CASE 1 · X BIDS ITS VALUE $5.00');
      s += step(238, 81, 1);
      s += txt(250, 122, 'ranking: X $5.00 → slot 1 · Y → slot 2', { fs: 12 });
      s += txt(250, 142, 'X pays the next bid: $4.00 / click', { fs: 12 });
      s += txt(250, 162, 'surplus = ($5.00 − $4.00) × 200 clicks', { fs: 12 });
      s += txt(250, 183, '= $200 / day', { fs: 12.5, w: 600 });

      // case 2: shaded
      s += aflow(534, 81, 566, 81);
      s += box(570, 66, 280, 30, 'CASE 2 · X SHADES DOWN TO $3.00');
      s += step(558, 66, 2);
      s += txt(570, 122, 'ranking: Y $4.00 → slot 1 · X → slot 2', { fs: 12 });
      s += txt(570, 142, 'X pays the next bid: $2.00 / click', { fs: 12 });
      s += txt(570, 162, 'surplus = ($5.00 − $2.00) × 100 clicks', { fs: 12 });
      s += hl(566, 170, 130, 20);
      s += txt(570, 185, '= $300 / day', { fs: 12.5, w: 600 });
      s += txt(704, 185, '← shading wins', { fs: 12, fill: GREEN, w: 600 });

      // headline
      s += step(48, 326, 3);
      s += hl(60, 316, 620, 20);
      s += txt(64, 331, 'RESULT: shading earns $300 > $200 — truth-telling is NOT an equilibrium in GSP', { fs: 12.5, w: 600 });
      s += note(60, 360, [
        'why GSP survives anyway: its envy-free equilibria exist and revenue-dominate',
        'VCG’s truthful equilibrium — bidders shade, but the mechanism still clears well.'
      ]);
      return wrap(392, s);
    } },

  /* -------------------------------------- Varian & Harris, VCG, 2014 */
  { id: 'vcg-theory-practice-2014',
    paper: 'VCG in Theory and Practice — Varian & Harris, AER P&P 2014',
    claim: 'VCG charges each winner the value everyone else loses because of its presence, restoring dominant-strategy truthfulness — Facebook launched on it, while Google’s search auction stayed GSP out of inertia and retraining cost.',
    svg() {
      let s = '';
      s += txt(60, 32, 'VCG · PAY THE HARM YOU CAUSE — AND WHO ACTUALLY SHIPPED IT', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      // the market
      s += box(60, 66, 150, 40, 'SLOT 1|100 clicks', { fill: SOFT });
      s += box(60, 116, 150, 40, 'SLOT 2|60 clicks', { fill: SOFT });
      s += step(72, 58, 1);
      s += txt(60, 182, 'bids: A $10 · B $6 · C $4', { fs: 12 });
      s += txt(60, 200, 'A → slot 1, B → slot 2', { fs: 11.5, fill: GREY });

      // the payment, worked
      s += step(292, 74, 2);
      s += txt(310, 78, 'WHAT A PAYS — the welfare others lose because A exists:', { fs: 12, w: 600 });
      s += txt(310, 104, 'without A:  B→slot 1, C→slot 2  =  6×100 + 4×60  =  $840', { fs: 12 });
      s += txt(310, 124, 'with A:     B→slot 2, C→nothing  =  6×60           =  $360', { fs: 12 });
      s += hl(306, 134, 342, 20);
      s += txt(310, 149, 'A pays 840 − 360 = $480   ($4.80 / click)', { fs: 12.5, w: 600 });
      s += txt(310, 174, 'B pays what C loses: 4×60 = $240   ($4.00 / click)', { fs: 12, fill: GREY });
      s += txt(310, 194, 'an unfamiliar, hard-to-explain price — the cost of truthfulness', { fs: 11.5, fill: GREY });

      // the property
      s += hl(60, 226, 660, 20);
      s += txt(64, 241, 'truth-telling is a DOMINANT strategy — your own bid never touches your own price', { fs: 12.5, w: 600 });

      // deployments
      s += step(72, 300, 3);
      s += box(60, 288, 250, 56, 'META / FACEBOOK|launched ads on VCG — the major|deployment; generalises to feeds', { fill: MARK });
      s += box(340, 288, 250, 56, 'GOOGLE SEARCH|stayed GSP — inertia and|advertiser retraining costs');
      s += arrow(310, 316, 336, 316, { color: GREY, sw: 1 });
      s += note(620, 300, [
        'same theory, opposite choices:',
        'switching costs decide, not',
        'elegance — stated with candour'
      ]);
      return wrap(376, s);
    } },

  /* ------------------------------------ LinkedIn throttling, KDD 2014 */
  { id: 'linkedin-throttling-2014',
    paper: 'Budget Pacing for Targeted Online Advertisements at LinkedIn — Agarwal et al., KDD 2014',
    claim: 'Forecast eligible impressions, lay the budget over the forecast, and gate auction participation with a probability — so the budget throttles delivery while the auction itself never sees it.',
    svg() {
      let s = '';
      s += txt(60, 32, 'LINKEDIN 2014 · PROBABILISTIC THROTTLING — THE AUCTION NEVER SEES THE BUDGET', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      // the pipeline
      s += box(60, 70, 180, 62, 'FORECAST|eligible impressions|per campaign · per hour', { fill: SOFT });
      s += step(72, 62, 1);
      s += arrow(240, 101, 266, 101);
      s += box(270, 70, 190, 62, 'ALLOCATE|budget over the forecast|= target spend curve');
      s += step(282, 62, 2);
      s += arrow(460, 101, 486, 101);
      s += box(490, 70, 180, 62, 'PACING GATE|enter the auction with|probability p(t)', { fill: SOFT });
      s += step(502, 62, 3);
      s += aflow(670, 101, 696, 101);
      s += box(700, 70, 160, 62, 'AUCTION|bid × pCTR|bid untouched', { fill: MARK });

      // the control loop
      s += apulse(box(490, 190, 180, 56, 'CONTROLLER|actual vs target spend|→ adjust p(t)'));
      s += `<path d="M780,132 L780,218 L674,218" fill="none" stroke="${GREY}" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ahg)"/>`;
      s += txt(700, 240, 'spend feedback', { fs: 11, fill: GREY });
      s += arrow(560, 186, 560, 136, { color: BLUE });
      s += txt(574, 166, 'raise / lower p', { fs: 11.5, fill: BLUE });
      s += txt(490, 272, 'ahead of curve → lower p · behind → raise p', { fs: 11.5, fill: GREY });

      // the spend picture
      s += `<line x1="60" y1="330" x2="400" y2="330" stroke="${INK}" stroke-width="1"/>`;
      s += `<line x1="60" y1="330" x2="60" y2="190" stroke="${INK}" stroke-width="1"/>`;
      s += `<path d="M60,326 C110,290 130,215 165,205 L400,203" fill="none" stroke="${RED}" stroke-width="1.8"/>`;
      s += `<path d="M60,328 C150,322 230,296 310,252 C360,224 385,212 400,207" fill="none" stroke="${GREEN}" stroke-width="1.8"/>`;
      s += txt(60, 182, 'cumulative spend, one budget, one day', { fs: 11, fill: GREY });
      s += txt(172, 222, 'unpaced: gone by mid-morning', { fs: 11.5, fill: RED });
      s += txt(180, 300, 'paced: tracks forecast supply', { fs: 11.5, fill: GREEN });
      s += txt(60, 346, '00:00', { fs: 10.5, fill: GREY });
      s += txt(216, 346, '12:00', { a: 'middle', fs: 10.5, fill: GREY });
      s += txt(400, 346, '24:00', { a: 'end', fs: 10.5, fill: GREY });

      // headline
      s += hl(60, 362, 684, 20);
      s += txt(64, 377, 'RESULT: gate PARTICIPATION, never the bid — incentives stay intact, budget lasts all day', { fs: 12.5, w: 600 });
      s += note(60, 404, ['why: a budget folded into the bid distorts truthful bidding; a gate before the auction leaves the mechanism clean.']);
      return wrap(420, s);
    } },

  /* ----------------------------------------- Smart pacing, KDD 2015 */
  { id: 'smart-pacing-2015',
    paper: 'Smart Pacing for Effective Online Ad Campaign Optimization — Xu et al., KDD 2015',
    claim: 'Group requests by predicted response rate, give each group its own pacing rate, and adjust the rates by online feedback — turning even-delivery-versus-performance into an explicit dial.',
    svg() {
      let s = '';
      s += txt(60, 32, 'SMART PACING · GROUP BY PREDICTED RESPONSE, PACE EACH GROUP', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      s += box(60, 108, 150, 56, 'AD REQUESTS|one campaign,|one budget');
      s += step(72, 100, 1);
      s += arrow(214, 122, 266, 84, { color: INK });
      s += aflow(214, 136, 266, 136, { label: 'pCTR', ly: -9 });
      s += arrow(214, 150, 266, 188, { color: INK });

      s += box(270, 62, 190, 44, 'GROUP 1 · high pCTR|participate 90%', { fill: MARK });
      s += box(270, 114, 190, 44, 'GROUP 2 · mid pCTR|participate 45%');
      s += box(270, 166, 190, 44, 'GROUP 3 · low pCTR|participate 10%', { fill: SOFT });
      s += step(282, 54, 2);

      s += arrow(460, 84, 536, 122, { color: GREY, sw: 1.2 });
      s += arrow(460, 136, 536, 136, { color: GREY, sw: 1.2 });
      s += arrow(460, 188, 536, 150, { color: GREY, sw: 1.2 });
      s += box(540, 108, 150, 56, 'AUCTIONS|winners spend|toward the budget');
      s += arrow(690, 136, 726, 136);
      s += box(730, 108, 130, 56, 'FEEDBACK|actual spend|vs plan');
      s += step(742, 100, 3);
      s += `<path d="M795,104 C795,26 480,18 368,58" fill="none" stroke="${BLUE}" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ahb)"/>`;
      s += txt(552, 66, 'adjust group rates online', { fs: 11.5, fill: BLUE });
      s += txt(270, 232, 'behind plan → open lower groups · ahead → close them, highest-value first', { fs: 11.5, fill: GREY });

      // the dial
      s += `<line x1="100" y1="300" x2="720" y2="300" stroke="${INK}" stroke-width="1.5"/>`;
      s += `<line x1="100" y1="293" x2="100" y2="307" stroke="${INK}" stroke-width="1.5"/>`;
      s += `<line x1="720" y1="293" x2="720" y2="307" stroke="${INK}" stroke-width="1.5"/>`;
      s += `<circle cx="410" cy="300" r="7" fill="${BLUE}"/>`;
      s += txt(100, 326, 'EVEN DELIVERY', { fs: 11.5, w: 600 });
      s += txt(100, 341, 'uniform random throttle', { fs: 10.5, fill: GREY });
      s += txt(720, 326, 'PERFORMANCE', { a: 'end', fs: 11.5, w: 600 });
      s += txt(720, 341, 'all budget to high response', { a: 'end', fs: 10.5, fill: GREY });
      s += txt(410, 282, 'smart pacing: pick the point', { a: 'middle', fs: 11.5, fill: BLUE });

      // headline
      s += hl(60, 354, 732, 20);
      s += txt(64, 369, 'RESULT: even-delivery vs performance becomes an EXPLICIT dial — set it with the contract in hand', { fs: 12.5, w: 600 });
      s += note(60, 392, [
        'why it matters: skewed participation quietly converts even delivery into performance',
        'delivery — making the dial visible turns it into a decision made on purpose.'
      ]);
      return wrap(420, s);
    } },

  /* ------------------------------------- Bid shading, CIKM 2020 */
  { id: 'bid-shading-first-price-2020',
    paper: 'Bid Shading in The Brave New World of First-Price Auctions — Zhou et al., CIKM 2020',
    claim: 'After the 2019 first-price migration a truthful bid earns zero surplus, so DSPs learn P(win|bid) from auction feedback and bid the surplus-maximising shade below value.',
    svg() {
      let s = '';
      s += txt(60, 32, 'FIRST PRICE 2019 · YOU PAY YOUR OWN BID — SO LEARN HOW FAR TO SHADE', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      s += box(60, 66, 260, 66, 'SECOND PRICE · pre-2019|bid your value, pay the runner-up|no shading model needed', { fill: SOFT });
      s += step(72, 58, 1);
      s += aflow(320, 99, 386, 99, { label: 'exchange', ly: -12 });
      s += txt(353, 112, 'migration', { a: 'middle', fs: 11.5, fill: INK });
      s += box(390, 66, 230, 66, 'FIRST PRICE · 2019 on|you pay your own bid —|bidding value earns $0');
      s += step(402, 58, 2);

      // worked shading table, value $8.00
      s += txt(60, 168, 'value v = $8.00 · sweep the bid against a learned win curve:', { fs: 12, fill: GREY });
      s += txt(60, 194, 'bid $7.00 · P(win) .88 · surplus (8−7)×.88 = $0.88', { fs: 12 });
      s += txt(60, 214, 'bid $6.00 · P(win) .75 · surplus (8−6)×.75 = $1.50', { fs: 12 });
      s += hl(56, 220, 400, 20);
      s += txt(60, 235, 'bid $5.00 · P(win) .55 · surplus (8−5)×.55 = $1.65', { fs: 12, w: 600 });
      s += txt(466, 235, '← optimal: shade ≈38%', { fs: 11.5, fill: GREEN, w: 600 });
      s += txt(60, 255, 'bid $4.00 · P(win) .35 · surplus (8−4)×.35 = $1.40', { fs: 12, fill: GREY });

      // the learning loop
      s += box(660, 160, 200, 48, 'AUCTION FEEDBACK|win / loss per bid');
      s += arrow(760, 208, 760, 228);
      s += box(660, 232, 200, 48, 'WIN-RATE MODEL|P(win ∣ bid) per segment');
      s += arrow(760, 280, 760, 300);
      s += box(660, 304, 200, 48, 'SHADED BID|argmax (v−b)·P(win∣b)', { fill: MARK });
      s += step(672, 296, 3);
      s += `<path d="M862,326 C876,300 876,210 862,184" fill="none" stroke="${GREY}" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#ahg)"/>`;
      s += txt(760, 372, 'the loop runs continuously', { a: 'middle', fs: 10.5, fill: GREY });

      // headline
      s += hl(60, 292, 574, 20);
      s += txt(64, 307, 'RESULT: shading went from research topic to vendor feature in about a year', { fs: 12.5, w: 600 });
      s += note(60, 336, [
        'the migration moved the modelling burden onto every bidder in the market —',
        'a problem the second-price world simply never had.'
      ]);
      return wrap(380, s);
    } },

  /* --------------------------- Autobidding with constraints, WINE 2019 */
  { id: 'autobidding-uniform-2019',
    paper: 'Autobidding with Constraints — Aggarwal, Badanidiyuru & Mehta, WINE 2019 (+ Auto-bidding and Auctions survey, 2024)',
    claim: 'For a value maximiser with a budget in a truthful auction, one multiplier is enough: bidding v/(1+λ) uniformly on every query is optimal, with λ set by dual descent on spend.',
    svg() {
      let s = '';
      s += txt(60, 32, 'AUTOBIDDING · ONE MULTIPLIER λ TURNS A BUDGET INTO A UNIFORM SHADE', { fs: 12.5, w: 600, fill: BLUE });
      s += `<line x1="60" y1="42" x2="840" y2="42" stroke="${BLUE}" stroke-width="1"/>`;

      s += box(60, 70, 200, 66, 'VALUE MAXIMISER|max Σ xᵢ·vᵢ|s.t. Σ xᵢ·cᵢ ≤ B', { fill: SOFT });
      s += step(72, 62, 1);
      s += aflow(260, 103, 368, 103, { label: 'relax with λ', ly: -9 });
      s += box(372, 70, 180, 66, 'DUAL VARIABLE λ|the price of one more|dollar of budget');
      s += step(384, 62, 2);
      s += arrow(552, 103, 636, 103, { label: 'closed form', ly: -9 });
      s += apulse(box(640, 70, 220, 66, 'BID  vᵢ / (1+λ)|the SAME shade on|every single query', { fill: MARK }));
      s += step(652, 62, 3);

      // worked example
      s += txt(60, 176, 'worked shade · λ = 0.25, so divide every value by 1.25:', { fs: 12, fill: GREY });
      s += txt(60, 200, 'v = $5.00 → bid $4.00', { fs: 12 });
      s += txt(60, 220, 'v = $2.50 → bid $2.00', { fs: 12 });
      s += txt(60, 240, 'v = $0.75 → bid $0.60', { fs: 12 });
      s += txt(300, 220, '← one number carries the whole budget', { fs: 11.5, fill: GREY });

      // dual descent loop
      s += box(560, 186, 300, 50, 'DUAL DESCENT|λ ← λ + η·(spend_rate − B/T)');
      s += arrow(650, 182, 510, 142, { color: GREY, dash: '3 3', sw: 1.2 });
      s += txt(668, 262, 'spending too fast → λ rises → bids fall', { a: 'middle', fs: 11.5, fill: GREY });
      s += txt(668, 278, 'the idealised form of every PID pacer', { a: 'middle', fs: 11, fill: GREY });

      // headline
      s += hl(60, 296, 716, 20);
      s += txt(64, 311, 'RESULT: uniform bidding v/(1+λ) is OPTIMAL for budgeted value maximisers in truthful auctions', { fs: 12.5, w: 600 });
      s += note(60, 340, [
        'pacing controllers are this theorem in disguise: the bid multiplier IS the Lagrangian λ.',
        '2024 survey: when every bidder is an algorithm, truthfulness and welfare results change shape.'
      ]);
      return wrap(376, s);
    } }
]

};
