/* =========================================================================
   App: routing, reading level, speech capture, answer grading.
   Everything is local. The optional /api/grade endpoint upgrades grading
   from rubric-matching to a real model, if the server has a key.
   ========================================================================= */

const LESSON_ORDER = COURSE.parts.flatMap(p => p.lessons);
const KEY = 'hmd.v1';
const S = load();

function load() {
  try { return JSON.parse(localStorage.getItem(KEY)) || blank(); }
  catch { return blank(); }
}
function blank() { return { level: 'beginner', done: {}, answers: {}, mcq: {} }; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch {} }

/* ------------------------------------------------------------- helpers */
const el = (h) => { const d = document.createElement('div'); d.innerHTML = h.trim(); return d.firstElementChild; };
/* <x-ref to="sid">3.3</x-ref>  ->  a real link, with the lesson title as the hint.
   Written once here so a renamed lesson can never leave a dead reference behind. */
function xrefs(html) {
  return String(html).replace(/<x-ref\s+to="([a-z_]+)">(.*?)<\/x-ref>/g, (m, id, label) => {
    const L = COURSE.lessons[id];
    if (!L) return label;
    return `<a class="xref" href="#/${id}" title="${L.title}">${lessonNum(id)} ${L.title.toLowerCase()}</a>`;
  });
}

/* A lesson body may embed figures at the exact paragraph they illustrate:
   <x-fig name="embed_ex"></x-fig>. Text segments keep the reading measure;
   figures get the full figure width. The checker validates every name. */
function bodyWithFigs(html) {
  const parts = String(html).split(/<x-fig name="([a-z_0-9]+)"><\/x-fig>/);
  let out = '';
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) {
      if (parts[i].trim()) out += `<div class="body">${xrefs(parts[i])}</div>`;
    } else if (FIG[parts[i]]) {
      out += `<figure><div class="figbox">${safeFig(FIG[parts[i]])}</div></figure>`;
    }
  }
  return out;
}

/* A lesson's number is derived from the parts, never written by hand,
   so reordering the syllabus can never leave a stale "3.2" anywhere. */
function lessonNum(id) {
  const p = COURSE.parts.find(p => p.lessons.includes(id));
  return p ? p.n + '.' + (p.lessons.indexOf(id) + 1) : '';
}
function lessonLink(id) {
  const L = COURSE.lessons[id];
  return L ? `<a class="xref" href="#/${id}" title="${L.title}">${lessonNum(id)} ${L.title.toLowerCase()}</a>` : id;
}

const escape_ = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

/* Any figure that throws degrades to one visible gap, never a dead page. */
function safeFig(fn) {
  try { return fn(); }
  catch (e) { return '<div class="missing">This figure failed to draw.</div>'; }
}
const q = (s) => document.querySelector(s);

/* --------------------------------------------------------------- grade */
/* Rubric matching. Deliberately generous on synonyms, strict on structure:
   we score whether the *concepts* are present, never whether the wording
   matches. Every miss comes back with the reason it mattered.            */
function gradeLocal(text, rub) {
  const t = ' ' + text.toLowerCase().replace(/\s+/g, ' ') + ' ';
  const hit = (item) => item.any.some(p => new RegExp(p, 'i').test(t));
  const must = rub.must.map(m => ({ ...m, ok: hit(m) }));
  const bonus = (rub.bonus || []).map(m => ({ ...m, ok: hit(m) }));
  const traps = (rub.traps || []).filter(hit);
  const got = must.filter(m => m.ok).length;
  const score = must.length ? got / must.length : 0;
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return { must, bonus, traps, got, total: must.length, score, words, source: 'rubric' };
}

function verdictFor(g) {
  if (g.words < 12) return "That's short enough that I can't tell what you know. Try again with a couple more sentences — writing it out is most of the learning.";
  if (g.score === 1 && g.bonus.some(b => b.ok)) return 'Complete, and you went past the core. This is the level to aim for.';
  if (g.score === 1) return 'You covered every core idea. Have a look at the extras below for where to push next.';
  if (g.score >= 0.5) return 'Most of the way there. The missing piece below is the one that changes the answer.';
  if (g.score > 0) return "You've got a piece of it. Read the model answer, then try explaining it again from memory tomorrow.";
  return "Not there yet — and that's useful information. Read the model answer, then come back to the figure above.";
}

async function gradeRemote(text, lesson) {
  try {
    const r = await fetch('/api/grade', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: lesson.open.prompt,
        answer: text,
        must: lesson.open.must.map(m => m.name),
        traps: (lesson.open.traps || []).map(t => ({ name: t.name, why: t.why })),
        model: lesson.open.model
      })
    });
    if (!r.ok) return null;
    const j = await r.json();
    return j && j.ok ? j : null;
  } catch { return null; }
}

/* --------------------------------------------------------------- speech */
let rec = null, recTarget = null, recBase = '';
function speechAvailable() {
  return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}
function toggleRecord(btn, ta) {
  const R = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!R) return;
  if (rec) { rec.stop(); return; }
  rec = new R();
  rec.continuous = true; rec.interimResults = true; rec.lang = 'en-US';
  recTarget = ta; recBase = ta.value ? ta.value.trim() + ' ' : '';
  btn.classList.add('live'); btn.textContent = '■ stop recording';
  rec.onresult = (e) => {
    let out = '';
    for (let i = 0; i < e.results.length; i++) out += e.results[i][0].transcript;
    recTarget.value = recBase + out;
    recTarget.dispatchEvent(new Event('input'));
  };
  rec.onerror = () => stopRecord(btn);
  rec.onend = () => stopRecord(btn);
  rec.start();
}
function stopRecord(btn) {
  rec = null;
  btn.classList.remove('live');
  btn.textContent = '● speak your answer';
}

/* ----------------------------------------------------------------- toc */
function renderToc() {
  const body = q('#tocBody');
  body.innerHTML = COURSE.parts.map(p => `
    <div class="part">
      <div class="pname"><i>${p.n}</i> ${escape_(p.name)}</div>
      <p class="pblurb">${escape_(p.blurb)}</p>
      ${p.lessons.map((id, i) => {
        const L = COURSE.lessons[id];
        return `<a class="les" href="#/${id}" data-id="${id}">${escape_(L.title)}
          ${S.done[id] ? '<span class="tick">✓</span>' : ''}</a>`;
      }).join('')}
    </div>`).join('');
  markCurrent();
}
const DEPTH_TABS = [
  ['engineering', 'engineering', ['when', 'When to reach for it'], ['how', 'How to use it'],
   ['where', 'Where it fits'], ['breaks', 'How it breaks']],
  ['research', 'research', ['maths', 'The mathematics'], ['papers', 'What the papers claim'],
   ['scratch', 'Implementing it from scratch'], ['evals', 'How success is measured']],
  ['industry', 'in production', ['big', 'Big technology companies'], ['startups', 'What smaller teams do'],
   ['open_source', 'Open source you can read']],
  ['next', "what's next", ['problems', 'Open problems'], ['watch', 'What to follow']]
];

function renderDepth(id) {
  const D = DEPTH[id];
  if (!D) return `<div class="depth"><div class="label">Going deeper</div>
    <div class="missing">The deeper layers for this lesson — engineering, mathematics, who runs it in
    production, and what is still open — are not written yet. Lesson 3.3 has all four if you want to
    see the shape they take. <code>tools/check.py</code> lists every lesson still missing them, so
    this gap is tracked rather than forgotten.</div></div>`;

  /* Each drawn paper renders right after the written claims it illustrates,
     and a single malformed figure degrades to one visible gap, never a blank
     page. */
  const gallery = (PAPERFIGS[id] || []).map(p => {
    let svg;
    try { svg = p.svg().replace('role="img"', `role="img" aria-label="${escape_(p.claim)}"`); }
    catch (e) { return `<div class="missing">The figure for ${escape_(p.paper)} failed to draw.</div>`; }
    return `<div class="pfig">
      <div class="pfh">${escape_(p.paper)}</div>
      <div class="figbox">${svg}</div>
      <p class="pfclaim">${escape_(p.claim)}</p>
    </div>`;
  }).join('');

  const panes = DEPTH_TABS.filter(t => D[t[0]]).map(([key, label, ...fields], i) => {
    const body = fields.filter(([f]) => D[key][f]).map(([f, h]) =>
      `<h4>${h}</h4>${xrefs(D[key][f])}` +
      (key === 'research' && f === 'papers' && gallery ? `<h4>The papers, drawn</h4>${gallery}` : '')
    ).join('');
    return { key, label, i, body };
  });
  const code = D.code ? `
    <div class="codeblk">
      <div class="ch"><span class="t">${escape_(D.code.title)}</span><span class="l">${escape_(D.code.lang)}</span></div>
      <p class="note">${xrefs(D.code.note)}</p>
      <pre><code>${escape_(D.code.body)}</code></pre>
    </div>` : '';

  return `<div class="depth">
    <div class="label">Going deeper — four ways into the same idea</div>
    <div class="tabs">
      ${panes.map(p => `<button data-t="${p.key}" class="${p.i === 0 ? 'on' : ''}">${p.label}</button>`).join('')}
      ${code ? '<button data-t="code">code</button>' : ''}
    </div>
    ${panes.map(p => `<div class="pane ${p.i === 0 ? 'on' : ''}" data-t="${p.key}">${p.body}</div>`).join('')}
    ${code ? `<div class="pane" data-t="code">${code}</div>` : ''}
  </div>`;
}

/* Every lesson ends with the places it is actually used: the capstones that
   build on it, the projects that lean on it, the library entries that pair
   with it. All three are generated from data the checker validates, so a
   renamed lesson can never orphan a connection. */
function renderConnections(id) {
  const caps = Object.values(CAPSTONES).filter(c => (c.builds_on || []).includes(id));
  const projs = PROJECTS.filter(p => (p.uses || []).includes(id));
  const libs = LIBRARY.flatMap(g => g.x.filter(x => (x[4] || []).includes(id)));
  if (!caps.length && !projs.length && !libs.length) return '';
  return `<div class="connect">
    <div class="label">Where this lesson is used</div>
    ${caps.map(c => `<div class="clink"><span class="ck">capstone</span>
      <div><a href="#/capstone/${c.id}">${escape_(c.title)}</a>
      <p>${escape_(c.one)}</p></div></div>`).join('')}
    ${projs.map(p => `<div class="clink"><span class="ck">build</span>
      <div><a href="#/projects/${p.id}">${escape_(p.title)}</a>
      <p>${escape_(p.one)}</p></div></div>`).join('')}
    ${libs.map(([t, k, why, url]) => `<div class="clink"><span class="ck">${escape_(k)}</span>
      <div><a href="${url}" target="_blank" rel="noopener">${escape_(t)}</a>
      <p>${escape_(why)}</p></div></div>`).join('')}
  </div>`;
}

function wireTabs() {
  const tabs = document.querySelectorAll('.tabs button');
  tabs.forEach(b => b.onclick = () => {
    tabs.forEach(x => x.classList.toggle('on', x === b));
    document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.dataset.t === b.dataset.t));
  });
}

function renderCapstoneIndex() {
  q('#main').innerHTML = `
  <div class="hero">
    <div class="kick">Capstones</div>
    <h1>Specified, not suggested.</h1>
    <p>A capstone here comes with an architecture, named datasets and how to preprocess them, the
    machine it runs on, a repository layout, a week-by-week order, and the metrics that decide whether
    it worked. Anything less is a project idea, and project ideas are free.</p>
  </div>
  ${Object.values(CAPSTONES).map(c => `
    <div class="proj">
      <div class="ph"><h2><a href="#/capstone/${c.id}">${escape_(c.title)}</a></h2>
        <span class="diff">${escape_(c.weeks)}</span></div>
      <p class="one">${escape_(c.one)}</p>
      <p class="uses">builds on — ${c.builds_on.map(lessonLink).join('  ·  ')}</p>
      <p class="why">${escape_(c.why)}</p>
    </div>`).join('')}
  <div class="callout" style="margin-top:38px">
    <div class="label">The other four</div>
    <p>Made-to-measure sizing, the plant doctor, the Jyotisha system and the options research lab are
    outlined on the <a href="#/projects">build</a> page. They will be specified to this depth one at a
    time; a half-specified capstone is worse than an outlined one, because it looks finished.</p>
  </div>`;
}

function renderCapstone(id) {
  const c = CAPSTONES[id];
  if (!c) { location.hash = '#/capstones'; return; }
  q('#main').innerHTML = `
  <article class="cap">
    <div class="num">Capstone · ${escape_(c.weeks)}</div>
    <h1>${escape_(c.title)}</h1>
    <p class="hook">${escape_(c.one)}</p>
    <p class="uses">builds on — ${c.builds_on.map(lessonLink).join('  ·  ')}</p>

    <figure>
      <div class="figbox">${safeFig(FIG[c.fig])}</div>
      <figcaption><b>Architecture.</b> ${escape_(c.figCap)}</figcaption>
    </figure>

    <div class="callout"><div class="label">Why this one</div><p>${xrefs(c.why)}</p></div>

    <h2>Datasets</h2>
    <table class="dstable">
      <thead><tr><th>Dataset</th><th>Use</th><th>Why · how to prepare it</th></tr></thead>
      <tbody>${c.datasets.map(([n, tag, why, prep]) => `<tr>
        <td>${escape_(n)}</td><td><span class="tag">${escape_(tag)}</span></td>
        <td>${escape_(why)}<br><br><em>${escape_(prep)}</em></td></tr>`).join('')}</tbody>
    </table>

    <h2>On scraping</h2>
    <p>${xrefs(c.scraping)}</p>

    <h2>The machine</h2>
    ${xrefs(c.machine)}

    <h2>Repository layout</h2>
    ${c.repo}

    <h2>Week by week</h2>
    <div class="ladder">
      ${c.steps.map(([w, t, d]) => `<div class="rung"><span class="w">${escape_(w)}</span>
        <p><b>${escape_(t)}</b><br>${xrefs(d)}</p></div>`).join('')}
    </div>

    <h2>Offline metrics</h2>
    ${c.offline.map(([m, d]) => `<div class="metric"><span class="m">${escape_(m)}</span><p>${xrefs(d)}</p></div>`).join('')}

    <h2>Online metrics</h2>
    ${c.online.map(([m, d]) => `<div class="metric"><span class="m">${escape_(m)}</span><p>${xrefs(d)}</p></div>`).join('')}

    <h2>What you ship</h2>
    <p>${escape_(c.output)}</p>
  </article>`;
}

function markPage() {
  const h = location.hash.replace('#/', '');
  const p = h.startsWith('capstone') ? 'capstones'
          : h.startsWith('projects') ? 'projects'
          : h === 'library' ? 'library' : 'course';
  document.querySelectorAll('.pnav a').forEach(a => a.classList.toggle('on', a.dataset.p === p));
}

function renderProjects() {
  q('#main').innerHTML = `
  <div class="hero">
    <div class="kick">Five things worth building</div>
    <h1>Pick one, and finish step one this weekend.</h1>
    <p>None of these gets finished in a weekend. All of them have a first rung you can reach in one,
    and that is how every one of them actually gets built. Each project says which lessons it leans on,
    what makes it genuinely hard, and the specific mistake that sinks most attempts.</p>
  </div>
  ${PROJECTS.map((pr, i) => `
    <div class="proj" id="p-${pr.id}">
      <div class="ph"><h2>${escape_(pr.title)}</h2><span class="diff">${escape_(pr.hard)}</span></div>
      <p class="one">${escape_(pr.one)}</p>
      <p class="uses">leans on — ${pr.uses.map(lessonLink).join('  ·  ')}</p>
      <p class="why">${escape_(pr.why)}</p>
      <div class="ladder">
        ${pr.ladder.map(([w, t]) => `<div class="rung"><span class="w">${escape_(w)}</span><p>${escape_(t)}</p></div>`).join('')}
      </div>
      <div class="trap"><b>${pr.id === 'options' ? 'Read this before you start' : 'The mistake that sinks it'}</b>${escape_(pr.trap)}</div>
    </div>`).join('')}`;
}

function renderLibrary() {
  q('#main').innerHTML = `
  <div class="hero">
    <div class="kick">Sorted by job, not by fame</div>
    <h1>Where to read next.</h1>
    <p>A video made to give you a feeling and a video made to teach you to implement something are
    different tools, and confusing them is the main reason people give up on learning this from the
    internet. Every group below says what it is for. Use them in that spirit.</p>
  </div>
  ${LIBRARY.map(g => `
    <div class="libgrp">
      <h2>${escape_(g.t)}</h2>
      <p class="f">${escape_(g.f)}</p>
      ${g.x.map(([t, k, why, url, ls]) => `
        <div class="src">
          <span class="kind">${escape_(k)}</span>
          <div><a class="t" href="${url}" target="_blank" rel="noopener">${escape_(t)}</a>
          <p class="srcwhy">${escape_(why)}</p>
          ${ls && ls.length ? `<p class="pairs">pairs with — ${ls.map(lessonLink).join('  ·  ')}</p>` : ''}</div>
        </div>`).join('')}
    </div>`).join('')}`;
}

function markCurrent() {
  const cur = location.hash.replace('#/', '');
  document.querySelectorAll('.toc a.les').forEach(a => a.classList.toggle('cur', a.dataset.id === cur));
  const n = Object.keys(S.done).filter(k => S.done[k]).length;
  q('#prog').innerHTML = `<b>${n}</b> / ${LESSON_ORDER.length}`;
}

/* -------------------------------------------------------------- margin */
function renderMargin() {
  const body = q('#marginBody');
  const entries = LESSON_ORDER.filter(id => S.answers[id]);
  if (!entries.length) {
    body.innerHTML = `<p class="mempty">Answers you write in your own words collect here, next to the lesson they came from.
      Come back in a week and read them — you'll see exactly where your understanding was thin.</p>`;
    return;
  }
  body.innerHTML = entries.slice().reverse().map(id => {
    const L = COURSE.lessons[id];
    const a = S.answers[id];
    return `<div class="mnote"><span class="mh">${escape_(L.title)}</span>
      <p>${escape_(a.text.length > 220 ? a.text.slice(0, 220) + '…' : a.text)}</p></div>`;
  }).join('');
}

/* --------------------------------------------------------------- pages */
function renderIndex() {
  const n = Object.keys(S.done).filter(k => S.done[k]).length;
  q('#main').innerHTML = `
  <div class="hero">
    <div class="kick">A course in figures</div>
    <h1>How machines decide what to show you, what to say, and what to do next.</h1>
    <p>${LESSON_ORDER.length} lessons. Each one is a question, a figure built to be read on its own, the concept in plain
    language, the same concept for someone who has to ship it, and a question you answer in your own words —
    typed or spoken — that gets marked against a rubric.</p>
    <div class="meta">Running locally · progress saved on this machine · ${n} of ${LESSON_ORDER.length} complete</div>
  </div>

  <figure>
    <div class="figbox">${safeFig(FIG.course_map)}</div>
    <figcaption><b>The map.</b> Six parts, read left to right and top to bottom. The arrows say why each part exists: what it needs from the one before it, or what it makes possible next.</figcaption>
  </figure>

  <div class="callout">
    <div class="label">How to use this</div>
    <p><b>Beginner or expert</b> is a toggle in the top bar, and the two tracks have different jobs:
    beginner teaches each idea from nothing, and expert continues the same story into the decisions
    you would face building the real thing. Read beginner first, even if you're experienced — expert
    assumes its characters and numbers — then switch and read the sequel.</p>
    <p>The written question at the end of each lesson matters more than the multiple choice. Recalling
    something and putting it in your own words is what moves it into memory — recognising a right answer
    among four is not the same act.</p>
  </div>

  <div class="parts">
    ${COURSE.parts.map(p => `
      <div class="pcard">
        <div class="ph"><span class="pn">PART ${p.n}</span><h2>${escape_(p.name)}</h2></div>
        <p class="pb">${escape_(p.blurb)}</p>
        <ol>${p.lessons.map((id, i) => {
          const L = COURSE.lessons[id];
          return `<li><a href="#/${id}">
            <span class="ln">${p.n}.${i + 1}</span>
            <span><span class="lt">${escape_(L.title)}</span> — <span class="lh">${escape_(L.hook)}</span></span>
            ${S.done[id] ? '<span class="tick">✓</span>' : ''}</a></li>`;
        }).join('')}</ol>
      </div>`).join('')}
  </div>

  <div class="callout" style="margin-top:44px">
    <div class="label">On the figures</div>
    <p>Every figure is built to carry the concept without the prose around it. If you can't read one on its
    own, that's a defect in the figure, not in you. Where somebody has already made the definitive
    explanation of something — attention, linear algebra — the lesson links to them instead of competing.
    Those links say what to watch for.</p>
  </div>`;
}

function renderSettings() {
  q('#main').innerHTML = `
  <div class="hero">
    <div class="kick">Settings</div>
    <h1>Local options</h1>
    <p>Nothing here leaves your machine unless you deliberately turn on model grading, which routes through
    your own local server using your own key.</p>
  </div>
  <div class="settings">
    <div class="callout">
      <div class="label">Answer grading</div>
      <p><b>Default:</b> answers are marked locally against a per-question rubric. No network, no key,
      works offline. It checks whether each required idea is present and tells you which one is missing and
      why it matters.</p>
      <p><b>Optional upgrade:</b> start the server with an <code>ANTHROPIC_API_KEY</code> environment
      variable and answers are additionally marked by a model against the same rubric, which handles
      unusual phrasing better than pattern matching can. The rubric result is always shown too, so you can
      see where the two disagree.</p>
    </div>
    <div class="callout">
      <div class="label">Speaking your answers</div>
      <p>The microphone button uses your browser's built-in speech recognition. It works in Chrome and Edge;
      Safari and Firefox will show the button greyed out, and you can type instead. Audio is never recorded
      or stored — only the transcribed text, and only on this machine.</p>
      <p>Status here: <b>${speechAvailable() ? 'available in this browser' : 'not available in this browser — type instead'}</b></p>
    </div>
    <div class="callout">
      <div class="label">Reset</div>
      <p>Clearing removes your progress, your written answers and your notebook. There's no undo.</p>
      <p><button class="btn alt" id="resetBtn" style="margin-top:8px">clear everything</button></p>
    </div>
  </div>`;
  q('#resetBtn').onclick = () => {
    if (!confirm('Delete all progress and answers on this machine?')) return;
    Object.assign(S, blank()); save(); renderToc(); renderMargin(); route();
  };
}

function renderLesson(id) {
  const L = COURSE.lessons[id];
  if (!L) { location.hash = '#/'; return; }
  const idx = LESSON_ORDER.indexOf(id);
  const part = COURSE.parts.find(p => p.lessons.includes(id));
  const sub = part.lessons.indexOf(id) + 1;
  const prev = idx > 0 ? LESSON_ORDER[idx - 1] : null;
  const next = idx < LESSON_ORDER.length - 1 ? LESSON_ORDER[idx + 1] : null;
  const body = S.level === 'expert' ? L.expert : L.beginner;
  const saved = S.answers[id];

  q('#main').innerHTML = `
  <article class="lesson">
    <div class="num">Part ${part.n} · ${escape_(part.name)} · ${part.n}.${sub}</div>
    <h1>${escape_(L.title)}</h1>
    <p class="hook">${escape_(L.hook)}</p>

    <figure>
      <div class="figbox">${safeFig(FIG[L.fig])}</div>
      <figcaption><b>Figure ${part.n}.${sub}.</b> ${escape_(L.figCap)}</figcaption>
    </figure>

    <nav class="jump" aria-label="On this page">
      <button class="jl" data-s="bodyText">the idea</button>
      <button class="jl" data-s="sec-terms">terms</button>
      <button class="jl" data-s="sec-sources">sources</button>
      <button class="jl" data-s="sec-practice">practice</button>
      <button class="jl" data-s="sec-depth">go deeper</button>
    </nav>

    <div id="bodyText">${bodyWithFigs(body)}</div>

    <div class="terms" id="sec-terms">
      <div class="label">Terms as they're used in the literature</div>
      <dl>${L.terms.map(([t, d]) => `<dt>${escape_(t)}</dt><dd>${escape_(d)}</dd>`).join('')}</dl>
    </div>

    <div id="sec-sources">
    <div class="label">Sources — and what to look for in each</div>
    ${L.sources.map(([t, k, why, url]) => `
      <div class="src">
        <span class="kind">${escape_(k)}</span>
        <div><a class="t" href="${url}" target="_blank" rel="noopener">${escape_(t)}</a>
        <p class="srcwhy">${escape_(why)}</p></div>
      </div>`).join('')}
    </div>

    <div class="check" id="mcq">
      <span id="sec-practice"></span>
      <div class="label">Quick check</div>
      <p class="q">${escape_(L.mcq.q)}</p>
      <div id="opts">${L.mcq.o.map((o, i) => `<button class="opt" data-i="${i}">${escape_(o)}</button>`).join('')}</div>
      <div class="why" id="mcqWhy">${escape_(L.mcq.why)}</div>
    </div>

    <div class="write">
      <div class="label">Explain it yourself</div>
      <p class="q">${escape_(L.open.prompt)}</p>
      <p class="hint">Three or four sentences. Type it, or press the button and say it out loud — speaking
      an explanation catches gaps that writing lets you skate over.</p>
      <textarea id="ans" placeholder="In your own words…">${saved ? escape_(saved.text) : ''}</textarea>
      <div class="wrow">
        <button class="btn" id="mark">mark my answer</button>
        <button class="btn rec" id="recBtn" ${speechAvailable() ? '' : 'disabled title="Not supported in this browser"'}>● speak your answer</button>
        <button class="btn alt" id="clr">clear</button>
        <span class="wcount" id="wc">0 words</span>
      </div>
      <div class="grade" id="grade"></div>
    </div>

    <div id="sec-depth">${renderDepth(id)}</div>

    ${renderConnections(id)}

    <div class="pager">
      ${prev ? `<a href="#/${prev}">← previous<span>${escape_(COURSE.lessons[prev].title)}</span></a>` : '<span></span>'}
      ${next ? `<a class="next" href="#/${next}">next →<span>${escape_(COURSE.lessons[next].title)}</span></a>`
             : `<a class="next" href="#/">back to contents<span>You've reached the end</span></a>`}
    </div>
  </article>`;

  /* mcq */
  const opts = q('#opts');
  const settle = (chosen) => {
    [...opts.children].forEach((b, i) => {
      b.disabled = true;
      if (i === L.mcq.a) { b.classList.add('yes'); b.textContent = '✓  ' + b.textContent; }
      else if (i === chosen) { b.classList.add('no'); b.textContent = '✕  ' + b.textContent; }
    });
    q('#mcqWhy').classList.add('on');
  };
  if (S.mcq[id] !== undefined) settle(S.mcq[id]);
  [...opts.children].forEach(b => b.onclick = () => {
    const i = +b.dataset.i;
    S.mcq[id] = i; save(); settle(i);
  });

  /* written answer */
  const ta = q('#ans'), wc = q('#wc');
  const count = () => { wc.textContent = (ta.value.trim().split(/\s+/).filter(Boolean).length) + ' words'; };
  ta.addEventListener('input', count); count();
  q('#recBtn').onclick = (e) => toggleRecord(e.currentTarget, ta);
  q('#clr').onclick = () => { ta.value = ''; count(); q('#grade').classList.remove('on'); };
  q('#mark').onclick = () => markAnswer(id, L);
  if (saved) showGrade(id, L, gradeLocal(saved.text, L.open), saved.remote);

  document.querySelectorAll('.jump .jl').forEach(b => b.onclick = () => {
    const t = document.getElementById(b.dataset.s);
    const smooth = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (t) t.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
  });

  wireTabs();
  markCurrent();
  window.scrollTo(0, 0);
}

/* Figures and paper cards fade up as they enter the viewport. Purely
   presentational: everything is visible immediately without JS or with
   reduced motion. */
function reveal() {
  const all = document.querySelectorAll('.figbox, .pfig');
  if (!('IntersectionObserver' in window)) {
    all.forEach(n => n.classList.add('vis'));
    return;
  }
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add('vis'); io.unobserve(e.target); }
  }), { threshold: 0.12 });
  all.forEach(n => io.observe(n));
}

async function markAnswer(id, L) {
  const ta = q('#ans');
  const text = ta.value.trim();
  if (!text) { ta.focus(); return; }
  const g = gradeLocal(text, L.open);
  const btn = q('#mark');
  btn.textContent = 'marking…'; btn.disabled = true;
  const remote = await gradeRemote(text, L);
  btn.textContent = 'mark my answer'; btn.disabled = false;

  S.answers[id] = { text, at: Date.now(), remote };
  if (g.score >= 0.5 && g.words >= 12) S.done[id] = true;
  save(); renderToc(); renderMargin();
  showGrade(id, L, g, remote);
}

function showGrade(id, L, g, remote) {
  const box = q('#grade');
  const pct = Math.round(g.score * 100);
  const hits = g.must.filter(m => m.ok).length;

  const idx = LESSON_ORDER.indexOf(id);
  const nid = g.score >= 0.5 && idx > -1 && idx < LESSON_ORDER.length - 1 ? LESSON_ORDER[idx + 1] : null;
  box.innerHTML = `
    <div class="scoreline">
      <span class="s">${hits}/${g.total}</span>
      <span class="verdict">${escape_(verdictFor(g))}</span>
      ${nid ? `<a class="next" href="#/${nid}">next → ${escape_(COURSE.lessons[nid].title)}</a>` : ''}
    </div>
    <div class="meter">
      ${g.must.map(m => `<i class="${m.ok ? 'hit' : 'miss'}" style="width:${100 / g.total}%"></i>`).join('')}
    </div>

    <div class="label">Core ideas</div>
    ${g.must.map(m => `
      <div class="gitem ${m.ok ? 'hit' : 'miss'}">
        <span class="mk">${m.ok ? '✓' : '✕'}</span>
        <span><span class="n">${escape_(m.name)}</span>${m.ok ? '' : ' <span class="ex">— not found in your answer</span>'}</span>
      </div>`).join('')}

    ${g.bonus.length ? `<div class="label" style="margin-top:16px">Going further</div>
      ${g.bonus.map(b => `<div class="gitem ${b.ok ? 'bonus' : 'miss'}">
        <span class="mk">${b.ok ? '+' : '·'}</span>
        <span><span class="n">${escape_(b.name)}</span></span></div>`).join('')}` : ''}

    ${g.traps.map(t => `<div class="trap"><b>Worth reconsidering</b>${escape_(t.why)}</div>`).join('')}

    ${remote ? `<div class="modelans"><div class="label">Marked by the model</div>
      <p>${escape_(remote.feedback)}</p></div>` : ''}

    <div class="modelans">
      <details class="reveal" ${g.score === 1 ? 'open' : ''}>
        <summary>A strong answer</summary>
        <p style="margin-top:12px">${escape_(L.open.model)}</p>
      </details>
    </div>`;
  box.classList.add('on');
}

/* --------------------------------------------------------------- route */
function route() {
  const h = location.hash.replace('#/', '');
  if (!h) renderIndex();
  else if (h === 'settings') renderSettings();
  else if (h.startsWith('projects')) {
    renderProjects();
    const pid = h.split('/')[1];
    const card = pid && document.getElementById('p-' + pid);
    if (card) { card.classList.add('flash'); card.scrollIntoView({ block: 'start' }); }
  }
  else if (h === 'library') renderLibrary();
  else if (h === 'capstones') renderCapstoneIndex();
  else if (h.startsWith('capstone/')) renderCapstone(h.split('/')[1]);
  else renderLesson(h);
  reveal();
  markCurrent(); markPage();
  q('#toc').classList.remove('open');
}

/* ----------------------------------------------------------------- init */
function setLevel(lv) {
  S.level = lv; save();
  q('#lvBeg').setAttribute('aria-pressed', lv === 'beginner');
  q('#lvExp').setAttribute('aria-pressed', lv === 'expert');
  const cur = location.hash.replace('#/', '');
  const bt = q('#bodyText');
  if (bt && cur && COURSE.lessons[cur]) {
    const L = COURSE.lessons[cur];
    const y = window.scrollY;
    bt.innerHTML = bodyWithFigs(lv === 'expert' ? L.expert : L.beginner);
    reveal();
    /* The two tracks differ in length, so a saved mid-page offset would land
       in unrelated prose — return to the top of the body instead. */
    if (y > bt.offsetTop) bt.scrollIntoView();
    else window.scrollTo(0, y);
  }
}
/* The sticky bar wraps to more rows on narrow screens; everything offset
   against it reads the measured height, never a constant. */
function setBarH() {
  document.documentElement.style.setProperty('--barh', q('.bar').offsetHeight + 'px');
}
addEventListener('resize', setBarH);
setBarH();
if (document.fonts && document.fonts.ready) document.fonts.ready.then(setBarH);

q('#lvBeg').onclick = () => setLevel('beginner');
q('#lvExp').onclick = () => setLevel('expert');
q('#setBtn').onclick = () => location.hash = '#/settings';
q('#menuBtn').onclick = () => q('#toc').classList.toggle('open');
addEventListener('hashchange', route);

setLevel(S.level || 'beginner');
renderToc(); renderMargin(); route();
