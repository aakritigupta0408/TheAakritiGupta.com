/* =========================================================================
   DEPTH LAYERS
   Every lesson can carry four extra layers beneath the two prose levels:

     engineering  — when to reach for it, how to use it, where it breaks
     research     — the mathematics, what the papers actually claim
     industry     — who runs this in production, and what they reported
     next         — open problems, and what to read to work on them

   Plus code: a runnable snippet or notebook cell per lesson.

   REFERENCE IMPLEMENTATION: `sid` is filled in completely. Every other
   lesson gets the same schema. tools/check.py reports which are missing
   so the gap is visible rather than forgotten.
   ========================================================================= */

const DEPTH = {

funnel: {
  engineering: {
    when: `Reach for a cascade when <b>scoring the whole catalogue with your best model breaks the
    latency budget</b> — and not before. A funnel is a tax you pay for scale, not an architectural
    virtue. At tens of thousands of items, run the good model on everything and skip the whole
    apparatus; every stage you add is a training pipeline, a serving fleet and a new place for the
    system to silently disagree with itself.`,
    how: `Standard shape, four decisions:
    <ul>
      <li><b>Retrieval is a union, not a model.</b> Production candidate sets come from many sources
      merged: a dual-encoder against an ANN index (<x-ref to="embed">2.1</x-ref>,
      <x-ref to="ann">2.2</x-ref>), co-visitation counts, graph walks, fresh-item pools, popularity.
      Each source exists to cover a failure mode of the others.</li>
      <li><b>Coarse ranking imitates fine ranking.</b> The pre-ranker's job is not to predict clicks;
      it is to predict <em>what the heavy ranker would say</em>, cheaply. Distil from the ranker's
      scores rather than training both independently on labels — independently trained stages drift
      apart and the pre-ranker starts pruning exactly the items the ranker wanted.</li>
      <li><b>Do the budget arithmetic before choosing models.</b> Per stage,
      <code>n_items × cost_per_item</code> must fit; the model class falls out of that, not the
      other way round.</li>
      <li><b>Log candidate coverage from the first request:</b> for every engagement you later
      observe, was the item in the candidate set? It is the single most useful diagnostic in the
      whole system and almost nobody logs it.</li>
    </ul>`,
    where: `Every feed, search box, ads stack and marketplace — and, less obviously, most LLM
    systems: retrieve-then-rerank in <x-ref to="rag">4.1</x-ref> is this exact funnel with documents
    instead of videos, a bi-encoder as stage one and a cross-encoder as stage two. Learn the failure
    modes once here and you have learned them everywhere.`,
    breaks: `<ul>
      <li><b>The recall ceiling.</b> A ranker can only reorder what retrieval hands it, so ranking
      wins that don't move online metrics are usually coverage problems. Check coverage before
      touching the model.</li>
      <li><b>Stage disagreement.</b> The pre-ranker and ranker were trained on different data with
      different features; where they disagree, good items die at the seam. Measure agreement on the
      overlap (rank correlation of pre-rank vs rank scores on surviving candidates) — a drop is an
      earlier warning than any online metric.</li>
      <li><b>The feedback loop.</b> Training data is engagement on items the funnel already chose to
      show. Items the funnel never surfaces never get labels, so every stage re-learns the previous
      deployment's blind spots. Reserve a small exploration slice or the ceiling quietly becomes
      permanent.</li>
      <li><b>Candidate-source rot.</b> Sources get added for launches and never deleted. Attribute
      final-slate items back to the source that produced them, per request; sources contributing
      nothing are pure latency.</li>
    </ul>`
  },
  research: {
    evals: `The number that decides whether a funnel change ships is <b>end-to-end candidate
recall</b>: of the items users later engaged with, what fraction survived every stage into the
slate. Everything else is a decomposition of that.
<ul>
  <li><b>Candidate recall@k</b> — probability the eventually-engaged item was in the retrieved
  set; its trap is that it is computed on engagement the current funnel generated, so the metric
  cannot see the items the incumbent never showed.</li>
  <li><b>Per-stage survival rate</b> — where in the cascade good items die; averaged over all
  traffic it hides a query class losing everything at one seam, so slice before you trust it.</li>
  <li><b>Pre-rank/rank agreement</b> — rank correlation between cheap and expensive scores on
  shared candidates; a suspiciously high value can mean the pre-ranker inherited the ranker's
  biases along with its wisdom.</li>
  <li><b>Slate quality vs a full-scoring oracle</b> — how close the funnel gets to running the
  heavy ranker on everything; the trap is that the oracle requires an expensive shadow job, so
  most teams substitute proxies and never learn their true ceiling.</li>
</ul>
<pre>retrieval tower:  L = −log [ exp s(u,i⁺) / (exp s(u,i⁺) + Σ_j exp(s(u,i_j) − log q(i_j))) ]
pre-ranker:       L = Σ_i (f_cheap(i) − f_heavy(i))²        # imitate the ranker's scores
final ranker:     L = −[y·log p̂ + (1−y)·log(1−p̂)]           # calibrated pointwise CE on labels</pre>
Offline recall gains routinely fail to move online metrics because the new candidates displace
old ones the ranker already knew how to score — the seam, not the stage, absorbs the win.`,

    maths: `An item reaches the slate only by surviving every stage, so the funnel's recall is a
    product of per-stage survival probabilities:
    <pre>P(i on slate) = P(retrieved) · P(survives coarse | retrieved) · P(survives fine | coarse)

recall(funnel) = Π_s recall_s   ≤   min_s recall_s</pre>
    One leaky stage caps the whole system, and no downstream stage can restore mass that an earlier
    stage discarded — this is the recall ceiling, and it is an inequality about probability flow,
    not about model quality. The cost side is a budget constraint:
    <pre>C = Σ_s n_s · c_s   ≤   latency budget
n_1 ≫ n_2 ≫ n_3     while     c_1 ≪ c_2 ≪ c_3</pre>
    Geometric decay in <code>n</code> against geometric growth in <code>c</code> keeps every term
    <code>n_s·c_s</code> comparable — a well-tuned funnel spends similar budget at each stage.
    Why not just raise <code>k</code>? Model retrieval as a noisy channel: cheap score =
    true utility + noise. Recall@k is the probability a true-top item's noisy rank lands inside k,
    which rises steeply for small k and flattens as you push into items the cheap model already
    confidently rejected — while cost in <code>k</code> rises linearly. Marginal recall per unit
    cost falls monotonically; the knee of that curve is where production systems sit, and the code
    below makes it visible.`,
    papers: `<ul>
      <li><b>A Cascade Ranking Model for Efficient Ranked Retrieval</b> (Wang, Lin & Metzler,
      SIGIR 2011) is the formal ancestor: successively richer features over successively smaller
      candidate sets, trained jointly against an objective that balances effectiveness with feature
      cost — and it showed the cascade can improve <em>both</em> at once, not merely trade them.</li>
      <li><b>Deep Neural Networks for YouTube Recommendations</b> (Covington et al., RecSys 2016)
      made the two-stage split the industry default: a candidate-generation network whose ANN lookup
      returns hundreds of candidates from a corpus of millions, then a separate ranking network.
      Section 3's framing — retrieval optimises recall, ranking optimises precision — is the whole
      argument in two sentences.</li>
      <li><b>COLD</b> (Wang et al., 2020, Alibaba) treats pre-ranking as algorithm–system co-design:
      the paper describes the pre-ranking stage receiving hundreds of thousands of candidates from
      matching and emitting the top thousands, and optimises the model jointly with the computing
      power it costs rather than fixing an architecture first.</li>
      <li><b>OneRec</b> (Kuaishou, 2025) is the counter-thesis: unify retrieval and ranking in one
      generative model so the stages can no longer disagree, at the price of giving up the cost
      structure that justified the cascade.</li>
    </ul>`,
    scratch: `Build the simulation before reading anything else — it is twenty minutes of numpy and
    it permanently changes how you read ranking papers:
    <ol>
      <li>Draw true utilities for a large corpus. Cheap retriever = truth + large noise; expensive
      ranker = truth + small noise.</li>
      <li>Sweep candidate count k. For each k, record candidate recall of the true top items and
      final slate quality relative to scoring everything with the ranker.</li>
      <li>Watch slate quality track retrieval recall almost exactly, no matter how good the ranker
      is. Then make the ranker <em>perfect</em> (zero noise) and watch nothing change.</li>
      <li>Extension: distil — refit the cheap scorer to imitate the ranker on logged candidates and
      measure how much recall the same k now buys. That is pre-ranking research in one plot.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Meta / Instagram Explore</b> is the best-documented funnel. The 2023 engineering blog
      describes retrieval narrowing billions of items to thousands, a Two Towers first-stage ranker
      over those thousands, a Multi-Task Multi-Label network over the 100 best, then a final re-rank
      applying integrity filters and diversity rules. An earlier Instagram engineering post reported
      the Explore ranking funnel extracting 65 billion features and making 90 million model
      predictions every second — the scale that makes the cost arithmetic non-negotiable.</li>
      <li><b>Alibaba</b> runs matching → pre-ranking → ranking; the COLD paper notes the pre-ranking
      candidate set is tens to hundreds of times larger than the ranking stage's, which is why
      pre-ranking gets its own research literature rather than being a smaller ranker.</li>
      <li><b>Pinterest</b>'s Pixie (Eksombatchai et al., WWW 2018) does retrieval by real-time
      random walks over a graph of 3 billion nodes and 17 billion edges; the paper reports that over
      half of pins saved daily came from Pixie-backed systems, with up to 50% higher engagement than
      the prior Hadoop-based batch pipeline. Retrieval need not be a neural model at all.</li>
      <li><b>Kuaishou</b> is the live experiment in deleting the funnel: OneRec replaces the
      retrieval / coarse / fine cascade with one generative model decoding semantic IDs
      (<x-ref to="sid">3.3</x-ref>), reported at roughly 25% of queries on the main app and Lite
      with a +1.68% total watch-time gain in the technical report. Whether that wins because
      end-to-end optimisation removes seam losses, or simply because it concentrates compute in one
      model, is genuinely unresolved.</li>
    </ul>`,
    startups: `Do not build four stages. Two is enough for years: a managed vector index for
    retrieval and one ranker you actually own. The funnel's stage count at big platforms is an
    artefact of their n, not a template — your defensible asset is the ranking features and the
    logging (coverage, source attribution) that the platforms had to learn to build, not the number
    of stages. Add a pre-ranker only when profiling shows the ranker is the bottleneck at a
    candidate count you have verified you need.`,
    open_source: `<b>FAISS</b>, <b>ScaNN</b> and <b>hnswlib</b> cover stage one
    (<x-ref to="ann">2.2</x-ref>). <b>Vespa</b> is the one engine where the cascade is a
    first-class citizen — first-phase and second-phase rank expressions run inside the index, so a
    cheap function scores everything and an ONNX model rescores the survivors without a second
    service. <b>TorchRec</b> (Meta) and <b>NVIDIA Merlin</b> cover the ranking-model side;
    Elasticsearch/OpenSearch rescoring windows give you a poor man's two-stage funnel on
    infrastructure you may already run.`
  },
  next: {
    problems: `<ul>
      <li><b>Joint training across stages.</b> Retrieval is still mostly trained as if it faced the
      user directly, when its real job is feeding the ranker. Distillation-based pre-ranking
      consistency is the active thread; a principled end-to-end objective for a non-differentiable
      pruning pipeline is not solved.</li>
      <li><b>The generative challenge.</b> If OneRec-style single models keep winning at Kuaishou
      scale, the funnel stops being the default and becomes a serving optimisation. If they don't
      travel beyond short-video feeds, the cascade's cost argument stands. Either outcome
      restructures the field.</li>
      <li><b>Exploration through the funnel.</b> Every stage filters the training data of every
      other stage; quantifying and correcting that closed loop (beyond a fixed random slice) is
      wide open.</li>
      <li><b>Evaluation.</b> Per-stage offline metrics are computed on logs the funnel itself
      produced. Counterfactual full-funnel evaluation — what would coverage have been under a
      different retriever — remains mostly untractable and mostly unattempted.</li>
    </ul>`,
    watch: `Watch three things: pre-ranking papers out of Alibaba, Meituan and Kuaishou (search
    arXiv cs.IR for "pre-ranking" and "cascade ranking"); the OneRec line and its imitators for
    whether cascade-collapse generalises; and RAG rerankers quietly reinventing funnel lessons —
    candidate coverage arguments are appearing in retrieval-augmentation papers under new names.`
  },
  code: {
    title: 'The recall ceiling, made visible',
    note: `Runnable as-is on CPU in a few seconds. A cheap noisy retriever picks k candidates; a
    near-perfect ranker orders them. The printout shows slate quality pinned to retrieval recall —
    the ranker's excellence never escapes the ceiling — while cost climbs linearly in k.`,
    lang: 'python',
    body: `import numpy as np

rng = np.random.default_rng(0)
N, SLATE, TRIALS = 100_000, 10, 30      # catalogue, slate size, repeats

def run_funnel(k, cheap_noise=2.0, fine_noise=0.1):
    shown_q, oracle_q, recall = [], [], []
    for _ in range(TRIALS):
        true = rng.standard_normal(N)                 # true utility (hidden)
        cheap = true + cheap_noise * rng.standard_normal(N)   # stage 1 scores
        cand = np.argpartition(-cheap, k)[:k]                 # retrieve top-k
        fine = true[cand] + fine_noise * rng.standard_normal(k)  # stage 2
        slate = cand[np.argsort(-fine)[:SLATE]]               # final 10
        best = np.argsort(-true)[:SLATE]              # what we SHOULD show
        shown_q.append(true[slate].mean())
        oracle_q.append(true[best].mean())
        recall.append(np.isin(best, cand).mean())     # true top-10 in candidates?
    return np.mean(recall), np.mean(shown_q) / np.mean(oracle_q)

print("ranker noise is 0.1 -- nearly perfect. Watch it not matter.")
print(f"{'k':>7}  {'retrieval recall':>16}  {'slate quality':>13}  {'cost units':>10}")
for k in [30, 100, 300, 1000, 3000, 10000]:
    r, q = run_funnel(k)
    cost = N * 1 + k * 100        # cheap model 1 unit/item, ranker 100 units/item
    print(f"{k:>7}  {r:>15.0%}  {q:>12.0%}  {cost:>10,}")

r_perfect, q_perfect = run_funnel(300, fine_noise=0.0)
print()
print(f"k=300 with a PERFECT ranker: quality {q_perfect:.0%} (was 85% with noise)")
print("perfecting the ranker bought ~1 point; raising k from 300 to 3000 bought 10.")
print("the ranker can only reorder what retrieval hands it -- quality moves with")
print("recall, not ranker skill, while cost grows linearly in k. That is the funnel.")`
  }
},

features: {
  engineering: {
    when: `Reach for explicit feature machinery when your signal lives in <b>high-cardinality
    categoricals and their combinations</b> — user × item × context, millions of distinct values,
    tabular rather than sequential structure. If your problem is a few hundred dense columns,
    gradient-boosted trees will embarrass a DLRM and cost a tenth as much to run. The embedding-table
    apparatus earns its keep only once one-hot encoding stops fitting in memory.`,
    how: `The decisions that actually matter, in the order you hit them:
    <ul>
      <li><b>Vocabulary or hash?</b> A managed vocabulary gives every ID its own row but needs an
      assignment service and grows without bound. Hashing is stateless and fixed-size but collides —
      run the collision arithmetic (see the maths tab) before picking a table size, not after the
      offline metrics look odd.</li>
      <li><b>Size tables by frequency, not uniformly.</b> A tail item seen 12 times cannot train a
      64-dim vector; give heads wide embeddings and the tail narrow ones or a shared bucket.</li>
      <li><b>Pick the interaction module last.</b> Dot-product interaction (DLRM), cross layers
      (DCN-v2), or an FM term (DeepFM) — the deltas between them are real but small next to getting
      features and freshness right. DCN-v2's low-rank cross layers are the sensible default; that is
      roughly what Google reported deploying across its ranking systems (Wang et al., WWW 2021).</li>
      <li><b>Log features at serving time and train on the log.</b> Recomputing features offline is
      how training–serving skew gets in. This one habit prevents a whole class of silent bugs.</li>
    </ul>`,
    where: `The ranking stage of the <x-ref to="funnel">funnel</x-ref> is where this machinery
    concentrates — retrieval towers (<x-ref to="embed">2.1</x-ref>) take a leaner feature set because
    the item tower must precompute. Counters and crosses also feed calibration and bidding models,
    which is why feature pipelines outlive the models on top of them.`,
    breaks: `<ul>
      <li><b>Hash collisions on the head.</b> A collision between two tail items costs nothing you
      can measure; a collision between a blockbuster and a tail item drags both. Reserve exact slots
      for the top of the distribution and hash only the tail.</li>
      <li><b>Staleness.</b> Behavioural counters (clicks-last-hour, CTR-so-far) decay in value within
      hours. ByteDance's Monolith paper (Liu et al., 2022) exists because batch-nightly training
      measurably loses to online training with minute-level parameter sync on exactly these features.</li>
      <li><b>Unbounded table growth.</b> New users and items arrive forever. Without expiry and
      frequency filtering — both of which Monolith builds in — the table eats the cluster.</li>
      <li><b>Leakage through counters.</b> A feature like "clicks on this item today" computed at
      end-of-day includes the click you are trying to predict. Point-in-time correctness is tedious
      and non-negotiable.</li>
    </ul>`
  },
  research: {
    evals: `The shipping decision for a feature or interaction change rests on <b>normalised
logloss</b> — cross-entropy relative to a constant base-rate predictor — because ranking and
bidding both consume the probabilities, not the ordering.
<ul>
  <li><b>Normalised cross-entropy</b> — how much better than predicting the base rate; its trap
  is head dominance: the tail segments where new features should help barely move the average.</li>
  <li><b>AUC</b> — pure ordering quality; blind to calibration, so a model can gain AUC while its
  probabilities drift far enough to break every downstream consumer of them.</li>
  <li><b>Segment calibration ratio</b> — predicted over observed positives per slice; the trap is
  cancellation, where over-prediction in one segment offsets under-prediction in another and the
  aggregate looks perfect.</li>
  <li><b>Staleness ablation</b> — re-score with counter features artificially delayed by hours;
  skipping this is how offline replay credits freshness your serving path cannot deliver.</li>
</ul>
<pre>L = −(1/N) Σ_i [ y_i log σ(f(x_i)) + (1−y_i) log(1−σ(f(x_i))) ]   # binary CE on logged impressions
NE = L / L_base,   L_base = CE of always predicting the empirical positive rate</pre>
The offline→online gap here is mechanical: training data contains only impressions the previous
model chose to make, so a logloss win on that log says little about behaviour on the traffic the
new model will newly surface.`,

    maths: `<b>Why crosses must be modelled explicitly.</b> A linear model over binary features
    x₁, x₂ scores w₁x₁ + w₂x₂ + b. Suppose the label is the XOR-style bargain pattern — positive
    only when cheap AND five-stars:
    <pre>(0,0)→0:  b ≤ 0
(1,0)→0:  w₁ + b ≤ 0
(0,1)→0:  w₂ + b ≤ 0
(1,1)→1:  w₁ + w₂ + b > 0   ← contradicts the sum of the middle two</pre>
    No weights satisfy all four. Add one product feature x₁x₂ and it is trivially separable — which
    is the entire justification for FM terms, cross layers and the wide branch of Wide &amp; Deep.
    An MLP can approximate the product, but it spends capacity learning multiplication; the explicit
    cross gets it for free.
    <br><br>
    <b>Hashing and the birthday bound.</b> Hash N IDs uniformly into M buckets. The probability a
    given ID shares its bucket with at least one other:
    <pre>P(collide) = 1 − (1 − 1/M)^(N−1) ≈ 1 − e^(−N/M)</pre>
    At M = N that is 1 − 1/e ≈ 63% of IDs sharing a vector — hashing into a table "as big as the
    vocabulary" is nowhere near collision-free. Expected colliding pairs are N(N−1)/2M, the birthday
    bound, which is why collisions appear long before the table looks full.
    <br><br>
    <b>The DLRM interaction.</b> With F embedded features e₁..e_F (shared dimension d), DLRM
    (Naumov et al., 2019) computes all pairwise dot products — F(F−1)/2 scalars — concatenates them
    with the dense-MLP output, and feeds a top MLP. Second-order interaction is hard-wired; the MLP
    only has to weigh it. DCN's cross layer instead computes
    <pre>x_{l+1} = x₀ ⊙ (W·x_l + b) + x_l</pre>
    so each layer raises the polynomial degree of interactions by one — l layers give degree-(l+1)
    crosses with parameters linear in the input width.`,
    papers: `<ul>
      <li><b>Wide &amp; Deep</b> (Cheng et al., 2016): a memorising linear branch over hand-crossed
      features joint-trained with a generalising MLP; reported significantly increased app
      acquisitions on Google Play. The paper that made "memorisation vs generalisation" the
      standard framing.</li>
      <li><b>DCN</b> (Wang et al., 2017) and <b>DCN-v2</b> (Wang et al., WWW 2021): learn the crosses
      instead of hand-writing them; v2 is the production report — low-rank mixture cross layers
      deployed in Google's web-scale learning-to-rank systems.</li>
      <li><b>DLRM</b> (Naumov et al., 2019): less an architecture paper than a systems statement —
      the reference design Meta open-sourced so hardware and systems people had a real workload.</li>
      <li><b>Feature hashing</b> (Weinberger et al., ICML 2009): the hashing trick with its analysis;
      the reason collisions hurt less than intuition says on sparse data, and the trick Vowpal Wabbit
      built its whole design around.</li>
      <li><b>Monolith</b> (Liu et al., 2022): ByteDance's case that collisionless tables plus online
      training beat bigger models — the feature-freshness paper.</li>
      <li><b>HSTU / "Actions Speak Louder than Words"</b> (Zhai et al., ICML 2024): the argument that
      sequentialising raw actions can replace much of this chapter — see the next tab.</li>
    </ul>`,
    scratch: `Build the collision story before believing anyone's table-size folklore:
    <ol>
      <li>Run the code tab: collision rate and retained signal vs table size, empirically.</li>
      <li>Fit a logistic regression on (cheap, five-stars) without the product feature, then with it.
      Watch accuracy jump from coin-flip to perfect. That is Wide &amp; Deep's wide branch in
      miniature.</li>
      <li>Implement one DCN cross layer — it is three lines — and check it recovers x₁x₂ exactly.</li>
      <li>Then read DLRM's Figure 1. After steps 1–3 the architecture reads as an inventory of
      decisions you have already made once.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Meta</b> published the clearest numbers on scale: embedding tables at terabyte scale,
      and the ZionEX training platform (Mudigere et al., ISCA 2022) demonstrated DLRMs up to
      <b>12 trillion parameters</b> with a reported 40× time-to-solution speedup — nearly all of
      those parameters sitting in embedding tables, which is the "big in storage, small in
      reasoning" shape made concrete. TorchRec is the open-sourced descendant of that stack.</li>
      <li><b>Google</b> reported DCN-v2 deployed across many web-scale ranking systems with
      significant offline and online gains (Wang et al., WWW 2021) — notable for being a practical-
      lessons paper: low-rank crosses because serving budgets, not benchmarks, set the design.</li>
      <li><b>ByteDance</b> runs Monolith under its recommender products: collisionless cuckoo-hashed
      tables, expirable embeddings, and minute-level online sync of sparse parameters.</li>
      <li><b>Meta again, against itself:</b> HSTU generative recommenders (Zhai et al., ICML 2024) —
      1.5-trillion-parameter models reporting <b>+12.4%</b> in online A/B tests on surfaces with
      billions of users — were motivated precisely by the claim that DLRM-style feature interaction
      had stopped scaling with compute (<x-ref to="seqrec">3.2</x-ref>).</li>
    </ul>`,
    startups: `Do not build sharded embedding serving. Below tens of millions of IDs a single-node
    table (or hashing into one) is fine, and below that GBDTs on engineered features remain the
    best accuracy-per-engineer in tabular ranking. The high-leverage thing to copy from big-tech
    practice is not the architecture — it is <b>logging features at serving time</b> and keeping
    counter features fresh. Freshness is the cheapest win in this chapter and the one small teams
    most often skip.`,
    open_source: `<b>TorchRec</b> (github.com/meta-pytorch/torchrec) is the serious option: sharded
    embedding tables (row-, column-, table-wise), FBGEMM kernels, pipelined training; Meta's current
    DLRM builds on it and it powers Meta production models. The original <b>facebookresearch/dlrm</b>
    reference code is the readable one, and DLRM became MLPerf's recommendation benchmark — a decent
    signal of how central this workload is to hardware vendors. <b>Vowpal Wabbit</b> is still the
    cleanest demonstration that hashing plus a linear model gets you shockingly far.`
  },
  next: {
    problems: `<ul>
      <li><b>Does feature engineering survive the generative turn?</b> HSTU's core argument is that
      most engineered features are lossy summaries of the raw action sequence, so a model over the
      sequence itself (<x-ref to="seqrec">3.2</x-ref>) subsumes the crosses. LinkedIn's LiGR paper
      claimed deprecation of most manual feature engineering, using a handful of features against
      hundreds in the baseline — and the authors later withdrew the arXiv submission citing
      discrepancies in the claims. Both halves of that story are the state of the field: the
      direction is probably right, and the reported magnitudes deserve scepticism.</li>
      <li><b>Scaling laws for sparse models.</b> Whether recommendation quality follows compute
      scaling once interactions are stacked properly (Meta's Wukong line) is genuinely open; the
      DLRM plateau might have been an architecture problem, not a paradigm one.</li>
      <li><b>Collisions are not solved.</b> Work on multi-probe zero-collision hashing (2026)
      continues because collisionless tables cost memory and hashed ones cost accuracy and
      freshness; the trade has no clean winner yet. <x-ref to="sid">Semantic IDs</x-ref> attack the
      same wound from the content side: give related items related codes so sharing a row is a
      feature, not a bug.</li>
      <li><b>Tail features under privacy constraints.</b> Per-user counters are exactly the features
      regulation squeezes first; nobody has a good public answer.</li>
    </ul>`,
    watch: `Watch whether Meta and ByteDance keep publishing DLRM-era systems papers or go quiet in
    favour of generative-recommender ones — the publication mix is the honest signal of where the
    parameter budget went. On arXiv cs.IR, "CTR prediction" survey papers now read as history;
    follow "feature interaction" plus "scaling law" instead, and read any withdrawal notice you
    find twice.`
  },
  code: {
    title: 'What hash collisions actually cost',
    note: `Runnable as-is on CPU, numpy only. Hashes N sparse IDs into tables of various sizes M.
    A hashed embedding can at best learn one vector per bucket, so the bucket-mean predictor is the
    ceiling — the "kept signal" column is the R-squared of that ceiling. Note the collision column
    matches 1 − e^(−N/M): the birthday bound from the maths tab, live.`,
    lang: 'python',
    body: `import numpy as np

rng = np.random.default_rng(0)
N = 100_000                          # distinct sparse IDs in training data
ctr = rng.beta(2, 20, size=N)        # each ID's true click-through rate

print(f"vocabulary N = {N:,} IDs, signal variance = {ctr.var():.5f}")
print(f"{'table M':>12} {'IDs collided':>13} {'birthday pred':>14} {'kept signal':>12}")

for M in [10_000, 50_000, 100_000, 1_000_000, 10_000_000]:
    bucket = rng.integers(0, M, size=N)          # stands in for hash(id) % M
    counts = np.bincount(bucket, minlength=M)

    # fraction of IDs sharing their bucket with at least one other ID
    collided = (counts[bucket] > 1).mean()
    theory = 1 - (1 - 1/M) ** (N - 1)            # birthday bound

    # best possible hashed model: one embedding per bucket.
    # its optimal prediction for every ID in a bucket is the bucket mean,
    # so within-bucket variance is signal irrecoverably destroyed by hashing.
    sums = np.bincount(bucket, weights=ctr, minlength=M)
    pred = sums[bucket] / counts[bucket]
    kept = 1 - np.mean((ctr - pred) ** 2) / ctr.var()   # R^2 of the ceiling

    print(f"{M:>12,} {collided:>12.1%} {theory:>13.1%} {kept:>11.1%}")

# What the table teaches:
#  - M == N is NOT collision-free: ~63% of IDs share a vector (1 - 1/e).
#  - kept signal ~63% at M == N: over a third of the per-ID signal is
#    gone before training even starts, and no optimiser can get it back.
#  - you need M ~ 10-100x N for hashing to be nearly free -- or exact
#    slots for head IDs and hashing for the tail, which is what ships.`
  }
},

embed: {
  engineering: {
    when: `Reach for learned embeddings when <b>vocabulary mismatch is your bottleneck</b> — users say
    "trainers", your catalogue says "sneakers", and no amount of synonym lists keeps up. If your queries
    mostly contain part numbers, error codes and exact names, BM25 is already winning and a dense encoder
    will quietly lose those queries — which is why <x-ref to="hybrid">4.2</x-ref> exists. The two-tower
    split specifically is for when the corpus is too large to score item-by-item at query time: it is a
    precomputation strategy wearing an architecture costume.`,
    how: `In order of increasing commitment:
    <ul>
      <li><b>Off-the-shelf encoder, no training.</b> A modern open encoder plus an ANN index is a working
      system in a day. Measure recall on your own labelled queries before believing any leaderboard.</li>
      <li><b>Fine-tune contrastively on your pairs.</b> In-batch negatives with the logQ popularity
      correction (Yi et al., RecSys 2019), then hard negatives mined from your own index. The negatives
      pipeline is where the recall lives; the encoder architecture barely matters by comparison.</li>
      <li><b>Engineer the vector itself.</b> Matryoshka-trained models truncate gracefully — serve 256
      dims, keep 1024 for rescoring. Binary quantization cuts memory 32× and retains ≈96% of retrieval
      quality with a float-query rescoring pass (Hugging Face / mixedbread reports, 2024). Vector cost
      is a product decision, not a footnote.</li>
    </ul>`,
    where: `The first stage of every funnel in <x-ref to="funnel">1.1</x-ref>: search retrieval, recommender
    candidate generation, RAG. Also everywhere similarity is the product itself — dedup, clustering,
    "more like this", routing tickets to teams. The vectors only become useful at scale through the index
    in <x-ref to="ann">2.2</x-ref>; embedding and indexing are one system and should be versioned as one.`,
    breaks: `<ul>
      <li><b>Leaderboard shopping.</b> MTEB now lists 400+ models separated by decimals, and training-data
      overlap with the benchmark is endemic. A model that drops double-digit nDCG from MTEB to your private
      domain was memorising the eval, not modelling meaning. Always hold out your own queries.</li>
      <li><b>Encoder drift.</b> Vectors from encoder v1 and v2 live in different spaces even when trained
      identically. Version encoder and index together; treat an encoder swap as a full re-embed.</li>
      <li><b>False negatives in mining.</b> The hardest negatives your index can find are often unlabelled
      positives; train on them raw and you teach the model to reject correct answers. Denoise with a score
      threshold or a cross-encoder pass first.</li>
      <li><b>Metric mismatch.</b> Train with cosine, serve with dot product (or vice versa) and ranking
      quietly changes, because unnormalized norms encode popularity. Pick one, enforce it in code.</li>
    </ul>`
  },
  research: {
    evals: `One metric decides an encoder swap: <b>recall@k on a held-out set of your own labelled
queries</b>, scored against your own corpus — never a public leaderboard position.
<ul>
  <li><b>Recall@k</b> — gold document present in the top k; on shared benchmarks its trap is
  train–test osmosis, where the score certifies familiarity with the benchmark rather than with
  meaning.</li>
  <li><b>MRR and nDCG@10</b> — position-weighted variants; they reward nailing easy head queries
  and can rise while paraphrase-heavy tail queries quietly collapse, so slice by query type.</li>
  <li><b>Alignment and uniformity</b> — geometric diagnostics of the embedding space; the trap is
  treating pretty geometry as evidence of retrieval quality, which it does not guarantee.</li>
  <li><b>Recall under truncation and quantisation</b> — the metric re-run at the dimensions and
  precision you will actually serve; skipping it means shipping a vector you never evaluated.</li>
</ul>
<pre>InfoNCE (temperature τ, in-batch + mined negatives):
  L = −log [ exp(sim(q, d⁺)/τ) / Σ_{d∈batch∪hard} exp(sim(q, d)/τ) ]
triplet alternative:  L = max(0, m − sim(q,d⁺) + sim(q,d⁻))   # margin m, when pairs are scarce</pre>
Offline recall on frozen queries fails to predict live quality because users reformulate: a miss
in production becomes a second query the eval never sees, and the encoder's real job is making
that second query unnecessary.`,

    maths: `<b>Skip-gram with negative sampling</b> (Mikolov et al., 2013) trains word vector u and
    context vector v to classify real co-occurrences against k sampled fakes:
    <pre>L = Σ_(w,c) [ log σ(u_w·v_c) + k · E_(n~P_n) log σ(−u_w·v_n) ]
P_n ∝ unigram(n)^0.75,   k ≈ 5–20 for small corpora (Mikolov et al., 2013)</pre>
    Levy & Goldberg (NeurIPS 2014) showed what this actually optimises: at the optimum,
    <pre>u_w · v_c = PMI(w, c) − log k</pre>
    — SGNS is implicit factorization of the shifted pointwise-mutual-information matrix. "Meaning as
    coordinates" is not a metaphor; it is a low-rank log co-occurrence model, and the famous analogies
    are a side effect of that log-linear structure.
    <br><br>
    <b>InfoNCE</b> (Oord et al., 2018) is the same trick generalised to modern encoders:
    <pre>L = −E [ log  exp(s(x, y⁺)/τ) / Σ_(j=1..N) exp(s(x, y_j)/τ) ]
I(x; y) ≥ log N − L</pre>
    The loss is a lower bound on mutual information that saturates at log N — a batch of 4096 can certify
    at most 12 bits — which is the mathematical reason larger batches and harder negatives keep paying off
    long after the architecture stops mattering.
    <br><br>
    <b>Cosine vs dot.</b> a·b = ‖a‖‖b‖cos θ, so unnormalized dot product gives every item a learnable
    magnitude — a popularity prior smuggled into geometry, sometimes wanted (recommenders), usually not
    (semantic search). Normalise and scores live in [−1, 1], which makes the temperature mandatory: at
    τ = 1 the softmax over near-unit scores is almost uniform and gradients vanish, so contrastive
    recipes run τ around 0.01–0.1.`,
    papers: `<ul>
      <li><b>word2vec</b> (Mikolov et al., 2013) — the efficiency argument: replace a full softmax over
      the vocabulary with k noise samples, and suddenly you can train on billions of tokens on a CPU.</li>
      <li><b>Neural Word Embedding as Implicit Matrix Factorization</b> (Levy & Goldberg, NeurIPS 2014)
      — the paper that told you what word2vec was doing. Short and worth reading in full.</li>
      <li><b>Sentence-BERT</b> (Reimers & Gurevych, EMNLP 2019) — cross-encoding all pairs of 10,000
      sentences: ≈65 hours. A siamese encoder with cosine similarity: ≈5 seconds (their abstract). That
      four-orders-of-magnitude gap is the entire case for the tower split.</li>
      <li><b>Dense Passage Retrieval</b> (Karpukhin et al., 2020) and <b>Sampling-Bias-Corrected Neural
      Modeling</b> (Yi et al., RecSys 2019) — the same dual encoder discovered independently in QA and in
      YouTube-scale recommendation; the logQ correction comes from the latter.</li>
      <li><b>E5</b> (Wang et al., 2022) and <b>GTE</b> (Li et al., 2023) — weakly supervised contrastive
      pretraining on web-mined pairs, then supervised fine-tuning: the now-standard recipe.</li>
      <li><b>Matryoshka Representation Learning</b> (Kusupati et al., NeurIPS 2022) — nest losses at
      prefix lengths so one vector serves every budget; up to 14× smaller embeddings at matched
      ImageNet-1k accuracy in the original paper.</li>
    </ul>`,
    scratch: `Build in this order, one afternoon each:
    <ol>
      <li>SGNS on a toy corpus (the code tab). Watch topic structure appear in nearest neighbours with no
      labels anywhere — the moment the whole field clicked, miniaturised.</li>
      <li>An InfoNCE dual encoder with in-batch negatives. Log recall@10 as you grow the batch: the
      log N bound shows up in your own curves.</li>
      <li>Hard-negative mining against your own index. Recall jumps, then degrades when you mine too
      hard — you have just rediscovered the false-negative problem empirically.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Google/YouTube</b> set the production template: the sampling-bias-corrected two-tower
      (Yi et al., RecSys 2019) for corpus-scale candidate retrieval, item tower precomputed into an ANN
      index, user tower run per request. Nearly every large feed and search stack since is a variation.</li>
      <li><b>MongoDB</b> acquired <b>Voyage AI</b> (Tengyu Ma's embedding and reranking company) in
      February 2025 — approximately $220M by press reports — to fold embedding and reranking natively
      into Atlas. Read the signal: databases are absorbing the embedding layer, betting retrieval
      quality is a storage-adjacent feature rather than a standalone API business. Anthropic, notably,
      ships no first-party embedding model and has pointed developers at providers like Voyage instead.</li>
    </ul>`,
    startups: `The embedding-API market is a squeeze play: open-weight models close the quality gap from
    below while databases integrate from above — the Voyage acquisition is the proof point. What defends
    a position is domain depth (legal, code, finance rerankers) and the evaluation-plus-fine-tuning loop
    on the customer's own data, not another general-purpose model two MTEB points up. The base encoder is
    becoming a commodity you inherit, not an asset you own.`,
    open_source: `<b>sentence-transformers</b> is the default training and inference library, and its
    embedding-quantization docs carry the 32× / ≈96%-retained binary recipe. <b>Nomic Embed</b> (Nussbaum
    et al., 2024) is the reference open release: 137M parameters, 8192-token context, Apache-2 weights
    <em>plus</em> training data and code — reproducibility no proprietary encoder offers. <b>E5</b> and
    <b>GTE</b> are the workhorse baselines; <b>MTEB</b> (Muennighoff et al., 2022) is the shared harness —
    run its toolkit on your own data even after you stop trusting its leaderboard.`
  },
  next: {
    problems: `<ul>
      <li><b>Evaluation is the crisis.</b> Contaminated leaderboards mean public numbers no longer predict
      private-domain behaviour. MMTEB-style refreshes are patches; benchmark design that resists
      training-set osmosis is unsolved.</li>
      <li><b>One vector is not enough.</b> A single point per document must serve every query intent at
      once. Late-interaction (multi-vector) models trade the cheap dot product for per-token matching;
      where that frontier settles decides how <x-ref to="chunk">4.3</x-ref>-style pipelines get built.</li>
      <li><b>Instruction-following encoders.</b> "Embed this for recency" versus "embed this for
      topicality" from one model — early results are promising, evaluation is thin.</li>
      <li><b>The cost frontier.</b> Matryoshka truncation, int8/binary quantization and dimension pruning
      are becoming one joint optimisation problem; there is no clean theory yet for what precision buys
      at which dimension.</li>
    </ul>`,
    watch: `Watch the MTEB leaderboard for what people optimise, not for what is true, and watch the
    database vendors' bundling moves — they show where the margin in this layer is going. On arXiv,
    "text embeddings", "late interaction" and "embedding quantization" in cs.IR/cs.CL cover the field
    at a paper or two a month.`
  },
  code: {
    title: 'Skip-gram with negative sampling, from scratch',
    note: `Runs on CPU in a few seconds, numpy only. The corpus is synthetic — sentences that stay on one
    topic — so co-occurrence IS topic membership. Nobody labels the topics; the geometry recovers them.
    That emergence is the word2vec result, miniaturised.`,
    lang: 'python',
    body: `import numpy as np

rng = np.random.default_rng(0)
topics = {
    "animals": ["cat", "dog", "horse", "wolf", "otter", "fox"],
    "food":    ["bread", "cheese", "apple", "rice", "soup", "cake"],
    "tools":   ["hammer", "wrench", "saw", "drill", "chisel", "pliers"],
}
vocab = [w for ws in topics.values() for w in ws]
idx = {w: i for i, w in enumerate(vocab)}
V, D, K, lr = len(vocab), 16, 5, 0.02

# each sentence sticks to one topic -> co-occurrence carries the meaning
names = list(topics)
corpus = [list(rng.choice(topics[names[rng.integers(3)]], size=8))
          for _ in range(800)]

W = rng.normal(0, 0.1, (V, D))   # word vectors     u_w
C = rng.normal(0, 0.1, (V, D))   # context vectors  v_c

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def neighbours(w, k=3):
    E = W / np.linalg.norm(W, axis=1, keepdims=True)   # cosine = normalised dot
    sims = E @ E[idx[w]]
    return [vocab[i] for i in np.argsort(-sims)[1:k + 1]]

print("before:", {w: neighbours(w) for w in ["cat", "bread", "saw"]})

for epoch in range(4):
    for sent in corpus:
        ids = [idx[w] for w in sent]
        for i, w in enumerate(ids):
            for c in ids[max(0, i - 2):i] + ids[i + 1:i + 3]:
                g = sigmoid(W[w] @ C[c]) - 1.0            # positive: pull in
                W[w] -= lr * g * C[c]; C[c] -= lr * g * W[w]
                for n in rng.integers(0, V, K):           # negatives: push out
                    g = sigmoid(W[w] @ C[n])
                    W[w] -= lr * g * C[n]; C[n] -= lr * g * W[w]

print("after:")
for w in ["cat", "bread", "saw"]:
    print(f"  {w:>6} -> {neighbours(w)}")
# Before training the neighbour lists are random. After, each word's
# neighbours are its topic-mates -- structure no one labelled, recovered
# purely from what co-occurs with what. Scale this up and you have word2vec.`
  }
},

ann: {
  engineering: {
    when: `Reach for ANN when <b>brute force stops fitting your latency budget</b>, and not before.
    Under a million vectors, exact search with a matrix multiply is a few milliseconds on CPU and it
    is <em>correct</em> — no recall measurement, no index build, no tuning. The crossover is later
    than people think. Above that, the decision is really a memory decision: do the arithmetic on
    corpus size × dimension × 4 bytes first, because it picks your index family for you.`,
    how: `Three families, chosen by where the vectors live:
    <ul>
      <li><b>HNSW</b> when everything fits in RAM. Best recall/QPS tradeoff on CPU, the default in
      nearly every vector database. Tune <code>M</code> once (16–32 is almost always right), then
      treat <code>efSearch</code> as your runtime dial — measured against brute-force ground truth
      on a held-out query sample, not guessed.</li>
      <li><b>IVF-PQ</b> when RAM is the constraint. Coarse k-means partitioning plus product
      quantization gets 64× compression before you feel it in recall; <code>nprobe</code> becomes
      the dial. This is what FAISS was built around.</li>
      <li><b>DiskANN-style</b> when the corpus is billions and you want one machine: graph on SSD,
      compressed vectors in RAM to steer traversal, full vectors on disk to re-rank.</li>
    </ul>
    If you are already on Postgres, pgvector's HNSW (added in 0.5.0, 2023) is genuinely competitive
    and saves you an entire system — start there and leave only when you must.`,
    where: `The retrieval stage of every <x-ref to="funnel">funnel</x-ref>: candidate generation over
    item embeddings, and the R in <x-ref to="rag">RAG</x-ref>. Anywhere <x-ref to="embed">embeddings</x-ref>
    exist, an ANN index is the thing that makes them queryable — the embedding model decides what
    "near" means, the index decides whether you can afford to ask.`,
    breaks: `<ul>
      <li><b>Filtered search.</b> The big one. Post-filtering a selective predicate returns almost
      nothing; pre-filtering degenerates to a scan. Below roughly 1% selectivity the
      predicate-satisfying subgraph disconnects and graph traversal collapses — the finding that
      motivated Filtered-DiskANN and ACORN. Ask your engine how it filters <em>during</em> traversal
      before you ask about QPS.</li>
      <li><b>Deletes and updates.</b> HNSW has no clean delete; tombstones accumulate and recall
      quietly degrades until you rebuild. Plan the rebuild cadence on day one.</li>
      <li><b>Embedding model retrains.</b> New encoder means every stored vector is stale — a full
      re-embed and re-index, the exact drift problem <x-ref to="sid">semantic IDs</x-ref> hit with
      their encoder. Version the model with the index.</li>
      <li><b>Recall measured on the wrong distribution.</b> Benchmark recall on SIFT says little
      about your production queries, especially out-of-distribution ones (text queries against an
      image corpus). Measure on your own traffic.</li>
    </ul>`
  },
  research: {
    evals: `An index earns deployment at one point: <b>recall@k at your latency budget</b>, measured
against exact brute-force ground truth on your own query traffic.
<ul>
  <li><b>Recall@k vs exact scan</b> — the honest baseline nobody can argue with; its trap is
  query provenance, since recall measured on corpus-drawn queries flatters an index that will
  serve out-of-distribution traffic.</li>
  <li><b>QPS at fixed recall</b> — the frontier's other axis; vendor figures are steady-state,
  single-tenant and filter-free, none of which describes production.</li>
  <li><b>p99 latency under concurrent writes</b> — the tail is what pages you, and inserts plus
  tombstones degrade it in ways a read-only benchmark never shows.</li>
  <li><b>Bytes per vector, fully loaded</b> — codes plus graph edges plus any full-precision
  copies kept for re-ranking; quoting only the compressed codes is the standard fudge.</li>
</ul>
<pre>nothing trains — the objective is a constrained search over index parameters:
  maximise  recall@k(index; your queries)
  s.t.      p99 latency ≤ budget,   memory ≤ fleet RAM / shards
  dials:    efSearch / nprobe move you along the frontier; M / codebooks move the frontier itself</pre>
The frozen-snapshot number decays silently: recall measured at build time fails to predict the
index six months later, after deletes, distribution drift and an encoder retrain have each taken
their unmeasured bite.`,

    maths: `<b>The memory arithmetic that motivates everything.</b> A 128-dim float32 vector is
    512 bytes; one billion of them is 512 GB before any index structure. Product quantization
    (Jégou, Douze &amp; Schmid, TPAMI 2011) splits the vector into m subvectors and quantizes each
    against its own codebook of K=256 centroids, so the code is m bytes:
    <pre>x ∈ R^128  →  (c₁, …, c₈),  cⱼ = argmin_k ‖ xⱼ − Cⱼ[k] ‖²      # 8 bytes, 64× smaller
effective codebook size: K^m = 256⁸ ≈ 1.8×10¹⁹ centroids
stored: m·K·(D/m) floats — the cross product is free</pre>
    Distance is estimated without decompressing, via lookup tables built once per query:
    <pre>d(q, x)² ≈ Σⱼ ‖ qⱼ − Cⱼ[cⱼ] ‖²     # asymmetric distance computation
cost per query: m·K subvector distances up front, then m table lookups per point</pre>
    <b>Why HNSW is roughly logarithmic.</b> Each point's top layer is drawn from a geometric
    distribution, so the layer count grows as O(log N); upper layers are sparse long-range edges,
    and greedy descent does a bounded number of hops per layer. Malkov &amp; Yashunin claim — and
    show empirically — O(log N) search scaling; it is a strong empirical regularity, not a
    worst-case theorem for arbitrary data.
    <br><br>
    <b>The curse of dimensionality, stated honestly.</b> For truly high-dimensional data with no
    structure, distances concentrate — the nearest and farthest neighbour become nearly
    indistinguishable — and every exact space-partitioning index (kd-trees and kin) degrades to a
    linear scan. ANN works in practice because real embeddings are not arbitrary: they concentrate
    near much lower-dimensional manifolds, and graphs and quantizers exploit exactly that structure.
    The guarantee you give up is the price of the structure you exploit.`,
    papers: `<ul>
      <li><b>HNSW</b> (Malkov &amp; Yashunin, 2016, <a href="https://arxiv.org/abs/1603.09320">arXiv 1603.09320</a>)
      — navigable small-world graphs made hierarchical. The layered-graph figure is the algorithm.</li>
      <li><b>Product quantization</b> (Jégou et al., TPAMI 2011) — 128-dim SIFT vectors to 8–16
      bytes with usable recall; IVFADC, the IVF-PQ pattern everyone still ships.</li>
      <li><b>Billion-scale similarity search with GPUs</b> (Johnson, Douze &amp; Jégou, 2017) and
      <b>The Faiss library</b> (Douze et al., 2024, <a href="https://arxiv.org/abs/2401.08281">arXiv 2401.08281</a>)
      — the engineering canon for quantization-based indexes.</li>
      <li><b>DiskANN</b> (Subramanya et al., NeurIPS 2019) — one billion points on a 64 GB
      workstation with an NVMe SSD: 5000+ QPS at under 3 ms mean latency with 95%+ 1-recall@1,
      where equal-memory IVF baselines plateaued near 50%.</li>
      <li><b>ScaNN</b> (Guo et al., ICML 2020) — anisotropic quantization: penalize error parallel
      to the vector direction more, because that is what corrupts inner products. Took the top of
      ann-benchmarks glove-1.2M on publication.</li>
      <li><b>RaBitQ</b> (Gao &amp; Long, SIGMOD 2024) — randomly rotate, then binarize, with a sharp
      theoretical error bound; the paper reports it beats PQ variants on the accuracy–efficiency
      tradeoff, and it is driving the current quantization-first turn.</li>
    </ul>`,
    scratch: `Build IVF before reading another paper — it is k-means plus bookkeeping, an afternoon
    of work (the code tab below is a complete one). Order: brute-force ground truth first, since
    without it recall is unmeasurable; then the inverted lists; then sweep nprobe and plot recall
    against vectors scanned. That curve is the entire field on one axis. Then add PQ inside each
    list and watch recall dip while memory falls 64×. Compare against hnswlib on the same data with
    ann-benchmarks' harness — matching the methodology of the standard benchmark keeps you honest.`
  },
  industry: {
    big: `<ul>
      <li><b>Meta</b> built FAISS and runs it at the far end of the scale axis: the Faiss
      engineering documentation describes a 1.5-trillion-vector index, sharded by ID and by
      inverted list across machines. At that size the index <em>is</em> a distributed system and
      quantization is not optional.</li>
      <li><b>Microsoft</b> built the DiskANN line and reports using it across its services; it
      ships as the vector index in Azure database products. The 2019 result — a billion points,
      64 GB RAM, one commodity SSD — is still the reference point for cost-per-vector arguments.</li>
      <li><b>Google</b> published ScaNN and runs its anisotropic quantization under Vertex AI
      Vector Search; the same lineage serves embedding retrieval inside Google's own products.</li>
      <li><b>NVIDIA</b>'s CAGRA (2023) rebuilt the graph index for GPUs: the paper reports 33–77×
      higher throughput than HNSW on CPU at 90–95% recall in large-batch search, and 2.2–27× faster
      graph construction. It is now inside FAISS via cuVS, and Milvus, OpenSearch and Elasticsearch
      integrations followed.</li>
    </ul>`,
    startups: `The vector-database gold rush — Pinecone, Weaviate, Qdrant, Zilliz/Milvus, Chroma —
    ran on the premise that the index is the product. It is not: everyone converged on HNSW plus
    quantization, and then the incumbents (pgvector in Postgres, OpenSearch, Elasticsearch, Redis,
    MongoDB) added vector columns to databases you already run. What actually differentiates
    engines now is the unglamorous part: filtered search done during traversal (Weaviate shipped an
    ACORN-style strategy for exactly this), freshness under heavy updates, and operational cost per
    billion vectors. If you are evaluating vendors, benchmark <em>your</em> filters at <em>your</em>
    selectivity — the headline QPS numbers are all measured without them.`,
    open_source: `<b>FAISS</b> is the reference toolkit and the best-documented tradeoff catalogue
    in the field; read its wiki like a textbook. <b>hnswlib</b> (Malkov's own implementation) is the
    small, readable one. <b>pgvector</b> made ANN a Postgres index type. <b>DiskANN</b> and
    <b>cuVS/CAGRA</b> are open source. And <b>ann-benchmarks</b> (Aumüller, Bernhardsson &amp;
    Faithfull) is the standard recall-vs-QPS comparison — every serious index publishes its curve
    there, and the Pareto frontier, not any single number, is the thing to read.`
  },
  next: {
    problems: `<ul>
      <li><b>Filtered ANN.</b> Still the gap between benchmarks and production. ACORN
      (SIGMOD 2024) is predicate-agnostic where Filtered-DiskANN needs predicates known at build
      time, but low-selectivity behaviour — the sub-1% regime where graphs fragment — remains an
      active research front with new entries every quarter.</li>
      <li><b>Streaming indexes.</b> Inserts, deletes and recall stability without periodic full
      rebuilds (the FreshDiskANN direction). Most deployed systems still just rebuild.</li>
      <li><b>Quantization-first design.</b> RaBitQ-style binary codes with error bounds, then
      graphs built <em>over</em> the compressed representation rather than compression bolted onto
      a graph. Watch this replace the HNSW-plus-afterthought-quantization default.</li>
      <li><b>Out-of-distribution queries.</b> Indexes optimised on the corpus distribution degrade
      when queries come from elsewhere — cross-modal retrieval being the common case. Big-ANN's OOD
      track exists precisely because this was being ignored.</li>
    </ul>`,
    watch: `The big-ann-benchmarks competition tracks (started at NeurIPS 2021: filtered, streaming,
    sparse, OOD) are the field's agenda made explicit — each track is a named open problem. The
    venue shifted too: the interesting ANN papers now land at SIGMOD and VLDB as often as NeurIPS,
    because the open problems are systems problems. Skim ann-benchmarks' updated curves twice a
    year; the Pareto frontier moves slowly enough that this is sufficient.`
  },
  code: {
    title: 'An IVF index in 40 lines',
    note: `Runnable as-is on CPU, numpy only. K-means the corpus into cells, probe a few cells per
    query, and the printout traces the recall/speedup curve as nprobe grows — that curve appearing
    in your terminal is the whole lesson. nprobe=64 probes everything: recall 1.0, speedup 1×,
    which is just brute force wearing an index costume.`,
    lang: 'python',
    body: `import numpy as np

rng = np.random.default_rng(0)
N, D, K = 20000, 64, 64            # corpus size, dims, IVF cells
centers = rng.normal(size=(32, D)) * 3.0        # clustered corpus: real
X = centers[rng.integers(0, 32, N)] + rng.normal(size=(N, D))  # data has
Q = centers[rng.integers(0, 32, 200)] + rng.normal(size=(200, D))  # structure

def sqdist(A, B):                  # pairwise squared euclidean, (a, b)
    return (A * A).sum(1)[:, None] - 2.0 * A @ B.T + (B * B).sum(1)

# --- build: k-means, a few iterations is plenty for an index -------------
C = X[rng.choice(N, K, replace=False)].copy()
for _ in range(10):
    assign = sqdist(X, C).argmin(1)
    for k in range(K):
        members = X[assign == k]
        if len(members):
            C[k] = members.mean(0)
cells = [np.flatnonzero(assign == k) for k in range(K)]

# --- ground truth: the honest O(N) scan we are trying to avoid -----------
truth = np.argsort(sqdist(Q, X), 1)[:, :10]

print(f"corpus={N}  dims={D}  cells={K}   (brute force scans {N}/query)")
cell_order = np.argsort(sqdist(Q, C), 1)        # nearest cells per query
for nprobe in (1, 2, 4, 8, 16, K):
    hits, scanned = 0, 0
    for i in range(len(Q)):
        cand = np.concatenate([cells[k] for k in cell_order[i, :nprobe]])
        scanned += len(cand)
        d = ((X[cand] - Q[i]) ** 2).sum(1)      # exact, but only in-cell
        top = cand[np.argsort(d)[:10]]
        hits += len(np.intersect1d(top, truth[i]))
    recall = hits / (len(Q) * 10)
    speedup = N / (scanned / len(Q))
    print(f"nprobe={nprobe:3d}   recall@10={recall:5.3f}   speedup={speedup:6.1f}x")

# Recall climbs toward 1.0 as speedup falls toward 1x. Every ANN system --
# HNSW, IVF-PQ, DiskANN -- is a cleverer version of exactly this dial.`
  }
},

attention: {
  engineering: {
    when: `Reach for a transformer when <b>order and context change the answer</b> — the same token
    means different things depending on what surrounds it. If a bag of features scores nearly as well
    as a sequence model on your task, you don't need attention yet; you need better features
    (<x-ref to="embed">2.1</x-ref>). The honest test: shuffle the input order and re-measure. If the
    metric barely moves, you are paying quadratic cost for permutation invariance you could get from
    mean pooling.`,
    how: `The training-side decisions are mostly settled; the <b>serving-side decisions are the job</b>:
    <ul>
      <li><b>Know your two phases.</b> Prefill (reading the prompt) is compute-bound and parallel;
      decode (generating) is memory-bandwidth-bound and serial. Every serving optimisation targets
      one phase or the other, and confusing them wastes weeks.</li>
      <li><b>The KV cache is the object you operate.</b> Every generated token re-reads all previous
      keys and values, so you store them. That store, not the weights, usually caps batch size at
      long context. Do the arithmetic before you pick hardware (formula in Research).</li>
      <li><b>Cache the prompt prefix.</b> Requests sharing a prefix — system prompt, few-shot
      examples, a document — share identical, reusable prefill. The single cheapest latency win in
      production, and providers now sell it directly (Industry).</li>
      <li><b>Cut KV heads, not quality.</b> Grouped-query attention shares K/V across query heads; a
      group factor of 4–8 shrinks the cache several-fold at near-zero quality cost (Ainslie et al.,
      2023). A pre-training decision — you mostly inherit it from the base model.</li>
    </ul>`,
    where: `Anywhere the problem can be phrased as "given everything so far, predict the next thing":
    language, code, and — the reason this course cares — user behaviour, where the tokens are items
    and actions rather than words (<x-ref to="seqrec">3.2</x-ref>). The same machinery also decides
    what a retrieval-augmented system actually does with the documents you stuff into its context
    (<x-ref to="rag">4.1</x-ref>).`,
    breaks: `<ul>
      <li><b>Quadratic prefill.</b> A 10× longer prompt is ~100× more attention compute. Long
      context is sold as a feature; it is billed, correctly, as a cost.</li>
      <li><b>KV cache blowout.</b> At long context the cache dwarfs everything else in memory and
      silently caps concurrency: throughput collapses as prompts lengthen while GPU utilisation
      looks fine.</li>
      <li><b>Long context ≠ used context.</b> Models attend unevenly — "Lost in the Middle" (Liu et
      al., 2023) showed retrieval quality dropping when the relevant passage sits mid-context.
      Fitting a document in the window is not the same as the model reading it.</li>
      <li><b>No plan exists.</b> Each token is chosen greedily-in-context; nothing holds the model to
      an outline. Long generations drift unless you put the outline <em>in</em> the context — which
      is half of why agent scaffolding exists (<x-ref to="agent">4.4</x-ref>).</li>
    </ul>`
  },
  research: {
    evals: `Pre-training is steered by one number — <b>held-out per-token cross-entropy</b> —
and the discipline of this topic is knowing exactly how far that number can be trusted.
<ul>
  <li><b>Perplexity / bits-per-token</b> — the exponential of mean NLL; averaged over all tokens
  it is dominated by easy ones, so a model can improve it while precise long-range recall gets
  worse.</li>
  <li><b>Needle-in-context probes</b> — can the model quote a planted fact from position p; the
  trap is that synthetic needles are lexically salient, so passing certifies readability of the
  position, not use of it.</li>
  <li><b>Downstream benchmark suites</b> — task accuracy after training; contamination is the
  standing trap, since the test set leaks into web-scale corpora faster than benchmarks rotate.</li>
  <li><b>Tokens/sec and time-to-first-token</b> — the serving metrics; measured at batch one on
  short prompts they hide the KV-cache pressure that dictates real concurrency.</li>
</ul>
<pre>L = −(1/n) Σ_t log p_θ(x_t | x_1 … x_{t−1})      # next-token CE, every position supervised at once
perplexity = exp(L);  bits/token = L / ln 2</pre>
The gap this topic teaches: small differences in validation loss fail to predict instruction
following, tool use or refusal behaviour — which is why post-training exists as a separate stage
with separate evals, and why loss curves alone never green-light a release.`,

    maths: `Scaled dot-product attention over a sequence of hidden states packed as rows of Q, K, V:
    <pre>Attention(Q, K, V) = softmax( QKᵀ / √d_k ) V</pre>
    <b>Why √d_k:</b> if the components of q and k are independent with mean 0 and variance 1, the dot
    product q·k has variance d_k. Unscaled, logits grow with dimension, softmax saturates to a
    one-hot, and gradients through it vanish. Dividing by √d_k pins the logit variance at 1
    regardless of head size — a one-line fix stated in Vaswani et al. (2017), section 3.2.1.
    <br><br>
    <b>Why the causal mask makes it a next-token machine:</b> mask the upper triangle (position i
    sees only ≤ i) and the model's output at every position is a conditional distribution
    p(x_{i+1} | x_1..x_i). The joint factorises by the chain rule:
    <pre>p(x_1..x_n) = Π_i  p(x_i | x_1 .. x_{i−1})</pre>
    so <em>one</em> forward pass over a length-n sequence yields n training examples at once. That
    parallelism — not the attention pattern itself — is why transformers displaced RNNs.
    <br><br>
    <b>KV cache size</b> (the serving equation you will actually use):
    <pre>bytes = 2 · n_layers · n_kv_heads · d_head · seq_len · bytes_per_elem</pre>
    The 2 is K and V. Worked example, a 70B-class shape: 80 layers, 8 KV heads (GQA), d_head 128,
    fp16 → 2·80·8·128·2 = 327,680 bytes ≈ 320 KB per token, so a single 128K-token sequence holds
    ≈ 42 GB of cache — more than the fp16 weights of many models it might be serving alongside.
    With full multi-head attention (64 heads instead of 8) it would be 8× worse; that ratio is the
    entire case for GQA.`,
    papers: `<ul>
      <li><b>Attention Is All You Need</b> (Vaswani et al., 2017,
      https://arxiv.org/abs/1706.03762) — dropped recurrence entirely; reported 28.4 BLEU on WMT14
      EN→DE, with the big model trained in 3.5 days on 8 GPUs. The efficiency claim aged even better
      than the quality claim.</li>
      <li><b>FlashAttention</b> (Dao et al., NeurIPS 2022, https://arxiv.org/abs/2205.14135) —
      changes <em>zero</em> FLOPs and zero maths; it is IO-aware tiling that never materialises the
      n×n matrix in GPU HBM, computing softmax blockwise in on-chip SRAM. Exact attention, memory
      linear in sequence length, up to ~3× end-to-end training speedup. The lesson: the bottleneck
      was memory traffic, not arithmetic.</li>
      <li><b>Scaling laws.</b> Kaplan et al. (2020) fit power laws of loss in parameters, data and
      compute. <b>Chinchilla</b> (Hoffmann et al., 2022, https://arxiv.org/abs/2203.15556) corrected
      the allocation: compute-optimal training wants roughly <b>20 tokens per parameter</b> — their
      70B model trained on 1.4T tokens beat much larger under-trained peers. Production models now
      routinely train far past this ratio because inference cost, ignored by both papers, favours
      smaller models trained longer.</li>
      <li><b>Speculative decoding</b> (Leviathan et al., 2023, https://arxiv.org/abs/2211.17192) —
      a small draft model proposes tokens, the big model verifies them in one parallel pass;
      rejection sampling keeps the output distribution <em>exactly</em> unchanged. Reported 2–3×
      decode speedup. It works precisely because decode is bandwidth-bound: verification rides along
      almost free.</li>
      <li><b>StreamingLLM</b> (Xiao et al., 2023, https://arxiv.org/abs/2309.17453) — found that
      keeping just ~4 initial "attention sink" tokens plus a sliding window preserves quality on
      unbounded streams. Softmax must put its mass somewhere; the model parks it on the first tokens.</li>
    </ul>`,
    scratch: `Write single-head causal attention in ~30 lines (the Code tab is a complete version),
    then run three experiments:
    <ol>
      <li>Train on a repeated-sequence copy task and <b>print the attention matrix</b>. A diagonal
      stripe appears at the copy offset — the model learnt a lookup table, visibly.</li>
      <li>Add a KV cache to your generate loop and assert the logits are bitwise-identical to the
      cache-free version. The cache is an optimisation, not an approximation — prove it once.</li>
      <li>Time a forward pass at n and 2n tokens. The 4× wall-clock ratio makes quadratic cost
      visceral in a way the formula never will.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Anthropic</b> sells the KV-cache insight directly as prompt caching on the Claude API:
      cache writes billed at a 25% premium over base input tokens, cache <em>reads</em> at 10% of the
      base input price — Anthropic's announcement claims up to 90% cost and up to 85% latency
      reduction on long prompts. The pricing is a public admission of the arithmetic above: prefill
      you can reuse is nearly free; prefill you recompute is the product's main cost.</li>
      <li><b>Sliding windows shipped quietly.</b> Multiple 2025 production open-weight model families
      interleave a global-attention layer with several sliding-window layers (a ~1:5 ratio with
      windows around 512–1024 tokens plus a small attention sink is the pattern documented in 2025
      hybrid-architecture analyses). Full quadratic attention at every layer is already the
      exception, not the rule.</li>
      <li><b>State-space hybrids</b> reached frontier credibility: Mamba (Gu & Dao, 2023,
      https://arxiv.org/abs/2312.00752) made recurrence competitive again, and the 2025–2026
      production consensus is hybrid stacks — mostly linear-time layers, a minority of attention
      layers retained for precise recall. Attention is becoming an ingredient, not the recipe.</li>
    </ul>`,
    startups: `Inference serving became a company category because the gap between naive and careful
    attention serving is the whole margin: continuous batching, prefix caching, quantised KV caches,
    speculative decoding. None of these change the model; they change who can serve it cheaply. The
    other surface is long-context products (contract review, codebase question-answering), where the
    real question is "pay quadratic prefill on everything" versus "retrieve first" — and retrieval
    usually wins on cost even when the window fits.`,
    open_source: `<b>flash-attention</b> (Dao-AILab) is the kernel nearly every training stack links
    against — read the README before the paper. <b>vLLM</b> introduced PagedAttention (Kwon et al.,
    2023, https://arxiv.org/abs/2309.06180): virtual-memory-style paging for the KV cache, reporting
    memory waste cut to under 4% and 2–4× serving throughput over prior systems — the clearest
    evidence that the cache, not the compute, was the serving bottleneck. For learning, Karpathy's
    from-scratch language-model lecture and its companion repo remain the shortest path from zero to
    a trained model; this course's sources link it.`
  },
  next: {
    problems: `<ul>
      <li><b>The subquadratic quality gap.</b> Linear-attention and state-space layers still lose
      to exact attention on precise long-range recall. Hybrids paper over this with a few attention
      layers — how few, and why those suffice, is not understood.</li>
      <li><b>Length generalisation.</b> Models trained at one context length degrade beyond it even
      with rotary-embedding tricks; "trained on 8K, reliable at 1M" is marketed, not solved.</li>
      <li><b>KV-cache compression.</b> Quantisation, eviction, low-rank latent caches — every month
      brings a scheme; a principled account of what the cache must keep does not exist.</li>
      <li><b>Scaling laws under data exhaustion.</b> Chinchilla assumed unlimited fresh tokens. With
      high-quality text finite, the question is repeated and synthetic data — and whether the power
      laws survive either.</li>
    </ul>`,
    watch: `Track three lineages: the FlashAttention line (kernels adapting to each GPU generation),
    hybrid-architecture papers (the attention-to-SSM layer ratio is quietly becoming a headline
    hyperparameter), and serving-team engineering posts — arXiv cs.CL/cs.LG for "KV cache" and
    "hybrid attention" monthly. When the fraction of attention layers in frontier hybrids stops
    falling, that number is telling you what attention uniquely does.`
  },
  code: {
    title: 'Causal attention learns a visible lookup table',
    note: `Runs on CPU in seconds. One attention head, trained to continue a repeated sequence — the
    only way to succeed is to look back exactly 7 positions, and the printed attention matrix shows
    it doing precisely that. This is the "attention is a learned lookup" claim made visible.`,
    lang: 'python',
    body: `import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
V, L, D = 16, 8, 32              # vocab, half-length, model dim
T = 2 * L

def batch(n=64):
    first = torch.randint(0, V, (n, L))
    seq = torch.cat([first, first], 1)       # second half repeats the first
    return seq[:, :-1], seq[:, 1:]           # input, next-token target

class OneHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.out = nn.Linear(D, V)
    def forward(self, x):
        t = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(t))
        q, k, v = self.q(h), self.k(h), self.v(h)
        att = (q @ k.transpose(-2, -1)) / D ** 0.5          # scaled dot product
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float('-inf')).softmax(-1)  # causal: no future
        return self.out(att @ v), att

model = OneHead()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(400):
    x, y = batch()
    logits, _ = model(x)
    loss = F.cross_entropy(logits[:, L-1:].reshape(-1, V), y[:, L-1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.3f}")   # hits 0.000 fast

x, y = batch(1)
logits, att = model(x)
print("target :", y[0, L-1:].tolist())
print("predict:", logits[0, L-1:].argmax(-1).tolist())
print("where each second-half position looks (expect a stripe at i-7):")
for i in range(L - 1, T - 1):
    peak = att[0, i].argmax().item()
    bar = "".join("#" if w > 0.5 else "+" if w > 0.1 else "." for w in att[0, i].tolist())
    print(f"pos {i:2d} reads pos {peak:2d}   {bar}")
# The '#' marches down a perfect off-diagonal: the head learnt, from data alone,
# that "the answer lives exactly 7 tokens back" -- a lookup, made visible.`
  }
},

seqrec: {
  engineering: {
    when: `Reach for sequential modelling when <b>order carries signal your features can't</b> — session
    intent, binge patterns, the fact that what someone did five minutes ago outranks their whole profile.
    If your users generate a handful of interactions a year, a well-tuned feature model
    (<x-ref to="features">1.2</x-ref>) will match a transformer at a fraction of the serving cost;
    the sequence only pays when there is enough of it per user to read.`,
    how: `The realistic adoption ladder, in order of increasing commitment:
    <ul>
      <li><b>Sequence as a feature.</b> Encode the last N actions with a small transformer and feed the
      pooled output into your existing ranker. This is Pinterest's TransAct pattern: real-time action
      sequences as one input among many, not a re-platform. Ship this first.</li>
      <li><b>Sequence as the model.</b> SASRec-style: causal self-attention over item IDs, next-item
      cross-entropy, retrieval by nearest-neighbour over the final hidden state
      (<x-ref to="ann">2.2</x-ref>). One model, one objective, easy to reason about.</li>
      <li><b>Sequence as the platform.</b> The HSTU move: interleave items and actions into one token
      stream, replace the ranker itself, serve with candidate micro-batching. Only sensible if
      maintaining many specialised models already hurts — the consolidation story Netflix told about
      its 2025 foundation model.</li>
    </ul>`,
    where: `Ranking first, retrieval second. The sequence encoder's user state is also the natural input
    to everything downstream — Netflix's blog describes one foundation model whose embeddings feed home
    page, search and messaging. Pair it with compact item identifiers
    (<x-ref to="sid">3.1</x-ref>) and long histories stop being a memory problem.`,
    breaks: `<ul>
      <li><b>Evaluation lies to you first.</b> Rank the true next item against 100 sampled negatives and
      your metric is not a noisy version of the real one — Krichene & Rendle (KDD 2020) showed sampled
      metrics can reverse model orderings, and at small sample sizes everything collapses toward AUC.
      Evaluate against the full catalogue or apply their corrections; never trust a leaderboard that
      doesn't say which it did.</li>
      <li><b>Training budget sensitivity.</b> Petrov & Macdonald (RecSys 2022) could not reproduce
      BERT4Rec with its default configuration and needed up to 30× the default training time to match
      the published numbers. Half the literature comparing it to SASRec compared an undertrained model.</li>
      <li><b>Latency at sequence length.</b> Attention cost grows quadratically and users keep acting;
      you need a truncation policy, prefix caching, or an HSTU-style block before anyone asks for
      8k-length histories.</li>
      <li><b>Non-stationarity.</b> Catalogue and behaviour both drift; a recommender trained like a
      static language model goes stale in days. Incremental training is not optional here.</li>
    </ul>`
  },
  research: {
    evals: `The decisive number is <b>hit@k and NDCG@k against the full catalogue under a temporal
split</b> — both qualifiers doing real work, since each is where published numbers go wrong.
<ul>
  <li><b>Full-catalogue hit@k</b> — the true next item ranked against everything; the sampled
  variant is the famous trap, inflating scores and sometimes reordering the systems compared.</li>
  <li><b>Split protocol validity</b> — leave-one-out slices a user's last event out regardless of
  when it happened, letting global trends leak backwards; a single time cut across all users is
  the honest split.</li>
  <li><b>Catalogue coverage of predictions</b> — distinct items appearing in anyone's top-k;
  hit-rate can climb while the model degenerates into a popularity chart, and only this metric
  notices.</li>
  <li><b>Short-history slice</b> — the same metrics for users with few events; aggregates are
  carried by heavy users, hiding that new users get near-random slates.</li>
</ul>
<pre>next-item CE (full catalogue when it fits): L = −Σ_t log softmax(h_t·E)[i_{t+1}]
sampled with correction:  L = −log [ exp s⁺ / (exp s⁺ + Σ_j exp(s_j − log q(j))) ]
pairwise BPR:             L = −Σ log σ( s(u,i⁺) − s(u,i⁻) )    # when only ordering matters</pre>
Offline hit-rate gains evaporate online with depressing regularity: the logged next item was
itself served by the previous policy, so the eval partly measures agreement with the incumbent
rather than with the user.`,

    maths: `The whole field is one factorization. Given a user's ordered history
    <code>i₁ … i_t</code>, model the next item autoregressively:
    <pre>p(i₁,…,i_T) = Πₜ p(i_{t+1} | i₁ … i_t)
L = −Σₜ log softmax(hₜ · E)[i_{t+1}]     # hₜ = encoder state, E = item embeddings</pre>
    That softmax is over the <b>entire catalogue</b>, which for 10⁸ items is the expensive part — so
    everyone samples negatives. But sampled softmax with proposal <code>q</code> is only unbiased if you
    subtract <code>log q(j)</code> from each sampled logit (the logQ correction); skip it and the model
    learns to suppress popular items exactly as hard as they are over-sampled. The same trap appears at
    evaluation time, which is the Krichene–Rendle result above.
    <br><br>
    <b>What HSTU actually changes.</b> Standard attention normalizes:
    <pre>Attn(X) = softmax(QKᵀ/√d) V</pre>
    HSTU (Zhai et al., ICML 2024) drops the softmax and adds gating:
    <pre>A     = φ(QKᵀ + rabᵖ·ᵗ)          # φ = SiLU, pointwise — no normalization
Y     = f( Norm(A V) ⊙ U(X) )     # U is a learned elementwise gate</pre>
    <code>rabᵖ·ᵗ</code> is a relative attention bias over both position gap and bucketized time gap —
    "how long ago" is a first-class input, not a positional afterthought. Removing softmax means
    attention weights keep their <em>magnitude</em>: a user who hammered one topic produces large
    aggregate weights, where softmax would renormalize that intensity away. In engagement data the
    intensity is the signal (contrast with <x-ref to="attention">2.1</x-ref>, where relative weighting
    is exactly what you want).`,
    papers: `<ul>
      <li><b>SASRec</b> (Kang & McAuley, 2018, https://arxiv.org/abs/1808.09781) — causal self-attention
      over item sequences. Still the baseline that refuses to die; read it first because everything since
      is a delta against it.</li>
      <li><b>BERT4Rec</b> (Sun et al., 2019, https://arxiv.org/abs/1904.06690) — masked-item prediction,
      bidirectional context. Then read <b>Petrov & Macdonald</b>
      (https://arxiv.org/abs/2207.07483), the replicability study that found the original results only
      reproduce with up to 30× longer training — a masterclass in why you rerun baselines yourself.</li>
      <li><b>Actions Speak Louder than Words</b> (Zhai et al., ICML 2024,
      https://arxiv.org/abs/2402.17152) — HSTU, generative recommenders, the 1.5T-parameter deployment
      with +12.4% in online A/B tests, and the claim that quality scales as a power law of training
      compute across three orders of magnitude. The scaling claim is the important one: it makes compute
      a purchasable lever for recommendation, which it never reliably was.</li>
      <li><b>On Sampled Metrics for Item Recommendation</b> (Krichene & Rendle, KDD 2020) — why your
      offline numbers may be fiction.</li>
      <li><b>TransAct V2</b> (Pinterest, 2025, https://arxiv.org/abs/2506.02267) — lifelong action
      sequences plus a next-action auxiliary loss inside a pointwise CTR ranker; the pragmatic
      counterpoint to Meta's full reframing.</li>
    </ul>`,
    scratch: `Build order that teaches the most per hour:
    <ol>
      <li>Write SASRec in under a hundred lines (the code tab is a start). Train on MovieLens-1M with
      full-softmax cross-entropy — it fits on a laptop and spares you the sampling trap entirely.</li>
      <li>Evaluate hit@10 twice: full catalogue, then against 100 sampled negatives. Watch the sampled
      number flatter you. That gap is the Krichene–Rendle paper in one experiment.</li>
      <li>Swap the softmax attention for SiLU pointwise attention and add a time-gap bias. You now have
      a toy HSTU block, and you can check whether magnitude-preserving attention helps when you plant
      repeat-intensity signal in synthetic data.</li>
      <li>Only then read Meta's open-source generative-recommenders repo — it will make sense instead of
      being 285 config flags.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Meta</b> deployed HSTU-based generative recommenders at 1.5 trillion parameters across
      surfaces with billions of users, reporting +12.4% on online A/B metrics (Zhai et al., ICML 2024).
      The paper also reports HSTU up to 65.8% better NDCG than baselines offline and 5.3–15.2× faster
      than FlashAttention2-based transformers on 8192-length sequences. Serving works because
      <b>M-FALCON</b> batches the candidates into the forward pass: the user-history attention is
      computed once and shared, so scoring m candidates costs roughly one sequence pass plus m cheap
      extensions. That amortization is how a model 285× more FLOPs-heavy than the DLRM it replaced ran
      at 1.50–2.48× <em>higher</em> throughput scoring 1024–16384 candidates on the same budget.</li>
      <li><b>Pinterest</b> took the incremental path twice: TransAct (KDD 2023) put realtime action
      sequences into Homefeed ranking; TransAct V2 (2025) extended to lifelong sequences with a
      next-action loss and reported gains in engagement volume and diversity in online A/B tests. Note
      what they did <em>not</em> do: replace the ranker with a generative model.</li>
      <li><b>Netflix</b> (2025 TechBlog) consolidated many specialised personalization models into one
      interaction-sequence foundation model — tokenized events, sliding-window sampling over long
      histories, incremental training — whose embeddings serve multiple downstream applications.</li>
      <li><b>Kuaishou</b> pushed furthest toward end-to-end generation, but that story belongs to
      semantic IDs — see <x-ref to="sid">3.1</x-ref> for OneRec.</li>
    </ul>`,
    startups: `The honest read: trillion-parameter generative ranking is a hyperscaler sport, because
    the win comes from enormous engagement streams plus the serving engineering to amortize inference.
    What transfers down-market is the TransAct shape — a modest transformer over the last few hundred
    actions feeding an existing ranker — and vendors selling recommendation infrastructure now pitch
    exactly that. If you are a small team, the sequence encoder is the cheapest 80%; the scaling law is
    someone else's capex.`,
    open_source: `Meta's <b>generative-recommenders</b> repo is the reference HSTU and M-FALCON
    implementation, and the model file is genuinely short — read it before forming opinions. It also
    became the basis of MLPerf's DLRMv3 inference benchmark (MLCommons, 2026), which is the clearest
    sign the industry expects this architecture to stick. <b>RecBole</b> and Petrov & Macdonald's
    <b>bert4rec_repro</b> repo are the places to get honest, comparable SASRec/BERT4Rec baselines
    without re-fighting the reproducibility war yourself.`
  },
  next: {
    problems: `<ul>
      <li><b>Does the scaling law generalize?</b> One paper, one company's data. Whether power-law
      scaling of recommendation quality holds across domains, sparser platforms and public datasets is
      the field's live question — and public replication is bottlenecked on data nobody can share.</li>
      <li><b>Lifelong context.</b> Years of behaviour per user at inference-time latency budgets:
      truncation, retrieval over one's own history, or compressed state are all being tried and none
      has clearly won.</li>
      <li><b>Objective mismatch.</b> Next-item likelihood is not long-term satisfaction; every team that
      scaled it rediscovers clickbait amplification and reaches for preference-style corrections from
      the reinforcement-learning playbook.</li>
      <li><b>Evaluation infrastructure.</b> After Krichene–Rendle and the BERT4Rec saga, the field still
      lacks a benchmark where offline gains predict online ones. That gap, not architecture, is where
      progress is cheapest.</li>
    </ul>`,
    watch: `Follow engineering blogs, not just arXiv — Meta, Pinterest, Netflix and Kuaishou publish
    deployment numbers papers omit. On arXiv cs.IR, watch for any independent replication of the HSTU
    scaling curves; it will be the most important recsys result of whichever year it lands.`
  },
  code: {
    title: 'A minimal SASRec, and why order beats popularity',
    note: `Runs on CPU in under a minute. The synthetic data has a planted rule — item i is usually
    followed by item i+1 — plus a few popular items that appear everywhere. A popularity baseline can
    only exploit the second; causal attention reads the first. The printed gap is the entire argument
    for sequential recommendation.`,
    lang: 'python',
    body: `import torch, torch.nn as nn
torch.manual_seed(0)
V, D, L, N = 50, 32, 8, 2000   # catalogue, dim, seq len, users

def make_seqs(n):
    # planted behaviour: i -> i+1, with 15% random jumps to popular items 0-4
    seqs = torch.zeros(n, L, dtype=torch.long)
    for s in range(n):
        cur = torch.randint(0, V, (1,)).item()
        for t in range(L):
            if torch.rand(1).item() < 0.15:
                cur = torch.randint(0, 5, (1,)).item()
            seqs[s, t] = cur
            cur = (cur + 1) % V
    return seqs

train, test = make_seqs(N), make_seqs(400)

class TinySASRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(L, D)
        self.attn = nn.MultiheadAttention(D, 2, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T))
        causal = torch.triu(torch.ones(T, T), 1).bool()   # no peeking ahead
        a, _ = self.attn(h, h, h, attn_mask=causal)
        h = h + a
        h = h + self.ff(h)
        return h @ self.emb.weight.t()   # full-catalogue logits: no sampling trap

model = TinySASRec()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(30):
    logits = model(train[:, :-1])                          # predict positions 1..L-1
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, V), train[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    pred = model(test[:, :-1])[:, -1].argmax(1)            # next item after prefix
    hit = (pred == test[:, -1]).float().mean().item()
    top = torch.bincount(train.flatten(), minlength=V).argmax()
    pop_hit = (test[:, -1] == top).float().mean().item()

print(f"popularity baseline hit@1: {pop_hit:.2f}")
print(f"tiny SASRec hit@1:         {hit:.2f}")
print("the gap is the order information a bag-of-items model cannot see")`
  }
},


sid: {
  engineering: {
    when: `Reach for semantic IDs when <b>cold start is your bottleneck</b> — a large fraction of
    your catalogue turns over faster than it accumulates interactions, so random-ID embeddings never
    get trained. If your catalogue is small and stable, they buy you very little and cost you a
    quantizer to maintain.`,
    how: `Two deployment paths, and choosing between them is the actual decision:
    <ul>
      <li><b>As features in an existing ranker.</b> Freeze a content encoder, train the RQ-VAE once,
      map every item to its code tuple, then learn embeddings for those code tokens alongside your
      existing sparse features. No re-platforming, measurable in one experiment, and it builds the
      SID infrastructure the generative path would need later. This is the de-risking move.</li>
      <li><b>As the retrieval mechanism.</b> Decode the ID tuple autoregressively with constrained
      beam search over the code tree. There is no ANN index at all. Much bigger change, much bigger
      payoff, and a serving problem you must solve before you start.</li>
    </ul>`,
    where: `Retrieval and ranking both. Also useful as a compact history representation: a few
    integers per item instead of a dense vector is what makes long user sequences affordable in
    memory and bandwidth — which is precisely how this connects to <x-ref to="seqrec">3.2</x-ref>.`,
    breaks: `<ul>
      <li><b>Codebook collapse.</b> A handful of codes absorb everything and the rest go unused.
      Monitor code utilization per level from the first training run; it is the single most useful
      diagnostic and almost nobody logs it.</li>
      <li><b>Encoder drift.</b> Retrain the content encoder and every previously assigned ID silently
      means something different. Version the encoder, version the codebooks, and treat an encoder
      change as a full re-index.</li>
      <li><b>Distribution mismatch.</b> The quantizer saw your catalogue snapshot; production sees
      live traffic skewed toward new and popular items. Re-fit on a representative sample.</li>
      <li><b>Embedding table growth.</b> Three levels of 256 codes is small; naive per-tuple
      embeddings are not. Embed per level and combine, don't embed the tuple.</li>
    </ul>`
  },
  research: {
    evals: `Semantic IDs are judged on the slice they were invented for: <b>recall@k restricted to
items with few or no interactions</b>. If that slice does not move, the tokenizer is decoration.
<ul>
  <li><b>Cold-slice recall@k</b> — retrieval quality on low-interaction items; the trap is
  upstream, where core-filtering the dataset deletes precisely the items the method exists to
  serve, leaving nothing to measure.</li>
  <li><b>Code utilisation per level</b> — entropy of code usage across the codebook; necessary
  but not sufficient, since random assignment is perfectly uniform and semantically empty.</li>
  <li><b>Residual norm per level</b> — how much of the embedding each level explains; its trap is
  measuring fidelity to the content encoder, which may itself encode nothing users care about.</li>
  <li><b>Prefix coherence audit</b> — do items sharing a level-1 code belong together to a human
  reader; audited only on head items it will pass while the tail is scrambled.</li>
</ul>
<pre>tokenizer: the RQ-VAE objective — reconstruction plus the two stop-gradient commitment
           terms — exactly as written in this lesson's maths tab; do not retrain it casually
generator: L = −Σ_k log p(c_k | user context, c_1 … c_{k−1})   # CE over code tokens, level by level</pre>
The offline→online gap is the quantizer's snapshot problem: codes fitted to yesterday's
catalogue meet today's uploads, and cold-start gains measured on held-out old items fail to
predict behaviour on genuinely new ones.`,

    maths: `Residual quantization applies vector quantization to successive residuals. Given an item
    embedding <code>x</code> and codebooks <code>C₁..C_m</code>, each of size K:
    <pre>r₀ = x
for k = 1..m:
    cₖ = argmin_j ‖ rₖ₋₁ − Cₖ[j] ‖²      # nearest codeword
    rₖ = rₖ₋₁ − Cₖ[cₖ]                    # what is left over
semantic ID = (c₁, c₂, …, c_m)
x̂ = Σₖ Cₖ[cₖ]                             # reconstruction</pre>
    The RQ-VAE trains encoder, decoder and codebooks jointly with a reconstruction term plus the
    standard VQ commitment terms, using the straight-through estimator because argmin has no gradient:
    <pre>L = ‖x − x̂‖²  +  Σₖ ( ‖sg[rₖ₋₁] − Cₖ[cₖ]‖²  +  β‖rₖ₋₁ − sg[Cₖ[cₖ]]‖² )</pre>
    <code>sg</code> is stop-gradient. The first codebook term pulls codewords toward the data; the
    second (weighted by β, typically 0.25) keeps the encoder from running away from the codebook.
    <br><br>
    <b>Why the hierarchy emerges:</b> level k only ever sees <code>rₖ₋₁</code>, the error left by
    levels 1..k−1. Level 1 therefore has to explain the largest variance direction — which in a
    content embedding space is broad category — and later levels get progressively finer structure.
    Quantize <code>x</code> independently m times instead and you get m noisy copies of the same
    information, and a shared prefix means nothing.`,
    papers: `<ul>
      <li><b>TIGER</b> (Rajput et al., NeurIPS 2023) introduced hierarchical semantic IDs from an
      RQ-VAE over frozen Sentence-T5 content embeddings, and generative retrieval by autoregressive
      SID decoding. Reported ≈ +17% Recall@5 and +29% NDCG@5 over the strongest baselines, with the
      cold-start gain coming from partial prefix matching at inference.</li>
      <li><b>Better Generalization with Semantic IDs</b> (Singh et al., 2023) is the production
      counterpart: SIDs as ranker features under latency constraints, where hashing and adaptation
      through embeddings is the practical crux rather than decoding.</li>
      <li><b>LETTER</b> adds collaborative signal and code-assignment diversity regularization to the
      tokenizer, addressing collapse directly. <b>ETEGRec</b> alternates tokenizer and generator
      optimization instead of freezing the tokenizer.</li>
      <li>Kuaishou's <b>OneRec</b> line replaced RQ-VAE with <b>RQ-Kmeans</b> over a
      collaborative-aware multimodal tokenizer that fuses title, tags, audio and images with behaviour.</li>
    </ul>`,
    scratch: `Implementable in an afternoon and worth doing before reading further. Order:
    <ol>
      <li>Get item text embeddings from any sentence encoder. Don't train it.</li>
      <li>Write the quantizer as a loop over levels with nearest-codeword lookup. Initialize each
      codebook by k-means on the residuals of the previous level — this alone prevents most collapse.</li>
      <li>Log code utilization per level every epoch. If level 1 uses 12 of 256 codes, stop and fix
      the initialization before touching anything else.</li>
      <li>Sanity check the hierarchy: sample items sharing a level-1 code and read their titles. If
      they aren't obviously related, the encoder or the k-means init is wrong, not the idea.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Kuaishou</b> runs the most aggressive public deployment. OneRec replaced the entire
      recall / coarse-rank / fine-rank / re-rank cascade with a single encoder-decoder that generates
      semantic IDs from user context. At an August 2025 investor event Kuaishou disclosed it covered
      roughly 25% of traffic on the main app and Lite with plans for 50–60%, and reported GMV up over
      20% in local services. OneRec-V2 moved to a lazy decoder-only design with preference alignment
      from real user feedback and FP8 post-training quantization in production; its main reported
      deployment ran one week on 5% traffic with online moves like Watch Time +0.044% and
      Comment +0.805%, characterized as no degradation while cutting compute.</li>
      <li><b>Kuaishou advertising</b> ships <b>Taiji</b>, an LLM-enhanced recommender trading off
      semantics and IDs, deployed since May 2026 across more than 400 million daily users, with a
      reported +2.83% advertiser value and +3.30% revenue.</li>
      <li><b>Snap</b> published a practitioner account of semantic IDs at Snapchat — use cases,
      technical challenges and design choices — which is unusually candid about what didn't work.</li>
      <li><b>Meta</b>'s generative-recommender line (<x-ref to="seqrec">3.2</x-ref>) uses the
      sequence reframing without SIDs as the identifier; the two ideas are separable and often
      confused.</li>
    </ul>`,
    startups: `The pattern to notice: almost nobody outside the largest platforms runs generative
    <em>retrieval</em> yet, because serving cost and constrained decoding are hard. What smaller teams
    do ship is the feature-side version — SIDs as compact item representations feeding an existing
    ranker or a vector index — which gets most of the cold-start benefit for a fraction of the
    engineering. If you are building a company here, that asymmetry is the opportunity: the tokenizer
    is reusable infrastructure and the decoder is not.`,
    open_source: `<b>OpenOneRec</b> (Kuaishou) releases 1.7B and 8B foundation models on a Qwen3
    backbone plus the full training pipeline — data processing, co-pretraining and post-training —
    explicitly to make scaling-law research in recommendation reproducible. <b>GRID</b> (Snap
    Research) is a smaller, more readable implementation of generative recommendation with semantic
    IDs, and the better one to read first.`
  },
  next: {
    problems: `<ul>
      <li><b>Tokenizer–generator co-training.</b> Freezing the tokenizer is convenient and almost
      certainly suboptimal. ETEGRec-style alternation and end-to-end variants are active work.</li>
      <li><b>Business constraints inside decoding.</b> Organic relevance and paid placement have
      different objectives; injecting bids or policy into constrained beam search without wrecking
      the likelihood model is genuinely unsolved.</li>
      <li><b>Scaling behaviour of SIDs specifically.</b> Recent work studies generative
      recommendation with semantic IDs from a model-scaling view — whether the recommendation
      scaling law survives the quantization bottleneck is not settled.</li>
      <li><b>Evaluation.</b> Reported gains vary enormously with preprocessing and negative sampling.
      Benchmarks like Kwai26 exist partly because the field's numbers stopped being comparable.</li>
    </ul>`,
    watch: `Search arXiv cs.IR for "semantic ID", "generative retrieval" and "generative
    recommendation" monthly. The field is moving fast enough that a six-month-old survey is stale,
    and slow enough that reading one paper a month keeps you current.`
  },
  code: {
    title: 'Residual quantization from scratch',
    note: `Runnable as-is on CPU. Deliberately not a library call — the loop is the lesson. Swap the
    random embeddings for real sentence-encoder output and this is a working tokenizer.`,
    lang: 'python',
    body: `import torch, torch.nn as nn

class ResidualQuantizer(nn.Module):
    """m codebooks of K codes each. Returns the code tuple and the reconstruction."""
    def __init__(self, dim=256, levels=3, codes=256):
        super().__init__()
        self.books = nn.ParameterList(
            [nn.Parameter(torch.randn(codes, dim) * 0.02) for _ in range(levels)]
        )

    def forward(self, x):                       # x: (batch, dim)
        residual, ids, recon = x, [], torch.zeros_like(x)
        for book in self.books:
            # nearest codeword by squared distance
            d = (residual.pow(2).sum(1, keepdim=True)
                 - 2 * residual @ book.t()
                 + book.pow(2).sum(1))
            idx = d.argmin(1)                   # (batch,)
            picked = book[idx]
            ids.append(idx)
            recon = recon + picked
            residual = residual - picked        # <- the whole idea lives on this line
        return torch.stack(ids, 1), recon, residual

def rq_loss(x, recon, residual, beta=0.25):
    # straight-through: argmin has no gradient, so we route it around the lookup
    commit = beta * (residual.detach() - residual).pow(2).mean()
    return (x - recon).pow(2).mean() + commit

# --- what to actually look at -------------------------------------------
q = ResidualQuantizer()
x = torch.randn(4096, 256)
ids, recon, res = q(x)

for lvl in range(ids.shape[1]):
    used = ids[:, lvl].unique().numel()
    print(f"level {lvl+1}: {used}/256 codes used")   # collapse shows up here first

print("residual norm ratio:", (res.norm(dim=1) / x.norm(dim=1)).mean().item())
# untrained this is ~1.0. After training it should fall level by level.`
  }
},

rag: {
  engineering: {
    when: `Reach for RAG when the knowledge is <b>fresh, large, or must be citable</b> — requirements no
    amount of fine-tuning satisfies. And know when not to: Anthropic's contextual-retrieval post is blunt
    that if your knowledge base is under ~200K tokens (about 500 pages), skip retrieval entirely, put the
    whole corpus in the prompt, and let prompt caching absorb the cost. RAG is what you build when the
    corpus outgrows the context window, not a default.`,
    how: `The pipeline is the retrieval cascade you already know with a generator bolted on:
    <ul>
      <li><b>Chunk</b> the corpus (<x-ref to="chunk">the chunking lesson</x-ref>) — and prepend document
      context to each chunk before indexing. Anthropic reports this one change (contextual embeddings)
      cut top-20 retrieval failure rate by 35% (5.7% → 3.7%).</li>
      <li><b>Index twice</b>: dense embeddings (<x-ref to="embed">embeddings</x-ref>) in an ANN index
      plus contextualised BM25 — the <x-ref to="hybrid">hybrid</x-ref> combination took the failure-rate
      reduction to 49% in the same study.</li>
      <li><b>Rerank</b> a wide candidate set with a cross-encoder before the prompt: 67% total reduction
      (5.7% → 1.9%) with reranking added. Retrieve ~150, rerank, pass the top 20.</li>
      <li><b>Generate with required citations</b>, then check them (<x-ref to="verify">verification</x-ref>).</li>
    </ul>
    The discipline that matters more than any component: <b>measure retrieval recall separately from
    answer quality</b>. If the evidence never reached the prompt, no prompt engineering downstream can fix it.`,
    where: `Between the query and the generator — but the honest answer is "wherever your evaluation says
    the failures are". Teams habitually blame the model when the retriever missed. Log, for every query:
    what was retrieved, whether the gold evidence was in it, and whether the answer cited it. That single
    log table is the difference between debugging and guessing.`,
    breaks: `<ul>
      <li><b>Silent retrieval failure.</b> The model answers fluently from its weights when retrieval
      misses, and nothing looks wrong. This is the failure mode; everything else is a special case of it.</li>
      <li><b>Chunking that severs context.</b> "The company grew 3%" retrieved without knowing which
      company or which quarter. Contextual retrieval exists precisely because naive chunking does this.</li>
      <li><b>Lost in the middle.</b> Liu et al. (TACL 2024) showed accuracy is highest when the relevant
      passage is at the start or end of the context and degrades when it sits in the middle — stuffing
      more chunks in is not monotonically better.</li>
      <li><b>Stale indexes and permission leaks.</b> The index is a copy of the corpus; it drifts, and it
      happily serves a document to a user who lost access yesterday. Filter by permission at query time,
      never at index time alone.</li>
    </ul>`
  },
  research: {
    evals: `One metric governs the whole pipeline: <b>evidence recall@k</b> — the gold passage made it
into the prompt — because nothing downstream can recover from its failure.
<ul>
  <li><b>Context recall@k</b> — gold evidence retrieved; the trap is labelling cost, so teams
  substitute judge-scored relevance and inherit every bias of the judge.</li>
  <li><b>Faithfulness</b> — each answer claim supported by retrieved text; a fluent paraphrase of
  an unsupported claim reads as grounded to most judges, which is exactly the case that
  matters.</li>
  <li><b>Answer correctness</b> — exact match, F1 or judged; its trap is parametric leakage,
  where the model answers from its weights and the metric silently credits the retriever.</li>
  <li><b>Citation precision</b> — cited chunks actually entail the claim they decorate; models
  learn to attach plausible-looking citations, and only span-level checking catches it.</li>
</ul>
<pre>retriever (DPR-style contrastive):  L = −log [ exp(q·d⁺/τ) / Σ_j exp(q·d_j/τ) ]
   negatives: in-batch plus mined — the same machinery as <x-ref to="embed">embeddings</x-ref>
reader, if fine-tuned:  token-level CE on answers conditioned on gold evidence</pre>
Offline recall on a curated question set fails to predict production because real users ask
questions the corpus cannot answer — and the system's behaviour on unanswerables, not on the
eval set, is what decides whether anyone trusts it.`,

    maths: `The original formulation (Lewis et al., 2020) treats the retrieved document as a latent
    variable and marginalises over it:
    <pre>p(y | x) ≈  Σ_{z ∈ top-k(p_η(·|x))}  p_η(z | x) · p_θ(y | x, z)

p_η(z | x) ∝ exp( d(z)ᵀ q(x) )     # dense retriever: DPR-style dot product</pre>
    RAG-Sequence uses one document for the whole answer; RAG-Token re-marginalises per token, letting
    different tokens draw on different documents. Modern systems collapse the sum — concatenate top-k
    and generate once — but the latent-variable reading survives as the right mental model: <b>the answer
    distribution is a mixture weighted by retrieval</b>.
    <br><br>
    <b>Why retrieval recall bounds answer recall.</b> Let R = P(gold evidence in retrieved set),
    a = P(correct | evidence present), ε = P(correct guess | evidence absent):
    <pre>P(correct) = R·a + (1 − R)·ε  ≤  R + ε</pre>
    With a strong reader (a ≈ 1) and an honest one (ε ≈ 0), end-to-end accuracy <em>is</em> retrieval
    recall. Every point of recall you fail to measure is a point of accuracy you cannot explain.
    <br><br>
    <b>The economics.</b> Cost per query ≈ retrieved tokens × input price, paid on every call. On the
    Claude API, prompt caching changes the calculus: cache writes cost 1.25× base input, cache reads 0.1×
    (per Anthropic's pricing docs) — so a stable corpus prefix is nearly free to re-send, which is exactly
    why the &lt;200K-token "no RAG" regime exists, and why Anthropic's contextual-chunk generation costs a
    one-time $1.02 per million document tokens with caching on.`,
    papers: `<ul>
      <li><b>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</b> (Lewis et al., 2020,
      https://arxiv.org/abs/2005.11401) — the original. RAG-Sequence hit 44.5 exact match on Natural
      Questions, state of the art at publication. Short and readable; read it before any survey.</li>
      <li><b>Dense Passage Retrieval</b> (Karpukhin et al., 2020, https://arxiv.org/abs/2004.04906) —
      the retriever half; dual encoders trained on question–passage pairs.</li>
      <li><b>Self-RAG</b> (Asai et al., 2023, https://arxiv.org/abs/2310.11511) — reflection tokens so
      the model learns <em>when</em> to retrieve and whether its output is supported.</li>
      <li><b>RAGAS</b> (Es et al., https://arxiv.org/abs/2309.15217) — reference-free evaluation:
      faithfulness and answer relevance score the generation; context precision and recall score the
      retrieval. The decomposition matters more than the specific scorers.</li>
      <li><b>Lost in the Middle</b> (Liu et al., https://arxiv.org/abs/2307.03172) — the position-bias
      result that makes "just add more chunks" a measurably bad plan.</li>
    </ul>`,
    scratch: `Build the evaluation before the system — an afternoon, in order:
    <ol>
      <li>Collect 50 real questions and hand-label which document answers each. This labelled set is
      worth more than any component you could build instead.</li>
      <li>Run BM25 only. Record recall@5 and recall@20. This is your baseline and it is embarrassingly
      strong on in-vocabulary queries.</li>
      <li>Add dense retrieval, then fuse. Watch <em>which questions</em> flip — dense wins on
      paraphrase, BM25 on identifiers and exact names.</li>
      <li>Only now touch generation. Any answer-quality gain you cannot trace to a recall gain is either
      prompt luck or the model answering from its weights — both are findings.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Anthropic</b> published the most useful production numbers in the field: contextual
      embeddings −35% retrieval failures, + contextual BM25 −49%, + reranking −67% (5.7% → 1.9% at
      top-20), with the contextualisation pass costing a one-time $1.02 per million document tokens using
      prompt caching. Rare in being a fully specified, reproducible recipe.</li>
      <li><b>Glean</b> built a company on the unglamorous parts: hybrid retrieval over an enterprise
      knowledge graph of people, projects and documents, with <b>permission-aware ranking from the first
      millisecond</b> — every result filtered by what the querying user may see. The lesson: in
      enterprise RAG, the connector and permission layer is the product; the LLM is a commodity.</li>
      <li><b>Code assistants</b> (Claude Code among them) push retrieval agentic: instead of one
      embedding lookup, the model greps, reads files, and follows imports — retrieval as tool use rather
      than as a fixed pre-generation step.</li>
    </ul>`,
    startups: `The vector-database land grab of 2023 taught an expensive lesson: the index was never the
    moat. What customers pay for is ingestion from messy sources, permission enforcement, freshness, and
    evaluation they can trust — the parts that look like plumbing. If you are building here, the
    asymmetry to exploit is that everyone demos retrieval and almost nobody can <em>measure</em> theirs;
    an evaluation-first product posture is still rare enough to be a differentiator.`,
    open_source: `<b>RAGAS</b> for evaluation — use its retrieval/generation metric split even if you
    replace the scorers. <b>FAISS</b> for the ANN index (<x-ref to="ann">the ANN lesson</x-ref>),
    <b>BM25S</b> or Lucene for the sparse side, <b>sentence-transformers</b> for both bi-encoders and
    cross-encoder rerankers. The full stack runs on one machine; resist orchestration frameworks until
    the eval numbers say the simple pipeline is the bottleneck.`
  },
  next: {
    problems: `<ul>
      <li><b>Agentic retrieval.</b> Letting the model decide when and what to retrieve, re-query on
      failure, and follow references (Self-RAG's question, now mainstream) multiplies cost per answer.
      When multi-hop retrieval beats two well-formed parallel queries is an empirical question most
      teams have not actually run.</li>
      <li><b>Long context vs RAG is a moving boundary, not a debate.</b> Anthropic's 200K-token
      threshold for skipping RAG will keep rising with context windows and cheaper cache reads; the
      durable question is which corpus sizes and freshness requirements keep retrieval necessary.</li>
      <li><b>Evaluation is still the weakest link.</b> Judge models carry documented verbosity, position
      and self-preference biases; faithfulness scores are only as good as the judge, and validating the
      judge against human labels remains manual.</li>
      <li><b>Index freshness and consistency.</b> Sub-minute update-to-servable latency without
      re-embedding the world is an open systems problem hiding behind every "answers from this morning's
      document" demo.</li>
    </ul>`,
    watch: `Follow Anthropic's engineering blog for production retrieval recipes with numbers attached,
    and arXiv cs.CL/cs.IR for "retrieval-augmented" and "agentic retrieval". Watch what happens to RAG
    papers' baselines as context windows grow — the honest ones now compare against long-context
    prompting with caching, and the ones that don't are dodging the question.`
  },
  code: {
    title: 'Answer quality is bounded by retrieval recall',
    note: `A toy corpus, a deliberately weak bag-of-words retriever, and a reader that is right 95% of
    the time when the evidence is in front of it. The printout shows end-to-end accuracy pinned to
    retrieval recall — which is why you evaluate retrieval directly instead of eyeballing answers.`,
    lang: 'python',
    body: `import numpy as np
from collections import Counter

corpus = {  # id -> passage (what the index actually contains)
    "refund":  "annual plan refunds allowed within 30 days of purchase",
    "cancel":  "subscriptions may be terminated from account settings",
    "sla":     "uptime commitment of 99.9 percent applies to enterprise",
    "export":  "workspace records may be downloaded as csv by admins",
    "sso":     "single sign on requires purchase of the enterprise tier",
}
questions = [  # (question, id of the only passage that answers it)
    ("how many days for annual plan refunds", "refund"),    # wording matches
    ("what uptime commitment applies to enterprise", "sla"),# wording matches
    ("how do i cancel my subscription plan", "cancel"),     # paraphrase: terminated
    ("can i export my data", "export"),                     # paraphrase: downloaded
    ("is sso included in the pro plan", "sso"),             # acronym mismatch
]

ids = list(corpus)
vocab = sorted({w for t in corpus.values() for w in t.split()})
def bow(text):  # bag-of-words over CORPUS vocabulary: lexical mismatch = zeros
    c = Counter(text.split())
    v = np.array([c[w] for w in vocab], float)
    n = np.linalg.norm(v)
    return v / n if n else v

D = np.stack([bow(corpus[i]) for i in ids])
READER_WITH, READER_WITHOUT = 0.95, 0.10  # P(correct | evidence in/out of prompt)

print("k  recall  end-to-end  gap to perfect reader")
for k in (1, 2, 3):
    hits, acc = 0, 0.0
    for q, gold in questions:
        top = np.argsort(-(D @ bow(q)))[:k]
        got = gold in [ids[j] for j in top]
        hits += got
        acc += READER_WITH if got else READER_WITHOUT  # expected accuracy
    r, a = hits / len(questions), acc / len(questions)
    print(f"{k}  {r:6.2f}  {a:10.2f}  {r - a:+.2f}")

print()
print("accuracy = recall*0.95 + (1-recall)*0.10 -- a ceiling set by retrieval.")
print("no prompt fixes an answer whose evidence never reached the prompt.")`
  }
},

hybrid: {
  engineering: {
    when: `Reach for hybrid the moment your queries contain <b>rare exact strings</b> — part numbers,
    error codes, ticket IDs, legal citations, people's names. A dense encoder (<x-ref to="embed">2.1</x-ref>)
    represents "E-4471" by its neighbourhood, so it retrieves things that look like error codes rather
    than that one — a category failure, not a quality failure, so a better encoder just blurs more
    confidently. If every query is conversational paraphrase over prose, dense alone may suffice; most
    real corpora are not like that, which is why hybrid is the production default and pure-dense is
    the demo default.`,
    how: `The standard recipe, in order of decreasing certainty:
    <ul>
      <li><b>Run both retrievers in parallel.</b> BM25 over an inverted index, dense over an ANN index
      (<x-ref to="ann">2.2</x-ref>). Take the top 50–200 from each.</li>
      <li><b>Fuse with RRF, not score averaging.</b> BM25 scores are unbounded positives; cosine lives
      in [−1, 1]. Any fixed weighting is arbitrary and shifts as the corpus grows. Ranks are comparable
      by construction, and RRF with k=60 needs no tuning to be respectable.</li>
      <li><b>Rerank the fused list with a cross-encoder</b>, then keep 5–15 for the generator
      (<x-ref to="rag">4.1</x-ref>). This is the same funnel discipline as everywhere else: cheap and
      recall-oriented first, expensive and precise last.</li>
    </ul>
    Tuned weighted fusion can beat RRF once you have labelled queries to tune against. Until then you
    are guessing, and RRF is the guess that fails least.`,
    where: `Anywhere retrieval feeds generation, and most places it feeds people: RAG pipelines,
    enterprise and e-commerce search, support-ticket dedup, code search (identifiers are the ultimate
    exact-match tokens). Every serious engine ships it natively — Elasticsearch has an RRF retriever,
    Qdrant's Query API prefetches sparse and dense sub-queries and fuses them, Weaviate exposes an
    alpha knob and a choice of fusion modes. You should almost never build the plumbing yourself.`,
    breaks: `<ul>
      <li><b>Score fusion across incompatible scales.</b> The classic self-inflicted wound: min-max
      normalizing BM25 makes the top score 1.0 whether the best hit was superb or merely least-bad.
      Rank fusion sidesteps this; score fusion needs per-query normalization and still drifts.</li>
      <li><b>Fusing garbage.</b> RRF rewards documents that appear in both lists. If one retriever is
      systematically wrong for a query class, its confident junk can outvote the other's lone correct
      answer. Fusion is not a substitute for either retriever being decent.</li>
      <li><b>Tokenization mismatch.</b> Your analyzer lowercases and strips hyphens, so "E-4471"
      becomes two tokens and exact match quietly stops being exact. Test the lexical side with your
      actual gnarly strings, not with prose.</li>
      <li><b>Two indexes, two consistency problems.</b> A document updated in one index but not the
      other produces fusion results nobody can debug. Index both sides in the same write path.</li>
    </ul>`
  },
  research: {
    evals: `A fusion change ships on <b>nDCG@10 over a labelled query set stratified by query
class</b> — exact-identifier queries and paraphrase queries reported separately, never pooled.
<ul>
  <li><b>Per-class nDCG@10</b> — the stratified headline; pooled, its trap is that one retriever
  carries one class and regressions in the other average away invisibly.</li>
  <li><b>Fused recall@100</b> — what the reranker is given to work with; measuring only the final
  output lets a strong cross-encoder mask a fusion regression for months.</li>
  <li><b>Identifier-slice exact hit rate</b> — the gnarly strings; the trap is evaluating with a
  cleaner analyzer than production runs, so the eval passes on tokens production splits apart.</li>
  <li><b>Weight-sensitivity sweep</b> — metric as a function of the fusion parameter; a sharp
  peak means you fitted the snapshot, and the peak will move when the corpus does.</li>
</ul>
<pre>fusion: untrained — RRF has one constant, chosen once and left alone
cross-encoder reranker:  L = BCE(σ(f(q,d)), rel)   or   L = −log σ(f(q,d⁺) − f(q,d⁻))
learned sparse: ranking loss + a sparsity regulariser on term activations (index cost is in the loss)</pre>
Offline fusion wins fail to transfer when the query mix shifts: the labelled set fixes the ratio
of lexical to semantic queries, and production renegotiates that ratio every week.`,

    maths: `BM25 scores a document D for query terms q₁..qₙ:
    <pre>score(D, Q) = Σᵢ IDF(qᵢ) · tf(qᵢ, D) · (k₁ + 1)
                      ─────────────────────────────────────────
                      tf(qᵢ, D) + k₁ · (1 − b + b · |D| / avgdl)</pre>
    <code>k₁</code> (typically 1.2–2.0) controls term-frequency saturation: the second occurrence of a
    term is worth less than the first, and the curve flattens — repeating a keyword fifty times cannot
    buy fifty times the score. <code>b</code> (typically 0.75) controls length normalization: at b=1 a
    long document is fully penalized for its length, at b=0 not at all. The IDF term is why BM25 owns
    rare strings: a token that appears in one document out of a million carries enormous weight, which
    is exactly the signal a dense encoder averages away.
    <br><br>
    Reciprocal Rank Fusion combines ranked lists by position alone:
    <pre>RRF(d) = Σ_r  1 / (k + rank_r(d))        k = 60</pre>
    Rank-based fusion sidesteps score incompatibility entirely: whatever units each retriever prints,
    "second place" means the same thing in both lists. The constant k damps the top of each list — at
    k=60 the gap between rank 1 and rank 2 (1/61 vs 1/62) is small, so one retriever's confident first
    place cannot steamroll broad agreement lower down; as k→0 the top rank dominates, as k→∞ every
    position counts equally. k=60 is the empirical setting from Cormack et al., and nearly every vendor
    kept it.`,
    papers: `<ul>
      <li><b>Reciprocal Rank Fusion</b> (Cormack, Clarke &amp; Buettcher, SIGIR 2009). Three pages.
      RRF beat Condorcet fusion and individual learned rankers on TREC data; k=60 comes from here.
      Probably the highest impact-per-page ratio in IR.</li>
      <li><b>BEIR</b> (Thakur et al., NeurIPS 2021) is the load-bearing evidence for hybrid: across 18
      zero-shot datasets, the best dense retriever evaluated, TAS-B, beat BM25 on only 8 of 18 — while
      the same class of dense models beats BM25 comfortably in-domain on MS MARCO. In-domain wins do
      not transfer; lexical robustness does.</li>
      <li><b>SPLADE / SPLADE v2</b> (Formal et al., 2021) learns sparse expansions over the vocabulary
      so a document about jams also indexes "stuck" — learned semantics served from an ordinary
      inverted index, with strong zero-shot BEIR results.</li>
      <li><b>ColBERT</b> (Khattab &amp; Zaharia, 2020) and <b>ColBERTv2</b>: per-token embeddings with
      a MaxSim operator — query and document interact late, so the index is still precomputable.
      <b>PLAID</b> (Santhanam et al., CIKM 2022) made it cheap: 3.7× faster on GPU and 22× on CPU than
      vanilla ColBERTv2 with no quality loss, tens of milliseconds at 140M passages.</li>
    </ul>`,
    scratch: `Buildable in an evening, and you should:
    <ol>
      <li>Implement BM25 from the formula over a few hundred documents. It is thirty lines.</li>
      <li>Embed the same corpus with any sentence encoder into a brute-force cosine scorer.</li>
      <li>Write ten queries: five with rare exact strings, five pure paraphrases. Score both retrievers
      per class — you will watch each one fail exactly where theory says it must.</li>
      <li>RRF-fuse and re-score. Then try weighted score averaging and watch it whipsaw as you vary
      the weight; that instability is the entire argument for rank fusion.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Anthropic</b> published the cleanest public ablation, in the Contextual Retrieval
      engineering post (2024): on their benchmark, contextual embeddings cut top-20 retrieval failure
      from 5.7% to 3.7% (−35%); adding contextual BM25 — hybrid — cut it to 2.9% (−49%); adding a
      reranker on top cut it to 1.9% (−67%). The one-time contextualization cost was $1.02 per million
      document tokens with prompt caching. Note the shape: each stage of the stack pays, and BM25 is
      still in the winning configuration in 2024.</li>
      <li><b>Elastic</b> and <b>OpenSearch</b> ship hybrid as a first-class query type, which tells you
      where enterprise demand landed. Elasticsearch went with RRF; OpenSearch's hybrid query normalizes
      and combines scores in a search pipeline, with RRF added later — the two defaults are a live
      A/B of the fusion debate.</li>
      <li><b>Vespa</b> (ex-Yahoo) treats fusion as just another ranking expression and pushes phased
      ranking — cheap first phase, expensive second phase — which is the retrieval funnel from part 1
      expressed as a query language.</li>
    </ul>`,
    startups: `The vector-database generation (Qdrant, Weaviate, Milvus, Pinecone) all launched
    dense-first and all retrofitted sparse — that migration is the market admitting BEIR was right.
    Reranking became its own product category: Cohere's Rerank endpoint and a crowd of cross-encoder
    APIs exist because a hosted reranker is the highest quality-per-line-of-code upgrade a retrieval
    stack can buy. If you are building here, note what is commoditized: the indexes and the fusion are
    free; the defensible work is evaluation on the customer's own queries.`,
    open_source: `<b>Pyserini / Anserini</b> is the reproducibility workhorse — Lucene BM25 plus dense
    baselines with published numbers to check yourself against. <b>rank_bm25</b> is the fifty-line
    Python version for prototypes. <b>ColBERT/PLAID</b> ships from the Stanford repo. <b>SPLADE</b>
    weights are downloadable and servable from any inverted index. Postgres people get surprisingly
    far with pgvector plus tsvector and RRF in twenty lines of SQL.`
  },
  next: {
    problems: `<ul>
      <li><b>Query-adaptive fusion.</b> "error E-4471" should weight lexical; "why is my printer
      unhappy" should weight dense. Global k or alpha is a compromise across query types; classifying
      the query and routing the weights is obvious, mostly unshipped, and evaluation-starved.</li>
      <li><b>One index instead of two.</b> Learned sparse (SPLADE) and late interaction (ColBERT) each
      claim to collapse the hybrid stack into a single model. Neither has displaced BM25-plus-dense in
      practice — serving maturity, not quality, is the bottleneck.</li>
      <li><b>Fusion under corpus drift.</b> RRF's k and any tuned weights are fitted to a snapshot;
      nobody has a good story for keeping fusion calibrated as the catalogue and query mix move.</li>
      <li><b>Chunking interacts with everything upstream</b> (<x-ref to="chunk">4.3</x-ref>): the
      contextual-retrieval result says the biggest recent gain came from changing what gets indexed,
      not how it is scored. That direction is underexplored relative to fusion tweaks.</li>
    </ul>`,
    watch: `Track the BEIR and MTEB retrieval leaderboards for the dense–sparse gap out of domain, and
    arXiv cs.IR for "hybrid retrieval", "learned sparse" and "late interaction". The tell to watch for:
    the year a learned-sparse or late-interaction model becomes the default in a mainstream engine.
    Until then, BM25 plus dense plus RRF plus rerank is the boring stack that keeps winning.`
  },
  code: {
    title: 'BM25 + toy dense retrieval, fused with RRF',
    note: `Runnable as-is on CPU, stdlib + numpy. The "dense encoder" is a hand-built synonym map — a
    deliberately tiny stand-in for a sentence encoder that still reproduces the real failure mode:
    it understands paraphrase and drops rare exact codes. Watch each retriever miss where the other
    hits, and the fused list get both right.`,
    lang: 'python',
    body: `import numpy as np
from collections import Counter

docs = ["printer error e-4471 paper jam in tray two",
        "printer warning e-9020 toner running low",
        "clearing stuck pages from the paper tray",
        "resetting the machine after a fault",
        "ordering replacement toner cartridges"]
SYN = {"printer":"device","machine":"device","device":"device",
       "error":"fault","fault":"fault","warning":"fault","complains":"fault",
       "jam":"jam","stuck":"jam","paper":"jam","pages":"jam",
       "toner":"ink","ink":"ink","cartridges":"ink",
       "clearing":"repair","resetting":"repair"}

toks = lambda s: s.lower().split()
dtoks = [toks(d) for d in docs]; N = len(docs)
df = Counter(t for d in dtoks for t in set(d))
avgdl = sum(map(len, dtoks)) / N

def bm25(q, k1=1.5, b=0.75):                      # scores exact terms, IDF-weighted
    out = []
    for i, d in enumerate(dtoks):
        tf, s = Counter(d), 0.0
        for t in toks(q):
            if tf[t] == 0: continue
            idf = np.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)  # rare term -> big weight
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        if s > 0: out.append((i, s))                # real engines return only matches
    return sorted(out, key=lambda x: -x[1])
CONC = sorted(set(SYN.values()))
def embed(s):                                      # toy encoder: bag of concepts
    v = np.zeros(len(CONC))
    for t in toks(s):
        if t in SYN: v[CONC.index(SYN[t])] += 1   # rare codes are OOV -> dropped
    n = np.linalg.norm(v); return v / n if n else v

D = np.stack([embed(d) for d in docs])
def dense(q):
    sims = D @ embed(q); return [(i, sims[i]) for i in np.argsort(-sims) if sims[i] > 0]
def rrf(lists, k=60):                              # fuse by rank, never by score
    sc = Counter()
    for lst in lists:
        for r, (i, _) in enumerate(lst): sc[i] += 1.0 / (k + r + 1)
    return sc.most_common()

for q, rel in [("error e-4471", 0), ("device complains it is short of ink", 1)]:
    b, dn = bm25(q), dense(q)
    top = lambda l: "none" if not l else "doc%d %s" % (l[0][0], "HIT" if l[0][0] == rel else "MISS")
    print("query: %-38s relevant: doc%d" % (q, rel))
    print("  bm25: %-12s dense: %-12s fused: %s" % (top(b), top(dn), top(rrf([b, dn]))))`
  }
},

chunk: {
  engineering: {
    when: `Chunking is a decision you make the moment your corpus outgrows what you can stuff into
    context — Anthropic's contextual-retrieval post draws the line at roughly <b>200k tokens (~500
    pages)</b>: below that, skip RAG and put the whole corpus in the prompt. Above it, the cut you
    choose is the <b>highest-leverage, hardest-to-reverse</b> decision in the pipeline: changing it
    later means re-embedding everything, so it is coupled to your embedding version
    (<x-ref to="embed">2.1</x-ref>) and quietly becomes migration-shaped infrastructure.`,
    how: `Order of operations, cheapest first:
    <ul>
      <li><b>Cut on structure, not on counts.</b> Headings, paragraphs, list items, functions.
      Markdown and HTML hand you the boundaries for free; for code, tree-sitter hands you the AST.
      A recursive splitter that falls back paragraph → sentence → token is the honest baseline, and
      Chroma's evaluation found it competitive with far fancier methods when properly sized.</li>
      <li><b>Prepend a situating sentence</b> (contextual retrieval): ask a small model to say what
      this chunk is about given the whole document, and index chunk + context together. Anthropic
      reported a 35% cut in top-20 retrieval failure rate from this alone (5.7% → 3.7%), 49% with
      contextual BM25 added, at a one-time cost of $1.02 per million document tokens with prompt
      caching.</li>
      <li><b>Decouple the retrieval unit from the generation unit.</b> Parent-document retrieval:
      embed small chunks for precise matching, but hand the model the enclosing section. Small
      finds, big informs.</li>
      <li><b>Store the address</b> — source, section path, page — as metadata, or citations and
      filtered search (<x-ref to="hybrid">4.2</x-ref>) are impossible later.</li>
    </ul>`,
    where: `Index time, between parsing and embedding — which is exactly why it is neglected: it
    lives in the unglamorous ETL layer, not the model layer. Every downstream stage inherits its
    mistakes. The reranker cannot rank a fact that was severed in half; the generator in
    <x-ref to="rag">4.1</x-ref> cannot cite a chunk with no address.`,
    breaks: `<ul>
      <li><b>Boundary-severed facts.</b> The failure this lesson is named for: cause in one chunk,
      effect in the next, neither retrievable by a query that mentions both. Overlap only shrinks
      the window of loss — it never closes it (see the maths).</li>
      <li><b>Tables and PDFs.</b> Fixed windows shred table rows from their headers; a cell reading
      "12.4" is unfindable forever. Layout-aware parsing before chunking is not optional for
      documents that were ever paginated.</li>
      <li><b>Defaults nobody chose.</b> Chroma measured a popular default configuration — 800-token
      chunks with 400-token overlap — at below-average recall and the worst precision of everything
      they tested. Defaults are somebody else's experiment from somebody else's corpus.</li>
      <li><b>Overlap as storage tax.</b> 400 of every 800 tokens duplicated doubles your index for
      a marginal recall change; Chroma found removing overlap cost little. Budget overlap like the
      redundancy it is.</li>
      <li><b>Silent re-chunk drift.</b> Change the chunker and every stored ID and eval pair means
      something different. Version the chunker like you version the embedding model.</li>
    </ul>`
  },
  research: {
    evals: `The metric with authority here is <b>token-level recall@k</b>: what fraction of the gold
answer span's tokens arrived inside the retrieved chunks.
<ul>
  <li><b>Token recall@k</b> — span tokens retrieved; document-level recall is the trap, awarding
  a hit when the right document's wrong section came back.</li>
  <li><b>Token precision / IoU</b> — retrieved tokens that were wanted; optimised alone it drives
  chunk width toward tiny fragments that match precisely and inform nobody.</li>
  <li><b>Severance rate</b> — gold spans crossing a chunk boundary; computable only with span
  labels, which is why almost every corpus ships without anyone knowing theirs.</li>
  <li><b>Index inflation</b> — stored tokens divided by corpus tokens; overlap multiplies this
  quietly, and the bill arrives as embedding cost and slower search.</li>
</ul>
<pre>nothing trains — the objective is a corpus-specific sweep:
  maximise  token recall@k(w, o, boundary rule)
  s.t.      token precision ≥ floor,   stored tokens ≤ budget
  both recall and precision move as 1/w in opposite directions: the optimum is interior, so measure</pre>
Span recall fails to predict answer quality on its own because delivering the evidence is not
the same as the generator using it — position effects inside the prompt take their cut after the
chunker has done everything right.`,

    maths: `<b>When does a fixed window sever a fact?</b> Take windows of width <code>w</code>
    tokens with overlap <code>o</code>, so stride <code>t = w − o</code>. A fact spans
    <code>s ≤ w</code> contiguous tokens starting at position <code>u</code>. Some chunk contains
    it whole iff a window start lands close enough before it:
    <pre>intact  ⇔  ∃k :  k·t ≤ u  and  u + s ≤ k·t + w
        ⇔  u mod t ∈ {0, 1, …, w − s}

P(intact)      = min(1, (w − s + 1) / (w − o))     (u uniform)
P(always split) = max(0, 1 − (w − s + 1)/(w − o))</pre>
    Concretely: w = 500, o = 50, s = 120 gives P(always split) = 1 − 381/450 ≈ <b>15%</b> — one in
    seven such facts is unfindable at any k, while the overlap inflates storage by w/t ≈ 11%.
    Overlap trades index size for a linear reduction in severance probability; it reaches zero only
    at o ≥ s − 1, i.e. when you already know the longest fact you must preserve.
    <br><br>
    <b>The size tradeoff, formally.</b> Token-level precision of a retrieved chunk is bounded by
    <code>s/w</code>: precision decays as 1/w, which is why Chroma scores IoU and precision and why
    their best precision came from 200-token chunks. Recall pulls the other way twice: larger w
    means fewer boundaries (severance falls as 1/(w−o)), but a mean-pooled embedding dilutes —
    model the chunk vector as <code>e = (s·f + (w−s)·g)/w</code> for fact direction f and off-topic
    mass g, and the query–chunk similarity shrinks roughly with s/w. Two opposing 1/w effects give
    an interior optimum, which is corpus-dependent — hence: measure, never inherit.`,
    papers: `<ul>
      <li><b>Evaluating Chunking Strategies for Retrieval</b> (Smith &amp; Troynikov, Chroma
      technical report, July 2024). The field's first serious apples-to-apples comparison — token
      windows, recursive splitters, and three semantic chunkers, over five corpora, scored with
      token-level recall, precision and IoU rather than "did the right document come back". Spread
      between strategies: up to 9 points of recall. Their LLM-driven chunker had the best recall
      (91.9%), their ClusterSemanticChunker at 200 tokens the best precision/IoU — and well-tuned
      heuristic splitters sat close behind at a fraction of the cost. Sobering for semantic-chunking
      enthusiasm: embedding-discontinuity chunkers were not reliably better than a tuned recursive
      splitter.</li>
      <li><b>Introducing Contextual Retrieval</b> (Anthropic, 2024). Prepend a Claude-generated
      situating sentence to each chunk before embedding and BM25 indexing: −35% retrieval failures
      from contextual embeddings, −49% adding contextual BM25, −67% adding reranking (5.7% → 1.9%).
      The rare technique whose numbers survived independent replication attempts.</li>
      <li><b>Late Chunking</b> (Günther et al., Jina AI, https://arxiv.org/abs/2409.04701). Run the
      long-context embedding model over the <em>whole</em> document first, then pool token vectors
      per chunk afterwards. Each chunk's embedding has already attended to its neighbours, so the
      pronoun problem dissolves without generating any text. No training needed; a dedicated
      fine-tune helps further.</li>
      <li><b>RAPTOR</b> (Sarthi et al., https://arxiv.org/abs/2401.18059). Recursive clustering and
      summarization builds a tree, so retrieval can answer at the paragraph, section, or
      whole-document level of abstraction — the honest response to questions no single chunk can
      answer.</li>
    </ul>`,
    scratch: `An afternoon, in order:
    <ol>
      <li>Take 20 documents you actually care about. Hand-write 50 (query → answer-span) pairs
      where the span is a token range, not a document ID.</li>
      <li>Implement three chunkers: fixed token window, recursive paragraph/sentence splitter,
      and the recursive splitter plus a one-line situating prefix.</li>
      <li>Score token-level recall and precision at k, exactly as the Chroma report does
      (their harness is public — brandonstarxel/chunking_evaluation).</li>
      <li>Now sweep w from 100 to 800. Watch precision fall as 1/w while recall rises then flattens.
      You will trust this curve because you drew it, and distrust every default forever after.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Anthropic</b> published contextual retrieval with unusually complete numbers (35/49/67%
      failure-rate reductions; $1.02 per million document tokens with prompt caching) and the advice
      that below ~200k tokens you should skip retrieval entirely and use context. Prompt caching is
      what makes the technique economical: the full document sits cached while Claude annotates each
      chunk against it.</li>
      <li><b>Chroma</b> is a database vendor doing public retrieval science — the chunking report
      exists because their customers' failures were chunking failures wearing a database costume.</li>
      <li><b>Jina AI</b> ships late chunking in its embedding stack, betting that the chunking
      problem is partly an artifact of embedding chunks in isolation and belongs inside the encoder
      (<x-ref to="embed">2.1</x-ref>).</li>
      <li>Every serious enterprise document-QA deployment converges on the same unglamorous
      truth: the accuracy lives in the parsing-and-chunking layer and dies in PDF tables.</li>
    </ul>`,
    startups: `A whole category exists because chunking is downstream of parsing: Unstructured and
    a dozen document-AI startups sell structure recovery — headings, tables, reading order —
    precisely so chunking has real boundaries to cut on. The pattern to notice: nobody sells
    "better fixed windows"; the value is always in recovering the document's own structure, which
    the fixed window threw away. If your catalogue is PDFs, you are buying or building a parser
    before you get to choose a chunker at all.`,
    open_source: `<b>brandonstarxel/chunking_evaluation</b> (the Chroma report's harness) — start
    here; evaluating your own corpus beats reading anyone's benchmark. <b>LangChain text
    splitters</b> — the ubiquitous recursive splitters; fine as baselines, dangerous as unexamined
    defaults. <b>jina-ai/late-chunking</b> — reference implementation with the paper's
    experiments. <b>tree-sitter</b> — for code, the chunk boundaries are called functions, and the
    parser is sitting right there. <b>Unstructured</b> — layout-aware parsing for the PDF mines.`
  },
  next: {
    problems: `<ul>
      <li><b>Facts that span any cut.</b> Multi-hop answers assembled from chunks in different
      sections defeat every chunker; RAPTOR-style summary trees and agentic retrieval that reads
      whole documents (<x-ref to="rag">4.1</x-ref>) are competing answers, and neither is settled.</li>
      <li><b>Is chunking an artifact?</b> Late chunking argues the isolated-chunk embedding was the
      bug; long-context models argue retrieval granularity should shrink to "find the document, read
      it all". Chunking may end up an index-time compression detail rather than a semantic
      commitment — or context costs may keep it load-bearing for years. Genuinely open.</li>
      <li><b>Evaluation is still document-shaped.</b> Chroma had to build token-level relevance
      scoring because standard IR benchmarks only score whole-document retrieval. Public benchmarks
      with span-level labels across formats (tables, code, transcripts) barely exist.</li>
      <li><b>Structure recovery as the bottleneck.</b> The best chunker on recovered-clean text
      loses to a mediocre chunker on properly parsed tables. Parsing quality × chunking strategy is
      under-studied as a joint problem because the two communities barely talk.</li>
    </ul>`,
    watch: `Watch Chroma's research page; search arXiv cs.IR for "chunking" and "contextual chunk
    embeddings" quarterly — the volume is low enough that you can read all of it, which is rare and
    will not last. Each time long-context pricing drops, re-run the "should we even chunk?"
    calculation for your corpus; that answer has a half-life of about six months.`
  },
  code: {
    title: 'Retrieval failure caused purely by the cut',
    note: `Stdlib only, runs anywhere. One corpus, one query, one deliberately crude scorer — the
    only variable is where the text was cut. The fixed window severs the cause from "30 percent",
    and the fact becomes unfindable even though every word of it is in the index. Optimise the
    chunker before the model.`,
    lang: 'python',
    body: `DOC = (
    "Release notes, Q3 platform team.\\n\\n"
    "Search infrastructure. We rebuilt the index pipeline and moved candidate "
    "generation onto the new ANN service. The migration to the shard-local "
    "cache cut p99 latency by 30 percent across all regions.\\n\\n"
    "Billing. Invoices are now generated nightly instead of weekly. "
    "No customer action is required.\\n\\n"
    "Deprecations. The v1 REST endpoints will be removed next quarter."
)

QUERY = "what cut p99 latency by 30 percent"
KEY = ("cache", "30 percent")          # cause and effect: both must survive the cut

def fixed_chunks(text, width=100):     # the naive default: cut every 100 characters
    return [text[i:i + width] for i in range(0, len(text), width)]

def para_chunks(text):                 # structure-aware: cut on blank lines
    return [p.strip() for p in text.split("\\n\\n") if p.strip()]

def score(query, chunk):               # crude lexical retrieval: shared words
    q = set(query.lower().split())
    return len(q & set(chunk.lower().replace(".", " ").split()))

for name, chunks in [("fixed-100-chars", fixed_chunks(DOC)),
                     ("paragraph-aware", para_chunks(DOC))]:
    best = max(chunks, key=lambda c: score(QUERY, c))
    intact = all(k in best for k in KEY)
    print("--- " + name + "  (" + str(len(chunks)) + " chunks)")
    print("best chunk : " + repr(best))
    print("score      : " + str(score(QUERY, best)))
    print("cause+effect in one chunk: " + str(intact))
    print()

print("Same corpus, same query, same scorer. Only the cut changed.")`
  }
},

agent: {
  engineering: {
    when: `Reach for an agent when <b>you cannot enumerate the steps in advance</b> — the path depends
    on what each action reveals. Anthropic's "Building effective agents" post is blunt about this: most
    production wins come from <em>workflows</em> (LLM calls on predefined code paths — prompt chaining,
    routing, parallelization, orchestrator–workers, evaluator–optimizer), and you graduate to an agent,
    where the model directs its own process, only when the simpler pattern demonstrably fails. If you
    can draw the flowchart, write the flowchart.`,
    how: `The loop is ten lines: model emits a thought and a tool call, runtime executes it, the
    observation goes back into context, repeat until a final answer or a cap. Everything
    production-grade sits around the loop:
    <ul>
      <li><b>Tool design first.</b> Anthropic's "Writing effective tools for agents" guidance:
      consolidate — one <code>get_customer</code> returning structured data beats five field-level
      endpoints; namespace clearly (their evals found prefix- vs suffix-naming measurably changes
      tool-selection behaviour); phrase errors as instructions the model can act on.</li>
      <li><b>Context management.</b> Long runs outgrow the window. Compaction — summarize the
      trajectory, continue from the summary — is how Claude Code survives multi-hour sessions. What
      survives compaction is a design decision; it is the agent's equivalent of the
      <x-ref to="attention">attention budget</x-ref>.</li>
      <li><b>MCP for the tool surface.</b> The Model Context Protocol standardizes how models discover
      and call tools — one server per system instead of one integration per model×system pair.
      Open-sourced November 2024; over 10,000 active public MCP servers by Anthropic's December 2025
      count.</li>
      <li><b>Enforcement outside the model.</b> max_turns, token budgets, tool timeouts, idempotency
      keys on writes, a kill switch that needs no deploy. Prompts are advisory; caps are not.</li>
    </ul>`,
    where: `Coding is the beachhead because the environment pushes back: compilers, tests and linters
    are free verifiers, which is why agents got good there first (see <x-ref to="verify">5.2</x-ref>).
    Beyond that: research over live systems — where <x-ref to="rag">RAG</x-ref> becomes one tool among
    several rather than the whole architecture — operations runbooks, and customer-facing tasks with
    narrow, well-instrumented tool sets.`,
    breaks: `<ul>
      <li><b>Error compounding.</b> 95% per-step reliability is 36% over twenty steps. The fix is not
      a smarter model first — it is fewer steps, cheaper recovery, and verifiable checkpoints.</li>
      <li><b>Runaway loops.</b> An agent retrying a failing tool all night is the canonical incident.
      Budget enforcement must live in infrastructure the model cannot talk its way past.</li>
      <li><b>Context rot.</b> Stale observations from turn 3 steering turn 40. Compaction that keeps
      decisions and discards transcripts helps; stored full trajectories are what let you debug it.</li>
      <li><b>Tool sprawl.</b> Forty overlapping tools make the selection problem harder than the task.
      Prune ruthlessly; evaluate tool-choice accuracy as its own metric.</li>
      <li><b>Silent side effects.</b> A hallucinated argument to a write-capable tool is an incident,
      not a wrong answer. Dry-run modes and human confirmation on anything irreversible.</li>
    </ul>`
  },
  research: {
    evals: `Agents are judged on a ratio, not a rate: <b>cost per solved task</b> — success on a
held-out suite divided into the tokens and wall-clock it burned getting there.
<ul>
  <li><b>Task success rate</b> — end-state verified, not transcript vibes; benchmark suites are
  cleaner than production, with no broken tools, stale docs or ambiguous specs, so treat the
  number as a ceiling.</li>
  <li><b>Cost per solved task</b> — total spend over total solved; the distribution is the truth
  here, since a fat tail of runaway trajectories hides comfortably inside a mean.</li>
  <li><b>Recovery rate</b> — of runs hitting a tool error, how many still finish; per-step
  accuracy flatters agents that never learned to notice failure.</li>
  <li><b>Tool-selection accuracy</b> — right tool, right arguments; probed single-turn it looks
  fine, and degrades mid-trajectory once the context has accumulated forty turns of residue.</li>
</ul>
<pre>the loop is not trained here — the objective being engineered is:
  minimise  E[ tokens × price  per solved task ]
  s.t.      success ≥ target,   irreversible side-effects = 0,   turns ≤ cap
  levers: tool shape, what compaction keeps, where verification gates the loop — not the model</pre>
Offline suites fail to predict production because production pushes back: real environments
inject the flaky tool, the interrupting user and the moving deadline that no frozen benchmark
contains, and the ranking of scaffolds can invert under that pressure.`,

    maths: `An agent is a POMDP the model never sees whole. Environment state <code>sₜ</code> (the
    repo, the database, the world) is hidden; the model picks an action from its tool set plus a
    terminal answer, and receives an observation:
    <pre>aₜ ∈ A = {tool₁(args), …, toolₖ(args), respond}
oₜ ~ O(· | sₜ, aₜ)          # tool output, error text, file contents
bₜ = f(o₁..oₜ, a₁..aₜ)      # belief state ≈ the context window</pre>
    The context window <em>is</em> the belief state, and compaction is deliberate, lossy belief-state
    maintenance — which is why what you keep matters more than how much.
    <br><br>
    <b>Why long chains fail:</b> with independent per-step success probability p, the chance a run of
    n steps has no fatal error is pⁿ:
    <pre>p = 0.95, n = 10 :  0.95¹⁰ ≈ 0.60
p = 0.95, n = 20 :  0.95²⁰ ≈ 0.36
p = 0.99, n = 20 :  0.99²⁰ ≈ 0.82</pre>
    Two honest caveats. Errors are not independent — one bad observation poisons later decisions, so
    reality is often worse than pⁿ. And recovery breaks the gloom: an agent that detects failure from
    the observation and retries survives individual bad steps, which is why structured errors and
    verifiable checkpoints beat raw per-step accuracy. Formally this is the setting of
    <x-ref to="rl">reinforcement learning</x-ref> — trajectory-level reward, credit assignment across
    steps — and RL on tool-use trajectories is how frontier labs now train the policy.`,
    papers: `<ul>
      <li><b>ReAct</b> (Yao et al., 2022, <a href="https://arxiv.org/abs/2210.03629">arXiv
      2210.03629</a>) established the pattern: interleaving reasoning traces with actions beats either
      alone, reporting absolute success-rate gains of 34% on ALFWorld and 10% on WebShop over imitation
      and RL baselines. Every agent framework since is this loop with better plumbing.</li>
      <li><b>SWE-bench</b> (Jimenez et al., ICLR 2024, <a href="https://arxiv.org/abs/2310.06770">arXiv
      2310.06770</a>): 2,294 real GitHub issues; the best assisted model in the original paper resolved
      1.96% of them. <b>SWE-bench Verified</b> (August 2024) is the human-validated 500-problem subset
      that fixed unsolvable and under-specified tasks; it is the number everyone now quotes.</li>
      <li><b>SWE-agent</b> (Yang et al., 2024, <a href="https://arxiv.org/abs/2405.15793">arXiv
      2405.15793</a>) showed the <em>agent–computer interface</em> is a first-class design object: the
      same model with a better-shaped file viewer and edit commands solves far more issues. Tool design
      as a research result, not a taste question.</li>
      <li><b>Building effective agents</b> (Anthropic engineering, December 2024) — not a paper, but
      the field's most-cited taxonomy, and the advice nobody wants to hear: use the simplest thing
      that works.</li>
    </ul>`,
    scratch: `Build the loop before reading another framework announcement:
    <ol>
      <li>Write the runtime: a loop with max_turns, a tool registry, errors caught and returned as
      observations. This is the code sample below; it is an afternoon.</li>
      <li>Swap the scripted policy for a Claude API call with tool definitions. Watch real
      trajectories; every failure mode in the engineering tab appears within an hour.</li>
      <li>Log trajectories to disk, then break a tool on purpose and see whether the agent recovers,
      retries forever, or hallucinates the result. That experiment teaches more than any survey.</li>
      <li>Only then add memory, subagents or MCP — one at a time, each justified by a named failure
      in your logs.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Anthropic</b> ships Claude Code, the clearest production example of a single agent with
      good tools: one loop, file/search/bash tools, compaction for long sessions, permission gates on
      writes. On the benchmark arc that tracks this field: the original SWE-bench paper's best result
      was 1.96% of issues (Jimenez et al., 2023); Claude 3.5 Sonnet reported 49.0% on SWE-bench
      Verified (Anthropic, October 2024); Claude Opus 4.5 reported 80.9% (Anthropic, November 2025).
      Two years from research curiosity to shipped product category.</li>
      <li><b>Anthropic's multi-agent research system</b> (June 2025 engineering post): a lead Claude
      Opus 4 orchestrating parallel Claude Sonnet 4 subagents beat single-agent Claude Opus 4 by 90.2%
      on internal research evals — at roughly 15× the tokens of a chat, with token spend alone
      explaining 80% of performance variance. Multi-agent buys parallel exploration of a breadth-first
      problem, and you pay for every branch.</li>
      <li><b>MCP governance</b>: Anthropic donated the Model Context Protocol to the Agentic AI
      Foundation, a directed fund under the Linux Foundation, in December 2025 — the standard now
      outlives any one vendor, which is what makes enterprises willing to build on it.</li>
    </ul>`,
    startups: `The moat is rarely the loop, which is a commodity, and rarely the model, which is
    rented. It is the tool layer and the evaluation harness — deep, well-shaped tools over one
    domain's systems, plus trajectory data showing where runs fail. Coding-agent startups discovered
    this first; the same shape is repeating in legal, finance and ops. If your pitch is "an agent for
    X", the defensible part is the verifier and the tools for X, not the agent.`,
    open_source: `MCP's reference servers and SDKs (now under the Agentic AI Foundation) are the
    de-facto standard for the tool surface; the SWE-bench harnesses are the field's shared
    scoreboard. The honest catalogue advice: most agent frameworks are thin wrappers around a loop
    you can write yourself in fifty lines — read one, then write your own so you understand what you
    are debugging at 2am.`
  },
  next: {
    problems: `<ul>
      <li><b>Reliability over long horizons.</b> pⁿ is the enemy; checkpointing, recovery and
      self-verification are where the effort is going, because per-step model gains arrive slower
      than tasks grow.</li>
      <li><b>Context management as a discipline.</b> What to keep, compress and discard across
      thousand-turn trajectories is barely theorized — today it is folklore in compaction prompts.</li>
      <li><b>Evaluation.</b> Benchmarks measure completion; production cares about cost, latency,
      side-effect safety and behaviour under broken tools. Trajectory-level evals are immature
      everywhere.</li>
      <li><b>Security.</b> Any text a tool returns is untrusted input steering a system with write
      access. Least-privilege scoping is necessary and known to be insufficient.</li>
      <li><b>Learning from experience.</b> Agents restart from zero each run. Turning stored
      trajectories into improved policies — <x-ref to="rl">RL</x-ref> on your own traces — is the
      obvious next step and operationally hard.</li>
    </ul>`,
    watch: `Follow Anthropic's engineering blog — the agents, tools and context-management posts are
    the field's working notes — the MCP specification changelog under the Agentic AI Foundation, and
    the SWE-bench Verified leaderboard: not for the headline number but for how fast it moves, which
    tells you how quickly today's architecture advice goes stale.`
  },
  code: {
    title: 'The agent loop, with the model mocked out',
    note: `Runnable as-is on CPU, stdlib only, no network. The policy is a script standing in for the
    model so the control flow — think, act, observe, repeat, cap — is the visible lesson. Swapping the
    mock for a real Claude API call is a three-line change to policy(); the runtime — loop, caps,
    error handling — does not change, which is precisely the point: that part is yours either way.`,
    lang: 'python',
    body: `import ast, operator

# --- real tools: deterministic work lives here, never in the model ------
OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv}
def calc(expr):
    def ev(n):
        if isinstance(n, ast.BinOp): return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Constant): return n.value
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)

FILES = {"notes.txt": "invoice total is 1187.50 across 5 items"}
def read_file(name): return FILES.get(name, "ERROR: no such file. Try list_files.")
def list_files(_=""): return ", ".join(sorted(FILES))

TOOLS = {"calc": calc, "read_file": read_file, "list_files": list_files}

# --- mock policy: a script standing in for the model --------------------
SCRIPT = [
    ("I need the invoice data before I can compute anything.", "list_files", ""),
    ("notes.txt looks right; read it.", "read_file", "notes.txt"),
    ("1187.50 over 5 items. Division is exact -> tool, not me.",
     "calc", "1187.50 / 5"),
    ("I have what I need; stop.", "final", "Each item costs 237.50."),
]
def policy(history, turn):
    # swap these lines for a Claude API call; nothing else changes
    return SCRIPT[min(turn, len(SCRIPT) - 1)]

def run(task, max_turns=8):                       # the cap is the seatbelt
    history = [("task", task)]
    for turn in range(max_turns):
        thought, action, arg = policy(history, turn)
        print("THINK ", thought)
        if action == "final":
            print("FINAL ", arg)
            return arg
        try:
            obs = str(TOOLS[action](arg))         # errors become observations,
        except Exception as e:                    # never uncaught exceptions
            obs = "ERROR: " + repr(e)
        print("ACT   ", action + "(" + repr(arg) + ")")
        print("OBS   ", obs, "\\n")
        history.append((action, obs))
    print("STOP   max_turns reached")             # enforcement, not advice

run("What is the per-item cost on the invoice?")`
  }
},

verify: {
  engineering: {
    when: `Reach for an LLM judge when the output is <b>not mechanically checkable</b> — tone, faithfulness,
    helpfulness, "did the agent actually do the task". If a unit test, a schema validator, or a string
    match can decide the question, use that instead: it is cheaper, deterministic, and cannot be
    flattered. The whole idea rests on the <b>generator–verifier gap</b> — checking an answer is usually
    easier than producing it — and when that gap closes (the judge is no better at the task than the
    writer), the second opinion is theatre. The checker must be at least as capable as the writer.`,
    how: `Decisions that matter, in order:
    <ul>
      <li><b>Pairwise vs pointwise.</b> Pairwise ("which of these two is better?") is far more reliable
      than absolute scores — models are bad at calibrated 1–10 scales and good at comparisons. Use
      pointwise only when you need a threshold, and then prefer a rubric with named criteria over a
      bare number.</li>
      <li><b>Swap the order.</b> Judges favour the first (sometimes last) position. Run every pairwise
      comparison both ways and treat disagreement between the two orders as a tie, exactly as Zheng et
      al. do for MT-Bench. This is one extra call and non-negotiable.</li>
      <li><b>Different judge than generator.</b> Panickssery et al. (NeurIPS 2024) showed evaluators
      recognize their own outputs and score them higher. Judge with a different model, or at minimum a
      different prompt persona.</li>
      <li><b>Validate before trusting.</b> Label ~100 examples by hand, measure agreement with Cohen's
      kappa, not raw percent. An unvalidated judge is a confident random number generator. This site's
      own server.py is the small-scale version: written answers go to a Claude model with an explicit
      rubric, and the browser's keyword matcher is the fallback — two graders, one of them mechanical.</li>
    </ul>`,
    where: `Four places, increasingly load-bearing: <b>offline evals</b> (regression suites in CI, gating
    prompt and model changes); <b>online guardrails</b> (a verification pass over claims before the answer
    ships, which is the faithfulness check <x-ref to="rag">4.1</x-ref> needs); <b>inside agent loops</b>
    (<x-ref to="agent">5.1</x-ref> — checking a step before acting on it, where a wrong "looks fine"
    compounds); and as the <b>reward signal in RL</b> (<x-ref to="rl">6.1</x-ref> — the most dangerous
    seat, because there the judge is optimised against, not merely consulted).`,
    breaks: `<ul>
      <li><b>Position bias.</b> The ranking flips when you swap which answer appears first. Documented in
      Zheng et al.; cured by swap-and-aggregate, and by nothing else.</li>
      <li><b>Verbosity bias.</b> Longer answers score higher at equal quality. Swapping does not touch
      this — you need the rubric to say so explicitly, or a length-controlled comparison.</li>
      <li><b>Self-preference.</b> A model judging its own family's output inflates it, and the effect
      scales with how well it recognizes its own text (Panickssery et al.).</li>
      <li><b>Goodhart under optimisation.</b> Consult a judge and its biases are noise; train against it
      and they are gradients. 2025 studies of rubric-based RL show judge scores climbing while held-out
      accuracy peaks early and falls — models learn markup tricks and confident phrasing that fool the
      grader. Any judge in an RL loop needs a held-out check it never sees.</li>
      <li><b>Judge drift.</b> Upgrade the judge model and every historical score changes meaning. Version
      the judge like a schema; re-baseline on the same labelled set after every change.</li>
    </ul>`
  },
  research: {
    evals: `A judge earns trust through one number: <b>Cohen's kappa against human labels on your
own task</b> — remeasured every time the judge, the rubric or the traffic changes.
<ul>
  <li><b>Kappa vs human labels</b> — chance-corrected agreement; raw percent agreement is the
  trap, looking superb whenever one label dominates, which in production evals it always does.</li>
  <li><b>Position-flip rate</b> — pairwise verdicts that reverse when slots swap; measured once
  and forgotten it silently regrows with every judge-model upgrade.</li>
  <li><b>False-pass rate at the gate threshold</b> — bad outputs the judge waves through; tuning
  the threshold for a comfortable pass rate optimises for exactly the wrong tail.</li>
  <li><b>Score–length correlation</b> — the cheapest bias audit you can run; a judge whose scores
  track answer length is grading effort, and effort is trivially gamed.</li>
</ul>
<pre>reward model (Bradley–Terry on preference pairs):  L = −E log σ( r_φ(x, y_w) − r_φ(x, y_l) )
process reward model: step-level CE against per-step correctness labels
prompted judge: no parameters train — validation against held-out human labels is the entire
                warrant for every downstream decision it makes</pre>
Offline agreement fails to predict behaviour under optimisation: the moment a judge's score
becomes a training signal or a gate, systems drift toward its blind spots, and the kappa you
measured on yesterday's distribution stops covering you.`,

    maths: `<b>Why majority voting works — and when it doesn't.</b> For n independent judges each correct
    with probability p, the majority is wrong with probability
    <pre>P(err) = Σ_{k > n/2} C(n,k) (1−p)^k p^(n−k)</pre>
    Condorcet's jury theorem: if p &gt; 0.5 this falls monotonically as n grows. Worked example, p = 0.7:
    <pre>n = 1:  accuracy 0.700
n = 3:  3(0.3)²(0.7) + (0.3)³ = 0.216 wrong  →  0.784
n = 5:  10p³q² + 5p⁴q + p⁵                  →  0.837</pre>
    Three mediocre judges beat one, five beat three. The catch is the word <em>independent</em>: three
    samples from the same model share the same verbosity and position biases, so their errors correlate
    and the effective n is much smaller than 3. In the limit of perfectly correlated judges, majority
    voting buys exactly nothing — you get p back. Self-consistency works as well as it does because
    sampled <em>reasoning paths</em> diverge even when the model is fixed; judge ensembles from one model
    mostly don't.
    <br><br>
    <b>Why raw agreement misleads.</b> Cohen's kappa corrects observed agreement p_o for the agreement
    p_e two annotators would reach by chance given their label distributions:
    <pre>κ = (p_o − p_e) / (1 − p_e),   p_e = Σ_c p₁(c)·p₂(c)</pre>
    Worked example: your judge and your human both say "pass" 90% of the time. Chance agreement is
    0.9·0.9 + 0.1·0.1 = 0.82. If observed agreement is 0.85 — which sounds excellent — then
    κ = 0.03/0.18 ≈ 0.17: barely better than chance. On skewed label distributions, which is what
    production evals always have, percent agreement is close to meaningless. Report kappa.`,
    papers: `<ul>
      <li><b>Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena</b> (Zheng et al., NeurIPS 2023,
      https://arxiv.org/abs/2306.05685) is the paper to read first. Strong LLM judges reached over 80%
      agreement with human preferences — the same level as human–human agreement — and the paper names
      the failure modes everyone now cites: position, verbosity and self-enhancement bias. It released
      80 questions, 3K expert votes and 30K human-preference conversations.</li>
      <li><b>Self-Consistency</b> (Wang et al., https://arxiv.org/abs/2203.11171): sample several
      reasoning chains, majority-vote the answers. Reported +17.9% on GSM8K, +11.0% on SVAMP, +12.2% on
      AQuA over greedy chain-of-thought. Verification without any judge at all — agreement as evidence.</li>
      <li><b>Let's Verify Step by Step</b> (Lightman et al., https://arxiv.org/abs/2305.20050): grade the
      <em>process</em>, not just the outcome. Their process-supervised reward model solved 78% of a MATH
      test subset, beating outcome supervision, and shipped PRM800K — 800K step-level human labels.
      Process reward models are what turned verification into a search signal.</li>
      <li><b>Constitutional AI</b> (Bai et al., https://arxiv.org/abs/2212.08073): verification pushed
      into training time — the model critiques and revises its own outputs against written principles,
      and an AI preference model replaces most human harmlessness labels. A rubric, applied at scale,
      before deployment rather than after.</li>
      <li><b>LLM Evaluators Recognize and Favor Their Own Generations</b> (Panickssery et al.,
      https://arxiv.org/abs/2404.13076): self-preference measured properly, with the linear link to
      self-recognition.</li>
    </ul>`,
    scratch: `An afternoon, no GPU, and it will change how you argue about evals:
    <ol>
      <li>Take 50 question–answer pairs you can label yourself. Label them.</li>
      <li>Write a pointwise judge prompt and a pairwise one. Run both, both orders.</li>
      <li>Compute percent agreement <em>and</em> kappa against your labels. Watch them disagree.</li>
      <li>Count position flips: how often does the pairwise verdict change when you swap slots? If it is
      over a few percent, you have measured the bias yourself and will never skip the swap again.</li>
      <li>Majority-vote three sampled judgements. Note how little it helps, and why: correlated
      errors. You have re-derived the Condorcet caveat empirically.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Anthropic</b> runs verification at training time: Constitutional AI's critique-revise-RLAIF
      loop is a judge inside the training pipeline, and the published principles are the rubric. The
      2025–26 shift extends this: rubric-based grading as RL reward for domains without mechanical
      checkers, next to verifiable-rewards RL (math and code, where the checker is a test harness and
      cannot be flattered).</li>
      <li><b>LMSYS / LMArena</b> is the field's human-preference ground truth: crowdsourced pairwise
      battles, Elo-style aggregation. Its main use for practitioners is calibration — it is the human
      label set that judge agreement numbers like Zheng et al.'s 80% are measured against.</li>
      <li><b>Every serious agent product</b> ships some claim-level verification pass — the
      decompose-adjudicate-delete loop from this lesson — because it is the cheapest fix for the
      hallucinated-citation failure that kills trust fastest.</li>
    </ul>`,
    startups: `Eval infrastructure became its own product category: Braintrust, LangSmith, Humanloop,
    Arize Phoenix all sell versions of judge-plus-labelled-baseline-plus-regression-dashboard. The
    pattern worth noticing: the moat is not the judge prompt, it is the accumulated human-labelled
    preference data that keeps the judge calibrated. If you build here, the calibrated domain-specific
    judge — legal, medical, financial — is the defensible piece; the generic scoring harness is a
    weekend project.`,
    open_source: `<b>FastChat</b> (LMSYS) ships the original MT-Bench judge prompts — read them; they are
    shorter than you expect. <b>promptfoo</b> and <b>DeepEval</b> are practical harnesses for judge-based
    regression testing in CI. <b>Ragas</b> implements faithfulness and answer-relevance metrics for
    retrieval pipelines. <b>PRM800K</b> (from Lightman et al.) is the
    dataset to start from if you want to train a process reward model rather than prompt a judge.`
  },
  next: {
    problems: `<ul>
      <li><b>The verifier gap governs RL.</b> You can only train with RL on what you can verify at scale,
      so capabilities advance fastest where checking is mechanical (code, math) and stall where it isn't.
      Closing the gap for open-ended domains — rubrics, debate, process rewards — is arguably the central
      open problem in post-training right now (<x-ref to="rl">6.1</x-ref>).</li>
      <li><b>Judges under adversarial pressure.</b> Self-play work like "More Convincing, Not More
      Correct" shows reference-free judges rewarding persuasive-sounding text over correct text, and
      scoring ensembles failing to survive adversarial optimisation. A judge that is fine as a metric can
      be catastrophic as a target.</li>
      <li><b>Who verifies the verifier?</b> Validating judges needs human labels; scaling oversight past
      what humans can label — recursive reward modelling, debate — is unsolved, and every scheme bottoms
      out somewhere.</li>
    </ul>`,
    watch: `Watch three threads: rubric-as-reward papers in RL post-training, process-reward-model scaling
    for verifier-guided search, and the steady stream of judge-bias audits. Re-run your own judge-vs-human
    kappa quarterly — the field's numbers age faster than the papers do, and so do yours.`
  },
  code: {
    title: 'A biased judge, measured and repaired',
    note: `Stdlib only, runs on CPU in a second. The simulated judge has a position bias and a verbosity
    bias, like the real ones documented in Zheng et al. Watch the ranking flip with slot order, then watch
    which countermeasure fixes which bias — swapping cures position, only the rubric cures verbosity, and
    majority voting cannot remove a bias every vote shares.`,
    lang: 'python',
    body: `import random
random.seed(7)
TRIALS = 2000

A = {"q": 0.72, "len": 120}   # short answer, genuinely better
B = {"q": 0.58, "len": 600}   # verbose answer, worse

def score(ans, slot, length_blind=False):
    s = ans["q"] + random.gauss(0, 0.10)     # noisy read of true quality
    if not length_blind:
        s += 0.0001 * ans["len"]             # verbosity bias: long reads as good
    if slot == 1:
        s += 0.15                            # position bias: first slot favoured
    return s

def duel(first, second, **kw):               # 1 = first wins, 2 = second wins
    return 1 if score(first, 1, **kw) >= score(second, 2, **kw) else 2

def rate(f):
    return sum(f() for _ in range(TRIALS)) / TRIALS

# 1. one call, fixed order: the verdict depends on the slot, not the answer
print("A first :  A wins %.0f%%" % (100 * rate(lambda: duel(A, B) == 1)))
print("A second:  A wins %.0f%%   <- same judge, ranking flipped by order"
      % (100 * rate(lambda: duel(B, A) == 2)))

# 2. swap protocol: ask both orders, keep only verdicts that agree
def swap_verdict(**kw):
    v1, v2 = duel(A, B, **kw), duel(B, A, **kw)
    if v1 == 1 and v2 == 2: return "A"
    if v1 == 2 and v2 == 1: return "B"
    return None                              # contradiction = position artefact
vs = [swap_verdict() for _ in range(TRIALS)]
dec = [v for v in vs if v]
print("swap-consistent:  A wins %.0f%% of decisive verdicts (%.0f%% decisive)"
      % (100 * dec.count("A") / len(dec), 100 * len(dec) / TRIALS))

# 3. majority of 3 votes, order randomised per vote
def vote(**kw):
    return duel(A, B, **kw) == 1 if random.random() < 0.5 else duel(B, A, **kw) == 2
print("majority-of-3, random order:  A wins %.0f%%   <- shared verbosity bias remains"
      % (100 * rate(lambda: sum(vote() for _ in range(3)) >= 2)))

# 4. one countermeasure per bias: swap kills position, rubric kills verbosity
fixed = [swap_verdict(length_blind=True) for _ in range(TRIALS)]
fdec = [v for v in fixed if v]
print("swap + length-blind rubric:  A wins %.0f%% of decisive verdicts"
      % (100 * fdec.count("A") / len(fdec)))
print("each bias needs its own countermeasure; averaging biased calls is not one")`
  }
},

rl: {
  engineering: {
    when: `Reach for RL when <b>the training signal is a consequence, not a label</b> — a click, a
    passing test suite, a thumbs-up — and when acting changes what data you see next. If a supervised
    label exists, use it; supervised learning is cheaper, stabler, and easier to debug. The honest
    hierarchy in production: bandits are routine, off-policy learning from logs is common, full
    online RL with long horizons is rare and usually a sign someone is showing off.`,
    how: `Two shipping patterns, in ascending order of pain:
    <ul>
      <li><b>Contextual bandits</b> for slots where the action doesn't change the world much:
      which artwork, which headline, which module ordering on the page.
      One decision, one reward, no credit assignment problem. Start here.</li>
      <li><b>Off-policy learning from logs.</b> You already log what you showed and what happened.
      Importance-weight by the logged propensities and you can evaluate or train a new policy without
      serving it — which is why <em>logging propensities</em> is the single highest-leverage
      infrastructure decision in this whole area. No propensities, no off-policy anything.</li>
    </ul>
    The third pattern — optimising language models against human or verifier judgement — changed
    enough hands and enough machinery to earn its own lesson: <x-ref to="rlhf">6.2</x-ref>.`,
    where: `Recommenders (explore/exploit on fresh items — the same cold-start pressure that motivates
    <x-ref to="sid">semantic IDs</x-ref>), ad and layout optimisation, and LLM post-training. Also the
    tuning layer for agents: an <x-ref to="agent">agent loop</x-ref> whose trajectories end in a
    checkable outcome is exactly the shape GRPO wants.`,
    breaks: `<ul>
      <li><b>Reward hacking.</b> The policy optimises what you measure, not what you meant.
      Anthropic's reward-tampering paper (Denison et al., 2024) showed models trained on gameable
      environments generalize zero-shot from sycophancy to editing their own reward function.
      Assume your proxy will be exploited; the only question is how visibly.</li>
      <li><b>Feedback loops.</b> The policy changes the logs, the logs train the policy. Clicks drift
      toward whatever the policy already shows. Reserve a small uniform-random slice of traffic
      forever; it is your only unbiased measurement instrument.</li>
      <li><b>Importance weights blow up.</b> When the new policy likes actions the old one rarely
      took, a handful of logged events dominate the estimate. Clip the weights and report effective
      sample size, not just the point estimate.</li>
    </ul>`
  },
  research: {
    evals: `What ships is decided by <b>estimated policy value with a confidence interval</b>, computed
from logs the old policy wrote — the whole discipline of this lesson in one number.
<ul>
  <li><b>IPS / SNIPS / DR value ± CI</b> — counterfactual value from logs; the trap is that a
  few low-propensity rows can carry the estimate, so the interval without the effective sample
  size is an incomplete sentence.</li>
  <li><b>Cumulative regret in simulation</b> — algorithm quality where truth is known; you chose
  the world, and a forgiving one certifies nothing about the real one.</li>
  <li><b>Max importance weight and weight tail</b> — the concentration diagnostic; one enormous
  weight means the estimate belongs to one logged row, whatever the mean says.</li>
</ul>
<pre>The quantities being engineered, not trained:
  regret R(T)         — minimised by the exploration policy
  V̂(π) ± CI            — the shipping decision, from IPS / SNIPS / DR above
  ESS = (Σw)²/Σw²      — the data you actually have after reweighting</pre>
The offline→online gap is the field's founding problem: deploying the policy changes the data
distribution the logs came from, so every logged estimate is a statement about a world the
deployment itself retires.`,

    maths: `<b>Bandits.</b> With arm means μ₁..μ_K and μ* = maxₐ μₐ, regret after T pulls is
    <pre>R(T) = T·μ*  −  E[ Σₜ μ_{aₜ} ]</pre>
    A fixed A/B split pulls bad arms at a constant rate, so R(T) grows <em>linearly</em>. UCB1
    (Auer et al., 2002) pulls the arm maximising
    <pre>x̄ₐ + √( 2 ln t / nₐ )</pre>
    — empirical mean plus an uncertainty bonus that shrinks as an arm is tried. The bonus is a
    confidence bound: with high probability the true mean lies below it, so a bad arm can only be
    pulled while its bound still overlaps the best arm's, which happens O(ln T / Δₐ²) times.
    Regret is logarithmic; the gap between ln T and T is the entire economic argument for bandits.
    <br><br>
    <b>Off-policy evaluation.</b> With logs (xᵢ, aᵢ, pᵢ, rᵢ) written by policy μ, where pᵢ = μ(aᵢ|xᵢ)
    is the recorded propensity, the value of a new policy π is estimated by
    <pre>IPS:    V̂(π) = (1/n) Σᵢ  wᵢ · rᵢ,          wᵢ = π(aᵢ|xᵢ) / pᵢ
SNIPS:  V̂(π) = Σᵢ wᵢ rᵢ / Σᵢ wᵢ              (self-normalised — biased, far lower variance)
DR:     V̂(π) = (1/n) Σᵢ [ q̂(xᵢ, π) + wᵢ·(rᵢ − q̂(xᵢ, aᵢ)) ]</pre>
    IPS is unbiased exactly when the propensities are correct, and its variance is governed by the
    weights: E[w²] blows up wherever π likes what μ avoided. Doubly robust (Dudík et al., 2011) adds
    a reward model q̂ and stays unbiased if <em>either</em> the propensities or q̂ are right — the model
    absorbs variance, the correction term removes its bias. Effective sample size
    <pre>ESS = (Σ wᵢ)² / Σ wᵢ²</pre>
    is the honest measure of how much data you really have after reweighting.`,
    papers: `<ul>
      <li><b>A Contextual-Bandit Approach to Personalized News Article Recommendation</b>
      (Li et al., WWW 2010). LinUCB on Yahoo! front-page news: a 12.5% click lift over a
      context-free bandit on 33M+ events, with the gap widening as data got scarcer. Still the
      cleanest production-bandit paper to read first.</li>
      <li><b>Top-K Off-Policy Correction for a REINFORCE Recommender</b> (Chen et al., 2019) and
      <b>Doubly Robust Policy Evaluation</b> (Dudík et al., 2011) — the ranking↔RL bridge and the
      estimator that makes logged evaluation trustworthy.</li>
    </ul>`,
    scratch: `Buildable in a weekend, in order:
    <ol>
      <li>Simulate a 5-arm CTR problem and implement epsilon-greedy, UCB1, and a fixed split.
      Plot cumulative regret. The linear-vs-logarithmic picture is the field's founding image.</li>
      <li>Log (context, action, propensity, reward) from one policy; estimate another policy's value
      with inverse propensity scoring. Watch the variance explode as the policies diverge — now you
      understand why doubly robust exists.</li>
      <li>REINFORCE on a toy task with and without a baseline. The variance difference is not subtle,
      and it is the same trick <x-ref to="rlhf">6.2</x-ref> plays at scale.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Yahoo!</b> ran the landmark bandit deployment (Li et al., 2010, above). <b>Netflix</b>
      described contextual bandits choosing per-user title artwork on its tech blog — the canonical
      "one decision, immediate reward" slot. <b>Spotify</b> published bandit-based playlist and
      explanation ranking ("Explore, Exploit, and Explain", McInerney et al., RecSys 2018).</li>
      <li><b>YouTube</b> shipped Top-K off-policy REINFORCE inside a production recommender
      (Chen et al., 2019): REINFORCE with clipped importance weights over logged traffic, plus a
      correction for recommending a slate of K — RL made deployable precisely by refusing to
      explore online. Connects directly to <x-ref to="seqrec">sequential recommendation</x-ref>.</li>
    </ul>`,
    startups: `Most teams still run A/B tests where a bandit would pay for itself — regret is a real
    cost that never appears on a dashboard, which is why nobody is accountable for it. The teams that
    do adopt bandits usually arrive via the logging side: once propensities are recorded for
    evaluation, exploration policies become nearly free to try, which is the quiet argument for
    building the log schema first and the algorithm second.`,
    open_source: `<b>Vowpal Wabbit</b> has carried production contextual bandits (including doubly
    robust OPE) for over a decade and remains the fastest way to ship one. <b>Open Bandit Pipeline</b>
    (ZOZO) is the reference for off-policy estimators and pairs with the only large public dataset
    that logs true propensities — the capstone built on this lesson uses both.`
  },
  next: {
    problems: `<ul>
      <li><b>Off-policy evaluation people believe.</b> IPS and doubly robust are unbiased in theory
      and high-variance in practice; A/B tests still overrule them. Confidence intervals tight
      enough to skip the experiment would change how the industry ships.</li>
      <li><b>Exploration under non-stationarity.</b> Real reward distributions drift — items age,
      trends pass, seasons turn — and the theory mostly assumes they don't. Discounted and
      sliding-window bandits exist, but choosing their forgetting rate is itself an
      explore/exploit problem nobody has closed.</li>
      <li><b>Long-horizon value in recommenders.</b> A click today versus a subscriber next year:
      the reward that matters most arrives too late to attribute. Surrogate objectives and
      progressively longer-horizon experiments are the current practice, not a solution.</li>
    </ul>`,
    watch: `In recsys, watch the bandit and off-policy tracks at RecSys and KDD — the tooling is
    finally commodity, so adoption is the story now. For the preference-training frontier, the
    watching guide lives in <x-ref to="rlhf">6.2</x-ref>.`
  },
  code: {
    title: 'The cost of not adapting: A/B split vs epsilon-greedy vs UCB1',
    note: `Runnable as-is on CPU, numpy only. Five ads, unknown CTRs, 50k impressions. The fixed A/B
    split sends 20% of traffic to every arm forever; the bandits stop paying for what they already
    know. Cumulative regret — clicks left on the table versus always showing the best ad — is the
    number a dashboard never shows you. The second lesson is free: textbook UCB1 over-explores on
    tiny CTRs, and the fix is the same exploration knob Li et al. tuned at Yahoo!.`,
    lang: 'python',
    body: `import numpy as np

rng = np.random.default_rng(0)
p = np.array([0.020, 0.024, 0.031, 0.018, 0.042])   # true CTRs; arm 4 is best
T, K = 50_000, len(p)
best = p.max()

def run(policy):
    n, s, regret = np.zeros(K), np.zeros(K), 0.0
    for t in range(1, T + 1):
        a = policy(t, n, s)
        s[a] += rng.random() < p[a]                  # Bernoulli click
        n[a] += 1
        regret += best - p[a]                        # expected clicks lost
    return regret, n

def ab_split(t, n, s):
    return t % K                                     # uniform forever, decide never

def eps_greedy(t, n, s, eps=0.1):
    if n.min() == 0 or rng.random() < eps:
        return int(rng.integers(K))
    return int(np.argmax(s / n))

def ucb(c):                                          # bonus scale c is the knob
    def pol(t, n, s):
        if n.min() == 0:
            return int(np.argmin(n))                 # try everything once
        return int(np.argmax(s / n + np.sqrt(c * np.log(t) / n)))
    return pol

policies = [("A/B split", ab_split), ("eps-greedy 10%", eps_greedy),
            ("UCB1 (textbook c=2)", ucb(2.0)), ("UCB (tuned c=0.02)", ucb(0.02))]
print(f"{'policy':<20} {'regret (clicks lost)':>21} {'traffic to best arm':>21}")
for name, pol in policies:
    reg, n = run(pol)
    print(f"{name:<20} {reg:>21.1f} {n[K-1] / T:>20.1%}")

# Two lessons in one table. The fixed split's regret grows linearly with T
# (re-run with T=500_000: its loss scales 10x, the tuned bandit's barely moves).
# And textbook UCB1 over-explores here: its Hoeffding bonus assumes rewards
# span [0,1], but CTRs live near 0.03 — every production bandit tunes c.`
  }
},

rlhf: {
  engineering: {
    when: `Reach for preference training when <b>quality is judgeable but not specifiable</b> — no
    label or metric captures "a good answer", yet a person (or a checker) can reliably compare two
    candidates. In practice that means the post-training stage of language models, and increasingly
    the finishing stage of <x-ref to="agent">agents</x-ref> whose trajectories end in something
    checkable. If a supervised label or a deterministic metric exists, use it first; preference
    machinery is what you pay for when it doesn't.`,
    how: `The 2026 stack, in the order you should try it:
    <ul>
      <li><b>DPO for broad preference shaping.</b> Offline pairs, a classification loss, no reward
      model in memory, no sampling loop — it runs like supervised fine-tuning and fails like it too,
      which is a compliment. Most teams should stop here.</li>
      <li><b>GRPO against verifiable rewards</b> where a checker exists — unit tests, maths answers,
      compilable code — and you can afford generation-time compute. Sampling several completions per
      prompt is the cost; deleting the value network and the gameable judge is what it buys.</li>
      <li><b>PPO with a learned reward model</b> is now the expensive path you justify, not the
      default: online sampling, a critic to tune, and a judge that can be flattered. You take it when
      the signal genuinely must be a learned model of human taste, at scale.</li>
    </ul>`,
    where: `The last stage of every serious assistant's training, after pretraining
    (<x-ref to="attention">3.1</x-ref>) and supervised fine-tuning. The same machinery is spreading
    to anything with a checkable outcome: retrieval-augmented answering graded for faithfulness,
    agents graded by task completion, code models graded by test suites.`,
    breaks: `<ul>
      <li><b>KL drift.</b> Loosen the KL penalty and the model finds the reward model's blind
      spots; reward goes up while actual quality falls. Track KL-to-reference as a first-class metric,
      and spot-check with evals the reward model never saw — the same discipline as
      <x-ref to="verify">verification</x-ref>.</li>
      <li><b>Preference data rot.</b> Comparisons collected under one distribution of model outputs
      stop discriminating once the model improves past them; a reward model trained on last quarter's
      failures grades this quarter's model on the wrong axis. Refresh pairs from the current policy's
      own outputs.</li>
      <li><b>Judge–generator kinship.</b> Grade a model with a judge that shares its lineage and you
      inherit self-preference bias; the fix is the verification lesson's rule — different judge than
      generator, rubric over vibes.</li>
      <li><b>Length and style capture.</b> Reward models overweight verbosity and confident tone.
      If your win rate rises while answers get longer, you may have trained an essayist, not an
      assistant. Regress reward against length before believing it.</li>
    </ul>`
  },
  research: {
    evals: `What ships is decided by <b>held-out win rate at a controlled KL budget</b> — preference
for the tuned model on prompts nothing in the training signal ever touched.
<ul>
  <li><b>Held-out win rate</b> — preference of the tuned model over reference on prompts the
  reward signal never saw; win rate on prompts the reward model scored is self-grading.</li>
  <li><b>KL to reference, sliced</b> — the leash; a modest mean hides concentrated drift on
  exactly the prompt families where the reward is being gamed.</li>
  <li><b>Reward-model accuracy on fresh pairs</b> — agreement with held-out human comparisons,
  re-measured as the policy improves; a static accuracy number ages badly.</li>
  <li><b>Verifiable-suite pass rate</b> — where checkers exist, the pass rate on unseen tasks is
  the one number that cannot be flattered; its trap is contamination of the suite itself.</li>
</ul>
<pre>PPO:  L = E_t[ min( ρ_t Â_t,  clip(ρ_t, 1−ε, 1+ε) Â_t ) ],   ρ_t = π_θ(a_t|s_t)/π_old(a_t|s_t)
GRPO: the same clipped surrogate with the group-normalised advantage from the maths above,
      no value network, KL-to-reference as an added penalty
DPO:  pairwise logistic loss on β-scaled log-probability-ratio margins, chosen minus rejected —
      supervised in form, equivalent in optimum to the KL-constrained RLHF objective</pre>
The offline→online gap here is overoptimisation: every offline reward curve keeps rising after
the model has started getting worse, so the reward plot alone can never tell you when to stop.`,
    maths: `<b>RLHF</b> optimises reward under a leash to the reference model:
    <pre>max_π  E_{x, y∼π}[ r_φ(x,y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )</pre>
    The reward model r_φ is fit on comparisons via Bradley–Terry: p(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l)).
    <br><br>
    <b>DPO</b> (Rafailov et al., 2023) notes this objective has a closed-form optimum,
    π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β), inverts it to write the reward as
    β·log(π(y|x)/π_ref(y|x)) + β·log Z(x), and substitutes into the Bradley–Terry preference model.
    The intractable Z(x) cancels between the chosen and rejected completions, leaving
    <pre>L = −E log σ( β log π(y_w)/π_ref(y_w)  −  β log π(y_l)/π_ref(y_l) )</pre>
    No explicit reward model, no sampling loop, no critic — preference tuning becomes a
    classification loss on paired data.
    <br><br>
    <b>GRPO</b> (DeepSeekMath, Shao et al., 2024) keeps the RL loop but deletes the value network:
    sample G completions per prompt, score them, and use the group statistics as the baseline:
    <pre>Âᵢ = ( rᵢ − mean(r₁..r_G) ) / std(r₁..r_G)</pre>
    Siblings from the same prompt are the critic. That removes an entire policy-sized model from
    memory and one more thing to tune — the reason a value-free method won by being cheaper,
    not cleverer.`,
    papers: `<ul>
      <li><b>Deep RL from Human Preferences</b> (Christiano et al., 2017) — the RLHF origin: complex
      behaviours learned with human feedback on less than 1% of the agent's interactions.</li>
      <li><b>InstructGPT paper</b> (Ouyang et al., 2022) — historically the pipeline everyone copied:
      SFT, reward model from pairwise comparisons, PPO with a KL penalty. Its labellers preferred a
      1.3B RLHF-tuned model over a 100×-larger untuned one, which is the sentence that launched the
      alignment-as-post-training era.</li>
      <li><b>DPO</b> (Rafailov et al., 2023) and <b>DeepSeekMath / GRPO</b> (Shao et al., 2024) —
      the two simplifications above.</li>
      <li><b>Tülu 3</b> (Lambert et al., 2024) named <b>RLVR</b> — reinforcement learning with
      verifiable rewards, replacing the reward model with a deterministic checker.</li>
    </ul>`,
    scratch: `A weekend, in order:
    <ol>
      <li>Implement the Bradley–Terry loss and fit a reward model on synthetic pairs where you chose
      the true reward. Check it recovers the ordering — you now understand what a reward model is.</li>
      <li>Implement DPO on a tiny model with a few hundred preference pairs. It is genuinely just a
      loss function; that realization is the point.</li>
      <li>Optimise hard against your learned reward from step 1 and track the true reward you defined.
      Watch measured reward rise while true reward peaks and falls — overoptimisation reproduced on
      your laptop is worth a dozen papers about it.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Anthropic</b> trains Claude with RLHF and Constitutional AI (RLAIF — the preference
      labels come from AI feedback guided by explicit principles), and publishes the failure-mode
      research too (Denison et al., 2024). <b>DeepSeek</b>'s GRPO line made verifiable-reward RL
      the default recipe for reasoning training across the industry in 2025.</li>
      <li><b>AI2's Tülu 3</b> is the openly documented end of the same stack — data, recipes and
      evaluations published, which makes it the deployment you can actually study rather than take
      on faith.</li>
    </ul>`,
    startups: `The quiet pattern: preference data and environments are the business, not the
    algorithm. Labs buy RLVR environments, graded agent tasks, and preference datasets from a wave
    of data companies, because DPO and GRPO are a few hundred lines while a reward signal that
    resists hacking is scarce. If you are building here, the durable asset is a verifier or a
    well-instrumented task suite, not a training loop.`,
    open_source: `<b>TRL</b> (Hugging Face) is the practitioner default for DPO/GRPO/PPO fine-tuning.
    <b>verl</b> (ByteDance) and <b>OpenRLHF</b> are the serious distributed RL-for-LLMs stacks.
    Read TRL's GRPO trainer source; it is shorter than the paper.`
  },
  next: {
    problems: `<ul>
      <li><b>Reward models saturate and then mislead.</b> Optimise long enough against any learned
      reward and true quality peaks then falls (documented as reward-model overoptimisation by
      Gao et al., 2022). Scalable oversight — rewarding what humans can't cheaply check — is the
      open problem behind the open problem.</li>
      <li><b>RLVR's boundary.</b> Verifiable rewards work where a checker exists: math, code,
      structured extraction. Extending the same training signal to essays, advice, and taste without
      reintroducing a hackable reward model is the live question of 2026 — one reason
      <x-ref to="verify">verification</x-ref> is becoming its own discipline.</li>
      <li><b>Credit assignment over agent horizons.</b> A 50-step tool-using trajectory with one
      terminal reward is REINFORCE at its worst. Process rewards, step-level verifiers, and
      tree-structured advantage estimates are all partial answers, none settled.</li>
    </ul>`,
    watch: `Follow the RLVR and process-reward literature on arXiv (cs.LG / cs.CL) and the
    specification-gaming line from the safety teams — Anthropic's reward-tampering work is the
    canary genre. The other thread worth tracking is preference-data provenance: who labelled what,
    under which instructions, is becoming the reproducibility question of the field.`
  },
  code: {
    title: 'DPO is just a loss function',
    note: `Runnable as-is on CPU, torch only. A 40-line "language model" — a table of logits over a
    tiny vocabulary — tuned on three preference pairs with the DPO loss. The printout shows the
    chosen completions' probabilities rising, the rejected ones falling, and the implicit reward
    margin β·log-ratio growing — the entire mechanism, with no reward model anywhere.`,
    lang: 'python',
    body: `import torch

V = 8                                  # vocabulary of 8 "answers"
pi  = torch.zeros(V, requires_grad=True)   # policy logits
ref = torch.zeros(V)                       # frozen reference logits
beta = 0.5

# preference pairs: (chosen answer id, rejected answer id)
pairs = [(0, 4), (1, 5), (2, 6)]

def logp(logits, i):
    return logits.log_softmax(0)[i]

opt = torch.optim.Adam([pi], lr=0.1)
for step in range(200):
    loss = 0.0
    for w, l in pairs:
        margin = beta * ((logp(pi, w) - logp(ref, w))
                         - (logp(pi, l) - logp(ref, l)))
        loss = loss - torch.nn.functional.logsigmoid(margin)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        p = pi.detach().softmax(0)
        print(f"step {step:3d}  loss {loss.item():.3f}  "
              f"p(chosen)={p[[0,1,2]].sum():.2f}  p(rejected)={p[[4,5,6]].sum():.2f}")

p = pi.detach().softmax(0)
print("final:", " ".join(f"{x:.2f}" for x in p))
print("the three chosen answers end EQUAL - no pair ever ranked them against")
print("each other, so the loss has no opinion between them. DPO only moves")
print("what a pair justifies; everything else is left to the reference.")`
  }
},


ads: {
  engineering: {
    when: `Reach for auction pricing the moment <b>demand exceeds slots and buyers hold private
    values</b> — a posted price either leaves money on the table or leaves slots empty, and an
    auction discovers the price per impression, per user, per millisecond. Reach for pacing the
    moment <b>any buyer has a budget</b>: an unpaced budget is a race to spend it first, and the
    same controller reappears wherever a finite allowance must survive a period — coupon pools,
    notification quotas, compute credits. If you have one advertiser or no scarcity, skip all of
    it and post a price.`,
    how: `Four decisions, in the order they bite:
    <ul>
      <li><b>Pricing rule.</b> Second-price/VCG keeps bidders honest and their bidding logic
      trivial; GSP is the entrenched search variant; first price (display's rule since the 2019
      exchange migrations) shifts the modelling burden onto every bidder as a shading model.
      You are choosing who has to be clever.</li>
      <li><b>The ranking score.</b> <code>bid × pCTR</code> plus quality terms. Here calibration
      beats discrimination: the pCTR multiplies money, so a model that ranks perfectly but runs
      hot by a constant factor still misprices every runner-up payment. Calibrate per slice, not
      just globally.</li>
      <li><b>The pacing loop.</b> Forecast eligible supply per campaign, lay a target cumulative
      spend curve over it, measure actual spend from a near-real-time counter, and close the loop
      with a controller — commonly PID — driving one of two actuators: participation probability
      (throttling) or a bid multiplier (shading). Throttling preserves auction incentives;
      multipliers double as the Lagrangian λ of the budget constraint.</li>
      <li><b>Frequency caps.</b> A per-user-per-campaign counter enforced at eligibility, before
      the auction. It is both policy and model hygiene — fatigued impressions depress CTR, and a
      capped campaign stops poisoning its own pCTR training data.</li>
    </ul>`,
    where: `Search ads (GSP with quality score), feed ads (VCG-style, Meta), exchange-traded
    display (first price), and retail-media marketplaces. The pCTR under every bid is the ranking
    model of <x-ref to="features">1.2</x-ref>; the slate where paid results blend with organic
    ones is the re-ranking stage of <x-ref to="funnel">1.1</x-ref>; and the explore/exploit
    pressure on new ads is the same one <x-ref to="rl">6.1</x-ref> treats with bandits.`,
    breaks: `<ul>
      <li><b>Miscalibrated pCTR misprices the market.</b> An overconfident model inflates an ad's
      rank and, through the GSP formula, inflates what the ad <em>above</em> it pays. Advertisers
      notice CPC drift before your AUC dashboard does.</li>
      <li><b>The midnight thundering herd.</b> Budgets reset at once, every controller opens its
      throttle at once, clearing prices spike, and the overshoot arrives before the spend counters
      do. Stagger resets, or feed the reset into the controller as a known disturbance rather than
      letting the integral term discover it.</li>
      <li><b>Spend-counter lag means flying blind.</b> Billing events land asynchronously; a
      controller reading a stale counter keeps buying past the budget. Platforms eat the
      overdelivery. The pacing loop is only as good as the freshness of its feedback signal.</li>
      <li><b>Controllers are coupled through prices.</b> Your throttle changes the clearing prices
      every other campaign sees; campaigns collectively chasing the cheap hours un-cheapen them.
      Tuning one controller in isolation on replayed logs misses the market's response.</li>
      <li><b>"Smart" throttling is a product decision in disguise.</b> A uniform-random throttle
      delivers evenly; skewing participation toward high-pCTR requests quietly converts
      even delivery into performance delivery. Xu et al. (KDD 2015) made that dial explicit —
      decide its setting on purpose, with the advertiser's contract in hand.</li>
    </ul>`
  },
  research: {
    evals: `Pacing and pricing are judged together on <b>delivered value under satisfied
constraints</b> — did campaigns spend the budget, hit their cost targets, and buy the impressions
worth buying, all at once.
<ul>
  <li><b>Budget utilisation</b> — fraction of budget delivered by flight end; its trap is that a
  greedy spender maxes it trivially, which is the exact failure pacing exists to prevent — never
  read it without smoothness beside it.</li>
  <li><b>Spend-curve deviation</b> — distance between actual and target cumulative spend; its trap
  is that perfectly even delivery is cheap if you ignore performance, since a smooth schedule of
  worthless impressions scores flawlessly.</li>
  <li><b>Constraint violation (tCPA / ROAS)</b> — share of campaigns ending outside their cost
  target; its trap is flight-level averaging, where a terrible final day hides inside a good
  week — report per-window violation too.</li>
  <li><b>Counterfactual auction replay</b> — revenue and welfare of a rule change replayed over
  logged bids; its trap is that the logged bids were optimised for the old rule, so replay is
  silent about the equilibrium the change will provoke.</li>
</ul>
<pre>pacing as the budgeted programme:
  max Σᵢ xᵢ·vᵢ   s.t.  Σᵢ xᵢ·cᵢ ≤ B     →  bid vᵢ/(1+λ), λ by dual descent
the controller that approximates it online:
  e(t) = target_spend(t) − actual_spend(t)
  u(t) = Kp·e(t) + Ki·Σe + Kd·Δe          →  throttle probability or bid multiplier</pre>
The offline→online gap here is strategic, not statistical: logs hold every rival's bids fixed,
but shipping a new price rule or pacing policy moves the equilibrium those rivals are
optimising against — the one response no replay contains.`,
    maths: `<b>Vickrey truthfulness</b> is a two-case argument. Let your value be v and the top
    rival bid be r. Bidding b &gt; v changes the outcome only when v &lt; r &lt; b — auctions you
    now win at price r &gt; v, a loss. Bidding b &lt; v changes the outcome only when
    b &lt; r &lt; v — profitable auctions you now lose. Every deviation only ever adds bad events,
    so b = v dominates regardless of opponents.
    <br><br>
    <b>GSP</b> generalises the price, not the property. With slots ordered by bid × pCTR, slot i
    pays per click
    <pre>p_i = b_{i+1} · q_{i+1} / q_i        # least bid that still holds slot i</pre>
    Edelman, Ostrovsky &amp; Schwarz (2007) show truth-telling is not an equilibrium once there is
    more than one slot — shading down can hold a cheaper slot at nearly the same click volume —
    but GSP's envy-free equilibria exist and revenue-dominate VCG's truthful one. <b>VCG</b>
    instead charges each winner the value lost by everyone else because of its presence, which
    restores dominant-strategy truthfulness at the cost of an unfamiliar, hard-to-explain price.
    <br><br>
    <b>Pacing is a Lagrangian.</b> For a value maximiser with budget B facing a truthful auction,
    relaxing the budget constraint with multiplier λ turns the optimal policy into "bid
    v/(1+λ) everywhere": one number converts the budget into a uniform shade. Dual descent
    <pre>λ ← λ + η·(spend_rate − B/T)</pre>
    is the idealised form of the PID loop above, and it is why Aggarwal, Badanidiyuru &amp; Mehta
    (WINE 2019) find uniform bidding optimal for constrained autobidders under truthful
    mechanisms. Probabilistic throttling solves the same programme with a different action space —
    participation instead of price — trading a little optimality for leaving incentives intact.`,
    papers: `<ul>
      <li><b>Internet Advertising and the Generalized Second-Price Auction</b> (Edelman, Ostrovsky
      &amp; Schwarz, AER 2007). The founding analysis: GSP is not Vickrey, truth-telling is not an
      equilibrium, and the envy-free equilibrium concept that explains why the mechanism works
      anyway.</li>
      <li><b>VCG in Theory and Practice</b> (Varian &amp; Harris, 2014). Two Google economists on
      why VCG's truthfulness is worth having, why Facebook launched with it, and why Google's own
      search auction stayed GSP — inertia and retraining costs, stated with unusual candour.</li>
      <li><b>Budget Pacing for Targeted Online Advertisements at LinkedIn</b> (Agarwal et al.,
      KDD 2014). Probabilistic throttling in production: forecast eligible impressions, allocate
      budget over the forecast, gate participation — chosen so the auction itself never sees the
      budget.</li>
      <li><b>Smart Pacing for Effective Online Ad Campaign Optimization</b> (Xu et al., KDD 2015).
      The even-delivery-versus-performance dial made formal: requests grouped by predicted response
      rate, group pacing rates adjusted by online feedback control.</li>
      <li><b>Bid Shading in The Brave New World of First-Price Auctions</b> (Zhou et al., CIKM
      2020). The DSP response to the 2019 migration — learning how far below value to bid from
      auction feedback, the problem the second-price world never had.</li>
      <li><b>Autobidding with Constraints</b> (Aggarwal, Badanidiyuru &amp; Mehta, WINE 2019) and
      the <b>Auto-bidding and Auctions survey</b> (Aggarwal et al., 2024). The theory rebuilt for
      value maximisers under tCPA/ROAS constraints: uniform bidding optimality, and what breaks in
      auction design when every bidder is an algorithm.</li>
    </ul>`,
    scratch: `Buildable in an afternoon, in order:
    <ol>
      <li>Simulate one slot, lognormal rival bids, and sweep your bid around your value under both
      price rules. Plot surplus versus bid: second price plateaus exactly at truth-telling, first
      price peaks below value. That single figure is most of the auction theory you'll use.</li>
      <li>Implement GSP and VCG payments for two slots and three bidders. Find a case where a
      bidder profits by shading under GSP — you have reproduced the EOS non-truthfulness result
      by hand.</li>
      <li>Add a budget and pace by dual descent on λ, bidding v/(1+λ). Watch λ settle to the
      market-clearing shade.</li>
      <li>Swap the λ actuator for a participation probability driven by a PID loop — the code tab —
      and compare what each version pays and wins. The difference is the throttle-versus-shade
      choice made tangible.</li>
    </ol>`
  },
  industry: {
    big: `<ul>
      <li><b>Google</b> runs the two regimes side by side: search ads on GSP with quality score,
      and display via Ad Manager on the unified first-price auction it rolled out through 2019,
      removing "last look" in the process (Ad Manager blog). Much of the autobidding theory —
      including the WINE 2019 constraints paper and the 2024 survey — comes from its researchers,
      which tells you where the production pain is.</li>
      <li><b>Meta</b> has run VCG since Facebook's early ads system — Varian &amp; Harris (2014)
      note it as the major deployment — a mechanism choice that generalises cleanly to feed
      formats that never looked like a ranked column of links.</li>
      <li><b>LinkedIn</b> published the canonical throttling system (KDD 2014); <b>Yahoo</b> and
      its successor DSP produced the smart-pacing (KDD 2015) and bid-shading lines — between them
      the practical pacing literature is largely these two shops writing down what broke.</li>
      <li><b>Kuaishou</b> is the live example of this machinery meeting generative ranking — its
      ad-side system and the reported numbers live in <x-ref to="sid">3.3</x-ref>, where the open
      problem of injecting bids into constrained decoding is exactly this lesson colliding with
      that one.</li>
    </ul>`,
    startups: `The auction is commodity; the loops around it are the business. DSPs differentiate
    almost entirely on bid landscape modelling, shading and pacing quality — after the 2019
    first-price migration, shading went from research topic to line-item vendor feature in about a
    year. The other durable niche is measurement: incrementality and attribution startups exist
    because auctions price impressions while advertisers want caused conversions, and the gap
    between those two is where budgets leak. If you are building here, note that every mechanism
    change by a major exchange instantly obsoletes and recreates the bidding-tools market.`,
    open_source: `Thin, tellingly — the interesting parts are attached to real money. The
    <b>iPinYou</b> dataset remains the standard public corpus of real bid logs with winning prices,
    the substrate for most academic bidding and shading papers; <b>Criteo</b>'s click and
    attribution datasets serve the pCTR side. The <b>rtb-papers</b> collection on GitHub is the
    field's de facto reading list. There is no open-source ad exchange worth running — but the
    pacing controller in the code tab is small enough that you don't need one.`
  },
  next: {
    problems: `<ul>
      <li><b>Auction design for autobidders.</b> Classical guarantees assume utility maximisers;
      the floor is now value maximisers under constraints, where truthfulness and efficiency
      results change shape. Welfare bounds for first- and second-price auctions in the autobidding
      world are an active frontier (Aggarwal et al. survey, 2024).</li>
      <li><b>Pacing equilibria.</b> Thousands of budget controllers coupled through clearing
      prices form a game, not a control problem. When equilibria exist, whether the decentralised
      loops find them, and whether the platform should compute the allocation centrally instead is
      open in both theory and infrastructure.</li>
      <li><b>Incrementality-aware bidding.</b> Bidding on pCTR buys correlated clicks, not caused
      conversions; folding causal lift into the bid without an unaffordable volume of holdout
      experiments is unsolved at scale.</li>
      <li><b>Pricing under signal loss.</b> Privacy changes degrade the pCTR that every bid
      multiplies, repricing the whole market at once. Auctions that stay efficient with aggregated
      or on-device signals are a mechanism-design and an ML problem simultaneously.</li>
    </ul>`,
    watch: `Follow the AdKDD workshop and the auctions tracks at EC and WWW for the theory; follow
    exchange product blogs for the practice — the 2019 first-price migration, the biggest
    mechanism change in the field's history, was announced as a product post, not a paper. The
    autobidding survey's open-problem list is the clearest map of where the next five years of
    this literature will come from.`
  },
  code: {
    title: 'Morning exhaustion vs a PID pacer',
    note: `Runnable as-is on CPU, numpy only. One day of second-price auctions: competition is
    fierce and expensive during business hours, then cheap in the evening — partly because rivals
    who didn't pace are already broke. Both bidders value an impression at $4.00 and bid
    truthfully; the naive one enters every auction, the paced one gates participation with a PID
    loop against a supply-weighted spend plan, LinkedIn-style. With this seed the naive bidder is
    broke by 09:00 while the pacer buys until 23:00 — winning more impressions for the same
    budget at a lower average price, because the cheap evening still exists for it.`,
    lang: 'python',
    body: `import numpy as np

rng = np.random.default_rng(7)
traffic  = [5,3,2,2,3,5,20,90,150,200,220,230,230,230,230,240,250,260,280,300,310,280,180,90]
price_mu = [1.2,1.1,1.0,1.0,1.0,1.1,1.6,2.8,3.4,3.6,3.5,3.4,3.3,3.2,3.1,3.0,2.8,2.5,2.1,1.7,1.5,1.4,1.3,1.2]
VALUE, BUDGET = 4.00, 650.0                       # true value per impression; daily budget
plan = BUDGET * np.array(traffic) / sum(traffic)  # LinkedIn-style: spend where supply is

def hour(h):                                      # second-price: bid VALUE, pay top rival
    comp = rng.lognormal(np.log(price_mu[h]), 0.30, traffic[h])
    return comp[comp < VALUE]                     # prices paid for the auctions we win

def naive():
    left, spend, wins = BUDGET, np.zeros(24), np.zeros(24)
    for h in range(24):
        paid = np.cumsum(hour(h))
        k = int(np.searchsorted(paid, left, side="right"))
        spend[h], wins[h] = (paid[k-1] if k else 0.0), k
        left -= spend[h]
    return spend, wins

def pid():                                        # throttle participation, never the bid
    p, i_err, prev = 0.10, 0.0, 0.0
    left, spend, wins = BUDGET, np.zeros(24), np.zeros(24)
    for h in range(24):
        take = hour(h)
        paid = np.cumsum(take[rng.random(take.size) < p])
        k = int(np.searchsorted(paid, left, side="right"))
        spend[h], wins[h] = (paid[k-1] if k else 0.0), k
        left -= spend[h]
        err = (plan[h] - spend[h]) / (BUDGET / 24)   # positive: behind plan
        i_err += err
        p = float(np.clip(p + 0.04*err + 0.012*i_err + 0.008*(err - prev), 0.02, 1.0))
        prev = err
    return spend, wins

(ns, nw), (ps, pw) = naive(), pid()
print("hour    naive $  wins    paced $  wins")
for h in range(24):
    print(f"{h:02d}:00 {ns[h]:9.1f} {nw[h]:5.0f} {ps[h]:10.1f} {pw[h]:5.0f}")
print(f"total {ns.sum():9.1f} {nw.sum():5.0f} {ps.sum():10.1f} {pw.sum():5.0f}")
print(f"avg price paid:  naive {ns.sum()/nw.sum():.2f}   paced {ps.sum()/pw.sum():.2f}")
dead = int(np.argmax(np.cumsum(ns) >= 0.99 * BUDGET))
print(f"naive bidder broke at {dead:02d}:00; the paced bidder was still buying at 23:00")`
  }
},

};

/* =========================================================================
   CAPSTONES
   A capstone is specified, not suggested: architecture, datasets, repo
   layout, the machine it runs on, what to build in what order, and the
   metrics that decide whether it worked.
   ========================================================================= */

const CAPSTONES = {

genrec: {
  id: 'genrec',
  title: 'Build a generative recommender',
  fig: 'capstone_genrec',
  figCap: 'The full pipeline. Everything left of the dashed line is offline and runs once; everything right of it runs per request.',
  one: `Take a public interaction dataset, learn semantic IDs for its catalogue, train a sequence
  model that generates the next item's ID, and beat a well-tuned SASRec baseline on a protocol you
  can defend. Then serve it and measure what it costs.`,
  builds_on: ['funnel', 'embed', 'attention', 'seqrec', 'sid', 'verify'],
  weeks: '6–8 weeks of evenings',
  why: `This is the smallest project that forces you through the entire stack: representation,
  sequence modelling, retrieval, serving and honest evaluation. It is also the one where the
  literature is fresh enough that a careful hobby build can produce a genuinely publishable ablation.`,

  datasets: [
    ['Amazon Reviews 2023', 'start here',
     'Category subsets (Beauty, Sports, Toys, CDs) are the standard benchmark for this exact task. Rich item text — title, price, brand, category hierarchy — which is what the tokenizer needs. Small enough to iterate on a laptop.',
     'Follow the TIGER/GRID convention: 5-core filter (drop users and items with fewer than 5 interactions), keep title, price, brand and categories as the item text, order by timestamp.'],
    ['MovieLens-1M / 20M', 'sanity check',
     'Small, clean, universally reported. Useful for catching bugs, not for claiming results — item text is thin (title and genres only), which starves the tokenizer.',
     'Filter users under 5 interactions, truncate sequences to 100.'],
    ['Yambda-5B (Yandex Music)', 'when you outgrow the above',
     'Roughly 4.79 billion interactions across about 1M users and 9.39M tracks, with pretrained audio embeddings, an is_organic flag separating recommender-driven from organic actions, and 50M / 500M subsamples so you can start small.',
     'Uses a global temporal split rather than leave-one-out, which does not break temporal dependencies — closer to how a real system is tested. Prefer this protocol.'],
    ['Kwai26', 'if you want an industrial protocol',
     'A billion-scale benchmark for multi-objective generative retrieval released with predefined splits and an evaluation protocol, alongside the Multi-Decoder OneRec work.',
     'Use it for the protocol even if you cannot afford the scale.']
  ],
  scraping: `You almost certainly should not scrape for this one. The public datasets are better than
  anything you'd assemble, already deduplicated, and comparable to published numbers — which is the
  entire point of a benchmark. If you do need custom data (a niche catalogue, a language the public
  sets don't cover), scrape <em>item metadata</em> from a source whose terms permit it and pair it
  with your own product's interaction logs. Never scrape user interaction data; it is other people's
  behaviour and usually other people's property.`,

  machine: `<ul>
    <li><b>Laptop is enough to start.</b> Amazon-Beauty with a 3-level tokenizer and a small decoder
    trains in under an hour on CPU, and in minutes on any GPU. Do not rent anything yet.</li>
    <li><b>One consumer GPU</b> (12–24 GB) carries you through the real experiments on Amazon and
    ML-1M. This is where most of the project lives.</li>
    <li><b>Rented A100/H100 hours</b> only once you are scaling to Yambda-500M or running the scaling
    ablation. Budget the run before you start it; an unbounded sweep is how hobby projects die.</li>
    <li>Environment: Python 3.11, PyTorch, a sentence encoder from Hugging Face, and nothing else you
    can avoid. Every extra dependency is a future breakage.</li>
  </ul>`,

  repo: `<pre>genrec/
├── README.md               <- the results table lives here, updated every run
├── pyproject.toml
├── configs/
│   ├── tokenizer.yaml      <- levels, codebook size, encoder name
│   └── model.yaml          <- layers, dim, sequence length, lr
├── data/
│   ├── download.py         <- fetch + verify checksums
│   ├── prepare.py          <- 5-core filter, chronological split, freeze to parquet
│   └── stats.py            <- prints catalogue size, sparsity, sequence-length histogram
├── tokenizer/
│   ├── encode.py           <- item text -> content embeddings (cached to disk)
│   ├── rqvae.py            <- the quantizer from the semantic-ID lesson
│   └── report.py           <- code utilization per level; run this every time
├── model/
│   ├── baseline_sasrec.py  <- build this FIRST
│   ├── genrec.py           <- decoder over SID tokens
│   └── decode.py           <- constrained beam search over the code tree
├── eval/
│   ├── protocol.py         <- the split, frozen and hashed
│   ├── metrics.py          <- Recall@k, NDCG@k, coverage, cold-start slice
│   └── compare.py          <- baseline vs model, same protocol, one table
├── serve/
│   ├── api.py              <- FastAPI, one endpoint
│   └── bench.py            <- p50/p99 latency and cost per 1k requests
└── tests/
    ├── test_quantizer.py   <- reconstruction improves per level
    ├── test_protocol.py    <- no future leakage; assert train.max_ts < test.min_ts
    └── test_decode.py      <- every decoded tuple maps to a real item</pre>`,

  steps: [
    ['Week 1', 'Data, frozen',
     `Download, apply the 5-core filter, split chronologically, write to parquet, and hash the split
      file. Print the statistics. Write <code>test_protocol.py</code> now — an assertion that no test
      timestamp precedes any training timestamp. Future leakage is the bug that will otherwise
      invalidate every number you produce for the next two months.`],
    ['Week 2', 'The baseline you have to beat',
     `Implement SASRec. Tune it properly — a badly tuned baseline is how most papers manufacture
      their improvement, and you'll know if yours is one of them. Record its Recall@10 and NDCG@10 in
      the README. This number is the project's reference point.`],
    ['Week 3', 'Tokenizer',
     `Embed item text with a frozen sentence encoder, cache to disk. Train the RQ-VAE. Run
      <code>tokenizer/report.py</code>: code utilization per level, residual norm ratio per level, and
      a printout of ten items sharing a level-1 code. If those ten items aren't obviously related,
      stop — nothing downstream will work.`],
    ['Week 4', 'Generative model',
     `A small causal decoder over SID tokens. Predict the next item's tuple. Start with 4 layers and
      256 dimensions; resist scaling until the small one is correct.`],
    ['Week 5', 'Constrained decoding',
     `Beam search restricted to valid prefixes in the code tree, so every generated tuple is a real
      item. <code>test_decode.py</code> asserts this. Compare against the baseline on the frozen
      protocol, ranking against the full catalogue — never against sampled negatives.`],
    ['Week 6', 'Evaluation that survives scrutiny',
     `Slice by item popularity and item age. The cold-start slice is where semantic IDs should win
      and where aggregate metrics hide the effect. Also report coverage: what fraction of the
      catalogue is ever generated. A model that only ever emits the head is not a retrieval system.`],
    ['Week 7', 'Serve it',
     `FastAPI, one endpoint, and a benchmark script. Report p50 and p99 latency and cost per thousand
      requests next to the quality numbers. This is the step that turns an experiment into an
      engineering result, and the step almost every hobby project skips.`],
    ['Week 8', 'One honest ablation',
     `Pick a single question — codebook size, number of levels, k-means initialization versus random,
      frozen versus co-trained tokenizer — and answer it properly with the protocol you already froze.
      Write it up including the negative result if that's what you get.`]
  ],

  offline: [
    ['Recall@10, NDCG@10', 'Headline quality. Rank against the full catalogue, never sampled negatives — sampled-negative numbers are not comparable across papers and inflate everything.'],
    ['Cold-start slice', 'The same metrics restricted to items with fewer than N interactions. This is the claim semantic IDs actually make; if it does not move here, they are not helping.'],
    ['Catalogue coverage', 'Fraction of distinct items appearing in any top-10. Falling coverage with rising Recall means you learned popularity, not preference.'],
    ['Code utilization per level', 'Tokenizer health. Level-1 utilization below roughly half your codebook size means collapse.'],
    ['Reconstruction residual per level', 'Should fall monotonically. If level 3 does not reduce the residual, you have too many levels.']
  ],
  online: [
    ['A/B on engagement', 'Only if you have a real surface. Otherwise say so plainly rather than implying you ran one.'],
    ['p50 / p99 latency', 'Constrained beam search is the expensive part. Measure it, do not estimate it.'],
    ['Cost per 1k requests', 'Quality per dollar is the number that decides whether a generative retriever ships anywhere.'],
    ['Candidate survival', 'If you feed a downstream ranker, measure what fraction of generated candidates survive into the served slate — see <x-ref to="funnel">1.1</x-ref>.']
  ],
  output: `A repository whose README opens with one table: baseline versus your model, on a frozen
  protocol, with cold-start and coverage columns and a latency figure. Plus one paragraph on what did
  not work. That artifact is worth more in an interview than any number of finished-looking demos,
  because it is the only kind that can be checked.`
},

rlab: {
  id: 'rlab',
  title: 'A bandit lab: learning without deploying',
  fig: 'capstone_rlab',
  figCap: 'Two lanes converging on one table. The simulator lane proves your algorithms work where truth is known; the logged-data lane estimates what they would do in the world, without deploying anything.',
  one: `Build the smallest honest reinforcement-learning stack: bandit algorithms proved in a
  simulator where ground truth is known, then evaluated against real logged data with recorded
  propensities — so you can say what a new policy would have earned without ever running it live.`,
  builds_on: ['rl', 'features', 'verify'],
  weeks: '3–4 weeks of evenings',
  why: `This is the easiest capstone on the list and the one with the highest skill-per-hour. Every
  serious use of RL in industry — headline selection, ad placement, recommender exploration, even
  evaluating LLM policies — runs through the same two disciplines this builds: calibrated exploration
  and off-policy evaluation. Everything trains in minutes on a laptop, so the loop between idea and
  evidence is as short as it ever gets in <x-ref to="rl">6.1</x-ref>.`,

  datasets: [
    ['Your own simulator', 'start here',
     'Five arms with known click probabilities you chose. The only place ground truth exists, which makes it the only place you can check an estimator against the answer.',
     'Write it in an afternoon with numpy. Keep the true means in a config file so every experiment is reproducible.'],
    ['Open Bandit Dataset (ZOZO)', 'the real thing',
     'About 26M impressions from a 7-day experiment on the ZOZOTOWN fashion platform across three campaigns, with the served action, the click, and — the rare part — the true propensity of the logging policy for every row. Built explicitly for off-policy evaluation research.',
     'Start with the smallest campaign. Histogram the propensities before anything else; effective sample size, not row count, is your real budget.'],
    ['Open Bandit Pipeline (obp)', 'the reference implementation',
     'The companion Python library: estimators (IPS, SNIPS, DR and more) plus dataset loaders. Use it to check your own implementations, not instead of writing them.',
     'Implement each estimator yourself first, then diff against obp on the same split. Disagreement means your code is wrong or your propensity handling is.'],
    ['MovieLens-1M', 'optional grounding',
     'Turn ratings into a simulated contextual bandit (recommend a genre, reward if the held-out rating is high) when you want contexts richer than the simulator but a world you still control.',
     'Keep the simulation rules in one file. The moment reward logic is scattered, your results stop being checkable.']
  ],
  scraping: `None, and this time it is not just ethics: off-policy evaluation needs the logging
  policy's action probabilities, and no amount of scraping recovers a propensity that was never
  recorded. Without propensities you are doing guesswork with error bars drawn from nowhere. Use the
  public dataset, or instrument a surface you own so it logs (context, action, propensity, reward)
  from day one — that one logging decision is what makes future evaluation possible at all.`,

  machine: `<ul>
    <li><b>A laptop is the whole lab.</b> numpy and pandas; every experiment here runs in seconds to
    minutes on CPU. There is nothing to rent.</li>
    <li>Optionally <code>obp</code> for cross-checking estimators, and matplotlib for regret curves.
    Nothing else.</li>
    <li>The absence of infrastructure is the point: when evidence is this cheap, the only excuse for
    an unchecked claim is not having asked.</li>
  </ul>`,

  repo: `<pre>rlab/
├── README.md               <- the policy × estimator table lives here
├── config/
│   └── sim.yaml            <- true arm means, horizon, seeds
├── sim/
│   ├── world.py            <- the bandit environment; truth lives here only
│   └── run.py              <- regret curves: eps-greedy vs UCB vs Thompson vs fixed A/B
├── data/
│   └── load_obd.py         <- fetch + verify the campaign; propensity histogram
├── policies/
│   ├── baselines.py        <- uniform, popularity, fixed A/B
│   ├── egreedy.py
│   ├── ucb.py
│   └── linucb.py           <- the contextual step
├── ope/
│   ├── ips.py              <- inverse propensity scoring, clipped and not
│   ├── snips.py            <- self-normalized variant
│   ├── dr.py               <- doubly robust: model + IPS correction
│   └── ci.py               <- bootstrap confidence intervals
├── report/
│   └── table.py            <- policies × estimators ± CI, one honest table
└── tests/
    ├── test_estimators.py  <- IPS of the logging policy ≈ empirical mean
    └── test_no_peeking.py  <- policies never see the reward of unchosen arms</pre>`,

  steps: [
    ['Week 1', 'The simulator, and the cost of not adapting',
     `Build the five-arm world and run epsilon-greedy, UCB and Thompson sampling against a fixed
      50/50 A/B split. Plot cumulative regret. The gate: your bandits reliably beat the static split,
      and you can explain the one plot to someone in two minutes. This is the lesson's code tab grown
      into an experiment.`],
    ['Week 2', 'Real logs, and respect for propensities',
     `Load the smallest Open Bandit Dataset campaign. Before any estimation: histogram propensities,
      compute effective sample size, and check the logging policy's empirical click rate. Implement
      replay-style evaluation. The gate is <code>test_estimators.py</code>: your IPS estimate of the
      logging policy itself must match its empirical mean — if it doesn't, nothing downstream means
      anything.`],
    ['Week 3', 'The estimator ladder',
     `IPS, then SNIPS, then doubly robust, each with bootstrap confidence intervals. Watch the
      variance fall as the estimators get smarter, and record the max importance weight — one row
      with propensity 0.001 can own your whole estimate, and knowing that is the discipline.`],
    ['Week 4', 'The decision table',
     `Evaluate three candidate policies — popularity, learned epsilon-greedy, LinUCB — with every
      estimator. The deliverable is one table: policy × estimator, value ± CI, effective sample size.
      Then the sentence that makes it a decision and not a dashboard: which policy ships, and how
      sure are you.`],
    ['If it works', 'Close the loop',
     `Run the winning policy on any surface you actually own — even reordering projects on a personal
      site counts — and compare the online result to the offline prediction. The calibration gap
      between them is the finding, whichever way it goes; write it down either way, as
      <x-ref to="verify">5.2</x-ref> demands.`]
  ],

  offline: [
    ['Cumulative regret (simulator)', 'The exploration cost of each algorithm against known truth. If a bandit does not beat the fixed split here, stop and debug before touching real data.'],
    ['Policy value by IPS / SNIPS / DR ± CI', 'The headline table. Report all three; when they disagree, the disagreement is diagnostic, not noise to average away.'],
    ['Effective sample size', 'What your millions of rows are actually worth after importance weighting. This number decides whether your confidence intervals are honest.'],
    ['Max importance weight', 'The single-row concentration risk. Report it next to every IPS figure; clipping without reporting is quiet dishonesty.'],
    ['Estimator self-check', 'Every estimator applied to the logging policy itself must recover the empirical mean. This is the test that catches propensity bugs.']
  ],
  online: [
    ['Prediction vs outcome', 'If you close the loop on a surface you own: the offline estimate against what actually happened. Calibration of this gap is the entire promise of OPE.'],
    ['Click-through under the new policy', 'Only meaningful with a guardrail: predefine the stopping rule before you start, or you are running the exploratory analysis the lesson warns about.'],
    ['Exploration budget spent', 'What fraction of traffic went to uncertain arms. The cost side of the ledger that engagement numbers alone hide.']
  ],
  output: `A repository whose README opens with the policy × estimator table, confidence intervals
  included, and one sentence committing to a decision. Plus the regret plot from week one. Anyone who
  can read that table knows you understand exploration, importance weighting and honest uncertainty —
  which is more than most production dashboards demonstrate.`
}

};
