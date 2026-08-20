/* =========================================================================
   COURSE CONTENT
   Each lesson has two bodies (beginner / expert), one figure, one open
   question with a grading rubric, and sources with a reason to open them.
   ========================================================================= */

const COURSE = {
  title: "How Machines Decide",
  parts: [
    {
      n: 1,
      name: "Choosing what to show",
      blurb: "Every feed, every search box, every store. One shape underneath all of them.",
      lessons: ["funnel", "features", "ads"]
    },
    {
      n: 2,
      name: "Meaning as geometry",
      blurb: "Turning things into numbers, then searching the numbers.",
      lessons: ["embed", "ann"]
    },
    {
      n: 3,
      name: "Sequences",
      blurb: "Why 'guess the next thing' turned out to be the whole game.",
      lessons: ["attention", "seqrec", "sid"]
    },
    {
      n: 4,
      name: "Looking things up",
      blurb: "Giving a model real documents, and making it show its work.",
      lessons: ["rag", "hybrid", "chunk"]
    },
    {
      n: 5,
      name: "Acting",
      blurb: "Loops, tools, and the second opinion that makes them trustworthy.",
      lessons: ["agent", "verify"]
    },
    {
      n: 6,
      name: "Learning from consequences",
      blurb: "No answer key. Just a score that arrives late.",
      lessons: ["rl", "rlhf"]
    }
  ],

  lessons: {

/* ---------------------------------------------------------------- 1.1 */
funnel: {
  id: "funnel",
  title: "The funnel",
  hook: "How do you pick 10 things out of a billion, in 100 milliseconds?",
  fig: "funnel",
  figCap: "The story from the lesson, with typical numbers. Read it left to right: a billion candidates become a thousand, become fifty, become the ten on screen — and each stage is allowed a slower, smarter model precisely because it faces fewer items.",
  beginner: `
<p>Open a shopping app and type "running shoes". Behind that search box sits a catalogue of —
let's say — a billion items, and the app answers in about a tenth of a second. Something that
should be impossible just happened. This lesson is the trick that makes it possible.</p>

<p>First, why it's impossible the obvious way. Imagine the app has one really good judging model:
show it an item plus everything you've browsed, and it predicts how much you'd like it. Careful
judging like that costs about a millisecond per item. A billion items at a millisecond each is
eleven <em>days</em> of computing for one search. Even spreading the work across a hundred machines
only gets you to a few hours. To answer in a tenth of a second, the careful judge can look at
<mark>a few hundred items at most — out of a billion</mark>. That is the whole problem in one
sentence: everything depends on choosing which few hundred.</p>

<p>The trick is the one airports use. Security doesn't interview every passenger — a fast, rough
check filters everyone, and only the flagged few get the slow, careful search. Recommenders work in
the same stages. Stage one, called <b>retrieval</b>, runs an extremely cheap test — a single
similarity number per item, and <x-ref to="embed">2.1</x-ref> shows where that number comes from —
across the entire billion, and keeps roughly a thousand plausible candidates. It is rough on purpose
and it makes mistakes; that is allowed. Stage two, <b>ranking</b>, is the careful judge from before,
and now it can afford to be careful, because it faces a thousand items instead of a billion — a
thousand milliseconds of judging squeezed into about forty of clock time by scoring the pile in
parallel batches across a few dozen machines. Stage
three, <b>re-ranking</b>, looks at the survivors <em>as a group</em>: it removes near-duplicates,
mixes in some variety, and applies house rules like "no more than two sponsored results".</p>

<p>Follow one product through. Your ideal blue trainers are item #482,119,204 of the billion.
Retrieval's cheap test scores them 0.71 — above its bar — so they survive into the thousand. Ranking
reads your history properly and scores them 0.93, third-best in the pile. Re-ranking notices the two
items above them are nearly identical to each other and drops one. Your trainers appear as the
second tile on your screen, about 55 milliseconds after you pressed enter: roughly 5 for the sweep,
40 for the careful judging, 10 for the tidy-up.</p>

<x-fig name="funnel_ex"></x-fig>

<p>Now the failure worth remembering. Suppose retrieval's rough test misjudges your perfect trainers
— scores them 0.12, below the bar. The careful judge would have loved them. It never gets the
chance. <mark>A mistake in the first stage is invisible and unrecoverable</mark>, because every later
stage can only work with what the first one hands over. That is why experienced engineers obsess
over the unglamorous rough filter at least as much as the clever judge — and why the figure below
puts the stages side by side, so you can see where the power actually sits.</p>`,
  expert: `
<p>The beginner track showed the funnel working. Building one is a different experience: it is
four decisions, each forced on you by arithmetic, and this track walks through them the way you
will actually meet them.</p>

<x-fig name="funnel_ex"></x-fig>

<p><b>Decision one: what can the first stage afford?</b> Write the constraint down before any
architecture diagram, because every choice falls out of it: <code>cost_per_item × n_items ≤
latency_budget</code>. Give stage one half the ~100 ms budget over a billion items, spread the
sweep across a thousand shards working in parallel, and each shard still gets only about
50 <em>nano</em>seconds per item — enough for one precomputed-vector comparison and nothing
else. Read that number again, because it is doing a lot of quiet work: it means the item's
representation must exist <em>before the query arrives</em>, which is why the item side of
the two-tower model in <x-ref to="embed">2.1</x-ref> is forbidden from looking at the user. Anything
that needs the query and the item in the same computation is exiled to stage two — by physics,
not preference.</p>

<p><b>Decision two: where do candidates really come from?</b> The clean story says "the ANN
index". Production says: a union of sources. A taste-vector lookup for users with history, a
popularity list for users without, a followed-accounts feed, items your friends engaged with
yesterday. Each source exists because some slice of users had an empty or terrible candidate set
without it. That is the instinct worth stealing: <mark>you add a retrieval source when a
identifiable group has no good candidates, not when an average metric looks low</mark> — averages
hide exactly the people the next source would save.</p>

<p><b>Decision three: what does the careful judge actually maximise?</b> "Predict the click" is
where everyone starts and almost nobody ends. Clicks are plentiful but cheap — they reward
curiosity-bait. Purchases, saves, watch-to-the-end and reports-of-regret arrive rarer but mean
more, so production rankers predict several outcomes at once and combine them with hand-set
weights: something like <code>score = p(click) + 8·p(purchase) − 40·p(report)</code>, where the
weights are the product strategy written as numbers. Changing them is the most consequential
edit in the company, and it never appears in a model file.</p>

<p><b>The war story you will live through once.</b> A new ranking model wins offline by a clear
margin. It ships. Online metrics do not move. Weeks of confusion later, someone checks the one
number nobody instruments: of the items users actually engaged with (found via search, links,
anywhere), what fraction ever appeared in the ranker's candidate pool? If the answer is "not
many", the brilliant judge was reordering the wrong pile — <mark>a ranker cannot rescue an item
retrieval never surfaced</mark>. Ship the coverage metric before you ship the clever model; it
explains more dead launches than every model diagnostic combined.</p>

<p>What separates people who have built this: they treat the seams between stages as the product.
The models are replaceable; the instrumentation of what each stage denied to the next is the
asset that compounds.</p>`,
  terms: [
    ["Retrieval", "Stage one. Cheap, high recall, narrows a corpus to a candidate set."],
    ["Ranking", "Stage two. Expensive scoring of the candidate set."],
    ["Re-ranking", "Stage three. Operates on the slate: diversity, dedup, rules."],
    ["Candidate coverage", "Fraction of eventual engagements that appeared in the candidate set."]
  ],
  sources: [
    ["Deep Neural Networks for YouTube Recommendations", "paper", "Covington et al., RecSys 2016. The paper that made this two-stage split the industry default. Read section 3 for the candidate generation / ranking split and skip the rest.", "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/"],
    ["Wide & Deep Learning", "paper", "Cheng et al., 2016. Where 'memorize with a linear part, generalize with a deep part' comes from.", "https://arxiv.org/abs/1606.07792"]
  ],
  mcq: {
    q: "A new ranking model scores 4% better in offline tests. Live clicks don't move at all. What do you check first?",
    o: ["Train for more epochs", "Whether retrieval surfaces the items the new ranker prefers", "Raise the candidate count to 5000", "Switch to a listwise loss"],
    a: 1,
    why: "Stage two can only reorder stage one's output. If the ranker's favourites are never candidates, the offline gain has nowhere to land. Measure candidate coverage before touching the model."
  },
  open: {
    prompt: "A colleague says: 'let's delete the retrieval stage and just rank everything with the big model.' Explain what breaks, in your own words.",
    must: [
      { name: "latency / time budget", any: ["latenc", "millisecond", "\\bms\\b", "too slow", "time budget", "speed", "fast enough", "response time"] },
      { name: "scale — cost grows with the number of items", any: ["billion", "million", "every item", "all items", "whole catalog", "entire catalog", "corpus", "scale", "n items", "number of items"] },
      { name: "cost of the expensive model per item", any: ["expensive", "compute", "cost", "gpu", "flops", "cross feature", "cross-encoder", "heavy"] }
    ],
    bonus: [
      { name: "the ranker has no independent item representation to precompute/index", any: ["precompute", "index", "cache", "ann", "cannot be indexed", "no index", "offline"] },
      { name: "quality of the funnel is capped by recall, not just precision", any: ["recall", "coverage", "candidate"] }
    ],
    traps: [
      { name: "accuracy is not the reason", any: ["less accurate", "worse accuracy", "not accurate enough", "lower quality model"], why: "The big model would actually be *more* accurate on each item. The problem is not quality — it's that you cannot afford to run it a billion times per request." }
    ],
    model: "Scoring one item with the ranking model is expensive — cross features, a deep network, sometimes an ensemble. That's fine for a thousand items and impossible for a billion, because total cost is cost-per-item times number-of-items, and the whole request has to finish in around a tenth of a second. Retrieval exists to shrink n. Its model is deliberately cheap and its item representations don't depend on the user, so they can be computed offline and indexed. Delete it and you have no way to get n down, so the request times out — the ranker being more accurate per item doesn't help."
  }
},

/* ---------------------------------------------------------------- 1.2 */
features: {
  id: "features",
  title: "Features and their combinations",
  hook: "Why is 'cheap AND five stars' worth more than 'cheap' plus 'five stars'?",
  fig: "features",
  figCap: "Start at the left: the tall columns are the embedding tables — the warehouse of cards,\none row for every user, product, brand and category, and the reason the picture is so lopsided.\nFollow one highlighted row out of each table as a lookup pulls it into the middle, where the cards\nmeet dense numeric features like price. Then trace the flow into the small interaction network on\nthe right — the few shallow layers hunting for pairings like cheap-with-five-stars — and out to the\nsingle click probability at the end. Before moving on, compare areas: the tables dwarf the network,\nwhich is the whole problem of this lesson drawn to scale.",
  beginner: `
<p>You're scrolling a shopping app for headphones when two facts about one listing catch your eye:
the price is £9, and the rating is five stars. Neither fact is interesting on its own — the
catalogue is stuffed with cheap things and stuffed with highly rated things. Together they mean
something neither means alone: a genuine bargain. Now swap one partner. A £9 watch carrying a famous
luxury brand is no bargain — it is almost certainly a counterfeit. The clue "cheap" just flipped
from good news to bad news purely because of what stood beside it. Every signal a recommender
receives — price, rating, brand, time of day, what you browsed yesterday — is called a
<b>feature</b>, and this lesson is about why the <em>pairings</em> of features carry the meaning.</p>

<p>Here is why the obvious design fails. The obvious design is a scorecard: award points for each
clue, add them up, and recommend anything above a bar — say 10 points. Try to set the points for
just two clues, "cheap" and "luxury brand". Bargain hunters click cheap items from ordinary brands,
so "cheap" on its own must be worth more than 10. Shoppers also click genuine luxury pieces, so
"luxury brand" on its own must be worth more than 10 as well. But now the £9 fake watch holds both
clues and scores over 20 — the highest score on the site, awarded to the one listing nobody should
see. <mark>No choice of points can repair this, because adding scores can never express that two
clues change each other's meaning.</mark> The scorecard isn't badly tuned; it is the wrong shape.</p>

<p>The repair is to hand the model each important pairing as a clue in its own right: put "cheap
together with luxury brand" on the scorecard as its own line, worth minus 30, and the fake sinks
while both honest listings stay up. A manufactured clue like that is called a <b>feature cross</b>.
Engineers once wrote crosses by hand; modern recommenders learn which pairings matter. To manage
clues that are names rather than numbers — a brand, a product code, your user ID — the model keeps
what is best pictured as a warehouse of index cards: one card per brand, per product, per user,
each card holding a short list of learned numbers describing that thing. Such a lookup, one card
per possible value, is an <b>embedding table</b> (the same idea behind the vectors of
<x-ref to="embed">2.1</x-ref>), and a catalogue of a billion items means a billion cards.</p>

<p>Follow one prediction through the warehouse. You open the audio section, and the model must
score those £9 headphones for you. It fetches a few dozen cards — yours, the product's, its brand's,
its category's — at, say, 32 numbers per card, and lays them beside plain numeric features like
price and rating. Then a small network, only a few layers deep, mixes the cards together — this
mixing is where cheap-meets-five-stars gets discovered — and squeezes everything down to one
number: 0.87, its guess at the chance you'll click. Notice the workload split. Fetching cards is
just lookup; the mixing costs a few hundred thousand multiplications, microseconds of arithmetic.
Yet the warehouse standing behind that instant holds tens of billions of learned numbers.</p>

<x-fig name="features_ex"></x-fig>

<p>That lopsidedness gives the classical deep recommender — the reference design is known as a
<b>DLRM</b> — a peculiar build: nearly every parameter sits in the lookup tables, while the
thinking stacked on top stays shallow. Enormous on disk, tiny in arithmetic. And here is the
failure worth remembering: as hardware got faster, engineers expected quality to climb the way it
climbed elsewhere in machine learning, and it refused to. <mark>Faster chips only run the same
shallow mixing sooner, because the model's knowledge lives in the cards — and a lookup does not
become wiser by being quicker.</mark> Language models are built the opposite way round, capacity in
the layers, so more compute keeps buying more quality — a contrast that eventually pushed the field
to rebuild recommenders in their image, the story <x-ref to="seqrec">3.2</x-ref> picks up. The
figure below draws this odd shape to scale, so start there.</p>`,
  expert: `
<p>The beginner track walked you through the warehouse and let you admire its lopsided shape.
This track puts you in charge of building one. Between that picture and a model taking live
traffic sit four decisions, and every one of them is settled by a small piece of arithmetic
that is cheap to run in advance and expensive to discover afterwards.</p>

<x-fig name="features_ex"></x-fig>

<p><b>Decision one: how do the pairings get in?</b> You might hope the network simply finds
cheap-meets-luxury-brand on its own. In practice a plain MLP over concatenated embeddings is a
poor multiplier: it has to burn depth and data re-inventing the product of two inputs before it
can exploit it. So every production family wires multiplication in somewhere — an FM term in
DeepFM, the all-pairs dot products of DLRM, cross layers in DCN where each extra layer lifts the
polynomial degree of the interactions by one, a memorising linear branch in Wide &amp; Deep. Here
is the uncomfortable truth about choosing between them: the gaps are real but small, dwarfed by
what feature quality and freshness decide. Pick the interaction module <em>last</em> — low-rank
cross layers are a sensible default, and roughly what Google reported shipping across its
rankers — and spend the argument you were about to have on the next three decisions instead.</p>

<p><b>Decision two: managed vocabulary or hashing?</b> A vocabulary service gives every user and
item its own row, at the cost of running an ID-assignment system that grows forever. Hashing is
stateless and fixed-size, but IDs start sharing rows — and the sharing arrives far earlier than
intuition says. The number to know, meaning first: it is the chance that a given ID lands in a
bucket some other ID already occupies. Written out, that is one minus e to the power of minus N
over M, where N is how many IDs you must store and M is how many rows the table has. Make the
table exactly vocabulary-sized — M equal to N — and the chance is already about 63%: two-thirds
of your entities are answering to someone else's vector. What rescues hashing in practice is
that <mark>a collision is priced by the traffic flowing through it, not by its existence</mark> —
two obscure items sharing a row costs nothing you can measure, a blockbuster sharing with anything
drags both. Reserve exact rows for the head of the distribution and let only the tail share.</p>

<p><b>Decision three: where does the memory go?</b> The tables hold nearly all the parameters, so
this model scales along the memory axis, not the compute axis — tens of billions of weights served
by microseconds of arithmetic. Three consequences follow. The tables will not fit on one machine,
so they shard across many, and each prediction becomes a scatter of network lookups; in a mature
system that lookup traffic, not the maths, is what saturates first. Row width should follow
frequency: an item observed a dozen times cannot earn a wide vector, so give the head generous
rows and push the tail into narrow ones or a shared bucket. And extra FLOPs have no purchase here
— the knowledge sits in rows a faster chip merely fetches sooner, which is why this paradigm
flatlined as hardware improved and why <x-ref to="seqrec">3.2</x-ref> rebuilds the recommender
with its capacity in the layers instead.</p>

<p><b>Decision four: how fresh is fresh enough?</b> Behavioural counters — clicks in the last
hour, conversion rate so far today — are among your strongest features and among your most
perishable: their value decays within hours. Retraining nightly quietly forfeits that value, which
is the case ByteDance's Monolith work makes with measurements — online training with minute-level
parameter sync beats a bigger, staler model. The same lifecycle thinking applies to the tables
themselves: users and items arrive forever, so without frequency filtering on the way in and
expiry on the way out, the warehouse eats the cluster. <mark>In a mature system, the age of the
features usually moves the metric more than the shape of the model does</mark> — budget your
effort accordingly.</p>

<p><b>The war story this lesson's teams live through.</b> Someone adds a counter feature —
clicks on this item today — and the offline evaluation looks like a career-making breakthrough.
It ships; online, the model is mediocre or worse. The counter was computed at end of day, so for
every training example it already contained the very click the model was supposed to predict —
the feature was a leaked answer key, useless the moment it had to be computed honestly at serving
time. The vaccination is a habit, not a heroic fix: compute features in the serving path, log
exactly what was served, and train only on that log. Point-in-time correctness is tedious, and
it is the tax that kills both leakage and training–serving skew in one payment.</p>

<p>What separates the people who have shipped this: they know the model on top is the perishable
part. Interaction modules get swapped in an afternoon; the ID space, the collision policy, the
sharded tables and the serving-time feature log are the infrastructure that every future model —
including the sequential ones of part 3 — will stand on. They build the pipeline as the product
and treat the network as this quarter's tenant.</p>`,
  terms: [
    ["Feature", "One input signal. Sparse (categorical, looked up) or dense (numeric)."],
    ["Feature interaction", "A modelled combination of two or more features."],
    ["Embedding table", "A learned lookup: one vector per category value. Where the parameters go."],
    ["DLRM", "Deep Learning Recommendation Model — the open reference architecture from Meta."]
  ],
  sources: [
    ["DLRM: Deep Learning Recommendation Model", "paper", "Naumov et al., 2019. Look at Figure 1 first — the architecture picture tells you 80% of the story before any equations.", "https://arxiv.org/abs/1906.00091"],
    ["Deep Interest Network", "paper", "Zhou et al., 2018. Introduces attention over a user's history in an ads ranker — the bridge toward part 3.", "https://arxiv.org/abs/1706.06978"]
  ],
  mcq: {
    q: "Why did adding more compute stop improving classical DLRMs?",
    o: ["They overfit", "Capacity sits in embedding lookups, so extra FLOPs have nothing to learn with", "GPUs ran out of memory", "Feature crosses can't be learned"],
    a: 1,
    why: "Memory-bound, not compute-bound. The same shallow interaction just runs faster. Fixing this required changing the paradigm, not the hardware."
  },
  open: {
    prompt: "In your own words: what does it mean that a recommender is 'big in parameters but small in compute', and why is that a problem?",
    must: [
      { name: "parameters are mostly embedding tables / lookups", any: ["embedding", "lookup", "table", "one vector per", "row per", "memory"] },
      { name: "the computation per example is shallow / small", any: ["shallow", "few layers", "small network", "little compute", "not much computation", "cheap to run", "flops"] },
      { name: "so adding compute doesn't buy quality", any: ["scal", "more compute", "doesn't improve", "no benefit", "stops improving", "diminish", "plateau"] }
    ],
    bonus: [
      { name: "contrast with language models, which do scale with compute", any: ["language model", "\\bllm\\b", "\\bgpt\\b", "transformer", "scaling law"] }
    ],
    traps: [
      { name: "confusing parameter count with model intelligence", any: ["more parameters means smarter", "bigger is better", "more parameters is better"], why: "Parameter count and useful capacity are different things here. Most of those parameters are a dictionary, not a brain." }
    ],
    model: "Nearly all the parameters are embedding tables — one row of numbers for every product, user or category — so the model is enormous on disk. But once you've looked up those rows, the network that combines them is only a few layers deep, so the actual arithmetic per prediction is small. That means throwing more GPU at it mostly makes the same shallow computation run faster, instead of letting the model learn something deeper. Language models don't have this problem: their capacity is in the layers, so more compute reliably buys more quality. Fixing it in recommenders meant changing the architecture, not the hardware."
  }
},

/* ---------------------------------------------------------------- 1.3 */
ads: {
  id: "ads",
  title: "The price of a slot",
  hook: "Why does the winner of an ad auction almost never pay what they bid?",
  fig: "ads",
  figCap: "Read the figure as one request's journey, top to bottom. First eligibility: is the ad\nallowed to appear here at all, and is this viewer's frequency cap still unspent? Then the pacing\ngate: given how fast the campaign's budget is draining, does it compete in this auction or sit it\nout? Only then the auction proper — candidates ordered by bid times predicted click chance, the\ntop one taking the slot. Notice which number sets the bill: the winner's bid decides the ranking,\nbut the runner-up's decides the price.",
  beginner: `
<p>Type "running shoes" into a search engine and look at the very top of the page: above the
ordinary results sits an entry marked "Sponsored". It was not chosen the way the rest were. In the
few milliseconds between you pressing enter and the page appearing, the platform held an auction —
several advertisers put money on that exact slot, in front of you, at this instant, and one of them
won. This lesson is about the three questions that auction must answer at that speed: who wins,
what the winner is charged, and how any advertiser's budget survives a whole day of this.</p>

<p>Start with the money. Three shoe brands want the slot: Anna bids $4.00, Ben bids $2.50, Carla
bids $1.00. Anna wins — but her bill is $2.50, Ben's bid, not her own. (In real systems the
ranking is a little richer: the platform multiplies each bid by its prediction of whether you will
click, because a $4.00 bid on an ad nobody wants earns less than a $2.50 bid on one people love —
and that prediction is exactly the model of <x-ref to="features">1.2</x-ref>. Hold the picture of
plain bids for now.) <mark>Charging the winner the runner-up's price rather than their own is the
oldest and strangest rule in advertising, and it exists to make lying pointless.</mark></p>

<p>To see why, put yourself in the auction. Suppose a click is worth exactly $3.00 to you: at any
price under $3 you profit, at any price above it you lose. First imagine the obvious rule — you
pay whatever you bid. Would you bid your honest $3? Never: winning at $3 gains you precisely
nothing, so you nudge your bid down to $2.60, or $2.10, forever guessing what everyone else will
do — the rule has turned every bidder into a poker player. Now switch to the strange rule — a
<b>second-price auction</b>: you pay
the next bid down, not your own. Bid your honest $3 and, if the best rival bid is $2.50, you win
and pay $2.50 — fifty cents of profit. Could bidding $3.60 ever help? It changes the outcome only
when a rival sits between $3.00 and $3.60 — auctions you would now win at a loss. Could bidding
$2.40 help? It changes the outcome only by losing auctions you would have won at a profit.
<mark>When your bid decides whether you win but never what you pay, bidding what the click is
truly worth to you beats every clever alternative — honesty wins by arithmetic, not virtue.</mark>
Some marketplaces do run the pay-your-own-bid rule — a <b>first-price auction</b> — instead, and there the advice flips: your bid
is your price, so you must bid below your $3 and estimate how far down you can go while still
winning often enough.</p>

<x-fig name="ads_ex"></x-fig>

<p>Now give yourself a budget: $100 to spend on these auctions today. They arrive by the thousand,
each one individually worth entering, and the morning is thick with early bargain-hunters — win
everything you can and the $100 is gone by 9am. For the rest of the day you are invisible,
including to the after-work shoppers who were your best customers, and who now see your rivals'
adverts at lower prices because you are no longer there bidding the market up. Platforms prevent
this in one of two ways. The first: before each morning auction the campaign in effect flips a
coin — heads it competes with its full honest bid, tails it sits this one out — so the money
stretches across the day; this is <b>throttling</b>. The second: the campaign enters every
auction but with its bid nudged down, so it wins fewer of them; this is <b>bid shading</b> used
as a brake. <mark>These are genuinely different levers: throttling skips some opportunities but
leaves every auction it does enter untouched — same honest bid, same price — whereas shading
changes which auctions the campaign wins and what it pays in them.</mark></p>

<p>One more safeguard fits in a single sentence: however much anyone bids, no person should be
shown the same advert forty times, so the platform counts repeats per viewer and stops serving
past a limit — the frequency cap. And here is the failure worth remembering: the advertiser who
went broke by breakfast lost not by bidding badly but by winning too eagerly — every auction
looked worth taking, and taking all of them was the mistake. <mark>In a budgeted world,
deliberately sitting out an auction you could have won is often the profitable move.</mark> The
figure below follows one ad request through the whole machine: the eligibility checks, the pacing
gate that decides whether a campaign competes at all, and the auction that picks a winner and then
bills them someone else's price.</p>`,
  expert: `
<p>The beginner track sat you in Anna's chair, deciding what one click was worth. Now take the
other chair — the platform's — where the job is not to win auctions but to design them: millions
an hour, and every rule you set quietly retrains how thousands of strangers bid. Built from that
seat, the system is four decisions, and this track takes them in the order they will actually
land on your desk.</p>

<x-fig name="ads_ex"></x-fig>

<p><b>Decision one: what does the auction sort by?</b> Sorting by bid alone hands the slot to the
deepest pocket, and a deep pocket attached to an ad nobody wants is a disaster twice over — the
platform earns nothing when the charge is per click, and the viewer's feed fills with things they
scroll past. So the sort key becomes expected revenue per impression, <code>bid × pCTR</code>:
the money on offer times the chance anyone takes it. That click predictor is the ranking model of
<x-ref to="features">1.2</x-ref> moonlighting as a pricing engine, and the moonlighting changes
what "good" means: a model that ranks perfectly but runs ten percent hot doesn't just reorder ads,
it inflates every bill downstream, because under the generalised second price rule each slot pays
the least it could have bid and still held its position — <code>p_i = b_(i+1) · q_(i+1) / q_i</code>,
the runner-up's expected revenue converted back through your own quality. Read that denominator
slowly: <mark>a higher predicted click rate is a discount — the platform charges better ads less
for the same slot, because they bring the clicks that get billed.</mark></p>

<p><b>Decision two: who pays what — and who has to be clever?</b> The beginner arithmetic proved
honesty dominant for one slot. Stretch to a ranked column of slots and the proof quietly dies:
Edelman, Ostrovsky and Schwarz showed in 2007 that GSP only resembles the honest auction, because
with several positions a bidder can shade down, drop one slot, keep most of the clicks and pay
meaningfully less — though the equilibria bidders settle into remain well behaved. VCG repairs
truthfulness by charging each winner the value its presence takes from everyone else, and the
deployment history is a lesson in switching costs: Facebook, starting fresh with no bidders to
retrain, launched on VCG, while search auctions stayed GSP largely because migrating would mean
telling every advertiser their hard-won bids were now wrong. Then 2019 inverted the question for
display: Google Ad Manager unified everything onto first price, deleting "last look" along the
way — and deleting the honesty property with it, since when your bid is your price, bidding true
value guarantees zero surplus. The burden of cleverness moved across the table: every bidder now
needs a shading model, estimated from feedback the exchange censors, because losers are mostly
told only that they lost. That is the real content of the decision — not which rule is elegant,
but which side of the market has to run the estimation problem.</p>

<p><b>Decision three: how does a budget change the bid?</b> Meaning before machinery: once money
is finite, every pound spent at breakfast is a pound that cannot chase the evening's shoppers, so
each auction carries an invisible surcharge — the opportunity cost of the budget it consumes. Call
that surcharge λ. The constrained problem — maximise total value, keep spend under the budget —
collapses to a startlingly simple policy: bid <code>v / (1 + λ)</code> everywhere, and steer the
one number λ upward when spend runs ahead of plan, downward when it lags. In production that
steering is a feedback controller, often literally a PID, watching the gap between actual spend
and a target curve shaped by forecast traffic. Notice what fell out: pacing-by-bid and shading are
the same lever held for different reasons.</p>

<p>The other actuator leaves bids alone entirely —
enter each auction with some probability and sit out the rest, which is the throttling design
LinkedIn published, chosen partly because an auction you do enter is untouched: same honest bid,
same price, incentives intact. Shading, by contrast, moves you down the ranking and changes both
which auctions you win and what they cost. This machinery is also why the bidders themselves
mutated: advertisers increasingly hand the platform a constraint — a budget, a cost target — and
autobidding systems run the λ loop on their behalf, with uniform shading provably the right move
for such constrained bidders when the underlying auction is truthful. A market where nearly every
participant is a controller holding a constraint is not the market the 2007 theory priced, and
rebuilding that theory is live research, not history.</p>

<p><b>Decision four: where does the auction end and the feed begin?</b> Someone must decide how
many slots exist at all, and that decision is made in the re-ranking stage of the
<x-ref to="funnel">1.1</x-ref> cascade, where paid results blend into the organic slate. The
honest way to think about a slot is as a displacement: every ad shown is an organic item not
shown, so the ad has to be worth more to the platform than the content it evicts — which is what
house rules like "no more than two sponsored results" are pricing crudely. Two guards live at this
boundary. Frequency caps sit before the auction, at eligibility, and they are model hygiene as
much as manners: a viewer served the same advert past fatigue stops clicking, and those dead
impressions would otherwise leak into the pCTR training data and rot the very number the auction
multiplies. And brand-new ads arrive with no click history at all, so the boundary inherits the
explore-or-exploit pressure that <x-ref to="rl">6.1</x-ref> treats properly with bandits.</p>

<p><b>The war story this lesson's demo replays in code.</b> A campaign with a healthy daily budget
and an unpaced controller went dark before the morning coffee round: the budget reset, the
controller saw a stream of individually sensible auctions, and it won them — straight through the
business-hours ramp, the most expensive impressions of the day — until the money ran out around
nine. For the remaining fourteen hours it was invisible, including through the cheap evening when
clearing prices sag, partly because rivals who paced the same way were broke too. The paced
version in the depth tab's naive-versus-paced comparison tells the ending: same budget, a
supply-weighted spend plan, a PID gate on participation — and it buys until eleven at night,
winning more impressions at a lower average price. <mark>The instinct that separates the two runs:
a pacing controller's real job is refusal — the auctions it declines to enter are what it earns
with, not the ones it wins.</mark></p>

<p>What separates people who have shipped this: they stop treating the auction as a function and
start treating it as a market that answers back. Every pacing loop's throttle moves the clearing
prices every other loop sees; every pricing-rule change moves the bids your logs were recorded
under, so replay is silent about the equilibrium you are about to provoke. The people who last in
this field instrument the feedback — pCTR calibration by slice, spend-counter freshness, the gap
between plan and delivery — because in a system built of coupled controllers, the measurements are
the product and the mechanism is just the stage.</p>`,
  terms: [
    ["Second-price auction", "The winner pays the runner-up's price. Your bid decides if you win, never what you pay — so bidding your true value is the safe strategy."],
    ["GSP", "Generalised second price: each slot pays just enough to keep its position. Looks like Vickrey; isn't truthful with multiple slots."],
    ["eCPM", "Expected revenue per thousand impressions: bid × predicted CTR. What the auction actually sorts by."],
    ["Bid shading", "Bidding below your value in a first-price auction, tuned against an estimated winning-price distribution."],
    ["Pacing", "The control loop that spreads a budget over its flight — by throttling auction participation or scaling bids."]
  ],
  sources: [
    ["Internet Advertising and the Generalized Second-Price Auction", "paper", "Edelman, Ostrovsky & Schwarz, AER 2007. Read the comparison with VCG first: why GSP merely resembles Vickrey, why truth-telling fails, and the envy-free equilibrium that rescues the mechanism anyway.", "https://www.aeaweb.org/articles?id=10.1257/aer.97.1.242"],
    ["An update on first price auctions for Google Ad Manager", "post", "Google, 2019. A market-wide mechanism change delivered as a product post: one unified first-price auction, 'last look' removed. Notice what is framed as fairness and what as simplification.", "https://blog.google/products/admanager/update-first-price-auctions-google-ad-manager/"],
    ["Budget pacing for targeted online advertisements at LinkedIn", "paper", "Agarwal et al., KDD 2014. The probabilistic-throttling recipe: forecast eligible traffic, spread the budget over the forecast, and gate participation so the auction's incentives stay untouched.", "https://dl.acm.org/doi/10.1145/2623330.2623366"],
    ["Auto-bidding and Auctions in Online Advertising: A Survey", "paper", "Aggarwal et al., 2024. Auction theory once every bidder is an algorithm holding a constraint. Start with the bidding-algorithms section; the open-problems list is a research menu.", "https://arxiv.org/abs/2408.07685"]
  ],
  mcq: {
    q: "A campaign is burning its daily budget twice as fast as planned. The platform can throttle it (enter fewer auctions) or shade it (enter every auction with lower bids). What actually differs?",
    o: [
      "Nothing — both halve the spend rate, so they are interchangeable",
      "Throttling skips opportunities but leaves each won auction unchanged; shading changes which auctions are won and what is paid",
      "Shading is impossible under a second-price rule",
      "Throttling only works for campaigns billed per click"
    ],
    a: 1,
    why: "A throttled campaign attends fewer auctions but bids its true value in each, so incentives and prices are undisturbed — LinkedIn chose throttling for exactly this reason. Lowering bids moves the campaign down the ranking, changes its win-rate and prices, and in a second-price world breaks the bid-your-value story."
  },
  open: {
    prompt: "Explain why, in a second-price auction, you should bid exactly what the impression is worth to you — and why that advice becomes wrong the day the exchange switches to first price.",
    must: [
      { name: "the price is the runner-up's bid, not your own", any: ["second.?highest", "runner.?up", "next.?highest", "someone else", "other bidder", "not (what )?you bid", "independent of (your|the) bid", "pay.*other"] },
      { name: "so truthful bidding is dominant — misreporting can't help", any: ["dominant", "truth", "honest", "no (reason|incentive|benefit)", "can'?t (do|gain|win) (better|more|anything)", "best strategy", "never helps"] },
      { name: "under first price your bid is your price, so you must bid below value", any: ["first.?price", "pay (your|their) own bid", "bid (is|becomes|sets) (the|your) price", "shad", "below (your |true |the )?value", "less than.*worth", "under.?bid"] }
    ],
    bonus: [
      { name: "shading trades win probability against margin", any: ["probabilit", "win.?rate", "chance of winning", "trade.?off", "margin", "surplus", "win less often"] },
      { name: "shading needs beliefs about others' bids / the winning-price distribution", any: ["distribution", "estimate", "landscape", "belief", "model.*(other|winning|clearing)", "guess.*(other|price)", "censored"] },
      { name: "the case analysis: overbid wins only losses, underbid loses only profits", any: ["overbid.*(loss|lose|overpay)", "win.*at a loss", "more than it.?s worth", "underbid.*(lose|miss)", "would have (won|been profitable)", "case"] }
    ],
    traps: [
      { name: "claiming second price is truthful because it is cheaper", any: ["truthful because.*(cheap|pay less|save)", "honest because.*(cheap|less money)", "platform (earns|makes) less so", "to save money"], why: "Cheapness is not the mechanism. Truthfulness comes from your bid being decoupled from your price — and revenue equivalence says the platform doesn't simply earn less under second price, because first-price bidders shade." }
    ],
    model: "In a second-price auction your bid decides only whether you win; the price is set by the runner-up, which you don't control. Bidding above your value changes the outcome only when the runner-up sits between your value and your inflated bid — precisely the auctions you'd win at a loss. Bidding below value changes the outcome only by losing auctions you'd have won profitably. So bidding your true value is dominant no matter what anyone else does. Under first price the coupling returns: your bid is your price, so bidding value guarantees zero surplus even when you win. The optimal bid drops below value — shading — and how far to shade depends on the distribution of competing bids, which you can only estimate, often from censored feedback. The 2019 switch turned bidding from an honesty problem into an estimation problem."
  }
},

/* ---------------------------------------------------------------- 2.1 */
embed: {
  id: "embed",
  title: "Meaning as coordinates",
  hook: "How do you make a computer understand that 'sneakers' and 'trainers' are the same thing?",
  fig: "embed",
  figCap: "Start at the bottom, where two separate towers stand side by side: the left one takes in\na query, the right one takes in a product, and neither can peek at the other's input. Follow each\ntower upward to the point it emits — a short list of coordinates — and see that both points land on\nthe same shared board, where their closeness is read off as a single similarity score. Then notice\nwhat the right-hand tower's isolation buys you: every product's point can be computed ahead of time\nand stored, so at search time only the left tower runs.",
  beginner: `
<p>You run an online shoe shop. A customer in London types "white trainers for the gym" into the
search box. Your catalogue was written by an American supplier, so every matching product is titled
something like "Court white leather sneakers". A search engine that works by matching words looks
at the query, looks at the title, and finds almost nothing in common. The customer sees an empty
page and leaves — even though the shop had exactly what she wanted.</p>

<p>Count the overlap and you can see the failure precisely. "White trainers for the gym" shares one
word with "Court white leather sneakers": <em>white</em>. It shares the same one word with "white
gloss paint, 2.5 litres". By word overlap, the paint and the shoes are equally good answers. The
obvious patch is a synonym dictionary — tell the system trainers = sneakers. But then you need
jumper = sweater, aubergine = eggplant, "mobile cover" = "phone case", and thousands more, in every
language you sell in, updated every time slang shifts. And no dictionary will ever tell you that
"shoes for a marathon" means running shoes, because that is not a synonym — it is an inference.
Matching the letters of words was always the wrong tool; what we need to compare is what the words
<em>mean</em>.</p>

<p>Here is the trick. Imagine a giant pinboard, and place every product on it as a pin, following
one rule: products that the same sorts of people buy, click and return go near each other. You
never read a label while pinning — position comes entirely from behaviour. Under that rule, the
trainers and the sneakers end up as neighbouring pins, because the same runners buy both; the
toaster ends up in a distant corner with the kettles. A pin's position is just a pair of numbers —
its coordinates — and that little list of numbers is called an <b>embedding</b>. Real systems use a
few hundred coordinates rather than two, but the idea survives intact: <mark>similarity of meaning
becomes closeness of position, so "find me something like this" turns into "find the nearest
pins".</mark></p>

<p>Let's actually do it with made-up coordinates. Put the query "white trainers for the gym" at
(8, 2). Product A, "Court white leather sneakers", sits at (7, 3). Product B, "white gloss paint",
sits at (1, 9). Distance to A: one step across and one step down, about 1.4 units. Distance to B:
seven across and seven up, about 9.9 units. The sneakers win by a factor of seven — despite sharing
no more words with the query than the paint did. The word-matcher had no way to prefer A over B;
the pinboard makes it obvious. Finding those nearest pins quickly among a billion is its own
problem, and <x-ref to="ann">2.2</x-ref> is about exactly that.</p>

<x-fig name="embed_ex"></x-fig>

<p>So who decides the coordinates? A model with two halves — one half reads queries, the other
reads products — trained so that a query and the product it led to land close together. This is
the <b>dual encoder</b>, or <b>two-tower</b> model. Notice something deliberate in the design: the
product half looks only at the product. It never sees the customer or the query. That restriction
is the whole business plan, because it means you can compute the pin for every product in the
catalogue tonight, store all billion positions in an index, and at search time do only one small
piece of work — embed the query and look up its neighbours. <mark>If a product's position depended
on who was asking, you would have to redraw the entire board for every single search.</mark></p>

<p>Training the towers means showing them pairs: pull a query towards the product that was clicked,
push it away from products that were not — the pushed-away ones are called <b>negatives</b>. The
most instructive negatives are near-misses that look right but are wrong, known as <b>hard
negatives</b>. And that is where the failure worth remembering lives: hunt for the very hardest
negatives and you will often dredge up products that are genuinely correct answers nobody happened
to label — and training on those teaches the model to shove right answers away from the queries
they satisfy, quietly making search worse while every training chart looks healthy. Careful teams
filter suspicious negatives first and blend hard ones with easy ones. The figure below shows the
two towers and the shared board they both write onto — walk through it with this paragraph in
mind.</p>`,
  expert: `
<p>The beginner track left you holding a picture: two towers, a shared board, meaning as
closeness. Turning that picture into a system people search with is surprisingly little
modelling and mostly four decisions — and every one of them is easy to get quietly wrong.</p>

<x-fig name="embed_ex"></x-fig>

<p><b>Decision one: what are the towers allowed to see?</b> You already know the rule — the item
side never looks at the customer — but in production it stops being a fact about the diagram and
becomes a boundary you defend in meetings. Someone will propose a harmless-sounding feature:
let the item encoding shift with the user's country, or blend in the current session. Say yes
and the nightly job that embeds the whole catalogue and builds the index searched in
<x-ref to="ann">2.2</x-ref> has nothing left to build, because there is no longer one vector per
product — there is one per product per customer. The arithmetic from
<x-ref to="funnel">1.1</x-ref> is the tiebreaker in every such argument: retrieval's budget
works out to tens of nanoseconds per item per shard,
which buys one comparison against a stored vector and nothing else. So run each proposed signal
through a single question — does it belong to the query alone, or to the item alone? Whatever
needs both at once is ranker work. <mark>Tower independence is not a choice you make once at
design time; it is a line you hold for the life of the system.</mark></p>

<p><b>Decision two: where do the matching pairs come from?</b> Nobody hand-labels "this query
goes with this product" at the millions-of-examples scale training needs, so you harvest
behaviour from your logs. A click is a weak vote, an add-to-basket a stronger one, a purchase
the strongest — and each is scarcer than the last: a month of traffic might yield fifty million
click pairs but only two million purchase pairs. Train on clicks alone and the towers learn to
imitate whatever your existing search already shows, blind spots included; insist on purchases
alone and the tail of the catalogue never appears in training at all. Shipping teams blend the
signals and then spend unglamorous effort on hygiene — discarding clicks with sub-second dwell,
collapsing repeat buys, capping how many pairs any single bestseller may contribute. That
pipeline moves recall more than any swap of encoder architecture will.</p>

<p><b>Decision three: what do you push away, and how hard?</b> Pulling queries towards their
matches is half the training; the other half needs something to push against. The free source is
<b>in-batch negatives</b>: load 1,024 pairs into a batch and each query can treat the other
1,023 items as wrong answers, so every step stages a small contest. The loss that referees the
contest is the contrastive objective, <b>InfoNCE</b>, and it is friendlier than its name. Take
the query's similarity to every item in the batch; divide each by a <b>temperature</b>, a small
knob that sharpens or softens the competition; convert the scores to probabilities with a
softmax; then charge the model the negative log of the probability it gave the true item. In
words: <em>the fine is proportional to how unconvincingly the right answer won.</em> One bias
ships free with this trick — bestsellers appear in nearly every batch, so they get shoved away
from thousands of unrelated queries and drift towards no-man's-land. The standard repair is the
logQ correction: subtract from each item's score the log of how often the sampling process
serves it up, so popularity stops being taxed.</p>

<p><b>Decision four: how much vector can you afford?</b> Width feels like a modelling knob until
you multiply it out. A billion items at 1,024 float dimensions is roughly 4 terabytes of raw
vectors before the index adds its own overhead; at 256 dimensions it is about 1 terabyte; binary-quantise
those and you are down near 32 gigabytes — the difference between a fleet of machines and a few
generous RAM sticks. Published measurements put binary vectors with a float rescoring pass at
roughly 96% of full retrieval quality, which is why serving narrow-and-coarse, then rescoring
the shortlist precisely, has become the default economy. Treat dimensionality as a product cost
with a quality dial attached, not a hyperparameter you inherit from a paper.</p>

<p><b>The war story you will live through once.</b> Eventually you will run the experiment that
sounds obviously right: mine the hardest negatives your own index can produce — the items it
places closest to each query without a click attached — and retrain on them. The loss falls
beautifully, and metrics scored against your labelled pairs stay healthy, because the labels
agree that the mined items are wrong. Then live search quietly degrades, and the diagnosis takes
longer than the damage did: the items an index ranks closest to a query with no click against
them are, disproportionately, items no user was ever shown. Your mining run dredged up correct
answers that were merely unlabelled, and training dutifully taught the towers to hold right
answers at arm's length from the queries they satisfy. <mark>The hardest negative you can find
is indistinguishable from a positive you failed to label.</mark> The teams that survive this
denoise before they train: drop mined candidates whose similarity is suspiciously high, run a
slow cross-encoder over the rest as a second opinion, and blend the hard survivors with plain
random negatives instead of going all-in.</p>

<p>What separates people who have shipped this: they treat the pair-and-negative pipeline as the
product and the encoder as the replaceable part; they version encoder and index as a single
artefact, because vectors from two different encoders share no geometry and mixing them returns
confident nonsense; and they judge every candidate model on held-out queries from their own
traffic, having learned that a leaderboard champion can shed double-digit recall the moment it
meets a catalogue it never trained on.</p>`,
  terms: [
    ["Embedding", "A vector representation. Similar inputs → nearby vectors."],
    ["Dual encoder / two-tower", "Separate encoders for query and item, joined only by a dot product."],
    ["In-batch negatives", "Using other items in the training batch as negative examples."],
    ["Hard negative", "A wrong answer that looks right. Where most of the learning signal is."]
  ],
  sources: [
    ["Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations", "paper", "Yi et al., RecSys 2019. The logQ correction, and the clearest statement of why the towers must stay separate.", "https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/"],
    ["Dense Passage Retrieval", "paper", "Karpukhin et al., 2020. Same architecture in the text world — useful for seeing that this is one idea, not two.", "https://arxiv.org/abs/2004.04906"],
    ["3Blue1Brown — Essence of Linear Algebra", "video", "Watch this if 'a space with 300 dimensions' doesn't feel like anything yet. Episodes 1–4 are enough. Nothing else explains it as well, so we won't try.", "https://www.3blue1brown.com/topics/linear-algebra"]
  ],
  mcq: {
    q: "Why must the item tower be independent of the user?",
    o: ["Fewer parameters", "So item vectors can be precomputed and indexed", "To prevent overfitting", "Because of the loss function"],
    a: 1,
    why: "Independence is what allows offline indexing. Any user–item interaction inside the encoder means you'd have to score the whole corpus at query time — which is the thing retrieval exists to avoid."
  },
  open: {
    prompt: "Someone asks: 'if hard negatives improve the model so much, why not train only on the hardest ones you can find?' What's your answer?",
    must: [
      { name: "some hard negatives are actually relevant / mislabelled", any: ["false negative", "actually relevant", "actually correct", "unlabelled", "unlabeled", "really is relevant", "true positive", "mislabel"] },
      { name: "training on them teaches the model to reject correct answers", any: ["push away", "reject", "penalis", "penaliz", "learns to avoid", "hurts recall", "worse", "degrad"] }
    ],
    bonus: [
      { name: "need denoising — threshold, cross-encoder, or human check", any: ["threshold", "cross-encoder", "cross encoder", "filter", "denois", "verify", "review"] },
      { name: "mixing easy and hard negatives / curriculum", any: ["mix", "curriculum", "ratio", "blend", "both"] }
    ],
    traps: [
      { name: "answering only 'it's slow'", any: ["too slow", "expensive to mine", "takes too long"], why: "Cost is real but secondary. The correctness problem is the one that silently destroys recall." }
    ],
    model: "Because the hardest negatives are, by construction, the items most similar to the query — and some of those are genuinely relevant, just never labelled. Training on them teaches the model to push away correct answers, so recall drops even though the loss looks like it's improving. In practice you denoise: drop candidates above a similarity threshold, or run a cross-encoder over mined negatives first, and mix hard negatives with easier ones rather than going all-in."
  }
},

/* ---------------------------------------------------------------- 2.2 */
ann: {
  id: "ann",
  title: "Searching a billion points",
  hook: "Comparing against every item is O(n). What replaces it?",
  fig: "ann",
  figCap: "Read the layers from the top down, like the journey to the lighthouse. The sparse top\nlayer is the motorway map: few points, long links, one hop covers a huge stretch of the space. Each\nlayer beneath adds more points and shorter links, down to the bottom layer where every point sits\namong its close neighbours — the local streets. A search enters at the top, greedily hops to\nwhichever linked point is nearer the query, and drops a layer each time it can get no closer;\nefSearch sets how many candidate points it keeps alive while doing so. Count the hops along the\ndrawn path — a few dozen, standing in for a billion distance checks.",
  beginner: `
<p>In <x-ref to="embed">2.1</x-ref>, every song, photo and product became a point in space, and
"things like this one" became "points near this one". So picture a music app: a listener finishes a
track, and somewhere in a cloud of a billion song-points sits the handful closest to it. The app has
roughly ten milliseconds to name them. This lesson is about how anyone finds the nearest points in a
billion without looking at a billion points.</p>

<p>Start with the honest method, because its failure is the whole story. To be certain you have the
nearest point, you measure the distance from your query to point one, then point two, and so on
through the entire collection — nothing lets you skip any, since the winner could be anywhere.
Suppose one distance measurement takes 200 nanoseconds, which is quick. A billion of them is 200
seconds: over three minutes of flat-out computing to serve one listener who expected an answer in a
hundredth of that. <mark>Checking every point is a three-minute job squeezed into a ten-millisecond
budget — twenty thousand times too slow</mark> — and buying twenty thousand machines per query is
not a plan.</p>

<p>Here is how you actually find things in a big space. Suppose a friend tells you only that she
lives nearer to the old lighthouse than anyone else in the country. You would never knock on every
door from coast to coast. You would jump to the right region, then the right town, then walk her
street asking at each corner "which way gets me closer?" — big leaps first, small steps last, and at
every step you only consider a handful of options. Searching a cloud of points works the same way
once you pre-build the roads: link the points into a web of connections, drop in anywhere, and
repeatedly hop to whichever linked point is nearer the query. This idea has a technical name,
<b>approximate nearest neighbour</b> search — <b>ANN</b> — and the most widely used road network is
called <b>HNSW</b>, a layered graph whose top layers hold a few motorway-length links between distant
regions and whose bottom layer holds dense little streets between close neighbours.
<mark>A few dozen hops down that web replaces a billion measurements</mark>.</p>

<p>Trace it with numbers small enough to hold in your head: 12 points, grouped into 3 neighbourhoods
of 4 around a centre each. A query arrives. Comparing against just the 3 centres, neighbourhood B's
is nearest, so we walk B's 4 members and the best of them sits at distance 0.31. Seven measurements
instead of twelve — but notice what could go wrong: the genuinely closest point, at distance 0.28,
lives just across the border in neighbourhood C, and we never entered C. The fix is a dial, not a
redesign: widen the search to the two nearest neighbourhoods and we make eleven measurements — the
three centres plus both neighbourhoods' eight members — find the 0.28
one, and pay a little more time for it. HNSW's version of that dial is called <b>efSearch</b> — how
many candidate points the hopping search keeps alive as it explores. <mark>Raise efSearch and you
visit more of the graph: answers get more reliable and each query gets slower; lower it and the
reverse</mark>. It is purely a query-time knob — the index's memory footprint is fixed by how many
links each point stores (a build-time setting called <b>M</b>), and the time to build the graph was
spent long before any query arrived.</p>

<x-fig name="ann_ex"></x-fig>

<p>The failure worth remembering is the borderline miss from our 12-point example, scaled up: an
approximate index will sometimes hand back the second- or fifth-nearest point and quietly skip the
true winner. That sounds alarming until you price it. Getting 99 of the true 100 neighbours in ten
milliseconds is worth far more to a live product than getting all 100 in three minutes, and in a
real system the miss is cushioned anyway — retrieval only nominates candidates, and the careful
ranking stage from <x-ref to="funnel">1.1</x-ref> re-scores whatever survives, so a near-nearest
substitute usually costs nothing a user would notice. <mark>You are not choosing between right and
wrong; you are selling a sliver of certainty for a factor of twenty thousand in speed, at a price
you set yourself</mark>. The figure below shows the road network that makes the deal possible — read
it as the lighthouse journey, motorways first.</p>`,
  expert: `
<p>The beginner track ended with you at the foot of the graph, holding a good-enough answer after
a few dozen hops. Building the thing that made those hops possible turns out to be three decisions
and one ambush, and each decision is settled by arithmetic you can do on a napkin before you touch
any software.</p>

<x-fig name="ann_ex"></x-fig>

<p><b>Decision one: do you need an index at all?</b> Under about a million vectors, exact search is
one matrix multiply — a few milliseconds on an ordinary CPU — and it is <em>correct</em>: no recall
to measure, no build step, no tuning, no rebuild schedule. That crossover sits later than almost
everyone assumes, and every hour spent tuning an index a matrix multiply would beat is an hour
stolen from the parts of the system that actually need you. <mark>Brute force until the arithmetic
forbids it</mark> is the first instinct that marks someone who has shipped this before.</p>

<p><b>Decision two: graph or partitions?</b> This sounds like an algorithms question and is really
a memory question, so do the memory sum first: corpus size × dimensions × 4 bytes. A billion
768-dimensional float32 vectors is 3,072 bytes each — about 3 TB before a single graph edge, which
no sensible machine holds in RAM. If the sum fits in memory, take <b>HNSW</b>, the layered graph
from the beginner track: <code>M</code> controls edges per node (16–32 is almost always right, and
it fixes the memory overhead), <code>efConstruction</code> controls how carefully the graph is
built, and the recall-per-millisecond it delivers on CPU is why it is the default in nearly every
vector database. If the sum does not fit, take <b>IVF-PQ</b>: partition the space with k-means so a
query probes only a few cells (<code>nprobe</code> becomes your dial), and shrink each vector with
product quantisation. The idea, meaning first: stop storing 768 measured numbers and instead chop
the vector into 96 eight-dimensional pieces, describing each piece by its nearest entry in a shared
256-entry codebook — one byte per piece. The vector becomes 96 bytes, 32 times smaller, and the
3 TB corpus becomes roughly 96 GB; distances are then estimated straight from the codes via lookup
tables, without ever decompressing. And when the corpus is billions but you want one machine, the
<b>DiskANN</b> shape splits the difference: the graph and full vectors live on SSD, compressed
codes live in RAM to steer the traversal, and the true vectors are fetched only to re-score the
final few. ScaNN's refinement is worth knowing here too — its quantiser deliberately spends its
error budget unevenly, protecting the directions that matter for inner-product scores.</p>

<p><b>Decision three: who sets the dial?</b> <code>efSearch</code> — or <code>nprobe</code> — is
the runtime trade between recall and latency, and the mistake is treating it as an engineer's
config default. First make the trade measurable: run exact brute-force search over a sample of
<em>your own</em> query traffic to get ground truth, then plot recall against latency as the dial
turns — benchmark queries drawn from someone else's corpus will flatter you. Then notice the number
is a product decision. Whether 95 per cent recall at 4 ms beats 99 per cent at 12 ms depends on
what sits downstream: inside a <x-ref to="funnel">1.1</x-ref>-style funnel, the ranker re-scores
whatever retrieval nominates, so a near-nearest substitute is usually invisible and the cheaper
setting wins. The person who owns that trade owns the user experience, not the index.</p>

<p><b>The ambush: the metadata filter.</b> The system works beautifully in the demo. Then someone
adds a perfectly reasonable constraint — in-stock items only, say, matching 3 per cent of the
catalogue — implemented the obvious way: fetch the approximate top 100, then filter. Three results
survive. The page is nearly empty, and no alert fires, because index recall is monitored without
filters and is still a healthy 0.98. Flip the order and it gets worse in a different way:
pre-filtering to the 3 per cent leaves a subgraph so sparse the traversal disconnects, and the
engine quietly degenerates towards the linear scan you built the index to avoid. The real fix is an
engine that applies the predicate <em>during</em> traversal, so the search keeps walking until it
has 100 survivors — and that is why <mark>the first question to ask any vector database is how it
filters during traversal, not what its headline QPS is</mark>. The benchmark page will not
volunteer this; production, especially <x-ref to="rag">RAG</x-ref> with its per-user document
permissions, will.</p>

<p>What separates people who have shipped this: they treat the index as a decaying artefact, not a
built one. Deletes leave tombstones that erode recall until a rebuild; a retrained embedding model
from <x-ref to="embed">2.1</x-ref> silently stales every stored vector; the recall you measured at
build time is not the recall six months later. So they version the encoder with the index, plan the
rebuild cadence on day one, and keep the brute-force ground-truth job running — because an
approximate system is only trustworthy while something exact is still watching it.</p>`,
  terms: [
    ["ANN", "Approximate nearest neighbour search."],
    ["HNSW", "Hierarchical Navigable Small World — the default graph index."],
    ["efSearch", "Runtime frontier width. The main recall-versus-latency dial."],
    ["Filtered search", "Combining metadata constraints with vector search. Where engines really differ."]
  ],
  sources: [
    ["Efficient and robust approximate nearest neighbor search using HNSW", "paper", "Malkov & Yashunin, 2016. Look at the layered-graph figure even if you skip the proofs — the picture is the algorithm.", "https://arxiv.org/abs/1603.09320"],
    ["FAISS: Billion-scale similarity search with GPUs", "paper", "Johnson, Douze & Jégou, 2017. The reference for quantization-based indexes.", "https://arxiv.org/abs/1702.08734"]
  ,
    ["SOAR: faster vector search with ScaNN","blog","Google Research, 2024. Why clustering indexes miss neighbours when a vector's residual aligns with the query, and how deliberate redundancy fixes it. The quantisation-side companion to the HNSW paper's graph side.","https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/"],
    ["Filtered vector search: what ACORN fixes","blog","Qdrant, 2026. What a metadata filter does to an HNSW graph — connectivity collapses into islands — and the two repairs, extra edges at build time versus two-hop traversal at query time, with honest benchmarks of where each loses.","https://qdrant.tech/blog/filtered-vector-search-acorn/"]
  ],
  mcq: {
    q: "You raise efSearch. What happens?",
    o: ["Higher recall, higher latency", "Lower memory use", "Faster index build", "Better metadata filtering"],
    a: 0,
    why: "efSearch widens the frontier explored during traversal: more of the graph visited, better recall, more time spent. Memory is set by M; build time by efConstruction."
  },
  open: {
    prompt: "Explain to a teammate why an 'approximate' search is the right choice here, and what you give up.",
    must: [
      { name: "exact search is linear in corpus size, too slow at scale", any: ["exact", "every item", "all points", "brute force", "linear", "o\\(n\\)", "too slow", "billion"] },
      { name: "approximate is orders of magnitude faster", any: ["faster", "milliseconds", "speed", "fewer comparisons", "hops", "efficient"] },
      { name: "you give up occasionally missing true nearest neighbours (recall)", any: ["miss", "recall", "not always", "sometimes wrong", "imperfect", "may not find"] }
    ],
    bonus: [
      { name: "the trade is tunable at query time", any: ["tune", "dial", "parameter", "efsearch", "trade-off", "tradeoff", "adjust"] },
      { name: "downstream ranking can recover from small retrieval errors", any: ["ranker", "rerank", "second stage", "downstream"] }
    ],
    traps: [
      { name: "claiming approximate is more accurate", any: ["more accurate than exact", "better than exact"], why: "It's strictly less accurate by definition. The argument is that the accuracy you lose is worth far less than the time you gain." }
    ],
    model: "Exact search means measuring distance to every point, which grows linearly with the corpus — fine for ten thousand items, impossible for a billion inside a 100ms budget. A graph index gets there in a few dozen hops instead. What you give up is recall: occasionally the true nearest neighbour isn't found. That's acceptable because the loss is small, it's tunable at query time by widening the search frontier, and a ranking stage downstream can recover from a slightly imperfect candidate set anyway."
  }
},

/* ---------------------------------------------------------------- 3.1 */
attention: {
  id: "attention",
  title: "Predict the next thing",
  hook: "How does guessing one word at a time produce a coherent argument?",
  fig: "attention",
  figCap: "Read the grid one row at a time: each row is a position choosing its next word, and the\nfilled cells along that row show which earlier positions it interviewed and how strongly. Row 6 can\nreach columns 1 to 6 and nothing beyond — the blank upper-right triangle is the causal mask, the\nhouse rule that no position may read to its right, because at writing time those words do not exist\nyet.",
  beginner: `
<p>Ask Claude to write your aunt a birthday poem and watch the reply arrive: the text streams onto
the screen a little at a time. That is not a loading animation. The model genuinely commits to one
word, then stops, re-reads absolutely everything so far — your request plus its own words — and only
then picks the next one. <mark>An entire coherent answer is built out of thousands of tiny one-word
decisions, each made with no memory of any plan, because no plan was ever made.</mark> This lesson
is about the machinery that makes such a strange procedure work.</p>

<p>Start with why the obvious approach fails. Suppose you tried to predict the next word using fixed
rules — say, "look at the previous two words and pick what usually follows them". Try it on this:
"The trophy didn't fit in the suitcase because it was too ___". If "it" means the trophy, the next
word is "big"; if "it" means the suitcase, the next word is "small". The previous two words — "was
too" — tell you nothing. Worse, no rule of grammar settles what "it" points at: swap "big" for
"small" and the referent flips, even though the sentence's structure is identical. <mark>The right
next word depends on meaning that is smeared across the whole sentence, in ways no fixed-size window
or hand-written rule can capture.</mark> Whatever mechanism predicts the next word has to be able to
consult any earlier word, and to decide for itself which ones matter this time.</p>

<p>Here is the mechanism, as a picture first. Treat each position in the sentence as a reporter
about to file one word. Before writing, the reporter is allowed to interview every word already on
the page, asking each the same question: "how much do you matter to what I say next?" Each earlier
word answers with a score, the scores are rescaled so they add up to 1 — they become shares of the
reporter's notice — and the reporter blends the interviewed words' meanings (which live as number
lists, the vectors from <x-ref to="embed">2.1</x-ref>) in exactly those proportions. This
interview-and-blend step is what researchers call <b>attention</b>. It comes with one strict house
rule: a reporter may interview words already written, never words yet to come. That rule is the
<b>causal mask</b>, and it exists for honesty, not tidiness. The model learns by practising on real
text where the "future" words are sitting right there on the page; <mark>without the mask it could
peek at the very word it is supposed to predict, score perfectly, and learn nothing except
copying</mark> — and the habit would be useless anyway at answer-writing time, when the future
words genuinely do not exist yet.</p>

<p>Now trace one prediction end to end. The sentence so far: "The bees stung the boy, so he began
to". The position after "to" runs its interviews. Made-up but sensible scores: "stung" gets 0.40 of
its notice, "boy" 0.25, "he" 0.15, "bees" 0.10, and the small connecting words share the last 0.10
— a full 1.00, spent mostly on the sting and its victim. The blend that comes back says, roughly,
"something painful just happened to a child", and from that the model's guess for the next word is
"cry". The word is appended, and the whole procedure runs again from scratch for the position after
"cry" — fresh interviews, fresh scores — which is how a different word can dominate at every step.</p>

<x-fig name="attention_ex"></x-fig>

<p>The failure worth remembering follows straight from the design. Because every word is chosen in
the moment, the final sentence of a paragraph is decided only when the model arrives there, shaped
by whatever it happens to have written above — there is no outline being consulted, so long answers
can drift, and the only fix is to put an outline into the text itself where the interviews can reach
it. The same trick, run on shopping histories instead of sentences, is where
<x-ref to="seqrec">3.2</x-ref> is headed. The figure below draws the interview grid directly: work out
which half of it the causal mask has emptied before reading on.</p>`,
  expert: `
<p>The beginner track left the reporter mid-interview. Building the interview for real comes down
to four decisions, and every one of them resurfaces later on an invoice.</p>

<x-fig name="attention_ex"></x-fig>

<p><b>Decision one: give the interview three vocabularies.</b> In the story, each position asks
every earlier word "how much do you matter to me?" — but a single embedding per word cannot play
all the parts in that exchange. Take "stung" in the bee sentence: it is at once a question-answerer
about pain, a sign advertising what it can be asked, and a package of meaning it hands over when
chosen. So the model learns three projection matrices and pushes every embedding (the vectors of
<x-ref to="embed">2.1</x-ref>) through each of them: the <b>query</b> q is the question a position
asks, the <b>key</b> k is a word's advert of what it can answer, and the <b>value</b> v is what it
actually contributes when quoted. One interview score is the dot product q·k; a row of scores
pushed through softmax becomes shares summing to 1; the output is the values blended in those
shares. Written for all positions at once: <code>softmax(QKᵀ/√d_k)V</code>, where Q stacks every
question, Kᵀ scores each question against every advert, softmax turns each row into shares, V is
the material being mixed, and d_k is the length of the q and k vectors. Notice what the formula
never mentions: word order. Shuffle the inputs and the outputs shuffle identically, so position
must be injected separately — learned offsets, or rotations folded into q and k — or the model
reads your sentence as a bag.</p>

<p><b>Decision two: keep the referee awake.</b> That √d_k is not decoration. If the components of
q and k each sit at mean 0 and variance 1, their dot product has variance d_k — at a head size of
128, raw scores swing far wider than softmax can referee. Feed it logits that large and it
saturates: one share rounds to 1, the rest to 0, and the gradient through every losing word
vanishes, so the model can never learn to redistribute its notice. Dividing by √d_k pins the score
variance at 1 whatever the head size — a one-line fix that decides whether training works at all.
Softmax has a second consequence worth logging now: because the shares must sum to 1, attention
records only <em>relative</em> importance and discards magnitude. Harmless for words; genuinely
arguable once the tokens are engagements whose intensity carries signal, which is exactly where
<x-ref to="seqrec">3.2</x-ref> picks the thread up.</p>

<p><b>Decision three: cache the past, because the mask froze it.</b> The causal mask exists for
honesty — practising on real text, an unmasked model could read the very word it is supposed to
predict and learn nothing beyond copying — but it pays two rents. In training, every position of
an n-token document becomes a legitimate exam question at once, so a single pass yields n graded
answers; that free parallelism, more than the attention pattern itself, is what retired recurrent
models. In generation, the mask means the past never changes: the key and value for position 41
are identical whether you are predicting token 42 or token 4,002, so you compute them once and
keep them. That store is the <b>KV cache</b>, and its size is arithmetic to run before ordering
hardware: 2 (for K and V) × layers × KV heads × head dimension × tokens × bytes per number. A
70B-class shape — 80 layers, 8 KV heads, head size 128, 16-bit — works out near 320 KB per token,
so one 128K-token conversation carries roughly 42 GB of cache, more than the weights of many
models it could be serving beside. <mark>Decode speed is therefore set by memory bandwidth — how
fast the chip can re-read that cache for each new token — not by arithmetic</mark>, which is why
grouped-query attention, sharing 8 KV heads where full attention would keep 64, buys an 8× smaller
cache at almost no quality cost.</p>

<p><b>Decision four: decide who pays the quadratic bill.</b> Every position interviews every
earlier one, so reading a prompt costs on the order of n² scores: a 10× longer document is about
100× the attention compute before the first token of the reply exists. Long context is marketed as
capacity and billed as cost, and the market has priced the difference honestly — providers now
sell prompt-prefix reuse directly, with cached reads on the Claude API at a tenth of the base
input price, the economics of reusable prefill published as a rate card. So the question to ask of
any long-context feature is never "does it fit in the window?" but "who pays the n² on each
request, and could a shared prefix or retrieval (<x-ref to="rag">4.1</x-ref>) shrink n before
attention ever sees it?"</p>

<p><b>The war story worth pre-living.</b> A team demos contract review by pasting whole agreements
into a long window. The demo is magnificent. Then real traffic arrives and two invoices land
together: prefill compute growing with the square of contract length, and a KV cache quietly
capping how many users fit on each GPU — throughput sags while the utilisation graphs look
healthy, the signature of a bandwidth-bound system. Worse, the accuracy audit finds the model
missing clauses from the middle of long contracts: fitting a document into the window was never
the same as the model reading it. The rescue is unglamorous — retrieve the relevant clauses first
(<x-ref to="chunk">4.3</x-ref> is that craft), cache the shared instruction prefix, keep the
window short. <mark>Treat the context window as a budget to spend, never a bin to fill.</mark></p>

<p>What separates people who have shipped this: they hold both ledgers at once. Attention the
training device — n supervised answers per pass — is what made these models possible; attention
the serving liability — a cache to store and a square to pay — is what decides whether a product
built on them survives its own success. The formula fits on a napkin. The people worth hiring can
also write the invoice underneath it.</p>`,
  terms: [
    ["Attention", "Weighted mixing over positions, weights computed from content."],
    ["Causal mask", "Blocks attention to future positions. Required for generation."],
    ["Token", "The unit being predicted — a word piece, an item ID, an action."],
    ["Context window", "How many tokens the model can attend over at once."]
  ],
  sources: [
    ["Attention Is All You Need", "paper", "Vaswani et al., 2017. Read Figure 1 and section 3.2, then come back for the rest later.", "https://arxiv.org/abs/1706.03762"],
    ["3Blue1Brown — Attention in transformers", "video", "The best visual explanation of attention that exists. We deliberately did not build our own version of this; watch it, then return to the figure above.", "https://www.3blue1brown.com/lessons/attention"],
    ["Karpathy — Let's build GPT from scratch", "video", "Two hours, and you will have written the thing. The single highest-value video in this whole course.", "https://www.youtube.com/watch?v=kCc8FmEb1nY"]
  ],
  mcq: {
    q: "When does a model decide the last sentence of a paragraph it's writing?",
    o: ["Before it starts", "While writing it, token by token, conditioned on everything already written", "Halfway through", "It drafts, then revises internally"],
    a: 1,
    why: "There is no stored plan. Each token is chosen in the moment. This also explains why models drift off-topic in long outputs — nothing is holding them to an outline unless you put one in the context."
  },
  open: {
    prompt: "Explain the causal mask: what it does, and why generation would break without it.",
    must: [
      { name: "it blocks attention to future positions", any: ["future", "later", "ahead", "upper triangle", "after it", "subsequent", "forward"] },
      { name: "without it the model sees the answer / leaks the target", any: ["cheat", "leak", "see the answer", "copy", "trivial", "already knows", "peek"] }
    ],
    bonus: [
      { name: "at inference future tokens don't exist, so training must match", any: ["inference", "generation time", "doesn't exist yet", "not available", "mismatch", "train.*test"] },
      { name: "allows all positions to be trained in one forward pass", any: ["one pass", "parallel", "simultaneously", "every position", "efficient"] }
    ],
    traps: [
      { name: "describing it as a speed optimization", any: ["makes it faster", "for speed", "efficiency reason", "saves compute"], why: "It does enable parallel training, but the mask exists for correctness first: without it the next-token task is trivially solvable by looking ahead." }
    ],
    model: "The causal mask stops each position from attending to anything after it. Without it, predicting token 5 would be trivial because the model can simply look at token 5 in its own input — it learns nothing and collapses to copying. It also has to match inference: when the model is actually generating, future tokens don't exist yet, so training with access to them would create a train/test mismatch. A useful side effect is that one masked forward pass produces a supervised signal for every position at once, which is a large part of why transformers train so efficiently."
  }
},

/* ---------------------------------------------------------------- 3.2 */
seqrec: {
  id: "seqrec",
  title: "Your history is a sequence",
  hook: "What happens when you point a language model at behaviour instead of words?",
  fig: "seqrec",
  figCap: "Read the top half as before-and-after. Before: a fixed tick-list of facts about the user feeds a scoring network, and nothing records what happened a minute ago. After: the same user as a single time-ordered stream in which items and the actions taken on them alternate as tokens, with the curved arcs showing every earlier token being consulted to fill the highlighted \"next\" slot — the arcs darken towards the right because recent behaviour weighs most. Bottom left is the chart that sold the idea: quality plotted against training compute, the older approach's dashed curve flattening out while the sequence approach keeps climbing.",
  beginner: `
<p>Meet two customers of the same online shop, Asha and Ben. Over the past year each has bought
exactly the same five things: hiking boots, a tent, a waterproof jacket, a head torch and a
paperback thriller. To an old-style recommender they are the same person — identical purchases,
identical profile, so identical suggestions for both.</p>

<p>But look at <em>when</em>. Ben bought the thriller back in January and the tent yesterday: a
camping trip is clearly coming, and the thing he needs next is a sleeping bag. Asha bought her tent
in spring and the thriller yesterday: her trip is done and she is curled up on the sofa, so the
right suggestion is another novel. The classic approach — describing a user with a tick-list of
stable facts, a so-called <b>bag of features</b> (<x-ref to="features">1.2</x-ref>) — cannot tell
them apart, because a bag has no order. <mark>Two histories made of the same events in a different
order belong to different people wanting different things next, and a system that only counts what
you did is blind to the difference.</mark></p>

<p>Here is the idea that fixes it, and it is worth carrying through the whole lesson: read a
person's history the way you read a sentence. Each thing they did is a word, the meaning lives in
the order, and the recommender's entire job becomes finishing the sentence — guessing what comes
next. That is precisely the game a chatbot plays with text, one predicted word at a time, and the
field's big move was to borrow that machinery wholesale. In the jargon this is
<b>sequential recommendation</b>: every item and every action (watched, skipped, bought) becomes a
<b>token</b>, the tokens are laid out in time order, and a model from the same family that powers
chatbots — the attention machinery of <x-ref to="attention">3.1</x-ref> — is trained to predict the
next token. <mark>The model never learns that it has stopped reading English; only the vocabulary
changed.</mark></p>

<p>Trace one tiny history through. In March you buy a kettle. In April, a teapot. This week, a box
of six mugs. The model must now finish the sentence "kettle, teapot, mugs, …". A similarity-based
system keys on your latest purchase and offers more of the same — another kettle, or a slightly
fancier one, the online-shopping absurdity everyone has met. The sequence model instead reads the
three purchases in order, picks up the story they tell — someone is kitting out a tea corner, each
step moving further from hardware towards the drink itself — and predicts the natural next word: a
tin of loose-leaf tea, perhaps, or a strainer. <mark>The guess comes from the direction the story
is heading, not from what the last item looked like.</mark></p>

<x-fig name="seqrec_ex"></x-fig>

<p>Why did this reframing cause such excitement? Because chatbot-style models come with a famous
property: give them ten times the computing power and their quality improves by a roughly
predictable amount, again and again — a pattern called a <b>scaling law</b>. Recommenders never had
one. You could beat last year's model by some percentage, but that was a one-off result, a single
number that says nothing about what the next round of hardware will buy you. The <b>HSTU</b> work
from Meta (2024) showed that once histories are treated as token sequences, the
predictable-improvement pattern appears in recommendation too. <mark>That turns progress from a
string of lucky architecture discoveries into something you can budget: spend more compute, get a
known amount better, repeatably.</mark></p>

<p>One habit from language did have to be unlearned, and it is the failure worth remembering.
Inside a chatbot, attention decides which earlier words matter by dealing out shares that must sum
to 100% — a forced split performed by a function called <b>softmax</b>. For words that is exactly
right: all you need is which words matter <em>relative to</em> the others. But behaviour has
volume. A listener who replayed one song forty times and a listener who played it once both come
out of the forced split as "100% of attention on that song" — the sheer how-much, the enthusiasm
itself, gets thrown away, and in engagement data that is often the strongest signal available.
HSTU therefore drops the forced split and lets attention keep its raw strengths, a change its
authors call <b>pointwise aggregated attention</b>. <mark>When you copy machinery between fields,
ask what it silently normalises away — one field's tidy bookkeeping is another field's
signal.</mark> The figure below shows the whole move in one picture: the old tick-list up top, the
token stream beneath it, and the rising curve that made everyone take the idea seriously.</p>`,
  expert: `
<p>Asha's thriller and Ben's tent made the case for reading histories in order. Now you have to
build the reader, and the first surprise is that hardly any of the hard choices are about the
transformer — that part you can lift almost unchanged from <x-ref to="attention">3.1</x-ref>. The
choices that will keep you up at night are about what a token is, how long you can afford to read,
and how you score the answer. This track walks through them in the order they will hit you.</p>

<x-fig name="seqrec_ex"></x-fig>

<p><b>Decision one: what is a token?</b> The cheap answer is "an item ID", which is what SASRec
used and what you should prototype with. But an item ID says <em>what</em> happened, not
<em>how</em>: a skipped song and a song replayed forty times collapse into the same token, and
those are close to opposite training signals. The HSTU answer is to interleave items with the
actions taken on them — watched, skipped, bought, dwelled — so one real event becomes two tokens.
The price is explicit: a 4,000-event history becomes an 8,000-token sequence, roughly quadrupling
attention cost for the same span of behaviour, and your item vocabulary is still a hundred million
IDs wide, which is where <x-ref to="sid">semantic IDs</x-ref> earn their keep. <mark>The
tokeniser is the actual product decision here; the transformer behind it is close to a
commodity.</mark></p>

<p><b>Decision two: how long a sequence can you serve?</b> Attention cost grows with the square
of length, so going from 512 tokens to 8,192 is not 16× the work, it is 256× — and unlike a
chatbot, you pay it for every one of the thousand candidates the funnel
(<x-ref to="funnel">1.1</x-ref>) hands you, because naively each candidate needs its own forward
pass over the whole history. That naive bill is what M-FALCON deletes: the user's sequence is
encoded once per request and the expensive pass is amortised across a micro-batch of candidates,
so a thousand scores cost roughly one long forward pass plus a thousand cheap tails — the same
instinct as prefix caching in LLM serving. That, plus an attention block leaner than the standard
one (the paper reports 5.3–15.2× faster than FlashAttention2 transformers at 8,192-length
sequences), is the only reason long histories are servable at all. Budget the sequence length
against this arithmetic before you fall in love with it.</p>

<p><b>Decision three: what do you do about the softmax over the catalogue?</b> The training loss
is the least exotic thing in the system — predict the next token, exactly as a language model
does: <code>L = −Σₜ log softmax(hₜ·E)[iₜ₊₁]</code>, where <code>hₜ</code> is the encoder's state
after reading the history so far and <code>E</code> holds every item's embedding. The catch is
that the softmax runs over the <em>entire</em> catalogue, and at 10⁸ items nobody can afford
that, so everyone samples a few hundred negatives instead. Sampling is fine — <em>if</em> you
subtract <code>log q(j)</code> from each sampled logit, where <code>q</code> is the probability
that item j got sampled. Skip that correction and the model learns to suppress popular items
exactly as hard as they were over-sampled, a bias you will misread as "the model prefers niche
content" until you find the missing term.</p>

<p><b>Decision four: does magnitude matter in your data?</b> Standard attention ends in a softmax
that forces the weights over history to sum to one — perfect for language, where all you want is
which words matter relative to the others. But engagement has volume: the forced split makes the
forty-replay listener and the one-play listener identical, both "100% on that song", discarding
the enthusiasm itself. HSTU's answer is pointwise aggregated attention —
<code>φ(QKᵀ + rab)</code> with φ a SiLU, no normalisation, plus a relative bias over both
position gap and time gap so "how long ago" is a first-class input. Weights keep their raw size,
so a user who hammered one topic produces large aggregate signal where softmax would have
flattened it. If your strongest label is intensity — replays, dwell, repeat purchase — this is
the change that pays for the rest.</p>

<p><b>The war story: the offline number that reverses.</b> Two models, and an evaluation that
ranks the true next item against 100 sampled negatives because the full catalogue is slow. Model
B beats model A, cleanly, and B ships. Krichene and Rendle (KDD 2020) showed why you should not
sleep well after that launch: sampled metrics are not a noisy version of the full-catalogue
metric, they are a <em>different</em> metric, and they can order models the opposite way — teams
have shipped the worse model while holding a genuinely better number for it. At small sample
sizes everything drifts towards AUC and the ranking you measured says little about the ranking
users see. <mark>Evaluate against the full catalogue or apply the published corrections; treat
any leaderboard that does not say which it did as decoration.</mark> The scaling-law headline —
quality tracking training compute as a power law across three orders of magnitude, 12.4% online
at 1.5T parameters — was only believable because Meta could show it on evaluations that dodge
this trap.</p>

<p>What separates people who have shipped this: they treat the glamorous part as settled and the
boring parts as the job. The architecture is a few dozen lines; the tokenisation scheme, the
sampling corrections in both training and evaluation, and the incremental-training loop that
keeps the model from going stale in a drifting catalogue are where the system actually lives —
and none of them appear in the paper's headline table.</p>`,
  terms: [
    ["Sequential recommendation", "Model the ordered action history, predict the next item."],
    ["HSTU", "Hierarchical Sequential Transduction Unit. The architecture behind generative recommenders at scale."],
    ["Scaling law", "Quality improving as a predictable power law of training compute."],
    ["M-FALCON", "Amortizing one user-sequence pass across many scored candidates."]
  ],
  sources: [
    ["Actions Speak Louder than Words", "paper", "Zhai et al., ICML 2024. The central paper of this part. Read the abstract, then Figure 7 (the scaling curves), then the HSTU block definition.", "https://arxiv.org/abs/2402.17152"],
    ["SASRec", "paper", "Kang & McAuley, 2018. The clean, small starting point. Read this before the HSTU paper, not after.", "https://arxiv.org/abs/1808.09781"],
    ["meta-recsys/generative-recommenders", "code", "Reference implementation of HSTU and M-FALCON. Read the model file; it's shorter than you expect.", "https://github.com/meta-recsys/generative-recommenders"]
  ],
  mcq: {
    q: "Why does HSTU replace softmax attention with pointwise aggregated attention?",
    o: ["Softmax is slower", "Softmax normalizes away intensity, which carries information in engagement data", "To reduce parameters", "To allow bidirectional context"],
    a: 1,
    why: "Softmax makes attention purely relative. In language that's fine. In streaming engagement data, how strongly a user engaged is signal, and normalization discards it."
  },
  open: {
    prompt: "Why is 'recommendation now has a scaling law' a bigger claim than 'this model beat the baseline by 65%'?",
    must: [
      { name: "a scaling law makes compute a predictable, repeatable lever", any: ["predict", "reliable", "repeatab", "compute", "invest", "buy.*improvement", "lever", "keep improving"] },
      { name: "a single benchmark win is a one-off / doesn't tell you what's next", any: ["one-off", "one off", "single", "point result", "doesn't tell", "no roadmap", "static", "one number"] }
    ],
    bonus: [
      { name: "same dynamic that drove LLM progress", any: ["language model", "\\bllm\\b", "\\bgpt\\b", "same as.*language"] },
      { name: "changes planning: budget hardware rather than search architectures", any: ["roadmap", "plan", "budget", "hardware", "invest", "strategy"] }
    ],
    traps: [
      { name: "treating them as equivalent claims", any: ["same thing", "both just mean", "equivalent"], why: "One is a measurement of a model. The other is a claim about the shape of future progress — it says what happens when you spend more, which is a planning statement, not a result." }
    ],
    model: "A benchmark win is one number: this model beat that model, once. A scaling law says something about the future — that if you spend 10× the compute, you get a predictable amount of extra quality, again and again. That converts research into planning: you can budget hardware instead of hoping the next architecture search finds something. It's the same dynamic that made language models improve so relentlessly, and recommenders had been visibly missing it — more compute simply wasn't buying more quality in the DLRM paradigm."
  }
},

/* ---------------------------------------------------------------- 3.3 */
sid: {
  id: "sid",
  title: "IDs that mean something",
  hook: "A new product has no history. Why should it start from a random vector?",
  fig: "sid",
  figCap: "Read the figure left to right as the shoes' code being built. The box x ∈ ℝ⁷⁶⁸ is the\nitem's position in meaning-space, produced from its text alone; the first level picks the nearest\nbroad region (our 12) and inevitably misses a little. Follow the arrow: the miss itself — the leftover — is handed to level\ntwo, which picks the code that best explains it (our 7), leaving a smaller miss for level three\n(our 3). Notice what each level is aiming at: never the original dot, always the previous level's\nerror. That is the residual idea, and it is why the numbers come out coarse-to-fine and why two\nitems sharing a prefix must share broad meaning.",
  beginner: `
<p>An hour ago, somewhere on a shopping site, a seller uploaded a brand-new pair of trail-running
shoes. Nobody has clicked them. Nobody has bought them. Nobody has even seen them. And yet tonight
the recommender has to decide, millions of times, whether these shoes deserve one of the ten slots
on somebody's screen. This lesson is about giving that decision a fighting chance.</p>

<p>Here is why the usual setup makes it nearly hopeless. When the shoes arrive, the system stamps
them with the next free number — say item #482,119,204 — and that number is pure bookkeeping. It
tells you the shoes were uploaded after item #482,119,203 and before #482,119,205, and absolutely
nothing else. Everything a recommender knows about an item normally accumulates <em>on</em> its
number, click by click, like reviews pinned to a door. <mark>A serial number with zero clicks
attached is a locked door with nothing pinned to it: the system cannot tell whether it hides
running shoes or a garden hose.</mark> Until the clicks arrive, the new item is invisible — and
because it is invisible, the clicks never arrive. That trap has a name you have probably heard:
cold start.</p>

<p>Libraries solved a version of this centuries before computers existed. Walk to the sports
section and pull a book labelled 796.42 RUN. You have never read it, no borrower has stamped its
card, yet the label already tells you it is about running — and the books shelved either side of
it, at 796.41 and 796.43, are about closely related sports. The identifier itself carries the
book's subject, and <em>closeness of numbers means closeness of topic</em>. When we build item
identifiers the same way — codes derived from what the item <em>is</em>, arranged so that similar
items get similar codes — we call the result a <b>semantic ID</b>. The contrast to hold onto:
<mark>a serial number records when an item arrived; a semantic ID records what an item is, and
only one of those is useful on day one.</mark></p>

<p>Watch it work on our shoes. First, their title, photos and description are turned into a point
on the meaning-map from <x-ref to="embed">2.1</x-ref>, landing among other trail gear. Then that
point is squashed into a short code, one number at a time. The first number is chosen from a menu
of broad regions to get as close to the point as possible — it picks 12, the footwear region.
Being one number, it is only roughly right, so the second number describes just the
<em>leftover</em>, the gap between "footwear" and where the shoes actually sit — it picks 7,
nudging towards running gear. The third number describes the error still remaining and picks 3,
the trail-specific corner. The shoes are now item (12, 7, 3), and <mark>because each number only
cleans up what the previous numbers missed, the first number is forced to carry the broadest
meaning and the later ones the finer detail</mark> — coarse to fine, exactly like the digits of a
call number. So every item starting (12, 7, …) is some kind of running shoe, and our newcomer,
one hour old, quietly inherits everything the model ever learned from years of clicks on its
(12, 7, …) shelf-mates.</p>

<x-fig name="sid_ex"></x-fig>

<p>Now the failure worth remembering. A tempting shortcut is to pick all three numbers
independently, each one aiming at the original point rather than at the leftover from the number
before. It feels equivalent; it is fatal. Three numbers all answering "roughly where is this
item?" are three blurry copies of one fact, so no number is broader or finer than another, and a
shared first number no longer promises anything about a shared category. <mark>Break the
leftover-by-leftover structure and the prefix stops meaning "same shelf" — which silently deletes
the entire cold-start benefit you adopted semantic IDs for.</mark> The figure below walks through
this build honestly, one leftover at a time, so you can see why the order of the numbers is the
whole trick.</p>`,
  expert: `
<p>The beginner track left our hour-old shoes wearing a code that already meant something. Getting
a production system to mint codes like that — and to keep them meaning what they meant — comes down
to three decisions and one habit of instrumentation, and this track takes them in the order you
will actually face them.</p>

<x-fig name="sid_ex"></x-fig>

<p><b>Decision one: which deployment path, and in which order.</b> Semantic IDs can enter your
stack in two places, and they are genuinely different projects. The incremental path treats the
tuple as features: freeze a content encoder, fit the residual quantiser once, map the catalogue to
tuples, and let your existing ranker learn embeddings for the code tokens alongside its other
sparse features — one small table per level, combined, never a table per tuple. Nothing about the
<x-ref to="funnel">1.1</x-ref> cascade changes, the whole bet fits in a single experiment, and as a
bonus a few integers per item is what makes the long histories of <x-ref to="seqrec">3.2</x-ref>
affordable. The radical path is generative retrieval in the TIGER style: a sequence model decodes
the tuple token by token under constrained beam search over the code tree, and the ANN index of
<x-ref to="ann">2.2</x-ref> simply ceases to exist — reported gains of roughly +17% Recall@5 and
+29% NDCG@5 over strong baselines, at the price of a serving problem most teams have never
operated. The de-risking order is not a compromise, it is the plan: the feature path proves the
tokeniser earns its keep, and it builds the exact assets — encoder, codebooks, item-to-tuple map —
the generative path would need anyway.</p>

<p><b>Decision two: how many levels of how many codes.</b> The arithmetic is mercifully small and
worth doing aloud. Capacity multiplies across levels: three levels of 256 codes address 256³ — about
16.7 million distinct tuples — from a vocabulary of only 768 learnable tokens, which is the whole
appeal over one flat codebook of 16.7 million entries that nothing could train or search. But
capacity on paper is not capacity in practice. Plot codes-used-per-level as a bar chart during
training: health looks like mass spread across most of the codebook; collapse looks like a skyline
with three towers, a handful of codes absorbing everything while the rest flatline at zero. A level
running at 12 codes of 256 has quietly become a twelve-way classifier, and every capacity number
you just computed is fiction. Initialising each codebook by k-means on the previous level's
residuals prevents most of it — an unglamorous line of code that outperforms most clever fixes.</p>

<p><b>Decision three: when is the tokeniser allowed to change?</b> The right first answer is
never: fit the RQ-VAE once against a frozen encoder and treat the mapping as infrastructure. Later
work co-trains — alternating tokeniser and generator optimisation, or injecting collaborative
signal into the codes — and buys real accuracy, but every step the tokeniser takes re-labels part
of your catalogue while everything downstream goes on trusting the old labels. Which exposes the
contract too few teams write down: <mark>a semantic ID is only meaningful relative to the exact
encoder and codebooks that minted it</mark>. Retrain the content encoder — even an innocent
upgrade to a better text model — and every stored tuple silently changes meaning while remaining
bit-for-bit identical in your logs. So version the encoder, version the codebooks, stamp both onto
every stored ID, and budget an encoder change as a full re-index — the same drift discipline
<x-ref to="embed">2.1</x-ref> teaches for raw embeddings, with higher stakes here because a tuple
of small integers looks so deceptively stable.</p>

<p>Beneath all three decisions sits one mechanism, best held meaning-first. The quantiser never
approximates the item twice. Level one picks the codeword nearest the embedding and is inevitably
wrong by some remainder; that remainder — not the original vector — is the only thing level two
ever sees, and level two's remainder is all that reaches level three. In symbols: start from
<code>r0 = x</code>, then at each level pick the nearest codeword to the current residual, subtract
it, and pass what is left onward; the reconstruction is just the sum of the chosen codewords.
Because level one alone must account for as much of <code>x</code> as a single codeword can, it is
forced onto the largest directions of variance — which, in a content-embedding space, is broad
category — and each later level inherits a smaller, finer error to explain. The hierarchy is not
designed in; it falls out of each level's target being the previous level's mistake. Training
needs one trick — the nearest-codeword choice has no gradient, so a straight-through estimator and
commitment terms hold encoder and codebooks together — but the residual chain is where the meaning
lives.</p>

<p><b>The war story to inoculate yourself with.</b> A team ships the feature path. Reconstruction
loss looks lovely offline and keeps improving. Cold-start recall on the very slice that justified
the project refuses to move, and for six weeks the suspects are the ranker, the feature plumbing,
the traffic split. Then someone finally plots utilisation per level and finds level one has been
running on a dozen codes since the second epoch: the tokeniser collapsed almost immediately,
reconstruction politely improved anyway because the later levels compensated, and every
"semantic" prefix in production was a near-arbitrary label. Nothing crashed and no dashboard went
red, because <mark>collapse is silent — a degenerate code is still a valid code, so only a
utilisation chart can tell you the hierarchy has died</mark>. The chart costs ten lines to log and
would have caught it on day one.</p>

<p>What separates people who have shipped this: they stop treating the tokeniser as a model and
start treating it as a registry — infrastructure with versions, contracts and audits, whose output
other systems memorise. The clever part, the residual chain, fits on an index card. The career
knowledge is that these IDs are promises, and the job is instrumenting every way a promise can
quietly break.</p>`,
  terms: [
    ["Semantic ID", "A tuple of discrete codes derived from content, replacing a random item ID."],
    ["RQ-VAE", "Residual-quantized autoencoder. Produces the hierarchical codes."],
    ["Codebook collapse", "Degenerate solution where few codes are used. The main training pathology."],
    ["Generative retrieval", "Decoding the item identifier instead of searching an index for it."]
  ],
  sources: [
    ["Recommender Systems with Generative Retrieval (TIGER)", "paper", "Rajput et al., NeurIPS 2023. The originating paper for semantic IDs in recsys. Figure 2 and 3 carry the idea.", "https://arxiv.org/abs/2305.05065"],
    ["Better Generalization with Semantic IDs", "paper", "Singh et al., 2023. The production-ranking version, including the practical hashing details TIGER doesn't need.", "https://arxiv.org/abs/2306.08121"],
    ["snap-research/GRID", "code", "Open implementation of generative recommendation with semantic IDs. Good for seeing the quantizer training loop concretely.", "https://github.com/snap-research/GRID"]
  ,
    ["Generative Recommendation with Semantic IDs: A Practitioner's Handbook","paper","Snap, CIKM 2025. The paper behind the GRID repo below: systematic ablations of every semantic-ID design choice — quantiser, codebook size, decoding. Read it when you want to know which knobs actually matter.","https://arxiv.org/abs/2507.22224"]
  ],
  mcq: {
    q: "Why does each quantization level operate on the residual rather than the original vector?",
    o: ["It's faster", "It makes the code coarse-to-fine, so shared prefixes mean shared broad meaning", "It avoids collapse", "It needs fewer codebooks"],
    a: 1,
    why: "Level 2 only has to explain the error left by level 1. That's what creates the shelf-number property where a shared prefix implies semantic similarity."
  },
  open: {
    prompt: "A teammate proposes using semantic IDs but sampling each level's code independently from its own codebook, not from residuals. What goes wrong?",
    must: [
      { name: "the codes stop being hierarchical / coarse-to-fine", any: ["hierarch", "coarse", "fine", "levels? mean", "no structure", "not nested", "flat"] },
      { name: "shared prefixes no longer imply similarity", any: ["prefix", "first code", "shared", "similar", "neighbour", "neighbor", "related"] }
    ],
    bonus: [
      { name: "cold-start benefit disappears", any: ["cold start", "cold-start", "new item", "generalis", "generaliz"] },
      { name: "levels become redundant — each re-encodes the same information", any: ["redundant", "same information", "duplicate", "repeat", "correlated"] }
    ],
    traps: [
      { name: "assuming it only affects compression size", any: ["only.*smaller", "just compression", "only about size"], why: "Compression is a side effect. The structural property — that a prefix means something — is the entire reason to use semantic IDs instead of random ones." }
    ],
    model: "Quantizing the original vector three times just gives you three noisy copies of the same information — the levels become redundant instead of complementary. Residuals are what make the code hierarchical: level 2 only encodes what level 1 got wrong, so level 1 ends up carrying broad category and later levels carry fine distinctions. Lose that and a shared prefix no longer implies two items are related, which kills the cold-start benefit — a new item can no longer land near similar items and inherit their statistical strength."
  }
},

/* ---------------------------------------------------------------- 4.1 */
rag: {
  id: "rag",
  title: "Give it the book",
  hook: "Why can't you just tell a model 'only say true things'?",
  fig: "rag",
  figCap: "Follow your parental-leave question through the diagram, left to right: the question goes first to a search over your documents, the fetched passages join the question inside the prompt, and only then does the model write — quoting the passage that supports it. Notice where the handbook sits: outside the model, in a store anyone can edit today and audit tomorrow. That placement is the whole idea, and the search arrow is the link that every later lesson teaches you to strengthen and to measure.",
  beginner: `
<p>Ask Claude a question about your own workplace — say, "how many weeks of paid parental leave do
I get at Meridian Foods?" — and watch what comes back. Claude has never seen your company's HR
handbook; it lives on an internal drive that no training run ever touched. Yet the reply arrives
fluent and specific: "Meridian Foods offers twelve weeks of paid parental leave." It sounds like a
fact. It is a guess wearing the costume of a fact, and <mark>nothing about the wording warns you
which of the two you got</mark>.</p>

<p>Two obvious fixes suggest themselves, and both collapse on contact. Fix one: train the model on
the handbook. But your leave policy was revised in March and will be revised again — nobody can
rerun an expensive training job every time a PDF gets a new paragraph, and even after the training
finished, the numbers would be smeared into the model's general habits with no page reference
attached. If the answer must reflect a document edited this morning, and must point at its source,
<mark>training simply cannot deliver that — only looking the document up at question time
can</mark>. Fix two: add "never make things up" to the instructions. This fails for a sneakier
reason. The model carries no internal marker distinguishing text it genuinely absorbed from text
that merely flows well — it does not know what it does not know, so an instruction to be truthful
gives it nothing to act on.</p>

<p>Here is the fix that works, and you already know it from school. A closed-book exam tests what a
student happens to remember, half-memories included. An open-book exam changes the rules: the
textbook sits on the desk, and full marks require pointing at the page that supports each answer.
The student and the brain are unchanged — but the honest move and the high-scoring move have become
the same move.
Doing this with a model is called <b>retrieval-augmented generation</b>: before the model writes a
word, a search step fetches the most relevant passages from your actual documents, places them into
the prompt as the open book, and the model is required to answer from those passages and name the
one it used.</p>

<p>Trace one question through. You ask about parental leave. The search step scans the indexed
handbook and hands back three passages: passage 1 covers annual holiday allowance, passage 2 is the
parental-leave section ("employees receive eighteen weeks of paid parental leave, effective 1
March"), passage 3 covers sick-day rules. The model reads all three and writes: "Eighteen weeks of
paid leave, per the parental-leave section [passage 2]." Notice what changed about your position as
the reader: <mark>you no longer have to trust the model — you can open passage 2 and check</mark>.
And when the policy changes next year, someone edits one document, the index picks it up, and every
future answer is current. No retraining happened anywhere.</p>

<p>Now run the same question when the search step misses — suppose it returns three passages about
holidays and sick days and none about parental leave. A well-behaved system says "the provided
documents don't cover this". But a model under pressure to be helpful often slides back into its
old habit and produces the twelve-weeks guess again, just as fluently as before, and the screen
looks identical to a good answer. This is the failure worth remembering: <mark>the answer can never
be better than the passages that were fetched, so the fetching step must be measured on its own</mark>
— did the right passage make it into the prompt, yes or no? — separately from how good the final
wording sounds. That is why the next two lessons exist: <x-ref to="chunk">4.3</x-ref> is about
splitting documents so the fetchable pieces carry their meaning with them, and
<x-ref to="hybrid">4.2</x-ref> is about combining two kinds of search so fewer right passages slip
through. Even when everything works, be honest about the limit — a model can still misread a
passage it was given, so this reduces invented answers rather than abolishing them. The figure
below lays the whole pipeline out; follow the question through it left to right.</p>

<x-fig name="rag_ex"></x-fig>`,
  expert: `
<p>The beginner track left the Meridian handbook open on the desk and the model pointing at
passage 2. Setting that desk up yourself turns out to be four decisions, and the first is really
a boundary line you draw before writing any code: fine-tuning changes how a model behaves — tone,
format, habits — while retrieval changes what it can see at the moment of answering. Any
requirement shaped like "reflect this morning's edit" or "point at the page" lands on the
retrieval side of the line, and no training budget moves it back.</p>

<x-fig name="rag_ex"></x-fig>

<p><b>Decision one: how many passages go into the prompt?</b> Before picking a number, look at
the arithmetic that governs it. The chance of a correct answer is the chance the right passage
was fetched, times the chance the model uses it faithfully, plus a small lucky-guess term when
it wasn't: <code>P(correct) = R·a + (1 − R)·ε</code>. Since the reader term a can never exceed
one, <mark>the whole system is capped at roughly its retrieval recall</mark> — a brilliant
generator behind a 60%-recall fetch is a 60% system wearing good prose. Stuffing in more
passages does raise R, but it quietly lowers a: models read the start and end of a long context
far better than the middle, so passage forty is close to invisible. The working compromise is
to fetch wide and pass narrow — pull a hundred-plus candidates with the cheap searches, let a
cross-encoder reranker pick, and hand the generator twenty or fewer.</p>

<p><b>Decision two: build the measuring stick before the machine.</b> Spend the first afternoon
collecting a hundred real questions and hand-labelling which passage answers each one. That gold
set is worth more than any component you could build in the same afternoon, because it lets you
score the fetch on its own — did the labelled passage reach the prompt, yes or no — separately
from how good the final wording sounds. Once you can measure, the published recipe becomes
usable: prepending document context to each chunk before indexing (the move
<x-ref to="chunk">4.3</x-ref> unpacks) cut top-20 retrieval failures from 5.7% to 3.7% in
Anthropic's study; adding a keyword search beside the embedding one
(<x-ref to="hybrid">4.2</x-ref>) took the reduction to 49%; reranking took it to 67%, down to
1.9%. Every one of those gains was only visible because someone had a labelled set to see it
with — tune nothing you cannot score.</p>

<p><b>Decision three: the citation contract.</b> The generator's side of the bargain is that
every claim names the passage supporting it, and a sentence that cannot name one is deleted, not
softened — the checking machinery lives in <x-ref to="verify">5.2</x-ref>. This contract is doing
a subtler job than tidiness. The model carries no internal flag for "I don't actually know this",
so the contract manufactures one from outside: when the fetch comes back empty-handed, "the
documents don't cover this" becomes the only answer that satisfies the rules, and the honest move
and the passing move collapse into the same move — the open-book exam's trick, now enforced by
software. If you score this with a judge model, calibrate it against human labels first; judges
reliably over-credit a fluent paraphrase of an unsupported claim, which is precisely the case
you built the contract to catch.</p>

<p><b>Decision four: whether to build retrieval at all.</b> Below roughly 200K tokens of corpus
— about five hundred pages — the honest answer is don't: put the whole thing in the prompt and
let caching absorb the cost, since re-reading a stable prefix is priced at about a tenth of
normal input. Above that line the economics flip, because you pay for every shipped token on
every query and the corpus no longer fits anyway; retrieval is what you build when the library
outgrows the desk. The line itself keeps moving as context windows grow, so revisit it yearly —
and note the agentic variants moving the other way, where the model decides mid-answer to search
again; cap those loops and check whether the extra hops change the answer before paying for them.</p>

<p><b>The war story to keep.</b> A team shipping an internal support assistant spent a month on
the generation side — new prompts, a bigger model, elaborate answer templates — while wrong
answers kept arriving in beautifully confident prose. Eventually someone did the afternoon of
labelling from decision two and measured the fetch: the right passage was reaching the prompt
40% of the time. Run that through the formula and the month explains itself — with R at 0.4,
even a flawless reader caps out around 40%, so <mark>every hour spent on the generator was spent
above a ceiling the fetch had already set</mark>. Two unglamorous weeks on chunking and hybrid
search lifted recall to the mid-eighties, and the "model" got dramatically smarter without
anyone touching the model. Generation was never the problem; it almost never is at first.</p>

<p>What separates people who have shipped this: they treat the fetch as a product with its own
metric and the generator as a replaceable part. Their gold question set and their per-query log
— what was retrieved, was the evidence present, was it cited — outlive every model swap, and
that record is the asset that compounds.</p>`,
  terms: [
    ["RAG", "Retrieval-augmented generation."],
    ["Parametric vs non-parametric memory", "Knowledge in the weights vs knowledge in a searchable store."],
    ["Faithfulness", "Whether each claim is supported by the retrieved evidence."],
    ["Grounding", "Attaching output to specific retrieved sources."]
  ],
  sources: [
    ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "paper", "Lewis et al., 2020. The original formulation. Short and readable.", "https://arxiv.org/abs/2005.11401"],
    ["Self-RAG", "paper", "Asai et al., 2023. Trains reflection tokens so the model learns when retrieval is needed and whether its output is supported.", "https://arxiv.org/abs/2310.11511"]
  ],
  mcq: {
    q: "A model must answer from a document updated an hour ago and cite it. Fine-tune or retrieve?",
    o: ["Fine-tune on the new document", "Retrieve — fine-tuning changes behaviour, not live knowledge, and can't cite", "Either works", "Neither is possible"],
    a: 1,
    why: "Freshness and provenance are retrieval properties. Fine-tuning blurs facts into behaviour, can't be updated hourly, and produces no citation."
  },
  open: {
    prompt: "Explain why 'just tell the model not to make things up' doesn't work, and what actually fixes it.",
    must: [
      { name: "the model has no internal way to distinguish true from plausible", any: ["no way to check", "cannot check", "can't check", "no mechanism", "plausib", "likely", "doesn't know.*true", "no notion of truth", "trained to predict"] },
      { name: "so give it real sources to answer from", any: ["retriev", "document", "source", "corpus", "\\brag\\b", "look up", "search"] },
      { name: "and require citation / grounding so it can be checked", any: ["cite", "citation", "provenance", "point to", "reference", "verif", "check the answer", "attribut"] }
    ],
    bonus: [
      { name: "documents can be updated without retraining", any: ["update", "fresh", "retrain", "current", "new information"] },
      { name: "reduces but doesn't eliminate hallucination", any: ["not eliminate", "still", "reduce", "doesn't fully", "not perfect", "can still"] }
    ],
    traps: [
      { name: "suggesting a bigger model fixes it", any: ["bigger model", "larger model", "more parameters", "better model would"], why: "Scale reduces the rate but doesn't change the mechanism — a larger model still has no way to tell recall from confabulation. The fix is structural, not size." }
    ],
    model: "The model was trained to produce likely text, not true text, and it has no internal signal separating 'I read this' from 'this pattern fits'. An instruction to avoid making things up has nothing to act on, because there's no check to perform. The fix is structural: retrieve real documents relevant to the question, put them in the context, and require the answer to cite which passage supports each claim. Now the knowledge lives in a store you can update and audit rather than in the weights, and a human or a second model can verify the citations. It reduces hallucination substantially — it doesn't eliminate it, since the model can still misread or over-extend a source."
  }
},

/* ---------------------------------------------------------------- 4.2 */
hybrid: {
  id: "hybrid",
  title: "Two searches, one list",
  hook: "Why does the best embedding model still fail on 'error code E-4471'?",
  fig: "hybrid",
  figCap: "Read the figure top to bottom. One query goes to two independent searches: the left\ncolumn is the keyword librarian's ranked list, the right column is the semantic librarian's. Notice\nthe raw scores attached to each list are in different units — that is why no arrow ever averages\nthem. Instead, each document earns one divided by (a constant plus its position) in every list it\nappears in, and the sums decide the final order. Find the document that sits in both columns: its\ntwo contributions add, which is how agreement between the searches outweighs one search's lone\nfavourite.",
  beginner: `
<p>Picture the search box on an appliance maker's support site. Behind it sit ten thousand manuals,
repair guides and help articles, and a frustrated customer has just typed: "my dishwasher shows
error code E-4471 and won't start". Half of that query is a precise string that appears in exactly
one document; the other half is an everyday complaint that could be phrased a hundred ways. This
lesson is about why no single kind of search can handle both halves — and the small, elegant trick
that lets two kinds work as one.</p>

<p>Start with the meaning-based search you met in <x-ref to="embed">2.1</x-ref>: it turns every
query and document into a point in space, and "nearby" means "about the same thing". Feed it
"error code E-4471" and something disappointing happens. To an embedding, that code is just a rare
token shaped like every other error code, so the search returns pages about E-4408 and E-4462 —
confident, plausible, wrong. <mark>Meaning-based search represents a rare exact string by its
neighbourhood, and the neighbourhood is precisely what you did not ask for.</mark> Old-fashioned
word matching has no such trouble: the string E-4471 either appears in a document or it does not,
so it walks straight to the one right page. Now flip the query. Search "my laptop won't switch on"
against a document titled "computer fails to boot": the two share not a single useful word, so word
matching comes back empty-handed — while meaning-based search recognises at once that they describe
the same problem. Each approach is brilliant exactly where the other is blind.</p>

<p>So imagine a library staffed by two librarians. One maintains an immaculate card index of every
word in every book: give her an exact phrase and she finds it in seconds, but paraphrase your
request and the cards are silent. The other has read the whole collection and grasps what you
<em>mean</em>, but glazes over on serial numbers and obscure codes. You would not sack either of
them — you would put the question to both. In search engines the card-index librarian is
<b>keyword search</b> (the classic recipe is called BM25) and the well-read librarian is
<b>semantic search</b>, and running them together is called hybrid search. The snag is combining
their answers. Keyword search reports its confidence as an unbounded number like 12.7; semantic
search reports a similarity between −1 and 1, like 0.83. <mark>Averaging those is like averaging a
temperature in Celsius with one in Fahrenheit — the units simply do not match — so you combine by
position in each list instead of by score.</mark> "Second place" means the same thing to both
librarians, whatever numbers they scribble in the margin.</p>

<p>Watch it work on our dishwasher query. The keyword librarian returns her top three: first the
<em>E-4471 fault code reference</em> (it contains the exact string), second the <em>dishwasher
won't start troubleshooting guide</em> (it mentions the code once), third an old parts catalogue.
The semantic librarian returns hers: first that same troubleshooting guide (it is all about
machines refusing to start), second a general article on <em>appliances that fail to begin a
cycle</em>, third a maintenance checklist. Now give each document one divided by its position, and
add across the two lists. The troubleshooting guide: 1/2 from the keyword list plus 1/1 from the
semantic list = 1.5. The fault code reference: 1/1 plus nothing = 1.0. The cycle article: 1/2 =
0.5. The guide wins, and it deserves to: <mark>a document that both searches rank highly beats a
document that only one of them loves.</mark> This one-over-the-position recipe is called
<b>reciprocal rank fusion</b> — "reciprocal" just means "one over" — and real systems add a
constant of about sixty to each position before dividing, so that a single first place cannot
steamroll broad agreement further down the lists.</p>

<x-fig name="hybrid_ex"></x-fig>

<p>Now the failure worth remembering. When a pipeline like the one in <x-ref to="rag">4.1</x-ref>
keeps fumbling queries that contain product codes, the tempting fix is a bigger, fancier embedding
model. Resist it. <mark>The blur on rare exact strings is a failure of category, not of quality —
a stronger encoder just retrieves the wrong neighbourhood with more confidence.</mark> The actual
fix costs almost nothing: add a keyword index alongside the embeddings and fuse the two lists by
position. The figure below traces this merge visually — follow one document through both columns
and see the same lesson at production settings: it runs the recipe with the constant of sixty, so
the numbers are smaller than ours, but the shape is identical — the document that appears in both
columns beats the document that topped only one.</p>`,
  expert: `
<p>The beginner track ended with two librarians and a one-over-the-position rule for adding up
their opinions. Standing up the real thing is four decisions, and each one has a wrong default
that feels reasonable right up until a customer types an error code.</p>

<x-fig name="hybrid_ex"></x-fig>

<p><b>Decision one: tune the keyword side before you touch anything else.</b> BM25 looks like a
dusty formula, but it is really two dials wrapped around one idea. The idea: a term that appears
in one document out of a million is worth a fortune, and that weighting — inverse document
frequency — is exactly what a dense encoder averages away, which is why the lexical side owns
identifiers. The first dial, <code>k1</code> (usually 1.2–2.0), sets <em>saturation</em>: the
second occurrence of a word earns less than the first, and the curve flattens fast, so a page
that repeats "dishwasher" fifty times cannot buy fifty times the score. The second dial,
<code>b</code> (usually 0.75), sets <em>length normalisation</em>: at 1 a long manual is fully
punished for its length, at 0 not at all — and for reference documentation, where the longest
page is often the right page, easing b down is a legitimate move. But the dial that actually
kills exact match is not in the formula at all: it is the tokeniser. An analyser that lowercases
and splits on hyphens turns E-4471 into two ordinary tokens, and your "exact" search quietly
stops being exact. Test the lexical side with your gnarliest real strings, never with prose.</p>

<p><b>Decision two: fuse by rank, or learn a score combination?</b> Reciprocal rank fusion —
<code>score(d) = Σ 1/(k + rank(d))</code>, summed over the lists a document appears in — is the
rank-based answer, and the constant matters more than it looks. With k around 60, first place
scores 1/61 and second place 1/62: a nearly invisible gap, which means one retriever's confident
favourite cannot trample a document that both retrievers placed mid-list. Shrink k and the top of
each list dominates; grow it and every position counts the same. The learned alternative — fit
per-retriever weights, or a small model over normalised scores — genuinely wins once you hold a
labelled query set that mirrors production traffic, because scores carry information ranks throw
away. The trade-off is fragility: normalised score fusion drifts as the corpus grows, and if a
sweep over the fusion weight shows a sharp peak, you have fitted this month's snapshot and the
peak will wander off as the document mix changes. Start with ranks; earn your way to scores.</p>

<p><b>Decision three: stratify your evaluation, or your evaluation will lie to you.</b> Split the
labelled queries into at least two classes — exact-identifier and paraphrase — and report each
separately, because they are answered by different retrievers and a pooled average lets one class
quietly collapse. This is the beginner story again wearing a metrics costume: pooled nDCG can
climb while every query containing a code gets worse, if paraphrase queries outnumber them.
The zero-shot evidence backs the paranoia — on the BEIR suite, the best dense retriever tested
beat BM25 on only 8 of 18 datasets, and your corpus was not in anyone's training mix. <mark>An
average over query classes is not a measurement, it is a blindfold; the instinct that transfers
is to name the classes first and only then read the numbers.</mark></p>

<p><b>Decision four: where does the cross-encoder pay its rent?</b> The bi-encoder from
<x-ref to="embed">2.1</x-ref> keeps query and document apart so the document side can live in an
index served by <x-ref to="ann">2.2</x-ref>; a cross-encoder reads them together and is sharply
more accurate, at a cost that scales with every candidate it reads. So it sits after fusion, not
instead of it: pull 50–200 candidates a side, fuse, let the cross-encoder reorder the fused pool,
and pass 5–15 survivors to the generator in <x-ref to="rag">4.1</x-ref>. Before you pay that
latency permanently, run the cheap experiment: generate answers with the reranker on and off. If
the downstream output barely moves, the reranker is decoration — spend the milliseconds where the
final answer actually changes.</p>

<p><b>A war story to keep.</b> One retrieval team spent a full quarter A/B-testing embedding
models — swap the encoder, re-embed, watch pooled nDCG twitch up a point, celebrate, repeat.
Meanwhile every support-ticket query carrying an error code failed, all quarter, under every
encoder, because there was no lexical index at all and codes are rare tokens each model blurred
in its own way. Pooled metrics never flinched: code-bearing queries were a modest slice of
traffic, and the encoder upgrades genuinely helped the paraphrase majority. It took one support
engineer pasting five failing tickets into a channel to reveal what a quarter of dashboards had
hidden. The fix was an afternoon: stand up BM25, fuse by rank. <mark>When a failure is
categorical, iterating on model quality is motion, not progress — the fix is a second system,
not a better version of the first.</mark></p>

<p>What separates people who have shipped hybrid retrieval: they stopped asking "which retriever
is better" — they know the honest answer is "better at which query class", so they keep a
stratified eval, a tokeniser test full of ugly identifiers, and a reranker they can justify with
an ablation. The fusion formula is the easy part; knowing which librarian just failed you, and
proving it with numbers split the right way, is the job.</p>`,
  terms: [
    ["BM25", "Classical lexical scoring. Strong on rare exact terms."],
    ["Hybrid search", "Running lexical and dense retrieval together."],
    ["RRF", "Reciprocal Rank Fusion. Merges ranked lists using positions."],
    ["Cross-encoder", "Scores query and document jointly. Accurate, not indexable."]
  ],
  sources: [
    ["Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods", "paper", "Cormack, Clarke & Buettcher, SIGIR 2009. Three pages. The whole method is one formula.", "https://dl.acm.org/doi/10.1145/1571941.1572114"],
    ["ColBERT", "paper", "Khattab & Zaharia, 2020. Late interaction — the middle ground between bi- and cross-encoders.", "https://arxiv.org/abs/2004.12832"]
  ],
  mcq: {
    q: "Why merge by rank rather than by weighted score average?",
    o: ["Ranks are more accurate", "The two score distributions are on incompatible scales", "It's faster", "Ranks are smaller numbers"],
    a: 1,
    why: "BM25 scores are unbounded positives; cosine similarity is bounded in [−1,1]. Any fixed weighting of the two is arbitrary and shifts with the corpus. Ranks are comparable by construction."
  },
  open: {
    prompt: "Your RAG system fails on queries containing product codes, even though your embedding model tops the public leaderboards. Diagnose and fix.",
    must: [
      { name: "dense retrieval blurs rare exact tokens", any: ["exact", "rare", "token", "dense.*bad at", "semantic.*miss", "blur", "similar but wrong", "paraphrase"] },
      { name: "add lexical / BM25 search alongside", any: ["bm25", "lexical", "keyword", "exact match", "sparse", "inverted index", "full.?text"] },
      { name: "fuse the two result lists", any: ["fuse", "fusion", "merge", "combine", "\\brrf\\b", "hybrid"] }
    ],
    bonus: [
      { name: "merge by rank, not by raw score", any: ["rank", "position", "not.*score", "incompatible", "different scale"] },
      { name: "leaderboard rank doesn't predict domain performance", any: ["mteb", "leaderboard", "own data", "domain", "your corpus", "benchmark.*not"] }
    ],
    traps: [
      { name: "reaching for a bigger embedding model", any: ["bigger embedding", "better embedding model", "larger model", "upgrade the embedding"], why: "This is the instinct to resist. The failure is categorical, not a matter of quality: dense retrieval represents a rare code by its neighbourhood, so a better encoder gives you a better-blurred answer." }
    ],
    model: "This is a category failure, not a quality failure. A dense encoder represents 'E-4471' by its semantic neighbourhood, so it retrieves things that look like product codes rather than that exact one — and a better encoder just does that more confidently. The fix is to add a lexical retriever like BM25, which matches rare exact strings well, run both in parallel, and fuse the ranked lists with reciprocal rank fusion. Fuse by position rather than by score, because the two scoring scales are incompatible. Also worth noting: leaderboard rank says very little about performance on your specific corpus, so evaluate retrieval on your own labelled queries."
  }
},

/* ---------------------------------------------------------------- 4.3 */
chunk: {
  id: "chunk",
  title: "Where you cut matters",
  hook: "The answer is in the corpus, the wording matches, and retrieval still misses it.",
  fig: "chunk",
  figCap: "Read panel A on the left first: the stored chunk opens with \"This applies…\", and nothing in it — not one word — is something a question about carrying over leave could match. Then panel B on the right: the added situating line names the handbook, the section and the topic, and those are precisely the words an employee's question arrives carrying. The passage itself is untouched between the two panels; findability came entirely from the one line the chunker prepended.",
  beginner: `
<p>A company wires a chatbot to its staff handbook. An employee asks: "Can I carry over unused
leave in my first year?" The handbook settles this question explicitly — it even contains the
exact phrase "carry over unused leave" — and still the bot either finds nothing useful or,
worse, confidently gives half the rule. The question is fine. The model is fine. <mark>The
document was quietly broken before either of them got involved</mark>, at a preparation step most
people never think about, and this lesson is about that step.</p>

<p>Here is why it happens. A search system never compares your question against a whole book at
once; before anything is stored, each document is sliced into smaller pieces, and every piece is
matched against the question on its own (<x-ref to="embed">2.1</x-ref> explains how a piece of
text becomes something a computer can compare). Now watch what one careless slice does to two
harmless sentences from the handbook: "Employees may carry over unused leave. This applies only
after two full years of service." Suppose the knife falls between them — a page ends there, or a
size limit was hit. The first piece now announces that carrying over is allowed, with no
conditions attached. The second piece says "This applies only after two full years of service" —
applies to <em>what</em>? <mark>Neither half can answer the question on its own, and the search
only ever sees the halves.</mark></p>

<p>Picture tearing a recipe card in two between the ingredients and the method. Hand a friend
just the bottom half and they read "simmer gently for twenty minutes" — twenty minutes of what?
Hand them just the top half and they hold butter, flour and lemons with no idea what to do. Both
halves are perfectly legible, and neither is a recipe any more. In retrieval, each stored piece
of a document is called a <b>chunk</b>, and deciding where the knife falls is called
<b>chunking</b>. Keep the recipe card in mind throughout: <mark>a good chunk is a complete
thought, and a cut placed mid-thought turns every piece into half a recipe.</mark></p>

<p>Let's run the experiment end to end. Our entire miniature handbook section is 49 words —
paragraph one: "Annual leave must be requested through the staff portal at least two weeks in
advance. Your manager will confirm within three working days. During busy periods, requests are
granted in the order they arrive." Paragraph two: "Employees may carry over unused leave. This
applies only after two full years of service." <b>Chunking A</b> cuts every 40 words, and word
40 falls exactly at the end of the first carry-over sentence: chunk A1 is the whole booking
paragraph plus "Employees may carry over unused leave.", and chunk A2 is the orphan "This
applies only after two full years of service." <b>Chunking B</b> cuts at the paragraph break
instead: chunk B1 is the booking paragraph, and chunk B2 keeps both carry-over sentences
together. Now ask both versions the same question: "Can I carry over leave in my first year?"
Under B, chunk B2 matches on "carry over" and "leave", arrives whole, and the system answers
correctly: not yet — two full years first. Under A, chunk A1 matches those same words, so
retrieval proudly returns it and the system answers "yes", because the restriction lives in A2 —
which shares not a single word with the question and is never fetched. <mark>Same document, same
question, same search engine; only the position of the cut changed, and it flipped the answer
from right to wrong.</mark></p>

<x-fig name="chunk_ex"></x-fig>

<p>Three habits keep you out of this trap. First, cut where the document already has seams:
paragraph breaks, headings with their sections, list items — and over a codebase, a whole
function or class per chunk, since the programming language draws those boundaries for you.
Second, and this rescues even pieces that had to be cut awkwardly, <mark>prepend a short
situating sentence to every chunk before it is stored — one line naming the document, the
section and the topic.</mark> Give our orphan the header "Staff handbook, annual leave:
carry-over rules" and the words the question needs are suddenly stapled to it. Third, keep each
chunk's address — source, section, page — as structured metadata beside the text, so results can
be filtered by document and, just as importantly, cited. So when a chunk holds the exact answer
yet never gets retrieved, repair the data first — situating sentence, metadata — long before
shopping for a bigger model.</p>

<p>The failure worth remembering is that a bad cut makes no noise. Nothing crashes and no error
is logged; the severed sentence sits in the index the whole time while every first-year employee
who asks receives the wrong answer, and you will be tempted to blame the model at the end of the
pipeline (<x-ref to="rag">4.1</x-ref> shows how the generation step inherits whatever retrieval
hands it). The figure below puts the rescue under a magnifying glass: the same orphaned passage
twice, once bare and once wearing its situating line, so you can point at the exact words that
turn it findable.</p>`,
  expert: `
<p>The beginner track watched one careless cut flip a right answer to a wrong one. When you build
the ingest pipeline yourself, that cut stops being an accident and becomes a choice — four
choices, really, all made before a single query arrives, and all more expensive to revisit than
any model swap, because changing your mind about chunking means re-embedding the corpus. This
track walks through them in the order they will actually confront you.</p>

<x-fig name="chunk_ex"></x-fig>

<p><b>Decision one: how wide is a chunk?</b> Treat width as a dial between finding and informing,
because that is what it is. A narrow chunk embeds as a sharp, single-topic vector
(<x-ref to="embed">2.1</x-ref> explains why mixed topics blur one), so it matches queries
beautifully — and then arrives at the generator carrying two sentences when the answer needed
twelve. A wide chunk brings the whole neighbourhood but smears several subjects into one vector
that matches nothing crisply. The trap is assuming one unit must serve both jobs. It doesn't:
the <b>parent-document pattern</b> retrieves on the narrow unit and then hands the generator the
enclosing section that unit came from. Search with the scalpel, answer with the page. Once you
separate the two roles, most agonising over the "right" chunk size simply dissolves.</p>

<p><b>Decision two: where does the knife fall?</b> The author already did this work. Headings,
paragraph breaks, list items, contract clauses, function definitions — every corpus arrives
pre-scored with the boundaries of its own complete thoughts, and a chunker that ignores them to
count characters is discarding free labour. The evidence is comfortingly boring: Chroma's
evaluation found a plain recursive splitter, sensibly sized, competitive with far cleverer
methods — while a popular library default of 800-token chunks with 400 tokens of overlap turned
in below-average recall and the worst precision of anything they measured. Sit with what that
overlap means before admiring it as insurance: duplicating 400 of every 800 tokens stores your
corpus nearly twice over, so you pay embedding twice and search a fatter index, all for a recall
change Chroma found marginal. A default is somebody else's experiment on somebody else's
documents; run your own.</p>

<p><b>Decision three: does each chunk know where it lives?</b> A chunk ripped from its document
loses its referents — the pronouns, the section it sits under, the product its "this" pointed at
— and no embedding model can recover what the text no longer contains. The repair is the
situating line: at ingest, ask a small model to write one sentence saying what this chunk is
about given the whole document, and index the two together. The cost lands once, at ingest —
Anthropic priced it at about $1.02 per million document tokens with prompt caching — and what it
buys is a 35% drop in top-20 retrieval failures (5.7% down to 3.7%), stretching to 49% when
contextual BM25 rides along (<x-ref to="hybrid">4.2</x-ref>). Read that trade the way a
practitioner does: roughly a dollar per million tokens, paid one time, against wrong answers
that recur on every unlucky query forever. It is the cheapest large win in this entire
pipeline.</p>

<p><b>Decision four: what is the citation contract?</b> Store each chunk's address — source
document, section path, page or line range — as structured fields beside the text, and design
that schema before you pick a vector database, not after. Metadata is not decoration. It is
what lets search restrict itself to the right document before comparing a single vector, and it
is the only reason the generator in <x-ref to="rag">4.1</x-ref> can ever say <em>where</em> an
answer came from. A retrieval system that cannot cite is a rumour mill with good latency, and
no amount of embedding quality retrofits an address onto a chunk that was stored anonymous.</p>

<p><b>The war story you will live through once.</b> A team's retrieval eval keeps missing, and
the eval appears to deliver a verdict: the current embedding model scores poorly, a bigger one
scores better, procurement begins. Then someone does the unglamorous thing — pulls up the failed
queries and actually reads the chunks that should have matched. Every gold answer had been
guillotined mid-fact by a fixed-width chunker; the "better" model was merely better at
retrieving fragments of evidence, which is not the same as retrieving evidence. Re-chunked on
document structure, the original cheap model passed the same eval. The data was guilty, and the
model very nearly served its sentence. The habit that prevents this costs an afternoon:
hand-label a hundred or so query-to-chunk pairs, track recall at k over them, and
<mark>whenever retrieval misses, read the stored chunks for the failing queries before you shop
for a model</mark> — the bug is usually visible to the naked eye.</p>

<p>What separates people who have shipped this: they treat the chunker as versioned
infrastructure, not a preprocessing footnote — because every downstream stage inherits its
mistakes and none can undo them. They have read their own index the way a chef tastes the stock.
And when the system answers wrongly, their reflex runs upstream: <mark>data first, model
last</mark>, because in retrieval the model is rarely the culprit and always the most expensive
suspect to charge.</p>`,
  terms: [
    ["Chunk", "The stored, retrievable unit of a document."],
    ["Contextual retrieval", "Prepending a situating summary to each chunk before embedding."],
    ["Metadata filtering", "Constraining search by structured fields alongside the vector."],
    ["Recall@k", "Fraction of queries whose correct chunk appears in the top k."]
  ],
  sources: [
    ["Introducing Contextual Retrieval", "blog", "Anthropic, 2024. Short, practical, with before/after numbers. Implement this before trying anything more exotic.", "https://www.anthropic.com/news/contextual-retrieval"],
    ["RAPTOR", "paper", "Sarthi et al., 2024. Recursive summary trees, so retrieval can happen at several levels of abstraction.", "https://arxiv.org/abs/2401.18059"]
  ,
    ["Late Chunking in Long-Context Embedding Models","blog","Jina, 2024. Run the transformer over the whole document first, then pool into chunk vectors afterwards, so every chunk keeps its referents. Same disease as contextual retrieval, cured from inside the model instead of by prepending text.","https://jina.ai/news/late-chunking-in-long-context-embedding-models/"]
  ],
  mcq: {
    q: "A chunk contains the exact answer but is never retrieved, and the wording matches the query. Best first move?",
    o: ["Use a larger embedding model", "Add a situating sentence and structured metadata before embedding", "Make chunks smaller", "Increase k"],
    a: 1,
    why: "The chunk lost the referents that made it findable. Restoring context is a data fix, costs almost nothing, and typically outperforms any model upgrade."
  },
  open: {
    prompt: "You're building RAG over a codebase. Describe your chunking strategy and justify each choice.",
    must: [
      { name: "chunk on semantic units — functions, classes, files", any: ["function", "class", "method", "module", "semantic unit", "logical", "block", "definition"] },
      { name: "add context — file path, module, surrounding purpose", any: ["file path", "filename", "context", "situat", "header", "module name", "what it does", "docstring", "summary"] },
      { name: "keep metadata for citation / filtering", any: ["metadata", "path", "line number", "cite", "citation", "repo", "language", "filter"] }
    ],
    bonus: [
      { name: "handle chunks larger than a function — split with overlap or summarize", any: ["overlap", "too long", "split", "large function", "summar", "hierarch"] },
      { name: "evaluate retrieval on labelled query→chunk pairs", any: ["evaluat", "recall@", "measure", "test set", "golden", "labelled", "labeled"] }
    ],
    traps: [
      { name: "fixed character or token windows", any: ["500 char", "fixed size", "every \\d+ (char|token)", "sliding window of \\d+", "chunk size of \\d+"], why: "Fixed windows cut through function bodies and separate signatures from their implementations. In code, the syntactic boundaries are handed to you for free — use them." }
    ],
    model: "Chunk on syntactic boundaries — one function or class per chunk — because those are already complete thoughts with names, and the parser gives you the boundaries for free. Prepend context: the file path, the module, the class it belongs to, and a one-line description of what it does, so a chunk isn't an anonymous body of code. Store metadata for filtering and citation: repo, path, line range, language. For functions too long for one chunk, split with overlap and repeat the signature in each part. Then build a small labelled set of question-to-function pairs and measure Recall@k, because you can't tune any of this by intuition."
  }
},

/* ---------------------------------------------------------------- 5.1 */
agent: {
  id: "agent",
  title: "The loop",
  hook: "What actually separates an agent from a very good chatbot?",
  fig: "agent",
  figCap: "Read the circle first: a thought chooses an action, the action goes out through a tool, and whatever comes back — result or error — is the observation that feeds the next thought. Then read the box around the circle: the turn cap, the spend budget and the tool timeouts sit outside the loop on purpose, because they are the guarantees the prompt cannot provide, and the check at the exit is the verifier that audits the answer before anyone acts on it.",
  beginner: `
<p>Ask an ordinary chatbot "is my 14:30 meeting still on?" and watch what it does. It answers —
fluently, politely, instantly — and the answer is worthless, because a chatbot's only ability is
producing sentences. It has never seen your calendar. Now ask Claude the same question when it has
been connected to your calendar: it opens the calendar, reads today's entries, notices the 14:30 slot
was cancelled an hour ago, and tells you so. <mark>Same question, same language skills — the
difference is that one of them can go and check.</mark> This lesson is about that difference.</p>

<p>Why does talking alone fail? Because a language model on its own is sealed off from the world.
Everything it says is assembled from patterns it absorbed during training, which ended before today
existed. Your meeting, your inbox, this morning's prices — none of it is in there. Worse, the model
won't say "I can't see that"; it will produce something confident-sounding anyway, because producing
confident-sounding text is precisely what it was built to do. A brain in a jar, however articulate,
cannot look anything up.</p>

<p>Here is the fix, and it's humbler than it sounds. Picture a new employee on their first day.
They don't know where anything lives, so they work in a rhythm: say what they're about to try
("I'll look in the shared drive"), try it, study what actually happened ("empty folder — so the
reports must be kept somewhere else"), and adjust before the next attempt. Nobody handed them a
complete plan; the plan emerges one attempt at a time. That rhythm is the whole idea. When Claude
works this way we call it the <b>agent loop</b>: <b>think</b> (decide the next step), <b>act</b>
(actually do it), <b>observe</b> (read what came back), and repeat. The acting happens through
<b>tools</b> — a search function, a calendar API, a bit of code Claude can run — which are the hands
attached to the brain in the jar. And one rule governs what the hands do versus what the brain does:
<mark>when a question has exactly one correct answer — the number of business days between two
dates, a sum, a database lookup — a tool computes it, because a model asked to count will instead
compose something that merely sounds counted.</mark></p>

<p>Watch the loop run on a small job: "find the cheapest of these three flights and hold a seat."
<em>Think:</em> three airlines, so I need three prices before I can compare. <em>Act:</em> call the
fare tool for airline A. <em>Observe:</em> £212. <em>Think:</em> one down. <em>Act:</em> fare tool
for airline B. <em>Observe:</em> £185 — new cheapest. <em>Act:</em> fare tool for airline C.
<em>Observe:</em> error — "unknown route code". <em>Think:</em> the request failed; that doesn't
mean no flight exists, it means I wrote the route wrong, so look up C's code for this airport and
retry. <em>Observe:</em> £169. <em>Think:</em> C wins. <em>Act:</em> call the hold-seat tool.
<em>Observe:</em> confirmation number — done. Notice what happened at the error: <mark>the failure
came back as just another observation, and the next thought steered around it</mark>. A script would
have crashed there. The loop's entire value is that it reads its own results, including the bad
ones, and corrects course.</p>

<x-fig name="agent_ex"></x-fig>

<p>That self-correction has a dark twin: self-destruction. Suppose the retry had failed too, and the
one after that. A loop with no boundary will happily repeat a doomed action all night, spending real
money on every lap. You might think the cure is writing "give up after three failures" into the
instructions — but an instruction in a prompt is advice, and advice is something the model follows
<em>usually</em>. "Usually" is not a guarantee, and the rare run that ignores it is exactly the one
that burns your budget overnight. <mark>So real systems put the limits outside the loop: a hard cap
on how many turns the agent may take and how much it may spend, enforced by plain code the model
cannot talk its way past.</mark> The prompt shapes behaviour; the cap ends it.</p>

<p>One more failure worth carrying with you: small errors multiply. A step that goes right 95% of
the time feels safe, but chain twenty such steps and only around a third of runs come through with
no mistake at all — each wobble feeds the next thought, and the loop can drift far from where it
should be while sounding reasonable the whole way. That is why serious agents end with a checker
that audits the finished answer against evidence — the second opinion of
<x-ref to="verify">5.2</x-ref> — rather than trusting twenty consecutive strokes of luck. The
figure below draws all of this in one picture: the think–act–observe circle in the middle, and the
rails around it that keep the circle honest.</p>`,
  expert: `
<p>The beginner track left the loop holding a seat reservation and looking rather clever. Between
that demo and anything you would connect to a real booking system sit four decisions, and this
track takes them in the order they will actually find you.</p>

<x-fig name="agent_ex"></x-fig>

<p><b>Decision one: should this be an agent at all?</b> The honest answer, most days, is no. Fetch
three fares, compare, hold the cheapest — you could list those steps before running any of them,
and a step list you can write in advance is just a programme: ordinary code with a model call
placed exactly where judgement lives, testable, predictable, and a fraction of the cost. Anthropic's
own guidance on building agents is unusually blunt here — most production wins come from workflows,
model calls arranged along paths you fixed beforehand. The open loop earns its keep only when the
path depends on what each action reveals: debugging an unfamiliar codebase, research where the
second question is written by the first answer. When the flowchart is drawable, code the flowchart;
save the loop for problems that refuse to be drawn. And whatever you build, the arithmetic stays
out of the model — a date difference or a fare comparison has exactly one right answer, so a tool
computes it and the model merely decides to ask.</p>

<p><b>Decision two: what does the model actually experience?</b> Prompts get all the attention, but
inside a loop the interface the model lives with is the tool result — each observation is the only
evidence the next thought has. So design results the way you would brief a colleague: did it work,
what came back, what would a sensible next move be. That last clause is where errors stop being
failures and become information. "Unknown route code — codes for this airline look like LHR-BOM"
is a course correction; a bare stack trace is a dead end wearing a uniform. SWE-agent made this
measurable: the same model with a better-shaped file viewer and edit commands solved far more real
GitHub issues, which promoted tool design from taste to result. <mark>Treat every tool result,
errors included, as a message you are writing to the model</mark> — and prefer a few consolidated
tools returning structured answers over a thicket of thin endpoints, because forty overlapping
tools turn tool choice into a harder problem than the task.</p>

<p><b>Decision three: what do you do about the multiplication?</b> Finish the sum the beginner
track started. A step that goes right 95% of the time feels solid; run twenty of them and the
chance nothing goes wrong is 0.95 to the twentieth power, about 0.36. In words: a per-step
reliability nobody would apologise for quietly becomes a two-in-three failure rate at agent length
— and reality is usually worse, because the steps are not independent; one polluted observation
leans on every thought after it. You have three levers. Shorten the chain: checkpoints that verify
progress mid-run mean an error costs you the distance back to the last good state, not the whole
journey. Raise the per-step odds: better tools and recoverable errors do more here than a smarter
model. And gate the exit: a verifier that audits the finished answer against evidence
(<x-ref to="verify">5.2</x-ref>) converts "probably fine" into "checked". Note what the caps from
the beginner track do <em>not</em> do — they never improve these odds; they bound the price of the
runs that were doomed regardless. Pushing the per-step number itself is now a training problem,
reward assigned across whole trajectories, which is <x-ref to="rl">6.1</x-ref>'s territory.</p>

<p><b>Decision four: what is the loop allowed to remember?</b> Everything that enters the context
window stays there steering later thoughts — turn three's stale directory listing whispering into
turn forty — and long runs outgrow the window entirely. So production agents compact: the
trajectory gets summarised and the run continues from the summary. What survives that summary is a
genuine design decision, the loop's version of the attention budget from
<x-ref to="attention">3.1</x-ref> — decisions made and questions still open survive; raw
transcripts do not. It also quietly reframes retrieval: inside an agent,
<x-ref to="rag">4.1</x-ref> stops being the architecture and becomes one tool among several, called
when the loop needs something it never saw.</p>

<p><b>The war story, because everyone gets one.</b> An agent watching a data feed met an API that
started returning 503s at ten in the evening. The prompt said, in perfectly clear English, "if a
tool fails three times in a row, stop and report". For hundreds of runs the model had honoured
that sentence; this run, mid-recovery, it reasoned its way into "the service seems flaky, I should
keep trying" and retried in a tight loop until morning. The day's budget was gone before anyone
was awake to notice. The post-mortem fits in one line: the limit lived in the prompt, where it was
a preference the model weighs, instead of in the runtime, where it would have been a wall. Every
limit you actually depend on — turns, spend, per-tool retries, timeouts, a kill switch that needs
no deploy — belongs in code outside the loop. The prompt shapes what the agent tries; the runtime
decides what it may.</p>

<p>What separates people who have shipped this: they debug trajectories, not answers. A wrong
final output tells you almost nothing; the stored record of every thought, action and observation
tells you which turn went sideways and what the model was looking at when it did. The loop itself
is fifty lines anyone can write in an afternoon. The tools, the enforced limits and the archive of
what actually happened are the part that compounds.</p>`,
  terms: [
    ["ReAct", "Reason + act interleaved in one trajectory."],
    ["Tool", "A function the model can call. Deterministic work belongs here."],
    ["max_turns", "Hard cap on loop iterations. The seatbelt."],
    ["Trajectory", "The full recorded sequence of thoughts, actions and observations."]
  ],
  sources: [
    ["ReAct: Synergizing Reasoning and Acting", "paper", "Yao et al., 2022. The pattern every agent framework implements.", "https://arxiv.org/abs/2210.03629"],
    ["Building effective agents", "blog", "Anthropic engineering. Unusually honest about when a workflow beats an agent — read it before you build one.", "https://www.anthropic.com/engineering/building-effective-agents"],
    ["Hugging Face AI Agents Course", "course", "Free and certificated. Unit 1 teaches exactly this loop; do it hands-on rather than reading about it.", "https://huggingface.co/learn/agents-course/en/unit1/introduction"]
  ,
    ["Effective context engineering for AI agents","blog","Anthropic engineering, late 2025. Why long-running agents degrade as context grows, and the fixes that actually ship: compaction, note-taking, sub-agents. The sequel to the piece above.","https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents"],
    ["Writing effective tools for agents","blog","Anthropic engineering. Tools are contracts between deterministic code and a non-deterministic caller — this is how to design and evaluate them, learnt from building Claude Code.","https://www.anthropic.com/engineering/writing-tools-for-agents"]
  ],
  mcq: {
    q: "Which of these belongs in a tool rather than in model output?",
    o: ["A summary of a document", "The number of business days between two dates", "A gift suggestion", "An explanation of a concept"],
    a: 1,
    why: "Exactly one right answer, trivially computable, and embarrassing to get wrong. Everything with that shape goes to a tool."
  },
  open: {
    prompt: "Your agent gets stuck retrying a failing tool and burns through a large budget overnight. What do you change, and why isn't a better prompt enough?",
    must: [
      { name: "hard caps — max turns and/or token budget", any: ["max.?turn", "turn limit", "budget", "cap", "limit", "maximum", "quota"] },
      { name: "prompts are advisory, not enforcement", any: ["advisor", "not enforce", "suggestion", "can ignore", "doesn't guarantee", "no guarantee", "soft", "cannot rely"] }
    ],
    bonus: [
      { name: "backoff / retry limits on the tool", any: ["backoff", "retry limit", "exponential", "circuit breaker", "timeout"] },
      { name: "alerting, kill switch, or budget enforcement outside the model", any: ["kill switch", "alert", "monitor", "circuit", "shut off", "outside the model", "infrastructure"] },
      { name: "return structured errors as observations so it can adapt", any: ["structured error", "error message", "observation", "feedback", "tell the model"] }
    ],
    traps: [
      { name: "fixing it only in the prompt", any: ["^.{0,200}(better prompt|improve the prompt|tell it to stop|instruct it)(?!.*(cap|limit|budget|max))"], why: "The model may follow that instruction 95% of the time. The 5% is what ran overnight. Anything you actually need guaranteed has to be enforced outside the model." }
    ],
    model: "Add enforcement outside the model: a hard max_turns cap, a per-run token and cost budget, timeouts and a retry limit on the tool itself, and a kill switch that doesn't need a deploy. Also return the tool's failure as a structured observation so the agent can try something different rather than repeating the same call. A better prompt isn't enough because prompts are advisory — the model may follow them almost always, and 'almost always' is exactly what burned the budget overnight. Anything you need to be guaranteed must be enforced in code that the model cannot talk its way past."
  }
},

/* ---------------------------------------------------------------- 5.2 */
verify: {
  id: "verify",
  title: "The second opinion",
  hook: "Who checks the model's work, and what should they do when a claim has no source?",
  fig: "verify",
  figCap: "Follow the worked example's three claims from left to right: each is matched against the\nsource document, and each lands in one of two places — kept because a page supports it, or removed\nbecause the source contradicts it or never mentions it. Look for the column the figure does not have:\nthere is no lane for \"kept, but softened with a hedge\". An unsourced claim leaves the answer\nentirely, which is why the rewritten summary on the right is shorter than the draft on the left —\nand honest.",
  beginner: `
<p>Ask a model to summarise a solar-energy company's annual report and it hands back three tidy
sentences: the company fitted 12,000 panels last year, installations dipped over the winter, and an
expansion into Spain is planned for next year. It reads like the work of someone who knows the
report cold. Here is the trouble: the Spanish expansion appears nowhere in the document. The model
invented it, and delivered the invention in exactly the same assured tone as the true parts.
<mark>The confidence is the problem — a fluent model sounds equally certain whether it is reporting
or inventing, so tone gives you nothing to grade truth by.</mark></p>

<p>The obvious remedy is to reply "are you sure? double-check that" — and it mostly fails, for a
concrete reason. Whatever internal gap led the model to find "expansion into Spain" plausible enough
to write is still there when it re-reads the sentence, so the same gap that produced the error now
inspects it and waves it through. Nine times out of ten you get back a polite "yes, confirmed",
which is the original mistake with a fresh coat of certainty. You have not added a check; you have
asked the blind spot to look for itself.</p>

<p>Newspapers solved this decades ago: the reporter who wrote a story is the one person barred from
fact-checking it. A separate reader — someone with no attachment to the draft — goes through with a
checklist, pointing at each factual statement and asking one question: show me where this comes
from. Machine systems borrow both pieces. The second reader is called a <b>judge</b> — in this
course's examples, the judge is Claude — and the checklist it works from is called a <b>rubric</b>:
a written list of what a good answer must do, so the judge grades against stated criteria instead of
its own mood.</p>

<p>Watch the check run on our three-sentence summary, with the actual report open beside it — a
document that, in a bigger system, retrieval would have fetched (<x-ref to="rag">4.1</x-ref>).
Sentence one: "12,000 panels fitted last year." The report's operations page says 12,000. Supported
— it stays. Sentence two: "installations dipped over the winter." The report says winter was the
busiest quarter. Contradicted — it goes, replaced by what the source actually says. Sentence three:
"expansion into Spain planned." No page mentions Spain at all. What does the checker do with a claim
that has no source? It deletes it. <mark>Not "reportedly planning", not "may be planning" — a hedge
still plants the idea in the reader's head while dressing it up as caution, so an unsupported claim
comes out entirely, and the honest summary simply says nothing about Spain.</mark> You have already
met this machinery, by the way: when this course grades your written answers, that is Claude reading
your text against a rubric — this lesson is describing the grader that marks you.</p>

<p>A good rubric also asks more than "was the final answer right?", because a right answer can be an
accident. If an agent (<x-ref to="agent">5.1</x-ref>) reached the correct figure by guessing, or by
a sequence of steps that only landed well by luck, the next question will expose it — so the judge
also examines the <em>route</em>: were sensible steps taken in a sensible order? It weighs the bill,
too: an answer two percent better that takes ten times the money and time is a defeat, not a win,
and only shows up as one if price and speed sit on the scorecard next to quality. And it rewards
knowing when to stop — <mark>a system that says "the report doesn't cover this" when the evidence
truly is absent is behaving better than one that always produces something, even though a bare
accuracy number will never show the difference.</mark></p>

<p>Now the failure worth remembering: the judge is a model too, and it arrives with its own tilts.
Judges reliably prefer the longer of two answers even when the extra words add nothing, and they
tend to favour whichever answer they read first. Worse, a judge that shares the writer's blind spots
inherits them — set a model to mark its own family's homework and it nods along at precisely the
inventions it would have made itself. Two honest fixes exist. Run every comparison twice with the
answers' positions swapped, and trust only verdicts that survive the swap. And keep the writer and
the judge different — a different model, so the newspaper rule holds inside the machine as well as
in the newsroom. The figure below traces the three claims through the check, and the column that
isn't there is the point.</p>

<x-fig name="verify_ex"></x-fig>`,
  expert: `
<p>The beginner track watched a judge delete a Spanish expansion that was never real. Now make the
deleting your job: you are shipping the checker, and other people's decisions will rest on its
verdicts. Four decisions determine whether you have built an instrument or an ornament, and they
arrive in roughly this order.</p>

<x-fig name="verify_ex"></x-fig>

<p><b>Decision one: what question does the judge actually answer?</b> The tempting rubric is
holistic — "rate this answer one to ten" — and it is the one to resist, because models hold
comparisons far more steadily than absolute scales. Ask for a score twice and the seven drifts to
a five; put two answers side by side and ask which is better, and the verdict holds. So the first
fork is pairwise wherever you are choosing — between prompts, models, versions. And when you need
a gate rather than a choice — ship or don't — decompose instead of scoring: split the draft into
atomic claims and put a question to each that yes or no can settle, "does the source support this
sentence?". Ten small verdicts are easier to trust than one large mood, and they hand you
something no score ever does — the exact sentence to remove. The price is cost: ten claims means
ten adjudications per draft, so budget the judge like any other model call, because that is what
it is.</p>

<p><b>Decision two: how do you know the judge is any good?</b> The same way you would trust a new
thermometer — by checking it against something you already trust. A hundred or so outputs,
labelled by a human who cares about the answer, is enough to start. The trap is the obvious
statistic. Percent agreement looks most reassuring exactly when your labels are skewed, and
production labels are always skewed — most outputs pass. When both graders say "pass" nearly all
the time, two weighted coins would also agree impressively often by pure luck. Kappa is nothing
more exotic than agreement with that luck subtracted: it asks how much of the observed agreement
survives once you remove what chance alone would have produced. <mark>Agreement only begins to
mean something above the floor that chance sets, and on skewed labels that floor is startlingly
high.</mark></p>

<p><b>Decision three: which tilts do you engineer away, and how?</b> The beginner track named the
judge's biases; the practitioner's good news is how cheap the countermeasures are — all
mechanical, none requiring a smarter model. Position: run every pairwise comparison in both
orders and score a flip as a tie; one extra call, and it is not optional. Verbosity: the swap does
nothing here, so the rubric must say outright that extra words earn no credit — and you should
watch whether scores correlate with answer length, because a judge that grades effort is grading
the one thing any model can fake for free. Self-preference: keep the writer and the judge
different models — where this course names one, the judge is Claude — because a grader reading
its own family's prose tends to mistake familiar habits for correctness.</p>

<p><b>Decision four: one judge or a panel?</b> The jury intuition says three judges outvote each
other's mistakes, and the arithmetic honestly delivers — provided the mistakes land in different
places. That proviso is the entire decision. Sample the same model three times and you have not
hired three judges; you have asked one judge three times, and it carries the same fondness for
length and the same blind spots into every reading, so the panel agrees most confidently exactly
where it is wrong together. Independence is the thing you are paying for. <mark>Spend the
second-judge budget on a differently built judge — another model, another rubric structure — or
on more human labels; a second sample of the same judge buys you almost nothing.</mark></p>

<p><b>The war story worth carrying.</b> A team ships an eval dashboard, and the tile everyone
quotes says the judge agrees with human spot-checks 85% of the time. A quarter of the quarter's
ship-or-hold calls cite that tile as evidence. Then someone computes the chance floor: the labels
ran nine-to-one towards "pass", and two graders who each say pass nine times in ten will agree
82% of the time by luck alone. The celebrated 85% sat three points above a weighted coin. The fix
involved no retraining at all — the labelled set was rebalanced with harder failures and the tile
switched to kappa, which promptly said what was true: the judge needed work, and the roadmap had
been steering by an instrument nobody had calibrated.</p>

<p>What separates people who have shipped this: they treat the judge as an instrument with a
calibration certificate. The prompt is versioned, a judge-model upgrade triggers a re-baseline
against the same human-labelled set, and that labelled set — not the judge — is the asset they
guard, because it is the only thing that keeps every future judge honest. And they know where the
instrument must never be pointed casually: the moment a judge's score becomes a training signal
(<x-ref to="rlhf">6.2</x-ref>) or steers an agent mid-task (<x-ref to="agent">5.1</x-ref>), its
tilts stop being noise in a report and become the direction the system grows.</p>`,
  terms: [
    ["Verifier", "A separate pass that adjudicates claims against evidence."],
    ["Claim decomposition", "Splitting an answer into atomic checkable statements."],
    ["LLM-as-judge", "Using a model to score outputs. Must be validated against humans."],
    ["Refusal correctness", "Whether the system declines when it should."]
  ],
  sources: [
    ["Reflexion", "paper", "Shinn et al., 2023. Verbal self-critique as a learning signal within a trajectory.", "https://arxiv.org/abs/2303.11366"],
    ["FACTSCORE", "paper", "Min et al., 2023. Decomposing generations into atomic facts and scoring each. The method behind most faithfulness metrics.", "https://arxiv.org/abs/2305.14251"]
  ,
    ["Demystifying Evals for AI Agents","article","Anthropic, 2026. Code, model and human graders compared honestly, transcript versus outcome grading, and per-agent-type strategies from real deployments. The practitioner companion to this lesson.","https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents"],
    ["Using LLM-as-a-Judge: A Complete Guide","article","Husain, 2024. Binary pass/fail judgements calibrated against one domain expert, distilled from thirty deployments. The working antidote to piles of uncalibrated 1–5 scores.","https://hamel.dev/blog/posts/llm-judge/"]
  ],
  mcq: {
    q: "The verifier finds a claim with no supporting evidence. What should happen?",
    o: ["Add a hedge like 'possibly'", "Delete the claim", "Flag it for the user to judge", "Lower the confidence score"],
    a: 1,
    why: "Hedging keeps an unsupported claim in front of the reader while looking responsible. If there's no evidence, the honest output contains nothing about it."
  },
  open: {
    prompt: "Why is 'the answer was correct' an insufficient way to evaluate an agent? Name what else you'd measure.",
    must: [
      { name: "trajectory / process matters, not just the final answer", any: ["trajector", "process", "steps", "how it got", "tool calls", "path", "reasoning"] },
      { name: "cost and/or latency must be reported alongside quality", any: ["cost", "latenc", "expensive", "token", "budget", "speed", "time"] }
    ],
    bonus: [
      { name: "faithfulness / grounding of claims", any: ["faithful", "ground", "cite", "source", "supported", "evidence"] },
      { name: "correct answers can come from luck or a broken path", any: ["luck", "by accident", "wrong reason", "coincidence", "guess", "fluke"] },
      { name: "refusal correctness — declining when it should", any: ["refus", "decline", "abstain", "don't know", "unknown", "insufficient"] }
    ],
    traps: [
      { name: "treating a single accuracy number as sufficient with caveats", any: ["accuracy is enough", "just measure accuracy", "only need accuracy"], why: "A single number hides the cases you most need to see: right answer for the wrong reason, right answer at 20× the cost, and confident answers where it should have declined." }
    ],
    model: "Because a correct final answer can come from a broken process — the agent guessed, or took a wrong path that happened to land somewhere right — and that won't hold on the next input. I'd measure trajectory validity: were the right tools called, in a sensible order, with valid arguments, and did it recover from errors. Faithfulness: is every claim traceable to a retrieved source or a tool result. Cost and latency, reported alongside quality, since something 3% better and five times more expensive is a regression. And refusal correctness — whether it declines when the evidence genuinely isn't there, because a system that always produces an answer is broken in a way accuracy can't detect."
  }
},

/* ---------------------------------------------------------------- 6.1 */
rl: {
  id: "rl",
  title: "Learning from consequences",
  hook: "How do you learn something nobody can demonstrate, from feedback that arrives late?",
  fig: "rl",
  figCap: "Read it from the right, where the only genuine feedback in the whole picture lands: the\ncrash (or the reward) at the final step. Then watch value seep backwards one step at a time, each\nstate inheriting a share of what followed it — this is credit assignment happening, the heel-drift\nat second three slowly accumulating its share of the blame for second twelve. Once every state\ncarries a value, the arrows underneath fall out almost for free: point each state at its\nbest-valued neighbour, and that set of arrows — always step toward the highest value — is the\nstrategy the values imply.",
  beginner: `
<p>Watch a child learning to skateboard. No coach on earth can hand over a printed sheet of correct
muscle commands — "lean 4 degrees left at second two" — because nobody, including the coach, knows
what the sheet should say. So the child pushes off, wobbles, falls, gets up, and does slightly more
of whatever hurt slightly less. Every lesson so far in this course assumed an answer key: a label
saying "this was a shoe" or "this ranking was right". Skateboarding has no answer key. It has only
consequences — and this lesson is about the machinery for learning when consequences are all you
get.</p>

<p>Here is what makes that genuinely hard, in one concrete run. The board rolls for twelve seconds
and ends in a crash. The actual error — a heel drifting an inch too far back — happened at second
three. In between sit nine seconds of decisions, nearly all of them fine. A supervised learner would
be told, move by move, "correct" or "wrong"; the skateboarder gets one bruise for the whole
sequence, delivered nine seconds after the mistake that earned it. <mark>The defining problem of
learning from consequences is deciding which earlier action deserves the blame for a result that
arrives late</mark> — the field calls it credit assignment.</p>

<p>The standard answer works backwards. Give the final moment the value its outcome earned — the
crash makes second twelve strongly negative — then let each earlier moment inherit a share of the
value of whatever followed it, step by step, until the heel-drift at second three carries its slice
of the blame. Once every moment has a value, acting well becomes simple: from wherever you are, step
toward the neighbouring state with the highest value. The figure below shows exactly this happening
on a small grid.</p>

<p>Before credit can even be assigned, there is a more basic dilemma: do you repeat what has worked,
or gamble on something untried? Order your usual dish every visit and you will never discover your
true favourite; order at random every visit and you eat a lot of disappointing dinners. Now put
numbers on it. A news site is choosing between two headlines for the same story. Headline A has been
shown 1,000 times and drawn 20 clicks — a solid 2%. Headline B has been shown only 100 times and
drawn 3 — nominally 3%, but three clicks is barely evidence; luck alone could explain it. Which do
you trust? A, clearly. Which should you show next? Not only A. <mark>The strategy "always play the
current best" quietly loses, because an option that starts with an unlucky streak gets buried
forever</mark> — if B's true rate really is 3%, showing A every time costs you a click per hundred
visitors, indefinitely, and you never find out. The fix is to keep trying options roughly in
proportion to how uncertain you still are about them: exploit where the evidence is firm, explore
where it is thin. This trade-off is called explore versus exploit, and this simple no-memory version
of the problem — a fixed menu, a score per choice — is called a bandit.</p>

<p>Now for the quiet superpower of this lesson. Suppose you have designed a new headline-picking
strategy and want to know how it would have done last month — without risking a month of live
traffic on it. You can, provided the old strategy kept an honest diary: for every visitor, what it
saw, what it chose, <em>the probability it gave that choice at the moment of choosing</em>, and what
happened. Take one diary row. The old policy showed article A with probability 0.1 — one visit in
ten — and the visitor clicked. Your new policy, given the same visitor, would have shown A with
probability 0.5. So that one click is entered into the books at a weight of five — 0.5 divided by
0.1 — standing in for all the identical visits on which the new policy, unlike the old, would have
put A on screen and had the chance to earn it. A row where the new policy would
almost never have made the logged choice gets scaled down towards zero instead. Average the
re-weighted results across the diary and you have priced the new policy without serving it once.
<mark>The recorded 0.1 — the propensity — is the whole trick: it is the denominator, and no volume
of extra logs can recover a probability that was never written down</mark>.</p>

<x-fig name="rl_ex"></x-fig>

<p>The failure worth remembering lives in that same arithmetic. If the new policy loves an action
the old one almost never took — logged with propensity 0.001, say — then one lucky click on that row
is multiplied five hundred-fold and can single-handedly drag the whole average. The estimate is
still honest on average, but it is resting on a handful of rare rows, so practitioners cap the
weights and always ask how many diary entries are really carrying the answer. Keep the two
lessons paired: record the chances, and distrust an estimate built on choices the diary barely
contains. With that, you are ready for the figure — and for <x-ref to="rlhf">6.2</x-ref>, where the
same learning-from-consequences loop, with people's preferences as the consequence, is how chat
assistants get their manners.</p>`,
  expert: `
<p>The beginner track left you holding a diary and a bruise. Turning those into a shipping system
is not really a modelling exercise — it is three decisions you will make early, cheaply, and
mostly without realising they were decisions. This track walks through them in the order they
will find you.</p>

<x-fig name="rl_ex"></x-fig>

<p><b>Decision one: how far up the ladder does your problem actually go?</b> RL is sold as one
subject but it is really a staircase, and each step up costs you data, variance and debuggability.
At the bottom sits the plain <b>bandit</b>: a fixed menu, a reward per pull, no memory of state —
UCB and Thompson sampling, and honestly most of what industry ships under the letters "RL". One
step up, the <b>contextual bandit</b> lets you look at the visitor before choosing, which is
already enough to run a headline slot, an artwork picker, or the ad slots of
<x-ref to="ads">1.3</x-ref>. The <b>full MDP</b> rung — where today's action reshapes tomorrow's options and
blame has to travel backwards through time, via value methods like Q-learning and DQN or policy
methods like REINFORCE and PPO — is the rung the textbooks spend nine chapters on and most product
teams never need. Above it, <b>learning purely from logs</b> takes no fresh actions at all, and
the top rung, <b>learning from preference</b>, replaces the environment's reward with a human or
mechanical judge — that is the machinery of <x-ref to="rlhf">6.2</x-ref>, kept honest by the
checkers of <x-ref to="verify">5.2</x-ref>. The temptation is
always to climb: sessions feel like trajectories, so surely the recommender is an MDP? Usually
what you wanted was a sequence model doing supervised learning — <x-ref to="seqrec">3.2</x-ref> —
with a bandit on top for exploration. <mark>Take the lowest rung that expresses your problem, and
climb only when the rung below has failed in a way you can name</mark>.</p>

<p>If you do climb to policy gradients, one piece of arithmetic carries the whole family. The
update <code>∇J = E[∇log π(a|s) · A]</code> reads as an instruction: nudge up the probability of
whatever did better than you had any right to expect. The load-bearing symbol is
<code>A</code>, the advantage — not the raw reward, but the reward minus a baseline estimating
what that state would have paid anyway. On average the subtraction changes nothing, which is
exactly why it is legal; what it does change is the noise, collapsing it enough that training
becomes possible at all. Nearly every trick in this field has that same shape — spend a little
bias, or nothing, to buy a lot of variance — and you will meet it twice more below. The cleanest
bridge from ranking land is YouTube's Top-K off-policy REINFORCE, which trained a production
recommender from logged traffic using clipped importance weights plus a correction for serving a
slate of K items rather than a single action.</p>

<p><b>Decision two: what goes in the log?</b> This is the decision teams get wrong precisely
because it does not look like one — it is a schema review, someone shaving fields to save bytes.
The field at stake is the propensity: the probability your live policy assigned, at the moment of
choosing, to the action it actually took. Written down, it is the denominator that lets you price
any future policy against last month's traffic. Omitted, it is gone — no volume of later data,
no cleverness, no retroactive model reconstructs a probability that was never recorded, and every
counterfactual question your team asks for the next two years comes back "we can't know".
<mark>Log the propensity on day one, before the first model is any good, because the logs you
write now are the experiments you are allowed to run later</mark>. While you are in that schema
review, reserve a sliver of uniform-random traffic too: it is the one slice of the log no
feedback loop can bend.</p>

<p><b>Decision three: how hard do you clamp the weights?</b> The estimator itself is one line —
<code>V̂(π) = mean over the log of (π(a|x) / μ(a|x)) × reward</code> — and its meaning is
bookkeeping, not magic: replay the diary, but let each entry count in proportion to how much more
the new policy would have been there than the old one was. The old policy served an article one
visit in ten; yours would serve it five in ten; a logged click on that row is booked at weight
0.5/0.1 = 5, standing in for the visits your policy would have collected and the old one skipped.
The same ratio running the other way shrinks a row towards zero. The trouble is the tail: a row
logged at propensity 0.001 that your policy loves arrives with a weight in the hundreds, and the
"unbiased" average quietly becomes a bet on a handful of flukes. So you clip — cap the weights at
ten or twenty — and accept a known, bounded bias in exchange for an estimate that no longer
swings on single rows. The honest way to report the result is the value, the interval, <em>and
the effective sample size</em>: how many rows, after weighting, are genuinely carrying the
answer. An OPE number without its ESS is an incomplete sentence.</p>

<p><b>The war story, because everyone gets one.</b> A team I will not name priced a new
recommendation policy against a month of logs: +30% on the primary reward, comfortably the best
result of the quarter, launch review booked. One reviewer asked for the maximum importance
weight. It was enormous — and pulling the thread revealed that twelve rows, all logged at
propensity 0.001, were contributing most of the lift; the "month of evidence" was a dozen lucky
clicks multiplied by several hundred each. Re-run with weights clipped and ESS reported, the
uplift fell inside the noise band, and the launch decision reversed on the spot. The postscript
is the useful part: they resolved it with a plain, fixed A/B test — two weeks, no adaptation,
small stakes — because that is the tool that is actually right when you need clean inference more
than you need efficiency. A bandit that shifts traffic mid-flight is the better earner; the
frozen split is the better witness.</p>

<p>What separates people who have shipped this: they treat the log, not the policy, as the asset.
Policies are retrained monthly and forgotten; a log with honest propensities and a random slice
compounds for years, because it prices every idea anyone has later. And they never present a
counterfactual estimate as one number — it is always value, interval, and how many rows are
really holding it up.</p>`,
  terms: [
    ["Policy", "The strategy: a mapping from state to action distribution."],
    ["Advantage", "How much better an action did than the baseline expectation."],
    ["Explore vs exploit", "Trying the uncertain versus repeating the known-good."],
    ["Regret", "Reward left on the table versus always playing the best action — the cost of not adapting."],
    ["Off-policy evaluation", "Estimating a new policy's value from logs written by an old one, using the recorded propensities."]
  ],
  sources: [
    ["Sutton & Barto, Reinforcement Learning: An Introduction", "book", "Free online and genuinely the canon. Chapters 3–6 are readable with high-school maths and are all you need to start.", "http://incompleteideas.net/book/the-book-2nd.html"],
    ["A Contextual-Bandit Approach to Personalized News Recommendation", "paper", "Li et al., 2010. The landmark production bandit — LinUCB on Yahoo! front-page news. Read it for the evaluation method as much as the algorithm.", "https://arxiv.org/abs/1003.0146"],
    ["Open Bandit Dataset and Pipeline", "paper", "Saito et al., 2020. The dataset and estimator library the capstone uses; the introduction is the best short survey of off-policy evaluation.", "https://arxiv.org/abs/2008.07146"],
    ["Top-K Off-Policy Correction for a REINFORCE Recommender System", "paper", "Chen et al., 2019. RL applied to a real recommender at scale. The bridge paper if you come from ranking.", "https://arxiv.org/abs/1812.02353"]
  ],
  mcq: {
    q: "You want to estimate how a new policy would perform using only logs collected under the old one. What makes an unbiased estimate possible?",
    o: ["The old policy's recorded probability for each action it took", "A large enough number of logged clicks", "Retraining the new policy on the logged rewards", "The two policies agreeing on most actions"],
    a: 0,
    why: "Importance weighting divides by the propensity the logging policy gave the logged action. Without that denominator recorded at logging time, no amount of data recovers it — which is why logging propensities is the one decision that makes every later evaluation possible."
  },
  open: {
    prompt: "Your team wants to know how a new recommendation policy would have performed last month, using only the logs the old policy wrote. Explain why that is possible — and what must have been recorded for the answer to be trustworthy.",
    must: [
      { name: "re-weight each logged event by how differently the two policies would act", any: ["reweigh", "re-weigh", "weight", "importance", "count.*(more|less)", "ratio"] },
      { name: "the logging policy's action probabilities (propensities) must be recorded", any: ["propensit", "probabilit.*(record|log|action|chance)", "chance.*(record|written|logged)", "how likely"] }
    ],
    bonus: [
      { name: "variance / effective sample size when the policies diverge", any: ["variance", "effective sample", "blow", "explode", "diverge", "dominat", "few events"] },
      { name: "the estimate is unbiased when propensities are real", any: ["unbiased", "in expectation"] },
      { name: "a reward model / doubly robust reduces the variance", any: ["doubly robust", "reward model", "model.*(absorb|reduce|assist)"] }
    ],
    traps: [
      { name: "claiming more data substitutes for missing propensities", any: ["enough data", "(more|enough|big) data.*(fix|recover|solve|substitute)", "volume.*(fix|recover)"], why: "No volume of logs recovers a probability that was never written down. Without the propensity there is no denominator, and the estimate isn't noisy — it's undefined." }
    ],
    model: "It's possible because the logs record not only what the old policy did but what happened afterwards, and each logged event can be re-weighted by how much more or less the new policy would have liked that action than the old one did — events the new policy would repeat count for more, events it would avoid count for less. That re-weighting needs a denominator: the probability the old policy assigned to the action it actually took, recorded at decision time. If those propensities are real, the estimate is unbiased; no amount of extra data substitutes for them if they were never logged. The catch is variance: when the new policy loves actions the old one rarely took, a handful of rows dominate the estimate, which is why you report effective sample size and why doubly robust estimators pair the weights with a reward model."
  }
},

/* ---------------------------------------------------------------- 6.2 */
rlhf: {
  id: "rlhf",
  title: "Learning from preference",
  hook: "Nobody can write down the reward for a good answer. Anybody can pick the better of two.",
  fig: "rlhf",
  figCap: "Follow the main road left to right: pairs of answers with human picks train the judge, and the assistant then practises against that judge while the leash holds it near its starting point. The two detours below are the lesson's shortcuts — one feeds the picks to the assistant directly and deletes the judge box entirely, the other swaps the judge for a checker wherever the answer can be verified, the one grader with no blind spots to exploit.",
  beginner: `
<p>Read these two replies to the same question — "My sourdough starter smells of nail-polish
remover. Have I killed it?" Reply A: "No — that solvent smell is a hunger signal. The yeast have
run through their food and acids are building up. Give it two feeds over the next day and the
smell will fade." Reply B: "Starters can produce many different smells, and smell alone is hard
to interpret. There are several possible causes. You may wish to consult a baking forum for
further guidance." You had chosen A before you finished reading B. Now try to give each a number
out of ten instead. Is A a seven, or a nine? Is B a three, or a five? Ask two people and their
scores will disagree by a couple of points — yet both will name A as the winner without
hesitating. <mark>Humans are unreliable at scoring answers but remarkably consistent at picking
the better of two</mark>, and that lopsided fact is what this whole lesson is built on.</p>

<p>It matters because of a gap left open in <x-ref to="rl">6.1</x-ref>: learning by trial and
error needs a reward — a number attached to every attempt — and for "a good answer to a question"
nobody can write that number down. Here is the workaround, told as a kitchen story. An apprentice
cook trains under a master whose palate no recipe book captures. The master has no time to taste
every practice dish, but has left behind thousands of verdicts of one simple form: "the plate on
the left beats the plate on the right". So the kitchen builds a stand-in — a <b>taste-predictor</b>
that studies those thousands of verdicts until, shown any new dish, it can guess how the master
would rate it. Now the apprentice can cook all night, every dish graded instantly, with the master
nowhere in the room.</p>

<p>Swap the kitchen for an assistant and you have the pipeline. Gather a few hundred thousand
questions, each with two candidate answers and a human's pick — exactly the sourdough choice you
just made. Train a second model, called the <b>reward model</b>, to predict those picks; it is the
taste-predictor. Then let the assistant practise: it drafts an answer to the sourdough question and
the reward model returns 0.58; a later draft that opens with the direct reassurance scores 0.66;
across millions of such rounds the assistant is nudged, step by step, toward whatever its judge
scores highly.</p>

<p>Run that loop with no safeguard and something sly happens: <mark>the apprentice stops studying
cooking and starts studying the critic</mark>. Suppose the reward model — trained on limited data,
an imperfect copy of human taste — happens to overrate answers that sound confident and arrive as
tidy lists. The assistant will find that seam. Soon it produces answers like "Great news — this is
completely solvable! Three key facts: 1) Smells are temporary. 2) Starters are resilient. 3)
Feeding resolves most issues." — which says nothing, and scores 0.91. The defence is a leash,
formally the <b>KL leash</b>: a penalty for drifting far from the model it started as. The logic
is plain — <mark>the judge's verdicts are only trustworthy near the kinds of answers it was
trained on, so the further the assistant wanders from familiar territory, the less its rising
score means</mark>. The leash deliberately sacrifices some measured score to keep the practice
inside the region where the judge still knows what it is talking about.</p>

<x-fig name="rlhf_ex"></x-fig>

<p>Three shortcuts trim this machinery, each for a plain reason. You can skip building the judge
at all (the method called DPO) and let each pair tug the model directly toward its preferred
answer — legitimate because the judge's opinion was never anything more than a summary of those
same pairs. During practice, instead of maintaining yet another model whose only job is estimating
how well things are currently going, <mark>the assistant can write several answers to the same
question and grade each one against the average of its own siblings</mark> (the method called
GRPO) — the siblings are a free measuring stick, so an entire extra network gets deleted. And when
an answer is objectively checkable — code that must pass tests, arithmetic with one right value —
you replace the learned judge with the checker itself, because a test suite cannot be
sweet-talked.</p>

<p>The failure worth remembering: plot the judge's score during training and it climbs; train
longer and it climbs further — while real people, shown the same answers, report they peaked a
while ago and are now getting worse. <mark>The score keeps rising after true quality has turned
down, so the score alone can never tell you when to stop</mark>; what is improving late in
training is skill at pleasing the judge, not skill at answering. That is why teams watch the leash
and test on questions the judge never graded — the discipline of <x-ref to="verify">5.2</x-ref>
applied to training itself. The figure lays this out as one main road with two detours: trace
picks → judge → practice-on-a-leash first, then see which boxes each shortcut removes.</p>`,
  expert: `
<p>The beginner track left the apprentice cooking all night against a stand-in critic. Standing
that kitchen up yourself is a different experience: the trial-and-error machinery is exactly the
one <x-ref to="rl">6.1</x-ref> built, and almost none of your time will go on it. Every decision
that matters is about the grader — who it is, how far to trust it, and what it can and cannot
see.</p>

<x-fig name="rlhf_ex"></x-fig>

<p><b>Decision one: collect comparisons, not scores.</b> This looks like an interface detail and
is actually where the statistics get won or lost. Ask raters for a mark out of ten and two honest
people disagree by a couple of points on the same answer — your labelling budget is buying noise.
Ask which of two answers is better and they converge, reliably, across thousands of raters.
Inter-rater agreement is the real currency here: every point of agreement is signal your money
actually purchased, and the pairwise interface buys far more of it per pound than any scoring
rubric. The price of the pairwise form is calibration — a pick says A beats B, never by how much,
and never whether both were dreadful — so teams that need an absolute floor bolt on a small
rule-checked set rather than trying to turn humans into consistent scorers. No interface achieves
that, because the inconsistency is in the humans, not the form.</p>

<p><b>Decision two: delete the judge, or keep it and stay online?</b> DPO deletes the reward
model by taking seriously something the beginner track hinted at: the judge was only ever a
compressed summary of the pairs, so let the pairs teach directly. The mechanics, meaning-first:
the leashed objective has a best-possible policy you can write down, and rearranging that
expression reveals a reward hiding inside every policy — how much more probable the tuned model
finds an answer than its reference is a score in disguise. Feed that disguised score into the same
win-or-lose comparison you collected, and the one quantity nobody can compute drops out, because
both answers share a prompt. What remains trains on log-probabilities the transformer already
produces in a forward pass (<x-ref to="attention">3.1</x-ref>) — no judge in memory, no
generation loop, and failure modes you already know how to debug from supervised training. (The
depth tab walks the algebra.) What deletion costs you: the model can only be pulled toward
answers that already exist in someone's dataset. Online RL is the opposite bargain — you pay for
sampling several answers per prompt (siblings that, as the beginner track showed, double as the
baseline), and in exchange the model practises on its own current mistakes, which is the only way
to learn behaviour no dataset yet contains. The working split: offline pairs suffice for taste —
tone, format, refusal style; online practice is for capability — reasoning and tool use, where
the good answer must be discovered rather than imitated.</p>

<p><b>Decision three: β, the price of trusting a proxy.</b> The whole objective fits on one line —
<code>max_π E[r(x,y)] − β·KL(π‖π_ref)</code> — and reads as a sentence: chase the judge's score,
but pay a fine for every step away from the model you started as. β is the exchange rate between
those two currencies. Set it high and you are saying the judge is barely trustworthy: the model
will hardly move, and hardly improve. Set it low and you are declaring the judge reliable even on
answers unlike anything it was trained on — a declaration that is false for every learned reward
model and true only for a checker. <mark>So set β by the quality of your reward signal, not by
copying a paper's value: a slack leash is something a grader has to earn</mark>. During the run,
read cumulative KL as a budget being spent — reward bought at high KL was bought in territory
where the judge is guessing.</p>

<p><b>Decision four: where does the reward actually come from?</b> This is the strategic one, the
one that outlives any algorithm choice. Three sources, three blind spots. Human comparisons are
the gold standard for taste, but they are slow, costly, and cannot grade what the rater cannot
perceive — a subtly wrong proof reads exactly like a right one. AI feedback under written
principles — the recipe Anthropic uses for Claude, with the principles published — scales to
millions of labels and makes the grading auditable, but the grader is itself a model, carrying a
model's biases and the same perceptual ceiling. Checkers — test suites, compilers, exact answers —
are the one grader with no seam to exploit, which is why this lesson and the discipline of
<x-ref to="verify">5.2</x-ref> converge; their limit is reach, because most of what an assistant
does has no checker. Production reward signals are therefore portfolios: checkers wherever they
exist, principled AI feedback across the broad middle, and scarce human comparisons reserved for
the frontier where taste is genuinely contested.</p>

<p><b>The war story worth internalising.</b> A preference-tuned run whose win rate climbed for
three straight weeks — while the user-complaint queue grew. Every dashboard said better; every
support ticket said worse. The check that finally caught it took one afternoon: regress the
reward model's score against answer length over a fixed evaluation set, and length explained most
of the variance. Raters had leaned, slightly, toward answers that looked thorough; the reward
model amplified that lean into a rule; the policy amplified the rule into essays; and the win
rate, graded by the same captured judge, applauded the entire spiral. <mark>Before trusting any
learned reward, regress it against length and the other cheap surrogates — confident phrasing,
list formatting — because whatever it correlates with is what you are about to train more
of</mark>. The fix was mundane: length-balanced pairs and a retrained judge. The habit is the
part that transfers.</p>

<p>What separates people who have shipped this: they stopped watching the reward curve. The
optimiser is a few hundred lines and interchangeable; the reward signal is the product. The
numbers they defend are held-out human win rate and KL spent — the two the judge cannot flatter —
because anyone can make measured reward go up, and the ones who have shipped know that was never
the hard part.</p>`,
  terms: [
    ["Reward model", "A model trained on human comparisons to predict which answer a person would prefer."],
    ["KL leash", "The penalty keeping the policy near the reference model, priced by β."],
    ["DPO", "Direct preference optimization. The closed-form substitution that turns preference tuning into classification."],
    ["GRPO", "Group-relative policy optimization. Group mean replaces the critic."],
    ["Verifiable reward", "A deterministic checker — tests, a compiler, an exact answer — standing in for the learned judge."]
  ],
  sources: [
    ["Deep RL from Human Preferences", "paper", "Christiano et al., 2017. The origin: complex behaviour from comparisons on under 1% of interactions. Read the clip-comparison interface design first.", "https://arxiv.org/abs/1706.03741"],
    ["Training language models to follow instructions", "paper", "Ouyang et al., 2022. The SFT → reward model → PPO pipeline, and the 1.3B-beats-175B preference result that started the post-training era.", "https://arxiv.org/abs/2203.02155"],
    ["Direct Preference Optimization", "paper", "Rafailov et al., 2023. Read Section 4 for the substitution trick — the algebra is short and the consequence is an entire deleted subsystem.", "https://arxiv.org/abs/2305.18290"],
    ["DeepSeekMath (GRPO)", "paper", "Shao et al., 2024. Section 4 introduces GRPO — shorter and simpler than PPO; read it once you understand advantages.", "https://arxiv.org/abs/2402.03300"],
    ["Proximal Policy Optimization", "paper", "Schulman et al., 2017. The optimizer under the original pipeline. Read alongside 'The 37 Implementation Details of PPO' before implementing anything.", "https://arxiv.org/abs/1707.06347"]
  ],
  mcq: {
    q: "What replaces the learned value function in GRPO?",
    o: ["A fixed constant", "The mean reward of a group of completions sampled for the same prompt", "The reward model's output", "Nothing — it uses raw returns"],
    a: 1,
    why: "Siblings from the same prompt form an empirical baseline. That removes the critic network entirely, halving memory and eliminating a model that needed its own tuning."
  },
  open: {
    prompt: "Why does RLHF keep a KL penalty to the reference model — what actually goes wrong if you remove it?",
    must: [
      { name: "the reward model is an imperfect, exploitable proxy", any: ["proxy", "imperfect", "exploit", "hack", "game", "blind spot", "approxim"] },
      { name: "without the leash the policy drifts into regions where the proxy is wrong", any: ["drift", "wander", "far from", "leash", "stay close", "near the reference", "off.?distribution", "outside.*training"] }
    ],
    bonus: [
      { name: "overoptimisation: measured reward rises while true quality falls", any: ["overoptim", "reward.*(rises|up).*quality.*(falls|down)", "goodhart", "peaks.*(then|and)"] },
      { name: "β prices the trade-off / KL as a budget", any: ["beta", "β", "budget", "price", "trade.?off"] },
      { name: "verifiable rewards avoid the problem where a checker exists", any: ["verifiab", "checker", "unit test", "compiler", "can't be flattered", "cannot be flattered"] }
    ],
    traps: [
      { name: "claiming the KL term improves reward", any: ["kl.*(increases|improves|maximis).*reward", "helps.*get more reward"], why: "The KL term constrains reward-seeking; it deliberately gives up measured reward to protect true quality. It costs reward — that's its job." }
    ],
    model: "The reward model is trained on a limited set of human comparisons, so it is only accurate near the distribution it saw; it is a proxy, and every proxy has exploitable blind spots. Unconstrained optimisation will push the policy exactly toward those blind spots, because that is where measured reward is cheapest — so reward keeps rising while genuine quality peaks and then degrades. The KL penalty keeps the policy close to the reference model, effectively limiting how far the optimiser may trust the proxy, with β setting the price of drift. That is also why verifiable rewards change the game where they exist: a compiler or test suite has no blind spots to find, so the leash matters less."
  }
}

  }
};
