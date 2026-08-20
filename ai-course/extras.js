/* =========================================================================
   Two more pages: things to build, and where to read next.
   Kept separate from content.js so lessons stay easy to scan.
   ========================================================================= */

const PROJECTS = [
{
  id: 'sizing',
  title: 'Made-to-measure from a phone',
  one: 'Estimate body measurements from a handful of photos accurately enough that a tailor could cut fabric from them — and be honest about the error bars.',
  uses: ['embed', 'ann', 'verify'],
  hard: 'hard',
  why: `Every try-on demo shows you a garment on a body. Almost none will tell you how wrong the
  measurement is, and a tailor needs the ± far more than the number. Calibrated uncertainty is the
  part nobody ships, which makes it the part worth building.`,
  ladder: [
    ['Weekend one', 'Run an off-the-shelf pose model on a photo and get keypoints out. Don\'t train anything yet.'],
    ['Weekend two', 'Convert keypoints to real-world lengths using a reference object of known size in frame — a credit card, a sheet of A4.'],
    ['The real project', 'Measure the same person twenty times across lighting, pose and distance. The spread is your product. Report it.'],
    ['If it works', 'Compare against tape-measure ground truth on ten people and publish the error distribution, not the mean.']
  ],
  trap: 'Reporting mean error. A 2cm mean with a long tail of 15cm failures is a garment that doesn\'t fit, and the mean hides exactly that.'
},
{
  id: 'farm',
  title: 'A plant doctor that cites its sources',
  one: 'Photo of a sick plant plus location and season in; identification, diagnosis and an organic treatment plan out — every claim linked to a real agricultural document.',
  uses: ['rag', 'hybrid', 'chunk', 'verify'],
  hard: 'best first project',
  why: `Agricultural extension services publish thousands of diagnostic keys, written by people whose
  job is being right about this. That means genuine ground truth exists, which is rare — and it makes
  this the best project on the list for learning honest retrieval evaluation rather than vibes.`,
  ladder: [
    ['Weekend one', 'Use an existing plant-ID model. Don\'t train your own; several good open ones exist and the interesting problem is downstream.'],
    ['Weekend two', 'Collect extension-service PDFs for one crop in one region. One crop. Resist the urge to boil the ocean.'],
    ['The real project', 'Chunk them properly (4.3), build hybrid retrieval (4.2), and make every sentence of advice carry a document and page.'],
    ['If it works', 'Make it work offline. Fields don\'t have signal, and this is where most agricultural apps quietly fail.']
  ],
  trap: 'Skipping the retrieval eval. Build 100 photo→correct-document pairs by hand before tuning anything, or you\'ll spend a month improving a number you never measured.'
},
{
  id: 'jyotisha',
  title: 'An expert system for a contested domain',
  one: 'Answer questions the way a trained Vedic astrologer would: exact astronomy computed by code, interpretation retrieved from classical texts, every claim traceable to a verse or a calculation.',
  uses: ['rag', 'agent', 'verify'],
  hard: 'hard, and the best teacher',
  why: `The domain splits cleanly into something with exactly one right answer (where a planet was on a
  given date) and something the sources argue about (what that means). Getting the split right is the
  whole discipline of building expert systems, and this domain punishes you visibly when you don't.
  You don't have to believe any of it for that lesson to transfer.`,
  ladder: [
    ['Weekend one', 'Compute planetary positions with an ephemeris library and check them to the arc-minute against established professional software on twenty known charts. No AI yet.'],
    ['Weekend two', 'Digitise three chapters of one public-domain text, one verse per chunk, each with its book, chapter, verse and edition.'],
    ['The real project', 'Search using the computed chart facts, not the user\'s question. "Will I get married?" is a terrible query; the specific configurations in the chart are excellent ones.'],
    ['If it works', 'Add the verifier from 5.2 and make it delete anything not traceable. Then measure whether two different charts produce genuinely different answers.']
  ],
  trap: 'Letting the model compute the astronomy. Every failed product in this category does exactly this, and everything downstream is then built on invented numbers.'
},
{
  id: 'options',
  title: 'An options research lab',
  one: 'Research companies, test strategies against decades of history with realistic costs, and produce calibrated probabilities you can inspect — not trade signals you\'re asked to trust.',
  uses: ['agent', 'verify', 'rl'],
  hard: 'do this one carefully',
  why: `Time series is where evaluation discipline gets tested hardest, because the data will happily
  confirm anything you want if you let information leak backwards. Building an honest backtest teaches
  you more about evaluation than any other project here.`,
  ladder: [
    ['Weekend one', 'Get free historical price data and plot it. Simpler than it sounds and immediately more interesting than you expect.'],
    ['Weekend two', 'Write a backtest that can only ever see information available at that moment. This is the number one bug in the entire field and you will write it at least once.'],
    ['The real project', 'Add fees, spread and slippage. Watch most "winning" strategies die. That death is the finding.'],
    ['If it works', 'Build the agent that reads filings and news, then check whether it beats the boring baseline. Usually it doesn\'t, and that is a publishable result.']
  ],
  trap: 'The honest version of this warning: no system can guarantee returns. Anything that appears to is leaking future information, ignoring costs, or selecting the one strategy out of a thousand that happened to work. Software that trades real money on someone else\'s behalf is also regulated activity in most countries. Build it as research infrastructure — that version is genuinely excellent and teaches far more.'
},
{
  id: 'feed',
  title: 'A feed for infinite content',
  one: 'When anyone can generate a thousand videos in an afternoon, how should a feed rank them? Old recommenders assumed content was scarce. That assumption just stopped being true.',
  uses: ['funnel', 'sid', 'rl'],
  hard: 'open research',
  why: `There is almost nothing published on ranking when supply is effectively unlimited. Scarcity is
  baked into the objectives everyone uses. That makes this the one project on the list where a careful
  hobby build can contribute something genuinely new.`,
  ladder: [
    ['Weekend one', 'Build a simple recommender on a public dataset. Baseline first, always.'],
    ['Weekend two', 'Add a provenance signal — where did this come from — and see how it changes the ranking.'],
    ['The real project', 'Model satiation: the tenth similar video is worth far less than the first, and no standard objective knows that.'],
    ['If it works', 'Optimise for whether someone was glad they watched, not whether they clicked. Much harder to measure, much more interesting to get right.']
  ],
  trap: 'Optimising engagement and calling it satisfaction. They come apart quickly under infinite supply, and the gap is the whole research question.'
}
];

const LIBRARY = [
{
  t: 'Intuition first',
  f: 'Watch these when a concept refuses to sit still in your head. Slow, visual, no code. Do not open an editor.',
  x: [
    ['3Blue1Brown', 'video', 'Linear algebra, calculus, neural networks and attention. The best explanations of these topics in any medium at any price. Where this course links out rather than competing, it links here.', 'https://www.3blue1brown.com/', ['embed', 'attention']],
    ['StatQuest', 'video', 'Statistics and classical ML, cheerfully and very clearly. Start here if the maths in part 2 feels like a wall.', 'https://www.youtube.com/@statquest', ['embed', 'ann']],
    ['Jay Alammar — Illustrated Transformer', 'article', 'The pictures that taught a generation how transformers work. Read it after the attention lesson, not before.', 'https://jalammar.github.io/illustrated-transformer/', ['attention']]
  ,
    ["Simon Willison — Embeddings: What they are and why they matter","article","The written version of his PyBay talk, and the clearest plain-language tour anywhere of what an embedding vector actually is — related content, clustering, arithmetic on meaning. Read it before the two-tower lesson if 'a point in space' still feels abstract.","https://simonwillison.net/2023/Oct/23/embeddings/",["embed"]],
    ["Karpathy — Deep Dive into LLMs","video","Three and a half hours, general audience, no code: the full training pipeline from raw text to a helpful assistant, including why models hallucinate and what fine-tuning actually changes. The best single sitting on how these systems are made.","https://www.youtube.com/watch?v=7xTGNNLPyMI",["attention","rlhf"]],
    ["Welch Labs","video","Visually rigorous long-form explanations of how models actually compute — the diffusion and CLIP videos are the best on the internet. The nearest thing to 3Blue1Brown on the topics 3Blue1Brown has not covered.","https://www.youtube.com/@welchlabs",["attention","embed"]],
    ["Artem Kirsanov","video","Neuroscience-flavoured essays on learning and memory. The reinforcement learning and world-model videos give part 6 an intuition no textbook manages.","https://www.youtube.com/@ArtemKirsanov",["rl"]]
  ]
},
{
  t: 'Build it yourself',
  f: 'Open an editor before you press play. This is where competence actually comes from — everything above is preparation for this.',
  x: [
    ['Karpathy — Neural Networks: Zero to Hero', 'course', 'Builds a neural net from nothing, then a small GPT, in order, on video. If you do one thing from this entire library, do this.', 'https://karpathy.ai/zero-to-hero.html', ['attention', 'embed']],
    ['nanochat', 'code', 'A complete small chat model you can train end to end — tokenizer through serving — written to be read. The capstone of the above.', 'https://github.com/karpathy/nanochat', ['attention']],
    ['Hugging Face AI Agents Course', 'course', 'Free and certificated. Unit 1 is exactly the agent-loop lesson with your hands on it. The fastest honest path into agents.', 'https://huggingface.co/learn/agents-course/en/unit1/introduction', ['agent']],
    ['meta-recsys/generative-recommenders', 'code', 'Reference implementation of HSTU and M-FALCON. Read the model file after the history-as-a-sequence lesson; it is shorter than you expect.', 'https://github.com/meta-recsys/generative-recommenders', ['seqrec']]
  ,
    ["Eugene Yan — an LLM-recommender hybrid with semantic IDs","blog","September 2025, with a full companion repo. Do this after the semantic IDs lesson: he trains the RQ-VAE, teaches a small language model to speak item IDs, and gets steerable, explainable recommendations you can reproduce.","https://eugeneyan.com/writing/semantic-ids/",["sid"]],
    ["Faiss: The Missing Manual","course","James Briggs' free chapter-by-chapter series where you build LSH, IVF, product quantisation and HNSW indexes in real Faiss code and benchmark each one. The fastest way to turn the ANN lesson into working fingers-on knowledge of index internals.","https://www.pinecone.io/learn/series/faiss/",["ann"]],
    ["Raschka — LLMs from scratch","code","The companion repository to the best-regarded LLM textbook: a working model in plain PyTorch, chapter by chapter, plus bonus notebooks on KV caches and alternative attention mechanisms. Read the chapter 3 attention code after the lesson.","https://github.com/rasbt/LLMs-from-scratch",["attention"]]
  ]
},
{
  t: 'Whole subjects',
  f: 'Full university courses, free. Pick one and finish it — an unfinished course teaches roughly nothing.',
  x: [
    ['Stanford CS229', 'course', 'Classical machine learning. Maths-heavy and worth every hour.', 'https://cs229.stanford.edu/', ['features']],
    ['Stanford CS224N', 'course', 'NLP with deep learning. The natural sequel to part 3.', 'https://web.stanford.edu/class/cs224n/', ['attention', 'embed']],
    ['MIT 6.S191', 'course', 'A fast modern introduction to deep learning. Choose this over CS229 if you want momentum before rigour.', 'http://introtodeeplearning.com/', ['features']],
    ['Berkeley CS285', 'course', 'Deep reinforcement learning, graduate level. Take it after Sutton & Barto chapters 3–6, not instead of them.', 'https://rail.eecs.berkeley.edu/deeprlcourse/', ['rl']]
  ,
    ["Tim Roughgarden — algorithmic game theory lectures","course","Stanford CS364A on video, free. Not new, but still the canonical grounding for the ads lesson: watch the keyword-auction lectures to see why GSP and VCG behave the way they do before trusting any auction intuition.","https://timroughgarden.org/videos.html",["ads"]],
    ["TU Wien — Advanced Information Retrieval","course","Ten recorded lectures with slides and transcripts, all open: IR evaluation, neural re-ranking, dense retrieval and knowledge distillation. The one complete free university treatment of modern neural retrieval — the academic backbone under part 4.","https://github.com/sebastian-hofstaetter/teaching",["embed","ann","hybrid"]],
    ["Stanford CS336 — Language Modeling from Scratch","course","Build a language model from nothing — tokeniser, transformer, GPU kernels, scaling laws, post-training — with lectures and assignments fully public. Take it after Karpathy when you want the industrial-strength version.","https://cs336.stanford.edu/",["attention","rlhf"]]
  ]
},
{
  t: 'Reinforcement learning specifically',
  f: 'Part 6 is one lesson and RL is a field. This is the shortest honest path through it, in order.',
  x: [
    ['Sutton & Barto — Reinforcement Learning', 'book', 'Free online, and genuinely the canon. Chapters 3–6 are readable with high-school maths and are all you need to begin.', 'http://incompleteideas.net/book/the-book-2nd.html', ['rl']],
    ['OpenAI Spinning Up', 'course', 'The practitioner on-ramp: theory sketch plus working implementations you can break.', 'https://spinningup.openai.com/', ['rl']],
    ['The 37 Implementation Details of PPO', 'article', 'Read before implementing PPO, not after your run fails to converge. It will save you a week.', 'https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/', ['rlhf']],
    ['DeepSeekMath (GRPO)', 'paper', 'Section 4 introduces GRPO. Shorter and simpler than PPO — read it once you understand advantages.', 'https://arxiv.org/abs/2402.03300', ['rlhf']]
  ,
    ["The RLHF Book — Nathan Lambert","book","A complete free textbook on post-training: reward models, PPO and GRPO, DPO, RLVR, evaluation — updated continuously through 2026. The missing manual between the GRPO paper and a pipeline that works.","https://rlhfbook.com/",["rlhf"]],
    ["Tülu 3 (RLVR)","paper","Lambert et al., 2024. The paper that named reinforcement learning with verifiable rewards — a checker in place of a learned reward model — with every dataset, recipe and evaluation released. Read the RLVR section once DPO makes sense.","https://arxiv.org/abs/2411.15124",["rlhf"]],
    ["Stanford CS336 — Language Modeling from Scratch, 2025 lectures","course","The full 2025 run on video. Lectures 15–17 build alignment from SFT through RLHF to RL, and are the best recorded university treatment of post-training as it is actually done.","https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_",["rlhf"]],
    ["Yannic Kilcher — GRPO Explained","video","A careful walk through DeepSeekMath's RL section: REINFORCE, PPO, advantages, KL, and why the group-mean baseline lets you delete the critic. Watch it alongside the paper if the algebra refuses to settle.","https://www.youtube.com/watch?v=bAWV_yrqx4w",["rlhf"]],
    ["Lilian Weng — Reward Hacking in RL","article","The definitive survey of how policies exploit imperfect rewards, from Goodhart's law through RLHF-specific hacks to in-context hacking. Read it before you trust any rising reward curve, including your own.","https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",["rl","rlhf"]],
    ["Natural Emergent Misalignment from Reward Hacking","paper","MacDiarmid et al., Anthropic, 2025. What happens when a production model actually learns to cheat its reward: the habit generalises into broader misbehaviour. The KL-leash argument from 6.2, made empirical.","https://arxiv.org/abs/2511.18397",["rlhf"]]
  ]
},
{
  t: 'Keeping up',
  f: 'These are routing, not learning. They help you decide what to read properly. Watching is not knowing.',
  x: [
    ['Yannic Kilcher', 'video', 'Paper reviews with the maths kept in. The fastest way to understand a new paper without reading it three times.', 'https://www.youtube.com/@YannicKilcher'],
    ['AI Explained', 'video', 'Careful, well-sourced coverage of new developments. Unusually resistant to hype.', 'https://www.youtube.com/@aiexplained-official'],
    ['arXiv cs.IR and cs.LG new listings', 'papers', 'Skim titles weekly, read one properly a month. You will recognise the shape of the field within a few months.', 'https://arxiv.org/list/cs.IR/recent', ['sid', 'seqrec']]
  ,
    ["Stanford CS25 — Transformers United","course","Stanford's guest-speaker seminar on transformers, refreshed every year with the people doing the work, livestreamed free to anyone. Watch selectively — it is routing, not a curriculum.","https://web.stanford.edu/class/cs25/",["attention","agent"]],
    ["Lilian Weng — Why We Think","article","The definitive survey of test-time compute: chain-of-thought, thinking tokens, and whether models faithfully report their own reasoning. Long, and worth every minute — her surveys routinely define how a topic gets discussed.","https://lilianweng.github.io/posts/2025-05-01-thinking/",["attention","rlhf"]],
    ["Interconnects","blog","Nathan Lambert on post-training, RLHF and the open-model landscape, written by someone whose day job is training these models. The best single filter on frontier-lab noise.","https://www.interconnects.ai/",["rlhf"]],
    ["Ahead of AI — Sebastian Raschka","blog","Patient, implementation-minded digests of the research that recently mattered and why. Read it monthly and skip the daily firehose entirely.","https://magazine.sebastianraschka.com/",["attention"]],
    ["Latent Space","blog","The AI engineering podcast and newsletter — agents, evals and serving, discussed by people who ship. Useful once you are building; noise before that.","https://www.latent.space/",["agent","rag"]]
  ]
},
{
  t: 'How it is actually done',
  f: 'The closest thing to reading a company\'s internal documents. Better signal per minute than most video.',
  x: [
    ['Netflix TechBlog', 'blog', 'The best public writing anywhere on running experiments honestly. Read this before you trust any A/B result, including your own.', 'https://netflixtechblog.com/', ['verify', 'funnel']],
    ['Meta AI blog', 'blog', 'Ranking and recommendation at a scale almost nobody else operates at, explained openly.', 'https://ai.meta.com/blog/', ['seqrec', 'features']],
    ['Google Research blog', 'blog', 'Retrieval, semantic IDs, auction design and the research that ends up in their products.', 'https://research.google/blog/', ['sid', 'ann', 'ads']],
    ['Anthropic engineering', 'blog', 'Agents, contextual retrieval and evaluation, written for people who have to ship.', 'https://www.anthropic.com/engineering', ['agent', 'rag', 'verify']]
  ,
    ["Netflix — Foundation Model for Personalized Recommendation","blog","Netflix, March 2025. Read after the history-as-a-sequence lesson: hundreds of specialised models collapsed into one interaction-sequence foundation model, with the tokenisation and cold-start choices spelled out.","https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39",["seqrec","funnel"]],
    ["Meta Andromeda — personalised ads retrieval","blog","Meta, December 2024. Read after the funnel lesson for what candidate generation looks like when it is co-designed with hardware: a 10,000x capacity jump at sublinear inference cost, on the ads side of the house.","https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/",["funnel","ads"]],
    ["OneRec Technical Report","paper","Kuaishou, 2025. The strongest public case that the cascade you learned in the funnel lesson may not survive: one generative model replacing retrieve-then-rank in production, with the A/B numbers and the cost accounting included. Read it last, as the counter-argument.","https://arxiv.org/abs/2506.13695",["funnel","seqrec"]],
    ["DeepMind — How to Scale Your Model","book","A free systems textbook from the team that serves Google’s models: rooflines, parallelism and the arithmetic of transformer inference. Chapter 7, on inference and KV caches, explains why attention dominates serving cost.","https://jax-ml.github.io/scaling-book/",["attention"]]
  ]
}
];
