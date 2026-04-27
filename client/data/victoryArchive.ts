export interface VictorySource {
  label: string;
  url: string;
  kind: "Official history" | "Research paper" | "Lab blog" | "University release";
}

export interface VictoryRecord {
  id: string;
  game: string;
  aiName: string;
  opponent: string;
  year: number;
  location: string;
  icon: string;
  scoreLabel: string;
  recordType: "Champion match" | "Benchmark leap";
  significance: string;
  summary: string;
  methods: string[];
  whyItMattered: string;
  todayContext: string;
  format: string;
  playableDemo: boolean;
  gradient: string;
  accent: string;
  sources: VictorySource[];
}

export const victories: VictoryRecord[] = [
  {
    id: "deep-blue-chess",
    game: "Chess",
    aiName: "Deep Blue",
    opponent: "Garry Kasparov",
    year: 1997,
    location: "New York City",
    icon: "♛",
    scoreLabel: "3.5 - 2.5",
    recordType: "Champion match",
    significance:
      "First computer system to defeat a reigning world chess champion in a standard match.",
    summary:
      "IBM's Deep Blue beat Garry Kasparov in a six-game rematch under standard tournament controls, showing that massive search, domain evaluation, and hardware co-design could overwhelm elite human play in a canonical strategy benchmark.",
    methods: [
      "Massively parallel search",
      "Custom chess processors",
      "Alpha-beta pruning",
      "Evaluation functions",
      "Opening and endgame databases",
    ],
    whyItMattered:
      "Deep Blue established games as public proof points for AI progress and made search, evaluation, and specialized compute feel like serious engineering advantages rather than academic curiosities.",
    todayContext:
      "Modern frontier systems no longer look like Deep Blue architecturally, but the lesson still holds: capability jumps often come from pairing better algorithms with much better infrastructure and tooling.",
    format: "Six-game classical match under standard tournament time controls.",
    playableDemo: true,
    gradient: "from-sky-500 to-indigo-700",
    accent: "border-sky-400/60",
    sources: [
      {
        label: "IBM Deep Blue history",
        url: "https://www.ibm.com/history/deep-blue",
        kind: "Official history",
      },
      {
        label: "IBM Research: Deep Blue",
        url: "https://research.ibm.com/publications/deep-blue",
        kind: "Research paper",
      },
    ],
  },
  {
    id: "alphago-go",
    game: "Go",
    aiName: "AlphaGo",
    opponent: "Lee Sedol",
    year: 2016,
    location: "Seoul, South Korea",
    icon: "⚫",
    scoreLabel: "4 - 1",
    recordType: "Champion match",
    significance:
      "World-champion Go victory that convinced the field deep reinforcement learning could master intuition-heavy planning problems.",
    summary:
      "DeepMind's AlphaGo defeated Lee Sedol four games to one, years ahead of expert expectations, and turned Go into the defining case study for deep neural networks paired with search and self-play.",
    methods: [
      "Policy networks",
      "Value networks",
      "Monte Carlo tree search",
      "Reinforcement learning",
      "Self-play fine-tuning",
    ],
    whyItMattered:
      "AlphaGo shifted mainstream belief about what neural networks could handle. It also made planning plus learning feel like a practical recipe for much harder domains than image classification alone.",
    todayContext:
      "The AlphaGo lineage directly informs later work such as AlphaZero, MuZero, AlphaDev, and broader planning-heavy AI systems used in science and optimization.",
    format: "Five-game professional Go match on a full 19x19 board.",
    playableDemo: true,
    gradient: "from-slate-700 to-black",
    accent: "border-slate-400/50",
    sources: [
      {
        label: "DeepMind AlphaGo overview",
        url: "https://deepmind.google/en/research/alphago/",
        kind: "Lab blog",
      },
      {
        label: "Nature: Mastering the game of Go",
        url: "https://www.nature.com/articles/nature16961",
        kind: "Research paper",
      },
    ],
  },
  {
    id: "libratus-poker",
    game: "Heads-Up No-Limit Texas Hold'em",
    aiName: "Libratus",
    opponent: "Top Human Professionals",
    year: 2017,
    location: "Pittsburgh, Pennsylvania",
    icon: "🂡",
    scoreLabel: "$1.8M in chips",
    recordType: "Champion match",
    significance:
      "First AI to defeat elite human specialists in a flagship imperfect-information game.",
    summary:
      "Carnegie Mellon's Libratus beat four top heads-up no-limit poker professionals over 120,000 hands, proving that AI could outperform humans in domains defined by hidden information, bluffing, and adaptive strategy.",
    methods: [
      "Blueprint strategy computation",
      "Nested subgame solving",
      "Self-improver updates",
      "Game-theoretic reasoning",
      "Large-scale compute",
    ],
    whyItMattered:
      "Poker mattered because it broke the 'perfect information only' narrative. After chess and Go, Libratus showed that uncertainty and deception were no longer safe territory for human advantage.",
    todayContext:
      "This line of work still matters for negotiation, cybersecurity, auctions, and multi-party decision systems where hidden information and adaptive opponents are the norm.",
    format: "20-day, 120,000-hand heads-up no-limit Texas Hold'em competition.",
    playableDemo: true,
    gradient: "from-rose-500 to-red-700",
    accent: "border-rose-400/60",
    sources: [
      {
        label: "CMU: Libratus beats top poker pros",
        url: "https://www.cs.cmu.edu/news/2017/carnegie-mellon-artificial-intelligence-beats-top-poker-pros",
        kind: "University release",
      },
      {
        label: "PubMed: Libratus paper",
        url: "https://pubmed.ncbi.nlm.nih.gov/29249696/",
        kind: "Research paper",
      },
    ],
  },
  {
    id: "alphazero-multiple",
    game: "Chess, Shogi, and Go",
    aiName: "AlphaZero",
    opponent: "Best-in-class engines",
    year: 2018,
    location: "London, United Kingdom",
    icon: "🎯",
    scoreLabel: "Rules only → superhuman play",
    recordType: "Benchmark leap",
    significance:
      "General self-play system that mastered three iconic games from rules alone.",
    summary:
      "AlphaZero learned chess, shogi, and Go from scratch using only the rules, then outperformed the strongest published engines in each domain, demonstrating that a single general reinforcement-learning recipe could replace large amounts of handcrafted expertise.",
    methods: [
      "Self-play reinforcement learning",
      "Single neural network policy-value head",
      "Monte Carlo tree search",
      "No human opening books",
      "General game-playing architecture",
    ],
    whyItMattered:
      "AlphaZero made a strong case that general-purpose learning systems could discover high-level strategy faster than expert-coded engines built over decades.",
    todayContext:
      "Its core lesson shows up far beyond games: if you can define an environment, rewards, and a planning loop, self-play can produce surprisingly transferable strategies.",
    format: "Science evaluation across chess, shogi, and Go benchmark suites.",
    playableDemo: false,
    gradient: "from-violet-500 to-purple-700",
    accent: "border-violet-400/60",
    sources: [
      {
        label: "DeepMind AlphaZero overview",
        url: "https://deepmind.google/en/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/",
        kind: "Lab blog",
      },
      {
        label: "Nature: AlphaGo Zero",
        url: "https://www.nature.com/articles/nature24270",
        kind: "Research paper",
      },
    ],
  },
  {
    id: "openai-five-dota",
    game: "Dota 2",
    aiName: "OpenAI Five",
    opponent: "Team OG",
    year: 2019,
    location: "San Francisco, California",
    icon: "🏆",
    scoreLabel: "2 - 0",
    recordType: "Champion match",
    significance:
      "First AI system to beat reigning world champions in a live esports match.",
    summary:
      "OpenAI Five beat OG, the reigning Dota 2 world champions, in back-to-back live games. The result showed that large-scale reinforcement learning could operate in long-horizon, real-time, multi-agent environments with partial information.",
    methods: [
      "Proximal Policy Optimization",
      "Large-scale self-play",
      "Distributed training",
      "LSTM policies",
      "Continual transfer across patches",
    ],
    whyItMattered:
      "This was the clearest signal that reinforcement learning could handle team coordination and fast-changing environments, not just static boards or turn-based play.",
    todayContext:
      "OpenAI Five is still one of the strongest references for multi-agent coordination, long-horizon credit assignment, and scaling laws inside reinforcement learning systems.",
    format: "OpenAI Five Finals best-of-three exhibition versus reigning Dota 2 champions.",
    playableDemo: false,
    gradient: "from-emerald-500 to-teal-700",
    accent: "border-emerald-400/60",
    sources: [
      {
        label: "OpenAI Five Finals",
        url: "https://openai.com/index/openai-five-finals",
        kind: "Lab blog",
      },
      {
        label: "OpenAI Five defeats Dota 2 world champions",
        url: "https://openai.com/five/",
        kind: "Lab blog",
      },
    ],
  },
  {
    id: "pluribus-poker",
    game: "Six-Player No-Limit Texas Hold'em",
    aiName: "Pluribus",
    opponent: "World-Class Professionals",
    year: 2019,
    location: "Pittsburgh, Pennsylvania",
    icon: "🃏",
    scoreLabel: "Statistically significant win",
    recordType: "Champion match",
    significance:
      "First AI to beat elite professionals in multiplayer poker, not just heads-up play.",
    summary:
      "Pluribus defeated world-class players in six-player no-limit Texas Hold'em, extending poker mastery from two-player settings to the messier multiplayer case where Nash-style guarantees are far harder to exploit directly.",
    methods: [
      "Self-play blueprint strategy",
      "Limited-lookahead search",
      "Imperfect-information solving",
      "Balanced mixed strategies",
      "Low-cost inference at runtime",
    ],
    whyItMattered:
      "Multiplayer poker is much closer to real strategic settings than heads-up poker. Pluribus showed that AI could reason effectively when several strong opponents interact at once.",
    todayContext:
      "This work remains relevant anywhere multiple strategic actors compete under uncertainty, including marketplaces, auctions, and security planning.",
    format: "Professional six-player no-limit Texas Hold'em evaluations across two experiments.",
    playableDemo: false,
    gradient: "from-amber-500 to-orange-700",
    accent: "border-amber-400/60",
    sources: [
      {
        label: "CMU and Facebook AI announce Pluribus",
        url: "https://www.cs.cmu.edu/news/2019/carnegie-mellon-and-facebook-ai-beats-professionals-six-player-poker",
        kind: "University release",
      },
      {
        label: "PubMed: Pluribus paper",
        url: "https://pubmed.ncbi.nlm.nih.gov/31296650/",
        kind: "Research paper",
      },
    ],
  },
  {
    id: "alphastar-starcraft",
    game: "StarCraft II",
    aiName: "AlphaStar",
    opponent: "Top Professional Players",
    year: 2019,
    location: "London, United Kingdom",
    icon: "🚀",
    scoreLabel: "5 - 0 vs MaNa",
    recordType: "Champion match",
    significance:
      "First AI to defeat a top professional StarCraft II player under competitive conditions.",
    summary:
      "DeepMind's AlphaStar beat Team Liquid's Grzegorz 'MaNa' Komincz and later reached Grandmaster level on Battle.net, showing that multi-agent reinforcement learning could scale to one of the most complex real-time strategy games ever used as an AI benchmark.",
    methods: [
      "League training",
      "Imitation learning from replays",
      "Multi-agent reinforcement learning",
      "Recurrent neural networks",
      "Action-rate and camera constraints",
    ],
    whyItMattered:
      "StarCraft II forced AI to handle partial observability, delayed rewards, large action spaces, and messy long-term strategy at the same time.",
    todayContext:
      "AlphaStar remains a strong reference point for general agents that need planning, adaptation, and coordination in dynamic environments closer to real operations than board games are.",
    format: "Professional StarCraft II matches followed by anonymous Battle.net Grandmaster evaluation.",
    playableDemo: false,
    gradient: "from-cyan-500 to-blue-700",
    accent: "border-cyan-400/60",
    sources: [
      {
        label: "DeepMind AlphaStar debut",
        url: "https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/",
        kind: "Lab blog",
      },
      {
        label: "DeepMind AlphaStar Grandmaster update",
        url: "https://deepmind.google/en/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/",
        kind: "Lab blog",
      },
    ],
  },
  {
    id: "cicero-diplomacy",
    game: "Diplomacy",
    aiName: "Cicero",
    opponent: "Human Online Players",
    year: 2022,
    location: "webDiplomacy.net (online)",
    icon: "🤝",
    scoreLabel: "2× avg human score · Top 10%",
    recordType: "Champion match",
    significance:
      "First AI to achieve human-level strategic play in Diplomacy — a game demanding natural language negotiation, alliance-building, and long-horizon planning simultaneously.",
    summary:
      "Meta AI's Cicero played 40 anonymous games on webDiplomacy.net, scoring 25.8% average against the human average of 12.4% and finishing in the top 10% of all participants (second among players with 5+ games). Diplomacy requires building and breaking alliances through natural language across many rounds — capabilities language-only or strategy-only models had never reliably combined at a competitive level.",
    methods: [
      "Strategic language model dialogue",
      "Reinforcement learning planning",
      "Intent-conditioning",
      "Imitation learning from human games",
      "Controllable text generation",
    ],
    whyItMattered:
      "Cicero proved that AI could negotiate through natural language at a competitive level — managing alliances, breaking deals, and tracking trust across a full multi-player competition rather than just reasoning in isolation.",
    todayContext:
      "Cicero's architecture — a language model steered by a strategic planner — anticipates how modern agents combine reasoning models with language output. The Science paper remains a key reference for AI in multi-agent negotiation settings.",
    format: "40 anonymous online games on webDiplomacy.net (August 19 – October 13, 2022).",
    playableDemo: false,
    gradient: "from-pink-500 to-rose-700",
    accent: "border-pink-400/60",
    sources: [
      {
        label: "Science: Human-level play in Diplomacy",
        url: "https://www.science.org/doi/10.1126/science.ade9097",
        kind: "Research paper",
      },
      {
        label: "Meta AI: Cicero blog post",
        url: "https://ai.meta.com/blog/cicero-ai-negotiates-persuades-and-cooperates-with-people/",
        kind: "Lab blog",
      },
    ],
  },
  {
    id: "alphacode2-programming",
    game: "Competitive Programming",
    aiName: "AlphaCode 2",
    opponent: "Competitive Programmers",
    year: 2023,
    location: "Codeforces (online)",
    icon: "💻",
    scoreLabel: "85th percentile on Codeforces",
    recordType: "Benchmark leap",
    significance:
      "First AI to reach the top 15% of competitive programmers on a major live coding platform.",
    summary:
      "Google DeepMind's AlphaCode 2 achieved an estimated Codeforces rating placing it at the 85th percentile of competitive programmers — solving 43% of competition problems across 12 recent contests. It combined Gemini model fine-tuning with large-scale solution sampling and scoring, far exceeding its predecessor's 50th-percentile baseline from 2022.",
    methods: [
      "Gemini model fine-tuning",
      "Large-scale solution sampling",
      "Code scoring and filtering",
      "Execution-based verification",
      "Ensemble selection",
    ],
    whyItMattered:
      "Competitive programming is a strong proxy for rigorous engineering and reasoning. Reaching the 85th percentile meant surpassing most professional developers and many specialist engineers in systematic problem-solving under time constraints.",
    todayContext:
      "AlphaCode 2 validated that large-scale language models, specialized fine-tuning, and compute-intensive sampling could crack previously human-dominant intellectual benchmarks — a pattern that now defines how frontier AI capability is evaluated and extended.",
    format: "Evaluation across 12 recent Codeforces contests totaling 77 problems (December 2023).",
    playableDemo: false,
    gradient: "from-blue-500 to-indigo-700",
    accent: "border-blue-400/60",
    sources: [
      {
        label: "AlphaCode 2 technical report",
        url: "https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf",
        kind: "Research paper",
      },
      {
        label: "DeepMind: Competitive programming with AlphaCode",
        url: "https://deepmind.google/discover/blog/competitive-programming-with-alphacode/",
        kind: "Lab blog",
      },
    ],
  },
  {
    id: "alphaproof-imo-2024",
    game: "International Math Olympiad",
    aiName: "AlphaProof + AlphaGeometry 2",
    opponent: "World's Best Math Students",
    year: 2024,
    location: "Bath, United Kingdom",
    icon: "📐",
    scoreLabel: "28 pts · Silver medal",
    recordType: "Benchmark leap",
    significance:
      "First AI system to solve IMO competition problems at silver-medal level, with solutions formally verified by official IMO judges.",
    summary:
      "DeepMind's AlphaProof and AlphaGeometry 2 solved four of six problems from IMO 2024, earning 28 of 42 points and meeting the silver-medal threshold. Solutions were formally verified and graded by IMO officials Timothy Gowers and Joseph Myers — the same standards applied to human contestants.",
    methods: [
      "Formal theorem proving in Lean 4",
      "Reinforcement learning on proof trees",
      "Geometric reasoning (AlphaGeometry 2)",
      "Self-training from generated problems",
      "Symbolic search over proof steps",
    ],
    whyItMattered:
      "IMO problems require non-routine mathematical discovery that cannot be retrieved or pattern-matched. A silver-medal score meant genuinely constructing proofs that professional mathematicians had to verify — not just passing standardized tests.",
    todayContext:
      "AlphaProof demonstrated that AI could conduct formal mathematical reasoning at world-class competition level. One year later, Gemini Deep Think improved to gold — confirming that AI mathematical reasoning is advancing rapidly enough to reframe collaboration between researchers and AI.",
    format: "Six problems from the 2024 International Mathematical Olympiad; solutions formally verified by IMO officials.",
    playableDemo: false,
    gradient: "from-teal-500 to-cyan-700",
    accent: "border-teal-400/60",
    sources: [
      {
        label: "DeepMind: AI solves IMO problems at silver medal level",
        url: "https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/",
        kind: "Lab blog",
      },
    ],
  },
  {
    id: "o3-arc-agi",
    game: "ARC-AGI Abstract Reasoning",
    aiName: "OpenAI o3",
    opponent: "Human Baseline",
    year: 2024,
    location: "ARC Prize (online)",
    icon: "🧩",
    scoreLabel: "87.5% (human baseline ~85%)",
    recordType: "Benchmark leap",
    significance:
      "First AI to surpass the human performance threshold on ARC-AGI, the leading benchmark for abstract and novel pattern reasoning.",
    summary:
      "OpenAI's o3 scored 87.5% on the ARC-AGI semi-private evaluation in December 2024, exceeding the established human baseline of approximately 85%. ARC-AGI was specifically designed to resist pattern-matching by requiring novel analogical reasoning on tasks structurally unlike anything in training data.",
    methods: [
      "Extended chain-of-thought reasoning",
      "Test-time compute scaling",
      "Reinforcement learning from feedback",
      "Deliberative inference",
      "Iterative self-correction",
    ],
    whyItMattered:
      "ARC-AGI was engineered to be hard for LLMs — novel visual pattern induction rather than knowledge retrieval. Clearing the human baseline was a milestone for AI generalization beyond memorization and the most contested proof point in the AGI debate.",
    todayContext:
      "o3's result shifted the conversation from 'can AI generalize?' to 'what does generalization cost in compute?' It prompted the ARC Prize to release ARC-AGI-2 targeting harder reasoning, acknowledging that the original benchmark no longer reliably separates human and AI capability.",
    format: "800-task semi-private evaluation set from the 2024 ARC Prize competition (December 2024).",
    playableDemo: false,
    gradient: "from-orange-500 to-amber-700",
    accent: "border-orange-400/60",
    sources: [
      {
        label: "ARC Prize: o3 breakthrough",
        url: "https://arcprize.org/blog/oai-o3-pub-breakthrough",
        kind: "Lab blog",
      },
      {
        label: "OpenAI: Introducing o3 and o4-mini",
        url: "https://openai.com/index/introducing-o3-and-o4-mini/",
        kind: "Lab blog",
      },
    ],
  },
  {
    id: "gemini-imo-2025",
    game: "International Math Olympiad",
    aiName: "Gemini Deep Think",
    opponent: "World's Best Math Students",
    year: 2025,
    location: "Gold Coast, Australia",
    icon: "🥇",
    scoreLabel: "35 pts · Gold medal",
    recordType: "Benchmark leap",
    significance:
      "First AI to achieve gold-medal performance at the International Mathematical Olympiad, solving 5 of 6 problems end-to-end in natural language within competition time limits.",
    summary:
      "Google DeepMind's Gemini with Deep Think solved 5 of 6 problems at IMO 2025, scoring 35 of 42 points — gold-medal standard. Critically, it operated entirely in natural language within the 4.5-hour competition window, with no formal proof language required. This built directly on AlphaProof's 2024 silver-medal result.",
    methods: [
      "Extended chain-of-thought reasoning",
      "Deep Think inference mode",
      "Natural language proof generation",
      "Test-time compute scaling",
      "Competition mathematics fine-tuning",
    ],
    whyItMattered:
      "Unlike AlphaProof's 2024 silver, Gemini Deep Think competed end-to-end in natural language under real competition time constraints — the same conditions as human contestants. Gold-medal performance requires discovering original mathematical arguments, not just verifying them.",
    todayContext:
      "The step from silver (2024) to gold (2025) in a single year illustrates the pace at which frontier AI is advancing on hard reasoning. This result has already shifted how research mathematicians think about AI as a genuine collaborator on open problems, not just a verification assistant.",
    format: "Six problems from the 2025 International Mathematical Olympiad; graded under official competition conditions.",
    playableDemo: false,
    gradient: "from-yellow-500 to-orange-600",
    accent: "border-yellow-400/60",
    sources: [
      {
        label: "DeepMind: Gemini achieves gold at IMO 2025",
        url: "https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/",
        kind: "Lab blog",
      },
    ],
  },
];
