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
];
