export type RefreshableRoute =
  | "/ai-playground"
  | "/ai-discoveries"
  | "/ai-tools"
  | "/ai-companies"
  | "/ai-projects"
  | "/prompt-engineering"
  | "/ai-agent-training"
  | "/ai-champions"
  | "/resume-builder"
  | "/games";

export interface PageRefreshContent {
  route: RefreshableRoute;
  eyebrow: string;
  title: string;
  description: string;
  chips: string[];
  refreshSummary: string;
  updatedAtLabel: string;
}

export interface SiteRefreshMeta {
  headline: string;
  description: string;
  updatedAtLabel: string;
}

export const siteRefreshMeta: SiteRefreshMeta = {
  "headline": "Weekly AI site refresh",
  "description": "A scheduled agent pulls the week's most relevant AI research, product launches, and startup moves into the main AI pages, then deploys the result through the normal GitHub-to-hosting pipeline.",
  "updatedAtLabel": "April 14, 2026"
};

export const pageRefreshContentByRoute: Record<
  RefreshableRoute,
  PageRefreshContent
> = {
  "/ai-playground": {
    "route": "/ai-playground",
    "eyebrow": "Interactive AI demos",
    "title": "Hands‑on AI experiences that showcase current product power",
    "description": "Playable generators and demos for image synthesis, language reasoning, and agent behavior, alongside a weekly radar of product capabilities worth trying right now.",
    "chips": [
      "Current product radar",
      "Live demo framing",
      "Deep‑dive showcase"
    ],
    "refreshSummary": "The product radar and demo commentary refresh weekly; the interactive generators and featured deep dives stay stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/ai-discoveries": {
    "route": "/ai-discoveries",
    "eyebrow": "AI history and frontier research",
    "title": "Foundations of modern AI and the breakthroughs shaping tomorrow",
    "description": "A timeline of AI milestones from early perceptrons to today's frontier models, with filters, embedded demos, and a research module tracking what's changing now.",
    "chips": [
      "Historical milestones",
      "Frontier research updates",
      "Filter by decade"
    ],
    "refreshSummary": "The frontier‑research module refreshes weekly; the historical archive and interactive demos stay stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/ai-tools": {
    "route": "/ai-tools",
    "eyebrow": "AI workflow index",
    "title": "AI tools mapped to real professional workflows",
    "description": "AI tools organized by profession, with current launches and use cases highlighted so it's easy to find the right tool for real work.",
    "chips": [
      "Profession‑based recommendations",
      "Latest launches and use cases",
      "Filter by impact"
    ],
    "refreshSummary": "The launch snapshot and use‑case framing refresh weekly; the profession catalog and filters stay stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/ai-companies": {
    "route": "/ai-companies",
    "eyebrow": "AI market map",
    "title": "The AI company landscape, from frontier labs to workflow startups",
    "description": "Frontier labs, applied‑AI leaders, and new entrants — filterable by sector, valuation, and headcount, with a live watchlist of recent additions.",
    "chips": [
      "Established labs and newer entrants",
      "Filter by sector, valuation, and headcount",
      "Live watchlist for recent additions"
    ],
    "refreshSummary": "The startup watchlist and market context refresh weekly; the larger company directory stays stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/ai-projects": {
    "route": "/ai-projects",
    "eyebrow": "AI build guide",
    "title": "Common AI project patterns, with enough depth to choose and build the right one",
    "description": "Build‑now tracks, filters by difficulty and category, and ready‑to‑study code examples so you can pick the right project and ship it end to end.",
    "chips": [
      "Build‑now project tracks",
      "Difficulty and category filters",
      "Code examples and implementation notes"
    ],
    "refreshSummary": "The build‑now tracks refresh weekly; the evergreen project library and code examples stay stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/prompt-engineering": {
    "route": "/prompt-engineering",
    "eyebrow": "Prompt design lab",
    "title": "Prompting patterns, examples, and practice workflows for modern AI systems",
    "description": "Prompt patterns, worked examples, and a practice playground covering modern agent‑style prompting across research, coding, and operations.",
    "chips": [
      "Examples, techniques, and playground",
      "Modern agent‑style prompting signals",
      "Practice‑ready analyzer flow"
    ],
    "refreshSummary": "The prompt‑pattern signals refresh weekly; the examples, techniques, and analyzer flow stay stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/ai-agent-training": {
    "route": "/ai-agent-training",
    "eyebrow": "Agent systems workshop",
    "title": "How to frame, train, and evaluate AI agents that operate in real workflows",
    "description": "Production agent examples, advanced training and evaluation techniques, and an interactive builder playground for sketching real agent systems.",
    "chips": [
      "Production‑agent examples",
      "Training and evaluation techniques",
      "Interactive builder playground"
    ],
    "refreshSummary": "The frontier‑signals module refreshes weekly; the examples, techniques, and builder stay stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/ai-champions": {
    "route": "/ai-champions",
    "eyebrow": "AI competition history",
    "title": "Historic moments when AI systems beat world‑class human champions",
    "description": "Case studies of AI systems — Deep Blue, AlphaGo, AlphaZero, Libratus, and more — defeating the best human players across chess, Go, poker, and real‑time strategy.",
    "chips": [
      "Historic matchups and context",
      "Playable demos where available",
      "Cross‑links into discoveries and games"
    ],
    "refreshSummary": "The overview narrative and cross‑links refresh weekly; the matchup library and embedded demos stay stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/resume-builder": {
    "route": "/resume-builder",
    "eyebrow": "Career toolkit",
    "title": "Resume assets, profile links, and reusable prompt templates in one place",
    "description": "Public resume, profile links, and copy‑ready prompt templates for recruiter‑safe candidate summaries — a single toolkit for career conversations.",
    "chips": [
      "Current public resume",
      "LinkedIn and GitHub references",
      "Copy‑ready AI prompt templates"
    ],
    "refreshSummary": "The framing and prompt‑template positioning refresh weekly; the direct resume and profile links stay stable.",
    "updatedAtLabel": "April 14, 2026"
  },
  "/games": {
    "route": "/games",
    "eyebrow": "Interactive portfolio games",
    "title": "Playable experiences that showcase strategy, AI thinking, and experimentation",
    "description": "Strategy, arcade, and educational games built as interactive portfolio pieces, cross‑linked with the AI‑competition history.",
    "chips": [
      "Strategy, arcade, and educational games",
      "Playable components embedded in‑page",
      "Cross‑links into AI competition history"
    ],
    "refreshSummary": "Section priorities and framing refresh weekly; the playable components and cross‑links stay stable.",
    "updatedAtLabel": "April 14, 2026"
  }
};

export function getPageRefreshContent(route: RefreshableRoute) {
  return pageRefreshContentByRoute[route];
}
