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
  "updatedAtLabel": "June 15, 2026"
};

export const pageRefreshContentByRoute: Record<
  RefreshableRoute,
  PageRefreshContent
> = {
  "/ai-playground": {
    "route": "/ai-playground",
    "eyebrow": "Interactive AI demos",
    "title": "Live AI Experiments that Show What’s Possible Today",
    "description": "Explore real‑time generators for image synthesis, language reasoning, and agent behavior, and see the newest product capabilities highlighted in our weekly radar.",
    "chips": [
      "Current product radar",
      "Live demo cards",
      "Deep‑dive showcases"
    ],
    "refreshSummary": "The product radar and demo commentary refresh weekly; the interactive generators and featured deep dives stay stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/ai-discoveries": {
    "route": "/ai-discoveries",
    "eyebrow": "AI history and frontier research",
    "title": "From Perceptrons to Frontier Models: The AI Timeline You Can Explore",
    "description": "A curated timeline of AI milestones, paired with a frontier research module that explains why today’s breakthroughs matter, complete with embedded demos and filters.",
    "chips": [
      "Historical milestones",
      "Frontier research updates",
      "Filter by decade"
    ],
    "refreshSummary": "The frontier‑research module refreshes weekly; the historical archive and interactive demos stay stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/ai-tools": {
    "route": "/ai-tools",
    "eyebrow": "AI workflow index",
    "title": "AI Tools Aligned to Real‑World Workflows",
    "description": "Browse tools organized by profession, see the latest launches and use‑case framing, and filter by impact to find the right solution fast.",
    "chips": [
      "Profession‑based recommendations",
      "Latest launches and use cases",
      "Filter by impact"
    ],
    "refreshSummary": "The launch snapshot and use‑case framing refresh weekly; the profession catalog and filters stay stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/ai-companies": {
    "route": "/ai-companies",
    "eyebrow": "AI market map",
    "title": "The AI Company Landscape, From Labs to Startups",
    "description": "Explore frontier labs, applied‑AI leaders, and new entrants with filters by sector, valuation, and headcount, plus a live watchlist of recent additions.",
    "chips": [
      "Established labs and newer entrants",
      "Filter by sector, valuation, and headcount",
      "Live watchlist for recent additions"
    ],
    "refreshSummary": "The startup watchlist and market context refresh weekly; the larger company directory stays stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/ai-projects": {
    "route": "/ai-projects",
    "eyebrow": "AI build guide",
    "title": "Build‑Now Projects: Pick, Learn, Ship",
    "description": "Choose from build‑now tracks, filter by difficulty and category, and study ready‑to‑use code examples to launch a project end to end.",
    "chips": [
      "Build‑now project tracks",
      "Difficulty and category filters",
      "Code examples and implementation notes"
    ],
    "refreshSummary": "The build‑now tracks refresh weekly; the evergreen project library and code examples stay stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/prompt-engineering": {
    "route": "/prompt-engineering",
    "eyebrow": "Prompt design lab",
    "title": "Prompting Patterns for Modern Agent Workflows",
    "description": "Explore prompt patterns, worked examples, and a practice playground that covers agent‑style prompting across research, coding, and operations.",
    "chips": [
      "Examples, techniques, and playground",
      "Modern agent‑style prompting signals",
      "Practice‑ready analyzer flow"
    ],
    "refreshSummary": "The prompt‑pattern signals refresh weekly; the examples, techniques, and analyzer flow stay stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/ai-agent-training": {
    "route": "/ai-agent-training",
    "eyebrow": "Agent systems workshop",
    "title": "Train and Evaluate AI Agents for Real Workflows",
    "description": "See production agent examples, advanced training and evaluation techniques, and an interactive builder playground to sketch real agent systems.",
    "chips": [
      "Production‑agent examples",
      "Training and evaluation techniques",
      "Interactive builder playground"
    ],
    "refreshSummary": "The frontier‑signals module refreshes weekly; the examples, techniques, and builder stay stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/ai-champions": {
    "route": "/ai-champions",
    "eyebrow": "AI competition history",
    "title": "Historic AI Victories Over Human Champions",
    "description": "Read case studies of AI systems that beat world‑class players, play demos where available, and explore links to discoveries and games.",
    "chips": [
      "Historic matchups and context",
      "Playable demos where available",
      "Cross‑links into discoveries and games"
    ],
    "refreshSummary": "The overview narrative and cross‑links refresh weekly; the matchup library and embedded demos stay stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/resume-builder": {
    "route": "/resume-builder",
    "eyebrow": "Career toolkit",
    "title": "Resume Assets and Prompt Templates in One Place",
    "description": "Access a public resume, profile links, and copy‑ready prompt templates for recruiter‑safe candidate summaries—your polished career toolkit.",
    "chips": [
      "Current public resume",
      "LinkedIn and GitHub references",
      "Copy‑ready AI prompt templates"
    ],
    "refreshSummary": "The framing and prompt‑template positioning refresh weekly; the direct resume and profile links stay stable.",
    "updatedAtLabel": "June 15, 2026"
  },
  "/games": {
    "route": "/games",
    "eyebrow": "Interactive portfolio games",
    "title": "Playful AI Experiments That Showcase Strategy and Thinking",
    "description": "Explore strategy, arcade, and educational games built as portfolio pieces, with cross‑links to AI competition history and research milestones.",
    "chips": [
      "Strategy, arcade, and educational games",
      "Playable components embedded in‑page",
      "Cross‑links into AI competition history"
    ],
    "refreshSummary": "Section priorities and framing refresh weekly; the playable components and cross‑links stay stable.",
    "updatedAtLabel": "June 15, 2026"
  }
};

export function getPageRefreshContent(route: RefreshableRoute) {
  return pageRefreshContentByRoute[route];
}
