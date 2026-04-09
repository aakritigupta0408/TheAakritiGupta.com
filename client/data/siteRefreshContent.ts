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
  "description": "Refreshes the latest-information sections, page shell copy, and content priorities across the main AI pages, then deploys the result through the normal GitHub-to-hosting pipeline.",
  "updatedAtLabel": "April 9, 2026"
};

export const pageRefreshContentByRoute: Record<
  RefreshableRoute,
  PageRefreshContent
> = {
  "/ai-playground": {
    "route": "/ai-playground",
    "eyebrow": "Interactive AI demos",
    "title": "Hands‑on AI experiences that showcase current product power",
    "description": "The playground now opens with the shared level‑one shell, highlighting the latest product radar and demo framing while keeping interactive generators and deep‑dive demos intact.",
    "chips": [
      "Current product radar",
      "Live demo framing",
      "Deep‑dive showcase"
    ],
    "refreshSummary": "Update the product radar and demo framing while preserving the interactive generators and featured deep‑dive demos.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/ai-discoveries": {
    "route": "/ai-discoveries",
    "eyebrow": "AI history and frontier research",
    "title": "Foundations of modern AI and the breakthroughs shaping tomorrow",
    "description": "The archive, filters, and demo overlays remain intact, but the page now enters through the same theme‑compatible hero and level‑one wayfinding used across the rest of the site.",
    "chips": [
      "Historical milestones",
      "Frontier research updates",
      "Filter by decade"
    ],
    "refreshSummary": "Refresh the frontier‑discoveries module and shell copy without disturbing the long historical timeline and interactive demos.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/ai-tools": {
    "route": "/ai-tools",
    "eyebrow": "AI workflow index",
    "title": "AI tools mapped to real professional workflows",
    "description": "This page now sits inside the shared level‑one shell: clearer hierarchy up top, faster sibling‑page navigation, and the original tool research preserved inside a single content frame.",
    "chips": [
      "Profession‑based recommendations",
      "Latest launches and use cases",
      "Filter by impact"
    ],
    "refreshSummary": "Refresh the launch snapshot and practical‑use framing while keeping the profession catalog and filters stable.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/ai-companies": {
    "route": "/ai-companies",
    "eyebrow": "AI market map",
    "title": "The AI company landscape, from frontier labs to workflow startups",
    "description": "The page keeps its company research and filters, but now shares the same theme‑compatible hero, level‑one navigation, and content framing as the other main sections.",
    "chips": [
      "Established labs and newer entrants",
      "Filter by sector, valuation, and headcount",
      "Live watchlist for recent additions"
    ],
    "refreshSummary": "Refresh the startup watchlist, market context, and page framing while preserving the larger company directory.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/ai-projects": {
    "route": "/ai-projects",
    "eyebrow": "AI build guide",
    "title": "Common AI project patterns, with enough depth to choose and build the right one",
    "description": "The long‑form project library now opens with the same shared hierarchy and sibling navigation as the other level‑one pages, while keeping the project cards, filters, code snippets, and modal details intact.",
    "chips": [
      "Build‑now project tracks",
      "Difficulty and category filters",
      "Code examples and implementation notes"
    ],
    "refreshSummary": "Refresh the build‑now tracks and summary framing while preserving the underlying project library and code examples.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/prompt-engineering": {
    "route": "/prompt-engineering",
    "eyebrow": "Prompt design lab",
    "title": "Prompting patterns, examples, and practice workflows for modern AI systems",
    "description": "The examples, techniques, and playground stay intact, but the page now opens with the same level‑one hierarchy and navigation as the other main sections.",
    "chips": [
      "Examples, techniques, and playground",
      "Modern agent‑style prompting signals",
      "Faster navigation across sibling pages"
    ],
    "refreshSummary": "Refresh the latest prompt‑pattern signals and shell copy while preserving the examples, techniques, and analyzer flow.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/ai-agent-training": {
    "route": "/ai-agent-training",
    "eyebrow": "Agent systems workshop",
    "title": "How to frame, train, and evaluate AI agents that operate in real workflows",
    "description": "This page keeps its examples, techniques, and playground, but now uses the shared level‑one shell so the dense training content starts with cleaner hierarchy and faster sibling navigation.",
    "chips": [
      "Production‑agent examples",
      "Training and evaluation techniques",
      "Interactive builder playground"
    ],
    "refreshSummary": "Refresh the frontier‑signals module and page framing while keeping the examples, techniques, and builder intact.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/ai-champions": {
    "route": "/ai-champions",
    "eyebrow": "AI competition history",
    "title": "Historic moments when AI systems beat world‑class human champions",
    "description": "The matchup cards, modal deep dives, and playable demos are still here, but the page now enters through the same shell and level‑one wayfinding as the rest of the site.",
    "chips": [
      "Historic matchups and context",
      "Playable demos where available",
      "Cross‑links into discoveries and games"
    ],
    "refreshSummary": "Refresh the overview framing and UX priorities while preserving the matchup library and embedded demos.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/resume-builder": {
    "route": "/resume-builder",
    "eyebrow": "Career toolkit",
    "title": "Resume assets, profile links, and reusable prompt templates in one place",
    "description": "This page now follows the shared level‑one shell so the practical resume resources feel like part of the same product system instead of a visually separate microsite.",
    "chips": [
      "Current public resume",
      "LinkedIn and GitHub references",
      "Copy‑ready AI prompt templates"
    ],
    "refreshSummary": "Refresh the framing copy and prompt‑template positioning while keeping the direct resume and profile links stable.",
    "updatedAtLabel": "April 9, 2026"
  },
  "/games": {
    "route": "/games",
    "eyebrow": "Interactive portfolio games",
    "title": "Playable experiences that showcase strategy, AI thinking, and experimentation",
    "description": "The games page now enters through the same shared shell as the other level‑one routes, while keeping the game grid, live demos, and supporting context below.",
    "chips": [
      "Strategy, arcade, and educational games",
      "Playable components embedded in‑page",
      "Cross‑links into AI competition history"
    ],
    "refreshSummary": "Refresh the framing and section priorities while preserving the playable components and supporting cross‑links.",
    "updatedAtLabel": "April 9, 2026"
  }
};

export function getPageRefreshContent(route: RefreshableRoute) {
  return pageRefreshContentByRoute[route];
}
