import type { RefreshableRoute } from "../../client/data/siteRefreshContent";

export interface PageRefreshSpec {
  route: RefreshableRoute;
  label: string;
  sourceFile: string;
  updateTargets: string[];
  preserve: string[];
  uxGoals: string[];
}

export const pageRefreshSpecs: PageRefreshSpec[] = [
  {
    route: "/ai-playground",
    label: "AI Playground",
    sourceFile: "client/pages/AIPlayground.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "product radar framing",
      "the top-level explanation of what the demos teach right now",
    ],
    preserve: [
      "interactive demo cards",
      "featured deep-dive demos",
      "generator mechanics and buttons",
    ],
    uxGoals: [
      "lead with current relevance instead of generic excitement",
      "make the first screen scan quickly on mobile and desktop",
      "keep the tone premium and product-like",
    ],
  },
  {
    route: "/ai-discoveries",
    label: "AI Discoveries",
    sourceFile: "client/pages/AIDiscoveries.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "frontier discoveries module",
      "context that bridges historical milestones to current research",
    ],
    preserve: [
      "historical discovery archive",
      "filtering and sorting controls",
      "interactive demos and modal details",
    ],
    uxGoals: [
      "reduce cognitive load before the long timeline begins",
      "clarify why the frontier module matters now",
      "keep the page feeling scholarly but readable",
    ],
  },
  {
    route: "/ai-tools",
    label: "AI Tools",
    sourceFile: "client/pages/AITools.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "latest product launches module",
      "current AI use-case framing",
    ],
    preserve: [
      "profession cards",
      "tool comparisons",
      "filters and sorting",
    ],
    uxGoals: [
      "make practical value obvious in the first viewport",
      "keep the top of the page current without bloating it",
      "preserve fast scanning across professions",
    ],
  },
  {
    route: "/ai-companies",
    label: "AI Companies",
    sourceFile: "client/pages/AICompanies.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "startup watchlist",
      "the context that explains who matters now and why",
    ],
    preserve: [
      "large company directory",
      "filters and sort controls",
      "company detail cards and modal content",
    ],
    uxGoals: [
      "make the market map feel contemporary and curated",
      "avoid overwhelming users before the grid",
      "keep the page readable despite its size",
    ],
  },
  {
    route: "/ai-projects",
    label: "AI Projects",
    sourceFile: "client/pages/AIProjects.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "build-now tracks",
      "current framing for high-leverage projects",
    ],
    preserve: [
      "project library",
      "filters and code samples",
      "difficulty and category structure",
    ],
    uxGoals: [
      "help users choose what to build first",
      "separate timely project ideas from the evergreen library",
      "keep the page ambitious but navigable",
    ],
  },
  {
    route: "/prompt-engineering",
    label: "Prompt Engineering",
    sourceFile: "client/pages/PromptEngineering.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "prompt-pattern signals for each tab",
      "context around modern prompt design",
    ],
    preserve: [
      "examples and techniques tabs",
      "playground analyzer",
      "modal deep dives",
    ],
    uxGoals: [
      "make the page feel like a working lab, not a static tutorial",
      "keep the tab system easy to understand",
      "tie prompting advice to current agent workflows",
    ],
  },
  {
    route: "/ai-agent-training",
    label: "AI Agent Training",
    sourceFile: "client/pages/AIAgentTraining.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "frontier signals module",
      "framing around production agent patterns",
    ],
    preserve: [
      "training examples",
      "advanced techniques tab",
      "agent builder playground",
    ],
    uxGoals: [
      "make the first screen more decisive and less generic",
      "highlight what changed in real agent systems",
      "maintain credibility for technical readers",
    ],
  },
  {
    route: "/ai-champions",
    label: "AI Champions",
    sourceFile: "client/pages/AIChampions.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "overview framing before the victory grid",
      "how the page connects to discovery and games pages",
    ],
    preserve: [
      "historic victory cards",
      "playable demos",
      "footer context and navigation",
    ],
    uxGoals: [
      "make the page feel intentional rather than arcade-like",
      "keep the matchup grid readable and energetic",
      "improve orientation before users open a modal",
    ],
  },
  {
    route: "/resume-builder",
    label: "Resume Builder",
    sourceFile: "client/pages/ResumeBuilder.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "prompt-template positioning",
      "how the page explains the resource set",
    ],
    preserve: [
      "public resume link",
      "profile links",
      "existing prompt templates",
    ],
    uxGoals: [
      "keep the page practical and calm",
      "make it immediately clear what the user can do here",
      "treat it like a polished toolkit, not a placeholder",
    ],
  },
  {
    route: "/games",
    label: "Games",
    sourceFile: "client/pages/Games.tsx",
    updateTargets: [
      "top-of-page framing copy",
      "top-level framing before the game grid",
      "how the page explains its relationship to the rest of the portfolio",
    ],
    preserve: [
      "game cards",
      "embedded game experiences",
      "footer navigation and skills context",
    ],
    uxGoals: [
      "balance playfulness with a premium portfolio tone",
      "make the page feel cohesive with the other subpages",
      "keep the game grid easy to scan",
    ],
  },
];
