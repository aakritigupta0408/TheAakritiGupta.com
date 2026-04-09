import {
  agentTabSignals,
  aiUseCasesNow,
  latestAIProductLaunches,
  latestAIResearchBreakthroughs,
  promptTabSignals,
  startupWatchlist,
} from "./aiSignals";
import { companies } from "./companyArchive";
import { projects } from "./projectArchive";
import { professions } from "./toolArchive";
import { victories } from "./victoryArchive";

export interface SiteSearchEntry {
  id: string;
  title: string;
  url: string;
  description: string;
  type: "Page" | "Profile" | "Company" | "Tool" | "Research" | "Project";
  section: string;
  keywords: string[];
}

const pageEntries: SiteSearchEntry[] = [
  {
    id: "page-home",
    title: "Home",
    url: "/",
    description:
      "Overview of Aakriti Gupta's AI work, engineering background, achievements, demos, and resume links.",
    type: "Page",
    section: "Main Pages",
    keywords: ["aakriti gupta", "ml engineer", "ai researcher", "home"],
  },
  {
    id: "page-ai-playground",
    title: "Interactive Demos",
    url: "/ai-playground",
    description:
      "Hands-on AI demos for text generation, image prompting, coding, translation, and practical model workflows.",
    type: "Page",
    section: "Main Pages",
    keywords: ["interactive demos", "playground", "code generator", "story generator"],
  },
  {
    id: "page-trade-recommendation-demo",
    title: "AI Trade Recommendation System Demo",
    url: "/ai-playground/trade-recommendation-system",
    description:
      "Interactive replay of a local-first AI trade recommendation workflow with request budgeting, recommendation gating, loop state, and end-of-day updates.",
    type: "Page",
    section: "Main Pages",
    keywords: ["trade recommendation system", "paper trading", "daily-only", "alphavantage", "interactive replay"],
  },
  {
    id: "page-ai-champions",
    title: "AI vs Humans",
    url: "/ai-champions",
    description:
      "Historic AI victories in chess, Go, poker, and other competitive strategy domains with interactive views.",
    type: "Page",
    section: "Main Pages",
    keywords: ["ai vs humans", "alphago", "deep blue", "libratus", "champions"],
  },
  {
    id: "page-ai-discoveries",
    title: "AI Discoveries",
    url: "/ai-discoveries",
    description:
      "Historical breakthroughs and current frontier discoveries across science, robotics, forecasting, and document AI.",
    type: "Page",
    section: "Main Pages",
    keywords: ["discoveries", "research papers", "breakthroughs", "timeline", "pioneers"],
  },
  {
    id: "page-ai-tools",
    title: "AI Tools",
    url: "/ai-tools",
    description:
      "AI tools, profession-specific recommendations, workflows, pricing, and practical use cases across industries.",
    type: "Page",
    section: "Main Pages",
    keywords: ["tools", "software developers", "teachers", "marketing", "productivity"],
  },
  {
    id: "page-ai-companies",
    title: "AI Companies",
    url: "/ai-companies",
    description:
      "Major AI labs, startups, infrastructure companies, and enterprise players with current product and category snapshots.",
    type: "Page",
    section: "Main Pages",
    keywords: ["companies", "labs", "startups", "openai", "anthropic", "mistral"],
  },
  {
    id: "page-ai-projects",
    title: "AI Projects",
    url: "/ai-projects",
    description:
      "Project ideas, code examples, solution patterns, and practical ways to build common AI systems.",
    type: "Page",
    section: "Main Pages",
    keywords: ["projects", "code examples", "computer vision", "nlp", "automation"],
  },
  {
    id: "page-prompt-mastery",
    title: "Prompt Mastery",
    url: "/prompt-engineering",
    description:
      "Prompt examples, modern agent-style prompting techniques, and an interactive prompt improvement playground.",
    type: "Page",
    section: "Main Pages",
    keywords: ["prompt engineering", "prompt mastery", "chain of thought", "playground"],
  },
  {
    id: "page-agent-training",
    title: "AI Agent Training",
    url: "/ai-agent-training",
    description:
      "Examples, techniques, and builder guidance for production AI agents, tool use, memory, and eval-driven training.",
    type: "Page",
    section: "Main Pages",
    keywords: ["agent training", "agent builder", "tool use", "memory", "evals"],
  },
  {
    id: "page-resume-builder",
    title: "Resume Builder",
    url: "/resume-builder",
    description:
      "Resume prompts, profile positioning guidance, and resources for tailoring job applications.",
    type: "Page",
    section: "Main Pages",
    keywords: ["resume", "job applications", "career", "linkedin", "profiles"],
  },
  {
    id: "page-games",
    title: "Games",
    url: "/games",
    description:
      "Playable games and AI-inspired mechanics spanning strategy, gradients, and classic game loops.",
    type: "Page",
    section: "Main Pages",
    keywords: ["games", "snake", "pacman", "chess", "mario"],
  },
];

const profileEntries: SiteSearchEntry[] = [
  {
    id: "profile-ai-researcher",
    title: "AI Researcher",
    url: "/talent/ai-researcher",
    description:
      "Research excellence, Yann LeCun recognition, scalable ML systems, and computer vision or NLP work.",
    type: "Profile",
    section: "Know More About AG",
    keywords: ["yann lecun", "iclr", "research", "computer vision", "machine learning"],
  },
  {
    id: "profile-social-entrepreneur",
    title: "Social Entrepreneur",
    url: "/talent/social-entrepreneur",
    description:
      "Swarnawastra, ethical innovation, accessibility, and technology built for social impact.",
    type: "Profile",
    section: "Know More About AG",
    keywords: ["swarnawastra", "social entrepreneur", "impact", "luxury tech"],
  },
  {
    id: "profile-marksman",
    title: "Marksman",
    url: "/talent/marksman",
    description:
      "Precision, discipline, and focus applied across training, systems thinking, and execution.",
    type: "Profile",
    section: "Know More About AG",
    keywords: ["marksman", "precision", "discipline", "focus"],
  },
  {
    id: "profile-equestrian",
    title: "Equestrian",
    url: "/talent/equestrian",
    description:
      "Professional horse riding, partnership, athletic discipline, and strategic control.",
    type: "Profile",
    section: "Know More About AG",
    keywords: ["equestrian", "horse riding", "riding", "partnership"],
  },
  {
    id: "profile-aviator",
    title: "Aviator",
    url: "/talent/aviator",
    description:
      "Pilot training, navigation systems, risk management, and systems-level decision making.",
    type: "Profile",
    section: "Know More About AG",
    keywords: ["aviator", "pilot", "aviation", "navigation", "flight"],
  },
  {
    id: "profile-motorcyclist",
    title: "Motorcyclist",
    url: "/talent/motorcyclist",
    description:
      "High-performance riding, precision control, resilience, and speed with discipline.",
    type: "Profile",
    section: "Know More About AG",
    keywords: ["motorcyclist", "motorcycle", "riding", "speed"],
  },
  {
    id: "profile-pianist",
    title: "Pianist",
    url: "/talent/pianist",
    description:
      "Classical and contemporary piano performance, practice discipline, and musical interpretation.",
    type: "Profile",
    section: "Know More About AG",
    keywords: ["pianist", "piano", "music", "classical"],
  },
];

const companyEntries: SiteSearchEntry[] = companies.map((company) => ({
  id: `company-${company.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
  title: company.name,
  url: "/ai-companies",
  description: `${company.description} ${company.currentFocus}`,
  type: "Company",
  section: "AI Companies",
  keywords: [
    company.name,
    company.category,
    ...company.keyProducts,
    ...company.landmarkContributions,
  ],
}));

const toolEntries: SiteSearchEntry[] = professions.map((profession) => ({
  id: `tool-${profession.title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
  title: profession.title,
  url: "/ai-tools",
  description: `${profession.description} ${profession.workflowNow}`,
  type: "Tool",
  section: "AI Tools",
  keywords: [
    profession.title,
    profession.impactLevel,
    profession.primaryTool.name,
    profession.primaryTool.category,
    ...profession.alternativeTools.map((tool) => tool.name),
  ],
}));

const projectEntries: SiteSearchEntry[] = projects.map((project) => ({
  id: `project-${project.title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
  title: project.title,
  url: "/ai-projects",
  description: `${project.summary} ${project.buildNow}`,
  type: "Project",
  section: "AI Projects",
  keywords: [
    project.title,
    project.category,
    ...project.tags,
    ...project.useCases,
  ],
}));

const researchEntries: SiteSearchEntry[] = [
  ...victories.map((victory) => ({
    id: `victory-${victory.id}`,
    title: `${victory.aiName} vs ${victory.opponent}`,
    url: "/ai-champions",
    description: `${victory.game}. ${victory.significance} ${victory.todayContext}`,
    type: "Research" as const,
    section: "AI Champions",
    keywords: [
      victory.aiName,
      victory.opponent,
      victory.game,
      victory.recordType,
      ...victory.methods,
    ],
  })),
  ...latestAIResearchBreakthroughs.map((signal) => ({
    id: `research-${signal.id}`,
    title: signal.title,
    url: "/ai-discoveries",
    description: `${signal.org}. ${signal.summary} ${signal.impact}`,
    type: "Research" as const,
    section: "AI Discoveries",
    keywords: [signal.category, signal.org, signal.date, ...signal.title.toLowerCase().split(" ")],
  })),
  ...latestAIProductLaunches.map((signal) => ({
    id: `launch-${signal.id}`,
    title: signal.title,
    url:
      signal.category === "Coding Agents" || signal.category === "Coding Workspace"
        ? "/ai-tools"
        : signal.category === "Research Agents"
          ? "/prompt-engineering"
          : "/ai-companies",
    description: `${signal.org}. ${signal.summary} ${signal.impact}`,
    type: "Research" as const,
    section: "Current AI Signals",
    keywords: [signal.category, signal.org, signal.date, ...signal.title.toLowerCase().split(" ")],
  })),
  ...aiUseCasesNow.map((signal) => ({
    id: `use-case-${signal.id}`,
    title: signal.title,
    url:
      signal.id === "coding-agents"
        ? "/ai-tools"
        : signal.id === "enterprise-work-ai"
          ? "/ai-companies"
          : "/ai-discoveries",
    description: `${signal.summary} ${signal.signal} ${signal.examples.join(", ")}.`,
    type: "Research" as const,
    section: "Current AI Signals",
    keywords: [signal.id.replace(/-/g, " "), ...signal.examples],
  })),
  ...startupWatchlist.map((item) => ({
    id: `startup-${item.id}`,
    title: item.name,
    url: "/ai-companies",
    description: `${item.focus}. ${item.latestMove} ${item.whyItMatters}`,
    type: "Company" as const,
    section: "Startup Watchlist",
    keywords: [item.name, item.focus, item.date],
  })),
  ...Object.entries(agentTabSignals).flatMap(([tabId, signals]) =>
    signals.map((signal) => ({
      id: `agent-${tabId}-${signal.id}`,
      title: signal.title,
      url: "/ai-agent-training",
      description: `${signal.org}. ${signal.summary} ${signal.impact}`,
      type: "Research" as const,
      section: "AI Agent Training",
      keywords: [tabId, signal.category, signal.org, signal.date],
    })),
  ),
  ...Object.entries(promptTabSignals).flatMap(([tabId, signals]) =>
    signals.map((signal) => ({
      id: `prompt-${tabId}-${signal.id}`,
      title: signal.title,
      url: "/prompt-engineering",
      description: `${signal.org}. ${signal.summary} ${signal.impact}`,
      type: "Research" as const,
      section: "Prompt Mastery",
      keywords: [tabId, signal.category, signal.org, signal.date],
    })),
  ),
];

export const siteSearchEntries: SiteSearchEntry[] = [
  ...pageEntries,
  ...profileEntries,
  ...companyEntries,
  ...toolEntries,
  ...projectEntries,
  ...researchEntries,
];

const normalize = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

export function searchSiteContent(query: string): SiteSearchEntry[] {
  const normalizedQuery = normalize(query);

  if (!normalizedQuery) {
    return [];
  }

  const terms = normalizedQuery.split(" ").filter(Boolean);

  const scoredResults = siteSearchEntries
    .map((entry) => {
      const haystacks = {
        title: normalize(entry.title),
        description: normalize(entry.description),
        section: normalize(entry.section),
        keywords: normalize(entry.keywords.join(" ")),
      };

      let score = 0;

      if (haystacks.title.includes(normalizedQuery)) score += 120;
      if (haystacks.keywords.includes(normalizedQuery)) score += 90;
      if (haystacks.section.includes(normalizedQuery)) score += 40;
      if (haystacks.description.includes(normalizedQuery)) score += 60;

      for (const term of terms) {
        if (haystacks.title.includes(term)) score += 18;
        if (haystacks.keywords.includes(term)) score += 14;
        if (haystacks.description.includes(term)) score += 10;
        if (haystacks.section.includes(term)) score += 6;
      }

      return { entry, score };
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || a.entry.title.localeCompare(b.entry.title));

  const deduped: SiteSearchEntry[] = [];
  const seen = new Set<string>();

  for (const { entry } of scoredResults) {
    const dedupeKey = `${entry.url}::${entry.title}`;
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    deduped.push(entry);

    if (deduped.length === 8) {
      break;
    }
  }

  return deduped;
}
