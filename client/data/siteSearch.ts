import {
  agentTabSignals,
  aiUseCasesNow,
  latestAIProductLaunches,
  latestAIResearchBreakthroughs,
  promptTabSignals,
  startupWatchlist,
} from "./aiSignals";

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

const companyEntries: SiteSearchEntry[] = [
  {
    id: "company-openai",
    title: "OpenAI",
    url: "/ai-companies",
    description:
      "Frontier AI lab behind ChatGPT, Codex, Sora, deep research, and multimodal model products.",
    type: "Company",
    section: "AI Companies",
    keywords: ["chatgpt", "codex", "sora", "openai"],
  },
  {
    id: "company-anthropic",
    title: "Anthropic",
    url: "/ai-companies",
    description:
      "AI safety company behind Claude, Claude Code, MCP, and enterprise-focused agent capabilities.",
    type: "Company",
    section: "AI Companies",
    keywords: ["anthropic", "claude", "mcp", "ai safety"],
  },
  {
    id: "company-google-deepmind",
    title: "Google DeepMind",
    url: "/ai-companies",
    description:
      "Research organization behind Gemini, AlphaFold, robotics work, and major scientific AI breakthroughs.",
    type: "Company",
    section: "AI Companies",
    keywords: ["deepmind", "gemini", "alphafold", "robotics"],
  },
  {
    id: "company-microsoft",
    title: "Microsoft",
    url: "/ai-companies",
    description:
      "Enterprise AI platform and product company spanning Copilot, Azure OpenAI Service, and Microsoft 365.",
    type: "Company",
    section: "AI Companies",
    keywords: ["microsoft", "copilot", "azure openai", "github copilot"],
  },
  {
    id: "company-meta-ai",
    title: "Meta AI",
    url: "/ai-companies",
    description:
      "Meta's AI organization behind Llama, PyTorch, computer vision work, and smart glasses products.",
    type: "Company",
    section: "AI Companies",
    keywords: ["meta", "llama", "pytorch", "ray-ban meta"],
  },
  {
    id: "company-nvidia",
    title: "Nvidia",
    url: "/ai-companies",
    description:
      "Core AI infrastructure company providing GPUs, CUDA, and data-center hardware for modern model training.",
    type: "Company",
    section: "AI Companies",
    keywords: ["nvidia", "cuda", "h100", "ai infrastructure", "gpu"],
  },
  {
    id: "company-mistral",
    title: "Mistral AI",
    url: "/ai-companies",
    description:
      "European AI company focused on open-weight models, document AI, research workflows, and enterprise products.",
    type: "Company",
    section: "AI Companies",
    keywords: ["mistral", "le chat", "ocr", "document ai"],
  },
  {
    id: "company-cursor",
    title: "Cursor",
    url: "/ai-companies",
    description:
      "Agent-first coding workspace built for multi-file edits, repo understanding, and developer supervision loops.",
    type: "Company",
    section: "AI Companies",
    keywords: ["cursor", "coding workspace", "ide", "agents"],
  },
  {
    id: "company-perplexity",
    title: "Perplexity",
    url: "/ai-companies",
    description:
      "Research-oriented AI product company known for answer engines, deep research, and grounded web synthesis.",
    type: "Company",
    section: "AI Companies",
    keywords: ["perplexity", "search", "deep research", "answer engine"],
  },
  {
    id: "company-glean",
    title: "Glean",
    url: "/ai-companies",
    description:
      "Enterprise search and agent platform centered on grounded execution, permissions, and connected knowledge.",
    type: "Company",
    section: "AI Companies",
    keywords: ["glean", "enterprise search", "agents", "knowledge"],
  },
  {
    id: "company-harvey",
    title: "Harvey",
    url: "/ai-companies",
    description:
      "Vertical AI company building professional workflows for legal and enterprise knowledge work.",
    type: "Company",
    section: "AI Companies",
    keywords: ["harvey", "legal ai", "professional services"],
  },
  {
    id: "company-sierra",
    title: "Sierra",
    url: "/ai-companies",
    description:
      "AI company focused on conversational agents and customer-facing execution across business channels.",
    type: "Company",
    section: "AI Companies",
    keywords: ["sierra", "voice ai", "customer agents", "conversation"],
  },
  {
    id: "company-elevenlabs",
    title: "ElevenLabs",
    url: "/ai-companies",
    description:
      "Speech and voice AI company focused on multilingual voice generation and conversational systems.",
    type: "Company",
    section: "AI Companies",
    keywords: ["elevenlabs", "voice ai", "speech", "audio"],
  },
  {
    id: "company-runway",
    title: "Runway",
    url: "/ai-companies",
    description:
      "Creative AI company working on video generation, media tooling, and broader AI-native creative platforms.",
    type: "Company",
    section: "AI Companies",
    keywords: ["runway", "video generation", "creative ai", "media"],
  },
];

const toolEntries: SiteSearchEntry[] = [
  {
    id: "tool-software-developers",
    title: "Software Developers",
    url: "/ai-tools",
    description:
      "Coding assistants and agentic development tools for generation, debugging, documentation, testing, and review.",
    type: "Tool",
    section: "AI Tools",
    keywords: ["software developers", "github copilot", "codex", "cursor", "coding"],
  },
  {
    id: "tool-content-writers",
    title: "Content Writers",
    url: "/ai-tools",
    description:
      "Writing assistants and editorial workflows for drafting, editing, structuring, and content strategy.",
    type: "Tool",
    section: "AI Tools",
    keywords: ["content writers", "chatgpt", "claude", "jasper", "writing"],
  },
  {
    id: "tool-data-scientists",
    title: "Data Scientists",
    url: "/ai-tools",
    description:
      "Data and AI platforms for analysis, AutoML, model development, and enterprise workflows.",
    type: "Tool",
    section: "AI Tools",
    keywords: ["data scientists", "databricks", "h2o", "datarobot", "analysis"],
  },
  {
    id: "tool-teachers",
    title: "Teachers & Educators",
    url: "/ai-tools",
    description:
      "Education-oriented assistants and tutoring tools for lesson planning, feedback, and differentiated support.",
    type: "Tool",
    section: "AI Tools",
    keywords: ["teachers", "educators", "khanmigo", "education", "lesson planning"],
  },
  {
    id: "tool-healthcare",
    title: "Healthcare Professionals",
    url: "/ai-tools",
    description:
      "Clinical and operational AI tools used for documentation, triage support, and workflow acceleration.",
    type: "Tool",
    section: "AI Tools",
    keywords: ["healthcare", "clinical ai", "medical", "documentation"],
  },
];

const projectEntries: SiteSearchEntry[] = [
  {
    id: "project-image-classification",
    title: "Image Classification",
    url: "/ai-projects",
    description:
      "Computer vision starter project with code examples, implementation steps, and practical deployment framing.",
    type: "Project",
    section: "AI Projects",
    keywords: ["image classification", "computer vision", "project", "python"],
  },
  {
    id: "project-object-detection",
    title: "Object Detection",
    url: "/ai-projects",
    description:
      "A practical project for detecting and localizing objects in images or video with model and code guidance.",
    type: "Project",
    section: "AI Projects",
    keywords: ["object detection", "computer vision", "yolo", "project"],
  },
  {
    id: "project-recommendation-systems",
    title: "Recommendation Systems",
    url: "/ai-projects",
    description:
      "Applied AI project patterns for personalized ranking, suggestions, and user-facing recommendations.",
    type: "Project",
    section: "AI Projects",
    keywords: ["recommendation systems", "ranking", "personalization", "project"],
  },
];

const researchEntries: SiteSearchEntry[] = [
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
