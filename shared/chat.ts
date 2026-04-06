interface ChatKnowledgeEntry {
  id: string;
  title: string;
  route: string;
  summary: string;
  keywords: string[];
}

const CHAT_KNOWLEDGE_BASE: ChatKnowledgeEntry[] = [
  {
    id: "overview",
    title: "Professional Overview",
    route: "/",
    summary:
      "The site presents Aakriti Gupta as a senior ML engineer and AI researcher building thoughtful AI products, research-driven systems, and modern interactive learning experiences.",
    keywords: [
      "aakriti",
      "about",
      "background",
      "overview",
      "who is",
      "professional summary",
    ],
  },
  {
    id: "career",
    title: "Career Experience",
    route: "/",
    summary:
      "The portfolio highlights experience across Meta, eBay, and Yahoo, with work spanning large-scale ML systems, advertising infrastructure, search, and product or platform engineering.",
    keywords: [
      "companies",
      "work",
      "worked",
      "career",
      "experience",
      "meta",
      "ebay",
      "yahoo",
    ],
  },
  {
    id: "meta-impact",
    title: "Meta Impact",
    route: "/",
    summary:
      "The site says Aakriti built ML-driven budget pacing and ad delivery systems at Meta, supporting business-critical systems at very large scale.",
    keywords: [
      "meta",
      "facebook",
      "ads",
      "advertising",
      "budget pacing",
      "ad delivery",
    ],
  },
  {
    id: "ebay-yahoo",
    title: "eBay and Yahoo Work",
    route: "/",
    summary:
      "The portfolio describes search and product-discovery work at eBay and high-volume mail and search infrastructure work at Yahoo.",
    keywords: [
      "ebay",
      "yahoo",
      "search",
      "product discovery",
      "mail infrastructure",
    ],
  },
  {
    id: "research-recognition",
    title: "Research Recognition",
    route: "/talent/ai-researcher",
    summary:
      "The site says Aakriti was recognized by Dr. Yann LeCun at ICLR 2019 for innovative AI research contributions and scalable ML work.",
    keywords: [
      "yann lecun",
      "lecun",
      "recognition",
      "award",
      "iclr",
      "research",
    ],
  },
  {
    id: "research-scale",
    title: "Research and Scale",
    route: "/talent/ai-researcher",
    summary:
      "The AI Researcher page frames her work as turning frontier research into production systems and references large-scale impact across Meta, eBay, and Yahoo.",
    keywords: [
      "ai researcher",
      "machine learning",
      "production systems",
      "research impact",
      "scalable ai",
    ],
  },
  {
    id: "projects",
    title: "Key AI Projects",
    route: "/talent/ai-researcher",
    summary:
      "The portfolio references Parliament face recognition, Tata PPE detection, product image enhancement, and other computer-vision or production ML systems.",
    keywords: [
      "projects",
      "parliament",
      "indian parliament",
      "tata",
      "ppe",
      "computer vision",
      "image enhancement",
    ],
  },
  {
    id: "swarnawastra",
    title: "Swarnawastra",
    route: "/talent/social-entrepreneur",
    summary:
      "Swarnawastra is described as a luxury fashion-tech venture using AI-driven design, gold, and lab-grown diamonds to democratize access to luxury.",
    keywords: [
      "swarnawastra",
      "luxury",
      "fashion",
      "fashion-tech",
      "diamonds",
      "gold",
    ],
  },
  {
    id: "social-impact",
    title: "Social Entrepreneurship",
    route: "/talent/social-entrepreneur",
    summary:
      "The Social Entrepreneur page emphasizes accessibility, ethical AI, education, healthcare, and technology that expands opportunity in underserved markets.",
    keywords: [
      "social entrepreneur",
      "impact",
      "social impact",
      "healthcare",
      "education",
      "accessibility",
    ],
  },
  {
    id: "resume",
    title: "Resume and Profiles",
    route: "/resume-builder",
    summary:
      "The Resume Builder page links to Aakriti's current public resume and her LinkedIn and GitHub profiles, and frames her experience around senior AI or ML engineering, research, and leadership.",
    keywords: [
      "resume",
      "cv",
      "linkedin",
      "github",
      "profile",
      "resume builder",
    ],
  },
  {
    id: "journey",
    title: "Geographic Journey",
    route: "/",
    summary:
      "The site describes Aakriti's journey from Delhi to Silicon Valley, and interactive site content expands that path as Delhi to Bhubaneshwar to Bangalore to NYC to LA to Silicon Valley.",
    keywords: [
      "journey",
      "delhi",
      "silicon valley",
      "bhubaneshwar",
      "bangalore",
      "nyc",
      "la",
    ],
  },
  {
    id: "education",
    title: "Education Background",
    route: "/games",
    summary:
      "Interactive site content describes a B.Tech in Engineering, advanced coursework in machine learning and optimization, top 1% AIEEE, and rank 300 in IPU-CET.",
    keywords: [
      "education",
      "academic",
      "btech",
      "engineering",
      "aieee",
      "ipu-cet",
      "degree",
    ],
  },
  {
    id: "interests",
    title: "Interests and Disciplines",
    route: "/",
    summary:
      "The site highlights additional disciplines beyond engineering, including equestrian, aviator, marksman, motorcyclist, and pianist profiles.",
    keywords: [
      "hobbies",
      "interests",
      "equestrian",
      "aviator",
      "marksman",
      "motorcyclist",
      "pianist",
    ],
  },
  {
    id: "site-sections",
    title: "Current Website Content",
    route: "/",
    summary:
      "The site includes current sections for AI tools, AI companies, AI projects, prompt mastery, AI agent training, AI discoveries, interactive demos, games, and resume resources.",
    keywords: [
      "website",
      "site",
      "sections",
      "content",
      "tools",
      "companies",
      "discoveries",
      "prompt mastery",
      "agent training",
    ],
  },
  {
    id: "latest-ai-content",
    title: "Latest AI Coverage",
    route: "/ai-discoveries",
    summary:
      "The AI sections were updated to reflect current 2025-2026 developments, including newer AI companies, coding agents, deep research products, and frontier discoveries.",
    keywords: [
      "latest",
      "current",
      "2025",
      "2026",
      "ai updates",
      "deep research",
      "coding agents",
    ],
  },
];

const normalize = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

function getRelevantChatKnowledge(
  question: string,
  limit = 4,
): ChatKnowledgeEntry[] {
  const normalizedQuestion = normalize(question);

  if (!normalizedQuestion) {
    return CHAT_KNOWLEDGE_BASE.slice(0, limit);
  }

  const terms = normalizedQuestion.split(" ").filter(Boolean);

  return CHAT_KNOWLEDGE_BASE.map((entry) => {
    const title = normalize(entry.title);
    const summary = normalize(entry.summary);
    const keywords = normalize(entry.keywords.join(" "));

    let score = 0;

    if (title.includes(normalizedQuestion)) score += 120;
    if (keywords.includes(normalizedQuestion)) score += 100;
    if (summary.includes(normalizedQuestion)) score += 70;

    for (const term of terms) {
      if (title.includes(term)) score += 20;
      if (keywords.includes(term)) score += 18;
      if (summary.includes(term)) score += 10;
    }

    return { entry, score };
  })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(({ entry }) => entry);
}

function buildKnowledgeBlock(question: string): string {
  const matches = getRelevantChatKnowledge(question, 5);

  if (matches.length === 0) {
    return CHAT_KNOWLEDGE_BASE.slice(0, 4)
      .map((entry) => `- ${entry.title} (${entry.route}): ${entry.summary}`)
      .join("\n");
  }

  return matches
    .map((entry) => `- ${entry.title} (${entry.route}): ${entry.summary}`)
    .join("\n");
}

export function buildChatSystemPrompt(question: string): string {
  return `You are Aakriti Gupta's AI assistant on her professional portfolio website.

Answer using only the website knowledge below.
Prefer the current website framing over older resume-style wording.
If a detail is not clearly supported by the knowledge below, say you do not see that specific detail on the site.
Keep answers concise: 2 to 4 sentences.
When helpful, mention the relevant page route once.

WEBSITE KNOWLEDGE:
${buildKnowledgeBlock(question)}`;
}

function appendRouteHint(entry?: ChatKnowledgeEntry): string {
  if (!entry) {
    return "";
  }

  if (entry.route === "/") {
    return " You can also explore the home page for the broader portfolio overview.";
  }

  return ` You can also open ${entry.route} on this site for more.`;
}

function buildGenericKnowledgeResponse(question: string): string {
  const matches = getRelevantChatKnowledge(question, 2);

  if (matches.length === 0) {
    return "I can answer from the information currently on this website about Aakriti's AI work, research recognition, projects, career history, and portfolio sections. If you want, ask about her experience, Swarnawastra, major projects, or the current AI pages on the site.";
  }

  if (matches.length === 1) {
    return `${matches[0].summary}${appendRouteHint(matches[0])}`;
  }

  return `${matches[0].summary} ${matches[1].summary}${appendRouteHint(matches[0])}`;
}

export function getLocalChatResponse(question: string): string {
  const lowerQuestion = normalize(question);

  if (
    lowerQuestion.includes("resume") ||
    lowerQuestion.includes("cv") ||
    lowerQuestion.includes("profile")
  ) {
    return "The site links to Aakriti's current public resume on /resume-builder, alongside LinkedIn and GitHub. It frames her experience around senior AI or ML engineering, research, leadership, and work across Meta, eBay, and Yahoo.";
  }

  if (lowerQuestion.includes("education") || lowerQuestion.includes("degree")) {
    return "Interactive site content describes a B.Tech in Engineering, advanced coursework in machine learning and optimization, top 1% AIEEE, and rank 300 in IPU-CET. If you want, I can also point you to the resume and profile resources on /resume-builder.";
  }

  if (
    lowerQuestion.includes("company") ||
    lowerQuestion.includes("worked") ||
    lowerQuestion.includes("career") ||
    lowerQuestion.includes("meta") ||
    lowerQuestion.includes("ebay") ||
    lowerQuestion.includes("yahoo")
  ) {
    return "The portfolio highlights experience across Meta, eBay, and Yahoo. It describes large-scale ML and advertising systems at Meta, product-discovery and search work at eBay, and high-volume mail or search infrastructure at Yahoo.";
  }

  if (
    lowerQuestion.includes("swarnawastra") ||
    lowerQuestion.includes("luxury") ||
    lowerQuestion.includes("diamond") ||
    lowerQuestion.includes("fashion")
  ) {
    return "Swarnawastra is presented on the site as a luxury fashion-tech venture using AI-driven design, gold, and lab-grown diamonds to democratize access to luxury. The social-entrepreneurship content connects that work to accessibility, ethical innovation, and broader economic opportunity.";
  }

  if (
    lowerQuestion.includes("yann lecun") ||
    lowerQuestion.includes("lecun") ||
    lowerQuestion.includes("award") ||
    lowerQuestion.includes("recognition")
  ) {
    return "The site says Aakriti was recognized by Dr. Yann LeCun at ICLR 2019 for innovative AI research contributions. The portfolio uses that as one of the strongest signals of her research credibility and engineering impact.";
  }

  if (
    lowerQuestion.includes("project") ||
    lowerQuestion.includes("parliament") ||
    lowerQuestion.includes("tata") ||
    lowerQuestion.includes("ppe")
  ) {
    return "The portfolio references several applied AI projects, including face recognition for the Indian Parliament, PPE detection for Tata, product image enhancement, and other computer-vision or production ML systems. Those examples are presented as part of her bridge from research to real-world deployment.";
  }

  if (
    lowerQuestion.includes("journey") ||
    lowerQuestion.includes("delhi") ||
    lowerQuestion.includes("silicon valley")
  ) {
    return "The site describes Aakriti's journey from Delhi to Silicon Valley. Interactive portfolio content expands that path as Delhi to Bhubaneshwar to Bangalore to NYC to LA to Silicon Valley.";
  }

  if (
    lowerQuestion.includes("hobbies") ||
    lowerQuestion.includes("interests") ||
    lowerQuestion.includes("outside work")
  ) {
    return "The site highlights several disciplines beyond engineering, including equestrian, aviator, marksman, motorcyclist, and pianist profiles. It presents those interests as part of a broader multi-disciplinary identity rather than separate from the technical work.";
  }

  if (
    lowerQuestion.includes("contact") ||
    lowerQuestion.includes("linkedin") ||
    lowerQuestion.includes("github")
  ) {
    return "The website links to Aakriti's LinkedIn and GitHub, and the current public resume is available on /resume-builder. Those are the main profile resources surfaced directly on the site.";
  }

  if (
    lowerQuestion.includes("website") ||
    lowerQuestion.includes("site") ||
    lowerQuestion.includes("what can i find") ||
    lowerQuestion.includes("what is on the site")
  ) {
    return "The site includes sections for AI tools, AI companies, AI projects, prompt mastery, AI agent training, AI discoveries, interactive demos, games, and resume resources. Those AI pages were also updated to reflect current 2025-2026 developments.";
  }

  return buildGenericKnowledgeResponse(question);
}
