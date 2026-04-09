export type ImpactLevel = "Critical" | "High" | "Medium";

export interface ToolEntry {
  name: string;
  link: string;
  sourceLabel: string;
  sourceKind: "Official site" | "Docs" | "Help" | "Press";
  description: string;
  pricingSignal: string;
  category: string;
}

export interface ProfessionProfile {
  id: number;
  title: string;
  icon: string;
  description: string;
  impactLevel: ImpactLevel;
  aiAdoption: number;
  workflowNow: string;
  timeSaved: string;
  primaryTool: ToolEntry;
  alternativeTools: ToolEntry[];
}

export const professions: ProfessionProfile[] = [
  {
    id: 1,
    title: "Software Developers",
    icon: "💻",
    description:
      "Coding work has shifted from autocomplete to multi-step agents that can plan, edit, run commands, and hand back reviewable diffs.",
    impactLevel: "Critical",
    aiAdoption: 96,
    workflowNow:
      "The strongest current workflow is supervised agentic development: quick chat for iteration, cloud agents for longer tasks, and code review kept under human control.",
    timeSaved: "40-70% on scoped implementation and debugging work",
    primaryTool: {
      name: "GitHub Copilot",
      link: "https://github.com/features/copilot",
      sourceLabel: "GitHub Copilot",
      sourceKind: "Official site",
      description:
        "Broadest mainstream coding assistant stack across IDEs, GitHub, CLI, and the newer cloud agent workflow.",
      pricingSignal: "Free trial / paid plans",
      category: "Coding assistant",
    },
    alternativeTools: [
      {
        name: "Codex",
        link: "https://openai.com/index/introducing-codex/",
        sourceLabel: "Introducing Codex",
        sourceKind: "Press",
        description:
          "Cloud software engineering agent for parallel tasks, repo reasoning, and longer-running supervised execution.",
        pricingSignal: "Included with supported ChatGPT plans / API pricing",
        category: "Coding agent",
      },
      {
        name: "Cursor",
        link: "https://cursor.com/product",
        sourceLabel: "Cursor Agent",
        sourceKind: "Official site",
        description:
          "Agent-first editor that combines codebase search, inline edits, cloud agents, and review-oriented workflows.",
        pricingSignal: "Free trial / paid plans",
        category: "AI IDE",
      },
    ],
  },
  {
    id: 2,
    title: "Content Writers",
    icon: "✍️",
    description:
      "Writers now use AI for outlining, restructuring, headline testing, source-backed drafting, and revision loops rather than one-shot blog generation.",
    impactLevel: "High",
    aiAdoption: 91,
    workflowNow:
      "The best writing setups combine fast drafting, voice/tone control, and a second pass for verification and editing.",
    timeSaved: "45-70% on first-draft and revision work",
    primaryTool: {
      name: "ChatGPT",
      link: "https://openai.com/chatgpt/overview/",
      sourceLabel: "ChatGPT overview",
      sourceKind: "Official site",
      description:
        "Flexible writing assistant for outlining, rewriting, summarizing, brainstorming, and web-backed research.",
      pricingSignal: "Free / Plus / Pro / business plans",
      category: "Writing assistant",
    },
    alternativeTools: [
      {
        name: "Claude",
        link: "https://www.anthropic.com/claude",
        sourceLabel: "Claude",
        sourceKind: "Official site",
        description:
          "Strong at long-form drafting, editing, structured revision, and handling larger context windows for document work.",
        pricingSignal: "Free / Pro / Team / Enterprise",
        category: "Writing assistant",
      },
      {
        name: "Jasper",
        link: "https://www.jasper.ai/",
        sourceLabel: "Jasper",
        sourceKind: "Official site",
        description:
          "Marketing-oriented writing platform focused on campaign execution, brand voice, and repeatable content workflows.",
        pricingSignal: "Paid plans / enterprise",
        category: "Marketing writing",
      },
    ],
  },
  {
    id: 3,
    title: "Graphic Designers",
    icon: "🎨",
    description:
      "Design work is increasingly split between ideation, generation, editing, and brand-safe production rather than prompt-only image experiments.",
    impactLevel: "High",
    aiAdoption: 86,
    workflowNow:
      "The strongest stack blends fast concept generation with editing controls, moodboarding, and production-ready design surfaces.",
    timeSaved: "50-80% on concepting and repetitive design iterations",
    primaryTool: {
      name: "Midjourney",
      link: "https://www.midjourney.com/",
      sourceLabel: "Midjourney",
      sourceKind: "Official site",
      description:
        "Creator-favorite image-generation environment for high-aesthetic concepting and visual exploration.",
      pricingSignal: "Paid plans",
      category: "Image generation",
    },
    alternativeTools: [
      {
        name: "Adobe Firefly",
        link: "https://www.adobe.com/products/firefly/",
        sourceLabel: "Adobe Firefly",
        sourceKind: "Official site",
        description:
          "Commercially oriented image, video, and design generation linked into Adobe’s editing workflow.",
        pricingSignal: "Free / Standard / Pro plans",
        category: "Generative design",
      },
      {
        name: "Canva AI",
        link: "https://www.canva.com/ai-image-generator/",
        sourceLabel: "Canva AI image generator",
        sourceKind: "Official site",
        description:
          "Accessible design workflow that mixes generation, editing, layout, and brand-kit production inside one visual suite.",
        pricingSignal: "Free / paid plans",
        category: "Design platform",
      },
    ],
  },
  {
    id: 4,
    title: "Data Scientists",
    icon: "📊",
    description:
      "AI platforms now need to support governed data pipelines, experimentation, model operations, and agentic workflows rather than isolated notebooks alone.",
    impactLevel: "Critical",
    aiAdoption: 94,
    workflowNow:
      "The modern data-science stack blends lakehouse or warehouse access, evaluation tooling, notebooks, model serving, and agent frameworks.",
    timeSaved: "55-80% on data prep, experimentation, and model operations",
    primaryTool: {
      name: "Databricks",
      link: "https://www.databricks.com/product/data-intelligence-platform",
      sourceLabel: "Data Intelligence Platform",
      sourceKind: "Official site",
      description:
        "Enterprise data and AI platform built around governed analytics, model development, and Mosaic AI workflows.",
      pricingSignal: "Platform pricing / enterprise",
      category: "Data platform",
    },
    alternativeTools: [
      {
        name: "H2O.ai",
        link: "https://h2o.ai/",
        sourceLabel: "H2O.ai",
        sourceKind: "Official site",
        description:
          "Private-data-focused predictive and generative AI platform with AutoML, vertical agents, and on-prem deployment strength.",
        pricingSignal: "Open source / enterprise",
        category: "ML platform",
      },
      {
        name: "DataRobot",
        link: "https://www.datarobot.com/product/agentic-ai/",
        sourceLabel: "DataRobot Agentic AI",
        sourceKind: "Official site",
        description:
          "Production-first platform for developing, deploying, observing, and governing agentic and predictive AI apps.",
        pricingSignal: "Sales-led / enterprise",
        category: "AI platform",
      },
    ],
  },
  {
    id: 5,
    title: "Marketing Professionals",
    icon: "📈",
    description:
      "The current marketing AI wave is operational, not experimental: agents, compliance, brand voice, CRM context, and content reuse at scale.",
    impactLevel: "High",
    aiAdoption: 88,
    workflowNow:
      "Teams are centralizing campaign planning, content generation, CRM intelligence, and compliance review inside fewer platforms.",
    timeSaved: "35-60% across campaign prep and content operations",
    primaryTool: {
      name: "HubSpot Breeze",
      link: "https://www.hubspot.com/products/artificial-intelligence",
      sourceLabel: "HubSpot Breeze",
      sourceKind: "Official site",
      description:
        "Integrated marketing, sales, and service AI layer with assistants, agents, and CRM-aware automation.",
      pricingSignal: "Free tools / platform plans",
      category: "Go-to-market AI",
    },
    alternativeTools: [
      {
        name: "Jasper",
        link: "https://www.jasper.ai/",
        sourceLabel: "Jasper",
        sourceKind: "Official site",
        description:
          "Purpose-built for marketing teams that need brand control, campaign assets, and repeatable content systems.",
        pricingSignal: "Paid plans / enterprise",
        category: "Marketing content",
      },
      {
        name: "Persado",
        link: "https://www.persado.com/platform/",
        sourceLabel: "Persado platform",
        sourceKind: "Official site",
        description:
          "Performance- and compliance-driven marketing language platform used heavily in regulated industries.",
        pricingSignal: "Sales-led / enterprise",
        category: "Marketing compliance AI",
      },
    ],
  },
  {
    id: 6,
    title: "Customer Service Representatives",
    icon: "🎧",
    description:
      "Support AI has moved from scripted bots to agentic systems that can reason, act, and resolve across channels.",
    impactLevel: "High",
    aiAdoption: 82,
    workflowNow:
      "Best-in-class service setups now combine AI agents for deflection, copilots for reps, and analytics for quality control.",
    timeSaved: "40-70% on response handling and ticket resolution",
    primaryTool: {
      name: "Zendesk AI Agents",
      link: "https://www.zendesk.com/service/ai/ai-agents/",
      sourceLabel: "Zendesk AI Agents",
      sourceKind: "Official site",
      description:
        "Agentic customer-service automation designed to resolve issues across channels using knowledge, policies, and workflows.",
      pricingSignal: "Zendesk plans / add-ons",
      category: "Customer support AI",
    },
    alternativeTools: [
      {
        name: "Intercom Fin",
        link: "https://www.intercom.com/fin",
        sourceLabel: "What is Fin?",
        sourceKind: "Help",
        description:
          "High-resolution support agent for chat, email, and phone with strong policy and workflow handling.",
        pricingSignal: "Platform pricing",
        category: "Support agent",
      },
      {
        name: "Sierra",
        link: "https://sierra.ai/",
        sourceLabel: "Sierra",
        sourceKind: "Official site",
        description:
          "Brand-controlled customer-experience agent platform spanning chat, voice, SMS, and email interactions.",
        pricingSignal: "Sales-led",
        category: "CX agent platform",
      },
    ],
  },
  {
    id: 7,
    title: "Translators",
    icon: "🌐",
    description:
      "Language work now blends machine translation, glossary control, live voice, and post-edit review instead of raw sentence replacement.",
    impactLevel: "Critical",
    aiAdoption: 92,
    workflowNow:
      "The practical stack combines secure document translation, terminology control, and human review for nuanced output.",
    timeSaved: "70-90% on first-pass translation and terminology lookup",
    primaryTool: {
      name: "DeepL",
      link: "https://www.deepl.com/en/translator",
      sourceLabel: "DeepL Translator",
      sourceKind: "Official site",
      description:
        "Translation platform with strong document support, business security positioning, and polished multilingual output.",
      pricingSignal: "Free / Pro",
      category: "Translation",
    },
    alternativeTools: [
      {
        name: "Google Translate",
        link: "https://translate.google.com/",
        sourceLabel: "Google Translate",
        sourceKind: "Official site",
        description:
          "Widely available translation utility with unmatched casual reach and strong coverage for quick multilingual tasks.",
        pricingSignal: "Free / Cloud API",
        category: "Translation",
      },
      {
        name: "Microsoft Translator",
        link: "https://www.microsoft.com/en-us/translator/",
        sourceLabel: "Microsoft Translator",
        sourceKind: "Official site",
        description:
          "Translation stack with conversation, speech, and Azure integration options for enterprise environments.",
        pricingSignal: "Free app / Azure pricing",
        category: "Translation",
      },
    ],
  },
  {
    id: 8,
    title: "Financial Analysts",
    icon: "💰",
    description:
      "Research-heavy finance workflows now depend on AI that can search premium data, build briefings, monitor companies, and connect qualitative and quantitative signals.",
    impactLevel: "High",
    aiAdoption: 78,
    workflowNow:
      "The high-leverage setup is source-backed intelligence: filings, transcripts, market data, and deep-research agents in one workflow.",
    timeSaved: "40-70% on market intelligence and memo prep",
    primaryTool: {
      name: "AlphaSense",
      link: "https://www.alpha-sense.com/platform/",
      sourceLabel: "AlphaSense platform",
      sourceKind: "Official site",
      description:
        "Unified market-intelligence and deep-research platform for filings, transcripts, financial data, and workflow agents.",
      pricingSignal: "Free trial / enterprise plans",
      category: "Market intelligence",
    },
    alternativeTools: [
      {
        name: "FactSet AI",
        link: "https://investor.factset.com/news-releases/news-release-details/factset-accelerates-innovation-banking-launch-new-ai-native/",
        sourceLabel: "FactSet AI for Banking",
        sourceKind: "Press",
        description:
          "AI-native workflow tooling built into institutional finance data and analysis workflows.",
        pricingSignal: "Institutional / enterprise",
        category: "Financial data AI",
      },
      {
        name: "Kensho",
        link: "https://www.kensho.com/",
        sourceLabel: "Kensho",
        sourceKind: "Official site",
        description:
          "AI-driven analytics and search tooling used for finance, enterprise knowledge, and data-heavy decision workflows.",
        pricingSignal: "Enterprise",
        category: "Financial AI",
      },
    ],
  },
  {
    id: 9,
    title: "Lawyers",
    icon: "⚖️",
    description:
      "Legal AI has become one of the clearest vertical-AI markets, centered on research, review, due diligence, and drafting in secure environments.",
    impactLevel: "High",
    aiAdoption: 71,
    workflowNow:
      "The practical pattern is domain-specific AI layered onto existing legal sources, precedents, and review workflows rather than generic chat alone.",
    timeSaved: "50-85% on review, drafting, and legal research tasks",
    primaryTool: {
      name: "Harvey",
      link: "https://www.harvey.ai/",
      sourceLabel: "Harvey",
      sourceKind: "Official site",
      description:
        "Legal AI platform designed for research, drafting, due diligence, and knowledge-heavy professional workflows.",
      pricingSignal: "Sales-led",
      category: "Legal AI",
    },
    alternativeTools: [
      {
        name: "CoCounsel Legal",
        link: "https://legal.thomsonreuters.com/en/products/cocounsel-legal",
        sourceLabel: "CoCounsel Legal",
        sourceKind: "Official site",
        description:
          "Thomson Reuters legal assistant for review, drafting, search, and workflow augmentation.",
        pricingSignal: "Platform plans / enterprise",
        category: "Legal research AI",
      },
      {
        name: "Luminance",
        link: "https://www.luminance.com/",
        sourceLabel: "Luminance",
        sourceKind: "Official site",
        description:
          "Contract and due-diligence platform focused on legal document analysis and workflow control.",
        pricingSignal: "Sales-led",
        category: "Document review",
      },
    ],
  },
  {
    id: 10,
    title: "Healthcare Professionals",
    icon: "🏥",
    description:
      "Healthcare AI is shifting from narrow demos to workflow tools that cut documentation burden, surface evidence, and coordinate care in production settings.",
    impactLevel: "Critical",
    aiAdoption: 73,
    workflowNow:
      "Winning deployments focus on notes, imaging triage, coding support, and clinical workflow integration with auditability.",
    timeSaved: "30-60% on documentation and imaging workflow tasks",
    primaryTool: {
      name: "Dragon Copilot",
      link: "https://www.microsoft.com/en-us/health-solutions/clinical-workflow/dragon-copilot",
      sourceLabel: "Microsoft Dragon Copilot",
      sourceKind: "Official site",
      description:
        "Clinical AI assistant for documentation, evidence surfacing, and workflow automation across care settings.",
      pricingSignal: "Enterprise / healthcare contracts",
      category: "Clinical workflow AI",
    },
    alternativeTools: [
      {
        name: "Aidoc",
        link: "https://www.aidoc.com/platform/aios/",
        sourceLabel: "Aidoc aiOS",
        sourceKind: "Official site",
        description:
          "Clinical AI orchestration platform for imaging, triage, governance, and systemwide deployment.",
        pricingSignal: "Sales-led",
        category: "Clinical imaging AI",
      },
      {
        name: "PathAI",
        link: "https://www.pathai.com/",
        sourceLabel: "PathAI",
        sourceKind: "Official site",
        description:
          "Pathology-focused AI platform for diagnosis support, precision medicine, and lab workflows.",
        pricingSignal: "Enterprise / lab partnerships",
        category: "Pathology AI",
      },
    ],
  },
  {
    id: 11,
    title: "Teachers & Educators",
    icon: "🎓",
    description:
      "Education AI is strongest when it saves prep time, supports differentiation, and keeps teachers in control of student-facing guidance.",
    impactLevel: "High",
    aiAdoption: 68,
    workflowNow:
      "The practical stack is lesson support plus structured classroom tools rather than unsupervised student chatbots.",
    timeSaved: "40-70% on prep, differentiation, and grading support",
    primaryTool: {
      name: "Khanmigo",
      link: "https://support.khanacademy.org/hc/en-us/articles/41626114291469-Which-languages-can-I-use-for-Khanmigo-and-free-teacher-tools",
      sourceLabel: "Khanmigo teacher tools",
      sourceKind: "Help",
      description:
        "Khan Academy’s AI tutor and teacher support layer for guided learning, assignments, and classroom monitoring.",
      pricingSignal: "District / supported plans",
      category: "Tutoring and classroom AI",
    },
    alternativeTools: [
      {
        name: "MagicSchool",
        link: "https://www.magicschool.ai/",
        sourceLabel: "MagicSchool",
        sourceKind: "Official site",
        description:
          "Teacher-first AI platform for lesson plans, assessment support, communication, and differentiation.",
        pricingSignal: "Free for teachers / district plans",
        category: "Teacher workflow AI",
      },
      {
        name: "Gradescope",
        link: "https://www.gradescope.com/",
        sourceLabel: "Gradescope",
        sourceKind: "Official site",
        description:
          "Assessment platform that accelerates grading, rubric use, and analytics across assignments and code work.",
        pricingSignal: "Instructor / institutional pricing",
        category: "Assessment workflow",
      },
    ],
  },
  {
    id: 12,
    title: "Sales Professionals",
    icon: "💼",
    description:
      "Sales AI is now much more than note-taking: it is becoming a revenue operating system for prospecting, forecasting, coaching, and account execution.",
    impactLevel: "High",
    aiAdoption: 84,
    workflowNow:
      "The strongest workflows combine conversation capture, CRM context, and agentic follow-up or pipeline actions.",
    timeSaved: "30-55% on call prep, follow-up, and pipeline hygiene",
    primaryTool: {
      name: "Gong",
      link: "https://www.gong.io/platform/",
      sourceLabel: "Gong Revenue AI OS",
      sourceKind: "Official site",
      description:
        "Revenue AI operating system spanning call intelligence, forecasting, enablement, and AI agents for GTM teams.",
      pricingSignal: "Sales-led",
      category: "Revenue AI",
    },
    alternativeTools: [
      {
        name: "Salesforce AI",
        link: "https://www.salesforce.com/einstein",
        sourceLabel: "Salesforce AI",
        sourceKind: "Official site",
        description:
          "CRM-native predictive, generative, and agentic AI across customer, service, and sales workflows.",
        pricingSignal: "Platform pricing / enterprise",
        category: "CRM AI",
      },
      {
        name: "Outreach",
        link: "https://www.outreach.io/",
        sourceLabel: "Outreach",
        sourceKind: "Official site",
        description:
          "Sales execution platform focused on prospecting, sequencing, seller workflow automation, and pipeline execution.",
        pricingSignal: "Sales-led",
        category: "Sales engagement",
      },
    ],
  },
  {
    id: 13,
    title: "HR Professionals",
    icon: "👥",
    description:
      "Recruiting AI is being rebuilt around skill validation, candidate experience, and higher-volume sourcing rather than resume keyword filtering alone.",
    impactLevel: "High",
    aiAdoption: 74,
    workflowNow:
      "Recruiting teams now mix conversational screening, AI-assisted sourcing, and skills intelligence inside existing HCM systems.",
    timeSaved: "45-75% on sourcing, screening, and coordination",
    primaryTool: {
      name: "HireVue",
      link: "https://www.hirevue.com/platform/ai-hiring-agents",
      sourceLabel: "HireVue AI Hiring Agents",
      sourceKind: "Official site",
      description:
        "Hiring workflow platform focused on validated skills, conversational screening, and AI-led recruiting steps.",
      pricingSignal: "Sales-led",
      category: "Hiring AI",
    },
    alternativeTools: [
      {
        name: "Workday Talent Acquisition",
        link: "https://www.workday.com/en-us/products/talent-management/talent-acquisition.html",
        sourceLabel: "Workday Talent Acquisition",
        sourceKind: "Official site",
        description:
          "AI-powered recruiting suite for hiring, candidate engagement, internal mobility, and workflow automation.",
        pricingSignal: "Sales-led",
        category: "HCM recruiting",
      },
      {
        name: "Eightfold",
        link: "https://eightfold.ai/talent-intelligence-platform/",
        sourceLabel: "Eightfold Talent Intelligence",
        sourceKind: "Official site",
        description:
          "Talent-intelligence platform that uses skills and real-time work signals for hiring, planning, and retention decisions.",
        pricingSignal: "Sales-led",
        category: "Talent intelligence",
      },
    ],
  },
  {
    id: 14,
    title: "Accountants",
    icon: "🧮",
    description:
      "Finance AI is getting practical where the work is repetitive, document-heavy, and audit-sensitive: close, reconciliation, expense review, and compliance.",
    impactLevel: "High",
    aiAdoption: 69,
    workflowNow:
      "The most useful tools are automating close tasks, matching, spend review, and audit trails with reviewability built in.",
    timeSaved: "45-80% on repetitive close and review workflows",
    primaryTool: {
      name: "FloQast",
      link: "https://www.floqast.com/",
      sourceLabel: "FloQast Accounting Transformation Platform",
      sourceKind: "Official site",
      description:
        "AI-powered accounting transformation platform for close management, reconciliation, and audit-ready workflows.",
      pricingSignal: "Sales-led",
      category: "Accounting operations",
    },
    alternativeTools: [
      {
        name: "AppZen",
        link: "https://www.appzen.com/expense-report-auditing",
        sourceLabel: "AppZen expense audit",
        sourceKind: "Official site",
        description:
          "Finance AI platform for expense auditing, AP automation, fraud checks, and spend controls.",
        pricingSignal: "Sales-led",
        category: "Finance AI",
      },
      {
        name: "Botkeeper",
        link: "https://www.botkeeper.com/",
        sourceLabel: "Botkeeper",
        sourceKind: "Official site",
        description:
          "Bookkeeping automation platform focused on reconciliations, reporting support, and accounting workflow efficiency.",
        pricingSignal: "Sales-led",
        category: "Bookkeeping automation",
      },
    ],
  },
  {
    id: 15,
    title: "Journalists",
    icon: "📰",
    description:
      "Journalism AI is most defensible when it accelerates sourcing, transcript review, and structured drafting without replacing verification.",
    impactLevel: "High",
    aiAdoption: 61,
    workflowNow:
      "The best newsroom workflow combines source-backed research agents, drafting help, and human verification at the final stage.",
    timeSaved: "35-60% on background research and draft assembly",
    primaryTool: {
      name: "Perplexity Deep Research",
      link: "https://docs.perplexity.ai/docs/sonar/models/sonar-deep-research",
      sourceLabel: "Sonar Deep Research",
      sourceKind: "Docs",
      description:
        "Citation-heavy research workflow for background reporting, source collection, and synthesis.",
      pricingSignal: "Free / paid plans",
      category: "Research assistant",
    },
    alternativeTools: [
      {
        name: "ChatGPT",
        link: "https://openai.com/chatgpt/overview/",
        sourceLabel: "ChatGPT overview",
        sourceKind: "Official site",
        description:
          "Useful for structuring interviews, rewriting leads, transcript summarization, and outline generation.",
        pricingSignal: "Free / Plus / Pro / business plans",
        category: "Writing assistant",
      },
      {
        name: "Grammarly",
        link: "https://www.grammarly.com/",
        sourceLabel: "Grammarly",
        sourceKind: "Official site",
        description:
          "Editing and clarity layer for refining copy, consistency, and publishing polish.",
        pricingSignal: "Free / paid plans",
        category: "Editing assistant",
      },
    ],
  },
  {
    id: 16,
    title: "Video Editors",
    icon: "🎬",
    description:
      "Video AI is strongest where it speeds ideation, rough cuts, enhancement, voice, and generative shot creation without forcing editors into a brand-new stack.",
    impactLevel: "High",
    aiAdoption: 79,
    workflowNow:
      "The strongest editor workflow is hybrid: generate or transform assets, then refine inside an editing-oriented product surface.",
    timeSaved: "40-70% on rough cuts, transforms, and content repurposing",
    primaryTool: {
      name: "Runway",
      link: "https://runwayml.com/product",
      sourceLabel: "Runway product",
      sourceKind: "Official site",
      description:
        "Creative toolkit for image and video generation, transformation, workflows, and multimodal production tasks.",
      pricingSignal: "Free / paid plans",
      category: "Video generation",
    },
    alternativeTools: [
      {
        name: "Descript",
        link: "https://www.descript.com/ai",
        sourceLabel: "Descript AI video editor",
        sourceKind: "Official site",
        description:
          "Edit video and podcasts like documents, with an AI co-editor for clips, avatars, cleanup, and repurposing.",
        pricingSignal: "Free / paid plans",
        category: "Video editing",
      },
      {
        name: "Adobe Firefly",
        link: "https://www.adobe.com/products/firefly/",
        sourceLabel: "Adobe Firefly",
        sourceKind: "Official site",
        description:
          "Useful for generative video, visual ideation, and production-ready asset creation inside Adobe workflows.",
        pricingSignal: "Free / Standard / Pro plans",
        category: "Generative video",
      },
    ],
  },
  {
    id: 17,
    title: "Photographers",
    icon: "📸",
    description:
      "Photography AI now matters most in finishing: enhancement, cleanup, expansion, restoration, and high-speed delivery workflows.",
    impactLevel: "Medium",
    aiAdoption: 74,
    workflowNow:
      "The practical stack pairs a main editor with specialized AI enhancement tools for speed and consistency.",
    timeSaved: "45-75% on post-processing and cleanup tasks",
    primaryTool: {
      name: "Photoshop with Firefly",
      link: "https://www.adobe.com/products/firefly/",
      sourceLabel: "Adobe Firefly",
      sourceKind: "Official site",
      description:
        "Generative fill, image expansion, and editing assist inside professional Adobe workflows.",
      pricingSignal: "Creative Cloud / Firefly plans",
      category: "Photo editing",
    },
    alternativeTools: [
      {
        name: "Topaz Photo",
        link: "https://www.topazlabs.com/topaz-photo-ai/",
        sourceLabel: "Topaz Photo",
        sourceKind: "Official site",
        description:
          "High-accuracy image enhancement for sharpening, denoising, unblurring, and rescue edits.",
        pricingSignal: "Starting at monthly subscription pricing",
        category: "Photo enhancement",
      },
      {
        name: "Luminar Neo",
        link: "https://skylum.com/luminar",
        sourceLabel: "Luminar Neo",
        sourceKind: "Official site",
        description:
          "AI-assisted photo editing suite for masking, enhancement, sky replacement, and fast polishing.",
        pricingSignal: "Paid plans",
        category: "Photo editing",
      },
    ],
  },
  {
    id: 18,
    title: "Musicians & Composers",
    icon: "🎵",
    description:
      "Music AI is increasingly split across song generation, composition support, and finishing tools like mastering and release preparation.",
    impactLevel: "Medium",
    aiAdoption: 52,
    workflowNow:
      "The strongest use today is ideation plus finishing: create sketches fast, then refine, arrange, or master inside a music-native workflow.",
    timeSaved: "25-55% on ideation, demo creation, and finishing tasks",
    primaryTool: {
      name: "AIVA",
      link: "https://www.aiva.ai/technology",
      sourceLabel: "AIVA technology",
      sourceKind: "Official site",
      description:
        "AI composition assistant for generating and editing songs in many styles with ownership options on paid plans.",
      pricingSignal: "Free / paid plans",
      category: "Composition AI",
    },
    alternativeTools: [
      {
        name: "Suno",
        link: "https://suno.com/about",
        sourceLabel: "Suno",
        sourceKind: "Official site",
        description:
          "Fast song-generation platform for turning prompts into full tracks and demos.",
        pricingSignal: "Free creation / paid plans",
        category: "Song generation",
      },
      {
        name: "LANDR",
        link: "https://www.landr.com/",
        sourceLabel: "LANDR",
        sourceKind: "Official site",
        description:
          "AI mastering and release workflow suite for getting tracks polished and distributed faster.",
        pricingSignal: "Subscription plans",
        category: "Mastering and release",
      },
    ],
  },
  {
    id: 19,
    title: "Researchers",
    icon: "🔬",
    description:
      "Research AI is strongest when it plans multi-step investigation, finds papers, and keeps citations visible instead of acting like an untraceable summary engine.",
    impactLevel: "Critical",
    aiAdoption: 72,
    workflowNow:
      "The modern research stack mixes deep web research, academic search, and paper synthesis with explicit source review.",
    timeSaved: "45-75% on literature review and source gathering",
    primaryTool: {
      name: "OpenAI Deep Research",
      link: "https://openai.com/index/introducing-deep-research/",
      sourceLabel: "Introducing deep research",
      sourceKind: "Press",
      description:
        "Multi-step research workflow that plans, searches, synthesizes, and returns cited reports for knowledge work.",
      pricingSignal: "Included with supported ChatGPT plans",
      category: "Research agent",
    },
    alternativeTools: [
      {
        name: "Elicit",
        link: "https://elicit.com/",
        sourceLabel: "Elicit",
        sourceKind: "Official site",
        description:
          "Research assistant focused on finding papers, extracting findings, and structuring evidence reviews.",
        pricingSignal: "Free / paid plans",
        category: "Paper review",
      },
      {
        name: "Semantic Scholar",
        link: "https://www.semanticscholar.org/",
        sourceLabel: "Semantic Scholar",
        sourceKind: "Official site",
        description:
          "Academic discovery engine for citation trails, paper relevance, and field exploration.",
        pricingSignal: "Free",
        category: "Academic search",
      },
    ],
  },
  {
    id: 20,
    title: "Project Managers",
    icon: "📋",
    description:
      "Project-management AI now works best when it can summarize status, generate plans, automate routine work, and connect to real execution systems.",
    impactLevel: "High",
    aiAdoption: 66,
    workflowNow:
      "The most effective setup puts AI directly inside the project system so updates, risks, briefs, and workflows stay grounded in actual work.",
    timeSaved: "25-45% on coordination, planning, and status synthesis",
    primaryTool: {
      name: "Asana AI",
      link: "https://asana.com/product/ai",
      sourceLabel: "Asana AI",
      sourceKind: "Official site",
      description:
        "Work-management AI for project updates, AI teammates, workflow automation, and execution support.",
      pricingSignal: "Included on paid tiers / AI Studio on supported plans",
      category: "Project management AI",
    },
    alternativeTools: [
      {
        name: "monday.com AI",
        link: "https://monday.com/p/press-release/monday-com-welcomes-ai-agents-to-its-platform-marking-a-shift-in-how-work-gets-done/",
        sourceLabel: "monday.com AI agents",
        sourceKind: "Press",
        description:
          "Work platform evolving toward agent-executable workflows, reporting, and cross-team coordination.",
        pricingSignal: "Platform pricing",
        category: "Work management",
      },
      {
        name: "Notion AI",
        link: "https://www.notion.com/ai",
        sourceLabel: "Notion AI",
        sourceKind: "Official site",
        description:
          "All-in-one workspace with AI for notes, search, meeting capture, research, and project support.",
        pricingSignal: "Included with workspace plans / AI pricing",
        category: "Workspace AI",
      },
    ],
  },
];
