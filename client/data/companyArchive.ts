export interface CompanySource {
  label: string;
  url: string;
  kind: "Official site" | "Product" | "Docs" | "Blog" | "Press";
}

export interface Company {
  id: number;
  name: string;
  founded: string;
  founders: string[];
  headquarters: string;
  scaleSignal: string;
  operatingSignal: string;
  description: string;
  journey: string;
  currentFocus: string;
  landmarkContributions: string[];
  keyProducts: string[];
  achievements: string[];
  category: string;
  logo: string;
  website: string;
  stockSymbol?: string;
  isRecentAddition?: boolean;
  sortScale: number;
  sortOperating: number;
  sortFounded: number;
  sources: CompanySource[];
}

export const companies: Company[] = [
  {
    id: 1,
    name: "OpenAI",
    founded: "2015",
    founders: ["Sam Altman", "Elon Musk", "Greg Brockman", "Ilya Sutskever"],
    headquarters: "San Francisco, CA",
    scaleSignal: "Global consumer and enterprise AI platform",
    operatingSignal: "ChatGPT, Codex, Sora, deep research, and API products",
    description:
      "OpenAI operates one of the most widely used frontier AI product stacks, spanning consumer chat, software agents, image generation, video generation, and developer APIs.",
    journey:
      "OpenAI started as a research lab, commercialized through the API and ChatGPT, and is now leaning hard into long-running agent workflows rather than single-turn chat alone.",
    currentFocus:
      "Current product emphasis centers on research agents, coding agents, multimodal generation, and authenticated workflows that can use external tools and trusted sources.",
    landmarkContributions: [
      "GPT family and mainstream large-language-model deployment",
      "ChatGPT as the consumer adoption breakout for generative AI",
      "Codex as a cloud software engineering agent",
      "Deep research workflows for multi-step web synthesis",
    ],
    keyProducts: ["ChatGPT", "Codex", "Deep Research", "Sora", "OpenAI API"],
    achievements: [
      "Turned LLMs into a mass-market product category",
      "Expanded from chat into agentic coding and research systems",
      "Helped set the pace for multimodal product launches",
    ],
    category: "AI Research",
    logo: "🤖",
    website: "https://openai.com",
    sortScale: 900,
    sortOperating: 940,
    sortFounded: 2015,
    sources: [
      {
        label: "Introducing Codex",
        url: "https://openai.com/index/introducing-codex/",
        kind: "Blog",
      },
      {
        label: "Introducing deep research",
        url: "https://openai.com/index/introducing-deep-research/",
        kind: "Blog",
      },
    ],
  },
  {
    id: 2,
    name: "Google DeepMind",
    founded: "2010 / 2023 merge",
    founders: ["Demis Hassabis", "Shane Legg", "Mustafa Suleyman"],
    headquarters: "London, UK / Mountain View, CA",
    scaleSignal: "Alphabet-scale frontier research and deployment engine",
    operatingSignal: "Gemini, AlphaFold, Veo, and robotics platforms",
    description:
      "Google DeepMind combines Google Brain and DeepMind capabilities across frontier models, scientific systems, robotics, and product infrastructure.",
    journey:
      "DeepMind began as an AGI-first research lab, delivered headline scientific breakthroughs, and now sits closer to product and platform deployment inside Alphabet.",
    currentFocus:
      "Its current frontier spans Gemini model families, science systems such as AlphaFold, world-model and video systems like Veo, and physical agents through Gemini Robotics.",
    landmarkContributions: [
      "AlphaGo and deep reinforcement learning leadership",
      "AlphaFold as a scientific AI breakthrough",
      "Transformer lineage through Google research",
      "Gemini Robotics for embodied multimodal systems",
    ],
    keyProducts: [
      "Gemini",
      "AlphaFold 3",
      "AlphaFold Server",
      "Veo",
      "Gemini Robotics 1.5",
    ],
    achievements: [
      "Made AI scientifically credible beyond consumer apps",
      "Turned AlphaFold into a globally used research utility",
      "Pushed multimodal and embodied AI deeper into production",
    ],
    category: "AI Research",
    logo: "🧠",
    website: "https://deepmind.google",
    sortScale: 950,
    sortOperating: 920,
    sortFounded: 2010,
    sources: [
      {
        label: "AlphaFold",
        url: "https://deepmind.google/science/alphafold/",
        kind: "Product",
      },
      {
        label: "Gemini Robotics",
        url: "https://deepmind.google/models/gemini-robotics/",
        kind: "Product",
      },
    ],
  },
  {
    id: 3,
    name: "Anthropic",
    founded: "2021",
    founders: ["Dario Amodei", "Daniela Amodei"],
    headquarters: "San Francisco, CA",
    scaleSignal: "Frontier model company with enterprise and safety weight",
    operatingSignal: "Claude, Claude Code, and MCP-centered workflows",
    description:
      "Anthropic builds frontier models with a safety-heavy positioning and has become a major platform player for enterprise assistants, coding, and tool-connected agents.",
    journey:
      "Founded by former OpenAI leaders, Anthropic differentiated through model behavior, constitutional training ideas, and enterprise-grade deployment rather than consumer virality alone.",
    currentFocus:
      "Its clearest current wedge is reliable work AI: Claude for reasoning and writing, Claude Code for software tasks, and MCP for connecting models to tools and data.",
    landmarkContributions: [
      "Constitutional AI as a steerability and safety framing",
      "Claude family as a strong enterprise LLM line",
      "Claude Code for agentic developer workflows",
      "Model Context Protocol ecosystem leadership",
    ],
    keyProducts: ["Claude", "Claude Code", "Anthropic API", "MCP"],
    achievements: [
      "Made safety language commercially legible to enterprises",
      "Helped normalize tool-connected coding agents",
      "Turned MCP into a cross-vendor integration standard",
    ],
    category: "AI Safety",
    logo: "🛡️",
    website: "https://anthropic.com",
    sortScale: 610,
    sortOperating: 700,
    sortFounded: 2021,
    sources: [
      {
        label: "Claude product overview",
        url: "https://www.anthropic.com/claude",
        kind: "Product",
      },
      {
        label: "Claude Code with MCP",
        url: "https://docs.anthropic.com/en/docs/claude-code/mcp",
        kind: "Docs",
      },
    ],
  },
  {
    id: 4,
    name: "Nvidia",
    founded: "1993",
    founders: ["Jensen Huang", "Chris Malachowsky", "Curtis Priem"],
    headquarters: "Santa Clara, CA",
    scaleSignal: "Foundational compute supplier for model training and inference",
    operatingSignal: "Blackwell systems, CUDA, DGX, and full-stack AI software",
    description:
      "Nvidia remains the backbone of modern AI infrastructure, pairing accelerated hardware with CUDA, networking, and deployment tooling used across most frontier-model stacks.",
    journey:
      "It moved from graphics into general-purpose accelerated computing and then became the default supplier for large-scale training and inference infrastructure.",
    currentFocus:
      "The company is now selling full AI factories, not just chips: accelerated servers, networking, inference software, and optimized deployment paths for agentic and multimodal systems.",
    landmarkContributions: [
      "CUDA as the dominant GPU software layer",
      "DGX systems as turnkey AI infrastructure",
      "Grace Blackwell systems for rack-scale AI",
      "NIM and inference software for enterprise deployment",
    ],
    keyProducts: ["Blackwell", "DGX", "CUDA", "NVIDIA NIM", "Omniverse"],
    achievements: [
      "Turned GPUs into the default substrate for deep learning",
      "Captured the center of training and inference demand",
      "Expanded into software and systems-level AI deployment",
    ],
    category: "AI Infrastructure",
    logo: "🎮",
    website: "https://www.nvidia.com",
    stockSymbol: "NVDA",
    sortScale: 980,
    sortOperating: 980,
    sortFounded: 1993,
    sources: [
      {
        label: "Blackwell-powered DGX SuperPOD",
        url: "https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Launches-Blackwell-Powered-DGX-SuperPOD-for-Generative-AI-Supercomputing-at-Trillion-Parameter-Scale/default.aspx",
        kind: "Press",
      },
      {
        label: "CUDA platform",
        url: "https://developer.nvidia.com/cuda-zone",
        kind: "Product",
      },
    ],
  },
  {
    id: 5,
    name: "Meta AI",
    founded: "2013 (FAIR) / 2004 (Meta)",
    founders: ["Mark Zuckerberg", "Yann LeCun (FAIR)"],
    headquarters: "Menlo Park, CA",
    scaleSignal: "Big-tech AI organization with open-model and product reach",
    operatingSignal: "Llama, Meta AI, PyTorch, and Ray-Ban Meta experiences",
    description:
      "Meta AI mixes open-model releases, large-scale recommendation and ranking systems, wearable AI products, and long-running research programs across vision and language.",
    journey:
      "FAIR established Meta as a major research lab early, and the company later pushed consumer AI through assistants, creator tooling, and smart glasses while open-sourcing Llama.",
    currentFocus:
      "Meta is currently using open models, recommendation systems, and hardware surfaces such as smart glasses to make AI a default part of social and ambient computing.",
    landmarkContributions: [
      "PyTorch as a major deep-learning framework",
      "Llama as a defining open-model line",
      "Segment Anything and computer vision tooling",
      "Smart-glasses distribution for consumer AI",
    ],
    keyProducts: [
      "Llama",
      "Meta AI assistant",
      "PyTorch",
      "Segment Anything",
      "Ray-Ban Meta glasses",
    ],
    achievements: [
      "Helped normalize open-weight frontier model releases",
      "Maintained major research credibility while shipping consumer AI",
      "Turned wearable AI into a serious product category",
    ],
    category: "Big Tech AI",
    logo: "📘",
    website: "https://ai.meta.com",
    stockSymbol: "META",
    sortScale: 960,
    sortOperating: 930,
    sortFounded: 2013,
    sources: [
      {
        label: "Open Source AI at Meta",
        url: "https://ai.meta.com/opensourceAI/",
        kind: "Product",
      },
      {
        label: "Meta AI research",
        url: "https://ai.meta.com/research/",
        kind: "Official site",
      },
    ],
  },
  {
    id: 6,
    name: "Microsoft",
    founded: "1975",
    founders: ["Bill Gates", "Paul Allen"],
    headquarters: "Redmond, WA",
    scaleSignal: "Global enterprise AI distribution across cloud and productivity",
    operatingSignal: "Microsoft 365 Copilot, GitHub Copilot, Azure Copilot, and Copilot Studio",
    description:
      "Microsoft’s AI strategy is defined by distribution: work applications, developer tools, Azure infrastructure, and agent-building surfaces all tied into enterprise software.",
    journey:
      "Its OpenAI partnership accelerated the early generative AI wave, but Microsoft has since broadened into its own copilot and agent stack across work, cloud, and development.",
    currentFocus:
      "The company is now packaging copilots as an operating layer for work: search, chat, agents, app connections, and deployment tooling across Microsoft 365 and Azure.",
    landmarkContributions: [
      "GitHub Copilot commercialization of AI coding",
      "Copilot distribution inside knowledge work apps",
      "Azure AI as enterprise model infrastructure",
      "Copilot Studio for custom agents and flows",
    ],
    keyProducts: [
      "Microsoft 365 Copilot",
      "GitHub Copilot",
      "Azure Copilot",
      "Azure OpenAI Service",
      "Copilot Studio",
    ],
    achievements: [
      "Put generative AI inside mainstream enterprise workflows",
      "Scaled AI coding to very large engineering organizations",
      "Built a strong cloud and compliance story for enterprise AI",
    ],
    category: "Big Tech AI",
    logo: "🪟",
    website: "https://www.microsoft.com/ai",
    stockSymbol: "MSFT",
    sortScale: 1000,
    sortOperating: 1000,
    sortFounded: 1975,
    sources: [
      {
        label: "Microsoft 365 Copilot",
        url: "https://www.microsoft.com/en-us/microsoft-365-copilot/enterprise",
        kind: "Product",
      },
      {
        label: "Azure Copilot",
        url: "https://azure.microsoft.com/en-us/products/copilot/",
        kind: "Product",
      },
    ],
  },
  {
    id: 7,
    name: "Tesla",
    founded: "2003",
    founders: ["Martin Eberhard", "Marc Tarpenning", "Elon Musk"],
    headquarters: "Austin, TX",
    scaleSignal: "Autonomy and robotics program attached to a massive vehicle fleet",
    operatingSignal: "FSD (Supervised), Optimus, in-house AI chips, and fleet training loops",
    description:
      "Tesla’s AI work is organized around real-world autonomy: vehicles, robotics, simulation, custom hardware, and large-scale data collection from fleet operations.",
    journey:
      "What started as advanced driver assistance became a full AI and robotics effort spanning perception, planning, custom silicon, humanoids, and robotaxi ambitions.",
    currentFocus:
      "Tesla is currently pushing vision-first driving systems, robotaxi operations, shared autonomy tooling, and Optimus as a second large product bet beyond vehicles.",
    landmarkContributions: [
      "Vision-first large-scale autonomy stack",
      "Custom AI inference hardware in production vehicles",
      "Optimus as a humanoid robotics program",
      "Large fleet-data learning loops for real-world driving",
    ],
    keyProducts: ["FSD (Supervised)", "Optimus", "Tesla AI Computer", "Dojo"],
    achievements: [
      "Made autonomy a core consumer-vehicle software story",
      "Built one of the largest real-world driving datasets",
      "Extended automotive AI work into general robotics",
    ],
    category: "Autonomous Systems",
    logo: "🚗",
    website: "https://www.tesla.com/AI",
    stockSymbol: "TSLA",
    sortScale: 870,
    sortOperating: 900,
    sortFounded: 2003,
    sources: [
      {
        label: "Tesla AI & Robotics",
        url: "https://www.tesla.com/AI",
        kind: "Product",
      },
      {
        label: "Full Self-Driving (Supervised)",
        url: "https://www.tesla.com/fsd",
        kind: "Product",
      },
    ],
  },
  {
    id: 8,
    name: "Stability AI",
    founded: "2019",
    founders: ["Emad Mostaque", "Cyrus Hodes", "Shan Shan Fu"],
    headquarters: "London, UK",
    scaleSignal: "Open generative model company with licensing and deployment focus",
    operatingSignal: "Stable Diffusion, Stable Video, and self-hosted commercial licensing",
    description:
      "Stability AI remains identified with open generative media models and commercial licensing for teams that want more control than closed creative platforms provide.",
    journey:
      "It broke into the mainstream through open image generation and then expanded toward audio, video, and enterprise-friendly deployment options.",
    currentFocus:
      "Its clearest value today is adaptable generative media infrastructure: open or licensable models that teams can run within their own products and environments.",
    landmarkContributions: [
      "Stable Diffusion as a defining open text-to-image model",
      "Commercial self-hosting for generative media models",
      "Stable Video for deployable text-to-video workflows",
    ],
    keyProducts: ["Stable Diffusion", "Stable Video Diffusion", "Stable Audio"],
    achievements: [
      "Made open image generation mainstream",
      "Gave developers and enterprises more deployable media models",
      "Kept open-source creative AI in the commercial conversation",
    ],
    category: "Generative AI",
    logo: "🎨",
    website: "https://stability.ai",
    sortScale: 400,
    sortOperating: 420,
    sortFounded: 2019,
    sources: [
      {
        label: "Stable Video",
        url: "https://stability.ai/stable-video",
        kind: "Product",
      },
      {
        label: "Stability AI homepage",
        url: "https://stability.ai",
        kind: "Official site",
      },
    ],
  },
  {
    id: 9,
    name: "Midjourney",
    founded: "2021",
    founders: ["David Holz"],
    headquarters: "San Francisco, CA",
    scaleSignal: "Creator-first image and video generation platform",
    operatingSignal: "Create, Explore, Organize, Edit, and Stealth mode workflows on the web",
    description:
      "Midjourney remains one of the strongest consumer creative brands in AI, centered on high-aesthetic image generation and increasingly web-native creation workflows.",
    journey:
      "The company moved from Discord-native generation to a broader browser workflow with organization, editing, video, and privacy controls for creators and teams.",
    currentFocus:
      "The product direction is less about publishing technical benchmarks and more about creative control surfaces, community inspiration, and fast end-to-end generation on the web.",
    landmarkContributions: [
      "Made image generation culturally mainstream for designers and creators",
      "Built a recognizable aesthetic brand in consumer AI",
      "Extended image workflows into editing, organization, and video",
    ],
    keyProducts: ["Create", "Explore", "Editor", "Organize", "Stealth Mode"],
    achievements: [
      "Became a default reference point for AI-native image creation",
      "Maintained strong product identity without an API-first strategy",
      "Expanded beyond prompt-only creation into workflow tooling",
    ],
    category: "Generative AI",
    logo: "🌌",
    website: "https://www.midjourney.com",
    sortScale: 390,
    sortOperating: 410,
    sortFounded: 2021,
    sources: [
      {
        label: "Website overview",
        url: "https://docs.midjourney.com/hc/en-us/articles/33329460426765-Website-Overview",
        kind: "Docs",
      },
      {
        label: "Creating on Web",
        url: "https://docs.midjourney.com/hc/en-us/articles/33390732264589-Creating-on-Web",
        kind: "Docs",
      },
    ],
  },
  {
    id: 10,
    name: "Databricks",
    founded: "2013",
    founders: [
      "Ali Ghodsi",
      "Ion Stoica",
      "Matei Zaharia",
      "Reynold Xin",
      "Patrick Wendell",
      "Andy Konwinski",
    ],
    headquarters: "San Francisco, CA",
    scaleSignal: "Data platform with AI lifecycle and agent tooling built in",
    operatingSignal: "Data Intelligence Platform, Mosaic AI, Agent Bricks, MLflow, and governance",
    description:
      "Databricks now frames AI as a data platform problem, combining lakehouse infrastructure, governance, evaluation, serving, and agent-building into one operating stack.",
    journey:
      "It grew from Spark and lakehouse infrastructure into a broader AI platform vendor that now competes for agent, retrieval, and enterprise model deployment workloads.",
    currentFocus:
      "Its current positioning is strongly enterprise and operations-heavy: use your own data, govern the full lifecycle, and build agents without leaving the data platform.",
    landmarkContributions: [
      "Lakehouse architecture for data and ML",
      "Mosaic AI for end-to-end GenAI development",
      "Agent Bricks and evaluation tooling for production AI agents",
      "MLflow as a major MLOps and evaluation standard",
    ],
    keyProducts: [
      "Data Intelligence Platform",
      "Mosaic AI",
      "Agent Bricks",
      "MLflow",
      "Unity Catalog",
    ],
    achievements: [
      "Brought data governance and GenAI closer together",
      "Made enterprise AI deployment more ops-friendly",
      "Stayed relevant as teams moved from dashboards to agents",
    ],
    category: "Enterprise AI",
    logo: "🧱",
    website: "https://www.databricks.com",
    sortScale: 700,
    sortOperating: 760,
    sortFounded: 2013,
    sources: [
      {
        label: "What is Databricks?",
        url: "https://docs.databricks.com/gcp/en/introduction/",
        kind: "Docs",
      },
      {
        label: "AI and ML on Databricks",
        url: "https://docs.databricks.com/aws/en/machine-learning",
        kind: "Docs",
      },
    ],
  },
  {
    id: 11,
    name: "Scale AI",
    founded: "2016",
    founders: ["Alexandr Wang", "Lucy Guo"],
    headquarters: "San Francisco, CA",
    scaleSignal: "Enterprise and public-sector AI data and deployment platform",
    operatingSignal: "Data Engine, GenAI Platform, evaluations, and model adaptation services",
    description:
      "Scale AI sits at the infrastructure layer between frontier labs and large enterprises, focusing on data quality, evaluation, alignment, and deployment readiness.",
    journey:
      "It began with data labeling and moved steadily up-stack into RLHF, evaluations, public-sector programs, and enterprise GenAI platform offerings.",
    currentFocus:
      "Scale’s strongest current story is operational AI delivery: custom data pipelines, evals, enterprise apps, and deployment systems for teams moving from experiments to production.",
    landmarkContributions: [
      "Data Engine services for high-quality model training",
      "Enterprise GenAI platform and evaluation infrastructure",
      "Strong positioning in government and regulated deployments",
    ],
    keyProducts: ["Scale Data Engine", "Scale GenAI Platform", "Evals", "RLHF services"],
    achievements: [
      "Moved data quality and evals into the center of enterprise AI buying",
      "Bridged model labs, government, and enterprise deployment programs",
      "Expanded beyond annotation into full-stack delivery",
    ],
    category: "Enterprise AI",
    logo: "📐",
    website: "https://scale.com",
    sortScale: 620,
    sortOperating: 700,
    sortFounded: 2016,
    sources: [
      {
        label: "Scale resources",
        url: "https://scale.com/resources",
        kind: "Official site",
      },
      {
        label: "Scale GenAI Platform docs",
        url: "https://scale.com/docs/genai-platform",
        kind: "Docs",
      },
    ],
  },
  {
    id: 12,
    name: "Cohere",
    founded: "2019",
    founders: ["Aidan Gomez", "Nick Frosst", "Ivan Zhang"],
    headquarters: "Toronto, Canada",
    scaleSignal: "Enterprise-first model and workflow platform",
    operatingSignal: "North, private deployment, and business-focused models",
    description:
      "Cohere focuses on enterprise AI where privacy, deployment control, and measurable business workflows matter more than general consumer chat adoption.",
    journey:
      "It started as a foundation-model company and has increasingly packaged those capabilities into business products and deployable work systems.",
    currentFocus:
      "North is now the clearest product expression: context-grounded search, creation, and workflow automation for teams that want AI inside governed enterprise environments.",
    landmarkContributions: [
      "Enterprise-oriented model deployment",
      "Retrieval and reranking leadership in practical enterprise use cases",
      "North as a workflow-centric AI workspace",
      "Flexible VPC, on-prem, and model-vault deployment options",
    ],
    keyProducts: ["North", "Command", "Rerank", "Embed", "Cohere Transcribe"],
    achievements: [
      "Stayed focused on enterprise security and deployment realism",
      "Turned retrieval quality into a practical product differentiator",
      "Moved from model vendor toward workflow platform",
    ],
    category: "Enterprise AI",
    logo: "🧭",
    website: "https://cohere.com",
    sortScale: 570,
    sortOperating: 650,
    sortFounded: 2019,
    sources: [
      {
        label: "Cohere homepage",
        url: "https://cohere.com/",
        kind: "Official site",
      },
      {
        label: "North",
        url: "https://cohere.com/north",
        kind: "Product",
      },
    ],
  },
  {
    id: 13,
    name: "Hugging Face",
    founded: "2016",
    founders: ["Clément Delangue", "Julien Chaumond", "Thomas Wolf"],
    headquarters: "New York, NY / Paris, France",
    scaleSignal: "Default open-source distribution hub for models, datasets, and demos",
    operatingSignal: "Hub, Spaces, datasets, transformers ecosystem, and inference providers",
    description:
      "Hugging Face remains the central open-source AI distribution layer, where models, datasets, evaluation assets, and demo apps are discovered, shared, and deployed.",
    journey:
      "It evolved from an NLP startup into the most important community platform in open-source AI, linking research, builders, and commercial infrastructure.",
    currentFocus:
      "Today the company is combining openness with deployment convenience: model hosting, serverless inference, enterprise features, and ecosystem distribution in one place.",
    landmarkContributions: [
      "Hub as the default index for open models and datasets",
      "Transformers and diffusion tooling for developers",
      "Spaces for rapid demos and product experiments",
      "Inference Providers for unified hosted inference access",
    ],
    keyProducts: ["Hub", "Transformers", "Datasets", "Spaces", "Inference Providers"],
    achievements: [
      "Made open AI assets easy to discover and reuse",
      "Reduced friction between research release and production trial",
      "Built the strongest distribution network in open-source AI",
    ],
    category: "AI Platform",
    logo: "🤗",
    website: "https://huggingface.co",
    sortScale: 590,
    sortOperating: 840,
    sortFounded: 2016,
    sources: [
      {
        label: "Hub documentation",
        url: "https://huggingface.co/docs/hub/main/index",
        kind: "Docs",
      },
      {
        label: "Inference Providers",
        url: "https://huggingface.co/docs/hub/en/models-inference",
        kind: "Docs",
      },
    ],
  },
  {
    id: 14,
    name: "Palantir",
    founded: "2003",
    founders: ["Peter Thiel", "Alex Karp", "Joe Lonsdale", "Stephen Cohen", "Nathan Gettings"],
    headquarters: "Denver, CO",
    scaleSignal: "Operational AI platform vendor for regulated and mission-critical environments",
    operatingSignal: "AIP, Foundry, Ontology, and agent deployment in real workflows",
    description:
      "Palantir’s AI story is about operational context: connecting models to governed enterprise objects, decisions, and execution environments rather than standalone chat experiences.",
    journey:
      "After years in data integration and defense-heavy analytics, Palantir repositioned around AIP to connect LLMs and agents to real business and mission operations.",
    currentFocus:
      "Its current edge is secure operationalization: models connected to the ontology, real systems of record, agent monitoring, and deployment inside regulated organizations.",
    landmarkContributions: [
      "Operational data modeling through the Ontology",
      "AIP for attaching generative AI to enterprise workflows",
      "Strong regulated-industry and government deployment story",
    ],
    keyProducts: ["AIP", "Foundry", "Apollo", "Ontology"],
    achievements: [
      "Made agent deployment legible to operations-heavy enterprises",
      "Connected LLM systems to governed enterprise objects and actions",
      "Turned AI from prototype chat into workflow execution",
    ],
    category: "Enterprise AI",
    logo: "🛰️",
    website: "https://palantir.com",
    stockSymbol: "PLTR",
    sortScale: 760,
    sortOperating: 760,
    sortFounded: 2003,
    sources: [
      {
        label: "AIP architecture",
        url: "https://www.palantir.com/docs/foundry/architecture-center/aip-architecture",
        kind: "Docs",
      },
      {
        label: "Ontology system",
        url: "https://www.palantir.com/docs/foundry/architecture-center/ontology-system",
        kind: "Docs",
      },
    ],
  },
  {
    id: 15,
    name: "UiPath",
    founded: "2005",
    founders: ["Daniel Dines", "Marius Tirca"],
    headquarters: "New York, NY",
    scaleSignal: "Large enterprise automation base moving into agentic orchestration",
    operatingSignal: "Maestro, agentic automation, RPA, and multi-system orchestration",
    description:
      "UiPath is translating its automation base into the agentic era by combining AI agents, robots, process intelligence, and human review in governed workflows.",
    journey:
      "What started as RPA has been repositioned as agentic automation, with orchestration now framed as the control layer for agents, robots, systems, and people together.",
    currentFocus:
      "The current product push is Maestro and agentic orchestration: getting enterprises beyond isolated copilots into long-running business processes with oversight.",
    landmarkContributions: [
      "RPA adoption across large enterprises",
      "Agentic orchestration through UiPath Maestro",
      "Blending agents, robots, and people inside one control plane",
    ],
    keyProducts: ["UiPath Maestro", "Agent Builder", "Automation Cloud", "Studio"],
    achievements: [
      "Used existing automation footprints to enter the agent market fast",
      "Framed orchestration as the missing layer for enterprise AI execution",
      "Extended AI into process-heavy operations rather than chat-only use cases",
    ],
    category: "Process Automation",
    logo: "⚙️",
    website: "https://www.uipath.com",
    stockSymbol: "PATH",
    sortScale: 680,
    sortOperating: 780,
    sortFounded: 2005,
    sources: [
      {
        label: "UiPath Maestro",
        url: "https://www.uipath.com/platform/agentic-automation/agentic-orchestration",
        kind: "Product",
      },
      {
        label: "Agentic automation platform launch",
        url: "https://www.uipath.com/newsroom/uipath-launches-first-enterprise-grade-platform-for-agentic-automation",
        kind: "Press",
      },
    ],
  },
  {
    id: 16,
    name: "SenseTime",
    founded: "2014",
    founders: ["Xu Li", "Tang Xiao'ou", "Lin Dahua", "Liu Yongsheng", "Wang Xiaogang"],
    headquarters: "Hong Kong / Shanghai",
    scaleSignal: "Large multimodal AI company with model and deployment breadth in Asia",
    operatingSignal: "SenseNova V6.5, SenseNova-MARS, and agentic multimodal reasoning",
    description:
      "SenseTime remains a major AI company in Asia, spanning multimodal foundation models, industry deployments, and open-source releases tied to search and reasoning.",
    journey:
      "Known first for computer vision, it has shifted toward full multimodal foundation-model systems and agentic reasoning products under the SenseNova brand.",
    currentFocus:
      "Recent work is focused on agentic multimodal reasoning and search, moving beyond perception into models that can plan, invoke tools, and act over mixed media.",
    landmarkContributions: [
      "Computer vision at large commercial scale",
      "SenseNova as a multimodal model platform",
      "SenseNova-MARS for search and reasoning",
    ],
    keyProducts: ["SenseNova V6.5", "SenseNova-MARS", "Raccoon"],
    achievements: [
      "Transitioned from pure vision leadership into multimodal model systems",
      "Open-sourced a competitive reasoning-oriented visual model",
      "Maintained relevance in an increasingly agent-centered market",
    ],
    category: "Computer Vision",
    logo: "👁️",
    website: "https://www.sensetime.com",
    sortScale: 510,
    sortOperating: 560,
    sortFounded: 2014,
    sources: [
      {
        label: "SenseNova V6.5 launch",
        url: "https://www.sensetime.com/en/news-detail/51169861",
        kind: "Blog",
      },
      {
        label: "SenseNova-MARS open source release",
        url: "https://www.sensetime.com/en/news-detail/51170506%3FcategoryId%3D1072",
        kind: "Blog",
      },
    ],
  },
  {
    id: 17,
    name: "DataRobot",
    founded: "2012",
    founders: ["Jeremy Achin", "Tom de Godoy"],
    headquarters: "Boston, MA",
    scaleSignal: "Enterprise AI platform pivoting from AutoML into governed agent operations",
    operatingSignal: "Agent Workforce Platform, observability, and cross-cloud deployment",
    description:
      "DataRobot now emphasizes business outcomes and governed production systems, not just model building, with a strong push into agent development and operations.",
    journey:
      "It rose through AutoML, added observability and GenAI tooling, and has now reframed itself around deploying and governing AI agents at enterprise scale.",
    currentFocus:
      "The company is pushing an agent workforce thesis: build, deploy, monitor, and govern agentic applications across cloud, hybrid, and on-prem settings.",
    landmarkContributions: [
      "AutoML adoption in enterprises",
      "Enterprise AI observability",
      "Agent Workforce Platform for productionized agents",
    ],
    keyProducts: ["Agent Workforce Platform", "Agentic AI", "AI Observability", "Enterprise AI Suite"],
    achievements: [
      "Stayed relevant by moving from model automation to AI operations",
      "Focused on governance and deployment rather than model hype alone",
      "Used NVIDIA partnerships to sharpen its infrastructure story",
    ],
    category: "Enterprise AI",
    logo: "📊",
    website: "https://www.datarobot.com",
    sortScale: 520,
    sortOperating: 620,
    sortFounded: 2012,
    sources: [
      {
        label: "Agentic AI product",
        url: "https://www.datarobot.com/product/agentic-ai/",
        kind: "Product",
      },
      {
        label: "Agent Workforce Platform",
        url: "https://www.datarobot.com/newsroom/press/datarobot-announces-agent-workforce-platform-built-with-nvidia/",
        kind: "Press",
      },
    ],
  },
  {
    id: 18,
    name: "C3.ai",
    founded: "2009",
    founders: ["Thomas Siebel"],
    headquarters: "Redwood City, CA",
    scaleSignal: "Long-cycle enterprise AI vendor expanding into agentic app generation",
    operatingSignal: "Turnkey enterprise applications, C3 AI Platform, and C3 Code",
    description:
      "C3.ai sells enterprise AI as an application and platform layer for industrial, public-sector, and regulated workflows where governance and deployment matter.",
    journey:
      "The company built around enterprise AI well before the generative AI boom and is now repositioning that foundation around agentic and code-generation workflows.",
    currentFocus:
      "Its current direction is more productized: turnkey apps, the agentic platform, and now C3 Code for turning natural-language requirements into deployed enterprise applications.",
    landmarkContributions: [
      "Early enterprise AI application suites",
      "Vertical AI deployments in industrial and government settings",
      "C3 Code for agentic app development",
    ],
    keyProducts: ["C3 AI Platform", "C3 AI Applications", "C3 AI Studio", "C3 Code"],
    achievements: [
      "Stayed positioned as a pure enterprise AI software company",
      "Maintained deep traction in industrial and government use cases",
      "Adapted the platform narrative toward agentic development",
    ],
    category: "Enterprise AI",
    logo: "🏢",
    website: "https://c3.ai",
    stockSymbol: "AI",
    sortScale: 640,
    sortOperating: 600,
    sortFounded: 2009,
    sources: [
      {
        label: "C3 AI homepage",
        url: "https://c3.ai/",
        kind: "Official site",
      },
      {
        label: "C3 Code launch",
        url: "https://c3.ai/c3-ai-announces-c3-code/",
        kind: "Press",
      },
    ],
  },
  {
    id: 19,
    name: "Cerebras",
    founded: "2016",
    founders: ["Andrew Feldman", "Gary Lauterbach", "Michael James", "Sean Lie"],
    headquarters: "Sunnyvale, CA",
    scaleSignal: "Specialized AI hardware and cloud platform for extreme throughput",
    operatingSignal: "Training Cloud, inference products, and wafer-scale compute",
    description:
      "Cerebras differentiates on speed and architectural boldness, using wafer-scale hardware and cloud services aimed at large training and ultra-fast inference workloads.",
    journey:
      "The company started as a hardware moonshot and has increasingly packaged that hardware through cloud training and inference services that are easier to consume.",
    currentFocus:
      "Its clearest commercial message is speed: large models trained or served faster with less systems complexity than traditional distributed GPU stacks.",
    landmarkContributions: [
      "Wafer-scale AI compute architecture",
      "Cerebras Training Cloud",
      "High-throughput inference positioning for agent workloads",
    ],
    keyProducts: ["Training Cloud", "Inference", "Wafer-Scale Engine"],
    achievements: [
      "Stayed differentiated in a GPU-dominated market",
      "Turned custom silicon into a cloud-delivered service story",
      "Built a strong speed brand around AI inference",
    ],
    category: "AI Hardware",
    logo: "🧪",
    website: "https://www.cerebras.ai",
    sortScale: 500,
    sortOperating: 520,
    sortFounded: 2016,
    sources: [
      {
        label: "Cerebras Cloud",
        url: "https://www.cerebras.ai/cloud/",
        kind: "Product",
      },
      {
        label: "Cerebras Inference",
        url: "https://www.cerebras.ai/inference",
        kind: "Product",
      },
    ],
  },
  {
    id: 20,
    name: "Graphcore",
    founded: "2016",
    founders: ["Nigel Toon", "Simon Knowles", "Roger James"],
    headquarters: "Bristol, UK",
    scaleSignal: "Alternative AI compute platform built around the IPU architecture",
    operatingSignal: "IPUs, Bow Pods, and Graphcloud access to machine-intelligence compute",
    description:
      "Graphcore continues to argue for a non-GPU approach to AI acceleration, centered on its Intelligence Processing Unit and pod-based deployment systems.",
    journey:
      "It emerged as one of the highest-profile alternative AI chip startups and now emphasizes deployment simplicity and cloud access alongside the IPU architecture.",
    currentFocus:
      "Its strongest message is still architectural differentiation: fine-grained parallelism, pod systems, and easier access to IPU hardware for research and production workloads.",
    landmarkContributions: [
      "IPU architecture for machine intelligence workloads",
      "Bow Pod systems for scaled deployment",
      "Graphcloud access to IPU-based compute",
    ],
    keyProducts: ["IPU", "Bow Pod", "Graphcloud", "Poplar SDK"],
    achievements: [
      "Stayed visible as a serious non-GPU hardware alternative",
      "Built a full systems and software story around the IPU",
      "Kept focus on novel model classes and scale-out training",
    ],
    category: "AI Hardware",
    logo: "🔷",
    website: "https://graphcore.ai",
    sortScale: 420,
    sortOperating: 410,
    sortFounded: 2016,
    sources: [
      {
        label: "IPU processors",
        url: "https://www.graphcore.ai/products/ipu",
        kind: "Product",
      },
      {
        label: "Graphcloud",
        url: "https://www.graphcore.ai/graphcloud",
        kind: "Product",
      },
    ],
  },
  {
    id: 21,
    name: "Mistral AI",
    founded: "2023",
    founders: ["Arthur Mensch", "Timothée Lacroix", "Guillaume Lample"],
    headquarters: "Paris, France",
    scaleSignal: "Fast-rising European model and enterprise AI company",
    operatingSignal: "Le Chat, Document AI, OCR, and open-weight frontier models",
    description:
      "Mistral blends open-weight model releases with a growing enterprise product stack, making it one of the most important independent AI companies in Europe.",
    journey:
      "It launched with open models and quickly expanded into assistants, document AI, OCR, and enterprise deployment options while keeping a strong research identity.",
    currentFocus:
      "Recent momentum comes from enterprise assistants, document intelligence, and flexible deployment models rather than open releases alone.",
    landmarkContributions: [
      "Open-weight frontier model releases from Europe",
      "Le Chat as a full enterprise assistant surface",
      "Strong OCR and document AI productization",
    ],
    keyProducts: ["Le Chat", "Document AI", "OCR Processor", "Mistral API"],
    achievements: [
      "Established a credible European counterweight in frontier AI",
      "Moved quickly from model launches into enterprise products",
      "Built strong document-intelligence differentiation",
    ],
    category: "AI Research",
    logo: "🌬️",
    website: "https://mistral.ai",
    isRecentAddition: true,
    sortScale: 430,
    sortOperating: 520,
    sortFounded: 2023,
    sources: [
      {
        label: "Le Chat",
        url: "https://mistral.ai/products/le-chat",
        kind: "Product",
      },
      {
        label: "Document AI overview",
        url: "https://docs.mistral.ai/capabilities/OCR/document_ai_overview/",
        kind: "Docs",
      },
    ],
  },
  {
    id: 22,
    name: "Cursor",
    founded: "2022",
    founders: ["Michael Truell", "Sualeh Asif", "Arvid Lunnemark", "Aman Sanger"],
    headquarters: "San Francisco, CA",
    scaleSignal: "Agent-first developer workspace with rapid enterprise adoption",
    operatingSignal: "IDE agent, cloud agents, review tooling, and cross-surface task delegation",
    description:
      "Cursor is one of the clearest examples of the coding-agent transition: a development environment where autocomplete, chat, planning, review, and multi-step agents live together.",
    journey:
      "It started as an AI-native editor and has expanded into cloud agents, review automation, team workflows, and broader software-delivery orchestration.",
    currentFocus:
      "The product direction now centers on agentic development: delegate work, review outcomes, and move tasks across IDE, CLI, web, and collaboration surfaces.",
    landmarkContributions: [
      "Agentic coding in a familiar IDE shell",
      "Codebase understanding and indexing as a first-class feature",
      "Cross-surface agent workflows and Bugbot review",
    ],
    keyProducts: ["Cursor Editor", "Agent", "Cloud Agents", "Bugbot", "CLI"],
    achievements: [
      "Helped define the modern AI code-editor category",
      "Pushed coding UX beyond chat into delegated execution",
      "Won rapid adoption inside large engineering organizations",
    ],
    category: "AI Platform",
    logo: "⌨️",
    website: "https://cursor.com",
    isRecentAddition: true,
    sortScale: 440,
    sortOperating: 660,
    sortFounded: 2022,
    sources: [
      {
        label: "Cursor features",
        url: "https://cursor.com/features",
        kind: "Product",
      },
      {
        label: "Cursor agent product",
        url: "https://cursor.com/product/",
        kind: "Product",
      },
    ],
  },
  {
    id: 23,
    name: "Perplexity",
    founded: "2022",
    founders: ["Aravind Srinivas", "Denis Yarats", "Johnny Ho", "Andy Konwinski"],
    headquarters: "San Francisco, CA",
    scaleSignal: "Research-first answer engine and enterprise knowledge platform",
    operatingSignal: "Enterprise search, deep research, and web-grounded product APIs",
    description:
      "Perplexity combines search, citation-grounded answers, deep research workflows, and enterprise knowledge integration into a research-oriented AI platform.",
    journey:
      "It moved from consumer answer engine to a broader enterprise offering that mixes files, internal tools, web research, and API access to search-grounded reasoning models.",
    currentFocus:
      "The strongest current positioning is source-backed research at speed: deep research, enterprise search, trusted citations, and grounded AI over internal and web data.",
    landmarkContributions: [
      "Popularized citation-heavy answer engines",
      "Made deep research a product category",
      "Brought grounded search behavior into enterprise AI",
    ],
    keyProducts: ["Enterprise Pro", "Sonar Deep Research", "Perplexity API", "Research"],
    achievements: [
      "Kept search and grounding central in an agent-heavy market",
      "Extended consumer research behavior into enterprise workflows",
      "Built a strong identity around citations and verification",
    ],
    category: "AI Platform",
    logo: "🔎",
    website: "https://www.perplexity.ai",
    isRecentAddition: true,
    sortScale: 450,
    sortOperating: 680,
    sortFounded: 2022,
    sources: [
      {
        label: "Perplexity Enterprise",
        url: "https://enterprise-prod.perplexity.ai/",
        kind: "Product",
      },
      {
        label: "Sonar Deep Research",
        url: "https://docs.perplexity.ai/docs/sonar/models/sonar-deep-research",
        kind: "Docs",
      },
    ],
  },
  {
    id: 24,
    name: "Glean",
    founded: "2019",
    founders: ["Arvind Jain", "Tony Gentilcore", "Piyush Prahladka", "T.R. Vishwanath"],
    headquarters: "Palo Alto, CA",
    scaleSignal: "Enterprise work AI platform grounded in permissions-aware knowledge",
    operatingSignal: "Search, Assistant, Agents, and APIs over company context",
    description:
      "Glean is turning enterprise search into a broader work AI platform, connecting assistants and agents to company knowledge, permissions, and task execution.",
    journey:
      "It began with workplace search and has expanded into assistant, agent, orchestration, and API surfaces that treat company context as the core product asset.",
    currentFocus:
      "The current story is work AI that acts, not just finds: secure search, answers, agents, connectors, and orchestration across internal company systems.",
    landmarkContributions: [
      "Permissions-aware enterprise search",
      "Strong enterprise-context layer for assistants and agents",
      "APIs for building custom agents on company knowledge",
    ],
    keyProducts: ["Glean Search", "Glean Assistant", "Glean Agents", "Glean APIs", "Glean Protect"],
    achievements: [
      "Expanded search into one of the strongest enterprise AI categories",
      "Built a clear enterprise-context and connectors moat",
      "Turned knowledge retrieval into an agent platform story",
    ],
    category: "Enterprise AI",
    logo: "🗂️",
    website: "https://www.glean.com",
    isRecentAddition: true,
    sortScale: 455,
    sortOperating: 690,
    sortFounded: 2019,
    sources: [
      {
        label: "Work AI platform overview",
        url: "https://www.glean.com/searchengine",
        kind: "Product",
      },
      {
        label: "Glean APIs",
        url: "https://www.glean.com/product/api",
        kind: "Product",
      },
    ],
  },
  {
    id: 25,
    name: "Harvey",
    founded: "2022",
    founders: ["Winston Weinberg", "Gabriel Pereyra"],
    headquarters: "San Francisco, CA",
    scaleSignal: "Vertical AI company focused on legal and professional services",
    operatingSignal: "Legal research, due diligence, deal work, and complex professional workflows",
    description:
      "Harvey is a leading example of vertical AI: a company using foundation models to build workflow-native products for law firms and professional-service teams.",
    journey:
      "It gained traction quickly by focusing on a narrow, high-value domain rather than competing as a general assistant platform.",
    currentFocus:
      "The business is centered on secure, domain-specific AI for research, drafting, contract analysis, and matter workflows in legal and adjacent professional services.",
    landmarkContributions: [
      "Proved vertical AI can outrun general-purpose horizontal tools",
      "Helped define the legal-AI product category",
      "Packaged foundation-model capabilities inside domain workflows",
    ],
    keyProducts: ["Harvey Platform", "Legal Research", "Deal Management", "Due Diligence"],
    achievements: [
      "Established legal AI as a serious enterprise software category",
      "Won adoption among top law firms and in-house teams",
      "Made domain workflow design central to AI product strategy",
    ],
    category: "Enterprise AI",
    logo: "⚖️",
    website: "https://www.harvey.ai",
    isRecentAddition: true,
    sortScale: 430,
    sortOperating: 520,
    sortFounded: 2022,
    sources: [
      {
        label: "Harvey platform",
        url: "https://www.harvey.ai/",
        kind: "Official site",
      },
    ],
  },
  {
    id: 26,
    name: "Sierra",
    founded: "2023",
    founders: ["Bret Taylor", "Clay Bavor"],
    headquarters: "San Francisco, CA",
    scaleSignal: "Customer-experience AI company built around branded agents",
    operatingSignal: "Cross-channel agents for chat, SMS, voice, email, and more",
    description:
      "Sierra focuses on customer experience agents, helping brands deploy AI representatives across service and engagement channels with outcome-based pricing.",
    journey:
      "It entered the market as a focused company rather than a platform for everything, betting that customer-facing agent quality would be its own major category.",
    currentFocus:
      "The core product goal is brand-safe customer interaction: one agent deployed consistently across multiple communication channels with measurable CX outcomes.",
    landmarkContributions: [
      "Customer-facing AI agents as a standalone product category",
      "Outcome-based pricing for conversational AI deployments",
      "Cross-channel deployment across major support surfaces",
    ],
    keyProducts: ["Sierra platform", "Customer agents", "Cross-channel deployment"],
    achievements: [
      "Built a sharp product identity around customer experience",
      "Focused enterprise AI value on brand and support outcomes",
      "Positioned AI agents as a CX layer, not just a call-center tool",
    ],
    category: "Enterprise AI",
    logo: "🏔️",
    website: "https://sierra.ai",
    isRecentAddition: true,
    sortScale: 410,
    sortOperating: 500,
    sortFounded: 2023,
    sources: [
      {
        label: "Sierra homepage",
        url: "https://sierra.ai/",
        kind: "Official site",
      },
    ],
  },
  {
    id: 27,
    name: "ElevenLabs",
    founded: "2022",
    founders: ["Mati Staniszewski", "Piotr Dabkowski"],
    headquarters: "London, UK / New York, NY",
    scaleSignal: "Voice AI platform spanning creation, dubbing, and real-time agents",
    operatingSignal: "Conversational AI, support agents, multilingual voice, and low-latency speech",
    description:
      "ElevenLabs is one of the strongest voice-native AI companies, combining speech generation, conversational agents, dubbing, and real-time developer tooling.",
    journey:
      "It broke out through synthetic voice quality and expanded quickly into broader speech infrastructure and customer-service agent products.",
    currentFocus:
      "The platform now targets both creators and enterprises, with fast voice and chat agents that can be grounded in company workflows and multilingual support operations.",
    landmarkContributions: [
      "High-quality synthetic voice generation",
      "Real-time conversational AI for support and service",
      "Multilingual voice tooling at developer and enterprise layers",
    ],
    keyProducts: ["Conversational AI", "ElevenAgents", "Voice generation", "Dubbing"],
    achievements: [
      "Made voice AI feel materially more human and expressive",
      "Expanded from creative tooling into enterprise operations",
      "Established a strong speech-first product identity",
    ],
    category: "Generative AI",
    logo: "🗣️",
    website: "https://elevenlabs.io",
    isRecentAddition: true,
    sortScale: 425,
    sortOperating: 610,
    sortFounded: 2022,
    sources: [
      {
        label: "Conversational AI platform",
        url: "https://elevenlabs.io/conversational-ai/",
        kind: "Product",
      },
      {
        label: "ElevenAgents for Support",
        url: "https://elevenlabs.io/agents/elevenagents-for-support",
        kind: "Product",
      },
    ],
  },
  {
    id: 28,
    name: "Runway",
    founded: "2018",
    founders: ["Cristóbal Valenzuela", "Alejandro Matamala", "Anastasis Germanidis"],
    headquarters: "New York, NY",
    scaleSignal: "Creative AI suite for video, image, and editing workflows",
    operatingSignal: "Gen-4.5, creative tooling, video editing, and multi-model workflow surfaces",
    description:
      "Runway sits at the intersection of generative media research and creator software, combining proprietary video models with editing and content-production tooling.",
    journey:
      "It started as an ML toolkit for creators and has become a full creative AI environment for generating, editing, and refining video and image content.",
    currentFocus:
      "Its product direction now emphasizes an all-in-one creative workspace with best-in-class media models, editing surfaces, and production-friendly tooling.",
    landmarkContributions: [
      "Helped define AI-native video generation as a product category",
      "Combined model access with creator-facing editing workflows",
      "Kept creative tooling central while the market shifted toward agents",
    ],
    keyProducts: ["Gen-4.5", "Gen-4 Image", "Act-Two", "Runway creative suite"],
    achievements: [
      "Built one of the strongest brands in AI-native video creation",
      "Maintained a product-led creator focus amid fast model churn",
      "Expanded from pure generation into broader production tooling",
    ],
    category: "Generative AI",
    logo: "🎬",
    website: "https://runwayml.com",
    isRecentAddition: true,
    sortScale: 435,
    sortOperating: 580,
    sortFounded: 2018,
    sources: [
      {
        label: "Runway product overview",
        url: "https://runwayml.com/product",
        kind: "Product",
      },
      {
        label: "Available models on Runway",
        url: "https://help.runwayml.com/hc/en-us/articles/48649877897107-Available-Models-on-Runway",
        kind: "Docs",
      },
    ],
  },
];

export const companyCategories = [
  "All",
  ...Array.from(new Set(companies.map((company) => company.category))).sort(),
];
