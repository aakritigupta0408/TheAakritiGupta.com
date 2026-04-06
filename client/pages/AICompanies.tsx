import React, { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Navigation from "../components/Navigation";
import {
  latestAIResearchBreakthroughs,
  startupWatchlist,
} from "../data/aiSignals";

interface Company {
  id: number;
  name: string;
  founded: string;
  founders: string[];
  headquarters: string;
  valuation: string;
  employees: string;
  description: string;
  journey: string;
  landmarkDiscoveries: string[];
  keyProducts: string[];
  achievements: string[];
  category: string;
  logo: string;
  website: string;
  stockSymbol?: string;
  isRecentAddition?: boolean;
  sortScale?: number;
  sortEmployees?: number;
  sortFounded?: number;
}

const companies: Company[] = [
  {
    id: 1,
    name: "OpenAI",
    founded: "2015",
    founders: ["Sam Altman", "Elon Musk", "Greg Brockman", "Ilya Sutskever"],
    headquarters: "San Francisco, CA",
    valuation: "$80 billion",
    employees: "1,000+",
    description:
      "Leading AI research company focused on artificial general intelligence (AGI) that benefits humanity.",
    journey:
      "Started as a non-profit research organization in 2015, transitioned to a capped-profit model in 2019. Gained worldwide attention with ChatGPT launch in 2022, becoming the fastest-growing consumer application in history.",
    landmarkDiscoveries: [
      "GPT series (GPT-1, GPT-2, GPT-3, GPT-4)",
      "DALL-E image generation",
      "Reinforcement Learning from Human Feedback (RLHF)",
      "ChatGPT conversational AI breakthrough",
    ],
    keyProducts: [
      "ChatGPT",
      "ChatGPT agent and deep research",
      "Codex",
      "gpt-image-1",
      "Whisper (speech recognition)",
      "Sora",
    ],
    achievements: [
      "100M+ ChatGPT users in 2 months",
      "Pioneered large language model applications",
      "Set new standards for AI safety research",
      "Expanded from chat into research and coding agents",
    ],
    category: "AI Research",
    logo: "🤖",
    website: "https://openai.com",
  },
  {
    id: 2,
    name: "Google DeepMind",
    founded: "2010 (DeepMind) / 2023 (merged)",
    founders: ["Demis Hassabis", "Shane Legg", "Mustafa Suleyman"],
    headquarters: "London, UK / Mountain View, CA",
    valuation: "Part of Alphabet ($1.7T)",
    employees: "2,500+",
    description:
      "AI research laboratory that combines DeepMind and Google Brain to advance artificial general intelligence.",
    journey:
      "DeepMind founded in 2010, acquired by Google in 2014 for $500M. Merged with Google Brain in 2023 to form Google DeepMind, creating one of the world's largest AI research organizations.",
    landmarkDiscoveries: [
      "AlphaGo - first AI to beat world Go champion",
      "AlphaFold - protein structure prediction",
      "Transformer architecture (Google Brain)",
      "Deep Q-Networks (DQN)",
    ],
    keyProducts: [
      "Gemini app and Gemini model family",
      "Gemini Robotics",
      "AlphaFold 3 and AlphaFold Server",
      "Veo and multimodal generation tools",
      "YouTube recommendations",
    ],
    achievements: [
      "Solved 50-year protein folding problem",
      "Defeated world champions in Go, Chess, StarCraft II",
      "Breakthrough in weather prediction with GraphCast",
      "Advanced materials discovery with GNoME",
    ],
    category: "AI Research",
    logo: "🧠",
    website: "https://deepmind.google",
  },
  {
    id: 3,
    name: "Anthropic",
    founded: "2021",
    founders: ["Dario Amodei", "Daniela Amodei"],
    headquarters: "San Francisco, CA",
    valuation: "$18.4 billion",
    employees: "500+",
    description:
      "AI safety company building reliable, interpretable, and steerable AI systems with a focus on constitutional AI.",
    journey:
      "Founded by former OpenAI researchers focusing on AI safety. Raised $450M in Series B, then $4B from Amazon. Launched Claude AI assistant in 2022, competing directly with ChatGPT.",
    landmarkDiscoveries: [
      "Constitutional AI training method",
      "Claude AI assistant architecture",
      "AI safety research breakthroughs",
      "Honest AI response generation",
    ],
    keyProducts: [
      "Claude (AI assistant)",
      "Claude Opus 4.5",
      "Claude Code",
      "Model Context Protocol (MCP)",
      "Computer Use and agent tooling",
    ],
    achievements: [
      "Pioneered constitutional AI approach",
      "Leading AI safety research",
      "Amazon's $4B strategic investment",
      "High performance on reasoning benchmarks",
    ],
    category: "AI Safety",
    logo: "🛡️",
    website: "https://anthropic.com",
  },
  {
    id: 4,
    name: "Nvidia",
    founded: "1993",
    founders: ["Jensen Huang", "Chris Malachowsky", "Curtis Priem"],
    headquarters: "Santa Clara, CA",
    valuation: "$1.8 trillion",
    employees: "29,600+",
    description:
      "Computing platform company that has become the backbone of AI infrastructure with GPU technology.",
    journey:
      "Started as graphics card company in 1993. Pivoted to AI/ML in 2010s when researchers discovered GPUs' parallel processing power for neural networks. Now dominates AI chip market with 90%+ share.",
    landmarkDiscoveries: [
      "CUDA parallel computing platform",
      "GPU acceleration for deep learning",
      "Transformer architecture optimization",
      "Real-time ray tracing",
    ],
    keyProducts: [
      "H100 AI chips",
      "A100 data center GPUs",
      "RTX consumer graphics cards",
      "CUDA software platform",
      "Omniverse collaboration platform",
    ],
    achievements: [
      "Enabled the deep learning revolution",
      "90%+ AI training market share",
      "First trillion-dollar chip company",
      "H100 demand exceeds supply globally",
    ],
    category: "AI Infrastructure",
    logo: "🎮",
    website: "https://nvidia.com",
    stockSymbol: "NVDA",
  },
  {
    id: 5,
    name: "Meta AI",
    founded: "2013 (FAIR) / 2004 (Meta)",
    founders: ["Mark Zuckerberg", "Yann LeCun (FAIR)"],
    headquarters: "Menlo Park, CA",
    valuation: "$825 billion",
    employees: "77,000+",
    description:
      "Social media giant's AI division developing foundational AI models and metaverse technologies.",
    journey:
      "Facebook AI Research (FAIR) established in 2013 under Yann LeCun. Rebranded to Meta in 2021, investing $10B+ annually in AI and metaverse. Open-sourced Llama models in 2023.",
    landmarkDiscoveries: [
      "Convolutional Neural Networks advancement",
      "Llama large language models",
      "Computer vision breakthroughs",
      "Self-supervised learning research",
    ],
    keyProducts: [
      "Llama open model family",
      "Ray-Ban Meta smart glasses",
      "Instagram/Facebook AI features",
      "PyTorch framework",
      "Segment Anything Model (SAM)",
    ],
    achievements: [
      "Open-sourced competitive LLM (Llama)",
      "Pioneered social media AI applications",
      "Leading computer vision research",
      "PyTorch adoption by researchers globally",
    ],
    category: "Big Tech AI",
    logo: "📘",
    website: "https://ai.meta.com",
    stockSymbol: "META",
  },
  {
    id: 6,
    name: "Microsoft",
    founded: "1975",
    founders: ["Bill Gates", "Paul Allen"],
    headquarters: "Redmond, WA",
    valuation: "$2.8 trillion",
    employees: "221,000+",
    description:
      "Technology giant leveraging AI across cloud services, productivity tools, and enterprise solutions.",
    journey:
      "Invested $1B in OpenAI in 2019, followed by $10B in 2023. Integrated GPT models into Bing, Office 365, and Azure. Became OpenAI's exclusive cloud provider and key partner.",
    landmarkDiscoveries: [
      "Partnership with OpenAI",
      "Azure AI cloud services",
      "Copilot AI assistant integration",
      "AI-powered search and work integrations",
    ],
    keyProducts: [
      "Microsoft Copilot",
      "Microsoft 365 Copilot",
      "Azure OpenAI Service",
      "Copilot Studio",
      "GitHub Copilot",
    ],
    achievements: [
      "Strategic OpenAI partnership",
      "AI integration across product suite",
      "Azure AI cloud leadership",
      "First major search engine AI integration",
    ],
    category: "Big Tech AI",
    logo: "🪟",
    website: "https://microsoft.com/ai",
    stockSymbol: "MSFT",
  },
  {
    id: 7,
    name: "Tesla",
    founded: "2003",
    founders: ["Elon Musk", "Martin Eberhard", "Marc Tarpenning"],
    headquarters: "Austin, TX",
    valuation: "$800 billion",
    employees: "140,000+",
    description:
      "Electric vehicle and clean energy company pioneering autonomous driving and robotics with AI.",
    journey:
      "Started as EV company in 2003. Developed Full Self-Driving (FSD) technology using neural networks. Announced Optimus humanoid robot in 2021. Built custom AI training infrastructure with Dojo supercomputer.",
    landmarkDiscoveries: [
      "End-to-end neural networks for driving",
      "Real-world AI training from fleet data",
      "Dojo supercomputer architecture",
      "Optimus humanoid robot",
    ],
    keyProducts: [
      "Full Self-Driving (FSD)",
      "Autopilot",
      "Optimus robot",
      "Dojo supercomputer",
      "Neural network training platform",
    ],
    achievements: [
      "Largest autonomous vehicle fleet",
      "Real-world AI training at scale",
      "Custom AI chip development",
      "Advancing humanoid robotics",
    ],
    category: "Autonomous Systems",
    logo: "🚗",
    website: "https://tesla.com",
    stockSymbol: "TSLA",
  },
  {
    id: 8,
    name: "Stability AI",
    founded: "2020",
    founders: ["Emad Mostaque"],
    headquarters: "London, UK",
    valuation: "$4 billion",
    employees: "200+",
    description:
      "Open-source AI company democratizing access to AI through models like Stable Diffusion.",
    journey:
      "Founded in 2020 with mission to democratize AI. Released Stable Diffusion in 2022, sparking the generative AI art revolution. Faced leadership changes but continues developing open-source AI models.",
    landmarkDiscoveries: [
      "Stable Diffusion image generation",
      "Open-source AI model distribution",
      "Latent diffusion model innovation",
      "Community-driven AI development",
    ],
    keyProducts: [
      "Stable Diffusion",
      "DreamStudio platform",
      "Stable Video Diffusion",
      "Stable Audio",
      "Open-source model ecosystem",
    ],
    achievements: [
      "Democratized AI art generation",
      "Open-source AI movement leadership",
      "Billions of images generated",
      "Enabled creative AI revolution",
    ],
    category: "Generative AI",
    logo: "🎨",
    website: "https://stability.ai",
  },
  {
    id: 9,
    name: "Midjourney",
    founded: "2021",
    founders: ["David Holz"],
    headquarters: "San Francisco, CA",
    valuation: "$3 billion (estimated)",
    employees: "40+",
    description:
      "Independent research lab creating AI-powered tools for human imagination and creativity.",
    journey:
      "Founded in 2021 by David Holz. Launched AI art generator in 2022 via Discord bot. Achieved profitability with 15M+ users despite small team size. Known for high-quality, artistic AI-generated images.",
    landmarkDiscoveries: [
      "Discord-based AI art generation",
      "High-quality artistic AI images",
      "Community-driven AI creativity",
      "Efficient AI model deployment",
    ],
    keyProducts: [
      "Midjourney AI art generator",
      "Discord bot interface",
      "Community galleries",
      "Style transfer capabilities",
    ],
    achievements: [
      "15M+ active users",
      "Profitable with small team",
      "Industry-leading image quality",
      "Creator economy transformation",
    ],
    category: "Generative AI",
    logo: "🖼️",
    website: "https://midjourney.com",
  },
  {
    id: 10,
    name: "Databricks",
    founded: "2013",
    founders: ["Ali Ghodsi", "Ion Stoica", "Matei Zaharia"],
    headquarters: "San Francisco, CA",
    valuation: "$43 billion",
    employees: "6,000+",
    description:
      "Data and AI platform company providing unified analytics for data engineering and machine learning.",
    journey:
      "Founded by Apache Spark creators in 2013. Built unified data platform for big data and ML. Went public considerations in 2023, focusing on AI/ML workloads and data governance.",
    landmarkDiscoveries: [
      "Apache Spark framework",
      "Unified data and AI platform",
      "Delta Lake data format",
      "MLflow ML lifecycle management",
    ],
    keyProducts: [
      "Databricks Platform",
      "Apache Spark",
      "Delta Lake",
      "MLflow",
      "Databricks SQL",
    ],
    achievements: [
      "Apache Spark ecosystem leadership",
      "Enterprise AI platform adoption",
      "$43B valuation achievement",
      "Open-source community building",
    ],
    category: "AI Platform",
    logo: "🧱",
    website: "https://databricks.com",
  },
  {
    id: 11,
    name: "Scale AI",
    founded: "2016",
    founders: ["Alexandr Wang"],
    headquarters: "San Francisco, CA",
    valuation: "$13.8 billion",
    employees: "1,000+",
    description:
      "Data platform for AI providing high-quality training data for machine learning models.",
    journey:
      "Founded by 19-year-old Alexandr Wang in 2016. Started with autonomous vehicle data labeling. Expanded to serve government, enterprises. Became critical infrastructure for AI training data across industries.",
    landmarkDiscoveries: [
      "Large-scale data labeling platform",
      "AI training data standardization",
      "Human-in-the-loop AI systems",
      "Multi-modal data annotation",
    ],
    keyProducts: [
      "Scale Data Platform",
      "Scale Nucleus",
      "Scale Government solutions",
      "RLHF platform",
      "3D sensor data processing",
    ],
    achievements: [
      "Youngest billionaire CEO",
      "Critical AI infrastructure provider",
      "Government AI contracts",
      "Enterprise AI data standards",
    ],
    category: "AI Infrastructure",
    logo: "📊",
    website: "https://scale.com",
  },
  {
    id: 12,
    name: "Cohere",
    founded: "2019",
    founders: ["Aidan Gomez", "Ivan Zhang", "Nick Frosst"],
    headquarters: "Toronto, Canada",
    valuation: "$5.5 billion",
    employees: "300+",
    description:
      "Enterprise AI platform providing large language models and natural language processing tools.",
    journey:
      "Founded by former Google researchers who co-authored the Transformer paper. Focused on enterprise LLMs and enterprise-grade AI solutions. Raised $270M+ from leading VCs and strategic investors.",
    landmarkDiscoveries: [
      "Co-invented Transformer architecture",
      "Enterprise-focused LLM development",
      "Multilingual language models",
      "Retrieval-augmented generation",
    ],
    keyProducts: [
      "Command language model",
      "Embed text embeddings",
      "Classify text classification",
      "Generate API",
      "Coral chat interface",
    ],
    achievements: [
      "Co-created Transformer architecture",
      "Leading enterprise AI adoption",
      "Multilingual AI capabilities",
      "Strong enterprise partnerships",
    ],
    category: "Enterprise AI",
    logo: "🔗",
    website: "https://cohere.com",
  },
  {
    id: 13,
    name: "Hugging Face",
    founded: "2016",
    founders: ["Clément Delangue", "Julien Chaumond", "Thomas Wolf"],
    headquarters: "New York, NY",
    valuation: "$4.5 billion",
    employees: "200+",
    description:
      "AI community platform hosting models, datasets, and applications for machine learning collaboration.",
    journey:
      "Started as chatbot company in 2016. Pivoted to open-source ML tools in 2018. Became the 'GitHub of AI' with millions of users sharing models. Raised $235M to build AI community ecosystem.",
    landmarkDiscoveries: [
      "Transformers library for NLP",
      "Open-source AI model ecosystem",
      "Gradio interface framework",
      "Spaces AI app platform",
    ],
    keyProducts: [
      "Transformers library",
      "Model Hub",
      "Datasets library",
      "Gradio",
      "Spaces platform",
    ],
    achievements: [
      "100M+ model downloads monthly",
      "Leading open-source AI platform",
      "Active developer community",
      "Democratizing AI access",
    ],
    category: "AI Platform",
    logo: "🤗",
    website: "https://huggingface.co",
  },
  {
    id: 14,
    name: "Palantir",
    founded: "2003",
    founders: ["Peter Thiel", "Alex Karp", "Joe Lonsdale"],
    headquarters: "Denver, CO",
    valuation: "$55 billion",
    employees: "3,500+",
    description:
      "Big data analytics company specializing in AI-powered data integration and analysis platforms.",
    journey:
      "Founded in 2003 for counter-terrorism. Expanded to commercial markets. Went public in 2020. Integrated AI/ML capabilities into data platforms for government and enterprise customers.",
    landmarkDiscoveries: [
      "Large-scale data integration",
      "AI-powered analytics platform",
      "Government AI applications",
      "Edge AI deployment",
    ],
    keyProducts: [
      "Palantir Gotham",
      "Palantir Foundry",
      "AIP (AI Platform)",
      "Edge AI solutions",
    ],
    achievements: [
      "Critical government AI infrastructure",
      "Public company transition",
      "Large-scale data processing",
      "AI/ML enterprise adoption",
    ],
    category: "Enterprise AI",
    logo: "👁️",
    website: "https://palantir.com",
    stockSymbol: "PLTR",
  },
  {
    id: 15,
    name: "UiPath",
    founded: "2005",
    founders: ["Daniel Dines", "Marius Tirca"],
    headquarters: "New York, NY",
    valuation: "$12 billion",
    employees: "4,000+",
    description:
      "Robotic Process Automation company using AI to automate business processes and workflows.",
    journey:
      "Founded in Romania in 2005. Moved to US, became RPA leader. Went public in 2021. Integrated AI/ML into automation platform for intelligent process automation across enterprises.",
    landmarkDiscoveries: [
      "Robotic Process Automation platform",
      "AI-powered process mining",
      "Computer vision for automation",
      "Natural language processing automation",
    ],
    keyProducts: [
      "UiPath Platform",
      "Process Mining",
      "Task Mining",
      "Document Understanding",
      "AI Fabric",
    ],
    achievements: [
      "RPA market leadership",
      "Public company achievement",
      "Enterprise automation adoption",
      "AI-powered workflow optimization",
    ],
    category: "Process Automation",
    logo: "🤖",
    website: "https://uipath.com",
    stockSymbol: "PATH",
  },
  {
    id: 16,
    name: "SenseTime",
    founded: "2014",
    founders: ["Tang Xiaoou", "Xu Li"],
    headquarters: "Hong Kong / Shanghai",
    valuation: "$12 billion",
    employees: "5,000+",
    description:
      "Chinese AI company specializing in computer vision, facial recognition, and smart city solutions.",
    journey:
      "Founded by Chinese University of Hong Kong professors in 2014. Became world's most valuable AI startup by 2018. Faced US sanctions in 2021. Went public in Hong Kong in 2021.",
    landmarkDiscoveries: [
      "Deep learning computer vision",
      "Facial recognition technology",
      "Smart city AI platforms",
      "Autonomous driving perception",
    ],
    keyProducts: [
      "SenseFoundry platform",
      "SenseAuto autonomous driving",
      "SenseStudy education AI",
      "SenseHealth medical AI",
    ],
    achievements: [
      "World's most valuable AI startup (2018)",
      "Leading computer vision technology",
      "Smart city deployments",
      "Hong Kong IPO completion",
    ],
    category: "Computer Vision",
    logo: "👁️‍🗨️",
    website: "https://sensetime.com",
  },
  {
    id: 17,
    name: "DataRobot",
    founded: "2012",
    founders: ["Jeremy Achin", "Tom de Godoy"],
    headquarters: "Boston, MA",
    valuation: "$6.3 billion",
    employees: "2,000+",
    description:
      "Enterprise AI platform automating machine learning model development and deployment.",
    journey:
      "Founded in 2012 to democratize machine learning. Built automated ML platform for enterprises. Raised $1B+ in funding. Focuses on responsible AI and enterprise ML operations.",
    landmarkDiscoveries: [
      "Automated machine learning platform",
      "Enterprise ML operations",
      "Responsible AI framework",
      "Time series forecasting automation",
    ],
    keyProducts: [
      "DataRobot Platform",
      "MLOps solutions",
      "AI Cloud",
      "Automated time series",
      "Model monitoring",
    ],
    achievements: [
      "Enterprise ML platform leadership",
      "Automated ML innovation",
      "Responsible AI advancement",
      "Large enterprise customer base",
    ],
    category: "Enterprise AI",
    logo: "🎯",
    website: "https://datarobot.com",
  },
  {
    id: 18,
    name: "C3.ai",
    founded: "2009",
    founders: ["Tom Siebel"],
    headquarters: "Redwood City, CA",
    valuation: "$2.5 billion",
    employees: "1,500+",
    description:
      "Enterprise AI software platform providing AI applications for digital transformation.",
    journey:
      "Founded by CRM pioneer Tom Siebel in 2009. Built enterprise AI platform for large organizations. Went public in 2020. Focuses on AI applications for energy, manufacturing, and government.",
    landmarkDiscoveries: [
      "Enterprise AI application platform",
      "Model-driven development",
      "AI suite for digital transformation",
      "Industry-specific AI solutions",
    ],
    keyProducts: [
      "C3 AI Suite",
      "C3 AI Applications",
      "Energy AI solutions",
      "Manufacturing AI",
      "Ex Machina platform",
    ],
    achievements: [
      "Enterprise AI platform pioneer",
      "Public company achievement",
      "Large enterprise deployments",
      "Industry-specific AI solutions",
    ],
    category: "Enterprise AI",
    logo: "🏭",
    website: "https://c3.ai",
    stockSymbol: "AI",
  },
  {
    id: 19,
    name: "Cerebras",
    founded: "2016",
    founders: ["Andrew Feldman", "Sean Lie"],
    headquarters: "Sunnyvale, CA",
    valuation: "$4 billion",
    employees: "500+",
    description:
      "AI chip company creating the world's largest computer processors for AI training and inference.",
    journey:
      "Founded in 2016 to build specialized AI hardware. Developed world's largest computer chip - CS-2 wafer-scale engine. Targets AI training workloads requiring massive compute power.",
    landmarkDiscoveries: [
      "Wafer-scale processor technology",
      "CS-2 AI training chip",
      "Specialized AI hardware architecture",
      "High-bandwidth memory integration",
    ],
    keyProducts: [
      "CS-2 Wafer-Scale Engine",
      "Cerebras software stack",
      "AI training systems",
      "Sparse neural networks",
    ],
    achievements: [
      "World's largest computer processor",
      "AI hardware innovation leadership",
      "Specialized training optimization",
      "Alternative to GPU dominance",
    ],
    category: "AI Hardware",
    logo: "🧠",
    website: "https://cerebras.net",
  },
  {
    id: 20,
    name: "Graphcore",
    founded: "2016",
    founders: ["Nigel Toon", "Simon Knowles"],
    headquarters: "Bristol, UK",
    valuation: "$2.8 billion",
    employees: "700+",
    description:
      "AI chip company developing Intelligence Processing Units (IPUs) for machine learning workloads.",
    journey:
      "Founded in 2016 by Bristol veterans. Developed IPU architecture optimized for AI/ML. Raised $700M+ from investors. Competes with Nvidia in AI training market with specialized processors.",
    landmarkDiscoveries: [
      "Intelligence Processing Unit (IPU)",
      "Poplar software framework",
      "Sparse computation optimization",
      "Graph-based processing architecture",
    ],
    keyProducts: [
      "IPU processors",
      "Poplar software stack",
      "Bow Pod systems",
      "IPU-POD datacenter solutions",
    ],
    achievements: [
      "IPU architecture innovation",
      "Alternative AI chip architecture",
      "European AI hardware leadership",
      "Partnership with major cloud providers",
    ],
    category: "AI Hardware",
    logo: "⚡",
    website: "https://graphcore.ai",
  },
  {
    id: 21,
    name: "Mistral AI",
    founded: "2023",
    founders: ["Arthur Mensch", "Timothée Lacroix", "Guillaume Lample"],
    headquarters: "Paris, France",
    valuation: "Le Chat Enterprise + frontier model platform",
    employees: "Fast-growing global team",
    description:
      "Frontier AI company building open-weight and enterprise-ready models, assistants, document intelligence, and sovereign deployment options.",
    journey:
      "Founded in 2023 by former DeepMind and Meta researchers, Mistral quickly became one of Europe's most important AI companies. Its 2025-2026 product cycle emphasized Le Chat Enterprise, deep research, OCR, and flexible cloud or self-hosted deployments for customers that want strong models with infrastructure control.",
    landmarkDiscoveries: [
      "Mistral and Mixtral open-weight model releases",
      "Le Chat Enterprise platform for work AI",
      "Mistral OCR for document-heavy workflows",
      "Enterprise deployment model spanning cloud and self-hosted options",
    ],
    keyProducts: [
      "Le Chat",
      "Le Chat Enterprise",
      "Mistral OCR",
      "Deep Research in Le Chat",
      "AI Studio and enterprise deployments",
    ],
    achievements: [
      "Established Europe as a serious frontier-model contender",
      "Positioned around privacy, sovereignty, and deployment flexibility",
      "Expanded beyond chat into document AI and research workflows",
      "Joined NVIDIA's Nemotron Coalition in March 2026",
    ],
    category: "AI Research",
    logo: "🇫🇷",
    website: "https://mistral.ai",
    isRecentAddition: true,
    sortScale: 185,
    sortEmployees: 450,
    sortFounded: 2023,
  },
  {
    id: 22,
    name: "Cursor",
    founded: "2022",
    founders: ["Michael Truell", "Sualeh Asif", "Arvid Lunnemark", "Aman Sanger"],
    headquarters: "San Francisco, CA",
    valuation: "$1B+ annualized revenue signal",
    employees: "Growing research + product team",
    description:
      "Agentic coding company turning the IDE into a multi-agent software workspace for editing, testing, browsing, and preparing PRs.",
    journey:
      "Cursor moved from AI-assisted editing into full agentic software development in 2025 and 2026. Its company updates highlighted a Series D and more than $1B in annualized revenue, while Cursor 3 reframed the product around persistent agents, multi-repo workflows, and clearer human review surfaces.",
    landmarkDiscoveries: [
      "Agent-first coding workspace design",
      "Composer 2 research for longer-horizon coding tasks",
      "Marketplace plugin ecosystem for tool-connected development",
      "Local and cloud agent handoff inside the same workflow",
    ],
    keyProducts: [
      "Cursor 3",
      "Composer 2",
      "Cursor Marketplace",
      "Self-hosted cloud agents",
      "JetBrains IDE support",
    ],
    achievements: [
      "Pushed coding tools beyond autocomplete into agent supervision",
      "Added multi-agent and multi-repo workflows in April 2026",
      "Expanded ecosystem integrations through the Cursor Marketplace",
      "Shared that it had passed $1B in annualized revenue in November 2025",
    ],
    category: "AI Platform",
    logo: "⌨️",
    website: "https://cursor.com",
    isRecentAddition: true,
    sortScale: 240,
    sortEmployees: 180,
    sortFounded: 2022,
  },
  {
    id: 23,
    name: "Perplexity",
    founded: "2022",
    founders: ["Aravind Srinivas", "Denis Yarats", "Johnny Ho", "Andy Konwinski"],
    headquarters: "San Francisco, CA",
    valuation: "Deep Research + Computer platform",
    employees: "Fast-growing research and product team",
    description:
      "Research and answer engine company expanding from citation-first search into computer-use, finance workflows, and enterprise automation.",
    journey:
      "Perplexity built early momentum with answer-engine search, then expanded into deeper reasoning and tool use. By early 2026 it had moved into Perplexity Computer, Personal Computer, Comet Enterprise, and finance tooling, broadening the company from search assistant into a more capable work system.",
    landmarkDiscoveries: [
      "Citation-first AI answer engine pattern",
      "Perplexity Deep Research workflow",
      "Computer-use orchestration across tools and web workflows",
      "Finance and enterprise context integration",
    ],
    keyProducts: [
      "Perplexity Search",
      "Deep Research",
      "Perplexity Computer",
      "Personal Computer",
      "Comet Enterprise",
    ],
    achievements: [
      "Helped define the answer-engine category",
      "Scaled from search into agentic research and computer use",
      "Added enterprise and finance-specific product surfaces in 2026",
      "Emphasized auditable, source-grounded analysis workflows",
    ],
    category: "AI Platform",
    logo: "🔎",
    website: "https://www.perplexity.ai",
    isRecentAddition: true,
    sortScale: 210,
    sortEmployees: 220,
    sortFounded: 2022,
  },
  {
    id: 24,
    name: "Glean",
    founded: "2019",
    founders: ["Arvind Jain", "Tony Gentilcore", "Piyush Prahladka", "T.R. Vishwanath"],
    headquarters: "Palo Alto, CA",
    valuation: ">$200M ARR in Dec 2025",
    employees: "Enterprise-scale team",
    description:
      "Work AI platform focused on enterprise search, grounded assistants, and autonomous agents built on organizational context and permissions.",
    journey:
      "Glean started with enterprise search, then expanded into a broader Work AI platform. Its 2025-2026 releases emphasized Glean Agents, Enterprise Context, proactive assistants, and autonomous agents that can work across tools like Salesforce, Jira, GitHub, and Microsoft ecosystems.",
    landmarkDiscoveries: [
      "Enterprise Graph and organizational context layer",
      "Open horizontal agent platform for work AI",
      "Autonomous agents grounded in connectors, memory, and governance",
      "Permission-aware enterprise search plus action workflows",
    ],
    keyProducts: [
      "Glean Assistant",
      "Glean Agents",
      "Enterprise Context",
      "Agent Builder",
      "Deep Research",
    ],
    achievements: [
      "Reported surpassing $200M ARR in December 2025",
      "Launched autonomous agents with enterprise context",
      "Expanded connectors, actions, and governance across enterprise tools",
      "Positioned itself as a core enterprise AI platform rather than a point assistant",
    ],
    category: "Enterprise AI",
    logo: "🏢",
    website: "https://www.glean.com",
    isRecentAddition: true,
    sortScale: 220,
    sortEmployees: 900,
    sortFounded: 2019,
  },
  {
    id: 25,
    name: "Harvey",
    founded: "2022",
    founders: ["Winston Weinberg", "Gabriel Pereyra"],
    headquarters: "San Francisco, CA",
    valuation: "Global legal AI platform momentum",
    employees: "Growing legal AI team",
    description:
      "Vertical AI company building legal research, drafting, review, and workflow systems tailored to law firms and in-house legal teams.",
    journey:
      "Harvey became one of the clearest examples of vertical AI becoming real software. Its 2026 momentum centered on agent-powered legal workflows, Microsoft 365 Copilot integration, expanding global benchmarks, and high-profile enterprise rollouts across firms and regulated organizations.",
    landmarkDiscoveries: [
      "BigLaw Bench evaluation framework",
      "Agentic legal workflow orchestration",
      "Jurisdiction-aware legal research benchmarking",
      "Document review and matter-centric analysis flows",
    ],
    keyProducts: [
      "Harvey Assistant",
      "Workflow Agents",
      "Vault and Review Tables",
      "Deep Analysis",
      "Microsoft 365 Copilot integration",
    ],
    achievements: [
      "Expanded BigLaw Bench with global and research-focused datasets in 2026",
      "Embedded legal intelligence into Microsoft 365 Copilot in March 2026",
      "Won major enterprise deployments including HSBC",
      "Broadened regional legal data coverage and workflow tooling",
    ],
    category: "Enterprise AI",
    logo: "⚖️",
    website: "https://www.harvey.ai",
    isRecentAddition: true,
    sortScale: 205,
    sortEmployees: 300,
    sortFounded: 2022,
  },
  {
    id: 26,
    name: "Sierra",
    founded: "2023",
    founders: ["Bret Taylor", "Clay Bavor"],
    headquarters: "San Francisco, CA",
    valuation: ">$150M ARR signal in Feb 2026",
    employees: "Growing enterprise CX team",
    description:
      "Customer-experience AI company building on-brand agents that can reason, take action, use memory, and operate across channels.",
    journey:
      "Sierra launched as a customer-service agent platform and quickly evolved toward a full Agent OS. Its late-2025 and early-2026 updates focused on Agent OS 2.0, Agent Data Platform, multi-model orchestration, voice quality, and metrics tied to real end-to-end task completion.",
    landmarkDiscoveries: [
      "Agent OS architecture for customer operations",
      "Agent Data Platform for memory and context",
      "Constellation-of-models orchestration across 15+ models",
      "Tau-bench expansions for knowledge and voice evaluation",
    ],
    keyProducts: [
      "Sierra Agent OS",
      "Agent Studio 2.0",
      "Agent Data Platform",
      "Voice performance tooling",
      "Ghostwriter agent builder",
    ],
    achievements: [
      "Reported over $150M ARR in February 2026",
      "Expanded from answers into memory, action, and continuous improvement",
      "Built a differentiated multi-model architecture for brand-safe agents",
      "Advanced benchmarking for knowledge and voice-based agents",
    ],
    category: "Enterprise AI",
    logo: "🎯",
    website: "https://sierra.ai",
    isRecentAddition: true,
    sortScale: 215,
    sortEmployees: 250,
    sortFounded: 2023,
  },
  {
    id: 27,
    name: "ElevenLabs",
    founded: "2022",
    founders: ["Mati Staniszewski", "Piotr Dabkowski"],
    headquarters: "London, UK / New York, NY",
    valuation: "33M+ conversations handled in 2026",
    employees: "200+",
    description:
      "Voice AI company expanding from high-quality text-to-speech into full conversational agents that can talk, type, retrieve knowledge, and take actions.",
    journey:
      "ElevenLabs became widely known for speech synthesis, then broadened into low-latency conversational systems. Its March 2026 updates reframed the platform as ElevenLabs Agents, layering turn-taking, RAG, multimodality, telephony, and enterprise readiness onto its core voice stack.",
    landmarkDiscoveries: [
      "Human-like low-latency text-to-speech",
      "Turn-taking model for live voice agents",
      "Multimodal voice + text conversational agents",
      "Integrated RAG inside voice workflows",
    ],
    keyProducts: [
      "ElevenLabs Agents",
      "Conversational AI 2.0",
      "Multimodal Conversational AI",
      "Voice Library",
      "Telephony and SIP integrations",
    ],
    achievements: [
      "Repositioned from speech tooling to a full agent platform in March 2026",
      "Shared that customers created more than 2 million agents",
      "Reported more than 33 million conversations handled in 2026",
      "Added stronger enterprise-readiness, multimodality, and broader telephony support",
    ],
    category: "Generative AI",
    logo: "🎙️",
    website: "https://elevenlabs.io",
    isRecentAddition: true,
    sortScale: 200,
    sortEmployees: 220,
    sortFounded: 2022,
  },
  {
    id: 28,
    name: "Runway",
    founded: "2018",
    founders: ["Cristóbal Valenzuela", "Alejandro Matamala", "Anastasis Germanidis"],
    headquarters: "New York, NY",
    valuation: "$315M Series E in Feb 2026",
    employees: "Growing media + research team",
    description:
      "Creative AI company building video, image, and world-simulation products for media creation, digital characters, and interactive visual experiences.",
    journey:
      "Runway helped define generative video, then spent 2025 and 2026 broadening into world models, API products, real-time characters, an incubator, and a startup fund. The company now positions itself as both a media-generation platform and a broader world-simulation company.",
    landmarkDiscoveries: [
      "Gen-4 and Gen-4.5 for controllable video generation",
      "World model framing for media and simulation",
      "Runway Characters real-time video agent API",
      "API-first tooling for creative and product builders",
    ],
    keyProducts: [
      "Runway Gen-4",
      "Runway Gen-4.5",
      "Runway Characters",
      "Runway Builders",
      "Runway Fund",
    ],
    achievements: [
      "Raised a $315M Series E in February 2026",
      "Shipped real-time video agent tooling with Runway Characters",
      "Expanded beyond creation tools into Runway Builders and Runway Fund",
      "Continued pushing consistency and controllability in AI media generation",
    ],
    category: "Generative AI",
    logo: "🎬",
    website: "https://runwayml.com",
    isRecentAddition: true,
    sortScale: 230,
    sortEmployees: 400,
    sortFounded: 2018,
  },
];

export default function AICompanies() {
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>("All");
  const [sortBy, setSortBy] = useState<string>("valuation");

  const recentAdditionsCount = useMemo(
    () => companies.filter((company) => company.isRecentAddition).length,
    [],
  );

  const parseScaleValue = (value: string) => {
    const normalized = value.toLowerCase();
    const match = normalized.match(/(\d+(\.\d+)?)/);

    if (!match) return 0;

    const amount = parseFloat(match[1]);

    if (/\btrillion\b|t(?=[)\s]|$)/.test(normalized)) {
      return amount * 1000;
    }

    if (/\bmillion\b|m(?=[)\s]|$)/.test(normalized)) {
      return amount / 1000;
    }

    return amount;
  };

  const categories = [
    "All",
    "AI Research",
    "Big Tech AI",
    "AI Infrastructure",
    "Generative AI",
    "Enterprise AI",
    "AI Platform",
    "Computer Vision",
    "AI Hardware",
    "Autonomous Systems",
    "Process Automation",
    "AI Safety",
  ];

  const parseEmployeeValue = (value: string) => {
    const match = value.replace(/,/g, "").match(/(\d+)/);
    return match ? parseInt(match[1], 10) : 0;
  };

  const parseFoundedValue = (value: string) => {
    const match = value.match(/(\d{4})/);
    return match ? parseInt(match[1], 10) : 0;
  };

  const filteredCompanies = useMemo(() => {
    let filtered = [...companies];

    if (filterCategory !== "All") {
      filtered = filtered.filter(
        (company) => company.category === filterCategory,
      );
    }

    if (sortBy === "valuation") {
      filtered.sort((a, b) => {
        const valA = a.sortScale ?? parseScaleValue(a.valuation);
        const valB = b.sortScale ?? parseScaleValue(b.valuation);
        return valB - valA;
      });
    } else if (sortBy === "founded") {
      filtered.sort(
        (a, b) =>
          (b.sortFounded ?? parseFoundedValue(b.founded)) -
          (a.sortFounded ?? parseFoundedValue(a.founded)),
      );
    } else if (sortBy === "employees") {
      filtered.sort((a, b) => {
        const empA = a.sortEmployees ?? parseEmployeeValue(a.employees);
        const empB = b.sortEmployees ?? parseEmployeeValue(b.employees);
        return empB - empA;
      });
    }

    return filtered;
  }, [filterCategory, sortBy]);

  const visibleRecentAdditions = filteredCompanies.filter(
    (company) => company.isRecentAddition,
  ).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-900 via-teal-900 to-cyan-900 relative overflow-x-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 left-20 w-96 h-96 bg-emerald-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-60 right-16 w-80 h-80 bg-cyan-500/20 rounded-full blur-3xl animate-bounce"></div>
        <div className="absolute bottom-20 left-1/3 w-72 h-72 bg-teal-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-40 left-1/2 w-88 h-88 bg-blue-500/20 rounded-full blur-3xl animate-bounce delay-500"></div>
        <div className="absolute bottom-40 right-1/4 w-64 h-64 bg-green-500/20 rounded-full blur-3xl animate-pulse delay-700"></div>
      </div>

      <Navigation />

      <div className="container mx-auto px-6 pt-28 pb-12 sm:pt-32">
        {/* Header */}
        <div className="text-center mb-16 relative z-10">
          <motion.div
            className="inline-block p-1 rounded-full bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 mb-8"
            initial={{ opacity: 0, scale: 0.92, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 1.2, ease: "backOut" }}
          >
            <h1 className="text-4xl sm:text-5xl md:text-7xl xl:text-8xl font-black bg-gradient-to-r from-white via-emerald-100 to-cyan-100 bg-clip-text text-transparent px-6 py-5 sm:px-8 sm:py-6">
              AI Companies Revolution
            </h1>
          </motion.div>

          <motion.p
            className="text-xl text-gray-100 max-w-5xl mx-auto mb-8 leading-relaxed"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            🚀 Explore the companies shaping the AI landscape right now. The
            grid below now includes the newer AI leaders that surged after
            August 2025, alongside the established labs, infrastructure firms,
            and enterprise platforms already on the page. ✨
          </motion.p>

          <motion.div
            className="flex flex-wrap justify-center gap-4 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-bold">🏢 Industry Giants</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-bold">📊 Scale Snapshots</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-bold">🔬 Innovations</span>
            </div>
          </motion.div>

          {/* Statistics */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-6xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-300"
              whileHover={{ y: -8, scale: 1.03 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-emerald-400 to-green-500 bg-clip-text text-transparent">
                {companies.length}
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                Company Cards
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-300"
              whileHover={{ y: -8, scale: 1.03 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                {recentAdditionsCount}
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                New Since Aug 2025
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-300"
              whileHover={{ y: -8, scale: 1.03 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-teal-400 to-emerald-500 bg-clip-text text-transparent">
                {categories.length - 1}
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                Filter Categories
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-300"
              whileHover={{ y: -8, scale: 1.03 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-blue-400 to-cyan-500 bg-clip-text text-transparent">
                {filteredCompanies.length}
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                Matching Current Filter
              </div>
            </motion.div>
          </motion.div>
        </div>

        {/* Current Company Watch */}
        <motion.div
          className="mb-14 relative z-10 space-y-8"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.95 }}
        >
          <div className="text-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-300/30 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100">
              Startup and Scale-Up Watch · April 2026
            </div>
            <h2 className="mt-4 text-3xl md:text-4xl font-black text-white">
              Who Has Momentum Right Now
            </h2>
            <p className="mt-3 max-w-4xl mx-auto text-gray-100 leading-relaxed">
              Beyond the established giants, the market is being reshaped by
              startups building agentic coding, enterprise work AI, legal AI,
              customer-experience agents, voice systems, and AI-native media
              platforms.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {startupWatchlist.slice(0, 6).map((item, index) => (
              <motion.a
                key={item.id}
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1 + index * 0.07 }}
                className="group rounded-3xl border border-white/15 bg-slate-950/25 backdrop-blur-xl p-6 shadow-2xl transition-all duration-300 hover:-translate-y-1 hover:bg-slate-950/35"
              >
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div>
                    <div className="rounded-full border border-emerald-300/30 bg-emerald-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-emerald-100 inline-flex mb-3">
                      {item.focus}
                    </div>
                    <h3 className="text-2xl font-black text-white group-hover:text-emerald-200">
                      {item.name}
                    </h3>
                  </div>
                  <span className="text-xs font-semibold text-gray-300">
                    {item.date}
                  </span>
                </div>
                <p className="text-sm text-gray-100 leading-relaxed mb-4">
                  {item.latestMove}
                </p>
                <p className="text-sm text-cyan-100 leading-relaxed">
                  {item.whyItMatters}
                </p>
              </motion.a>
            ))}
          </div>

          <div className="rounded-[2rem] border border-white/15 bg-white/10 backdrop-blur-xl p-8">
            <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between mb-6">
              <div>
                <h3 className="text-2xl md:text-3xl font-black text-white">
                  Research Frontier Driving Company Strategy
                </h3>
                <p className="text-gray-100 mt-2 max-w-3xl">
                  The companies with the strongest stories right now are the
                  ones translating frontier research into deployable systems.
                </p>
              </div>
              <p className="text-sm text-emerald-100 font-medium">
                Official lab announcements
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              {latestAIResearchBreakthroughs.slice(0, 3).map((signal, index) => (
                <motion.a
                  key={signal.id}
                  href={signal.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 1.15 + index * 0.08 }}
                  className="rounded-3xl border border-white/15 bg-black/20 p-5 transition-all duration-300 hover:bg-black/30"
                >
                  <div className="flex items-center justify-between gap-4 mb-3">
                    <span className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                      {signal.category}
                    </span>
                    <span className="text-xs font-semibold text-gray-300">
                      {signal.date}
                    </span>
                  </div>
                  <h4 className="text-lg font-black text-white mb-2">
                    {signal.title}
                  </h4>
                  <p className="text-sm font-semibold text-emerald-100 mb-2">
                    {signal.org}
                  </p>
                  <p className="text-sm text-gray-100 leading-relaxed mb-3">
                    {signal.summary}
                  </p>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    {signal.impact}
                  </p>
                </motion.a>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Filters and Sort */}
        <div className="mb-12 space-y-6 relative z-10">
          <div className="mx-auto flex max-w-5xl flex-col gap-4 rounded-3xl border border-white/15 bg-white/10 p-5 text-center backdrop-blur-xl md:flex-row md:items-center md:justify-between md:text-left">
            <div>
              <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-100">
                Company Grid Status
              </p>
              <p className="mt-2 text-lg font-semibold text-white">
                Showing {filteredCompanies.length} of {companies.length} companies
              </p>
              <p className="mt-1 text-sm text-gray-200">
                {visibleRecentAdditions} recent additions are visible in this
                view.
              </p>
            </div>
            {filterCategory !== "All" && (
              <motion.button
                onClick={() => setFilterCategory("All")}
                className="rounded-full border border-emerald-300/40 bg-emerald-400/15 px-5 py-3 text-sm font-bold text-white transition-all duration-300 hover:bg-emerald-400/25"
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
              >
                Clear Category Filter
              </motion.button>
            )}
          </div>

          <div className="flex flex-wrap gap-4 justify-center">
            <span className="text-sm font-black text-white px-6 py-3 bg-white/10 backdrop-blur-md rounded-full border border-white/20">
              🎨 Filter by Category:
            </span>
            {categories.slice(0, 6).map((category) => (
              <motion.button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`px-6 py-3 rounded-full text-sm font-bold transition-all duration-300 ${
                  filterCategory === category
                    ? "bg-gradient-to-r from-emerald-500 to-cyan-600 text-white shadow-2xl scale-105 border border-emerald-400/50"
                    : "bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/20 hover:scale-105"
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {category}
              </motion.button>
            ))}
          </div>

          {categories.length > 6 && (
            <div className="flex flex-wrap gap-4 justify-center">
              {categories.slice(6).map((category) => (
                <motion.button
                  key={category}
                  onClick={() => setFilterCategory(category)}
                  className={`px-6 py-3 rounded-full text-sm font-bold transition-all duration-300 ${
                    filterCategory === category
                      ? "bg-gradient-to-r from-emerald-500 to-cyan-600 text-white shadow-2xl scale-105 border border-emerald-400/50"
                      : "bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/20 hover:scale-105"
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {category}
                </motion.button>
              ))}
            </div>
          )}

          <div className="flex flex-wrap gap-4 justify-center">
            <span className="text-sm font-black text-white px-6 py-3 bg-white/10 backdrop-blur-md rounded-full border border-white/20">
              🔄 Sort by:
            </span>
            {[
              { value: "valuation", label: "Scale Snapshot", emoji: "📊" },
              { value: "founded", label: "Founded Date", emoji: "📅" },
              { value: "employees", label: "Company Size", emoji: "👥" },
            ].map((sort) => (
              <motion.button
                key={sort.value}
                onClick={() => setSortBy(sort.value)}
                className={`px-6 py-3 rounded-full text-sm font-bold transition-all duration-300 ${
                  sortBy === sort.value
                    ? "bg-gradient-to-r from-teal-500 to-cyan-600 text-white shadow-2xl scale-105 border border-teal-400/50"
                    : "bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/20 hover:scale-105"
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {sort.emoji} {sort.label}
              </motion.button>
            ))}
          </div>
        </div>

        {/* Companies Grid */}
        {filteredCompanies.length === 0 ? (
          <div className="mb-12 rounded-[2rem] border border-white/15 bg-white/10 p-10 text-center shadow-2xl backdrop-blur-xl relative z-10">
            <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-100">
              No Matches Right Now
            </p>
            <h3 className="mt-3 text-3xl font-black text-white">
              The current filters hid every company card.
            </h3>
            <p className="mx-auto mt-3 max-w-2xl text-gray-100">
              Reset the category filter to bring the full company grid back and
              compare the newly added 2025-2026 companies with the older
              leaders.
            </p>
            <motion.button
              onClick={() => {
                setFilterCategory("All");
                setSortBy("valuation");
              }}
              className="mt-6 rounded-full border border-emerald-300/40 bg-gradient-to-r from-emerald-500 to-cyan-600 px-6 py-3 text-sm font-bold text-white shadow-2xl"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Reset Filters
            </motion.button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8 mb-12 relative z-10">
            {filteredCompanies.map((company, index) => (
              <motion.div
                key={company.id}
                initial={{ opacity: 0, y: 50, scale: 0.8 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{
                  duration: 0.6,
                  delay: Math.min(index * 0.04, 0.32),
                  ease: "backOut",
                }}
                className="relative bg-white/10 backdrop-blur-xl rounded-3xl border border-white/20 overflow-hidden transition-all duration-500 cursor-pointer group shadow-2xl hover:shadow-emerald-500/25 will-change-transform"
                onClick={() => setSelectedCompany(company)}
                whileHover={{
                  y: -8,
                  scale: 1.02,
                }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="p-8">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-6">
                    <motion.div
                      className="text-5xl group-hover:scale-110 transition-transform duration-300"
                      whileHover={{ rotate: 10, scale: 1.2 }}
                    >
                      {company.logo}
                    </motion.div>
                    <motion.div
                      className="text-xs bg-emerald-500/30 text-emerald-200 px-4 py-2 rounded-full font-bold border border-emerald-400/50 backdrop-blur-md"
                      whileHover={{ scale: 1.1 }}
                    >
                      {company.category}
                    </motion.div>
                  </div>

                  {company.isRecentAddition && (
                    <div className="mb-4 inline-flex rounded-full border border-cyan-300/40 bg-cyan-400/10 px-3 py-1 text-[11px] font-black uppercase tracking-[0.2em] text-cyan-100">
                      New Since Aug 2025
                    </div>
                  )}

                  {/* Company Name & Basic Info */}
                  <h3 className="text-xl font-black text-white mb-4 group-hover:bg-gradient-to-r group-hover:from-emerald-400 group-hover:to-cyan-500 group-hover:bg-clip-text group-hover:text-transparent transition-all duration-300">
                    {company.name}
                  </h3>

                  <div className="space-y-3 mb-6">
                    <div className="flex justify-between text-sm bg-white/10 backdrop-blur-md rounded-xl p-3 border border-white/20">
                      <span className="text-gray-200 font-medium">
                        📅 Founded:
                      </span>
                      <span className="font-bold text-white">
                        {company.founded}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm bg-white/10 backdrop-blur-md rounded-xl p-3 border border-white/20">
                      <span className="text-gray-200 font-medium">
                        📊 Scale:
                      </span>
                      <span className="font-bold text-emerald-300">
                        {company.valuation}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm bg-white/10 backdrop-blur-md rounded-xl p-3 border border-white/20">
                      <span className="text-gray-200 font-medium">
                        👥 Employees:
                      </span>
                      <span className="font-bold text-white">
                        {company.employees}
                      </span>
                    </div>
                  </div>

                  {/* Description */}
                  <p className="text-gray-200 text-sm line-clamp-3 mb-6 leading-relaxed">
                    {company.description}
                  </p>

                  {/* Key Products Preview */}
                  <div className="border-t border-white/20 pt-6">
                    <p className="text-xs font-bold text-cyan-300 mb-3">
                      🚀 Key Products:
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {company.keyProducts.slice(0, 3).map((product, idx) => (
                        <motion.span
                          key={idx}
                          className="text-xs bg-white/20 text-gray-200 px-3 py-2 rounded-full font-medium border border-white/30 backdrop-blur-md"
                          whileHover={{
                            scale: 1.05,
                          }}
                        >
                          {product}
                        </motion.span>
                      ))}
                      {company.keyProducts.length > 3 && (
                        <span className="text-xs text-gray-300 font-medium bg-white/10 rounded-full px-3 py-2">
                          +{company.keyProducts.length - 3} more
                        </span>
                      )}
                    </div>
                  </div>

                  {/* View Details Button */}
                  <div className="mt-6 text-center">
                    <motion.div
                      className="text-sm text-white font-bold group-hover:text-emerald-300 transition-colors duration-300 bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 rounded-full px-4 py-2 border border-emerald-400/30"
                      whileHover={{ scale: 1.05 }}
                    >
                      ✨ Click to explore journey →
                    </motion.div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}

        {/* Detailed Modal */}
        <AnimatePresence>
          {selectedCompany && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
              onClick={() => setSelectedCompany(null)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="bg-white rounded-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="p-6 sm:p-8">
                  {/* Header */}
                  <div className="mb-8 flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-6">
                      <div className="text-5xl sm:text-6xl">
                        {selectedCompany.logo}
                      </div>
                      <div>
                        <h2 className="mb-2 text-3xl sm:text-4xl font-bold text-black">
                          {selectedCompany.name}
                        </h2>
                        <div className="flex flex-col gap-2 text-base sm:flex-row sm:items-center sm:gap-4 sm:text-lg">
                          <span className="text-gray-600">
                            Founded: <strong>{selectedCompany.founded}</strong>
                          </span>
                          <span className="text-green-600 font-bold">
                            {selectedCompany.valuation}
                          </span>
                        </div>
                        <div className="mt-2">
                          <span className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-sm font-medium">
                            {selectedCompany.category}
                          </span>
                          {selectedCompany.isRecentAddition && (
                            <span className="ml-2 inline-block rounded-full bg-emerald-100 px-3 py-1 text-sm font-medium text-emerald-700">
                              New since Aug 2025
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedCompany(null)}
                      className="self-end text-3xl font-light text-gray-500 hover:text-gray-700 md:self-auto"
                    >
                      ×
                    </button>
                  </div>

                  {/* Company Details Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                    {/* Basic Info */}
                    <div className="lg:col-span-1">
                      <h3 className="text-xl font-bold mb-4">Company Info</h3>
                      <div className="space-y-3 bg-gray-50 p-4 rounded-lg">
                        <div>
                          <span className="font-semibold">Headquarters:</span>
                          <p className="text-gray-700">
                            {selectedCompany.headquarters}
                          </p>
                        </div>
                        <div>
                          <span className="font-semibold">Employees:</span>
                          <p className="text-gray-700">
                            {selectedCompany.employees}
                          </p>
                        </div>
                        <div>
                          <span className="font-semibold">Founders:</span>
                          <p className="text-gray-700">
                            {selectedCompany.founders.join(", ")}
                          </p>
                        </div>
                        {selectedCompany.stockSymbol && (
                          <div>
                            <span className="font-semibold">Stock:</span>
                            <p className="text-gray-700">
                              {selectedCompany.stockSymbol}
                            </p>
                          </div>
                        )}
                        <div>
                          <span className="font-semibold">Website:</span>
                          <a
                            href={selectedCompany.website}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:underline block"
                          >
                            {selectedCompany.website.replace("https://", "")}
                          </a>
                        </div>
                      </div>
                    </div>

                    {/* Journey & Description */}
                    <div className="lg:col-span-2">
                      <h3 className="text-xl font-bold mb-4">
                        Company Journey
                      </h3>
                      <div className="space-y-4">
                        <div>
                          <h4 className="font-semibold text-lg mb-2">About</h4>
                          <p className="text-gray-700">
                            {selectedCompany.description}
                          </p>
                        </div>
                        <div>
                          <h4 className="font-semibold text-lg mb-2">
                            Journey
                          </h4>
                          <p className="text-gray-700">
                            {selectedCompany.journey}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Landmark Discoveries */}
                  <div className="mb-8">
                    <h3 className="text-xl font-bold mb-4">
                      🏆 Landmark Discoveries
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {selectedCompany.landmarkDiscoveries.map(
                        (discovery, idx) => (
                          <div
                            key={idx}
                            className="bg-blue-50 p-4 rounded-lg border border-blue-200"
                          >
                            <div className="flex items-start gap-3">
                              <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                                {idx + 1}
                              </div>
                              <p className="text-gray-800">{discovery}</p>
                            </div>
                          </div>
                        ),
                      )}
                    </div>
                  </div>

                  {/* Key Products */}
                  <div className="mb-8">
                    <h3 className="text-xl font-bold mb-4">🚀 Key Products</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {selectedCompany.keyProducts.map((product, idx) => (
                        <div
                          key={idx}
                          className="bg-green-50 p-4 rounded-lg border border-green-200"
                        >
                          <p className="font-semibold text-green-800">
                            {product}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Achievements */}
                  <div className="mb-8">
                    <h3 className="text-xl font-bold mb-4">
                      🎯 Major Achievements
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {selectedCompany.achievements.map((achievement, idx) => (
                        <div
                          key={idx}
                          className="bg-purple-50 p-4 rounded-lg border border-purple-200"
                        >
                          <div className="flex items-start gap-3">
                            <div className="text-purple-600 text-xl">✓</div>
                            <p className="text-gray-800">{achievement}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Bottom Navigation */}
        <div className="text-center relative z-10">
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link
              to="/"
              className="inline-block bg-gradient-to-r from-emerald-500 to-cyan-600 text-white font-bold px-8 py-4 rounded-full hover:from-emerald-600 hover:to-cyan-700 transition-all duration-300 shadow-2xl border border-white/20 backdrop-blur-md"
            >
              ← Back to Portfolio
            </Link>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
