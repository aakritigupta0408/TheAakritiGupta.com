import React, { useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Navigation from "../components/Navigation";

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
      "GPT-4 API",
      "DALL-E 3",
      "Whisper (speech recognition)",
      "Codex (code generation)",
    ],
    achievements: [
      "100M+ ChatGPT users in 2 months",
      "Pioneered large language model applications",
      "Set new standards for AI safety research",
      "First to achieve GPT-4 level performance",
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
      "Bard (conversational AI)",
      "Gemini (multimodal AI)",
      "AlphaFold database",
      "Google Search AI integration",
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
      "Claude Pro",
      "Constitutional AI framework",
      "AI safety research publications",
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
      "Llama 2 (open-source LLM)",
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
      "Bing Chat search enhancement",
    ],
    keyProducts: [
      "Microsoft Copilot",
      "Bing Chat",
      "Azure OpenAI Service",
      "Office 365 AI features",
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
];

export default function AICompanies() {
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>("All");
  const [sortBy, setSortBy] = useState<string>("valuation");

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

  const getFilteredCompanies = () => {
    let filtered = [...companies];

    if (filterCategory !== "All") {
      filtered = filtered.filter(
        (company) => company.category === filterCategory,
      );
    }

    if (sortBy === "valuation") {
      filtered.sort((a, b) => {
        const valA = parseFloat(a.valuation.replace(/[$TB,\s]/g, ""));
        const valB = parseFloat(b.valuation.replace(/[$TB,\s]/g, ""));
        return valB - valA;
      });
    } else if (sortBy === "founded") {
      filtered.sort((a, b) => parseInt(b.founded) - parseInt(a.founded));
    } else if (sortBy === "employees") {
      filtered.sort((a, b) => {
        const empA = parseInt(a.employees.replace(/[,+\s]/g, ""));
        const empB = parseInt(b.employees.replace(/[,+\s]/g, ""));
        return empB - empA;
      });
    }

    return filtered;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white">
      <Navigation />

      <div className="container mx-auto px-6 py-12">
        {/* Header */}
        <div className="text-center mb-16">
          <motion.h1
            className="text-6xl font-bold text-black mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            AI Companies Leading the Revolution
          </motion.h1>
          <motion.p
            className="text-xl text-gray-600 max-w-4xl mx-auto mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Explore the top 20 companies shaping the AI landscape with their
            groundbreaking discoveries, innovative products, and transformative
            journeys from startups to industry giants.
          </motion.p>

          {/* Statistics */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
              <div className="text-3xl font-bold text-blue-600">$2.8T+</div>
              <div className="text-sm text-gray-600">Combined Valuation</div>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
              <div className="text-3xl font-bold text-green-600">500K+</div>
              <div className="text-sm text-gray-600">Total Employees</div>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
              <div className="text-3xl font-bold text-purple-600">100+</div>
              <div className="text-sm text-gray-600">Breakthrough Products</div>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
              <div className="text-3xl font-bold text-orange-600">20+</div>
              <div className="text-sm text-gray-600">
                Years Combined History
              </div>
            </div>
          </motion.div>
        </div>

        {/* Filters and Sort */}
        <div className="mb-8 space-y-4">
          <div className="flex flex-wrap gap-2 justify-center">
            <span className="text-sm font-medium text-gray-700 px-3 py-2">
              Filter by Category:
            </span>
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all ${
                  filterCategory === category
                    ? "bg-blue-500 text-white border-blue-500"
                    : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
                }`}
              >
                {category}
              </button>
            ))}
          </div>

          <div className="flex gap-2 justify-center">
            <span className="text-sm font-medium text-gray-700 px-3 py-2">
              Sort by:
            </span>
            {[
              { value: "valuation", label: "Valuation" },
              { value: "founded", label: "Founded Date" },
              { value: "employees", label: "Company Size" },
            ].map((sort) => (
              <button
                key={sort.value}
                onClick={() => setSortBy(sort.value)}
                className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all ${
                  sortBy === sort.value
                    ? "bg-blue-500 text-white border-blue-500"
                    : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
                }`}
              >
                {sort.label}
              </button>
            ))}
          </div>
        </div>

        {/* Companies Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 mb-12">
          {getFilteredCompanies().map((company, index) => (
            <motion.div
              key={company.id}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden hover:shadow-xl transition-all duration-300 cursor-pointer group"
              onClick={() => setSelectedCompany(company)}
            >
              <div className="p-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="text-4xl">{company.logo}</div>
                  <div className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-medium">
                    {company.category}
                  </div>
                </div>

                {/* Company Name & Basic Info */}
                <h3 className="text-lg font-bold text-black mb-2 group-hover:text-blue-600 transition-colors">
                  {company.name}
                </h3>

                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Founded:</span>
                    <span className="font-semibold">{company.founded}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Valuation:</span>
                    <span className="font-semibold text-green-600">
                      {company.valuation}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Employees:</span>
                    <span className="font-semibold">{company.employees}</span>
                  </div>
                </div>

                {/* Description */}
                <p className="text-gray-600 text-sm line-clamp-3 mb-4">
                  {company.description}
                </p>

                {/* Key Products Preview */}
                <div className="border-t pt-4">
                  <p className="text-xs font-semibold text-gray-800 mb-2">
                    Key Products:
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {company.keyProducts.slice(0, 3).map((product, idx) => (
                      <span
                        key={idx}
                        className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded"
                      >
                        {product}
                      </span>
                    ))}
                    {company.keyProducts.length > 3 && (
                      <span className="text-xs text-gray-500">
                        +{company.keyProducts.length - 3} more
                      </span>
                    )}
                  </div>
                </div>

                {/* View Details Button */}
                <div className="mt-4 text-center">
                  <div className="text-xs text-blue-600 font-medium group-hover:underline">
                    Click to explore journey →
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

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
                <div className="p-8">
                  {/* Header */}
                  <div className="flex justify-between items-start mb-8">
                    <div className="flex items-center gap-6">
                      <div className="text-6xl">{selectedCompany.logo}</div>
                      <div>
                        <h2 className="text-4xl font-bold text-black mb-2">
                          {selectedCompany.name}
                        </h2>
                        <div className="flex items-center gap-4 text-lg">
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
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedCompany(null)}
                      className="text-gray-500 hover:text-gray-700 text-3xl font-light"
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
        <div className="text-center">
          <Link to="/" className="button-secondary">
            ← Back to Portfolio
          </Link>
        </div>
      </div>
    </div>
  );
}
