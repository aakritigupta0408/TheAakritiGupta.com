import React, { useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Navigation from "../components/Navigation";

interface AITool {
  name: string;
  link: string;
  description: string;
  pricing: string;
  category: string;
  rating: number;
}

interface Profession {
  id: number;
  title: string;
  icon: string;
  description: string;
  impactLevel: "High" | "Medium" | "Critical";
  aiAdoption: number;
  primaryTool: AITool;
  alternativeTools: AITool[];
  useCase: string;
  timesSaved: string;
}

const professions: Profession[] = [
  {
    id: 1,
    title: "Software Developers",
    icon: "💻",
    description:
      "AI transforms coding with intelligent code completion, bug detection, and automated testing.",
    impactLevel: "Critical",
    aiAdoption: 95,
    primaryTool: {
      name: "GitHub Copilot",
      link: "https://github.com/features/copilot",
      description:
        "AI pair programmer that suggests code and entire functions in real-time as you type.",
      pricing: "$10/month",
      category: "Code Generation",
      rating: 4.8,
    },
    alternativeTools: [
      {
        name: "Tabnine",
        link: "https://www.tabnine.com/",
        description: "AI code completion that learns from your codebase",
        pricing: "Free / $12/month",
        category: "Code Completion",
        rating: 4.5,
      },
      {
        name: "Cursor",
        link: "https://cursor.sh/",
        description: "AI-first code editor built for pair programming with AI",
        pricing: "$20/month",
        category: "IDE",
        rating: 4.7,
      },
    ],
    useCase: "Code generation, debugging, documentation, and code reviews",
    timesSaved: "40-60% development time",
  },
  {
    id: 2,
    title: "Content Writers",
    icon: "✍️",
    description:
      "AI assists with ideation, drafting, editing, and optimizing content for various platforms.",
    impactLevel: "High",
    aiAdoption: 88,
    primaryTool: {
      name: "ChatGPT",
      link: "https://chat.openai.com/",
      description:
        "Advanced AI assistant for writing, editing, brainstorming, and content strategy.",
      pricing: "Free / $20/month",
      category: "Writing Assistant",
      rating: 4.6,
    },
    alternativeTools: [
      {
        name: "Jasper",
        link: "https://www.jasper.ai/",
        description: "AI marketing copywriter with brand voice training",
        pricing: "$49/month",
        category: "Marketing Copy",
        rating: 4.4,
      },
      {
        name: "Copy.ai",
        link: "https://www.copy.ai/",
        description: "AI copywriting platform for marketing and sales content",
        pricing: "Free / $36/month",
        category: "Marketing Copy",
        rating: 4.3,
      },
    ],
    useCase:
      "Blog posts, marketing copy, social media content, and email campaigns",
    timesSaved: "50-70% writing time",
  },
  {
    id: 3,
    title: "Graphic Designers",
    icon: "🎨",
    description:
      "AI revolutionizes visual creation with generative design, image editing, and style transfer.",
    impactLevel: "High",
    aiAdoption: 82,
    primaryTool: {
      name: "Midjourney",
      link: "https://midjourney.com/",
      description:
        "AI image generator creating high-quality artwork from text descriptions.",
      pricing: "$10/month",
      category: "Image Generation",
      rating: 4.7,
    },
    alternativeTools: [
      {
        name: "Adobe Firefly",
        link: "https://www.adobe.com/products/firefly.html",
        description:
          "Adobe's AI image generator integrated with Creative Suite",
        pricing: "Included in CC",
        category: "Image Generation",
        rating: 4.5,
      },
      {
        name: "Canva AI",
        link: "https://www.canva.com/ai-image-generator/",
        description: "AI-powered design tools within Canva platform",
        pricing: "Free / $15/month",
        category: "Design Platform",
        rating: 4.4,
      },
    ],
    useCase:
      "Concept art, logo design, marketing materials, and visual ideation",
    timesSaved: "60-80% design iteration time",
  },
  {
    id: 4,
    title: "Data Scientists",
    icon: "📊",
    description:
      "AI accelerates data analysis, model building, and insight generation from complex datasets.",
    impactLevel: "Critical",
    aiAdoption: 92,
    primaryTool: {
      name: "DataRobot",
      link: "https://www.datarobot.com/",
      description:
        "Automated machine learning platform that accelerates data science workflows.",
      pricing: "Contact Sales",
      category: "AutoML",
      rating: 4.6,
    },
    alternativeTools: [
      {
        name: "H2O.ai",
        link: "https://h2o.ai/",
        description: "Open source machine learning and AI platform",
        pricing: "Free / Enterprise",
        category: "ML Platform",
        rating: 4.5,
      },
      {
        name: "Databricks",
        link: "https://databricks.com/",
        description: "Unified analytics platform for data science and ML",
        pricing: "Pay-as-you-go",
        category: "Data Platform",
        rating: 4.4,
      },
    ],
    useCase:
      "Predictive modeling, data exploration, automated feature engineering",
    timesSaved: "70-85% model development time",
  },
  {
    id: 5,
    title: "Marketing Professionals",
    icon: "📈",
    description:
      "AI transforms campaign optimization, customer segmentation, and personalized messaging.",
    impactLevel: "High",
    aiAdoption: 85,
    primaryTool: {
      name: "HubSpot AI",
      link: "https://www.hubspot.com/artificial-intelligence",
      description:
        "AI-powered marketing automation with predictive analytics and content optimization.",
      pricing: "$45/month",
      category: "Marketing Automation",
      rating: 4.5,
    },
    alternativeTools: [
      {
        name: "Marketo Engage",
        link: "https://business.adobe.com/products/marketo/adobe-marketo.html",
        description: "AI-driven marketing automation and lead management",
        pricing: "$1,195/month",
        category: "Enterprise Marketing",
        rating: 4.3,
      },
      {
        name: "Persado",
        link: "https://www.persado.com/",
        description: "AI platform for generating persuasive marketing language",
        pricing: "Contact Sales",
        category: "Content Optimization",
        rating: 4.4,
      },
    ],
    useCase: "Campaign optimization, A/B testing, customer journey mapping",
    timesSaved: "40-55% campaign setup time",
  },
  {
    id: 6,
    title: "Customer Service Representatives",
    icon: "🎧",
    description:
      "AI chatbots and sentiment analysis transform customer support efficiency and quality.",
    impactLevel: "High",
    aiAdoption: 78,
    primaryTool: {
      name: "Zendesk Answer Bot",
      link: "https://www.zendesk.com/service/answer-bot/",
      description:
        "AI-powered chatbot that resolves customer inquiries automatically.",
      pricing: "$5/agent/month",
      category: "Customer Support",
      rating: 4.3,
    },
    alternativeTools: [
      {
        name: "Intercom Resolution Bot",
        link: "https://www.intercom.com/resolution-bot",
        description: "AI customer service bot with natural language processing",
        pricing: "$87/month",
        category: "Customer Support",
        rating: 4.4,
      },
      {
        name: "LivePerson",
        link: "https://www.liveperson.com/",
        description: "Conversational AI platform for customer engagement",
        pricing: "Contact Sales",
        category: "Conversational AI",
        rating: 4.2,
      },
    ],
    useCase: "Automated responses, ticket routing, sentiment analysis",
    timesSaved: "50-70% response time reduction",
  },
  {
    id: 7,
    title: "Translators",
    icon: "🌐",
    description:
      "AI translation tools provide instant, context-aware translations across multiple languages.",
    impactLevel: "Critical",
    aiAdoption: 90,
    primaryTool: {
      name: "DeepL",
      link: "https://www.deepl.com/",
      description:
        "AI-powered translator with superior accuracy and natural language understanding.",
      pricing: "Free / $6.99/month",
      category: "Translation",
      rating: 4.7,
    },
    alternativeTools: [
      {
        name: "Google Translate",
        link: "https://translate.google.com/",
        description: "Free AI translation service supporting 100+ languages",
        pricing: "Free",
        category: "Translation",
        rating: 4.2,
      },
      {
        name: "Microsoft Translator",
        link: "https://www.microsoft.com/en-us/translator/",
        description: "AI translation with real-time conversation features",
        pricing: "Free / Pay-per-use",
        category: "Translation",
        rating: 4.1,
      },
    ],
    useCase: "Document translation, real-time interpretation, localization",
    timesSaved: "80-90% translation time",
  },
  {
    id: 8,
    title: "Financial Analysts",
    icon: "💰",
    description:
      "AI enhances financial modeling, risk assessment, and market prediction accuracy.",
    impactLevel: "High",
    aiAdoption: 75,
    primaryTool: {
      name: "AlphaSense",
      link: "https://www.alpha-sense.com/",
      description:
        "AI-powered market intelligence platform for financial research and analysis.",
      pricing: "Contact Sales",
      category: "Financial Research",
      rating: 4.6,
    },
    alternativeTools: [
      {
        name: "Kensho",
        link: "https://www.kensho.com/",
        description: "AI analytics platform for financial markets",
        pricing: "Enterprise",
        category: "Market Analytics",
        rating: 4.5,
      },
      {
        name: "FactSet",
        link: "https://www.factset.com/",
        description: "Financial data platform with AI-driven insights",
        pricing: "Contact Sales",
        category: "Financial Data",
        rating: 4.4,
      },
    ],
    useCase: "Market analysis, portfolio optimization, risk modeling",
    timesSaved: "60-75% research time",
  },
  {
    id: 9,
    title: "Lawyers",
    icon: "⚖️",
    description:
      "AI assists with legal research, document review, contract analysis, and case prediction.",
    impactLevel: "Medium",
    aiAdoption: 65,
    primaryTool: {
      name: "LawGeex",
      link: "https://www.lawgeex.com/",
      description:
        "AI platform for automated contract review and legal document analysis.",
      pricing: "Contact Sales",
      category: "Legal Tech",
      rating: 4.5,
    },
    alternativeTools: [
      {
        name: "Westlaw Edge",
        link: "https://legal.thomsonreuters.com/en/products/westlaw",
        description: "AI-enhanced legal research platform",
        pricing: "Subscription",
        category: "Legal Research",
        rating: 4.3,
      },
      {
        name: "Luminance",
        link: "https://www.luminance.com/",
        description: "AI for legal document review and due diligence",
        pricing: "Contact Sales",
        category: "Document Review",
        rating: 4.4,
      },
    ],
    useCase: "Contract analysis, legal research, document review, compliance",
    timesSaved: "70-85% document review time",
  },
  {
    id: 10,
    title: "Healthcare Professionals",
    icon: "🏥",
    description:
      "AI transforms diagnostics, treatment planning, and patient care through medical imaging and data analysis.",
    impactLevel: "Critical",
    aiAdoption: 70,
    primaryTool: {
      name: "IBM Watson Health",
      link: "https://www.ibm.com/watson-health",
      description:
        "AI platform for healthcare data analysis, diagnostics, and treatment recommendations.",
      pricing: "Enterprise",
      category: "Healthcare AI",
      rating: 4.3,
    },
    alternativeTools: [
      {
        name: "Zebra Medical Vision",
        link: "https://www.zebra-med.com/",
        description: "AI medical imaging analysis for radiology",
        pricing: "Contact Sales",
        category: "Medical Imaging",
        rating: 4.4,
      },
      {
        name: "PathAI",
        link: "https://www.pathai.com/",
        description: "AI-powered pathology for accurate diagnosis",
        pricing: "Enterprise",
        category: "Pathology",
        rating: 4.5,
      },
    ],
    useCase: "Medical imaging, diagnosis assistance, treatment planning",
    timesSaved: "40-60% diagnostic time",
  },
  {
    id: 11,
    title: "Teachers & Educators",
    icon: "🎓",
    description:
      "AI personalizes learning, automates grading, and provides intelligent tutoring systems.",
    impactLevel: "Medium",
    aiAdoption: 60,
    primaryTool: {
      name: "Gradescope",
      link: "https://www.gradescope.com/",
      description:
        "AI-assisted grading platform that streamlines assessment and feedback.",
      pricing: "Free / $3/student",
      category: "Educational Assessment",
      rating: 4.6,
    },
    alternativeTools: [
      {
        name: "Khan Academy",
        link: "https://www.khanacademy.org/",
        description: "AI-powered personalized learning platform",
        pricing: "Free",
        category: "Learning Platform",
        rating: 4.7,
      },
      {
        name: "Carnegie Learning",
        link: "https://www.carnegielearning.com/",
        description: "AI tutoring systems for mathematics education",
        pricing: "Contact Sales",
        category: "Intelligent Tutoring",
        rating: 4.4,
      },
    ],
    useCase:
      "Automated grading, personalized learning paths, student assessment",
    timesSaved: "50-70% grading time",
  },
  {
    id: 12,
    title: "Sales Professionals",
    icon: "💼",
    description:
      "AI optimizes lead scoring, sales forecasting, and customer relationship management.",
    impactLevel: "High",
    aiAdoption: 80,
    primaryTool: {
      name: "Gong.io",
      link: "https://www.gong.io/",
      description:
        "AI platform that analyzes sales calls and provides actionable insights.",
      pricing: "Contact Sales",
      category: "Sales Intelligence",
      rating: 4.7,
    },
    alternativeTools: [
      {
        name: "Outreach",
        link: "https://www.outreach.io/",
        description: "AI-powered sales engagement platform",
        pricing: "$100/user/month",
        category: "Sales Engagement",
        rating: 4.5,
      },
      {
        name: "Salesforce Einstein",
        link: "https://www.salesforce.com/products/einstein/",
        description: "AI integrated into Salesforce CRM platform",
        pricing: "$25/user/month",
        category: "CRM AI",
        rating: 4.4,
      },
    ],
    useCase:
      "Call analysis, lead scoring, sales forecasting, opportunity insights",
    timesSaved: "35-50% sales cycle time",
  },
  {
    id: 13,
    title: "HR Professionals",
    icon: "👥",
    description:
      "AI streamlines recruitment, employee assessment, and workforce analytics.",
    impactLevel: "High",
    aiAdoption: 72,
    primaryTool: {
      name: "HireVue",
      link: "https://www.hirevue.com/",
      description: "AI-powered video interviewing and assessment platform.",
      pricing: "Contact Sales",
      category: "Recruitment",
      rating: 4.3,
    },
    alternativeTools: [
      {
        name: "Pymetrics",
        link: "https://www.pymetrics.ai/",
        description: "AI bias-free hiring and talent matching platform",
        pricing: "Contact Sales",
        category: "Talent Assessment",
        rating: 4.4,
      },
      {
        name: "Workday",
        link: "https://www.workday.com/",
        description: "AI-enhanced HR management and workforce planning",
        pricing: "Contact Sales",
        category: "HR Platform",
        rating: 4.2,
      },
    ],
    useCase: "Candidate screening, employee assessment, workforce analytics",
    timesSaved: "60-80% screening time",
  },
  {
    id: 14,
    title: "Accountants",
    icon: "🧮",
    description:
      "AI automates bookkeeping, fraud detection, and financial reporting processes.",
    impactLevel: "High",
    aiAdoption: 68,
    primaryTool: {
      name: "MindBridge AI",
      link: "https://www.mindbridge.ai/",
      description:
        "AI-powered financial risk discovery and audit analytics platform.",
      pricing: "Contact Sales",
      category: "Financial Analytics",
      rating: 4.5,
    },
    alternativeTools: [
      {
        name: "AppZen",
        link: "https://www.appzen.com/",
        description: "AI platform for expense report auditing and compliance",
        pricing: "Contact Sales",
        category: "Expense Management",
        rating: 4.4,
      },
      {
        name: "Botkeeper",
        link: "https://www.botkeeper.com/",
        description: "AI-powered bookkeeping and accounting automation",
        pricing: "$399/month",
        category: "Bookkeeping",
        rating: 4.3,
      },
    ],
    useCase: "Automated bookkeeping, fraud detection, financial analysis",
    timesSaved: "70-85% manual processing time",
  },
  {
    id: 15,
    title: "Journalists",
    icon: "📰",
    description:
      "AI assists with research, fact-checking, automated reporting, and content optimization.",
    impactLevel: "Medium",
    aiAdoption: 55,
    primaryTool: {
      name: "Quill by Narrative Science",
      link: "https://narrativescience.com/",
      description:
        "AI platform that generates written narratives from data automatically.",
      pricing: "Contact Sales",
      category: "Automated Writing",
      rating: 4.2,
    },
    alternativeTools: [
      {
        name: "Wordsmith",
        link: "https://automatedinsights.com/wordsmith",
        description: "Natural language generation platform for data stories",
        pricing: "Contact Sales",
        category: "Data Journalism",
        rating: 4.1,
      },
      {
        name: "Grammarly",
        link: "https://www.grammarly.com/",
        description: "AI writing assistant for editing and style improvement",
        pricing: "Free / $12/month",
        category: "Writing Assistant",
        rating: 4.5,
      },
    ],
    useCase:
      "Automated reporting, fact-checking, content editing, data analysis",
    timesSaved: "40-60% research and writing time",
  },
  {
    id: 16,
    title: "Video Editors",
    icon: "🎬",
    description:
      "AI revolutionizes video editing with automated cutting, effects, and content generation.",
    impactLevel: "High",
    aiAdoption: 75,
    primaryTool: {
      name: "Runway ML",
      link: "https://runwayml.com/",
      description:
        "AI-powered creative suite for video editing, generation, and effects.",
      pricing: "$12/month",
      category: "Video AI",
      rating: 4.6,
    },
    alternativeTools: [
      {
        name: "Descript",
        link: "https://www.descript.com/",
        description:
          "AI video editor with text-based editing and voice cloning",
        pricing: "Free / $12/month",
        category: "Video Editing",
        rating: 4.5,
      },
      {
        name: "Luma AI",
        link: "https://lumalabs.ai/",
        description: "AI for 3D capture and realistic video generation",
        pricing: "Free / Pro plans",
        category: "3D Video",
        rating: 4.4,
      },
    ],
    useCase:
      "Automated editing, background removal, style transfer, content generation",
    timesSaved: "50-70% editing time",
  },
  {
    id: 17,
    title: "Photographers",
    icon: "📸",
    description:
      "AI enhances photo editing, object removal, style transfer, and image enhancement.",
    impactLevel: "Medium",
    aiAdoption: 70,
    primaryTool: {
      name: "Luminar AI",
      link: "https://skylum.com/luminar",
      description:
        "AI-powered photo editing software with intelligent enhancement tools.",
      pricing: "$79 one-time",
      category: "Photo Editing",
      rating: 4.4,
    },
    alternativeTools: [
      {
        name: "Topaz Labs",
        link: "https://www.topazlabs.com/",
        description: "AI photo enhancement for noise reduction and upscaling",
        pricing: "$79-199",
        category: "Photo Enhancement",
        rating: 4.6,
      },
      {
        name: "Adobe Photoshop AI",
        link: "https://www.adobe.com/products/photoshop.html",
        description:
          "AI-powered features in Photoshop including neural filters",
        pricing: "$20.99/month",
        category: "Professional Editing",
        rating: 4.5,
      },
    ],
    useCase:
      "Automated editing, background replacement, noise reduction, upscaling",
    timesSaved: "60-80% post-processing time",
  },
  {
    id: 18,
    title: "Musicians & Composers",
    icon: "🎵",
    description:
      "AI creates original compositions, assists with mixing, and generates musical ideas.",
    impactLevel: "Medium",
    aiAdoption: 45,
    primaryTool: {
      name: "AIVA",
      link: "https://www.aiva.ai/",
      description:
        "AI composer that creates original music in various styles and genres.",
      pricing: "Free / €11/month",
      category: "Music Composition",
      rating: 4.3,
    },
    alternativeTools: [
      {
        name: "Amper Music",
        link: "https://www.ampermusic.com/",
        description: "AI music composition platform for content creators",
        pricing: "Contact Sales",
        category: "Background Music",
        rating: 4.1,
      },
      {
        name: "LANDR",
        link: "https://www.landr.com/",
        description: "AI mastering and music production platform",
        pricing: "$4/month",
        category: "Music Production",
        rating: 4.2,
      },
    ],
    useCase: "Music composition, automated mastering, melody generation",
    timesSaved: "30-50% composition time",
  },
  {
    id: 19,
    title: "Researchers",
    icon: "🔬",
    description:
      "AI accelerates literature review, data analysis, and hypothesis generation in research.",
    impactLevel: "High",
    aiAdoption: 65,
    primaryTool: {
      name: "Semantic Scholar",
      link: "https://www.semanticscholar.org/",
      description:
        "AI-powered academic search engine that understands scientific literature.",
      pricing: "Free",
      category: "Academic Research",
      rating: 4.5,
    },
    alternativeTools: [
      {
        name: "Elicit",
        link: "https://elicit.org/",
        description: "AI research assistant for finding and summarizing papers",
        pricing: "Free / $10/month",
        category: "Research Assistant",
        rating: 4.4,
      },
      {
        name: "ResearchRabbit",
        link: "https://www.researchrabbit.ai/",
        description: "AI tool for discovering and organizing research papers",
        pricing: "Free",
        category: "Literature Discovery",
        rating: 4.3,
      },
    ],
    useCase:
      "Literature review, paper discovery, research synthesis, data analysis",
    timesSaved: "50-70% literature review time",
  },
  {
    id: 20,
    title: "Project Managers",
    icon: "📋",
    description:
      "AI optimizes project planning, risk assessment, and resource allocation.",
    impactLevel: "Medium",
    aiAdoption: 58,
    primaryTool: {
      name: "Monday.com AI",
      link: "https://monday.com/",
      description:
        "Project management platform with AI-powered insights and automation.",
      pricing: "$8/user/month",
      category: "Project Management",
      rating: 4.6,
    },
    alternativeTools: [
      {
        name: "Asana Intelligence",
        link: "https://asana.com/intelligence",
        description: "AI features in Asana for smart project insights",
        pricing: "$10.99/user/month",
        category: "Project Management",
        rating: 4.5,
      },
      {
        name: "Notion AI",
        link: "https://www.notion.so/ai",
        description: "AI writing and planning assistant integrated in Notion",
        pricing: "$8/user/month",
        category: "Productivity",
        rating: 4.4,
      },
    ],
    useCase:
      "Project planning, risk assessment, timeline optimization, task automation",
    timesSaved: "25-40% planning and coordination time",
  },
];

export default function AITools() {
  const [selectedProfession, setSelectedProfession] =
    useState<Profession | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>("All");
  const [sortBy, setSortBy] = useState<string>("impact");

  const impactColors = {
    Critical: "bg-red-100 text-red-800 border-red-200",
    High: "bg-orange-100 text-orange-800 border-orange-200",
    Medium: "bg-yellow-100 text-yellow-800 border-yellow-200",
  };

  const getFilteredProfessions = () => {
    let filtered = [...professions];

    if (filterCategory !== "All") {
      filtered = filtered.filter((p) => p.impactLevel === filterCategory);
    }

    if (sortBy === "impact") {
      const impactOrder = { Critical: 3, High: 2, Medium: 1 };
      filtered.sort(
        (a, b) => impactOrder[b.impactLevel] - impactOrder[a.impactLevel],
      );
    } else if (sortBy === "adoption") {
      filtered.sort((a, b) => b.aiAdoption - a.aiAdoption);
    } else if (sortBy === "name") {
      filtered.sort((a, b) => a.title.localeCompare(b.title));
    }

    return filtered;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-400/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-20 w-96 h-96 bg-purple-400/20 rounded-full blur-3xl animate-bounce"></div>
        <div className="absolute bottom-20 left-1/4 w-80 h-80 bg-pink-400/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-60 left-1/2 w-64 h-64 bg-cyan-400/20 rounded-full blur-3xl animate-bounce delay-500"></div>
      </div>

      <Navigation />

      <div className="container mx-auto px-6 py-12">
        {/* Header */}
        <div className="text-center mb-16 relative z-10">
          <motion.div
            className="inline-block p-1 rounded-full bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 mb-6"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, ease: "backOut" }}
          >
            <h1 className="text-6xl md:text-7xl font-black bg-gradient-to-r from-white via-blue-100 to-purple-100 bg-clip-text text-transparent px-8 py-4">
              AI Tools for Every Profession
            </h1>
          </motion.div>

          <motion.p
            className="text-xl text-gray-100 max-w-4xl mx-auto mb-8 leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            Discover the most impactful AI tools transforming the top 20
            professions. Each recommendation includes pricing, features, and
            time-saving potential. ✨
          </motion.p>

          <motion.div
            className="flex justify-center gap-4 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-medium">🚀 Trending Tools</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-medium">💡 Pro Tips</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-medium">⚡ AI Powered</span>
            </div>
          </motion.div>

          {/* Statistics */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-5xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
              whileHover={{ y: -10 }}
            >
              <div className="text-4xl font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                20
              </div>
              <div className="text-sm text-gray-200 font-medium">
                Professions Covered
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
              whileHover={{ y: -10 }}
            >
              <div className="text-4xl font-black bg-gradient-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent">
                60+
              </div>
              <div className="text-sm text-gray-200 font-medium">
                AI Tools Reviewed
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
              whileHover={{ y: -10 }}
            >
              <div className="text-4xl font-black bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
                50-85%
              </div>
              <div className="text-sm text-gray-200 font-medium">
                Average Time Saved
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
              whileHover={{ y: -10 }}
            >
              <div className="text-4xl font-black bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
                $2.8T
              </div>
              <div className="text-sm text-gray-200 font-medium">
                Market Impact
              </div>
            </motion.div>
          </motion.div>
        </div>

        {/* Filters and Sort */}
        <div className="mb-12 space-y-6 relative z-10">
          <div className="flex flex-wrap gap-3 justify-center">
            <span className="text-sm font-bold text-white px-4 py-3 bg-white/10 backdrop-blur-md rounded-full border border-white/20">
              🎯 Filter by Impact:
            </span>
            {["All", "Critical", "High", "Medium"].map((category) => (
              <motion.button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`px-6 py-3 rounded-full text-sm font-bold transition-all duration-300 ${
                  filterCategory === category
                    ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-2xl scale-105"
                    : "bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/20 hover:scale-105"
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {category}
              </motion.button>
            ))}
          </div>

          <div className="flex flex-wrap gap-3 justify-center">
            <span className="text-sm font-bold text-white px-4 py-3 bg-white/10 backdrop-blur-md rounded-full border border-white/20">
              🔄 Sort by:
            </span>
            {[
              { value: "impact", label: "Impact Level", emoji: "⚡" },
              { value: "adoption", label: "AI Adoption", emoji: "📈" },
              { value: "name", label: "Name", emoji: "🔤" },
            ].map((sort) => (
              <motion.button
                key={sort.value}
                onClick={() => setSortBy(sort.value)}
                className={`px-6 py-3 rounded-full text-sm font-bold transition-all duration-300 ${
                  sortBy === sort.value
                    ? "bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-2xl scale-105"
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

        {/* Profession Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8 mb-12 relative z-10">
          {getFilteredProfessions().map((profession, index) => (
            <motion.div
              key={profession.id}
              initial={{ opacity: 0, y: 50, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{
                duration: 0.6,
                delay: index * 0.1,
                ease: "backOut",
              }}
              className="bg-white/10 backdrop-blur-xl rounded-3xl border border-white/20 overflow-hidden hover:scale-105 transition-all duration-500 cursor-pointer group shadow-2xl hover:shadow-cyan-500/25"
              onClick={() => setSelectedProfession(profession)}
              whileHover={{
                scale: 1.05,
                rotateY: 5,
                rotateX: 5,
              }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="p-8">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <motion.div
                    className="text-5xl transform group-hover:scale-110 transition-transform duration-300"
                    whileHover={{ rotate: 10, scale: 1.2 }}
                  >
                    {profession.icon}
                  </motion.div>
                  <motion.div
                    className={`px-4 py-2 rounded-full text-xs font-bold border-2 backdrop-blur-md ${
                      profession.impactLevel === "Critical"
                        ? "bg-red-500/20 border-red-400/50 text-red-200"
                        : profession.impactLevel === "High"
                          ? "bg-orange-500/20 border-orange-400/50 text-orange-200"
                          : "bg-yellow-500/20 border-yellow-400/50 text-yellow-200"
                    }`}
                    whileHover={{ scale: 1.1 }}
                  >
                    {profession.impactLevel} Impact
                  </motion.div>
                </div>

                {/* Title */}
                <h3 className="text-xl font-black text-white mb-4 group-hover:bg-gradient-to-r group-hover:from-cyan-400 group-hover:to-blue-500 group-hover:bg-clip-text group-hover:text-transparent transition-all duration-300">
                  {profession.title}
                </h3>

                {/* AI Adoption */}
                <div className="mb-6">
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm text-gray-200 font-medium">
                      🤖 AI Adoption
                    </span>
                    <span className="text-sm font-bold text-white bg-white/20 rounded-full px-3 py-1">
                      {profession.aiAdoption}%
                    </span>
                  </div>
                  <div className="w-full bg-white/20 rounded-full h-3 overflow-hidden">
                    <motion.div
                      className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 h-3 rounded-full relative"
                      initial={{ width: 0 }}
                      animate={{ width: `${profession.aiAdoption}%` }}
                      transition={{
                        duration: 1.5,
                        delay: index * 0.1 + 0.8,
                        ease: "easeOut",
                      }}
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse"></div>
                    </motion.div>
                  </div>
                </div>

                {/* Primary Tool */}
                <div className="border-t border-white/20 pt-6">
                  <div className="text-sm font-bold text-gray-200 mb-2">
                    🚀 Recommended Tool:
                  </div>
                  <div className="text-cyan-300 font-bold text-base mb-2">
                    {profession.primaryTool.name}
                  </div>
                  <div className="text-xs text-gray-300 bg-white/10 rounded-full px-3 py-1 inline-block">
                    ⚡ {profession.timesSaved}
                  </div>
                </div>

                {/* View Details Button */}
                <div className="mt-6 text-center">
                  <motion.div
                    className="text-sm text-white font-bold group-hover:text-cyan-300 transition-colors duration-300 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-full px-4 py-2 border border-cyan-400/30"
                    whileHover={{ scale: 1.05 }}
                  >
                    ✨ Click to explore tools →
                  </motion.div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Detailed Modal */}
        <AnimatePresence>
          {selectedProfession && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50"
              onClick={() => setSelectedProfession(null)}
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0, rotateY: -20 }}
                animate={{ scale: 1, opacity: 1, rotateY: 0 }}
                exit={{ scale: 0.8, opacity: 0, rotateY: 20 }}
                transition={{ type: "spring", bounce: 0.3 }}
                className="bg-gradient-to-br from-indigo-900/95 via-purple-900/95 to-pink-900/95 backdrop-blur-xl rounded-3xl max-w-6xl w-full max-h-[90vh] overflow-y-auto border border-white/20 shadow-2xl"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="p-10">
                  {/* Header */}
                  <div className="flex justify-between items-start mb-8">
                    <div className="flex items-center gap-6">
                      <motion.div
                        className="text-7xl"
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        transition={{ delay: 0.2, type: "spring", bounce: 0.6 }}
                      >
                        {selectedProfession.icon}
                      </motion.div>
                      <div>
                        <motion.h2
                          className="text-4xl font-black bg-gradient-to-r from-white via-cyan-200 to-purple-200 bg-clip-text text-transparent mb-3"
                          initial={{ opacity: 0, x: -50 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.3 }}
                        >
                          {selectedProfession.title}
                        </motion.h2>
                        <motion.div
                          className={`inline-block px-4 py-2 rounded-full text-sm font-bold border-2 backdrop-blur-md ${
                            selectedProfession.impactLevel === "Critical"
                              ? "bg-red-500/30 border-red-400/50 text-red-200"
                              : selectedProfession.impactLevel === "High"
                                ? "bg-orange-500/30 border-orange-400/50 text-orange-200"
                                : "bg-yellow-500/30 border-yellow-400/50 text-yellow-200"
                          }`}
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.4 }}
                        >
                          {selectedProfession.impactLevel} Impact •{" "}
                          {selectedProfession.aiAdoption}% Adoption
                        </motion.div>
                      </div>
                    </div>
                    <motion.button
                      onClick={() => setSelectedProfession(null)}
                      className="text-white hover:text-red-400 text-3xl font-bold bg-white/10 backdrop-blur-md rounded-full w-12 h-12 flex items-center justify-center border border-white/20 hover:bg-red-500/20 transition-all duration-300"
                      whileHover={{ scale: 1.1, rotate: 90 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      ×
                    </motion.button>
                  </div>

                  {/* Description */}
                  <motion.div
                    className="mb-10"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                  >
                    <p className="text-gray-100 text-xl leading-relaxed mb-6">
                      {selectedProfession.description}
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                        <span className="font-bold text-cyan-300 text-lg block mb-2">
                          🎯 Primary Use Case:
                        </span>
                        <p className="text-gray-200">
                          {selectedProfession.useCase}
                        </p>
                      </div>
                      <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                        <span className="font-bold text-green-300 text-lg block mb-2">
                          ⚡ Time Savings:
                        </span>
                        <p className="text-green-300 font-bold text-xl">
                          {selectedProfession.timesSaved}
                        </p>
                      </div>
                    </div>
                  </motion.div>

                  {/* Primary Tool */}
                  <motion.div
                    className="mb-10"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                  >
                    <h3 className="text-3xl font-black text-white mb-6">
                      🏆 Recommended Tool
                    </h3>
                    <div className="bg-gradient-to-r from-cyan-500/20 to-blue-600/20 backdrop-blur-xl p-8 rounded-3xl border border-white/30 shadow-2xl">
                      <div className="flex flex-col md:flex-row justify-between items-start gap-6">
                        <div className="flex-1">
                          <h4 className="text-2xl font-black text-cyan-300 mb-3">
                            {selectedProfession.primaryTool.name}
                          </h4>
                          <div className="flex flex-wrap items-center gap-3 mb-4">
                            <span className="text-sm bg-cyan-500/30 text-cyan-200 px-4 py-2 rounded-full font-bold border border-cyan-400/50">
                              {selectedProfession.primaryTool.category}
                            </span>
                            <span className="text-sm text-yellow-300 font-bold">
                              ⭐ {selectedProfession.primaryTool.rating}
                            </span>
                          </div>
                          <p className="text-gray-200 text-lg">
                            {selectedProfession.primaryTool.description}
                          </p>
                        </div>
                        <div className="text-center md:text-right">
                          <div className="text-2xl font-black text-green-400 mb-4">
                            {selectedProfession.primaryTool.pricing}
                          </div>
                          <motion.a
                            href={selectedProfession.primaryTool.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-block bg-gradient-to-r from-green-500 to-emerald-600 text-white font-bold px-8 py-4 rounded-full hover:from-green-600 hover:to-emerald-700 transition-all duration-300 shadow-xl border border-green-400/30"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                          >
                            🚀 Try Now →
                          </motion.a>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Alternative Tools */}
                  <motion.div
                    className="mb-8"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                  >
                    <h3 className="text-3xl font-black text-white mb-6">
                      🛠️ Alternative Tools
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {selectedProfession.alternativeTools.map((tool, idx) => (
                        <motion.div
                          key={idx}
                          className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 hover:bg-white/15 transition-all duration-300 group"
                          initial={{ opacity: 0, x: idx % 2 === 0 ? -20 : 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.8 + idx * 0.1 }}
                          whileHover={{ scale: 1.02, y: -5 }}
                        >
                          <div className="flex flex-col gap-4">
                            <div>
                              <h5 className="font-black text-white text-lg mb-2 group-hover:text-cyan-300 transition-colors">
                                {tool.name}
                              </h5>
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="text-xs bg-purple-500/30 text-purple-200 px-3 py-1 rounded-full font-bold border border-purple-400/50">
                                  {tool.category}
                                </span>
                                <span className="text-xs text-yellow-300 font-bold">
                                  ⭐ {tool.rating}
                                </span>
                              </div>
                            </div>
                            <p className="text-gray-200">{tool.description}</p>
                            <div className="flex justify-between items-center pt-2 border-t border-white/20">
                              <div className="text-lg font-bold text-green-400">
                                {tool.pricing}
                              </div>
                              <motion.a
                                href={tool.link}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-cyan-300 hover:text-cyan-100 font-bold transition-colors"
                                whileHover={{ scale: 1.1 }}
                              >
                                Visit →
                              </motion.a>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
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
              className="inline-block bg-gradient-to-r from-purple-500 to-pink-600 text-white font-bold px-8 py-4 rounded-full hover:from-purple-600 hover:to-pink-700 transition-all duration-300 shadow-2xl border border-white/20 backdrop-blur-md"
            >
              ← Back to Portfolio
            </Link>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
