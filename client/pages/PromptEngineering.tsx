import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Navigation from "../components/Navigation";

interface PromptExample {
  id: string;
  title: string;
  category: string;
  description: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  badPrompt: string;
  goodPrompt: string;
  explanation: string;
  tips: string[];
  useCase: string;
  icon: string;
  resources: {
    tutorials: {
      title: string;
      url: string;
      platform: string;
      duration?: string;
    }[];
    videos: { title: string; url: string; creator: string; duration: string }[];
    tools: {
      name: string;
      url: string;
      description: string;
      pricing: string;
    }[];
    templates: {
      name: string;
      url: string;
      description: string;
      category: string;
    }[];
  };
}

interface PromptTechnique {
  id: string;
  name: string;
  description: string;
  example: string;
  whenToUse: string;
  icon: string;
  color: string;
  resources: {
    guides: { title: string; url: string; author: string; type: string }[];
    papers: { title: string; url: string; authors: string; year: string }[];
    courses: { title: string; url: string; provider: string; level: string }[];
    communities: {
      name: string;
      url: string;
      platform: string;
      members: string;
    }[];
  };
}

const promptTechniques: PromptTechnique[] = [
  {
    id: "chain-of-thought",
    name: "Chain of Thought",
    description: "Break down complex problems into step-by-step reasoning",
    example:
      "Let's think step by step: 1) First identify... 2) Then analyze... 3) Finally conclude...",
    whenToUse: "Complex reasoning, math problems, multi-step analysis",
    icon: "🔗",
    color: "from-blue-500 to-cyan-500",
    resources: {
      guides: [
        {
          title:
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
          url: "https://arxiv.org/abs/2201.11903",
          author: "Google Research",
          type: "Research Paper",
        },
        {
          title: "Complete Guide to Chain of Thought Prompting",
          url: "https://www.promptingguide.ai/techniques/cot",
          author: "Prompting Guide",
          type: "Tutorial",
        },
        {
          title: "Advanced CoT Techniques",
          url: "https://learnprompting.org/docs/intermediate/chain_of_thought",
          author: "Learn Prompting",
          type: "Course Material",
        },
      ],
      papers: [
        {
          title:
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
          url: "https://arxiv.org/abs/2201.11903",
          authors: "Wei et al. (Google)",
          year: "2022",
        },
        {
          title: "Self-Consistency Improves Chain of Thought Reasoning",
          url: "https://arxiv.org/abs/2203.11171",
          authors: "Wang et al. (Google)",
          year: "2022",
        },
        {
          title: "Large Language Models are Zero-Shot Reasoners",
          url: "https://arxiv.org/abs/2205.11916",
          authors: "Kojima et al.",
          year: "2022",
        },
      ],
      courses: [
        {
          title: "Prompt Engineering for Developers",
          url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/",
          provider: "DeepLearning.AI",
          level: "Intermediate",
        },
        {
          title: "Advanced Prompt Engineering",
          url: "https://www.coursera.org/learn/prompt-engineering",
          provider: "Coursera",
          level: "Advanced",
        },
      ],
      communities: [
        {
          name: "r/PromptEngineering",
          url: "https://reddit.com/r/PromptEngineering",
          platform: "Reddit",
          members: "50K+",
        },
        {
          name: "Prompt Engineering Discord",
          url: "https://discord.gg/promptengineering",
          platform: "Discord",
          members: "25K+",
        },
      ],
    },
  },
  {
    id: "few-shot",
    name: "Few-Shot Learning",
    description: "Provide examples to guide the AI's response format",
    example:
      "Here are examples: Input: X → Output: Y. Input: A → Output: B. Now: Input: Z → Output: ?",
    whenToUse: "Specific formatting, pattern recognition, classification tasks",
    icon: "📚",
    color: "from-green-500 to-emerald-500",
    resources: {
      guides: [
        {
          title: "Few-Shot Learning with Language Models",
          url: "https://www.promptingguide.ai/techniques/fewshot",
          author: "Prompting Guide",
          type: "Tutorial",
        },
        {
          title: "In-Context Learning and Few-Shot Examples",
          url: "https://platform.openai.com/docs/guides/prompt-engineering/strategy-provide-examples",
          author: "OpenAI",
          type: "Documentation",
        },
      ],
      papers: [
        {
          title: "Language Models are Few-Shot Learners",
          url: "https://arxiv.org/abs/2005.14165",
          authors: "Brown et al. (OpenAI)",
          year: "2020",
        },
        {
          title: "What Makes Good In-Context Examples for GPT-3?",
          url: "https://arxiv.org/abs/2101.06804",
          authors: "Liu et al.",
          year: "2021",
        },
      ],
      courses: [
        {
          title: "Few-Shot Learning Masterclass",
          url: "https://www.udemy.com/course/few-shot-learning/",
          provider: "Udemy",
          level: "Intermediate",
        },
      ],
      communities: [
        {
          name: "AI/ML Twitter Community",
          url: "https://twitter.com/search?q=%23FewShotLearning",
          platform: "Twitter",
          members: "100K+",
        },
      ],
    },
  },
  {
    id: "role-playing",
    name: "Role Playing",
    description: "Assign a specific role or persona to the AI",
    example:
      "You are an expert data scientist with 10 years of experience. Explain...",
    whenToUse: "Expert advice, specific perspectives, creative writing",
    icon: "🎭",
    color: "from-purple-500 to-pink-500",
    resources: {
      guides: [
        {
          title: "Role-Based Prompting Strategies",
          url: "https://learnprompting.org/docs/basics/roles",
          author: "Learn Prompting",
          type: "Tutorial",
        },
        {
          title: "Persona Patterns in AI",
          url: "https://www.anthropic.com/news/claude-character",
          author: "Anthropic",
          type: "Research Article",
        },
      ],
      papers: [
        {
          title: "Constitutional AI: Harmlessness from AI Feedback",
          url: "https://arxiv.org/abs/2212.08073",
          authors: "Bai et al. (Anthropic)",
          year: "2022",
        },
      ],
      courses: [
        {
          title: "Creative AI Writing",
          url: "https://www.masterclass.com/classes/ai-writing",
          provider: "MasterClass",
          level: "Beginner",
        },
      ],
      communities: [
        {
          name: "Character AI Community",
          url: "https://character.ai/",
          platform: "Character.AI",
          members: "1M+",
        },
      ],
    },
  },
  {
    id: "constraint-based",
    name: "Constraint-Based",
    description: "Set clear boundaries and limitations for the response",
    example:
      "In exactly 3 bullet points, using only technical terms, explain...",
    whenToUse: "Specific formats, word limits, style requirements",
    icon: "⚖️",
    color: "from-orange-500 to-red-500",
    resources: {
      guides: [
        {
          title: "Constraint-Based Prompt Design",
          url: "https://www.promptingguide.ai/techniques/constraints",
          author: "Prompting Guide",
          type: "Tutorial",
        },
      ],
      papers: [
        {
          title: "Controllable Text Generation",
          url: "https://arxiv.org/abs/1909.05858",
          authors: "Keskar et al.",
          year: "2019",
        },
      ],
      courses: [
        {
          title: "Structured Output Generation",
          url: "https://www.edx.org/course/structured-ai",
          provider: "edX",
          level: "Advanced",
        },
      ],
      communities: [
        {
          name: "Structured AI Generation",
          url: "https://discord.gg/structuredai",
          platform: "Discord",
          members: "5K+",
        },
      ],
    },
  },
  {
    id: "iterative-refinement",
    name: "Iterative Refinement",
    description: "Build on previous responses to improve quality",
    example:
      "Based on your previous answer, can you expand on point 2 and add more specific examples?",
    whenToUse: "Complex projects, detailed analysis, creative development",
    icon: "🔄",
    color: "from-indigo-500 to-purple-500",
    resources: {
      guides: [
        {
          title: "Iterative Prompt Refinement",
          url: "https://learnprompting.org/docs/intermediate/iterative_prompting",
          author: "Learn Prompting",
          type: "Tutorial",
        },
      ],
      papers: [
        {
          title: "Constitutional AI: Harmlessness from AI Feedback",
          url: "https://arxiv.org/abs/2212.08073",
          authors: "Bai et al.",
          year: "2022",
        },
      ],
      courses: [
        {
          title: "Advanced Conversation Design",
          url: "https://www.coursera.org/learn/conversation-design",
          provider: "Coursera",
          level: "Advanced",
        },
      ],
      communities: [
        {
          name: "Prompt Optimization",
          url: "https://reddit.com/r/PromptOptimization",
          platform: "Reddit",
          members: "15K+",
        },
      ],
    },
  },
  {
    id: "metacognitive",
    name: "Metacognitive Prompting",
    description: "Ask the AI to explain its thinking process",
    example:
      "Explain your reasoning process and confidence level for each step of your answer.",
    whenToUse:
      "Quality assurance, understanding AI logic, educational purposes",
    icon: "🧠",
    color: "from-teal-500 to-cyan-500",
    resources: {
      guides: [
        {
          title: "Metacognitive Prompting Techniques",
          url: "https://www.promptingguide.ai/techniques/metacognitive",
          author: "Prompting Guide",
          type: "Tutorial",
        },
        {
          title: "AI Transparency and Explainability",
          url: "https://www.anthropic.com/news/measuring-model-persuasiveness",
          author: "Anthropic",
          type: "Research",
        },
      ],
      papers: [
        {
          title: "Teaching Models to Express Their Uncertainty",
          url: "https://arxiv.org/abs/2205.14334",
          authors: "Lin et al.",
          year: "2022",
        },
        {
          title: "Self-Refine: Iterative Refinement with Self-Feedback",
          url: "https://arxiv.org/abs/2303.17651",
          authors: "Madaan et al.",
          year: "2023",
        },
      ],
      courses: [
        {
          title: "AI Interpretability",
          url: "https://www.fast.ai/posts/2019-08-27-interpretability.html",
          provider: "Fast.ai",
          level: "Advanced",
        },
      ],
      communities: [
        {
          name: "AI Safety & Alignment",
          url: "https://www.lesswrong.com/tag/ai-alignment",
          platform: "LessWrong",
          members: "20K+",
        },
      ],
    },
  },
];

const promptExamples: PromptExample[] = [
  {
    id: "code-review",
    title: "Code Review Assistant",
    category: "Programming",
    description: "Get comprehensive code reviews with actionable feedback",
    difficulty: "Intermediate",
    badPrompt: "Review this code",
    goodPrompt:
      "As a senior software engineer, please review this Python function for: 1) Code quality and readability, 2) Performance optimizations, 3) Security vulnerabilities, 4) Best practices violations. Provide specific line-by-line feedback with severity levels (critical/warning/suggestion) and example fixes:\n\n```python\n[YOUR CODE HERE]\n```",
    explanation:
      "The good prompt specifies the reviewer's role, defines review criteria, requests structured feedback with severity levels, and asks for actionable solutions.",
    tips: [
      "Define the reviewer's expertise level",
      "Specify what aspects to focus on",
      "Request structured output format",
      "Ask for specific examples and fixes",
    ],
    useCase:
      "Code quality improvement, learning best practices, preparing for code reviews",
    icon: "👨‍💻",
    resources: {
      tutorials: [
        {
          title: "AI-Powered Code Review Best Practices",
          url: "https://github.blog/2023-06-20-how-to-write-better-prompts-for-github-copilot/",
          platform: "GitHub",
          duration: "15 min",
        },
        {
          title: "Effective Code Review Prompts",
          url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/",
          platform: "DeepLearning.AI",
          duration: "1 hour",
        },
        {
          title: "Code Quality with AI Assistants",
          url: "https://docs.github.com/en/copilot/using-github-copilot/getting-started-with-github-copilot",
          platform: "GitHub Docs",
          duration: "20 min",
        },
      ],
      videos: [
        {
          title: "Using AI for Code Reviews - Complete Guide",
          url: "https://www.youtube.com/watch?v=8c5NQbJNQkE",
          creator: "GitHub",
          duration: "28 min",
        },
        {
          title: "ChatGPT for Developers: Code Review Automation",
          url: "https://www.youtube.com/watch?v=jkrNMKz9pWU",
          creator: "Programming with Mosh",
          duration: "35 min",
        },
        {
          title: "AI-Assisted Code Quality Improvement",
          url: "https://www.youtube.com/watch?v=v8dDj28x2nE",
          creator: "Tech With Tim",
          duration: "42 min",
        },
      ],
      tools: [
        {
          name: "GitHub Copilot",
          url: "https://github.com/features/copilot",
          description: "AI pair programmer for code suggestions and reviews",
          pricing: "$10/month",
        },
        {
          name: "CodeGPT",
          url: "https://marketplace.visualstudio.com/items?itemName=DanielSanMedium.dscodegpt",
          description: "VS Code extension for AI-powered code assistance",
          pricing: "Free",
        },
        {
          name: "Tabnine",
          url: "https://www.tabnine.com/",
          description: "AI code completion and review assistant",
          pricing: "Free/Pro plans available",
        },
      ],
      templates: [
        {
          name: "Code Review Prompt Template",
          url: "https://github.com/f/awesome-chatgpt-prompts#act-as-a-code-reviewer",
          description: "Structured template for comprehensive code reviews",
          category: "Programming",
        },
        {
          name: "Security Review Prompts",
          url: "https://owasp.org/www-project-code-review-guide/",
          description: "Security-focused code review guidelines",
          category: "Security",
        },
      ],
    },
  },
  {
    id: "data-analysis",
    title: "Data Analysis Expert",
    category: "Data Science",
    description: "Generate insights and recommendations from complex datasets",
    difficulty: "Advanced",
    badPrompt: "Analyze this data",
    goodPrompt:
      "As a senior data scientist, analyze this dataset and provide: 1) Key statistical insights with confidence intervals, 2) Trend analysis with seasonality detection, 3) Anomaly identification with severity scores, 4) Actionable business recommendations with expected impact. Use statistical significance testing and explain your methodology:\n\n[DATA DESCRIPTION]\nDataset: Sales data (2020-2024)\nColumns: date, product_id, sales_amount, region, customer_segment\nSize: 1M+ records",
    explanation:
      "This prompt establishes expertise, defines analysis scope, requests statistical rigor, and connects insights to business value.",
    tips: [
      "Specify statistical requirements",
      "Request methodology explanation",
      "Connect insights to business impact",
      "Ask for confidence levels and significance testing",
    ],
    useCase:
      "Business intelligence, research analysis, data-driven decision making",
    icon: "📊",
    resources: {
      tutorials: [
        {
          title: "AI for Data Analysis: Complete Guide",
          url: "https://www.kaggle.com/learn/intro-to-machine-learning",
          platform: "Kaggle",
          duration: "4 hours",
        },
        {
          title: "Prompt Engineering for Data Science",
          url: "https://www.coursera.org/learn/prompt-engineering-for-data-science",
          platform: "Coursera",
          duration: "6 weeks",
        },
        {
          title: "Statistical Analysis with AI Assistants",
          url: "https://www.datacamp.com/courses/statistical-thinking-in-python-part-1",
          platform: "DataCamp",
          duration: "4 hours",
        },
      ],
      videos: [
        {
          title: "Data Analysis with ChatGPT and Python",
          url: "https://www.youtube.com/watch?v=vmEHCJofslg",
          creator: "freeCodeCamp",
          duration: "2 hr 15 min",
        },
        {
          title: "AI-Powered Business Intelligence",
          url: "https://www.youtube.com/watch?v=Ksv9GxNKFT4",
          creator: "Microsoft Power BI",
          duration: "45 min",
        },
        {
          title: "Advanced Data Science Prompts",
          url: "https://www.youtube.com/watch?v=wd7TZ4w1mSw",
          creator: "Weights & Biases",
          duration: "1 hr 8 min",
        },
      ],
      tools: [
        {
          name: "Pandas AI",
          url: "https://github.com/gventuri/pandas-ai",
          description: "AI-powered data analysis with natural language queries",
          pricing: "Open Source",
        },
        {
          name: "Julius AI",
          url: "https://julius.ai/",
          description: "AI data analyst for instant insights",
          pricing: "Free tier available",
        },
        {
          name: "DataGPT",
          url: "https://www.datagpt.com/",
          description: "Conversational data analytics platform",
          pricing: "Contact for pricing",
        },
      ],
      templates: [
        {
          name: "Data Analysis Prompt Library",
          url: "https://github.com/awesome-data-science/awesome-data-science",
          description: "Comprehensive data science prompt collection",
          category: "Data Science",
        },
        {
          name: "Statistical Analysis Templates",
          url: "https://www.statsmodels.org/stable/examples/index.html",
          description: "Statistical modeling examples and prompts",
          category: "Statistics",
        },
      ],
    },
  },
  {
    id: "creative-writing",
    title: "Creative Writing Coach",
    category: "Writing",
    description:
      "Develop compelling narratives with rich character development",
    difficulty: "Intermediate",
    badPrompt: "Write a story",
    goodPrompt:
      "As a creative writing mentor, help me develop a short story with these elements: 1) Genre: Sci-fi thriller, 2) Setting: Mars colony in 2157, 3) Protagonist: Reluctant engineer with trust issues, 4) Central conflict: Sabotage threatens colony survival. \n\nProvide: Character backstory, plot outline with 3 acts, key dialogue samples, and writing techniques to build suspense. Focus on showing vs telling and create authentic character voices.",
    explanation:
      "The prompt defines genre, setting, character, and conflict while requesting specific writing craft elements and techniques.",
    tips: [
      "Specify genre and setting clearly",
      "Define character traits and motivations",
      "Outline the central conflict",
      "Request specific writing techniques",
    ],
    useCase:
      "Creative writing, storytelling, narrative development, character creation",
    icon: "✍️",
    resources: {
      tutorials: [
        {
          title: "AI-Assisted Creative Writing",
          url: "https://www.masterclass.com/classes/ai-writing",
          platform: "MasterClass",
          duration: "3 hours",
        },
        {
          title: "Story Development with AI",
          url: "https://learnprompting.org/docs/applications/creative_writing",
          platform: "Learn Prompting",
          duration: "45 min",
        },
      ],
      videos: [
        {
          title: "Creative Writing with ChatGPT",
          url: "https://www.youtube.com/watch?v=1OW8mVbARq0",
          creator: "Writer's Digest",
          duration: "32 min",
        },
      ],
      tools: [
        {
          name: "Sudowrite",
          url: "https://www.sudowrite.com/",
          description: "AI writing partner for creative fiction",
          pricing: "$10-25/month",
        },
        {
          name: "NovelAI",
          url: "https://novelai.net/",
          description: "AI storytelling and creative writing assistant",
          pricing: "$10-25/month",
        },
      ],
      templates: [
        {
          name: "Creative Writing Prompts",
          url: "https://github.com/f/awesome-chatgpt-prompts#act-as-a-novelist",
          description: "Collection of creative writing prompt templates",
          category: "Writing",
        },
      ],
    },
  },
  {
    id: "marketing-strategy",
    title: "Marketing Strategy Advisor",
    category: "Business",
    description:
      "Create comprehensive marketing campaigns with measurable outcomes",
    difficulty: "Advanced",
    badPrompt: "Help with marketing",
    goodPrompt:
      "As a marketing strategist with expertise in B2B SaaS, develop a go-to-market strategy for our new AI analytics platform targeting mid-market companies (100-1000 employees). Include: 1) Customer persona profiles with pain points, 2) Value proposition canvas, 3) Multi-channel campaign strategy, 4) Content marketing roadmap, 5) KPI framework with benchmarks. \n\nBudget: $50K/month, Timeline: 6 months, Target: 100 qualified leads/month.",
    explanation:
      "This prompt establishes domain expertise, defines target market, specifies deliverables, and provides constraints for realistic recommendations.",
    tips: [
      "Define target market precisely",
      "Specify budget and timeline constraints",
      "Request measurable outcomes",
      "Include competitive landscape context",
    ],
    useCase:
      "Product launches, marketing campaigns, business strategy, competitive analysis",
    icon: "📈",
    resources: {
      tutorials: [
        {
          title: "AI Marketing Strategy Development",
          url: "https://www.hubspot.com/artificial-intelligence-marketing",
          platform: "HubSpot Academy",
          duration: "2 hours",
        },
        {
          title: "Growth Hacking with AI",
          url: "https://growthhackers.com/articles/ai-growth-hacking",
          platform: "Growth Hackers",
          duration: "1 hour",
        },
      ],
      videos: [
        {
          title: "AI for Marketing Strategy",
          url: "https://www.youtube.com/watch?v=qGhyKNl5Cs8",
          creator: "Neil Patel",
          duration: "34 min",
        },
      ],
      tools: [
        {
          name: "Copy.ai",
          url: "https://www.copy.ai/",
          description: "AI-powered marketing copy generation",
          pricing: "Free tier, $35+/month",
        },
        {
          name: "Jasper",
          url: "https://www.jasper.ai/",
          description: "AI marketing content creation platform",
          pricing: "$39+/month",
        },
      ],
      templates: [
        {
          name: "Marketing Strategy Templates",
          url: "https://blog.hubspot.com/marketing/marketing-plan-template-generator",
          description: "Comprehensive marketing strategy frameworks",
          category: "Marketing",
        },
      ],
    },
  },
  {
    id: "learning-tutor",
    title: "Personalized Learning Tutor",
    category: "Education",
    description: "Create adaptive learning experiences based on student needs",
    difficulty: "Beginner",
    badPrompt: "Teach me math",
    goodPrompt:
      "As an experienced math tutor, help me master calculus derivatives using these approaches: 1) Start with my current level (completed algebra, struggling with limits), 2) Use visual explanations and real-world examples, 3) Provide practice problems with step-by-step solutions, 4) Check my understanding with questions, 5) Adapt explanations based on my responses.\n\nMy learning style: Visual learner, prefer hands-on examples, need concepts broken into small steps.",
    explanation:
      "The prompt establishes current knowledge level, specifies learning preferences, requests adaptive teaching methods, and emphasizes understanding verification.",
    tips: [
      "Specify current knowledge level",
      "Define learning style preferences",
      "Request interactive engagement",
      "Ask for understanding verification",
    ],
    useCase:
      "Online education, skill development, concept explanation, personalized learning",
    icon: "🎓",
    resources: {
      tutorials: [
        {
          title: "AI Tutoring Systems",
          url: "https://www.edx.org/course/artificial-intelligence-in-education",
          platform: "edX",
          duration: "6 weeks",
        },
        {
          title: "Personalized Learning with AI",
          url: "https://www.coursera.org/learn/ai-for-education",
          platform: "Coursera",
          duration: "4 weeks",
        },
      ],
      videos: [
        {
          title: "Creating AI Tutors",
          url: "https://www.youtube.com/watch?v=education-ai",
          creator: "Khan Academy",
          duration: "25 min",
        },
      ],
      tools: [
        {
          name: "Socratic by Google",
          url: "https://socratic.org/",
          description: "AI-powered homework help and tutoring",
          pricing: "Free",
        },
        {
          name: "Tutorly",
          url: "https://www.tutorly.ai/",
          description: "AI tutoring platform for personalized learning",
          pricing: "$15-30/month",
        },
      ],
      templates: [
        {
          name: "Educational Prompt Templates",
          url: "https://github.com/f/awesome-chatgpt-prompts#act-as-a-math-teacher",
          description: "Teaching and tutoring prompt collection",
          category: "Education",
        },
      ],
    },
  },
  {
    id: "debugging-assistant",
    title: "Debug Detective",
    category: "Programming",
    description: "Systematically identify and fix complex software bugs",
    difficulty: "Advanced",
    badPrompt: "Fix this bug",
    goodPrompt:
      "As a debugging expert, help me resolve this production issue using systematic debugging methodology: 1) Analyze error logs and stack traces, 2) Identify potential root causes with likelihood scores, 3) Suggest debugging steps in priority order, 4) Provide code fixes with explanations, 5) Recommend prevention strategies.\n\nError: 'Memory leak in user session management'\nEnvironment: Node.js + Redis, 10K concurrent users\nSymptoms: Memory usage grows 50MB/hour, crashes after 6 hours\nRecent changes: Session timeout logic updated last week",
    explanation:
      "This prompt establishes systematic approach, provides context about the environment and symptoms, and requests prioritized solutions with prevention.",
    tips: [
      "Provide complete error context",
      "Describe environment and scale",
      "List recent changes that might be related",
      "Request systematic debugging approach",
    ],
    useCase:
      "Production debugging, performance optimization, system troubleshooting",
    icon: "🐛",
    resources: {
      tutorials: [
        {
          title: "AI-Assisted Debugging Techniques",
          url: "https://github.blog/2023-06-20-how-to-write-better-prompts-for-github-copilot/",
          platform: "GitHub",
          duration: "20 min",
        },
        {
          title: "Systematic Debugging with AI",
          url: "https://www.pluralsight.com/courses/debugging-applications-ai",
          platform: "Pluralsight",
          duration: "3 hours",
        },
      ],
      videos: [
        {
          title: "Debugging with AI Tools",
          url: "https://www.youtube.com/watch?v=debugging-ai",
          creator: "Microsoft Developer",
          duration: "38 min",
        },
      ],
      tools: [
        {
          name: "GitHub Copilot",
          url: "https://github.com/features/copilot",
          description: "AI-powered debugging assistance",
          pricing: "$10/month",
        },
        {
          name: "Sourcegraph Cody",
          url: "https://sourcegraph.com/cody",
          description: "AI coding assistant for debugging",
          pricing: "Free tier available",
        },
      ],
      templates: [
        {
          name: "Debugging Prompt Templates",
          url: "https://github.com/debugging-prompts/collection",
          description: "Systematic debugging prompt library",
          category: "Programming",
        },
      ],
    },
  },
];

export default function PromptEngineering() {
  const [selectedExample, setSelectedExample] = useState<PromptExample | null>(
    null,
  );
  const [selectedTechnique, setSelectedTechnique] =
    useState<PromptTechnique | null>(null);
  const [activeTab, setActiveTab] = useState<
    "examples" | "techniques" | "playground"
  >("examples");
  const [userPrompt, setUserPrompt] = useState("");
  const [improvedPrompt, setImprovedPrompt] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyzePrompt = () => {
    if (!userPrompt.trim()) return;

    setIsAnalyzing(true);

    // Simulate AI analysis
    setTimeout(() => {
      const improvements = [
        "Add specific role definition (e.g., 'As an expert in...')",
        "Include clear output format requirements",
        "Specify constraints and parameters",
        "Add context and background information",
        "Request step-by-step reasoning",
        "Include examples or patterns to follow",
      ];

      const randomImprovements = improvements.slice(
        0,
        3 + Math.floor(Math.random() * 3),
      );

      setImprovedPrompt(`**Improved Prompt:**

As a [DEFINE ROLE/EXPERTISE], ${userPrompt.toLowerCase()}

Please provide:
${randomImprovements.map((imp, idx) => `${idx + 1}. ${imp}`).join("\n")}

**Analysis:**
- Added role specification for expert context
- Structured output requirements for clarity
- Enhanced with specific constraints
- Improved formatting for better results`);

      setIsAnalyzing(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-indigo-900 relative overflow-hidden">
      {/* Enhanced Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,_rgba(120,119,198,0.3),_transparent_50%),radial-gradient(circle_at_80%_20%,_rgba(255,119,198,0.3),_transparent_50%),radial-gradient(circle_at_40%_40%,_rgba(120,200,255,0.3),_transparent_50%)]"></div>

        {/* Floating prompt bubbles */}
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-32 h-32 bg-white/5 rounded-full blur-xl"
            animate={{
              x: [0, 100, 0],
              y: [0, -100, 0],
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.6, 0.3],
            }}
            transition={{
              duration: 10 + i * 2,
              repeat: Infinity,
              delay: i * 1.5,
            }}
            style={{
              left: `${20 + i * 15}%`,
              top: `${10 + i * 10}%`,
            }}
          />
        ))}
      </div>

      <Navigation />

      <div className="container mx-auto px-6 py-24 relative z-10">
        {/* Header */}
        <div className="text-center mb-16">
          <motion.div
            className="inline-block p-1 rounded-full bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 mb-8"
            initial={{ opacity: 0, scale: 0.5, rotateY: -180 }}
            animate={{ opacity: 1, scale: 1, rotateY: 0 }}
            transition={{ duration: 1.2, ease: "backOut" }}
          >
            <h1 className="text-5xl md:text-7xl font-black bg-gradient-to-r from-white via-cyan-100 to-purple-100 bg-clip-text text-transparent px-8 py-6">
              Prompt Engineering Mastery
            </h1>
          </motion.div>

          <motion.p
            className="text-xl text-gray-100 max-w-4xl mx-auto mb-12 leading-relaxed"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            🚀 Master the art of communicating with AI! Learn advanced
            techniques, explore real-world examples, and practice with
            interactive demos to unlock the full potential of AI systems. ✨
          </motion.p>

          {/* Navigation Tabs */}
          <motion.div
            className="flex justify-center gap-2 mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            {[
              {
                id: "examples",
                label: "📚 Examples",
                desc: "Real-world prompts",
              },
              {
                id: "techniques",
                label: "🧠 Techniques",
                desc: "Core methods",
              },
              {
                id: "playground",
                label: "🛠️ Playground",
                desc: "Practice & improve",
              },
            ].map((tab) => (
              <motion.button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-6 py-4 rounded-2xl font-bold text-sm transition-all duration-300 ${
                  activeTab === tab.id
                    ? "bg-gradient-to-r from-cyan-500 to-purple-600 text-white shadow-2xl scale-105"
                    : "bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/20"
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div>{tab.label}</div>
                <div className="text-xs opacity-80">{tab.desc}</div>
              </motion.button>
            ))}
          </motion.div>
        </div>

        {/* Examples Section */}
        {activeTab === "examples" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
              {promptExamples.map((example, index) => (
                <motion.div
                  key={example.id}
                  className="bg-white/10 backdrop-blur-xl rounded-3xl p-6 border border-white/20 hover:bg-white/15 transition-all duration-300 cursor-pointer group"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.02, y: -5 }}
                  onClick={() => setSelectedExample(example)}
                >
                  <div className="flex items-start gap-4 mb-4">
                    <span className="text-4xl">{example.icon}</span>
                    <div>
                      <h3 className="text-xl font-bold text-white group-hover:text-cyan-300 transition-colors">
                        {example.title}
                      </h3>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-xs bg-purple-500/30 text-purple-200 px-2 py-1 rounded-full">
                          {example.category}
                        </span>
                        <span
                          className={`text-xs px-2 py-1 rounded-full ${
                            example.difficulty === "Beginner"
                              ? "bg-green-500/30 text-green-200"
                              : example.difficulty === "Intermediate"
                                ? "bg-yellow-500/30 text-yellow-200"
                                : "bg-red-500/30 text-red-200"
                          }`}
                        >
                          {example.difficulty}
                        </span>
                      </div>
                    </div>
                  </div>

                  <p className="text-gray-200 text-sm leading-relaxed">
                    {example.description}
                  </p>

                  <div className="mt-4 text-cyan-300 text-sm font-bold group-hover:text-cyan-200 transition-colors">
                    Click to explore →
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Techniques Section */}
        {activeTab === "techniques" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
              {promptTechniques.map((technique, index) => (
                <motion.div
                  key={technique.id}
                  className="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 cursor-pointer group"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.02, y: -5 }}
                  onClick={() => setSelectedTechnique(technique)}
                >
                  <div className="flex items-start gap-6">
                    <div
                      className={`w-16 h-16 bg-gradient-to-r ${technique.color} rounded-2xl flex items-center justify-center text-2xl shadow-lg`}
                    >
                      {technique.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-2xl font-bold text-white group-hover:text-cyan-300 transition-colors mb-3">
                        {technique.name}
                      </h3>
                      <p className="text-gray-200 leading-relaxed mb-4">
                        {technique.description}
                      </p>
                      <div className="text-sm text-cyan-300 font-medium">
                        Best for: {technique.whenToUse}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Playground Section */}
        {activeTab === "playground" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="max-w-4xl mx-auto">
              <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
                <h2 className="text-3xl font-bold text-white mb-6 text-center">
                  🛠️ Prompt Improvement Workshop
                </h2>

                <div className="space-y-6">
                  <div>
                    <label className="block text-white font-bold mb-3">
                      Enter your prompt for analysis:
                    </label>
                    <textarea
                      value={userPrompt}
                      onChange={(e) => setUserPrompt(e.target.value)}
                      placeholder="e.g., 'Help me write a business plan' or 'Explain machine learning'"
                      className="w-full h-32 bg-white/10 border border-white/30 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 resize-none"
                    />
                  </div>

                  <motion.button
                    onClick={analyzePrompt}
                    disabled={!userPrompt.trim() || isAnalyzing}
                    className="w-full bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-bold py-4 rounded-xl hover:from-cyan-600 hover:to-purple-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {isAnalyzing ? (
                      <div className="flex items-center justify-center gap-3">
                        <motion.div
                          className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                          animate={{ rotate: 360 }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            ease: "linear",
                          }}
                        />
                        Analyzing your prompt...
                      </div>
                    ) : (
                      "🔍 Analyze & Improve Prompt"
                    )}
                  </motion.button>

                  {improvedPrompt && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-black/30 rounded-xl p-6 border border-green-400/30"
                    >
                      <h3 className="text-xl font-bold text-green-300 mb-4">
                        ✨ Improved Version
                      </h3>
                      <pre className="text-gray-200 whitespace-pre-wrap font-mono text-sm leading-relaxed">
                        {improvedPrompt}
                      </pre>
                    </motion.div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Example Detail Modal */}
        <AnimatePresence>
          {selectedExample && (
            <motion.div
              className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedExample(null)}
            >
              <motion.div
                className="bg-gradient-to-br from-slate-900/95 via-purple-900/95 to-indigo-900/95 backdrop-blur-xl rounded-3xl max-w-6xl w-full max-h-[90vh] overflow-y-auto border border-white/20 p-8"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-start mb-8">
                  <div className="flex items-center gap-4">
                    <span className="text-6xl">{selectedExample.icon}</span>
                    <div>
                      <h2 className="text-3xl font-black text-white">
                        {selectedExample.title}
                      </h2>
                      <p className="text-gray-200">
                        {selectedExample.description}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedExample(null)}
                    className="text-white hover:text-red-400 text-2xl"
                  >
                    ×
                  </button>
                </div>

                <div className="space-y-8">
                  {/* Bad vs Good Prompt Comparison */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-red-500/10 border border-red-400/30 rounded-xl p-6">
                      <h3 className="text-xl font-bold text-red-300 mb-4">
                        ❌ Weak Prompt
                      </h3>
                      <div className="bg-black/30 rounded-lg p-4 font-mono text-sm text-gray-300">
                        {selectedExample.badPrompt}
                      </div>
                    </div>

                    <div className="bg-green-500/10 border border-green-400/30 rounded-xl p-6">
                      <h3 className="text-xl font-bold text-green-300 mb-4">
                        ✅ Strong Prompt
                      </h3>
                      <div className="bg-black/30 rounded-lg p-4 font-mono text-sm text-gray-300 whitespace-pre-wrap">
                        {selectedExample.goodPrompt}
                      </div>
                    </div>
                  </div>

                  {/* Explanation */}
                  <div className="bg-white/10 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-cyan-300 mb-4">
                      💡 Why This Works Better
                    </h3>
                    <p className="text-gray-200 leading-relaxed">
                      {selectedExample.explanation}
                    </p>
                  </div>

                  {/* Tips */}
                  <div className="bg-white/10 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-purple-300 mb-4">
                      🎯 Key Tips
                    </h3>
                    <ul className="space-y-2">
                      {selectedExample.tips.map((tip, idx) => (
                        <li
                          key={idx}
                          className="flex items-start gap-3 text-gray-200"
                        >
                          <span className="text-purple-400 mt-1">•</span>
                          <span>{tip}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Use Case */}
                  <div className="bg-blue-500/10 border border-blue-400/30 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-blue-300 mb-4">
                      🚀 Best Used For
                    </h3>
                    <p className="text-gray-200">{selectedExample.useCase}</p>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Technique Detail Modal */}
        <AnimatePresence>
          {selectedTechnique && (
            <motion.div
              className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedTechnique(null)}
            >
              <motion.div
                className="bg-gradient-to-br from-slate-900/95 via-purple-900/95 to-indigo-900/95 backdrop-blur-xl rounded-3xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-white/20 p-8"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-start mb-8">
                  <div className="flex items-center gap-6">
                    <div
                      className={`w-20 h-20 bg-gradient-to-r ${selectedTechnique.color} rounded-2xl flex items-center justify-center text-3xl shadow-lg`}
                    >
                      {selectedTechnique.icon}
                    </div>
                    <div>
                      <h2 className="text-3xl font-black text-white">
                        {selectedTechnique.name}
                      </h2>
                      <p className="text-gray-200 text-lg">
                        {selectedTechnique.description}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedTechnique(null)}
                    className="text-white hover:text-red-400 text-2xl"
                  >
                    ×
                  </button>
                </div>

                <div className="space-y-6">
                  <div className="bg-white/10 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-cyan-300 mb-4">
                      📝 Example Usage
                    </h3>
                    <div className="bg-black/30 rounded-lg p-4 font-mono text-sm text-gray-300">
                      {selectedTechnique.example}
                    </div>
                  </div>

                  <div className="bg-white/10 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-green-300 mb-4">
                      🎯 When to Use
                    </h3>
                    <p className="text-gray-200 leading-relaxed">
                      {selectedTechnique.whenToUse}
                    </p>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
