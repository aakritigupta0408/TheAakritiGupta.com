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
}

interface PromptTechnique {
  id: string;
  name: string;
  description: string;
  example: string;
  whenToUse: string;
  icon: string;
  color: string;
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
