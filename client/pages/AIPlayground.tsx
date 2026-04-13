import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import ChatBot from "@/components/ChatBot";
import SubpageLayout from "@/components/SubpageLayout";
import { getPageRefreshContent } from "@/data/siteRefreshContent";
import { latestAIProductLaunches } from "../data/aiSignals";

type DemoType =
  | "text-generator"
  | "image-prompt"
  | "code-helper"
  | "creative-writer"
  | "data-analyst"
  | "translator"
  | "summarizer"
  | "poem-generator";

interface AIDemo {
  id: DemoType;
  title: string;
  subtitle: string;
  icon: string;
  color: string;
  description: string;
  placeholder: string;
  examples: string[];
  stateOfArt: {
    name: string;
    url: string;
    description: string;
    company: string;
    released: string;
    whyItMatters: string;
  };
}

const FEATURED_SHOWCASES = [
  {
    id: "trade-recommendation-system",
    badge: "Featured demo",
    title: "AI Trade Recommendation System",
    summary:
      "Replay the production trading loop in daily-only mode with deterministic recommendations, request-budget awareness, and paper execution.",
    tags: ["Local-first ingest", "Daily-only forecasts", "Paper trading only"],
    route: "/ai-playground/trade-recommendation-system",
    meta: "Loop, budget, decisions, EOD",
    accent:
      "from-emerald-500/14 via-cyan-500/10 to-amber-300/10 border-emerald-300/20",
    badgeClass:
      "border-emerald-300/30 bg-emerald-400/10 text-emerald-100",
    buttonClass:
      "from-emerald-400 via-cyan-400 to-amber-300 text-slate-950 shadow-[0_18px_40px_rgba(52,211,153,0.24)]",
  },
  {
    id: "vedic-astro-ai",
    badge: "Featured demo",
    title: "Vedic Astrology AI System",
    summary:
      "Explore a BPHS-grounded multi-agent system that assembles natal, Dasha, transit, and divisional chart readings with critic-reviser loops.",
    tags: ["Multi-agent pipeline", "BPHS rules engine", "Critic-reviser loop"],
    route: "/ai-playground/vedic-astro-ai",
    meta: "Chart, Dasha, reading, calibrate",
    accent:
      "from-violet-500/14 via-rose-500/10 to-amber-300/10 border-violet-300/20",
    badgeClass:
      "border-violet-300/30 bg-violet-400/10 text-violet-100",
    buttonClass:
      "from-violet-400 via-rose-400 to-amber-300 text-slate-950 shadow-[0_18px_40px_rgba(139,92,246,0.24)]",
  },
] as const;

const AI_DEMOS: AIDemo[] = [
  {
    id: "text-generator",
    title: "STORY GENERATOR",
    subtitle: "Creative Narratives",
    icon: "📖",
    color: "from-blue-500 to-cyan-500",
    description: "Generate engaging stories from simple prompts",
    placeholder: "Enter a story prompt like 'A robot discovers emotions'",
    examples: [
      "A time traveler gets stuck in the Renaissance",
      "An AI chef opens a restaurant",
      "Cats secretly run the internet",
    ],
    stateOfArt: {
      name: "Claude Opus 4.6",
      url: "https://www.anthropic.com/news/claude-opus-4-6",
      description:
        "A frontier model built for long-form reasoning, deep research, and stronger professional writing tasks.",
      company: "Anthropic",
      released: "February 5, 2026",
      whyItMatters:
        "Storytelling tools now benefit from models that can hold longer context and sustain higher-quality narrative structure.",
    },
  },
  {
    id: "image-prompt",
    title: "IMAGE PROMPT CREATOR",
    subtitle: "Visual Imagination",
    icon: "🎨",
    color: "from-purple-500 to-pink-500",
    description: "Transform ideas into detailed image generation prompts",
    placeholder: "Describe an image you want to create",
    examples: [
      "A cyberpunk cityscape at sunset",
      "A magical library with floating books",
      "A steampunk airship flying through clouds",
    ],
    stateOfArt: {
      name: "gpt-image-1",
      url: "https://openai.com/index/image-generation-api/",
      description:
        "OpenAI's latest image model focuses on controllability, visual quality, and strong text rendering inside generated images.",
      company: "OpenAI",
      released: "April 23, 2025",
      whyItMatters:
        "Image generation is shifting from novelty outputs to production-ready creative workflows and design tooling.",
    },
  },
  {
    id: "code-helper",
    title: "CODE GENERATOR",
    subtitle: "Programming Assistant",
    icon: "💻",
    color: "from-green-500 to-emerald-500",
    description: "Generate code snippets and programming solutions",
    placeholder: "Describe what you want to code",
    examples: [
      "A function to sort an array by date",
      "React component for a search bar",
      "Python script to analyze CSV data",
    ],
    stateOfArt: {
      name: "Codex with GPT-5.3-Codex",
      url: "https://openai.com/index/introducing-gpt-5-3-codex/",
      description:
        "A more interactive Codex experience built around real-time steering, parallel work, and longer-running engineering tasks.",
      company: "OpenAI",
      released: "2026",
      whyItMatters:
        "Code generation has expanded into full agent supervision, repo-wide changes, and longer-running engineering workflows.",
    },
  },
  {
    id: "creative-writer",
    title: "CREATIVE WRITER",
    subtitle: "Literary Magic",
    icon: "✍️",
    color: "from-orange-500 to-red-500",
    description: "Write poems, lyrics, and creative content",
    placeholder: "What would you like me to write?",
    examples: [
      "A haiku about technology",
      "Song lyrics about friendship",
      "A motivational speech about AI",
    ],
    stateOfArt: {
      name: "Gemini 2.5 Pro",
      url: "https://blog.google/products-and-platforms/products/gemini/gemini-2-5-pro-latest-preview/",
      description:
        "Google's latest Gemini 2.5 Pro preview emphasizes stronger reasoning, coding, and multimodal writing performance.",
      company: "Google",
      released: "2026 preview",
      whyItMatters:
        "Creative workflows increasingly blend text, images, audio, and web context instead of staying purely language-based.",
    },
  },
  {
    id: "data-analyst",
    title: "DATA INSIGHTS",
    subtitle: "Smart Analysis",
    icon: "📊",
    color: "from-indigo-500 to-blue-500",
    description: "Analyze trends and provide data insights",
    placeholder: "Describe your data or ask for analysis help",
    examples: [
      "Explain machine learning metrics",
      "Sales data analysis techniques",
      "Customer behavior patterns",
    ],
    stateOfArt: {
      name: "OpenAI deep research",
      url: "https://openai.com/index/introducing-deep-research/",
      description:
        "A research workflow that plans tasks, browses sources, connects to trusted sources, and delivers documented reports with citations.",
      company: "OpenAI",
      released: "February 10, 2026 update",
      whyItMatters:
        "Analytical work now includes AI systems that can independently gather evidence and return structured reasoning.",
    },
  },
  {
    id: "translator",
    title: "SMART TRANSLATOR",
    subtitle: "Global Communication",
    icon: "🌍",
    color: "from-teal-500 to-green-500",
    description: "Translate text with cultural context",
    placeholder: "Enter text to translate or language questions",
    examples: [
      "Translate 'Hello beautiful world' to French",
      "How do you say 'machine learning' in Japanese?",
      "Spanish business greetings",
    ],
    stateOfArt: {
      name: "Le Chat",
      url: "https://mistral.ai/en/products/le-chat",
      description:
        "Mistral's multilingual assistant blends web search, voice, document work, and multilingual reasoning into a single interface.",
      company: "Mistral AI",
      released: "July 17, 2025",
      whyItMatters:
        "Translation is becoming part of a broader multilingual workflow that includes summarization, search, and voice interaction.",
    },
  },
  {
    id: "summarizer",
    title: "SMART SUMMARIZER",
    subtitle: "Key Insights",
    icon: "📝",
    color: "from-yellow-500 to-orange-500",
    description: "Summarize complex text into key points",
    placeholder: "Paste text or describe what you need summarized",
    examples: [
      "Summarize the benefits of renewable energy",
      "Key points about AI in healthcare",
      "Overview of quantum computing",
    ],
    stateOfArt: {
      name: "Le Chat Deep Research",
      url: "https://help.mistral.ai/en/articles/424376-generate-reports-with-deep-research",
      description:
        "Le Chat's Deep Research mode autonomously browses the web and assembles structured reports with citations.",
      company: "Mistral AI",
      released: "July 17, 2025",
      whyItMatters:
        "Summarization is moving toward evidence-backed synthesis rather than short paraphrasing of a single document.",
    },
  },
  {
    id: "poem-generator",
    title: "POEM CREATOR",
    subtitle: "Poetic Expression",
    icon: "🌸",
    color: "from-pink-500 to-rose-500",
    description: "Create beautiful poems in various styles",
    placeholder: "What theme or emotion for your poem?",
    examples: [
      "A sonnet about artificial intelligence",
      "Free verse about ocean waves",
      "A limerick about coffee",
    ],
    stateOfArt: {
      name: "Claude Opus 4.6",
      url: "https://www.anthropic.com/news/claude-opus-4-6",
      description:
        "A frontier writing model suited for tone control, longer context windows, and more deliberate creative generation.",
      company: "Anthropic",
      released: "February 5, 2026",
      whyItMatters:
        "Creative generation now benefits from models built for depth, editing quality, and more reliable stylistic control.",
    },
  },
];

export default function AIPlayground() {
  const navigate = useNavigate();
  const [selectedDemo, setSelectedDemo] = useState<DemoType | null>(null);
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async (demo: AIDemo) => {
    if (!inputText.trim()) return;

    setIsGenerating(true);
    setOutputText("");

    // Simulate AI generation with realistic delays
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Generate demo responses based on the demo type
    const responses = generateDemoResponse(demo.id, inputText);

    // Typewriter effect
    for (let i = 0; i <= responses.length; i++) {
      setOutputText(responses.slice(0, i));
      await new Promise((resolve) => setTimeout(resolve, 30));
    }

    setIsGenerating(false);
  };

  const generateDemoResponse = (demoId: DemoType, input: string): string => {
    switch (demoId) {
      case "text-generator":
        return `Once upon a time, ${input.toLowerCase()}...\n\nIn a world where technology and imagination collide, our story unfolds with unexpected twists. The characters discover that the most powerful force isn't magic or science, but the connections they forge along the way.\n\nAs the sun set behind the digital horizon, they realized their adventure was just beginning. Each step forward revealed new possibilities, and with each challenge overcome, they grew stronger, wiser, and more determined to shape their destiny.\n\nThe end? Not quite. This is where your story truly begins...`;

      case "image-prompt":
        return `🎨 **Detailed Image Prompt:**\n\n"${input}", rendered in stunning 4K quality with cinematic lighting, intricate details, and photorealistic textures. The composition features dynamic angles with rich color palette, atmospheric effects, and professional photography techniques. Shot with depth of field, dramatic shadows, and golden hour lighting. Trending on ArtStation, award-winning digital art style.\n\n**Additional Elements:**\n• Hyperrealistic rendering\n• Volumetric lighting\n• Rich environmental details\n• Cinematic composition\n• Professional color grading`;

      case "code-helper":
        return `\`\`\`javascript\n// Solution for: ${input}\n\nfunction solutionFunction(data) {\n  // Implementation based on your requirements\n  const result = data\n    .filter(item => item.isValid)\n    .map(item => ({\n      ...item,\n      processed: true,\n      timestamp: new Date().toISOString()\n    }))\n    .sort((a, b) => new Date(b.date) - new Date(a.date));\n  \n  return result;\n}\n\n// Usage example:\nconst processedData = solutionFunction(yourData);\nconsole.log('Processed:', processedData);\n\`\`\`\n\n**Key Features:**\n• Error handling included\n• Optimized performance\n• Clean, readable code\n• Follows best practices`;

      case "creative-writer":
        return `✨ **Creative Response to: "${input}"**\n\nIn the realm where words dance and imagination soars,\nYour request blooms into artistic expression.\nEach syllable carefully chosen,\nEach phrase a brushstroke on the canvas of creativity.\n\nHere's your personalized creation:\n\n*[Imagine a beautifully crafted piece here, tailored specifically to your request, flowing with rhythm and meaning, designed to inspire and captivate your imagination.]*\n\nMay these words spark joy and inspiration in your heart! 🌟`;

      case "data-analyst":
        return `📊 **Data Analysis Insights for: "${input}"**\n\n**Key Findings:**\n• Pattern Analysis: Strong correlation identified\n• Trend Direction: Positive growth trajectory\n• Statistical Significance: 95% confidence level\n• Predictive Accuracy: High reliability\n\n**Recommendations:**\n1. Focus on high-performing segments\n2. Optimize underperforming areas\n3. Implement data-driven strategies\n4. Monitor key performance indicators\n\n**Next Steps:**\n• Set up automated reporting\n• Establish baseline metrics\n• Create actionable dashboards\n• Schedule regular reviews`;

      case "translator":
        return `🌍 **Translation & Cultural Context:**\n\nFor "${input}":\n\n**Primary Translation:**\n[Translated text with proper grammar and cultural nuance]\n\n**Cultural Notes:**\n• Context matters: Consider formal vs. informal usage\n• Regional variations may apply\n• Cultural sensitivity recommendations\n• Best practices for this language\n\n**Alternative Expressions:**\n• Formal version\n• Casual version\n• Business context\n• Creative interpretation\n\nLanguage is a bridge between cultures! 🌉`;

      case "summarizer":
        return `📝 **Smart Summary of: "${input}"**\n\n**Key Points:**\n• Main concept clearly defined\n• Primary benefits highlighted\n• Important considerations noted\n• Actionable insights provided\n\n**Executive Summary:**\nThe core message revolves around [key theme], emphasizing the importance of [main benefit] while addressing [key challenge]. This approach offers significant value through [specific advantages].\n\n**Takeaways:**\n1. Clear understanding of the topic\n2. Practical applications identified\n3. Strategic implications outlined\n4. Future considerations mapped\n\n*Summary optimized for clarity and actionability.*`;

      case "poem-generator":
        return `🌸 **Your Custom Poem: "${input}"**\n\nIn verses bright and words so true,\nA poem crafted just for you.\nWith rhythm, rhyme, and heartfelt grace,\nTo bring a smile upon your face.\n\nThe theme you chose inspires the lines,\nWhere creativity combines\nWith artificial intelligence,\nTo create something magnificent.\n\nEach stanza flows with purpose clear,\nTo touch the heart and calm the fear,\nThat beauty lives in simple things,\nAnd joy in what tomorrow brings.\n\n*May this poem bring you inspiration and delight!* ✨`;

      default:
        return "Amazing results generated based on your input! AI capabilities are truly limitless.";
    }
  };

  const handleExampleClick = (example: string) => {
    setInputText(example);
  };

  const selectedDemoData = AI_DEMOS.find((demo) => demo.id === selectedDemo);
  const pageRefresh = getPageRefreshContent("/ai-playground");
  const productRadar = latestAIProductLaunches.slice(0, 4);

  return (
    <SubpageLayout
      route="/ai-playground"
      eyebrow={pageRefresh.eyebrow}
      title={pageRefresh.title}
      description={pageRefresh.description}
      accent="rose"
      chips={pageRefresh.chips}
      refreshSummary={pageRefresh.refreshSummary}
      updatedAtLabel={pageRefresh.updatedAtLabel}
      metrics={[
        {
          value: AI_DEMOS.length.toString(),
          label: "Interactive demos",
        },
        {
          value: FEATURED_SHOWCASES.length.toString(),
          label: "Featured deep dives",
        },
      ]}
    >

      <section className="relative z-20 px-4 py-6 sm:px-6 sm:py-8">
        <div className="mx-auto max-w-7xl space-y-5">
          <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/30 p-5 backdrop-blur-xl sm:p-6">
            <div className="mb-4 flex items-baseline justify-between gap-3">
              <h2 className="text-lg font-semibold text-white">
                Current AI launches
              </h2>
              <span className="text-[11px] uppercase tracking-[0.2em] text-gray-400">
                Product radar
              </span>
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {productRadar.map((signal) => (
                <a
                  key={signal.id}
                  href={signal.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:bg-white/10"
                >
                  <div className="mb-2 flex items-center justify-between gap-2 text-[11px] uppercase tracking-[0.18em] text-gray-400">
                    <span className="bg-gradient-to-r from-fuchsia-300 to-cyan-300 bg-clip-text text-transparent">
                      {signal.category}
                    </span>
                    <span>{signal.date}</span>
                  </div>
                  <h3 className="text-sm font-semibold leading-snug text-white">
                    {signal.title}
                  </h3>
                  <p className="mt-1 text-xs text-cyan-100">{signal.org}</p>
                </a>
              ))}
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            {FEATURED_SHOWCASES.map((showcase, index) => (
              <motion.section
                key={showcase.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.45, delay: index * 0.08 }}
                className={`overflow-hidden rounded-[1.9rem] border bg-gradient-to-br p-5 backdrop-blur-xl sm:p-6 ${showcase.accent}`}
              >
                <div className="flex flex-col gap-5">
                  <div>
                    <div
                      className={`inline-flex rounded-full border px-3 py-1 text-[11px] font-bold uppercase tracking-[0.2em] ${showcase.badgeClass}`}
                    >
                      {showcase.badge}
                    </div>
                    <h2 className="mt-4 text-2xl font-black text-white sm:text-3xl">
                      {showcase.title}
                    </h2>
                    <p className="mt-3 text-sm leading-6 text-slate-100">
                      {showcase.summary}
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2 text-xs text-slate-100">
                    {showcase.tags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded-full border border-white/15 bg-white/8 px-3 py-1.5"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>

                  <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
                    <div className="rounded-[1.4rem] border border-white/15 bg-slate-950/45 p-4 text-sm text-slate-100">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                        What it shows
                      </p>
                      <p className="mt-2 font-semibold">{showcase.meta}</p>
                    </div>
                    <motion.button
                      onClick={() => navigate(showcase.route)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={`w-full rounded-2xl bg-gradient-to-r px-5 py-3 text-sm font-black tracking-[0.12em] sm:w-auto ${showcase.buttonClass}`}
                    >
                      VIEW DEMO
                    </motion.button>
                  </div>
                </div>
              </motion.section>
            ))}
          </div>
        </div>
      </section>

      {/* Demo Grid */}
      <section className="relative z-20 px-4 pb-8 pt-2 sm:px-6 sm:pb-10">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-8"
          >
            <motion.h2
              className="text-2xl font-black text-white sm:text-3xl"
              initial={{ opacity: 0, scale: 0.96 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
            >
              Try an interactive generator
            </motion.h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
              Select a demo, test the prompt flow, and compare it with the
              current leading product for that use case.
            </p>
          </motion.div>

          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-4 mb-8">
            {AI_DEMOS.map((demo, index) => (
              <motion.button
                key={demo.id}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                whileHover={{ scale: 1.02, y: -3 }}
                onClick={() => {
                  setSelectedDemo(demo.id);
                  setInputText("");
                  setOutputText("");
                }}
                className={`tom-ford-card rounded-[1.5rem] p-4 text-left transition-all duration-500 ${
                  selectedDemo === demo.id
                    ? "border-4 border-yellow-500 transform scale-[1.01]"
                    : "hover:border-yellow-400/50"
                }`}
              >
                <div className="text-3xl mb-3">{demo.icon}</div>
                <h3 className="tom-ford-subheading luxury-text-primary text-sm mb-1 tracking-widest">
                  {demo.title}
                </h3>
                <p className="luxury-text-muted text-xs mb-2">
                  {demo.subtitle}
                </p>
                <p className="luxury-text-muted text-xs font-light leading-relaxed">
                  {demo.description}
                </p>
              </motion.button>
            ))}
          </div>

          {/* Demo Interface */}
          <AnimatePresence mode="wait">
            {selectedDemo && selectedDemoData && (
              <motion.div
                key={selectedDemo}
                initial={{ opacity: 0, y: 60 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -60 }}
                transition={{ duration: 0.8 }}
                className="tom-ford-glass overflow-hidden rounded-[1.8rem]"
              >
                <div className="border-b-2 border-black p-5 sm:p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-4">
                      <div
                        className={`flex h-14 w-14 items-center justify-center rounded-xl border-2 border-black bg-gradient-to-r text-2xl ${selectedDemoData.color}`}
                      >
                        {selectedDemoData.icon}
                      </div>
                      <div>
                        <h3 className="tom-ford-heading text-xl luxury-text-primary sm:text-2xl">
                          {selectedDemoData.title}
                        </h3>
                        <p className="tom-ford-subheading luxury-text-muted text-sm">
                          {selectedDemoData.description}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedDemo(null)}
                      className="luxury-text-muted hover:luxury-text-primary transition-colors text-xl"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                <div className="bg-white/50 p-5 sm:p-6">
                  <div className="grid gap-6 lg:grid-cols-2">
                    {/* Input Section */}
                    <div>
                      <label className="tom-ford-subheading luxury-text-primary text-sm tracking-wider mb-4 block">
                        YOUR INPUT
                      </label>

                      {/* Example buttons */}
                      <div className="mb-4">
                        <p className="tom-ford-subheading luxury-text-muted text-xs mb-2">
                          TRY THESE EXAMPLES:
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {selectedDemoData.examples.map((example, index) => (
                            <button
                              key={index}
                              onClick={() => handleExampleClick(example)}
                              className="px-3 py-1 bg-yellow-100 border border-black rounded-lg text-xs luxury-text-primary hover:bg-yellow-200 transition-colors"
                            >
                              {example}
                            </button>
                          ))}
                        </div>
                      </div>

                      <textarea
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder={selectedDemoData.placeholder}
                        className="h-32 w-full resize-none rounded-xl border-2 border-black bg-white px-4 py-3 font-light luxury-text-primary placeholder-gray-500 transition-colors focus:border-yellow-500 focus:outline-none"
                      />

                      <motion.button
                        onClick={() => handleGenerate(selectedDemoData)}
                        disabled={!inputText.trim() || isGenerating}
                        whileHover={{ scale: inputText.trim() ? 1.02 : 1 }}
                        whileTap={{ scale: inputText.trim() ? 0.98 : 1 }}
                        className={`w-full mt-4 rounded-xl py-3 text-sm font-semibold tracking-wide transition-all duration-300 ${
                          inputText.trim()
                            ? "bg-slate-900 text-white hover:bg-slate-800"
                            : "cursor-not-allowed border-2 border-gray-300 bg-gray-200 text-gray-400"
                        }`}
                      >
                        {isGenerating ? (
                          <div className="flex items-center justify-center gap-3">
                            <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                            Rendering sample…
                          </div>
                        ) : (
                          "Show sample response"
                        )}
                      </motion.button>
                    </div>

                    {/* Output Section */}
                    <div>
                      <div className="mb-3 flex items-center justify-between gap-2">
                        <label className="text-sm font-semibold tracking-wide text-slate-900">
                          Sample response
                        </label>
                        <span className="rounded-full border border-amber-300 bg-amber-50 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-amber-700">
                          Illustrative · not live AI
                        </span>
                      </div>
                      <div className="h-64 overflow-y-auto rounded-xl border-2 border-black bg-white p-4">
                        {outputText ? (
                          <pre className="whitespace-pre-wrap text-sm font-light leading-relaxed text-slate-900">
                            {outputText}
                          </pre>
                        ) : (
                          <div className="flex h-full items-center justify-center">
                            <p className="text-center text-sm text-slate-500">
                              {isGenerating
                                ? "Assembling a representative response…"
                                : "Pick an example, then render a canned illustration of the response shape. For real generation, open the linked product below."}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-5">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                          Leading product
                        </p>
                        <h5 className="mt-1 text-lg font-semibold text-slate-900">
                          {selectedDemoData.stateOfArt.name}
                          <span className="ml-2 text-sm font-normal text-slate-500">
                            · {selectedDemoData.stateOfArt.company}
                          </span>
                        </h5>
                        <p className="mt-2 text-sm leading-relaxed text-slate-700">
                          {selectedDemoData.stateOfArt.whyItMatters}
                        </p>
                        <p className="mt-2 text-xs text-slate-500">
                          Released {selectedDemoData.stateOfArt.released}
                        </p>
                      </div>
                      <motion.a
                        href={selectedDemoData.stateOfArt.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.97 }}
                        className="inline-flex shrink-0 items-center gap-2 rounded-full bg-gradient-to-r from-fuchsia-500 via-rose-500 to-amber-400 px-5 py-2.5 text-sm font-semibold text-white shadow-[0_10px_30px_rgba(244,63,94,0.28)]"
                      >
                        Open {selectedDemoData.stateOfArt.name}
                        <svg
                          className="h-4 w-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                          />
                        </svg>
                      </motion.a>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      <ChatBot />
    </SubpageLayout>
  );
}
