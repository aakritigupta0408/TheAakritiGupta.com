import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

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
  };
}

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
      name: "ChatGPT-4",
      url: "https://chat.openai.com",
      description:
        "The most advanced conversational AI for creative writing and storytelling",
      company: "OpenAI",
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
      name: "Midjourney",
      url: "https://midjourney.com",
      description:
        "Leading AI image generation platform for artistic and photorealistic images",
      company: "Midjourney Inc.",
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 relative overflow-x-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-20 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-60 right-16 w-80 h-80 bg-blue-500/20 rounded-full blur-3xl animate-bounce"></div>
        <div className="absolute bottom-20 left-1/3 w-72 h-72 bg-pink-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-40 left-1/2 w-88 h-88 bg-cyan-500/20 rounded-full blur-3xl animate-bounce delay-500"></div>
        <div className="absolute bottom-40 right-1/4 w-64 h-64 bg-indigo-500/20 rounded-full blur-3xl animate-pulse delay-700"></div>
      </div>

      <Navigation />

      {/* Header Section */}
      <section className="relative z-20 pt-32 pb-20">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="text-center mb-16 relative z-10"
          >
            <motion.div
              className="w-32 h-32 bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 rounded-3xl flex items-center justify-center text-white text-5xl font-bold mx-auto mb-8 shadow-2xl border border-white/20 backdrop-blur-md"
              whileHover={{ scale: 1.1, rotate: 5 }}
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              🤖
            </motion.div>

            <motion.div
              className="inline-block p-1 rounded-full bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 mb-8"
              initial={{ opacity: 0, scale: 0.5, rotateY: -180 }}
              animate={{ opacity: 1, scale: 1, rotateY: 0 }}
              transition={{ duration: 1.2, ease: "backOut" }}
            >
              <h1 className="text-6xl md:text-8xl font-black bg-gradient-to-r from-white via-purple-100 to-cyan-100 bg-clip-text text-transparent px-8 py-6">
                AI PLAYGROUND
              </h1>
            </motion.div>

            <motion.p
              className="text-xl text-gray-100 max-w-5xl mx-auto mb-8 leading-relaxed"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              🎮 EXPLORE THE MAGIC OF ARTIFICIAL INTELLIGENCE THROUGH
              INTERACTIVE DEMOS! Ready to dive into the future? ✨
            </motion.p>

            <motion.div
              className="flex flex-wrap justify-center gap-4 mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
            >
              <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
                <span className="text-white font-bold">🧠 AI Powered</span>
              </div>
              <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
                <span className="text-white font-bold">🚀 Interactive</span>
              </div>
              <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
                <span className="text-white font-bold">✨ Creative</span>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Demo Grid */}
      <section className="relative z-20 py-20">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.h2
              className="text-5xl font-black text-white mb-6"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
            >
              CHOOSE YOUR
              <br />
              <span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                AI ADVENTURE
              </span>
            </motion.h2>
            <p className="tom-ford-subheading luxury-text-muted text-lg tracking-widest max-w-3xl mx-auto">
              CLICK ANY DEMO TO START EXPLORING AI CAPABILITIES
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
            {AI_DEMOS.map((demo, index) => (
              <motion.button
                key={demo.id}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                whileHover={{ scale: 1.02, y: -4 }}
                onClick={() => {
                  setSelectedDemo(demo.id);
                  setInputText("");
                  setOutputText("");
                }}
                className={`tom-ford-card rounded-xl p-6 text-center transition-all duration-500 ${
                  selectedDemo === demo.id
                    ? "border-4 border-yellow-500 transform scale-105"
                    : "hover:border-yellow-400/50"
                }`}
              >
                <div className="text-4xl mb-4">{demo.icon}</div>
                <h3 className="tom-ford-subheading luxury-text-primary text-sm mb-2 tracking-widest">
                  {demo.title}
                </h3>
                <p className="luxury-text-muted text-xs mb-3">
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
                className="tom-ford-glass rounded-xl overflow-hidden"
              >
                <div className="p-8 border-b-2 border-black">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-4">
                      <div
                        className={`w-16 h-16 bg-gradient-to-r ${selectedDemoData.color} rounded-xl flex items-center justify-center text-2xl border-2 border-black`}
                      >
                        {selectedDemoData.icon}
                      </div>
                      <div>
                        <h3 className="tom-ford-heading text-2xl luxury-text-primary">
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

                <div className="p-8 bg-white/50">
                  <div className="grid lg:grid-cols-2 gap-8">
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
                        className="w-full h-32 px-4 py-3 border-2 border-black rounded-xl luxury-text-primary placeholder-gray-500 focus:border-yellow-500 focus:outline-none transition-colors font-light resize-none bg-white"
                      />

                      <motion.button
                        onClick={() => handleGenerate(selectedDemoData)}
                        disabled={!inputText.trim() || isGenerating}
                        whileHover={{ scale: inputText.trim() ? 1.02 : 1 }}
                        whileTap={{ scale: inputText.trim() ? 0.98 : 1 }}
                        className={`w-full mt-4 py-4 rounded-xl font-light tracking-widest transition-all duration-300 ${
                          inputText.trim()
                            ? "tom-ford-button text-white"
                            : "bg-gray-200 text-gray-400 cursor-not-allowed border-2 border-gray-300"
                        }`}
                      >
                        {isGenerating ? (
                          <div className="flex items-center justify-center gap-3">
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            GENERATING...
                          </div>
                        ) : (
                          `✨ GENERATE WITH AI`
                        )}
                      </motion.button>
                    </div>

                    {/* Output Section */}
                    <div>
                      <label className="tom-ford-subheading luxury-text-primary text-sm tracking-wider mb-4 block">
                        AI OUTPUT
                      </label>
                      <div className="h-64 p-4 border-2 border-black rounded-xl bg-white overflow-y-auto">
                        {outputText ? (
                          <pre className="luxury-text-primary font-light text-sm leading-relaxed whitespace-pre-wrap">
                            {outputText}
                          </pre>
                        ) : (
                          <div className="flex items-center justify-center h-full">
                            <p className="luxury-text-muted text-sm text-center">
                              {isGenerating
                                ? "AI is working its magic..."
                                : "Enter your input and click generate to see AI in action!"}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* Call to Action */}
      <section className="relative z-20 py-20 border-t-2 border-black">
        <div className="max-w-7xl mx-auto px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="tom-ford-glass p-16 rounded-xl"
          >
            <div className="text-6xl mb-8">🚀</div>
            <h2 className="tom-ford-heading text-4xl luxury-text-primary mb-8">
              READY TO BUILD
              <br />
              <span className="gold-shimmer">YOUR OWN AI?</span>
            </h2>
            <p className="luxury-text-muted text-lg mb-8 max-w-2xl mx-auto">
              These demos showcase just a fraction of AI's potential. Let's
              discuss how we can build custom AI solutions for your specific
              needs.
            </p>
            <div className="flex justify-center gap-6 flex-wrap">
              <motion.button
                onClick={() => navigate("/")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="tom-ford-button px-8 py-4 rounded-xl text-white font-light tracking-wider"
              >
                EXPLORE MY PORTFOLIO
              </motion.button>
              <motion.button
                onClick={() => navigate("/games")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="border-2 border-black luxury-text-primary px-8 py-4 rounded-xl font-light tracking-wider hover:bg-yellow-100 transition-all duration-300"
              >
                TRY INTERACTIVE GAMES
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>

      <ChatBot />
    </div>
  );
}
