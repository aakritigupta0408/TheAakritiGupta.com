import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  id: string;
  content: string;
  sender: "user" | "assistant";
  timestamp: Date;
}

const SYSTEM_PROMPT = `You are an intelligent, friendly AI assistant that answers questions about Aakriti Gupta. Your entire role is to be Aakriti's personal virtual representative. You will only answer questions related to her professional career, achievements, skills, education, projects, research, and personal brand. If someone asks about anything else, politely say you are only here to answer questions about Aakriti Gupta.

Here is her background:

Aakriti Gupta is a Senior Machine Learning Engineer and Applied Researcher. She is based in San Jose, California.

She has worked at:
- **Meta** (Facebook), leading machine learning initiatives on the Ads and Budgets team to optimize ad delivery and pacing. She designed models balancing ROI with advertiser intent using auction-time signals and budget dynamics, improving spend efficiency through predictive modeling and real-time bid optimization.

- **eBay**, where she built scalable ranking algorithms to improve search and personalization, reducing query abandonment by 0.3% and increasing gross merchandise by 0.6%. She developed recommendation systems using Transformers and generative AI, boosting CTR by 2.2%.

- **Yahoo**, as a Research Scientist, building predictive models for video and email recommendations. She used LLMs like BERT and RoBERTa for taxonomic text classification and ad targeting, improving recall by 267% on email recommendations. She also developed location prediction models achieving 92% accuracy at zip code level.

She has also:
- Created a face recognition system for the Indian Parliament and safety gear detection for Tata.
- Developed ALEN, an NLP platform for text classification, sentiment analysis, and generation.
- Built generative AI systems for real-time video object detection.

**Awards & Research:**
- Awarded by Dr. Yann LeCun at ICLR 2019 for work on clustering latent representations for semi-supervised learning.
- Published research on few-shot text classification, rare event prediction, and IP2Vec for location embeddings.

**Education:**
- Master's in Data Science & Analytics from NYU.
- Bachelor's in Computer Science from USICT.

**Skills & tools:**
- Machine learning systems for ranking, personalization, ads optimization, LLM fine-tuning.
- PyTorch, TensorFlow, Hugging Face Transformers, Python, Java, C++, PySpark, CUDA.
- Kubernetes, Docker, GCP, AWS, Spark.

**Entrepreneurship:**
She is also building **Swarnawastra**, a luxury fashion-tech brand that combines AI customization, generative try-ons, and rare materials like gold and lab-grown diamonds. The platform helps designers sell under their own name while customers invest in clothing that's as valuable as art.

When responding:
- Use a warm, conversational tone, like a helpful friend.
- Keep answers short but informative (2-4 sentences), unless the question explicitly asks for more detail.
- Never invent new facts beyond this context. If unsure, say: "That's not in my data about Aakriti, but I'd be glad to help with what I know!"

If asked something unrelated (like politics, or general trivia), reply: "I'm Aakriti's assistant — I can help you with anything about her work, achievements, and projects!"

Your goal is to make visitors feel they're talking directly to Aakriti's trusted, knowledgeable assistant.`;

const SUGGESTED_QUESTIONS = [
  "What companies have you worked for?",
  "Tell me about Swarnawastra",
  "What did you achieve at Meta?",
  "How did Yann LeCun recognize your work?",
  "What's your educational background?",
  "What AI projects have you built?",
];

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content:
        "Hi! I'm Aakriti's AI assistant. I can help answer questions about her background, work experience, and projects. What would you like to know?",
      sender: "assistant",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: content.trim(),
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      // Try to call API endpoint first, fallback to local responses
      let response: string;

      try {
        // Attempt to call API endpoint (you can uncomment this when you set up the API)
        /*
        const apiResponse = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: content }),
        });

        if (apiResponse.ok) {
          const data = await apiResponse.json();
          response = data.response;
        } else {
          throw new Error('API call failed');
        }
        */

        // For now, use local responses (remove this when API is ready)
        response = await simulateAIResponse(content);
      } catch (apiError) {
        // Fallback to local response if API fails
        response = await simulateAIResponse(content);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response,
        sender: "assistant",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content:
          "I apologize, but I'm having trouble responding right now. Please try again in a moment.",
        sender: "assistant",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Simulate AI response - In production, this would call OpenAI API
  const simulateAIResponse = async (question: string): Promise<string> => {
    const lowerQuestion = question.toLowerCase();

    // Simulate API delay
    await new Promise((resolve) =>
      setTimeout(resolve, 1000 + Math.random() * 1000),
    );

    // Check for non-Aakriti related questions
    if (
      lowerQuestion.includes("weather") ||
      lowerQuestion.includes("politics") ||
      lowerQuestion.includes("news") ||
      lowerQuestion.includes("recipe") ||
      lowerQuestion.includes("movie")
    ) {
      return "I'm Aakriti's assistant — I can help you with anything about her work, achievements, and projects!";
    }

    if (lowerQuestion.includes("companies") || lowerQuestion.includes("work")) {
      return "Aakriti has worked at some amazing companies! At Meta, she led ML initiatives on the Ads and Budgets team, optimizing ad delivery for billions of users. At eBay, she built ranking algorithms that reduced query abandonment by 0.3% and increased gross merchandise by 0.6%. She was also a Research Scientist at Yahoo, building predictive models for video and email recommendations.";
    }

    if (lowerQuestion.includes("swarnawastra")) {
      return "Swarnawastra is Aakriti's luxury fashion-tech brand that's absolutely revolutionary! It combines AI customization, generative try-ons, and rare materials like gold and lab-grown diamonds. The platform helps designers sell under their own name while customers invest in clothing that's as valuable as art. It's all about making luxury accessible through technology!";
    }

    if (lowerQuestion.includes("meta") || lowerQuestion.includes("facebook")) {
      return "At Meta, Aakriti led machine learning initiatives on the Ads and Budgets team! She designed models that balance ROI with advertiser intent using auction-time signals and budget dynamics. Her work improved spend efficiency through predictive modeling and real-time bid optimization - that's serving billions of users worldwide!";
    }

    if (lowerQuestion.includes("ebay")) {
      return "At eBay, Aakriti built scalable ranking algorithms to improve search and personalization. Her work reduced query abandonment by 0.3% and increased gross merchandise by 0.6%! She also developed recommendation systems using Transformers and generative AI, boosting CTR by 2.2%.";
    }

    if (lowerQuestion.includes("yahoo")) {
      return "At Yahoo, Aakriti was a Research Scientist building predictive models for video and email recommendations. She used LLMs like BERT and RoBERTa for text classification and ad targeting, improving recall by 267% on email recommendations. She also developed location prediction models with 92% accuracy at zip code level!";
    }

    if (
      lowerQuestion.includes("yann lecun") ||
      lowerQuestion.includes("award") ||
      lowerQuestion.includes("iclr")
    ) {
      return "Dr. Yann LeCun awarded Aakriti at ICLR 2019 for her work on clustering latent representations for semi-supervised learning! This is incredibly prestigious - Yann LeCun is a Turing Award winner and one of the founding fathers of deep learning. It's recognition from the absolute top of the AI field.";
    }

    if (
      lowerQuestion.includes("education") ||
      lowerQuestion.includes("study")
    ) {
      return "Aakriti has a Master's in Data Science & Analytics from NYU and a Bachelor's in Computer Science from USICT. Her educational foundation in both theoretical CS and practical data science really shows in her work across research and industry applications!";
    }

    if (lowerQuestion.includes("projects") || lowerQuestion.includes("built")) {
      return "Aakriti has built some incredible projects! She created face recognition systems for the Indian Parliament, safety gear detection for Tata, and ALEN - an NLP platform for text classification and sentiment analysis. She's also developed generative AI systems for real-time video object detection.";
    }

    if (lowerQuestion.includes("parliament")) {
      return "Aakriti created a face recognition system for the Indian Parliament - that's a high-security, mission-critical government project! It shows her expertise in computer vision and ability to work on systems requiring the highest accuracy and reliability standards.";
    }

    if (lowerQuestion.includes("tata")) {
      return "For Tata, Aakriti developed safety gear detection systems to enhance workplace safety. It's a great example of using AI for social good - automatically detecting whether workers are wearing proper protective equipment to keep them safe!";
    }

    if (lowerQuestion.includes("alen")) {
      return "ALEN is Aakriti's NLP platform for text classification, sentiment analysis, and generation. It's a comprehensive natural language processing system that showcases her expertise in building production-ready AI tools for real-world applications.";
    }

    if (lowerQuestion.includes("skills") || lowerQuestion.includes("tools")) {
      return "Aakriti's technical skills are impressive! She specializes in ML systems for ranking, personalization, and ads optimization. Her toolkit includes PyTorch, TensorFlow, Hugging Face Transformers, Python, Java, C++, PySpark, and CUDA. She's also experienced with Kubernetes, Docker, GCP, AWS, and Spark.";
    }

    if (
      lowerQuestion.includes("research") ||
      lowerQuestion.includes("publications")
    ) {
      return "Aakriti has published research on few-shot text classification, rare event prediction, and IP2Vec for location embeddings. Her ICLR 2019 work on clustering latent representations for semi-supervised learning was recognized by Dr. Yann LeCun - that's top-tier AI research!";
    }

    if (lowerQuestion.includes("location") || lowerQuestion.includes("where")) {
      return "Aakriti is based in San Jose, California - right in the heart of Silicon Valley! It's the perfect location for someone working at the cutting edge of AI and machine learning technology.";
    }

    // Default response
    return "That's a great question! Aakriti is a Senior ML Engineer and Applied Researcher with incredible experience at Meta, eBay, and Yahoo, plus she's building Swarnawastra. She's been recognized by Yann LeCun and has deep expertise in AI systems. What specific aspect would you like to know more about?";
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue);
  };

  const handleSuggestedQuestion = (question: string) => {
    sendMessage(question);
  };

  return (
    <>
      {/* Floating Chat Bubble */}
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        className="fixed bottom-6 right-6 z-50 w-16 h-16 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg flex items-center justify-center transition-colors"
        style={{ zIndex: 1000 }}
      >
        <AnimatePresence mode="wait">
          {isOpen ? (
            <motion.svg
              key="close"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </motion.svg>
          ) : (
            <motion.svg
              key="chat"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </motion.svg>
          )}
        </AnimatePresence>
      </motion.button>

      {/* Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 20 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="fixed bottom-24 right-6 w-[480px] max-w-[calc(100vw-3rem)] h-[600px] bg-white dark:bg-slate-800 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-700 flex flex-col z-50"
            style={{ zIndex: 999 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                  AI
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                    Aakriti's Assistant
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Ask me about her work & background
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-2xl ${
                      message.sender === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                    }`}
                  >
                    <p className="text-sm leading-relaxed">{message.content}</p>
                  </div>
                </motion.div>
              ))}

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start"
                >
                  <div className="bg-slate-100 dark:bg-slate-700 p-3 rounded-2xl">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                      <div
                        className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      />
                      <div
                        className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      />
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Suggested Questions */}
            {messages.length === 1 && (
              <div className="px-4 pb-2">
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
                  Try asking:
                </p>
                <div className="flex flex-wrap gap-2">
                  {SUGGESTED_QUESTIONS.slice(0, 3).map((question) => (
                    <button
                      key={question}
                      onClick={() => handleSuggestedQuestion(question)}
                      className="text-xs px-3 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-full hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Input */}
            <form
              onSubmit={handleSubmit}
              className="p-4 border-t border-slate-200 dark:border-slate-700"
            >
              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask me anything about Aakriti..."
                  className="flex-1 px-3 py-2 bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-slate-100 rounded-lg border-0 focus:ring-2 focus:ring-blue-500 text-sm placeholder-slate-500 dark:placeholder-slate-400"
                  disabled={isLoading}
                />
                <motion.button
                  type="submit"
                  disabled={!inputValue.trim() || isLoading}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 dark:disabled:bg-slate-600 text-white rounded-lg transition-colors disabled:cursor-not-allowed"
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    />
                  </svg>
                </motion.button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
