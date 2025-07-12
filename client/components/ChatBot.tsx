import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  id: string;
  content: string;
  sender: "user" | "assistant";
  timestamp: Date;
}

const SYSTEM_PROMPT = `You are Aakriti Gupta's AI assistant on her professional portfolio website. You should answer questions about her background, work, and achievements in a friendly, conversational manner. Here's what you should know about Aakriti:

PROFESSIONAL BACKGROUND:
- AI Engineer and Technology Leader with expertise in machine learning and large-scale systems
- Worked at major tech companies: Meta (Facebook), eBay, Yahoo
- At Meta: Built ML-driven advertising systems serving billions of users worldwide
- At eBay: Scaled e-commerce infrastructure handling millions of daily transactions
- At Yahoo: Early engineering experience in large-scale web technologies
- Founded an AI company specializing in product image transformation and enhancement

CURRENT BUSINESS - SWARNAWASTRA:
- Founder of Swarnawastra, a luxury fashion-tech brand
- Mission: Democratizing access to luxury through AI, gold, and lab-grown diamonds
- Uses AI to make luxury design and high-end fashion more accessible
- Innovative approach combining technology with traditional luxury craftsmanship

NOTABLE PROJECTS:
- Developed face recognition systems for the Indian Parliament for government security
- Created PPE detection systems for Tata to enhance workplace safety
- Built ML systems for product image enhancement and transformation
- Worked on large-scale advertising and e-commerce platforms

ACHIEVEMENTS & RECOGNITION:
- Recognized by Yann LeCun (Turing Award winner) for innovative AI contributions
- Successfully founded and led technology companies
- Expert in AI, machine learning, computer vision, and large-scale systems
- Strong engineering background with proven track record at top tech companies

CURRENT FOCUS:
- Scaling Swarnawastra to democratize luxury with AI
- Developing innovative AI applications for fashion and luxury goods
- Working with gold and lab-grown diamonds in luxury products
- Building technology that makes high-end design accessible to more people

Keep responses conversational, helpful, and focused on her professional achievements. If asked about something you don't know, politely say you don't have that specific information but offer to help with what you do know about her work.`;

const SUGGESTED_QUESTIONS = [
  "What companies have you worked for?",
  "Tell me about Swarnawastra",
  "What is your biggest achievement?",
  "What kind of AI projects have you worked on?",
  "How did Yann LeCun recognize your work?",
  "What makes Swarnawastra unique?",
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
      // In a real implementation, you would call your API endpoint here
      // For now, I'll simulate an API call with a more sophisticated local response
      const response = await simulateAIResponse(content);

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

    if (lowerQuestion.includes("companies") || lowerQuestion.includes("work")) {
      return "Aakriti has worked at several major tech companies! She was at Meta (Facebook) where she built ML-driven advertising systems serving billions of users. Before that, she worked at eBay scaling e-commerce infrastructure for millions of daily transactions, and at Yahoo for large-scale web technologies. She's also founded her own AI company focused on product image transformation.";
    }

    if (lowerQuestion.includes("swarnawastra")) {
      return "Swarnawastra is Aakriti's luxury fashion-tech brand that's really innovative! The mission is to democratize access to luxury through AI, gold, and lab-grown diamonds. She's using AI to make luxury design and high-end fashion more accessible to people, combining cutting-edge technology with traditional luxury craftsmanship.";
    }

    if (
      lowerQuestion.includes("achievement") ||
      lowerQuestion.includes("award")
    ) {
      return "One of Aakriti's biggest achievements is being recognized by Yann LeCun - he's a Turing Award winner and one of the most respected figures in AI! She's also successfully founded multiple technology companies and has made significant contributions to large-scale AI systems at top tech companies.";
    }

    if (
      lowerQuestion.includes("yann lecun") ||
      lowerQuestion.includes("recognition")
    ) {
      return "Yann LeCun, who won the Turing Award for his contributions to deep learning, recognized Aakriti for her innovative AI contributions. This is a huge honor since he's considered one of the founding fathers of modern AI and deep learning!";
    }

    if (lowerQuestion.includes("projects") || lowerQuestion.includes("ai")) {
      return "Aakriti has worked on some fascinating AI projects! She developed face recognition systems for the Indian Parliament for security purposes, created PPE detection systems for Tata to enhance workplace safety, and built ML systems for product image enhancement. Her work spans computer vision, large-scale systems, and practical AI applications.";
    }

    if (
      lowerQuestion.includes("parliament") ||
      lowerQuestion.includes("government")
    ) {
      return "Aakriti developed face recognition systems for the Indian Parliament - that's a high-security, mission-critical project for government security. It shows her expertise in computer vision and her ability to work on systems that require the highest levels of accuracy and reliability.";
    }

    if (lowerQuestion.includes("tata") || lowerQuestion.includes("safety")) {
      return "For Tata, Aakriti created PPE (Personal Protective Equipment) detection systems to enhance workplace safety. This is a great example of using AI for social good - helping protect workers by automatically detecting whether proper safety equipment is being used.";
    }

    if (lowerQuestion.includes("meta") || lowerQuestion.includes("facebook")) {
      return "At Meta (Facebook), Aakriti built ML-driven advertising systems that serve billions of users worldwide. That's incredibly complex work - handling that scale requires deep expertise in machine learning, distributed systems, and performance optimization.";
    }

    if (
      lowerQuestion.includes("luxury") ||
      lowerQuestion.includes("diamonds")
    ) {
      return "Through Swarnawastra, Aakriti is working with gold and lab-grown diamonds to create luxury products. She's using AI to make luxury design more accessible and democratic, which is a unique approach to combining technology with traditional luxury craftsmanship.";
    }

    // Default response
    return "That's a great question! Aakriti has extensive experience in AI, machine learning, and technology leadership. She's worked at Meta, eBay, and Yahoo, founded Swarnawastra, and has been recognized by AI leaders like Yann LeCun. Is there something specific about her background or work you'd like to know more about?";
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
            className="fixed bottom-24 right-6 w-96 max-w-[calc(100vw-3rem)] h-[500px] bg-white dark:bg-slate-800 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-700 flex flex-col z-50"
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
