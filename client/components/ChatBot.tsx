import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { callOpenAI } from "@/api/chat";

interface Message {
  id: string;
  content: string;
  sender: "user" | "assistant";
  timestamp: Date;
}

const SUGGESTED_QUESTIONS = [
  "Tell me about your journey from Delhi to Silicon Valley",
  "What companies have you worked for?",
  "How did Yann LeCun recognize your work?",
  "Tell me about Swarnawastra",
  "What are your hobbies and interests?",
  "What's your educational background?",
];

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content:
        "Hi. I can answer questions about Aakriti Gupta's AI work, engineering background, research recognition, projects, and journey from Delhi to Silicon Valley.",
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
      const response = await callOpenAI(content);

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
          "I hit a temporary issue. Please try again and I will keep helping.",
        sender: "assistant",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
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
      {/* Persistent Chat Entry */}
      <motion.div
        className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-6"
        style={{ zIndex: 1000 }}
      >
        {/* Prompt Card */}
        {!isOpen && (
          <motion.div
            initial={{ opacity: 0, x: 50, scale: 0.8 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="max-w-sm rounded-[2rem] border border-slate-200/90 bg-white/92 px-6 py-5 shadow-[0_22px_60px_rgba(15,23,42,0.14)] backdrop-blur-2xl"
          >
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#0071e3]">
              AG AI Assistant
            </div>
            <div className="mb-4 text-sm leading-6 text-slate-600">
              Ask about Aakriti's work in AI, her experience at Meta, eBay, and
              Yahoo, or the projects and research featured across the site.
            </div>
            <div className="flex items-center gap-3 text-[11px] font-medium tracking-[0.08em] text-slate-400">
              <span>Always available</span>
              <span>•</span>
              <span>Fast answers</span>
            </div>
          </motion.div>
        )}

        {/* Chat Toggle */}
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          whileHover={{ scale: 1.04, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className="flex h-20 w-20 flex-col items-center justify-center rounded-full border border-sky-300/60 bg-[#0071e3] text-white shadow-[0_22px_50px_rgba(0,113,227,0.35)] transition-all duration-300"
          animate={{
            boxShadow: [
              "0 18px 40px rgba(0, 113, 227, 0.28)",
              "0 22px 56px rgba(0, 113, 227, 0.38)",
              "0 18px 40px rgba(0, 113, 227, 0.28)",
            ],
          }}
          transition={{
            boxShadow: {
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut",
            },
          }}
        >
          <AnimatePresence mode="wait">
            {isOpen ? (
              <motion.div
                key="close"
                initial={{ rotate: -90, opacity: 0 }}
                animate={{ rotate: 0, opacity: 1 }}
                exit={{ rotate: 90, opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="flex flex-col items-center"
              >
                <div className="mb-1 text-2xl">✕</div>
                <div className="text-[10px] font-medium uppercase tracking-[0.12em]">
                  Close
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="chat"
                initial={{ rotate: -90, opacity: 0 }}
                animate={{ rotate: 0, opacity: 1 }}
                exit={{ rotate: 90, opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="flex flex-col items-center"
              >
                <div className="mb-1 text-2xl">✦</div>
                <div className="text-[10px] font-medium uppercase tracking-[0.12em]">
                  Ask AG
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.button>
      </motion.div>

      {/* Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.85, y: 40 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.85, y: 40 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="fixed bottom-32 right-6 z-50 flex h-[750px] max-h-[calc(100vh-8rem)] w-[600px] max-w-[calc(100vw-3rem)] flex-col rounded-[2rem] border border-slate-200 bg-[#fbfbfd]/96 shadow-[0_30px_80px_rgba(15,23,42,0.16)] backdrop-blur-2xl"
            style={{ zIndex: 999 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-200 px-6 py-5">
              <div className="flex items-center gap-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#0071e3] text-white shadow-[0_12px_24px_rgba(0,113,227,0.28)]">
                  <span className="text-2xl">✦</span>
                </div>
                <div>
                  <h3 className="mb-1 text-lg font-semibold tracking-[-0.02em] text-slate-900">
                    Ask Aakriti's AI
                  </h3>
                  <p className="text-sm text-slate-500">
                    Work, research, projects, and background
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="rounded-full p-2 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700"
              >
                ✕
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 space-y-6 overflow-y-auto bg-gradient-to-b from-white via-[#f8f9fb] to-[#f2f4f8] p-6">
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-[1.5rem] px-5 py-4 ${
                      message.sender === "user"
                        ? "bg-[#0071e3] text-white shadow-[0_14px_28px_rgba(0,113,227,0.28)]"
                        : "border border-slate-200 bg-white text-slate-700 shadow-sm"
                    }`}
                  >
                    <p className="text-sm leading-6">
                      {message.content}
                    </p>
                  </div>
                </motion.div>
              ))}

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start"
                >
                  <div className="rounded-[1.5rem] border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="flex gap-3">
                      <div className="h-3 w-3 animate-bounce rounded-full bg-[#0071e3]" />
                      <div
                        className="h-3 w-3 animate-bounce rounded-full bg-[#0071e3]"
                        style={{ animationDelay: "0.1s" }}
                      />
                      <div
                        className="h-3 w-3 animate-bounce rounded-full bg-[#0071e3]"
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
              <div className="px-6 pb-5">
                <p className="mb-4 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">
                  Try one of these
                </p>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  {SUGGESTED_QUESTIONS.slice(0, 4).map((question) => (
                    <button
                      key={question}
                      onClick={() => handleSuggestedQuestion(question)}
                      className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-left text-xs font-medium leading-5 text-slate-700 transition-all duration-300 hover:-translate-y-0.5 hover:border-sky-200 hover:text-slate-900 hover:shadow-sm"
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
              className="border-t border-slate-200 bg-white/90 p-6"
            >
              <div className="flex flex-col gap-4 sm:flex-row">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask about Aakriti's work, experience, or projects..."
                  className="flex-1 rounded-2xl border border-slate-200 bg-[#f5f5f7] px-5 py-4 text-sm text-slate-900 placeholder-slate-400 transition-colors focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-100"
                  disabled={isLoading}
                />
                <motion.button
                  type="submit"
                  disabled={!inputValue.trim() || isLoading}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  className="rounded-2xl bg-[#0071e3] px-6 py-4 font-medium text-white transition-all duration-300 hover:bg-[#0077ed] disabled:cursor-not-allowed disabled:bg-slate-300 sm:px-6"
                >
                  <span className="text-lg">→</span>
                </motion.button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
