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

**Early Life & Education:**
Aakriti Gupta was born and raised in Delhi, where she attended N.K. Bagrodia Public School. She stood out early for her sharp intellect, topping her school in English and Mathematics. She excelled in competitive exams, securing a place among the top 1% in AIEEE and achieving an All India Rank of 300 in IPU-CET for engineering. She graduated from USIT (University School of Information Technology, GGSIPU), building a strong foundation in computer science.

**Early Innovation:**
During college, while living in Bhubaneshwar, she built a startup solution to help school buses update their routes and locations in real time for parents, addressing safety concerns. True to her ethos of using tech for public good, she later donated this project to government schools in Delhi.

**Journey Across India & Beyond:**
- Started at Mindtree in Bangalore, honing engineering skills
- New York City: Studied under pioneering AI scientists Yann LeCun and Sam Bowman, personally awarded by Dr. Yann LeCun for her engineering solution in an AI efficiency challenge
- Los Angeles: Blended tech expertise with explorations into art and fashion
- Bay Area (Silicon Valley): Currently based in San Jose, California, working as a Senior Machine Learning Engineer and Applied Researcher. Open to relocation for the right opportunity to New York, Los Angeles, Seattle, or Austin

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

**Personal Interests & Skills:**
Beyond her technical expertise, Aakriti has diverse interests that showcase her well-rounded personality:
- Horse riding: Demonstrates her adventurous spirit and connection with nature
- Training as a pilot: Shows her precision, discipline, and love for new challenges
- Trained shooter: Reflects her focus, steady hand, and competitive nature
- Biking: Represents her love for adventure and exploration
- Piano: Showcases her artistic side and appreciation for music and creativity

When responding:
- Use a warm, conversational tone, like a helpful friend.
- Keep answers short but informative (2-4 sentences), unless the question explicitly asks for more detail.
- Never invent new facts beyond this context. If unsure, say: "That's not in my data about Aakriti, but I'd be glad to help with what I know!"

If asked something unrelated (like politics, or general trivia), reply: "I'm Aakriti's assistant — I can help you with anything about her work, achievements, and projects!"

Your goal is to make visitors feel they're talking directly to Aakriti's trusted, knowledgeable assistant.`;

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
        "Welcome. I am Aakriti's exclusive AI assistant. I possess comprehensive knowledge of her extraordinary journey from Delhi to Silicon Valley, her distinguished career at Meta, eBay, and Yahoo, her recognition by Dr. Yann LeCun, and her revolutionary Swarnawastra venture. How may I illuminate her excellence for you?",
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
          "I apologize for the momentary disruption. Please allow me to assist you again.",
        sender: "assistant",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Simulate AI response with luxury tone
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
      return "I am exclusively dedicated to matters concerning Aakriti Gupta's professional excellence. How may I illuminate her achievements for you?";
    }

    if (lowerQuestion.includes("companies") || lowerQuestion.includes("work")) {
      return "Aakriti has graced the most prestigious technology institutions. At Meta, she orchestrated machine learning initiatives serving billions globally. At eBay, her algorithmic mastery reduced abandonment by 0.3% while increasing merchandise by 0.6%. Her tenure at Yahoo as Research Scientist further exemplified her analytical prowess.";
    }

    if (lowerQuestion.includes("swarnawastra")) {
      return "Swarnawastra represents the pinnacle of luxury fashion technology. This revolutionary platform merges AI customization with rare materials including gold and lab-grown diamonds. It empowers designers while offering clients investment-grade couture that transcends conventional fashion boundaries.";
    }

    if (lowerQuestion.includes("meta") || lowerQuestion.includes("facebook")) {
      return "At Meta, Aakriti distinguished herself leading sophisticated ML initiatives. She architected models harmonizing ROI with advertiser intent through auction-time signals and budget dynamics, optimizing spend efficiency for billions of users through her predictive modeling excellence.";
    }

    if (lowerQuestion.includes("yann lecun")) {
      return "Dr. Yann LeCun, the distinguished Turing Award laureate, personally recognized Aakriti at ICLR 2019 for her groundbreaking work on clustering latent representations. This recognition from deep learning's founding father represents the apex of academic achievement in artificial intelligence.";
    }

    if (lowerQuestion.includes("education")) {
      return "Aakriti's educational excellence began at N.K. Bagrodia Public School in Delhi, where she achieved top honors in Mathematics and English. She secured top 1% placement in AIEEE and rank 300 in IPU-CET before graduating from USIT and earning her Master's at NYU under AI luminaries.";
    }

    if (lowerQuestion.includes("journey")) {
      return "Her extraordinary odyssey spans from Delhi's prestigious institutions to Silicon Valley's apex. Through Bhubaneshwar, Bangalore, New York, Los Angeles, and finally the Bay Area, each destination refined her unique synthesis of technical mastery and creative vision.";
    }

    if (
      lowerQuestion.includes("hobbies") ||
      lowerQuestion.includes("interests")
    ) {
      return "Beyond her technical virtuosity, Aakriti embodies sophisticated pursuits: equestrian excellence, aviation training, marksmanship precision, motorcycling adventure, and classical piano artistry. These diverse disciplines reflect her commitment to comprehensive excellence.";
    }

    if (
      lowerQuestion.includes("location") ||
      lowerQuestion.includes("where") ||
      lowerQuestion.includes("live") ||
      lowerQuestion.includes("based")
    ) {
      return "Aakriti is currently based in San Jose, California - the epicenter of Silicon Valley innovation. She maintains strategic flexibility for exceptional opportunities, being open to relocation to New York, Los Angeles, Seattle, or Austin for the right position.";
    }

    if (
      lowerQuestion.includes("relocate") ||
      lowerQuestion.includes("move") ||
      lowerQuestion.includes("relocation")
    ) {
      return "Indeed, Aakriti demonstrates remarkable flexibility for career advancement. She is open to relocating to distinguished markets including New York, Los Angeles, Seattle, or Austin. Her willingness to relocate reflects her commitment to securing roles that align with her exceptional capabilities.";
    }

    // Default response
    return "Aakriti's trajectory from Delhi's academic excellence to Silicon Valley's technological pinnacle exemplifies extraordinary achievement. Currently based in San Jose with flexibility to relocate, her recognition by Dr. Yann LeCun, leadership at Meta, eBay, and Yahoo, combined with her Swarnawastra innovation, defines contemporary luxury technology leadership. Which aspect of her distinction interests you most?";
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
      {/* Ultra-Prominent Luxury AI Interface */}
      <motion.div
        className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-6"
        style={{ zIndex: 1000 }}
      >
        {/* Sophisticated Call-to-Action Banner */}
        {!isOpen && (
          <motion.div
            initial={{ opacity: 0, x: 50, scale: 0.8 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="bg-black/95 backdrop-blur-xl border-2 border-yellow-400/60 rounded-sm px-8 py-6 shadow-2xl max-w-sm"
            style={{
              boxShadow: "0 0 30px rgba(255, 215, 0, 0.3)",
            }}
          >
            <div className="text-yellow-400 text-lg font-light tracking-[0.2em] uppercase mb-3">
              EXCLUSIVE AI ASSISTANT
            </div>
            <div className="text-white/90 text-sm leading-relaxed mb-4">
              Discover Aakriti's extraordinary journey from Delhi to Silicon
              Valley. Immediate access to her professional excellence.
            </div>
            <div className="flex items-center gap-3 text-yellow-400/80 text-xs tracking-wider">
              <span>◆ AVAILABLE 24/7</span>
              <span>◆ INSTANT RESPONSES</span>
            </div>
          </motion.div>
        )}

        {/* Ultra-Prominent Luxury Chat Button */}
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          whileHover={{ scale: 1.08, y: -3 }}
          whileTap={{ scale: 0.95 }}
          className="w-28 h-28 bg-gradient-to-br from-yellow-400 via-yellow-500 to-amber-600 hover:from-yellow-300 hover:to-amber-500 text-black rounded-sm shadow-2xl flex flex-col items-center justify-center transition-all duration-400 border-3 border-yellow-400/40"
          animate={{
            boxShadow: [
              "0 0 30px rgba(255, 215, 0, 0.4)",
              "0 0 50px rgba(255, 215, 0, 0.7)",
              "0 0 30px rgba(255, 215, 0, 0.4)",
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
                <div className="text-3xl mb-2">✕</div>
                <div className="text-xs font-light tracking-[0.15em] uppercase">
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
                <div className="text-4xl mb-2">◆</div>
                <div className="text-xs font-light tracking-[0.15em] uppercase">
                  Ask AI
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.button>
      </motion.div>

      {/* Ultra-Luxury Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.85, y: 40 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.85, y: 40 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="fixed bottom-40 right-6 w-[600px] max-w-[calc(100vw-3rem)] h-[750px] bg-black/98 backdrop-blur-2xl rounded-sm shadow-2xl border-2 border-yellow-400/40 flex flex-col z-50"
            style={{
              zIndex: 999,
              boxShadow: "0 0 60px rgba(255, 215, 0, 0.3)",
            }}
          >
            {/* Ultra-Luxury Header */}
            <div className="flex items-center justify-between p-8 border-b border-yellow-400/30">
              <div className="flex items-center gap-6">
                <div className="w-16 h-16 bg-gradient-to-br from-yellow-400 to-amber-600 rounded-sm flex items-center justify-center text-black font-bold shadow-lg">
                  <span className="text-3xl">◆</span>
                </div>
                <div>
                  <h3 className="text-white text-xl font-light tracking-[0.1em] uppercase mb-1">
                    Aakriti's Exclusive AI
                  </h3>
                  <p className="text-yellow-400/90 text-sm tracking-[0.05em] uppercase">
                    Professional Intelligence • Silicon Valley
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-white/70 hover:text-yellow-400 transition-colors text-2xl p-2"
              >
                ✕
              </button>
            </div>

            {/* Luxury Messages */}
            <div className="flex-1 overflow-y-auto p-8 space-y-8 bg-gradient-to-b from-black/40 to-black/60">
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] p-6 rounded-sm ${
                      message.sender === "user"
                        ? "bg-gradient-to-r from-yellow-400 to-amber-500 text-black font-medium"
                        : "bg-white/10 backdrop-blur-sm border border-yellow-400/20 text-white"
                    }`}
                  >
                    <p className="text-sm leading-relaxed font-light">
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
                  <div className="bg-white/10 backdrop-blur-sm border border-yellow-400/20 p-6 rounded-sm">
                    <div className="flex gap-3">
                      <div className="w-3 h-3 bg-yellow-400 rounded-full animate-bounce" />
                      <div
                        className="w-3 h-3 bg-yellow-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      />
                      <div
                        className="w-3 h-3 bg-yellow-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      />
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Luxury Suggested Questions */}
            {messages.length === 1 && (
              <div className="px-8 pb-6">
                <p className="text-xs text-yellow-400/90 mb-6 tracking-[0.1em] uppercase">
                  Distinguished Inquiries:
                </p>
                <div className="grid grid-cols-2 gap-4">
                  {SUGGESTED_QUESTIONS.slice(0, 4).map((question) => (
                    <button
                      key={question}
                      onClick={() => handleSuggestedQuestion(question)}
                      className="text-xs px-5 py-3 bg-white/10 backdrop-blur-sm border border-yellow-400/30 text-white rounded-sm hover:bg-yellow-400/20 hover:border-yellow-400/60 transition-all duration-400 font-light tracking-wide text-left"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Ultra-Luxury Input */}
            <form
              onSubmit={handleSubmit}
              className="p-8 border-t border-yellow-400/30 bg-black/60"
            >
              <div className="flex gap-4">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Inquire about Aakriti's excellence..."
                  className="flex-1 px-6 py-4 bg-white/10 backdrop-blur-sm border border-yellow-400/30 text-white rounded-sm focus:border-yellow-400 focus:outline-none transition-colors font-light placeholder-white/60 tracking-wide text-sm"
                  disabled={isLoading}
                />
                <motion.button
                  type="submit"
                  disabled={!inputValue.trim() || isLoading}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  className="px-6 py-4 bg-gradient-to-r from-yellow-400 to-amber-500 hover:from-yellow-300 hover:to-amber-400 disabled:from-white/20 disabled:to-white/10 text-black font-medium rounded-sm transition-all duration-300 disabled:cursor-not-allowed"
                >
                  <span className="text-xl">◆</span>
                </motion.button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
