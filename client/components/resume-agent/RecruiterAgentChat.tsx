import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { ResumeAgentChatTurn } from "@shared/api";
import type { ResumeAgentProfile } from "@shared/resume-agent";
import { chatWithResumeAgent } from "@/api/resume-agent";

interface RecruiterAgentChatProps {
  profile: ResumeAgentProfile;
  mode: "preview" | "recruiter";
  shareUrl?: string;
  redirectUrl?: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

function toHistory(messages: ChatMessage[]): ResumeAgentChatTurn[] {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
}

export default function RecruiterAgentChat({
  profile,
  mode,
  shareUrl,
  redirectUrl = "https://www.theaakritigupta.com",
}: RecruiterAgentChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: `Hi. I can answer recruiter questions about ${profile.candidateName} using only the resume and project material shared for this chat. If a detail is missing, I will say so instead of guessing.`,
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">(
    "idle",
  );
  const [isFinishing, setIsFinishing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const sendMessage = async (content: string) => {
    const trimmed = content.trim();

    if (!trimmed || isLoading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: trimmed,
    };

    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await chatWithResumeAgent({
        profile,
        message: trimmed,
        history: toHistory(messages.filter((message) => message.id !== "welcome")),
      });

      setMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          content: response.response,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const copyShareUrl = async () => {
    if (!shareUrl) {
      return;
    }

    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopyStatus("copied");
      window.setTimeout(() => setCopyStatus("idle"), 2000);
    } catch (error) {
      void error;
      setCopyStatus("failed");
      window.setTimeout(() => setCopyStatus("idle"), 2000);
    }
  };

  const finishAndRedirect = () => {
    setIsFinishing(true);
    window.setTimeout(() => {
      window.location.assign(redirectUrl);
    }, 1000);
  };

  return (
    <div className="rounded-[2rem] border border-white/15 bg-white/10 p-6 shadow-2xl backdrop-blur-xl">
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="mb-3 inline-flex rounded-full border border-emerald-300/30 bg-emerald-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-emerald-100">
            Recruiter-safe agent
          </div>
          <h2 className="text-2xl font-black text-white md:text-3xl">
            {profile.candidateName}
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-200">
            {profile.professionalHeadline}
          </p>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-300">
            {profile.summary}
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          {mode === "preview" && shareUrl && (
            <motion.button
              type="button"
              onClick={copyShareUrl}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-5 py-3 text-sm font-bold text-white transition-colors hover:bg-cyan-400/15"
            >
              {copyStatus === "copied"
                ? "Link copied"
                : copyStatus === "failed"
                  ? "Copy failed"
                  : "Copy recruiter link"}
            </motion.button>
          )}

          {mode === "recruiter" && (
            <motion.button
              type="button"
              onClick={finishAndRedirect}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="rounded-full border border-amber-300/30 bg-amber-400/10 px-5 py-3 text-sm font-bold text-white transition-colors hover:bg-amber-400/15"
            >
              {isFinishing ? "Redirecting..." : "Finish review"}
            </motion.button>
          )}
        </div>
      </div>

      <div className="mb-5 rounded-3xl border border-white/10 bg-black/20 p-5">
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-100">
          Grounding policy
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-200">
          This chatbot answers only from the uploaded resume and project notes.
          If a recruiter asks for something that is not in the candidate
          material, the chatbot refuses instead of inventing details.
        </p>
      </div>

      <div className="mb-5 grid gap-4 lg:grid-cols-2">
        {profile.sections.slice(0, 4).map((section) => (
          <div
            key={section.id}
            className="rounded-3xl border border-white/10 bg-black/20 p-5"
          >
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-violet-100">
              {section.title}
            </p>
            <div className="mt-3 space-y-2">
              {section.bullets.slice(0, 3).map((bullet) => (
                <p key={bullet} className="text-sm leading-relaxed text-slate-200">
                  - {bullet}
                </p>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mb-5 rounded-3xl border border-white/10 bg-slate-950/40">
        <div className="border-b border-white/10 px-5 py-4">
          <p className="text-sm font-black uppercase tracking-[0.18em] text-white">
            Recruiter chat
          </p>
        </div>

        <div className="max-h-[26rem] space-y-4 overflow-y-auto px-5 py-5">
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-[1.5rem] px-5 py-4 text-sm leading-6 ${
                  message.role === "user"
                    ? "bg-[#0071e3] text-white"
                    : "border border-white/10 bg-white/10 text-slate-100"
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
            </motion.div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="rounded-[1.5rem] border border-white/10 bg-white/10 px-5 py-4">
                <div className="flex gap-2">
                  <div className="h-2.5 w-2.5 animate-bounce rounded-full bg-cyan-200" />
                  <div
                    className="h-2.5 w-2.5 animate-bounce rounded-full bg-cyan-200"
                    style={{ animationDelay: "0.1s" }}
                  />
                  <div
                    className="h-2.5 w-2.5 animate-bounce rounded-full bg-cyan-200"
                    style={{ animationDelay: "0.2s" }}
                  />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {messages.length === 1 && profile.suggestedQuestions.length > 0 && (
        <div className="mb-5">
          <p className="mb-3 text-xs font-bold uppercase tracking-[0.18em] text-slate-300">
            Suggested recruiter prompts
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            {profile.suggestedQuestions.map((question) => (
              <button
                key={question}
                type="button"
                onClick={() => sendMessage(question)}
                className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-left text-sm font-medium text-slate-100 transition-colors hover:bg-white/10"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      <form
        onSubmit={(event) => {
          event.preventDefault();
          sendMessage(inputValue);
        }}
        className="flex flex-col gap-4 sm:flex-row"
      >
        <input
          type="text"
          value={inputValue}
          onChange={(event) => setInputValue(event.target.value)}
          placeholder="Ask about skills, projects, or experience..."
          className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-5 py-4 text-sm text-white placeholder:text-slate-400 focus:border-cyan-300/40 focus:outline-none"
          disabled={isLoading || isFinishing}
        />
        <motion.button
          type="submit"
          disabled={!inputValue.trim() || isLoading || isFinishing}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          className="rounded-2xl bg-[#0071e3] px-6 py-4 text-sm font-bold text-white transition-colors hover:bg-[#0077ed] disabled:cursor-not-allowed disabled:bg-slate-500"
        >
          Send
        </motion.button>
      </form>

      <AnimatePresence>
        {mode === "recruiter" && isFinishing && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 12 }}
            className="mt-4 rounded-2xl border border-amber-300/20 bg-amber-400/10 px-4 py-3 text-sm text-amber-100"
          >
            Redirecting back to the main site.
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
