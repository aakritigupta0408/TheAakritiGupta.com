import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

const photos = [
  {
    url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2Fed0bc18cd21244e1939892616f236f8f?format=webp&width=800",
    title: "AI Researcher",
  },
  {
    url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F8eb1e0d8ff0f4e7e8a3cb9a919e054b1?format=webp&width=800",
    title: "Technology Leader",
  },
  {
    url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F84cf6b44dba445fcaeced4f15fd299f1?format=webp&width=800",
    title: "Luxury Visionary",
  },
  {
    url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F4d9e8bcd67214b5b963eb37e44602024?format=webp&width=800",
    title: "Multi-disciplinary Expert",
  },
];

const pages = [
  { path: "/ai-playground", label: "AI Playground", icon: "🎮" },
  { path: "/ai-tools", label: "AI Tools", icon: "🛠️" },
  { path: "/ai-projects", label: "Projects", icon: "🚀" },
  { path: "/prompt-engineering", label: "Prompting", icon: "✨" },
  { path: "/ai-agent-training", label: "Agents", icon: "🤖" },
  { path: "/ai-companies", label: "Companies", icon: "🏢" },
  { path: "/ai-discoveries", label: "Discoveries", icon: "🔬" },
  { path: "/ai-champions", label: "Champions", icon: "🏆" },
  { path: "/games", label: "Games", icon: "🎯" },
  { path: "/resume-builder", label: "Resume", icon: "📄" },
];

export default function Index() {
  const navigate = useNavigate();
  const [activePhoto, setActivePhoto] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActivePhoto((prev) => (prev + 1) % photos.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative min-h-screen bg-[#f5f5f7] text-slate-900">
      <Navigation />

      <div className="h-20 lg:h-28" aria-hidden="true" />

      {/* When AI Always Agrees banner */}
      <Link
        to="/ai-safety"
        className="group block w-full bg-gradient-to-r from-teal-800 via-teal-700 to-amber-600 px-6 py-4 text-left text-white shadow-lg transition-all hover:brightness-110 sm:py-5"
      >
        <div className="mx-auto flex max-w-7xl flex-col items-start gap-1 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
          <div className="flex items-center gap-3">
            <span className="text-2xl sm:text-3xl" aria-hidden="true">
              🧠
            </span>
            <div>
              <div className="text-lg font-extrabold tracking-tight sm:text-2xl">
                Why Your Favorite Chatbot Can Cause the Next Pandemic — the
                Mental Health Pandemic
              </div>
              <div className="text-xs text-teal-100 sm:text-sm">
                Court filings · company disclosures · peer-reviewed research ·
                2022–2026 · every claim cited
              </div>
            </div>
          </div>
          <span className="shrink-0 rounded-full bg-white/15 px-4 py-2 text-sm font-bold backdrop-blur transition-all group-hover:bg-white group-hover:text-teal-800">
            Read the evidence →
          </span>
        </div>
      </Link>

      <main className="mx-auto flex h-[calc(100vh-10rem)] max-w-7xl items-start gap-8 px-6 lg:h-[calc(100vh-12rem)] lg:items-center">
        {/* Left — identity + navigation */}
        <div className="flex min-w-0 flex-1 flex-col gap-5">
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
              Aakriti Gupta
            </h1>
            <p className="mt-2 text-base text-slate-500 sm:text-lg">
              Senior ML Engineer · AI Researcher · Meta · eBay · Yahoo
            </p>
            <p className="mt-1 text-sm text-slate-400">
              Recognized by Dr. Yann LeCun at ICLR 2019 · Building AI to make life simpler
            </p>
          </div>

          {/* Page grid */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
            {pages.map((page) => (
              <button
                key={page.path}
                onClick={() => navigate(page.path)}
                className="group flex items-center gap-2 rounded-xl border border-slate-200 bg-white/80 px-3 py-2.5 text-left text-xs font-medium text-slate-700 shadow-sm transition-all hover:-translate-y-0.5 hover:bg-white hover:shadow-md"
              >
                <span className="text-base">{page.icon}</span>
                <span className="truncate">{page.label}</span>
              </button>
            ))}
          </div>

          {/* Quick links */}
          <div className="flex flex-wrap items-center gap-3">
            <a
              href="https://drive.google.com/file/d/1Mnmk6nP9l_Av0LvpgJQ5Tkjb7BqhY7nb/view?usp=sharing"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white shadow-md transition-all hover:bg-slate-800"
            >
              Resume ↗
            </a>
            <a
              href="https://www.linkedin.com/in/aakritigupta4894/"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition-all hover:bg-slate-50"
            >
              LinkedIn
            </a>
            <a
              href="https://github.com/aakritigupta0408"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition-all hover:bg-slate-50"
            >
              GitHub
            </a>
            <a
              href="/paperleaks/"
              className="rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition-all hover:bg-slate-50"
            >
              Paper Leak Data
            </a>
          </div>

          {/* Talents row */}
          <div className="flex flex-wrap gap-2 text-xs text-slate-400">
            {["🎯 Marksman", "🏇 Equestrian", "✈️ Aviator", "🏍️ Motorcyclist", "🎹 Pianist", "💎 Swarnawastra"].map(
              (talent) => (
                <span
                  key={talent}
                  className="rounded-full border border-slate-200/60 bg-white/60 px-2.5 py-1"
                >
                  {talent}
                </span>
              ),
            )}
          </div>
        </div>

        {/* Right — photo */}
        <div className="hidden aspect-[3/4] w-72 shrink-0 overflow-hidden rounded-2xl border border-white/80 bg-white/80 shadow-xl lg:block xl:w-80">
          <AnimatePresence mode="wait">
            <motion.img
              key={activePhoto}
              src={photos[activePhoto].url}
              alt={photos[activePhoto].title}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.6 }}
              className="h-full w-full object-cover"
            />
          </AnimatePresence>
        </div>

        {/* Far right — sponsor rail */}
        <aside
          aria-label="Sponsored"
          className="hidden w-44 shrink-0 flex-col gap-3 self-start xl:flex"
        >
          <div className="text-center text-[10px] font-semibold uppercase tracking-widest text-slate-400">
            Sponsored
          </div>
          <div className="rounded-xl border border-violet-200 bg-violet-50/80 p-3 text-center shadow-sm">
            <div className="text-2xl" aria-hidden="true">
              🔮
            </div>
            <div className="mt-1 text-xs font-bold text-violet-900">
              Astrologers
            </div>
            <p className="mt-1 text-[11px] leading-snug text-violet-700">
              Advertise your practice to readers of the Vedic Astrology AI.
            </p>
          </div>
          <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-3 text-center shadow-sm">
            <div className="text-2xl" aria-hidden="true">
              💎
            </div>
            <div className="mt-1 text-xs font-bold text-amber-900">
              Jewelers
            </div>
            <p className="mt-1 text-[11px] leading-snug text-amber-700">
              Showcase your jewelry and gemstones here.
            </p>
          </div>
          <a
            href="https://www.linkedin.com/in/aakritigupta4894/"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-center text-[11px] font-semibold text-slate-600 shadow-sm transition-all hover:bg-slate-50"
          >
            Advertise here →
          </a>
        </aside>
      </main>

      <ChatBot />
    </div>
  );
}
