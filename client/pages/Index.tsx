import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";
import { saveEmailToLocalStorage } from "@/api/save-email";
import { siteRefreshMeta } from "@/data/siteRefreshContent";
import type {
  SaveEmailRequest,
  SaveEmailResponse,
  SiteRefreshTriggerResponse,
} from "@shared/api";

// Photo gallery with creative transitions
const PhotoGallery = () => {
  const [activePhoto, setActivePhoto] = useState(0);

  const photos = [
    {
      url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2Fed0bc18cd21244e1939892616f236f8f?format=webp&width=800",
      title: "AI Researcher",
      subtitle: "Pushing boundaries in machine learning",
    },
    {
      url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F8eb1e0d8ff0f4e7e8a3cb9a919e054b1?format=webp&width=800",
      title: "Technology Leader",
      subtitle: "Leading innovation at top tech companies",
    },
    {
      url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F84cf6b44dba445fcaeced4f15fd299f1?format=webp&width=800",
      title: "Luxury Visionary",
      subtitle: "Founder of Swarnawastra",
    },
    {
      url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F4d9e8bcd67214b5b963eb37e44602024?format=webp&width=800",
      title: "Multi-disciplinary Expert",
      subtitle: "From Delhi to Silicon Valley",
    },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActivePhoto((prev) => (prev + 1) % photos.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative w-full h-full">
      <AnimatePresence mode="wait">
        <motion.div
          key={activePhoto}
          initial={{ opacity: 0, scale: 1.1 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="relative w-full h-full"
        >
          <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent z-10 rounded-2xl" />
          <img
            src={photos[activePhoto].url}
            alt={photos[activePhoto].title}
            className="w-full h-full object-cover rounded-2xl shadow-2xl"
          />
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="absolute bottom-6 left-6 z-20 text-white"
          >
            <h3 className="text-xl font-bold mb-1">
              {photos[activePhoto].title}
            </h3>
            <p className="text-white/80 text-sm">
              {photos[activePhoto].subtitle}
            </p>
          </motion.div>
        </motion.div>
      </AnimatePresence>

      {/* Photo indicators */}
      <div className="absolute bottom-4 right-4 z-20 flex gap-2">
        {photos.map((_, idx) => (
          <button
            key={idx}
            onClick={() => setActivePhoto(idx)}
            className={`w-3 h-3 rounded-full transition-all duration-300 ${
              idx === activePhoto ? "bg-white" : "bg-white/40"
            }`}
          />
        ))}
      </div>
    </div>
  );
};

// Floating achievement badges
const AchievementBadges = () => {
  const achievements = [
    { icon: "🏆", label: "Yann LeCun Award", delay: 0 },
    { icon: "🤖", label: "AI Expert", delay: 0.5 },
    { icon: "💎", label: "Luxury Tech", delay: 1 },
    { icon: "🎯", label: "Marksman", delay: 1.5 },
    { icon: "🏇", label: "Equestrian", delay: 2 },
    { icon: "✈️", label: "Pilot", delay: 2.5 },
    { icon: "🏍️", label: "Motorcyclist", delay: 3 },
    { icon: "🎹", label: "Pianist", delay: 3.5 },
  ];

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {achievements.map((achievement, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0, scale: 0, rotate: -180 }}
          animate={{
            opacity: [0, 1, 1, 0],
            scale: [0, 1, 1, 0],
            rotate: [0, 360],
            x: [
              Math.max(
                0,
                Math.random() *
                  (typeof window !== "undefined" && window.innerWidth > 0
                    ? window.innerWidth
                    : 1200),
              ),
              Math.max(
                0,
                Math.random() *
                  (typeof window !== "undefined" && window.innerWidth > 0
                    ? window.innerWidth
                    : 1200),
              ),
            ],
            y: [
              Math.max(
                0,
                Math.random() *
                  (typeof window !== "undefined" && window.innerHeight > 0
                    ? window.innerHeight
                    : 800),
              ),
              Math.max(
                0,
                Math.random() *
                  (typeof window !== "undefined" && window.innerHeight > 0
                    ? window.innerHeight
                    : 800),
              ),
            ],
          }}
          transition={{
            duration: 8,
            delay: achievement.delay,
            repeat: Infinity,
            repeatDelay: 10,
          }}
          className="absolute"
        >
          <div className="bg-white/10 backdrop-blur-md rounded-full px-4 py-2 border border-white/20 shadow-lg">
            <div className="flex items-center gap-2">
              <span className="text-2xl">{achievement.icon}</span>
              <span className="text-white text-sm font-medium">
                {achievement.label}
              </span>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

// Company logo carousel
const CompanyCarousel = () => {
  const companies = [
    {
      name: "Meta",
      logo: "f",
      color: "bg-blue-500",
      gradient: "from-blue-500 to-blue-700",
    },
    {
      name: "eBay",
      logo: "eB",
      color: "bg-gradient-to-r from-red-500 via-yellow-400 to-blue-500",
      gradient: "from-red-500 to-blue-500",
    },
    {
      name: "Yahoo",
      logo: "Y!",
      color: "bg-purple-600",
      gradient: "from-purple-500 to-purple-700",
    },
  ];

  return (
    <motion.div
      className="flex gap-6 justify-center"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1, duration: 0.8 }}
    >
      {companies.map((company, idx) => (
        <motion.div
          key={company.name}
          whileHover={{ scale: 1.1, y: -5 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 1.2 + idx * 0.2, duration: 0.5 }}
          className={`relative group cursor-pointer`}
        >
          <div
            className={`w-16 h-16 ${company.color} rounded-2xl flex items-center justify-center text-white font-bold text-xl shadow-xl border-2 border-white/20 group-hover:border-white/40 transition-all duration-300`}
          >
            {company.logo}
          </div>
          <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <div className="bg-black/80 backdrop-blur-sm text-white text-xs px-3 py-1 rounded-full">
              {company.name}
            </div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

export default function Index() {
  const navigate = useNavigate();
  const manualSiteRefreshUrl =
    "https://github.com/aakritigupta0408/TheAakritiGupta.com/actions/workflows/site-refresh.yml";
  const [email, setEmail] = useState("");
  const [emailSubmitted, setEmailSubmitted] = useState(false);
  const [isSavingEmail, setIsSavingEmail] = useState(false);
  const [emailMessage, setEmailMessage] = useState<string | null>(null);
  const [emailMessageTone, setEmailMessageTone] = useState<
    "success" | "error" | null
  >(null);
  const [isTriggeringSiteRefresh, setIsTriggeringSiteRefresh] = useState(false);
  const [siteRefreshMessage, setSiteRefreshMessage] = useState<string | null>(
    null,
  );
  const [siteRefreshMessageTone, setSiteRefreshMessageTone] = useState<
    "success" | "error" | null
  >(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const normalizedEmail = email.trim();

    if (!normalizedEmail || isSavingEmail) {
      return;
    }

    setIsSavingEmail(true);
    setEmailMessage(null);
    setEmailMessageTone(null);

    const savedLocally = saveEmailToLocalStorage(normalizedEmail);
    let savedRemotely = false;

    try {
      const requestBody: SaveEmailRequest = { email: normalizedEmail };
      const response = await fetch("/api/save-email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorBody = (await response
          .json()
          .catch(() => null)) as SaveEmailResponse | null;

        throw new Error(
          errorBody?.message || "Server failed to save the email.",
        );
      }

      savedRemotely = true;
    } catch (serverError) {
      console.log("Server save failed, using localStorage:", serverError);
    } finally {
      setIsSavingEmail(false);
    }

    if (savedLocally || savedRemotely) {
      setEmailSubmitted(true);
      setEmailMessageTone("success");
      setEmailMessage(
        savedRemotely
          ? "Thanks. You are on the list."
          : "Saved locally for now. It will sync once the server is available.",
      );
      setTimeout(() => {
        setEmailSubmitted(false);
        setEmail("");
        setEmailMessage(null);
        setEmailMessageTone(null);
      }, 3000);
      return;
    }

    setEmailSubmitted(false);
    setEmailMessageTone("error");
    setEmailMessage("Could not save your email right now. Please try again.");
  };

  const handleSiteRefreshTrigger = async () => {
    if (isTriggeringSiteRefresh) {
      return;
    }

    setIsTriggeringSiteRefresh(true);
    setSiteRefreshMessage(null);
    setSiteRefreshMessageTone(null);

    try {
      const response = await fetch("/api/site-refresh/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "homepage" }),
      });

      const payload = (await response
        .json()
        .catch(() => null)) as SiteRefreshTriggerResponse | null;

      if (!response.ok || !payload?.success) {
        throw new Error(
          payload?.message || "Could not queue the site refresh workflow.",
        );
      }

      setSiteRefreshMessageTone("success");
      setSiteRefreshMessage(payload.message);
    } catch (error) {
      if (typeof window !== "undefined") {
        window.open(manualSiteRefreshUrl, "_blank", "noopener,noreferrer");
      }

      setSiteRefreshMessageTone("success");
      setSiteRefreshMessage(
        "This site is currently served from GitHub Pages, so the manual trigger opens the GitHub Actions workflow page in a new tab. Weekly scheduled runs still use the same workflow automatically.",
      );
    } finally {
      setIsTriggeringSiteRefresh(false);
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#f5f5f7] text-slate-900">
      <Navigation />

      <div className="pointer-events-none absolute inset-0 z-0">
        <div className="absolute left-1/2 top-0 h-[32rem] w-[32rem] -translate-x-1/2 rounded-full bg-white blur-3xl opacity-90" />
        <motion.div
          className="absolute h-80 w-80 rounded-full bg-sky-100/60 blur-3xl"
          style={{
            left: mousePosition.x / 12,
            top: mousePosition.y / 12,
          }}
          animate={{ opacity: [0.35, 0.55, 0.35] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      {/* Hero Section */}
      <section className="relative z-20 flex min-h-screen items-center justify-center px-6 py-24 pt-32 lg:pt-40">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          {/* Left Side - Content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="space-y-8"
          >
            {/* Name and Title with animated gradient */}
            <div className="space-y-6">
              <motion.h1
                className="text-5xl font-semibold leading-[0.92] tracking-[-0.06em] text-slate-900 sm:text-6xl lg:text-8xl"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.8 }}
                whileHover={{ scale: 1.02 }}
              >
                AAKRITI
                <br />
                GUPTA
              </motion.h1>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.8 }}
                className="space-y-3"
              >
                <div className="max-w-xl text-lg leading-8 text-slate-600">
                  Building thoughtful AI products, research-driven systems, and
                  modern interactive learning experiences.
                </div>
                <div className="space-y-4">
                  {[
                    {
                      title: "Senior ML Engineer",
                      icon: "🤖",
                      gradient: "from-pink-500 to-purple-500",
                      desc: "Building AI that matters",
                    },
                    {
                      title: "AI Researcher",
                      icon: "🔬",
                      gradient: "from-blue-500 to-cyan-500",
                      desc: "Pushing boundaries of AI",
                    },
                    {
                      title: "Luxury Tech Visionary",
                      icon: "💎",
                      gradient: "from-yellow-500 to-orange-500",
                      desc: "Where elegance meets innovation",
                    },
                  ].map((role, idx) => (
                    <motion.div
                      key={idx}
                      className="group flex items-center gap-4 rounded-3xl border border-slate-200/80 bg-white/75 p-4 shadow-[0_12px_30px_rgba(15,23,42,0.06)] backdrop-blur-xl transition-all duration-300 hover:-translate-y-0.5 hover:bg-white"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + idx * 0.1 }}
                      whileHover={{ scale: 1.01 }}
                    >
                      <motion.div
                        className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 text-lg shadow-inner"
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          delay: idx * 0.5,
                        }}
                      >
                        {role.icon}
                      </motion.div>
                      <div>
                        <p className="text-xl font-semibold text-slate-900 transition-colors">
                          {role.title}
                        </p>
                        <p className="text-sm text-slate-500 transition-colors">
                          {role.desc}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </div>

            {/* Achievement highlights */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.8 }}
              className="rounded-[2rem] border border-slate-200/80 bg-white/78 p-6 shadow-[0_18px_40px_rgba(15,23,42,0.08)] backdrop-blur-xl"
            >
              <h3 className="mb-4 flex items-center gap-2 font-semibold text-slate-900">
                <span className="text-2xl">🏆</span>
                Recognition & Achievements
              </h3>
              <div className="space-y-2 text-slate-600">
                <p>• Recognized by Dr. Yann LeCun at ICLR 2019</p>
                <p>• Building AI to make life simpler</p>
                <p>• From Delhi to Silicon Valley</p>
              </div>
            </motion.div>

            {/* Company logos */}
            <CompanyCarousel />

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2, duration: 0.8 }}
              className="flex flex-col gap-4"
            >
              <motion.a
                href="https://drive.google.com/file/d/1Mnmk6nP9l_Av0LvpgJQ5Tkjb7BqhY7nb/view?usp=sharing"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                className="group relative overflow-hidden rounded-full bg-slate-900 px-8 py-4 text-lg font-medium text-white shadow-[0_16px_36px_rgba(15,23,42,0.18)] transition-all duration-300"
              >
                <div className="relative flex items-center justify-center gap-3">
                  <span className="text-2xl">📄</span>
                  <span>Download Resume</span>
                  <span className="group-hover:translate-x-1 transition-transform duration-300">
                    →
                  </span>
                </div>
              </motion.a>

              <div className="grid grid-cols-2 gap-4">
                <motion.button
                  onClick={() => navigate("/games")}
                  whileHover={{ scale: 1.02, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  className="group relative flex items-center justify-center gap-3 overflow-hidden rounded-3xl border border-slate-200 bg-white/82 px-6 py-4 font-medium text-slate-900 shadow-sm transition-all duration-300 hover:bg-white"
                >
                  <motion.span
                    className="text-2xl relative z-10"
                    animate={{ rotate: [0, 5, -5, 0] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    🎮
                  </motion.span>
                  <span className="relative z-10 transition-colors">Games</span>
                </motion.button>

                <motion.button
                  onClick={() => navigate("/ai-playground")}
                  whileHover={{ scale: 1.02, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  className="group relative flex items-center justify-center gap-3 overflow-hidden rounded-3xl border border-slate-200 bg-white/82 px-6 py-4 font-medium text-slate-900 shadow-sm transition-all duration-300 hover:bg-white"
                >
                  <motion.span
                    className="text-2xl relative z-10"
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    🤖
                  </motion.span>
                  <span className="relative z-10">
                    Interactive Demos
                  </span>
                </motion.button>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <motion.button
                  onClick={() => navigate("/ai-discoveries")}
                  whileHover={{ scale: 1.02, y: -1 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex items-center justify-center gap-2 rounded-full border border-slate-200 bg-transparent px-6 py-3 font-medium text-slate-700 transition-all duration-300 hover:bg-white"
                >
                  <span className="text-xl">🧠</span>
                  <span>AI History</span>
                </motion.button>

                <motion.button
                  onClick={() => navigate("/ai-tools")}
                  whileHover={{ scale: 1.02, y: -1 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex items-center justify-center gap-2 rounded-full border border-slate-200 bg-transparent px-6 py-3 font-medium text-slate-700 transition-all duration-300 hover:bg-white"
                >
                  <span className="text-xl">🛠️</span>
                  <span>Pro Tools</span>
                </motion.button>
              </div>

              {/* New Prompt Engineering Button */}
              <motion.button
                onClick={() => navigate("/prompt-engineering")}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="group relative flex w-full items-center justify-center gap-3 overflow-hidden rounded-3xl border border-slate-200 bg-white/82 px-6 py-4 text-slate-900 shadow-sm transition-all duration-300 hover:bg-white"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.4 }}
              >
                <motion.span
                  className="text-2xl relative z-10"
                  animate={{
                    rotate: [0, 5, -5, 0],
                    scale: [1, 1.1, 1],
                  }}
                  transition={{ duration: 3, repeat: Infinity }}
                >
                  ✨
                </motion.span>
                <div className="relative z-10 text-center">
                  <div className="font-semibold transition-colors">
                    Prompt Engineering Mastery
                  </div>
                  <div className="text-xs text-slate-500 transition-opacity">
                    Master AI Communication
                  </div>
                </div>
              </motion.button>

              {/* AI Agent Training Button */}
              <motion.button
                onClick={() => navigate("/ai-agent-training")}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="group relative flex w-full items-center justify-center gap-3 overflow-hidden rounded-3xl border border-slate-200 bg-white/82 px-6 py-4 text-slate-900 shadow-sm transition-all duration-300 hover:bg-white"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.5 }}
              >
                <motion.span
                  className="text-2xl relative z-10"
                  animate={{
                    rotate: [0, 10, -10, 0],
                    scale: [1, 1.2, 1],
                  }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  🤖
                </motion.span>
                <div className="relative z-10 text-center">
                  <div className="font-semibold transition-colors">
                    AI Agent Training
                  </div>
                  <div className="text-xs text-slate-500 transition-opacity">
                    Build Intelligent Agents
                  </div>
                </div>
              </motion.button>

              {/* AI Champions Button */}
              <motion.button
                onClick={() => navigate("/ai-champions")}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="group relative flex w-full items-center justify-center gap-3 overflow-hidden rounded-3xl border border-slate-200 bg-white/82 px-6 py-4 text-slate-900 shadow-sm transition-all duration-300 hover:bg-white"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.6 }}
              >
                <motion.span
                  className="text-2xl relative z-10"
                  animate={{
                    rotate: [0, 10, -10, 0],
                    scale: [1, 1.2, 1],
                  }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  🏆
                </motion.span>
                <span className="relative z-10">
                  AI vs Champions
                </span>
              </motion.button>

              <motion.div
                className="rounded-[2rem] border border-slate-200 bg-white/82 p-5 shadow-sm"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.7 }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                      {siteRefreshMeta.headline}
                    </div>
                    <div className="mt-2 text-base font-semibold text-slate-900">
                      Refresh the latest AI sections and redeploy
                    </div>
                    <p className="mt-2 text-sm leading-6 text-slate-600">
                      {siteRefreshMeta.description}
                    </p>
                    <div className="mt-3 text-xs font-medium text-slate-500">
                      Last refresh template updated {siteRefreshMeta.updatedAtLabel}
                    </div>
                  </div>

                  <motion.button
                    onClick={handleSiteRefreshTrigger}
                    whileHover={{ scale: 1.02, y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    disabled={isTriggeringSiteRefresh}
                    className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-[0_14px_30px_rgba(15,23,42,0.16)] transition-all duration-300 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    {isTriggeringSiteRefresh ? "Queuing..." : "Run agent now"}
                  </motion.button>
                </div>

                {siteRefreshMessage && (
                  <div
                    className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${
                      siteRefreshMessageTone === "success"
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-rose-200 bg-rose-50 text-rose-700"
                    }`}
                  >
                    {siteRefreshMessage}
                  </div>
                )}
              </motion.div>
            </motion.div>
          </motion.div>

          {/* Right Side - Creative Photo Gallery */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
            className="relative"
          >
            <div className="mx-auto aspect-[4/5] max-w-md rounded-[2rem] border border-white/70 bg-white/80 p-4 shadow-[0_28px_80px_rgba(15,23,42,0.12)] backdrop-blur-xl">
              <PhotoGallery />
            </div>

            {/* Decorative elements */}
            <motion.div
              className="absolute -right-8 -top-8 h-32 w-32 rounded-full bg-sky-100/80 blur-3xl"
              animate={{ scale: [1, 1.08, 1] }}
              transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            />
            <motion.div
              className="absolute -bottom-8 -left-8 h-28 w-28 rounded-full bg-slate-200/80 blur-3xl"
              animate={{ scale: [1.06, 1, 1.06] }}
              transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
            />
          </motion.div>
        </div>
      </section>

      {/* Skills & Talents Showcase */}
      <section className="relative z-20 border-t border-slate-200 py-20">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="mb-6 text-5xl font-semibold text-slate-900">
              Multi-Disciplinary
              <span className="text-slate-500">
                {" "}
                Expertise
              </span>
            </h2>
            <p className="mx-auto max-w-3xl text-xl text-slate-600">
              A unique blend of technical mastery and diverse talents, from AI
              research to luxury innovation
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: "🔬",
                title: "AI Research",
                desc: "Published research, Yann LeCun recognition",
                color: "from-blue-500 to-cyan-500",
              },
              {
                icon: "💼",
                title: "Tech Leadership",
                desc: "Meta, eBay, Yahoo experience",
                color: "from-purple-500 to-pink-500",
              },
              {
                icon: "💎",
                title: "Luxury Tech",
                desc: "Founding Swarnawastra",
                color: "from-yellow-500 to-orange-500",
              },
              {
                icon: "🎯",
                title: "Marksman",
                desc: "Precision shooting expertise",
                color: "from-red-500 to-rose-500",
              },
              {
                icon: "🏇",
                title: "Equestrian",
                desc: "Professional horse riding",
                color: "from-green-500 to-emerald-500",
              },
              {
                icon: "✈️",
                title: "Aviation",
                desc: "Pilot training & certification",
                color: "from-blue-500 to-indigo-500",
              },
              {
                icon: "🏍️",
                title: "Motorcycling",
                desc: "High-performance riding",
                color: "from-gray-500 to-slate-500",
              },
              {
                icon: "🎹",
                title: "Music",
                desc: "Classical piano mastery",
                color: "from-violet-500 to-purple-500",
              },
            ].map((skill, idx) => (
              <motion.div
                key={skill.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.1, duration: 0.6 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="group cursor-pointer rounded-[2rem] border border-slate-200 bg-white/82 p-6 shadow-[0_16px_34px_rgba(15,23,42,0.06)] backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:bg-white"
              >
                <div
                  className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 text-2xl transition-transform duration-300 group-hover:scale-105"
                >
                  {skill.icon}
                </div>
                <h3 className="mb-2 text-lg font-semibold text-slate-900">
                  {skill.title}
                </h3>
                <p className="text-sm text-slate-500">{skill.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="relative z-20 border-t border-slate-200 py-20">
        <div className="max-w-4xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="mb-6 text-4xl font-semibold text-slate-900">
              Let's Build the
              <span className="text-slate-500">
                {" "}
                Future Together
              </span>
            </h2>
            <p className="text-lg text-slate-600">
              Available for consulting, speaking engagements, and collaboration
              opportunities
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            {/* Contact Links */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="space-y-6"
            >
              <motion.a
                href="https://www.linkedin.com/in/aakritigupta4894/"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02, x: 5 }}
                className="group flex items-center gap-4 rounded-[2rem] border border-slate-200 bg-white/82 p-6 shadow-[0_14px_32px_rgba(15,23,42,0.06)] transition-all duration-300 hover:-translate-y-0.5 hover:bg-white"
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-[#0071e3] text-xl font-semibold text-white transition-transform duration-300 group-hover:scale-105">
                  in
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900">LinkedIn</h3>
                  <p className="text-sm text-slate-500">
                    Professional network & experience
                  </p>
                </div>
                <div className="ml-auto text-slate-300 transition-transform duration-300 group-hover:translate-x-2 group-hover:text-[#0071e3]">
                  →
                </div>
              </motion.a>

              <motion.a
                href="https://github.com/aakritigupta0408?tab=achievements"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02, x: 5 }}
                className="group flex items-center gap-4 rounded-[2rem] border border-slate-200 bg-white/82 p-6 shadow-[0_14px_32px_rgba(15,23,42,0.06)] transition-all duration-300 hover:-translate-y-0.5 hover:bg-white"
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-900 text-lg font-semibold text-white transition-transform duration-300 group-hover:scale-105">
                  Git
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900">GitHub</h3>
                  <p className="text-sm text-slate-500">
                    Open source contributions & projects
                  </p>
                </div>
                <div className="ml-auto text-slate-300 transition-transform duration-300 group-hover:translate-x-2 group-hover:text-slate-900">
                  →
                </div>
              </motion.a>
            </motion.div>

            {/* Email Subscription */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="rounded-[2rem] border border-slate-200 bg-white/82 p-8 shadow-[0_18px_40px_rgba(15,23,42,0.08)] backdrop-blur-xl"
            >
              <h3 className="mb-4 text-xl font-semibold text-slate-900">
                Get Exclusive Updates
              </h3>
              <p className="mb-6 text-slate-600">
                Stay informed about latest AI research, projects, and speaking
                engagements
              </p>

              <form onSubmit={handleEmailSubmit} className="space-y-4">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your@email.com"
                  className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 placeholder-slate-400 transition-colors focus:border-sky-500 focus:outline-none"
                  required
                />
                <motion.button
                  type="submit"
                  disabled={emailSubmitted || isSavingEmail}
                  whileHover={{ scale: emailSubmitted || isSavingEmail ? 1 : 1.02 }}
                  whileTap={{ scale: emailSubmitted || isSavingEmail ? 1 : 0.98 }}
                  className={`w-full py-3 rounded-xl font-semibold transition-all duration-300 ${
                    emailSubmitted
                      ? "bg-green-600 text-white"
                      : isSavingEmail
                        ? "cursor-wait bg-slate-800/80 text-white"
                        : "bg-slate-900 text-white hover:bg-slate-800"
                  }`}
                >
                  {emailSubmitted
                    ? "✓ Subscribed!"
                    : isSavingEmail
                      ? "Saving..."
                      : "Subscribe"}
                </motion.button>
                {emailMessage && (
                  <p
                    className={`text-sm ${
                      emailMessageTone === "error"
                        ? "text-red-600"
                        : "text-emerald-600"
                    }`}
                  >
                    {emailMessage}
                  </p>
                )}
              </form>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-20 border-t border-slate-200 py-12">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-sm text-slate-500">
            © 2026 Aakriti Gupta • Senior ML Engineer • AI Researcher • Luxury
            Tech Visionary
          </p>
          <div className="mt-4 flex justify-center gap-8 text-xs text-slate-400">
            <span>Delhi to Silicon Valley</span>
            <span>•</span>
            <span>Meta • eBay • Yahoo</span>
            <span>•</span>
            <span>AI • Luxury • Innovation</span>
          </div>
        </div>
      </footer>

      <ChatBot />
    </div>
  );
}
