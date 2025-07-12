import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Chess from "@/components/Chess";
import BaghChal from "@/components/BaghChal";
import Pacman from "@/components/Pacman";
import Snake from "@/components/Snake";
import MarioGradientDescent from "@/components/MarioGradientDescent";
import ChatBot from "@/components/ChatBot";

type GameTab = "chess" | "bagh-chal" | "pacman" | "snake" | "mario-gradient";

interface FloatingSkill {
  id: string;
  label: string;
  icon: string;
  color: string;
  x: number;
  y: number;
  speed: number;
  direction: number;
}

const FLOATING_SKILLS: Omit<
  FloatingSkill,
  "x" | "y" | "speed" | "direction"
>[] = [
  {
    id: "ai-researcher",
    label: "AI Researcher",
    icon: "🧠",
    color: "from-blue-500 to-cyan-500",
  },
  {
    id: "engineer",
    label: "Engineer",
    icon: "⚙️",
    color: "from-gray-600 to-slate-600",
  },
  {
    id: "horse-rider",
    label: "Horse Rider",
    icon: "🐎",
    color: "from-amber-600 to-yellow-600",
  },
  {
    id: "pilot",
    label: "Training Pilot",
    icon: "✈️",
    color: "from-sky-500 to-blue-500",
  },
  {
    id: "shooter",
    label: "Trained Shooter",
    icon: "🎯",
    color: "from-red-500 to-orange-500",
  },
  {
    id: "biker",
    label: "Biker",
    icon: "🏍️",
    color: "from-green-600 to-emerald-600",
  },
  {
    id: "pianist",
    label: "Pianist",
    icon: "🎹",
    color: "from-purple-500 to-violet-500",
  },
];

// Floating Skills Component
const FloatingSkills = () => {
  const [skills, setSkills] = useState<FloatingSkill[]>([]);

  useEffect(() => {
    // Initialize skills with random positions and movement properties
    const initialSkills = FLOATING_SKILLS.map((skill, index) => ({
      ...skill,
      x: Math.random() * (window.innerWidth - 200),
      y: Math.random() * (window.innerHeight - 100),
      speed: 0.3 + Math.random() * 0.4, // 0.3 to 0.7
      direction: Math.random() * Math.PI * 2,
    }));
    setSkills(initialSkills);
  }, []);

  useEffect(() => {
    if (skills.length === 0) return;

    const animateSkills = () => {
      setSkills((prevSkills) =>
        prevSkills.map((skill) => {
          let { x, y, direction, speed } = skill;

          // Move in current direction
          x += Math.cos(direction) * speed;
          y += Math.sin(direction) * speed;

          // Bounce off edges
          if (x <= 0 || x >= window.innerWidth - 200) {
            direction = Math.PI - direction;
            x = Math.max(0, Math.min(window.innerWidth - 200, x));
          }
          if (y <= 0 || y >= window.innerHeight - 100) {
            direction = -direction;
            y = Math.max(0, Math.min(window.innerHeight - 100, y));
          }

          return { ...skill, x, y, direction };
        }),
      );
    };

    const interval = setInterval(animateSkills, 50);
    return () => clearInterval(interval);
  }, [skills.length]);

  return (
    <div className="fixed inset-0 pointer-events-none z-10 overflow-hidden">
      {skills.map((skill) => (
        <motion.div
          key={skill.id}
          className="absolute"
          style={{
            left: skill.x,
            top: skill.y,
          }}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 0.7, scale: 1 }}
          transition={{
            duration: 2,
            delay: Math.random() * 3,
            ease: "easeOut",
          }}
        >
          <div
            className={`px-4 py-2 rounded-full bg-gradient-to-r ${skill.color} text-white text-sm font-medium shadow-lg backdrop-blur-sm bg-opacity-90 flex items-center gap-2 border border-white/20`}
          >
            <span className="text-lg">{skill.icon}</span>
            <span className="whitespace-nowrap">{skill.label}</span>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

export default function Index() {
  const [activeTab, setActiveTab] = useState<GameTab>("bagh-chal");
  const [email, setEmail] = useState("");
  const [emailSubmitted, setEmailSubmitted] = useState(false);

  const handleEmailSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      console.log("Email submitted:", email);
      setEmailSubmitted(true);
      setTimeout(() => {
        setEmailSubmitted(false);
        setEmail("");
      }, 3000);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 relative">
      {/* Floating Skills Background */}
      <FloatingSkills />
      {/* Professional Header */}
      <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg border-b border-slate-200 dark:border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-slate-100 mb-2">
              Aakriti Gupta
            </h1>
            <p className="text-xl text-slate-600 dark:text-slate-300 font-medium">
              AI Engineer & Technology Leader
            </p>
            <p className="text-lg text-slate-500 dark:text-slate-400 mt-2">
              Interactive Portfolio & Professional Games
            </p>
          </motion.div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <motion.nav
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.6 }}
        className="max-w-4xl mx-auto px-4 py-8"
      >
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700 p-2">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            <button
              onClick={() => setActiveTab("chess")}
              className={`px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                activeTab === "chess"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              ♟️ Chess
            </button>
            <button
              onClick={() => setActiveTab("bagh-chal")}
              className={`px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                activeTab === "bagh-chal"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              🐅 Bagh-Chal
            </button>
            <button
              onClick={() => setActiveTab("pacman")}
              className={`px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                activeTab === "pacman"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              🟡 Pacman
            </button>
            <button
              onClick={() => setActiveTab("snake")}
              className={`px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                activeTab === "snake"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              🐍 Snake
            </button>
            <button
              onClick={() => setActiveTab("mario-gradient")}
              className={`px-6 py-4 rounded-lg font-semibold transition-all duration-200 ${
                activeTab === "mario-gradient"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              🍄 Mario ML
            </button>
          </div>
        </div>
      </motion.nav>

      {/* Game Content */}
      <main className="max-w-7xl mx-auto px-4 pb-8">
        <AnimatePresence mode="wait">
          {activeTab === "chess" && (
            <motion.div
              key="chess"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Chess />
            </motion.div>
          )}
          {activeTab === "bagh-chal" && (
            <motion.div
              key="bagh-chal"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <BaghChal />
            </motion.div>
          )}
          {activeTab === "pacman" && (
            <motion.div
              key="pacman"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Pacman />
            </motion.div>
          )}
          {activeTab === "snake" && (
            <motion.div
              key="snake"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Snake />
            </motion.div>
          )}
          {activeTab === "mario-gradient" && (
            <motion.div
              key="mario-gradient"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <MarioGradientDescent />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Professional Contact & Links Section */}
      <footer className="bg-slate-900 dark:bg-slate-950 text-slate-100">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="max-w-7xl mx-auto px-4 py-16"
        >
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Professional Summary */}
            <div>
              <h2 className="text-3xl font-bold mb-6">Connect with Aakriti</h2>
              <p className="text-slate-300 text-lg mb-8 leading-relaxed">
                Experienced AI Engineer and Technology Leader with expertise in
                machine learning, large-scale systems, and innovative product
                development. Proven track record at Meta, eBay, Yahoo, and as a
                successful entrepreneur.
              </p>

              <div className="space-y-4">
                <motion.a
                  href="https://www.linkedin.com/in/aakritigupta4894/"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.02 }}
                  className="flex items-center gap-4 p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-blue-500 transition-all group"
                >
                  <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
                    <svg
                      className="w-6 h-6 text-white"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                    </svg>
                  </div>
                  <div>
                    <div className="font-semibold text-slate-100 group-hover:text-blue-400 transition-colors">
                      LinkedIn Profile
                    </div>
                    <div className="text-slate-400 text-sm">
                      Professional experience & network
                    </div>
                  </div>
                  <div className="ml-auto text-slate-400 group-hover:text-blue-400 transition-colors">
                    →
                  </div>
                </motion.a>

                <motion.a
                  href="https://github.com/aakritigupta0408?tab=achievements"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.02 }}
                  className="flex items-center gap-4 p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-green-500 transition-all group"
                >
                  <div className="w-12 h-12 bg-slate-700 rounded-lg flex items-center justify-center">
                    <svg
                      className="w-6 h-6 text-white"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                    </svg>
                  </div>
                  <div>
                    <div className="font-semibold text-slate-100 group-hover:text-green-400 transition-colors">
                      GitHub Profile
                    </div>
                    <div className="text-slate-400 text-sm">
                      Open source projects & code
                    </div>
                  </div>
                  <div className="ml-auto text-slate-400 group-hover:text-green-400 transition-colors">
                    →
                  </div>
                </motion.a>
              </div>
            </div>

            {/* Resume Request */}
            <div className="bg-slate-800 rounded-xl p-8 border border-slate-700">
              <h3 className="text-2xl font-bold mb-4">Get Resume</h3>
              <p className="text-slate-300 mb-6">
                Request access to Aakriti's complete professional resume and
                portfolio details.
              </p>

              <form onSubmit={handleEmailSubmit} className="space-y-4">
                <div>
                  <label
                    htmlFor="email"
                    className="block text-sm font-medium text-slate-300 mb-2"
                  >
                    Business Email Address
                  </label>
                  <input
                    type="email"
                    id="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="your.email@company.com"
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    required
                  />
                </div>
                <motion.button
                  type="submit"
                  disabled={emailSubmitted}
                  whileHover={{ scale: emailSubmitted ? 1 : 1.02 }}
                  whileTap={{ scale: emailSubmitted ? 1 : 0.98 }}
                  className={`w-full py-3 rounded-lg font-semibold transition-all duration-200 ${
                    emailSubmitted
                      ? "bg-green-600 text-white cursor-default"
                      : "bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl"
                  }`}
                >
                  {emailSubmitted ? "✓ Resume Sent" : "Request Resume"}
                </motion.button>
              </form>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-16 pt-8 border-t border-slate-700 text-center">
            <p className="text-slate-400">
              © 2024 Aakriti Gupta. All rights reserved.
            </p>
          </div>
        </motion.div>
      </footer>

      {/* AI Assistant ChatBot */}
      <ChatBot />
    </div>
  );
}
