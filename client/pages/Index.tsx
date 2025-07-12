import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Chess from "@/components/Chess";
import BaghChal from "@/components/BaghChal";
import Pacman from "@/components/Pacman";

type GameTab = "chess" | "bagh-chal" | "pacman";

// Floating decorative elements
const FloatingPetals = () => {
  const petals = ["🌸", "🌺", "🌻", "🦋", "✨", "🌙"];

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {Array.from({ length: 15 }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute text-2xl opacity-70"
          style={{
            left: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 8}s`,
          }}
          animate={{
            y: [0, window.innerHeight + 100],
            x: [0, Math.random() * 100 - 50],
            rotate: [0, 360],
          }}
          transition={{
            duration: 8 + Math.random() * 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          {petals[Math.floor(Math.random() * petals.length)]}
        </motion.div>
      ))}
    </div>
  );
};

// Cute cloud decorations
const CloudDecorations = () => (
  <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
    <motion.div
      className="absolute top-20 left-10 text-6xl opacity-40"
      animate={{ x: [-20, 20], y: [-5, 5] }}
      transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
    >
      ☁️
    </motion.div>
    <motion.div
      className="absolute top-32 right-20 text-4xl opacity-30"
      animate={{ x: [20, -20], y: [5, -5] }}
      transition={{
        duration: 10,
        repeat: Infinity,
        ease: "easeInOut",
        delay: 2,
      }}
    >
      ☁️
    </motion.div>
    <motion.div
      className="absolute top-10 right-1/3 text-5xl opacity-35"
      animate={{ x: [-15, 15], y: [-8, 8] }}
      transition={{
        duration: 12,
        repeat: Infinity,
        ease: "easeInOut",
        delay: 4,
      }}
    >
      ☁️
    </motion.div>
  </div>
);

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
    <div className="min-h-screen ghibli-gradient relative overflow-hidden">
      <FloatingPetals />
      <CloudDecorations />

      {/* Main content with higher z-index */}
      <div className="relative z-10">
        {/* Header with magical styling */}
        <div className="text-center pt-12 pb-8">
          <motion.div
            initial={{ opacity: 0, y: -30, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{
              duration: 0.8,
              ease: "easeOut",
              type: "spring",
              stiffness: 100,
            }}
            className="inline-block"
          >
            <h1 className="text-4xl md:text-7xl font-bold ghibli-text-gradient mb-4 ghibli-float">
              🌸 TheAakritiGupta.com 🌸
            </h1>
            <div className="flex justify-center gap-2 text-2xl mb-4">
              <motion.span
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                ✨
              </motion.span>
              <motion.span
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
              >
                🧚‍♀️
              </motion.span>
              <motion.span
                animate={{ rotate: [0, -10, 10, 0] }}
                transition={{ duration: 2, repeat: Infinity, delay: 1 }}
              >
                ✨
              </motion.span>
            </div>
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="text-xl md:text-2xl text-purple-800 dark:text-purple-200 mt-2 font-medium"
          >
            ✨ Magical Interactive Games Portfolio ✨
          </motion.p>

          <motion.p
            initial={{ opacity: 0, y: -15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="text-lg text-pink-700 dark:text-pink-300 mt-3 max-w-3xl mx-auto font-light"
          >
            🌟 Journey through Aakriti's enchanted professional world! Play
            whimsical games and discover her magical talents! 🌟
          </motion.p>
        </div>

        {/* Cute Tab Navigation */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="flex justify-center mb-12"
        >
          <div className="ghibli-glass rounded-3xl p-3 shadow-2xl border-2 border-white/30">
            <div className="flex gap-3">
              <motion.button
                onClick={() => setActiveTab("chess")}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className={`px-8 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 ${
                  activeTab === "chess"
                    ? "ghibli-button text-white shadow-lg transform -translate-y-1"
                    : "text-purple-700 dark:text-purple-300 hover:bg-white/20 hover:text-purple-600"
                }`}
              >
                ♟️ Chess
              </motion.button>
              <motion.button
                onClick={() => setActiveTab("bagh-chal")}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className={`px-8 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 ${
                  activeTab === "bagh-chal"
                    ? "ghibli-button text-white shadow-lg transform -translate-y-1"
                    : "text-purple-700 dark:text-purple-300 hover:bg-white/20 hover:text-purple-600"
                }`}
              >
                🐅 Bagh-Chal
              </motion.button>
              <motion.button
                onClick={() => setActiveTab("pacman")}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className={`px-8 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 ${
                  activeTab === "pacman"
                    ? "ghibli-button text-white shadow-lg transform -translate-y-1"
                    : "text-purple-700 dark:text-purple-300 hover:bg-white/20 hover:text-purple-600"
                }`}
              >
                🟡 Pacman
              </motion.button>
            </div>
          </div>
        </motion.div>

        {/* Game Content with magical transitions */}
        <div className="px-6">
          <AnimatePresence mode="wait">
            {activeTab === "chess" && (
              <motion.div
                key="chess"
                initial={{ opacity: 0, x: -50, scale: 0.9 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 50, scale: 0.9 }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
              >
                <Chess />
              </motion.div>
            )}
            {activeTab === "bagh-chal" && (
              <motion.div
                key="bagh-chal"
                initial={{ opacity: 0, y: 50, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -50, scale: 0.9 }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
              >
                <BaghChal />
              </motion.div>
            )}
            {activeTab === "pacman" && (
              <motion.div
                key="pacman"
                initial={{ opacity: 0, x: 50, scale: 0.9 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: -50, scale: 0.9 }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
              >
                <Pacman />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Magical Email Signup & Social Links Section */}
        <motion.div
          initial={{ opacity: 0, y: 80 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2, duration: 0.8 }}
          className="mt-20 max-w-5xl mx-auto px-6 pb-20"
        >
          <div className="ghibli-glass rounded-3xl p-10 shadow-2xl border-2 border-white/30 backdrop-blur-xl">
            <div className="text-center mb-10">
              <motion.h2
                className="text-4xl font-bold ghibli-text-gradient mb-6 ghibli-float"
                animate={{ scale: [1, 1.02, 1] }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                🌸 Connect with Aakriti 🌸
              </motion.h2>
              <p className="text-purple-700 dark:text-purple-300 text-lg mb-4 max-w-3xl mx-auto">
                ✨ Discover her magical journey through AI, engineering, and
                luxury tech! ✨
              </p>
              <div className="flex justify-center gap-4 text-2xl">
                <motion.span
                  animate={{ y: [0, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  🦋
                </motion.span>
                <motion.span
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 4, repeat: Infinity }}
                >
                  🌟
                </motion.span>
                <motion.span
                  animate={{ y: [0, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity, delay: 1 }}
                >
                  🦋
                </motion.span>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-10 items-center">
              {/* Cute Email Signup */}
              <motion.div
                className="ghibli-glass rounded-2xl p-8 border border-white/30"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
              >
                <h3 className="text-2xl font-semibold text-purple-800 dark:text-purple-200 mb-6 text-center">
                  📄 Get Aakriti's Magical Resume! 📄
                </h3>
                <form onSubmit={handleEmailSubmit} className="space-y-6">
                  <motion.div whileFocus={{ scale: 1.02 }}>
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="✨ Enter your magical email address ✨"
                      className="w-full px-6 py-4 border-2 border-pink-200 dark:border-pink-600 rounded-2xl bg-white/80 dark:bg-purple-900/50 text-purple-900 dark:text-purple-100 focus:ring-4 focus:ring-pink-300 focus:border-pink-400 transition-all text-center placeholder-pink-400"
                      required
                    />
                  </motion.div>
                  <motion.button
                    type="submit"
                    disabled={emailSubmitted}
                    whileHover={{
                      scale: emailSubmitted ? 1 : 1.05,
                      y: emailSubmitted ? 0 : -2,
                    }}
                    whileTap={{ scale: emailSubmitted ? 1 : 0.95 }}
                    className={`w-full py-4 rounded-2xl font-bold text-lg transition-all duration-300 ${
                      emailSubmitted
                        ? "bg-green-400 text-white cursor-default border-2 border-green-300"
                        : "ghibli-button text-white shadow-xl hover:shadow-2xl"
                    }`}
                  >
                    {emailSubmitted
                      ? "✅ Magic Sent! 🌟"
                      : "🌸 Summon Resume 🌸"}
                  </motion.button>
                </form>
              </motion.div>

              {/* Whimsical Social Links */}
              <motion.div
                className="ghibli-glass rounded-2xl p-8 border border-white/30"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
              >
                <h3 className="text-2xl font-semibold text-purple-800 dark:text-purple-200 mb-6 text-center">
                  🔗 Magical Professional Links 🔗
                </h3>
                <div className="space-y-6">
                  <motion.a
                    href="https://www.linkedin.com/in/aakritigupta4894/"
                    target="_blank"
                    rel="noopener noreferrer"
                    whileHover={{ scale: 1.05, x: 5 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-4 p-5 bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/40 dark:to-purple-900/40 rounded-2xl border-2 border-blue-200 dark:border-blue-700 hover:border-blue-300 transition-all group shadow-lg"
                  >
                    <div className="w-12 h-12 bg-blue-500 rounded-2xl flex items-center justify-center text-white text-xl font-bold shadow-lg">
                      💼
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-purple-800 dark:text-purple-200 text-lg">
                        LinkedIn Kingdom
                      </div>
                      <div className="text-purple-600 dark:text-purple-300">
                        Professional magic & network ✨
                      </div>
                    </div>
                    <motion.div
                      className="text-2xl text-blue-500"
                      animate={{ x: [0, 5, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      🦋
                    </motion.div>
                  </motion.a>

                  <motion.a
                    href="https://github.com/aakritigupta0408?tab=achievements"
                    target="_blank"
                    rel="noopener noreferrer"
                    whileHover={{ scale: 1.05, x: 5 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-4 p-5 bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/40 dark:to-pink-900/40 rounded-2xl border-2 border-purple-200 dark:border-purple-700 hover:border-purple-300 transition-all group shadow-lg"
                  >
                    <div className="w-12 h-12 bg-purple-600 rounded-2xl flex items-center justify-center text-white text-xl shadow-lg">
                      ⚡
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-purple-800 dark:text-purple-200 text-lg">
                        GitHub Enchantments
                      </div>
                      <div className="text-purple-600 dark:text-purple-300">
                        Open source spells & code 🌟
                      </div>
                    </div>
                    <motion.div
                      className="text-2xl text-purple-500"
                      animate={{ rotate: [0, 360] }}
                      transition={{ duration: 3, repeat: Infinity }}
                    >
                      ✨
                    </motion.div>
                  </motion.a>
                </div>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
