import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Chess from "@/components/Chess";
import BaghChal from "@/components/BaghChal";

type GameTab = "chess" | "bagh-chal";

export default function Index() {
  const [activeTab, setActiveTab] = useState<GameTab>("bagh-chal");
  const [email, setEmail] = useState("");
  const [emailSubmitted, setEmailSubmitted] = useState(false);

  const handleEmailSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      // Here you would typically send the email to your backend
      console.log("Email submitted:", email);
      setEmailSubmitted(true);
      // Reset after 3 seconds
      setTimeout(() => {
        setEmailSubmitted(false);
        setEmail("");
      }, 3000);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-yellow-50 to-orange-50 dark:from-slate-900 dark:via-purple-900 dark:to-indigo-900">
      {/* Header */}
      <div className="text-center pt-8 pb-4">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-amber-600 via-yellow-500 to-orange-500 bg-clip-text text-transparent"
        >
          TheAakritiGupta.com
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-lg md:text-xl text-slate-600 dark:text-slate-300 mt-2"
        >
          Interactive Strategy Games Portfolio
        </motion.p>
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="text-sm text-slate-500 dark:text-slate-400 mt-2 max-w-2xl mx-auto"
        >
          Experience Aakriti's professional journey through strategic games.
          Choose between chess and traditional Bagh-Chal.
        </motion.p>
      </div>

      {/* Tab Navigation */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex justify-center mb-8"
      >
        <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-2xl p-2 shadow-lg border border-white/20 dark:border-slate-700/20">
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab("chess")}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 ${
                activeTab === "chess"
                  ? "bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-lg"
                  : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              ♟️ Chess
            </button>
            <button
              onClick={() => setActiveTab("bagh-chal")}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 ${
                activeTab === "bagh-chal"
                  ? "bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-lg"
                  : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
              }`}
            >
              🐅 Bagh-Chal
            </button>
          </div>
        </div>
      </motion.div>

      {/* Game Content */}
      <div className="px-4">
        <AnimatePresence mode="wait">
          {activeTab === "chess" && (
            <motion.div
              key="chess"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.3 }}
            >
              <Chess />
            </motion.div>
          )}
          {activeTab === "bagh-chal" && (
            <motion.div
              key="bagh-chal"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <BaghChal />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Email Signup & Social Links Section */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.0 }}
        className="mt-16 max-w-4xl mx-auto px-4 pb-16"
      >
        <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20 dark:border-slate-700/20">
          <div className="text-center">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-amber-600 via-yellow-500 to-orange-500 bg-clip-text text-transparent mb-4">
              Connect with Aakriti
            </h2>
            <p className="text-slate-600 dark:text-slate-300 mb-8 max-w-2xl mx-auto">
              Get her complete resume and connect with her professional journey
              in AI, engineering, and luxury tech.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 items-center">
            {/* Email Signup */}
            <div>
              <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-200 mb-4">
                📄 Get Aakriti's Resume
              </h3>
              <form onSubmit={handleEmailSubmit} className="space-y-4">
                <div>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter your email address"
                    className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-colors"
                    required
                  />
                </div>
                <motion.button
                  type="submit"
                  disabled={emailSubmitted}
                  whileHover={{ scale: emailSubmitted ? 1 : 1.05 }}
                  whileTap={{ scale: emailSubmitted ? 1 : 0.95 }}
                  className={`w-full py-3 rounded-lg font-semibold transition-all duration-200 ${
                    emailSubmitted
                      ? "bg-green-500 text-white cursor-default"
                      : "bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white shadow-lg"
                  }`}
                >
                  {emailSubmitted ? "✅ Resume Sent!" : "Get Resume"}
                </motion.button>
              </form>
            </div>

            {/* Social Links */}
            <div>
              <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-200 mb-4">
                🔗 Professional Links
              </h3>
              <div className="space-y-4">
                <motion.a
                  href="https://www.linkedin.com/in/aakritigupta4894/"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors group"
                >
                  <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center text-white text-sm font-bold">
                    in
                  </div>
                  <div>
                    <div className="font-semibold text-slate-800 dark:text-slate-200">
                      LinkedIn Profile
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      Professional experience & network
                    </div>
                  </div>
                  <div className="ml-auto text-blue-600 group-hover:translate-x-1 transition-transform">
                    →
                  </div>
                </motion.a>

                <motion.a
                  href="https://github.com/aakritigupta0408?tab=achievements"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex items-center gap-3 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg border border-slate-200 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-700/80 transition-colors group"
                >
                  <div className="w-8 h-8 bg-slate-800 dark:bg-slate-600 rounded flex items-center justify-center text-white text-lg">
                    ⚡
                  </div>
                  <div>
                    <div className="font-semibold text-slate-800 dark:text-slate-200">
                      GitHub Profile
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      Open source projects & code
                    </div>
                  </div>
                  <div className="ml-auto text-slate-600 group-hover:translate-x-1 transition-transform">
                    →
                  </div>
                </motion.a>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
