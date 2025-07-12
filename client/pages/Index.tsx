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

// Enhanced Floating Skills Component
const FloatingSkills = () => {
  const [skills, setSkills] = useState<FloatingSkill[]>([]);

  useEffect(() => {
    const initialSkills = FLOATING_SKILLS.map((skill, index) => ({
      ...skill,
      x: Math.random() * (window.innerWidth - 300),
      y: Math.random() * (window.innerHeight - 100),
      speed: 0.4 + Math.random() * 0.6,
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

          x += Math.cos(direction) * speed;
          y += Math.sin(direction) * speed;

          if (x <= 0 || x >= window.innerWidth - 300) {
            direction = Math.PI - direction;
            x = Math.max(0, Math.min(window.innerWidth - 300, x));
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
          style={{ left: skill.x, top: skill.y }}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 0.8, scale: 1 }}
          transition={{
            duration: 2,
            delay: Math.random() * 3,
            ease: "easeOut",
          }}
        >
          <div
            className={`px-6 py-3 rounded-full bg-gradient-to-r ${skill.color} text-white text-lg font-bold shadow-2xl backdrop-blur-sm bg-opacity-90 flex items-center gap-3 border-2 border-white/30`}
          >
            <span className="text-2xl">{skill.icon}</span>
            <span className="whitespace-nowrap">{skill.label}</span>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

export default function Index() {
  const [activeGame, setActiveGame] = useState<GameTab | null>(null);
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

  const gameCards = [
    {
      id: "chess" as GameTab,
      title: "Chess Mastery",
      description:
        "Strategic chess battles revealing professional stories through captures",
      icon: "♟️",
      gradient: "from-amber-500 to-orange-500",
    },
    {
      id: "bagh-chal" as GameTab,
      title: "Bagh-Chal Strategy",
      description:
        "Traditional Nepalese game with advanced AI vs AI demonstrations",
      icon: "🐅",
      gradient: "from-green-500 to-emerald-500",
    },
    {
      id: "pacman" as GameTab,
      title: "Pacman Adventure",
      description:
        "Arcade-style game collecting diamonds that reveal professional strengths",
      icon: "🟡",
      gradient: "from-yellow-500 to-amber-500",
    },
    {
      id: "snake" as GameTab,
      title: "Snake Journey",
      description:
        "Guide the snake to discover Aakriti's professional milestones",
      icon: "🐍",
      gradient: "from-purple-500 to-pink-500",
    },
    {
      id: "mario-gradient" as GameTab,
      title: "Mario ML Adventure",
      description:
        "Learn gradient descent with Super Mario in this educational game",
      icon: "🍄",
      gradient: "from-red-500 to-pink-500",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 relative overflow-x-hidden">
      {/* Giant Floating Skills Background */}
      <FloatingSkills />

      {/* Hero Section */}
      <section className="relative z-20 min-h-screen flex items-center justify-center">
        <div className="max-w-7xl mx-auto px-4 py-16 grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Side - Text Content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="space-y-8"
          >
            <div>
              <motion.h1
                className="text-5xl md:text-7xl font-bold text-slate-900 dark:text-slate-100 mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                Aakriti Gupta
              </motion.h1>
              <motion.p
                className="text-2xl md:text-3xl text-blue-600 font-semibold mb-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                Senior ML Engineer & AI Researcher
              </motion.p>
              <motion.p
                className="text-lg text-slate-600 dark:text-slate-300 leading-relaxed"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                From Delhi to Silicon Valley • Meta, eBay, Yahoo • Recognized by
                Yann LeCun • Building Swarnawastra luxury fashion-tech • AI
                Researcher, Engineer, Horse Rider, Pilot, Shooter, Biker,
                Pianist
              </motion.p>
            </div>

            {/* Quick Links */}
            <motion.div
              className="flex flex-wrap gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
            >
              <motion.a
                href="https://www.linkedin.com/in/aakritigupta4894/"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.05 }}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold flex items-center gap-2 shadow-lg"
              >
                💼 LinkedIn
              </motion.a>
              <motion.a
                href="https://github.com/aakritigupta0408?tab=achievements"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.05 }}
                className="bg-slate-800 hover:bg-slate-900 text-white px-6 py-3 rounded-lg font-semibold flex items-center gap-2 shadow-lg"
              >
                ⚡ GitHub
              </motion.a>
              <motion.button
                onClick={() =>
                  document
                    .getElementById("games")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
                whileHover={{ scale: 1.05 }}
                className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-semibold flex items-center gap-2 shadow-lg"
              >
                🎮 Play Games
              </motion.button>
            </motion.div>
          </motion.div>

          {/* Right Side - Photo Gallery */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="relative"
          >
            <div className="grid grid-cols-2 gap-4">
              {/* Main large photo */}
              <motion.div
                className="col-span-2 relative"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
              >
                <img
                  src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2Fed0bc18cd21244e1939892616f236f8f?format=webp&width=800"
                  alt="Aakriti Gupta - Professional Portrait"
                  className="w-full h-80 object-cover rounded-2xl shadow-2xl"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent rounded-2xl" />
              </motion.div>

              {/* Smaller photos */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.3 }}
              >
                <img
                  src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F8eb1e0d8ff0f4e7e8a3cb9a919e054b1?format=webp&width=800"
                  alt="Aakriti Gupta - Professional Look"
                  className="w-full h-40 object-cover rounded-xl shadow-lg"
                />
              </motion.div>

              <motion.div
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.3 }}
              >
                <img
                  src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F84cf6b44dba445fcaeced4f15fd299f1?format=webp&width=800"
                  alt="Aakriti Gupta - Fashion Portrait"
                  className="w-full h-40 object-cover rounded-xl shadow-lg"
                />
              </motion.div>

              {/* Bottom row photos */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.3 }}
              >
                <img
                  src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F4d9e8bcd67214b5b963eb37e44602024?format=webp&width=800"
                  alt="Aakriti Gupta - Smart Casual"
                  className="w-full h-32 object-cover rounded-lg shadow-md"
                />
              </motion.div>

              <motion.div
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.3 }}
              >
                <img
                  src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2Fb2b65fd8e262453e9bd06e29a1b52798?format=webp&width=800"
                  alt="Aakriti Gupta - Elegant Style"
                  className="w-full h-32 object-cover rounded-lg shadow-md"
                />
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Professional Highlights Section */}
      <section className="relative z-20 py-20 bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-6">
              Professional Journey
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
              From Delhi classrooms to discussions with AI pioneers like Yann
              LeCun, building the future of luxury fashion with Swarnawastra.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl p-8 text-white"
            >
              <div className="text-4xl mb-4">🏢</div>
              <h3 className="text-2xl font-bold mb-4">Tech Giants</h3>
              <p className="text-lg opacity-90">
                Led ML initiatives at Meta, built ranking algorithms at eBay,
                research scientist at Yahoo serving billions of users.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl p-8 text-white"
            >
              <div className="text-4xl mb-4">🏆</div>
              <h3 className="text-2xl font-bold mb-4">Recognition</h3>
              <p className="text-lg opacity-90">
                Awarded by Dr. Yann LeCun at ICLR 2019 for innovative work on
                clustering latent representations for semi-supervised learning.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="bg-gradient-to-br from-emerald-500 to-teal-500 rounded-2xl p-8 text-white"
            >
              <div className="text-4xl mb-4">💎</div>
              <h3 className="text-2xl font-bold mb-4">Swarnawastra</h3>
              <p className="text-lg opacity-90">
                Building luxury fashion-tech brand combining AI customization,
                generative try-ons, and rare materials like gold and lab-grown
                diamonds.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Interactive Games Section */}
      <section id="games" className="relative z-20 py-20">
        <div className="max-w-7xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl font-bold text-slate-900 dark:text-slate-100 mb-6">
              🎮 Interactive Games Portfolio
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-4xl mx-auto">
              Discover Aakriti's professional journey through engaging
              interactive games. Each game reveals different aspects of her
              career, achievements, and expertise.
            </p>
          </motion.div>

          {/* Game Selection Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
            {gameCards.map((game, index) => (
              <motion.div
                key={game.id}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05 }}
                onClick={() => setActiveGame(game.id)}
                className={`cursor-pointer rounded-2xl p-8 shadow-2xl transition-all duration-300 ${
                  activeGame === game.id
                    ? "ring-4 ring-blue-500 transform scale-105"
                    : "hover:shadow-3xl"
                }`}
              >
                <div
                  className={`bg-gradient-to-br ${game.gradient} text-white rounded-2xl p-6 h-full`}
                >
                  <div className="text-5xl mb-4">{game.icon}</div>
                  <h3 className="text-2xl font-bold mb-4">{game.title}</h3>
                  <p className="text-lg opacity-90 mb-6">{game.description}</p>
                  <button className="bg-white/20 hover:bg-white/30 px-6 py-3 rounded-lg font-semibold transition-colors">
                    {activeGame === game.id ? "Playing Now" : "Play Game"}
                  </button>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Game Display Area */}
          <AnimatePresence mode="wait">
            {activeGame && (
              <motion.div
                key={activeGame}
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -50 }}
                transition={{ duration: 0.5 }}
                className="bg-white dark:bg-slate-800 rounded-3xl shadow-2xl border border-slate-200 dark:border-slate-700 overflow-hidden"
              >
                <div className="p-6 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
                  <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                    {gameCards.find((g) => g.id === activeGame)?.title}
                  </h3>
                  <button
                    onClick={() => setActiveGame(null)}
                    className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
                  >
                    <svg
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
                    </svg>
                  </button>
                </div>
                <div className="p-8">
                  {activeGame === "chess" && <Chess />}
                  {activeGame === "bagh-chal" && <BaghChal />}
                  {activeGame === "pacman" && <Pacman />}
                  {activeGame === "snake" && <Snake />}
                  {activeGame === "mario-gradient" && <MarioGradientDescent />}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Call to Action */}
          {!activeGame && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center mt-12"
            >
              <p className="text-xl text-slate-600 dark:text-slate-300 mb-6">
                Click on any game above to start playing and discover Aakriti's
                professional journey!
              </p>
              <div className="flex justify-center gap-4">
                <motion.button
                  onClick={() => setActiveGame("mario-gradient")}
                  whileHover={{ scale: 1.05 }}
                  className="bg-gradient-to-r from-red-500 to-pink-500 text-white px-8 py-4 rounded-xl font-bold text-lg shadow-lg"
                >
                  🍄 Start with Mario ML
                </motion.button>
                <motion.button
                  onClick={() => setActiveGame("chess")}
                  whileHover={{ scale: 1.05 }}
                  className="bg-gradient-to-r from-amber-500 to-orange-500 text-white px-8 py-4 rounded-xl font-bold text-lg shadow-lg"
                >
                  ♟️ Play Chess Battle
                </motion.button>
              </div>
            </motion.div>
          )}
        </div>
      </section>

      {/* Contact & Resume Section */}
      <section className="relative z-20 py-20 bg-slate-900 text-slate-100">
        <div className="max-w-7xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid lg:grid-cols-2 gap-12 items-center"
          >
            {/* Contact Information */}
            <div>
              <h2 className="text-4xl font-bold mb-8">Connect with Aakriti</h2>
              <p className="text-xl text-slate-300 mb-8 leading-relaxed">
                Senior ML Engineer based in San Jose, California. Available for
                consulting, speaking engagements, and collaboration
                opportunities in AI, luxury fashion-tech, and innovative
                technology solutions.
              </p>

              <div className="space-y-6">
                <motion.a
                  href="https://www.linkedin.com/in/aakritigupta4894/"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.02 }}
                  className="flex items-center gap-4 p-6 bg-slate-800 rounded-xl border border-slate-700 hover:border-blue-500 transition-all group"
                >
                  <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
                    <span className="text-2xl">💼</span>
                  </div>
                  <div>
                    <div className="font-bold text-slate-100 group-hover:text-blue-400 transition-colors">
                      LinkedIn Profile
                    </div>
                    <div className="text-slate-400 text-sm">
                      Professional experience & network
                    </div>
                  </div>
                </motion.a>

                <motion.a
                  href="https://github.com/aakritigupta0408?tab=achievements"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.02 }}
                  className="flex items-center gap-4 p-6 bg-slate-800 rounded-xl border border-slate-700 hover:border-green-500 transition-all group"
                >
                  <div className="w-12 h-12 bg-slate-700 rounded-lg flex items-center justify-center">
                    <span className="text-2xl">⚡</span>
                  </div>
                  <div>
                    <div className="font-bold text-slate-100 group-hover:text-green-400 transition-colors">
                      GitHub Achievements
                    </div>
                    <div className="text-slate-400 text-sm">
                      Open source projects & code
                    </div>
                  </div>
                </motion.a>
              </div>
            </div>

            {/* Resume Request Form */}
            <div className="bg-slate-800 rounded-2xl p-8 border border-slate-700">
              <h3 className="text-2xl font-bold mb-6">Get Complete Resume</h3>
              <p className="text-slate-300 mb-6">
                Request access to Aakriti's comprehensive professional resume,
                portfolio details, and project case studies.
              </p>

              <form onSubmit={handleEmailSubmit} className="space-y-6">
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
                  className={`w-full py-4 rounded-lg font-bold text-lg transition-all duration-200 ${
                    emailSubmitted
                      ? "bg-green-600 text-white cursor-default"
                      : "bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl"
                  }`}
                >
                  {emailSubmitted
                    ? "✓ Resume Sent Successfully!"
                    : "Request Complete Resume"}
                </motion.button>
              </form>
            </div>
          </motion.div>

          {/* Footer */}
          <div className="mt-16 pt-8 border-t border-slate-700 text-center">
            <p className="text-slate-400 text-lg">
              © 2024 Aakriti Gupta • Senior ML Engineer • AI Researcher •
              Swarnawastra Founder
            </p>
            <div className="mt-4 flex justify-center gap-8 text-sm text-slate-500">
              <span>🎯 Trained Shooter</span>
              <span>🐎 Horse Rider</span>
              <span>✈️ Training Pilot</span>
              <span>🏍️ Biker</span>
              <span>🎹 Pianist</span>
            </div>
          </div>
        </div>
      </section>

      {/* AI Assistant ChatBot */}
      <ChatBot />
    </div>
  );
}
