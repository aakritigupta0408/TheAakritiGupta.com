import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import Chess from "@/components/Chess";
import BaghChal from "@/components/BaghChal";
import Pacman from "@/components/Pacman";
import Snake from "@/components/Snake";
import MarioGradientDescent from "@/components/MarioGradientDescent";
import Helicopter from "@/components/Helicopter";

type GameTab =
  | "chess"
  | "bagh-chal"
  | "pacman"
  | "snake"
  | "mario-gradient"
  | "helicopter";

export default function Games() {
  const navigate = useNavigate();
  const [activeGame, setActiveGame] = useState<GameTab | null>(null);
  const [hoveredGame, setHoveredGame] = useState<GameTab | null>(null);

  const gameCards = [
    {
      id: "chess" as GameTab,
      title: "CHESS",
      description:
        "Strategic mastery revealing professional narratives through the royal game",
      icon: "♔",
      accent: "border-yellow-400",
      gradient: "from-yellow-400 to-amber-600",
      difficulty: "Expert",
      players: "2 Players",
      category: "Strategy",
    },
    {
      id: "bagh-chal" as GameTab,
      title: "BAGH-CHAL",
      description: "Traditional Nepali strategy with modern AI sophistication",
      icon: "🐅",
      accent: "border-orange-400",
      gradient: "from-orange-400 to-red-600",
      difficulty: "Advanced",
      players: "2 Players",
      category: "Traditional",
    },
    {
      id: "pacman" as GameTab,
      title: "PACMAN",
      description:
        "Arcade adventure unveiling professional strengths and achievements",
      icon: "👾",
      accent: "border-blue-400",
      gradient: "from-blue-400 to-purple-600",
      difficulty: "Medium",
      players: "1 Player",
      category: "Arcade",
    },
    {
      id: "snake" as GameTab,
      title: "SNAKE",
      description: "Journey through professional milestones and career growth",
      icon: "🐍",
      accent: "border-green-400",
      gradient: "from-green-400 to-emerald-600",
      difficulty: "Easy",
      players: "1 Player",
      category: "Classic",
    },
    {
      id: "mario-gradient" as GameTab,
      title: "GRADIENT DESCENT",
      description:
        "Machine learning education through interactive Mario-style gameplay",
      icon: "🎮",
      accent: "border-purple-400",
      gradient: "from-purple-400 to-pink-600",
      difficulty: "Expert",
      players: "1 Player",
      category: "Educational",
    },
    {
      id: "helicopter" as GameTab,
      title: "HELICOPTER",
      description:
        "Navigate challenges to discover achievements and unlock career milestones",
      icon: "🚁",
      accent: "border-cyan-400",
      gradient: "from-cyan-400 to-blue-600",
      difficulty: "Hard",
      players: "1 Player",
      category: "Action",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-black relative overflow-x-hidden">
      {/* Enhanced Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute top-10 left-10 w-72 h-72 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-20 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-bounce"></div>
        <div className="absolute bottom-20 left-1/4 w-80 h-80 bg-pink-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-60 left-1/2 w-64 h-64 bg-cyan-500/20 rounded-full blur-3xl animate-bounce delay-500"></div>
        <div className="absolute bottom-10 right-10 w-88 h-88 bg-yellow-500/20 rounded-full blur-3xl animate-pulse delay-700"></div>
      </div>

      <Navigation />

      {/* Hero Section */}
      <section className="relative z-20 pt-32 pb-16">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <motion.h1
              className="text-6xl md:text-8xl font-bold text-black mb-8"
              animate={{
                backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
              }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              style={{
                background:
                  "linear-gradient(90deg, #000000, #2563eb, #7c3aed, #000000)",
                backgroundSize: "200% 100%",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              INTERACTIVE
              <br />
              GAMES
            </motion.h1>
            <motion.p
              className="text-xl text-gray-100 tracking-wide max-w-4xl mx-auto mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              🎮 Discover professional mastery through sophisticated gameplay
              experiences! Ready to challenge yourself? ✨
            </motion.p>

            {/* Gaming Stats */}
            <motion.div
              className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <motion.div
                className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
                whileHover={{ y: -5 }}
              >
                <div className="text-3xl font-black bg-gradient-to-r from-blue-400 to-cyan-500 bg-clip-text text-transparent">
                  6
                </div>
                <div className="text-sm text-gray-200 font-bold">
                  Interactive Games
                </div>
              </motion.div>
              <motion.div
                className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
                whileHover={{ y: -5 }}
              >
                <div className="text-3xl font-black bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
                  AI-Powered
                </div>
                <div className="text-sm text-gray-200 font-bold">
                  Smart Gameplay
                </div>
              </motion.div>
              <motion.div
                className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
                whileHover={{ y: -5 }}
              >
                <div className="text-3xl font-black bg-gradient-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent">
                  Portfolio
                </div>
                <div className="text-sm text-gray-200 font-bold">
                  Integration
                </div>
              </motion.div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Game Selection Grid */}
      <section className="relative z-20 py-16">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-black mb-6">
              Choose Your
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                {" "}
                Adventure
              </span>
            </h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Select from our collection of interactive games that showcase
              different aspects of strategic thinking and problem-solving
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {gameCards.map((game, index) => (
              <motion.div
                key={game.id}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                whileHover={{ scale: 1.02, y: -5 }}
                onHoverStart={() => setHoveredGame(game.id)}
                onHoverEnd={() => setHoveredGame(null)}
                onClick={() => setActiveGame(game.id)}
                className={`cursor-pointer bg-white rounded-2xl shadow-lg border-2 transition-all duration-300 overflow-hidden group ${
                  activeGame === game.id
                    ? `${game.accent} shadow-2xl`
                    : hoveredGame === game.id
                      ? "border-gray-300 shadow-xl"
                      : "border-gray-200 hover:shadow-xl"
                }`}
              >
                {/* Game Header */}
                <div
                  className={`p-6 bg-gradient-to-r ${game.gradient} text-white relative overflow-hidden`}
                >
                  <motion.div
                    className="absolute inset-0 bg-white/10"
                    animate={{
                      x: hoveredGame === game.id ? [200, -200] : 200,
                    }}
                    transition={{ duration: 0.6 }}
                  />
                  <div className="relative z-10">
                    <div className="text-4xl mb-3">{game.icon}</div>
                    <h3 className="text-xl font-bold mb-2">{game.title}</h3>
                    <div className="flex gap-2 flex-wrap">
                      <span className="text-xs bg-white/20 px-2 py-1 rounded-full">
                        {game.category}
                      </span>
                      <span className="text-xs bg-white/20 px-2 py-1 rounded-full">
                        {game.difficulty}
                      </span>
                      <span className="text-xs bg-white/20 px-2 py-1 rounded-full">
                        {game.players}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Game Content */}
                <div className="p-6">
                  <p className="text-gray-600 text-sm leading-relaxed mb-4">
                    {game.description}
                  </p>

                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className={`w-full py-3 px-4 rounded-xl font-semibold transition-all duration-300 ${
                      activeGame === game.id
                        ? "bg-gradient-to-r from-green-500 to-emerald-600 text-white"
                        : "bg-gradient-to-r from-gray-100 to-gray-200 text-gray-700 hover:from-gray-200 hover:to-gray-300"
                    }`}
                  >
                    {activeGame === game.id ? "Currently Playing" : "Play Game"}
                  </motion.button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Game Display Area */}
      <AnimatePresence mode="wait">
        {activeGame && (
          <section className="relative z-20 py-16">
            <div className="max-w-7xl mx-auto px-8">
              <motion.div
                key={activeGame}
                initial={{ opacity: 0, y: 60, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -60, scale: 0.95 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden"
              >
                {/* Game Header */}
                <motion.div
                  className={`p-6 bg-gradient-to-r ${gameCards.find((g) => g.id === activeGame)?.gradient} text-white relative`}
                  initial={{ x: -100, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-4">
                      <div className="text-4xl">
                        {gameCards.find((g) => g.id === activeGame)?.icon}
                      </div>
                      <div>
                        <h3 className="text-2xl font-bold">
                          {gameCards.find((g) => g.id === activeGame)?.title}
                        </h3>
                        <p className="text-white/80 text-sm">
                          {gameCards.find((g) => g.id === activeGame)?.category}{" "}
                          Game
                        </p>
                      </div>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.1, rotate: 90 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => setActiveGame(null)}
                      className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition-colors"
                    >
                      <span className="text-xl">✕</span>
                    </motion.button>
                  </div>
                </motion.div>

                {/* Game Content */}
                <motion.div
                  className="p-8 bg-gray-50 min-h-[600px]"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  {activeGame === "chess" && <Chess />}
                  {activeGame === "bagh-chal" && <BaghChal />}
                  {activeGame === "pacman" && <Pacman />}
                  {activeGame === "snake" && <Snake />}
                  {activeGame === "mario-gradient" && <MarioGradientDescent />}
                  {activeGame === "helicopter" && <Helicopter />}
                </motion.div>
              </motion.div>
            </div>
          </section>
        )}
      </AnimatePresence>

      {/* Enhanced Footer */}
      <section className="relative z-20 py-16 bg-gray-100 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            {/* About Section */}
            <div>
              <h3 className="text-lg font-bold text-black mb-4">
                Interactive Gaming Portfolio
              </h3>
              <p className="text-gray-600 text-sm leading-relaxed">
                Experience strategic thinking and problem-solving through
                carefully crafted games that showcase professional expertise.
              </p>
            </div>

            {/* Game Categories */}
            <div>
              <h3 className="text-lg font-bold text-black mb-4">
                Game Categories
              </h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <span className="text-gray-600">Strategy</span>
                <span className="text-gray-600">Arcade</span>
                <span className="text-gray-600">Educational</span>
                <span className="text-gray-600">Traditional</span>
                <span className="text-gray-600">Action</span>
                <span className="text-gray-600">Classic</span>
              </div>
            </div>

            {/* Skills Showcase */}
            <div>
              <h3 className="text-lg font-bold text-black mb-4">
                Skills Demonstrated
              </h3>
              <div className="flex flex-wrap gap-2">
                {[
                  "Strategic Thinking",
                  "Problem Solving",
                  "AI/ML",
                  "Game Theory",
                  "Pattern Recognition",
                ].map((skill, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Copyright */}
          <div className="text-center pt-8 border-t border-gray-200">
            <p className="text-gray-500 text-sm">
              © 2024 Aakriti Gupta • Senior ML Engineer • Game Developer
            </p>
            <div className="mt-4 flex justify-center gap-8 text-xs text-gray-400">
              <span>Marksman</span>
              <span>Equestrian</span>
              <span>Aviator</span>
              <span>Motorcyclist</span>
              <span>Pianist</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
