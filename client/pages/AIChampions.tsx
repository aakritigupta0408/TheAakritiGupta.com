import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";

interface AIVictory {
  id: string;
  game: string;
  aiName: string;
  champion: string;
  year: number;
  location: string;
  icon: string;
  matchScore: string;
  significance: string;
  description: string;
  aiTechnology: string[];
  gameRules: string;
  playableDemo: boolean;
  historicalContext: string;
  impact: string;
  videoUrl?: string;
  gradient: string;
  accent: string;
}

const aiVictories: AIVictory[] = [
  {
    id: "deep-blue-chess",
    game: "Chess",
    aiName: "Deep Blue",
    champion: "Garry Kasparov",
    year: 1997,
    location: "New York City",
    icon: "♛",
    matchScore: "3.5 - 2.5",
    significance:
      "First computer to defeat a reigning world chess champion in a match",
    description:
      "IBM's Deep Blue made history by becoming the first computer to defeat a reigning world chess champion in a six-game match. This watershed moment demonstrated that brute-force computation could overcome human intuition and strategic thinking in complex games.",
    aiTechnology: [
      "Specialized chess processors",
      "Alpha-beta pruning",
      "Evaluation functions",
      "Opening book",
      "Endgame databases",
    ],
    gameRules: "Standard FIDE chess rules with classical time control",
    playableDemo: true,
    historicalContext:
      "Following its 1996 loss to Kasparov, IBM upgraded Deep Blue significantly, increasing its processing power from 100 million to 200 million positions per second.",
    impact:
      "Marked the beginning of the AI era in strategy games and sparked global interest in computer chess and artificial intelligence capabilities.",
    gradient: "from-blue-600 to-indigo-800",
    accent: "border-blue-500",
  },
  {
    id: "alphago-go",
    game: "Go",
    aiName: "AlphaGo",
    champion: "Lee Sedol",
    year: 2016,
    location: "Seoul, South Korea",
    icon: "⚫",
    matchScore: "4 - 1",
    significance:
      "Conquered the ancient game of Go, thought to be decades away from AI mastery",
    description:
      "DeepMind's AlphaGo stunned the world by defeating 18-time world champion Lee Sedol in Go, a game with more possible positions than atoms in the observable universe. This victory came a decade earlier than experts predicted.",
    aiTechnology: [
      "Deep neural networks",
      "Monte Carlo tree search",
      "Reinforcement learning",
      "Self-play training",
      "Value and policy networks",
    ],
    gameRules: "Standard 19x19 Go board with Chinese rules",
    playableDemo: true,
    historicalContext:
      "Go was considered the last bastion of human supremacy in board games due to its astronomical complexity and reliance on intuition rather than calculation.",
    impact:
      "Revolutionized AI research by demonstrating that neural networks could master intuitive, pattern-recognition based games, leading to breakthroughs in many other domains.",
    gradient: "from-gray-700 to-black",
    accent: "border-gray-600",
  },
  {
    id: "libratus-poker",
    game: "No-Limit Texas Hold'em Poker",
    aiName: "Libratus",
    champion: "Top Human Professionals",
    year: 2017,
    location: "Pittsburgh, PA",
    icon: "🂡",
    matchScore: "$1.8M profit",
    significance:
      "First AI to defeat top professionals in no-limit poker, mastering imperfect information",
    description:
      "Carnegie Mellon's Libratus defeated four of the world's best no-limit Texas Hold'em players in a 20-day tournament, winning by a statistically significant margin and demonstrating AI's ability to handle incomplete information and bluffing.",
    aiTechnology: [
      "Counterfactual regret minimization",
      "Abstraction techniques",
      "Real-time strategy computation",
      "Nash equilibrium approximation",
    ],
    gameRules: "No-limit Texas Hold'em heads-up format",
    playableDemo: true,
    historicalContext:
      "Poker presented unique challenges as an imperfect information game where players cannot see opponents' cards, requiring sophisticated deception and psychological reasoning.",
    impact:
      "Showed AI could excel in scenarios involving uncertainty, deception, and incomplete information - crucial for real-world applications like negotiations and security.",
    gradient: "from-red-600 to-red-900",
    accent: "border-red-500",
  },
  {
    id: "alphazero-multiple",
    game: "Chess, Shogi & Go",
    aiName: "AlphaZero",
    champion: "Stockfish, Elmo & AlphaGo",
    year: 2017,
    location: "London, UK",
    icon: "🎯",
    matchScore: "Dominated all three",
    significance:
      "Self-taught AI that mastered three different games using only the rules",
    description:
      "DeepMind's AlphaZero learned chess, shogi, and Go from scratch using only the game rules and self-play, defeating the world's best programs in each game within hours of training, including its predecessor AlphaGo.",
    aiTechnology: [
      "Self-play reinforcement learning",
      "Deep neural networks",
      "Monte Carlo tree search",
      "No domain knowledge",
      "General game-playing architecture",
    ],
    gameRules: "Standard rules for Chess, Shogi (Japanese chess), and Go",
    playableDemo: true,
    historicalContext:
      "Unlike previous AIs that relied on human knowledge and handcrafted features, AlphaZero started with only the rules and discovered strategies through pure self-play.",
    impact:
      "Demonstrated that AI could discover novel strategies and playing styles, often superior to centuries of human knowledge, purely through self-directed learning.",
    gradient: "from-purple-600 to-purple-900",
    accent: "border-purple-500",
  },
  {
    id: "openai-five-dota",
    game: "Dota 2",
    aiName: "OpenAI Five",
    champion: "Team OG",
    year: 2019,
    location: "San Francisco, CA",
    icon: "🏆",
    matchScore: "2 - 0",
    significance:
      "First AI to defeat world champions in a complex team-based video game",
    description:
      "OpenAI Five became the first AI system to defeat the reigning world champions of Dota 2, one of the most complex esports games requiring real-time strategy, teamwork, and adaptation to millions of possible game states.",
    aiTechnology: [
      "Proximal policy optimization",
      "Long short-term memory",
      "Hierarchical reinforcement learning",
      "Self-play at scale",
      "Multi-agent coordination",
    ],
    gameRules: "Standard Dota 2 5v5 format with minor restrictions",
    playableDemo: false,
    historicalContext:
      "Dota 2 presents challenges far beyond traditional board games: real-time action, partial observability, high-dimensional action spaces, and the need for team coordination.",
    impact:
      "Proved AI could handle the complexity of modern video games and multi-agent environments, opening possibilities for AI applications in dynamic, collaborative scenarios.",
    gradient: "from-green-600 to-teal-800",
    accent: "border-green-500",
  },
  {
    id: "pluribus-poker",
    game: "Six-Player No-Limit Texas Hold'em",
    aiName: "Pluribus",
    champion: "World-Class Professionals",
    year: 2019,
    location: "Online",
    icon: "🃏",
    matchScore: "Decisive victory",
    significance:
      "First AI to defeat multiple world-class players simultaneously in multiplayer poker",
    description:
      "Facebook AI's Pluribus achieved superhuman performance in six-player no-limit Texas Hold'em poker, demonstrating AI's ability to handle the complex dynamics of multiplayer games with shifting alliances and coalitions.",
    aiTechnology: [
      "Monte Carlo counterfactual regret minimization",
      "Abstraction techniques",
      "Blueprint strategy",
      "Real-time search",
      "Multi-player game theory",
    ],
    gameRules: "Six-player no-limit Texas Hold'em tournament format",
    playableDemo: true,
    historicalContext:
      "Multiplayer poker is exponentially more complex than heads-up poker due to the need to model multiple opponents and their interactions with each other.",
    impact:
      "Extended AI poker mastery to multiplayer scenarios, relevant for auction theory, cybersecurity, and any domain involving multiple competing agents.",
    gradient: "from-yellow-600 to-orange-700",
    accent: "border-yellow-500",
  },
  {
    id: "muzero-atari",
    game: "Atari Games",
    aiName: "MuZero",
    champion: "Human World Records",
    year: 2020,
    location: "London, UK",
    icon: "🕹️",
    matchScore: "57 of 57 games",
    significance:
      "Achieved superhuman performance in classic video games without knowing the rules",
    description:
      "DeepMind's MuZero mastered chess, shogi, Go, and 57 different Atari games without being programmed with the rules of any game, learning everything through interaction and achieving superhuman performance across all domains.",
    aiTechnology: [
      "Model-based reinforcement learning",
      "Learned world models",
      "Monte Carlo tree search",
      "Deep neural networks",
      "Planning in latent space",
    ],
    gameRules: "Various - from board games to classic arcade games",
    playableDemo: true,
    historicalContext:
      "MuZero represents the culmination of combining model-free and model-based RL, learning its own internal model of game dynamics while playing.",
    impact:
      "Demonstrated that a single AI architecture could master diverse game types without domain-specific programming, pointing toward artificial general intelligence.",
    gradient: "from-cyan-600 to-blue-800",
    accent: "border-cyan-500",
  },
];

type GameTab = string | null;

export default function AIChampions() {
  const navigate = useNavigate();
  const [activeGame, setActiveGame] = useState<GameTab>(null);
  const [hoveredGame, setHoveredGame] = useState<GameTab>(null);
  const [selectedVictory, setSelectedVictory] = useState<AIVictory | null>(
    null,
  );

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
                  "linear-gradient(90deg, #000000, #dc2626, #7c3aed, #000000)",
                backgroundSize: "200% 100%",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              AI vs HUMAN
              <br />
              CHAMPIONS
            </motion.h1>
            <motion.p
              className="text-xl text-gray-100 tracking-wide max-w-4xl mx-auto mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              🏆 Witness the historic moments when artificial intelligence
              defeated world champions in games once thought impossible for
              machines to master! Experience these legendary battles firsthand!
              ⚔️
            </motion.p>

            {/* Victory Stats */}
            <motion.div
              className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <motion.div
                className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
                whileHover={{ y: -5 }}
              >
                <div className="text-3xl font-black bg-gradient-to-r from-red-400 to-red-600 bg-clip-text text-transparent">
                  {aiVictories.length}
                </div>
                <div className="text-sm text-gray-200 font-bold">
                  Historic Victories
                </div>
              </motion.div>
              <motion.div
                className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
                whileHover={{ y: -5 }}
              >
                <div className="text-3xl font-black bg-gradient-to-r from-yellow-400 to-orange-500 bg-clip-text text-transparent">
                  1997-2020
                </div>
                <div className="text-sm text-gray-200 font-bold">
                  Era of Dominance
                </div>
              </motion.div>
              <motion.div
                className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 shadow-2xl hover:scale-105 transition-all duration-300"
                whileHover={{ y: -5 }}
              >
                <div className="text-3xl font-black bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-transparent">
                  Play Now
                </div>
                <div className="text-sm text-gray-200 font-bold">
                  Interactive Demos
                </div>
              </motion.div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* AI Victories Grid */}
      <section className="relative z-20 py-16">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Historic
              <span className="bg-gradient-to-r from-red-400 to-purple-600 bg-clip-text text-transparent">
                {" "}
                AI Victories
              </span>
            </h2>
            <p className="text-lg text-gray-300 max-w-3xl mx-auto">
              Experience the pivotal moments when artificial intelligence proved
              it could surpass human champions in their own domains
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {aiVictories.map((victory, index) => (
              <motion.div
                key={victory.id}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                whileHover={{ scale: 1.02, y: -5 }}
                onHoverStart={() => setHoveredGame(victory.id)}
                onHoverEnd={() => setHoveredGame(null)}
                onClick={() => setSelectedVictory(victory)}
                className={`cursor-pointer bg-white rounded-2xl shadow-lg border-2 transition-all duration-300 overflow-hidden group ${
                  selectedVictory?.id === victory.id
                    ? `${victory.accent} shadow-2xl`
                    : hoveredGame === victory.id
                      ? "border-gray-300 shadow-xl"
                      : "border-gray-200 hover:shadow-xl"
                }`}
              >
                {/* Victory Header */}
                <div
                  className={`p-6 bg-gradient-to-r ${victory.gradient} text-white relative overflow-hidden`}
                >
                  <motion.div
                    className="absolute inset-0 bg-white/10"
                    animate={{
                      x: hoveredGame === victory.id ? [200, -200] : 200,
                    }}
                    transition={{ duration: 0.6 }}
                  />
                  <div className="relative z-10">
                    <div className="text-4xl mb-3">{victory.icon}</div>
                    <h3 className="text-xl font-bold mb-2">{victory.game}</h3>
                    <p className="text-sm mb-2 opacity-90">
                      {victory.aiName} vs {victory.champion}
                    </p>
                    <div className="flex gap-2 flex-wrap">
                      <span className="text-xs bg-white/20 px-2 py-1 rounded-full">
                        {victory.year}
                      </span>
                      <span className="text-xs bg-white/20 px-2 py-1 rounded-full">
                        {victory.matchScore}
                      </span>
                      {victory.playableDemo && (
                        <span className="text-xs bg-green-500/30 px-2 py-1 rounded-full">
                          Playable
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Victory Content */}
                <div className="p-6">
                  <p className="text-gray-600 text-sm leading-relaxed mb-4">
                    {victory.significance}
                  </p>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Location:</span>
                      <span className="font-medium text-gray-700">
                        {victory.location}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Technology:</span>
                      <span className="font-medium text-gray-700">
                        {victory.aiTechnology[0]}
                      </span>
                    </div>
                  </div>

                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className={`w-full py-3 px-4 rounded-xl font-semibold transition-all duration-300 ${
                      selectedVictory?.id === victory.id
                        ? "bg-gradient-to-r from-green-500 to-emerald-600 text-white"
                        : "bg-gradient-to-r from-gray-100 to-gray-200 text-gray-700 hover:from-gray-200 hover:to-gray-300"
                    }`}
                  >
                    {selectedVictory?.id === victory.id
                      ? "Selected"
                      : "View Details"}
                  </motion.button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Victory Details Modal */}
      <AnimatePresence>
        {selectedVictory && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedVictory(null)}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0, y: 50 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.8, opacity: 0, y: 50 }}
              className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div
                className={`p-6 bg-gradient-to-r ${selectedVictory.gradient} text-white`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <div className="text-5xl mb-3">{selectedVictory.icon}</div>
                    <h2 className="text-3xl font-bold mb-2">
                      {selectedVictory.game}
                    </h2>
                    <p className="text-xl opacity-90">
                      {selectedVictory.aiName} defeats{" "}
                      {selectedVictory.champion}
                    </p>
                    <p className="text-lg opacity-80 mt-1">
                      {selectedVictory.year} • {selectedVictory.location}
                    </p>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.1, rotate: 90 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => setSelectedVictory(null)}
                    className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition-colors"
                  >
                    <span className="text-2xl">✕</span>
                  </motion.button>
                </div>
              </div>

              {/* Modal Content */}
              <div className="p-8">
                {/* Match Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                  <div>
                    <h3 className="text-xl font-bold text-gray-800 mb-4">
                      Match Results
                    </h3>
                    <div className="bg-gray-50 p-4 rounded-xl space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Final Score:</span>
                        <span className="font-bold text-gray-800">
                          {selectedVictory.matchScore}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Year:</span>
                        <span className="font-bold text-gray-800">
                          {selectedVictory.year}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Location:</span>
                        <span className="font-bold text-gray-800">
                          {selectedVictory.location}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Playable Demo:</span>
                        <span
                          className={`font-bold ${selectedVictory.playableDemo ? "text-green-600" : "text-red-600"}`}
                        >
                          {selectedVictory.playableDemo
                            ? "Available"
                            : "Not Available"}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xl font-bold text-gray-800 mb-4">
                      AI Technology
                    </h3>
                    <div className="space-y-2">
                      {selectedVictory.aiTechnology.map((tech, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          <span className="text-gray-700 text-sm">{tech}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Description */}
                <div className="mb-8">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">
                    The Historic Victory
                  </h3>
                  <p className="text-gray-700 leading-relaxed">
                    {selectedVictory.description}
                  </p>
                </div>

                {/* Historical Context */}
                <div className="mb-8">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">
                    Historical Context
                  </h3>
                  <p className="text-gray-700 leading-relaxed">
                    {selectedVictory.historicalContext}
                  </p>
                </div>

                {/* Impact */}
                <div className="mb-8">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">
                    Impact on AI
                  </h3>
                  <p className="text-gray-700 leading-relaxed">
                    {selectedVictory.impact}
                  </p>
                </div>

                {/* Game Rules */}
                <div className="mb-8">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">
                    Game Format
                  </h3>
                  <div className="bg-blue-50 p-4 rounded-xl">
                    <p className="text-gray-700">{selectedVictory.gameRules}</p>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-4 flex-wrap">
                  {selectedVictory.playableDemo && (
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        setSelectedVictory(null);
                        setActiveGame(selectedVictory.id);
                      }}
                      className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-blue-700 transition-all"
                    >
                      🎮 Play Demo
                    </motion.button>
                  )}
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => navigate("/ai-discoveries")}
                    className="px-6 py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl font-semibold hover:from-purple-600 hover:to-purple-700 transition-all"
                  >
                    📚 Learn More About AI
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => navigate("/games")}
                    className="px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl font-semibold hover:from-green-600 hover:to-green-700 transition-all"
                  >
                    🎯 More Games
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Interactive Game Demo */}
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
                  className={`p-6 bg-gradient-to-r ${aiVictories.find((v) => v.id === activeGame)?.gradient} text-white relative`}
                  initial={{ x: -100, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-4">
                      <div className="text-4xl">
                        {aiVictories.find((v) => v.id === activeGame)?.icon}
                      </div>
                      <div>
                        <h3 className="text-2xl font-bold">
                          {aiVictories.find((v) => v.id === activeGame)?.game}
                        </h3>
                        <p className="text-white/80 text-sm">
                          Historic AI vs Human Champion Demo
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
                  className="p-8 bg-gray-50 min-h-[600px] flex items-center justify-center"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  <div className="text-center">
                    <div className="text-6xl mb-4">🚧</div>
                    <h4 className="text-2xl font-bold text-gray-800 mb-4">
                      Demo Coming Soon
                    </h4>
                    <p className="text-gray-600 max-w-2xl mx-auto">
                      Interactive demo for{" "}
                      {aiVictories.find((v) => v.id === activeGame)?.game} is
                      under development. You'll soon be able to play against the
                      same AI that defeated world champions!
                    </p>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => navigate("/games")}
                      className="mt-6 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-blue-700 transition-all"
                    >
                      Play Other Games
                    </motion.button>
                  </div>
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
                AI vs Human Champions
              </h3>
              <p className="text-gray-600 text-sm leading-relaxed">
                Explore the pivotal moments in AI history when machines
                surpassed human champions in games that defined strategic
                thinking for centuries.
              </p>
            </div>

            {/* Victory Timeline */}
            <div>
              <h3 className="text-lg font-bold text-black mb-4">
                Victory Timeline
              </h3>
              <div className="space-y-2 text-sm">
                <div className="text-gray-600">
                  1997 • Deep Blue vs Kasparov
                </div>
                <div className="text-gray-600">2016 • AlphaGo vs Lee Sedol</div>
                <div className="text-gray-600">
                  2017 • Libratus vs Poker Pros
                </div>
                <div className="text-gray-600">
                  2019 • OpenAI Five vs Team OG
                </div>
                <div className="text-gray-600">2020 • MuZero vs All Games</div>
              </div>
            </div>

            {/* Technologies */}
            <div>
              <h3 className="text-lg font-bold text-black mb-4">
                AI Technologies
              </h3>
              <div className="flex flex-wrap gap-2">
                {[
                  "Deep Learning",
                  "Monte Carlo Tree Search",
                  "Reinforcement Learning",
                  "Neural Networks",
                  "Self-Play",
                  "Game Theory",
                ].map((tech, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-xs font-medium"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Copyright */}
          <div className="text-center pt-8 border-t border-gray-200">
            <p className="text-gray-500 text-sm">
              © 2024 Aakriti Gupta • Senior ML Engineer • AI Champion Analyst
            </p>
            <div className="mt-4 flex justify-center gap-8 text-xs text-gray-400">
              <span>Deep Blue Era</span>
              <span>AlphaGo Revolution</span>
              <span>Modern AI Supremacy</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
