import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import LevelOneLoadMoreButton from "@/components/LevelOneLoadMoreButton";
import Chess from "@/components/Chess";
import BaghChal from "@/components/BaghChal";
import Pacman from "@/components/Pacman";
import Snake from "@/components/Snake";
import MarioGradientDescent from "@/components/MarioGradientDescent";
import Helicopter from "@/components/Helicopter";
import SubpageLayout from "@/components/SubpageLayout";
import { getPageRefreshContent } from "@/data/siteRefreshContent";

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
  const [visibleCount, setVisibleCount] = useState(6);

  const gameCards = [
    {
      id: "chess" as GameTab,
      title: "CHESS",
      description:
        "Play against Aakriti's AI — capture pieces to unlock her professional story. White is her portfolio, black is yours.",
      icon: "♔",
      accent: "border-yellow-400",
      gradient: "from-yellow-400 to-amber-600",
      difficulty: "Expert",
      players: "1 vs AI",
      category: "Strategy",
    },
    {
      id: "bagh-chal" as GameTab,
      title: "BAGH-CHAL",
      description:
        "Classical Nepali board game pitting 4 tigers against 20 goats. Tigers hunt; goats surround. Asymmetric strategy at its sharpest.",
      icon: "🐅",
      accent: "border-orange-400",
      gradient: "from-orange-400 to-red-600",
      difficulty: "Advanced",
      players: "2 Players",
      category: "Traditional",
    },
    {
      id: "pacman" as GameTab,
      title: "PAC-MAN",
      description:
        "Classic maze chase with a twist — collect 💎 diamonds hidden in the maze to reveal Aakriti's professional achievements.",
      icon: "🟡",
      accent: "border-blue-400",
      gradient: "from-blue-400 to-purple-600",
      difficulty: "Medium",
      players: "1 Player",
      category: "Arcade",
    },
    {
      id: "snake" as GameTab,
      title: "SNAKE",
      description:
        "Guide the snake through milestones — each fruit you collect unlocks a chapter from Aakriti's career journey.",
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
        "Watch Mario optimize a loss function in real time. Adjust the learning rate and see how gradient descent finds the global minimum.",
      icon: "🎮",
      accent: "border-purple-400",
      gradient: "from-purple-400 to-pink-600",
      difficulty: "Educational",
      players: "1 Player",
      category: "ML Demo",
    },
    {
      id: "helicopter" as GameTab,
      title: "HELICOPTER",
      description:
        "Fly through obstacles and collect achievements. Each gold item you reach reveals a career milestone.",
      icon: "🚁",
      accent: "border-cyan-400",
      gradient: "from-cyan-400 to-blue-600",
      difficulty: "Hard",
      players: "1 Player",
      category: "Action",
    },
  ];

  const categoryCount = new Set(gameCards.map((game) => game.category)).size;
  const visibleGames = gameCards.slice(0, visibleCount);
  const hasMoreGames = visibleCount < gameCards.length;
  const pageRefresh = getPageRefreshContent("/games");

  return (
    <SubpageLayout
      route="/games"
      eyebrow={pageRefresh.eyebrow}
      title={pageRefresh.title}
      description={pageRefresh.description}
      accent="amber"
      chips={pageRefresh.chips}
      metrics={[
        {
          value: gameCards.length.toString(),
          label: "Playable games",
        },
        {
          value: categoryCount.toString(),
          label: "Categories",
        },
      ]}
    >

      {/* Game Selection Grid */}
      <section className="relative z-20 py-6 pt-4">
        <div className="max-w-7xl mx-auto px-8">
          <div className="mx-auto mb-10 flex max-w-5xl flex-col gap-4 rounded-3xl border border-white/15 bg-white/10 p-5 text-center backdrop-blur-xl md:flex-row md:items-center md:justify-between md:text-left">
            <div>
              <p className="text-sm font-bold uppercase tracking-[0.2em] text-amber-100">
                Playable collection
              </p>
              <p className="mt-2 text-lg font-semibold text-white">
                Showing {visibleGames.length} portfolio games across {categoryCount} categories
              </p>
            </div>
            <p className="text-sm text-slate-300 md:max-w-md">
              Pick a game, play it in the browser, then jump back for the next.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {visibleGames.map((game, index) => (
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
                  className={`p-4 bg-gradient-to-r ${game.gradient} text-white relative overflow-hidden`}
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
                <div className="p-4">
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

          {hasMoreGames && (
            <LevelOneLoadMoreButton
              label="Show 3 more games"
              onClick={() => setVisibleCount((current) => current + 3)}
            />
          )}
        </div>
      </section>

      {/* Game Display Area */}
      <AnimatePresence mode="wait">
        {activeGame && (
          <section className="relative z-20 py-6">
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
                  className={`p-4 bg-gradient-to-r ${gameCards.find((g) => g.id === activeGame)?.gradient} text-white relative`}
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

      {/* Footer */}
      <section className="relative z-20 py-6 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div>
              <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-gray-400 mb-3">
                About this collection
              </h3>
              <p className="text-sm text-slate-300 leading-relaxed">
                Six playable games — from classic arcade to ML demos — each surfacing a different layer of Aakriti's engineering and career story.
              </p>
            </div>

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-gray-400 mb-3">
                Game categories
              </h3>
              <div className="flex flex-wrap gap-2">
                {Array.from(new Set(gameCards.map((g) => g.category))).map((cat) => (
                  <span
                    key={cat}
                    className="rounded-full border border-white/15 bg-white/5 px-2.5 py-1 text-xs font-medium text-slate-300"
                  >
                    {cat}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-gray-400 mb-3">
                Explore more
              </h3>
              <motion.button
                onClick={() => navigate("/ai-champions")}
                whileHover={{ x: 4 }}
                className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 p-3 text-left hover:bg-white/10 transition-colors w-full"
              >
                <span className="text-2xl">🏆</span>
                <div>
                  <div className="text-sm font-semibold text-white">
                    AI vs Human Champions
                  </div>
                  <div className="text-xs text-gray-400">
                    Chess, Go, poker, math olympiads & more
                  </div>
                </div>
                <span className="ml-auto text-gray-400">→</span>
              </motion.button>
            </div>
          </div>
        </div>
      </section>
    </SubpageLayout>
  );
}
