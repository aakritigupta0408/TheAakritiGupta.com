import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import Chess from "@/components/Chess";
import BaghChal from "@/components/BaghChal";
import Pacman from "@/components/Pacman";
import Snake from "@/components/Snake";
import MarioGradientDescent from "@/components/MarioGradientDescent";
import Helicopter from "@/components/Helicopter";
import ChatBot from "@/components/ChatBot";

type GameTab =
  | "chess"
  | "bagh-chal"
  | "pacman"
  | "snake"
  | "mario-gradient"
  | "helicopter";

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
    label: "AI RESEARCHER",
    icon: "◆",
    color: "tom-ford-float",
  },
  {
    id: "engineer",
    label: "ENGINEER",
    icon: "◇",
    color: "tom-ford-float",
  },
  {
    id: "meta-expert",
    label: "META ENGINEER",
    icon: "◆",
    color: "tom-ford-float",
  },
  {
    id: "horse-rider",
    label: "EQUESTRIAN",
    icon: "◈",
    color: "tom-ford-float",
  },
  {
    id: "ml-specialist",
    label: "ML SPECIALIST",
    icon: "◇",
    color: "tom-ford-float",
  },
  {
    id: "pilot",
    label: "AVIATOR",
    icon: "◉",
    color: "tom-ford-float",
  },
  {
    id: "ebay-alumni",
    label: "EBAY VETERAN",
    icon: "◆",
    color: "tom-ford-float",
  },
  {
    id: "shooter",
    label: "MARKSMAN",
    icon: "◎",
    color: "tom-ford-float",
  },
  {
    id: "yahoo-scientist",
    label: "YAHOO SCIENTIST",
    icon: "◇",
    color: "tom-ford-float",
  },
  {
    id: "biker",
    label: "MOTORCYCLIST",
    icon: "◐",
    color: "tom-ford-float",
  },
  {
    id: "swarnawastra",
    label: "LUXURY TECH FOUNDER",
    icon: "◆",
    color: "tom-ford-float",
  },
  {
    id: "pianist",
    label: "PIANIST",
    icon: "◑",
    color: "tom-ford-float",
  },
  {
    id: "yann-lecun-awardee",
    label: "YANN LECUN AWARDEE",
    icon: "◇",
    color: "tom-ford-float",
  },
  {
    id: "silicon-valley",
    label: "SILICON VALLEY",
    icon: "◈",
    color: "tom-ford-float",
  },
  {
    id: "delhi-to-sv",
    label: "DELHI TO SILICON VALLEY",
    icon: "◉",
    color: "tom-ford-float",
  },
];

// Sophisticated Floating Skills Component
const FloatingSkills = () => {
  const [skills, setSkills] = useState<FloatingSkill[]>([]);

  useEffect(() => {
    // Use only original skills for less frequency
    const initialSkills = FLOATING_SKILLS.map((skill, index) => ({
      ...skill,
      x: Math.random() * (window.innerWidth - 300),
      y: Math.random() * (window.innerHeight - 100),
      speed: 0.15 + Math.random() * 0.25, // Reduced speed
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

    const interval = setInterval(animateSkills, 120); // Less frequent updates
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
          animate={{ opacity: 0.4, scale: 1 }}
          transition={{
            duration: 4,
            delay: Math.random() * 8,
            ease: "easeOut",
          }}
        >
          <div className="tom-ford-float px-4 py-2 rounded-sm text-white text-xs font-light tracking-wider flex items-center gap-2 shadow-lg">
            <span className="text-sm text-yellow-400">{skill.icon}</span>
            <span className="tom-ford-subheading text-xs">{skill.label}</span>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

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
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white relative overflow-x-hidden">
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
              className="text-xl text-gray-600 tracking-wide max-w-4xl mx-auto mb-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              Discover professional mastery through sophisticated gameplay
              experiences
            </motion.p>

            {/* Gaming Stats */}
            <motion.div
              className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <div className="bg-white p-4 rounded-xl shadow-lg border border-gray-200">
                <div className="text-2xl font-bold text-blue-600">6</div>
                <div className="text-sm text-gray-600">Interactive Games</div>
              </div>
              <div className="bg-white p-4 rounded-xl shadow-lg border border-gray-200">
                <div className="text-2xl font-bold text-purple-600">
                  AI-Powered
                </div>
                <div className="text-sm text-gray-600">Smart Gameplay</div>
              </div>
              <div className="bg-white p-4 rounded-xl shadow-lg border border-gray-200">
                <div className="text-2xl font-bold text-green-600">
                  Portfolio
                </div>
                <div className="text-sm text-gray-600">Integration</div>
              </div>
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
                      x: hoveredGame === game.id ? ["100%", "-100%"] : "100%",
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

      {/* Interactive Games Portfolio */}
      <section className="relative z-20 py-32 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-20"
          >
            <h2 className="tom-ford-heading text-5xl md:text-6xl text-white mb-8">
              INTERACTIVE
              <br />
              <span className="gold-shimmer">PORTFOLIO</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              DISCOVER PROFESSIONAL MASTERY THROUGH SOPHISTICATED GAMEPLAY
            </p>
          </motion.div>

          {/* Luxury Game Selection */}
          <div className="grid md:grid-cols-3 lg:grid-cols-6 gap-8 mb-16">
            {gameCards.map((game, index) => (
              <motion.div
                key={game.id}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.8 }}
                whileHover={{ scale: 1.02, y: -4 }}
                onClick={() => setActiveGame(game.id)}
                className={`cursor-pointer tom-ford-card rounded-sm p-8 text-center transition-all duration-500 ${
                  activeGame === game.id
                    ? `${game.accent} border-2 transform scale-105`
                    : "hover:border-yellow-400/50"
                }`}
              >
                <div className="text-4xl text-yellow-400 mb-6">{game.icon}</div>
                <h3 className="tom-ford-subheading text-white text-sm mb-4 tracking-widest">
                  {game.title}
                </h3>
                <p className="text-white/60 text-xs font-light leading-relaxed mb-6">
                  {game.description}
                </p>
                <div
                  className={`w-full h-0.5 ${activeGame === game.id ? "bg-yellow-400" : "bg-white/20"} transition-colors duration-300`}
                />
              </motion.div>
            ))}
          </div>

          {/* Game Display Area */}
          <AnimatePresence mode="wait">
            {activeGame && (
              <motion.div
                key={activeGame}
                initial={{ opacity: 0, y: 60 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -60 }}
                transition={{ duration: 0.8 }}
                className="tom-ford-glass rounded-sm overflow-hidden"
              >
                <div className="p-8 border-b border-white/10 flex justify-between items-center">
                  <h3 className="tom-ford-heading text-3xl text-white">
                    {gameCards.find((g) => g.id === activeGame)?.title}
                  </h3>
                  <button
                    onClick={() => setActiveGame(null)}
                    className="text-white/60 hover:text-yellow-400 transition-colors text-xl"
                  >
                    ✕
                  </button>
                </div>
                <div className="p-12 bg-black/50">
                  {activeGame === "chess" && <Chess />}
                  {activeGame === "bagh-chal" && <BaghChal />}
                  {activeGame === "pacman" && <Pacman />}
                  {activeGame === "snake" && <Snake />}
                  {activeGame === "mario-gradient" && <MarioGradientDescent />}
                  {activeGame === "helicopter" && <Helicopter />}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* Sophisticated Footer */}
      <section className="relative z-20 py-16 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8 text-center">
          <p className="tom-ford-subheading text-white/40 text-sm tracking-widest">
            © 2024 AAKRITI GUPTA • SENIOR ML ENGINEER • LUXURY TECH VISIONARY
          </p>
          <div className="mt-8 flex justify-center gap-12 text-xs text-white/30 tracking-wider">
            <span>MARKSMAN</span>
            <span>EQUESTRIAN</span>
            <span>AVIATOR</span>
            <span>MOTORCYCLIST</span>
            <span>PIANIST</span>
          </div>
        </div>
      </section>

      {/* AI Assistant ChatBot */}
      <ChatBot />
    </div>
  );
}
