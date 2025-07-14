import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
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

  const gameCards = [
    {
      id: "chess" as GameTab,
      title: "CHESS",
      description: "Strategic mastery revealing professional narratives",
      icon: "♔",
      accent: "border-yellow-400",
    },
    {
      id: "bagh-chal" as GameTab,
      title: "BAGH-CHAL",
      description: "Traditional strategy with modern AI sophistication",
      icon: "◆",
      accent: "border-yellow-400",
    },
    {
      id: "pacman" as GameTab,
      title: "PACMAN",
      description: "Arcade adventure unveiling professional strengths",
      icon: "●",
      accent: "border-yellow-400",
    },
    {
      id: "snake" as GameTab,
      title: "SNAKE",
      description: "Journey through professional milestones",
      icon: "◊",
      accent: "border-yellow-400",
    },
    {
      id: "mario-gradient" as GameTab,
      title: "GRADIENT DESCENT",
      description: "Machine learning education through interactive play",
      icon: "▲",
      accent: "border-yellow-400",
    },
    {
      id: "helicopter" as GameTab,
      title: "HELICOPTER",
      description: "Navigate challenges to discover achievements",
      icon: "🚁",
      accent: "border-yellow-400",
    },
  ];

  return (
    <div className="min-h-screen tom-ford-gradient relative overflow-x-hidden">
      {/* Sophisticated Floating Skills Background */}
      <FloatingSkills />

      {/* Header Section */}
      <section className="relative z-20 pt-20 pb-16">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <motion.button
              onClick={() => navigate("/")}
              whileHover={{ scale: 1.02, x: -4 }}
              className="inline-flex items-center gap-3 mb-8 text-white/60 hover:text-yellow-400 transition-colors duration-300 font-light tracking-wider"
            >
              <span className="text-xl">◀</span>
              <span className="tom-ford-subheading text-sm tracking-widest">
                BACK TO PORTFOLIO
              </span>
            </motion.button>

            <h1 className="tom-ford-heading text-6xl md:text-8xl text-white mb-8">
              INTERACTIVE
              <br />
              <span className="gold-shimmer">GAMES</span>
            </h1>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              DISCOVER PROFESSIONAL MASTERY THROUGH SOPHISTICATED GAMEPLAY
            </p>
          </motion.div>
        </div>
      </section>

      {/* Play Old Games Menu */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="tom-ford-heading text-4xl md:text-5xl text-white mb-6">
              PLAY OLD
              <br />
              <span className="gold-shimmer">GAMES</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-3xl mx-auto">
              CLASSIC GAMING EXPERIENCES SHOWCASING TECHNICAL MASTERY
            </p>
          </motion.div>

          <div className="grid md:grid-cols-6 gap-6">
            {gameCards.map((game, index) => (
              <motion.button
                key={game.id}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                whileHover={{ scale: 1.05, y: -8 }}
                onClick={() => setActiveGame(game.id)}
                className="tom-ford-card rounded-sm p-6 text-center transition-all duration-500 hover:border-yellow-400/60 group"
              >
                <div className="text-3xl text-yellow-400 mb-4 group-hover:scale-110 transition-transform duration-300">
                  {game.icon}
                </div>
                <h3 className="tom-ford-subheading text-white text-xs mb-3 tracking-widest">
                  {game.title}
                </h3>
                <div className="w-full h-0.5 bg-yellow-400/20 group-hover:bg-yellow-400/60 transition-colors duration-300" />
              </motion.button>
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
