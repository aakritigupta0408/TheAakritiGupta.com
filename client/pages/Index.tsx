import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Chess from "@/components/Chess";
import BaghChal from "@/components/BaghChal";
import Pacman from "@/components/Pacman";
import Snake from "@/components/Snake";
import MarioGradientDescent from "@/components/MarioGradientDescent";
import ChatBot from "@/components/ChatBot";
import { saveEmailToLocalStorage } from "@/api/save-email";

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
    // Create multiple instances of skills for more frequency
    const duplicatedSkills = [
      ...FLOATING_SKILLS,
      ...FLOATING_SKILLS,
      ...FLOATING_SKILLS,
    ];
    const initialSkills = duplicatedSkills.map((skill, index) => ({
      ...skill,
      id: `${skill.id}-${index}`,
      x: Math.random() * (window.innerWidth - 350),
      y: Math.random() * (window.innerHeight - 120),
      speed: 0.3 + Math.random() * 0.5, // Increased speed
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

    const interval = setInterval(animateSkills, 50); // More frequent updates
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
          animate={{ opacity: 0.6, scale: 1 }}
          transition={{
            duration: 3,
            delay: Math.random() * 4,
            ease: "easeOut",
          }}
        >
          <div className="tom-ford-float px-8 py-4 rounded-sm text-white text-base font-light tracking-wider flex items-center gap-4 shadow-2xl">
            <span className="text-2xl text-yellow-400">{skill.icon}</span>
            <span className="tom-ford-subheading text-sm">{skill.label}</span>
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

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      try {
        // Save email to localStorage as primary method
        saveEmailToLocalStorage(email);

        // Try to call server API if available
        try {
          await fetch("/api/save-email", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ email }),
          });
          console.log("Email saved to server:", email);
        } catch (serverError) {
          console.log("Server save failed, using localStorage:", serverError);
        }

        console.log("Email submitted and saved:", email);
        setEmailSubmitted(true);

        setTimeout(() => {
          setEmailSubmitted(false);
          setEmail("");
        }, 3000);
      } catch (error) {
        console.error("Error saving email:", error);
        // Still show success to user even if saving fails
        setEmailSubmitted(true);
        setTimeout(() => {
          setEmailSubmitted(false);
          setEmail("");
        }, 3000);
      }
    }
  };

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
  ];

  return (
    <div className="min-h-screen tom-ford-gradient relative overflow-x-hidden">
      {/* Sophisticated Floating Skills Background */}
      <FloatingSkills />

      {/* Hero Section - Tom Ford Style */}
      <section className="relative z-20 min-h-screen flex items-center justify-center">
        <div className="max-w-7xl mx-auto px-8 py-20 grid lg:grid-cols-2 gap-16 items-center">
          {/* Left Side - Sophisticated Typography */}
          <motion.div
            initial={{ opacity: 0, x: -60 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 1.2, ease: "easeOut" }}
            className="space-y-10"
          >
            <div>
              <motion.h1
                className="tom-ford-heading text-6xl md:text-8xl text-white mb-8"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 1 }}
              >
                AAKRITI
                <br />
                <span className="gold-shimmer">GUPTA</span>
              </motion.h1>

              <motion.div
                className="space-y-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6, duration: 1 }}
              >
                <p className="tom-ford-subheading text-yellow-400 text-lg tracking-widest">
                  SENIOR ML ENGINEER
                </p>
                <p className="tom-ford-subheading text-white/80 text-base tracking-wider">
                  AI RESEARCHER
                </p>
                <p className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                  LUXURY TECH VISIONARY
                </p>
              </motion.div>

              <motion.div
                className="mt-12 space-y-4 text-white/70 font-light text-lg leading-relaxed max-w-lg"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9, duration: 1 }}
              >
                <p>From Delhi to Silicon Valley</p>

                {/* Company Logos */}
                <div className="space-y-4">
                  <div className="flex items-center gap-6">
                    {/* Meta Logo */}
                    <div className="flex items-center gap-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-sm px-4 py-2">
                      <div className="w-8 h-8 bg-blue-500 rounded-sm flex items-center justify-center text-white font-bold text-sm">
                        f
                      </div>
                      <span className="text-white text-sm tracking-wider">
                        META
                      </span>
                    </div>

                    {/* eBay Logo */}
                    <div className="flex items-center gap-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-sm px-4 py-2">
                      <div className="w-8 h-8 bg-gradient-to-r from-red-500 via-yellow-400 to-blue-500 rounded-sm flex items-center justify-center text-white font-bold text-xs">
                        eB
                      </div>
                      <span className="text-white text-sm tracking-wider">
                        EBAY
                      </span>
                    </div>

                    {/* Yahoo Logo */}
                    <div className="flex items-center gap-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-sm px-4 py-2">
                      <div className="w-8 h-8 bg-purple-600 rounded-sm flex items-center justify-center text-white font-bold text-sm">
                        Y!
                      </div>
                      <span className="text-white text-sm tracking-wider">
                        YAHOO
                      </span>
                    </div>
                  </div>
                </div>

                <p>Recognized by Yann LeCun</p>
                <p>Building Swarnawastra</p>
              </motion.div>
            </div>

            {/* Sophisticated Action Buttons */}
            <motion.div
              className="flex flex-col gap-4 mt-16"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2, duration: 1 }}
            >
              <motion.a
                href="https://www.linkedin.com/in/aakritigupta4894/"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02, y: -2 }}
                className="tom-ford-button px-8 py-4 rounded-sm text-center font-light tracking-wider flex items-center justify-center gap-3"
              >
                <div className="w-6 h-6 bg-blue-600 rounded-sm flex items-center justify-center text-white text-sm font-bold">
                  in
                </div>
                PROFESSIONAL NETWORK
              </motion.a>

              <motion.a
                href="https://github.com/aakritigupta0408?tab=achievements"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02, y: -2 }}
                className="border border-white/30 text-white px-8 py-4 rounded-sm font-light tracking-wider hover:border-yellow-400 hover:text-yellow-400 transition-all duration-300 flex items-center justify-center gap-3"
              >
                <div className="w-6 h-6 bg-slate-800 rounded-sm flex items-center justify-center text-white text-sm font-bold">
                  Git
                </div>
                CODE PORTFOLIO
              </motion.a>

              <motion.button
                onClick={() =>
                  document
                    .getElementById("games")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
                whileHover={{ scale: 1.02, y: -2 }}
                className="border border-yellow-400/50 text-yellow-400 px-8 py-4 rounded-sm font-light tracking-wider hover:border-yellow-400 hover:bg-yellow-400/10 transition-all duration-300"
              >
                PLAY INTERACTIVE GAMES
              </motion.button>
            </motion.div>
          </motion.div>

          {/* Right Side - Sophisticated Photo Gallery */}
          <motion.div
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 1.2, delay: 0.3, ease: "easeOut" }}
            className="relative"
          >
            <div className="space-y-8">
              {/* Main Portrait - Tom Ford Style */}
              <motion.div
                className="relative group"
                whileHover={{ scale: 1.01 }}
                transition={{ duration: 0.6 }}
              >
                <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent z-10 rounded-sm" />
                <img
                  src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2Fed0bc18cd21244e1939892616f236f8f?format=webp&width=800"
                  alt="Aakriti Gupta - Professional Portrait"
                  className="w-full object-contain rounded-sm shadow-2xl border border-white/10"
                  style={{
                    maxHeight: "500px",
                    filter: "contrast(1.1) brightness(0.95)",
                  }}
                />
              </motion.div>

              {/* Secondary Photos Grid */}
              <div className="grid grid-cols-2 gap-6">
                <motion.div
                  whileHover={{ scale: 1.03 }}
                  transition={{ duration: 0.4 }}
                  className="relative group"
                >
                  <img
                    src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F8eb1e0d8ff0f4e7e8a3cb9a919e054b1?format=webp&width=800"
                    alt="Aakriti Gupta - Professional"
                    className="w-full object-contain rounded-sm shadow-xl border border-white/10"
                    style={{
                      maxHeight: "200px",
                      filter: "contrast(1.1) brightness(0.95)",
                    }}
                  />
                </motion.div>

                <motion.div
                  whileHover={{ scale: 1.03 }}
                  transition={{ duration: 0.4 }}
                  className="relative group"
                >
                  <img
                    src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F84cf6b44dba445fcaeced4f15fd299f1?format=webp&width=800"
                    alt="Aakriti Gupta - Fashion"
                    className="w-full object-contain rounded-sm shadow-xl border border-white/10"
                    style={{
                      maxHeight: "200px",
                      filter: "contrast(1.1) brightness(0.95)",
                    }}
                  />
                </motion.div>

                <motion.div
                  whileHover={{ scale: 1.03 }}
                  transition={{ duration: 0.4 }}
                  className="relative group"
                >
                  <img
                    src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F4d9e8bcd67214b5b963eb37e44602024?format=webp&width=800"
                    alt="Aakriti Gupta - Casual"
                    className="w-full object-contain rounded-sm shadow-xl border border-white/10"
                    style={{
                      maxHeight: "150px",
                      filter: "contrast(1.1) brightness(0.95)",
                    }}
                  />
                </motion.div>

                <motion.div
                  whileHover={{ scale: 1.03 }}
                  transition={{ duration: 0.4 }}
                  className="relative group"
                >
                  <img
                    src="https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2Fb2b65fd8e262453e9bd06e29a1b52798?format=webp&width=800"
                    alt="Aakriti Gupta - Elegant"
                    className="w-full object-contain rounded-sm shadow-xl border border-white/10"
                    style={{
                      maxHeight: "150px",
                      filter: "contrast(1.1) brightness(0.95)",
                    }}
                  />
                </motion.div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Professional Excellence Section */}
      <section className="relative z-20 py-32 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 1 }}
            className="text-center mb-20"
          >
            <h2 className="tom-ford-heading text-5xl md:text-6xl text-white mb-8">
              PROFESSIONAL
              <br />
              <span className="gold-shimmer">EXCELLENCE</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-3xl mx-auto">
              REDEFINING THE INTERSECTION OF ARTIFICIAL INTELLIGENCE AND LUXURY
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-12">
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1, duration: 1 }}
              className="tom-ford-card rounded-sm p-10 text-center"
            >
              <div className="text-6xl text-yellow-400 mb-8">◆</div>
              <h3 className="tom-ford-heading text-2xl text-white mb-6">
                TECHNOLOGY LEADERSHIP
              </h3>
              <p className="text-white/70 font-light leading-relaxed mb-6">
                Leading machine learning initiatives at premier technology
                institutions. Serving billions of users through advanced AI
                systems.
              </p>

              {/* Company Logos - Vertical Stack */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 bg-white/5 backdrop-blur-sm border border-yellow-400/20 rounded-sm px-3 py-2">
                  <div className="w-6 h-6 bg-blue-500 rounded-sm flex items-center justify-center text-white font-bold text-xs">
                    f
                  </div>
                  <span className="text-yellow-400/90 text-xs tracking-wider">
                    META
                  </span>
                </div>

                <div className="flex items-center gap-2 bg-white/5 backdrop-blur-sm border border-yellow-400/20 rounded-sm px-3 py-2">
                  <div className="w-6 h-6 bg-gradient-to-r from-red-500 via-yellow-400 to-blue-500 rounded-sm flex items-center justify-center text-white font-bold text-xs">
                    eB
                  </div>
                  <span className="text-yellow-400/90 text-xs tracking-wider">
                    EBAY
                  </span>
                </div>

                <div className="flex items-center gap-2 bg-white/5 backdrop-blur-sm border border-yellow-400/20 rounded-sm px-3 py-2">
                  <div className="w-6 h-6 bg-purple-600 rounded-sm flex items-center justify-center text-white font-bold text-xs">
                    Y!
                  </div>
                  <span className="text-yellow-400/90 text-xs tracking-wider">
                    YAHOO
                  </span>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 1 }}
              className="tom-ford-card rounded-sm p-10 text-center"
            >
              <div className="text-6xl text-yellow-400 mb-8">◇</div>
              <h3 className="tom-ford-heading text-2xl text-white mb-6">
                ACADEMIC RECOGNITION
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Awarded by Dr. Yann LeCun at ICLR 2019. Published research in
                advanced machine learning methodologies.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3, duration: 1 }}
              className="tom-ford-card rounded-sm p-10 text-center"
            >
              <div className="text-6xl text-yellow-400 mb-8">◈</div>
              <h3 className="tom-ford-heading text-2xl text-white mb-6">
                LUXURY INNOVATION
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Founding Swarnawastra. Pioneering AI-driven luxury fashion with
                gold and lab-grown diamonds.
              </p>
            </motion.div>
          </div>
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

          <div className="grid md:grid-cols-5 gap-6">
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
      <section
        id="games"
        className="relative z-20 py-32 border-t border-white/10"
      >
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
          <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-8 mb-16">
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
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* Luxury Contact Section */}
      <section className="relative z-20 py-32 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid lg:grid-cols-2 gap-16 items-center"
          >
            {/* Contact Information */}
            <div>
              <h2 className="tom-ford-heading text-5xl text-white mb-12">
                PROFESSIONAL
                <br />
                <span className="gold-shimmer">CONNECTION</span>
              </h2>

              <p className="text-white/70 text-lg font-light leading-relaxed mb-12 max-w-lg">
                Available for consulting, speaking engagements, and
                collaboration in artificial intelligence, luxury technology, and
                innovative solutions.
              </p>

              <div className="space-y-8">
                <motion.a
                  href="https://www.linkedin.com/in/aakritigupta4894/"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.01, x: 4 }}
                  className="flex items-center gap-6 tom-ford-card p-8 rounded-sm hover:border-yellow-400/50 transition-all duration-300 group"
                >
                  <div className="relative">
                    <div className="w-16 h-16 bg-blue-600 rounded-sm flex items-center justify-center border border-yellow-400/30 group-hover:border-blue-400 transition-colors">
                      <span className="text-3xl text-white font-bold">in</span>
                    </div>
                    <div className="absolute -top-1 -right-1 w-4 h-4 bg-yellow-400 rounded-full flex items-center justify-center">
                      <span className="text-xs text-black font-bold">✓</span>
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="tom-ford-subheading text-white text-sm tracking-wider">
                      LINKEDIN NETWORK
                    </div>
                    <div className="text-white/60 text-xs mt-1">
                      Professional experience & connections
                    </div>
                    <div className="text-blue-400/80 text-xs mt-2 font-light">
                      @aakritigupta4894
                    </div>
                  </div>
                  <div className="text-yellow-400 group-hover:translate-x-2 transition-transform">
                    ◆
                  </div>
                </motion.a>

                <motion.a
                  href="https://github.com/aakritigupta0408?tab=achievements"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.01, x: 4 }}
                  className="flex items-center gap-6 tom-ford-card p-8 rounded-sm hover:border-yellow-400/50 transition-all duration-300 group"
                >
                  <div className="relative">
                    <div className="w-16 h-16 bg-slate-800 rounded-sm flex items-center justify-center border border-yellow-400/30 group-hover:border-slate-400 transition-colors">
                      <span className="text-2xl text-white font-bold">Git</span>
                    </div>
                    <div className="absolute -top-1 -right-1 w-4 h-4 bg-yellow-400 rounded-full flex items-center justify-center">
                      <span className="text-xs text-black font-bold">✓</span>
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="tom-ford-subheading text-white text-sm tracking-wider">
                      GITHUB PORTFOLIO
                    </div>
                    <div className="text-white/60 text-xs mt-1">
                      Open source contributions & achievements
                    </div>
                    <div className="text-slate-400/80 text-xs mt-2 font-light">
                      @aakritigupta0408
                    </div>
                  </div>
                  <div className="text-yellow-400 group-hover:translate-x-2 transition-transform">
                    ◇
                  </div>
                </motion.a>
              </div>
            </div>

            {/* Luxury Resume Request */}
            <div className="tom-ford-card rounded-sm p-12">
              <h3 className="tom-ford-heading text-3xl text-white mb-8">
                EXECUTIVE
                <br />
                <span className="gold-shimmer">PORTFOLIO</span>
              </h3>

              <p className="text-white/70 font-light mb-10 leading-relaxed">
                Request comprehensive professional documentation, detailed
                project portfolios, and executive summary.
              </p>

              <form onSubmit={handleEmailSubmit} className="space-y-8">
                <div>
                  <label className="tom-ford-subheading block text-white/60 text-xs tracking-wider mb-4">
                    PROFESSIONAL EMAIL
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="executive@company.com"
                    className="w-full px-6 py-4 bg-black/50 border border-white/20 rounded-sm text-white placeholder-white/40 focus:border-yellow-400 focus:outline-none transition-colors font-light"
                    required
                  />
                </div>

                <motion.button
                  type="submit"
                  disabled={emailSubmitted}
                  whileHover={{
                    scale: emailSubmitted ? 1 : 1.01,
                    y: emailSubmitted ? 0 : -2,
                  }}
                  className={`w-full py-5 rounded-sm font-light tracking-widest transition-all duration-300 ${
                    emailSubmitted
                      ? "bg-green-600/80 text-white border border-green-500"
                      : "tom-ford-button"
                  }`}
                >
                  {emailSubmitted
                    ? "PORTFOLIO TRANSMITTED"
                    : "REQUEST EXECUTIVE PORTFOLIO"}
                </motion.button>
              </form>
            </div>
          </motion.div>

          {/* Sophisticated Footer */}
          <div className="mt-32 pt-16 border-t border-white/10 text-center">
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
        </div>
      </section>

      {/* AI Assistant ChatBot */}
      <ChatBot />
    </div>
  );
}
