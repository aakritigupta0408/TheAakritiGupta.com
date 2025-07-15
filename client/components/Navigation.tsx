import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate, useLocation } from "react-router-dom";

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);

  const talents = [
    {
      id: "ai-researcher",
      title: "AI RESEARCHER",
      subtitle: "Innovation & Discovery",
      icon: "◆",
      color: "from-cyan-500 to-cyan-700",
      description: "Advanced machine learning research and AI innovation",
    },
    {
      id: "social-entrepreneur",
      title: "SOCIAL ENTREPRENEUR",
      subtitle: "Impact & Vision",
      icon: "◇",
      color: "from-teal-500 to-teal-700",
      description: "Building technology for social good and impact",
    },
    {
      id: "marksman",
      title: "MARKSMAN",
      subtitle: "Precision & Focus",
      icon: "◎",
      color: "from-red-500 to-red-700",
      description: "Expert precision shooting and tactical training",
    },
    {
      id: "equestrian",
      title: "EQUESTRIAN",
      subtitle: "Grace & Partnership",
      icon: "◈",
      color: "from-amber-500 to-amber-700",
      description: "Professional horse riding and equestrian arts",
    },
    {
      id: "aviator",
      title: "AVIATOR",
      subtitle: "Sky Mastery",
      icon: "◉",
      color: "from-blue-500 to-blue-700",
      description: "Pilot training and aviation excellence",
    },
    {
      id: "motorcyclist",
      title: "MOTORCYCLIST",
      subtitle: "Speed & Freedom",
      icon: "◐",
      color: "from-purple-500 to-purple-700",
      description: "High-performance motorcycle expertise",
    },
    {
      id: "pianist",
      title: "PIANIST",
      subtitle: "Musical Artistry",
      icon: "◑",
      color: "from-green-500 to-green-700",
      description: "Classical and contemporary piano mastery",
    },
  ];

  const handleTalentClick = (talentId: string) => {
    navigate(`/talent/${talentId}`);
    setIsOpen(false);
  };

  const isHomePage = location.pathname === "/";

  return (
    <>
      {/* Main Navigation Header */}
      <motion.header
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="luxury-nav fixed top-0 left-0 right-0 z-50"
      >
        <div className="max-w-7xl mx-auto px-8 py-4">
          <div className="flex justify-between items-center">
            {/* Logo/Home */}
            <motion.button
              onClick={() => navigate("/")}
              whileHover={{ scale: 1.02 }}
              className="flex items-center gap-3"
            >
              <div className="w-10 h-10 bg-gradient-to-r from-yellow-400 to-yellow-600 rounded-lg flex items-center justify-center border-2 border-black">
                <span className="text-white font-bold text-lg">◆</span>
              </div>
              <div className="text-left">
                <div className="tom-ford-heading luxury-text-primary text-lg tracking-wider">
                  AAKRITI GUPTA
                </div>
                <div className="tom-ford-subheading luxury-text-accent text-xs tracking-widest">
                  ML ENGINEER & VISIONARY
                </div>
              </div>
            </motion.button>

            {/* Main Navigation */}
            <nav className="hidden lg:flex items-center gap-8">
              <motion.button
                onClick={() => navigate("/")}
                whileHover={{ y: -2 }}
                className={`tom-ford-subheading text-sm tracking-widest transition-colors duration-300 ${
                  isHomePage
                    ? "text-yellow-400 border-b border-yellow-400 pb-1"
                    : "text-white/80 hover:text-yellow-400"
                }`}
              >
                PORTFOLIO
              </motion.button>

              <motion.button
                onClick={() => navigate("/games")}
                whileHover={{ y: -2 }}
                className={`tom-ford-subheading text-sm tracking-widest transition-colors duration-300 ${
                  location.pathname === "/games"
                    ? "luxury-text-accent border-b border-yellow-400 pb-1"
                    : "luxury-text-muted hover:luxury-text-accent"
                }`}
              >
                GAMES
              </motion.button>

              <motion.button
                onClick={() => navigate("/ai-playground")}
                whileHover={{ y: -2 }}
                className={`tom-ford-subheading text-sm tracking-widest transition-colors duration-300 ${
                  location.pathname === "/ai-playground"
                    ? "luxury-text-accent border-b border-yellow-400 pb-1"
                    : "luxury-text-muted hover:luxury-text-accent"
                }`}
              >
                AI PLAYGROUND
              </motion.button>

              {/* Talents Dropdown */}
              <div className="relative">
                <motion.button
                  onClick={() => setIsOpen(!isOpen)}
                  whileHover={{ y: -2 }}
                  className={`tom-ford-subheading text-sm tracking-widest transition-colors duration-300 flex items-center gap-2 ${
                    location.pathname.startsWith("/talent")
                      ? "text-yellow-400"
                      : "text-white/80 hover:text-yellow-400"
                  }`}
                >
                  TALENTS
                  <motion.span
                    animate={{ rotate: isOpen ? 180 : 0 }}
                    transition={{ duration: 0.3 }}
                    className="text-yellow-400"
                  >
                    ▼
                  </motion.span>
                </motion.button>

                {/* Dropdown Menu */}
                <AnimatePresence>
                  {isOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-full right-0 mt-4 w-96 bg-black/95 backdrop-blur-sm border border-white/20 rounded-sm shadow-2xl overflow-hidden"
                    >
                      <div className="p-2">
                        {talents.map((talent, index) => (
                          <motion.button
                            key={talent.id}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            onClick={() => handleTalentClick(talent.id)}
                            className="w-full p-4 text-left hover:bg-white/5 rounded-sm transition-all duration-300 group"
                          >
                            <div className="flex items-center gap-4">
                              <div
                                className={`w-12 h-12 bg-gradient-to-r ${talent.color} rounded-sm flex items-center justify-center text-white text-xl font-bold shadow-lg group-hover:scale-110 transition-transform duration-300`}
                              >
                                {talent.icon}
                              </div>
                              <div className="flex-1">
                                <div className="tom-ford-subheading text-white text-sm tracking-wider group-hover:text-yellow-400 transition-colors">
                                  {talent.title}
                                </div>
                                <div className="text-white/60 text-xs mt-1 group-hover:text-white/80 transition-colors">
                                  {talent.subtitle}
                                </div>
                                <div className="text-white/40 text-xs mt-1 font-light">
                                  {talent.description}
                                </div>
                              </div>
                              <div className="text-yellow-400/60 group-hover:text-yellow-400 group-hover:translate-x-1 transition-all duration-300">
                                ▶
                              </div>
                            </div>
                          </motion.button>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Resume Link */}
              <motion.a
                href="https://drive.google.com/file/d/1Mnmk6nP9l_Av0LvpgJQ5Tkjb7BqhY7nb/view?usp=sharing"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-2 bg-gradient-to-r from-yellow-400/20 to-yellow-600/20 border border-yellow-400/50 rounded-sm text-yellow-400 tom-ford-subheading text-sm tracking-widest hover:border-yellow-400 hover:bg-yellow-400/10 transition-all duration-300"
              >
                RESUME
              </motion.a>
            </nav>

            {/* Mobile Menu Button */}
            <motion.button
              onClick={() => setIsOpen(!isOpen)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="lg:hidden text-white p-2"
            >
              <div className="w-6 h-6 flex flex-col justify-center items-center">
                <motion.span
                  animate={{
                    rotate: isOpen ? 45 : 0,
                    y: isOpen ? 8 : 0,
                  }}
                  className="w-full h-0.5 bg-yellow-400 mb-1.5 origin-center transition-all duration-300"
                />
                <motion.span
                  animate={{ opacity: isOpen ? 0 : 1 }}
                  className="w-full h-0.5 bg-white mb-1.5 transition-all duration-300"
                />
                <motion.span
                  animate={{
                    rotate: isOpen ? -45 : 0,
                    y: isOpen ? -8 : 0,
                  }}
                  className="w-full h-0.5 bg-yellow-400 origin-center transition-all duration-300"
                />
              </div>
            </motion.button>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="lg:hidden bg-black/95 backdrop-blur-sm border-t border-white/10"
            >
              <div className="max-w-7xl mx-auto px-8 py-6 space-y-4">
                <motion.button
                  onClick={() => {
                    navigate("/");
                    setIsOpen(false);
                  }}
                  className="block w-full text-left tom-ford-subheading text-white/80 hover:text-yellow-400 transition-colors py-2"
                >
                  PORTFOLIO
                </motion.button>
                <motion.button
                  onClick={() => {
                    navigate("/games");
                    setIsOpen(false);
                  }}
                  className="block w-full text-left tom-ford-subheading text-white/80 hover:text-yellow-400 transition-colors py-2"
                >
                  GAMES
                </motion.button>
                <div className="border-t border-white/10 pt-4">
                  <div className="tom-ford-subheading text-yellow-400 text-sm mb-3 tracking-wider">
                    TALENTS
                  </div>
                  {talents.map((talent) => (
                    <motion.button
                      key={talent.id}
                      onClick={() => handleTalentClick(talent.id)}
                      className="block w-full text-left p-3 hover:bg-white/5 rounded-sm transition-all duration-300 mb-2"
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-8 h-8 bg-gradient-to-r ${talent.color} rounded-sm flex items-center justify-center text-white text-sm font-bold`}
                        >
                          {talent.icon}
                        </div>
                        <div>
                          <div className="text-white text-sm tom-ford-subheading tracking-wider">
                            {talent.title}
                          </div>
                          <div className="text-white/60 text-xs">
                            {talent.subtitle}
                          </div>
                        </div>
                      </div>
                    </motion.button>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.header>

      {/* Click outside to close */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsOpen(false)}
            className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm lg:hidden"
          />
        )}
      </AnimatePresence>
    </>
  );
};

export default Navigation;
