import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";
import { saveEmailToLocalStorage } from "@/api/save-email";

// Photo gallery with creative transitions
const PhotoGallery = () => {
  const [activePhoto, setActivePhoto] = useState(0);

  const photos = [
    {
      url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2Fed0bc18cd21244e1939892616f236f8f?format=webp&width=800",
      title: "AI Researcher",
      subtitle: "Pushing boundaries in machine learning",
    },
    {
      url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F8eb1e0d8ff0f4e7e8a3cb9a919e054b1?format=webp&width=800",
      title: "Technology Leader",
      subtitle: "Leading innovation at top tech companies",
    },
    {
      url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F84cf6b44dba445fcaeced4f15fd299f1?format=webp&width=800",
      title: "Luxury Visionary",
      subtitle: "Founder of Swarnawastra",
    },
    {
      url: "https://cdn.builder.io/api/v1/image/assets%2Ff2155d07c4314be389b158f0dc3f31dc%2F4d9e8bcd67214b5b963eb37e44602024?format=webp&width=800",
      title: "Multi-disciplinary Expert",
      subtitle: "From Delhi to Silicon Valley",
    },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActivePhoto((prev) => (prev + 1) % photos.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative w-full h-full">
      <AnimatePresence mode="wait">
        <motion.div
          key={activePhoto}
          initial={{ opacity: 0, scale: 1.1 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="relative w-full h-full"
        >
          <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent z-10 rounded-2xl" />
          <img
            src={photos[activePhoto].url}
            alt={photos[activePhoto].title}
            className="w-full h-full object-cover rounded-2xl shadow-2xl"
          />
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="absolute bottom-6 left-6 z-20 text-white"
          >
            <h3 className="text-xl font-bold mb-1">
              {photos[activePhoto].title}
            </h3>
            <p className="text-white/80 text-sm">
              {photos[activePhoto].subtitle}
            </p>
          </motion.div>
        </motion.div>
      </AnimatePresence>

      {/* Photo indicators */}
      <div className="absolute bottom-4 right-4 z-20 flex gap-2">
        {photos.map((_, idx) => (
          <button
            key={idx}
            onClick={() => setActivePhoto(idx)}
            className={`w-3 h-3 rounded-full transition-all duration-300 ${
              idx === activePhoto ? "bg-white" : "bg-white/40"
            }`}
          />
        ))}
      </div>
    </div>
  );
};

// Floating achievement badges
const AchievementBadges = () => {
  const achievements = [
    { icon: "🏆", label: "Yann LeCun Award", delay: 0 },
    { icon: "🤖", label: "AI Expert", delay: 0.5 },
    { icon: "💎", label: "Luxury Tech", delay: 1 },
    { icon: "🎯", label: "Marksman", delay: 1.5 },
    { icon: "🏇", label: "Equestrian", delay: 2 },
    { icon: "✈️", label: "Pilot", delay: 2.5 },
    { icon: "🏍️", label: "Motorcyclist", delay: 3 },
    { icon: "🎹", label: "Pianist", delay: 3.5 },
  ];

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {achievements.map((achievement, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0, scale: 0, rotate: -180 }}
          animate={{
            opacity: [0, 1, 1, 0],
            scale: [0, 1, 1, 0],
            rotate: [0, 360],
            x: [
              Math.random() *
                (typeof window !== "undefined" ? window.innerWidth : 1200),
              Math.random() *
                (typeof window !== "undefined" ? window.innerWidth : 1200),
            ],
            y: [
              Math.random() *
                (typeof window !== "undefined" ? window.innerHeight : 800),
              Math.random() *
                (typeof window !== "undefined" ? window.innerHeight : 800),
            ],
          }}
          transition={{
            duration: 8,
            delay: achievement.delay,
            repeat: Infinity,
            repeatDelay: 10,
          }}
          className="absolute"
        >
          <div className="bg-white/10 backdrop-blur-md rounded-full px-4 py-2 border border-white/20 shadow-lg">
            <div className="flex items-center gap-2">
              <span className="text-2xl">{achievement.icon}</span>
              <span className="text-white text-sm font-medium">
                {achievement.label}
              </span>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

// Company logo carousel
const CompanyCarousel = () => {
  const companies = [
    {
      name: "Meta",
      logo: "f",
      color: "bg-blue-500",
      gradient: "from-blue-500 to-blue-700",
    },
    {
      name: "eBay",
      logo: "eB",
      color: "bg-gradient-to-r from-red-500 via-yellow-400 to-blue-500",
      gradient: "from-red-500 to-blue-500",
    },
    {
      name: "Yahoo",
      logo: "Y!",
      color: "bg-purple-600",
      gradient: "from-purple-500 to-purple-700",
    },
  ];

  return (
    <motion.div
      className="flex gap-6 justify-center"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1, duration: 0.8 }}
    >
      {companies.map((company, idx) => (
        <motion.div
          key={company.name}
          whileHover={{ scale: 1.1, y: -5 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 1.2 + idx * 0.2, duration: 0.5 }}
          className={`relative group cursor-pointer`}
        >
          <div
            className={`w-16 h-16 ${company.color} rounded-2xl flex items-center justify-center text-white font-bold text-xl shadow-xl border-2 border-white/20 group-hover:border-white/40 transition-all duration-300`}
          >
            {company.logo}
          </div>
          <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <div className="bg-black/80 backdrop-blur-sm text-white text-xs px-3 py-1 rounded-full">
              {company.name}
            </div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

export default function Index() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [emailSubmitted, setEmailSubmitted] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      try {
        saveEmailToLocalStorage(email);

        try {
          await fetch("/api/save-email", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email }),
          });
        } catch (serverError) {
          console.log("Server save failed, using localStorage:", serverError);
        }

        setEmailSubmitted(true);
        setTimeout(() => {
          setEmailSubmitted(false);
          setEmail("");
        }, 3000);
      } catch (error) {
        console.error("Error saving email:", error);
        setEmailSubmitted(true);
        setTimeout(() => {
          setEmailSubmitted(false);
          setEmail("");
        }, 3000);
      }
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-black relative overflow-hidden">
      <Navigation />

      {/* Floating achievement badges */}
      <AchievementBadges />

      {/* Dynamic cursor glow */}
      <motion.div
        className="fixed w-96 h-96 bg-blue-500/20 rounded-full blur-3xl pointer-events-none z-0"
        style={{
          left: mousePosition.x - 192,
          top: mousePosition.y - 192,
        }}
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Hero Section */}
      <section className="relative z-20 min-h-screen flex items-center justify-center px-6 py-20">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          {/* Left Side - Content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="space-y-8"
          >
            {/* Name and Title with animated gradient */}
            <div className="space-y-6">
              <motion.h1
                className="text-6xl lg:text-8xl font-black bg-gradient-to-r from-white via-cyan-200 to-purple-200 bg-clip-text text-transparent leading-tight"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.8 }}
                whileHover={{ scale: 1.02 }}
                style={{
                  backgroundSize: "200% 200%",
                  animation: "gradientShift 4s ease infinite",
                }}
              >
                AAKRITI
                <br />
                GUPTA
              </motion.h1>

              {/* Add gradient animation keyframes */}
              <style>{`
                @keyframes gradientShift {
                  0% {
                    background-position: 0% 50%;
                  }
                  50% {
                    background-position: 100% 50%;
                  }
                  100% {
                    background-position: 0% 50%;
                  }
                }
              `}</style>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.8 }}
                className="space-y-3"
              >
                <div className="space-y-4">
                  {[
                    {
                      title: "Senior ML Engineer",
                      icon: "🤖",
                      gradient: "from-pink-500 to-purple-500",
                      desc: "Building AI that matters",
                    },
                    {
                      title: "AI Researcher",
                      icon: "🔬",
                      gradient: "from-blue-500 to-cyan-500",
                      desc: "Pushing boundaries of AI",
                    },
                    {
                      title: "Luxury Tech Visionary",
                      icon: "💎",
                      gradient: "from-yellow-500 to-orange-500",
                      desc: "Where elegance meets innovation",
                    },
                  ].map((role, idx) => (
                    <motion.div
                      key={idx}
                      className="group flex items-center gap-4 p-4 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 transition-all duration-300"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + idx * 0.1 }}
                      whileHover={{ scale: 1.02, x: 10 }}
                    >
                      <motion.div
                        className={`w-12 h-12 bg-gradient-to-r ${role.gradient} rounded-full flex items-center justify-center text-lg shadow-lg`}
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          delay: idx * 0.5,
                        }}
                      >
                        {role.icon}
                      </motion.div>
                      <div>
                        <p className="text-xl text-white font-bold group-hover:text-cyan-300 transition-colors">
                          {role.title}
                        </p>
                        <p className="text-sm text-gray-300 group-hover:text-white transition-colors">
                          {role.desc}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </div>

            {/* Achievement highlights */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.8 }}
              className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10"
            >
              <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                <span className="text-2xl">🏆</span>
                Recognition & Achievements
              </h3>
              <div className="space-y-2 text-white/80">
                <p>• Recognized by Dr. Yann LeCun at ICLR 2019</p>
                <p>• Building AI to make life simpler</p>
                <p>• From Delhi to Silicon Valley</p>
              </div>
            </motion.div>

            {/* Company logos */}
            <CompanyCarousel />

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2, duration: 0.8 }}
              className="flex flex-col gap-4"
            >
              <motion.a
                href="https://drive.google.com/file/d/1Mnmk6nP9l_Av0LvpgJQ5Tkjb7BqhY7nb/view?usp=sharing"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.98 }}
                className="relative overflow-hidden bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-2xl font-semibold text-lg shadow-2xl hover:shadow-blue-500/25 transition-all duration-300 group"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative flex items-center justify-center gap-3">
                  <span className="text-2xl">📄</span>
                  <span>Download Resume</span>
                  <span className="group-hover:translate-x-1 transition-transform duration-300">
                    →
                  </span>
                </div>
              </motion.a>

              <div className="grid grid-cols-2 gap-4">
                <motion.button
                  onClick={() => navigate("/games")}
                  whileHover={{ scale: 1.05, y: -3 }}
                  whileTap={{ scale: 0.95 }}
                  className="relative group bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-md border border-white/30 text-white px-6 py-4 rounded-2xl font-bold hover:from-purple-500/30 hover:to-pink-500/30 transition-all duration-300 flex items-center justify-center gap-3 shadow-lg overflow-hidden"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-600/0 to-pink-600/0 group-hover:from-purple-600/20 group-hover:to-pink-600/20 transition-all duration-500"></div>
                  <motion.span
                    className="text-2xl relative z-10"
                    animate={{ rotate: [0, 5, -5, 0] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    🎮
                  </motion.span>
                  <span className="relative z-10 group-hover:text-pink-200 transition-colors">
                    Games
                  </span>
                </motion.button>

                <motion.button
                  onClick={() => navigate("/ai-playground")}
                  whileHover={{ scale: 1.05, y: -3 }}
                  whileTap={{ scale: 0.95 }}
                  className="relative group bg-gradient-to-r from-blue-500/20 to-cyan-500/20 backdrop-blur-md border border-white/30 text-white px-6 py-4 rounded-2xl font-bold hover:from-blue-500/30 hover:to-cyan-500/30 transition-all duration-300 flex items-center justify-center gap-3 shadow-lg overflow-hidden"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-600/0 to-cyan-600/0 group-hover:from-blue-600/20 group-hover:to-cyan-600/20 transition-all duration-500"></div>
                  <motion.span
                    className="text-2xl relative z-10"
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    🤖
                  </motion.span>
                  <span className="relative z-10 group-hover:text-cyan-200 transition-colors">
                    AI Tools
                  </span>
                </motion.button>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <motion.button
                  onClick={() => navigate("/ai-discoveries")}
                  whileHover={{ scale: 1.02, y: -1 }}
                  whileTap={{ scale: 0.98 }}
                  className="bg-white/10 backdrop-blur-md border border-white/20 text-white px-6 py-3 rounded-xl font-medium hover:bg-white/20 transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <span className="text-xl">🧠</span>
                  <span>AI History</span>
                </motion.button>

                <motion.button
                  onClick={() => navigate("/ai-tools")}
                  whileHover={{ scale: 1.02, y: -1 }}
                  whileTap={{ scale: 0.98 }}
                  className="bg-white/10 backdrop-blur-md border border-white/20 text-white px-6 py-3 rounded-xl font-medium hover:bg-white/20 transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <span className="text-xl">🛠️</span>
                  <span>Pro Tools</span>
                </motion.button>
              </div>

              {/* New Prompt Engineering Button */}
              <motion.button
                onClick={() => navigate("/prompt-engineering")}
                whileHover={{ scale: 1.05, y: -3 }}
                whileTap={{ scale: 0.95 }}
                className="relative group w-full bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20 backdrop-blur-md border border-white/30 text-white px-6 py-4 rounded-2xl font-bold hover:from-violet-500/30 hover:to-fuchsia-500/30 transition-all duration-300 flex items-center justify-center gap-3 shadow-lg overflow-hidden"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.4 }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-violet-600/0 to-fuchsia-600/0 group-hover:from-violet-600/20 group-hover:to-fuchsia-600/20 transition-all duration-500"></div>
                <motion.span
                  className="text-2xl relative z-10"
                  animate={{
                    rotate: [0, 5, -5, 0],
                    scale: [1, 1.1, 1],
                  }}
                  transition={{ duration: 3, repeat: Infinity }}
                >
                  ✨
                </motion.span>
                <div className="relative z-10 text-center">
                  <div className="group-hover:text-fuchsia-200 transition-colors font-black">
                    Prompt Engineering Mastery
                  </div>
                  <div className="text-xs opacity-80 group-hover:opacity-100 transition-opacity">
                    Master AI Communication
                  </div>
                </div>
              </motion.button>

              {/* AI Agent Training Button */}
              <motion.button
                onClick={() => navigate("/ai-agent-training")}
                whileHover={{ scale: 1.05, y: -3 }}
                whileTap={{ scale: 0.95 }}
                className="relative group w-full bg-gradient-to-r from-cyan-500/20 to-blue-500/20 backdrop-blur-md border border-white/30 text-white px-6 py-4 rounded-2xl font-bold hover:from-cyan-500/30 hover:to-blue-500/30 transition-all duration-300 flex items-center justify-center gap-3 shadow-lg overflow-hidden"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.5 }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-600/0 to-blue-600/0 group-hover:from-cyan-600/20 group-hover:to-blue-600/20 transition-all duration-500"></div>
                <motion.span
                  className="text-2xl relative z-10"
                  animate={{
                    rotate: [0, 10, -10, 0],
                    scale: [1, 1.2, 1],
                  }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  🤖
                </motion.span>
                <div className="relative z-10 text-center">
                  <div className="group-hover:text-cyan-200 transition-colors font-black">
                    AI Agent Training
                  </div>
                  <div className="text-xs opacity-80 group-hover:opacity-100 transition-opacity">
                    Build Intelligent Agents
                  </div>
                </div>
              </motion.button>

              {/* AI Champions Button */}
              <motion.button
                onClick={() => navigate("/ai-champions")}
                whileHover={{ scale: 1.05, y: -3 }}
                whileTap={{ scale: 0.95 }}
                className="relative group w-full bg-gradient-to-r from-red-500/20 to-orange-500/20 backdrop-blur-md border border-white/30 text-white px-6 py-4 rounded-2xl font-bold hover:from-red-500/30 hover:to-orange-500/30 transition-all duration-300 flex items-center justify-center gap-3 shadow-lg overflow-hidden"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.6 }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-red-600/0 to-orange-600/0 group-hover:from-red-600/20 group-hover:to-orange-600/20 transition-all duration-500"></div>
                <motion.span
                  className="text-2xl relative z-10"
                  animate={{
                    rotate: [0, 10, -10, 0],
                    scale: [1, 1.2, 1],
                  }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  🏆
                </motion.span>
                <span className="relative z-10 group-hover:text-orange-200 transition-colors">
                  AI vs Champions
                </span>
              </motion.button>
            </motion.div>
          </motion.div>

          {/* Right Side - Creative Photo Gallery */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
            className="relative"
          >
            <div className="aspect-[4/5] max-w-md mx-auto">
              <PhotoGallery />
            </div>

            {/* Decorative elements */}
            <motion.div
              className="absolute -top-4 -right-4 w-32 h-32 bg-gradient-to-br from-blue-500/30 to-purple-500/30 rounded-full blur-2xl"
              animate={{
                scale: [1, 1.2, 1],
                rotate: [0, 180, 360],
              }}
              transition={{
                duration: 10,
                repeat: Infinity,
                ease: "linear",
              }}
            />
            <motion.div
              className="absolute -bottom-4 -left-4 w-24 h-24 bg-gradient-to-br from-pink-500/30 to-yellow-500/30 rounded-full blur-2xl"
              animate={{
                scale: [1.2, 1, 1.2],
                rotate: [360, 180, 0],
              }}
              transition={{
                duration: 8,
                repeat: Infinity,
                ease: "linear",
              }}
            />
          </motion.div>
        </div>
      </section>

      {/* Skills & Talents Showcase */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl font-bold text-white mb-6">
              Multi-Disciplinary
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                {" "}
                Expertise
              </span>
            </h2>
            <p className="text-white/70 text-xl max-w-3xl mx-auto">
              A unique blend of technical mastery and diverse talents, from AI
              research to luxury innovation
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: "🔬",
                title: "AI Research",
                desc: "Published research, Yann LeCun recognition",
                color: "from-blue-500 to-cyan-500",
              },
              {
                icon: "💼",
                title: "Tech Leadership",
                desc: "Meta, eBay, Yahoo experience",
                color: "from-purple-500 to-pink-500",
              },
              {
                icon: "💎",
                title: "Luxury Tech",
                desc: "Founding Swarnawastra",
                color: "from-yellow-500 to-orange-500",
              },
              {
                icon: "🎯",
                title: "Marksman",
                desc: "Precision shooting expertise",
                color: "from-red-500 to-rose-500",
              },
              {
                icon: "🏇",
                title: "Equestrian",
                desc: "Professional horse riding",
                color: "from-green-500 to-emerald-500",
              },
              {
                icon: "✈️",
                title: "Aviation",
                desc: "Pilot training & certification",
                color: "from-blue-500 to-indigo-500",
              },
              {
                icon: "🏍️",
                title: "Motorcycling",
                desc: "High-performance riding",
                color: "from-gray-500 to-slate-500",
              },
              {
                icon: "🎹",
                title: "Music",
                desc: "Classical piano mastery",
                color: "from-violet-500 to-purple-500",
              },
            ].map((skill, idx) => (
              <motion.div
                key={skill.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.1, duration: 0.6 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10 hover:border-white/20 transition-all duration-300 cursor-pointer group"
              >
                <div
                  className={`w-16 h-16 bg-gradient-to-br ${skill.color} rounded-2xl flex items-center justify-center text-2xl mb-4 group-hover:scale-110 transition-transform duration-300`}
                >
                  {skill.icon}
                </div>
                <h3 className="text-white font-semibold text-lg mb-2">
                  {skill.title}
                </h3>
                <p className="text-white/70 text-sm">{skill.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-4xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-white mb-6">
              Let's Build the
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                {" "}
                Future Together
              </span>
            </h2>
            <p className="text-white/70 text-lg">
              Available for consulting, speaking engagements, and collaboration
              opportunities
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            {/* Contact Links */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="space-y-6"
            >
              <motion.a
                href="https://www.linkedin.com/in/aakritigupta4894/"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02, x: 5 }}
                className="flex items-center gap-4 bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10 hover:border-blue-500/50 transition-all duration-300 group"
              >
                <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center text-white font-bold text-xl group-hover:scale-110 transition-transform duration-300">
                  in
                </div>
                <div>
                  <h3 className="text-white font-semibold">LinkedIn</h3>
                  <p className="text-white/70 text-sm">
                    Professional network & experience
                  </p>
                </div>
                <div className="ml-auto text-blue-400 group-hover:translate-x-2 transition-transform duration-300">
                  →
                </div>
              </motion.a>

              <motion.a
                href="https://github.com/aakritigupta0408?tab=achievements"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02, x: 5 }}
                className="flex items-center gap-4 bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10 hover:border-gray-500/50 transition-all duration-300 group"
              >
                <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center text-white font-bold text-lg group-hover:scale-110 transition-transform duration-300">
                  Git
                </div>
                <div>
                  <h3 className="text-white font-semibold">GitHub</h3>
                  <p className="text-white/70 text-sm">
                    Open source contributions & projects
                  </p>
                </div>
                <div className="ml-auto text-gray-400 group-hover:translate-x-2 transition-transform duration-300">
                  →
                </div>
              </motion.a>
            </motion.div>

            {/* Email Subscription */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="bg-white/5 backdrop-blur-md rounded-2xl p-8 border border-white/10"
            >
              <h3 className="text-white font-semibold text-xl mb-4">
                Get Exclusive Updates
              </h3>
              <p className="text-white/70 mb-6">
                Stay informed about latest AI research, projects, and speaking
                engagements
              </p>

              <form onSubmit={handleEmailSubmit} className="space-y-4">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your@email.com"
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:border-blue-500 focus:outline-none transition-colors"
                  required
                />
                <motion.button
                  type="submit"
                  disabled={emailSubmitted}
                  whileHover={{ scale: emailSubmitted ? 1 : 1.02 }}
                  whileTap={{ scale: emailSubmitted ? 1 : 0.98 }}
                  className={`w-full py-3 rounded-xl font-semibold transition-all duration-300 ${
                    emailSubmitted
                      ? "bg-green-600 text-white"
                      : "bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700"
                  }`}
                >
                  {emailSubmitted ? "✓ Subscribed!" : "Subscribe"}
                </motion.button>
              </form>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-20 py-12 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-white/50 text-sm">
            © 2024 Aakriti Gupta • Senior ML Engineer • AI Researcher • Luxury
            Tech Visionary
          </p>
          <div className="mt-4 flex justify-center gap-8 text-xs text-white/30">
            <span>Delhi to Silicon Valley</span>
            <span>•</span>
            <span>Meta • eBay • Yahoo</span>
            <span>•</span>
            <span>AI • Luxury • Innovation</span>
          </div>
        </div>
      </footer>

      <ChatBot />
    </div>
  );
}
