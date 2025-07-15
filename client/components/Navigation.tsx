import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate, useLocation } from "react-router-dom";

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const talents = [
    {
      id: "ai-researcher",
      title: "AI RESEARCHER",
      subtitle: "Innovation & Discovery",
      icon: "🔬",
      color: "from-cyan-500 to-cyan-700",
      description: "Advanced machine learning research and AI innovation",
    },
    {
      id: "social-entrepreneur",
      title: "SOCIAL ENTREPRENEUR",
      subtitle: "Impact & Vision",
      icon: "🌍",
      color: "from-teal-500 to-teal-700",
      description: "Building technology for social good and impact",
    },
    {
      id: "marksman",
      title: "MARKSMAN",
      subtitle: "Precision & Focus",
      icon: "🎯",
      color: "from-red-500 to-red-700",
      description: "Expert precision shooting and tactical training",
    },
    {
      id: "equestrian",
      title: "EQUESTRIAN",
      subtitle: "Grace & Partnership",
      icon: "🐎",
      color: "from-amber-500 to-amber-700",
      description: "Professional horse riding and equestrian arts",
    },
    {
      id: "aviator",
      title: "AVIATOR",
      subtitle: "Sky Mastery",
      icon: "✈️",
      color: "from-blue-500 to-blue-700",
      description: "Pilot training and aviation excellence",
    },
    {
      id: "motorcyclist",
      title: "MOTORCYCLIST",
      subtitle: "Speed & Freedom",
      icon: "🏍️",
      color: "from-purple-500 to-purple-700",
      description: "High-performance motorcycle expertise",
    },
    {
      id: "pianist",
      title: "PIANIST",
      subtitle: "Musical Artistry",
      icon: "🎹",
      color: "from-green-500 to-green-700",
      description: "Classical and contemporary piano mastery",
    },
  ];

  const navLinks = [
    { path: "/", label: "PORTFOLIO", emoji: "🏠" },
    { path: "/games", label: "GAMES", emoji: "🎮" },
    { path: "/ai-playground", label: "AI PLAYGROUND", emoji: "🤖" },
    { path: "/ai-discoveries", label: "AI DISCOVERIES", emoji: "🔬" },
    { path: "/ai-tools", label: "AI TOOLS", emoji: "🛠️" },
    { path: "/ai-companies", label: "AI COMPANIES", emoji: "🏢" },
    { path: "/ai-projects", label: "AI PROJECTS", emoji: "🚀" },
  ];

  // Simulated Perplexity search function
  const searchWebsite = async (query: string) => {
    if (!query.trim()) return [];

    setIsSearching(true);

    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Mock search results based on website content
    const mockResults = [
      {
        title: "AI Tools for Every Profession",
        url: "/ai-tools",
        description:
          "Discover the most impactful AI tools transforming the top 20 professions. Each recommendation includes pricing, features, and time-saving potential.",
        type: "page",
      },
      {
        title: "20 Fundamental AI Discoveries",
        url: "/ai-discoveries",
        description:
          "Explore the groundbreaking discoveries that shaped artificial intelligence from 1950 to 2018 with interactive demos.",
        type: "page",
      },
      {
        title: "AI Companies Leading the Revolution",
        url: "/ai-companies",
        description:
          "Top 20 companies shaping the AI landscape with their groundbreaking discoveries and innovative products.",
        type: "page",
      },
      {
        title: "Interactive Games Collection",
        url: "/games",
        description:
          "A curated collection of classic and modern games showcasing AI algorithms and strategic thinking.",
        type: "page",
      },
      {
        title: "AI Projects & Solutions Guide",
        url: "/ai-projects",
        description:
          "Comprehensive guide to the most common AI projects with step-by-step training approaches and code examples.",
        type: "page",
      },
    ].filter(
      (result) =>
        result.title.toLowerCase().includes(query.toLowerCase()) ||
        result.description.toLowerCase().includes(query.toLowerCase()),
    );

    setSearchResults(mockResults);
    setIsSearching(false);
  };

  useEffect(() => {
    if (searchQuery) {
      const debounceTimer = setTimeout(() => {
        searchWebsite(searchQuery);
      }, 300);

      return () => clearTimeout(debounceTimer);
    } else {
      setSearchResults([]);
    }
  }, [searchQuery]);

  useEffect(() => {
    if (isSearchOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isSearchOpen]);

  const handleTalentClick = (talentId: string) => {
    navigate(`/talent/${talentId}`);
    setIsOpen(false);
  };

  const handleSearchResultClick = (url: string) => {
    navigate(url);
    setIsSearchOpen(false);
    setSearchQuery("");
    setSearchResults([]);
  };

  const isHomePage = location.pathname === "/";

  return (
    <>
      {/* Adaptive Navigation Header */}
      <motion.header
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="fixed top-0 left-0 right-0 z-50 bg-black/60 backdrop-blur-2xl border-b border-white/10 shadow-2xl"
        style={{
          background:
            "linear-gradient(135deg, rgba(0,0,0,0.7) 0%, rgba(30,30,60,0.8) 50%, rgba(0,0,0,0.7) 100%)",
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16 lg:h-20">
            {/* Logo/Home */}
            <motion.button
              onClick={() => navigate("/")}
              whileHover={{ scale: 1.02 }}
              className="flex items-center gap-3 flex-shrink-0"
            >
              <div className="w-10 h-10 lg:w-12 lg:h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center border border-white/20 shadow-lg">
                <span className="text-white font-black text-lg lg:text-xl">
                  AG
                </span>
              </div>
              <div className="hidden sm:block text-left">
                <div className="text-white font-black text-sm lg:text-base">
                  AAKRITI GUPTA
                </div>
                <div className="text-gray-300 text-xs lg:text-sm">
                  ML ENGINEER & AI RESEARCHER
                </div>
              </div>
            </motion.button>

            {/* Center Navigation - Hidden on small screens */}
            <nav className="hidden xl:flex items-center gap-1">
              {navLinks.map((link) => (
                <motion.button
                  key={link.path}
                  onClick={() => navigate(link.path)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`px-4 py-2 rounded-xl text-xs font-bold tracking-wider transition-all duration-300 ${
                    location.pathname === link.path
                      ? "bg-gradient-to-r from-blue-500/30 to-purple-500/30 text-white border border-blue-400/50"
                      : "text-gray-300 hover:text-white hover:bg-white/10"
                  }`}
                >
                  <span className="mr-1">{link.emoji}</span>
                  {link.label}
                </motion.button>
              ))}
            </nav>

            {/* Right Section */}
            <div className="flex items-center gap-2 lg:gap-4">
              {/* Enhanced Search Button */}
              <motion.button
                onClick={() => setIsSearchOpen(!isSearchOpen)}
                whileHover={{ scale: 1.1, rotate: 5 }}
                whileTap={{ scale: 0.9 }}
                className="relative p-2 lg:p-3 rounded-xl bg-gradient-to-r from-cyan-500/20 to-blue-500/20 backdrop-blur-md border border-white/30 text-white hover:from-cyan-500/30 hover:to-blue-500/30 transition-all duration-300 shadow-lg"
              >
                <svg
                  className="w-4 h-4 lg:w-5 lg:h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              </motion.button>

              {/* Talents Dropdown - Desktop */}
              <div className="hidden lg:block relative">
                <motion.button
                  onClick={() => setIsOpen(!isOpen)}
                  whileHover={{ scale: 1.05 }}
                  className="px-4 py-2 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 text-white hover:bg-white/20 transition-all duration-300 flex items-center gap-2 text-sm font-bold"
                >
                  🎭 TALENTS
                  <motion.span
                    animate={{ rotate: isOpen ? 180 : 0 }}
                    transition={{ duration: 0.3 }}
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
                      className="absolute top-full right-0 mt-2 w-80 bg-black/95 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl overflow-hidden"
                    >
                      <div className="p-2">
                        {talents.map((talent, index) => (
                          <motion.button
                            key={talent.id}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            onClick={() => handleTalentClick(talent.id)}
                            className="w-full p-3 text-left hover:bg-white/10 rounded-xl transition-all duration-300 group"
                          >
                            <div className="flex items-center gap-3">
                              <div className="text-2xl">{talent.icon}</div>
                              <div className="flex-1">
                                <div className="text-white text-sm font-bold group-hover:text-blue-400 transition-colors">
                                  {talent.title}
                                </div>
                                <div className="text-gray-400 text-xs mt-1">
                                  {talent.subtitle}
                                </div>
                              </div>
                              <div className="text-blue-400/60 group-hover:text-blue-400 group-hover:translate-x-1 transition-all duration-300">
                                →
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
                whileHover={{ scale: 1.05 }}
                className="hidden lg:block px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl text-sm font-bold hover:from-blue-600 hover:to-purple-700 transition-all duration-300 border border-blue-400/30"
              >
                📄 RESUME
              </motion.a>

              {/* Mobile Menu Button */}
              <motion.button
                onClick={() => setIsOpen(!isOpen)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="xl:hidden p-2 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 text-white"
              >
                <div className="w-5 h-5 flex flex-col justify-center items-center">
                  <motion.span
                    animate={{
                      rotate: isOpen ? 45 : 0,
                      y: isOpen ? 6 : 0,
                    }}
                    className="w-full h-0.5 bg-white mb-1 origin-center transition-all duration-300"
                  />
                  <motion.span
                    animate={{ opacity: isOpen ? 0 : 1 }}
                    className="w-full h-0.5 bg-white mb-1 transition-all duration-300"
                  />
                  <motion.span
                    animate={{
                      rotate: isOpen ? -45 : 0,
                      y: isOpen ? -6 : 0,
                    }}
                    className="w-full h-0.5 bg-white origin-center transition-all duration-300"
                  />
                </div>
              </motion.button>
            </div>
          </div>
        </div>

        {/* Search Overlay */}
        <AnimatePresence>
          {isSearchOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="border-t border-white/10 bg-black/95 backdrop-blur-xl"
            >
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                <div className="relative">
                  {/* Search Input */}
                  <div className="relative mb-4">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                      <svg
                        className="h-5 w-5 text-gray-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                        />
                      </svg>
                    </div>
                    <input
                      ref={searchInputRef}
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search the website with AI-powered search..."
                      className="w-full pl-12 pr-4 py-4 bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <span className="hidden sm:inline">Powered by</span>
                        <div className="flex items-center gap-1 bg-blue-500/20 px-2 py-1 rounded-lg border border-blue-400/30">
                          <div className="w-4 h-4 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full"></div>
                          <span className="text-blue-300 font-bold text-xs">
                            Perplexity
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Search Results */}
                  {isSearching && (
                    <div className="flex items-center justify-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                      <span className="ml-3 text-gray-300">
                        Searching with AI...
                      </span>
                    </div>
                  )}

                  {searchResults.length > 0 && !isSearching && (
                    <div className="space-y-2">
                      <div className="text-sm text-gray-400 mb-3">
                        Found {searchResults.length} result
                        {searchResults.length !== 1 ? "s" : ""}
                      </div>
                      {searchResults.map((result, index) => (
                        <motion.button
                          key={index}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.1 }}
                          onClick={() => handleSearchResultClick(result.url)}
                          className="w-full text-left p-4 bg-white/5 hover:bg-white/10 rounded-xl border border-white/10 hover:border-white/20 transition-all duration-300 group"
                        >
                          <div className="flex items-start gap-3">
                            <div className="text-2xl">
                              {result.type === "page" ? "📄" : "🔗"}
                            </div>
                            <div className="flex-1">
                              <h3 className="text-white font-bold group-hover:text-blue-400 transition-colors">
                                {result.title}
                              </h3>
                              <p className="text-gray-400 text-sm mt-1 line-clamp-2">
                                {result.description}
                              </p>
                              <div className="text-blue-400 text-xs mt-2 font-mono">
                                {result.url}
                              </div>
                            </div>
                            <div className="text-blue-400/60 group-hover:text-blue-400 group-hover:translate-x-1 transition-all duration-300">
                              →
                            </div>
                          </div>
                        </motion.button>
                      ))}
                    </div>
                  )}

                  {searchQuery &&
                    !isSearching &&
                    searchResults.length === 0 && (
                      <div className="text-center py-8 text-gray-400">
                        <div className="text-4xl mb-2">🔍</div>
                        <p>No results found for "{searchQuery}"</p>
                        <p className="text-sm mt-1">
                          Try searching for AI tools, discoveries, or companies
                        </p>
                      </div>
                    )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Mobile Menu */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="xl:hidden bg-black/95 backdrop-blur-xl border-t border-white/10"
            >
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-3">
                {navLinks.map((link) => (
                  <motion.button
                    key={link.path}
                    onClick={() => {
                      navigate(link.path);
                      setIsOpen(false);
                    }}
                    className="block w-full text-left p-3 text-white hover:bg-white/10 rounded-xl transition-all duration-300 font-bold"
                  >
                    <span className="mr-3">{link.emoji}</span>
                    {link.label}
                  </motion.button>
                ))}

                <div className="border-t border-white/10 pt-4">
                  <div className="text-blue-400 font-bold text-sm mb-3">
                    🎭 TALENTS
                  </div>
                  {talents.map((talent) => (
                    <motion.button
                      key={talent.id}
                      onClick={() => handleTalentClick(talent.id)}
                      className="block w-full text-left p-3 hover:bg-white/10 rounded-xl transition-all duration-300 mb-2"
                    >
                      <div className="flex items-center gap-3">
                        <div className="text-2xl">{talent.icon}</div>
                        <div>
                          <div className="text-white text-sm font-bold">
                            {talent.title}
                          </div>
                          <div className="text-gray-400 text-xs">
                            {talent.subtitle}
                          </div>
                        </div>
                      </div>
                    </motion.button>
                  ))}
                </div>

                <motion.a
                  href="https://drive.google.com/file/d/1Mnmk6nP9l_Av0LvpgJQ5Tkjb7BqhY7nb/view?usp=sharing"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full text-center p-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-bold hover:from-blue-600 hover:to-purple-700 transition-all duration-300 border border-blue-400/30"
                >
                  📄 RESUME
                </motion.a>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.header>

      {/* Click outside to close */}
      <AnimatePresence>
        {(isOpen || isSearchOpen) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => {
              setIsOpen(false);
              setIsSearchOpen(false);
            }}
            className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
          />
        )}
      </AnimatePresence>
    </>
  );
};

export default Navigation;
