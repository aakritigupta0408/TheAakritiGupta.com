import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate, useLocation } from "react-router-dom";
import { searchSiteContent, type SiteSearchEntry } from "@/data/siteSearch";

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [isResourcesOpen, setIsResourcesOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SiteSearchEntry[]>([]);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const resumeUrl =
    "https://drive.google.com/file/d/1Mnmk6nP9l_Av0LvpgJQ5Tkjb7BqhY7nb/view?usp=sharing";

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
    { path: "/", label: "HOME", emoji: "🏠" },
    { path: "/ai-playground", label: "INTERACTIVE DEMOS", emoji: "🎮" },
    { path: "/ai-champions", label: "AI VS HUMANS", emoji: "🏆" },
    { path: "/ai-discoveries", label: "AI DISCOVERIES", emoji: "🔬" },
    { path: "/ai-tools", label: "AI TOOLS", emoji: "🛠️" },
    { path: "/ai-companies", label: "AI COMPANIES", emoji: "🏢" },
    { path: "/ai-projects", label: "AI PROJECTS", emoji: "🚀" },
    { path: "/prompt-engineering", label: "PROMPT MASTERY", emoji: "✨" },
    { path: "/ai-agent-training", label: "AGENT TRAINING", emoji: "🤖" },
    { path: "/resume-builder", label: "RESUME BUILDER", emoji: "📄" },
    { path: "/games", label: "GAMES", emoji: "🎯" },
  ];

  const resourceLinks = [
    {
      href: resumeUrl,
      label: "RESUME",
      emoji: "📄",
      description: "Open Aakriti's current resume",
    },
  ];

  useEffect(() => {
    if (searchQuery) {
      setSearchResults(searchSiteContent(searchQuery));
      return;
    } else {
      setSearchResults([]);
    }
  }, [searchQuery]);

  useEffect(() => {
    if (isSearchOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isSearchOpen]);

  const closeMenus = () => {
    setIsAboutOpen(false);
    setIsResourcesOpen(false);
    setIsSearchOpen(false);
    setIsMobileMenuOpen(false);
  };

  const handleNavigate = (path: string) => {
    navigate(path);
    closeMenus();
    setSearchQuery("");
    setSearchResults([]);
  };

  const handleTalentClick = (talentId: string) => {
    navigate(`/talent/${talentId}`);
    closeMenus();
  };

  const handleSearchResultClick = (url: string) => {
    handleNavigate(url);
  };

  return (
    <>
      {/* Floating AI Assistant Button */}
      <motion.div
        className="fixed bottom-8 right-8 z-40"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 2, duration: 0.5, type: "spring", bounce: 0.5 }}
      >
        <motion.button
          aria-label="Open AI website search"
          whileHover={{ scale: 1.1, rotate: 5 }}
          whileTap={{ scale: 0.9 }}
          className="group relative flex h-14 w-14 items-center justify-center rounded-full border border-slate-200/80 bg-white/90 text-xl text-slate-900 shadow-[0_18px_40px_rgba(15,23,42,0.16)] backdrop-blur-xl"
          onClick={() => {
            setIsAboutOpen(false);
            setIsResourcesOpen(false);
            setIsMobileMenuOpen(false);
            setIsSearchOpen(true);
            if (searchInputRef.current) {
              searchInputRef.current.focus();
            }
          }}
        >
          <motion.span
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            🤖
          </motion.span>

          {/* Pulsing ring */}
          <motion.div
            className="absolute inset-0 rounded-full border border-sky-300/70"
            animate={{ scale: [1, 1.5], opacity: [1, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          />

          {/* Tooltip */}
          <div className="pointer-events-none absolute bottom-16 right-0 whitespace-nowrap rounded-full border border-slate-200 bg-white/95 px-3 py-2 text-xs font-medium text-slate-700 opacity-0 shadow-lg transition-opacity duration-300 group-hover:opacity-100">
            Search this site
          </div>
        </motion.button>
      </motion.div>

      {/* Adaptive Navigation Header */}
      <motion.header
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="luxury-nav fixed top-0 left-0 right-0 z-50"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between gap-3">
            {/* Logo/Home */}
            <motion.button
              onClick={() => navigate("/")}
              whileHover={{ scale: 1.01 }}
              className="flex flex-shrink-0 items-center gap-3 text-left"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-slate-200 bg-white shadow-[0_10px_24px_rgba(15,23,42,0.08)]">
                <span className="bg-gradient-to-br from-slate-900 to-slate-500 bg-clip-text text-base font-black text-transparent">
                  AG
                </span>
              </div>
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold tracking-[-0.02em] text-slate-900 sm:text-[15px]">
                  AAKRITI GUPTA
                </div>
                <div className="hidden text-[11px] font-medium tracking-[0.01em] text-slate-500 sm:block">
                  ML Engineer and AI Researcher
                </div>
              </div>
            </motion.button>

            {/* Right Section */}
            <div className="flex items-center gap-2">
              {/* Enhanced Search Button */}
              <motion.button
                aria-label="Open website search"
                onClick={() => {
                  setIsSearchOpen((current) => !current);
                  setIsAboutOpen(false);
                  setIsResourcesOpen(false);
                  setIsMobileMenuOpen(false);
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.9 }}
                className="rounded-full border border-slate-200 bg-white/85 p-2.5 text-slate-700 transition-all duration-300 hover:bg-white"
              >
                <svg
                  className="h-4 w-4"
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

              {/* Free Resources Dropdown - Desktop */}
              <div className="relative hidden lg:block">
                <motion.button
                  onClick={() => {
                    setIsResourcesOpen((current) => !current);
                    setIsAboutOpen(false);
                  }}
                  whileHover={{ scale: 1.02 }}
                  className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/85 px-3 py-2 text-[11px] font-medium tracking-[0.02em] text-slate-700 transition-all duration-300 hover:bg-white"
                >
                  Free Resources
                  <motion.span
                    animate={{ rotate: isResourcesOpen ? 180 : 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    ▼
                  </motion.span>
                </motion.button>

                <AnimatePresence>
                  {isResourcesOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-full right-0 mt-3 w-72 overflow-hidden rounded-3xl border border-slate-200 bg-white/96 backdrop-blur-xl shadow-[0_24px_60px_rgba(15,23,42,0.14)]"
                    >
                      <div className="p-2">
                        {resourceLinks.map((resource, index) => (
                          <motion.a
                            key={resource.label}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            href={resource.href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block w-full rounded-xl p-3 text-left transition-all duration-300 hover:bg-white/10 group"
                            onClick={() => closeMenus()}
                          >
                            <div className="flex items-center gap-3">
                              <div className="text-2xl">{resource.emoji}</div>
                              <div className="flex-1">
                                <div className="text-sm font-semibold text-slate-900 transition-colors group-hover:text-sky-600">
                                  {resource.label}
                                </div>
                                <div className="mt-1 text-xs text-slate-500">
                                  {resource.description}
                                </div>
                              </div>
                              <div className="text-slate-300 transition-all duration-300 group-hover:translate-x-1 group-hover:text-sky-600">
                                →
                              </div>
                            </div>
                          </motion.a>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* About Dropdown - Desktop */}
              <div className="relative hidden lg:block">
                <motion.button
                  onClick={() => {
                    setIsAboutOpen((current) => !current);
                    setIsResourcesOpen(false);
                  }}
                  whileHover={{ scale: 1.02 }}
                  className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/85 px-3 py-2 text-[11px] font-medium tracking-[0.02em] text-slate-700 transition-all duration-300 hover:bg-white"
                >
                  Know More About AG
                  <motion.span
                    animate={{ rotate: isAboutOpen ? 180 : 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    ▼
                  </motion.span>
                </motion.button>

                {/* Dropdown Menu */}
                <AnimatePresence>
                  {isAboutOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-full right-0 mt-3 w-80 overflow-hidden rounded-3xl border border-slate-200 bg-white/96 shadow-[0_24px_60px_rgba(15,23,42,0.14)] backdrop-blur-xl"
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
                                <div className="text-sm font-semibold text-slate-900 transition-colors group-hover:text-sky-600">
                                  {talent.title}
                                </div>
                                <div className="mt-1 text-xs text-slate-500">
                                  {talent.subtitle}
                                </div>
                              </div>
                              <div className="text-slate-300 transition-all duration-300 group-hover:translate-x-1 group-hover:text-sky-600">
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

              {/* Mobile Menu Button */}
              <motion.button
                aria-label="Open navigation menu"
                onClick={() => {
                  setIsMobileMenuOpen((current) => !current);
                  setIsAboutOpen(false);
                  setIsResourcesOpen(false);
                  setIsSearchOpen(false);
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="rounded-full border border-slate-200 bg-white/85 p-2 text-slate-700 lg:hidden"
              >
                <div className="w-5 h-5 flex flex-col justify-center items-center">
                  <motion.span
                    animate={{
                      rotate: isMobileMenuOpen ? 45 : 0,
                      y: isMobileMenuOpen ? 6 : 0,
                    }}
                    className="mb-1 h-0.5 w-full origin-center bg-slate-700 transition-all duration-300"
                  />
                  <motion.span
                    animate={{ opacity: isMobileMenuOpen ? 0 : 1 }}
                    className="mb-1 h-0.5 w-full bg-slate-700 transition-all duration-300"
                  />
                  <motion.span
                    animate={{
                      rotate: isMobileMenuOpen ? -45 : 0,
                      y: isMobileMenuOpen ? -6 : 0,
                    }}
                    className="h-0.5 w-full origin-center bg-slate-700 transition-all duration-300"
                  />
                </div>
              </motion.button>
            </div>
          </div>

          <div className="hidden h-12 items-center border-t border-slate-200/80 lg:flex">
            <nav className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto py-2 [scrollbar-width:none] [-ms-overflow-style:none]">
              {navLinks.map((link) => (
                <motion.button
                  key={link.path}
                  onClick={() => handleNavigate(link.path)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`shrink-0 whitespace-nowrap rounded-full px-3 py-2 text-[10px] font-medium tracking-[0.02em] transition-all duration-300 xl:text-[11px] ${
                    location.pathname === link.path
                      ? "border border-slate-300 bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-white hover:text-slate-900"
                  }`}
                >
                  {link.label}
                </motion.button>
              ))}
            </nav>
          </div>
        </div>

        {/* Search Overlay */}
        <AnimatePresence>
          {isSearchOpen && (
            <motion.div
              initial={{ opacity: 0, scaleY: 0 }}
              animate={{ opacity: 1, scaleY: 1 }}
              exit={{ opacity: 0, scaleY: 0 }}
              style={{ originY: 0 }}
              transition={{ duration: 0.3 }}
              className="border-t border-slate-200 bg-[#fbfbfd]/95 backdrop-blur-2xl"
            >
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                <div className="relative">
                  {/* Search Input */}
                  <div className="relative mb-4">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                      <svg
                      className="h-5 w-5 text-slate-400"
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
                      placeholder="Search pages, companies, tools, profiles, and topics on this website..."
                      className="w-full rounded-3xl border border-slate-200 bg-white px-12 py-4 pr-4 text-slate-900 placeholder-slate-400 shadow-sm focus:border-transparent focus:outline-none focus:ring-2 focus:ring-sky-500"
                    />
                    <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                      <div className="hidden rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500 sm:block">
                        Website Search
                      </div>
                    </div>
                  </div>

                  <div className="mb-4 text-sm text-slate-500">
                    Searches information already available across this website, including pages, companies, tools, projects, discoveries, and profiles.
                  </div>

                  {/* Search Results */}
                  {searchResults.length > 0 && (
                    <div className="space-y-2">
                      <div className="mb-3 text-sm text-slate-500">
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
                          className="group w-full rounded-3xl border border-slate-200 bg-white p-4 text-left shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md"
                        >
                          <div className="flex items-start gap-3">
                            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-100 text-lg text-slate-700">
                              {result.type === "Page"
                                ? "📄"
                                : result.type === "Profile"
                                  ? "👤"
                                  : result.type === "Company"
                                    ? "🏢"
                                    : result.type === "Tool"
                                      ? "🛠️"
                                      : result.type === "Project"
                                        ? "🚀"
                                        : "🔬"}
                            </div>
                            <div className="flex-1">
                              <div className="mb-2 flex flex-wrap items-center gap-2">
                                <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">
                                  {result.type}
                                </span>
                                <span className="text-xs font-medium text-slate-400">
                                  {result.section}
                                </span>
                              </div>
                              <h3 className="font-semibold text-slate-900 transition-colors group-hover:text-sky-600">
                                {result.title}
                              </h3>
                              <p className="mt-1 line-clamp-2 text-sm text-slate-500">
                                {result.description}
                              </p>
                              <div className="mt-2 font-mono text-xs text-sky-600">
                                {result.url}
                              </div>
                            </div>
                            <div className="text-slate-300 transition-all duration-300 group-hover:translate-x-1 group-hover:text-sky-600">
                              →
                            </div>
                          </div>
                        </motion.button>
                      ))}
                    </div>
                  )}

                  {searchQuery &&
                    searchResults.length === 0 && (
                      <div className="py-8 text-center text-slate-500">
                        <div className="text-4xl mb-2">🔍</div>
                        <p>No results found for "{searchQuery}"</p>
                        <p className="text-sm mt-1">
                          Try searching for company names, tools, talent profiles, projects, or research topics
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
          {isMobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, scaleY: 0 }}
              animate={{ opacity: 1, scaleY: 1 }}
              exit={{ opacity: 0, scaleY: 0 }}
              style={{ originY: 0 }}
              transition={{ duration: 0.3 }}
              className="border-t border-slate-200 bg-[#fbfbfd]/95 backdrop-blur-2xl lg:hidden"
            >
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-3">
                <div className="rounded-3xl border border-slate-200 bg-white p-3 shadow-sm">
                  <div className="mb-3 text-sm font-semibold text-slate-900">
                    Free Resources
                  </div>
                  {resourceLinks.map((resource) => (
                    <motion.a
                      key={resource.label}
                      href={resource.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={() => closeMenus()}
                      className="block w-full rounded-2xl p-3 text-left font-semibold text-slate-900 transition-all duration-300 hover:bg-slate-50"
                    >
                      <span className="mr-3">{resource.emoji}</span>
                      {resource.label}
                    </motion.a>
                  ))}
                </div>

                {navLinks.map((link) => (
                  <motion.button
                    key={link.path}
                    onClick={() => handleNavigate(link.path)}
                    className="block w-full rounded-2xl p-3 text-left font-semibold text-slate-900 transition-all duration-300 hover:bg-white"
                  >
                    {link.label}
                  </motion.button>
                ))}

                <div className="border-t border-slate-200 pt-4">
                  <div className="mb-3 text-sm font-semibold text-slate-900">
                    Know More About AG
                  </div>
                  {talents.map((talent) => (
                    <motion.button
                      key={talent.id}
                      onClick={() => handleTalentClick(talent.id)}
                      className="mb-2 block w-full rounded-2xl p-3 text-left transition-all duration-300 hover:bg-white"
                    >
                      <div className="flex items-center gap-3">
                        <div className="text-2xl">{talent.icon}</div>
                        <div>
                          <div className="text-sm font-semibold text-slate-900">
                            {talent.title}
                          </div>
                          <div className="text-xs text-slate-500">
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
        {(isAboutOpen || isResourcesOpen || isSearchOpen || isMobileMenuOpen) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeMenus}
            className="fixed inset-0 z-40 bg-slate-900/10 backdrop-blur-sm"
          />
        )}
      </AnimatePresence>
    </>
  );
};

export default Navigation;
