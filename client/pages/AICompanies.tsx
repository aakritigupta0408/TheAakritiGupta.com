import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import SubpageLayout from "@/components/SubpageLayout";
import {
  companies,
  companyCategories,
  type Company,
} from "@/data/companyArchive";
import { getPageRefreshContent } from "@/data/siteRefreshContent";
import {
  latestAIResearchBreakthroughs,
  startupWatchlist,
} from "../data/aiSignals";

type SortMode = "scale" | "founded" | "operating";

export default function AICompanies() {
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
  const [filterCategory, setFilterCategory] = useState("All");
  const [sortBy, setSortBy] = useState<SortMode>("scale");
  const [visibleCount, setVisibleCount] = useState(8);

  const pageRefresh = getPageRefreshContent("/ai-companies");

  const recentAdditionsCount = useMemo(
    () => companies.filter((company) => company.isRecentAddition).length,
    [],
  );

  const filteredCompanies = useMemo(() => {
    const filtered =
      filterCategory === "All"
        ? [...companies]
        : companies.filter((company) => company.category === filterCategory);

    filtered.sort((left, right) => {
      if (sortBy === "founded") {
        return right.sortFounded - left.sortFounded;
      }

      if (sortBy === "operating") {
        return right.sortOperating - left.sortOperating;
      }

      return right.sortScale - left.sortScale;
    });

    return filtered;
  }, [filterCategory, sortBy]);

  useEffect(() => {
    setVisibleCount(8);
  }, [filterCategory, sortBy]);

  const visibleCompanies = filteredCompanies.slice(0, visibleCount);
  const hasMoreCompanies = visibleCompanies.length < filteredCompanies.length;
  const visibleRecentAdditions = visibleCompanies.filter(
    (company) => company.isRecentAddition,
  ).length;

  return (
    <SubpageLayout
      route="/ai-companies"
      eyebrow={pageRefresh.eyebrow}
      title={pageRefresh.title}
      description={pageRefresh.description}
      accent="emerald"
      chips={pageRefresh.chips}
      refreshSummary={pageRefresh.refreshSummary}
      updatedAtLabel={pageRefresh.updatedAtLabel}
      metrics={[
        { value: companies.length.toString(), label: "Company cards" },
        { value: recentAdditionsCount.toString(), label: "Added since Aug 2025" },
        {
          value: (companyCategories.length - 1).toString(),
          label: "Filter categories",
        },
        {
          value: filteredCompanies.length.toString(),
          label: "Matches current filter",
        },
      ]}
    >
      <div className="container mx-auto px-6 py-10 sm:py-12">
        <motion.section
          className="mb-14 space-y-8"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <div className="text-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-300/30 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100">
              Startup and scale-up watch · April 2026
            </div>
            <h2 className="mt-4 text-3xl font-black text-white md:text-4xl">
              Who Has Momentum Right Now
            </h2>
            <p className="mx-auto mt-3 max-w-4xl text-gray-100 leading-relaxed">
              The archive below is now anchored to primary sources. Cards focus
              on current product direction, operating footprint, and why each
              company matters now instead of stale valuation trivia.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
            {startupWatchlist.slice(0, 6).map((item, index) => (
              <motion.a
                key={item.id}
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: 0.08 * index }}
                className="group rounded-3xl border border-white/15 bg-slate-950/25 p-6 shadow-2xl backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:bg-slate-950/35"
              >
                <div className="mb-4 flex items-start justify-between gap-4">
                  <div>
                    <div className="mb-3 inline-flex rounded-full border border-emerald-300/30 bg-emerald-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-emerald-100">
                      {item.focus}
                    </div>
                    <h3 className="text-2xl font-black text-white group-hover:text-emerald-200">
                      {item.name}
                    </h3>
                  </div>
                  <span className="text-xs font-semibold text-gray-300">
                    {item.date}
                  </span>
                </div>
                <p className="mb-4 text-sm leading-relaxed text-gray-100">
                  {item.latestMove}
                </p>
                <p className="text-sm leading-relaxed text-cyan-100">
                  {item.whyItMatters}
                </p>
              </motion.a>
            ))}
          </div>

          <div className="rounded-[2rem] border border-white/15 bg-white/10 p-8 backdrop-blur-xl">
            <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <h3 className="text-2xl font-black text-white md:text-3xl">
                  Research Frontier Driving Company Strategy
                </h3>
                <p className="mt-2 max-w-3xl text-gray-100">
                  The strongest companies now are translating frontier research
                  into deployable systems, not just publishing benchmarks.
                </p>
              </div>
              <p className="text-sm font-medium text-emerald-100">
                Official lab announcements
              </p>
            </div>

            <div className="grid gap-5 lg:grid-cols-3">
              {latestAIResearchBreakthroughs.slice(0, 3).map((signal, index) => (
                <motion.a
                  key={signal.id}
                  href={signal.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.45, delay: 0.1 * index }}
                  className="rounded-3xl border border-white/15 bg-black/20 p-5 transition-all duration-300 hover:bg-black/30"
                >
                  <div className="mb-3 flex items-center justify-between gap-4">
                    <span className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                      {signal.category}
                    </span>
                    <span className="text-xs font-semibold text-gray-300">
                      {signal.date}
                    </span>
                  </div>
                  <h4 className="mb-2 text-lg font-black text-white">
                    {signal.title}
                  </h4>
                  <p className="mb-2 text-sm font-semibold text-emerald-100">
                    {signal.org}
                  </p>
                  <p className="mb-3 text-sm leading-relaxed text-gray-100">
                    {signal.summary}
                  </p>
                  <p className="text-sm leading-relaxed text-gray-300">
                    {signal.impact}
                  </p>
                </motion.a>
              ))}
            </div>
          </div>
        </motion.section>

        <section className="mb-12 space-y-6">
          <div className="mx-auto flex max-w-5xl flex-col gap-4 rounded-3xl border border-white/15 bg-white/10 p-5 text-center backdrop-blur-xl md:flex-row md:items-center md:justify-between md:text-left">
            <div>
              <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-100">
                Company Grid Status
              </p>
              <p className="mt-2 text-lg font-semibold text-white">
                Showing {visibleCompanies.length} of {filteredCompanies.length} companies
              </p>
              <p className="mt-1 text-sm text-gray-200">
                {visibleRecentAdditions} recent additions are visible in this
                view.
              </p>
            </div>
            {filterCategory !== "All" && (
              <motion.button
                onClick={() => setFilterCategory("All")}
                className="rounded-full border border-emerald-300/40 bg-emerald-400/15 px-5 py-3 text-sm font-bold text-white transition-all duration-300 hover:bg-emerald-400/25"
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                Clear Category Filter
              </motion.button>
            )}
          </div>

          <div className="flex flex-wrap justify-center gap-4">
            <span className="rounded-full border border-white/20 bg-white/10 px-6 py-3 text-sm font-black text-white backdrop-blur-md">
              Filter by Category
            </span>
            {companyCategories.map((category) => (
              <motion.button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`rounded-full px-6 py-3 text-sm font-bold transition-all duration-300 ${
                  filterCategory === category
                    ? "scale-105 border border-emerald-400/50 bg-gradient-to-r from-emerald-500 to-cyan-600 text-white shadow-2xl"
                    : "border border-white/20 bg-white/10 text-white backdrop-blur-md hover:bg-white/20"
                }`}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                {category}
              </motion.button>
            ))}
          </div>

          <div className="flex flex-wrap justify-center gap-4">
            <span className="rounded-full border border-white/20 bg-white/10 px-6 py-3 text-sm font-black text-white backdrop-blur-md">
              Sort by
            </span>
            {[
              { value: "scale", label: "Market Relevance" },
              { value: "operating", label: "Operating Footprint" },
              { value: "founded", label: "Founded Date" },
            ].map((option) => (
              <motion.button
                key={option.value}
                onClick={() => setSortBy(option.value as SortMode)}
                className={`rounded-full px-6 py-3 text-sm font-bold transition-all duration-300 ${
                  sortBy === option.value
                    ? "scale-105 border border-teal-400/50 bg-gradient-to-r from-teal-500 to-cyan-600 text-white shadow-2xl"
                    : "border border-white/20 bg-white/10 text-white backdrop-blur-md hover:bg-white/20"
                }`}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                {option.label}
              </motion.button>
            ))}
          </div>
        </section>

        {filteredCompanies.length === 0 ? (
          <div className="mb-12 rounded-[2rem] border border-white/15 bg-white/10 p-10 text-center shadow-2xl backdrop-blur-xl">
            <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-100">
              No Matches Right Now
            </p>
            <h3 className="mt-3 text-3xl font-black text-white">
              The current filters hid every company card.
            </h3>
            <p className="mx-auto mt-3 max-w-2xl text-gray-100">
              Reset the category filter to compare frontier labs, enterprise
              platforms, infrastructure leaders, and AI-native startups again.
            </p>
          </div>
        ) : (
          <div className="mb-12 grid grid-cols-1 gap-8 md:grid-cols-2 xl:grid-cols-4">
            {visibleCompanies.map((company, index) => (
              <motion.button
                key={company.id}
                type="button"
                onClick={() => setSelectedCompany(company)}
                initial={{ opacity: 0, y: 28 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: Math.min(index * 0.04, 0.24) }}
                className="group rounded-3xl border border-white/20 bg-white/10 p-7 text-left shadow-2xl backdrop-blur-xl transition-all duration-300 hover:-translate-y-2 hover:bg-white/15"
              >
                <div className="mb-5 flex items-start justify-between gap-4">
                  <div className="text-5xl">{company.logo}</div>
                  <span className="rounded-full border border-emerald-300/30 bg-emerald-400/10 px-3 py-1 text-[11px] font-black uppercase tracking-[0.18em] text-emerald-100">
                    {company.category}
                  </span>
                </div>

                {company.isRecentAddition && (
                  <div className="mb-4 inline-flex rounded-full border border-cyan-300/40 bg-cyan-400/10 px-3 py-1 text-[11px] font-black uppercase tracking-[0.2em] text-cyan-100">
                    New since Aug 2025
                  </div>
                )}

                <h3 className="mb-3 text-xl font-black text-white group-hover:text-emerald-200">
                  {company.name}
                </h3>
                <p className="mb-5 line-clamp-3 text-sm leading-relaxed text-gray-200">
                  {company.description}
                </p>

                <div className="space-y-3">
                  <div className="rounded-2xl border border-white/15 bg-black/20 p-3">
                    <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gray-300">
                      Founded
                    </p>
                    <p className="mt-1 text-sm font-semibold text-white">
                      {company.founded}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/15 bg-black/20 p-3">
                    <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gray-300">
                      Scale signal
                    </p>
                    <p className="mt-1 text-sm font-semibold text-emerald-200">
                      {company.scaleSignal}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/15 bg-black/20 p-3">
                    <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gray-300">
                      Operating signal
                    </p>
                    <p className="mt-1 line-clamp-2 text-sm font-semibold text-white">
                      {company.operatingSignal}
                    </p>
                  </div>
                </div>

                <div className="mt-6 border-t border-white/15 pt-5">
                  <div className="flex flex-wrap gap-2">
                    {company.keyProducts.slice(0, 3).map((product) => (
                      <span
                        key={product}
                        className="rounded-full border border-white/20 bg-white/10 px-3 py-2 text-xs font-medium text-gray-100"
                      >
                        {product}
                      </span>
                    ))}
                  </div>
                  <p className="mt-4 text-sm font-bold text-cyan-200">
                    Open source-backed details and sources
                  </p>
                </div>
              </motion.button>
            ))}
          </div>
        )}

        {hasMoreCompanies && (
          <div className="mb-12 flex justify-center">
            <motion.button
              onClick={() => setVisibleCount((current) => current + 8)}
              className="rounded-full border border-white/20 bg-white/10 px-6 py-3 text-sm font-bold text-white shadow-xl backdrop-blur-md transition-all duration-300 hover:bg-white/15"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              Show 8 more companies
            </motion.button>
          </div>
        )}

        <AnimatePresence>
          {selectedCompany && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
              onClick={() => setSelectedCompany(null)}
            >
              <motion.div
                initial={{ scale: 0.96, opacity: 0, y: 12 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.96, opacity: 0, y: 12 }}
                className="max-h-[90vh] w-full max-w-6xl overflow-y-auto rounded-[2rem] border border-white/20 bg-[#f8fafc] shadow-2xl"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="p-6 sm:p-8">
                  <div className="mb-8 flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-6">
                      <div className="text-6xl">{selectedCompany.logo}</div>
                      <div>
                        <h2 className="text-3xl font-black text-slate-950 sm:text-4xl">
                          {selectedCompany.name}
                        </h2>
                        <div className="mt-3 flex flex-wrap gap-3">
                          <span className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white">
                            {selectedCompany.category}
                          </span>
                          <span className="rounded-full bg-emerald-100 px-4 py-2 text-sm font-semibold text-emerald-800">
                            Founded {selectedCompany.founded}
                          </span>
                          {selectedCompany.isRecentAddition && (
                            <span className="rounded-full bg-cyan-100 px-4 py-2 text-sm font-semibold text-cyan-800">
                              Added since Aug 2025
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedCompany(null)}
                      className="self-end rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 lg:self-start"
                    >
                      Close
                    </button>
                  </div>

                  <div className="mb-8 grid gap-6 lg:grid-cols-[1.05fr,1.4fr]">
                    <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                      <h3 className="text-lg font-black text-slate-950">
                        Source-Backed Snapshot
                      </h3>
                      <div className="mt-5 space-y-4 text-sm text-slate-700">
                        <div>
                          <p className="font-semibold text-slate-500">Headquarters</p>
                          <p className="mt-1">{selectedCompany.headquarters}</p>
                        </div>
                        <div>
                          <p className="font-semibold text-slate-500">Founders</p>
                          <p className="mt-1">{selectedCompany.founders.join(", ")}</p>
                        </div>
                        <div>
                          <p className="font-semibold text-slate-500">Scale signal</p>
                          <p className="mt-1">{selectedCompany.scaleSignal}</p>
                        </div>
                        <div>
                          <p className="font-semibold text-slate-500">Operating signal</p>
                          <p className="mt-1">{selectedCompany.operatingSignal}</p>
                        </div>
                        {selectedCompany.stockSymbol && (
                          <div>
                            <p className="font-semibold text-slate-500">Ticker</p>
                            <p className="mt-1">{selectedCompany.stockSymbol}</p>
                          </div>
                        )}
                        <div>
                          <p className="font-semibold text-slate-500">Website</p>
                          <a
                            href={selectedCompany.website}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mt-1 inline-block text-cyan-700 hover:underline"
                          >
                            {selectedCompany.website.replace(/^https?:\/\//, "")}
                          </a>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                      <h3 className="text-lg font-black text-slate-950">What Matters Now</h3>
                      <p className="mt-4 text-base leading-relaxed text-slate-700">
                        {selectedCompany.description}
                      </p>
                      <div className="mt-5 rounded-2xl bg-slate-50 p-4">
                        <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                          Current focus
                        </p>
                        <p className="mt-2 leading-relaxed text-slate-700">
                          {selectedCompany.currentFocus}
                        </p>
                      </div>
                      <div className="mt-5">
                        <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                          Company arc
                        </p>
                        <p className="mt-2 leading-relaxed text-slate-700">
                          {selectedCompany.journey}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mb-8 grid gap-6 lg:grid-cols-2">
                    <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                      <h3 className="text-lg font-black text-slate-950">
                        Landmark Contributions
                      </h3>
                      <div className="mt-4 space-y-3">
                        {selectedCompany.landmarkContributions.map((item) => (
                          <div
                            key={item}
                            className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm leading-relaxed text-slate-700"
                          >
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                      <h3 className="text-lg font-black text-slate-950">Key Products</h3>
                      <div className="mt-4 flex flex-wrap gap-3">
                        {selectedCompany.keyProducts.map((item) => (
                          <span
                            key={item}
                            className="rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-900"
                          >
                            {item}
                          </span>
                        ))}
                      </div>

                      <h4 className="mt-6 text-sm font-black uppercase tracking-[0.18em] text-slate-500">
                        Why teams still watch this company
                      </h4>
                      <div className="mt-3 space-y-3">
                        {selectedCompany.achievements.map((item) => (
                          <div
                            key={item}
                            className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm leading-relaxed text-slate-700"
                          >
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                      <div>
                        <h3 className="text-lg font-black text-slate-950">Primary Sources</h3>
                        <p className="mt-1 text-sm text-slate-600">
                          Updated from official company pages, docs, and release posts.
                        </p>
                      </div>
                    </div>
                    <div className="mt-5 grid gap-4 md:grid-cols-2">
                      {selectedCompany.sources.map((source) => (
                        <a
                          key={source.url}
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="rounded-2xl border border-slate-200 bg-slate-50 p-4 transition hover:border-cyan-300 hover:bg-cyan-50"
                        >
                          <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                            {source.kind}
                          </p>
                          <p className="mt-2 text-sm font-semibold text-slate-900">
                            {source.label}
                          </p>
                          <p className="mt-2 text-sm text-cyan-700">
                            Open source →
                          </p>
                        </a>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </SubpageLayout>
  );
}
