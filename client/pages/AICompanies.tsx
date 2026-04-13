import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import SubpageLayout from "@/components/SubpageLayout";
import {
  companies,
  companyCategories,
  type Company,
} from "@/data/companyArchive";
import { getPageRefreshContent } from "@/data/siteRefreshContent";
import { startupWatchlist } from "../data/aiSignals";

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
        { value: recentAdditionsCount.toString(), label: "Recently added" },
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
          className="mb-8 rounded-[1.75rem] border border-white/10 bg-slate-950/25 p-6 backdrop-blur-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-4 flex items-baseline justify-between gap-3">
            <h2 className="text-lg font-semibold text-white">
              Who has momentum
            </h2>
            <span className="text-[11px] uppercase tracking-[0.2em] text-gray-400">
              Startup & scale-up watch
            </span>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {startupWatchlist.slice(0, 6).map((item) => (
              <a
                key={item.id}
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:bg-white/10"
              >
                <div className="mb-2 flex items-center justify-between gap-2 text-[11px] uppercase tracking-[0.18em] text-gray-400">
                  <span className="bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">
                    {item.focus}
                  </span>
                  <span>{item.date}</span>
                </div>
                <h3 className="text-sm font-semibold leading-snug text-white">
                  {item.name}
                </h3>
                <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-gray-300">
                  {item.latestMove}
                </p>
              </a>
            ))}
          </div>
        </motion.section>

        <section className="mb-6 space-y-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 backdrop-blur-md">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400">
              Category
            </span>
            {companyCategories.map((category) => (
              <button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                  filterCategory === category
                    ? "bg-gradient-to-r from-emerald-500 to-cyan-500 text-white"
                    : "border border-white/15 bg-white/5 text-gray-200 hover:bg-white/10"
                }`}
              >
                {category}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <select
                value={sortBy}
                onChange={(event) => setSortBy(event.target.value as SortMode)}
                className="rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-semibold text-gray-100 focus:outline-none"
              >
                <option value="scale" className="bg-slate-900">
                  Market relevance
                </option>
                <option value="operating" className="bg-slate-900">
                  Operating footprint
                </option>
                <option value="founded" className="bg-slate-900">
                  Founded date
                </option>
              </select>
            </div>
            <span className="text-xs text-gray-400">
              {visibleCompanies.length} / {filteredCompanies.length}
            </span>
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
                    Recently added
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
                              Recently added
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
