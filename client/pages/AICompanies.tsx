import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import LevelOneLoadMoreButton from "@/components/LevelOneLoadMoreButton";
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
          className="mb-8 rounded-[1.75rem] border border-white/10 bg-white/[0.06] p-4 backdrop-blur-xl"
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
          <div className="mb-12 rounded-[2rem] border border-white/15 bg-white/10 p-6 text-center shadow-2xl backdrop-blur-xl">
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
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  type: "spring",
                  stiffness: 260,
                  damping: 24,
                  delay: Math.min(index * 0.04, 0.22),
                }}
                whileHover={{ y: -6, rotateX: 2, rotateY: -2 }}
                whileTap={{ scale: 0.98 }}
                style={{ transformStyle: "preserve-3d", perspective: 900 }}
                className="group relative overflow-hidden rounded-[22px] border border-white/10 bg-gradient-to-br from-white/[0.09] to-white/[0.03] p-5 text-left shadow-[0_22px_52px_rgba(8,12,24,0.45),inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-2xl transition-colors duration-300 hover:border-white/25"
              >
                <span className="pointer-events-none absolute inset-x-0 -top-px h-px bg-gradient-to-r from-transparent via-emerald-300/60 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
                <span className="pointer-events-none absolute -right-14 -top-14 h-44 w-44 rounded-full bg-gradient-to-br from-emerald-400/15 via-transparent to-transparent opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-100" />

                <div className="relative mb-4 flex items-start justify-between gap-3">
                  <div className="text-4xl drop-shadow-[0_4px_12px_rgba(52,211,153,0.35)] transition-transform duration-300 group-hover:scale-110">
                    {company.logo}
                  </div>
                  <span className="rounded-full bg-emerald-400/15 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-emerald-100 ring-1 ring-inset ring-emerald-300/25">
                    {company.category}
                  </span>
                </div>

                {company.isRecentAddition && (
                  <div className="relative mb-3 inline-flex rounded-full bg-gradient-to-r from-cyan-400/20 to-emerald-400/20 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-100 ring-1 ring-inset ring-cyan-300/30">
                    New · recently added
                  </div>
                )}

                <h3 className="relative mb-2 text-lg font-semibold leading-snug text-white transition-colors group-hover:text-emerald-200">
                  {company.name}
                </h3>
                <p className="relative mb-4 line-clamp-2 text-xs leading-relaxed text-slate-300">
                  {company.description}
                </p>

                <div className="relative grid grid-cols-2 gap-2">
                  <div className="rounded-xl border border-white/10 bg-white/[0.06] p-2.5">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                      Founded
                    </p>
                    <p className="mt-0.5 text-xs font-semibold text-white">
                      {company.founded}
                    </p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.06] p-2.5">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                      Scale
                    </p>
                    <p className="mt-0.5 line-clamp-1 text-xs font-semibold text-emerald-200">
                      {company.scaleSignal}
                    </p>
                  </div>
                </div>

                <div className="relative mt-3 border-t border-white/10 pt-3">
                  <div className="flex flex-wrap gap-1.5">
                    {company.keyProducts.slice(0, 3).map((product) => (
                      <span
                        key={product}
                        className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-medium text-slate-200"
                      >
                        {product}
                      </span>
                    ))}
                  </div>
                  <p className="mt-3 text-xs font-semibold text-cyan-200 transition-transform group-hover:translate-x-1">
                    Open sources →
                  </p>
                </div>
              </motion.button>
            ))}
          </div>
        )}

        {hasMoreCompanies && (
          <div className="mb-12 flex justify-center">
            <LevelOneLoadMoreButton
              label="Load 8 more"
              glowClassName="from-emerald-400/0 via-emerald-400/20 to-emerald-400/0"
              onClick={() => setVisibleCount((current) => current + 8)}
            />
          </div>
        )}

        <AnimatePresence>
          {selectedCompany && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/72 p-4 backdrop-blur-md"
              onClick={() => setSelectedCompany(null)}
            >
              <motion.div
                initial={{ scale: 0.96, opacity: 0, y: 12 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.96, opacity: 0, y: 12 }}
                className="relative max-h-[90vh] w-full max-w-6xl overflow-y-auto rounded-[2.25rem] border border-white/15 bg-gradient-to-br from-white/95 via-slate-50 to-emerald-50/70 shadow-[0_36px_120px_rgba(8,12,24,0.4)]"
                onClick={(event) => event.stopPropagation()}
              >
                <span className="pointer-events-none absolute inset-x-10 top-0 h-px bg-gradient-to-r from-transparent via-emerald-300/80 to-transparent" />
                <span className="pointer-events-none absolute -right-20 top-10 h-56 w-56 rounded-full bg-emerald-400/15 blur-3xl" />
                <span className="pointer-events-none absolute -left-16 bottom-8 h-44 w-44 rounded-full bg-cyan-400/12 blur-3xl" />

                <div className="relative p-5 sm:p-4">
                  <div className="mb-6 overflow-hidden rounded-[28px] border border-white/70 bg-gradient-to-br from-emerald-400/14 via-white/92 to-cyan-400/12 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.12)] sm:p-4">
                    <div className="mb-6 flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-5">
                        <div className="flex h-20 w-20 items-center justify-center rounded-[24px] border border-white/70 bg-white/80 text-5xl shadow-[0_20px_50px_rgba(16,185,129,0.18)]">
                          {selectedCompany.logo}
                        </div>
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-emerald-700/75">
                          Source-backed company brief
                        </p>
                        <h2 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-4xl">
                          {selectedCompany.name}
                        </h2>
                        <div className="mt-3 flex flex-wrap gap-2.5">
                          <span className="rounded-full bg-slate-950 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-white">
                            {selectedCompany.category}
                          </span>
                          <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-emerald-800">
                            Founded {selectedCompany.founded}
                          </span>
                          {selectedCompany.isRecentAddition && (
                            <span className="rounded-full border border-cyan-200 bg-cyan-50 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-800">
                              Recently added
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                      <motion.button
                        whileHover={{ rotate: 90, scale: 1.06 }}
                        whileTap={{ scale: 0.94 }}
                        onClick={() => setSelectedCompany(null)}
                        aria-label="Close company details"
                        className="self-end rounded-full border border-slate-200/80 bg-white/80 p-3 text-slate-500 shadow-sm transition-colors hover:border-slate-300 hover:text-slate-900 lg:self-start"
                      >
                        <svg
                          className="h-4 w-4"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M6 6l12 12M18 6L6 18"
                          />
                        </svg>
                      </motion.button>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-3">
                      <div className="rounded-[22px] border border-white/70 bg-white/78 p-4 shadow-[0_18px_40px_rgba(15,23,42,0.08)]">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                          Headquarters
                        </p>
                        <p className="mt-2 text-sm font-medium text-slate-800">
                          {selectedCompany.headquarters}
                        </p>
                      </div>
                      <div className="rounded-[22px] border border-white/70 bg-white/78 p-4 shadow-[0_18px_40px_rgba(15,23,42,0.08)]">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                          Scale signal
                        </p>
                        <p className="mt-2 text-sm font-medium text-slate-800">
                          {selectedCompany.scaleSignal}
                        </p>
                      </div>
                      <div className="rounded-[22px] border border-white/70 bg-white/78 p-4 shadow-[0_18px_40px_rgba(15,23,42,0.08)]">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                          Operating signal
                        </p>
                        <p className="mt-2 text-sm font-medium text-slate-800">
                          {selectedCompany.operatingSignal}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mb-8 grid gap-6 lg:grid-cols-[1.05fr,1.4fr]">
                    <div className="rounded-[1.75rem] border border-white/70 bg-white/80 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                      <h3 className="text-lg font-semibold text-slate-950">
                        Company facts
                      </h3>
                      <div className="mt-5 grid gap-4 text-sm text-slate-700 sm:grid-cols-2">
                        <div className="rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4 sm:col-span-2">
                          <p className="font-semibold uppercase tracking-[0.14em] text-slate-500">
                            Founders
                          </p>
                          <p className="mt-1">{selectedCompany.founders.join(", ")}</p>
                        </div>
                        {selectedCompany.stockSymbol && (
                          <div className="rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4">
                            <p className="font-semibold uppercase tracking-[0.14em] text-slate-500">
                              Ticker
                            </p>
                            <p className="mt-1">{selectedCompany.stockSymbol}</p>
                          </div>
                        )}
                        <div className="rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4">
                          <p className="font-semibold uppercase tracking-[0.14em] text-slate-500">
                            Website
                          </p>
                          <a
                            href={selectedCompany.website}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mt-1 inline-block font-medium text-cyan-700 transition-colors hover:text-cyan-900"
                          >
                            {selectedCompany.website.replace(/^https?:\/\//, "")}
                          </a>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-[1.75rem] border border-white/70 bg-white/80 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                      <h3 className="text-lg font-semibold text-slate-950">What Matters Now</h3>
                      <p className="mt-4 text-base leading-relaxed text-slate-700">
                        {selectedCompany.description}
                      </p>
                      <div className="mt-5 rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                          Current focus
                        </p>
                        <p className="mt-2 leading-relaxed text-slate-700">
                          {selectedCompany.currentFocus}
                        </p>
                      </div>
                      <div className="mt-5">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                          Company arc
                        </p>
                        <p className="mt-2 leading-relaxed text-slate-700">
                          {selectedCompany.journey}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mb-8 grid gap-6 lg:grid-cols-2">
                    <div className="rounded-[1.75rem] border border-white/70 bg-white/80 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                      <h3 className="text-lg font-semibold text-slate-950">
                        Landmark Contributions
                      </h3>
                      <div className="mt-4 space-y-3">
                        {selectedCompany.landmarkContributions.map((item) => (
                          <div
                            key={item}
                            className="rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4 text-sm leading-relaxed text-slate-700"
                          >
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-[1.75rem] border border-white/70 bg-white/80 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                      <h3 className="text-lg font-semibold text-slate-950">Key Products</h3>
                      <div className="mt-4 flex flex-wrap gap-3">
                        {selectedCompany.keyProducts.map((item) => (
                          <span
                            key={item}
                            className="rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-900 shadow-[0_6px_18px_rgba(16,185,129,0.08)]"
                          >
                            {item}
                          </span>
                        ))}
                      </div>

                      <h4 className="mt-6 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                        Why teams still watch this company
                      </h4>
                      <div className="mt-3 space-y-3">
                        {selectedCompany.achievements.map((item) => (
                          <div
                            key={item}
                            className="rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4 text-sm leading-relaxed text-slate-700"
                          >
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-[1.75rem] border border-white/70 bg-white/80 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                      <div>
                        <h3 className="text-lg font-semibold text-slate-950">Primary Sources</h3>
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
                          className="rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4 transition hover:border-cyan-300 hover:bg-cyan-50/80"
                        >
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                            {source.kind}
                          </p>
                          <p className="mt-2 text-sm font-semibold text-slate-900">
                            {source.label}
                          </p>
                          <p className="mt-2 text-sm font-medium text-cyan-700">
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
