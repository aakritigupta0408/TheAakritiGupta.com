import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import SubpageLayout from "@/components/SubpageLayout";
import { discoveries, type Discovery } from "@/data/discoveryArchive";
import { getPageRefreshContent } from "@/data/siteRefreshContent";
import { latestAIResearchBreakthroughs } from "../data/aiSignals";

type SortMode = "chronological" | "alphabetical";

export default function AIDiscoveries() {
  const [selectedDiscovery, setSelectedDiscovery] = useState<Discovery | null>(
    null,
  );
  const [sortBy, setSortBy] = useState<SortMode>("chronological");
  const [filterDecade, setFilterDecade] = useState("All");
  const [visibleCount, setVisibleCount] = useState(8);

  const pageRefresh = getPageRefreshContent("/ai-discoveries");

  const decades = useMemo(() => {
    const uniqueDecades = Array.from(
      new Set(
        discoveries.map((discovery) => {
          const year = parseInt(discovery.year, 10);
          return `${Math.floor(year / 10) * 10}`;
        }),
      ),
    ).sort();

    return ["All", ...uniqueDecades];
  }, []);

  const filteredDiscoveries = useMemo(() => {
    const filtered =
      filterDecade === "All"
        ? [...discoveries]
        : discoveries.filter((discovery) => {
            const year = parseInt(discovery.year, 10);
            return `${Math.floor(year / 10) * 10}` === filterDecade;
          });

    filtered.sort((left, right) => {
      if (sortBy === "alphabetical") {
        return left.title.localeCompare(right.title);
      }

      return parseInt(left.year, 10) - parseInt(right.year, 10);
    });

    return filtered;
  }, [filterDecade, sortBy]);

  useEffect(() => {
    setVisibleCount(8);
  }, [filterDecade, sortBy]);

  const visibleDiscoveries = filteredDiscoveries.slice(0, visibleCount);
  const hasMoreDiscoveries = visibleDiscoveries.length < filteredDiscoveries.length;

  return (
    <SubpageLayout
      route="/ai-discoveries"
      eyebrow={pageRefresh.eyebrow}
      title={pageRefresh.title}
      description={pageRefresh.description}
      accent="blue"
      chips={pageRefresh.chips}
      metrics={[
        {
          value: discoveries.length.toString(),
          label: "Milestones in the archive",
        },
        {
          value: (decades.length - 1).toString(),
          label: "Decades covered",
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
              Recent frontier signals
            </h2>
            <span className="text-xs uppercase tracking-[0.2em] text-gray-400">
              Linked to primary sources
            </span>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {latestAIResearchBreakthroughs.slice(0, 3).map((signal) => (
              <a
                key={signal.id}
                href={signal.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:bg-white/10"
              >
                <div className="mb-2 flex items-center justify-between gap-2 text-[11px] uppercase tracking-[0.18em] text-gray-400">
                  <span className="text-emerald-200">{signal.category}</span>
                  <span>{signal.date}</span>
                </div>
                <h3 className="mb-1 text-sm font-semibold leading-snug text-white">
                  {signal.title}
                </h3>
                <p className="text-xs text-cyan-100">{signal.org}</p>
              </a>
            ))}
          </div>
        </motion.section>

        <section className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-[20px] border border-white/10 bg-white/[0.04] px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_10px_30px_rgba(0,0,0,0.25)] backdrop-blur-2xl">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-gray-400">
              Decade
            </span>
            {decades.map((decade) => (
              <motion.button
                key={decade}
                onClick={() => setFilterDecade(decade)}
                whileHover={{ y: -1 }}
                whileTap={{ scale: 0.94 }}
                transition={{ type: "spring", stiffness: 420, damping: 24 }}
                className={`relative rounded-full px-3.5 py-1.5 text-xs font-semibold transition-colors ${
                  filterDecade === decade
                    ? "text-white shadow-[0_6px_18px_rgba(52,211,153,0.38)]"
                    : "text-gray-200 hover:text-white"
                }`}
              >
                {filterDecade === decade && (
                  <motion.span
                    layoutId="discoveryPill"
                    transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    className="absolute inset-0 rounded-full bg-gradient-to-br from-emerald-400 via-teal-500 to-cyan-500"
                  />
                )}
                <span className="relative z-10">
                  {decade === "All" ? "All" : `${decade}s`}
                </span>
              </motion.button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value as SortMode)}
              className="rounded-full border border-white/15 bg-white/5 px-3.5 py-1.5 text-xs font-semibold text-gray-100 transition hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
            >
              <option value="chronological" className="bg-slate-900">
                Chronological
              </option>
              <option value="alphabetical" className="bg-slate-900">
                Alphabetical
              </option>
            </select>
            <span className="font-mono text-[11px] text-gray-400">
              {visibleDiscoveries.length} / {filteredDiscoveries.length}
            </span>
          </div>
        </section>

        {filteredDiscoveries.length === 0 ? (
          <div className="mb-12 rounded-[2rem] border border-white/20 bg-white/10 p-12 text-center backdrop-blur-xl">
            <h3 className="text-3xl font-black text-white">No discoveries found</h3>
            <p className="mt-3 text-lg text-gray-200">
              Try a different decade or reset back to the full archive.
            </p>
          </div>
        ) : (
          <div className="mb-12 grid grid-cols-1 gap-5 lg:grid-cols-2 xl:grid-cols-4">
            {visibleDiscoveries.map((discovery, index) => (
              <motion.button
                key={discovery.id}
                type="button"
                onClick={() => setSelectedDiscovery(discovery)}
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  type: "spring",
                  stiffness: 260,
                  damping: 24,
                  delay: Math.min(index * 0.04, 0.2),
                }}
                whileHover={{ y: -6, rotateX: 2, rotateY: -2 }}
                whileTap={{ scale: 0.98 }}
                style={{ transformStyle: "preserve-3d", perspective: 900 }}
                className="group relative overflow-hidden rounded-[22px] border border-white/10 bg-gradient-to-br from-white/[0.09] to-white/[0.04] p-5 text-left shadow-[0_20px_48px_rgba(8,12,24,0.45),inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-2xl transition-colors duration-300 hover:border-white/25"
              >
                <span className="pointer-events-none absolute inset-x-0 -top-px h-px bg-gradient-to-r from-transparent via-cyan-300/60 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
                <span className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full bg-gradient-to-br from-cyan-400/15 via-transparent to-transparent opacity-0 blur-2xl transition-opacity duration-500 group-hover:opacity-100" />

                <div className="relative mb-4 flex items-center justify-between gap-3">
                  <span className="rounded-full bg-gradient-to-r from-cyan-400/25 to-blue-500/25 px-3 py-1 text-xs font-bold tracking-wide text-cyan-50 ring-1 ring-inset ring-cyan-300/30">
                    {discovery.year}
                  </span>
                  <span className="font-mono text-[11px] font-semibold text-gray-400">
                    #{discovery.id.toString().padStart(2, "0")}
                  </span>
                </div>

                <h3 className="relative mb-3 text-lg font-bold leading-snug text-white transition-colors group-hover:text-cyan-100">
                  {discovery.title}
                </h3>
                <p className="relative mb-5 line-clamp-3 text-sm leading-relaxed text-gray-300">
                  {discovery.summary}
                </p>

                <div className="relative border-t border-white/10 pt-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.14em] text-cyan-200/90">
                    {discovery.discoverer}
                  </p>
                  <p className="mt-2 line-clamp-3 text-xs leading-relaxed text-gray-400">
                    {discovery.todayContext}
                  </p>
                </div>
              </motion.button>
            ))}
          </div>
        )}

        {hasMoreDiscoveries && (
          <div className="mb-12 flex justify-center">
            <motion.button
              onClick={() => setVisibleCount((current) => current + 8)}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.96 }}
              transition={{ type: "spring", stiffness: 420, damping: 24 }}
              className="group relative overflow-hidden rounded-full border border-white/15 bg-white/[0.06] px-6 py-2.5 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(8,12,24,0.35)] backdrop-blur-xl transition-colors hover:border-white/30 hover:bg-white/10"
            >
              <span className="absolute inset-0 bg-gradient-to-r from-cyan-400/0 via-cyan-400/20 to-cyan-400/0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
              <span className="relative">Load 8 more</span>
            </motion.button>
          </div>
        )}

        <AnimatePresence>
          {selectedDiscovery && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
              onClick={() => setSelectedDiscovery(null)}
            >
              <motion.div
                initial={{ scale: 0.96, opacity: 0, y: 12 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.96, opacity: 0, y: 12 }}
                className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-[2rem] border border-white/20 bg-[#f8fafc] shadow-2xl"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="p-6 sm:p-8">
                  <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <h2 className="text-3xl font-black text-slate-950 sm:text-4xl">
                        {selectedDiscovery.title}
                      </h2>
                      <p className="mt-3 text-lg font-semibold text-cyan-700">
                        {selectedDiscovery.year} · {selectedDiscovery.discoverer}
                      </p>
                    </div>
                    <button
                      onClick={() => setSelectedDiscovery(null)}
                      className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
                    >
                      Close
                    </button>
                  </div>

                  <div className="grid gap-6 lg:grid-cols-[1.15fr,0.85fr]">
                    <div className="space-y-6">
                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">
                          What Changed
                        </h3>
                        <p className="mt-4 leading-relaxed text-slate-700">
                          {selectedDiscovery.summary}
                        </p>
                      </div>

                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">
                          Why It Still Matters
                        </h3>
                        <p className="mt-4 leading-relaxed text-slate-700">
                          {selectedDiscovery.whyItMatters}
                        </p>
                        <div className="mt-5 rounded-2xl bg-slate-50 p-4">
                          <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                            Today context
                          </p>
                          <p className="mt-2 leading-relaxed text-slate-700">
                            {selectedDiscovery.todayContext}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-6">
                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">
                          About the Discoverer
                        </h3>
                        <p className="mt-4 leading-relaxed text-slate-700">
                          {selectedDiscovery.discovererBio}
                        </p>
                      </div>

                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">
                          Primary Paper
                        </h3>
                        <p className="mt-4 text-sm font-semibold text-slate-900">
                          {selectedDiscovery.paperTitle}
                        </p>
                        <a
                          href={selectedDiscovery.paperLink}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="mt-4 inline-flex rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700"
                        >
                          Open paper
                        </a>
                      </div>

                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">
                          Sources
                        </h3>
                        <div className="mt-4 space-y-3">
                          {selectedDiscovery.sources.map((source) => (
                            <a
                              key={source.url}
                              href={source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block rounded-2xl border border-slate-200 bg-slate-50 p-4 transition hover:border-cyan-300 hover:bg-cyan-50"
                            >
                              <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                                {source.kind}
                              </p>
                              <p className="mt-2 text-sm font-semibold text-slate-900">
                                {source.label}
                              </p>
                            </a>
                          ))}
                        </div>
                      </div>
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
