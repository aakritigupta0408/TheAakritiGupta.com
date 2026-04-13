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
      refreshSummary={pageRefresh.refreshSummary}
      updatedAtLabel={pageRefresh.updatedAtLabel}
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

        <section className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 backdrop-blur-md">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400">
              Decade
            </span>
            {decades.map((decade) => (
              <button
                key={decade}
                onClick={() => setFilterDecade(decade)}
                className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                  filterDecade === decade
                    ? "bg-emerald-500 text-white"
                    : "border border-white/15 bg-white/5 text-gray-200 hover:bg-white/10"
                }`}
              >
                {decade === "All" ? "All" : `${decade}s`}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value as SortMode)}
              className="rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-semibold text-gray-100 focus:outline-none"
            >
              <option value="chronological" className="bg-slate-900">
                Chronological
              </option>
              <option value="alphabetical" className="bg-slate-900">
                Alphabetical
              </option>
            </select>
            <span className="text-xs text-gray-400">
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
          <div className="mb-12 grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-4">
            {visibleDiscoveries.map((discovery, index) => (
              <motion.button
                key={discovery.id}
                type="button"
                onClick={() => setSelectedDiscovery(discovery)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: Math.min(index * 0.05, 0.25) }}
                className="group rounded-3xl border border-white/20 bg-white/10 p-6 text-left shadow-2xl backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:bg-white/15"
              >
                <div className="mb-4 flex items-center justify-between gap-3">
                  <span className="rounded-full border border-blue-300/40 bg-blue-400/10 px-3 py-1 text-sm font-bold text-blue-100">
                    {discovery.year}
                  </span>
                  <span className="text-sm font-semibold text-gray-300">
                    #{discovery.id}
                  </span>
                </div>

                <h3 className="mb-3 text-xl font-black text-white group-hover:text-cyan-200">
                  {discovery.title}
                </h3>
                <p className="mb-5 line-clamp-4 text-sm leading-relaxed text-gray-100">
                  {discovery.summary}
                </p>

                <div className="border-t border-white/15 pt-5">
                  <p className="text-sm font-bold text-cyan-100">
                    {discovery.discoverer}
                  </p>
                  <p className="mt-2 line-clamp-3 text-sm text-gray-300">
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
              className="rounded-full border border-white/20 bg-white/10 px-6 py-3 text-sm font-bold text-white shadow-xl backdrop-blur-md transition-all duration-300 hover:bg-white/15"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              Show 8 more discoveries
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
