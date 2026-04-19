import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import DeepBlueChess from "@/components/games/DeepBlueChess";
import AlphaGoDemo from "@/components/games/AlphaGoDemo";
import LibratusPoker from "@/components/games/LibratusPoker";
import LevelOneLoadMoreButton from "@/components/LevelOneLoadMoreButton";
import SubpageLayout from "@/components/SubpageLayout";
import { getPageRefreshContent } from "@/data/siteRefreshContent";
import { victories, type VictoryRecord } from "@/data/victoryArchive";

type MatchFilter = "All" | VictoryRecord["recordType"];

const matchFilters: MatchFilter[] = ["All", "Champion match", "Benchmark leap"];

export default function AIChampions() {
  const navigate = useNavigate();
  const [activeGame, setActiveGame] = useState<string | null>(null);
  const [selectedVictory, setSelectedVictory] = useState<VictoryRecord | null>(
    null,
  );
  const [filterType, setFilterType] = useState<MatchFilter>("All");
  const [visibleCount, setVisibleCount] = useState(6);

  const pageRefresh = getPageRefreshContent("/ai-champions");

  const championMatchCount = useMemo(
    () =>
      victories.filter((victory) => victory.recordType === "Champion match")
        .length,
    [],
  );
  const playableVictoryCount = useMemo(
    () => victories.filter((victory) => victory.playableDemo).length,
    [],
  );

  const filteredVictories = useMemo(
    () =>
      filterType === "All"
        ? victories
        : victories.filter((victory) => victory.recordType === filterType),
    [filterType],
  );

  useEffect(() => {
    setVisibleCount(6);
  }, [filterType]);

  const visibleVictories = filteredVictories.slice(0, visibleCount);
  const hasMoreVictories = visibleVictories.length < filteredVictories.length;
  const activeVictory =
    victories.find((victory) => victory.id === activeGame) ?? null;
  const featuredVictories = victories.filter((victory) =>
    ["deep-blue-chess", "alphago-go", "openai-five-dota"].includes(victory.id),
  );

  return (
    <SubpageLayout
      route="/ai-champions"
      eyebrow={pageRefresh.eyebrow}
      title={pageRefresh.title}
      description={pageRefresh.description}
      accent="amber"
      chips={pageRefresh.chips}
      metrics={[
        {
          value: victories.length.toString(),
          label: "Landmark matchups",
        },
        {
          value: championMatchCount.toString(),
          label: "Direct human wins",
        },
        {
          value: playableVictoryCount.toString(),
          label: "Playable demos",
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
              Featured matchups
            </h2>
            <span className="text-[11px] uppercase tracking-[0.2em] text-gray-400">
              Chess · Go · Dota 2
            </span>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            {featuredVictories.map((victory) => (
              <button
                key={victory.id}
                type="button"
                onClick={() => setSelectedVictory(victory)}
                aria-label={`View ${victory.aiName} vs human champion in ${victory.game}`}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 text-left transition hover:bg-white/10"
              >
                <div className="mb-2 flex items-center justify-between gap-2 text-[11px] uppercase tracking-[0.18em] text-gray-400">
                  <span className="text-xl leading-none">{victory.icon}</span>
                  <span>{victory.year}</span>
                </div>
                <h3 className="text-sm font-semibold leading-snug text-white">
                  {victory.aiName}
                </h3>
                <p className="mt-1 text-xs text-amber-100">{victory.game}</p>
              </button>
            ))}
          </div>
        </motion.section>

        <section className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 backdrop-blur-md">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400">
              Record
            </span>
            {matchFilters.map((filter) => (
              <button
                key={filter}
                onClick={() => setFilterType(filter)}
                aria-label={`Filter by ${filter}`}
                className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                  filterType === filter
                    ? "bg-gradient-to-r from-amber-500 to-orange-500 text-white"
                    : "border border-white/15 bg-white/5 text-gray-200 hover:bg-white/10"
                }`}
              >
                {filter}
              </button>
            ))}
          </div>
          <span className="text-xs text-gray-400">
            {visibleVictories.length} / {filteredVictories.length}
          </span>
        </section>

        <div className="mb-12 grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
          {visibleVictories.map((victory, index) => (
            <motion.button
              key={victory.id}
              type="button"
              onClick={() => setSelectedVictory(victory)}
              aria-label={`${victory.aiName} vs ${victory.opponent} in ${victory.game}, ${victory.year}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                type: "spring",
                stiffness: 260,
                damping: 24,
                delay: Math.min(index * 0.05, 0.22),
              }}
              whileHover={{ y: -6, rotateX: 2, rotateY: -2 }}
              whileTap={{ scale: 0.98 }}
              style={{ transformStyle: "preserve-3d", perspective: 900 }}
              className={`group overflow-hidden rounded-[22px] border bg-white text-left shadow-[0_22px_52px_rgba(8,12,24,0.2)] transition-colors duration-300 hover:shadow-[0_30px_70px_rgba(8,12,24,0.25)] ${
                selectedVictory?.id === victory.id
                  ? `${victory.accent} ring-2 ring-amber-300/60`
                  : "border-gray-200"
              }`}
            >
              <div
                className={`bg-gradient-to-r ${victory.gradient} p-6 text-white`}
              >
                <div className="mb-4 flex items-start justify-between gap-4">
                  <div>
                    <div className="text-4xl">{victory.icon}</div>
                    <h3 className="mt-3 text-2xl font-black">{victory.game}</h3>
                    <p className="mt-1 text-sm font-semibold text-white/90">
                      {victory.aiName} vs {victory.opponent}
                    </p>
                  </div>
                  <span className="rounded-full border border-white/20 bg-white/10 px-3 py-2 text-xs font-semibold">
                    {victory.year}
                  </span>
                </div>

                <div className="flex flex-wrap gap-2">
                  <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs font-semibold">
                    {victory.recordType}
                  </span>
                  <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs font-semibold">
                    {victory.scoreLabel}
                  </span>
                  {victory.playableDemo && (
                    <span className="rounded-full border border-emerald-200/30 bg-emerald-400/20 px-3 py-1 text-xs font-semibold">
                      Playable demo
                    </span>
                  )}
                </div>
              </div>

              <div className="p-6">
                <p className="mb-4 text-sm leading-relaxed text-gray-700">
                  {victory.significance}
                </p>

                <div className="mb-5 rounded-2xl border border-gray-100 bg-gray-50 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-gray-500">
                    Why it mattered
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-gray-700">
                    {victory.whyItMattered}
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  {victory.methods.slice(0, 3).map((method) => (
                    <span
                      key={method}
                      className="rounded-full border border-gray-200 bg-white px-3 py-1 text-xs font-semibold text-gray-700"
                    >
                      {method}
                    </span>
                  ))}
                </div>
              </div>
            </motion.button>
          ))}
        </div>

        {hasMoreVictories && (
          <div className="mb-12 flex justify-center">
            <LevelOneLoadMoreButton
              label="Load 3 more"
              glowClassName="from-amber-400/0 via-amber-400/20 to-amber-400/0"
              onClick={() => setVisibleCount((current) => current + 3)}
            />
          </div>
        )}

        <AnimatePresence>
          {selectedVictory && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-md"
              onClick={() => setSelectedVictory(null)}
            >
              <motion.div
                initial={{ scale: 0.94, opacity: 0, y: 24 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.94, opacity: 0, y: 24 }}
                transition={{ duration: 0.25 }}
                className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-[2rem] border border-white/15 bg-slate-950 shadow-2xl"
                onClick={(event) => event.stopPropagation()}
              >
                <div
                  className={`bg-gradient-to-br ${selectedVictory.gradient} p-8 text-white`}
                >
                  <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
                    <div className="max-w-3xl">
                      <div className="mb-4 flex items-center gap-4">
                        <span className="text-5xl">{selectedVictory.icon}</span>
                        <div>
                          <div className="mb-2 inline-flex rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em]">
                            {selectedVictory.recordType}
                          </div>
                          <h2 className="text-3xl font-black md:text-4xl">
                            {selectedVictory.aiName}
                          </h2>
                        </div>
                      </div>
                      <p className="text-lg font-semibold text-white/90">
                        {selectedVictory.game} · {selectedVictory.year} ·{" "}
                        {selectedVictory.location}
                      </p>
                      <p className="mt-4 text-base leading-relaxed text-white/95">
                        {selectedVictory.summary}
                      </p>
                    </div>

                    <motion.button
                      whileHover={{ scale: 1.08, rotate: 90 }}
                      whileTap={{ scale: 0.92 }}
                      onClick={() => setSelectedVictory(null)}
                      className="flex h-12 w-12 items-center justify-center rounded-full border border-white/20 bg-white/10 text-2xl transition-colors hover:bg-white/15"
                    >
                      ✕
                    </motion.button>
                  </div>
                </div>

                <div className="p-8">
                  <div className="mb-8 grid gap-4 md:grid-cols-3">
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-amber-100">
                        Score or result
                      </p>
                      <p className="mt-3 text-2xl font-black text-white">
                        {selectedVictory.scoreLabel}
                      </p>
                    </div>
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-100">
                        Opponent
                      </p>
                      <p className="mt-3 text-2xl font-black text-white">
                        {selectedVictory.opponent}
                      </p>
                    </div>
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-100">
                        Playable
                      </p>
                      <p className="mt-3 text-2xl font-black text-white">
                        {selectedVictory.playableDemo ? "Yes" : "No"}
                      </p>
                    </div>
                  </div>

                  <div className="mb-8 grid gap-6 lg:grid-cols-[1fr,1fr]">
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-gray-300">
                        Why it mattered
                      </p>
                      <p className="mt-3 text-base leading-relaxed text-gray-100">
                        {selectedVictory.whyItMattered}
                      </p>
                    </div>
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-gray-300">
                        Today context
                      </p>
                      <p className="mt-3 text-base leading-relaxed text-gray-100">
                        {selectedVictory.todayContext}
                      </p>
                    </div>
                  </div>

                  <div className="mb-8 rounded-3xl border border-white/10 bg-white/5 p-6">
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-100">
                      Match format
                    </p>
                    <p className="mt-3 text-base leading-relaxed text-gray-100">
                      {selectedVictory.format}
                    </p>
                  </div>

                  <div className="mb-8 rounded-3xl border border-white/10 bg-white/5 p-6">
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-violet-100">
                      Core methods
                    </p>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {selectedVictory.methods.map((method) => (
                        <span
                          key={method}
                          className="rounded-full border border-white/10 bg-black/20 px-3 py-2 text-sm font-semibold text-white"
                        >
                          {method}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="mb-8 rounded-3xl border border-white/10 bg-white/5 p-6">
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-100">
                      Primary sources
                    </p>
                    <div className="mt-4 grid gap-4 md:grid-cols-2">
                      {selectedVictory.sources.map((source) => (
                        <motion.a
                          key={source.url}
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="rounded-2xl border border-white/10 bg-black/20 p-4 transition-colors hover:bg-black/30"
                        >
                          <p className="text-sm font-black text-white">
                            {source.label}
                          </p>
                          <p className="mt-2 text-xs font-semibold uppercase tracking-[0.16em] text-gray-300">
                            {source.kind}
                          </p>
                        </motion.a>
                      ))}
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-4">
                    {selectedVictory.playableDemo && (
                      <motion.button
                        whileHover={{ scale: 1.04 }}
                        whileTap={{ scale: 0.96 }}
                        onClick={() => {
                          setSelectedVictory(null);
                          setActiveGame(selectedVictory.id);
                        }}
                        className="rounded-full bg-gradient-to-r from-amber-500 to-orange-600 px-6 py-3 text-sm font-bold text-white shadow-lg transition-all"
                      >
                        Play demo
                      </motion.button>
                    )}
                    <motion.button
                      whileHover={{ scale: 1.04 }}
                      whileTap={{ scale: 0.96 }}
                      onClick={() => navigate("/ai-discoveries")}
                      className="rounded-full border border-white/15 bg-white/5 px-6 py-3 text-sm font-bold text-white transition-all hover:bg-white/10"
                    >
                      Explore AI discoveries
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.04 }}
                      whileTap={{ scale: 0.96 }}
                      onClick={() => navigate("/games")}
                      className="rounded-full border border-white/15 bg-white/5 px-6 py-3 text-sm font-bold text-white transition-all hover:bg-white/10"
                    >
                      Open games
                    </motion.button>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {activeVictory && (
            <section className="mb-12">
              <div className="rounded-[2rem] border border-white/15 bg-white shadow-2xl">
                <motion.div
                  key={activeVictory.id}
                  initial={{ opacity: 0, y: 32, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -24, scale: 0.98 }}
                  transition={{ duration: 0.45 }}
                >
                  <div
                    className={`flex flex-col gap-4 rounded-t-[2rem] bg-gradient-to-r ${activeVictory.gradient} p-6 text-white md:flex-row md:items-center md:justify-between`}
                  >
                    <div className="flex items-center gap-4">
                      <span className="text-4xl">{activeVictory.icon}</span>
                      <div>
                        <h3 className="text-2xl font-black">
                          {activeVictory.game}
                        </h3>
                        <p className="text-sm font-semibold text-white/90">
                          Historic AI demo based on {activeVictory.aiName}
                        </p>
                      </div>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.08, rotate: 90 }}
                      whileTap={{ scale: 0.92 }}
                      onClick={() => setActiveGame(null)}
                      className="flex h-10 w-10 items-center justify-center rounded-full border border-white/20 bg-white/10 text-xl transition-colors hover:bg-white/15"
                    >
                      ✕
                    </motion.button>
                  </div>

                  <div className="min-h-[600px] bg-gray-50">
                    {activeGame === "deep-blue-chess" && <DeepBlueChess />}
                    {activeGame === "alphago-go" && <AlphaGoDemo />}
                    {activeGame === "libratus-poker" && <LibratusPoker />}
                  </div>
                </motion.div>
              </div>
            </section>
          )}
        </AnimatePresence>

      </div>
    </SubpageLayout>
  );
}
