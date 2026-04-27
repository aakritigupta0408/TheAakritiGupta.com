import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import LevelOneLoadMoreButton from "@/components/LevelOneLoadMoreButton";
import SubpageLayout from "@/components/SubpageLayout";
import {
  professions,
  type ImpactLevel,
  type ProfessionProfile,
} from "@/data/toolArchive";
import { getPageRefreshContent } from "@/data/siteRefreshContent";
import { latestAIProductLaunches } from "../data/aiSignals";

type SortMode = "adoption" | "impact" | "alphabetical";

const impactRank: Record<ImpactLevel, number> = {
  Critical: 3,
  High: 2,
  Medium: 1,
};

const impactFilters: Array<"All" | ImpactLevel> = [
  "All",
  "Critical",
  "High",
  "Medium",
];

export default function AITools() {
  const [selectedProfession, setSelectedProfession] =
    useState<ProfessionProfile | null>(null);
  const [filterImpact, setFilterImpact] = useState<"All" | ImpactLevel>("All");
  const [sortBy, setSortBy] = useState<SortMode>("impact");
  const [visibleCount, setVisibleCount] = useState(8);

  const pageRefresh = getPageRefreshContent("/ai-tools");

  const criticalRoleCount = useMemo(
    () =>
      professions.filter((profession) => profession.impactLevel === "Critical")
        .length,
    [],
  );

  const uniqueToolCount = useMemo(() => {
    const names = new Set(
      professions.flatMap((profession) => [
        profession.primaryTool.name,
        ...profession.alternativeTools.map((tool) => tool.name),
      ]),
    );

    return names.size;
  }, []);

  const filteredProfessions = useMemo(() => {
    const filtered =
      filterImpact === "All"
        ? [...professions]
        : professions.filter(
            (profession) => profession.impactLevel === filterImpact,
          );

    filtered.sort((left, right) => {
      if (sortBy === "alphabetical") {
        return left.title.localeCompare(right.title);
      }

      if (sortBy === "adoption") {
        return right.aiAdoption - left.aiAdoption;
      }

      const impactDelta =
        impactRank[right.impactLevel] - impactRank[left.impactLevel];

      if (impactDelta !== 0) {
        return impactDelta;
      }

      return right.aiAdoption - left.aiAdoption;
    });

    return filtered;
  }, [filterImpact, sortBy]);

  useEffect(() => {
    setVisibleCount(8);
  }, [filterImpact, sortBy]);

  const visibleProfessions = filteredProfessions.slice(0, visibleCount);
  const hasMoreProfessions = visibleProfessions.length < filteredProfessions.length;

  return (
    <SubpageLayout
      route="/ai-tools"
      eyebrow={pageRefresh.eyebrow}
      title={pageRefresh.title}
      description={pageRefresh.description}
      accent="rose"
      chips={pageRefresh.chips}
      metrics={[
        {
          value: professions.length.toString(),
          label: "Role playbooks",
        },
        {
          value: uniqueToolCount.toString(),
          label: "Tools tracked",
        },
        {
          value: criticalRoleCount.toString(),
          label: "Critical-shift roles",
        },
      ]}
    >
      <div className="container mx-auto px-4 py-4">
        <motion.section
          className="mb-3 rounded-[1.75rem] border border-white/10 bg-white/[0.06] p-4 backdrop-blur-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-4 flex items-baseline justify-between gap-3">
            <h2 className="text-lg font-semibold text-white">
              Current product shifts
            </h2>
            <span className="text-[11px] uppercase tracking-[0.2em] text-gray-400">
              Pick the workflow, then the tool
            </span>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {latestAIProductLaunches.slice(0, 4).map((launch) => (
              <a
                key={launch.id}
                href={launch.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:bg-white/10"
              >
                <div className="mb-2 flex items-center justify-between gap-2 text-[11px] uppercase tracking-[0.18em] text-gray-400">
                  <span className="bg-gradient-to-r from-violet-300 to-cyan-300 bg-clip-text text-transparent">
                    {launch.category}
                  </span>
                  <span>{launch.date}</span>
                </div>
                <h3 className="text-sm font-semibold leading-snug text-white">
                  {launch.title}
                </h3>
                <p className="mt-1 text-xs text-violet-100">{launch.org}</p>
              </a>
            ))}
          </div>
        </motion.section>

        <section className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 backdrop-blur-md">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400">
              Impact
            </span>
            {impactFilters.map((impact) => (
              <button
                key={impact}
                onClick={() => setFilterImpact(impact)}
                className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                  filterImpact === impact
                    ? "bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white"
                    : "border border-white/15 bg-white/5 text-gray-200 hover:bg-white/10"
                }`}
              >
                {impact}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value as SortMode)}
              className="rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-semibold text-gray-100 focus:outline-none"
            >
              <option value="impact" className="bg-slate-900">
                Impact shift
              </option>
              <option value="adoption" className="bg-slate-900">
                AI adoption
              </option>
              <option value="alphabetical" className="bg-slate-900">
                Alphabetical
              </option>
            </select>
            <span className="text-xs text-gray-400">
              {visibleProfessions.length} / {filteredProfessions.length}
            </span>
          </div>
        </section>

        <div className="mb-4 grid grid-cols-1 gap-2 lg:grid-cols-2 xl:grid-cols-3">
          {visibleProfessions.map((profession, index) => (
            <motion.button
              key={profession.id}
              type="button"
              onClick={() => setSelectedProfession(profession)}
              initial={{ opacity: 0, y: 24 }}
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
              className="group relative overflow-hidden rounded-[22px] border border-white/10 bg-gradient-to-br from-white/[0.09] to-white/[0.03] p-5 text-left shadow-[0_22px_52px_rgba(8,12,24,0.45),inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-2xl transition-colors duration-300 hover:border-white/25"
            >
              <span className="pointer-events-none absolute inset-x-0 -top-px h-px bg-gradient-to-r from-transparent via-violet-300/60 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
              <span className="pointer-events-none absolute -right-14 -top-14 h-44 w-44 rounded-full bg-gradient-to-br from-violet-400/15 via-transparent to-transparent opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-100" />

              <div className="relative mb-4 flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.08] text-2xl shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] transition-transform duration-300 group-hover:scale-110">
                    {profession.icon}
                  </span>
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-violet-200/80">
                      {profession.impactLevel}
                    </p>
                    <h3 className="mt-0.5 text-lg font-semibold leading-snug text-white transition-colors group-hover:text-violet-200">
                      {profession.title}
                    </h3>
                  </div>
                </div>
                <span className="rounded-full bg-white/5 px-2.5 py-1 text-[10px] font-semibold text-slate-200 ring-1 ring-inset ring-white/10">
                  {profession.aiAdoption}%
                </span>
              </div>

              <p className="relative mb-4 line-clamp-2 text-xs leading-relaxed text-slate-300">
                {profession.description}
              </p>

              <div className="relative mb-4 grid gap-2 sm:grid-cols-2">
                <div className="rounded-xl border border-white/10 bg-white/[0.06] p-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-200">
                    Workflow now
                  </p>
                  <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-slate-300">
                    {profession.workflowNow}
                  </p>
                </div>
                <div className="rounded-xl border border-white/10 bg-white/[0.06] p-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-emerald-200">
                    Time saved
                  </p>
                  <p className="mt-1 text-xs font-semibold text-white">
                    {profession.timeSaved}
                  </p>
                  <p className="mt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                    Primary
                  </p>
                  <p className="text-xs text-cyan-100">
                    {profession.primaryTool.name}
                  </p>
                </div>
              </div>

              <div className="relative flex items-center justify-between gap-3 border-t border-white/10 pt-3">
                <p className="line-clamp-1 text-[11px] text-slate-300">
                  {profession.alternativeTools
                    .slice(0, 3)
                    .map((tool) => tool.name)
                    .join(" · ")}
                </p>
                <span className="text-xs font-semibold text-violet-200 transition-transform group-hover:translate-x-1">
                  Open →
                </span>
              </div>
            </motion.button>
          ))}
        </div>

        {hasMoreProfessions && (
          <LevelOneLoadMoreButton
            label="Load 4 more"
            onClick={() => setVisibleCount((current) => current + 4)}
          />
        )}

        <AnimatePresence>
          {selectedProfession && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-md"
              onClick={() => setSelectedProfession(null)}
            >
              <motion.div
                initial={{ scale: 0.94, opacity: 0, y: 24 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.94, opacity: 0, y: 24 }}
                transition={{ duration: 0.25 }}
                className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-[2rem] border border-white/15 bg-slate-800 shadow-2xl"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="border-b border-white/10 bg-gradient-to-br from-violet-600/30 via-slate-950 to-cyan-600/20 p-8">
                  <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
                    <div className="max-w-3xl">
                      <div className="mb-4 flex items-center gap-4">
                        <span className="flex h-16 w-16 items-center justify-center rounded-3xl border border-white/15 bg-white/10 text-4xl">
                          {selectedProfession.icon}
                        </span>
                        <div>
                          <div className="mb-2 inline-flex rounded-full border border-violet-300/30 bg-violet-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-violet-100">
                            {selectedProfession.impactLevel}
                          </div>
                          <h2 className="text-3xl font-black text-white md:text-4xl">
                            {selectedProfession.title}
                          </h2>
                        </div>
                      </div>
                      <p className="text-base leading-relaxed text-gray-100">
                        {selectedProfession.description}
                      </p>
                    </div>

                    <motion.button
                      whileHover={{ scale: 1.08, rotate: 90 }}
                      whileTap={{ scale: 0.92 }}
                      onClick={() => setSelectedProfession(null)}
                      className="flex h-12 w-12 items-center justify-center rounded-full border border-white/15 bg-white/10 text-2xl text-white transition-colors hover:bg-white/15"
                    >
                      ✕
                    </motion.button>
                  </div>
                </div>

                <div className="p-8">
                  <div className="mb-8 grid gap-4 md:grid-cols-3">
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-violet-100">
                        AI adoption
                      </p>
                      <p className="mt-3 text-3xl font-black text-white">
                        {selectedProfession.aiAdoption}%
                      </p>
                    </div>
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-100">
                        Typical time gain
                      </p>
                      <p className="mt-3 text-2xl font-black text-white">
                        {selectedProfession.timeSaved}
                      </p>
                    </div>
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-100">
                        Primary stack anchor
                      </p>
                      <p className="mt-3 text-2xl font-black text-white">
                        {selectedProfession.primaryTool.name}
                      </p>
                    </div>
                  </div>

                  <div className="mb-8 rounded-3xl border border-white/10 bg-white/5 p-4">
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-gray-300">
                      Workflow now
                    </p>
                    <p className="mt-3 text-base leading-relaxed text-gray-100">
                      {selectedProfession.workflowNow}
                    </p>
                  </div>

                  <div className="mb-8 grid gap-6 lg:grid-cols-[1fr,1fr]">
                    <motion.a
                      href={selectedProfession.primaryTool.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="rounded-3xl border border-violet-300/20 bg-violet-500/10 p-4 transition-colors hover:bg-violet-500/15"
                    >
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <span className="rounded-full border border-violet-300/30 bg-violet-400/10 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.18em] text-violet-100">
                          Primary tool
                        </span>
                        <span className="text-xs font-semibold text-gray-300">
                          {selectedProfession.primaryTool.category}
                        </span>
                      </div>
                      <h3 className="text-2xl font-black text-white">
                        {selectedProfession.primaryTool.name}
                      </h3>
                      <p className="mt-3 text-sm leading-relaxed text-gray-100">
                        {selectedProfession.primaryTool.description}
                      </p>
                      <div className="mt-4 space-y-2 text-sm text-gray-200">
                        <p>
                          <span className="font-semibold text-white">
                            Pricing signal:
                          </span>{" "}
                          {selectedProfession.primaryTool.pricingSignal}
                        </p>
                        <p>
                          <span className="font-semibold text-white">
                            Source:
                          </span>{" "}
                          {selectedProfession.primaryTool.sourceLabel} ·{" "}
                          {selectedProfession.primaryTool.sourceKind}
                        </p>
                      </div>
                    </motion.a>

                    <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
                      <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-100">
                        Alternative stack
                      </p>
                      <div className="mt-4 space-y-4">
                        {selectedProfession.alternativeTools.map((tool) => (
                          <motion.a
                            key={tool.name}
                            href={tool.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block rounded-2xl border border-white/10 bg-white/[0.05] p-4 transition-colors hover:bg-white/[0.08]"
                          >
                            <div className="flex items-center justify-between gap-3">
                              <h4 className="text-lg font-black text-white">
                                {tool.name}
                              </h4>
                              <span className="text-xs font-semibold text-gray-300">
                                {tool.category}
                              </span>
                            </div>
                            <p className="mt-2 text-sm leading-relaxed text-gray-100">
                              {tool.description}
                            </p>
                            <p className="mt-3 text-sm text-cyan-100">
                              {tool.pricingSignal}
                            </p>
                          </motion.a>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-100">
                      Official sources in this brief
                    </p>
                    <div className="mt-4 grid gap-4 md:grid-cols-3">
                      {[
                        selectedProfession.primaryTool,
                        ...selectedProfession.alternativeTools,
                      ].map((tool) => (
                        <motion.a
                          key={`${selectedProfession.id}-${tool.name}`}
                          href={tool.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="rounded-2xl border border-white/10 bg-white/[0.05] p-4 transition-colors hover:bg-white/[0.08]"
                        >
                          <p className="text-sm font-black text-white">
                            {tool.name}
                          </p>
                          <p className="mt-2 text-xs font-semibold uppercase tracking-[0.16em] text-gray-300">
                            {tool.sourceKind}
                          </p>
                          <p className="mt-2 text-sm text-cyan-100">
                            {tool.sourceLabel}
                          </p>
                        </motion.a>
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
