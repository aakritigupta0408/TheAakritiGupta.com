import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
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
      refreshSummary={pageRefresh.refreshSummary}
      updatedAtLabel={pageRefresh.updatedAtLabel}
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
      <div className="container mx-auto px-6 py-10 sm:py-12">
        <motion.section
          className="mb-8 rounded-[1.75rem] border border-white/10 bg-slate-950/25 p-6 backdrop-blur-xl"
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

        <div className="mb-12 grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
          {visibleProfessions.map((profession, index) => (
            <motion.button
              key={profession.id}
              type="button"
              onClick={() => setSelectedProfession(profession)}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, delay: Math.min(index * 0.05, 0.25) }}
              className="group rounded-3xl border border-white/20 bg-white/10 p-6 text-left shadow-2xl backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:bg-white/15"
            >
              <div className="mb-5 flex items-start justify-between gap-4">
                <div className="flex items-center gap-4">
                  <span className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/15 bg-black/20 text-3xl">
                    {profession.icon}
                  </span>
                  <div>
                    <div className="mb-2 inline-flex rounded-full border border-violet-300/30 bg-violet-400/10 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.18em] text-violet-100">
                      {profession.impactLevel}
                    </div>
                    <h3 className="text-2xl font-black text-white group-hover:text-violet-200">
                      {profession.title}
                    </h3>
                  </div>
                </div>
                <span className="rounded-full border border-white/15 bg-white/10 px-3 py-2 text-xs font-semibold text-gray-100">
                  {profession.aiAdoption}% adoption
                </span>
              </div>

              <p className="mb-4 text-sm leading-relaxed text-gray-100">
                {profession.description}
              </p>

              <div className="mb-5 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-white/15 bg-black/20 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-100">
                    Workflow now
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-gray-100">
                    {profession.workflowNow}
                  </p>
                </div>
                <div className="rounded-2xl border border-white/15 bg-black/20 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-100">
                    Time saved
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {profession.timeSaved}
                  </p>
                  <p className="mt-3 text-xs font-semibold uppercase tracking-[0.16em] text-gray-300">
                    Primary tool
                  </p>
                  <p className="mt-1 text-sm text-cyan-100">
                    {profession.primaryTool.name}
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between gap-4 border-t border-white/15 pt-5">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-gray-300">
                    Alternatives
                  </p>
                  <p className="mt-2 text-sm text-white">
                    {profession.alternativeTools.map((tool) => tool.name).join(" • ")}
                  </p>
                </div>
                <span className="rounded-full border border-white/15 bg-white/10 px-4 py-2 text-xs font-semibold text-gray-100">
                  View brief
                </span>
              </div>
            </motion.button>
          ))}
        </div>

        {hasMoreProfessions && (
          <div className="mb-12 flex justify-center">
            <motion.button
              onClick={() => setVisibleCount((current) => current + 4)}
              className="rounded-full border border-white/20 bg-white/10 px-6 py-3 text-sm font-bold text-white shadow-xl backdrop-blur-md transition-all duration-300 hover:bg-white/15"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              Show 4 more role playbooks
            </motion.button>
          </div>
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
                className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-[2rem] border border-white/15 bg-slate-950 shadow-2xl"
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

                  <div className="mb-8 rounded-3xl border border-white/10 bg-white/5 p-6">
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
                      className="rounded-3xl border border-violet-300/20 bg-violet-500/10 p-6 transition-colors hover:bg-violet-500/15"
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

                    <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
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
                            className="block rounded-2xl border border-white/10 bg-black/20 p-4 transition-colors hover:bg-black/30"
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

                  <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
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
                          className="rounded-2xl border border-white/10 bg-black/20 p-4 transition-colors hover:bg-black/30"
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
