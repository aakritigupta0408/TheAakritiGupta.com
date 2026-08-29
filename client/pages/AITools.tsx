import { type ReactNode, useEffect, useMemo, useState } from "react";
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

const impactTagStyles: Record<ImpactLevel, string> = {
  Critical: "border-[#A15C22]/30 bg-[#A15C22]/10 text-[#8a4d1c]",
  High: "border-[#2C5A73]/30 bg-[#2C5A73]/10 text-[#2C5A73]",
  Medium: "border-slate-300 bg-slate-100 text-slate-500",
};

/** The page's signature element: a tick-marked ruler filled to the
 *  profession's AI-adoption level. Ticks every 10% read as an instrument
 *  scale, not a progress bar. */
function AdoptionRuler({
  value,
  tall = false,
}: {
  value: number;
  tall?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={`relative flex-1 overflow-hidden rounded-sm border border-slate-300 bg-white ${
          tall ? "h-3.5" : "h-2.5"
        }`}
        role="img"
        aria-label={`${value}% AI adoption`}
        style={{
          backgroundImage:
            "repeating-linear-gradient(to right, transparent 0, transparent calc(10% - 1px), #d5dce2 calc(10% - 1px), #d5dce2 10%)",
        }}
      >
        <div
          className="absolute inset-y-0 left-0 bg-[#2C5A73] transition-[width] duration-700 ease-out motion-reduce:transition-none"
          style={{ width: `${value}%` }}
        />
      </div>
      <span className="font-mono text-[11px] font-semibold tabular-nums text-[#2C5A73]">
        {value}%
      </span>
    </div>
  );
}

function MonoLabel({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
      {children}
    </p>
  );
}

export default function AITools() {
  const [selectedProfession, setSelectedProfession] =
    useState<ProfessionProfile | null>(null);
  const [filterImpact, setFilterImpact] = useState<"All" | ImpactLevel>("All");
  const [sortBy, setSortBy] = useState<SortMode>("impact");
  const [visibleCount, setVisibleCount] = useState(9);

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
    setVisibleCount(9);
  }, [filterImpact, sortBy]);

  const visibleProfessions = filteredProfessions.slice(0, visibleCount);
  const hasMoreProfessions =
    visibleProfessions.length < filteredProfessions.length;

  return (
    <SubpageLayout
      route="/ai-tools"
      eyebrow={pageRefresh.eyebrow}
      title={pageRefresh.title}
      description={pageRefresh.description}
      accent="amber"
      frameClassName="bg-[#f5f5f7]"
      chips={pageRefresh.chips}
      metrics={[
        {
          value: professions.length.toString(),
          label: "Profession playbooks",
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
      <div className="container mx-auto px-4 py-6 text-slate-900">
        {/* This week's radar — live entries from the weekly refresh agent */}
        <section className="mb-8 rounded-xl border border-slate-200 bg-white p-5">
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <h2 className="font-serif text-lg font-bold text-slate-900">
              This week on the radar
            </h2>
            <MonoLabel>refreshed weekly</MonoLabel>
          </div>
          <div className="divide-y divide-slate-100">
            {latestAIProductLaunches.slice(0, 4).map((launch) => (
              <a
                key={launch.id}
                href={launch.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group flex flex-col gap-1 py-2.5 first:pt-0 last:pb-0 sm:flex-row sm:items-baseline sm:gap-4"
              >
                <span className="w-32 shrink-0 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-[#A15C22]">
                  {launch.category}
                </span>
                <span className="flex-1 text-sm font-medium text-slate-800 group-hover:underline">
                  {launch.title}
                </span>
                <span className="font-mono text-[11px] text-slate-400">
                  {launch.org} · {launch.date}
                </span>
              </a>
            ))}
          </div>
        </section>

        {/* Index controls */}
        <section className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-1.5">
            <MonoLabel>impact</MonoLabel>
            {impactFilters.map((impact) => (
              <button
                key={impact}
                onClick={() => setFilterImpact(impact)}
                className={`rounded-md border px-3 py-1.5 text-xs font-semibold transition ${
                  filterImpact === impact
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-300 bg-white text-slate-600 hover:border-slate-400"
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
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2C5A73]"
            >
              <option value="impact">Sort: impact shift</option>
              <option value="adoption">Sort: AI adoption</option>
              <option value="alphabetical">Sort: A to Z</option>
            </select>
            <span className="font-mono text-[11px] tabular-nums text-slate-400">
              {visibleProfessions.length}/{filteredProfessions.length}
            </span>
          </div>
        </section>

        {/* Profession index cards */}
        <div className="mb-6 grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-3">
          {visibleProfessions.map((profession, index) => (
            <motion.button
              key={profession.id}
              type="button"
              onClick={() => setSelectedProfession(profession)}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.35,
                delay: Math.min(index * 0.04, 0.2),
              }}
              className="group flex flex-col rounded-xl border border-slate-200 bg-white p-5 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md motion-reduce:hover:translate-y-0"
            >
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="flex h-11 w-11 items-center justify-center rounded-lg border border-slate-200 bg-[#f5f5f7] text-xl">
                    {profession.icon}
                  </span>
                  <h3 className="font-serif text-lg font-bold leading-snug text-slate-900">
                    {profession.title}
                  </h3>
                </div>
                <span
                  className={`shrink-0 rounded-md border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.12em] ${impactTagStyles[profession.impactLevel]}`}
                >
                  {profession.impactLevel}
                </span>
              </div>

              <div className="mb-3">
                <MonoLabel>ai adoption</MonoLabel>
                <div className="mt-1">
                  <AdoptionRuler value={profession.aiAdoption} />
                </div>
              </div>

              <p className="mb-4 line-clamp-2 text-[13px] leading-relaxed text-slate-500">
                {profession.workflowNow}
              </p>

              <div className="mt-auto border-t border-slate-100 pt-3">
                <div className="flex items-baseline justify-between gap-3">
                  <div className="min-w-0">
                    <MonoLabel>primary tool</MonoLabel>
                    <p className="truncate text-sm font-bold text-[#A15C22]">
                      {profession.primaryTool.name}
                    </p>
                  </div>
                  <div className="min-w-0 text-right">
                    <MonoLabel>saves</MonoLabel>
                    <p className="truncate text-sm font-semibold text-slate-700">
                      {profession.timeSaved}
                    </p>
                  </div>
                </div>
                <p className="mt-2 flex items-center justify-between gap-2">
                  <span className="line-clamp-1 font-mono text-[11px] text-slate-400">
                    also:{" "}
                    {profession.alternativeTools
                      .slice(0, 3)
                      .map((tool) => tool.name)
                      .join(" · ")}
                  </span>
                  <span className="shrink-0 text-xs font-semibold text-slate-500 transition-transform group-hover:translate-x-0.5 motion-reduce:group-hover:translate-x-0">
                    Open →
                  </span>
                </p>
              </div>
            </motion.button>
          ))}
        </div>

        {hasMoreProfessions && (
          <LevelOneLoadMoreButton
            variant="light"
            label="Load more playbooks"
            onClick={() => setVisibleCount((current) => current + 6)}
          />
        )}

        {/* Playbook detail sheet */}
        <AnimatePresence>
          {selectedProfession && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm"
              onClick={() => setSelectedProfession(null)}
            >
              <motion.div
                initial={{ scale: 0.97, opacity: 0, y: 16 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.97, opacity: 0, y: 16 }}
                transition={{ duration: 0.2 }}
                className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-2xl border border-slate-200 bg-[#fdfdfc] shadow-2xl"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="border-b border-slate-200 p-6 sm:p-8">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-4">
                      <span className="flex h-14 w-14 items-center justify-center rounded-xl border border-slate-200 bg-white text-3xl">
                        {selectedProfession.icon}
                      </span>
                      <div>
                        <span
                          className={`inline-block rounded-md border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.12em] ${impactTagStyles[selectedProfession.impactLevel]}`}
                        >
                          {selectedProfession.impactLevel} shift
                        </span>
                        <h2 className="mt-1 font-serif text-2xl font-bold text-slate-900 sm:text-3xl">
                          {selectedProfession.title}
                        </h2>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedProfession(null)}
                      aria-label="Close playbook"
                      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-lg text-slate-500 transition hover:border-slate-400 hover:text-slate-900"
                    >
                      ✕
                    </button>
                  </div>
                  <p className="mt-4 max-w-3xl text-sm leading-relaxed text-slate-600">
                    {selectedProfession.description}
                  </p>
                </div>

                <div className="p-6 sm:p-8">
                  <div className="mb-8 grid gap-px overflow-hidden rounded-xl border border-slate-200 bg-slate-200 sm:grid-cols-3">
                    <div className="bg-white p-4">
                      <MonoLabel>ai adoption</MonoLabel>
                      <div className="mt-2">
                        <AdoptionRuler
                          value={selectedProfession.aiAdoption}
                          tall
                        />
                      </div>
                    </div>
                    <div className="bg-white p-4">
                      <MonoLabel>typical time gain</MonoLabel>
                      <p className="mt-1.5 text-lg font-bold text-slate-900">
                        {selectedProfession.timeSaved}
                      </p>
                    </div>
                    <div className="bg-white p-4">
                      <MonoLabel>workflow now</MonoLabel>
                      <p className="mt-1.5 text-[13px] leading-relaxed text-slate-600">
                        {selectedProfession.workflowNow}
                      </p>
                    </div>
                  </div>

                  <div className="mb-8 grid gap-4 lg:grid-cols-2">
                    <a
                      href={selectedProfession.primaryTool.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group rounded-xl border-2 border-[#A15C22]/40 bg-white p-5 transition hover:border-[#A15C22]"
                    >
                      <div className="mb-2 flex items-center justify-between gap-3">
                        <span className="rounded-md bg-[#A15C22] px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-white">
                          Primary tool
                        </span>
                        <span className="font-mono text-[11px] text-slate-400">
                          {selectedProfession.primaryTool.category}
                        </span>
                      </div>
                      <h3 className="font-serif text-xl font-bold text-slate-900 group-hover:underline">
                        {selectedProfession.primaryTool.name} ↗
                      </h3>
                      <p className="mt-2 text-sm leading-relaxed text-slate-600">
                        {selectedProfession.primaryTool.description}
                      </p>
                      <dl className="mt-4 space-y-1.5 text-[13px] text-slate-600">
                        <div className="flex gap-2">
                          <dt className="font-semibold text-slate-900">
                            Pricing:
                          </dt>
                          <dd>
                            {selectedProfession.primaryTool.pricingSignal}
                          </dd>
                        </div>
                        <div className="flex gap-2">
                          <dt className="font-semibold text-slate-900">
                            Source:
                          </dt>
                          <dd>
                            {selectedProfession.primaryTool.sourceLabel} ·{" "}
                            {selectedProfession.primaryTool.sourceKind}
                          </dd>
                        </div>
                      </dl>
                    </a>

                    <div className="rounded-xl border border-slate-200 bg-white p-5">
                      <MonoLabel>alternatives</MonoLabel>
                      <div className="mt-3 divide-y divide-slate-100">
                        {selectedProfession.alternativeTools.map((tool) => (
                          <a
                            key={tool.name}
                            href={tool.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group block py-3 first:pt-0 last:pb-0"
                          >
                            <div className="flex items-baseline justify-between gap-3">
                              <h4 className="text-sm font-bold text-slate-900 group-hover:underline">
                                {tool.name} ↗
                              </h4>
                              <span className="font-mono text-[11px] text-slate-400">
                                {tool.category}
                              </span>
                            </div>
                            <p className="mt-1 line-clamp-2 text-[13px] leading-relaxed text-slate-600">
                              {tool.description}
                            </p>
                            <p className="mt-1 font-mono text-[11px] text-[#2C5A73]">
                              {tool.pricingSignal}
                            </p>
                          </a>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white p-5">
                    <MonoLabel>official sources in this playbook</MonoLabel>
                    <div className="mt-3 grid gap-3 sm:grid-cols-3">
                      {[
                        selectedProfession.primaryTool,
                        ...selectedProfession.alternativeTools,
                      ].map((tool) => (
                        <a
                          key={`${selectedProfession.id}-${tool.name}`}
                          href={tool.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="rounded-lg border border-slate-200 bg-[#f5f5f7] p-3 transition hover:border-slate-400"
                        >
                          <p className="text-sm font-bold text-slate-900">
                            {tool.name}
                          </p>
                          <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-slate-400">
                            {tool.sourceKind}
                          </p>
                          <p className="mt-1 text-[13px] text-slate-600">
                            {tool.sourceLabel}
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
