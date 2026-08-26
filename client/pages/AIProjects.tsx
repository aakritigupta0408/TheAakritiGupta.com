import { type ReactNode, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import LevelOneLoadMoreButton from "@/components/LevelOneLoadMoreButton";
import SubpageLayout from "@/components/SubpageLayout";
import {
  projects,
  projectCategories,
  type Project,
} from "@/data/projectArchive";
import { getPageRefreshContent } from "@/data/siteRefreshContent";
import { buildNowProjectTracks } from "../data/aiSignals";

const difficulties = ["All", "Beginner", "Intermediate", "Advanced"] as const;

const difficultyPips: Record<Project["difficulty"], number> = {
  Beginner: 1,
  Intermediate: 2,
  Advanced: 3,
};

function MonoLabel({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
      {children}
    </p>
  );
}

/** The page's signature element: difficulty as machinist's gauge pips.
 *  One filled block per level — read at a glance, no color decoding. */
function DifficultyPips({ level }: { level: Project["difficulty"] }) {
  const filled = difficultyPips[level];
  return (
    <span
      className="inline-flex items-center gap-1 font-mono text-[11px] font-semibold text-slate-700"
      aria-label={`Difficulty: ${level}`}
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          aria-hidden="true"
          className={`h-2 w-3 rounded-[2px] border ${
            i < filled
              ? "border-slate-800 bg-slate-800"
              : "border-slate-300 bg-white"
          }`}
        />
      ))}
      <span className="ml-1">{level}</span>
    </span>
  );
}

export default function AIProjects() {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [filterCategory, setFilterCategory] = useState("All");
  const [filterDifficulty, setFilterDifficulty] = useState("All");
  const [visibleCount, setVisibleCount] = useState(6);

  const pageRefresh = getPageRefreshContent("/ai-projects");

  const filteredProjects = useMemo(() => {
    return projects.filter((project) => {
      const categoryMatch =
        filterCategory === "All" || project.category === filterCategory;
      const difficultyMatch =
        filterDifficulty === "All" || project.difficulty === filterDifficulty;
      return categoryMatch && difficultyMatch;
    });
  }, [filterCategory, filterDifficulty]);

  useEffect(() => {
    setVisibleCount(6);
  }, [filterCategory, filterDifficulty]);

  const visibleProjects = filteredProjects.slice(0, visibleCount);
  const hasMoreProjects = visibleProjects.length < filteredProjects.length;

  return (
    <SubpageLayout
      route="/ai-projects"
      eyebrow={pageRefresh.eyebrow}
      title={pageRefresh.title}
      description={pageRefresh.description}
      accent="amber"
      frameClassName="bg-[#f5f5f7]"
      chips={pageRefresh.chips}
      metrics={[
        { value: projects.length.toString(), label: "Project blueprints" },
        {
          value: buildNowProjectTracks.length.toString(),
          label: "Build-now tracks",
        },
        {
          value: (projectCategories.length - 1).toString(),
          label: "Categories",
        },
      ]}
    >
      <div className="container mx-auto px-4 py-6 text-slate-900">
        {/* Featured course */}
        <section className="mb-4 rounded-xl border-2 border-[#A15C22]/40 bg-white p-6">
          <MonoLabel>featured · interactive course</MonoLabel>
          <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
            <div className="max-w-2xl">
              <h2 className="font-serif text-2xl font-bold text-slate-900">
                How Machines Decide
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                A 15-lesson course on how ranking, retrieval, language models,
                agents and reinforcement learning actually work — every lesson a
                beginner story plus a practitioner sequel, 100+ hand-drawn
                figures, one drawn summary per cited paper, and written
                questions graded against a rubric. Runs entirely in your
                browser.
              </p>
              <p className="mt-3 font-mono text-[11px] uppercase tracking-[0.14em] text-slate-400">
                15 lessons · 6 parts · 100+ figures · 2 capstones
              </p>
            </div>
            <a
              href="/ai-course/"
              className="rounded-lg bg-[#A15C22] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[#8a4d1c]"
            >
              Open the course →
            </a>
          </div>
        </section>

        {/* Build-now tracks — live entries from the weekly refresh agent */}
        <section className="mb-8 rounded-xl border border-slate-200 bg-white p-5">
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <h2 className="font-serif text-lg font-bold text-slate-900">
              Build-now tracks
            </h2>
            <MonoLabel>current stacks · refreshed weekly</MonoLabel>
          </div>
          <div className="divide-y divide-slate-100">
            {buildNowProjectTracks.map((track) => (
              <a
                key={track.id}
                href={track.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group flex flex-col gap-1 py-2.5 first:pt-0 last:pb-0 sm:flex-row sm:items-baseline sm:gap-4"
              >
                <span className="w-40 shrink-0 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-[#A15C22]">
                  {track.category}
                </span>
                <span className="flex-1 text-sm font-medium text-slate-800 group-hover:underline">
                  {track.title}
                  <span className="ml-2 font-normal text-slate-500">
                    — {track.outcome}
                  </span>
                </span>
                <span className="shrink-0 font-mono text-[11px] text-slate-400">
                  {track.difficulty}
                </span>
              </a>
            ))}
          </div>
        </section>

        {/* Blueprint index controls */}
        <section className="mb-5 space-y-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <MonoLabel>category</MonoLabel>
            {projectCategories.map((category) => (
              <button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`rounded-md border px-3 py-1.5 text-xs font-semibold transition ${
                  filterCategory === category
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-300 bg-white text-slate-600 hover:border-slate-400"
                }`}
              >
                {category}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-1.5">
              <MonoLabel>level</MonoLabel>
              {difficulties.map((difficulty) => (
                <button
                  key={difficulty}
                  onClick={() => setFilterDifficulty(difficulty)}
                  className={`rounded-md border px-3 py-1.5 text-xs font-semibold transition ${
                    filterDifficulty === difficulty
                      ? "border-slate-900 bg-slate-900 text-white"
                      : "border-slate-300 bg-white text-slate-600 hover:border-slate-400"
                  }`}
                >
                  {difficulty}
                </button>
              ))}
            </div>
            <span className="font-mono text-[11px] tabular-nums text-slate-400">
              {visibleProjects.length}/{filteredProjects.length}
            </span>
          </div>
        </section>

        {filteredProjects.length === 0 ? (
          <div className="mb-6 rounded-xl border border-slate-200 bg-white p-8 text-center">
            <h3 className="font-serif text-xl font-bold text-slate-900">
              No blueprints match these filters
            </h3>
            <p className="mt-2 text-sm text-slate-500">
              Clear a filter to see the full build library again.
            </p>
          </div>
        ) : (
          <div className="mb-6 grid grid-cols-1 gap-3 lg:grid-cols-2">
            {visibleProjects.map((project, index) => (
              <motion.button
                key={project.id}
                type="button"
                onClick={() => setSelectedProject(project)}
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
                      {project.icon}
                    </span>
                    <div>
                      <h3 className="font-serif text-lg font-bold leading-snug text-slate-900">
                        {project.title}
                      </h3>
                      <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-[#A15C22]">
                        {project.category}
                      </p>
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1">
                    <DifficultyPips level={project.difficulty} />
                    <span className="font-mono text-[11px] text-slate-400">
                      {project.timeToComplete}
                    </span>
                  </div>
                </div>

                <p className="mb-3 line-clamp-2 text-[13px] leading-relaxed text-slate-500">
                  {project.summary}
                </p>

                <div className="mb-3 rounded-lg border border-slate-100 bg-[#f5f5f7] p-3">
                  <MonoLabel>current build angle</MonoLabel>
                  <p className="mt-1 line-clamp-2 text-[13px] leading-relaxed text-slate-600">
                    {project.buildNow}
                  </p>
                </div>

                <div className="mt-auto flex items-center justify-between gap-3 border-t border-slate-100 pt-3">
                  <span className="line-clamp-1 font-mono text-[11px] text-slate-400">
                    {project.tags.slice(0, 4).join(" · ")}
                  </span>
                  <span className="shrink-0 text-xs font-semibold text-slate-500 transition-transform group-hover:translate-x-0.5 motion-reduce:group-hover:translate-x-0">
                    Open build sheet →
                  </span>
                </div>
              </motion.button>
            ))}
          </div>
        )}

        {hasMoreProjects && (
          <LevelOneLoadMoreButton
            variant="light"
            label="Load more blueprints"
            onClick={() => setVisibleCount((current) => current + 6)}
          />
        )}

        {/* Build sheet */}
        <AnimatePresence>
          {selectedProject && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm"
              onClick={() => setSelectedProject(null)}
            >
              <motion.div
                initial={{ scale: 0.97, opacity: 0, y: 16 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.97, opacity: 0, y: 16 }}
                transition={{ duration: 0.2 }}
                className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-2xl border border-slate-200 bg-[#fdfdfc] shadow-2xl"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="border-b border-slate-200 p-6 sm:p-8">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-4">
                      <span className="flex h-14 w-14 items-center justify-center rounded-xl border border-slate-200 bg-white text-3xl">
                        {selectedProject.icon}
                      </span>
                      <div>
                        <MonoLabel>build sheet</MonoLabel>
                        <h2 className="mt-1 font-serif text-2xl font-bold text-slate-900 sm:text-3xl">
                          {selectedProject.title}
                        </h2>
                        <div className="mt-2 flex flex-wrap items-center gap-3">
                          <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-[#A15C22]">
                            {selectedProject.category}
                          </span>
                          <DifficultyPips level={selectedProject.difficulty} />
                          <span className="font-mono text-[11px] text-slate-400">
                            {selectedProject.timeToComplete}
                          </span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedProject(null)}
                      aria-label="Close build sheet"
                      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-lg text-slate-500 transition hover:border-slate-400 hover:text-slate-900"
                    >
                      ✕
                    </button>
                  </div>
                  <p className="mt-4 max-w-3xl text-sm leading-relaxed text-slate-600">
                    {selectedProject.summary}
                  </p>
                  <div className="mt-4 rounded-lg border border-slate-200 bg-white p-4">
                    <MonoLabel>current build angle</MonoLabel>
                    <p className="mt-1 text-sm leading-relaxed text-slate-600">
                      {selectedProject.buildNow}
                    </p>
                  </div>
                </div>

                <div className="p-6 sm:p-8">
                  <div className="grid gap-6 lg:grid-cols-2">
                    <div className="space-y-6">
                      <div className="rounded-xl border border-slate-200 bg-white p-5">
                        <h3 className="font-serif text-lg font-bold text-slate-900">
                          Build steps
                        </h3>
                        <ol className="mt-4 space-y-3">
                          {selectedProject.buildSteps.map((step, index) => (
                            <li key={step} className="flex gap-3">
                              <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-slate-300 bg-[#f5f5f7] font-mono text-[11px] font-bold text-slate-700">
                                {index + 1}
                              </span>
                              <p className="text-sm leading-relaxed text-slate-600">
                                {step}
                              </p>
                            </li>
                          ))}
                        </ol>
                      </div>

                      <div className="rounded-xl border border-slate-200 bg-white p-5">
                        <h3 className="font-serif text-lg font-bold text-slate-900">
                          Recommended stack
                        </h3>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {selectedProject.recommendedStack.map((item) => (
                            <span
                              key={item}
                              className="rounded-md border border-[#2C5A73]/30 bg-[#2C5A73]/5 px-2.5 py-1 font-mono text-[12px] font-medium text-[#2C5A73]"
                            >
                              {item}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-xl border border-slate-200 bg-white p-5">
                        <h3 className="font-serif text-lg font-bold text-slate-900">
                          Use cases
                        </h3>
                        <ul className="mt-3 space-y-2">
                          {selectedProject.useCases.map((item) => (
                            <li
                              key={item}
                              className="border-l-2 border-slate-200 pl-3 text-sm leading-relaxed text-slate-600"
                            >
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="space-y-6">
                      <div className="rounded-xl border border-slate-200 bg-white p-5">
                        <h3 className="font-serif text-lg font-bold text-slate-900">
                          Resources
                        </h3>
                        <div className="mt-3 divide-y divide-slate-100">
                          {selectedProject.resources.map((resource) => (
                            <a
                              key={resource.url}
                              href={resource.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="group block py-3 first:pt-0 last:pb-0"
                            >
                              <div className="flex items-baseline justify-between gap-3">
                                <p className="text-sm font-bold text-slate-900 group-hover:underline">
                                  {resource.name} ↗
                                </p>
                                <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-slate-400">
                                  {resource.type}
                                </span>
                              </div>
                              <p className="mt-1 text-[13px] leading-relaxed text-slate-600">
                                {resource.note}
                              </p>
                            </a>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-xl border border-slate-200 bg-white p-5">
                        <h3 className="font-serif text-lg font-bold text-slate-900">
                          Key papers
                        </h3>
                        <div className="mt-3 space-y-2">
                          {selectedProject.keyPapers.map((paper) => (
                            <a
                              key={paper.url}
                              href={paper.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block border-l-2 border-[#A15C22]/40 pl-3 text-sm font-medium text-slate-700 transition hover:border-[#A15C22] hover:text-slate-900 hover:underline"
                            >
                              {paper.title} ↗
                            </a>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
                    <div className="flex items-center justify-between gap-3 border-b border-slate-700 px-5 py-3">
                      <h3 className="font-serif text-base font-bold text-white">
                        Starter code
                      </h3>
                      <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-400">
                        scaffold · adapt before shipping
                      </span>
                    </div>
                    <pre className="overflow-x-auto p-5 font-mono text-[13px] leading-relaxed text-slate-100">
                      <code>{selectedProject.codeExample}</code>
                    </pre>
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
