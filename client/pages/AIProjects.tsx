import React, { useEffect, useMemo, useState } from "react";
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
      accent="rose"
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
      <div className="container mx-auto px-6 py-10 sm:py-12">
        <motion.section
          className="mb-8 rounded-[1.75rem] border border-white/10 bg-white/[0.06] p-4 backdrop-blur-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-4 flex items-baseline justify-between gap-3">
            <h2 className="text-lg font-semibold text-white">
              Build-now tracks
            </h2>
            <span className="text-[11px] uppercase tracking-[0.2em] text-gray-400">
              Current stacks, realistic scope
            </span>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {buildNowProjectTracks.map((track) => (
              <a
                key={track.id}
                href={track.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:bg-white/10"
              >
                <div className="mb-2 flex items-center justify-between gap-2 text-[11px] uppercase tracking-[0.18em] text-gray-400">
                  <span className="bg-gradient-to-r from-pink-300 to-cyan-300 bg-clip-text text-transparent">
                    {track.category}
                  </span>
                  <span>{track.difficulty}</span>
                </div>
                <h3 className="text-sm font-semibold leading-snug text-white">
                  {track.title}
                </h3>
                <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-gray-300">
                  {track.outcome}
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
            {projectCategories.map((category) => (
              <button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                  filterCategory === category
                    ? "bg-gradient-to-r from-pink-500 to-fuchsia-500 text-white"
                    : "border border-white/15 bg-white/5 text-gray-200 hover:bg-white/10"
                }`}
              >
                {category}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400">
                Level
              </span>
              {difficulties.map((difficulty) => (
                <button
                  key={difficulty}
                  onClick={() => setFilterDifficulty(difficulty)}
                  className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                    filterDifficulty === difficulty
                      ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white"
                      : "border border-white/15 bg-white/5 text-gray-200 hover:bg-white/10"
                  }`}
                >
                  {difficulty}
                </button>
              ))}
            </div>
            <span className="text-xs text-gray-400">
              {visibleProjects.length} / {filteredProjects.length}
            </span>
          </div>
        </section>

        {filteredProjects.length === 0 ? (
          <div className="mb-12 rounded-[2rem] border border-white/20 bg-white/10 p-6 text-center backdrop-blur-xl">
            <h3 className="text-3xl font-black text-white">No projects found</h3>
            <p className="mt-3 text-lg text-gray-200">
              Reset the filters to compare the full build library again.
            </p>
          </div>
        ) : (
          <div className="mb-12 grid grid-cols-1 gap-8 lg:grid-cols-2">
            {visibleProjects.map((project, index) => (
              <motion.button
                key={project.id}
                type="button"
                onClick={() => setSelectedProject(project)}
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  type: "spring",
                  stiffness: 260,
                  damping: 24,
                  delay: Math.min(index * 0.06, 0.25),
                }}
                whileHover={{ y: -6, rotateX: 2, rotateY: -2 }}
                whileTap={{ scale: 0.98 }}
                style={{ transformStyle: "preserve-3d", perspective: 900 }}
                className="group relative overflow-hidden rounded-[24px] border border-white/10 bg-gradient-to-br from-white/[0.09] to-white/[0.03] p-4 text-left shadow-[0_24px_56px_rgba(8,12,24,0.45),inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-2xl transition-colors duration-300 hover:border-white/25"
              >
                <span className="pointer-events-none absolute inset-x-0 -top-px h-px bg-gradient-to-r from-transparent via-pink-300/60 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
                <span className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-gradient-to-br from-pink-400/15 via-transparent to-transparent opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-100" />

                <div className="relative mb-5 flex items-start justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="text-4xl drop-shadow-[0_4px_12px_rgba(244,114,182,0.35)] transition-transform duration-300 group-hover:scale-110">
                      {project.icon}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold leading-snug text-white transition-colors group-hover:text-pink-200">
                        {project.title}
                      </h3>
                      <p className="mt-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">
                        {project.category}
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-1.5">
                    <span
                      className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ring-1 ring-inset ${
                        project.difficulty === "Beginner"
                          ? "bg-emerald-500/15 text-emerald-100 ring-emerald-300/30"
                          : project.difficulty === "Intermediate"
                            ? "bg-amber-500/15 text-amber-100 ring-amber-300/30"
                            : "bg-rose-500/15 text-rose-100 ring-rose-300/30"
                      }`}
                    >
                      {project.difficulty}
                    </span>
                    <span className="rounded-full bg-sky-500/15 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-sky-100 ring-1 ring-inset ring-sky-300/30">
                      {project.timeToComplete}
                    </span>
                  </div>
                </div>

                <p className="relative mb-4 line-clamp-3 text-sm leading-relaxed text-slate-200">
                  {project.summary}
                </p>

                <div className="relative rounded-2xl border border-white/10 bg-white/[0.06] p-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-200">
                    Current build angle
                  </p>
                  <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-slate-300">
                    {project.buildNow}
                  </p>
                </div>

                <div className="relative mt-4 flex flex-wrap gap-1.5">
                  {project.tags.slice(0, 4).map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-[10px] font-medium text-slate-200"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                <p className="relative mt-5 text-xs font-semibold text-pink-200 transition-transform duration-300 group-hover:translate-x-1">
                  Open stack, papers, starter code →
                </p>
              </motion.button>
            ))}
          </div>
        )}

        {hasMoreProjects && (
          <div className="mb-12 flex justify-center">
            <LevelOneLoadMoreButton
              label="Load 6 more"
              glowClassName="from-pink-400/0 via-pink-400/20 to-pink-400/0"
              onClick={() => setVisibleCount((current) => current + 6)}
            />
          </div>
        )}

        <AnimatePresence>
          {selectedProject && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/72 p-4 backdrop-blur-md"
              onClick={() => setSelectedProject(null)}
            >
              <motion.div
                initial={{ scale: 0.96, opacity: 0, y: 12 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.96, opacity: 0, y: 12 }}
                className="relative max-h-[90vh] w-full max-w-6xl overflow-y-auto rounded-[2.25rem] border border-white/15 bg-gradient-to-br from-slate-50 via-white to-pink-50/70 shadow-[0_36px_120px_rgba(8,12,24,0.4)]"
                onClick={(event) => event.stopPropagation()}
              >
                <span className="pointer-events-none absolute inset-x-10 top-0 h-px bg-gradient-to-r from-transparent via-pink-300/80 to-transparent" />
                <span className="pointer-events-none absolute -right-20 top-10 h-56 w-56 rounded-full bg-pink-400/14 blur-3xl" />
                <span className="pointer-events-none absolute -left-16 bottom-8 h-44 w-44 rounded-full bg-sky-400/12 blur-3xl" />

                <div className="relative p-5 sm:p-4">
                  <div className="mb-6 overflow-hidden rounded-[28px] border border-white/70 bg-gradient-to-br from-pink-400/14 via-white/92 to-sky-400/12 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.12)] sm:p-4">
                    <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                      <div className="flex items-center gap-5">
                        <div className="flex h-20 w-20 items-center justify-center rounded-[24px] border border-white/70 bg-white/80 text-5xl shadow-[0_20px_50px_rgba(236,72,153,0.16)]">
                          {selectedProject.icon}
                        </div>
                        <div>
                          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-pink-700/75">
                            Build-now project brief
                          </p>
                          <h2 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-4xl">
                            {selectedProject.title}
                          </h2>
                          <div className="mt-3 flex flex-wrap gap-2.5">
                            <span className="rounded-full bg-slate-950 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-white">
                              {selectedProject.category}
                            </span>
                            <span className="rounded-full border border-cyan-200 bg-cyan-50 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-800">
                              {selectedProject.difficulty}
                            </span>
                            <span className="rounded-full border border-pink-200 bg-pink-50 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-pink-800">
                              {selectedProject.timeToComplete}
                            </span>
                          </div>
                        </div>
                      </div>
                      <motion.button
                        whileHover={{ rotate: 90, scale: 1.06 }}
                        whileTap={{ scale: 0.94 }}
                        onClick={() => setSelectedProject(null)}
                        aria-label="Close project details"
                        className="rounded-full border border-slate-200/80 bg-white/80 p-3 text-slate-500 shadow-sm transition-colors hover:border-slate-300 hover:text-slate-900"
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
                          Difficulty
                        </p>
                        <p className="mt-2 text-sm font-medium text-slate-800">
                          {selectedProject.difficulty}
                        </p>
                      </div>
                      <div className="rounded-[22px] border border-white/70 bg-white/78 p-4 shadow-[0_18px_40px_rgba(15,23,42,0.08)]">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                          Time to complete
                        </p>
                        <p className="mt-2 text-sm font-medium text-slate-800">
                          {selectedProject.timeToComplete}
                        </p>
                      </div>
                      <div className="rounded-[22px] border border-white/70 bg-white/78 p-4 shadow-[0_18px_40px_rgba(15,23,42,0.08)]">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                          Reference set
                        </p>
                        <p className="mt-2 text-sm font-medium text-slate-800">
                          {selectedProject.resources.length} resources +{" "}
                          {selectedProject.keyPapers.length} papers
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-6 lg:grid-cols-[1.05fr,1.25fr]">
                    <div className="space-y-6">
                      <div className="rounded-[1.75rem] border border-white/70 bg-white/82 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                        <h3 className="text-lg font-semibold text-slate-950">
                          Project Overview
                        </h3>
                        <p className="mt-4 leading-relaxed text-slate-700">
                          {selectedProject.summary}
                        </p>
                        <div className="mt-5 rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                            Current build angle
                          </p>
                          <p className="mt-2 leading-relaxed text-slate-700">
                            {selectedProject.buildNow}
                          </p>
                        </div>
                      </div>

                      <div className="rounded-[1.75rem] border border-white/70 bg-white/82 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                        <h3 className="text-lg font-semibold text-slate-950">Use Cases</h3>
                        <div className="mt-4 space-y-3">
                          {selectedProject.useCases.map((item) => (
                            <div
                              key={item}
                              className="rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4 text-sm leading-relaxed text-slate-700"
                            >
                              {item}
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-[1.75rem] border border-white/70 bg-white/82 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                        <h3 className="text-lg font-semibold text-slate-950">
                          Recommended Stack
                        </h3>
                        <div className="mt-4 flex flex-wrap gap-3">
                          {selectedProject.recommendedStack.map((item) => (
                            <span
                              key={item}
                              className="rounded-full border border-cyan-200 bg-cyan-50 px-4 py-2 text-sm font-medium text-cyan-900 shadow-[0_6px_18px_rgba(14,165,233,0.08)]"
                            >
                              {item}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-6">
                      <div className="rounded-[1.75rem] border border-white/70 bg-white/82 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                        <h3 className="text-lg font-semibold text-slate-950">
                          Build Steps
                        </h3>
                        <div className="mt-4 space-y-3">
                          {selectedProject.buildSteps.map((step, index) => (
                            <div
                              key={step}
                              className="flex gap-4 rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4"
                            >
                              <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-slate-950 text-sm font-bold text-white shadow-[0_10px_24px_rgba(15,23,42,0.14)]">
                                {index + 1}
                              </span>
                              <p className="text-sm leading-relaxed text-slate-700">
                                {step}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-[1.75rem] border border-white/70 bg-white/82 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                        <h3 className="text-lg font-semibold text-slate-950">
                          Current Resources
                        </h3>
                        <div className="mt-4 space-y-4">
                          {selectedProject.resources.map((resource) => (
                            <a
                            key={resource.url}
                            href={resource.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4 transition hover:border-cyan-300 hover:bg-cyan-50/80"
                          >
                            <div className="flex flex-wrap items-center justify-between gap-3">
                              <p className="text-sm font-semibold text-slate-900">
                                {resource.name}
                              </p>
                                <span className="rounded-full bg-slate-950 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-white">
                                  {resource.type}
                                </span>
                              </div>
                              <p className="mt-2 text-sm leading-relaxed text-slate-700">
                                {resource.note}
                              </p>
                            </a>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 grid gap-6 lg:grid-cols-[0.85fr,1.15fr]">
                    <div className="rounded-[1.75rem] border border-white/70 bg-white/82 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.08)]">
                      <h3 className="text-lg font-semibold text-slate-950">
                        Key Papers
                      </h3>
                      <div className="mt-4 space-y-3">
                        {selectedProject.keyPapers.map((paper) => (
                          <a
                            key={paper.url}
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block rounded-[22px] border border-slate-200/70 bg-slate-50/85 p-4 text-sm font-semibold text-slate-800 transition hover:border-pink-300 hover:bg-pink-50/80"
                          >
                            {paper.title}
                          </a>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-[1.75rem] border border-slate-800 bg-slate-800 p-4 shadow-[0_24px_60px_rgba(15,23,42,0.18)]">
                      <div className="flex items-center justify-between gap-3">
                        <h3 className="text-lg font-semibold text-white">Starter Code</h3>
                        <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300">
                          Starter scaffold
                        </span>
                      </div>
                      <pre className="mt-4 overflow-x-auto rounded-[22px] border border-white/10 bg-black/45 p-5 text-sm leading-relaxed text-slate-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                        <code>{selectedProject.codeExample}</code>
                      </pre>
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
