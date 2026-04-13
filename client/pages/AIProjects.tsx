import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
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
      refreshSummary={pageRefresh.refreshSummary}
      updatedAtLabel={pageRefresh.updatedAtLabel}
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
          className="mb-8 rounded-[1.75rem] border border-white/10 bg-slate-950/25 p-6 backdrop-blur-xl"
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
          <div className="mb-12 rounded-[2rem] border border-white/20 bg-white/10 p-12 text-center backdrop-blur-xl">
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
                transition={{ duration: 0.45, delay: Math.min(index * 0.08, 0.32) }}
                className="group rounded-3xl border border-white/20 bg-white/10 p-8 text-left shadow-2xl backdrop-blur-xl transition-all duration-300 hover:-translate-y-2 hover:bg-white/15"
              >
                <div className="mb-6 flex items-start justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className="text-5xl">{project.icon}</div>
                    <div>
                      <h3 className="text-2xl font-black text-white group-hover:text-pink-200">
                        {project.title}
                      </h3>
                      <p className="mt-2 inline-block rounded-full bg-white/10 px-3 py-1 text-sm font-medium text-gray-200">
                        {project.category}
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-col gap-3 text-right">
                    <span
                      className={`rounded-full border px-4 py-2 text-xs font-bold ${
                        project.difficulty === "Beginner"
                          ? "border-green-300/30 bg-green-500/20 text-green-100"
                          : project.difficulty === "Intermediate"
                            ? "border-orange-300/30 bg-orange-500/20 text-orange-100"
                            : "border-red-300/30 bg-red-500/20 text-red-100"
                      }`}
                    >
                      {project.difficulty}
                    </span>
                    <span className="rounded-full border border-blue-300/30 bg-blue-500/20 px-4 py-2 text-xs font-bold text-blue-100">
                      {project.timeToComplete}
                    </span>
                  </div>
                </div>

                <p className="mb-5 text-lg leading-relaxed text-gray-100">
                  {project.summary}
                </p>

                <div className="rounded-[1.5rem] border border-white/15 bg-black/20 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-200">
                    Current build angle
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-gray-200">
                    {project.buildNow}
                  </p>
                </div>

                <div className="mt-5 flex flex-wrap gap-2">
                  {project.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full border border-white/20 bg-white/10 px-3 py-2 text-xs font-bold text-blue-100"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                <p className="mt-6 text-sm font-bold text-pink-200">
                  Open current stack, papers, and code starter →
                </p>
              </motion.button>
            ))}
          </div>
        )}

        {hasMoreProjects && (
          <div className="mb-12 flex justify-center">
            <motion.button
              onClick={() => setVisibleCount((current) => current + 6)}
              className="rounded-full border border-white/20 bg-white/10 px-6 py-3 text-sm font-bold text-white shadow-xl backdrop-blur-md transition-all duration-300 hover:bg-white/15"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              Show 6 more projects
            </motion.button>
          </div>
        )}

        <AnimatePresence>
          {selectedProject && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
              onClick={() => setSelectedProject(null)}
            >
              <motion.div
                initial={{ scale: 0.96, opacity: 0, y: 12 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.96, opacity: 0, y: 12 }}
                className="max-h-[90vh] w-full max-w-6xl overflow-y-auto rounded-[2rem] border border-white/20 bg-[#f8fafc] shadow-2xl"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="p-6 sm:p-8">
                  <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div className="flex items-center gap-5">
                      <div className="text-6xl">{selectedProject.icon}</div>
                      <div>
                        <h2 className="text-3xl font-black text-slate-950 sm:text-4xl">
                          {selectedProject.title}
                        </h2>
                        <div className="mt-3 flex flex-wrap gap-3">
                          <span className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white">
                            {selectedProject.category}
                          </span>
                          <span className="rounded-full bg-cyan-100 px-4 py-2 text-sm font-semibold text-cyan-800">
                            {selectedProject.difficulty}
                          </span>
                          <span className="rounded-full bg-pink-100 px-4 py-2 text-sm font-semibold text-pink-800">
                            {selectedProject.timeToComplete}
                          </span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => setSelectedProject(null)}
                      className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
                    >
                      Close
                    </button>
                  </div>

                  <div className="grid gap-6 lg:grid-cols-[1.05fr,1.25fr]">
                    <div className="space-y-6">
                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">
                          Project Overview
                        </h3>
                        <p className="mt-4 leading-relaxed text-slate-700">
                          {selectedProject.summary}
                        </p>
                        <div className="mt-5 rounded-2xl bg-slate-50 p-4">
                          <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                            Current build angle
                          </p>
                          <p className="mt-2 leading-relaxed text-slate-700">
                            {selectedProject.buildNow}
                          </p>
                        </div>
                      </div>

                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">Use Cases</h3>
                        <div className="mt-4 space-y-3">
                          {selectedProject.useCases.map((item) => (
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
                        <h3 className="text-lg font-black text-slate-950">
                          Recommended Stack
                        </h3>
                        <div className="mt-4 flex flex-wrap gap-3">
                          {selectedProject.recommendedStack.map((item) => (
                            <span
                              key={item}
                              className="rounded-full border border-cyan-200 bg-cyan-50 px-4 py-2 text-sm font-medium text-cyan-900"
                            >
                              {item}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-6">
                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">
                          Build Steps
                        </h3>
                        <div className="mt-4 space-y-3">
                          {selectedProject.buildSteps.map((step, index) => (
                            <div
                              key={step}
                              className="flex gap-4 rounded-2xl border border-slate-200 bg-slate-50 p-4"
                            >
                              <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-slate-900 text-sm font-bold text-white">
                                {index + 1}
                              </span>
                              <p className="text-sm leading-relaxed text-slate-700">
                                {step}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-lg font-black text-slate-950">
                          Current Resources
                        </h3>
                        <div className="mt-4 space-y-4">
                          {selectedProject.resources.map((resource) => (
                            <a
                              key={resource.url}
                              href={resource.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block rounded-2xl border border-slate-200 bg-slate-50 p-4 transition hover:border-cyan-300 hover:bg-cyan-50"
                            >
                              <div className="flex flex-wrap items-center justify-between gap-3">
                                <p className="text-sm font-semibold text-slate-900">
                                  {resource.name}
                                </p>
                                <span className="rounded-full bg-slate-900 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.18em] text-white">
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
                    <div className="rounded-[1.5rem] border border-slate-200 bg-white p-6 shadow-sm">
                      <h3 className="text-lg font-black text-slate-950">
                        Key Papers
                      </h3>
                      <div className="mt-4 space-y-3">
                        {selectedProject.keyPapers.map((paper) => (
                          <a
                            key={paper.url}
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm font-semibold text-slate-800 transition hover:border-pink-300 hover:bg-pink-50"
                          >
                            {paper.title}
                          </a>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-[1.5rem] border border-slate-200 bg-slate-950 p-6 shadow-sm">
                      <h3 className="text-lg font-black text-white">Starter Code</h3>
                      <pre className="mt-4 overflow-x-auto rounded-2xl bg-black/50 p-5 text-sm leading-relaxed text-slate-100">
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
