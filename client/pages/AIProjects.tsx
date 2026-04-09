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
        { value: projects.length.toString(), label: "Project types" },
        {
          value: buildNowProjectTracks.length.toString(),
          label: "Build-now tracks",
        },
        {
          value: (projectCategories.length - 1).toString(),
          label: "Main categories",
        },
        {
          value: filteredProjects.length.toString(),
          label: "Results in current view",
        },
      ]}
    >
      <div className="container mx-auto px-6 py-10 sm:py-12">
        <motion.section
          className="mb-14"
          initial={{ opacity: 0, y: 22 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <div className="rounded-[2rem] border border-white/15 bg-slate-950/25 p-8 backdrop-blur-xl">
            <div className="mb-8 text-center">
              <div className="inline-flex items-center gap-2 rounded-full border border-pink-300/30 bg-pink-400/10 px-4 py-2 text-sm font-semibold text-pink-100">
                Updated for April 2026
              </div>
              <h2 className="mt-4 text-3xl font-black text-white md:text-4xl">
                Highest-Leverage AI Projects To Build Right Now
              </h2>
              <p className="mx-auto mt-3 max-w-4xl text-gray-100 leading-relaxed">
                The legacy project cards have been rewritten around current
                stacks, current papers, and realistic build paths. The focus is
                now production relevance, not outdated model name-dropping.
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
              {buildNowProjectTracks.map((track, index) => (
                <motion.a
                  key={track.id}
                  href={track.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.45, delay: index * 0.08 }}
                  className="group rounded-3xl border border-white/15 bg-white/10 p-6 transition-all duration-300 hover:-translate-y-1 hover:bg-white/15"
                >
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <span className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                      {track.category}
                    </span>
                    <span
                      className={`rounded-full px-3 py-1 text-xs font-bold ${
                        track.difficulty === "Beginner"
                          ? "border border-green-300/30 bg-green-500/20 text-green-100"
                          : track.difficulty === "Intermediate"
                            ? "border border-yellow-300/30 bg-yellow-500/20 text-yellow-100"
                            : "border border-pink-300/30 bg-pink-500/20 text-pink-100"
                      }`}
                    >
                      {track.difficulty}
                    </span>
                  </div>
                  <h3 className="mb-3 text-2xl font-black text-white group-hover:text-cyan-200">
                    {track.title}
                  </h3>
                  <p className="mb-4 text-sm leading-relaxed text-gray-100">
                    {track.summary}
                  </p>
                  <p className="mb-4 text-sm leading-relaxed text-cyan-100">
                    {track.outcome}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {track.stack.map((item) => (
                      <span
                        key={item}
                        className="rounded-full border border-white/15 bg-black/20 px-3 py-1 text-xs font-medium text-gray-100"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </motion.a>
              ))}
            </div>
          </div>
        </motion.section>

        <section className="mb-12 space-y-6">
          {(filterCategory !== "All" || filterDifficulty !== "All") && (
            <div className="text-center">
              <motion.button
                onClick={() => {
                  setFilterCategory("All");
                  setFilterDifficulty("All");
                }}
                className="rounded-full border border-red-400/30 bg-gradient-to-r from-red-500 to-pink-600 px-6 py-3 font-bold text-white shadow-xl transition-all duration-300"
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                Clear All Filters
              </motion.button>
            </div>
          )}

          <div className="flex flex-wrap justify-center gap-4">
            <span className="rounded-full border border-white/20 bg-white/10 px-6 py-3 text-sm font-black text-white backdrop-blur-md">
              Filter by Category
            </span>
            {projectCategories.map((category) => (
              <motion.button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`rounded-full px-6 py-3 text-sm font-bold transition-all duration-300 ${
                  filterCategory === category
                    ? "scale-105 border border-pink-400/50 bg-gradient-to-r from-pink-500 to-purple-600 text-white shadow-2xl"
                    : "border border-white/20 bg-white/10 text-white hover:bg-white/20"
                }`}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                {category}
              </motion.button>
            ))}
          </div>

          <div className="flex flex-wrap justify-center gap-4">
            <span className="rounded-full border border-white/20 bg-white/10 px-6 py-3 text-sm font-black text-white backdrop-blur-md">
              Filter by Difficulty
            </span>
            {difficulties.map((difficulty) => (
              <motion.button
                key={difficulty}
                onClick={() => setFilterDifficulty(difficulty)}
                className={`rounded-full px-6 py-3 text-sm font-bold transition-all duration-300 ${
                  filterDifficulty === difficulty
                    ? "scale-105 border border-cyan-400/50 bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-2xl"
                    : "border border-white/20 bg-white/10 text-white hover:bg-white/20"
                }`}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                {difficulty}
              </motion.button>
            ))}
          </div>

          <div className="text-center">
            <div className="inline-block rounded-full border border-white/20 bg-white/10 px-6 py-3 backdrop-blur-md">
              <span className="font-bold text-white">
                Showing {visibleProjects.length} of {filteredProjects.length} projects
                {filterCategory !== "All" && ` in ${filterCategory}`}
                {filterDifficulty !== "All" && ` (${filterDifficulty})`}
              </span>
            </div>
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
