import type { ReactNode } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  ChevronRight,
  Compass,
  Home,
} from "lucide-react";
import { Link } from "react-router-dom";
import Navigation from "@/components/Navigation";
import { cn } from "@/lib/utils";

type AccentTone = "blue" | "emerald" | "rose" | "amber";

interface SubpageMetric {
  value: string;
  label: string;
  detail?: string;
}

interface SubpageLayoutProps {
  route: string;
  eyebrow: string;
  title: string;
  description: string;
  metrics: SubpageMetric[];
  chips?: string[];
  accent?: AccentTone;
  children: ReactNode;
  frameClassName?: string;
}

const levelOnePages = [
  { path: "/ai-playground", label: "AI Playground" },
  { path: "/ai-discoveries", label: "AI Discoveries" },
  { path: "/ai-tools", label: "AI Tools" },
  { path: "/ai-companies", label: "AI Companies" },
  { path: "/ai-projects", label: "AI Projects" },
  { path: "/prompt-engineering", label: "Prompt Engineering" },
  { path: "/ai-agent-training", label: "AI Agent Training" },
  { path: "/ai-champions", label: "AI Champions" },
  { path: "/resume-builder", label: "Resume Builder" },
  { path: "/games", label: "Games" },
];

const accentStyles: Record<
  AccentTone,
  {
    badge: string;
    value: string;
    activeLink: string;
    glow: string;
  }
> = {
  blue: {
    badge:
      "border-sky-200/80 bg-sky-50/90 text-sky-700 shadow-[0_10px_24px_rgba(14,165,233,0.12)]",
    value: "from-slate-900 via-sky-700 to-blue-500",
    activeLink: "border-sky-200 bg-sky-50 text-sky-800",
    glow: "from-sky-200/70 via-blue-100/60 to-white",
  },
  emerald: {
    badge:
      "border-emerald-200/80 bg-emerald-50/90 text-emerald-700 shadow-[0_10px_24px_rgba(16,185,129,0.12)]",
    value: "from-slate-900 via-emerald-700 to-teal-500",
    activeLink: "border-emerald-200 bg-emerald-50 text-emerald-800",
    glow: "from-emerald-200/70 via-teal-100/60 to-white",
  },
  rose: {
    badge:
      "border-rose-200/80 bg-rose-50/90 text-rose-700 shadow-[0_10px_24px_rgba(244,63,94,0.12)]",
    value: "from-slate-900 via-fuchsia-700 to-rose-500",
    activeLink: "border-rose-200 bg-rose-50 text-rose-800",
    glow: "from-rose-200/70 via-fuchsia-100/60 to-white",
  },
  amber: {
    badge:
      "border-amber-200/80 bg-amber-50/90 text-amber-700 shadow-[0_10px_24px_rgba(245,158,11,0.12)]",
    value: "from-slate-900 via-amber-700 to-orange-500",
    activeLink: "border-amber-200 bg-amber-50 text-amber-800",
    glow: "from-amber-200/70 via-orange-100/60 to-white",
  },
};

export default function SubpageLayout({
  route,
  eyebrow,
  title,
  description,
  metrics,
  chips = [],
  accent = "blue",
  children,
  frameClassName,
}: SubpageLayoutProps) {
  const accentStyle = accentStyles[accent];
  const currentIndex = levelOnePages.findIndex((page) => page.path === route);
  const previousPage =
    currentIndex > 0 ? levelOnePages[currentIndex - 1] : undefined;
  const nextPage =
    currentIndex >= 0 && currentIndex < levelOnePages.length - 1
      ? levelOnePages[currentIndex + 1]
      : undefined;

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#f5f5f7] text-slate-900">
      <Navigation />

      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-1/2 top-0 h-[34rem] w-[34rem] -translate-x-1/2 rounded-full bg-white opacity-90 blur-3xl" />
        <div
          className={cn(
            "absolute -left-16 top-28 h-72 w-72 rounded-full blur-3xl",
            `bg-gradient-to-br ${accentStyle.glow}`,
          )}
        />
        <div className="absolute -right-20 top-40 h-96 w-96 rounded-full bg-slate-200/70 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-white/80 blur-3xl" />
        <div className="absolute inset-x-0 top-24 h-px bg-gradient-to-r from-transparent via-slate-300/50 to-transparent" />
      </div>

      <main className="relative z-10 px-4 pb-16 pt-24 sm:px-6 lg:px-8">
        <section className="mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            className="grid gap-6 rounded-[32px] border border-white/80 bg-white/72 p-6 shadow-[0_28px_90px_rgba(15,23,42,0.14)] backdrop-blur-2xl lg:grid-cols-[minmax(0,1.3fr)_22rem] lg:p-8"
          >
            <div>
              <div
                className={cn(
                  "inline-flex items-center gap-2 rounded-full border px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.24em]",
                  accentStyle.badge,
                )}
              >
                <Compass className="h-3.5 w-3.5" />
                {eyebrow}
              </div>

              <h1 className="mt-5 max-w-4xl text-4xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-5xl lg:text-6xl">
                {title}
              </h1>

              <p className="mt-5 max-w-3xl text-base leading-7 text-slate-600 sm:text-lg">
                {description}
              </p>

              {chips.length > 0 && (
                <div className="mt-6 flex flex-wrap gap-3">
                  {chips.map((chip) => (
                    <span
                      key={chip}
                      className="rounded-full border border-slate-200 bg-white/80 px-4 py-2 text-sm font-medium text-slate-600"
                    >
                      {chip}
                    </span>
                  ))}
                </div>
              )}

              <div className="mt-8 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {metrics.map((metric, index) => (
                  <motion.div
                    key={`${metric.label}-${metric.value}`}
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45, delay: 0.08 * index }}
                    className="rounded-[24px] border border-slate-200/80 bg-white/88 p-4 shadow-[0_14px_40px_rgba(15,23,42,0.08)]"
                  >
                    <div
                      className={cn(
                        "bg-gradient-to-r bg-clip-text text-3xl font-semibold tracking-[-0.04em] text-transparent",
                        accentStyle.value,
                      )}
                    >
                      {metric.value}
                    </div>
                    <div className="mt-2 text-sm font-semibold text-slate-800">
                      {metric.label}
                    </div>
                    {metric.detail && (
                      <p className="mt-2 text-sm leading-6 text-slate-500">
                        {metric.detail}
                      </p>
                    )}
                  </motion.div>
                ))}
              </div>
            </div>

            <motion.aside
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.15 }}
              className="rounded-[28px] border border-slate-200/90 bg-slate-50/86 p-5 shadow-[0_18px_44px_rgba(15,23,42,0.08)]"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                    Section Map
                  </div>
                  <h2 className="mt-2 text-xl font-semibold tracking-[-0.03em] text-slate-950">
                    Browse level-one pages
                  </h2>
                </div>
                <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-sm font-semibold text-slate-600">
                  {currentIndex + 1}/{levelOnePages.length}
                </div>
              </div>

              <div className="mt-5 grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
                {levelOnePages.map((page) => {
                  const isCurrent = page.path === route;

                  return (
                    <Link
                      key={page.path}
                      to={page.path}
                      className={cn(
                        "flex items-center justify-between rounded-2xl border px-4 py-3 text-sm font-medium transition-all duration-200",
                        isCurrent
                          ? accentStyle.activeLink
                          : "border-white bg-white/88 text-slate-600 hover:border-slate-200 hover:bg-white hover:text-slate-900",
                      )}
                    >
                      <span>{page.label}</span>
                      <ChevronRight className="h-4 w-4" />
                    </Link>
                  );
                })}
              </div>
            </motion.aside>
          </motion.div>
        </section>

        <section className="mx-auto mt-8 max-w-7xl">
          <div
            className={cn(
              "overflow-hidden rounded-[34px] border border-slate-900/8 bg-[linear-gradient(180deg,rgba(15,23,42,0.97),rgba(17,24,39,0.985))] shadow-[0_30px_100px_rgba(15,23,42,0.18)]",
              frameClassName,
            )}
          >
            {children}
          </div>
        </section>

        <section className="mx-auto mt-8 max-w-7xl">
          <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
            {previousPage ? (
              <Link
                to={previousPage.path}
                className="group rounded-[28px] border border-white/80 bg-white/72 p-5 shadow-[0_20px_50px_rgba(15,23,42,0.1)] backdrop-blur-xl transition-all duration-200 hover:-translate-y-0.5 hover:bg-white/90"
              >
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-500">
                  <ArrowLeft className="h-4 w-4" />
                  Previous
                </div>
                <div className="mt-3 text-lg font-semibold tracking-[-0.03em] text-slate-950">
                  {previousPage.label}
                </div>
              </Link>
            ) : (
              <div className="hidden md:block" />
            )}

            <Link
              to="/"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 shadow-[0_14px_36px_rgba(15,23,42,0.08)] transition-all duration-200 hover:-translate-y-0.5 hover:text-slate-950"
            >
              <Home className="h-4 w-4" />
              Back to home
            </Link>

            {nextPage ? (
              <Link
                to={nextPage.path}
                className="group rounded-[28px] border border-white/80 bg-white/72 p-5 text-right shadow-[0_20px_50px_rgba(15,23,42,0.1)] backdrop-blur-xl transition-all duration-200 hover:-translate-y-0.5 hover:bg-white/90"
              >
                <div className="flex items-center justify-end gap-2 text-sm font-semibold text-slate-500">
                  Next
                  <ArrowRight className="h-4 w-4" />
                </div>
                <div className="mt-3 text-lg font-semibold tracking-[-0.03em] text-slate-950">
                  {nextPage.label}
                </div>
              </Link>
            ) : (
              <div className="hidden md:block" />
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
