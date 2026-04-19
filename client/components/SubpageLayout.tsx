import type { ReactNode } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
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
  { path: "/ai-tools", label: "AI Tools" },
  { path: "/ai-projects", label: "AI Projects" },
  { path: "/prompt-engineering", label: "Prompt Engineering" },
  { path: "/ai-agent-training", label: "Agent Training" },
  { path: "/ai-companies", label: "AI Companies" },
  { path: "/ai-discoveries", label: "Discoveries" },
  { path: "/ai-champions", label: "Champions" },
  { path: "/games", label: "Games" },
  { path: "/resume-builder", label: "Resume Builder" },
];

const accentStyles: Record<
  AccentTone,
  { badge: string; value: string; activeLink: string }
> = {
  blue: {
    badge: "border-sky-200/80 bg-sky-50/90 text-sky-700",
    value: "text-sky-600",
    activeLink: "border-sky-200 bg-sky-50 text-sky-800",
  },
  emerald: {
    badge: "border-emerald-200/80 bg-emerald-50/90 text-emerald-700",
    value: "text-emerald-600",
    activeLink: "border-emerald-200 bg-emerald-50 text-emerald-800",
  },
  rose: {
    badge: "border-rose-200/80 bg-rose-50/90 text-rose-700",
    value: "text-rose-600",
    activeLink: "border-rose-200 bg-rose-50 text-rose-800",
  },
  amber: {
    badge: "border-amber-200/80 bg-amber-50/90 text-amber-700",
    value: "text-amber-600",
    activeLink: "border-amber-200 bg-amber-50 text-amber-800",
  },
};

export default function SubpageLayout({
  route,
  eyebrow,
  title,
  description,
  metrics,
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
    <div className="relative min-h-screen bg-slate-100 text-slate-900">
      <Navigation />

      <main className="relative z-10 px-4 pb-12 pt-20 sm:px-6 lg:px-8 lg:pt-32">
        <section className="mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="rounded-2xl border border-white/80 bg-white/70 px-5 py-4 shadow-[0_12px_40px_rgba(15,23,42,0.1)] backdrop-blur-xl sm:px-6"
          >
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="min-w-0">
                <div className="flex items-center gap-3">
                  <span
                    className={cn(
                      "shrink-0 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em]",
                      accentStyle.badge,
                    )}
                  >
                    {eyebrow}
                  </span>
                  <div className="hidden items-center gap-3 text-xs text-slate-400 sm:flex">
                    {metrics.slice(0, 4).map((m) => (
                      <span key={m.label}>
                        <span className={cn("font-bold", accentStyle.value)}>
                          {m.value}
                        </span>{" "}
                        {m.label}
                      </span>
                    ))}
                  </div>
                </div>
                <h1 className="mt-2 text-xl font-semibold tracking-tight text-slate-950 sm:text-2xl lg:text-3xl">
                  {title}
                </h1>
                <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-500 line-clamp-1 sm:line-clamp-none">
                  {description}
                </p>
              </div>
            </div>

            <div className="mt-3 flex gap-1.5 overflow-x-auto">
              {levelOnePages.map((page) => {
                const isCurrent = page.path === route;
                return (
                  <Link
                    key={page.path}
                    to={page.path}
                    className={cn(
                      "shrink-0 rounded-full border px-3 py-1 text-[11px] font-medium transition-all duration-150",
                      isCurrent
                        ? accentStyle.activeLink
                        : "border-transparent text-slate-400 hover:bg-slate-100 hover:text-slate-700",
                    )}
                  >
                    {page.label}
                  </Link>
                );
              })}
            </div>
          </motion.div>
        </section>

        <section className="mx-auto mt-3 max-w-7xl">
          <div
            className={cn(
              "overflow-hidden rounded-2xl border border-slate-700/40 bg-slate-800 shadow-[0_8px_30px_rgba(15,23,42,0.12)]",
              frameClassName,
            )}
          >
            {children}
          </div>
        </section>

        <section className="mx-auto mt-4 max-w-7xl">
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
            {previousPage ? (
              <Link
                to={previousPage.path}
                className="group rounded-2xl border border-white/80 bg-white/70 p-3 shadow-sm backdrop-blur-xl transition-all hover:-translate-y-0.5 hover:bg-white/90"
              >
                <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
                  <ArrowLeft className="h-3.5 w-3.5" />
                  Previous
                </div>
                <div className="mt-1 text-sm font-semibold text-slate-950">
                  {previousPage.label}
                </div>
              </Link>
            ) : (
              <div className="hidden md:block" />
            )}

            <Link
              to="/"
              className="inline-flex items-center justify-center gap-1.5 rounded-full border border-slate-200 bg-white px-4 py-2.5 text-xs font-semibold text-slate-600 shadow-sm transition-all hover:-translate-y-0.5 hover:text-slate-950"
            >
              <Home className="h-3.5 w-3.5" />
              Home
            </Link>

            {nextPage ? (
              <Link
                to={nextPage.path}
                className="group rounded-2xl border border-white/80 bg-white/70 p-3 text-right shadow-sm backdrop-blur-xl transition-all hover:-translate-y-0.5 hover:bg-white/90"
              >
                <div className="flex items-center justify-end gap-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
                  Next
                  <ArrowRight className="h-3.5 w-3.5" />
                </div>
                <div className="mt-1 text-sm font-semibold text-slate-950">
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
