import type { ReactNode } from "react";
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

const accentColors: Record<AccentTone, { badge: string; stat: string; active: string }> = {
  blue: {
    badge: "bg-sky-100 text-sky-800 border-sky-200",
    stat: "text-sky-600",
    active: "bg-sky-100 text-sky-900 border-sky-300",
  },
  emerald: {
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
    stat: "text-emerald-600",
    active: "bg-emerald-100 text-emerald-900 border-emerald-300",
  },
  rose: {
    badge: "bg-rose-100 text-rose-800 border-rose-200",
    stat: "text-rose-600",
    active: "bg-rose-100 text-rose-900 border-rose-300",
  },
  amber: {
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    stat: "text-amber-600",
    active: "bg-amber-100 text-amber-900 border-amber-300",
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
  const colors = accentColors[accent];
  const currentIndex = levelOnePages.findIndex((page) => page.path === route);
  const previousPage =
    currentIndex > 0 ? levelOnePages[currentIndex - 1] : undefined;
  const nextPage =
    currentIndex >= 0 && currentIndex < levelOnePages.length - 1
      ? levelOnePages[currentIndex + 1]
      : undefined;

  return (
    <div className="relative min-h-screen bg-white text-slate-900">
      <Navigation />

      {/* Spacer to clear the fixed nav — 7rem mobile, 9rem desktop */}
      <div className="h-28 lg:h-36" aria-hidden="true" />

      <header className="border-b border-slate-200 bg-white px-4 pb-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span
              className={cn(
                "rounded-full border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-widest",
                colors.badge,
              )}
            >
              {eyebrow}
            </span>
            {metrics.slice(0, 4).map((m) => (
              <span key={m.label} className="text-[11px] text-slate-400">
                <span className={cn("font-bold", colors.stat)}>{m.value}</span>{" "}
                {m.label}
              </span>
            ))}
          </div>

          <h1 className="mt-2 text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
            {title}
          </h1>

          <p className="mt-1 text-sm text-slate-500">{description}</p>

          <nav className="mt-3 flex gap-1 overflow-x-auto pb-0.5" aria-label="Page navigation">
            {levelOnePages.map((page) => {
              const isCurrent = page.path === route;
              return (
                <Link
                  key={page.path}
                  to={page.path}
                  aria-current={isCurrent ? "page" : undefined}
                  className={cn(
                    "shrink-0 rounded-full border px-3 py-1 text-[11px] font-medium transition-colors",
                    isCurrent
                      ? colors.active
                      : "border-transparent text-slate-400 hover:bg-slate-100 hover:text-slate-700",
                  )}
                >
                  {page.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="px-4 pb-10 pt-4 sm:px-6 lg:px-8">
        <section className="mx-auto max-w-7xl">
          <div
            className={cn(
              "overflow-hidden rounded-2xl bg-slate-800",
              frameClassName,
            )}
          >
            {children}
          </div>
        </section>

        <section className="mx-auto mt-4 max-w-7xl">
          <div className="flex items-center justify-between">
            {previousPage ? (
              <Link
                to={previousPage.path}
                className="flex items-center gap-1.5 rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
              >
                <ArrowLeft className="h-3.5 w-3.5" />
                {previousPage.label}
              </Link>
            ) : (
              <div />
            )}

            <Link
              to="/"
              className="flex items-center gap-1.5 text-xs font-medium text-slate-400 transition-colors hover:text-slate-700"
            >
              <Home className="h-3.5 w-3.5" />
              Home
            </Link>

            {nextPage ? (
              <Link
                to={nextPage.path}
                className="flex items-center gap-1.5 rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
              >
                {nextPage.label}
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            ) : (
              <div />
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
