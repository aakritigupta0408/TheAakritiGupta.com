import type { ReactNode } from "react";
import {
  ArrowLeft,
  ArrowRight,
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
  { path: "/ai-playground", label: "Playground" },
  { path: "/ai-tools", label: "Tools" },
  { path: "/ai-projects", label: "Projects" },
  { path: "/prompt-engineering", label: "Prompting" },
  { path: "/ai-agent-training", label: "Agents" },
  { path: "/ai-companies", label: "Companies" },
  { path: "/ai-discoveries", label: "Discoveries" },
  { path: "/ai-champions", label: "Champions" },
  { path: "/games", label: "Games" },
  { path: "/resume-builder", label: "Resume" },
];

const accentColors: Record<AccentTone, { badge: string; stat: string; active: string }> = {
  blue: {
    badge: "bg-sky-100 text-sky-800",
    stat: "text-sky-600",
    active: "bg-sky-100 text-sky-900 border-sky-300",
  },
  emerald: {
    badge: "bg-emerald-100 text-emerald-800",
    stat: "text-emerald-600",
    active: "bg-emerald-100 text-emerald-900 border-emerald-300",
  },
  rose: {
    badge: "bg-rose-100 text-rose-800",
    stat: "text-rose-600",
    active: "bg-rose-100 text-rose-900 border-rose-300",
  },
  amber: {
    badge: "bg-amber-100 text-amber-800",
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
    <div className="flex h-screen flex-col overflow-hidden bg-[#f5f5f7] text-slate-900">
      <Navigation />

      {/* Spacer for fixed nav */}
      <div className="shrink-0 h-16 lg:h-[7rem]" aria-hidden="true" />

      {/* Header bar */}
      <header className="shrink-0 border-b border-slate-200 bg-white px-4 py-2 sm:px-6">
        <div className="mx-auto flex max-w-7xl flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            <span
              className={cn(
                "shrink-0 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest",
                colors.badge,
              )}
            >
              {eyebrow}
            </span>
            <h1 className="truncate text-sm font-semibold text-slate-900 sm:text-base">
              {title}
            </h1>
            <div className="hidden items-center gap-2 text-[10px] text-slate-400 lg:flex">
              {metrics.slice(0, 3).map((m) => (
                <span key={m.label}>
                  <span className={cn("font-bold", colors.stat)}>{m.value}</span>{" "}
                  {m.label}
                </span>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-1 overflow-x-auto">
            {previousPage && (
              <Link
                to={previousPage.path}
                className="shrink-0 rounded-full border border-slate-200 px-2 py-0.5 text-[10px] text-slate-500 hover:bg-slate-50"
              >
                <ArrowLeft className="inline h-3 w-3" /> {previousPage.label}
              </Link>
            )}
            {levelOnePages.map((page) => {
              const isCurrent = page.path === route;
              return (
                <Link
                  key={page.path}
                  to={page.path}
                  aria-current={isCurrent ? "page" : undefined}
                  className={cn(
                    "shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors",
                    isCurrent
                      ? cn("border", colors.active)
                      : "text-slate-400 hover:text-slate-700",
                  )}
                >
                  {page.label}
                </Link>
              );
            })}
            {nextPage && (
              <Link
                to={nextPage.path}
                className="shrink-0 rounded-full border border-slate-200 px-2 py-0.5 text-[10px] text-slate-500 hover:bg-slate-50"
              >
                {nextPage.label} <ArrowRight className="inline h-3 w-3" />
              </Link>
            )}
          </div>
        </div>
      </header>

      {/* Content area — fills remaining height, scrolls internally */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 sm:px-6">
        <div
          className={cn(
            "mx-auto max-w-7xl rounded-xl bg-slate-800",
            frameClassName,
          )}
        >
          {children}
        </div>
      </div>
    </div>
  );
}
