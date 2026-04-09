import { motion } from "framer-motion";
import { ArrowLeft, ArrowRight, Compass, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

import ChatBot from "@/components/ChatBot";
import Navigation from "@/components/Navigation";
import {
  getTalentProfile,
  talentProfiles,
  type TalentRoute,
} from "@/data/talentProfiles";
import { cn } from "@/lib/utils";

interface TalentProfilePageProps {
  route: TalentRoute;
}

export default function TalentProfilePage({
  route,
}: TalentProfilePageProps) {
  const profile = getTalentProfile(route);
  const siblingProfiles = talentProfiles.filter((item) => item.route !== route);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#f5f5f7] text-slate-900">
      <Navigation />

      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-1/2 top-0 h-[34rem] w-[34rem] -translate-x-1/2 rounded-full bg-white opacity-90 blur-3xl" />
        <div
          className={cn(
            "absolute -left-12 top-32 h-72 w-72 rounded-full bg-gradient-to-br blur-3xl",
            profile.accent.glow,
          )}
        />
        <div className="absolute -right-16 top-28 h-80 w-80 rounded-full bg-slate-200/60 blur-3xl" />
        <div className="absolute bottom-10 left-1/3 h-80 w-80 rounded-full bg-white/70 blur-3xl" />
      </div>

      <main className="relative z-10 px-4 pb-16 pt-24 sm:px-6 lg:px-8">
        <section className="mx-auto max-w-7xl">
          <div className="grid gap-6 rounded-[34px] border border-white/80 bg-white/78 p-6 shadow-[0_28px_90px_rgba(15,23,42,0.12)] backdrop-blur-2xl lg:grid-cols-[minmax(0,1.25fr)_22rem] lg:p-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <Link
                to="/"
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-4 py-2 text-sm font-medium text-slate-600 transition-colors hover:text-slate-900"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to home
              </Link>

              <div
                className={cn(
                  "mt-6 inline-flex items-center gap-2 rounded-full border px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.24em]",
                  profile.accent.softBg,
                  profile.accent.softBorder,
                  profile.accent.text,
                )}
              >
                <Compass className="h-3.5 w-3.5" />
                {profile.eyebrow}
              </div>

              <div className="mt-5 flex items-start gap-4">
                <div
                  className={cn(
                    "flex h-16 w-16 shrink-0 items-center justify-center rounded-[22px] border text-3xl shadow-[0_18px_44px_rgba(15,23,42,0.08)]",
                    profile.accent.softBg,
                    profile.accent.softBorder,
                  )}
                >
                  {profile.symbol}
                </div>
                <div>
                  <h1 className="text-4xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-5xl lg:text-6xl">
                    {profile.role}
                  </h1>
                  <p className="mt-4 max-w-3xl text-base leading-7 text-slate-600 sm:text-lg">
                    {profile.headline}
                  </p>
                </div>
              </div>

              <p className="mt-6 max-w-3xl text-sm leading-7 text-slate-500 sm:text-base">
                {profile.summary}
              </p>

              <div className="mt-8 grid gap-3 sm:grid-cols-3">
                {profile.metrics.map((metric, index) => (
                  <motion.div
                    key={`${metric.label}-${metric.value}`}
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45, delay: 0.08 * index }}
                    className="rounded-[24px] border border-slate-200/80 bg-white/90 p-4 shadow-[0_14px_34px_rgba(15,23,42,0.06)]"
                  >
                    <div
                      className={cn(
                        "text-3xl font-semibold tracking-[-0.04em]",
                        profile.accent.text,
                      )}
                    >
                      {metric.value}
                    </div>
                    <div className="mt-2 text-sm font-semibold text-slate-800">
                      {metric.label}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            <motion.aside
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.15 }}
              className="rounded-[28px] border border-slate-200/90 bg-slate-50/88 p-5 shadow-[0_18px_44px_rgba(15,23,42,0.08)]"
            >
              <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                Why this page exists
              </div>
              <h2 className="mt-2 text-xl font-semibold tracking-[-0.03em] text-slate-950">
                Beyond the resume
              </h2>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                These talent pages translate non-obvious strengths into product,
                engineering, and leadership signals that would otherwise stay
                invisible in a traditional profile.
              </p>

              <div className="mt-5 grid gap-3">
                <div className="rounded-2xl border border-slate-200 bg-white/90 p-4">
                  <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                    Reading mode
                  </div>
                  <p className="mt-2 text-sm leading-6 text-slate-600">
                    Skim the focus areas first, then use the translation cards
                    to see how the discipline maps into work.
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white/90 p-4">
                  <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                    Tone
                  </div>
                  <p className="mt-2 text-sm leading-6 text-slate-600">
                    Lighter, calmer, and more structured than the older talent
                    pages while keeping the premium aesthetic intact.
                  </p>
                </div>
              </div>
            </motion.aside>
          </div>
        </section>

        <section className="mx-auto mt-8 max-w-7xl">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
            <div className="rounded-[32px] border border-slate-200/80 bg-white/80 p-6 shadow-[0_18px_54px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:p-8">
              <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                <Sparkles className="h-3.5 w-3.5" />
                Focus areas
              </div>
              <h2 className="mt-4 text-3xl font-semibold tracking-[-0.04em] text-slate-950">
                What this discipline builds
              </h2>
              <div className="mt-6 grid gap-4">
                {profile.focusAreas.map((item, index) => (
                  <motion.div
                    key={item.title}
                    initial={{ opacity: 0, y: 18 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.45, delay: index * 0.08 }}
                    className={cn(
                      "rounded-[24px] border bg-gradient-to-r p-5",
                      profile.accent.softBorder,
                      profile.accent.surface,
                    )}
                  >
                    <h3 className="text-lg font-semibold text-slate-900">
                      {item.title}
                    </h3>
                    <p className="mt-2 text-sm leading-7 text-slate-600">
                      {item.description}
                    </p>
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="rounded-[32px] border border-slate-200/80 bg-slate-950 p-6 text-white shadow-[0_20px_60px_rgba(15,23,42,0.14)] sm:p-8">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">
                Professional translation
              </div>
              <h2 className="mt-4 text-3xl font-semibold tracking-[-0.04em] text-white">
                How it changes the work
              </h2>
              <div className="mt-6 space-y-4">
                {profile.professionalTranslation.map((item, index) => (
                  <motion.div
                    key={item.title}
                    initial={{ opacity: 0, y: 18 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.45, delay: index * 0.08 }}
                    className="rounded-[24px] border border-white/10 bg-white/5 p-5"
                  >
                    <h3 className="text-lg font-semibold text-white">
                      {item.title}
                    </h3>
                    <p className="mt-2 text-sm leading-7 text-slate-300">
                      {item.description}
                    </p>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto mt-8 max-w-7xl">
          <div className="rounded-[32px] border border-slate-200/80 bg-white/82 p-6 shadow-[0_16px_46px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:p-8">
            <div className="max-w-4xl">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                Personal principle
              </div>
              <blockquote className="mt-4 text-2xl font-semibold tracking-[-0.03em] text-slate-950 sm:text-3xl">
                “{profile.quote}”
              </blockquote>
              <p className={cn("mt-4 text-sm font-semibold", profile.accent.text)}>
                {profile.quoteCredit}
              </p>
            </div>
          </div>
        </section>

        <section className="mx-auto mt-8 max-w-7xl">
          <div className="rounded-[32px] border border-slate-200/80 bg-white/82 p-6 shadow-[0_16px_46px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:p-8">
            <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                  Explore more
                </div>
                <h2 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-slate-950">
                  Other lenses on the same profile
                </h2>
              </div>
              <p className="max-w-2xl text-sm leading-6 text-slate-500">
                These pages are meant to feel like connected perspectives, not
                isolated microsites.
              </p>
            </div>

            <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {siblingProfiles.map((item) => (
                <Link
                  key={item.route}
                  to={item.route}
                  className="group flex items-center justify-between rounded-[24px] border border-slate-200 bg-slate-50/90 px-5 py-4 text-sm font-medium text-slate-700 transition-all duration-200 hover:border-slate-300 hover:bg-white"
                >
                  <div>
                    <div className="font-semibold text-slate-900">{item.role}</div>
                    <div className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-500">
                      {item.eyebrow}
                    </div>
                  </div>
                  <ArrowRight className="h-4 w-4 text-slate-400 transition-transform duration-200 group-hover:translate-x-1" />
                </Link>
              ))}
            </div>
          </div>
        </section>
      </main>

      <ChatBot />
    </div>
  );
}
