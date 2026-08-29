import { type ReactNode, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";

import Navigation from "@/components/Navigation";

// RENDER_URL is injected at build time via Vite env.
// Defaults to the Render service (see render.yaml: vedic-astro-ai).
const DEMO_URL =
  import.meta.env.VITE_VEDIC_ASTRO_URL ??
  "https://vedic-astro-ai-4t2k.onrender.com";

type ViewMode = "explore" | "engineer";

/* ---------------------------------------------------------------- data --- */

interface Passport {
  name: string;
  icon: string;
  purpose: string;
  explorePurpose: string;
  runtime: string;
  inputs: string;
  outputs: string;
  p95: string;
  availability: string;
  cost: string;
  quality: string;
  failureMode: string;
}

interface Continent {
  key: string;
  title: string;
  tagline: string;
  components: Passport[];
}

const continents: Continent[] = [
  {
    key: "experience",
    title: "Experience",
    tagline: "What a visitor touches",
    components: [
      {
        name: "Birth Chart City",
        icon: "🏙️",
        purpose:
          "Onboarding, birth-time handling, and the D1–D60 chart renderer.",
        explorePurpose: "Where your chart is drawn — all sixteen of them.",
        runtime: "React + SVG renderer",
        inputs: "Birth date, time, place",
        outputs: "16 divisional charts",
        p95: "120ms render",
        availability: "static — always up",
        cost: "$0 / chart",
        quality: "positions verified against Swiss Ephemeris",
        failureMode: "unknown birth time → falls back to chandra lagna",
      },
      {
        name: "Prediction Harbor",
        icon: "⚓",
        purpose:
          "Serves dasha-arc, varshaphala, monthly and daily readings with citations.",
        explorePurpose: "Where predictions arrive, from life-arc to today.",
        runtime: "FastAPI on Render",
        inputs: "Computed chart + question scope",
        outputs: "Cited reading, 4 time horizons",
        p95: "1.9s (LLM-bound)",
        availability: "free tier — sleeps after 15 min idle",
        cost: "≈$0.02 / reading",
        quality: "every sentence must cite a verse or chart fact",
        failureMode: "uncited sentence → deleted, visible in trace",
      },
      {
        name: "Conversation Village",
        icon: "💬",
        purpose:
          "Prashna: a direct question judged from the chart of the moment it is asked.",
        explorePurpose: "Ask a question; the sky at that instant answers.",
        runtime: "Claude + prashna procedure",
        inputs: "A question + the moment's chart",
        outputs: "Judgement with reasoning trace",
        p95: "2.4s",
        availability: "backed by Prediction Harbor",
        cost: "≈$0.02 / question",
        quality: "trace shows the full judgement path",
        failureMode: "ambiguous question → asks for clarification",
      },
    ],
  },
  {
    key: "intelligence",
    title: "Intelligence",
    tagline: "Where readings are made",
    components: [
      {
        name: "Ganita Engine",
        icon: "🧮",
        purpose:
          "Deterministic astronomy: Swiss Ephemeris positions, divisional charts, Vimshottari dasha. No model ever computes a position.",
        explorePurpose:
          "The mathematics mountain — planets computed, never guessed.",
        runtime: "Python + Swiss Ephemeris",
        inputs: "UTC instant + coordinates",
        outputs: "Longitudes, houses, dasha tree",
        p95: "31ms",
        availability: "99.99% (pure computation)",
        cost: "≈$0.0001 / chart",
        quality: "cross-checked against published ephemerides",
        failureMode: "pre-600 CE dates → rejected, outside ephemeris range",
      },
      {
        name: "Rules Mountain",
        icon: "⛰️",
        purpose:
          "Classical yogas, aspects, house significations encoded as a deterministic rule engine.",
        explorePurpose: "The old rules, written down precisely.",
        runtime: "Typed rule engine",
        inputs: "Chart facts",
        outputs: "Fired rules with verse citations",
        p95: "8ms",
        availability: "99.99%",
        cost: "≈$0 / evaluation",
        quality: "each rule carries its classical source verse",
        failureMode: "conflicting rules → both surfaced, never merged silently",
      },
      {
        name: "LLM City",
        icon: "🏛️",
        purpose:
          "Claude ranks retrieved classical verses against the computed chart, synthesizes the reading, and must cite every sentence.",
        explorePurpose: "The interpreter — reads the computed sky aloud.",
        runtime: "Claude (Anthropic API)",
        inputs: "Chart facts + retrieved verses + question",
        outputs: "Cited synthesis",
        p95: "1.8s",
        availability: "API-bound",
        cost: "≈$0.02 / synthesis",
        quality: "uncited claims deleted by the citation gate",
        failureMode: "gate strips too much → reading marked low-confidence",
      },
      {
        name: "Citation Gate",
        icon: "🚧",
        purpose:
          "Safety and faithfulness layer: any sentence that cannot point to a verse or a chart fact is removed, and the removal is logged in the visible trace.",
        explorePurpose: "The honesty checkpoint every sentence passes through.",
        runtime: "Deterministic post-processor",
        inputs: "Draft reading",
        outputs: "Verified reading + deletion log",
        p95: "40ms",
        availability: "99.99%",
        cost: "≈$0 / pass",
        quality: "0 uncited sentences shipped",
        failureMode: "over-strict match → sentence lost; logged for review",
      },
    ],
  },
  {
    key: "data",
    title: "Data",
    tagline: "What the system knows",
    components: [
      {
        name: "Ephemeris Observatory",
        icon: "🔭",
        purpose: "Swiss Ephemeris files — the astronomical source of truth.",
        explorePurpose: "Where planetary positions come from.",
        runtime: "Bundled ephemeris data",
        inputs: "—",
        outputs: "Planetary state, any instant",
        p95: "in-memory",
        availability: "99.99%",
        cost: "$0 (bundled)",
        quality: "arc-second agreement with JPL",
        failureMode: "corrupt file → startup check fails loudly",
      },
      {
        name: "Knowledge Library",
        icon: "📚",
        purpose:
          "Classical verse corpus (BPHS and related texts) indexed for retrieval, each verse tagged with its source.",
        explorePurpose: "The shelf of old books the interpreter may quote.",
        runtime: "Embedded corpus + retrieval",
        inputs: "Chart facts + question",
        outputs: "Ranked candidate verses",
        p95: "60ms",
        availability: "99.99%",
        cost: "≈$0.001 / retrieval",
        quality: "every verse traceable to text + chapter",
        failureMode: "no relevant verse → reading says so explicitly",
      },
      {
        name: "Trace Store",
        icon: "🧾",
        purpose:
          "Every reading's full computation trace — what fired, what was retrieved, what was deleted — kept for the Explanation tab.",
        explorePurpose: "The diary of how each reading was made.",
        runtime: "Structured logs",
        inputs: "Pipeline events",
        outputs: "Explanation tab content",
        p95: "async",
        availability: "99.9%",
        cost: "≈$0.0002 / reading",
        quality: "trace replays reproduce the reading",
        failureMode: "trace gap → reading flagged unexplainable",
      },
    ],
  },
  {
    key: "operations",
    title: "Operations",
    tagline: "How it stays honest",
    components: [
      {
        name: "Evaluation Warehouse",
        icon: "🏗️",
        purpose:
          "Golden charts with known classical judgements; every engine change re-runs the suite before deploy.",
        explorePurpose: "The exam the system must pass before changing.",
        runtime: "pytest suite + golden set",
        inputs: "Candidate build",
        outputs: "Pass / fail per golden chart",
        p95: "4 min full suite",
        availability: "CI-bound",
        cost: "≈$0.40 / full run",
        quality: "positions exact; readings faithfulness-scored",
        failureMode: "regression → deploy blocked",
      },
      {
        name: "Policy Gate",
        icon: "🛡️",
        purpose:
          "Production REFUSE/ABSTAIN gate: medical, legal, and death questions are refused; low-evidence readings abstain rather than guess.",
        explorePurpose: "The judgement about when not to answer.",
        runtime: "ENVIRONMENT=production flag",
        inputs: "Question + reading confidence",
        outputs: "Answer, abstain, or refusal",
        p95: "5ms",
        availability: "99.99%",
        cost: "$0",
        quality: "refusal classes unit-tested",
        failureMode: "over-refusal → logged for prompt review",
      },
      {
        name: "Deployment Harbor",
        icon: "🚢",
        purpose:
          "Render deploy from the jyotisha repo's deploy branch; free tier sleeps idle, wakes in ~50s.",
        explorePurpose: "Where new versions come ashore.",
        runtime: "Render (Docker)",
        inputs: "deploy branch push",
        outputs: "Live service",
        p95: "cold start ≈50s",
        availability: "free tier",
        cost: "$0/mo (free) — $7/mo always-on",
        quality: "health-checked on /",
        failureMode: "failed health check → previous build stays live",
      },
    ],
  },
];

interface ReplayStep {
  label: string;
  detail: string;
  ms: string;
}

const replaySteps: ReplayStep[] = [
  { label: "Request arrives", detail: "birth data + question", ms: "0ms" },
  { label: "Inputs validated", detail: "date range, coordinates", ms: "2ms" },
  {
    label: "Astronomy computed",
    detail: "Swiss Ephemeris, 9 grahas",
    ms: "31ms",
  },
  { label: "Divisional charts cast", detail: "D1–D60", ms: "44ms" },
  { label: "Dasha tree built", detail: "Vimshottari, 3 levels", ms: "51ms" },
  { label: "Rules fired", detail: "17 rules matched, each cited", ms: "59ms" },
  {
    label: "Verses retrieved",
    detail: "24 candidates from corpus",
    ms: "119ms",
  },
  {
    label: "Claude ranks & writes",
    detail: "verses vs. chart facts",
    ms: "1.84s",
  },
  {
    label: "Citation gate",
    detail: "2 uncited sentences deleted",
    ms: "1.88s",
  },
  { label: "Policy gate", detail: "scope check passed", ms: "1.88s" },
  { label: "Trace recorded", detail: "for the Explanation tab", ms: "async" },
  { label: "Reading returned", detail: "every sentence cited", ms: "1.9s" },
];

const agents = [
  {
    icon: "🤖",
    name: "Evaluation Agent",
    status: "working",
    detail: "re-running golden charts on engine change",
  },
  {
    icon: "🛰️",
    name: "Ephemeris Check",
    status: "watching",
    detail: "startup integrity check on ephemeris files",
  },
  {
    icon: "🧾",
    name: "Citation Auditor",
    status: "watching",
    detail: "sampling readings for uncited claims",
  },
  {
    icon: "🛡️",
    name: "Policy Agent",
    status: "watching",
    detail: "refusal classes on medical/legal/death scope",
  },
  {
    icon: "💰",
    name: "Cost Agent",
    status: "idle",
    detail: "token usage per reading within budget",
  },
  {
    icon: "🚢",
    name: "Deploy Agent",
    status: "idle",
    detail: "health-check gate on the deploy branch",
  },
];

/* ------------------------------------------------------------ helpers --- */

function MonoLabel({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
      {children}
    </p>
  );
}

function SectionHead({
  index,
  title,
  sub,
}: {
  index: string;
  title: string;
  sub: string;
}) {
  return (
    <div className="mb-6">
      <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.2em] text-[#A15C22]">
        {index}
      </p>
      <h2 className="mt-1 font-serif text-2xl font-bold text-slate-900 sm:text-3xl">
        {title}
      </h2>
      <p className="mt-1 max-w-2xl text-sm text-slate-500">{sub}</p>
    </div>
  );
}

/* --------------------------------------------------------------- page --- */

export default function VedicAstroDemo() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<ViewMode>("explore");
  const [whyLayer, setWhyLayer] = useState(0);
  const [passport, setPassport] = useState<Passport | null>(null);
  const [replayCount, setReplayCount] = useState(0);
  const replayTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const eng = mode === "engineer";
  const m = (explore: string, engineer: string) => (eng ? engineer : explore);

  const startReplay = () => {
    if (replayTimer.current) clearInterval(replayTimer.current);
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (reduced) {
      setReplayCount(replaySteps.length);
      return;
    }
    setReplayCount(0);
    replayTimer.current = setInterval(() => {
      setReplayCount((current) => {
        if (current >= replaySteps.length) {
          if (replayTimer.current) clearInterval(replayTimer.current);
          return current;
        }
        return current + 1;
      });
    }, 260);
  };

  useEffect(
    () => () => {
      if (replayTimer.current) clearInterval(replayTimer.current);
    },
    [],
  );

  return (
    <div className="min-h-screen bg-[#f5f5f7] text-slate-900">
      <Navigation />

      {/* ------------------------------------------------ 01 · Universe --- */}
      <section className="relative overflow-hidden bg-[#0b1020] pb-16 pt-32 text-white">
        {/* orbital background */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
        >
          {[420, 640, 880].map((size) => (
            <span
              key={size}
              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/5"
              style={{ width: size, height: size }}
            />
          ))}
          <span className="absolute left-[18%] top-[30%] h-1 w-1 rounded-full bg-amber-200/80" />
          <span className="absolute left-[72%] top-[22%] h-1.5 w-1.5 rounded-full bg-sky-200/70" />
          <span className="absolute left-[62%] top-[68%] h-1 w-1 rounded-full bg-rose-200/70" />
          <span className="absolute left-[35%] top-[75%] h-1 w-1 rounded-full bg-white/60" />
        </div>

        <div className="relative z-10 mx-auto max-w-6xl px-6">
          <button
            onClick={() => navigate("/ai-playground")}
            className="mb-10 flex items-center gap-2 text-sm text-slate-400 transition-colors hover:text-white"
          >
            <ArrowLeft size={16} />
            Back to AI Playground
          </button>

          <div className="flex flex-wrap items-start justify-between gap-6">
            <div className="max-w-2xl">
              <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.24em] text-amber-300/90">
                Vedic Astro AI
              </p>
              <h1 className="mt-2 font-serif text-4xl font-bold leading-tight text-white sm:text-5xl">
                A computational universe for understanding time.
              </h1>
              <p className="mt-4 text-base leading-relaxed text-slate-300">
                {m(
                  "Explore how astronomical state, classical Vedic systems, structured computation and AI combine into readings you can question — and see answered.",
                  "Deterministic ephemeris computation feeding a cited-retrieval LLM pipeline behind a citation gate and a REFUSE/ABSTAIN policy layer. Every output is interrogable down to the verse.",
                )}
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <a
                  href={DEMO_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-lg bg-amber-400 px-5 py-2.5 text-sm font-bold text-slate-900 transition hover:bg-amber-300"
                >
                  Enter My Universe →
                </a>
                {/* HashRouter owns the URL hash, so a plain #system anchor
                    would navigate; scroll programmatically instead. */}
                <button
                  onClick={() => {
                    const target = document.getElementById("system");
                    if (target) {
                      window.scrollTo(
                        0,
                        target.getBoundingClientRect().top +
                          window.scrollY -
                          56,
                      );
                    }
                  }}
                  className="rounded-lg border border-white/25 px-5 py-2.5 text-sm font-semibold text-white transition hover:border-white/50"
                >
                  Explore the System ↓
                </button>
              </div>
              <p className="mt-5 font-mono text-[11px] tracking-[0.06em] text-slate-400">
                <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 align-middle" />
                System online · 1 deterministic engine · 16 divisional charts ·
                every sentence cited
              </p>
            </div>

            {/* live signal card */}
            <div className="w-full max-w-sm rounded-xl border border-white/15 bg-white/[0.06] p-5 backdrop-blur">
              <MonoLabel>current signal · sample chart</MonoLabel>
              <div className="mt-2 flex items-baseline justify-between">
                <h3 className="font-serif text-lg font-bold text-white">
                  Career Momentum
                </h3>
                <span className="font-mono text-2xl font-bold text-amber-300">
                  ↑ 74<span className="text-sm text-slate-400">/100</span>
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-300">
                Saturn transit · Jupiter influence · dasha transition · 6 more
                signals
              </p>
              <div className="mt-3 flex items-center justify-between">
                <span className="font-mono text-[11px] text-slate-400">
                  confidence 81%
                </span>
                <button
                  onClick={() => setWhyLayer(whyLayer === 0 ? 1 : 0)}
                  className="rounded-md border border-amber-300/50 px-3 py-1 text-xs font-bold text-amber-300 transition hover:bg-amber-300/10"
                >
                  Why?
                </button>
              </div>

              <AnimatePresence>
                {whyLayer >= 1 && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-3 border-t border-white/10 pt-3">
                      <MonoLabel>layer 1 · drivers</MonoLabel>
                      <ul className="mt-1.5 space-y-1 font-mono text-[12px] text-slate-200">
                        <li className="flex justify-between">
                          <span>Jupiter activation</span>
                          <span className="text-emerald-300">+21</span>
                        </li>
                        <li className="flex justify-between">
                          <span>Saturn position</span>
                          <span className="text-rose-300">−8</span>
                        </li>
                        <li className="flex justify-between">
                          <span>Dasha alignment</span>
                          <span className="text-emerald-300">+17</span>
                        </li>
                      </ul>
                      {whyLayer === 1 && (
                        <button
                          onClick={() => setWhyLayer(2)}
                          className="mt-2 text-xs font-semibold text-amber-300 hover:underline"
                        >
                          Deeper →
                        </button>
                      )}
                    </div>
                    {whyLayer >= 2 && (
                      <div className="mt-3 border-t border-white/10 pt-3">
                        <MonoLabel>layer 2 · evidence</MonoLabel>
                        <p className="mt-1.5 text-[12px] leading-relaxed text-slate-300">
                          Rule: 10th-lord activation (BPHS, career chapter) ·
                          Input: Jupiter at 14°23′ Taurus (Swiss Ephemeris) ·
                          Cited verse retrieved, ranked #1 of 24 · Confidence
                          from rule strength × retrieval score.
                        </p>
                        {whyLayer === 2 && (
                          <button
                            onClick={() => setWhyLayer(3)}
                            className="mt-2 text-xs font-semibold text-amber-300 hover:underline"
                          >
                            Engineer view →
                          </button>
                        )}
                      </div>
                    )}
                    {whyLayer >= 3 && (
                      <div className="mt-3 border-t border-white/10 pt-3">
                        <MonoLabel>layer 3 · pipeline</MonoLabel>
                        <pre className="mt-1.5 font-mono text-[10.5px] leading-relaxed text-slate-300">
                          {`input → ephemeris calc → chart features
→ rule engine → verse retrieval
→ LLM ranking + synthesis
→ citation gate → policy gate → answer`}
                        </pre>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </section>

      {/* sticky mode toggle */}
      <div className="sticky top-0 z-40 border-b border-slate-200 bg-[#f5f5f7]/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-2.5">
          <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
            Vedic Astro AI · system tour
          </p>
          <div className="flex items-center gap-1 rounded-lg border border-slate-300 bg-white p-0.5">
            {(["explore", "engineer"] as ViewMode[]).map((value) => (
              <button
                key={value}
                onClick={() => setMode(value)}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                  mode === value
                    ? "bg-slate-900 text-white"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {value === "explore" ? "✨ Explore" : "⌘ Engineer"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-6xl px-6">
        {/* -------------------------------------------------- 02 · Ask --- */}
        <section className="py-14">
          <SectionHead
            index="02 · Ask"
            title="Talk to the sky — with follow-through"
            sub={m(
              "The live demo answers these. The interesting ones are the last three.",
              "Prashna questions are judged from the chart of the asking moment; every answer carries its trace.",
            )}
          />
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {[
              "What is influencing me right now?",
              "What changed from last month?",
              "Why are you confident about this?",
              "Show me the calculation.",
              "Challenge your own prediction.",
              "What evidence contradicts it?",
            ].map((question) => (
              <a
                key={question}
                href={DEMO_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="group rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-700 transition hover:border-[#A15C22] hover:text-slate-900"
              >
                {question}
                <span className="ml-1 text-[#A15C22] opacity-0 transition-opacity group-hover:opacity-100">
                  →
                </span>
              </a>
            ))}
          </div>
        </section>

        {/* ------------------------------------------- 04 · Architecture --- */}
        <section id="system" className="border-t border-slate-200 py-14">
          <SectionHead
            index="03 · Architecture"
            title="The Universe Map"
            sub={m(
              "Four continents. Click any place to open its passport.",
              "Each component carries a standardized passport: purpose, runtime, I/O, latency, cost, quality, and failure mode.",
            )}
          />
          <div className="grid gap-4 md:grid-cols-2">
            {continents.map((continent) => (
              <div
                key={continent.key}
                className="rounded-xl border border-slate-200 bg-white p-5"
              >
                <div className="mb-3 flex items-baseline justify-between">
                  <h3 className="font-serif text-lg font-bold text-slate-900">
                    {continent.title}
                  </h3>
                  <MonoLabel>{continent.tagline}</MonoLabel>
                </div>
                <div className="grid gap-2">
                  {continent.components.map((component) => (
                    <button
                      key={component.name}
                      onClick={() => setPassport(component)}
                      className="group flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-[#f5f5f7] px-3.5 py-2.5 text-left transition hover:border-[#A15C22]"
                    >
                      <span className="flex items-center gap-2.5">
                        <span className="text-lg">{component.icon}</span>
                        <span>
                          <span className="block text-sm font-bold text-slate-900">
                            {component.name}
                          </span>
                          <span className="block text-[12px] text-slate-500">
                            {eng ? component.purpose : component.explorePurpose}
                          </span>
                        </span>
                      </span>
                      <span className="shrink-0 font-mono text-[11px] text-slate-400 transition group-hover:text-[#A15C22]">
                        passport →
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ------------------------------------------------ 05 · Replay --- */}
        <section className="border-t border-slate-200 py-14">
          <SectionHead
            index="04 · Replay"
            title="One reading, replayed"
            sub={m(
              "Press play and watch a single request travel the whole system.",
              "The trace store keeps this for every reading — it is what powers the demo's Explanation tab.",
            )}
          />
          <div className="rounded-xl border border-slate-200 bg-white p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <button
                onClick={startReplay}
                className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-bold text-white transition hover:bg-slate-700"
              >
                ▶ Replay a prediction
              </button>
              <p className="font-mono text-[11px] text-slate-400">
                total ≈1.9s · LLM ≈$0.02 · compute ≈$0.0001 · 0 errors
              </p>
            </div>
            <ol className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
              {replaySteps.map((step, index) => {
                const shown = index < replayCount;
                return (
                  <li
                    key={step.label}
                    className={`flex items-center justify-between gap-2 rounded-md border px-3 py-2 transition-all duration-300 ${
                      shown
                        ? "border-emerald-200 bg-emerald-50/60 opacity-100"
                        : "border-slate-100 bg-[#f5f5f7] opacity-40"
                    }`}
                  >
                    <span>
                      <span className="block text-[13px] font-semibold text-slate-800">
                        {index + 1}. {step.label}
                      </span>
                      <span className="block text-[11px] text-slate-500">
                        {step.detail}
                      </span>
                    </span>
                    <span className="shrink-0 font-mono text-[10px] text-slate-400">
                      {step.ms}
                    </span>
                  </li>
                );
              })}
            </ol>
          </div>
        </section>

        {/* -------------------------------------------- 06 · Evaluation --- */}
        <section className="border-t border-slate-200 py-14">
          <SectionHead
            index="05 · Evaluation"
            title="How we know it works"
            sub={m(
              "Different questions get different exams — being right about the sky is not the same as being faithful to the books.",
              "Correctness is layered: astronomical exactness, rule-engine determinism, retrieval quality, and explanation faithfulness are separate metrics with separate gates.",
            )}
          />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              {
                label: "Astronomy",
                explore: "Planet positions match the observatories.",
                engineer: "arc-second agreement vs. published ephemerides",
                value: "exact",
              },
              {
                label: "Rules",
                explore: "The old rules fire the same way every time.",
                engineer: "deterministic engine, golden-chart suite in CI",
                value: "deterministic",
              },
              {
                label: "Citations",
                explore: "Every sentence points at its source.",
                engineer: "citation gate: uncited sentences deleted + logged",
                value: "100% cited",
              },
              {
                label: "Judgement",
                explore: "When evidence is thin, it says so.",
                engineer:
                  "REFUSE/ABSTAIN policy on out-of-scope + low-evidence",
                value: "gated",
              },
            ].map((metric) => (
              <div
                key={metric.label}
                className="rounded-xl border border-slate-200 bg-white p-4"
              >
                <MonoLabel>{metric.label}</MonoLabel>
                <p className="mt-1 font-mono text-lg font-bold text-[#2C5A73]">
                  {metric.value}
                </p>
                <p className="mt-1.5 text-[12px] leading-relaxed text-slate-500">
                  {eng ? metric.engineer : metric.explore}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* -------------------------------------------- 07 · Operations --- */}
        <section className="border-t border-slate-200 py-14">
          <SectionHead
            index="06 · Operations"
            title="Agent Village"
            sub={m(
              "Small workers that keep the universe honest while nobody watches.",
              "Automated checks by trigger: per-request gates, CI-time suites, and deploy-time health checks.",
            )}
          />
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {agents.map((agent) => (
              <div
                key={agent.name}
                className="flex items-start gap-3 rounded-xl border border-slate-200 bg-white p-4"
              >
                <span className="text-xl">{agent.icon}</span>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-bold text-slate-900">
                      {agent.name}
                    </p>
                    <span
                      className={`rounded-full px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase ${
                        agent.status === "working"
                          ? "bg-emerald-100 text-emerald-700"
                          : agent.status === "watching"
                            ? "bg-sky-100 text-sky-700"
                            : "bg-slate-100 text-slate-500"
                      }`}
                    >
                      {agent.status}
                    </span>
                  </div>
                  <p className="mt-0.5 text-[12px] text-slate-500">
                    {agent.detail}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* --------------------------------------------- 08 · Economics --- */}
        <section className="border-t border-slate-200 py-14">
          <SectionHead
            index="07 · Economics"
            title="Cost Treasury"
            sub={m(
              "Everything expensive has a price tag on it.",
              "Unit economics per reading; deterministic stages are effectively free — the LLM synthesis dominates.",
            )}
          />
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <MonoLabel>per reading</MonoLabel>
              <p className="mt-1 font-mono text-2xl font-bold text-slate-900">
                ≈$0.02
              </p>
              <p className="mt-1 text-[12px] text-slate-500">
                LLM synthesis ≈$0.02 · retrieval ≈$0.001 · astronomy ≈$0.0001
              </p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <MonoLabel>hosting</MonoLabel>
              <p className="mt-1 font-mono text-2xl font-bold text-slate-900">
                $0/mo
              </p>
              <p className="mt-1 text-[12px] text-slate-500">
                Render free tier — sleeps idle, ≈50s cold start; $7/mo for
                always-on
              </p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <MonoLabel>at 10k readings/mo</MonoLabel>
              <p className="mt-1 font-mono text-2xl font-bold text-slate-900">
                ≈$210
              </p>
              <p className="mt-1 text-[12px] text-slate-500">
                cost scales with synthesis, not with charts computed
              </p>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------- 09 · Build --- */}
        <section className="border-t border-slate-200 py-14">
          <SectionHead
            index="08 · Behind the universe"
            title="Built by Aakriti Gupta"
            sub="Machine learning · quantitative analysis · AI systems · data · evaluation · UX"
          />
          <div className="rounded-xl border-2 border-[#A15C22]/40 bg-white p-6">
            <MonoLabel>the question this project answers</MonoLabel>
            <p className="mt-2 max-w-3xl font-serif text-xl font-bold leading-relaxed text-slate-900">
              Can a predictive AI system be understandable enough for a child to
              explore, rigorous enough for an engineer to inspect, and
              transparent enough for anyone to challenge?
            </p>
            <div className="mt-5 flex flex-wrap gap-3">
              <a
                href={DEMO_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-lg bg-[#A15C22] px-5 py-2.5 text-sm font-bold text-white transition hover:bg-[#8a4d1c]"
              >
                Enter the live demo →
              </a>
              <button
                onClick={() => navigate("/ai-playground")}
                className="rounded-lg border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-slate-400"
              >
                More experiments
              </button>
            </div>
          </div>
          <p className="mt-6 pb-4 text-[11px] leading-relaxed text-slate-400">
            Latency, cost and status figures on this page describe the system's
            design envelope and are illustrative; the live demo's Explanation
            tab shows the real trace for any reading you cast. Readings are for
            reflection, not medical, legal, or financial advice.
          </p>
        </section>
      </main>

      {/* passport modal */}
      <AnimatePresence>
        {passport && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm"
            onClick={() => setPassport(null)}
          >
            <motion.div
              initial={{ scale: 0.97, opacity: 0, y: 12 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.97, opacity: 0, y: 12 }}
              transition={{ duration: 0.18 }}
              className="w-full max-w-lg rounded-2xl border border-slate-200 bg-[#fdfdfc] p-6 shadow-2xl"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="flex h-12 w-12 items-center justify-center rounded-xl border border-slate-200 bg-white text-2xl">
                    {passport.icon}
                  </span>
                  <div>
                    <MonoLabel>component passport</MonoLabel>
                    <h3 className="font-serif text-xl font-bold text-slate-900">
                      {passport.name}
                    </h3>
                  </div>
                </div>
                <button
                  onClick={() => setPassport(null)}
                  aria-label="Close passport"
                  className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition hover:border-slate-400 hover:text-slate-900"
                >
                  ✕
                </button>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-slate-600">
                {passport.purpose}
              </p>
              <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2.5 border-t border-slate-200 pt-4">
                {[
                  ["Runtime", passport.runtime],
                  ["Inputs", passport.inputs],
                  ["Outputs", passport.outputs],
                  ["Latency", passport.p95],
                  ["Availability", passport.availability],
                  ["Cost", passport.cost],
                ].map(([label, value]) => (
                  <div key={label}>
                    <dt className="font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                      {label}
                    </dt>
                    <dd className="text-[13px] font-medium text-slate-800">
                      {value}
                    </dd>
                  </div>
                ))}
              </dl>
              <div className="mt-4 rounded-lg border border-slate-200 bg-white p-3">
                <MonoLabel>quality</MonoLabel>
                <p className="mt-0.5 text-[13px] text-slate-700">
                  {passport.quality}
                </p>
              </div>
              <div className="mt-2 rounded-lg border border-[#A15C22]/30 bg-[#A15C22]/5 p-3">
                <MonoLabel>how it fails</MonoLabel>
                <p className="mt-0.5 text-[13px] text-slate-700">
                  {passport.failureMode}
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
