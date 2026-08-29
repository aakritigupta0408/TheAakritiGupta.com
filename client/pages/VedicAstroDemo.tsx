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

type ViewMode = "story" | "dev";

/* ---------------------------------------------------------------- data --- */

interface Guide {
  key: string;
  emoji: string;
  name: string;
  role: string;
  line: string;
  devLine: string;
}

const guides: Guide[] = [
  {
    key: "surya",
    emoji: "☀️",
    name: "Surya",
    role: "the King",
    line: "I mark where your light stands. Every chart begins with me.",
    devLine: "Solar longitude anchors the lagna math and the D1 frame.",
  },
  {
    key: "chandra",
    emoji: "🌙",
    name: "Chandra",
    role: "the Dreamer",
    line: "Your mind is my territory. The dasha clock starts from my nakshatra.",
    devLine: "Moon nakshatra seeds the Vimshottari dasha tree.",
  },
  {
    key: "mangala",
    emoji: "🔥",
    name: "Mangala",
    role: "the Warrior",
    line: "I bring the push. Where I sit, things get done — or fought over.",
    devLine: "Mars placements weight drive/conflict rules in the engine.",
  },
  {
    key: "budha",
    emoji: "📯",
    name: "Budha",
    role: "the Messenger",
    line: "Words, trades, wit — I carry them between houses.",
    devLine: "Mercury feeds communication and commerce significations.",
  },
  {
    key: "guru",
    emoji: "📖",
    name: "Guru",
    role: "the Teacher",
    line: "I expand what I touch. Ask me why — I always cite my verse.",
    devLine:
      "Jupiter rules the growth rules; every firing carries a BPHS citation.",
  },
  {
    key: "shukra",
    emoji: "🎨",
    name: "Shukra",
    role: "the Artist",
    line: "Beauty, bonds, and comforts pass through my hands.",
    devLine: "Venus weights relationship and aesthetics significations.",
  },
  {
    key: "shani",
    emoji: "⏳",
    name: "Shani",
    role: "the Timekeeper",
    line: "I am slow, and I am fair. What I delay, I make durable.",
    devLine: "Saturn drives delay/discipline rules and long transits.",
  },
  {
    key: "rahu",
    emoji: "🐍",
    name: "Rahu",
    role: "the Shadow",
    line: "I am hunger without a body. I amplify whatever house holds me.",
    devLine: "North node: amplification rules, no rulership, shadow points.",
  },
  {
    key: "ketu",
    emoji: "☄️",
    name: "Ketu",
    role: "the Comet-sage",
    line: "I am the tail that lets go. Where I sit, you detach — and see.",
    devLine: "South node: detachment significations, moksha-karaka rules.",
  },
];

interface Sheet {
  name: string;
  icon: string;
  storyLine: string;
  devLine: string;
  runtime: string;
  inputs: string;
  outputs: string;
  speed: string;
  uptime: string;
  cost: string;
  quality: string;
  weakness: string;
}

interface Zone {
  key: string;
  title: string;
  storyTitle: string;
  tagline: string;
  emoji: string;
  places: Sheet[];
}

const zones: Zone[] = [
  {
    key: "experience",
    title: "Experience",
    storyTitle: "Experience Isles",
    tagline: "What a visitor touches",
    emoji: "🏝️",
    places: [
      {
        name: "Birth Chart City",
        icon: "🏙️",
        storyLine: "Where your chart is drawn — all sixteen of them.",
        devLine:
          "Onboarding, birth-time handling, and the D1–D60 chart renderer.",
        runtime: "React + SVG renderer",
        inputs: "Birth date, time, place",
        outputs: "16 divisional charts",
        speed: "120ms render",
        uptime: "static — always up",
        cost: "$0 / chart",
        quality: "positions verified against Swiss Ephemeris",
        weakness: "unknown birth time → falls back to chandra lagna",
      },
      {
        name: "Prediction Harbor",
        icon: "⚓",
        storyLine: "Where predictions arrive, from life-arc to today.",
        devLine:
          "Serves dasha-arc, varshaphala, monthly and daily readings with citations.",
        runtime: "FastAPI on Render",
        inputs: "Computed chart + question scope",
        outputs: "Cited reading, 4 time horizons",
        speed: "1.9s (LLM-bound)",
        uptime: "free tier — sleeps after 15 min idle",
        cost: "≈$0.02 / reading",
        quality: "every sentence must cite a verse or chart fact",
        weakness: "uncited sentence → deleted, visible in trace",
      },
      {
        name: "Conversation Village",
        icon: "💬",
        storyLine: "Ask a question; the sky at that instant answers.",
        devLine:
          "Prashna: a direct question judged from the chart of the moment it is asked.",
        runtime: "Claude + prashna procedure",
        inputs: "A question + the moment's chart",
        outputs: "Judgement with reasoning trace",
        speed: "2.4s",
        uptime: "backed by Prediction Harbor",
        cost: "≈$0.02 / question",
        quality: "trace shows the full judgement path",
        weakness: "ambiguous question → asks for clarification",
      },
    ],
  },
  {
    key: "intelligence",
    title: "Intelligence",
    storyTitle: "Intelligence Peaks",
    tagline: "Where readings are made",
    emoji: "⛰️",
    places: [
      {
        name: "Ganita Engine",
        icon: "🧮",
        storyLine:
          "The mathematics mountain — planets computed, never guessed.",
        devLine:
          "Deterministic astronomy: Swiss Ephemeris positions, divisional charts, Vimshottari dasha. No model ever computes a position.",
        runtime: "Python + Swiss Ephemeris",
        inputs: "UTC instant + coordinates",
        outputs: "Longitudes, houses, dasha tree",
        speed: "31ms",
        uptime: "99.99% (pure computation)",
        cost: "≈$0.0001 / chart",
        quality: "cross-checked against published ephemerides",
        weakness: "pre-600 CE dates → rejected, outside ephemeris range",
      },
      {
        name: "Rules Mountain",
        icon: "📜",
        storyLine: "The old rules, written down precisely.",
        devLine:
          "Classical yogas, aspects, house significations encoded as a deterministic rule engine.",
        runtime: "Typed rule engine",
        inputs: "Chart facts",
        outputs: "Fired rules with verse citations",
        speed: "8ms",
        uptime: "99.99%",
        cost: "≈$0 / evaluation",
        quality: "each rule carries its classical source verse",
        weakness: "conflicting rules → both surfaced, never merged silently",
      },
      {
        name: "LLM City",
        icon: "🏛️",
        storyLine: "The interpreter — reads the computed sky aloud.",
        devLine:
          "Claude ranks retrieved classical verses against the computed chart, synthesizes the reading, and must cite every sentence.",
        runtime: "Claude (Anthropic API)",
        inputs: "Chart facts + retrieved verses + question",
        outputs: "Cited synthesis",
        speed: "1.8s",
        uptime: "API-bound",
        cost: "≈$0.02 / synthesis",
        quality: "uncited claims deleted by the citation gate",
        weakness: "gate strips too much → reading marked low-confidence",
      },
      {
        name: "Citation Gate",
        icon: "🚧",
        storyLine: "The honesty checkpoint every sentence passes through.",
        devLine:
          "Any sentence that cannot point to a verse or a chart fact is removed, and the removal is logged in the visible trace.",
        runtime: "Deterministic post-processor",
        inputs: "Draft reading",
        outputs: "Verified reading + deletion log",
        speed: "40ms",
        uptime: "99.99%",
        cost: "≈$0 / pass",
        quality: "0 uncited sentences shipped",
        weakness: "over-strict match → sentence lost; logged for review",
      },
    ],
  },
  {
    key: "data",
    title: "Data",
    storyTitle: "Data Depths",
    tagline: "What the system knows",
    emoji: "🌊",
    places: [
      {
        name: "Ephemeris Observatory",
        icon: "🔭",
        storyLine: "Where planetary positions come from.",
        devLine: "Swiss Ephemeris files — the astronomical source of truth.",
        runtime: "Bundled ephemeris data",
        inputs: "—",
        outputs: "Planetary state, any instant",
        speed: "in-memory",
        uptime: "99.99%",
        cost: "$0 (bundled)",
        quality: "arc-second agreement with JPL",
        weakness: "corrupt file → startup check fails loudly",
      },
      {
        name: "Knowledge Library",
        icon: "📚",
        storyLine: "The shelf of old books the interpreter may quote.",
        devLine:
          "Classical verse corpus (BPHS and related texts) indexed for retrieval, each verse tagged with its source.",
        runtime: "Embedded corpus + retrieval",
        inputs: "Chart facts + question",
        outputs: "Ranked candidate verses",
        speed: "60ms",
        uptime: "99.99%",
        cost: "≈$0.001 / retrieval",
        quality: "every verse traceable to text + chapter",
        weakness: "no relevant verse → reading says so explicitly",
      },
      {
        name: "Trace Store",
        icon: "🧾",
        storyLine: "The diary of how each reading was made.",
        devLine:
          "Every reading's full computation trace — what fired, what was retrieved, what was deleted — kept for the Explanation tab.",
        runtime: "Structured logs",
        inputs: "Pipeline events",
        outputs: "Explanation tab content",
        speed: "async",
        uptime: "99.9%",
        cost: "≈$0.0002 / reading",
        quality: "trace replays reproduce the reading",
        weakness: "trace gap → reading flagged unexplainable",
      },
    ],
  },
  {
    key: "operations",
    title: "Operations",
    storyTitle: "Ops Citadel",
    tagline: "How it stays honest",
    emoji: "🏰",
    places: [
      {
        name: "Evaluation Warehouse",
        icon: "🏗️",
        storyLine: "The exam the system must pass before changing.",
        devLine:
          "Golden charts with known classical judgements; every engine change re-runs the suite before deploy.",
        runtime: "pytest suite + golden set",
        inputs: "Candidate build",
        outputs: "Pass / fail per golden chart",
        speed: "4 min full suite",
        uptime: "CI-bound",
        cost: "≈$0.40 / full run",
        quality: "positions exact; readings faithfulness-scored",
        weakness: "regression → deploy blocked",
      },
      {
        name: "Policy Gate",
        icon: "🛡️",
        storyLine: "The judgement about when not to answer.",
        devLine:
          "Production REFUSE/ABSTAIN gate: medical, legal, and death questions are refused; low-evidence readings abstain rather than guess.",
        runtime: "ENVIRONMENT=production flag",
        inputs: "Question + reading confidence",
        outputs: "Answer, abstain, or refusal",
        speed: "5ms",
        uptime: "99.99%",
        cost: "$0",
        quality: "refusal classes unit-tested",
        weakness: "over-refusal → logged for prompt review",
      },
      {
        name: "Deployment Harbor",
        icon: "🚢",
        storyLine: "Where new versions come ashore.",
        devLine:
          "Render deploy from the jyotisha repo's deploy branch; free tier sleeps idle, wakes in ~50s.",
        runtime: "Render (Docker)",
        inputs: "deploy branch push",
        outputs: "Live service",
        speed: "cold start ≈50s",
        uptime: "free tier",
        cost: "$0/mo (free) — $7/mo always-on",
        quality: "health-checked on /",
        weakness: "failed health check → previous build stays live",
      },
    ],
  },
];

interface Quest {
  label: string;
  detail: string;
  ms: string;
}

const questSteps: Quest[] = [
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

const party = [
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

const PIXEL = { fontFamily: "'Silkscreen', monospace" } as const;

function PixelTag({
  children,
  tone = "gold",
}: {
  children: ReactNode;
  tone?: "gold" | "dim";
}) {
  return (
    <span
      style={PIXEL}
      className={`inline-block text-[10px] uppercase tracking-wider ${
        tone === "gold" ? "text-amber-300" : "text-slate-400"
      }`}
    >
      {children}
    </span>
  );
}

/** Segmented XP-style meter — 10 chunky cells, game HUD language. */
function XpBar({ value }: { value: number }) {
  const filled = Math.round(value / 10);
  return (
    <span
      className="inline-flex gap-[3px]"
      role="img"
      aria-label={`${value} out of 100`}
    >
      {Array.from({ length: 10 }, (_, i) => (
        <span
          key={i}
          className={`h-3 w-3.5 border ${
            i < filled
              ? "border-amber-300 bg-amber-400 shadow-[0_0_6px_#fbbf2488]"
              : "border-slate-600 bg-slate-800"
          }`}
        />
      ))}
    </span>
  );
}

function Frame({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`border-2 border-slate-600 bg-[#101830] shadow-[5px_5px_0_#000000aa] ${className}`}
    >
      {children}
    </div>
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
      <p
        style={PIXEL}
        className="text-[11px] uppercase tracking-widest text-amber-400"
      >
        {index}
      </p>
      <h2 className="mt-1 font-serif text-2xl font-bold text-white sm:text-3xl">
        {title}
      </h2>
      <p className="mt-1 max-w-2xl text-sm text-slate-400">{sub}</p>
    </div>
  );
}

/* --------------------------------------------------------------- page --- */

export default function VedicAstroDemo() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<ViewMode>("story");
  const [guide, setGuide] = useState<Guide>(guides[4]); // Guru greets first
  const [whyLayer, setWhyLayer] = useState(0);
  const [sheet, setSheet] = useState<Sheet | null>(null);
  const [questCount, setQuestCount] = useState(0);
  const questTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const dev = mode === "dev";
  const m = (story: string, devText: string) => (dev ? devText : story);

  const startQuest = () => {
    if (questTimer.current) clearInterval(questTimer.current);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setQuestCount(questSteps.length);
      return;
    }
    setQuestCount(0);
    questTimer.current = setInterval(() => {
      setQuestCount((current) => {
        if (current >= questSteps.length) {
          if (questTimer.current) clearInterval(questTimer.current);
          return current;
        }
        return current + 1;
      });
    }, 260);
  };

  useEffect(
    () => () => {
      if (questTimer.current) clearInterval(questTimer.current);
    },
    [],
  );

  const questDone = questCount >= questSteps.length;

  return (
    <div className="min-h-screen bg-[#0B1020] text-slate-100">
      <Navigation />

      {/* ------------------------------------------------------- HUD bar --- */}
      <div className="sticky top-0 z-40 border-b-2 border-slate-700 bg-[#0B1020]/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-6 py-2">
          <div className="flex items-center gap-3">
            <span
              style={PIXEL}
              className="text-[11px] uppercase tracking-widest text-amber-300"
            >
              ✦ Vedic Astro AI
            </span>
            <span
              style={PIXEL}
              className="hidden text-[10px] text-emerald-400 sm:inline"
            >
              <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400 align-middle" />
              online
            </span>
          </div>
          <div className="flex items-center gap-1 border-2 border-slate-600 bg-[#101830] p-0.5">
            {(["story", "dev"] as ViewMode[]).map((value) => (
              <button
                key={value}
                onClick={() => setMode(value)}
                style={PIXEL}
                className={`px-3 py-1 text-[10px] uppercase tracking-wider transition ${
                  mode === value
                    ? "bg-amber-400 text-slate-900"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {value === "story" ? "✦ Story" : "⌘ Dev"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ---------------------------------------------------- title crawl --- */}
      <header className="relative overflow-hidden pb-10 pt-24">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
        >
          {[380, 600, 840].map((size) => (
            <span
              key={size}
              className="absolute left-1/2 top-24 -translate-x-1/2 rounded-full border border-white/5"
              style={{ width: size, height: size }}
            />
          ))}
          <span className="absolute left-[15%] top-[20%] h-1 w-1 rounded-full bg-amber-200/80" />
          <span className="absolute left-[78%] top-[16%] h-1.5 w-1.5 rounded-full bg-sky-200/70" />
          <span className="absolute left-[65%] top-[60%] h-1 w-1 rounded-full bg-rose-200/70" />
        </div>
        <div className="relative z-10 mx-auto max-w-6xl px-6 text-center">
          <button
            onClick={() => navigate("/ai-playground")}
            className="mx-auto mb-8 flex items-center gap-2 text-sm text-slate-400 transition-colors hover:text-white"
          >
            <ArrowLeft size={16} />
            Back to AI Playground
          </button>
          <p
            style={PIXEL}
            className="text-[12px] uppercase tracking-[0.3em] text-amber-400"
          >
            a computational universe
          </p>
          <h1 className="mx-auto mt-3 max-w-3xl font-serif text-4xl font-bold leading-tight text-white sm:text-6xl">
            Understanding Time
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-base leading-relaxed text-slate-300">
            {m(
              "Nine planetary guides, four regions, one honest machine. Explore it like a world; question it like a lab.",
              "Deterministic ephemeris computation feeding a cited-retrieval LLM pipeline behind a citation gate and a REFUSE/ABSTAIN policy layer.",
            )}
          </p>
          <div className="mt-7 flex flex-wrap justify-center gap-4">
            <a
              href={DEMO_URL}
              target="_blank"
              rel="noopener noreferrer"
              style={PIXEL}
              className="border-2 border-amber-300 bg-amber-400 px-6 py-3 text-[12px] uppercase tracking-wider text-slate-900 shadow-[4px_4px_0_#00000088] transition hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_#00000088]"
            >
              ▶ Press Start
            </a>
            <button
              onClick={() => {
                const target = document.getElementById("worldmap");
                if (target) {
                  window.scrollTo(
                    0,
                    target.getBoundingClientRect().top + window.scrollY - 60,
                  );
                }
              }}
              style={PIXEL}
              className="border-2 border-slate-500 bg-transparent px-6 py-3 text-[12px] uppercase tracking-wider text-slate-200 shadow-[4px_4px_0_#00000055] transition hover:border-slate-300 hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_#00000055]"
            >
              World Map ↓
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6">
        {/* ------------------------------------------ 01 · character select --- */}
        <section className="py-12">
          <SectionHead
            index="01 · Choose your guide"
            title="Meet your sky team"
            sub={m(
              "Nine heroes, one story — yours. Tap a hero to hear its job; every reading below is their teamwork.",
              "The nine grahas. Each drives a family of rules in the deterministic engine; tap for its computational role.",
            )}
          />
          <div className="flex gap-2 overflow-x-auto pb-2">
            {guides.map((g) => (
              <button
                key={g.key}
                onClick={() => setGuide(g)}
                aria-pressed={guide.key === g.key}
                className={`flex min-w-[86px] flex-col items-center gap-1 border-2 px-3 py-3 transition ${
                  guide.key === g.key
                    ? "border-amber-300 bg-[#1a2440] shadow-[0_0_14px_#fbbf2444]"
                    : "border-slate-700 bg-[#101830] hover:border-slate-500"
                }`}
              >
                <span className="text-2xl">{g.emoji}</span>
                <span style={PIXEL} className="text-[10px] text-white">
                  {g.name}
                </span>
                <span className="text-[10px] text-slate-400">{g.role}</span>
              </button>
            ))}
          </div>
          {/* dialogue box */}
          <Frame className="mt-4 p-4">
            <div className="flex items-start gap-3">
              <span className="text-3xl">{guide.emoji}</span>
              <div>
                <p
                  style={PIXEL}
                  className="text-[11px] uppercase text-amber-300"
                >
                  {guide.name} · {guide.role}
                </p>
                <p className="mt-1 text-sm leading-relaxed text-slate-200">
                  “{dev ? guide.devLine : guide.line}”
                </p>
              </div>
            </div>
          </Frame>
        </section>

        {/* --------------------------------------------- 02 · current signal --- */}
        <section className="border-t-2 border-slate-800 py-12">
          <SectionHead
            index="02 · Current signal"
            title="Career Momentum"
            sub={m(
              "A sample reading from a sample chart. The Why? button is the whole point — press it until you hit bedrock.",
              "Sample output shape. Every production reading carries this drill-down in the demo's Explanation tab.",
            )}
          />
          <Frame className="max-w-xl p-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <PixelTag>power level</PixelTag>
                <p className="mt-1 font-serif text-3xl font-bold text-amber-300">
                  74<span className="text-lg text-slate-400">/100</span>
                </p>
              </div>
              <XpBar value={74} />
            </div>
            <p className="mt-2 text-xs text-slate-400">
              Saturn transit · Jupiter influence · dasha transition · 6 more
              signals · confidence 81%
            </p>
            <div className="mt-3 flex gap-2">
              <button
                onClick={() => setWhyLayer(whyLayer === 0 ? 1 : 0)}
                style={PIXEL}
                className="border-2 border-amber-300 px-4 py-1.5 text-[11px] uppercase tracking-wider text-amber-300 transition hover:bg-amber-300/10"
              >
                Why?
              </button>
              {whyLayer >= 1 && whyLayer < 3 && (
                <button
                  onClick={() => setWhyLayer(whyLayer + 1)}
                  style={PIXEL}
                  className="border-2 border-slate-600 px-4 py-1.5 text-[11px] uppercase tracking-wider text-slate-300 transition hover:border-slate-400"
                >
                  Deeper →
                </button>
              )}
            </div>
            <AnimatePresence>
              {whyLayer >= 1 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-4 border-t-2 border-dashed border-slate-700 pt-3">
                    <PixelTag>lv.1 drivers</PixelTag>
                    <ul className="mt-1.5 space-y-1 font-mono text-[13px] text-slate-200">
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
                  </div>
                  {whyLayer >= 2 && (
                    <div className="mt-3 border-t-2 border-dashed border-slate-700 pt-3">
                      <PixelTag>lv.2 evidence</PixelTag>
                      <p className="mt-1.5 text-[13px] leading-relaxed text-slate-300">
                        Rule: 10th-lord activation (BPHS, career chapter) ·
                        Input: Jupiter at 14°23′ Taurus (Swiss Ephemeris) ·
                        Cited verse retrieved, ranked #1 of 24 · Confidence from
                        rule strength × retrieval score.
                      </p>
                    </div>
                  )}
                  {whyLayer >= 3 && (
                    <div className="mt-3 border-t-2 border-dashed border-slate-700 pt-3">
                      <PixelTag>lv.3 pipeline</PixelTag>
                      <pre className="mt-1.5 font-mono text-[11px] leading-relaxed text-slate-300">{`input → ephemeris calc → chart features
→ rule engine → verse retrieval
→ LLM ranking + synthesis
→ citation gate → policy gate → answer`}</pre>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </Frame>
        </section>

        {/* ------------------------------------------------- 03 · world map --- */}
        <section id="worldmap" className="border-t-2 border-slate-800 py-12">
          <SectionHead
            index="03 · World map"
            title={m(
              "Four regions to explore",
              "The architecture, as territory",
            )}
            sub={m(
              "Enter any location to open its character sheet — stats, powers, and its one weakness.",
              "13 components, each with a standardized sheet: runtime, I/O, latency, cost, quality, and failure mode.",
            )}
          />
          <div className="grid gap-4 md:grid-cols-2">
            {zones.map((zone) => (
              <Frame key={zone.key} className="p-4">
                <div className="mb-3 flex items-baseline justify-between">
                  <h3 className="font-serif text-lg font-bold text-white">
                    <span className="mr-2">{zone.emoji}</span>
                    {m(zone.storyTitle, zone.title)}
                  </h3>
                  <PixelTag tone="dim">{zone.tagline}</PixelTag>
                </div>
                <div className="grid gap-1.5">
                  {zone.places.map((place) => (
                    <button
                      key={place.name}
                      onClick={() => setSheet(place)}
                      className="group flex items-center justify-between gap-3 border-2 border-slate-700 bg-[#0d1428] px-3 py-2 text-left transition hover:border-amber-300"
                    >
                      <span className="flex items-center gap-2.5">
                        <span className="text-lg">{place.icon}</span>
                        <span>
                          <span className="block text-sm font-bold text-white">
                            {place.name}
                          </span>
                          <span className="block text-[12px] text-slate-400">
                            {dev ? place.devLine : place.storyLine}
                          </span>
                        </span>
                      </span>
                      <span
                        style={PIXEL}
                        className="shrink-0 text-[10px] uppercase text-slate-500 transition group-hover:text-amber-300"
                      >
                        enter →
                      </span>
                    </button>
                  ))}
                </div>
              </Frame>
            ))}
          </div>
        </section>

        {/* ------------------------------------------------- 04 · quest log --- */}
        <section className="border-t-2 border-slate-800 py-12">
          <SectionHead
            index="04 · Quest log"
            title="One reading's journey"
            sub={m(
              "Press play and follow one request across the whole world, checkpoint by checkpoint.",
              "The trace store records this for every reading — it powers the demo's Explanation tab.",
            )}
          />
          <Frame className="p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <button
                onClick={startQuest}
                style={PIXEL}
                className="border-2 border-amber-300 bg-amber-400 px-5 py-2 text-[11px] uppercase tracking-wider text-slate-900 shadow-[3px_3px_0_#00000088] transition hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0_#00000088]"
              >
                ▶ Begin quest
              </button>
              <p className="font-mono text-[11px] text-slate-400">
                total ≈1.9s · LLM ≈$0.02 · compute ≈$0.0001 · 0 errors
              </p>
            </div>
            <ol className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
              {questSteps.map((step, index) => {
                const shown = index < questCount;
                return (
                  <li
                    key={step.label}
                    className={`flex items-center justify-between gap-2 border-2 px-3 py-2 transition-all duration-300 ${
                      shown
                        ? "border-emerald-500/60 bg-emerald-900/20 opacity-100"
                        : "border-slate-800 bg-[#0d1428] opacity-40"
                    }`}
                  >
                    <span>
                      <span className="block text-[13px] font-semibold text-slate-100">
                        {shown ? "✓" : "·"} {step.label}
                      </span>
                      <span className="block text-[11px] text-slate-500">
                        {step.detail}
                      </span>
                    </span>
                    <span className="shrink-0 font-mono text-[10px] text-slate-500">
                      {step.ms}
                    </span>
                  </li>
                );
              })}
            </ol>
            {questDone && (
              <p
                style={PIXEL}
                className="mt-4 text-center text-[12px] uppercase tracking-widest text-amber-300"
              >
                ★ Quest complete — every sentence cited ★
              </p>
            )}
          </Frame>
        </section>

        {/* ----------------------------------------- 05 · scoreboard + party --- */}
        <section className="border-t-2 border-slate-800 py-12">
          <SectionHead
            index="05 · Scoreboard"
            title="How we know it works"
            sub={m(
              "Different questions get different exams — being right about the sky is not the same as being faithful to the books.",
              "Layered correctness: astronomical exactness, rule determinism, citation coverage, and refusal policy are separate gates.",
            )}
          />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              {
                label: "Astronomy",
                value: "exact",
                story: "Planet positions match the observatories.",
                devText: "arc-second agreement vs. published ephemerides",
              },
              {
                label: "Rules",
                value: "deterministic",
                story: "The old rules fire the same way every time.",
                devText: "deterministic engine, golden-chart suite in CI",
              },
              {
                label: "Citations",
                value: "100% cited",
                story: "Every sentence points at its source.",
                devText: "citation gate: uncited sentences deleted + logged",
              },
              {
                label: "Judgement",
                value: "gated",
                story: "When evidence is thin, it says so.",
                devText: "REFUSE/ABSTAIN policy on out-of-scope + low-evidence",
              },
            ].map((metric) => (
              <Frame key={metric.label} className="p-4">
                <PixelTag tone="dim">{metric.label}</PixelTag>
                <p
                  style={PIXEL}
                  className="mt-1 text-[13px] uppercase text-emerald-300"
                >
                  {metric.value}
                </p>
                <p className="mt-1.5 text-[12px] leading-relaxed text-slate-400">
                  {dev ? metric.devText : metric.story}
                </p>
              </Frame>
            ))}
          </div>

          <div className="mt-8">
            <SectionHead
              index="06 · Party members"
              title="Who watches the watchers"
              sub={m(
                "Small companions that keep the world honest while nobody plays.",
                "Automated checks by trigger: per-request gates, CI suites, deploy-time health checks.",
              )}
            />
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {party.map((agent) => (
                <Frame
                  key={agent.name}
                  className="flex items-start gap-3 p-3.5"
                >
                  <span className="text-xl">{agent.icon}</span>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-bold text-white">
                        {agent.name}
                      </p>
                      <span
                        style={PIXEL}
                        className={`text-[9px] uppercase ${
                          agent.status === "working"
                            ? "text-emerald-400"
                            : agent.status === "watching"
                              ? "text-sky-400"
                              : "text-slate-500"
                        }`}
                      >
                        {agent.status}
                      </span>
                    </div>
                    <p className="mt-0.5 text-[12px] text-slate-400">
                      {agent.detail}
                    </p>
                  </div>
                </Frame>
              ))}
            </div>
          </div>
        </section>

        {/* ------------------------------------------------ 07 · gold + end --- */}
        <section className="border-t-2 border-slate-800 py-12">
          <SectionHead
            index="07 · Gold"
            title="What a reading costs"
            sub={m(
              "Everything expensive has a price tag on it.",
              "Unit economics per reading; deterministic stages are effectively free — LLM synthesis dominates.",
            )}
          />
          <div className="grid gap-3 sm:grid-cols-3">
            {[
              {
                label: "per reading",
                value: "≈$0.02",
                note: "LLM ≈$0.02 · retrieval ≈$0.001 · astronomy ≈$0.0001",
              },
              {
                label: "hosting",
                value: "$0/mo",
                note: "Render free tier — sleeps idle, ≈50s wake; $7/mo always-on",
              },
              {
                label: "at 10k readings/mo",
                value: "≈$210",
                note: "cost scales with synthesis, not with charts computed",
              },
            ].map((coin) => (
              <Frame key={coin.label} className="p-4">
                <PixelTag tone="dim">{coin.label}</PixelTag>
                <p className="mt-1 font-mono text-2xl font-bold text-amber-300">
                  🪙 {coin.value}
                </p>
                <p className="mt-1 text-[12px] text-slate-400">{coin.note}</p>
              </Frame>
            ))}
          </div>

          <Frame className="mt-10 p-6 text-center">
            <PixelTag>credits</PixelTag>
            <p className="mx-auto mt-2 max-w-2xl font-serif text-xl font-bold leading-relaxed text-white">
              Can a predictive AI system be a world a child can explore, a lab
              an engineer can inspect, and a claim anyone can challenge?
            </p>
            <p className="mt-2 text-sm text-slate-400">
              Built by Aakriti Gupta
            </p>
            <div className="mt-5 flex flex-wrap justify-center gap-3">
              <a
                href={DEMO_URL}
                target="_blank"
                rel="noopener noreferrer"
                style={PIXEL}
                className="border-2 border-amber-300 bg-amber-400 px-6 py-2.5 text-[11px] uppercase tracking-wider text-slate-900 shadow-[3px_3px_0_#00000088] transition hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0_#00000088]"
              >
                New game + → live demo
              </a>
              <button
                onClick={() => navigate("/ai-playground")}
                style={PIXEL}
                className="border-2 border-slate-600 px-6 py-2.5 text-[11px] uppercase tracking-wider text-slate-300 transition hover:border-slate-400"
              >
                More experiments
              </button>
            </div>
          </Frame>
          <p className="mt-6 pb-6 text-center text-[11px] leading-relaxed text-slate-500">
            Latency, cost and status figures describe the system's design
            envelope and are illustrative; the live demo's Explanation tab shows
            the real trace for any reading you cast. Readings are for
            reflection, not medical, legal, or financial advice.
          </p>
        </section>
      </main>

      {/* character sheet modal */}
      <AnimatePresence>
        {sheet && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
            onClick={() => setSheet(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0, y: 12 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 12 }}
              transition={{ duration: 0.18 }}
              className="w-full max-w-lg border-2 border-amber-300/70 bg-[#101830] p-6 shadow-[8px_8px_0_#000000cc]"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="flex h-12 w-12 items-center justify-center border-2 border-slate-600 bg-[#0d1428] text-2xl">
                    {sheet.icon}
                  </span>
                  <div>
                    <PixelTag>character sheet</PixelTag>
                    <h3 className="font-serif text-xl font-bold text-white">
                      {sheet.name}
                    </h3>
                  </div>
                </div>
                <button
                  onClick={() => setSheet(null)}
                  aria-label="Close character sheet"
                  style={PIXEL}
                  className="border-2 border-slate-600 px-2.5 py-1 text-[11px] text-slate-300 transition hover:border-slate-300 hover:text-white"
                >
                  ✕
                </button>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-slate-300">
                {dev ? sheet.devLine : sheet.storyLine}
              </p>
              <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2.5 border-t-2 border-dashed border-slate-700 pt-4">
                {[
                  ["class", sheet.runtime],
                  ["consumes", sheet.inputs],
                  ["produces", sheet.outputs],
                  ["speed", sheet.speed],
                  ["uptime", sheet.uptime],
                  ["upkeep", sheet.cost],
                ].map(([label, value]) => (
                  <div key={label}>
                    <dt
                      style={PIXEL}
                      className="text-[9px] uppercase text-slate-500"
                    >
                      {label}
                    </dt>
                    <dd className="text-[13px] font-medium text-slate-200">
                      {value}
                    </dd>
                  </div>
                ))}
              </dl>
              <div className="mt-4 border-2 border-emerald-700/50 bg-emerald-900/20 p-3">
                <PixelTag tone="dim">special power</PixelTag>
                <p className="mt-0.5 text-[13px] text-slate-200">
                  {sheet.quality}
                </p>
              </div>
              <div className="mt-2 border-2 border-rose-700/50 bg-rose-900/20 p-3">
                <PixelTag tone="dim">weakness</PixelTag>
                <p className="mt-0.5 text-[13px] text-slate-200">
                  {sheet.weakness}
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
