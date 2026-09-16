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

const BIRTH_KEY = "vedicAstroBirth";

/* ---------------------------------------------------------------- data --- */

interface Birth {
  date: string;
  time: string;
  offset: string;
  lat: string;
  lon: string;
  place: string;
}

const SAMPLE_BIRTH: Birth = {
  date: "1990-03-14",
  time: "10:30",
  offset: "+05:30",
  lat: "28.61",
  lon: "77.20",
  place: "Delhi",
};

const guides = [
  {
    emoji: "☀️",
    name: "Surya",
    role: "the King",
    line: "I mark where your light stands. Every chart begins with me.",
  },
  {
    emoji: "🌙",
    name: "Chandra",
    role: "the Dreamer",
    line: "Your mind is my territory. The dasha clock starts from my nakshatra.",
  },
  {
    emoji: "🔥",
    name: "Mangala",
    role: "the Warrior",
    line: "I bring the push. Where I sit, things get done — or fought over.",
  },
  {
    emoji: "📯",
    name: "Budha",
    role: "the Messenger",
    line: "Words, trades, wit — I carry them between houses.",
  },
  {
    emoji: "📖",
    name: "Guru",
    role: "the Teacher",
    line: "I expand what I touch. Ask me why — I always cite my verse.",
  },
  {
    emoji: "🎨",
    name: "Shukra",
    role: "the Artist",
    line: "Beauty, bonds, and comforts pass through my hands.",
  },
  {
    emoji: "⏳",
    name: "Shani",
    role: "the Timekeeper",
    line: "I am slow, and I am fair. What I delay, I make durable.",
  },
  {
    emoji: "🐍",
    name: "Rahu",
    role: "the Shadow",
    line: "I am hunger without a body. I amplify whatever house holds me.",
  },
  {
    emoji: "☄️",
    name: "Ketu",
    role: "the Comet-sage",
    line: "I am the tail that lets go. Where I sit, you detach — and see.",
  },
];

interface Sheet {
  name: string;
  icon: string;
  line: string;
  runtime: string;
  speed: string;
  cost: string;
  quality: string;
  weakness: string;
}

const zones: {
  title: string;
  emoji: string;
  tagline: string;
  places: Sheet[];
}[] = [
  {
    title: "Experience Isles",
    emoji: "🏝️",
    tagline: "What a visitor touches",
    places: [
      {
        name: "Birth Chart City",
        icon: "🏙️",
        line: "Where your chart is drawn — all sixteen of them.",
        runtime: "SVG renderer",
        speed: "120ms",
        cost: "$0 / chart",
        quality: "positions verified against Swiss Ephemeris",
        weakness: "unknown birth time → falls back to chandra lagna",
      },
      {
        name: "Prediction Harbor",
        icon: "⚓",
        line: "Where predictions arrive, from life-arc to today.",
        runtime: "Python on Render",
        speed: "1.9s (LLM-bound)",
        cost: "≈$0.02 / reading",
        quality: "every sentence must cite a verse or chart fact",
        weakness: "uncited sentence → deleted, visible in trace",
      },
      {
        name: "Conversation Village",
        icon: "💬",
        line: "Ask a question; the sky at that instant answers.",
        runtime: "Claude + prashna",
        speed: "2.4s",
        cost: "≈$0.02 / question",
        quality: "trace shows the full judgement path",
        weakness: "ambiguous question → asks for clarification",
      },
    ],
  },
  {
    title: "Intelligence Peaks",
    emoji: "⛰️",
    tagline: "Where readings are made",
    places: [
      {
        name: "Ganita Engine",
        icon: "🧮",
        line: "The mathematics mountain — planets computed, never guessed.",
        runtime: "Swiss Ephemeris",
        speed: "31ms",
        cost: "≈$0.0001 / chart",
        quality: "cross-checked against published ephemerides",
        weakness: "pre-600 CE dates → rejected, outside range",
      },
      {
        name: "Rules Mountain",
        icon: "📜",
        line: "The old rules, written down precisely.",
        runtime: "Typed rule engine",
        speed: "8ms",
        cost: "≈$0",
        quality: "each rule carries its classical source verse",
        weakness: "conflicting rules → both surfaced, never merged",
      },
      {
        name: "LLM City",
        icon: "🏛️",
        line: "The interpreter — reads the computed sky aloud.",
        runtime: "Claude API",
        speed: "1.8s",
        cost: "≈$0.02 / synthesis",
        quality: "uncited claims deleted by the citation gate",
        weakness: "gate strips too much → marked low-confidence",
      },
      {
        name: "Citation Gate",
        icon: "🚧",
        line: "The honesty checkpoint every sentence passes through.",
        runtime: "Post-processor",
        speed: "40ms",
        cost: "≈$0",
        quality: "0 uncited sentences shipped",
        weakness: "over-strict match → sentence lost; logged",
      },
    ],
  },
  {
    title: "Data Depths",
    emoji: "🌊",
    tagline: "What the system knows",
    places: [
      {
        name: "Ephemeris Observatory",
        icon: "🔭",
        line: "Where planetary positions come from.",
        runtime: "Bundled data",
        speed: "in-memory",
        cost: "$0",
        quality: "arc-second agreement with JPL",
        weakness: "corrupt file → startup check fails loudly",
      },
      {
        name: "Knowledge Library",
        icon: "📚",
        line: "The shelf of old books the interpreter may quote.",
        runtime: "Corpus + vector retrieval",
        speed: "60ms",
        cost: "≈$0.001",
        quality: "every verse traceable to text + chapter",
        weakness: "no relevant verse → the reading says so",
      },
      {
        name: "Trace Store",
        icon: "🧾",
        line: "The diary of how each reading was made.",
        runtime: "Structured logs",
        speed: "async",
        cost: "≈$0.0002",
        quality: "trace replays reproduce the reading",
        weakness: "trace gap → flagged unexplainable",
      },
    ],
  },
  {
    title: "Ops Citadel",
    emoji: "🏰",
    tagline: "How it stays honest",
    places: [
      {
        name: "Evaluation Warehouse",
        icon: "🏗️",
        line: "The exam the system must pass before changing.",
        runtime: "pytest + golden set",
        speed: "4 min suite",
        cost: "≈$0.40 / run",
        quality: "positions exact; faithfulness-scored",
        weakness: "regression → deploy blocked",
      },
      {
        name: "Policy Gate",
        icon: "🛡️",
        line: "The judgement about when not to answer.",
        runtime: "REFUSE/ABSTAIN flag",
        speed: "5ms",
        cost: "$0",
        quality: "refusal classes unit-tested",
        weakness: "over-refusal → logged for review",
      },
      {
        name: "Deployment Harbor",
        icon: "🚢",
        line: "Where new versions come ashore.",
        runtime: "Render (Docker)",
        speed: "cold start ≈50s",
        cost: "$0–7 /mo",
        quality: "health-checked",
        weakness: "failed check → previous build stays live",
      },
    ],
  },
];

const questSteps = [
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

const scoreboard = [
  {
    label: "Astronomy",
    value: "exact",
    note: "Planet positions match the observatories.",
  },
  {
    label: "Rules",
    value: "deterministic",
    note: "The old rules fire the same way every time.",
  },
  {
    label: "Citations",
    value: "100% cited",
    note: "Every sentence points at its source.",
  },
  {
    label: "Judgement",
    value: "gated",
    note: "When evidence is thin, it says so.",
  },
];

const gold = [
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
];

type TabKey =
  | "guides"
  | "signal"
  | "map"
  | "quest"
  | "score"
  | "party"
  | "gold";

const TABS: { key: TabKey; label: string; topic: string; chips: string[] }[] = [
  {
    key: "guides",
    label: "Sky team",
    topic: "the nine grahas of my chart and what each one governs",
    chips: [
      "Which graha is strongest in my chart?",
      "What does my Moon say about my mind?",
      "Explain Rahu like I'm 10",
    ],
  },
  {
    key: "signal",
    label: "Signal",
    topic: "the current influences on my chart and how a signal decomposes",
    chips: [
      "What is influencing me right now?",
      "Why are you confident about this?",
      "What evidence contradicts it?",
    ],
  },
  {
    key: "map",
    label: "World map",
    topic:
      "how this system computes and interprets — engine, rules, retrieval, LLM, gates",
    chips: [
      "How is my chart computed?",
      "What stops the AI from making things up?",
      "What is the citation gate?",
    ],
  },
  {
    key: "quest",
    label: "Quest log",
    topic: "the pipeline a reading travels, step by step",
    chips: [
      "Walk me through one reading of my chart",
      "Where do the verses come from?",
      "What gets deleted and why?",
    ],
  },
  {
    key: "score",
    label: "Scoreboard",
    topic: "how correctness and faithfulness are evaluated",
    chips: [
      "How do you test the astronomy?",
      "What does 100% cited mean?",
      "When do you refuse to answer?",
    ],
  },
  {
    key: "party",
    label: "Party",
    topic: "the automated agents that keep the system honest",
    chips: [
      "Who checks the checker?",
      "What happens when a check fails?",
      "Teach me how dashas work",
    ],
  },
  {
    key: "gold",
    label: "Gold",
    topic: "what readings cost and why",
    chips: [
      "Why does a reading cost two cents?",
      "What is free and what is not?",
      "What would 100k users cost?",
    ],
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
      className={`inline-block text-[10px] uppercase tracking-wider ${tone === "gold" ? "text-amber-300" : "text-slate-400"}`}
    >
      {children}
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

interface ChatMsg {
  role: "user" | "assistant";
  content: string;
  meta?: string;
}

/* The talking assistant. One thread per tab; every answer comes from the
   live engine's /api/chat — retrieval over the verse corpus, then the
   citation verifier — personalized by the birth details from the gate. */
function AskPanel({
  tab,
  birth,
}: {
  tab: (typeof TABS)[number];
  birth: Birth;
}) {
  const [threads, setThreads] = useState<Record<string, ChatMsg[]>>({});
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const thread = threads[tab.key] ?? [];
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [threads, busy]);

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || busy) return;
    setErr("");
    setInput("");
    setBusy(true);
    const next = [...thread, { role: "user" as const, content: q }];
    setThreads((t) => ({ ...t, [tab.key]: next }));
    try {
      const window8 = next.slice(-8);
      const messages = window8.map((m, i) =>
        i === window8.length - 1 && m.role === "user"
          ? {
              role: m.role,
              content: `(The user is on the “${tab.label}” tab, about ${tab.topic}.) ${m.content}`,
            }
          : { role: m.role, content: m.content },
      );
      const r = await fetch(`${DEMO_URL}/api/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages, birth }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
      const reply: ChatMsg = j.refused
        ? {
            role: "assistant",
            content: j.say ?? "I must abstain on that.",
            meta: "refused by the policy gate",
          }
        : {
            role: "assistant",
            content: j.reply || "(nothing survived the verifier)",
            meta: `${j.retrieved?.length ?? 0} sources · ${j.deleted?.length ?? 0} deleted · ${j.model ?? ""}`,
          };
      setThreads((t) => ({ ...t, [tab.key]: [...next, reply] }));
    } catch (e) {
      setErr(
        `${e instanceof Error ? e.message : "request failed"} — if the engine was asleep it takes ~1 min to wake; try again.`,
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Frame className="mt-8 p-4">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-3">
        <PixelTag>ask · {tab.label}</PixelTag>
        <span className="font-mono text-[10px] text-slate-500">
          live engine · retrieval + citation gate · your chart
        </span>
      </div>

      {thread.length === 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {tab.chips.map((c) => (
            <button
              key={c}
              onClick={() => send(c)}
              className="rounded-full border border-slate-600 px-3 py-1.5 text-[12px] text-slate-300 transition hover:border-amber-300 hover:text-amber-300"
            >
              {c}
            </button>
          ))}
        </div>
      )}

      <div className="max-h-80 space-y-3 overflow-y-auto pr-1">
        {thread.map((m, i) => (
          <div
            key={i}
            className={`max-w-[92%] border-2 px-3 py-2 text-[13.5px] leading-relaxed ${
              m.role === "user"
                ? "ml-auto border-amber-300/60 bg-amber-400/10 text-amber-100"
                : "border-slate-700 bg-[#0d1428] text-slate-200"
            }`}
          >
            <p className="whitespace-pre-wrap">{m.content}</p>
            {m.meta && (
              <p className="mt-1.5 font-mono text-[10px] text-slate-500">
                {m.meta}
              </p>
            )}
          </div>
        ))}
        {busy && (
          <div className="max-w-[92%] border-2 border-slate-700 bg-[#0d1428] px-3 py-2 text-[13px] text-slate-400">
            <span className="animate-pulse">consulting the shelf…</span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {err && (
        <p className="mt-2 border-l-2 border-rose-500 pl-2 text-[12px] text-rose-300">
          {err}
        </p>
      )}

      <form
        className="mt-3 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask anything — answers cite their verses…"
          className="w-full border-2 border-slate-600 bg-[#0B1224] px-3 py-2 text-[13.5px] text-slate-100 placeholder:text-slate-600 focus:border-amber-300 focus:outline-none"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          style={PIXEL}
          className="border-2 border-amber-300 bg-amber-400 px-4 text-[11px] uppercase text-slate-900 disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </Frame>
  );
}

/* geo lookup via open-meteo (same source the live instrument uses) */
interface GeoHit {
  name: string;
  latitude: number;
  longitude: number;
  admin1?: string;
  country?: string;
}

function BirthGate({ onCast }: { onCast: (b: Birth) => void }) {
  const [b, setB] = useState<Birth>(SAMPLE_BIRTH);
  const [hits, setHits] = useState<GeoHit[]>([]);
  const [searching, setSearching] = useState(false);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  const set = (k: keyof Birth, v: string) =>
    setB((old) => ({ ...old, [k]: v }));

  const searchPlace = (q: string) => {
    set("place", q);
    if (debounce.current) clearTimeout(debounce.current);
    if (q.trim().length < 2) {
      setHits([]);
      return;
    }
    debounce.current = setTimeout(async () => {
      setSearching(true);
      try {
        const r = await fetch(
          `https://geocoding-api.open-meteo.com/v1/search?count=5&name=${encodeURIComponent(q.trim())}`,
        );
        const j = await r.json();
        setHits(j.results ?? []);
      } catch {
        setHits([]);
      } finally {
        setSearching(false);
      }
    }, 350);
  };

  const pick = (h: GeoHit) => {
    setB((old) => ({
      ...old,
      place: `${h.name}${h.admin1 ? ", " + h.admin1 : ""}`,
      lat: h.latitude.toFixed(2),
      lon: h.longitude.toFixed(2),
    }));
    setHits([]);
  };

  const ready = b.date && b.time && b.lat && b.lon && b.offset;

  return (
    <section className="relative flex min-h-[calc(100vh-8rem)] items-center justify-center px-6 py-12">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(700px 460px at 20% 15%, #2A1F4D55 0%, transparent 70%)," +
            "radial-gradient(800px 500px at 85% 80%, #123A5A44 0%, transparent 70%)",
        }}
      />
      <div className="relative w-full max-w-xl">
        <p
          style={PIXEL}
          className="text-center text-[12px] uppercase tracking-[0.3em] text-amber-400"
        >
          new game · create your chart
        </p>
        <h1 className="mt-3 text-center font-serif text-4xl font-bold text-white sm:text-5xl">
          Understanding Time
        </h1>
        <p className="mt-3 text-center text-sm text-slate-400">
          The universe renders around <em>your</em> sky. Enter when and where
          you were born — the page, and every assistant on it, answers from that
          chart.
        </p>

        <Frame className="mt-8 p-5">
          <div className="relative">
            <PixelTag tone="dim">birth place — just start typing</PixelTag>
            <input
              value={b.place}
              onChange={(e) => searchPlace(e.target.value)}
              className="mt-1 w-full border-2 border-slate-600 bg-[#0B1224] px-3 py-2 font-mono text-sm text-slate-100 focus:border-amber-300 focus:outline-none"
              placeholder="Delhi"
            />
            {hits.length > 0 && (
              <div className="absolute inset-x-0 top-full z-10 mt-1 border-2 border-slate-600 bg-[#0d1428]">
                {hits.map((h) => (
                  <button
                    key={`${h.latitude}${h.longitude}`}
                    type="button"
                    onClick={() => pick(h)}
                    className="block w-full px-3 py-2 text-left text-[13px] text-slate-200 hover:bg-amber-400/10"
                  >
                    <b className="text-amber-300">{h.name}</b>
                    <span className="text-slate-500">
                      {" "}
                      {h.admin1 ? h.admin1 + " · " : ""}
                      {h.country}
                    </span>
                  </button>
                ))}
              </div>
            )}
            {searching && (
              <p className="mt-1 font-mono text-[10px] text-slate-500">
                searching…
              </p>
            )}
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <div>
              <PixelTag tone="dim">date of birth</PixelTag>
              <input
                type="date"
                value={b.date}
                onChange={(e) => set("date", e.target.value)}
                className="mt-1 w-full border-2 border-slate-600 bg-[#0B1224] px-3 py-2 font-mono text-sm text-slate-100 [color-scheme:dark] focus:border-amber-300 focus:outline-none"
              />
            </div>
            <div>
              <PixelTag tone="dim">time of birth</PixelTag>
              <input
                type="time"
                value={b.time}
                onChange={(e) => set("time", e.target.value)}
                className="mt-1 w-full border-2 border-slate-600 bg-[#0B1224] px-3 py-2 font-mono text-sm text-slate-100 [color-scheme:dark] focus:border-amber-300 focus:outline-none"
              />
            </div>
          </div>

          <div className="mt-4 grid grid-cols-3 gap-3">
            {(
              [
                ["offset", "utc offset"],
                ["lat", "lat · auto"],
                ["lon", "lon · auto"],
              ] as const
            ).map(([k, label]) => (
              <div key={k}>
                <PixelTag tone="dim">{label}</PixelTag>
                <input
                  value={b[k]}
                  onChange={(e) => set(k, e.target.value)}
                  className="mt-1 w-full border-2 border-slate-600 bg-[#0B1224] px-3 py-2 font-mono text-sm text-slate-100 focus:border-amber-300 focus:outline-none"
                />
              </div>
            ))}
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button
              onClick={() => ready && onCast(b)}
              disabled={!ready}
              style={PIXEL}
              className="border-2 border-amber-300 bg-amber-400 px-6 py-2.5 text-[12px] uppercase tracking-wider text-slate-900 shadow-[4px_4px_0_#00000088] transition hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_#00000088] disabled:opacity-40"
            >
              ▶ Cast my chart
            </button>
            <button
              onClick={() => onCast(SAMPLE_BIRTH)}
              className="text-[13px] text-slate-400 underline-offset-4 hover:text-white hover:underline"
            >
              or explore with the sample chart
            </button>
          </div>
        </Frame>
        <p className="mt-4 text-center font-mono text-[10px] leading-relaxed text-slate-600">
          Birth details stay in your browser and are sent only to the live
          engine to compute your chart. Readings are for reflection, not
          medical, legal, or financial advice.
        </p>
      </div>
    </section>
  );
}

/* --------------------------------------------------------------- page --- */

export default function VedicAstroDemo() {
  const navigate = useNavigate();
  const [birth, setBirth] = useState<Birth | null>(null);
  const [tabKey, setTabKey] = useState<TabKey>("guides");
  const [guide, setGuide] = useState(guides[4]);
  const [sheet, setSheet] = useState<Sheet | null>(null);
  const [whyOpen, setWhyOpen] = useState(false);
  const [questCount, setQuestCount] = useState(0);
  const questTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(BIRTH_KEY);
      if (saved) setBirth(JSON.parse(saved));
    } catch {
      /* private mode etc. */
    }
  }, []);

  const cast = (b: Birth) => {
    setBirth(b);
    try {
      localStorage.setItem(BIRTH_KEY, JSON.stringify(b));
    } catch {
      /* ignore */
    }
  };

  const startQuest = () => {
    if (questTimer.current) clearInterval(questTimer.current);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setQuestCount(questSteps.length);
      return;
    }
    setQuestCount(0);
    questTimer.current = setInterval(() => {
      setQuestCount((c) => {
        if (c >= questSteps.length) {
          if (questTimer.current) clearInterval(questTimer.current);
          return c;
        }
        return c + 1;
      });
    }, 260);
  };

  useEffect(
    () => () => {
      if (questTimer.current) clearInterval(questTimer.current);
    },
    [],
  );

  const tab = TABS.find((t) => t.key === tabKey)!;

  return (
    <div className="min-h-screen bg-[#05070F] text-slate-100">
      <Navigation />
      <div className="h-24" aria-hidden="true" />

      {!birth ? (
        <>
          <div className="mx-auto max-w-6xl px-6">
            <button
              onClick={() => navigate("/ai-playground")}
              className="flex items-center gap-2 text-sm text-slate-400 transition-colors hover:text-white"
            >
              <ArrowLeft size={16} />
              Back to AI Playground
            </button>
          </div>
          <BirthGate onCast={cast} />
        </>
      ) : (
        <main className="mx-auto max-w-6xl px-6 pb-16">
          {/* chart strip */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b-2 border-slate-800 pb-4">
            <div>
              <PixelTag>your chart</PixelTag>
              <p className="font-mono text-[13px] text-slate-300">
                {birth.place || `${birth.lat}, ${birth.lon}`} · {birth.date} ·{" "}
                {birth.time} ({birth.offset})
              </p>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setBirth(null)}
                className="text-[12px] text-slate-400 underline-offset-4 hover:text-white hover:underline"
              >
                edit birth details
              </button>
              <a
                href={DEMO_URL}
                target="_blank"
                rel="noopener noreferrer"
                style={PIXEL}
                className="border-2 border-amber-300 px-3 py-1.5 text-[10px] uppercase text-amber-300 transition hover:bg-amber-300/10"
              >
                full instrument →
              </a>
            </div>
          </div>

          {/* tab bar */}
          <div className="sticky top-20 z-30 -mx-6 border-b-2 border-slate-800 bg-[#05070F]/95 px-6 backdrop-blur">
            <div className="flex gap-1 overflow-x-auto py-2">
              {TABS.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setTabKey(t.key)}
                  style={PIXEL}
                  className={`whitespace-nowrap px-3.5 py-2 text-[10px] uppercase tracking-wider transition ${
                    tabKey === t.key
                      ? "bg-amber-400 text-slate-900"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* tab content */}
          <div className="pt-8">
            {tabKey === "guides" && (
              <section>
                <h2 className="font-serif text-2xl font-bold text-white sm:text-3xl">
                  Meet your sky team
                </h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-400">
                  Nine guides, one story — yours. Tap a guide, then ask the
                  assistant below where it sits in <em>your</em> chart.
                </p>
                <div className="mt-5 flex gap-2 overflow-x-auto pb-2">
                  {guides.map((g) => (
                    <button
                      key={g.name}
                      onClick={() => setGuide(g)}
                      aria-pressed={guide.name === g.name}
                      className={`flex min-w-[86px] flex-col items-center gap-1 border-2 px-3 py-3 transition ${
                        guide.name === g.name
                          ? "border-amber-300 bg-[#152040] shadow-[0_0_14px_#fbbf2444]"
                          : "border-slate-700 bg-[#101830] hover:border-slate-500"
                      }`}
                    >
                      <span className="text-2xl">{g.emoji}</span>
                      <span style={PIXEL} className="text-[10px] text-white">
                        {g.name}
                      </span>
                      <span className="text-[10px] text-slate-400">
                        {g.role}
                      </span>
                    </button>
                  ))}
                </div>
                <Frame className="mt-3 p-4">
                  <div className="flex items-start gap-3">
                    <span className="text-3xl">{guide.emoji}</span>
                    <div>
                      <PixelTag>
                        {guide.name} · {guide.role}
                      </PixelTag>
                      <p className="mt-1 text-sm leading-relaxed text-slate-200">
                        “{guide.line}”
                      </p>
                    </div>
                  </div>
                </Frame>
              </section>
            )}

            {tabKey === "signal" && (
              <section>
                <h2 className="font-serif text-2xl font-bold text-white sm:text-3xl">
                  Current signal
                </h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-400">
                  The shape every reading takes — a score, its drivers, and a
                  Why? that goes to bedrock. Ask below for <em>your</em> live
                  influences.
                </p>
                <Frame className="mt-5 max-w-xl p-5">
                  <PixelTag>example · career momentum</PixelTag>
                  <div className="mt-1 flex items-center justify-between gap-4">
                    <p className="font-serif text-3xl font-bold text-amber-300">
                      74<span className="text-lg text-slate-400">/100</span>
                    </p>
                    <span className="font-mono text-[11px] text-slate-500">
                      confidence 81%
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-slate-400">
                    Saturn transit · Jupiter influence · dasha transition · 6
                    more signals
                  </p>
                  <button
                    onClick={() => setWhyOpen(!whyOpen)}
                    style={PIXEL}
                    className="mt-3 border-2 border-amber-300 px-4 py-1.5 text-[11px] uppercase tracking-wider text-amber-300 transition hover:bg-amber-300/10"
                  >
                    Why?
                  </button>
                  {whyOpen && (
                    <div className="mt-3 border-t-2 border-dashed border-slate-700 pt-3">
                      <ul className="space-y-1 font-mono text-[13px] text-slate-200">
                        <li className="flex justify-between">
                          <span>Jupiter activation · transit</span>
                          <span className="text-emerald-300">+21</span>
                        </li>
                        <li className="flex justify-between">
                          <span>Saturn position · aspect</span>
                          <span className="text-rose-300">−8</span>
                        </li>
                        <li className="flex justify-between">
                          <span>Dasha alignment · period</span>
                          <span className="text-emerald-300">+17</span>
                        </li>
                      </ul>
                      <p className="mt-2 text-[12px] text-slate-400">
                        Every driver cites its rule and verse — ask below and
                        the engine answers with sources for your chart.
                      </p>
                    </div>
                  )}
                </Frame>
              </section>
            )}

            {tabKey === "map" && (
              <section>
                <h2 className="font-serif text-2xl font-bold text-white sm:text-3xl">
                  The world map
                </h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-400">
                  Four regions; every location opens a character sheet — stats,
                  powers, and its one weakness.
                </p>
                <div className="mt-5 grid gap-4 md:grid-cols-2">
                  {zones.map((zone) => (
                    <Frame key={zone.title} className="p-4">
                      <div className="mb-3 flex items-baseline justify-between">
                        <h3 className="font-serif text-lg font-bold text-white">
                          <span className="mr-2">{zone.emoji}</span>
                          {zone.title}
                        </h3>
                        <PixelTag tone="dim">{zone.tagline}</PixelTag>
                      </div>
                      <div className="grid gap-1.5">
                        {zone.places.map((place) => (
                          <button
                            key={place.name}
                            onClick={() => setSheet(place)}
                            className="group flex items-center justify-between gap-3 border-2 border-slate-700 bg-[#0A1122] px-3.5 py-2.5 text-left transition hover:border-amber-300"
                          >
                            <span className="flex items-center gap-2.5">
                              <span className="text-lg">{place.icon}</span>
                              <span>
                                <span className="block text-sm font-bold text-white">
                                  {place.name}
                                </span>
                                <span className="block text-[12px] text-slate-400">
                                  {place.line}
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
            )}

            {tabKey === "quest" && (
              <section>
                <h2 className="font-serif text-2xl font-bold text-white sm:text-3xl">
                  One reading's journey
                </h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-400">
                  Press play and follow one request across the whole world,
                  checkpoint by checkpoint.
                </p>
                <Frame className="mt-5 p-5">
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
                              : "border-slate-800 bg-[#0A1122] opacity-40"
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
                  {questCount >= questSteps.length && (
                    <p
                      style={PIXEL}
                      className="mt-4 text-center text-[12px] uppercase tracking-widest text-amber-300"
                    >
                      ★ Quest complete — every sentence cited ★
                    </p>
                  )}
                </Frame>
              </section>
            )}

            {tabKey === "score" && (
              <section>
                <h2 className="font-serif text-2xl font-bold text-white sm:text-3xl">
                  How we know it works
                </h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-400">
                  Different questions get different exams — being right about
                  the sky is not the same as being faithful to the books.
                </p>
                <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  {scoreboard.map((m) => (
                    <Frame key={m.label} className="p-4">
                      <PixelTag tone="dim">{m.label}</PixelTag>
                      <p
                        style={PIXEL}
                        className="mt-1 text-[13px] uppercase text-emerald-300"
                      >
                        {m.value}
                      </p>
                      <p className="mt-1.5 text-[12px] leading-relaxed text-slate-400">
                        {m.note}
                      </p>
                    </Frame>
                  ))}
                </div>
              </section>
            )}

            {tabKey === "party" && (
              <section>
                <h2 className="font-serif text-2xl font-bold text-white sm:text-3xl">
                  Who watches the watchers
                </h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-400">
                  Small companions that keep the world honest while nobody
                  plays.
                </p>
                <div className="mt-5 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
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
              </section>
            )}

            {tabKey === "gold" && (
              <section>
                <h2 className="font-serif text-2xl font-bold text-white sm:text-3xl">
                  What a reading costs
                </h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-400">
                  Everything expensive has a price tag on it.
                </p>
                <div className="mt-5 grid gap-3 sm:grid-cols-3">
                  {gold.map((coin) => (
                    <Frame key={coin.label} className="p-4">
                      <PixelTag tone="dim">{coin.label}</PixelTag>
                      <p className="mt-1 font-mono text-2xl font-bold text-amber-300">
                        🪙 {coin.value}
                      </p>
                      <p className="mt-1 text-[12px] text-slate-400">
                        {coin.note}
                      </p>
                    </Frame>
                  ))}
                </div>
              </section>
            )}

            <AskPanel tab={tab} birth={birth} />
          </div>

          <p className="mt-10 text-center text-[11px] leading-relaxed text-slate-600">
            Illustrative figures describe the system's design envelope; every
            assistant answer above comes from the live engine with its sources
            counted and deletions shown. Readings are for reflection, not
            medical, legal, or financial advice.
          </p>
        </main>
      )}

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
              transition={{ duration: 0.18 }}
              className="w-full max-w-lg border-2 border-amber-300/70 bg-[#101830] p-6 shadow-[8px_8px_0_#000000cc]"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="flex h-12 w-12 items-center justify-center border-2 border-slate-600 bg-[#0A1122] text-2xl">
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
                {sheet.line}
              </p>
              <dl className="mt-4 grid grid-cols-3 gap-x-4 gap-y-2.5 border-t-2 border-dashed border-slate-700 pt-4">
                {(
                  [
                    ["class", sheet.runtime],
                    ["speed", sheet.speed],
                    ["upkeep", sheet.cost],
                  ] as const
                ).map(([label, value]) => (
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
