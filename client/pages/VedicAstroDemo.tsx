import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

import Navigation from "@/components/Navigation";

// RENDER_URL is injected at build time via Vite env.
// Defaults to the Render service (see render.yaml: vedic-astro-ai).
const DEMO_URL =
  import.meta.env.VITE_VEDIC_ASTRO_URL ??
  "https://vedic-astro-ai-4t2k.onrender.com";

type Mode = "story" | "evidence" | "system";

/* ================================================================ data == */

const GOLD = "#E8B44C";
const BLUE = "#66B8FF";

interface HouseInfo {
  n: number;
  themes: string;
  planets: { g: string; name: string; note: string }[];
}

const houseData: Record<number, HouseInfo> = {
  1: {
    n: 1,
    themes: "Self · Body · Beginnings",
    planets: [{ g: "Sū", name: "Surya", note: "identity anchored, visible" }],
  },
  2: {
    n: 2,
    themes: "Wealth · Speech · Family",
    planets: [{ g: "Bu", name: "Budha", note: "quick, trade-minded speech" }],
  },
  3: { n: 3, themes: "Courage · Siblings · Effort", planets: [] },
  4: {
    n: 4,
    themes: "Home · Heart · Foundations",
    planets: [{ g: "Ch", name: "Chandra", note: "the mind rests at home" }],
  },
  5: {
    n: 5,
    themes: "Creativity · Children · Play",
    planets: [{ g: "Śu", name: "Shukra", note: "art comes easily" }],
  },
  6: { n: 6, themes: "Work · Health · Rivals", planets: [] },
  7: {
    n: 7,
    themes: "Partnership · Contracts",
    planets: [{ g: "Ma", name: "Mangala", note: "heat in partnerships" }],
  },
  8: { n: 8, themes: "Depth · Change · Research", planets: [] },
  9: {
    n: 9,
    themes: "Fortune · Teachers · Dharma",
    planets: [{ g: "Ke", name: "Ketu", note: "detached from doctrine" }],
  },
  10: {
    n: 10,
    themes: "Career · Status · Public life",
    planets: [
      { g: "Gu", name: "Guru", note: "supportive transit · +21" },
      { g: "Śa", name: "Shani", note: "structural restraint · −8" },
    ],
  },
  11: { n: 11, themes: "Gains · Networks · Allies", planets: [] },
  12: {
    n: 12,
    themes: "Loss · Rest · Release",
    planets: [{ g: "Ra", name: "Rahu", note: "restless in retreat" }],
  },
};

const PLANET_GLYPH: Record<string, string> = {
  Surya: "☉",
  Chandra: "☾",
  Mangala: "♂",
  Budha: "☿",
  Guru: "♃",
  Shukra: "♀",
  Shani: "♄",
  Rahu: "☊",
  Ketu: "☋",
};

/* North Indian chart geometry — viewBox 0 0 400 400.
   Outer square, both diagonals, inner diamond of edge midpoints. */
const P = {
  TL: [0, 0],
  TR: [400, 0],
  BR: [400, 400],
  BL: [0, 400],
  Tm: [200, 0],
  Rm: [400, 200],
  Bm: [200, 400],
  Lm: [0, 200],
  C: [200, 200],
  a: [100, 100],
  b: [300, 100],
  c: [300, 300],
  d: [100, 300],
};

const housePolys: Record<number, readonly (readonly number[])[]> = {
  1: [P.Tm, P.b, P.C, P.a],
  2: [P.TL, P.Tm, P.a],
  3: [P.TL, P.a, P.Lm],
  4: [P.Lm, P.a, P.C, P.d],
  5: [P.Lm, P.d, P.BL],
  6: [P.BL, P.d, P.Bm],
  7: [P.Bm, P.d, P.C, P.c],
  8: [P.Bm, P.c, P.BR],
  9: [P.BR, P.c, P.Rm],
  10: [P.Rm, P.c, P.C, P.b],
  11: [P.Rm, P.b, P.TR],
  12: [P.TR, P.b, P.Tm],
};

const houseLabelPos: Record<number, [number, number]> = {
  1: [200, 92],
  2: [100, 40],
  3: [40, 100],
  4: [92, 200],
  5: [40, 300],
  6: [100, 360],
  7: [200, 308],
  8: [300, 360],
  9: [360, 300],
  10: [308, 200],
  11: [360, 100],
  12: [300, 40],
};

const planetPos: Record<number, [number, number]> = {
  1: [200, 130],
  2: [140, 55],
  4: [130, 215],
  5: [55, 285],
  7: [200, 262],
  9: [345, 320],
  10: [292, 168],
  12: [318, 62],
};

/* signals over time, 2025 → 2030 */
const YEARS = [2025, 2026, 2027, 2028, 2029, 2030];
const signalTracks: Record<string, number[]> = {
  Career: [74, 78, 88, 84, 80, 76],
  Relationships: [61, 55, 42, 49, 58, 63],
  Finance: [67, 70, 73, 71, 75, 78],
};
const dashaByYear = [
  "Śani mahā · Budha antara",
  "Śani mahā · Ketu antara",
  "Budha mahā · Budha antara",
  "Budha mahā · Ketu antara",
  "Budha mahā · Śukra antara",
  "Budha mahā · Śukra antara",
];

const evidenceRows = [
  ["Jupiter transit", "+21"],
  ["10th-lord strength", "+18"],
  ["Current dasha", "+17"],
  ["Saturn restriction", "−8"],
  ["Mars pressure", "−5"],
  ["Other signals", "+31"],
];

const placements = [
  ["Surya", "12°04′ Siṃha", "H1"],
  ["Chandra", "27°41′ Vṛścika", "H4"],
  ["Budha", "03°18′ Kanyā", "H2"],
  ["Guru", "14°23′ Vṛṣabha", "H10"],
  ["Śani", "19°55′ Vṛṣabha", "H10"],
  ["Śukra", "08°12′ Dhanu", "H5"],
  ["Maṅgala", "22°37′ Kumbha", "H7"],
  ["Rāhu", "05°49′ Karka", "H12"],
  ["Ketu", "05°49′ Makara", "H9"],
];

interface Entity {
  key: string;
  glyph: string;
  name: string;
  orbit: string[];
  house: string;
  sign: string;
  nak: string;
  strength: number;
  transit: string;
}

const entities: Entity[] = [
  {
    key: "guru",
    glyph: "♃",
    name: "Guru — Jupiter",
    orbit: ["Expansion", "Wisdom", "Fortune", "Teachers"],
    house: "X",
    sign: "Vṛṣabha",
    nak: "Rohiṇī",
    strength: 84,
    transit: "High — direct over natal midheaven",
  },
  {
    key: "shani",
    glyph: "♄",
    name: "Śani — Saturn",
    orbit: ["Discipline", "Delay", "Structure", "Responsibility"],
    house: "X",
    sign: "Vṛṣabha",
    nak: "Kṛttikā",
    strength: 72,
    transit: "High — slow pass, restraining",
  },
  {
    key: "surya",
    glyph: "☉",
    name: "Sūrya — Sun",
    orbit: ["Identity", "Vitality", "Authority", "Visibility"],
    house: "I",
    sign: "Siṃha",
    nak: "Maghā",
    strength: 78,
    transit: "Moderate",
  },
  {
    key: "chandra",
    glyph: "☾",
    name: "Chandra — Moon",
    orbit: ["Mind", "Mood", "Memory", "Home"],
    house: "IV",
    sign: "Vṛścika",
    nak: "Jyeṣṭhā",
    strength: 58,
    transit: "Fast — colors the day",
  },
  {
    key: "mangala",
    glyph: "♂",
    name: "Maṅgala — Mars",
    orbit: ["Drive", "Conflict", "Courage", "Heat"],
    house: "VII",
    sign: "Kumbha",
    nak: "Śatabhiṣā",
    strength: 61,
    transit: "Short-term tension",
  },
];

/* aspect network */
const netNodes: Record<string, [number, number, string]> = {
  Jupiter: [210, 42, "♃"],
  Sun: [78, 150, "☉"],
  Moon: [340, 140, "☾"],
  Saturn: [140, 268, "♄"],
  Mars: [292, 270, "♂"],
};
const netEdges: {
  a: string;
  b: string;
  kind: "conjunction" | "opposition" | "trine";
  note: string;
  strength: number;
}[] = [
  {
    a: "Jupiter",
    b: "Moon",
    kind: "trine",
    note: "supportive influence",
    strength: 0.72,
  },
  {
    a: "Jupiter",
    b: "Saturn",
    kind: "conjunction",
    note: "growth under constraint",
    strength: 0.81,
  },
  {
    a: "Sun",
    b: "Saturn",
    kind: "opposition",
    note: "authority vs. duty",
    strength: 0.44,
  },
  {
    a: "Moon",
    b: "Mars",
    kind: "opposition",
    note: "mood under pressure",
    strength: 0.39,
  },
  {
    a: "Sun",
    b: "Jupiter",
    kind: "trine",
    note: "confidence, blessed",
    strength: 0.66,
  },
];

/* life constellation */
const constellation = [
  {
    name: "CAREER",
    x: 300,
    y: 60,
    s: 0.92,
    pulse: true,
    drivers: ["♃ Guru", "♄ Śani", "daśā"],
  },
  {
    name: "PURPOSE",
    x: 505,
    y: 130,
    s: 0.72,
    pulse: false,
    drivers: ["☉ Sūrya", "♃ Guru"],
  },
  {
    name: "LOVE",
    x: 545,
    y: 300,
    s: 0.55,
    pulse: false,
    drivers: ["♀ Śukra", "♂ Maṅgala"],
  },
  {
    name: "CREATIVITY",
    x: 420,
    y: 420,
    s: 0.62,
    pulse: false,
    drivers: ["♀ Śukra", "☾ Chandra"],
  },
  {
    name: "HOME",
    x: 180,
    y: 420,
    s: 0.5,
    pulse: false,
    drivers: ["☾ Chandra"],
  },
  {
    name: "MONEY",
    x: 60,
    y: 295,
    s: 0.68,
    pulse: false,
    drivers: ["☿ Budha", "♃ Guru"],
  },
  {
    name: "HEALTH",
    x: 95,
    y: 130,
    s: 0.6,
    pulse: true,
    drivers: ["♂ Maṅgala", "♄ Śani"],
  },
];

/* system galaxy */
const galaxy = [
  {
    key: "eph",
    name: "Ephemeris",
    sub: "the observatory",
    x: 60,
    y: 210,
    tel: "Swiss Ephemeris · arc-second truth",
  },
  {
    key: "chart",
    name: "Chart engine",
    sub: "mechanics",
    x: 185,
    y: 90,
    tel: "31ms · deterministic, never guessed",
  },
  {
    key: "rules",
    name: "Rules",
    sub: "knowledge constellation",
    x: 330,
    y: 180,
    tel: "classical yogas · every rule cites its verse",
  },
  {
    key: "retr",
    name: "Retrieval",
    sub: "the library",
    x: 470,
    y: 82,
    tel: "BPHS corpus · 24 candidates / reading",
  },
  {
    key: "llm",
    name: "Interpretation",
    sub: "the engine",
    x: 610,
    y: 190,
    tel: "≈1.9s · ≈$0.02 · every sentence cited",
  },
  {
    key: "gate",
    name: "Gates",
    sub: "honesty checkpoints",
    x: 740,
    y: 96,
    tel: "citation gate + REFUSE/ABSTAIN policy",
  },
  {
    key: "ans",
    name: "Answer",
    sub: "with its trace",
    x: 855,
    y: 205,
    tel: "full trace kept — the Explanation tab",
  },
];

const askPrompts = [
  "What is changing?",
  "Why do I feel stuck?",
  "What should I pay attention to?",
  "Show the strongest influence",
  "Challenge this interpretation",
  "Explain like I'm 10",
];

/* ============================================================= helpers == */

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function trackAt(track: number[], yearF: number) {
  const i = Math.min(Math.floor(yearF), track.length - 2);
  return Math.round(lerp(track[i], track[i + 1], yearF - i));
}

function Note({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-slate-500">
      {children}
    </p>
  );
}

function ChapterLine({
  k,
  children,
  sub,
}: {
  k: string;
  children: ReactNode;
  sub?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.7 }}
      className="mb-10"
    >
      <Note>{k}</Note>
      <h2 className="mt-4 max-w-4xl font-serif text-4xl font-bold leading-[1.04] tracking-tight text-white sm:text-6xl lg:text-7xl">
        {children}
      </h2>
      {sub && <p className="mt-4 max-w-xl text-[15px] text-slate-400">{sub}</p>}
    </motion.div>
  );
}

/* starfield canvas */
function Starfield() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    let w = (canvas.width = canvas.offsetWidth * devicePixelRatio);
    let h = (canvas.height = canvas.offsetHeight * devicePixelRatio);
    const stars = Array.from({ length: 140 }, () => ({
      r: Math.random() * Math.max(w, h) * 0.55 + 20,
      a: Math.random() * Math.PI * 2,
      sp: (Math.random() * 0.00012 + 0.00003) * (Math.random() > 0.5 ? 1 : -1),
      size: Math.random() * 1.6 + 0.4,
      tw: Math.random() * Math.PI * 2,
    }));
    let raf = 0;
    const draw = (t: number) => {
      ctx.clearRect(0, 0, w, h);
      const cx = w / 2;
      const cy = h * 0.46;
      for (const s of stars) {
        if (!reduced) s.a += s.sp * 16;
        const x = cx + Math.cos(s.a) * s.r;
        const y = cy + Math.sin(s.a) * s.r * 0.62;
        if (x < -10 || x > w + 10 || y < -10 || y > h + 10) continue;
        const alpha = 0.35 + 0.45 * Math.abs(Math.sin(t * 0.001 + s.tw));
        ctx.fillStyle = `rgba(232,224,200,${alpha})`;
        ctx.beginPath();
        ctx.arc(x, y, s.size * devicePixelRatio * 0.7, 0, Math.PI * 2);
        ctx.fill();
      }
      if (!reduced) raf = requestAnimationFrame(draw);
    };
    draw(0);
    const onResize = () => {
      w = canvas.width = canvas.offsetWidth * devicePixelRatio;
      h = canvas.height = canvas.offsetHeight * devicePixelRatio;
      if (reduced) draw(0);
    };
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, []);
  return (
    <canvas
      ref={ref}
      aria-hidden="true"
      className="absolute inset-0 h-full w-full"
    />
  );
}

/* the big chart */
function BirthChart({
  selected,
  onSelect,
  lagnaShift,
  transitYearF,
}: {
  selected: number;
  onSelect: (n: number) => void;
  lagnaShift: number;
  transitYearF: number;
}) {
  const transitAngle = -20 + transitYearF * 24;
  return (
    <svg
      viewBox="-14 -14 428 428"
      className="w-full max-w-[620px]"
      role="group"
      aria-label="Interactive North Indian birth chart"
    >
      <defs>
        <radialGradient id="houseGlow" cx="50%" cy="50%" r="70%">
          <stop offset="0%" stopColor={GOLD} stopOpacity="0.2" />
          <stop offset="100%" stopColor={GOLD} stopOpacity="0" />
        </radialGradient>
        <filter id="soft" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="2.2" />
        </filter>
      </defs>
      {/* luminous outer frame */}
      <rect
        x="0"
        y="0"
        width="400"
        height="400"
        fill="none"
        stroke={GOLD}
        strokeWidth="2.4"
        opacity="0.35"
        filter="url(#soft)"
      />
      <rect
        x="0"
        y="0"
        width="400"
        height="400"
        fill="none"
        stroke="#B9C2E8"
        strokeWidth="0.8"
        opacity="0.8"
      />
      {/* aspect lines + travelling sparks */}
      <g className="astro-dash" stroke={BLUE} strokeWidth="1" opacity="0.55">
        <line x1={292} y1={168} x2={130} y2={215} strokeDasharray="5 6" />
        <line x1={292} y1={168} x2={200} y2={130} strokeDasharray="5 6" />
      </g>
      <g className="astro-particles">
        <circle r="2.4" fill={BLUE} opacity="0.9">
          <animateMotion
            dur="4.2s"
            repeatCount="indefinite"
            path="M292,168 L130,215"
          />
        </circle>
        <circle r="2" fill={GOLD} opacity="0.9">
          <animateMotion
            dur="3.4s"
            repeatCount="indefinite"
            path="M292,168 L200,130"
          />
        </circle>
      </g>
      {Object.entries(housePolys).map(([n, pts]) => {
        const num = Number(n);
        const active = selected === num;
        return (
          <g key={n}>
            <polygon
              points={pts.map((p) => p.join(",")).join(" ")}
              fill={active ? "url(#houseGlow)" : "transparent"}
              stroke={active ? GOLD : "#4A5480"}
              strokeWidth={active ? 2.6 : 1.2}
              className="cursor-pointer transition-all duration-300"
              style={{
                opacity: selected && !active ? 0.32 : 1,
                filter: active ? `drop-shadow(0 0 10px ${GOLD}99)` : undefined,
              }}
              onMouseEnter={() => onSelect(num)}
              onClick={() => onSelect(num)}
            />
            <text
              x={houseLabelPos[num][0]}
              y={houseLabelPos[num][1]}
              textAnchor="middle"
              fontSize="11"
              fill={active ? GOLD : "#5A6390"}
              className="pointer-events-none select-none font-mono"
            >
              {num}
            </text>
            {houseData[num].planets.map((p, pi) => {
              const [bx, by] = planetPos[num] ?? houseLabelPos[num];
              const off = (pi - (houseData[num].planets.length - 1) / 2) * 26;
              return (
                <text
                  key={p.name}
                  x={bx + off}
                  y={by}
                  textAnchor="middle"
                  fontSize="19"
                  fill={GOLD}
                  className="pointer-events-none select-none"
                  style={{
                    opacity: selected && !active ? 0.3 : 1,
                    filter: `drop-shadow(0 0 6px ${GOLD}AA)`,
                  }}
                >
                  {PLANET_GLYPH[p.name] ?? p.g}
                </text>
              );
            })}
          </g>
        );
      })}
      {/* lagna tick — rotates with the what-if birth-time slider */}
      <g transform={`rotate(${lagnaShift} 200 200)`}>
        <line x1="200" y1="8" x2="200" y2="26" stroke={GOLD} strokeWidth="3" />
      </g>
      {/* transit marker — moves with the time machine */}
      <g transform={`rotate(${transitAngle} 200 200)`}>
        <circle cx="388" cy="200" r="5" fill={BLUE}>
          <title>transiting Guru</title>
        </circle>
      </g>
      <circle
        className="youring"
        cx="200"
        cy="200"
        r="10"
        fill="none"
        stroke={GOLD}
        strokeWidth="1"
      />
      <text
        x="200"
        y="205"
        textAnchor="middle"
        fontSize="12"
        fill="#C7CEE8"
        className="pointer-events-none select-none font-mono"
      >
        YOU
      </text>
    </svg>
  );
}

/* ================================================================ page == */

export default function VedicAstroDemo() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>("story");
  const [house, setHouse] = useState(10);
  const [whyOpen, setWhyOpen] = useState(false);
  const [whyNode, setWhyNode] = useState<string | null>(null);
  const [yearF, setYearF] = useState(0); // 0..5 → 2025..2030
  const [birthShift, setBirthShift] = useState(15); // minutes 0..30
  const [entity, setEntity] = useState<Entity | null>(null);
  const [edgeInfo, setEdgeInfo] = useState<(typeof netEdges)[number] | null>(
    null,
  );
  const [starSel, setStarSel] = useState<string | null>(null);
  const [orbOpen, setOrbOpen] = useState(false);
  const [sysSel, setSysSel] = useState<(typeof galaxy)[number] | null>(null);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const onHeroMove = useCallback((e: React.MouseEvent<HTMLElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setTilt({
      x: ((e.clientX - r.left) / r.width - 0.5) * 8,
      y: ((e.clientY - r.top) / r.height - 0.5) * -8,
    });
  }, []);

  const year = 2025 + yearF;
  const signals = useMemo(
    () =>
      Object.entries(signalTracks).map(([name, track]) => ({
        name,
        base: track[0],
        now: trackAt(track, yearF),
      })),
    [yearF],
  );

  const rel = Math.round(
    63 - Math.max(0, birthShift - 12) * 1.6 - Math.max(0, 6 - birthShift) * 0.8,
  );
  const relSensitive = rel < 50;

  const scrollToId = (id: string) => {
    const el = document.getElementById(id);
    if (el)
      window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 64);
  };

  const evid = mode !== "story";

  return (
    <div className="min-h-screen overflow-x-hidden bg-[#05070F] text-slate-200">
      <style>{`
        .astro-dash line{stroke-dashoffset:0;animation:astrodash 3.2s linear infinite}
        @keyframes astrodash{to{stroke-dashoffset:-44}}
        .orb-pulse{animation:orbp 3.4s ease-in-out infinite}
        @keyframes orbp{50%{box-shadow:0 0 64px 18px #E8B44C55, 0 0 120px 40px #66B8FF22}}
        .star-pulse{animation:starp 2.6s ease-in-out infinite}
        @keyframes starp{50%{opacity:.55}}
        .flow-dash{stroke-dasharray:6 10;animation:flowd 2.4s linear infinite}
        @keyframes flowd{to{stroke-dashoffset:-64}}
        @media (prefers-reduced-motion: reduce){
          .astro-dash line,.orb-pulse,.star-pulse,.flow-dash{animation:none}
        }
        input[type=range].cosmic{appearance:none;height:3px;border-radius:2px;
          background:linear-gradient(90deg,#3A4368,#E8B44C);outline:none}
        input[type=range].cosmic::-webkit-slider-thumb{appearance:none;width:18px;
          height:18px;border-radius:50%;background:#E8B44C;border:2px solid #05070F;
          box-shadow:0 0 12px #E8B44C99;cursor:grab}
        .floaty{animation:floaty 9s ease-in-out infinite}
        .floaty.f2{animation-duration:13s;animation-delay:-4s}
        .floaty.f3{animation-duration:11s;animation-delay:-7s}
        @keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-16px)}}
        .chev{animation:chev 2.2s ease-in-out infinite}
        @keyframes chev{0%,100%{transform:translateY(0);opacity:.5}50%{transform:translateY(7px);opacity:1}}
        .shoot{position:absolute;top:16%;left:-70px;width:70px;height:1.5px;
          transform:rotate(14deg);border-radius:2px;
          background:linear-gradient(90deg,transparent,#E8B44C99,#66B8FF);
          animation:shoot 11s linear infinite}
        .shoot.s2{top:64%;animation-delay:5.4s;animation-duration:14s}
        @keyframes shoot{0%,87%{opacity:0;transform:translateX(0) rotate(14deg)}
          90%{opacity:1}100%{opacity:0;transform:translateX(120vw) rotate(14deg)}}
        .youring{animation:youring 3.4s ease-out infinite}
        @keyframes youring{0%{r:8;opacity:.7}100%{r:26;opacity:0}}
        @media (prefers-reduced-motion: reduce){
          .floaty,.chev,.shoot,.youring{animation:none}
          .astro-particles{display:none}
        }
      `}</style>

      <Navigation />

      {/* mode control */}
      <div className="fixed right-4 top-24 z-40 flex gap-0.5 rounded-full border border-white/15 bg-[#0B1020]/80 p-1 backdrop-blur">
        {(["story", "evidence", "system"] as Mode[]).map((value) => (
          <button
            key={value}
            onClick={() => {
              setMode(value);
              if (value === "system") scrollToId("machine");
            }}
            className={`rounded-full px-3.5 py-1.5 font-mono text-[11px] uppercase tracking-wider transition ${
              mode === value
                ? "bg-[#E8B44C] text-[#05070F]"
                : "text-slate-400 hover:text-white"
            }`}
          >
            {value}
          </button>
        ))}
      </div>

      {/* ------------------------------------------------------- HERO ---- */}
      <section
        onMouseMove={onHeroMove}
        className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6"
      >
        {/* nebulae */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(720px 460px at 18% 18%, #2A1F4D55 0%, transparent 70%)," +
              "radial-gradient(820px 520px at 84% 74%, #123A5A44 0%, transparent 70%)," +
              "radial-gradient(520px 380px at 70% 12%, #4A2A1233 0%, transparent 70%)",
          }}
        />
        <Starfield />
        <span aria-hidden="true" className="shoot" />
        <span aria-hidden="true" className="shoot s2" />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 transition-transform duration-300 ease-out"
          style={{
            transform: `perspective(900px) rotateY(${tilt.x}deg) rotateX(${tilt.y}deg)`,
          }}
        >
          {[320, 500, 700, 920].map((size, i) => (
            <span
              key={size}
              className="absolute left-1/2 top-[46%] -translate-x-1/2 -translate-y-1/2 rounded-full border"
              style={{
                width: size,
                height: size * 0.62,
                borderColor: i % 2 ? "#E8B44C2E" : "#66B8FF26",
                boxShadow: i === 1 ? "0 0 40px #E8B44C11 inset" : undefined,
              }}
            />
          ))}
          <span className="floaty absolute left-[20%] top-[28%] text-2xl text-[#E8B44C]/80 [text-shadow:0_0_14px_#E8B44C88]">
            ♃
          </span>
          <span className="floaty f2 absolute right-[18%] top-[22%] text-xl text-[#66B8FF]/70 [text-shadow:0_0_12px_#66B8FF88]">
            ♄
          </span>
          <span className="floaty f3 absolute bottom-[24%] right-[28%] text-lg text-[#E8E4D8]/70 [text-shadow:0_0_10px_#E8E4D888]">
            ☾
          </span>
          <span className="floaty f2 absolute bottom-[32%] left-[26%] text-base text-[#F2846B]/60 [text-shadow:0_0_10px_#F2846B66]">
            ♂
          </span>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2 }}
          className="relative z-10 text-center"
        >
          <div className="mx-auto mb-6 h-px w-24 bg-gradient-to-r from-transparent via-[#E8B44C] to-transparent" />
          <h1 className="font-serif text-5xl font-bold tracking-[0.1em] text-white [text-shadow:0_0_44px_#E8B44C33] sm:text-7xl lg:text-8xl">
            VEDIC ASTRO AI
          </h1>
          <p className="mx-auto mt-5 max-w-xl font-serif text-xl italic text-slate-300 sm:text-2xl">
            Your life, interpreted as a moving system.
          </p>
          <div className="mt-10 flex flex-wrap items-center justify-center gap-6">
            <a
              href={DEMO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full border border-[#E8B44C] bg-[#E8B44C]/10 px-9 py-3.5 font-serif text-lg text-[#E8B44C] shadow-[0_0_36px_#E8B44C3D] backdrop-blur transition hover:bg-[#E8B44C] hover:text-[#05070F] hover:shadow-[0_0_54px_#E8B44C66]"
            >
              Enter your universe →
            </a>
            <button
              onClick={() => scrollToId("ch1")}
              className="text-sm text-slate-400 underline-offset-4 transition hover:text-white hover:underline"
            >
              Explore how it works
            </button>
          </div>
        </motion.div>
        <div className="absolute bottom-7 z-10 flex flex-col items-center gap-2">
          <p className="px-4 text-center font-mono text-[11px] tracking-[0.18em] text-slate-500">
            9 planets · 12 houses · 27 nakshatras · live transits · AI
            interpretation
          </p>
          <span aria-hidden="true" className="chev text-[#E8B44C]">
            ↓
          </span>
        </div>
      </section>

      <main className="relative mx-auto max-w-6xl px-6">
        {/* -------------------------------------------------- CH 01 · YOU -- */}
        <section id="ch1" className="py-24">
          <ChapterLine k="chapter 01 · you">
            YOU WERE BORN INTO A&nbsp;PARTICULAR&nbsp;SKY.
          </ChapterLine>

          <div className="flex flex-col items-start gap-10 lg:flex-row">
            <div className="w-full lg:w-[58%]">
              <BirthChart
                selected={house}
                onSelect={(n) => {
                  setHouse(n);
                  setWhyOpen(false);
                }}
                lagnaShift={(birthShift - 15) * 0.9}
                transitYearF={yearF}
              />
              <p className="mt-3 font-mono text-[11px] text-slate-500">
                a sample chart · click any house — the universe dims around it
              </p>
            </div>

            {/* house panel — enter-only animation: exit-gated swaps freeze
                in throttled tabs, so never block content on an exit. */}
            <div className="w-full lg:w-[42%] lg:pt-10">
              <motion.div
                key={house}
                initial={{ opacity: 0, x: 24 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Note>house {house}</Note>
                <h3 className="mt-2 font-serif text-3xl font-bold text-white">
                  {houseData[house].themes.split("·")[0].trim()}
                </h3>
                <p className="mt-1 text-sm text-slate-400">
                  {houseData[house].themes}
                </p>
                <div className="mt-5 space-y-2.5">
                  {houseData[house].planets.length === 0 && (
                    <p className="text-sm italic text-slate-500">
                      No graha resides here — this house answers to its lord.
                    </p>
                  )}
                  {houseData[house].planets.map((p) => (
                    <div
                      key={p.g}
                      className="flex items-baseline gap-3 border-l-2 border-[#E8B44C]/50 pl-3"
                    >
                      <span className="font-serif text-lg text-[#E8B44C]">
                        {p.name}
                      </span>
                      <span className="text-[13px] text-slate-400">
                        {p.note}
                      </span>
                    </div>
                  ))}
                </div>

                {house === 10 && (
                  <div className="mt-7 border-t border-white/10 pt-5">
                    <Note>current signal</Note>
                    <div className="mt-2 flex items-baseline gap-4">
                      <span className="font-serif text-2xl text-white">
                        Career momentum
                      </span>
                      <span className="font-mono text-3xl font-bold text-[#E8B44C]">
                        {signals[0].now}
                        <span className="text-base text-slate-500">/100</span>
                      </span>
                    </div>
                    <p className="mt-1 font-mono text-[11px] text-slate-500">
                      confidence 81% · {dashaByYear[Math.round(yearF)]}
                    </p>
                    <button
                      onClick={() => setWhyOpen(!whyOpen)}
                      className="mt-4 rounded-full border border-[#66B8FF]/60 px-6 py-2 font-serif text-[#66B8FF] transition hover:bg-[#66B8FF]/10"
                    >
                      Why?
                    </button>
                  </div>
                )}
              </motion.div>
            </div>
          </div>

          {/* the explosion */}
          <AnimatePresence>
            {whyOpen && house === 10 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-10 border-t border-white/10 pt-8">
                  <Note>the signal, decomposed</Note>
                  <svg viewBox="0 0 480 240" className="mt-4 w-full max-w-2xl">
                    <motion.line
                      x1="240"
                      y1="46"
                      x2="110"
                      y2="120"
                      stroke={BLUE}
                      strokeWidth="1.4"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.6 }}
                    />
                    <motion.line
                      x1="240"
                      y1="46"
                      x2="240"
                      y2="120"
                      stroke={BLUE}
                      strokeWidth="1.4"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.6, delay: 0.1 }}
                    />
                    <motion.line
                      x1="240"
                      y1="46"
                      x2="370"
                      y2="120"
                      stroke={BLUE}
                      strokeWidth="1.4"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.6, delay: 0.2 }}
                    />
                    <text
                      x="240"
                      y="30"
                      textAnchor="middle"
                      fill="#fff"
                      fontSize="17"
                      className="font-serif"
                    >
                      CAREER {signals[0].now}
                    </text>
                    {[
                      ["Jupiter", "+21", 110, "transit"],
                      ["Saturn", "−8", 240, "aspect"],
                      ["Dasha", "+17", 370, "period"],
                    ].map(([name, v, x, kind]) => (
                      <g
                        key={name as string}
                        className="cursor-pointer"
                        onClick={() =>
                          setWhyNode(whyNode === name ? null : (name as string))
                        }
                      >
                        <circle
                          cx={x as number}
                          cy={140}
                          r="34"
                          fill={whyNode === name ? "#E8B44C22" : "#0B1020"}
                          stroke={whyNode === name ? GOLD : "#3A4368"}
                        />
                        <text
                          x={x as number}
                          y={136}
                          textAnchor="middle"
                          fill="#E8E4D8"
                          fontSize="12"
                        >
                          {name}
                        </text>
                        <text
                          x={x as number}
                          y={154}
                          textAnchor="middle"
                          fill={
                            (v as string).startsWith("+")
                              ? "#7BE0A8"
                              : "#F2846B"
                          }
                          fontSize="13"
                          className="font-mono"
                        >
                          {v}
                        </text>
                        <text
                          x={x as number}
                          y={196}
                          textAnchor="middle"
                          fill="#5A6390"
                          fontSize="10"
                          className="font-mono"
                        >
                          {kind}
                        </text>
                      </g>
                    ))}
                  </svg>
                  {whyNode && (
                    <p className="max-w-xl border-l-2 border-[#E8B44C]/60 pl-4 text-sm text-slate-300">
                      {whyNode === "Jupiter" &&
                        "Transiting Guru at 14°23′ Vṛṣabha crosses the natal midheaven — the classical expansion signal for the tenth. Cited: BPHS, career chapter; retrieval rank #1 of 24."}
                      {whyNode === "Saturn" &&
                        "Natal Śani in the tenth restrains as it structures: −8, not a denial but a governor. Cited rule surfaces alongside, never silently merged."}
                      {whyNode === "Dasha" &&
                        "The running daśā lends its period lord to career matters: +17 while the antara holds. Computed from the Vimshottari tree, not estimated."}
                    </p>
                  )}
                  {evid && (
                    <div className="mt-6 max-w-md">
                      <Note>evidence · full decomposition</Note>
                      <table className="mt-2 w-full font-mono text-[13px]">
                        <tbody>
                          {evidenceRows.map(([k, v]) => (
                            <tr key={k} className="border-b border-white/5">
                              <td className="py-1.5 text-slate-400">{k}</td>
                              <td
                                className={`py-1.5 text-right ${v.startsWith("+") ? "text-emerald-300" : "text-rose-300"}`}
                              >
                                {v}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {/* ------------------------------------------- CH 02 · THE SKY NOW -- */}
        <section className="border-t border-white/5 py-24">
          <ChapterLine
            k="chapter 02 · the sky now"
            sub="Left: the sky you were born under, fixed forever. Right: the sky tonight — and where it touches yours."
          >
            BUT THE SKY KEPT MOVING.
          </ChapterLine>
          <div className="grid items-center gap-10 md:grid-cols-[1fr_auto_1fr]">
            <div className="md:text-right">
              <Note>your natal sky · fixed forever</Note>
              <p className="mt-3 font-serif text-3xl leading-snug text-white sm:text-4xl">
                Guru &amp; Śani,
                <br />
                seated in the tenth.
              </p>
              <p className="ml-auto mt-3 max-w-xs text-sm text-slate-400 md:mr-0">
                The reference frame every transit is judged against.
              </p>
            </div>
            <div
              aria-hidden="true"
              className="hidden flex-col items-center gap-1 md:flex"
            >
              <span className="text-2xl text-[#66B8FF] [text-shadow:0_0_12px_#66B8FF]">
                ♃
              </span>
              <span className="h-16 w-px bg-gradient-to-b from-[#66B8FF] to-[#E8B44C]" />
              <span className="chev text-[#E8B44C]">↓</span>
            </div>
            <div>
              <Note>today's sky · in motion</Note>
              <p className="mt-3 font-serif text-3xl leading-snug text-white sm:text-4xl">
                Today's Jupiter
                <br />
                crosses your midheaven.
              </p>
              <p className="mt-3 font-mono text-[13px] tracking-[0.14em] text-[#66B8FF]">
                TODAY'S ♃ → YOUR 10TH HOUSE · EXPANSION SIGNAL
              </p>
              <p className="mt-2 max-w-xs text-sm text-slate-400">
                This crossing is the +21 in the career decomposition above.
              </p>
            </div>
          </div>
        </section>

        {/* ------------------------------------------- CH 03 · THE FORCES -- */}
        <section className="border-t border-white/5 py-24">
          <ChapterLine
            k="chapter 03 · the forces"
            sub="Planets converse. Hover a line to hear what it says."
          >
            NINE BODIES, ONE&nbsp;CONVERSATION.
          </ChapterLine>

          <div className="flex flex-col gap-10 lg:flex-row">
            <svg viewBox="0 0 420 320" className="w-full max-w-lg">
              {netEdges.map((e) => {
                const [ax, ay] = netNodes[e.a];
                const [bx, by] = netNodes[e.b];
                const active = edgeInfo === e;
                return (
                  <line
                    key={e.a + e.b}
                    x1={ax}
                    y1={ay}
                    x2={bx}
                    y2={by}
                    stroke={
                      e.kind === "opposition"
                        ? "#F2846B"
                        : e.kind === "trine"
                          ? BLUE
                          : GOLD
                    }
                    strokeWidth={active ? 3 : 1 + e.strength * 1.6}
                    strokeDasharray={
                      e.kind === "opposition" ? "6 6" : undefined
                    }
                    opacity={active ? 1 : 0.55}
                    className="cursor-pointer transition-all"
                    onMouseEnter={() => setEdgeInfo(e)}
                    onClick={() => setEdgeInfo(e)}
                  />
                );
              })}
              {Object.entries(netNodes).map(([name, [x, y, glyph]]) => (
                <g key={name}>
                  <circle
                    cx={x}
                    cy={y}
                    r="24"
                    fill="#0B1020"
                    stroke="#3A4368"
                  />
                  <text
                    x={x}
                    y={y + 1}
                    textAnchor="middle"
                    fill={GOLD}
                    fontSize="16"
                  >
                    {glyph}
                  </text>
                  <text
                    x={x}
                    y={y + 40}
                    textAnchor="middle"
                    fill="#8890B8"
                    fontSize="11"
                  >
                    {name}
                  </text>
                </g>
              ))}
            </svg>
            <div className="lg:pt-8">
              <AnimatePresence>
                {edgeInfo ? (
                  <motion.div
                    key={edgeInfo.a + edgeInfo.b}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <Note>{edgeInfo.kind}</Note>
                    <p className="mt-2 font-serif text-2xl text-white">
                      {edgeInfo.a} → {edgeInfo.b}
                    </p>
                    <p className="mt-1 text-slate-400">{edgeInfo.note}</p>
                    <p className="mt-2 font-mono text-sm text-[#66B8FF]">
                      strength {edgeInfo.strength.toFixed(2)}
                    </p>
                  </motion.div>
                ) : (
                  <motion.p
                    key="hint"
                    className="text-sm italic text-slate-500"
                  >
                    hover a line…
                  </motion.p>
                )}
              </AnimatePresence>

              {/* entities */}
              <div className="mt-10">
                <Note>or meet a body directly</Note>
                <div className="mt-3 flex gap-3">
                  {entities.map((en) => (
                    <button
                      key={en.key}
                      onClick={() => setEntity(en)}
                      className="flex h-12 w-12 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] text-xl text-[#E8B44C] transition hover:border-[#E8B44C] hover:shadow-[0_0_18px_#E8B44C44]"
                      aria-label={en.name}
                    >
                      {en.glyph}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------ CH 04 · TIME -- */}
        <section className="border-t border-white/5 py-24">
          <ChapterLine
            k="chapter 04 · time"
            sub="Drag through the years. The chart's transit marker moves; the signals morph."
          >
            MOVE THROUGH TIME.
          </ChapterLine>

          <div className="max-w-3xl">
            <div className="flex items-center gap-5">
              <span className="font-mono text-sm text-slate-400">2025</span>
              <input
                type="range"
                min={0}
                max={5}
                step={0.02}
                value={yearF}
                onChange={(e) => setYearF(Number(e.target.value))}
                className="cosmic w-full"
                aria-label="Move through time"
              />
              <span className="font-mono text-sm text-slate-400">2030</span>
            </div>
            <p className="mt-2 text-center font-serif text-2xl text-[#E8B44C]">
              {year.toFixed(1)}
            </p>
            <p className="mt-1 text-center font-mono text-[11px] uppercase tracking-[0.2em] text-slate-500">
              {dashaByYear[Math.round(yearF)]}
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              {signals.map((s) => {
                const up = s.now >= s.base;
                return (
                  <div key={s.name} className="text-center">
                    <Note>{s.name}</Note>
                    <p className="mt-1 font-mono text-2xl text-white">
                      <span className="text-slate-500">{s.base} → </span>
                      <span
                        className={up ? "text-emerald-300" : "text-rose-300"}
                      >
                        {s.now}
                      </span>
                    </p>
                    {/* confidence as atmosphere */}
                    <div
                      className="mx-auto mt-2 h-2.5 rounded-full"
                      title={`signal ${s.now}/100`}
                      style={{
                        width: `${34 + s.now * 0.5}%`,
                        background: up ? "#7BE0A8" : "#F2846B",
                        filter: `blur(${Math.max(0, (80 - s.now) / 18)}px)`,
                        opacity: 0.85,
                      }}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* --------------------------------------- CH 05 · POSSIBILITIES -- */}
        <section className="border-t border-white/5 py-24">
          <ChapterLine
            k="chapter 05 · possibilities"
            sub="Uncertainty is part of the system — so we show what it does."
          >
            WHAT IF THE INPUT&nbsp;CHANGED?
          </ChapterLine>

          <div className="max-w-3xl">
            <Note>birth time</Note>
            <div className="mt-2 flex items-center gap-5">
              <span className="font-mono text-sm text-slate-400">2:01 PM</span>
              <input
                type="range"
                min={0}
                max={30}
                step={1}
                value={birthShift}
                onChange={(e) => setBirthShift(Number(e.target.value))}
                className="cosmic w-full"
                aria-label="Shift birth time"
              />
              <span className="font-mono text-sm text-slate-400">2:31 PM</span>
            </div>
            <p className="mt-2 font-mono text-[12px] text-slate-500">
              recorded 2:{String(1 + birthShift).padStart(2, "0")} PM — the
              lagna tick on the chart above rotates with you
            </p>

            <table className="mt-6 w-full max-w-md font-mono text-[14px]">
              <tbody>
                {[
                  [
                    "Career",
                    74,
                    74 - Math.round(Math.abs(birthShift - 15) / 7),
                  ],
                  ["Relationships", 63, rel],
                  [
                    "Finance",
                    81,
                    81 - Math.round(Math.abs(birthShift - 15) / 15),
                  ],
                  [
                    "Identity",
                    76,
                    76 - Math.round(Math.abs(birthShift - 15) / 10),
                  ],
                ].map(([name, a, b]) => (
                  <tr key={name as string} className="border-b border-white/5">
                    <td className="py-2 text-slate-300">{name}</td>
                    <td className="py-2 text-right text-slate-500">{a}</td>
                    <td className="w-10 text-center text-slate-600">→</td>
                    <td
                      className={`py-2 text-right ${
                        (b as number) < (a as number) - 8
                          ? "text-rose-300"
                          : "text-slate-200"
                      }`}
                    >
                      {b}
                      {name === "Relationships" && relSensitive && " ⚠"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 max-w-md border-l-2 border-[#F2846B]/70 pl-4 text-sm text-slate-300">
              Relationship interpretation is highly sensitive to birth-time
              uncertainty — the seventh's lord changes sign within this window.
              The system says so instead of guessing.
            </p>
          </div>
        </section>

        {/* --------------------------------------- CH 06 · CONSTELLATION -- */}
        <section className="border-t border-white/5 py-24">
          <ChapterLine
            k="chapter 06 · your constellation"
            sub="Brightness is current strength; pulse is transit pressure. Click a star to see who feeds it."
          >
            AND THOSE MOVEMENTS CREATE DIFFERENT&nbsp;PRESSURES.
          </ChapterLine>

          <svg viewBox="0 0 620 480" className="mx-auto w-full max-w-2xl">
            {starSel &&
              constellation
                .find((c) => c.name === starSel)!
                .drivers.map((d, i) => (
                  <motion.text
                    key={d}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    x={310}
                    y={250 + i * 22}
                    textAnchor="middle"
                    fontSize="13"
                    fill={GOLD}
                  >
                    {d}
                  </motion.text>
                ))}
            {constellation.map((c) => {
              const sel = starSel === c.name;
              return (
                <g
                  key={c.name}
                  className={`cursor-pointer ${c.pulse ? "star-pulse" : ""}`}
                  onClick={() => setStarSel(sel ? null : c.name)}
                >
                  {sel && (
                    <line
                      x1={310}
                      y1={240}
                      x2={c.x}
                      y2={c.y}
                      stroke={GOLD}
                      strokeWidth="1"
                      opacity="0.6"
                    />
                  )}
                  <circle
                    cx={c.x}
                    cy={c.y}
                    r={5 + c.s * 9}
                    fill={sel ? GOLD : "#E8E4D8"}
                    opacity={0.25 + c.s * 0.75}
                    style={{
                      filter: `drop-shadow(0 0 ${4 + c.s * 10}px #E8B44C)`,
                    }}
                  />
                  <text
                    x={c.x}
                    y={c.y - 18}
                    textAnchor="middle"
                    fontSize="12"
                    fill={sel ? GOLD : "#8890B8"}
                    className="font-mono"
                  >
                    {c.name}
                  </text>
                </g>
              );
            })}
            <circle
              cx={310}
              cy={240}
              r="7"
              fill="none"
              stroke="#fff"
              strokeWidth="1.5"
            />
            <text
              x={310}
              y={225}
              textAnchor="middle"
              fontSize="11"
              fill="#fff"
              className="font-mono"
            >
              YOU
            </text>
          </svg>
        </section>

        {/* ------------------------------------------------- CH 07 · ASK -- */}
        <section className="border-t border-white/5 py-24 text-center">
          <ChapterLine k="chapter 07 · ask">ASK YOUR CHART.</ChapterLine>
          <div className="relative mx-auto flex h-72 max-w-lg items-center justify-center">
            <button
              onClick={() => setOrbOpen(!orbOpen)}
              aria-expanded={orbOpen}
              className="orb-pulse relative z-10 h-28 w-28 rounded-full border border-[#E8B44C]/60 transition hover:scale-105"
              style={{
                background:
                  "radial-gradient(circle at 35% 30%, #F5D48A, #E8B44C 45%, #7A5A1E 100%)",
                boxShadow: "0 0 44px #E8B44C55",
              }}
            >
              <span className="sr-only">Ask your chart</span>
            </button>
            <AnimatePresence>
              {orbOpen &&
                askPrompts.map((q, i) => {
                  const angle =
                    (i / askPrompts.length) * Math.PI * 2 - Math.PI / 2;
                  return (
                    <motion.a
                      key={q}
                      href={DEMO_URL}
                      target="_blank"
                      rel="noopener noreferrer"
                      initial={{ opacity: 0, scale: 0.6 }}
                      animate={{
                        opacity: 1,
                        scale: 1,
                        x: Math.cos(angle) * 190,
                        y: Math.sin(angle) * 110,
                      }}
                      exit={{ opacity: 0, scale: 0.6, x: 0, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="absolute hidden whitespace-nowrap rounded-full border border-white/20 bg-[#0B1020]/90 px-4 py-2 text-[13px] text-slate-200 backdrop-blur transition hover:border-[#E8B44C] hover:text-[#E8B44C] md:block"
                    >
                      {q}
                    </motion.a>
                  );
                })}
            </AnimatePresence>
          </div>
          {/* mobile fallback */}
          {orbOpen && (
            <div className="mx-auto flex max-w-md flex-wrap justify-center gap-2 md:hidden">
              {askPrompts.map((q) => (
                <a
                  key={q}
                  href={DEMO_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-full border border-white/20 px-4 py-2 text-[13px] text-slate-200"
                >
                  {q}
                </a>
              ))}
            </div>
          )}
          <p className="mt-4 text-sm text-slate-500">
            every question opens the live instrument — answers arrive with their
            citations
          </p>
        </section>

        {/* -------------------------------------------- CH 08 · MACHINE -- */}
        <section id="machine" className="border-t border-white/5 py-24">
          <ChapterLine
            k="chapter 08 · the machine"
            sub={
              mode === "system"
                ? "System mode — the universe as its architecture. Click a body."
                : "Don't just trust the answer. Switch to SYSTEM (top right) and the universe shows its machinery."
            }
          >
            DON'T JUST TRUST THE&nbsp;ANSWER.
          </ChapterLine>

          <div
            className={`transition-opacity duration-500 ${
              mode === "system" ? "opacity-100" : "opacity-60"
            }`}
          >
            <svg viewBox="0 0 920 300" className="w-full">
              <path
                d="M60,210 C120,140 150,110 185,90 C250,110 290,170 330,180 C390,150 420,95 470,82 C540,110 570,175 610,190 C670,150 700,105 740,96 C800,130 830,185 855,205"
                fill="none"
                stroke={BLUE}
                strokeWidth="1.6"
                className="flow-dash"
                opacity="0.7"
              />
              {galaxy.map((gx) => {
                const sel = sysSel?.key === gx.key;
                return (
                  <g
                    key={gx.key}
                    className="cursor-pointer"
                    onClick={() => setSysSel(sel ? null : gx)}
                  >
                    <circle
                      cx={gx.x}
                      cy={gx.y}
                      r={sel ? 26 : 20}
                      fill="#0B1020"
                      stroke={sel ? GOLD : "#3A4368"}
                      strokeWidth={sel ? 2.4 : 1.4}
                      style={
                        sel
                          ? { filter: `drop-shadow(0 0 10px ${GOLD}88)` }
                          : undefined
                      }
                    />
                    <text
                      x={gx.x}
                      y={gx.y - 32}
                      textAnchor="middle"
                      fontSize="12.5"
                      fill="#E8E4D8"
                    >
                      {gx.name}
                    </text>
                    <text
                      x={gx.x}
                      y={gx.y + 42}
                      textAnchor="middle"
                      fontSize="10"
                      fill="#5A6390"
                      className="font-mono"
                    >
                      {gx.sub}
                    </text>
                  </g>
                );
              })}
            </svg>
            <AnimatePresence>
              {sysSel && (
                <motion.p
                  key={sysSel.key}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="mx-auto max-w-xl border-l-2 border-[#66B8FF]/60 pl-4 text-sm text-slate-300"
                >
                  <span className="font-serif text-lg text-white">
                    {sysSel.name}
                  </span>
                  <span className="mx-2 text-slate-600">·</span>
                  <span className="font-mono text-[13px] text-[#66B8FF]">
                    {sysSel.tel}
                  </span>
                </motion.p>
              )}
            </AnimatePresence>
          </div>

          {evid && (
            <div className="mt-12">
              <Note>evidence · exact placements (sample chart)</Note>
              <div className="mt-3 grid max-w-2xl grid-cols-1 gap-x-10 font-mono text-[13px] sm:grid-cols-2">
                {placements.map(([p, deg, h]) => (
                  <div
                    key={p}
                    className="flex justify-between border-b border-white/5 py-1.5"
                  >
                    <span className="text-slate-300">{p}</span>
                    <span className="text-slate-500">{deg}</span>
                    <span className="text-[#E8B44C]">{h}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-16 text-center">
            <p className="mx-auto max-w-2xl font-serif text-2xl leading-relaxed text-white sm:text-3xl">
              It begins as something magical — and ends by showing you the
              machine underneath.
            </p>
            <div className="mt-7 flex flex-wrap justify-center gap-4">
              <a
                href={DEMO_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full border border-[#E8B44C] bg-[#E8B44C]/10 px-8 py-3 font-serif text-lg text-[#E8B44C] transition hover:bg-[#E8B44C] hover:text-[#05070F]"
              >
                Enter your universe →
              </a>
              <button
                onClick={() => navigate("/ai-playground")}
                className="rounded-full border border-white/20 px-8 py-3 text-sm text-slate-300 transition hover:border-white/50"
              >
                More experiments
              </button>
            </div>
            <p className="mx-auto mt-10 max-w-2xl pb-4 text-[11px] leading-relaxed text-slate-600">
              The chart, signals and figures on this page are a sample
              demonstration of the system's design; the live instrument's
              Explanation tab carries the real trace for any reading you cast.
              Readings are for reflection, not medical, legal, or financial
              advice.
            </p>
          </div>
        </section>
      </main>

      {/* planet entity overlay */}
      <AnimatePresence>
        {entity && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-[#05070F]/90 p-6 backdrop-blur-md"
            onClick={() => setEntity(null)}
          >
            <motion.div
              initial={{ scale: 0.92, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.92, opacity: 0 }}
              className="relative w-full max-w-lg text-center"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="relative mx-auto flex h-44 w-44 items-center justify-center">
                <span
                  className="flex h-28 w-28 items-center justify-center rounded-full text-6xl text-[#E8B44C]"
                  style={{
                    background:
                      "radial-gradient(circle at 35% 30%, #1A2340, #0B1020)",
                    boxShadow: "0 0 60px #E8B44C33, inset 0 0 30px #66B8FF22",
                  }}
                >
                  {entity.glyph}
                </span>
                {entity.orbit.map((c, i) => {
                  const angle =
                    (i / entity.orbit.length) * Math.PI * 2 - Math.PI / 2;
                  return (
                    <span
                      key={c}
                      className="absolute font-mono text-[11px] uppercase tracking-wider text-slate-400"
                      style={{
                        transform: `translate(${Math.cos(angle) * 118}px, ${Math.sin(angle) * 88}px)`,
                      }}
                    >
                      {c}
                    </span>
                  );
                })}
              </div>
              <h3 className="mt-4 font-serif text-3xl font-bold text-white">
                {entity.name}
              </h3>
              <div className="mx-auto mt-4 grid max-w-sm grid-cols-2 gap-x-8 gap-y-2 text-left font-mono text-[13px]">
                {[
                  ["house", entity.house],
                  ["sign", entity.sign],
                  ["nakshatra", entity.nak],
                  ["strength", `${entity.strength} / 100`],
                ].map(([k, v]) => (
                  <div
                    key={k}
                    className="flex justify-between border-b border-white/10 py-1"
                  >
                    <span className="text-slate-500">{k}</span>
                    <span className="text-slate-200">{v}</span>
                  </div>
                ))}
              </div>
              <p className="mt-3 font-mono text-[12px] text-[#66B8FF]">
                transit influence: {entity.transit}
              </p>
              <button
                onClick={() => {
                  setEntity(null);
                  setHouse(
                    entity.house === "X"
                      ? 10
                      : entity.house === "I"
                        ? 1
                        : entity.house === "IV"
                          ? 4
                          : 7,
                  );
                  scrollToId("ch1");
                }}
                className="mt-6 rounded-full border border-[#E8B44C]/70 px-6 py-2.5 font-serif text-[#E8B44C] transition hover:bg-[#E8B44C]/10"
              >
                Show me where {entity.name.split(" ")[0]} matters →
              </button>
              <button
                onClick={() => setEntity(null)}
                aria-label="Close"
                className="absolute -right-2 -top-2 flex h-10 w-10 items-center justify-center rounded-full border border-white/20 text-slate-400 transition hover:border-white/60 hover:text-white"
              >
                ✕
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
