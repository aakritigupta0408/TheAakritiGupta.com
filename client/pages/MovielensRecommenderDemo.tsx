/**
 * MovieLens 25M Recommender — Offline Benchmark Dashboard
 *
 * Static-data showcase mirroring the Python Streamlit dashboard. All numbers
 * are pre-computed by the offline pipeline in `movielens_recommender/` and
 * served from `/data/movielens-recommender/*.json` (no backend dependency).
 *
 * Sections:
 *   1. Hero
 *   2. Head-to-head retrieval comparison (full-catalog protocol)
 *   3. Sampled-99-negs results (academic protocol)
 *   4. The leakage story (before/after the val-positive fix)
 *   5. Worked example (Blair Witch user)
 *   6. XGBoost feature importance
 *   7. Training curves
 *   8. Audit summary
 *   9. Documentation links
 */
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Award,
  BookOpen,
  Brain,
  CheckCircle2,
  ChevronRight,
  Database,
  Download,
  Github,
  Layers,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingUp,
  TriangleAlert,
  XCircle,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import Navigation from "@/components/Navigation";

// ── Types ──────────────────────────────────────────────────────────────────

type MetricRow = Record<string, number | string>;

interface MetricsBundle {
  // Each retrieval method (popularity, cooccurrence, als, full_pipeline) is a
  // flat MetricRow at the top level; `sampled_99_negatives` is a nested
  // namespace with the academic-protocol metrics keyed by method.
  popularity?: MetricRow;
  cooccurrence?: MetricRow;
  als?: MetricRow;
  full_pipeline?: MetricRow;
  sampled_99_negatives?: Record<string, MetricRow>;
  [key: string]: MetricRow | Record<string, MetricRow> | undefined;
}

interface AuditCheck {
  id: string;
  name: string;
  status: "PASS" | "WARN" | "FAIL";
  detail: string;
}

interface AuditReport {
  summary: Record<string, number>;
  checks: AuditCheck[];
}

interface Recommendation {
  rank: number;
  item_idx: number;
  movieId: number;
  title: string;
  genres: string[];
  year: number | null;
  score: number;
  source_pop: boolean;
  source_cooc: boolean;
  source_als: boolean;
}

interface ExampleRecs {
  user_idx: number;
  recommendations: Recommendation[];
}

interface FeatureImportance {
  feature_names: string[];
  importance: {
    gain: { feature: string; importance: number }[];
    cover: { feature: string; importance: number }[];
    weight: { feature: string; importance: number }[];
  };
}

interface TrainingCurve {
  model: string;
  rounds: number;
  metric: string;
  history: { round: number; train: number; val: number }[];
}

// ── Static loaders ─────────────────────────────────────────────────────────

const DATA_BASE = "/data/movielens-recommender";

async function loadJSON<T>(name: string): Promise<T> {
  const r = await fetch(`${DATA_BASE}/${name}`);
  if (!r.ok) throw new Error(`failed to load ${name}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

// ── Method color tokens ────────────────────────────────────────────────────

const METHOD_COLORS: Record<string, string> = {
  popularity: "#94a3b8", // slate-400
  cooccurrence: "#a78bfa", // violet-400
  als: "#f472b6", // pink-400
  full_pipeline: "#fbbf24", // amber-400 — the winner
  random: "#475569", // slate-600
};

const METHOD_LABEL: Record<string, string> = {
  popularity: "Popularity",
  cooccurrence: "Item-item cooc",
  als: "ALS (personalized)",
  full_pipeline: "Full pipeline (XGB)",
  random: "Random",
};

const METHOD_KIND: Record<string, string> = {
  popularity: "non-personalized",
  cooccurrence: "non-personalized",
  als: "personalized",
  full_pipeline: "fusion",
  random: "baseline",
};

// ── Component ──────────────────────────────────────────────────────────────

export default function MovielensRecommenderDemo() {
  const navigate = useNavigate();

  const [metrics, setMetrics] = useState<MetricsBundle | null>(null);
  const [audit, setAudit] = useState<AuditReport | null>(null);
  const [example, setExample] = useState<ExampleRecs | null>(null);
  const [importance, setImportance] = useState<FeatureImportance | null>(null);
  const [curve, setCurve] = useState<TrainingCurve | null>(null);
  const [importanceType, setImportanceType] = useState<"gain" | "cover" | "weight">("gain");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      loadJSON<MetricsBundle>("metrics.json"),
      loadJSON<AuditReport>("audit_report.json"),
      loadJSON<ExampleRecs>("example_recs.json"),
      loadJSON<FeatureImportance>("xgb_feature_importance.json"),
      loadJSON<TrainingCurve>("training_curve.json"),
    ])
      .then(([m, a, e, i, c]) => {
        setMetrics(m);
        setAudit(a);
        setExample(e);
        setImportance(i);
        setCurve(c);
      })
      .catch((err) => setError(err.message));
  }, []);

  const methods = ["popularity", "cooccurrence", "als", "full_pipeline"] as const;

  const fullCatalogChart = useMemo(() => {
    if (!metrics) return [];
    return methods.map((m) => ({
      method: METHOD_LABEL[m],
      key: m,
      "Recall@10": (metrics[m]?.["recall@10"] as number) ?? 0,
      "NDCG@10": (metrics[m]?.["ndcg@10"] as number) ?? 0,
      "Recall@50": (metrics[m]?.["recall@50"] as number) ?? 0,
      Coverage: (metrics[m]?.["coverage"] as number) ?? 0,
    }));
  }, [metrics]);

  const sampledChart = useMemo(() => {
    if (!metrics) return [];
    const s = metrics.sampled_99_negatives ?? {};
    return ["random", "popularity", "cooccurrence", "als", "full_pipeline"].map((m) => ({
      method: METHOD_LABEL[m],
      key: m,
      "HR@1": (s[m]?.["hr@1"] as number) ?? 0,
      "HR@10": (s[m]?.["hr@10"] as number) ?? 0,
      "NDCG@10": (s[m]?.["ndcg@10"] as number) ?? 0,
      MRR: (s[m]?.["mrr@100"] as number) ?? 0,
    }));
  }, [metrics]);

  const topFeatures = useMemo(() => {
    if (!importance) return [];
    return importance.importance[importanceType].slice(0, 15);
  }, [importance, importanceType]);

  // ── Render ──

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#0b0a1a] text-slate-50">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(167,139,250,0.18),transparent_26%),radial-gradient(circle_at_85%_15%,rgba(244,114,182,0.16),transparent_26%),radial-gradient(circle_at_50%_100%,rgba(251,191,36,0.1),transparent_30%)]" />
        <div className="absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] [background-size:72px_72px]" />
      </div>

      <Navigation />

      <main className="relative z-10 px-4 pb-24 pt-28 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-6xl">
          {/* ── Hero ── */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-8 rounded-[2rem] border border-white/10 bg-[linear-gradient(140deg,rgba(20,18,40,0.88),rgba(11,10,26,0.72))] p-7 shadow-[0_30px_90px_rgba(4,12,19,0.42)] backdrop-blur-xl sm:p-9"
          >
            <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <div className="inline-flex items-center gap-2 rounded-full border border-violet-400/30 bg-violet-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-violet-100">
                  <Sparkles className="h-4 w-4" />
                  Offline benchmark · MovieLens 25M
                </div>
                <h1 className="mt-5 text-4xl font-black leading-tight text-white sm:text-5xl lg:text-6xl">
                  MovieLens 25M
                  <span className="block bg-gradient-to-r from-violet-300 via-pink-200 to-amber-200 bg-clip-text text-transparent">
                    Recommender — head to head
                  </span>
                </h1>
                <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
                  Personalized (implicit-ALS matrix factorization) and
                  non-personalized (Wilson popularity, item-item cooccurrence)
                  retrieval, evaluated independently on the same held-out users,
                  then fused via an XGBoost LambdaMART ranker. 162,541 users ×
                  32,720 items × 24.9M ratings. Honest leave-one-out evaluation
                  on real users.
                </p>
              </div>
              <div className="flex shrink-0 flex-col gap-3">
                <div className="flex flex-wrap gap-2">
                  {[
                    { icon: <Database className="h-3.5 w-3.5" />, label: "Users", value: "162k" },
                    { icon: <Layers className="h-3.5 w-3.5" />, label: "Items", value: "32k" },
                    { icon: <Target className="h-3.5 w-3.5" />, label: "Ratings", value: "24.9M" },
                    {
                      icon: <ShieldCheck className="h-3.5 w-3.5" />,
                      label: "Audit",
                      value: audit
                        ? `${audit.summary.PASS ?? 0}P / ${audit.summary.WARN ?? 0}W / ${audit.summary.FAIL ?? 0}F`
                        : "—",
                    },
                  ].map((s) => (
                    <div
                      key={s.label}
                      className="flex items-center gap-2 rounded-full border border-violet-300/20 bg-violet-300/5 px-4 py-2 text-sm text-violet-100"
                    >
                      {s.icon}
                      <span className="font-bold">{s.value}</span>
                      <span className="text-xs opacity-75">{s.label}</span>
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => navigate("/ai-playground")}
                    className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 hover:bg-white/10"
                  >
                    <ArrowLeft className="h-4 w-4" /> Back
                  </button>
                  <a
                    href={`${DATA_BASE}/OFFLINE_RESULTS.pdf`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 rounded-full border border-violet-300/30 bg-violet-400/10 px-4 py-2 text-sm font-semibold text-violet-100 hover:bg-violet-400/20"
                  >
                    <Download className="h-4 w-4" /> Offline results (PDF)
                  </a>
                  <a
                    href="https://github.com/aakritigupta0408"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 hover:bg-white/10"
                  >
                    <Github className="h-4 w-4" /> Source
                  </a>
                </div>
              </div>
            </div>
          </motion.div>

          {error && (
            <div className="mb-6 flex items-center gap-3 rounded-2xl border border-rose-400/20 bg-rose-400/8 px-5 py-4 text-sm text-rose-200">
              <TriangleAlert className="h-4 w-4 shrink-0" />
              <span>Failed to load static data: {error}</span>
            </div>
          )}

          {/* ── Headline metric cards ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="mb-10"
          >
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-widest text-slate-500">
              Headline — full-catalog ranking
            </h2>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {methods.map((m) => {
                const r10 = (metrics?.[m]?.["recall@10"] as number) ?? 0;
                const ndcg = (metrics?.[m]?.["ndcg@10"] as number) ?? 0;
                const winner = m === "full_pipeline";
                return (
                  <div
                    key={m}
                    className={`group rounded-[1.6rem] border bg-white/4 p-5 backdrop-blur-sm transition ${
                      winner
                        ? "border-amber-300/30 bg-amber-300/5 shadow-[0_18px_60px_rgba(251,191,36,0.16)]"
                        : "border-white/10 hover:border-white/20"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs uppercase tracking-widest text-slate-500">
                        {METHOD_KIND[m]}
                      </span>
                      {winner && (
                        <Award className="h-4 w-4 text-amber-300" />
                      )}
                    </div>
                    <p className="mt-2 text-base font-semibold text-white">
                      {METHOD_LABEL[m]}
                    </p>
                    <div className="mt-4 flex items-baseline gap-4">
                      <div>
                        <p className="text-3xl font-black text-white">
                          {(r10 * 100).toFixed(1)}%
                        </p>
                        <p className="text-[11px] uppercase tracking-widest text-slate-500">
                          Recall@10
                        </p>
                      </div>
                      <div>
                        <p className="text-lg font-bold text-slate-300">
                          {ndcg.toFixed(3)}
                        </p>
                        <p className="text-[11px] uppercase tracking-widest text-slate-500">
                          NDCG@10
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-3 text-xs text-slate-500">
              Full pipeline wins by{" "}
              <span className="text-amber-300">
                +
                {metrics
                  ? Math.round(
                      (((metrics.full_pipeline?.["recall@10"] as number) ?? 0) /
                        Math.max(
                          (metrics.cooccurrence?.["recall@10"] as number) ?? 1,
                          1e-9,
                        ) -
                        1) *
                        100,
                    )
                  : "—"}
                %
              </span>{" "}
              relative Recall@10 over the best individual retrieval. Evaluated
              on 10,000 held-out test users.
            </p>
          </motion.section>

          {/* ── Full-catalog comparison chart ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-10 rounded-[1.6rem] border border-white/10 bg-white/4 p-6 backdrop-blur-sm"
          >
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
                Head-to-head — full-catalog protocol
              </h2>
              <span className="rounded-full border border-emerald-300/30 bg-emerald-300/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-widest text-emerald-100">
                production-honest
              </span>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={fullCatalogChart} margin={{ top: 16, right: 24, left: 0, bottom: 8 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
                  <XAxis dataKey="method" tick={{ fill: "#94a3b8", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(11,10,26,0.96)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                    formatter={(v: number) => v.toFixed(3)}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Recall@10" fill="#a78bfa" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="NDCG@10" fill="#f472b6" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="Recall@50" fill="#fbbf24" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.section>

          {/* ── Sampled-99-negs chart ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="mb-10 rounded-[1.6rem] border border-white/10 bg-white/4 p-6 backdrop-blur-sm"
          >
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
                Academic protocol — sampled 99 negatives
              </h2>
              <span className="rounded-full border border-amber-300/30 bg-amber-300/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-widest text-amber-100">
                known biased (Krichene & Rendle 2020)
              </span>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sampledChart} margin={{ top: 16, right: 24, left: 0, bottom: 8 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
                  <XAxis dataKey="method" tick={{ fill: "#94a3b8", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} tickLine={false} axisLine={false} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(11,10,26,0.96)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                    formatter={(v: number) => v.toFixed(3)}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="HR@1" fill="#fbbf24" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="HR@10" fill="#a78bfa" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="NDCG@10" fill="#f472b6" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-3 text-xs text-slate-500">
              HR@10 saturates for stronger methods because random negatives from
              32k items are mostly long-tail garbage. <span className="text-amber-300">HR@1 is
              the honest comparison</span> in this protocol — XGB fusion wins
              {metrics?.sampled_99_negatives && (
                <>
                  {" "}
                  with{" "}
                  <span className="text-white">
                    {((metrics.sampled_99_negatives.full_pipeline?.["hr@1"] as number) * 100).toFixed(
                      1,
                    )}
                    %
                  </span>{" "}
                  vs ALS's{" "}
                  <span className="text-white">
                    {((metrics.sampled_99_negatives.als?.["hr@1"] as number) * 100).toFixed(1)}%
                  </span>
                  .
                </>
              )}
            </p>
          </motion.section>

          {/* ── Leakage story ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-10 rounded-[1.6rem] border border-rose-400/20 bg-rose-400/5 p-6 backdrop-blur-sm"
          >
            <div className="mb-4 flex items-center gap-2">
              <TriangleAlert className="h-5 w-5 text-rose-300" />
              <h2 className="text-sm font-semibold uppercase tracking-widest text-rose-200">
                The leakage bug we caught and fixed
              </h2>
            </div>
            <p className="mb-5 text-sm leading-7 text-slate-300">
              First version of the XGB ranker used the user's <em>last train
              positive</em> as the training target. ALS had memorized that
              item — its score was artificially high. The ranker learned the
              degenerate rule <em>"high ALS score ⇒ positive"</em>, which
              doesn't transfer to held-out future items. <strong className="text-rose-200">Sampled HR@1
              collapsed to 0.0001 — worse than random.</strong>
            </p>
            <p className="mb-5 text-sm leading-7 text-slate-300">
              The fix was a 5-line change: use the user's <strong className="text-emerald-200">val
              positive</strong> (held out of retrieval training) as the
              ranker's training target. After the fix:
            </p>
            <div className="grid gap-4 sm:grid-cols-3">
              {[
                { label: "Sampled HR@1", before: "0.0001", after: "0.623", lift: "×6,200" },
                { label: "Full Recall@10", before: "0.020", after: "0.125", lift: "×6.3" },
                { label: "Full NDCG@10", before: "0.010", after: "0.066", lift: "×6.6" },
              ].map((row) => (
                <div
                  key={row.label}
                  className="rounded-[1.2rem] border border-white/10 bg-[rgba(11,10,26,0.5)] p-4"
                >
                  <p className="text-[11px] uppercase tracking-widest text-slate-500">
                    {row.label}
                  </p>
                  <div className="mt-2 flex items-baseline gap-2 font-mono">
                    <span className="text-base text-rose-300/80 line-through">{row.before}</span>
                    <ChevronRight className="h-4 w-4 text-slate-600" />
                    <span className="text-xl font-bold text-emerald-200">{row.after}</span>
                  </div>
                  <p className="mt-1 text-xs text-amber-300">{row.lift}</p>
                </div>
              ))}
            </div>
            <p className="mt-5 text-xs italic text-slate-500">
              Lesson: when stacking a ranker on top of retrieval, the ranker's
              training positives must be held out of retrieval training.
            </p>
          </motion.section>

          {/* ── Worked example ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
            className="mb-10 rounded-[1.6rem] border border-white/10 bg-white/4 p-6 backdrop-blur-sm"
          >
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
                Worked example — real user from test split
              </h2>
              {example && (
                <span className="text-xs text-slate-500">user_idx = {example.user_idx}</span>
              )}
            </div>
            {example?.recommendations.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-[10px] uppercase tracking-widest text-slate-500">
                    <tr className="border-b border-white/10">
                      <th className="px-2 py-2">#</th>
                      <th className="px-2 py-2">Title</th>
                      <th className="px-2 py-2">Year</th>
                      <th className="px-2 py-2">Genres</th>
                      <th className="px-2 py-2">Sources</th>
                      <th className="px-2 py-2 text-right">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {example.recommendations.map((r) => {
                      const sources: string[] = [];
                      if (r.source_pop) sources.push("pop");
                      if (r.source_cooc) sources.push("cooc");
                      if (r.source_als) sources.push("als");
                      return (
                        <tr
                          key={r.item_idx}
                          className="border-b border-white/5 last:border-0 hover:bg-white/3"
                        >
                          <td className="px-2 py-2 font-mono text-slate-500">{r.rank}</td>
                          <td className="px-2 py-2 font-medium text-white">{r.title}</td>
                          <td className="px-2 py-2 text-slate-400">{r.year ?? "—"}</td>
                          <td className="px-2 py-2 text-xs text-slate-400">
                            {r.genres.slice(0, 3).join(", ")}
                          </td>
                          <td className="px-2 py-2">
                            <div className="flex gap-1">
                              {sources.map((s) => (
                                <span
                                  key={s}
                                  className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase ${
                                    s === "pop"
                                      ? "border-slate-400/30 bg-slate-400/10 text-slate-300"
                                      : s === "cooc"
                                        ? "border-violet-400/30 bg-violet-400/10 text-violet-200"
                                        : "border-pink-400/30 bg-pink-400/10 text-pink-200"
                                  }`}
                                >
                                  {s}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-2 py-2 text-right font-mono text-amber-300">
                            {r.score.toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-slate-500">Loading example…</p>
            )}
            <p className="mt-4 text-xs text-slate-500">
              User had a mixed horror/comedy watch history (Blair Witch Project,
              Blob, Bone Collector, Blues Brothers, ...). Top recommendations
              include Ghostbusters, Army of Darkness, Beetlejuice — cross-genre
              comedy-horror matches. Source badges show which retrieval methods
              surfaced each candidate before XGB scored it.
            </p>
          </motion.section>

          {/* ── XGB feature importance ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mb-10 rounded-[1.6rem] border border-white/10 bg-white/4 p-6 backdrop-blur-sm"
          >
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
                XGBoost feature importance — top 15
              </h2>
              <div className="flex gap-1 rounded-full border border-white/10 bg-white/5 p-1">
                {(["gain", "cover", "weight"] as const).map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setImportanceType(t)}
                    className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-widest transition ${
                      importanceType === t
                        ? "bg-violet-400 text-slate-950"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={topFeatures}
                  layout="vertical"
                  margin={{ top: 8, right: 24, left: 80, bottom: 8 }}
                >
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: "#cbd5e1", fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={170}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(11,10,26,0.96)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="importance" fill="#a78bfa" radius={[0, 6, 6, 0]}>
                    {topFeatures.map((f, i) => (
                      <Cell
                        key={f.feature}
                        fill={i === 0 ? "#fbbf24" : i < 3 ? "#f472b6" : "#a78bfa"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-3 text-xs text-slate-500">
              <span className="font-mono text-amber-300">als_score</span>,{" "}
              <span className="font-mono text-pink-300">cooc_score</span>, and
              the <span className="font-mono">src_*</span> source flags
              dominate — exactly the design intent.
            </p>
          </motion.section>

          {/* ── Training curve ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="mb-10 rounded-[1.6rem] border border-white/10 bg-white/4 p-6 backdrop-blur-sm"
          >
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-widest text-slate-500">
              XGBoost LambdaMART training curve (NDCG@10)
            </h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curve?.history ?? []} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
                  <XAxis dataKey="round" tick={{ fill: "#94a3b8", fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis
                    domain={[0.7, 0.86]}
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => v.toFixed(2)}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(11,10,26,0.96)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                    formatter={(v: number) => v.toFixed(4)}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="train" name="train NDCG@10" stroke="#fbbf24" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val" name="val NDCG@10" stroke="#a78bfa" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-3 text-xs text-slate-500">
              Healthy curve. Train climbs to 0.857 (overfit on small 10-
              candidate groups); val plateaus at 0.79. No catastrophic overfit;
              no early stopping needed at this scale.
            </p>
          </motion.section>

          {/* ── Audit summary ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mb-10 rounded-[1.6rem] border border-white/10 bg-white/4 p-6 backdrop-blur-sm"
          >
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
                Audit — leakage & methodology checks
              </h2>
              <code className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] text-slate-400">
                python -m src.audit
              </code>
            </div>
            {audit && (
              <>
                <div className="mb-5 grid gap-3 sm:grid-cols-3">
                  {[
                    { key: "PASS", label: "Passed", color: "emerald", icon: CheckCircle2 },
                    { key: "WARN", label: "Warned", color: "amber", icon: TriangleAlert },
                    { key: "FAIL", label: "Failed", color: "rose", icon: XCircle },
                  ].map((s) => {
                    const Icon = s.icon;
                    return (
                      <div
                        key={s.key}
                        className={`flex items-center gap-3 rounded-[1.2rem] border p-4 ${
                          s.color === "emerald"
                            ? "border-emerald-400/20 bg-emerald-400/8"
                            : s.color === "amber"
                              ? "border-amber-400/20 bg-amber-400/8"
                              : "border-rose-400/20 bg-rose-400/8"
                        }`}
                      >
                        <Icon
                          className={`h-7 w-7 ${
                            s.color === "emerald"
                              ? "text-emerald-300"
                              : s.color === "amber"
                                ? "text-amber-300"
                                : "text-rose-300"
                          }`}
                        />
                        <div>
                          <p className="text-3xl font-black text-white">
                            {audit.summary[s.key] ?? 0}
                          </p>
                          <p className="text-[11px] uppercase tracking-widest text-slate-500">
                            {s.label}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="overflow-hidden rounded-[1.2rem] border border-white/10">
                  <table className="w-full text-xs">
                    <thead className="bg-white/5 text-left text-[10px] uppercase tracking-widest text-slate-500">
                      <tr>
                        <th className="px-3 py-2 w-12"></th>
                        <th className="px-3 py-2 w-12">ID</th>
                        <th className="px-3 py-2">Check</th>
                        <th className="px-3 py-2">Detail</th>
                      </tr>
                    </thead>
                    <tbody>
                      {audit.checks.map((c) => (
                        <tr
                          key={c.id}
                          className={`border-t border-white/5 ${
                            c.status === "FAIL"
                              ? "bg-rose-400/4"
                              : c.status === "WARN"
                                ? "bg-amber-400/4"
                                : ""
                          }`}
                        >
                          <td className="px-3 py-2 text-center">
                            {c.status === "PASS" && <CheckCircle2 className="inline h-4 w-4 text-emerald-300" />}
                            {c.status === "WARN" && <TriangleAlert className="inline h-4 w-4 text-amber-300" />}
                            {c.status === "FAIL" && <XCircle className="inline h-4 w-4 text-rose-300" />}
                          </td>
                          <td className="px-3 py-2 font-mono text-slate-500">{c.id}</td>
                          <td className="px-3 py-2 text-slate-200">{c.name}</td>
                          <td className="px-3 py-2 text-slate-400 leading-relaxed">{c.detail}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-4 text-xs italic text-slate-500">
                  Biggest disclosed eyebrow:{" "}
                  <span className="text-amber-300">
                    89.1% of users rate val and test within the same 1-hour
                    session
                  </span>{" "}
                  — not leakage (retrieval models never see either), but it
                  means the XGB-train and eval-test distributions are highly
                  correlated.
                </p>
              </>
            )}
          </motion.section>

          {/* ── Architecture diagram (ASCII style) ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.45 }}
            className="mb-10 rounded-[1.6rem] border border-white/10 bg-white/4 p-6 backdrop-blur-sm"
          >
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-widest text-slate-500">
              <Brain className="h-4 w-4" /> Architecture
            </h2>
            <pre className="overflow-x-auto whitespace-pre rounded-2xl border border-white/10 bg-black/40 p-5 text-[11px] leading-5 text-slate-300">
{`     ┌────────────────────────────┐         ┌─────────────────────────────────────┐
     │  OFFLINE  (build once)     │         │  SERVING  (~1.6 ms per user)         │
     ├────────────────────────────┤         ├─────────────────────────────────────┤
     │  ratings.csv → preprocess  │         │  user_idx                           │
     │            ↓               │         │       │                             │
     │  leave-one-out splits      │         │  ┌────┴──────────────────────┐      │
     │            ↓               │         │  ▼                ▼          ▼      │
     │  ┌────────┐ ┌────┐ ┌────┐  │         │ POP top-200  COOC top-200   ALS    │
     │  │ pop    │ │cooc│ │ALS │  │   ────► │  (Wilson)   (last-20 seeds) (MF)  │
     │  │Wilson  │ │cos │ │MF  │  │         │  └──────┬───────┬───────────┘      │
     │  └────────┘ └────┘ └────┘  │         │         ▼       ▼                  │
     │       ↓       ↓       ↓     │         │      UNION ≈ 500 candidates        │
     │       └───────┴───────┘     │         │             │                       │
     │   XGB LambdaMART (55 feats) │         │             ▼                       │
     │   trained on VAL positives  │ ──────► │   XGBoost LambdaMART predict       │
     └────────────────────────────┘         │             │                       │
                                            │             ▼                       │
                                            │      top-N to user                  │
                                            └─────────────────────────────────────┘
`}
            </pre>
          </motion.section>

          {/* ── Documentation links ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="rounded-[1.6rem] border border-white/10 bg-white/4 p-6 backdrop-blur-sm"
          >
            <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-widest text-slate-500">
              <BookOpen className="h-4 w-4" /> Repository documentation
            </h2>
            <div className="grid gap-3 sm:grid-cols-2">
              {[
                {
                  label: "Design doc (RFC)",
                  desc: "Formal engineering design — goals, alternatives, risks, open questions.",
                },
                {
                  label: "Technical document",
                  desc: "Full walkthrough with diagrams and per-stage detail.",
                },
                {
                  label: "Offline results report",
                  desc: "Every metric, every protocol, vs published baselines.",
                },
                {
                  label: "Audit report",
                  desc: "All leakage and methodology checks with narrative.",
                },
                {
                  label: "30-min lecture script",
                  desc: "Word-for-word presentation script with stage directions.",
                },
                {
                  label: "Slide deck (24 slides)",
                  desc: "Presentable Markdown deck with explicit slide breaks.",
                },
              ].map((d) => (
                <div
                  key={d.label}
                  className="rounded-[1.2rem] border border-white/10 bg-[rgba(11,10,26,0.5)] p-4"
                >
                  <div className="flex items-center gap-2 text-sm font-semibold text-white">
                    <TrendingUp className="h-4 w-4 text-violet-300" />
                    {d.label}
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-slate-400">{d.desc}</p>
                </div>
              ))}
            </div>
          </motion.section>
        </div>
      </main>
    </div>
  );
}
