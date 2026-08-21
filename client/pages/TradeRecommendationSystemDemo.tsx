/**
 * Live Trade Recommendation System Dashboard
 *
 * Fetches real-time data from the Python FastAPI service via Express proxy:
 *   GET /api/trade-system/recommendations  → scan results + scores
 *   GET /api/trade-system/users            → all 5 user states
 *   GET /api/trade-system/users/:id        → user detail
 *   GET /api/trade-system/summary          → combined overview
 *   POST /api/trade-system/scan/trigger    → manual scan trigger
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  ArrowLeft,
  BarChart2,
  RefreshCw,
  Target,
  TriangleAlert,
  Users,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import Navigation from "@/components/Navigation";

// ── Types ──────────────────────────────────────────────────────────────────

interface ScanResult {
  ticker: string;
  action: string; // BUY | SELL | HOLD
  composite_score: number;
  agent_score: number;
  fta_score: number;
  momentum_score: number;
  iv_regime_score: number;
  option_strategy_type: string;
  scanned_at: string;
  error: string | null;
  is_actionable: boolean;
}

interface TradeRecord {
  trade_id: string;
  ticker: string;
  action: string;
  instrument: string;
  quantity: number;
  entry_price: number;
  stop_price: number;
  target_price: number;
  status: string;
  pnl: number;
  opened_at: string;
  closed_at: string | null;
  composite_score: number;
  rationale: string;
}

interface UserState {
  user_id: string;
  name: string;
  avatar: string;
  description: string;
  equity: number;
  starting_capital: number;
  return_pct: number;
  open_positions: number;
  total_trades: number;
  win_rate: number;
  status: string; // active | target_reached | bust | paused
  open_trades: TradeRecord[];
  closed_trades: TradeRecord[];
  equity_curve: { ts: string; equity: number }[];
}

interface Summary {
  scan_in_progress: boolean;
  last_scan_at: string | null;
  scan_error: string | null;
  total_users: number;
  active_users: number;
  total_open_positions: number;
  top_recommendations: ScanResult[];
  users: {
    user_id: string;
    name: string;
    avatar: string;
    return_pct: number;
    status: string;
    open_positions: number;
    total_trades: number;
    win_rate: number;
  }[];
}

// ── API helpers ───────────────────────────────────────────────────────────

const API = "/api/trade-system";

async function fetchJSON<T>(path: string): Promise<T> {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

async function postJSON<T>(path: string): Promise<T> {
  const r = await fetch(API + path, { method: "POST" });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function fmt(n: number, digits = 2) {
  return n.toFixed(digits);
}

function pctColor(v: number) {
  if (v > 5) return "text-emerald-300";
  if (v > 0) return "text-emerald-200";
  if (v < -5) return "text-rose-400";
  if (v < 0) return "text-rose-300";
  return "text-slate-300";
}

function statusBadge(status: string) {
  switch (status) {
    case "target_reached":
      return "border-emerald-400/25 bg-emerald-400/10 text-emerald-200";
    case "bust":
      return "border-rose-400/25 bg-rose-400/10 text-rose-200";
    case "paused":
      return "border-amber-400/25 bg-amber-400/10 text-amber-200";
    default:
      return "border-cyan-400/25 bg-cyan-400/10 text-cyan-200";
  }
}

function actionBadge(action: string) {
  switch (action?.toUpperCase()) {
    case "BUY":
      return "border-emerald-400/25 bg-emerald-400/10 text-emerald-200";
    case "SELL":
      return "border-rose-400/25 bg-rose-400/10 text-rose-200";
    default:
      return "border-slate-400/20 bg-slate-400/8 text-slate-400";
  }
}

function ScoreBar({ value, color = "emerald" }: { value: number; color?: string }) {
  const colorMap: Record<string, string> = {
    emerald: "from-emerald-500 to-emerald-300",
    cyan: "from-cyan-500 to-cyan-300",
    amber: "from-amber-500 to-amber-300",
    rose: "from-rose-500 to-rose-300",
  };
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/8">
      <div
        className={`h-full rounded-full bg-gradient-to-r ${colorMap[color] ?? colorMap.emerald} transition-all duration-700`}
        style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
      />
    </div>
  );
}

// ── Equity curve mini chart ────────────────────────────────────────────────

function EquityCurve({
  data,
  startingCapital,
}: {
  data: { ts: string; equity: number }[];
  startingCapital: number;
}) {
  if (data.length < 2) {
    return (
      <div className="flex h-32 items-center justify-center text-xs text-slate-500">
        No equity data yet
      </div>
    );
  }
  const chartData = data.map((d) => ({
    t: new Date(d.ts).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
    equity: d.equity,
  }));
  const isUp = data[data.length - 1].equity >= startingCapital;
  return (
    <ResponsiveContainer width="100%" height={128}>
      <AreaChart data={chartData} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={isUp ? "#34d399" : "#f87171"} stopOpacity={0.25} />
            <stop offset="95%" stopColor={isUp ? "#34d399" : "#f87171"} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={false} />
        <YAxis
          tick={{ fill: "#64748b", fontSize: 9 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
          width={36}
        />
        <Tooltip
          contentStyle={{
            background: "rgba(7,19,28,0.95)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 12,
            fontSize: 11,
          }}
          formatter={(v: number) => [`$${v.toLocaleString()}`, "Equity"]}
          labelStyle={{ color: "#94a3b8" }}
        />
        <Area
          type="monotone"
          dataKey="equity"
          stroke={isUp ? "#34d399" : "#f87171"}
          strokeWidth={1.5}
          fill="url(#eqGrad)"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── User detail panel ──────────────────────────────────────────────────────

function UserDetailPanel({
  userId,
  onClose,
}: {
  userId: string;
  onClose: () => void;
}) {
  const [user, setUser] = useState<UserState | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"open" | "closed">("open");

  useEffect(() => {
    setLoading(true);
    fetchJSON<UserState>(`/users/${userId}`)
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, [userId]);

  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 40 }}
      className="fixed right-0 top-0 z-50 flex h-full w-full flex-col overflow-y-auto border-l border-white/10 bg-[rgba(7,19,28,0.97)] p-6 shadow-2xl backdrop-blur-2xl sm:max-w-lg"
    >
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={onClose}
          className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 hover:bg-white/10"
        >
          <ArrowLeft className="h-4 w-4" /> Back
        </button>
      </div>

      {loading && (
        <div className="mt-12 flex items-center justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
        </div>
      )}

      {!loading && user && (
        <>
          <div className="mt-6 flex items-center gap-4">
            <span className="text-4xl">{user.avatar}</span>
            <div>
              <h2 className="text-2xl font-black text-white">{user.name}</h2>
              <p className="text-sm text-slate-400">{user.description}</p>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-3 gap-3">
            {[
              { label: "Equity", value: `$${user.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
              { label: "Return", value: `${user.return_pct >= 0 ? "+" : ""}${fmt(user.return_pct)}%`, colored: true, val: user.return_pct },
              { label: "Win Rate", value: `${fmt(user.win_rate * 100, 0)}%` },
            ].map((m) => (
              <div key={m.label} className="rounded-2xl border border-white/8 bg-white/5 p-3 text-center">
                <p className="text-xs text-slate-500">{m.label}</p>
                <p className={`mt-1 text-lg font-black ${m.colored ? pctColor(m.val ?? 0) : "text-white"}`}>
                  {m.value}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-5">
            <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-500">Equity Curve</p>
            <EquityCurve data={user.equity_curve} startingCapital={user.starting_capital} />
          </div>

          <div className="mt-6 flex gap-2">
            {(["open", "closed"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
                  tab === t
                    ? "bg-cyan-400/20 text-cyan-200"
                    : "text-slate-400 hover:text-slate-300"
                }`}
              >
                {t === "open" ? `Open (${user.open_trades.length})` : `Closed (${user.closed_trades.length})`}
              </button>
            ))}
          </div>

          <div className="mt-3 flex flex-col gap-2">
            {(tab === "open" ? user.open_trades : user.closed_trades).map((trade) => (
              <div
                key={trade.trade_id}
                className="rounded-2xl border border-white/8 bg-white/4 p-4"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white">{trade.ticker}</span>
                  <span
                    className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${
                      trade.status === "closed_profit"
                        ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300"
                        : trade.status === "closed_loss"
                          ? "border-rose-400/25 bg-rose-400/10 text-rose-300"
                          : "border-cyan-400/25 bg-cyan-400/10 text-cyan-300"
                    }`}
                  >
                    {trade.status.replace("_", " ")}
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400">
                  <span>Action: <span className="text-slate-200">{trade.action}</span></span>
                  <span>Qty: <span className="text-slate-200">{trade.quantity}</span></span>
                  <span>Entry: <span className="text-slate-200">${fmt(trade.entry_price)}</span></span>
                  {trade.pnl !== 0 && (
                    <span>P&L: <span className={trade.pnl >= 0 ? "text-emerald-300" : "text-rose-300"}>${fmt(trade.pnl)}</span></span>
                  )}
                </div>
                {trade.rationale && (
                  <p className="mt-2 text-xs leading-5 text-slate-500">{trade.rationale}</p>
                )}
              </div>
            ))}
            {(tab === "open" ? user.open_trades : user.closed_trades).length === 0 && (
              <p className="py-6 text-center text-sm text-slate-600">
                No {tab} trades
              </p>
            )}
          </div>
        </>
      )}

      {!loading && !user && (
        <p className="mt-12 text-center text-slate-500">Failed to load user data.</p>
      )}
    </motion.div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

type SortKey = "composite_score" | "ticker" | "action" | "fta_score" | "agent_score";

export default function TradeRecommendationSystemDemo() {
  const navigate = useNavigate();

  // Data state
  const [summary, setSummary] = useState<Summary | null>(null);
  const [recs, setRecs] = useState<ScanResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [scanning, setScanning] = useState(false);

  // UI state
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "actionable">("actionable");
  const [sortKey, setSortKey] = useState<SortKey>("composite_score");
  const [sortAsc, setSortAsc] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [s, r] = await Promise.all([
        fetchJSON<Summary>("/summary"),
        fetchJSON<{ recommendations: ScanResult[] }>("/recommendations?limit=200"),
      ]);
      setSummary(s);
      setRecs(r.recommendations);
      setError(null);
      setLastRefresh(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load + 60s auto-refresh
  useEffect(() => {
    loadData();
    const id = window.setInterval(loadData, 60_000);
    return () => window.clearInterval(id);
  }, [loadData]);

  const triggerScan = async () => {
    if (scanning) return;
    setScanning(true);
    try {
      await postJSON("/scan/trigger");
      setTimeout(loadData, 3000); // re-load after 3s
    } catch {
      /* ignore */
    } finally {
      setTimeout(() => setScanning(false), 3000);
    }
  };

  const filteredRecs = useMemo(() => {
    let list = filter === "actionable" ? recs.filter((r) => r.is_actionable) : recs;
    list = [...list].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return list;
  }, [recs, filter, sortKey, sortAsc]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc((v) => !v);
    else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const scanInProgress = summary?.scan_in_progress ?? false;

  return (
    <div className="min-h-screen overflow-x-hidden bg-[#07131c] text-slate-50">
      {/* Background */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(52,211,153,0.18),transparent_24%),radial-gradient(circle_at_80%_20%,rgba(34,211,238,0.16),transparent_24%),radial-gradient(circle_at_50%_100%,rgba(251,191,36,0.1),transparent_30%)]" />
        <div className="absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] [background-size:72px_72px]" />
      </div>

      <Navigation />

      <main className="relative z-10 px-4 pb-24 pt-28 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">

          {/* Hero */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-8 rounded-[2rem] border border-white/10 bg-[linear-gradient(140deg,rgba(9,20,32,0.88),rgba(7,19,28,0.72))] p-7 shadow-[0_30px_90px_rgba(4,12,19,0.42)] backdrop-blur-xl sm:p-8"
          >
            <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-100">
                  <Activity className="h-4 w-4" />
                  {scanInProgress ? "Scan in progress…" : "Live trading system"}
                </div>
                <h1 className="mt-5 text-5xl font-black leading-tight text-white sm:text-6xl">
                  AI Trade
                  <span className="block bg-gradient-to-r from-emerald-300 via-cyan-200 to-amber-200 bg-clip-text text-transparent">
                    Recommendation System
                  </span>
                </h1>
                <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
                  NASDAQ-100 + SPY/QQQ scanned with FTA + agent scoring.
                  Five autonomous paper traders track each position until target or bust.
                  Auto-refreshes every 60 s.
                </p>
              </div>

              {/* Status bar */}
              <div className="flex shrink-0 flex-col gap-3">
                <div className="flex flex-wrap gap-2">
                  {[
                    {
                      icon: <Target className="h-3.5 w-3.5" />,
                      label: "Actionable",
                      value: recs.filter((r) => r.is_actionable).length,
                      color: "emerald",
                    },
                    {
                      icon: <Users className="h-3.5 w-3.5" />,
                      label: "Active users",
                      value: summary?.active_users ?? "—",
                      color: "cyan",
                    },
                    {
                      icon: <BarChart2 className="h-3.5 w-3.5" />,
                      label: "Open positions",
                      value: summary?.total_open_positions ?? "—",
                      color: "amber",
                    },
                  ].map((s) => (
                    <div
                      key={s.label}
                      className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm ${
                        s.color === "emerald"
                          ? "border-emerald-400/20 bg-emerald-400/8 text-emerald-200"
                          : s.color === "cyan"
                            ? "border-cyan-400/20 bg-cyan-400/8 text-cyan-200"
                            : "border-amber-400/20 bg-amber-400/8 text-amber-200"
                      }`}
                    >
                      {s.icon}
                      <span className="font-bold">{s.value}</span>
                      <span className="text-xs opacity-75">{s.label}</span>
                    </div>
                  ))}
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => navigate("/ai-playground")}
                    className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 hover:bg-white/10"
                  >
                    <ArrowLeft className="h-4 w-4" /> Back
                  </button>
                  <button
                    type="button"
                    onClick={loadData}
                    className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 hover:bg-white/10"
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                    Refresh
                  </button>
                  <button
                    type="button"
                    onClick={triggerScan}
                    disabled={scanning || scanInProgress}
                    className="flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-4 py-2 text-sm font-semibold text-cyan-200 hover:bg-cyan-400/20 disabled:opacity-50"
                  >
                    <Zap className={`h-4 w-4 ${scanning ? "animate-pulse" : ""}`} />
                    {scanning || scanInProgress ? "Scanning…" : "Trigger scan"}
                  </button>
                </div>
                {lastRefresh && (
                  <p className="text-right text-[10px] text-slate-600">
                    Last refresh: {lastRefresh.toLocaleTimeString()}
                  </p>
                )}
              </div>
            </div>
          </motion.div>

          {/* Error banner */}
          {error && (
            <div className="mb-6 flex items-center gap-3 rounded-2xl border border-amber-400/20 bg-amber-400/8 px-5 py-4 text-sm text-amber-200">
              <TriangleAlert className="h-4 w-4 shrink-0" />
              <span>
                {error.includes("unavailable")
                  ? "Python trading service is offline. Start it with: uvicorn src.api.server:app --port 8000"
                  : error}
              </span>
            </div>
          )}

          {/* ── Users row ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-8"
          >
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-widest text-slate-500">
              Autonomous traders
            </h2>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              {loading && !summary
                ? Array.from({ length: 5 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-40 animate-pulse rounded-[1.6rem] border border-white/8 bg-white/4"
                    />
                  ))
                : (summary?.users ?? []).map((u) => (
                    <button
                      key={u.user_id}
                      type="button"
                      onClick={() => setSelectedUserId(u.user_id)}
                      className="group rounded-[1.6rem] border border-white/10 bg-white/4 p-4 text-left backdrop-blur-sm transition hover:border-white/20 hover:bg-white/8"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-2xl">{u.avatar}</span>
                        <span
                          className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${statusBadge(u.status)}`}
                        >
                          {u.status.replace("_", " ")}
                        </span>
                      </div>
                      <p className="mt-2 font-bold text-white">{u.name}</p>
                      <p
                        className={`mt-0.5 text-xl font-black ${pctColor(u.return_pct)}`}
                      >
                        {u.return_pct >= 0 ? "+" : ""}
                        {fmt(u.return_pct)}%
                      </p>
                      <div className="mt-3 grid grid-cols-2 gap-x-2 gap-y-1 text-[11px] text-slate-400">
                        <span>
                          Pos:{" "}
                          <span className="text-slate-200">{u.open_positions}</span>
                        </span>
                        <span>
                          Win:{" "}
                          <span className="text-slate-200">
                            {fmt(u.win_rate * 100, 0)}%
                          </span>
                        </span>
                        <span className="col-span-2">
                          Trades:{" "}
                          <span className="text-slate-200">{u.total_trades}</span>
                        </span>
                      </div>
                    </button>
                  ))}
            </div>
          </motion.section>

          {/* ── Recommendations table ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
                Scan results
                {!loading && (
                  <span className="ml-2 text-slate-600">
                    ({filteredRecs.length} shown)
                  </span>
                )}
              </h2>
              <div className="flex gap-2">
                {(["all", "actionable"] as const).map((f) => (
                  <button
                    key={f}
                    type="button"
                    onClick={() => setFilter(f)}
                    className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
                      filter === f
                        ? "bg-cyan-400/20 text-cyan-200"
                        : "text-slate-400 hover:text-slate-300"
                    }`}
                  >
                    {f === "all" ? "All" : "Actionable"}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-hidden rounded-[1.8rem] border border-white/10 bg-white/4 backdrop-blur-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/8">
                      {(
                        [
                          { key: "ticker", label: "Ticker" },
                          { key: "action", label: "Signal" },
                          { key: "composite_score", label: "Score" },
                          { key: "agent_score", label: "Agent" },
                          { key: "fta_score", label: "FTA" },
                        ] as { key: SortKey; label: string }[]
                      ).map((col) => (
                        <th
                          key={col.key}
                          onClick={() => toggleSort(col.key)}
                          className="cursor-pointer px-5 py-3 text-left text-xs font-semibold uppercase tracking-widest text-slate-500 hover:text-slate-300"
                        >
                          {col.label}{" "}
                          {sortKey === col.key ? (sortAsc ? "↑" : "↓") : ""}
                        </th>
                      ))}
                      <th className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-widest text-slate-500">
                        Strategy
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {loading && filteredRecs.length === 0 ? (
                      Array.from({ length: 8 }).map((_, i) => (
                        <tr key={i} className="border-b border-white/5">
                          {Array.from({ length: 6 }).map((_, j) => (
                            <td key={j} className="px-5 py-3">
                              <div className="h-4 animate-pulse rounded-full bg-white/8" />
                            </td>
                          ))}
                        </tr>
                      ))
                    ) : filteredRecs.length === 0 ? (
                      <tr>
                        <td
                          colSpan={6}
                          className="px-5 py-12 text-center text-slate-600"
                        >
                          {error
                            ? "No data — service offline"
                            : "No recommendations yet. Trigger a scan to begin."}
                        </td>
                      </tr>
                    ) : (
                      filteredRecs.map((r) => (
                        <tr
                          key={r.ticker}
                          className="border-b border-white/5 transition hover:bg-white/4"
                        >
                          <td className="px-5 py-3 font-bold text-white">
                            {r.ticker}
                          </td>
                          <td className="px-5 py-3">
                            <span
                              className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${actionBadge(r.action)}`}
                            >
                              {r.action}
                            </span>
                          </td>
                          <td className="w-36 px-5 py-3">
                            <div className="flex items-center gap-2">
                              <ScoreBar
                                value={r.composite_score}
                                color={
                                  r.composite_score >= 0.7
                                    ? "emerald"
                                    : r.composite_score >= 0.55
                                      ? "cyan"
                                      : "amber"
                                }
                              />
                              <span className="w-8 shrink-0 text-right text-xs text-slate-300">
                                {fmt(r.composite_score, 2)}
                              </span>
                            </div>
                          </td>
                          <td className="w-28 px-5 py-3">
                            <div className="flex items-center gap-2">
                              <ScoreBar value={r.agent_score} color="cyan" />
                              <span className="w-8 shrink-0 text-right text-xs text-slate-400">
                                {fmt(r.agent_score, 2)}
                              </span>
                            </div>
                          </td>
                          <td className="w-28 px-5 py-3">
                            <div className="flex items-center gap-2">
                              <ScoreBar value={r.fta_score} color="amber" />
                              <span className="w-8 shrink-0 text-right text-xs text-slate-400">
                                {fmt(r.fta_score, 2)}
                              </span>
                            </div>
                          </td>
                          <td className="px-5 py-3 text-xs text-slate-500">
                            {r.option_strategy_type?.replace(/_/g, " ") ?? "—"}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.section>

          {/* Last scan info */}
          {summary?.last_scan_at && (
            <p className="mt-4 text-center text-xs text-slate-700">
              Last scan:{" "}
              {new Date(summary.last_scan_at).toLocaleString()}
              {summary.scan_error && (
                <span className="ml-2 text-amber-600"> · {summary.scan_error}</span>
              )}
            </p>
          )}
        </div>
      </main>

      {/* User detail side panel */}
      <AnimatePresence>
        {selectedUserId && (
          <UserDetailPanel
            userId={selectedUserId}
            onClose={() => setSelectedUserId(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
