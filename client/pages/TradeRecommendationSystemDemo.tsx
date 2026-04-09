import { useEffect, useMemo, useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  ArrowLeft,
  BrainCircuit,
  Clock3,
  Database,
  Pause,
  Play,
  RefreshCw,
  ShieldCheck,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import ChatBot from "@/components/ChatBot";
import Navigation from "@/components/Navigation";
import { TRADE_SYSTEM_FRAMES } from "@/data/tradeSystemDemo";

const REPLAY_INTERVAL_MS = 2200;

export default function TradeRecommendationSystemDemo() {
  const navigate = useNavigate();
  const [frameIndex, setFrameIndex] = useState(0);
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [isPlaying, setIsPlaying] = useState(false);

  const frame = TRADE_SYSTEM_FRAMES[frameIndex];
  const selectedTickerFrame =
    frame.tickers.find((item) => item.ticker === selectedTicker) ?? frame.tickers[0];

  useEffect(() => {
    if (!frame.tickers.some((item) => item.ticker === selectedTicker)) {
      setSelectedTicker(frame.tickers[0]?.ticker ?? "AAPL");
    }
  }, [frame, selectedTicker]);

  useEffect(() => {
    if (!isPlaying) {
      return;
    }
    const timer = window.setInterval(() => {
      setFrameIndex((current) => {
        if (current >= TRADE_SYSTEM_FRAMES.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, REPLAY_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const equityCurve = useMemo(
    () =>
      TRADE_SYSTEM_FRAMES.map((item) => ({
        label: item.label.split("·")[0].trim(),
        equity: item.systemPortfolio.equity,
      })),
    [],
  );

  const budgetCurve = useMemo(
    () =>
      TRADE_SYSTEM_FRAMES.map((item) => ({
        label: item.label.split("·")[0].trim(),
        used: item.requestBudget.used,
        remaining: item.requestBudget.remaining,
      })),
    [],
  );

  const advanceFrame = () => {
    setFrameIndex((current) => Math.min(current + 1, TRADE_SYSTEM_FRAMES.length - 1));
  };

  const resetReplay = () => {
    setIsPlaying(false);
    setFrameIndex(0);
    setSelectedTicker("AAPL");
  };

  return (
    <div className="min-h-screen overflow-x-hidden bg-[#07131c] text-slate-50">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-24 left-0 h-72 w-72 rounded-full bg-emerald-500/16 blur-3xl" />
        <div className="absolute top-36 right-0 h-80 w-80 rounded-full bg-cyan-500/12 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-96 w-96 rounded-full bg-amber-400/10 blur-3xl" />
      </div>

      <Navigation />

      <main className="relative z-10">
        <section className="px-6 pb-14 pt-28 sm:px-8">
          <div className="mx-auto max-w-7xl">
            <div className="grid gap-10 lg:grid-cols-[1.25fr_0.85fr]">
              <div>
                <motion.div
                  initial={{ opacity: 0, y: 24 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.7 }}
                  className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-100"
                >
                  <Activity className="h-4 w-4" />
                  Interactive system replay
                </motion.div>

                <motion.h1
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.1 }}
                  className="mt-6 max-w-4xl text-5xl font-black leading-[0.92] text-white sm:text-7xl"
                >
                  AI Trade
                  <span className="block bg-gradient-to-r from-emerald-300 via-cyan-200 to-amber-200 bg-clip-text text-transparent">
                    Recommendation System
                  </span>
                </motion.h1>

                <motion.p
                  initial={{ opacity: 0, y: 28 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="mt-6 max-w-3xl text-lg leading-8 text-slate-300"
                >
                  This demo replays the production architecture running in
                  daily-only mode under Alpha Vantage free-tier constraints:
                  local-first ingest, deterministic recommendations, market-hours
                  paper execution, and end-of-day learning.
                </motion.p>

                <motion.div
                  initial={{ opacity: 0, y: 28 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.3 }}
                  className="mt-8 flex flex-wrap gap-3"
                >
                  <button
                    type="button"
                    onClick={() => navigate("/ai-playground")}
                    className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
                  >
                    <ArrowLeft className="h-4 w-4" />
                    Back to demos
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsPlaying((current) => !current)}
                    className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-emerald-400 via-cyan-400 to-amber-300 px-5 py-3 text-sm font-semibold text-slate-950 shadow-[0_18px_40px_rgba(16,185,129,0.28)]"
                  >
                    {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    {isPlaying ? "Pause replay" : "Run replay"}
                  </button>
                </motion.div>
              </div>

              <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-[0_25px_80px_rgba(5,18,27,0.45)] backdrop-blur-xl"
              >
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                      Current frame
                    </p>
                    <h2 className="mt-2 text-2xl font-bold text-white">{frame.label}</h2>
                  </div>
                  <div className="rounded-full border border-white/10 bg-slate-950/50 px-4 py-2 text-sm text-slate-200">
                    {frameIndex + 1} / {TRADE_SYSTEM_FRAMES.length}
                  </div>
                </div>
                <p className="mt-4 text-sm leading-7 text-slate-300">{frame.summary}</p>
                <div className="mt-6 grid gap-3 sm:grid-cols-2">
                  <ControlButton
                    icon={<RefreshCw className="h-4 w-4" />}
                    label="Reset replay"
                    onClick={resetReplay}
                  />
                  <ControlButton
                    icon={<Clock3 className="h-4 w-4" />}
                    label="Next tick"
                    onClick={advanceFrame}
                    disabled={frameIndex >= TRADE_SYSTEM_FRAMES.length - 1}
                  />
                </div>
                <div className="mt-6 rounded-3xl border border-amber-300/15 bg-amber-300/8 p-5 text-sm leading-7 text-amber-50">
                  Public website note: this page is an interactive replay of the
                  live Python system. The production engine itself still runs
                  locally-first and would need its own hosted API/export layer to
                  stream true internet-visible telemetry.
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        <section className="px-6 pb-10 sm:px-8">
          <div className="mx-auto grid max-w-7xl gap-4 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              icon={<ShieldCheck className="h-5 w-5" />}
              label="System mode"
              value="daily_only"
              note="1d prediction source, swing-trade bias"
            />
            <MetricCard
              icon={<Database className="h-5 w-5" />}
              label="Alpha Vantage budget"
              value={`${frame.requestBudget.remaining} remaining`}
              note={`${frame.requestBudget.used} of ${frame.requestBudget.limit} requests used`}
            />
            <MetricCard
              icon={<TrendingUp className="h-5 w-5" />}
              label="Paper portfolio"
              value={`$${frame.systemPortfolio.equity.toLocaleString()}`}
              note={`${frame.systemPortfolio.openPositions} open · ${frame.systemPortfolio.closedTrades} closed`}
            />
            <MetricCard
              icon={<BrainCircuit className="h-5 w-5" />}
              label="Loop status"
              value={frame.loopStatus}
              note={`${frame.marketSession} · generated ${new Date(frame.generatedAt).toLocaleTimeString()}`}
            />
          </div>
        </section>

        <section className="px-6 pb-10 sm:px-8">
          <div className="mx-auto max-w-7xl rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
            <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-100">
                  Live watchlist
                </p>
                <h2 className="mt-2 text-3xl font-black text-white">
                  Recommendation feed
                </h2>
              </div>
              <p className="max-w-2xl text-sm leading-7 text-slate-300">
                Each ticker card reflects what the deterministic recommendation
                layer would surface after local-only feature assembly, forecast,
                FTA, meta-model gating, and paper execution.
              </p>
            </div>

            <div className="mt-8 grid gap-4 lg:grid-cols-3">
              {frame.tickers.map((tickerFrame) => {
                const selected = tickerFrame.ticker === selectedTickerFrame.ticker;
                return (
                  <button
                    key={tickerFrame.ticker}
                    type="button"
                    onClick={() => setSelectedTicker(tickerFrame.ticker)}
                    className={`rounded-[1.6rem] border p-5 text-left transition ${
                      selected
                        ? "border-emerald-300/45 bg-emerald-300/10 shadow-[0_18px_36px_rgba(16,185,129,0.18)]"
                        : "border-white/10 bg-slate-950/35 hover:border-cyan-200/30 hover:bg-white/8"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                          {tickerFrame.ticker}
                        </p>
                        <p className="mt-3 text-3xl font-black text-white">
                          ${tickerFrame.latestClose.toFixed(2)}
                        </p>
                      </div>
                      <span
                        className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${
                          tickerFrame.recommendation.action === "BUY"
                            ? "bg-emerald-300/15 text-emerald-100"
                            : tickerFrame.recommendation.action === "SELL"
                              ? "bg-rose-300/15 text-rose-100"
                              : "bg-amber-300/15 text-amber-100"
                        }`}
                      >
                        {tickerFrame.recommendation.action}
                      </span>
                    </div>

                    <div className="mt-6 flex items-center justify-between text-sm">
                      <span className={tickerFrame.changePct >= 0 ? "text-emerald-200" : "text-rose-200"}>
                        {(tickerFrame.changePct * 100).toFixed(2)}%
                      </span>
                      <span className="text-slate-400">
                        {tickerFrame.recommendation.positionAction}
                      </span>
                    </div>

                    <p className="mt-4 text-sm leading-7 text-slate-300">
                      {tickerFrame.recommendation.rationale}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </section>

        <section className="px-6 pb-10 sm:px-8">
          <div className="mx-auto grid max-w-7xl gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
              <div className="flex items-end justify-between gap-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-100">
                    Selected ticker
                  </p>
                  <h2 className="mt-2 text-3xl font-black text-white">
                    {selectedTickerFrame.ticker}
                  </h2>
                </div>
                <div className="rounded-full border border-white/10 bg-slate-950/45 px-4 py-2 text-sm text-slate-300">
                  {selectedTickerFrame.recommendation.tradeStyle.replace("_", " ")}
                </div>
              </div>

              <div className="mt-6 grid gap-4 lg:grid-cols-2">
                <div className="rounded-[1.4rem] border border-white/10 bg-slate-950/45 p-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Price replay
                  </p>
                  <div className="mt-4 h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={selectedTickerFrame.priceSeries}>
                        <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                        <XAxis dataKey="label" stroke="#94a3b8" tickLine={false} axisLine={false} />
                        <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} width={50} />
                        <Tooltip
                          contentStyle={{
                            background: "rgba(7, 19, 28, 0.95)",
                            borderRadius: "18px",
                            border: "1px solid rgba(255,255,255,0.1)",
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="close"
                          stroke="#34d399"
                          strokeWidth={3}
                          dot={{ r: 4, fill: "#fbbf24" }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="rounded-[1.4rem] border border-white/10 bg-slate-950/45 p-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    System equity curve
                  </p>
                  <div className="mt-4 h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={equityCurve}>
                        <defs>
                          <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.55} />
                            <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.04} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                        <XAxis dataKey="label" stroke="#94a3b8" tickLine={false} axisLine={false} />
                        <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} width={70} />
                        <Tooltip
                          contentStyle={{
                            background: "rgba(7, 19, 28, 0.95)",
                            borderRadius: "18px",
                            border: "1px solid rgba(255,255,255,0.1)",
                          }}
                        />
                        <Area type="monotone" dataKey="equity" stroke="#22d3ee" fill="url(#equityFill)" strokeWidth={3} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-100">
                  Latest recommendation
                </p>
                <div className="mt-4 rounded-[1.5rem] border border-white/10 bg-slate-950/55 p-5">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <h3 className="text-2xl font-black text-white">
                        {selectedTickerFrame.recommendation.action} · {selectedTickerFrame.recommendation.positionAction}
                      </h3>
                      <p className="mt-1 text-sm text-slate-300">
                        {selectedTickerFrame.recommendation.timeframeMode}
                      </p>
                    </div>
                    <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100">
                      conf {selectedTickerFrame.recommendation.forecastConfidence.toFixed(2)}
                    </div>
                  </div>

                  <div className="mt-5 grid grid-cols-2 gap-3 text-sm text-slate-200">
                    <InfoStat label="Entry" value={currency(selectedTickerFrame.recommendation.entryPrice)} />
                    <InfoStat label="Stop" value={currency(selectedTickerFrame.recommendation.stopPrice)} />
                    <InfoStat label="Target" value={currency(selectedTickerFrame.recommendation.target1)} />
                    <InfoStat label="Size" value={quantity(selectedTickerFrame.recommendation.positionSize)} />
                    <InfoStat label="FTA score" value={selectedTickerFrame.recommendation.ftaScore.toFixed(2)} />
                    <InfoStat label="Meta prob" value={selectedTickerFrame.recommendation.probabilityOfSuccess.toFixed(2)} />
                  </div>

                  <p className="mt-5 text-sm leading-7 text-slate-300">
                    {selectedTickerFrame.recommendation.rationale}
                  </p>
                  {selectedTickerFrame.recommendation.rejectionReason ? (
                    <p className="mt-3 text-sm font-semibold text-amber-200">
                      Rejection reason: {selectedTickerFrame.recommendation.rejectionReason}
                    </p>
                  ) : null}
                </div>
              </div>

              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-100">
                  Budget trace
                </p>
                <div className="mt-4 h-52">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={budgetCurve}>
                      <defs>
                        <linearGradient id="budgetFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.45} />
                          <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.04} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                      <XAxis dataKey="label" stroke="#94a3b8" tickLine={false} axisLine={false} />
                      <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} width={40} />
                      <Tooltip
                        contentStyle={{
                          background: "rgba(7, 19, 28, 0.95)",
                          borderRadius: "18px",
                          border: "1px solid rgba(255,255,255,0.1)",
                        }}
                      />
                      <Area type="monotone" dataKey="used" stroke="#fbbf24" fill="url(#budgetFill)" strokeWidth={3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="px-6 pb-16 sm:px-8">
          <div className="mx-auto grid max-w-7xl gap-6 xl:grid-cols-[0.92fr_1.08fr]">
            <div className="space-y-6">
              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-300">
                  Local-first data inventory
                </p>
                <div className="mt-5 grid gap-3">
                  <InventoryRow label="Daily bars" value={String(selectedTickerFrame.inventory.dailyBars)} />
                  <InventoryRow label="Intraday bars" value={String(selectedTickerFrame.inventory.intradayBars)} />
                  <InventoryRow
                    label="Real 1h feed"
                    value={selectedTickerFrame.inventory.hasRealIntraday ? "available" : "absent"}
                  />
                  <InventoryRow
                    label="Fundamentals"
                    value={selectedTickerFrame.inventory.fundamentalsFresh ? "fresh" : "stale"}
                  />
                  <InventoryRow
                    label="Earnings"
                    value={selectedTickerFrame.inventory.earningsFresh ? "fresh" : "stale"}
                  />
                  <InventoryRow
                    label="News"
                    value={selectedTickerFrame.inventory.newsFresh ? "fresh" : "reporting disabled"}
                  />
                  <InventoryRow
                    label="Provider indicators"
                    value={selectedTickerFrame.inventory.providerIndicators.join(", ")}
                  />
                </div>
              </div>

              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-300">
                  Ingest activity
                </p>
                <div className="mt-5 space-y-3">
                  {frame.ingestEvents.map((event) => (
                    <div
                      key={`${event.time}-${event.ticker}-${event.endpoint}`}
                      className="rounded-[1.35rem] border border-white/10 bg-slate-950/45 p-4"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="text-sm font-semibold text-white">
                            {event.ticker} · {event.endpoint}
                          </p>
                          <p className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-400">
                            {event.time}
                          </p>
                        </div>
                        <div className="text-right text-sm text-slate-200">
                          <p>{event.result}</p>
                          <p className="text-xs text-slate-400">{event.rows} rows</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-300">
                  Decision journal
                </p>
                <div className="mt-5 overflow-hidden rounded-[1.5rem] border border-white/10">
                  <table className="min-w-full divide-y divide-white/10 text-left text-sm">
                    <thead className="bg-slate-950/60 text-slate-300">
                      <tr>
                        <th className="px-4 py-3 font-semibold">Time</th>
                        <th className="px-4 py-3 font-semibold">Action</th>
                        <th className="px-4 py-3 font-semibold">Position</th>
                        <th className="px-4 py-3 font-semibold">Note</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 bg-slate-950/40 text-slate-100">
                      {selectedTickerFrame.decisionLog.map((item) => (
                        <tr key={`${item.time}-${item.note}`}>
                          <td className="px-4 py-3">{item.time}</td>
                          <td className="px-4 py-3">{item.action}</td>
                          <td className="px-4 py-3">{item.positionAction}</td>
                          <td className="px-4 py-3 text-slate-300">{item.note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-300">
                  System narrative
                </p>
                <div className="mt-5 rounded-[1.5rem] border border-white/10 bg-slate-950/45 p-5 text-sm leading-7 text-slate-300">
                  <p>
                    The current frame shows how the system behaves as one
                    product: data ingest is budget-aware, features are assembled
                    from local storage only, the decision layer is deterministic,
                    and the paper loop saves state every iteration.
                  </p>
                  {frame.eodSummary ? (
                    <div className="mt-4 rounded-[1.2rem] border border-emerald-300/20 bg-emerald-300/10 p-4 text-emerald-50">
                      <p className="font-semibold">
                        EOD summary: {frame.eodSummary.summary}
                      </p>
                      <p className="mt-2 text-sm text-emerald-100">
                        {frame.eodSummary.tradesFinalized} trades finalized · adaptive update: {frame.eodSummary.adaptiveUpdate} · meta-model retrained: {frame.eodSummary.metaModelRetrained ? "yes" : "no"}
                      </p>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <ChatBot />
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  note,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="rounded-[1.7rem] border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
      <div className="flex items-center gap-3 text-emerald-100">
        <div className="rounded-full border border-emerald-300/20 bg-emerald-300/10 p-2">
          {icon}
        </div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-300">
          {label}
        </p>
      </div>
      <p className="mt-5 text-2xl font-black text-white">{value}</p>
      <p className="mt-2 text-sm leading-6 text-slate-400">{note}</p>
    </div>
  );
}

function ControlButton({
  icon,
  label,
  onClick,
  disabled,
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {icon}
      {label}
    </button>
  );
}

function InfoStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-2 text-base font-semibold text-white">{value}</p>
    </div>
  );
}

function InventoryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-[1.2rem] border border-white/10 bg-slate-950/45 px-4 py-3 text-sm">
      <span className="text-slate-300">{label}</span>
      <span className="font-semibold text-white">{value}</span>
    </div>
  );
}

function currency(value?: number) {
  return typeof value === "number" ? `$${value.toFixed(2)}` : "n/a";
}

function quantity(value?: number) {
  return typeof value === "number" ? `${value.toFixed(0)} shares` : "n/a";
}
