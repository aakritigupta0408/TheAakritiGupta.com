import { useEffect, useMemo, useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Database,
  Pause,
  Play,
  RefreshCw,
  ShieldCheck,
  Target,
  TrendingUp,
  TriangleAlert,
  Workflow,
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
import {
  TRADE_SYSTEM_FRAMES,
  type DemoRecommendation,
  type DemoTickerFrame,
} from "@/data/tradeSystemDemo";

const REPLAY_INTERVAL_MS = 2200;

const CHART_TOOLTIP_STYLE = {
  background: "rgba(7, 19, 28, 0.95)",
  borderRadius: "18px",
  border: "1px solid rgba(255,255,255,0.1)",
};

export default function TradeRecommendationSystemDemo() {
  const navigate = useNavigate();
  const [frameIndex, setFrameIndex] = useState(0);
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [isPlaying, setIsPlaying] = useState(false);

  const frame = TRADE_SYSTEM_FRAMES[frameIndex];
  const selectedTickerFrame =
    frame.tickers.find((item) => item.ticker === selectedTicker) ?? frame.tickers[0];
  const recommendation = selectedTickerFrame.recommendation;
  const budgetUsedPct = Math.round(
    (frame.requestBudget.used / frame.requestBudget.limit) * 100,
  );
  const latestDecision =
    selectedTickerFrame.decisionLog[selectedTickerFrame.decisionLog.length - 1];
  const decisionStages = useMemo(
    () => buildDecisionStages(selectedTickerFrame),
    [selectedTickerFrame],
  );
  const frameSteps = useMemo(
    () =>
      TRADE_SYSTEM_FRAMES.map((item, index) => {
        const [time, phase] = item.label.split("·").map((part) => part.trim());
        return {
          index,
          time,
          phase: phase ?? item.label,
        };
      }),
    [],
  );

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
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(52,211,153,0.18),transparent_24%),radial-gradient(circle_at_80%_20%,rgba(34,211,238,0.16),transparent_24%),radial-gradient(circle_at_50%_100%,rgba(251,191,36,0.1),transparent_30%)]" />
        <div className="absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] [background-size:72px_72px]" />
      </div>

      <Navigation />

      <main className="relative z-10">
        <section className="px-6 pb-12 pt-28 sm:px-8">
          <div className="mx-auto max-w-7xl">
            <div className="grid gap-8 xl:grid-cols-[1.12fr_0.88fr]">
              <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7 }}
                className="rounded-[2.3rem] border border-white/10 bg-[linear-gradient(140deg,rgba(9,20,32,0.88),rgba(7,19,28,0.72))] p-7 shadow-[0_30px_90px_rgba(4,12,19,0.42)] backdrop-blur-xl sm:p-8"
              >
                <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-100">
                  <Activity className="h-4 w-4" />
                  Interactive system replay
                </div>

                <h1 className="mt-6 max-w-4xl text-5xl font-black leading-[0.92] text-white sm:text-7xl">
                  AI Trade
                  <span className="block bg-gradient-to-r from-emerald-300 via-cyan-200 to-amber-200 bg-clip-text text-transparent">
                    Recommendation System
                  </span>
                </h1>

                <p className="mt-6 max-w-3xl text-lg leading-8 text-slate-300">
                  This replay is designed like an operating room for the product,
                  not a generic dashboard. It surfaces how the system ingests
                  locally, respects Alpha Vantage free-tier limits, makes
                  deterministic decisions, and closes the day with adaptive paper
                  learning.
                </p>

                <div className="mt-8 flex flex-wrap gap-3 text-sm">
                  <HeroChip label="Local-first retrieval" />
                  <HeroChip label="Daily-only forecasts" />
                  <HeroChip label="Deterministic paper execution" />
                  <HeroChip label="Restart-safe loop state" />
                </div>

                <div className="mt-8 grid gap-4 lg:grid-cols-3">
                  <PrincipleCard
                    tone="emerald"
                    title="What the engine trusts"
                    body="Local normalized bars, indicators, fundamentals, earnings, and optional news snapshots."
                  />
                  <PrincipleCard
                    tone="cyan"
                    title="What it avoids"
                    body="Repeated pulls, hidden API reads inside model code, and forced trades when FTA says no."
                  />
                  <PrincipleCard
                    tone="amber"
                    title="Fallback behavior"
                    body="No real 1h feed is detected, so the system stays fully operational in clean daily-only mode."
                  />
                </div>
              </motion.div>

              <motion.aside
                initial={{ opacity: 0, y: 24, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.7, delay: 0.12 }}
                className="rounded-[2.3rem] border border-white/10 bg-white/5 p-6 shadow-[0_25px_80px_rgba(5,18,27,0.45)] backdrop-blur-xl"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                      Mission control
                    </p>
                    <h2 className="mt-2 text-2xl font-black text-white">{frame.label}</h2>
                  </div>
                  <div className="rounded-full border border-white/10 bg-slate-950/60 px-4 py-2 text-sm text-slate-200">
                    {frameIndex + 1} / {TRADE_SYSTEM_FRAMES.length}
                  </div>
                </div>

                <p className="mt-4 text-sm leading-7 text-slate-300">{frame.summary}</p>

                <div className="mt-6 grid gap-3 sm:grid-cols-2">
                  {frameSteps.map((step) => (
                    <button
                      key={step.index}
                      type="button"
                      onClick={() => {
                        setIsPlaying(false);
                        setFrameIndex(step.index);
                      }}
                      className={`rounded-[1.35rem] border px-4 py-4 text-left transition ${
                        step.index === frameIndex
                          ? "border-cyan-300/35 bg-cyan-400/10 shadow-[0_18px_36px_rgba(34,211,238,0.14)]"
                          : "border-white/10 bg-slate-950/35 hover:border-white/20 hover:bg-white/5"
                      }`}
                    >
                      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                        {step.time}
                      </p>
                      <p className="mt-2 text-sm font-semibold text-white">{step.phase}</p>
                    </button>
                  ))}
                </div>

                <div className="mt-6 rounded-[1.6rem] border border-white/10 bg-slate-950/45 p-5">
                  <div className="flex items-center justify-between gap-4 text-sm">
                    <span className="text-slate-300">Request budget used</span>
                    <span className="font-semibold text-white">
                      {frame.requestBudget.used} / {frame.requestBudget.limit}
                    </span>
                  </div>
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/8">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-cyan-400 to-amber-300"
                      style={{ width: `${budgetUsedPct}%` }}
                    />
                  </div>
                  <p className="mt-3 text-sm leading-7 text-slate-300">
                    {frame.requestBudget.remaining} requests remain because the
                    loop prefers the local store and skips already-covered data.
                  </p>
                </div>

                <div className="mt-6 grid gap-3 sm:grid-cols-2">
                  <ControlButton
                    icon={<ArrowLeft className="h-4 w-4" />}
                    label="Back to demos"
                    onClick={() => navigate("/ai-playground")}
                  />
                  <ControlButton
                    icon={isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    label={isPlaying ? "Pause replay" : "Run replay"}
                    onClick={() => setIsPlaying((current) => !current)}
                    highlighted
                  />
                  <ControlButton
                    icon={<Clock3 className="h-4 w-4" />}
                    label="Next tick"
                    onClick={advanceFrame}
                    disabled={frameIndex >= TRADE_SYSTEM_FRAMES.length - 1}
                  />
                  <ControlButton
                    icon={<RefreshCw className="h-4 w-4" />}
                    label="Reset replay"
                    onClick={resetReplay}
                  />
                </div>
              </motion.aside>
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
              icon={<Target className="h-5 w-5" />}
              label="Selected posture"
              value={`${recommendation.action} · ${recommendation.positionAction}`}
              note={selectedTickerFrame.ticker}
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
          <div className="mx-auto grid max-w-7xl gap-6 xl:grid-cols-[1.08fr_0.92fr]">
            <div className="space-y-6">
              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-100">
                      Watchlist overview
                    </p>
                    <h2 className="mt-2 text-3xl font-black text-white">
                      Recommendation feed
                    </h2>
                  </div>
                  <p className="max-w-2xl text-sm leading-7 text-slate-300">
                    Each card reflects the single decision layer after local-only
                    feature assembly, forecast generation, FTA gating, and the
                    meta-model check.
                  </p>
                </div>

                <div className="mt-8 grid gap-4 lg:grid-cols-3">
                  {frame.tickers.map((tickerFrame) => {
                    const selected = tickerFrame.ticker === selectedTickerFrame.ticker;
                    const actionTone = getActionTone(tickerFrame.recommendation.action);

                    return (
                      <button
                        key={tickerFrame.ticker}
                        type="button"
                        onClick={() => setSelectedTicker(tickerFrame.ticker)}
                        className={`rounded-[1.7rem] border p-5 text-left transition ${
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
                            className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${actionTone.badgeClass}`}
                          >
                            {tickerFrame.recommendation.action}
                          </span>
                        </div>

                        <div className="mt-6 flex items-center justify-between text-sm">
                          <span
                            className={
                              tickerFrame.changePct >= 0
                                ? "text-emerald-200"
                                : "text-rose-200"
                            }
                          >
                            {formatPercent(tickerFrame.changePct)}
                          </span>
                          <span className="text-slate-400">
                            {formatLabel(tickerFrame.recommendation.positionAction)}
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

              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-100">
                      Selected ticker
                    </p>
                    <div className="mt-2 flex flex-wrap items-center gap-3">
                      <h2 className="text-3xl font-black text-white">
                        {selectedTickerFrame.ticker}
                      </h2>
                      <span
                        className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${
                          selectedTickerFrame.changePct >= 0
                            ? "border border-emerald-300/20 bg-emerald-300/10 text-emerald-100"
                            : "border border-rose-300/20 bg-rose-300/10 text-rose-100"
                        }`}
                      >
                        {formatPercent(selectedTickerFrame.changePct)}
                      </span>
                      <span className="rounded-full border border-white/10 bg-slate-950/50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-300">
                        {formatLabel(recommendation.tradeStyle)}
                      </span>
                    </div>
                  </div>

                  <div className="rounded-[1.2rem] border border-white/10 bg-slate-950/45 px-4 py-3 text-right">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-400">
                      Last journal note
                    </p>
                    <p className="mt-2 text-sm font-semibold text-white">
                      {latestDecision?.note ?? "No decision journal entries"}
                    </p>
                  </div>
                </div>

                <div className="mt-6 grid gap-4 md:grid-cols-3">
                  <MiniStat
                    label="Latest close"
                    value={`$${selectedTickerFrame.latestClose.toFixed(2)}`}
                  />
                  <MiniStat
                    label="Forecast confidence"
                    value={recommendation.forecastConfidence.toFixed(2)}
                  />
                  <MiniStat
                    label="Meta-model probability"
                    value={recommendation.probabilityOfSuccess.toFixed(2)}
                  />
                </div>

                <div className="mt-6 grid gap-4 lg:grid-cols-2">
                  <div className="rounded-[1.5rem] border border-white/10 bg-slate-950/45 p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      Price replay
                    </p>
                    <div className="mt-4 h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={selectedTickerFrame.priceSeries}>
                          <CartesianGrid
                            stroke="rgba(255,255,255,0.08)"
                            vertical={false}
                          />
                          <XAxis
                            dataKey="label"
                            stroke="#94a3b8"
                            tickLine={false}
                            axisLine={false}
                          />
                          <YAxis
                            stroke="#94a3b8"
                            tickLine={false}
                            axisLine={false}
                            width={50}
                          />
                          <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
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

                  <div className="rounded-[1.5rem] border border-white/10 bg-slate-950/45 p-5">
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
                          <CartesianGrid
                            stroke="rgba(255,255,255,0.08)"
                            vertical={false}
                          />
                          <XAxis
                            dataKey="label"
                            stroke="#94a3b8"
                            tickLine={false}
                            axisLine={false}
                          />
                          <YAxis
                            stroke="#94a3b8"
                            tickLine={false}
                            axisLine={false}
                            width={70}
                          />
                          <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                          <Area
                            type="monotone"
                            dataKey="equity"
                            stroke="#22d3ee"
                            fill="url(#equityFill)"
                            strokeWidth={3}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-100">
                  Decision brief
                </p>
                <div className="mt-4 rounded-[1.7rem] border border-white/10 bg-slate-950/55 p-6">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <div className="flex flex-wrap items-center gap-3">
                        <ActionBadge recommendation={recommendation} />
                        <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100">
                          {recommendation.timeframeMode}
                        </span>
                      </div>
                      <h3 className="mt-4 text-3xl font-black text-white">
                        {recommendation.action} · {recommendation.positionAction}
                      </h3>
                    </div>
                    <div className="rounded-[1.2rem] border border-white/10 bg-white/5 px-4 py-3 text-right">
                      <p className="text-xs uppercase tracking-[0.18em] text-slate-400">
                        Confidence
                      </p>
                      <p className="mt-2 text-xl font-black text-white">
                        {recommendation.forecastConfidence.toFixed(2)}
                      </p>
                    </div>
                  </div>

                  <p className="mt-5 text-sm leading-7 text-slate-300">
                    {recommendation.rationale}
                  </p>

                  <div className="mt-6 grid grid-cols-2 gap-3 text-sm text-slate-200">
                    <InfoStat label="Entry" value={currency(recommendation.entryPrice)} />
                    <InfoStat label="Stop" value={currency(recommendation.stopPrice)} />
                    <InfoStat label="Target" value={currency(recommendation.target1)} />
                    <InfoStat label="Size" value={quantity(recommendation.positionSize)} />
                    <InfoStat label="FTA score" value={recommendation.ftaScore.toFixed(2)} />
                    <InfoStat
                      label="Meta prob"
                      value={recommendation.probabilityOfSuccess.toFixed(2)}
                    />
                  </div>

                  <div
                    className={`mt-5 rounded-[1.25rem] border px-4 py-4 text-sm leading-7 ${
                      recommendation.rejectionReason
                        ? "border-amber-300/20 bg-amber-300/10 text-amber-50"
                        : "border-emerald-300/20 bg-emerald-300/10 text-emerald-50"
                    }`}
                  >
                    {recommendation.rejectionReason ? (
                      <span>
                        Rejection reason: {recommendation.rejectionReason}
                      </span>
                    ) : (
                      <span>
                        Approved path: forecast, FTA, and meta-model aligned
                        strongly enough to open or close a paper position.
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <div className="flex items-center gap-3">
                  <div className="rounded-full border border-cyan-300/20 bg-cyan-400/10 p-2 text-cyan-100">
                    <Workflow className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-100">
                      Decision stack
                    </p>
                    <h3 className="mt-1 text-xl font-black text-white">
                      Why the system acted this way
                    </h3>
                  </div>
                </div>

                <div className="mt-6 space-y-3">
                  {decisionStages.map((stage) => (
                    <DecisionStageRow
                      key={stage.label}
                      label={stage.label}
                      value={stage.value}
                      detail={stage.detail}
                      tone={stage.tone}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="px-6 pb-16 sm:px-8">
          <div className="mx-auto grid max-w-7xl gap-6 xl:grid-cols-[0.92fr_0.9fr_1.18fr]">
            <div className="space-y-6">
              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-100">
                  Budget trace
                </p>
                <div className="mt-4 h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={budgetCurve}>
                      <defs>
                        <linearGradient id="budgetFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.45} />
                          <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.04} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                      <XAxis
                        dataKey="label"
                        stroke="#94a3b8"
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        stroke="#94a3b8"
                        tickLine={false}
                        axisLine={false}
                        width={40}
                      />
                      <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                      <Area
                        type="monotone"
                        dataKey="used"
                        stroke="#fbbf24"
                        fill="url(#budgetFill)"
                        strokeWidth={3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <p className="mt-4 text-sm leading-7 text-slate-300">
                  Already-covered ranges are skipped. Fresh daily tails and a few
                  high-value signals get budget priority instead.
                </p>
              </div>

              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-300">
                  Local-first data inventory
                </p>
                <div className="mt-5 grid gap-3">
                  <InventoryRow
                    label="Daily bars"
                    value={String(selectedTickerFrame.inventory.dailyBars)}
                  />
                  <InventoryRow
                    label="Intraday bars"
                    value={String(selectedTickerFrame.inventory.intradayBars)}
                  />
                  <InventoryRow
                    label="Real 1h feed"
                    value={
                      selectedTickerFrame.inventory.hasRealIntraday
                        ? "available"
                        : "absent"
                    }
                  />
                  <InventoryRow
                    label="Fundamentals"
                    value={
                      selectedTickerFrame.inventory.fundamentalsFresh
                        ? "fresh"
                        : "stale"
                    }
                  />
                  <InventoryRow
                    label="Earnings"
                    value={
                      selectedTickerFrame.inventory.earningsFresh ? "fresh" : "stale"
                    }
                  />
                  <InventoryRow
                    label="News"
                    value={
                      selectedTickerFrame.inventory.newsFresh
                        ? "fresh"
                        : "reporting disabled"
                    }
                  />
                  <InventoryRow
                    label="Provider indicators"
                    value={selectedTickerFrame.inventory.providerIndicators.join(", ")}
                  />
                </div>
              </div>
            </div>

            <div className="space-y-6">
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

              <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-300">
                  System narrative
                </p>
                <div className="mt-5 space-y-3">
                  <NarrativeRow
                    title="One coherent product"
                    body="Ingest, feature assembly, forecasting, filtering, execution, and EOD updates are presented as one operating flow."
                  />
                  <NarrativeRow
                    title="No hidden leakage"
                    body="Forecasting and model-facing logic depend on the persisted local store instead of provider calls inside the decision path."
                  />
                  <NarrativeRow
                    title="Graceful without intraday"
                    body="The replay makes it explicit that the engine keeps working even when intraday coverage is absent."
                  />
                </div>

                {frame.eodSummary ? (
                  <div className="mt-5 rounded-[1.3rem] border border-emerald-300/20 bg-emerald-300/10 p-4 text-emerald-50">
                    <p className="font-semibold">EOD summary: {frame.eodSummary.summary}</p>
                    <p className="mt-2 text-sm text-emerald-100">
                      {frame.eodSummary.tradesFinalized} trades finalized · adaptive
                      update: {frame.eodSummary.adaptiveUpdate} · meta-model
                      retrained: {frame.eodSummary.metaModelRetrained ? "yes" : "no"}
                    </p>
                  </div>
                ) : null}
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
                  Public website note
                </p>
                <div className="mt-5 rounded-[1.4rem] border border-amber-300/15 bg-amber-300/8 p-5 text-sm leading-7 text-amber-50">
                  This page is an interactive replay of the production Python
                  system. The runtime itself stays local-first and would need a
                  dedicated hosted telemetry layer before exposing live internet
                  state.
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
  highlighted,
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  highlighted?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
        highlighted
          ? "bg-gradient-to-r from-emerald-400 via-cyan-400 to-amber-300 text-slate-950 shadow-[0_18px_40px_rgba(16,185,129,0.28)]"
          : "border border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function HeroChip({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-slate-100">
      {label}
    </span>
  );
}

function PrincipleCard({
  title,
  body,
  tone,
}: {
  title: string;
  body: string;
  tone: "emerald" | "cyan" | "amber";
}) {
  const toneClasses = {
    emerald: "border-emerald-300/15 bg-emerald-300/8 text-emerald-50",
    cyan: "border-cyan-300/15 bg-cyan-300/8 text-cyan-50",
    amber: "border-amber-300/15 bg-amber-300/8 text-amber-50",
  };

  return (
    <div className={`rounded-[1.5rem] border p-5 ${toneClasses[tone]}`}>
      <p className="text-sm font-semibold">{title}</p>
      <p className="mt-3 text-sm leading-7 text-white/80">{body}</p>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.3rem] border border-white/10 bg-slate-950/45 px-4 py-4">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
        {label}
      </p>
      <p className="mt-3 text-lg font-black text-white">{value}</p>
    </div>
  );
}

function ActionBadge({ recommendation }: { recommendation: DemoRecommendation }) {
  const actionTone = getActionTone(recommendation.action);

  return (
    <span
      className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] ${actionTone.badgeClass}`}
    >
      {actionTone.label}
    </span>
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

function DecisionStageRow({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone: "good" | "warn" | "neutral";
}) {
  const toneMap = {
    good: {
      icon: <CheckCircle2 className="h-5 w-5" />,
      iconClass: "text-emerald-200",
      valueClass: "text-emerald-100",
      backgroundClass: "border-emerald-300/15 bg-emerald-300/8",
    },
    warn: {
      icon: <TriangleAlert className="h-5 w-5" />,
      iconClass: "text-amber-200",
      valueClass: "text-amber-100",
      backgroundClass: "border-amber-300/15 bg-amber-300/8",
    },
    neutral: {
      icon: <Database className="h-5 w-5" />,
      iconClass: "text-cyan-200",
      valueClass: "text-cyan-100",
      backgroundClass: "border-cyan-300/15 bg-cyan-300/8",
    },
  }[tone];

  return (
    <div
      className={`rounded-[1.35rem] border p-4 ${toneMap.backgroundClass}`}
    >
      <div className="flex items-start gap-3">
        <div className={toneMap.iconClass}>{toneMap.icon}</div>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-sm font-semibold text-white">{label}</p>
            <span className={`text-xs font-semibold uppercase tracking-[0.18em] ${toneMap.valueClass}`}>
              {value}
            </span>
          </div>
          <p className="mt-2 text-sm leading-7 text-white/80">{detail}</p>
        </div>
      </div>
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

function NarrativeRow({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-[1.35rem] border border-white/10 bg-slate-950/45 p-4">
      <p className="text-sm font-semibold text-white">{title}</p>
      <p className="mt-2 text-sm leading-7 text-slate-300">{body}</p>
    </div>
  );
}

function buildDecisionStages(tickerFrame: DemoTickerFrame) {
  const recommendation = tickerFrame.recommendation;

  return [
    {
      label: "Local store",
      value: `${tickerFrame.inventory.dailyBars} bars ready`,
      detail:
        "Feature assembly reads persisted daily data first, with optional intraday staying absent without breaking the flow.",
      tone: "good" as const,
    },
    {
      label: "Forecast signal",
      value:
        recommendation.forecastConfidence >= 0.7
          ? "constructive"
          : recommendation.forecastConfidence >= 0.58
            ? "mixed"
            : "soft",
      detail: `TimesFM confidence is ${recommendation.forecastConfidence.toFixed(2)} in the current frame.`,
      tone:
        recommendation.forecastConfidence >= 0.7
          ? ("good" as const)
          : recommendation.forecastConfidence >= 0.58
            ? ("neutral" as const)
            : ("warn" as const),
    },
    {
      label: "FTA hard filter",
      value:
        recommendation.ftaScore >= 0.6 ? "accepted" : "blocked",
      detail: `FTA score is ${recommendation.ftaScore.toFixed(2)} and remains authoritative.`,
      tone: recommendation.ftaScore >= 0.6 ? ("good" as const) : ("warn" as const),
    },
    {
      label: "Meta-model gate",
      value:
        recommendation.probabilityOfSuccess >= 0.6 ? "accepted" : "blocked",
      detail: `Meta-model probability is ${recommendation.probabilityOfSuccess.toFixed(2)}.`,
      tone:
        recommendation.probabilityOfSuccess >= 0.6
          ? ("good" as const)
          : ("warn" as const),
    },
    {
      label: "Execution outcome",
      value:
        recommendation.action === "HOLD"
          ? "no new trade"
          : `${recommendation.action.toLowerCase()} ${recommendation.positionAction.toLowerCase()}`,
      detail:
        recommendation.action === "HOLD"
          ? "The engine preserves capital and keeps the recommendation deterministic."
          : "The deterministic execution model converts the decision into a paper-trading action.",
      tone: recommendation.action === "HOLD" ? ("neutral" as const) : ("good" as const),
    },
  ];
}

function getActionTone(action: DemoRecommendation["action"]) {
  switch (action) {
    case "BUY":
      return {
        label: "Approved long bias",
        badgeClass: "border border-emerald-300/20 bg-emerald-300/15 text-emerald-100",
      };
    case "SELL":
      return {
        label: "Close or exit bias",
        badgeClass: "border border-rose-300/20 bg-rose-300/15 text-rose-100",
      };
    default:
      return {
        label: "Standby / hold",
        badgeClass: "border border-amber-300/20 bg-amber-300/15 text-amber-100",
      };
  }
}

function formatPercent(value: number) {
  const pct = (value * 100).toFixed(2);
  return `${value >= 0 ? "+" : ""}${pct}%`;
}

function formatLabel(value: string) {
  return value.replace(/_/g, " ");
}

function currency(value?: number) {
  return typeof value === "number" ? `$${value.toFixed(2)}` : "n/a";
}

function quantity(value?: number) {
  return typeof value === "number" ? `${value.toFixed(0)} shares` : "n/a";
}
