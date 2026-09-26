import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Activity, ArrowLeft, ExternalLink, RefreshCw } from "lucide-react";
import Navigation from "@/components/Navigation";
import {
  loadPaperSnapshot,
  observationAge,
  SNAPSHOT_URL,
  type PaperLeg,
  type SnapshotResult,
} from "@/lib/trade-system-snapshot";

const usd = (value: number | null) =>
  value == null
    ? "Unavailable"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
      }).format(value);
const words = (value: string) => value.replace(/_/g, " ");
const time = (value: string) =>
  new Date(value).toLocaleString(undefined, { timeZoneName: "short" });
const panel = "min-w-0 rounded-2xl border border-white/10 bg-white/[0.035] p-5 sm:p-6";
const button =
  "inline-flex items-center justify-center gap-2 rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm text-slate-200 hover:bg-white/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cyan-300 disabled:opacity-50";

function LegDetails({ legs }: { legs: PaperLeg[] }) {
  if (!legs.length)
    return (
      <p className="text-sm text-slate-400">No contract details recorded.</p>
    );
  return (
    <ul className="grid gap-3 text-sm">
      {legs.map((leg, index) => (
        <li key={index} className="grid gap-1">
          <span className="text-slate-200">
            {words(leg.side)} {leg.ticker} {usd(leg.strike)} {leg.option_type} ·{" "}
            {leg.expiry}
          </span>
          <span className="text-slate-400">
            Recorded bid {usd(leg.bid)} / ask {usd(leg.ask)} ·{" "}
            {leg.quote_sources.length
              ? leg.quote_sources.join(", ")
              : "source unavailable"}
          </span>
          <span className="text-slate-400">
            Source quote time:{" "}
            {leg.source_quote_at ? time(leg.source_quote_at) : "unavailable"}
          </span>
        </li>
      ))}
    </ul>
  );
}

export default function TradeRecommendationSystemDemo() {
  const [result, setResult] = useState<SnapshotResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const request = useRef<AbortController | null>(null);
  const refresh = useCallback(async () => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError(null);
    try {
      const next = await loadPaperSnapshot({ signal: controller.signal });
      if (!controller.signal.aborted) {
        setResult(next);
        setNow(Date.now());
      }
    } catch (err) {
      if (!controller.signal.aborted)
        setError(
          err instanceof Error ? err.message : "Unable to load observation.",
        );
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);
  useEffect(() => {
    void refresh();
    const interval = setInterval(() => setNow(Date.now()), 60_000);
    return () => {
      request.current?.abort();
      clearInterval(interval);
    };
  }, [refresh]);

  const snapshot = result?.snapshot;
  const legs = snapshot
    ? [...snapshot.recommendations, ...snapshot.position_observations].flatMap(
        (row) => row.legs,
      )
    : [];
  const missingQuoteClocks =
    !legs.length || legs.some((leg) => !leg.source_quote_at);

  return (
    <div className="min-h-screen bg-[#07131c] text-slate-50">
      <Navigation />
      <main className="mx-auto grid max-w-7xl gap-6 px-4 pb-20 pt-28 sm:px-6 lg:px-8">
        <header
          className={`${panel} bg-[linear-gradient(140deg,rgba(9,34,45,0.9),rgba(7,19,28,0.9))]`}
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <p className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-cyan-200">
              <Activity className="h-4 w-4" aria-hidden="true" />
              Daily paper snapshot
            </p>
            <div className="flex gap-2">
              <Link to="/ai-playground" className={button}>
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                Back
              </Link>
              <button
                type="button"
                className={button}
                onClick={() => void refresh()}
                disabled={loading}
              >
                <RefreshCw
                  className={`h-4 w-4 ${loading ? "animate-spin motion-reduce:animate-none" : ""}`}
                  aria-hidden="true"
                />
                {loading ? "Loading…" : "Refresh snapshot"}
              </button>
            </div>
          </div>
          <h1 className="mt-5 max-w-3xl text-3xl font-bold leading-tight text-white sm:text-5xl">
            Trade Recommendation System
          </h1>
          <p className="mt-4 max-w-3xl leading-7 text-slate-300">
            A recorded view of the SPY options research control: candidate
            recommendations, existing paper positions, and paper P&amp;L. This
            page is read-only and does not place orders.
          </p>
          <p className="mt-3 text-sm leading-6 text-slate-400">
            Observations are produced by the daily control run. Refresh checks
            for a newer published record; this is not a streaming market feed.
          </p>
        </header>
        {error && (
          <div
            role="alert"
            className="rounded-2xl border border-amber-300/30 bg-amber-300/10 p-5 text-sm leading-6 text-amber-100"
          >
            {error}
            {snapshot &&
              " The previous observation remains below; its recorded time has not changed."}
          </div>
        )}
        {loading && !snapshot && (
          <p role="status" className="py-12 text-center text-slate-300">
            Loading the published paper observation…
          </p>
        )}
        {snapshot && (
          <>
            <section
              aria-label="Observation status"
              className={`${panel} grid gap-4 md:grid-cols-2`}
            >
              <div>
                <h2 className="font-semibold text-white">
                  Observed {observationAge(snapshot.observed_at, now)}
                </h2>
                <p className="mt-2 text-sm text-slate-300">
                  <time dateTime={snapshot.observed_at}>
                    {time(snapshot.observed_at)}
                  </time>
                </p>
                <p className="mt-2 text-sm text-slate-400">
                  Session {snapshot.session_date} ·{" "}
                  {result.source === "observation_api"
                    ? "Observation endpoint"
                    : "Published snapshot"}
                </p>
                {result.fallback && (
                  <p className="mt-2 text-sm text-amber-200">
                    The observation endpoint is unavailable; showing the
                    published snapshot.
                  </p>
                )}
              </div>
              <div className="grid content-start gap-2 text-sm leading-6">
                <p className="text-cyan-200">
                  Paper entries{" "}
                  {snapshot.authority.paper_entries_enabled
                    ? "enabled"
                    : "disabled"}{" "}
                  in the recorded control state. Live execution{" "}
                  {snapshot.authority.live_execution_enabled
                    ? "enabled in recorded state"
                    : "disabled"}
                  .
                </p>
                <p className="text-slate-300">
                  {missingQuoteClocks
                    ? "Source quote timestamps are unavailable for some or all contracts. Quote freshness cannot be verified from this record."
                    : "Contract quote timestamps are listed below. They may be older than the observation."}
                </p>
              </div>
            </section>
            <section
              aria-label="Recorded paper results"
              className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
            >
              {[
                ["Realized paper P&L", usd(snapshot.metrics.realized_pnl)],
                ["Unrealized paper P&L", usd(snapshot.metrics.unrealized_pnl)],
                ["Open paper positions", snapshot.metrics.open_positions],
                ["Closed paper trades", snapshot.metrics.closed_trades],
              ].map(([label, value]) => (
                <div className={panel} key={label}>
                  <p className="text-sm text-slate-400">{label}</p>
                  <p className="mt-3 text-2xl font-semibold tabular-nums text-white">
                    {value}
                  </p>
                </div>
              ))}
            </section>
            <section className={panel}>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="text-xl font-semibold text-white">
                  Recorded candidate recommendations
                </h2>
                <span className="text-sm text-slate-400">
                  {snapshot.recommendations.length} candidates
                </span>
              </div>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                Candidates are research outputs, not executed fills. Agent
                confidence is a heuristic score, not a calibrated probability of
                profit. The recorded entry price is a proposed option premium
                per share.
              </p>
              {!snapshot.recommendations.length ? (
                <p className="mt-6 text-slate-300">
                  No candidates were recorded in this observation.
                </p>
              ) : (
                <div className="mt-5 overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <caption className="sr-only">
                      SPY candidate recommendations from the dated paper journal
                    </caption>
                    <thead>
                      <tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-400">
                        {[
                          "Rank",
                          "Strategy / agent",
                          "Heuristic confidence",
                          "Rank score",
                          "Proposed debit",
                          "Expiry",
                        ].map((label) => (
                          <th
                            scope="col"
                            className="px-3 py-3 font-medium"
                            key={label}
                          >
                            {label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[...snapshot.recommendations]
                        .sort((a, b) => a.rank - b.rank)
                        .map((row) => (
                          <tr
                            key={`${row.rank}-${row.agent}`}
                            className="border-b border-white/5 align-top"
                          >
                            <td className="px-3 py-4 tabular-nums">
                              {row.rank}
                            </td>
                            <td className="min-w-56 px-3 py-4">
                              <p className="font-medium text-white">
                                {row.ticker} · {words(row.strategy)}
                              </p>
                              <p className="mt-1 text-xs text-slate-400">
                                {row.agent}
                              </p>
                              <details className="mt-3">
                                <summary className="cursor-pointer text-cyan-200 focus-visible:outline focus-visible:outline-cyan-300">
                                  Recorded contract quotes
                                </summary>
                                <div className="mt-3">
                                  <LegDetails legs={row.legs} />
                                </div>
                              </details>
                            </td>
                            <td className="px-3 py-4 tabular-nums">
                              {(row.confidence * 100).toFixed(0)}%
                            </td>
                            <td className="px-3 py-4 tabular-nums">
                              {row.score.toFixed(4)}
                            </td>
                            <td className="px-3 py-4 tabular-nums">
                              {usd(row.entry_net_debit)}
                              {row.entry_net_credit !== null && (
                                <span className="block text-xs text-slate-400">
                                  Credit {usd(row.entry_net_credit)}
                                </span>
                              )}
                            </td>
                            <td className="whitespace-nowrap px-3 py-4">
                              {row.expiry}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
            <section className={panel}>
              <h2 className="text-xl font-semibold text-white">
                Existing paper positions
              </h2>
              <p className="mt-3 text-sm leading-6 text-slate-400">
                Recorded liquidation marks are estimates from the observed
                bid/ask data. Paper records do not establish broker execution or
                realized investment returns.
              </p>
              {!snapshot.position_observations.length ? (
                <p className="mt-5 text-slate-300">
                  No position observations were recorded.
                </p>
              ) : (
                <div className="mt-5 grid gap-4">
                  {snapshot.position_observations.map((position, index) => (
                    <article
                      key={index}
                      className="grid gap-4 rounded-xl border border-white/10 p-4 md:grid-cols-2"
                    >
                      <div className="grid content-start gap-2">
                        <h3 className="font-semibold text-white">
                          {position.ticker} · {words(position.strategy)}
                        </h3>
                        <p className="text-sm text-slate-300">
                          {position.contracts} contract
                          {position.contracts === 1 ? "" : "s"} · expires{" "}
                          {position.expiry}
                        </p>
                        <p className="text-sm text-cyan-200">
                          {position.action} · {words(position.reason)}
                        </p>
                        <p className="text-sm text-slate-400">
                          Paper entry {usd(position.entry_price)} · liquidation
                          mark {usd(position.liquidation_mark)} per share
                        </p>
                        <p className="text-sm text-slate-300">
                          Unrealized paper P&amp;L{" "}
                          {usd(position.unrealized_pnl)}
                        </p>
                      </div>
                      <LegDetails legs={position.legs} />
                    </article>
                  ))}
                </div>
              )}
            </section>
            <section className={`${panel} grid gap-5 md:grid-cols-2`}>
              <div>
                <h2 className="text-xl font-semibold text-white">
                  Where this record comes from
                </h2>
                <p className="mt-3 text-sm leading-6 text-slate-300">
                  The OptionsAgents MVP paper-control journal contains{" "}
                  {snapshot.source.record_count} recorded observations. This
                  page presents its latest timestamped record.
                </p>
                <p className="mt-3 text-sm leading-6 text-slate-400">
                  Selected decision inputs{" "}
                  {snapshot.source.input_snapshot_verified
                    ? "were verified against their saved content hash"
                    : "could not be verified against a saved input snapshot"}
                  . A complete upstream raw-data lineage is unavailable.
                </p>
              </div>
              <dl className="grid content-start gap-3 text-sm">
                <div>
                  <dt className="text-slate-400">Snapshot exported</dt>
                  <dd className="mt-1 text-slate-200">
                    <time dateTime={snapshot.generated_at}>
                      {time(snapshot.generated_at)}
                    </time>
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-400">Source journal</dt>
                  <dd className="mt-1 break-all font-mono text-xs text-slate-300">
                    {snapshot.source.journal}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-400">Journal SHA-256</dt>
                  <dd className="mt-1 break-all font-mono text-xs text-slate-300">
                    {snapshot.source.journal_sha256}
                  </dd>
                </div>
              </dl>
            </section>
          </>
        )}
        <footer className="flex flex-wrap items-center justify-between gap-4 border-t border-white/10 pt-6 text-sm">
          <a className={button} href={SNAPSHOT_URL} download>
            Download published snapshot
          </a>
          <a className={button} href="/btc-oracle/site/home.html">
            Separate project: BTC Oracle{" "}
            <ExternalLink className="h-4 w-4" aria-hidden="true" />
          </a>
        </footer>
      </main>
    </div>
  );
}
