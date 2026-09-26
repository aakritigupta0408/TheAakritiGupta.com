import { z } from "zod";

const timestamp = z.string().datetime({ offset: true });
const number = z.number().finite();
const count = number.int().nonnegative();
const date = z.string().regex(/^\d{4}-\d{2}-\d{2}$/);
const hash = z.string().regex(/^[a-f0-9]{64}$/);
const legSchema = z.object({
  ticker: z.string(),
  option_type: z.string(),
  side: z.string(),
  strike: number.nonnegative(),
  expiry: date,
  bid: number.nonnegative().nullable(),
  ask: number.nonnegative().nullable(),
  source_quote_at: timestamp.nullable(),
  quote_sources: z.array(z.string()),
});

export const paperSnapshotSchema = z.object({
  schema_version: z.literal(1),
  mode: z.literal("paper_observation"),
  generated_at: timestamp,
  observed_at: timestamp,
  session_date: date,
  source: z.object({
    journal: z.literal("logs/mvp_control/journal.jsonl"),
    journal_sha256: hash,
    record_count: count.positive(),
    input_snapshot_sha256: hash.nullable(),
    input_snapshot_verified: z.boolean(),
    raw_lineage: z.literal("unavailable"),
  }),
  authority: z.object({
    paper_entries_enabled: z.boolean(),
    live_execution_enabled: z.boolean(),
  }),
  metrics: z.object({
    realized_pnl: number,
    unrealized_pnl: number,
    open_positions: count,
    closed_trades: count,
  }),
  recommendations: z.array(
    z.object({
      rank: count.positive(),
      ticker: z.string(),
      strategy: z.string(),
      agent: z.string(),
      confidence: number.min(0).max(1),
      score: number,
      expiry: date,
      entry_net_debit: number.nonnegative().nullable(),
      entry_net_credit: number.nonnegative().nullable(),
      legs: z.array(legSchema),
    }),
  ),
  position_observations: z.array(
    z.object({
      ticker: z.string(),
      strategy: z.string(),
      contracts: count,
      expiry: date,
      action: z.string(),
      reason: z.string(),
      entry_price: number.nonnegative(),
      liquidation_mark: number.nullable(),
      unrealized_pnl: number.nullable(),
      legs: z.array(legSchema),
    }),
  ),
});

export type PaperSnapshot = z.infer<typeof paperSnapshotSchema>;
export type PaperLeg = z.infer<typeof legSchema>;
export const SNAPSHOT_URL = `${import.meta.env.BASE_URL}data/trade-system-snapshot.json`;
export type SnapshotResult = {
  snapshot: PaperSnapshot;
  source: "published_snapshot" | "observation_api";
  fallback: boolean;
};

// Static hosting has no /api routes. An optional observation endpoint must
// return this same versioned schema; a legacy trader summary is not accepted.
export async function loadPaperSnapshot({
  apiUrl = import.meta.env.VITE_TRADE_SYSTEM_SNAPSHOT_API as string | undefined,
  fetcher = fetch,
  signal,
}: {
  apiUrl?: string;
  fetcher?: typeof fetch;
  signal?: AbortSignal;
} = {}): Promise<SnapshotResult> {
  const urls = apiUrl ? [apiUrl, SNAPSHOT_URL] : [SNAPSHOT_URL];
  for (const [index, url] of urls.entries()) {
    if (signal?.aborted) throw new Error("Snapshot request cancelled");
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    const timeout = setTimeout(abort, 8000);
    try {
      const response = await fetcher(url, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Snapshot HTTP ${response.status}`);
      const snapshot = paperSnapshotSchema.parse(await response.json());
      if (Date.parse(snapshot.observed_at) > Date.now() + 5 * 60_000)
        throw new Error("Future observation timestamp");
      return {
        snapshot,
        source:
          apiUrl && index === 0 ? "observation_api" : "published_snapshot",
        fallback: Boolean(apiUrl && index > 0),
      };
    } catch {
      if (signal?.aborted) throw new Error("Snapshot request cancelled");
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
    }
  }
  throw new Error(
    "The paper observation could not be loaded. Please try refreshing later.",
  );
}

export function observationAge(observedAt: string, now = Date.now()): string {
  const minutes = Math.max(
    0,
    Math.floor((now - Date.parse(observedAt)) / 60_000),
  );
  if (minutes < 1) return "less than a minute ago";
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  return `${Math.floor(hours / 24)} days ago`;
}
