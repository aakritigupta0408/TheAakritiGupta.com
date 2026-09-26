import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  loadPaperSnapshot,
  observationAge,
  paperSnapshotSchema,
  SNAPSHOT_URL,
} from "./trade-system-snapshot";

const fixture = () =>
  JSON.parse(
    readFileSync(
      "public/data/trade-system-snapshot.json",
      "utf-8",
    ),
  );
const response = (body: unknown, status = 200) =>
  ({ ok: status === 200, status, json: async () => body }) as Response;
afterEach(() => vi.useRealTimers());

describe("published paper observations", () => {
  it("accepts the exported record with explicit unknown quote timestamps", () => {
    const data = paperSnapshotSchema.parse(fixture());
    expect(data.source.input_snapshot_verified).toBe(true);
    expect(data.recommendations[0].legs[0].source_quote_at).toBeNull();
  });
  it("rejects missing metrics, nonfinite values, and legacy summaries", () => {
    const missing = fixture();
    delete missing.metrics.realized_pnl;
    expect(paperSnapshotSchema.safeParse(missing).success).toBe(false);
    const invalid = fixture();
    invalid.metrics.unrealized_pnl = Infinity;
    expect(paperSnapshotSchema.safeParse(invalid).success).toBe(false);
    expect(
      paperSnapshotSchema.safeParse({ total_users: 5, users: [] }).success,
    ).toBe(false);
  });
  it("works on a static host without requesting any legacy API or mutation", async () => {
    const fetcher = vi.fn(async () => response(fixture()));
    const result = await loadPaperSnapshot({ fetcher });
    expect(result.source).toBe("published_snapshot");
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(fetcher).toHaveBeenCalledWith(
      SNAPSHOT_URL,
      expect.objectContaining({ method: "GET", cache: "no-store" }),
    );
  });
  it.each([404, 503])(
    "falls back when the configured observation endpoint returns %s",
    async (status) => {
      const fetcher = vi
        .fn()
        .mockResolvedValueOnce(response({}, status))
        .mockResolvedValueOnce(response(fixture()));
      const result = await loadPaperSnapshot({
        apiUrl: "/api/observations",
        fetcher,
      });
      expect(result.fallback).toBe(true);
      expect(result.source).toBe("published_snapshot");
    },
  );
  it("falls back from a successful HTML response or invalid JSON schema", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response("<!doctype html>"))
      .mockResolvedValueOnce(response(fixture()));
    expect(
      (await loadPaperSnapshot({ apiUrl: "/api/observations", fetcher }))
        .fallback,
    ).toBe(true);
  });
  it("uses a schema-valid configured endpoint and labels its provenance", async () => {
    const fetcher = vi.fn(async () => response(fixture()));
    const result = await loadPaperSnapshot({
      apiUrl: "/api/observations",
      fetcher,
    });
    expect(result.source).toBe("observation_api");
    expect(result.fallback).toBe(false);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it("rejects a future observation rather than labeling it fresh", async () => {
    const data = fixture();
    data.observed_at = new Date(Date.now() + 86_400_000).toISOString();
    await expect(
      loadPaperSnapshot({ fetcher: vi.fn(async () => response(data)) }),
    ).rejects.toThrow("could not be loaded");
  });
  it("does not retry after the page cancels its request", async () => {
    const controller = new AbortController();
    controller.abort();
    const fetcher = vi.fn();
    await expect(
      loadPaperSnapshot({ signal: controller.signal, fetcher }),
    ).rejects.toThrow("cancelled");
    expect(fetcher).not.toHaveBeenCalled();
  });
  it("times out an unavailable endpoint before falling back", async () => {
    vi.useFakeTimers();
    const fetcher = vi
      .fn()
      .mockImplementationOnce(
        (_url, init) =>
          new Promise((_resolve, reject) => {
            init.signal.addEventListener("abort", () =>
              reject(new Error("aborted")),
            );
          }),
      )
      .mockResolvedValueOnce(response(fixture()));
    const pending = loadPaperSnapshot({ apiUrl: "/api/observations", fetcher });
    await vi.advanceTimersByTimeAsync(8000);
    expect((await pending).source).toBe("published_snapshot");
  });
  it("calculates age from the observation, independent of export or refresh time", () => {
    expect(
      observationAge(
        "2026-09-20T12:00:00Z",
        Date.parse("2026-09-25T12:00:00Z"),
      ),
    ).toBe("5 days ago");
    expect(
      observationAge(
        "2026-09-25T11:00:00Z",
        Date.parse("2026-09-25T12:00:00Z"),
      ),
    ).toBe("1 hour ago");
  });
});
