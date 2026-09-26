// @vitest-environment happy-dom
import React from "react";
import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import TradeRecommendationSystemDemo from "./TradeRecommendationSystemDemo";

vi.mock("@/components/Navigation", () => ({
  default: () => <nav>Site navigation</nav>,
}));
const data = JSON.parse(
  readFileSync(
    "public/data/trade-system-snapshot.json",
    "utf-8",
  ),
);
const response = (body: unknown, status = 200) =>
  ({ ok: status === 200, status, json: async () => body }) as Response;
const renderPage = () =>
  render(
    <MemoryRouter>
      <TradeRecommendationSystemDemo />
    </MemoryRouter>,
  );
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("paper observation page", () => {
  it("shows a useful error when no snapshot is available", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({}, 404)),
    );
    const view = renderPage();
    expect((await view.findByRole("alert")).textContent).toContain(
      "could not be loaded",
    );
    expect(view.queryByText("Realized paper P&L")).toBeNull();
    expect(view.queryByText(/live trading system/i)).toBeNull();
  });
  it("retains the dated record with an error after a failed refresh, with no trading controls", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response(data))
      .mockResolvedValueOnce(response({}, 404));
    vi.stubGlobal("fetch", fetcher);
    const view = renderPage();
    await view.findByRole("heading", {
      name: /recorded candidate recommendations/i,
    });
    const observed = view.getByRole("heading", {
      name: /^observed /i,
    }).textContent;
    fireEvent.click(view.getByRole("button", { name: /refresh snapshot/i }));
    await waitFor(() =>
      expect(view.getByRole("alert").textContent).toContain(
        "previous observation remains",
      ),
    );
    expect(view.getByRole("heading", { name: /^observed /i }).textContent).toBe(
      observed,
    );
    expect(view.getByText(/not a calibrated probability/i)).not.toBeNull();
    expect(
      view.queryByRole("button", { name: /scan|buy|sell|execute/i }),
    ).toBeNull();
    expect(fetcher.mock.calls.every(([, init]) => init.method === "GET")).toBe(
      true,
    );
  });
});
