// @vitest-environment happy-dom

import React from "react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { installMatchMediaMock } from "@/test/testUtils";
import Navigation from "./Navigation";

vi.mock("framer-motion", async () => {
  const { createFramerMotionMock } = await import("@/test/testUtils");
  return createFramerMotionMock();
});

beforeAll(() => {
  installMatchMediaMock();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

function LocationDisplay() {
  const location = useLocation();

  return <div data-testid="location">{location.pathname}</div>;
}

const renderNavigation = (initialEntries = ["/"]) =>
  render(
    <MemoryRouter initialEntries={initialEntries}>
      <Routes>
        <Route
          path="*"
          element={
            <>
              <Navigation />
              <LocationDisplay />
            </>
          }
        />
      </Routes>
    </MemoryRouter>,
  );

describe("Navigation", () => {
  it("routes across the full primary navigation", async () => {
    const user = userEvent.setup();
    const view = renderNavigation(["/ai-tools"]);

    const navCases: Array<[RegExp, string]> = [
      [/home/i, "/"],
      [/interactive demos/i, "/ai-playground"],
      [/ai vs humans/i, "/ai-champions"],
      [/ai discoveries/i, "/ai-discoveries"],
      [/ai tools/i, "/ai-tools"],
      [/ai companies/i, "/ai-companies"],
      [/ai projects/i, "/ai-projects"],
      [/prompt mastery/i, "/prompt-engineering"],
      [/agent training/i, "/ai-agent-training"],
      [/resume builder/i, "/resume-builder"],
      [/games/i, "/games"],
    ];

    for (const [label, path] of navCases) {
      await user.click(view.getByRole("button", { name: label }));
      expect(view.getByTestId("location").textContent).toBe(path);
    }
  });

  it("opens the about dropdown and routes to a talent page", async () => {
    const user = userEvent.setup();
    const view = renderNavigation();

    await user.click(view.getByRole("button", { name: /know more about ag/i }));
    await user.click(view.getAllByRole("button", { name: /ai researcher/i }).at(-1)!);

    expect(view.getByTestId("location").textContent).toBe("/talent/ai-researcher");
  });

  it("opens free resources and exposes the resume link", async () => {
    const user = userEvent.setup();
    const view = renderNavigation();

    await user.click(view.getByRole("button", { name: /free resources/i }));

    const resumeLink = view.getByRole("link", { name: /resume/i });
    expect(resumeLink.getAttribute("href")).toContain("drive.google.com");
  });

  it("opens search, finds a matching page, and navigates from the result", async () => {
    const view = renderNavigation();

    fireEvent.click(
      view.getByRole("button", { name: /open website search/i }),
    );
    fireEvent.change(
      view.getByPlaceholderText(/search pages, companies, tools, profiles, and topics on this website/i),
      {
        target: {
          value: "anthropic",
        },
      },
    );

    const companyResult = (await view.findAllByRole("button", { name: /anthropic/i })).find(
      (button) => button.textContent?.includes("AI Companies"),
    );

    expect(companyResult).toBeDefined();
    fireEvent.click(companyResult!);

    expect(view.getByTestId("location").textContent).toBe("/ai-companies");
  });
});
