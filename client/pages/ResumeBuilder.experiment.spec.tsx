// @vitest-environment happy-dom

import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import ResumeBuilder from "./ResumeBuilder";

vi.mock("../components/Navigation", () => ({
  default: () => <div data-testid="navigation" />,
}));

vi.mock("@/components/Navigation", () => ({
  default: () => <div data-testid="navigation" />,
}));

vi.mock("framer-motion", async () => {
  const { createFramerMotionMock } = await import("@/test/testUtils");
  return createFramerMotionMock();
});

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe("ResumeBuilder experiment variants", () => {
  it("renders the compact variant when forced", () => {
    render(
      <MemoryRouter initialEntries={["/resume-builder?exp-resume-builder-layout=compact"]}>
        <ResumeBuilder />
      </MemoryRouter>,
    );

    expect(screen.getByText("What this page does")).toBeTruthy();
    expect(
      screen.getByText(/Create one grounded recruiter link/i),
    ).toBeTruthy();
  });

  it("renders the guided variant when forced", () => {
    render(
      <MemoryRouter initialEntries={["/resume-builder?exp-resume-builder-layout=guided"]}>
        <ResumeBuilder />
      </MemoryRouter>,
    );

    expect(screen.getByText("Static-safe recruiter handoff")).toBeTruthy();
    expect(
      screen.getByText(/Turn approved candidate evidence into one recruiter-safe link/i),
    ).toBeTruthy();
    expect(screen.getByText("Variant override active: guided")).toBeTruthy();
  });
});
