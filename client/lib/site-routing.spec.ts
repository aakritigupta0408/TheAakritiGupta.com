import { describe, expect, it, vi } from "vitest";
import {
  buildHashRoute,
  buildStaticSiteUrl,
  normalizeAppPath,
} from "./site-routing";

describe("site routing helpers", () => {
  it("normalizes app paths with a leading slash", () => {
    expect(normalizeAppPath("resume-builder")).toBe("/resume-builder");
    expect(normalizeAppPath("/ai-playground")).toBe("/ai-playground");
  });

  it("builds hash routes for static navigation", () => {
    expect(buildHashRoute("/ai-tools")).toBe("#/ai-tools");
  });

  it("builds absolute hash URLs for recruiter share links", () => {
    vi.stubGlobal("window", {
      location: {
        origin: "https://www.theaakritigupta.com",
      },
    });

    expect(
      buildStaticSiteUrl("/resume-builder/recruiter/demo", {
        agent: "token",
      }),
    ).toBe(
      "https://www.theaakritigupta.com/#/resume-builder/recruiter/demo?agent=token",
    );
  });
});
