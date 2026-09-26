// @vitest-environment happy-dom

import React from "react";
import { readFileSync } from "node:fs";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import AITools from "./AITools";
import AICompanies from "./AICompanies";
import AIProjects from "./AIProjects";
import AIChampions from "./AIChampions";
import AIPlayground from "./AIPlayground";
import TradeRecommendationSystemDemo from "./TradeRecommendationSystemDemo";
import PromptEngineering from "./PromptEngineering";
import AIAgentTraining from "./AIAgentTraining";
import AIDiscoveries from "./AIDiscoveries";
import ResumeBuilder from "./ResumeBuilder";
import { installMatchMediaMock } from "@/test/testUtils";
import { professions } from "@/data/toolArchive";
import { companies, companyCategories } from "@/data/companyArchive";
import { projects, projectCategories } from "@/data/projectArchive";
import { victories } from "@/data/victoryArchive";
import { discoveries } from "@/data/discoveryArchive";
import { createResumeAgentProfileFromInput } from "@shared/resume-agent";

const { buildResumeAgentMock } = vi.hoisted(() => ({
  buildResumeAgentMock: vi.fn(),
}));

vi.mock("../components/Navigation", () => ({
  default: () => <div data-testid="navigation" />,
}));

vi.mock("@/components/Navigation", () => ({
  default: () => <div data-testid="navigation" />,
}));

vi.mock("@/components/ChatBot", () => ({
  default: () => <div data-testid="chatbot" />,
}));

vi.mock("@/components/games/DeepBlueChess", () => ({
  default: () => <div data-testid="deep-blue-chess-demo" />,
}));

vi.mock("@/components/games/AlphaGoDemo", () => ({
  default: () => <div data-testid="alphago-go-demo" />,
}));

vi.mock("@/components/games/LibratusPoker", () => ({
  default: () => <div data-testid="libratus-poker-demo" />,
}));

vi.mock("@/components/resume-agent/RecruiterAgentChat", () => ({
  default: ({ profile }: { profile: { candidateName: string } }) => (
    <div data-testid="recruiter-agent-chat">{profile.candidateName}</div>
  ),
}));

vi.mock("@/api/resume-agent", () => ({
  buildResumeAgent: buildResumeAgentMock,
}));

vi.mock("framer-motion", async () => {
  const { createFramerMotionMock } = await import("@/test/testUtils");
  return createFramerMotionMock();
});

const paperSnapshot = JSON.parse(
  readFileSync(
    "public/data/trade-system-snapshot.json",
    "utf-8",
  ),
);

function jsonResponse(
  body: unknown,
  init?: { status?: number; statusText?: string },
) {
  const status = init?.status ?? 200;

  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: init?.statusText ?? "OK",
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

beforeAll(() => {
  installMatchMediaMock();

  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const rawUrl =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;
      const url = new URL(rawUrl, "http://localhost");
      const path = `${url.pathname}${url.search}`;

      if (path === "/api/chat") {
        return jsonResponse({ response: "Mock chat response" });
      }

      if (path === "/data/trade-system-snapshot.json") {
        return jsonResponse(paperSnapshot);
      }

      return jsonResponse(
        { error: `Unhandled fetch in tests: ${path}` },
        { status: 404, statusText: "Not Found" },
      );
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.clearAllMocks();
  buildResumeAgentMock.mockReset();
});

const renderPage = (ui: React.ReactElement) =>
  render(<MemoryRouter>{ui}</MemoryRouter>);

describe("AI page interactions", () => {
  it("filters AI champions and opens the playable demo for a selected match", () => {
    const targetVictory = victories.find(
      (victory) =>
        victory.id === "deep-blue-chess" &&
        victory.recordType === "Champion match" &&
        victory.playableDemo,
    )!;
    const hiddenVictory = victories.find(
      (victory) => victory.recordType === "Benchmark leap",
    )!;
    const view = renderPage(<AIChampions />);

    fireEvent.click(
      view.getAllByRole("button", { name: /Champion match/i })[0],
    );

    expect(view.queryByText(hiddenVictory.aiName)).toBeNull();

    // Champion matches (7 total) exceed the initial visibleCount of 6; load more to expose Deep Blue (1997)
    fireEvent.click(view.getByText("Load 3 more"));

    expect(view.getByText(targetVictory.aiName)).not.toBeNull();

    fireEvent.click(
      view.getByRole("button", {
        name: new RegExp(
          `${targetVictory.aiName}.*${targetVictory.opponent}`,
          "i",
        ),
      }),
    );

    expect(
      view.getAllByRole("heading", { name: targetVictory.aiName }).length,
    ).toBeGreaterThan(0);
    expect(view.getAllByText(targetVictory.scoreLabel).length).toBeGreaterThan(
      0,
    );

    fireEvent.click(view.getByRole("button", { name: /play demo/i }));

    expect(view.getByTestId("deep-blue-chess-demo")).not.toBeNull();
  });

  it("filters AI tools and opens a profession detail view", () => {
    const mediumProfession = [...professions]
      .filter((profession) => profession.impactLevel === "Medium")
      .sort((left, right) => right.aiAdoption - left.aiAdoption)[0]!;
    const hiddenCriticalProfession = professions.find(
      (profession) => profession.impactLevel === "Critical",
    )!;
    const view = renderPage(<AITools />);

    fireEvent.click(view.getByRole("button", { name: /^Medium$/i }));

    expect(view.getByText(mediumProfession.title)).not.toBeNull();
    expect(view.queryByText(hiddenCriticalProfession.title)).toBeNull();

    fireEvent.click(
      view.getByRole("button", {
        name: new RegExp(mediumProfession.title, "i"),
      }),
    );

    expect(
      view.getAllByRole("heading", { name: mediumProfession.title }).length,
    ).toBeGreaterThan(0);
    expect(
      view.getAllByText(mediumProfession.primaryTool.name).length,
    ).toBeGreaterThan(0);
  });

  it("filters AI companies and opens the selected company modal", () => {
    const targetCategory = companyCategories.find(
      (category) => category !== "All",
    )!;
    const targetCompany = [...companies]
      .filter((company) => company.category === targetCategory)
      .sort((left, right) => right.sortScale - left.sortScale)[0]!;
    const hiddenCompany = companies.find(
      (company) => company.category !== targetCategory,
    )!;
    const view = renderPage(<AICompanies />);

    fireEvent.click(
      view.getByRole("button", { name: new RegExp(targetCategory, "i") }),
    );

    expect(view.getByText(targetCompany.name)).not.toBeNull();
    expect(view.queryByText(hiddenCompany.name)).toBeNull();

    fireEvent.click(
      view.getByRole("button", { name: new RegExp(targetCompany.name, "i") }),
    );

    expect(
      view.getAllByRole("heading", { name: targetCompany.name }).length,
    ).toBeGreaterThan(0);
    expect(view.getAllByText(targetCompany.scaleSignal).length).toBeGreaterThan(
      0,
    );
  });

  it("applies AI project filters and reveals the selected project details", () => {
    const targetCategory = projectCategories.find(
      (category) => category !== "All",
    )!;
    const targetProject = projects.find(
      (project) =>
        project.category === targetCategory &&
        project.difficulty === "Beginner",
    )!;
    const hiddenProject =
      projects.find(
        (project) =>
          project.category === targetCategory &&
          project.title !== targetProject.title &&
          project.difficulty !== targetProject.difficulty,
      ) || projects.find((project) => project.category !== targetCategory)!;
    const view = renderPage(<AIProjects />);

    fireEvent.click(
      view.getAllByRole("button", { name: new RegExp(targetCategory, "i") })[0],
    );
    fireEvent.click(view.getByRole("button", { name: /^Beginner$/i }));

    expect(view.getByText(targetProject.title)).not.toBeNull();
    expect(view.queryByText(hiddenProject.title)).toBeNull();

    fireEvent.click(
      view.getByRole("button", { name: new RegExp(targetProject.title, "i") }),
    );

    expect(
      view.getAllByRole("heading", { name: targetProject.title }).length,
    ).toBeGreaterThan(0);
    expect(
      view.getAllByText(targetProject.recommendedStack[0]).length,
    ).toBeGreaterThan(0);
  });

  it("clears AI project filters and restores hidden projects", () => {
    const targetCategory = projectCategories.find(
      (category) => category !== "All",
    )!;
    const hiddenProject = projects.find(
      (project) => project.category !== targetCategory,
    )!;
    const view = renderPage(<AIProjects />);

    fireEvent.click(
      view.getAllByRole("button", { name: new RegExp(targetCategory, "i") })[0],
    );
    fireEvent.click(view.getByRole("button", { name: /^Beginner$/i }));

    expect(view.queryByText(hiddenProject.title)).toBeNull();

    fireEvent.click(view.getAllByRole("button", { name: /^All$/i })[0]);
    fireEvent.click(view.getAllByRole("button", { name: /^All$/i })[1]);

    expect(view.getAllByText(hiddenProject.title).length).toBeGreaterThan(0);
  });

  it("runs an AI playground generation flow from demo selection to output", async () => {
    vi.useFakeTimers();
    const view = renderPage(<AIPlayground />);

    fireEvent.click(view.getByRole("button", { name: /code generator/i }));
    fireEvent.click(
      view.getByRole("button", {
        name: /a function to sort an array by date/i,
      }),
    );
    fireEvent.click(
      view.getByRole("button", { name: /show sample response/i }),
    );

    await vi.runAllTimersAsync();

    expect(view.getByText(/solution for:/i)).not.toBeNull();

    vi.useRealTimers();
  }, 20000);

  it("loads the published paper journal without the unavailable legacy API", async () => {
    const view = renderPage(<TradeRecommendationSystemDemo />);
    expect(
      view.getByRole("heading", { name: /trade recommendation system/i }),
    ).not.toBeNull();
    await waitFor(() => {
      expect(
        view.getByRole("heading", {
          name: /recorded candidate recommendations/i,
        }),
      ).not.toBeNull();
    });
    expect(
      view.getByText(/quote freshness cannot be verified/i),
    ).not.toBeNull();
    expect(view.getByText(/paper entries disabled/i)).not.toBeNull();
    expect(view.queryByRole("button", { name: /trigger scan/i })).toBeNull();
    expect(
      view
        .getByRole("link", { name: /separate project: btc oracle/i })
        .getAttribute("href"),
    ).toBe("/btc-oracle/site/home.html");
    expect(
      view.getAllByText(paperSnapshot.recommendations[0].agent).length,
    ).toBeGreaterThan(0);
    const before = view.getByRole("heading", {
      name: /^observed /i,
    }).textContent;
    fireEvent.click(view.getByRole("button", { name: /refresh snapshot/i }));
    await waitFor(() =>
      expect(
        view.getByRole("button", { name: /refresh snapshot/i }),
      ).not.toBeNull(),
    );
    expect(view.getByRole("heading", { name: /^observed /i }).textContent).toBe(
      before,
    );
  });

  it("switches prompt engineering tabs and generates an improved prompt", async () => {
    vi.useFakeTimers();
    const view = renderPage(<PromptEngineering />);

    fireEvent.click(view.getByRole("button", { name: /techniques/i }));
    expect(
      view.getByText(/technique shifts driven by agentic ai/i),
    ).not.toBeNull();

    fireEvent.click(view.getByText("Chain of Thought"));
    expect(view.getAllByText("Chain of Thought").length).toBeGreaterThan(0);

    fireEvent.click(view.getByRole("button", { name: /playground/i }));

    fireEvent.change(
      view.getByPlaceholderText(/help me write a business plan/i),
      {
        target: {
          value: "Review this codebase and summarize the major risks.",
        },
      },
    );
    fireEvent.click(
      view.getByRole("button", { name: /analyze & improve prompt/i }),
    );

    await vi.runAllTimersAsync();

    expect(view.getByText(/improved prompt:/i)).not.toBeNull();
    vi.useRealTimers();
  }, 15000);

  it("switches agent-training tabs and generates a training strategy", async () => {
    vi.useFakeTimers();
    const view = renderPage(<AIAgentTraining />);

    expect(view.getByTestId("navigation")).not.toBeNull();
    expect(view.getByTestId("chatbot")).not.toBeNull();

    fireEvent.click(view.getByRole("button", { name: /agent builder/i }));
    fireEvent.change(
      view.getByPlaceholderText(/i want to build an ai agent/i),
      {
        target: {
          value: "An agent that reviews pull requests and summarizes risks.",
        },
      },
    );
    fireEvent.click(
      view.getByRole("button", { name: /generate training strategy/i }),
    );

    await vi.runAllTimersAsync();

    expect(view.getByText(/ai agent training analysis/i)).not.toBeNull();

    vi.useRealTimers();
  }, 15000);

  it("filters discoveries by decade and supports alphabetical sorting", () => {
    const modernDiscovery = [...discoveries]
      .filter((discovery) => discovery.year.startsWith("202"))
      .sort((left, right) => left.title.localeCompare(right.title))[0]!;
    const earlyDiscovery = discoveries.find(
      (discovery) => parseInt(discovery.year, 10) < 2000,
    )!;
    const view = renderPage(<AIDiscoveries />);

    fireEvent.click(view.getByRole("button", { name: /2020s/i }));
    fireEvent.change(view.getByRole("combobox"), {
      target: { value: "alphabetical" },
    });

    expect(view.getByText(modernDiscovery.title)).not.toBeNull();
    expect(view.queryByText(earlyDiscovery.title)).toBeNull();
  });

  it("clears the discoveries decade filter by switching back to all", () => {
    const earlyDiscovery = discoveries.find(
      (discovery) => parseInt(discovery.year, 10) < 2000,
    )!;
    const view = renderPage(<AIDiscoveries />);

    fireEvent.click(view.getByRole("button", { name: /2020s/i }));
    expect(view.queryByText(earlyDiscovery.title)).toBeNull();

    fireEvent.click(view.getByRole("button", { name: /^All$/i }));
    // Switch to oldest-first so the pre-2000 entry appears within the initial 8 results
    fireEvent.change(view.getByRole("combobox"), {
      target: { value: "oldest-first" },
    });

    expect(view.getByText(earlyDiscovery.title)).not.toBeNull();
  });

  it("builds a recruiter link from resume evidence and keeps LinkedIn import hidden", async () => {
    const profile = createResumeAgentProfileFromInput({
      candidateName: "Aakriti Gupta",
      resumeText:
        "Aakriti Gupta\nSenior AI engineer building grounded product workflows with React, TypeScript, and Python.",
      projectNotes:
        "Built a recruiter-safe resume agent that publishes a grounded share link and preserves factual constraints.",
    });
    buildResumeAgentMock.mockResolvedValue({
      profile,
      shareToken: "ra1.test-share-token",
      shareId: "share-123",
      usedModel: true,
    });

    const view = render(
      <MemoryRouter initialEntries={["/resume-builder"]}>
        <ResumeBuilder />
      </MemoryRouter>,
    );

    expect(view.queryByText(/LinkedIn import/i)).toBeNull();

    fireEvent.change(view.getByPlaceholderText(/optional/i), {
      target: { value: "Aakriti Gupta" },
    });
    fireEvent.change(
      view.getByPlaceholderText(
        /upload a resume file or paste resume text here/i,
      ),
      {
        target: {
          value:
            "Aakriti Gupta\nSenior AI engineer building grounded product workflows with React, TypeScript, and Python.",
        },
      },
    );
    fireEvent.change(view.getByPlaceholderText(/write in simple english/i), {
      target: {
        value:
          "Built a recruiter-safe resume agent that publishes a grounded share link and preserves factual constraints.",
      },
    });

    fireEvent.click(
      view.getByRole("button", { name: /build recruiter agent/i }),
    );

    await waitFor(() => {
      expect(buildResumeAgentMock).toHaveBeenCalledWith({
        candidateName: "Aakriti Gupta",
        resumeText:
          "Aakriti Gupta\nSenior AI engineer building grounded product workflows with React, TypeScript, and Python.",
        projectNotes:
          "Built a recruiter-safe resume agent that publishes a grounded share link and preserves factual constraints.",
      });
      expect(
        view.getByText(
          /Recruiter link is live and tied to the approved candidate facts/i,
        ),
      ).not.toBeNull();
    });

    expect(
      view.getByText(/resume-builder\/recruiter\/share-123/i),
    ).not.toBeNull();
    expect(
      view.getByText(/Persistent recruiter route created/i),
    ).not.toBeNull();
    expect(view.getByTestId("recruiter-agent-chat").textContent).toBe(
      "Aakriti Gupta",
    );
  });

  it("load-more buttons reveal additional items on every page that has them", () => {
    // AITools: 20 professions, initial 9, +6 per load
    const toolsView = renderPage(<AITools />);
    const toolsBefore = toolsView.getAllByText(/Open →/i).length;
    expect(toolsBefore).toBe(9);
    fireEvent.click(toolsView.getByText("Load more playbooks"));
    expect(toolsView.getAllByText(/Open →/i).length).toBe(15);
    cleanup();

    // AICompanies: 29 companies, initial 8
    const companiesView = renderPage(<AICompanies />);
    const companiesBefore =
      companiesView.container.querySelectorAll("h3").length;
    fireEvent.click(companiesView.getByText("Load 8 more"));
    const companiesAfter =
      companiesView.container.querySelectorAll("h3").length;
    expect(companiesAfter).toBeGreaterThan(companiesBefore);
    cleanup();

    // AIChampions: 12 victories, initial 6; keep clicking until all are revealed
    const championsView = renderPage(<AIChampions />);
    fireEvent.click(championsView.getByText("Load 3 more")); // 6 → 9
    fireEvent.click(championsView.getByText("Load 3 more")); // 9 → 12
    // All 12 victories are now visible; load-more button should be gone
    expect(championsView.queryByText("Load 3 more")).toBeNull();
  });
});
