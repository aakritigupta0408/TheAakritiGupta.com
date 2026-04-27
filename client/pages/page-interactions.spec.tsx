// @vitest-environment happy-dom

import React from "react";
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

const tradeRecommendations = [
  {
    ticker: "AAPL",
    action: "BUY",
    composite_score: 0.81,
    agent_score: 0.79,
    fta_score: 0.74,
    momentum_score: 0.72,
    iv_regime_score: 0.65,
    option_strategy_type: "bull_call_spread",
    scanned_at: "2026-04-10T09:30:00Z",
    error: null,
    is_actionable: true,
  },
  {
    ticker: "MSFT",
    action: "HOLD",
    composite_score: 0.48,
    agent_score: 0.5,
    fta_score: 0.46,
    momentum_score: 0.45,
    iv_regime_score: 0.41,
    option_strategy_type: "watchlist_only",
    scanned_at: "2026-04-10T09:30:00Z",
    error: null,
    is_actionable: false,
  },
];

const tradeUsers = [
  {
    user_id: "trader-alpha",
    name: "Alpha",
    avatar: "🧠",
    return_pct: 12.4,
    status: "active",
    open_positions: 1,
    total_trades: 8,
    win_rate: 0.75,
  },
  {
    user_id: "trader-beta",
    name: "Beta",
    avatar: "⚡",
    return_pct: -1.2,
    status: "paused",
    open_positions: 0,
    total_trades: 5,
    win_rate: 0.4,
  },
];

const tradeSummary = {
  scan_in_progress: false,
  last_scan_at: "2026-04-10T09:30:00Z",
  scan_error: null,
  total_users: 2,
  active_users: 1,
  total_open_positions: 1,
  top_recommendations: tradeRecommendations,
  users: tradeUsers,
};

const tradeUserDetail = {
  user_id: "trader-alpha",
  name: "Alpha",
  avatar: "🧠",
  description: "Momentum-focused trader",
  equity: 112400,
  starting_capital: 100000,
  return_pct: 12.4,
  open_positions: 1,
  total_trades: 8,
  win_rate: 0.75,
  status: "active",
  open_trades: [
    {
      trade_id: "trade-1",
      ticker: "AAPL",
      action: "BUY",
      instrument: "CALL",
      quantity: 2,
      entry_price: 6.25,
      stop_price: 4.8,
      target_price: 8.9,
      status: "open",
      pnl: 0,
      opened_at: "2026-04-10T09:35:00Z",
      closed_at: null,
      composite_score: 0.81,
      rationale: "Positive composite signal with broad agent agreement.",
    },
  ],
  closed_trades: [
    {
      trade_id: "trade-2",
      ticker: "NVDA",
      action: "BUY",
      instrument: "CALL",
      quantity: 1,
      entry_price: 5.1,
      stop_price: 4.3,
      target_price: 7.4,
      status: "closed_profit",
      pnl: 180,
      opened_at: "2026-04-09T14:35:00Z",
      closed_at: "2026-04-09T15:52:00Z",
      composite_score: 0.77,
      rationale: "Trend continuation with improving volatility setup.",
    },
  ],
  equity_curve: [
    { ts: "2026-04-10T09:30:00Z", equity: 100000 },
    { ts: "2026-04-10T10:30:00Z", equity: 105500 },
    { ts: "2026-04-10T11:30:00Z", equity: 112400 },
  ],
};

function jsonResponse(body: unknown, init?: { status?: number; statusText?: string }) {
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

      if (path === "/api/trade-system/summary") {
        return jsonResponse(tradeSummary);
      }

      if (path.startsWith("/api/trade-system/recommendations")) {
        return jsonResponse({ recommendations: tradeRecommendations });
      }

      if (path === "/api/trade-system/scan/trigger" && init?.method === "POST") {
        return jsonResponse({ queued: true });
      }

      if (path === "/api/trade-system/users/trader-alpha") {
        return jsonResponse(tradeUserDetail);
      }

      if (path === "/api/trade-system/users/trader-beta") {
        return jsonResponse({
          ...tradeUserDetail,
          user_id: "trader-beta",
          name: "Beta",
          avatar: "⚡",
          description: "Risk-controlled trader",
          return_pct: -1.2,
          status: "paused",
          open_trades: [],
        });
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

    fireEvent.click(view.getAllByRole("button", { name: /Champion match/i })[0]);

    expect(view.getByText(targetVictory.aiName)).not.toBeNull();
    expect(view.queryByText(hiddenVictory.aiName)).toBeNull();

    fireEvent.click(
      view.getByRole(
        "button",
        {
          name: new RegExp(
            `${targetVictory.aiName}.*${targetVictory.opponent}`,
            "i",
          ),
        },
      ),
    );

    expect(
      view.getAllByRole("heading", { name: targetVictory.aiName }).length,
    ).toBeGreaterThan(0);
    expect(view.getAllByText(targetVictory.scoreLabel).length).toBeGreaterThan(0);

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
      view.getByRole("button", { name: new RegExp(mediumProfession.title, "i") }),
    );

    expect(
      view.getAllByRole("heading", { name: mediumProfession.title }).length,
    ).toBeGreaterThan(0);
    expect(view.getAllByText(mediumProfession.primaryTool.name).length).toBeGreaterThan(0);
  });

  it("filters AI companies and opens the selected company modal", () => {
    const targetCategory = companyCategories.find((category) => category !== "All")!;
    const targetCompany = [...companies]
      .filter((company) => company.category === targetCategory)
      .sort((left, right) => right.sortScale - left.sortScale)[0]!;
    const hiddenCompany = companies.find(
      (company) => company.category !== targetCategory,
    )!;
    const view = renderPage(<AICompanies />);

    fireEvent.click(view.getByRole("button", { name: new RegExp(targetCategory, "i") }));

    expect(view.getByText(targetCompany.name)).not.toBeNull();
    expect(view.queryByText(hiddenCompany.name)).toBeNull();

    fireEvent.click(
      view.getByRole("button", { name: new RegExp(targetCompany.name, "i") }),
    );

    expect(
      view.getAllByRole("heading", { name: targetCompany.name }).length,
    ).toBeGreaterThan(0);
    expect(view.getAllByText(targetCompany.scaleSignal).length).toBeGreaterThan(0);
  });

  it("applies AI project filters and reveals the selected project details", () => {
    const targetCategory = projectCategories.find((category) => category !== "All")!;
    const targetProject = projects.find(
      (project) =>
        project.category === targetCategory && project.difficulty === "Beginner",
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
    expect(view.getAllByText(targetProject.recommendedStack[0]).length).toBeGreaterThan(0);
  });

  it("clears AI project filters and restores hidden projects", () => {
    const targetCategory = projectCategories.find((category) => category !== "All")!;
    const hiddenProject = projects.find((project) => project.category !== targetCategory)!;
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

  it(
    "runs an AI playground generation flow from demo selection to output",
    async () => {
      vi.useFakeTimers();
      const view = renderPage(<AIPlayground />);

      fireEvent.click(view.getByRole("button", { name: /code generator/i }));
      fireEvent.click(
        view.getByRole("button", { name: /a function to sort an array by date/i }),
      );
      fireEvent.click(view.getByRole("button", { name: /show sample response/i }));

      await vi.runAllTimersAsync();

      expect(view.getByText(/solution for:/i)).not.toBeNull();

      vi.useRealTimers();
    },
    20000,
  );

  it("loads the live trade system view and opens a trader detail panel", async () => {
    const view = renderPage(<TradeRecommendationSystemDemo />);

    expect(
      view.getByRole("heading", { name: /ai trade recommendation system/i }),
    ).not.toBeNull();

    await waitFor(() => {
      expect(view.getByText("Alpha")).not.toBeNull();
      expect(view.getByText("AAPL")).not.toBeNull();
    });

    expect(view.queryByText("MSFT")).toBeNull();
    fireEvent.click(view.getByRole("button", { name: /^All$/i }));
    expect(view.getByText("MSFT")).not.toBeNull();

    fireEvent.click(view.getByRole("button", { name: /alpha/i }));

    await waitFor(() => {
      expect(
        view.getAllByRole("heading", { name: /alpha/i }).length,
      ).toBeGreaterThan(0);
      expect(view.getByText(/Momentum-focused trader/i)).not.toBeNull();
    });
  });

  it(
    "switches prompt engineering tabs and generates an improved prompt",
    async () => {
      vi.useFakeTimers();
      const view = renderPage(<PromptEngineering />);

      fireEvent.click(view.getByRole("button", { name: /techniques/i }));
      expect(view.getByText(/technique shifts driven by agentic ai/i)).not.toBeNull();

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
    },
    15000,
  );

  it(
    "switches agent-training tabs and generates a training strategy",
    async () => {
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
    },
    15000,
  );

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

    fireEvent.change(
      view.getByPlaceholderText(/optional/i),
      { target: { value: "Aakriti Gupta" } },
    );
    fireEvent.change(
      view.getByPlaceholderText(/upload a resume file or paste resume text here/i),
      {
        target: {
          value:
            "Aakriti Gupta\nSenior AI engineer building grounded product workflows with React, TypeScript, and Python.",
        },
      },
    );
    fireEvent.change(
      view.getByPlaceholderText(/write in simple english/i),
      {
        target: {
          value:
            "Built a recruiter-safe resume agent that publishes a grounded share link and preserves factual constraints.",
        },
      },
    );

    fireEvent.click(view.getByRole("button", { name: /build recruiter agent/i }));

    await waitFor(() => {
      expect(buildResumeAgentMock).toHaveBeenCalledWith({
        candidateName: "Aakriti Gupta",
        resumeText:
          "Aakriti Gupta\nSenior AI engineer building grounded product workflows with React, TypeScript, and Python.",
        projectNotes:
          "Built a recruiter-safe resume agent that publishes a grounded share link and preserves factual constraints.",
      });
      expect(
        view.getByText(/Recruiter link is live and tied to the approved candidate facts/i),
      ).not.toBeNull();
    });

    expect(view.getByText(/resume-builder\/recruiter\/share-123/i)).not.toBeNull();
    expect(view.getByText(/Persistent recruiter route created/i)).not.toBeNull();
    expect(view.getByTestId("recruiter-agent-chat").textContent).toBe(
      "Aakriti Gupta",
    );
  });

  it("load-more buttons reveal additional items on every page that has them", () => {
    // AITools: 22 professions, initial 8
    const toolsView = renderPage(<AITools />);
    const toolsBefore = toolsView.getAllByText(/Open →/i).length;
    expect(toolsBefore).toBe(8);
    fireEvent.click(toolsView.getByText("Load 4 more"));
    expect(toolsView.getAllByText(/Open →/i).length).toBe(12);
    cleanup();

    // AICompanies: 29 companies, initial 8
    const companiesView = renderPage(<AICompanies />);
    const companiesBefore = companiesView.container.querySelectorAll("h3").length;
    fireEvent.click(companiesView.getByText("Load 8 more"));
    const companiesAfter = companiesView.container.querySelectorAll("h3").length;
    expect(companiesAfter).toBeGreaterThan(companiesBefore);
    cleanup();

    // AIChampions: 8 victories, initial 6
    const championsView = renderPage(<AIChampions />);
    fireEvent.click(championsView.getByText("Load 3 more"));
    // After loading, should show all 8
    expect(championsView.queryByText("Load 3 more")).toBeNull();
  });
});
