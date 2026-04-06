// @vitest-environment happy-dom

import React from "react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import AITools from "./AITools";
import AICompanies from "./AICompanies";
import AIProjects from "./AIProjects";
import AIPlayground from "./AIPlayground";
import PromptEngineering from "./PromptEngineering";
import AIAgentTraining from "./AIAgentTraining";
import AIDiscoveries from "./AIDiscoveries";

vi.mock("../components/Navigation", () => ({
  default: () => <div data-testid="navigation" />,
}));

vi.mock("@/components/Navigation", () => ({
  default: () => <div data-testid="navigation" />,
}));

vi.mock("@/components/ChatBot", () => ({
  default: () => <div data-testid="chatbot" />,
}));

vi.mock("framer-motion", async () => {
  const React = await import("react");

  const stripMotionProps = (props: Record<string, unknown>) => {
    const {
      animate,
      exit,
      initial,
      layout,
      layoutId,
      transition,
      viewport,
      whileHover,
      whileInView,
      whileTap,
      ...rest
    } = props;

    void animate;
    void exit;
    void initial;
    void layout;
    void layoutId;
    void transition;
    void viewport;
    void whileHover;
    void whileInView;
    void whileTap;

    return rest;
  };

  const createMock =
    (tag: keyof React.JSX.IntrinsicElements) =>
    React.forwardRef<HTMLElement, React.HTMLAttributes<HTMLElement>>(
      ({ children, ...props }, ref) =>
        React.createElement(tag, { ref, ...stripMotionProps(props) }, children),
    );

  const motion = new Proxy(
    {},
    {
      get: (_, tag: string) =>
        createMock((tag as keyof React.JSX.IntrinsicElements) || "div"),
    },
  );

  return {
    AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    motion,
  };
});

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const renderPage = (ui: React.ReactElement) =>
  render(<MemoryRouter>{ui}</MemoryRouter>);

describe("AI page interactions", () => {
  it("filters AI tools and opens a profession detail view", async () => {
    const view = renderPage(<AITools />);

    fireEvent.click(view.getByRole("button", { name: /^Medium$/i }));

    expect(view.queryByText("Software Developers")).toBeNull();
    expect(view.getByText("Teachers & Educators")).not.toBeNull();

    fireEvent.click(view.getByText("Teachers & Educators"));

    expect(view.getAllByRole("heading", { name: "Teachers & Educators" }).length).toBeGreaterThan(0);
    expect(view.getAllByText("Khanmigo").length).toBeGreaterThan(0);
  });

  it("filters AI companies and opens the selected company modal", async () => {
    const view = renderPage(<AICompanies />);

    fireEvent.click(view.getByRole("button", { name: /scale snapshot/i }));
    fireEvent.click(view.getByRole("button", { name: /ai safety/i }));

    expect(view.getAllByText(/click to explore journey/i)).toHaveLength(1);

    fireEvent.click(view.getAllByText("Anthropic").at(-1)!);

    expect(view.getAllByRole("heading", { name: "Anthropic" }).length).toBeGreaterThan(0);
    expect(view.getAllByText("Claude Opus 4.5").length).toBeGreaterThan(0);
  });

  it("shows the newer company additions in the main grid and keeps category views populated", () => {
    const view = renderPage(<AICompanies />);

    expect(view.getAllByText("Cursor").length).toBeGreaterThan(0);
    expect(view.getAllByText("Mistral AI").length).toBeGreaterThan(0);
    expect(view.getAllByText(/new since aug 2025/i).length).toBeGreaterThan(0);

    const categoryLabels = [
      /ai research/i,
      /big tech ai/i,
      /ai infrastructure/i,
      /generative ai/i,
      /enterprise ai/i,
      /ai platform/i,
      /computer vision/i,
      /ai hardware/i,
      /autonomous systems/i,
      /process automation/i,
      /ai safety/i,
    ];

    for (const label of categoryLabels) {
      fireEvent.click(view.getByRole("button", { name: label }));
      expect(view.getAllByText(/click to explore journey/i).length).toBeGreaterThan(0);
    }

    fireEvent.click(view.getByRole("button", { name: /clear category filter/i }));
    expect(view.getAllByText("Runway").length).toBeGreaterThan(0);
  });

  it("keeps AI company cards visible after scrolling the page", () => {
    const view = renderPage(<AICompanies />);

    fireEvent.scroll(window, { target: { scrollY: 900 } });

    expect(view.getAllByText(/click to explore journey/i).length).toBeGreaterThan(10);
    expect(view.getAllByText("OpenAI").length).toBeGreaterThan(0);
    expect(view.getAllByText("Runway").length).toBeGreaterThan(0);
  });

  it("applies AI project filters and reveals code for the remaining project", async () => {
    const view = renderPage(<AIProjects />);

    fireEvent.click(view.getAllByRole("button", { name: /computer vision/i })[0]);
    fireEvent.click(view.getByRole("button", { name: /beginner/i }));

    expect(view.getByText("Image Classification")).not.toBeNull();
    expect(view.queryByText("Object Detection")).toBeNull();

    const showCodeButton = view.getAllByRole("button", { name: /show code/i })[0];
    fireEvent.click(showCodeButton);

    expect(view.getByText(/image_classification\.py/i)).not.toBeNull();

    fireEvent.click(view.getByRole("button", { name: /full guide/i }));

    expect(view.getAllByRole("heading", { name: "Image Classification" }).length).toBeGreaterThan(0);
  });

  it("clears AI project filters and restores the hidden projects", async () => {
    const view = renderPage(<AIProjects />);

    fireEvent.click(view.getAllByRole("button", { name: /computer vision/i })[0]);
    fireEvent.click(view.getByRole("button", { name: /beginner/i }));

    expect(view.queryByText("Object Detection")).toBeNull();

    fireEvent.click(view.getByRole("button", { name: /clear all filters/i }));

    expect(view.getAllByText("Object Detection").length).toBeGreaterThan(0);
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
    fireEvent.click(view.getByRole("button", { name: /generate with ai/i }));

    await vi.runAllTimersAsync();

    expect(view.getByText(/solution for:/i)).not.toBeNull();

    vi.useRealTimers();
    },
    20000,
  );

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

    const input = view.getByPlaceholderText(
      /help me write a business plan/i,
    );
    fireEvent.change(input, {
      target: {
        value: "Review this codebase and summarize the major risks.",
      },
    });
    fireEvent.click(
      view.getByRole("button", { name: /analyze & improve prompt/i }),
    );

    await vi.runAllTimersAsync();

    expect(view.getByText(/improved version/i)).not.toBeNull();
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

    const input = view.getByPlaceholderText(
      /i want to build an ai agent/i,
    );
    fireEvent.change(input, {
      target: {
        value: "An agent that reviews pull requests and summarizes risks.",
      },
    });
    fireEvent.click(
      view.getByRole("button", { name: /generate training strategy/i }),
    );

    await vi.runAllTimersAsync();

    expect(view.getByText(/ai agent training analysis/i)).not.toBeNull();

    vi.useRealTimers();
    },
    15000,
  );

  it(
    "filters discoveries by decade and shows the modern filter controls",
    async () => {
    const view = renderPage(<AIDiscoveries />);

    fireEvent.click(view.getByRole("button", { name: /2020s/i }));
    expect(view.getByRole("button", { name: /clear decade filter/i })).not.toBeNull();
    expect(view.getByRole("button", { name: /alphabetical/i })).not.toBeNull();
    },
    15000,
  );

  it("clears the discoveries decade filter and restores earlier discoveries", async () => {
    const view = renderPage(<AIDiscoveries />);

    fireEvent.click(view.getByRole("button", { name: /2020s/i }));
    expect(view.queryByText("The Perceptron")).toBeNull();

    fireEvent.click(view.getByRole("button", { name: /clear decade filter/i }));

    expect(view.getByText("The Perceptron")).not.toBeNull();
  });
});
