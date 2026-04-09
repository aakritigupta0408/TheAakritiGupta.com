import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { z } from "zod";
import {
  agentTabSignals,
  aiUseCasesNow,
  buildNowProjectTracks,
  latestAIProductLaunches,
  latestAIResearchBreakthroughs,
  promptTabSignals,
  startupWatchlist,
} from "../../client/data/aiSignals";
import {
  pageRefreshContentByRoute,
  siteRefreshMeta,
} from "../../client/data/siteRefreshContent";
import { pageRefreshSpecs } from "./pageSpecs";

const SIGNALS_OUTPUT_PATH = resolve("client/data/aiSignals.ts");
const PAGE_SHELL_OUTPUT_PATH = resolve("client/data/siteRefreshContent.ts");

const ISO_DATE_LABEL = new Intl.DateTimeFormat("en-US", {
  month: "long",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

const aiSignalSchema = z.object({
  id: z.string().trim().min(1),
  title: z.string().trim().min(1),
  org: z.string().trim().min(1),
  date: z.string().trim().min(1),
  category: z.string().trim().min(1),
  summary: z.string().trim().min(1),
  impact: z.string().trim().min(1),
  url: z.string().trim().url(),
});

const aiUseCaseSignalSchema = z.object({
  id: z.string().trim().min(1),
  title: z.string().trim().min(1),
  summary: z.string().trim().min(1),
  signal: z.string().trim().min(1),
  examples: z.array(z.string().trim().min(1)).min(3).max(4),
});

const startupWatchSchema = z.object({
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
  focus: z.string().trim().min(1),
  latestMove: z.string().trim().min(1),
  date: z.string().trim().min(1),
  whyItMatters: z.string().trim().min(1),
  url: z.string().trim().url(),
});

const buildTrackSchema = z.object({
  id: z.string().trim().min(1),
  title: z.string().trim().min(1),
  category: z.enum([
    "Computer Vision",
    "Natural Language Processing",
    "Machine Learning",
  ]),
  difficulty: z.enum(["Beginner", "Intermediate", "Advanced"]),
  summary: z.string().trim().min(1),
  outcome: z.string().trim().min(1),
  stack: z.array(z.string().trim().min(1)).min(3).max(5),
  url: z.string().trim().url(),
});

const signalSetSchema = z.object({
  latestAIResearchBreakthroughs: z.array(aiSignalSchema).length(5),
  latestAIProductLaunches: z.array(aiSignalSchema).length(6),
  aiUseCasesNow: z.array(aiUseCaseSignalSchema).length(6),
  startupWatchlist: z.array(startupWatchSchema).length(5),
  buildNowProjectTracks: z.array(buildTrackSchema).length(6),
  agentTabSignals: z.object({
    examples: z.array(aiSignalSchema).length(3),
    techniques: z.array(aiSignalSchema).length(3),
    playground: z.array(aiSignalSchema).length(3),
  }),
  promptTabSignals: z.object({
    examples: z.array(aiSignalSchema).length(3),
    techniques: z.array(aiSignalSchema).length(3),
    playground: z.array(aiSignalSchema).length(3),
  }),
});

const refreshableRoutes = pageRefreshSpecs.map((spec) => spec.route);

const pageShellSchema = z.object({
  route: z
    .string()
    .refine((value) => refreshableRoutes.includes(value as (typeof refreshableRoutes)[number]), {
      message: "Unknown refreshable route",
    }),
  eyebrow: z.string().trim().min(1).max(60),
  title: z.string().trim().min(1).max(140),
  description: z.string().trim().min(1).max(320),
  chips: z.array(z.string().trim().min(1).max(80)).length(3),
  refreshSummary: z.string().trim().min(1).max(220),
});

const pageShellOutputSchema = z.object({
  siteRefreshMeta: z.object({
    headline: z.string().trim().min(1).max(80),
    description: z.string().trim().min(1).max(280),
  }),
  pages: z.array(pageShellSchema).length(pageRefreshSpecs.length),
});

interface ResponsesApiTextContent {
  text?: string;
  type?: string;
}

interface ResponsesApiOutputItem {
  content?: ResponsesApiTextContent[];
}

interface ResponsesApiResponse {
  output_text?: string;
  output?: ResponsesApiOutputItem[];
}

function requireEnv(name: string) {
  const value = process.env[name];

  if (!value) {
    throw new Error(`${name} is required for the site refresh agent.`);
  }

  return value;
}

function formatJsonValue(value: unknown) {
  return JSON.stringify(value, null, 2);
}

function stripCodeFences(text: string) {
  const trimmed = text.trim();

  if (trimmed.startsWith("```")) {
    return trimmed
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();
  }

  return trimmed;
}

function extractJsonText(text: string) {
  const cleaned = stripCodeFences(text);
  const firstBrace = cleaned.indexOf("{");
  const lastBrace = cleaned.lastIndexOf("}");

  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    return cleaned;
  }

  return cleaned.slice(firstBrace, lastBrace + 1);
}

function getOutputText(response: ResponsesApiResponse) {
  if (response.output_text?.trim()) {
    return response.output_text.trim();
  }

  const fallback = response.output
    ?.flatMap((item) => item.content ?? [])
    .map((content) => content.text ?? "")
    .join("")
    .trim();

  if (!fallback) {
    throw new Error("OpenAI response did not include output text.");
  }

  return fallback;
}

async function generateJsonWithWebSearch<T>(
  prompt: string,
  schema: z.ZodSchema<T>,
) {
  const apiKey = requireEnv("OPENAI_API_KEY");
  const model = process.env.OPENAI_SITE_REFRESH_MODEL || "gpt-5.4-mini";
  const reasoningEffort =
    process.env.OPENAI_SITE_REFRESH_REASONING_EFFORT || "low";

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      reasoning: { effort: reasoningEffort },
      tools: [{ type: "web_search_preview" }],
      input: prompt,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `OpenAI site refresh request failed (${response.status}): ${errorText}`,
    );
  }

  const payload = (await response.json()) as ResponsesApiResponse;
  const outputText = extractJsonText(getOutputText(payload));
  const parsed = JSON.parse(outputText);

  return schema.parse(parsed);
}

function serializeAiSignalsModule(data: z.infer<typeof signalSetSchema>) {
  return `/* This file is generated by scripts/site-refresh/refresh-agent.ts. */
export interface AISignal {
  id: string;
  title: string;
  org: string;
  date: string;
  category: string;
  summary: string;
  impact: string;
  url: string;
}

export interface AIUseCaseSignal {
  id: string;
  title: string;
  summary: string;
  signal: string;
  examples: string[];
}

export interface StartupWatchItem {
  id: string;
  name: string;
  focus: string;
  latestMove: string;
  date: string;
  whyItMatters: string;
  url: string;
}

export interface BuildTrack {
  id: string;
  title: string;
  category: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  summary: string;
  outcome: string;
  stack: string[];
  url: string;
}

export const latestAIResearchBreakthroughs: AISignal[] = ${formatJsonValue(data.latestAIResearchBreakthroughs)};

export const latestAIProductLaunches: AISignal[] = ${formatJsonValue(data.latestAIProductLaunches)};

export const aiUseCasesNow: AIUseCaseSignal[] = ${formatJsonValue(data.aiUseCasesNow)};

export const startupWatchlist: StartupWatchItem[] = ${formatJsonValue(data.startupWatchlist)};

export const buildNowProjectTracks: BuildTrack[] = ${formatJsonValue(data.buildNowProjectTracks)};

export const agentTabSignals: Record<
  "examples" | "techniques" | "playground",
  AISignal[]
> = ${formatJsonValue(data.agentTabSignals)};

export const promptTabSignals: Record<
  "examples" | "techniques" | "playground",
  AISignal[]
> = ${formatJsonValue(data.promptTabSignals)};
`;
}

function serializePageShellModule(
  data: z.infer<typeof pageShellOutputSchema>,
  updatedAtLabel: string,
) {
  const pagesByRoute = Object.fromEntries(
    data.pages.map((page) => [
      page.route,
      {
        ...page,
        updatedAtLabel,
      },
    ]),
  );

  return `export type RefreshableRoute =
  | "/ai-playground"
  | "/ai-discoveries"
  | "/ai-tools"
  | "/ai-companies"
  | "/ai-projects"
  | "/prompt-engineering"
  | "/ai-agent-training"
  | "/ai-champions"
  | "/resume-builder"
  | "/games";

export interface PageRefreshContent {
  route: RefreshableRoute;
  eyebrow: string;
  title: string;
  description: string;
  chips: string[];
  refreshSummary: string;
  updatedAtLabel: string;
}

export interface SiteRefreshMeta {
  headline: string;
  description: string;
  updatedAtLabel: string;
}

export const siteRefreshMeta: SiteRefreshMeta = ${formatJsonValue({
    ...data.siteRefreshMeta,
    updatedAtLabel,
  })};

export const pageRefreshContentByRoute: Record<
  RefreshableRoute,
  PageRefreshContent
> = ${formatJsonValue(pagesByRoute)};

export function getPageRefreshContent(route: RefreshableRoute) {
  return pageRefreshContentByRoute[route];
}
`;
}

function buildSignalsPrompt() {
  return `You are a weekly website-refresh agent for a public AI portfolio. Use web search to refresh the latest-information datasets that power the site.

Rules:
- Return JSON only. No markdown, no code fences, no commentary.
- Keep the exact top-level keys requested.
- Use absolute dates like "April 9, 2026" when possible. If a source only supports a broader time range, use a precise range string such as "2025-2026 rollout".
- Prefer official or primary sources for URLs.
- Keep the tone concise, factual, and useful for a premium portfolio site.
- Update only current-information datasets. Do not turn evergreen sections into news feeds.
- Preserve the current array lengths exactly.

Current dataset snapshot:
${JSON.stringify(
  {
    latestAIResearchBreakthroughs,
    latestAIProductLaunches,
    aiUseCasesNow,
    startupWatchlist,
    buildNowProjectTracks,
    agentTabSignals,
    promptTabSignals,
  },
  null,
  2,
)}

Return a JSON object with this exact shape:
{
  "latestAIResearchBreakthroughs": AISignal[5],
  "latestAIProductLaunches": AISignal[6],
  "aiUseCasesNow": AIUseCaseSignal[6],
  "startupWatchlist": StartupWatchItem[5],
  "buildNowProjectTracks": BuildTrack[6],
  "agentTabSignals": {
    "examples": AISignal[3],
    "techniques": AISignal[3],
    "playground": AISignal[3]
  },
  "promptTabSignals": {
    "examples": AISignal[3],
    "techniques": AISignal[3],
    "playground": AISignal[3]
  }
}

Field reminders:
- AISignal: { id, title, org, date, category, summary, impact, url }
- AIUseCaseSignal: { id, title, summary, signal, examples }
- StartupWatchItem: { id, name, focus, latestMove, date, whyItMatters, url }
- BuildTrack: { id, title, category, difficulty, summary, outcome, stack, url }
`;
}

function buildPageShellPrompt(
  refreshedSignals: z.infer<typeof signalSetSchema>,
  updatedAtLabel: string,
) {
  return `You are a principal content-and-UX editor updating the shell copy for a portfolio's main AI pages.

Rules:
- Return JSON only. No markdown, no code fences, no commentary.
- Keep the route values exactly as provided.
- Keep each page shell premium, concise, and current.
- Do not rewrite the entire page; describe what gets refreshed at the top of the page and what the weekly agent focuses on.
- Each page must have exactly 3 chips.
- Titles should be strong, specific, and readable on a hero section.
- refreshSummary should say what gets updated weekly and what remains stable.
- Set the site-wide updatedAtLabel to "${updatedAtLabel}".

Current page-shell snapshot:
${JSON.stringify(pageRefreshContentByRoute, null, 2)}

Current site-refresh meta:
${JSON.stringify(siteRefreshMeta, null, 2)}

Page briefs describing what to update, where to update it, and what to preserve:
${JSON.stringify(pageRefreshSpecs, null, 2)}

Latest refreshed signal snapshot:
${JSON.stringify(
  {
    latestAIResearchBreakthroughs: refreshedSignals.latestAIResearchBreakthroughs,
    latestAIProductLaunches: refreshedSignals.latestAIProductLaunches,
    aiUseCasesNow: refreshedSignals.aiUseCasesNow,
    startupWatchlist: refreshedSignals.startupWatchlist,
    buildNowProjectTracks: refreshedSignals.buildNowProjectTracks,
  },
  null,
  2,
)}

Return a JSON object with this exact shape:
{
  "siteRefreshMeta": {
    "headline": string,
    "description": string
  },
  "pages": [
    {
      "route": string,
      "eyebrow": string,
      "title": string,
      "description": string,
      "chips": [string, string, string],
      "refreshSummary": string
    }
  ]
}`;
}

async function main() {
  const updatedAtLabel = ISO_DATE_LABEL.format(new Date());

  console.log("Refreshing latest AI signal datasets...");
  const refreshedSignals = await generateJsonWithWebSearch(
    buildSignalsPrompt(),
    signalSetSchema,
  );

  console.log("Refreshing page-shell copy...");
  const refreshedPageShells = await generateJsonWithWebSearch(
    buildPageShellPrompt(refreshedSignals, updatedAtLabel),
    pageShellOutputSchema,
  );

  await writeFile(
    SIGNALS_OUTPUT_PATH,
    serializeAiSignalsModule(refreshedSignals),
    "utf8",
  );

  await writeFile(
    PAGE_SHELL_OUTPUT_PATH,
    serializePageShellModule(refreshedPageShells, updatedAtLabel),
    "utf8",
  );

  console.log("Site refresh data files updated.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
