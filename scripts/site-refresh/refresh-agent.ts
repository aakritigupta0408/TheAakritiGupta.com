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

interface TextContentPart {
  text?: string;
  type?: string;
}

interface HuggingFaceChatCompletionMessage {
  content?: string | TextContentPart[];
}

interface HuggingFaceChatCompletionChoice {
  message?: HuggingFaceChatCompletionMessage;
}

interface HuggingFaceChatCompletionResponse {
  choices?: HuggingFaceChatCompletionChoice[];
}

interface ResearchSearchTask {
  id: string;
  label: string;
  query: string;
  maxResults: number;
  previewCount: number;
}

interface ResearchSource {
  title: string;
  url: string;
  snippet: string;
  publishedAt?: string;
  preview?: string;
}

interface ResearchSearchResult {
  id: string;
  label: string;
  query: string;
  results: ResearchSource[];
}

interface ResearchDossier {
  generatedAt: string;
  searches: ResearchSearchResult[];
}

const REQUEST_HEADERS = {
  "User-Agent":
    "TheAakritiGuptaSiteRefresh/1.0 (+https://www.theaakritigupta.com)",
  "Accept-Language": "en-US,en;q=0.9",
};

const SEARCH_TASKS: ResearchSearchTask[] = [
  {
    id: "frontier-research",
    label: "Frontier research breakthroughs",
    query:
      'AI research breakthrough 2026 arXiv OpenAI Anthropic DeepMind Meta multimodal reasoning',
    maxResults: 4,
    previewCount: 2,
  },
  {
    id: "product-launches",
    label: "Latest AI product launches",
    query:
      'AI product launch 2026 OpenAI Anthropic Google Mistral Perplexity enterprise release',
    maxResults: 4,
    previewCount: 2,
  },
  {
    id: "agent-patterns",
    label: "Agent workflows and coding systems",
    query:
      'AI agent coding workspace deep research MCP workflow 2026 official announcement',
    maxResults: 4,
    previewCount: 2,
  },
  {
    id: "enterprise-use-cases",
    label: "Enterprise AI use cases",
    query:
      'enterprise AI use case rollout 2026 copilots agents workflow automation official',
    maxResults: 4,
    previewCount: 2,
  },
  {
    id: "startup-watch",
    label: "AI startups to watch",
    query: 'AI startup funding launch 2026 model infrastructure agent',
    maxResults: 4,
    previewCount: 1,
  },
  {
    id: "build-now-projects",
    label: "Build-now project signals",
    query:
      'AI GitHub Hugging Face project framework repository launch 2026 agents multimodal',
    maxResults: 4,
    previewCount: 2,
  },
  {
    id: "prompt-patterns",
    label: "Prompt engineering and reasoning patterns",
    query:
      'prompt engineering agent workflow reasoning guide 2026 OpenAI Anthropic Google',
    maxResults: 4,
    previewCount: 2,
  },
];

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

function compactWhitespace(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function truncate(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, maxLength - 1).trimEnd()}…`;
}

function decodeHtmlEntities(value: string) {
  return value
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/gi, "$1")
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&#x([0-9a-f]+);/gi, (_, code) =>
      String.fromCharCode(Number.parseInt(code, 16)),
    )
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'");
}

function stripTags(value: string) {
  return compactWhitespace(decodeHtmlEntities(value).replace(/<[^>]+>/g, " "));
}

function extractTagValue(block: string, tagName: string) {
  const match = block.match(
    new RegExp(`<${tagName}[^>]*>([\\s\\S]*?)<\\/${tagName}>`, "i"),
  );

  return match?.[1] ?? "";
}

async function fetchText(
  url: string,
  headers: Record<string, string>,
  timeoutMs = 12000,
) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      headers,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Request failed (${response.status}) for ${url}`);
    }

    return await response.text();
  } finally {
    clearTimeout(timeoutId);
  }
}

function parseBingRss(xml: string) {
  const items = xml.match(/<item\b[\s\S]*?<\/item>/gi) ?? [];

  return items
    .map((item) => {
      const title = stripTags(extractTagValue(item, "title"));
      const url = compactWhitespace(stripTags(extractTagValue(item, "link")));
      const snippet = truncate(
        stripTags(extractTagValue(item, "description")),
        280,
      );
      const publishedAt = compactWhitespace(
        stripTags(extractTagValue(item, "pubDate")),
      );

      return {
        title,
        url,
        snippet,
        publishedAt: publishedAt || undefined,
      } satisfies ResearchSource;
    })
    .filter((item) => item.title && item.url);
}

function normalizePreview(html: string) {
  const title = stripTags(extractTagValue(html, "title"));
  const metaDescriptionMatch = html.match(
    /<meta[^>]+(?:name|property)=["'](?:description|og:description)["'][^>]+content=["']([^"']+)["'][^>]*>/i,
  );
  const description = metaDescriptionMatch
    ? stripTags(metaDescriptionMatch[1])
    : "";
  const bodyText = stripTags(
    html
      .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
      .replace(/<style\b[\s\S]*?<\/style>/gi, " "),
  );

  return truncate(
    compactWhitespace([title, description, bodyText].filter(Boolean).join(" | ")),
    700,
  );
}

const previewCache = new Map<string, Promise<string | undefined>>();

function getPagePreview(url: string) {
  if (!previewCache.has(url)) {
    previewCache.set(
      url,
      (async () => {
        try {
          const html = await fetchText(
            url,
            {
              ...REQUEST_HEADERS,
              Accept: "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
            10000,
          );

          return normalizePreview(html);
        } catch {
          return undefined;
        }
      })(),
    );
  }

  return previewCache.get(url)!;
}

async function runSearchTask(task: ResearchSearchTask) {
  try {
    const rss = await fetchText(
      `https://www.bing.com/search?format=rss&q=${encodeURIComponent(task.query)}`,
      {
        ...REQUEST_HEADERS,
        Accept: "application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
      },
    );

    const uniqueResults = new Map<string, ResearchSource>();

    for (const item of parseBingRss(rss)) {
      if (!uniqueResults.has(item.url)) {
        uniqueResults.set(item.url, item);
      }
    }

    const results = Array.from(uniqueResults.values()).slice(0, task.maxResults);

    await Promise.all(
      results.slice(0, task.previewCount).map(async (result) => {
        result.preview = await getPagePreview(result.url);
      }),
    );

    return {
      id: task.id,
      label: task.label,
      query: task.query,
      results,
    } satisfies ResearchSearchResult;
  } catch (error) {
    console.warn(`Search task failed for "${task.label}":`, error);

    return {
      id: task.id,
      label: task.label,
      query: task.query,
      results: [],
    } satisfies ResearchSearchResult;
  }
}

async function buildResearchDossier() {
  console.log("Collecting research context for the weekly refresh...");

  const searches = await Promise.all(SEARCH_TASKS.map(runSearchTask));
  const sourceCount = searches.reduce(
    (count, search) => count + search.results.length,
    0,
  );

  if (sourceCount === 0) {
    throw new Error(
      "The site refresh agent could not collect any research sources from the web.",
    );
  }

  return {
    generatedAt: new Date().toISOString(),
    searches,
  } satisfies ResearchDossier;
}

function getChatCompletionText(payload: HuggingFaceChatCompletionResponse) {
  const message = payload.choices?.[0]?.message?.content;

  if (typeof message === "string" && message.trim()) {
    return message.trim();
  }

  if (Array.isArray(message)) {
    const fallback = message
      .map((item) => item.text ?? "")
      .join("")
      .trim();

    if (fallback) {
      return fallback;
    }
  }

  throw new Error("Hugging Face response did not include message content.");
}

function extractFailedGeneration(errorBody: string): string | null {
  try {
    const parsed = JSON.parse(errorBody);
    const candidate = parsed?.error?.failed_generation;
    if (typeof candidate === "string" && candidate.trim()) {
      return extractJsonText(candidate);
    }
  } catch {
    // fall through to regex fallback
  }
  const match = errorBody.match(/"failed_generation"\s*:\s*"((?:\\.|[^"\\])*)"/);
  if (match) {
    try {
      return extractJsonText(JSON.parse(`"${match[1]}"`));
    } catch {
      return null;
    }
  }
  return null;
}

function stripStrayQuotesBetweenStructures(raw: string) {
  let output = "";
  let inString = false;
  let escape = false;
  for (let i = 0; i < raw.length; i += 1) {
    const char = raw[i];
    if (escape) {
      output += char;
      escape = false;
      continue;
    }
    if (char === "\\") {
      output += char;
      escape = true;
      continue;
    }
    if (char === '"') {
      if (!inString) {
        let j = i + 1;
        while (j < raw.length && (raw[j] === " " || raw[j] === "\n" || raw[j] === "\t")) {
          j += 1;
        }
        const next = raw[j];
        if (next === "{" || next === "[") {
          i = j - 1;
          continue;
        }
        const prev = output.replace(/\s+$/, "").slice(-1);
        if (prev === "}" || prev === "]") {
          continue;
        }
      }
      inString = !inString;
      output += char;
      continue;
    }
    output += char;
  }
  return output;
}

function tryRepairTruncatedJson(raw: string) {
  let text = raw.trim();
  if (!text.startsWith("{")) {
    return null;
  }
  text = text.replace(/,\s*([}\]])/g, "$1");

  const stack: string[] = [];
  let inString = false;
  let escape = false;
  let safeEnd = 0;
  let safeStack: string[] = [];
  let truncated = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (char === "\\") {
      escape = true;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (char === "{" || char === "[") {
      stack.push(char);
    } else if (char === "}" || char === "]") {
      const open = stack[stack.length - 1];
      const expected = char === "}" ? "{" : "[";
      if (open === expected) {
        stack.pop();
        if (stack.length === 0) {
          safeEnd = i + 1;
          safeStack = [];
        }
      } else {
        truncated = true;
        break;
      }
    }
    if (!inString && stack.length > 0) {
      safeEnd = i + 1;
      safeStack = [...stack];
    }
  }

  if (truncated || inString || stack.length > 0) {
    let closed = text.slice(0, safeEnd);
    const closers = [...safeStack];
    if (inString && !truncated) closed += '"';
    closed = closed.replace(/,\s*$/, "").replace(/:\s*$/, "");
    while (closers.length) {
      const open = closers.pop();
      closed += open === "{" ? "}" : "]";
    }
    return closed === raw.trim() ? null : closed;
  }

  return null;
}

async function generateJsonWithHuggingFace<T>(
  prompt: string,
  schema: z.ZodSchema<T>,
  responseFormatName: string,
) {
  const token = requireEnv("HF_TOKEN");
  const model = process.env.HF_SITE_REFRESH_MODEL || "openai/gpt-oss-20b";
  const reasoningEffort =
    process.env.HF_SITE_REFRESH_REASONING_EFFORT || "low";

  const maxAttempts = 4;
  let attemptPrompt = prompt;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const response = await fetch(
      "https://router.huggingface.co/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          reasoning_effort: reasoningEffort,
          temperature: 0.1,
          max_tokens: 8000,
          response_format: { type: "json_object" },
          messages: [
            {
              role: "system",
              content:
                "Return one valid JSON object. Do not use markdown, code fences, or explanatory text. Close every [ with ] and every { with }.",
            },
            {
              role: "user",
              content: attemptPrompt,
            },
          ],
        }),
      },
    );

    if (!response.ok) {
      const errorText = await response.text();
      lastError = new Error(
        `HF ${response.status}: ${errorText.slice(0, 400)}`,
      );

      const failedGen = extractFailedGeneration(errorText);
      if (failedGen) {
        const salvageCandidates: string[] = [failedGen];
        const salvageCleaned = stripStrayQuotesBetweenStructures(failedGen);
        if (salvageCleaned !== failedGen) salvageCandidates.push(salvageCleaned);
        const salvageRepaired = tryRepairTruncatedJson(salvageCleaned);
        if (salvageRepaired) salvageCandidates.push(salvageRepaired);
        for (const candidate of salvageCandidates) {
          try {
            const parsed = JSON.parse(candidate);
            const validated = schema.parse(parsed);
            console.warn(
              `Hugging Face attempt ${attempt} returned ${response.status} but failed_generation was repaired locally.`,
            );
            return validated;
          } catch (error) {
            lastError = error;
          }
        }
      }

      console.warn(
        `Hugging Face attempt ${attempt}/${maxAttempts} failed (${response.status}). Retrying with a stricter prompt...`,
      );
      attemptPrompt = `${prompt}

The previous attempt produced invalid JSON. Return exactly one JSON object that matches the requested shape. Between array items write only "},{" with no other characters. Close every array with "]" and every object with "}". Do not emit trailing commas or unbalanced brackets.`;
      continue;
    }

    const payload = (await response.json()) as HuggingFaceChatCompletionResponse;
    const rawText = extractJsonText(getChatCompletionText(payload));

    const candidates: string[] = [rawText];
    const cleaned = stripStrayQuotesBetweenStructures(rawText);
    if (cleaned !== rawText) {
      candidates.push(cleaned);
    }
    const repairedRaw = tryRepairTruncatedJson(rawText);
    if (repairedRaw && repairedRaw !== rawText) {
      candidates.push(repairedRaw);
    }
    const repairedCleaned = tryRepairTruncatedJson(cleaned);
    if (
      repairedCleaned &&
      repairedCleaned !== cleaned &&
      repairedCleaned !== repairedRaw
    ) {
      candidates.push(repairedCleaned);
    }

    for (const candidate of candidates) {
      try {
        const parsed = JSON.parse(candidate);
        return schema.parse(parsed);
      } catch (error) {
        lastError = error;
      }
    }

    if (attempt === maxAttempts) {
      break;
    }

    attemptPrompt = `${prompt}

Your previous response did not parse correctly. Return one valid JSON object that exactly matches the requested shape. Close every array and object. Do not include trailing commas.`;
  }

  throw new Error(
    `Hugging Face output did not match the expected ${responseFormatName} schema after ${maxAttempts} attempts: ${String(
      lastError,
    )}`,
  );
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

function buildSignalsPrompt(researchDossier: ResearchDossier) {
  return `You are a weekly website-refresh agent for a public AI portfolio. Use the research dossier below to refresh the latest-information datasets that power the site.

Rules:
- Return JSON only. No markdown, no code fences, no commentary.
- Keep the exact top-level keys requested.
- Use absolute dates like "April 9, 2026" when possible. If a source only supports a broader time range, use a precise range string such as "2025-2026 rollout".
- Prefer official or primary sources for URLs when the dossier includes them.
- Keep the tone concise, factual, and useful for a premium portfolio site.
- Update only current-information datasets. Do not turn evergreen sections into news feeds.
- Preserve the current array lengths exactly.
- Only use sources that appear in the research dossier below.
- If the dossier is thin for a specific slot, keep the current item rather than inventing one.

Research dossier:
${JSON.stringify(researchDossier, null, 2)}

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
  const researchDossier = await buildResearchDossier();

  console.log("Refreshing latest AI signal datasets...");
  const refreshedSignals = await generateJsonWithHuggingFace(
    buildSignalsPrompt(researchDossier),
    signalSetSchema,
    "site refresh signals",
  );

  console.log("Refreshing page-shell copy...");
  const refreshedPageShells = await generateJsonWithHuggingFace(
    buildPageShellPrompt(refreshedSignals, updatedAtLabel),
    pageShellOutputSchema,
    "page shell refresh content",
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
