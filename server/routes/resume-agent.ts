import { RequestHandler } from "express";
import { z } from "zod";
import type {
  ResumeAgentBuildResponse,
  ResumeAgentChatResponse,
  ResumeAgentFetchResponse,
} from "../../shared/api";
import {
  buildResumeAgentSystemPrompt,
  createResumeAgentProfileFromInput,
  encodeResumeAgentProfile,
  getLocalResumeAgentResponse,
  sanitizeResumeAgentProfile,
  type ResumeAgentProfile,
} from "../../shared/resume-agent";
import {
  loadResumeAgentProfile,
  saveResumeAgentProfile,
} from "../lib/resume-agent-store";

const resumeAgentSectionSchema = z.object({
  id: z.string().trim().min(1).max(40),
  title: z.string().trim().min(1).max(60),
  bullets: z.array(z.string().trim().min(1).max(180)).min(1).max(5),
});

const resumeAgentProfileSchema = z.object({
  candidateName: z.string().trim().min(1).max(80),
  professionalHeadline: z.string().trim().min(1).max(140),
  summary: z.string().trim().min(1).max(320),
  sections: z.array(resumeAgentSectionSchema).min(1).max(6),
  suggestedQuestions: z.array(z.string().trim().min(1).max(120)).min(1).max(4),
});

const buildRequestSchema = z.object({
  candidateName: z.string().trim().max(80).optional(),
  resumeText: z.string().trim().min(1).max(50000),
  projectNotes: z.string().trim().max(20000).default(""),
});

const chatRequestSchema = z.object({
  profile: resumeAgentProfileSchema,
  message: z.string().trim().min(1).max(1000),
  history: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().trim().min(1).max(1000),
      }),
    )
    .max(12)
    .optional(),
});

interface OpenAIChatCompletionResponse {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
}

function extractJsonObject(value: string): string {
  const trimmed = value.trim();
  const fencedMatch = trimmed.match(/```json\s*([\s\S]*?)```/i);

  if (fencedMatch?.[1]) {
    return fencedMatch[1].trim();
  }

  return trimmed;
}

async function callOpenAI(messages: Array<{ role: "system" | "user" | "assistant"; content: string }>, options?: { maxTokens?: number; responseFormat?: { type: "json_object" } }) {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    return null;
  }

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: process.env.OPENAI_RESUME_AGENT_MODEL || process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini",
      messages,
      max_tokens: options?.maxTokens ?? 700,
      temperature: 0.1,
      ...(options?.responseFormat ? { response_format: options.responseFormat } : {}),
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
  }

  const data = (await response.json()) as OpenAIChatCompletionResponse;
  return data.choices?.[0]?.message?.content?.trim() ?? null;
}

function buildResumeAgentBuildPrompt(input: {
  candidateName?: string;
  resumeText: string;
  projectNotes: string;
}): string {
  return [
    "You build recruiter-facing candidate profiles from user-provided material.",
    "Use only the resume text and project notes supplied by the user.",
    "Do not invent dates, metrics, employers, certifications, responsibilities, technologies, or outcomes.",
    "If a detail is missing, omit it instead of guessing.",
    "Return valid JSON only.",
    "Target compact output suitable for a share link.",
    "JSON shape:",
    "{",
    '  "candidateName": string,',
    '  "professionalHeadline": string,',
    '  "summary": string,',
    '  "sections": [{ "id": string, "title": string, "bullets": string[] }],',
    '  "suggestedQuestions": string[]',
    "}",
    "Rules:",
    "- sections: 3 to 6 sections",
    "- each section: 1 to 5 bullets",
    "- bullets must be concise and factual",
    "- suggestedQuestions: 3 to 4 recruiter-facing questions",
    '- if candidate name is missing, use "Candidate"',
    "",
    `Candidate name hint: ${input.candidateName || "Not provided"}`,
    "Resume text:",
    input.resumeText,
    "",
    "Project notes:",
    input.projectNotes || "No extra project notes were provided.",
  ].join("\n");
}

export const handleResumeAgentBuild: RequestHandler = async (req, res) => {
  const parseResult = buildRequestSchema.safeParse(req.body);

  if (!parseResult.success) {
    res.status(400).json({
      profile: createResumeAgentProfileFromInput({
        candidateName: "Candidate",
        resumeText: "Please upload resume text.",
        projectNotes: "",
      }),
      shareToken: "",
      usedModel: false,
    } satisfies ResumeAgentBuildResponse);
    return;
  }

  const input = {
    candidateName: parseResult.data.candidateName,
    resumeText: parseResult.data.resumeText,
    projectNotes: parseResult.data.projectNotes,
  };
  const fallbackProfile = createResumeAgentProfileFromInput(input);

  const respondWithProfile = async (
    profile: ResumeAgentProfile,
    usedModel: boolean,
  ) => {
    try {
      const shareId = await saveResumeAgentProfile(profile);

      res.status(200).json({
        profile,
        shareToken: encodeResumeAgentProfile(profile),
        shareId,
        usedModel,
      } satisfies ResumeAgentBuildResponse);
    } catch (error) {
      console.error("Error persisting resume agent profile:", error);

      res.status(200).json({
        profile,
        shareToken: encodeResumeAgentProfile(profile),
        usedModel,
      } satisfies ResumeAgentBuildResponse);
    }
  };

  try {
    const content = await callOpenAI(
      [
        {
          role: "system",
          content:
            "You convert candidate material into strict recruiter-facing JSON. Never fabricate facts.",
        },
        {
          role: "user",
          content: buildResumeAgentBuildPrompt(input),
        },
      ],
      {
        maxTokens: 900,
        responseFormat: { type: "json_object" },
      },
    );

    if (!content) {
      throw new Error("No model content returned");
    }

    const parsed = resumeAgentProfileSchema.parse(
      JSON.parse(extractJsonObject(content)),
    );
    const profile = sanitizeResumeAgentProfile(parsed as ResumeAgentProfile);

    await respondWithProfile(profile, true);
  } catch (error) {
    console.error("Error building resume agent profile:", error);

    await respondWithProfile(fallbackProfile, false);
  }
};

export const handleResumeAgentFetch: RequestHandler = async (req, res) => {
  const agentId = z.string().trim().min(1).safeParse(req.params.agentId);

  if (!agentId.success) {
    res.status(400).json({
      profile: createResumeAgentProfileFromInput({
        candidateName: "Candidate",
        resumeText: "No candidate profile found.",
        projectNotes: "",
      }),
      shareId: "",
      usedModel: false,
    } satisfies ResumeAgentFetchResponse);
    return;
  }

  const profile = await loadResumeAgentProfile(agentId.data);

  if (!profile) {
    res.status(404).json({
      profile: createResumeAgentProfileFromInput({
        candidateName: "Candidate",
        resumeText: "No candidate profile found.",
        projectNotes: "",
      }),
      shareId: agentId.data,
      usedModel: false,
    } satisfies ResumeAgentFetchResponse);
    return;
  }

  res.status(200).json({
    profile,
    shareId: agentId.data,
    usedModel: false,
  } satisfies ResumeAgentFetchResponse);
};

export const handleResumeAgentChat: RequestHandler = async (req, res) => {
  const parseResult = chatRequestSchema.safeParse(req.body);

  if (!parseResult.success) {
    res.status(400).json({
      response: "Please send a grounded recruiter question.",
      usedModel: false,
    } satisfies ResumeAgentChatResponse);
    return;
  }

  const { profile: unsafeProfile, message, history = [] } = parseResult.data;
  const profile = sanitizeResumeAgentProfile({
    candidateName: unsafeProfile.candidateName,
    professionalHeadline: unsafeProfile.professionalHeadline,
    summary: unsafeProfile.summary,
    sections: unsafeProfile.sections.map((section) => ({
      id: section.id,
      title: section.title,
      bullets: [...section.bullets],
    })),
    suggestedQuestions: [...unsafeProfile.suggestedQuestions],
  });
  const fallbackResponse = getLocalResumeAgentResponse(profile, message);

  try {
    const content = await callOpenAI(
      [
        {
          role: "system",
          content: buildResumeAgentSystemPrompt(profile, message),
        },
        ...history.slice(-8).map((entry) => ({
          role: entry.role,
          content: entry.content,
        })),
        {
          role: "user",
          content: message,
        },
      ],
      { maxTokens: 280 },
    );

    if (!content) {
      throw new Error("No model content returned");
    }

    res.status(200).json({
      response: content,
      usedModel: true,
    } satisfies ResumeAgentChatResponse);
  } catch (error) {
    console.error("Error running resume agent chat:", error);

    res.status(200).json({
      response: fallbackResponse,
      usedModel: false,
    } satisfies ResumeAgentChatResponse);
  }
};
