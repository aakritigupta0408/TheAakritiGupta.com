export interface ResumeAgentSection {
  id: string;
  title: string;
  bullets: string[];
}

export interface ResumeAgentProfile {
  candidateName: string;
  professionalHeadline: string;
  summary: string;
  sections: ResumeAgentSection[];
  suggestedQuestions: string[];
}

export interface ResumeAgentChatTurn {
  role: "user" | "assistant";
  content: string;
}

const MAX_SECTION_COUNT = 6;
const MAX_BULLETS_PER_SECTION = 5;
const MAX_SUGGESTIONS = 4;
const TOKEN_PREFIX = "ra1.";
const SEARCH_STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "can",
  "candidate",
  "does",
  "have",
  "in",
  "is",
  "me",
  "of",
  "or",
  "please",
  "tell",
  "the",
  "this",
  "what",
]);

const normalizeWhitespace = (value: string) =>
  value.replace(/\r/g, "\n").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();

const normalizeSearchValue = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

function toBase64Url(value: string): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(value, "utf-8").toString("base64url");
  }

  return btoa(unescape(encodeURIComponent(value)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function fromBase64Url(value: string): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(value, "base64url").toString("utf-8");
  }

  const padded = value.replace(/-/g, "+").replace(/_/g, "/");
  const normalized = padded.padEnd(Math.ceil(padded.length / 4) * 4, "=");
  return decodeURIComponent(escape(atob(normalized)));
}

function clampText(value: string, maxLength: number): string {
  const normalized = normalizeWhitespace(value);

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength - 1).trim()}...`;
}

function titleFromSectionId(id: string): string {
  return id
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .trim();
}

function splitIntoBullets(value: string, maxItems: number): string[] {
  return normalizeWhitespace(value)
    .split(/\n|[.;]\s+/)
    .map((part) => clampText(part, 180))
    .filter((part) => part.length > 10)
    .slice(0, maxItems);
}

export function sanitizeResumeAgentProfile(
  input: Partial<ResumeAgentProfile>,
): ResumeAgentProfile {
  const sections = (input.sections ?? [])
    .map((section, index) => ({
      id: clampText(section.id || `section-${index + 1}`, 40)
        .toLowerCase()
        .replace(/[^a-z0-9-]+/g, "-")
        .replace(/^-+|-+$/g, ""),
      title: clampText(section.title || titleFromSectionId(section.id || "section"), 60),
      bullets: (section.bullets ?? [])
        .map((bullet) => clampText(bullet, 180))
        .filter(Boolean)
        .slice(0, MAX_BULLETS_PER_SECTION),
    }))
    .filter((section) => section.title && section.bullets.length > 0)
    .slice(0, MAX_SECTION_COUNT);

  const safeSections =
    sections.length > 0
      ? sections
      : [
          {
            id: "candidate-material",
            title: "Candidate Material",
            bullets: ["The uploaded material did not produce enough structured detail."],
          },
        ];

  const suggestions = (input.suggestedQuestions ?? [])
    .map((question) => clampText(question, 120))
    .filter(Boolean)
    .slice(0, MAX_SUGGESTIONS);

  return {
    candidateName: clampText(input.candidateName || "Candidate", 80),
    professionalHeadline: clampText(
      input.professionalHeadline || "Candidate profile built from uploaded resume material",
      140,
    ),
    summary: clampText(
      input.summary ||
        "This chat is grounded only in the resume and project notes shared for this candidate.",
      320,
    ),
    sections: safeSections,
    suggestedQuestions:
      suggestions.length > 0
        ? suggestions
        : [
            "What are the candidate's strongest skills?",
            "Can you summarize the most relevant projects?",
            "What experience stands out for this role?",
          ],
  };
}

export function createResumeAgentProfileFromInput(input: {
  candidateName?: string;
  resumeText: string;
  projectNotes: string;
}): ResumeAgentProfile {
  const resumeText = normalizeWhitespace(input.resumeText);
  const projectNotes = normalizeWhitespace(input.projectNotes);
  const name =
    clampText(input.candidateName || resumeText.split("\n")[0] || "Candidate", 80) ||
    "Candidate";

  const combinedLines = `${resumeText}\n${projectNotes}`
    .split("\n")
    .map((line) => normalizeWhitespace(line))
    .filter((line) => line.length > 0);

  const summaryBullets = combinedLines.slice(0, 3).map((line) => clampText(line, 180));

  const skillMatches = Array.from(
    new Set(
      `${resumeText}\n${projectNotes}`
        .match(
          /\b(react|typescript|javascript|python|java|sql|aws|gcp|azure|machine learning|deep learning|nlp|computer vision|pytorch|tensorflow|node|express|product|leadership|analytics|data engineering|llm|ai)\b/gi,
        ) ?? [],
    ),
  )
    .map((match) => match.trim())
    .slice(0, 8);

  const projectBullets = splitIntoBullets(projectNotes, 5);
  const experienceBullets = splitIntoBullets(resumeText, 5);

  return sanitizeResumeAgentProfile({
    candidateName: name,
    professionalHeadline:
      summaryBullets[0] || "Candidate profile built from uploaded resume material",
    summary:
      summaryBullets.join(" ") ||
      "This chat is grounded only in the resume and project notes shared for this candidate.",
    sections: [
      experienceBullets.length > 0
        ? {
            id: "experience",
            title: "Experience",
            bullets: experienceBullets,
          }
        : undefined,
      skillMatches.length > 0
        ? {
            id: "skills",
            title: "Skills",
            bullets: skillMatches.map((skill) => `Mentions ${skill}`),
          }
        : undefined,
      projectBullets.length > 0
        ? {
            id: "projects",
            title: "Projects",
            bullets: projectBullets,
          }
        : undefined,
    ].filter(Boolean) as ResumeAgentSection[],
    suggestedQuestions: [
      "What skills does this candidate highlight?",
      "Can you summarize the key projects?",
      "What experience is most relevant here?",
      "What should a recruiter know first?",
    ],
  });
}

export function encodeResumeAgentProfile(profile: ResumeAgentProfile): string {
  const sanitized = sanitizeResumeAgentProfile(profile);
  return `${TOKEN_PREFIX}${toBase64Url(JSON.stringify(sanitized))}`;
}

export function decodeResumeAgentProfile(token: string): ResumeAgentProfile | null {
  if (!token.startsWith(TOKEN_PREFIX)) {
    return null;
  }

  try {
    const decoded = fromBase64Url(token.slice(TOKEN_PREFIX.length));
    const parsed = JSON.parse(decoded) as Partial<ResumeAgentProfile>;
    return sanitizeResumeAgentProfile(parsed);
  } catch {
    return null;
  }
}

function getRelevantSections(
  profile: ResumeAgentProfile,
  question: string,
): ResumeAgentSection[] {
  const normalizedQuestion = normalizeSearchValue(question);
  const terms = normalizedQuestion
    .split(" ")
    .filter((term) => term.length > 2 && !SEARCH_STOP_WORDS.has(term));

  if (terms.length === 0) {
    return profile.sections.slice(0, 2);
  }

  return [...profile.sections]
    .map((section) => {
      const haystack = normalizeSearchValue(
        `${section.title} ${section.bullets.join(" ")}`,
      );
      let score = 0;

      if (haystack.includes(normalizedQuestion)) {
        score += 60;
      }

      for (const term of terms) {
        if (haystack.includes(term)) {
          score += 12;
        }
      }

      return { section, score };
    })
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score)
    .map((entry) => entry.section)
    .slice(0, 2);
}

export function getLocalResumeAgentResponse(
  profile: ResumeAgentProfile,
  message: string,
): string {
  const normalizedMessage = normalizeSearchValue(message);

  if (
    normalizedMessage.includes("tell me about") ||
    normalizedMessage.includes("summary") ||
    normalizedMessage.includes("overview")
  ) {
    const sectionPreview = profile.sections
      .slice(0, 2)
      .flatMap((section) => section.bullets.slice(0, 2))
      .slice(0, 4);

    return [
      `Based only on the material shared for ${profile.candidateName}:`,
      profile.summary,
      ...sectionPreview.map((bullet) => `- ${bullet}`),
    ].join("\n");
  }

  const relevantSections = getRelevantSections(profile, message);

  if (relevantSections.length === 0) {
    return `That detail is not in the candidate material shared for this chat. Ask about ${profile.sections
      .slice(0, 3)
      .map((section) => section.title.toLowerCase())
      .join(", ")}.`;
  }

  const bullets = relevantSections.flatMap((section) =>
    section.bullets.slice(0, 2).map((bullet) => `- ${section.title}: ${bullet}`),
  );

  return [
    `Using only the candidate material shared for ${profile.candidateName}:`,
    ...bullets.slice(0, 4),
  ].join("\n");
}

export function buildResumeAgentSystemPrompt(
  profile: ResumeAgentProfile,
  recruiterQuestion: string,
): string {
  return [
    "You are a recruiter-facing chat agent for one candidate.",
    "Answer using only the structured candidate material below.",
    "Do not invent facts, metrics, dates, employers, technologies, education, certifications, or project details.",
    "If a detail is missing, say exactly: That detail is not in the candidate material shared for this chat.",
    "Keep answers concise, recruiter-friendly, and grounded.",
    "You may synthesize across sections, but you may not add new information.",
    "",
    `Candidate name: ${profile.candidateName}`,
    `Professional headline: ${profile.professionalHeadline}`,
    `Summary: ${profile.summary}`,
    "Structured material:",
    ...profile.sections.map(
      (section) =>
        `${section.title}: ${section.bullets.map((bullet) => `- ${bullet}`).join(" ")}`,
    ),
    "",
    `Current recruiter question: ${recruiterQuestion}`,
  ].join("\n");
}
