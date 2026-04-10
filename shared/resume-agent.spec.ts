import { describe, expect, it } from "vitest";
import {
  decodeResumeAgentProfile,
  encodeResumeAgentProfile,
  getLocalResumeAgentResponse,
  sanitizeResumeAgentProfile,
} from "./resume-agent";

describe("resume agent utilities", () => {
  const profile = sanitizeResumeAgentProfile({
    candidateName: "Riya Sharma",
    professionalHeadline: "AI engineer focused on production ML systems",
    summary: "Built production ML systems and recruiter-facing project narratives.",
    sections: [
      {
        id: "skills",
        title: "Skills",
        bullets: ["Python", "React", "LLM evaluation"],
      },
      {
        id: "projects",
        title: "Projects",
        bullets: [
          "Built a recruiter chatbot grounded in uploaded resume data.",
          "Implemented evaluation checks so the agent refuses missing details.",
        ],
      },
    ],
    suggestedQuestions: ["What projects stand out?"],
  });

  it("encodes and decodes share tokens", () => {
    const token = encodeResumeAgentProfile(profile);
    const decoded = decodeResumeAgentProfile(token);

    expect(decoded).not.toBeNull();
    expect(decoded?.candidateName).toBe("Riya Sharma");
    expect(decoded?.sections[1]?.title).toBe("Projects");
  });

  it("grounds local responses in the provided profile", () => {
    const response = getLocalResumeAgentResponse(
      profile,
      "What projects does this candidate have?",
    );

    expect(response).toMatch(/uploaded resume data/i);
    expect(response).toMatch(/Projects:/);
  });

  it("refuses details not in the profile", () => {
    const response = getLocalResumeAgentResponse(
      profile,
      "What certifications does the candidate have?",
    );

    expect(response).toMatch(/not in the candidate material/i);
  });
});
