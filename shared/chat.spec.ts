import { describe, expect, it } from "vitest";
import { buildChatSystemPrompt, getLocalChatResponse } from "./chat";

describe("chat grounding", () => {
  it("returns current resume guidance instead of the older generic bio", () => {
    const response = getLocalChatResponse("Where can I find Aakriti's resume?");

    expect(response).toMatch(/resume-builder/i);
    expect(response).toMatch(/linkedin/i);
    expect(response).toMatch(/github/i);
  });

  it("answers education questions from current site knowledge", () => {
    const response = getLocalChatResponse("What is Aakriti's educational background?");

    expect(response).toMatch(/B\.Tech/i);
    expect(response).toMatch(/AIEEE/i);
    expect(response).toMatch(/IPU-CET/i);
  });

  it("grounds the system prompt in relevant website knowledge", () => {
    const prompt = buildChatSystemPrompt("Tell me about Swarnawastra");

    expect(prompt).toMatch(/answer using only the website knowledge below/i);
    expect(prompt).toMatch(/Swarnawastra/i);
    expect(prompt).toMatch(/talent\/social-entrepreneur/i);
  });
});
