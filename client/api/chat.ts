import type { ChatRequest, ChatResponse } from "@shared/api";
import { getLocalChatResponse } from "@shared/chat";

export async function callOpenAI(userMessage: string): Promise<string> {
  try {
    const requestBody: ChatRequest = { message: userMessage };
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`Chat API error: ${response.statusText}`);
    }

    const data = (await response.json()) as ChatResponse;
    return data.response;
  } catch {
    return getLocalChatResponse(userMessage);
  }
}
