import { RequestHandler } from "express";
import { z } from "zod";
import type { ChatResponse } from "../../shared/api";
import { CHAT_SYSTEM_PROMPT, getLocalChatResponse } from "../../shared/chat";

const chatRequestSchema = z.object({
  message: z.string().trim().min(1),
});

interface OpenAIChatCompletionResponse {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
}

export const handleChat: RequestHandler = async (req, res) => {
  const parseResult = chatRequestSchema.safeParse(req.body);

  if (!parseResult.success) {
    const response: ChatResponse = {
      response: "Please send a non-empty message.",
    };

    res.status(400).json(response);
    return;
  }

  const message = parseResult.data.message;
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    const response: ChatResponse = {
      response: getLocalChatResponse(message),
    };

    res.status(200).json(response);
    return;
  }

  try {
    const openAIResponse = await fetch(
      "https://api.openai.com/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini",
          messages: [
            {
              role: "system",
              content: CHAT_SYSTEM_PROMPT,
            },
            {
              role: "user",
              content: message,
            },
          ],
          max_tokens: 150,
          temperature: 0.7,
        }),
      },
    );

    if (!openAIResponse.ok) {
      throw new Error(`OpenAI API error: ${openAIResponse.statusText}`);
    }

    const data =
      (await openAIResponse.json()) as OpenAIChatCompletionResponse;
    const content = data.choices?.[0]?.message?.content?.trim();

    const response: ChatResponse = {
      response: content || getLocalChatResponse(message),
    };

    res.status(200).json(response);
  } catch (error) {
    console.error("Error calling OpenAI API:", error);

    const response: ChatResponse = {
      response: getLocalChatResponse(message),
    };

    res.status(200).json(response);
  }
};
