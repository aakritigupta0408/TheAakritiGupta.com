import fs from "fs";
import os from "os";
import path from "path";
import { RequestHandler } from "express";
import { z } from "zod";
import type { SaveEmailResponse } from "../../shared/api";

const saveEmailSchema = z.object({
  email: z.string().trim().email(),
});

export const handleSaveEmail: RequestHandler = async (req, res) => {
  const parseResult = saveEmailSchema.safeParse(req.body);

  if (!parseResult.success) {
    const response: SaveEmailResponse = {
      success: false,
      message: "Please provide a valid email address.",
    };

    res.status(400).json(response);
    return;
  }

  try {
    const timestamp = new Date().toISOString();
    const emailEntry = `${timestamp} - ${parseResult.data.email}\n`;
    const isServerlessRuntime =
      Boolean(process.env.NETLIFY) ||
      Boolean(process.env.AWS_LAMBDA_FUNCTION_NAME);
    const emailsDirectory = process.env.EMAIL_CAPTURE_DIR
      ? path.resolve(process.env.EMAIL_CAPTURE_DIR)
      : isServerlessRuntime
        ? os.tmpdir()
        : process.cwd();
    const emailsFilePath = path.join(emailsDirectory, "emails.txt");

    await fs.promises.mkdir(emailsDirectory, { recursive: true });
    await fs.promises.appendFile(emailsFilePath, emailEntry, "utf8");

    const response: SaveEmailResponse = {
      success: true,
      message: "Email saved successfully.",
    };

    res.status(201).json(response);
  } catch (error) {
    console.error("Error saving email to file:", error);

    const response: SaveEmailResponse = {
      success: false,
      message: "Unable to save email right now.",
    };

    res.status(500).json(response);
  }
};
