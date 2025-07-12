import { NextApiRequest, NextApiResponse } from "next";
import fs from "fs";
import path from "path";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  if (req.method !== "POST") {
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    const { email } = req.body;

    if (!email || typeof email !== "string") {
      return res.status(400).json({ message: "Valid email is required" });
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ message: "Invalid email format" });
    }

    const timestamp = new Date().toISOString();
    const emailEntry = `${timestamp} - ${email}\n`;

    // Define the path for the emails file
    const emailsFilePath = path.join(process.cwd(), "emails.txt");

    // Append the email to the file
    await fs.promises.appendFile(emailsFilePath, emailEntry, "utf8");

    console.log(`Email saved to file: ${email} at ${timestamp}`);

    res.status(200).json({
      message: "Email saved successfully",
      timestamp,
      email,
    });
  } catch (error) {
    console.error("Error saving email:", error);
    res.status(500).json({
      message: "Internal server error",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}
