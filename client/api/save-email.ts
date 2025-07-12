import fs from "fs";
import path from "path";

export const saveEmailToFile = async (email: string): Promise<void> => {
  try {
    const timestamp = new Date().toISOString();
    const emailEntry = `${timestamp} - ${email}\n`;

    // Define the path for the emails file
    const emailsFilePath = path.join(process.cwd(), "emails.txt");

    // Append the email to the file
    await fs.promises.appendFile(emailsFilePath, emailEntry, "utf8");

    console.log(`Email saved: ${email} at ${timestamp}`);
  } catch (error) {
    console.error("Error saving email to file:", error);
    throw error;
  }
};

// For client-side fallback - save to localStorage as backup
export const saveEmailToLocalStorage = (email: string): void => {
  try {
    const timestamp = new Date().toISOString();
    const existingEmails = localStorage.getItem("submittedEmails") || "";
    const newEmailEntry = `${timestamp} - ${email}\n`;
    const updatedEmails = existingEmails + newEmailEntry;

    localStorage.setItem("submittedEmails", updatedEmails);
    console.log("Email saved to localStorage as backup:", email);
  } catch (error) {
    console.error("Error saving email to localStorage:", error);
  }
};

// Export emails from localStorage to downloadable file
export const downloadEmailsFromLocalStorage = (): void => {
  try {
    const emails = localStorage.getItem("submittedEmails") || "No emails found";
    const blob = new Blob([emails], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `aakriti-portfolio-emails-${new Date().toISOString().split("T")[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
    console.log("Emails downloaded from localStorage");
  } catch (error) {
    console.error("Error downloading emails:", error);
  }
};
