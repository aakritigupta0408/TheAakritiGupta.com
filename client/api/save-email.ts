export const saveEmailToLocalStorage = (email: string): boolean => {
  try {
    if (typeof window === "undefined") {
      return false;
    }

    const timestamp = new Date().toISOString();
    const existingEmails = localStorage.getItem("submittedEmails") || "";
    const newEmailEntry = `${timestamp} - ${email}\n`;
    const updatedEmails = existingEmails + newEmailEntry;

    localStorage.setItem("submittedEmails", updatedEmails);
    return true;
  } catch {
    return false;
  }
};

export const downloadEmailsFromLocalStorage = (): void => {
  try {
    if (typeof window === "undefined") {
      return;
    }

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
  } catch {
    // Silent fallback — email download is best-effort
  }
};
