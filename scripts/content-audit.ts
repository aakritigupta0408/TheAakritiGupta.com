import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

type Severity = "error" | "warn";

interface Finding {
  severity: Severity;
  file: string;
  line: number;
  url: string;
  reason: string;
}

const SCAN_ROOTS = ["client/pages", "client/data"];
const FILE_EXTENSIONS = new Set([".ts", ".tsx"]);
const URL_REGEX = /https?:\/\/[^\s"'`)<>\]]+/g;
const COMMUNITY_HOSTS = new Set([
  "reddit.com",
  "www.reddit.com",
  "discord.gg",
]);
const MARKETPLACE_HOSTS = new Set([
  "www.udemy.com",
  "udemy.com",
  "www.coursera.org",
  "coursera.org",
  "www.edx.org",
  "edx.org",
  "www.masterclass.com",
  "masterclass.com",
]);
const VIDEO_HOSTS = new Set(["www.youtube.com", "youtube.com", "youtu.be"]);

function walk(directory: string): string[] {
  const entries = readdirSync(directory).sort();
  const files: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(directory, entry);
    const stats = statSync(fullPath);

    if (stats.isDirectory()) {
      files.push(...walk(fullPath));
      continue;
    }

    if (FILE_EXTENSIONS.has(path.extname(fullPath))) {
      files.push(fullPath);
    }
  }

  return files;
}

function getLineNumber(source: string, index: number) {
  return source.slice(0, index).split("\n").length;
}

function normalizeHost(url: URL) {
  return url.hostname.toLowerCase();
}

function classify(urlString: string): Omit<Finding, "file" | "line" | "url">[] {
  let parsed: URL;

  try {
    parsed = new URL(urlString);
  } catch {
    return [
      {
        severity: "error",
        reason: "Invalid URL syntax in source data.",
      },
    ];
  }

  const host = normalizeHost(parsed);
  const findings: Omit<Finding, "file" | "line" | "url">[] = [];

  if (host === "docs.microsoft.com") {
    findings.push({
      severity: "error",
      reason: "Deprecated Microsoft docs host. Use learn.microsoft.com instead.",
    });
  }

  if (
    host.includes("youtube.com") &&
    ["education-ai", "debugging-ai"].includes(parsed.searchParams.get("v") || "")
  ) {
    findings.push({
      severity: "error",
      reason: "Placeholder YouTube ID detected. Replace with a real source or remove it.",
    });
  }

  if (
    host === "platform.openai.com" &&
    parsed.pathname === "/docs/assistants/overview"
  ) {
    findings.push({
      severity: "warn",
      reason: "Legacy OpenAI Assistants docs path. Confirm the current Agents/Responses equivalent.",
    });
  }

  if (COMMUNITY_HOSTS.has(host)) {
    findings.push({
      severity: "warn",
      reason: "Community link detected. Prefer official docs, company blogs, arXiv, or Papers with Code for evergreen page content.",
    });
  }

  if (MARKETPLACE_HOSTS.has(host)) {
    findings.push({
      severity: "warn",
      reason: "Course marketplace link detected. Prefer primary sources for long-lived reference pages.",
    });
  }

  if (VIDEO_HOSTS.has(host)) {
    findings.push({
      severity: "warn",
      reason: "Video link detected. Verify there is also a stable written primary source for the same claim.",
    });
  }

  return findings;
}

function collectFindings() {
  const findings: Finding[] = [];

  for (const root of SCAN_ROOTS) {
    const directory = path.resolve(process.cwd(), root);
    const files = walk(directory);

    for (const file of files) {
      const source = readFileSync(file, "utf8");

      for (const match of source.matchAll(URL_REGEX)) {
        const url = match[0];
        const index = match.index ?? 0;
        const line = getLineNumber(source, index);

        for (const issue of classify(url)) {
          findings.push({
            ...issue,
            file: path.relative(process.cwd(), file),
            line,
            url,
          });
        }
      }
    }
  }

  return findings;
}

function main() {
  const strictWarnings = process.argv.includes("--fail-on-warn");
  const findings = collectFindings();
  const errors = findings.filter((finding) => finding.severity === "error");
  const warnings = findings.filter((finding) => finding.severity === "warn");

  if (findings.length === 0) {
    console.log("Content audit passed. No flagged links found.");
    return;
  }

  for (const severity of ["error", "warn"] as const) {
    const bucket = findings.filter((finding) => finding.severity === severity);

    if (bucket.length === 0) {
      continue;
    }

    console.log(`\n${severity.toUpperCase()}S (${bucket.length})`);

    for (const finding of bucket) {
      console.log(
        `- ${finding.file}:${finding.line} :: ${finding.reason}\n  ${finding.url}`,
      );
    }
  }

  if (errors.length > 0 || (strictWarnings && warnings.length > 0)) {
    process.exitCode = 1;
  }
}

main();
