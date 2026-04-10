import { createHash } from "crypto";
import { mkdir, readFile, writeFile } from "fs/promises";
import path from "path";
import type { ResumeAgentProfile } from "../../shared/resume-agent";

interface StoredResumeAgentRecord {
  id: string;
  profile: ResumeAgentProfile;
  createdAt: string;
  updatedAt: string;
}

const STORE_DIR =
  process.env.RESUME_AGENT_STORE_DIR ||
  path.join(process.cwd(), ".runtime-data", "resume-agents");

function getRecordPath(id: string) {
  return path.join(STORE_DIR, `${id}.json`);
}

function sanitizeId(id: string) {
  return id.toLowerCase().replace(/[^a-z0-9-]/g, "");
}

export function createResumeAgentId(profile: ResumeAgentProfile): string {
  const hash = createHash("sha256")
    .update(JSON.stringify(profile))
    .digest("hex")
    .slice(0, 12);

  const nameSlug = profile.candidateName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 20);

  return sanitizeId(`${nameSlug || "candidate"}-${hash}`);
}

export async function saveResumeAgentProfile(profile: ResumeAgentProfile): Promise<string> {
  const id = createResumeAgentId(profile);
  const recordPath = getRecordPath(id);
  const timestamp = new Date().toISOString();

  await mkdir(STORE_DIR, { recursive: true });

  const record: StoredResumeAgentRecord = {
    id,
    profile,
    createdAt: timestamp,
    updatedAt: timestamp,
  };

  try {
    const existing = await readFile(recordPath, "utf-8");
    const parsed = JSON.parse(existing) as StoredResumeAgentRecord;
    record.createdAt = parsed.createdAt || timestamp;
  } catch {
    // First write for this id.
  }

  await writeFile(recordPath, JSON.stringify(record, null, 2), "utf-8");

  return id;
}

export async function loadResumeAgentProfile(
  id: string,
): Promise<ResumeAgentProfile | null> {
  const safeId = sanitizeId(id);

  if (!safeId) {
    return null;
  }

  try {
    const content = await readFile(getRecordPath(safeId), "utf-8");
    const record = JSON.parse(content) as StoredResumeAgentRecord;
    return record.profile ?? null;
  } catch {
    return null;
  }
}
