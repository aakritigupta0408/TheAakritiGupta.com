import type { ResumeAgentProfile } from "@shared/resume-agent";
import { sanitizeResumeAgentProfile } from "@shared/resume-agent";

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL?.trim() || "";
const SUPABASE_PUBLISHABLE_KEY =
  import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY?.trim() || "";

function getSupabaseRpcUrl(functionName: string) {
  return `${SUPABASE_URL.replace(/\/+$/, "")}/rest/v1/rpc/${functionName}`;
}

function hasSupabasePersistenceConfig() {
  return Boolean(SUPABASE_URL && SUPABASE_PUBLISHABLE_KEY);
}

async function callSupabaseRpc<T>(
  functionName: string,
  payload: Record<string, unknown>,
): Promise<T | null> {
  if (!hasSupabasePersistenceConfig()) {
    return null;
  }

  const response = await fetch(getSupabaseRpcUrl(functionName), {
    method: "POST",
    headers: {
      apikey: SUPABASE_PUBLISHABLE_KEY,
      Authorization: `Bearer ${SUPABASE_PUBLISHABLE_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      `Supabase RPC ${functionName} failed: ${response.status} ${response.statusText} ${errorText}`.trim(),
    );
  }

  return (await response.json()) as T;
}

export async function persistResumeAgentProfile(
  profile: ResumeAgentProfile,
): Promise<string | null> {
  if (!hasSupabasePersistenceConfig()) {
    return null;
  }

  const result = await callSupabaseRpc<string | { id?: string }>(
    "create_resume_agent_link",
    {
      p_profile: sanitizeResumeAgentProfile(profile),
    },
  );

  if (!result) {
    return null;
  }

  if (typeof result === "string") {
    return result;
  }

  return typeof result.id === "string" ? result.id : null;
}

export async function fetchPersistedResumeAgentProfile(
  agentId: string,
): Promise<ResumeAgentProfile | null> {
  if (!hasSupabasePersistenceConfig()) {
    return null;
  }

  const result = await callSupabaseRpc<unknown>("get_resume_agent_profile", {
    p_id: agentId,
  });

  if (!result || typeof result !== "object") {
    return null;
  }

  return sanitizeResumeAgentProfile(result as Partial<ResumeAgentProfile>);
}

export function isResumeAgentPersistenceConfigured() {
  return hasSupabasePersistenceConfig();
}
