import type {
  ResumeAgentBuildRequest,
  ResumeAgentBuildResponse,
  ResumeAgentChatRequest,
  ResumeAgentChatResponse,
  ResumeAgentFetchResponse,
} from "@shared/api";
import {
  createResumeAgentProfileFromInput,
  encodeResumeAgentProfile,
  getLocalResumeAgentResponse,
  sanitizeResumeAgentProfile,
} from "@shared/resume-agent";
import {
  fetchPersistedResumeAgentProfile,
  persistResumeAgentProfile,
} from "@/lib/resume-agent-persistence";

async function attachPersistentShareId(
  result: ResumeAgentBuildResponse,
): Promise<ResumeAgentBuildResponse> {
  if (result.shareId) {
    return result;
  }

  try {
    const shareId = await persistResumeAgentProfile(result.profile);

    if (!shareId) {
      return result;
    }

    return {
      ...result,
      shareId,
    };
  } catch (error) {
    console.error("Resume agent persistence failed:", error);
    return result;
  }
}

export async function buildResumeAgent(
  request: ResumeAgentBuildRequest,
): Promise<ResumeAgentBuildResponse> {
  try {
    const response = await fetch("/api/resume-agent/build", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Resume agent build failed: ${response.statusText}`);
    }

    return await attachPersistentShareId(
      (await response.json()) as ResumeAgentBuildResponse,
    );
  } catch (error) {
    console.error("Resume agent build API failed, using local fallback:", error);
    const profile = createResumeAgentProfileFromInput(request);

    return await attachPersistentShareId({
      profile,
      shareToken: encodeResumeAgentProfile(profile),
      usedModel: false,
    });
  }
}

export async function fetchResumeAgent(
  agentId: string,
): Promise<ResumeAgentFetchResponse | null> {
  try {
    const response = await fetch(`/api/resume-agent/${encodeURIComponent(agentId)}`);

    if (response.ok) {
      return (await response.json()) as ResumeAgentFetchResponse;
    }
  } catch (error) {
    console.error("Resume agent fetch API failed:", error);
  }

  try {
    const profile = await fetchPersistedResumeAgentProfile(agentId);

    if (!profile) {
      return null;
    }

    return {
      profile,
      shareId: agentId,
      usedModel: false,
    };
  } catch (error) {
    console.error("Resume agent database fetch failed:", error);
    return null;
  }
}

export async function chatWithResumeAgent(
  request: ResumeAgentChatRequest,
): Promise<ResumeAgentChatResponse> {
  try {
    const response = await fetch("/api/resume-agent/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Resume agent chat failed: ${response.statusText}`);
    }

    return (await response.json()) as ResumeAgentChatResponse;
  } catch (error) {
    console.error("Resume agent chat API failed, using local fallback:", error);
    const profile = sanitizeResumeAgentProfile(request.profile);

    return {
      response: getLocalResumeAgentResponse(profile, request.message),
      usedModel: false,
    };
  }
}
