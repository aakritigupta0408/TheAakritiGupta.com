/**
 * Shared code between client and server
 * Useful to share types between client and server
 * and/or small pure JS functions that can be used on both client and server
 */

/**
 * Example response type for /api/demo
 */
export interface DemoResponse {
  message: string;
}

export interface SaveEmailRequest {
  email: string;
}

export interface SaveEmailResponse {
  success: boolean;
  message: string;
}

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  response: string;
}

export interface ResumeAgentBuildRequest {
  candidateName?: string;
  resumeText: string;
  projectNotes: string;
}

export interface ResumeAgentSection {
  id: string;
  title: string;
  bullets: string[];
}

export interface ResumeAgentProfile {
  candidateName: string;
  professionalHeadline: string;
  summary: string;
  sections: ResumeAgentSection[];
  suggestedQuestions: string[];
}

export interface ResumeAgentBuildResponse {
  profile: ResumeAgentProfile;
  shareToken: string;
  shareId?: string;
  usedModel: boolean;
}

export interface ResumeAgentChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ResumeAgentChatRequest {
  profile: ResumeAgentProfile;
  message: string;
  history?: ResumeAgentChatTurn[];
}

export interface ResumeAgentChatResponse {
  response: string;
  usedModel: boolean;
}

export interface ResumeAgentFetchResponse {
  profile: ResumeAgentProfile;
  shareId: string;
  usedModel: boolean;
}

export interface SiteRefreshTriggerRequest {
  source?: string;
}

export interface SiteRefreshTriggerResponse {
  success: boolean;
  message: string;
  workflowUrl?: string;
  queuedAt?: string;
  cooldownUntil?: string;
}
