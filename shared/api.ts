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
