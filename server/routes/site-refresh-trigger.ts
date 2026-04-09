import { RequestHandler } from "express";
import { z } from "zod";
import type {
  SiteRefreshTriggerRequest,
  SiteRefreshTriggerResponse,
} from "@shared/api";

const triggerRequestSchema = z.object({
  source: z.string().trim().max(80).optional(),
});

interface GitHubWorkflowRun {
  status?: string;
  created_at?: string;
}

interface GitHubWorkflowRunsResponse {
  workflow_runs?: GitHubWorkflowRun[];
}

const DEFAULT_REPOSITORY = "aakritigupta0408/TheAakritiGupta.com";
const DEFAULT_WORKFLOW_FILE = "site-refresh.yml";
const DEFAULT_REF = "main";
const TRIGGER_COOLDOWN_MS = 6 * 60 * 60 * 1000;

function getWorkflowUrl(repository: string, workflowFile: string) {
  return `https://github.com/${repository}/actions/workflows/${workflowFile}`;
}

function getGitHubHeaders(token: string) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json",
  };
}

export const handleSiteRefreshTrigger: RequestHandler = async (req, res) => {
  const parseResult = triggerRequestSchema.safeParse(req.body ?? {});

  if (!parseResult.success) {
    const response: SiteRefreshTriggerResponse = {
      success: false,
      message: "Invalid site refresh trigger request.",
    };

    res.status(400).json(response);
    return;
  }

  const body = parseResult.data as SiteRefreshTriggerRequest;
  const githubToken = process.env.GITHUB_WORKFLOW_TOKEN;
  const repository = process.env.SITE_REFRESH_REPOSITORY || DEFAULT_REPOSITORY;
  const workflowFile =
    process.env.SITE_REFRESH_WORKFLOW_FILE || DEFAULT_WORKFLOW_FILE;
  const workflowRef = process.env.SITE_REFRESH_REF || DEFAULT_REF;
  const workflowUrl = getWorkflowUrl(repository, workflowFile);

  if (!githubToken) {
    const response: SiteRefreshTriggerResponse = {
      success: false,
      message:
        "Site refresh trigger is not configured yet. Add GITHUB_WORKFLOW_TOKEN in the deployment environment.",
      workflowUrl,
    };

    res.status(503).json(response);
    return;
  }

  try {
    const latestRunResponse = await fetch(
      `https://api.github.com/repos/${repository}/actions/workflows/${workflowFile}/runs?per_page=1&branch=${workflowRef}`,
      {
        headers: getGitHubHeaders(githubToken),
      },
    );

    if (!latestRunResponse.ok) {
      throw new Error(
        `Failed to inspect workflow runs (${latestRunResponse.status})`,
      );
    }

    const latestRunData =
      (await latestRunResponse.json()) as GitHubWorkflowRunsResponse;
    const latestRun = latestRunData.workflow_runs?.[0];

    if (latestRun?.created_at) {
      const createdAtMs = new Date(latestRun.created_at).getTime();
      const cooldownUntil = new Date(
        createdAtMs + TRIGGER_COOLDOWN_MS,
      ).toISOString();
      const stillCoolingDown = Date.now() < createdAtMs + TRIGGER_COOLDOWN_MS;
      const isActive =
        latestRun.status === "queued" || latestRun.status === "in_progress";

      if (isActive || stillCoolingDown) {
        const response: SiteRefreshTriggerResponse = {
          success: false,
          message:
            "A site refresh already ran recently. Please wait before starting another run.",
          workflowUrl,
          cooldownUntil,
        };

        res.status(409).json(response);
        return;
      }
    }

    const dispatchResponse = await fetch(
      `https://api.github.com/repos/${repository}/actions/workflows/${workflowFile}/dispatches`,
      {
        method: "POST",
        headers: getGitHubHeaders(githubToken),
        body: JSON.stringify({
          ref: workflowRef,
          inputs: {
            trigger_source: body.source || "homepage",
            trigger_note: "Manual site refresh requested from the home page.",
          },
        }),
      },
    );

    if (!dispatchResponse.ok) {
      throw new Error(
        `Failed to dispatch workflow (${dispatchResponse.status})`,
      );
    }

    const response: SiteRefreshTriggerResponse = {
      success: true,
      message:
        "Site refresh queued. The workflow will update the latest-info sections and deploy once the commit lands.",
      workflowUrl,
      queuedAt: new Date().toISOString(),
    };

    res.status(202).json(response);
  } catch (error) {
    console.error("Error triggering site refresh workflow:", error);

    const response: SiteRefreshTriggerResponse = {
      success: false,
      message:
        "Could not trigger the site refresh workflow right now. Check the GitHub Actions workflow configuration and token.",
      workflowUrl,
    };

    res.status(500).json(response);
  }
};
