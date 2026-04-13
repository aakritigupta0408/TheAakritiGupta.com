import express from "express";
import cors from "cors";
import type { RequestHandler } from "express";
import { handleChat } from "./routes/chat";
import { handleDemo } from "./routes/demo";
import { handleSaveEmail } from "./routes/save-email";
import { handleSiteRefreshTrigger } from "./routes/site-refresh-trigger";
import { proxyRequest } from "./routes/trade-system";
import { createCorsOptions, isAllowedOrigin, REQUEST_BODY_LIMIT } from "./lib/http-security";
import {
  handleResumeAgentBuild,
  handleResumeAgentChat,
  handleResumeAgentFetch,
} from "./routes/resume-agent";

const rejectDisallowedOrigins: RequestHandler = (req, res, next) => {
  if (!isAllowedOrigin(req.headers.origin)) {
    res.status(403).json({ error: "Origin not allowed." });
    return;
  }

  next();
};

export function createServer() {
  const app = express();
  app.disable("x-powered-by");

  // Middleware
  app.use(rejectDisallowedOrigins);
  app.use(cors(createCorsOptions()));
  app.use(express.json({ limit: REQUEST_BODY_LIMIT }));
  app.use(express.urlencoded({ extended: true, limit: REQUEST_BODY_LIMIT }));

  // Example API routes
  app.get("/api/ping", (_req, res) => {
    res.json({ message: "Hello from Express server v2!" });
  });

  app.get("/api/demo", handleDemo);
  app.post("/api/chat", handleChat);
  app.get("/api/resume-agent/:agentId", handleResumeAgentFetch);
  app.post("/api/resume-agent/build", handleResumeAgentBuild);
  app.post("/api/resume-agent/chat", handleResumeAgentChat);
  app.post("/api/save-email", handleSaveEmail);
  app.post("/api/site-refresh/run", handleSiteRefreshTrigger);
  app.all("/api/trade-system/*", proxyRequest);
  app.all("/api/trade-system", proxyRequest);

  return app;
}
