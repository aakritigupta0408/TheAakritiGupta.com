import express from "express";
import cors from "cors";
import { handleChat } from "./routes/chat";
import { handleDemo } from "./routes/demo";
import { handleSaveEmail } from "./routes/save-email";
import { handleSiteRefreshTrigger } from "./routes/site-refresh-trigger";

export function createServer() {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Example API routes
  app.get("/api/ping", (_req, res) => {
    res.json({ message: "Hello from Express server v2!" });
  });

  app.get("/api/demo", handleDemo);
  app.post("/api/chat", handleChat);
  app.post("/api/save-email", handleSaveEmail);
  app.post("/api/site-refresh/run", handleSiteRefreshTrigger);

  return app;
}
