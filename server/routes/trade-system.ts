/**
 * Proxy all /api/trade-system/* requests to the Python FastAPI service.
 * FastAPI runs on TRADE_SYSTEM_PORT (default 8000) via uvicorn.
 *
 * Path rewrite: /api/trade-system/foo → /api/v1/foo
 *               /api/trade-system/health → /health
 */
import type { Request, Response } from "express";
import * as http from "http";

const PYTHON_HOST = "127.0.0.1";
const PYTHON_PORT = parseInt(process.env.TRADE_SYSTEM_PORT ?? "8000", 10);

function proxyRequest(req: Request, res: Response): void {
  // Rewrite path
  let upstreamPath = req.url.replace(/^\/api\/trade-system/, "");
  if (upstreamPath === "" || upstreamPath === "/") {
    upstreamPath = "/health";
  }
  // /health stays /health; everything else maps to /api/v1/...
  if (!upstreamPath.startsWith("/health") && !upstreamPath.startsWith("/api/")) {
    upstreamPath = "/api/v1" + upstreamPath;
  }

  const body = req.method !== "GET" && req.method !== "HEAD" ? JSON.stringify(req.body) : undefined;

  const options: http.RequestOptions = {
    hostname: PYTHON_HOST,
    port: PYTHON_PORT,
    path: upstreamPath,
    method: req.method,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(body ? { "Content-Length": Buffer.byteLength(body) } : {}),
    },
  };

  const upstream = http.request(options, (proxyRes) => {
    res.status(proxyRes.statusCode ?? 502);
    res.setHeader("Content-Type", "application/json");
    proxyRes.pipe(res, { end: true });
  });

  upstream.on("error", (err) => {
    const isRefused =
      (err as NodeJS.ErrnoException).code === "ECONNREFUSED" ||
      (err as NodeJS.ErrnoException).code === "ENOTFOUND";
    res.status(503).json({
      error: "trade_system_unavailable",
      message: isRefused
        ? "Python service is not running. Start with: uvicorn src.api.server:app --port 8000"
        : err.message,
    });
  });

  if (body) {
    upstream.write(body);
  }
  upstream.end();
}

export { proxyRequest };
