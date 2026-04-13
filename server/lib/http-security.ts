import type { CorsOptions } from "cors";

const DEFAULT_ALLOWED_ORIGINS = [
  "https://www.theaakritigupta.com",
  "https://theaakritigupta.com",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
  "http://localhost:8080",
  "http://127.0.0.1:8080",
];

export const REQUEST_BODY_LIMIT = "256kb";

function normalizeOrigin(origin: string) {
  return origin.trim().replace(/\/+$/, "");
}

function getConfiguredOrigins() {
  const configured = (process.env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean)
    .map(normalizeOrigin);

  return new Set([...DEFAULT_ALLOWED_ORIGINS, ...configured]);
}

export function isAllowedOrigin(origin?: string | null) {
  if (!origin) {
    return true;
  }

  return getConfiguredOrigins().has(normalizeOrigin(origin));
}

export function createCorsOptions(): CorsOptions {
  return {
    origin(origin, callback) {
      callback(null, isAllowedOrigin(origin));
    },
    methods: ["GET", "POST", "OPTIONS"],
  };
}
