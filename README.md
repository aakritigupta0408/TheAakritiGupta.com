# TheAakritiGupta.com

Production portfolio site and AI showcase built as a React SPA with an integrated Express API.

## Stack

- Frontend: React 18, React Router 6, TypeScript, Vite, TailwindCSS
- Backend: Express
- Testing: Vitest, Testing Library, Happy DOM
- Shared types and logic: `shared/`

## Repo layout

- [client/](/Users/aakritigupta/TheAakritiGupta.com/client): SPA routes, UI components, client-side API wrappers
- [server/](/Users/aakritigupta/TheAakritiGupta.com/server): API handlers and server bootstrap
- [shared/](/Users/aakritigupta/TheAakritiGupta.com/shared): pure logic and shared request/response types
- [scripts/](/Users/aakritigupta/TheAakritiGupta.com/scripts): scheduled site refresh agent and support scripts
- [python-service/](/Users/aakritigupta/TheAakritiGupta.com/python-service): optional companion service for the trade-system demo

## Main user flows

### 1. Marketing and portfolio pages

- The home page lives in [Index.tsx](/Users/aakritigupta/TheAakritiGupta.com/client/pages/Index.tsx)
- Level-one AI pages share the shell in [SubpageLayout.tsx](/Users/aakritigupta/TheAakritiGupta.com/client/components/SubpageLayout.tsx)
- Routes are defined in [App.tsx](/Users/aakritigupta/TheAakritiGupta.com/client/App.tsx)

This site is deployed as a static GitHub Pages SPA. Internal app links therefore use hash routing, for example `/#/ai-playground`, to avoid deep-link failures on static hosting.

## Deployment

- **Production**: GitHub Pages. Every push to `main` runs
  [.github/workflows/deploy-pages.yml](.github/workflows/deploy-pages.yml),
  which builds `dist/spa` and publishes it to the `gh-pages` branch. The
  canonical host is **https://www.theaakritigupta.com** (see `CNAME`).
  `public/404.html` redirects unknown paths to `/#/<path>` for the
  HashRouter — the workflow must not overwrite it.
- **Bare apex (`theaakritigupta.com`)**: currently broken — its DNS points at
  an old AWS load balancer that returns 405. Fix at the domain registrar:
  set the apex **A records to GitHub Pages** (185.199.108.153,
  185.199.109.153, 185.199.110.153, 185.199.111.153), keep `www` as a CNAME
  to `aakritigupta0408.github.io`, then enable "Enforce HTTPS" in the repo's
  Pages settings. No repo change can fix this; it is registrar-side.
- **Netlify**: `netlify.toml` is vestigial for production (kept for local
  reference and the express-function wiring); the live site is Pages.
- **Render**: `render.yaml` deploys only the Python vedic-astro-ai engine
  that the playground page embeds; the SPA itself never deploys to Render.
- **Data pushes**: `public/btc-oracle/` receives hourly data commits from a
  local cron. Note each such push also triggers a full Pages rebuild.

### 2. Resume builder and recruiter agent

- Builder UI: [ResumeBuilder.tsx](/Users/aakritigupta/TheAakritiGupta.com/client/pages/ResumeBuilder.tsx)
- Recruiter chat page: [RecruiterResumeAgent.tsx](/Users/aakritigupta/TheAakritiGupta.com/client/pages/RecruiterResumeAgent.tsx)
- Client API wrapper: [resume-agent.ts](/Users/aakritigupta/TheAakritiGupta.com/client/api/resume-agent.ts)
- Server endpoints: [resume-agent.ts](/Users/aakritigupta/TheAakritiGupta.com/server/routes/resume-agent.ts)
- Shared grounding logic: [resume-agent.ts](/Users/aakritigupta/TheAakritiGupta.com/shared/resume-agent.ts)

The recruiter agent is intentionally grounded-only. It should answer from the stored candidate profile and refuse missing facts.

### 3. Site refresh agent

- Scheduled content updater: [refresh-agent.ts](/Users/aakritigupta/TheAakritiGupta.com/scripts/site-refresh/refresh-agent.ts)
- Workflow: [.github/workflows/site-refresh.yml](/Users/aakritigupta/TheAakritiGupta.com/.github/workflows/site-refresh.yml)
- Trigger API: [site-refresh-trigger.ts](/Users/aakritigupta/TheAakritiGupta.com/server/routes/site-refresh-trigger.ts)

### 4. Optional trade-system demo

- Demo UI: [TradeRecommendationSystemDemo.tsx](/Users/aakritigupta/TheAakritiGupta.com/client/pages/TradeRecommendationSystemDemo.tsx)
- Express proxy: [trade-system.ts](/Users/aakritigupta/TheAakritiGupta.com/server/routes/trade-system.ts)
- Python companion service: [python-service/](/Users/aakritigupta/TheAakritiGupta.com/python-service)

This flow is optional. The portfolio builds without the Python service, but the trade-system demo only works fully when that service is running.

## Local development

Install dependencies:

```bash
npm ci
```

Start the app:

```bash
npm run dev
```

Run checks:

```bash
npm test
npm run typecheck
npm run build
npm run audit:content
```

## Environment variables

Copy [.env.example](/Users/aakritigupta/TheAakritiGupta.com/.env.example) and set only the values needed for your target flow.

High-signal variables:

- `OPENAI_API_KEY`: enables `/api/chat` and server-side recruiter-agent generation/chat
- `OPENAI_CHAT_MODEL`: default model for the site chatbot
- `OPENAI_RESUME_AGENT_MODEL`: recruiter-agent-specific model override
- `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY`: client-side recruiter-link persistence
- `VITE_GA_MEASUREMENT_ID`: optional Google Analytics ID for static-site experiment metrics
- `RESUME_AGENT_STORE_DIR`: server-side fallback storage for recruiter profiles
- `HF_TOKEN`: weekly site refresh agent
- `GITHUB_WORKFLOW_TOKEN`: manual trigger for the weekly refresh workflow
- `ALLOWED_ORIGINS`: comma-separated API origin allowlist
- `TRADE_SYSTEM_PORT`: port for the optional Python trade-service proxy

## Engineering rules for contributors

- Keep page data in dedicated data files when possible. Avoid giant inline arrays in route components.
- Prefer shared pure logic in `shared/` for anything used by both client and server.
- Treat recruiter-agent facts as high-integrity data. Do not add “smart” enrichment that invents information.
- Use primary sources for evergreen AI content when possible: official product docs/blogs, arXiv, Papers with Code, Nature, PubMed, or first-party company documentation.
- Run `npm run audit:content` after editing curated resource pages. It flags deprecated docs hosts, placeholder URLs, and lower-signal source types that should be reviewed.
- Use `?exp-resume-builder-layout=compact` or `?exp-resume-builder-layout=guided` on the hash-routed URL to force a resume-builder variant during QA.
- When changing page structure, update [client/pages/page-interactions.spec.tsx](/Users/aakritigupta/TheAakritiGupta.com/client/pages/page-interactions.spec.tsx) so the interaction suite tracks shipped behavior.
- If you touch deployment or background jobs, test with `npm run build` before pushing.

## Known operational constraints

- The site can run as a static SPA, but some flows degrade to client-only fallbacks when the Express API is unavailable.
- Because the site is static on GitHub Pages, shareable internal URLs should use the hash-routed form generated by the app, not clean BrowserRouter paths.
- The recruiter-agent token link contains embedded profile data when persistent storage is unavailable. Prefer persistent storage for production sharing.
- The trade-system demo depends on a separate Python backend and should be treated as an integration, not a standalone frontend page.
