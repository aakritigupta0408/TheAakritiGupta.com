# Supabase Resume Agent Setup

This project can persist recruiter-share links from `/resume-builder` directly from the browser, which allows the flow to work on the static live site at `https://www.theaakritigupta.com`.

## 1. Create a free Supabase project

Create a new free project in the Supabase dashboard.

You need:

- Project URL
- Publishable key

## 2. Run the SQL bootstrap

Open the Supabase SQL Editor and run:

- [scripts/supabase/resume-agent.sql](/Users/aakritigupta/TheAakritiGupta.com/scripts/supabase/resume-agent.sql)

This creates:

- a private table for stored recruiter-agent profiles
- a public RPC to create a share link
- a public RPC to fetch a stored profile by link id

## 3. Add build-time variables

Add these repository variables in GitHub:

- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_PUBLISHABLE_KEY`

If you also use Render for the static site, add the same two variables there.

## 4. Redeploy

After the next deploy, `/resume-builder` will:

- build the recruiter agent locally or via `/api` when available
- persist the sanitized recruiter profile to Supabase
- generate a stable recruiter route like `/resume-builder/recruiter/<id>`
- load that stored profile on the recruiter page even when the site is served statically

## Notes

- The publishable key is safe to expose in the client when the database is protected appropriately.
- The recruiter chat remains grounded to user-provided material. Persistence only stores the structured candidate profile used by that grounded chat.
