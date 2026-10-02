# Implementation log

## 2026-10-02 — Authentication foundation

- Inspected the backend scaffold and the two supplied project documents. The pasted auth milestone sets the current scope; the older complete API reference describes later screening and research work. Its device-level `userId` model must be replaced by the verified Supabase Auth UUID when those routes are built.
- Added FastAPI application startup in `main.py` and the three requested `/api/v1/auth` routes. Registration, login, refresh, and logout stay in Supabase Auth; no password or custom token code was added to FastAPI.
- Added `get_current_user` in `app/core/security.py`. It requires a Bearer token and asks the project's Supabase Auth `/auth/v1/user` endpoint to validate it. This documented route handles both asymmetric signing keys and legacy shared-secret tokens. The returned user UUID is the only identity passed to profile operations. Missing/invalid tokens return 401; Auth service outage returns 503.
- Added `ProfileService` using Supabase's Data API with the **caller's access token** and the publishable key. The backend secret key is not used, so database RLS applies. GET and PATCH filter by the verified UUID; the request body only permits `display_name` and rejects extra fields, including a supplied `user_id`.
- Added `sql/profiles.sql` for `public.profiles`, an `auth.users(id)` foreign key, ownership RLS policies, restricted grants, signup trigger, existing-user backfill, and `updated_at` trigger. This must be run in the Supabase SQL Editor before `/me` can read profiles. The table's UUID can be referenced by future screening/session tables.
- Added runtime dependencies to `pyproject.toml` and a pytest development extra.
- Extended the repository's existing `.gitignore` to exclude the live backend `.env`, virtual environments, test caches, and Python bytecode. This prevents local credentials and generated files from being committed with the auth work.

### Connection points

1. Mobile signs up and signs in through Supabase Auth using the **publishable** key; it stores and refreshes the Supabase session.
2. Mobile sends `Authorization: Bearer <access_token>` to FastAPI.
3. FastAPI validates that token with Supabase Auth, extracts the verified UUID, and queries `profiles` through Supabase's Data API. RLS independently limits reads and updates to that UUID.
4. The supplied `SUPABASE_JWKS_URL` is reserved for a future local-verification optimization. Auth `/user` verification works with both current asymmetric keys and older HS256 projects. `SUPABASE_SECRET_KEY` is intentionally unused and must remain backend-only.

### Verification and remaining deployment work

- Local route tests cover missing/invalid tokens, owned profile reads and updates, rejected user-ID manipulation, missing profiles, and upstream outages.
- Local tests passed (`9 passed`). A read-only probe of the configured Supabase project found one JWKS key and returned HTTP 404 for `profiles`, so the profile schema is not yet available through its Data API. Run `sql/profiles.sql` in the project's Supabase SQL Editor, then test with a real user access token. No real-token end-to-end test was possible without a registered test user/session.
- Logout and refresh are Supabase Auth operations in the mobile app. Supabase access tokens are stateless; a previously issued access token can remain valid until its expiry after sign-out. Mobile must clear its local session and stop sending that token. Choose a suitably short JWT lifetime in Supabase settings if immediate-ish revocation matters.

## 2026-10-02 — Live Supabase verification after schema installation

- Re-ran the existing auth suite: **9 passed**.
- Probed `public.profiles` through the Supabase Data API. An anonymous request now returns `401` with PostgreSQL code `42501` (the `anon` role has no SELECT grant), confirming the table is exposed and private.
- Created two temporary Supabase Auth users with confirmed test emails using the backend-only secret key. Both then signed in through the normal email/password token endpoint with the publishable key; no credentials or tokens were printed or stored in project files.
- Called the FastAPI routes in-process through ASGI while their Supabase calls used the live project. `GET /api/v1/auth/test` returned the verified user UUID, `GET /api/v1/auth/me` read the automatically created profile for each user, and `PATCH /api/v1/auth/me` persisted a display-name change that a later GET returned.
- Confirmed missing and malformed Bearer tokens return `401`. FastAPI rejected a PATCH body with a client-supplied `user_id` with `422`. A freshly refreshed Supabase access token also worked with `/auth/test`.
- Confirmed RLS directly through the Data API: user A could neither SELECT nor UPDATE user B's profile; user B's display name remained unchanged.
- Deleted both temporary test users through Supabase Auth Admin; both deletion requests returned `200`.
- No backend code changes were needed after the live checks. No manual step remains for these backend auth routes. Mobile sign-up, session storage, and logout remain separate frontend integration work by the agreed scope.
