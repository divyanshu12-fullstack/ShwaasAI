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

## 2026-10-02 — Session foundation

- Added `sql/sessions.sql` for `public.sessions`: UUID primary key, foreign key to the Supabase Auth user, input/cough types, status, nullable model-result fields, recording and server timestamps, checks, and an index for each user's newest-first history. The table is deliberately separate from `profiles` so future metadata and embeddings can reference `session_id`.
- Restricted the `authenticated` role to SELECT, DELETE, and column-limited INSERT of only `user_id`, `input_type`, `cough_type`, and `recorded_at`. RLS allows those operations only where `auth.uid() = user_id`. No client UPDATE grant or policy exists, so clients cannot mark a session complete or write a risk result. The future analysis backend must write those result fields internally after producing them.
- Added four `/api/v1/sessions` routes and `SessionService`. Each route requires the existing Supabase token dependency; the service sends the caller's token to the Data API and filters by the verified UUID. The create schema rejects client-supplied `user_id`, status, and model output. Missing or another user's session returns the same 404 response.
- Added limit/offset pagination with deterministic `created_at`, `session_id` descending order. Creation returns a pending session (201); deletion returns 204.
- Added focused route tests for ownership, invalid fields, pagination, missing/invalid auth, database outages, and delete behavior. The auth and session suites together pass: **18 passed**.
- A read-only live probe returned `PGRST205`/HTTP 404 for `sessions`, confirming this table has not yet been installed in the configured Supabase project. The SQL must be run in the Supabase SQL Editor before live session verification. No database credentials are present for direct SQL execution, and no dashboard tab was available in this task.

## 2026-10-02 — Live session verification after schema installation

- After `sql/sessions.sql` was run in the Supabase SQL Editor, a read-only anonymous Data API probe returned HTTP 401 / PostgreSQL `42501` instead of the prior missing-table `PGRST205`. This confirms the table is exposed while anonymous SELECT remains denied.
- Created two temporary Supabase Auth users and signed each in through the normal email/password token endpoint. Called all four FastAPI session routes in-process while their Auth and Data API requests reached the live Supabase project.
- Verified POST creates a `pending` row owned by the verified JWT user, with a server-generated session UUID, timestamps, and no model result. Verified each user's GET list and GET by ID return only their own row; the other user's ID returns 404.
- Verified the create schema rejects a supplied `user_id` or `risk_score` with 422. Direct Data API attempts to select or delete another user's row returned no rows; attempts to insert with another owner or write completed/model-result fields were denied by grants/RLS.
- Verified an unauthorized DELETE returns 404, an owner's DELETE returns 204, the deleted row then returns 404, and the other user's row remains intact. Missing and invalid Bearer tokens returned 401.
- Deleted both temporary test users; Supabase returned 200 for each deletion. No credentials or tokens were written into project files or printed.
- Re-ran the complete local test suite after live verification: **18 passed**. No backend changes were required. No manual step remains for the sessions foundation; result completion waits for the later analysis backend.

## 2026-10-02 — Sessions SQL rerun fix

- A second execution of the original `sql/sessions.sql` stopped at `ERROR 42P07: relation "sessions" already exists`. This is an installation-script idempotency issue, not a failure of the existing sessions table; the live API and RLS checks above had already succeeded.
- Updated `sql/sessions.sql` to preserve an existing table/index, create ownership policies and the timestamp trigger only when absent, and replace the timestamp function safely. Grants and RLS configuration are reapplied on rerun. This makes repeat execution safe for the same schema; changing an older/different table still needs a proper migration.
- Confirmed the live `sessions` endpoint remains present (anonymous requests return 401 / `42501`, consistent with denied anonymous SELECT). Re-ran the complete local suite: **18 passed**. The updated rerun path has not been executed in the SQL Editor from this environment.

## 2026-10-02 — History filters and screening metadata

- Extended `GET /api/v1/sessions` with optional `input_type=cough|breathing` and `sort=newest|oldest`, alongside the existing `limit` and `offset`. Filtering and ordering happen in Supabase before pagination; both sort directions use `session_id` as a stable tie breaker. Requests with unsupported values receive 422.
- Added `sql/session_metadata.sql` for a one-to-one `public.session_metadata` table keyed by `session_id` with a foreign key to `sessions`. It stores age, sex, fever, smoker, cough duration, night sweats, weight loss, and timestamps. The foreign key cascades deletion from the parent session. Age and text lengths have database checks.
- Added SELECT/INSERT/UPDATE RLS policies that require the parent session to belong to `auth.uid()`. Anonymous access and direct reassignment of `session_id` are not granted. Metadata is optional; unknown answers remain `NULL` instead of being confused with `false`.
- Added `GET` and `PUT /api/v1/sessions/{session_id}/metadata`. Both verify the session belongs to the JWT user before accessing metadata; Supabase RLS checks the parent again. `PUT` replaces all metadata fields and uses separate insert/update calls to avoid silently resetting server timestamps during an upsert. A concurrent initial insert conflict retries as an update.
- Added validation and route tests for filtering, both sort directions, metadata creation/replacement, cross-user denial, missing/invalid authorization, invalid values, and database outages. The full local suite passes: **22 passed**.
- At this stage, the metadata SQL needed installation in the Supabase SQL Editor before live metadata checks. No direct SQL connection or dashboard session was available from this task; installation and live verification are recorded below.
- Verified history against the live project using two temporary signed-in users and three sessions: `limit`, `offset`, `input_type=cough|breathing`, and both `sort` directions returned the expected rows. Rechecked owner isolation; both temporary users were deleted (HTTP 200 each).

## 2026-10-02 — Live screening metadata verification

- After `sql/session_metadata.sql` was installed in the Supabase SQL Editor, the Data API recognized `session_metadata`; anonymous SELECT returned HTTP 401 / PostgreSQL `42501`, consistent with its restricted grants.
- Created two temporary confirmed Supabase Auth users, signed them in normally, and called FastAPI in-process while its Auth and Data API requests reached the live project. `GET /api/v1/sessions/{session_id}/metadata` returned 404 before creation; `PUT` created metadata; `GET` read the persisted row; a second `PUT` replaced its fields, cleared omitted values, and preserved `created_at`.
- Verified FastAPI returned 404 for another user's GET and PUT, 401 for missing/invalid JWTs, and 422 for invalid age or a client-supplied `user_id`.
- Queried Supabase's Data API directly with the second user's JWT. RLS returned no rows for cross-user SELECT and UPDATE and denied cross-user INSERT; the owner's metadata remained unchanged.
- Created metadata for the second user's session, deleted that parent session, and verified through a backend-only Data API read that the metadata row was removed by the foreign-key cascade. Both temporary users were deleted through Auth Admin (HTTP 200 each). No test tokens or credentials were printed or saved in the project.
- Re-ran the complete local suite: **22 passed**. No backend code or SQL changes were needed after live verification. No manual step remains for the metadata routes.
- The configured publishable and secret API keys support Auth and Data API calls, not arbitrary schema SQL. This environment has no PostgreSQL connection string, Supabase Management API access token, or authenticated SQL Editor session; that is why the schema installation required a manual SQL Editor step. It does not indicate a backend connection or `.env` error.

## 2026-10-02 — Direct database connection and public SQL review

- Confirmed `DIRECT_URL` is present in the ignored backend `.env` and has a PostgreSQL URL with the expected `db.<project>.supabase.co` host and port 5432. A read-only PostgreSQL connection attempt did not reach authentication: Windows DNS returned `11001` for the database hostname. The project's Supabase HTTPS API hostname did resolve over IPv4. Supabase documents that direct database connections require IPv6 unless the project has its IPv4 add-on; the IPv4 session pooler is the practical alternative here. Database SQL execution remains unverified from this environment.
- Checked the three files in `backend/sql/` for credentials and connection strings; they contain schema, grants, RLS policies, and triggers, with no secrets. These SQL files are appropriate for public version control so the schema can be reviewed and reproduced. `backend/.env` is ignored, is not tracked, and has no commit in the local Git history. `backend/.env.example` contains only empty key placeholders, including `DIRECT_URL=`.

## 2026-10-02 — Session pooler connection verified

- The user replaced the value of `DIRECT_URL` with a Supabase session pooler URL on port 5432. A read-only PostgreSQL connection succeeded over IPv4 and authenticated to the configured project.
- Queried the PostgreSQL catalog directly: `public.profiles`, `public.sessions`, and `public.session_metadata` all exist and have RLS enabled. Read-only privilege checks confirmed the connected role can create objects in `public` and create schemas in the database. The connection string and credentials were not printed.
- The pooler now provides database access for future schema changes from this environment; no further SQL Editor step is needed for the current three tables. The environment variable name `DIRECT_URL` now holds a session pooler URL, so it is a label rather than the connection mode.

## 2026-10-02 — Pooler variable renamed

- The user renamed the local `.env` variable from `DIRECT_URL` to `SESSION_POOLER_URL`. Added a configuration comment, `.env.example` comment, and README guidance that this server-side connection is for SQL administration/migrations; request handlers continue to use the caller's JWT through the Supabase Data API. The runtime does not need to read the pooler URL for existing routes.
- Repeated the read-only PostgreSQL check using `SESSION_POOLER_URL`: connection succeeded over IPv4, all three application tables were present with RLS enabled, and the connected role retained schema creation privileges. No schema or data was changed.

## 2026-10-02 — Remove unused scaffold placeholders

- Removed 29 files that were empty, except for the two-byte whitespace-only `Notes/index.txt`, and then removed seven directories left empty by that cleanup. These were unimplemented admin/analyze/health/hear/model/user routers, database and ML placeholders, unused request/response and service modules, empty scripts and tests, and the empty Notes folder. No working module imported them.
- Kept the active auth, session, metadata, SQL, configuration, and test files; retained package `__init__.py` markers, project licensing, and the Supabase agent skills. This leaves the backend focused on the routes that actually exist while future features can add their files when implemented.
- Re-ran the full local test suite after the removals: **22 passed**.
