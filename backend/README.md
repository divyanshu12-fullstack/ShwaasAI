# ShwaasAI backend

Run `sql/profiles.sql`, `sql/sessions.sql`, and `sql/session_metadata.sql` in that order through the Supabase SQL Editor or a server-side PostgreSQL session pooler connection before using these routes. The sessions and metadata scripts are safe to rerun when the tables already have the documented schema. Keep `backend/.env` with `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` set. `SESSION_POOLER_URL` is for database administration/migrations; FastAPI routes use the caller's JWT through the Supabase Data API and do not use this connection string or the secret key.

From the repository root:

```powershell
uv sync --project backend --extra dev
uv run --project backend uvicorn backend.main:app --reload
uv run --project backend pytest backend/tests
```

The mobile app should sign up, sign in, refresh, and sign out with Supabase Auth. For protected FastAPI calls, send `Authorization: Bearer <access_token>`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/v1/auth/test` | Confirm a Supabase token is valid and return its user UUID |
| GET | `/api/v1/auth/me` | Read the signed-in user's profile |
| PATCH | `/api/v1/auth/me` | Update only the signed-in user's `display_name` |
| POST | `/api/v1/sessions` | Create a pending screening for the signed-in user |
| GET | `/api/v1/sessions` | List own sessions; `limit` 1–100, `offset` ≥ 0, optional `input_type=cough\|breathing`, `sort=newest\|oldest` |
| GET | `/api/v1/sessions/{session_id}` | Read one owned session |
| DELETE | `/api/v1/sessions/{session_id}` | Delete one owned session |
| GET | `/api/v1/sessions/{session_id}/metadata` | Read metadata for an owned session |
| PUT | `/api/v1/sessions/{session_id}/metadata` | Create or replace metadata for an owned session |

Example PATCH body: `{"display_name":"Asha"}`. Use `null` to clear the name. Auth and profile service outages return 503; missing profiles return 404.

Example session creation body:

```json
{"input_type":"cough","cough_type":"passive","recorded_at":"2026-10-02T12:00:00Z"}
```

`cough_type` and `recorded_at` are optional; `cough_type` must be omitted for breathing sessions. The backend derives `user_id` from the verified token, while Supabase creates the session ID, timestamps, and `pending` status. The mobile app must not send `user_id`, status, risk scores, or model output. The analysis backend will populate results in a later milestone. All session reads and deletes require ownership; an unknown or another user's session returns 404. Delete returns 204 on success.

History is sorted by server `created_at` and then `session_id` for stable pagination. The default sort is `newest`. For example: `/api/v1/sessions?input_type=cough&sort=oldest&limit=20&offset=0`.

Metadata is a separate one-to-one record linked to a session. After creating a session, send a `PUT` body such as:

```json
{"age":28,"sex":"Male","fever":false,"smoker":false,"cough_duration":"< 1 week","night_sweats":false,"weight_loss":false}
```

Every metadata field is optional and nullable. `PUT` replaces the whole metadata record: omitted fields become `null`, so clients should include values they want to keep. A missing metadata record returns 404 on GET. Deleting a session also deletes its metadata.
