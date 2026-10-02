# ShwaasAI backend auth

Run `sql/profiles.sql` once in the Supabase SQL Editor. Keep `backend/.env` with `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` set. The secret key is not needed by these routes.

From `backend/`:

```powershell
uv sync --extra dev
uv run uvicorn main:app --reload
uv run pytest
```

The mobile app should sign up, sign in, refresh, and sign out with Supabase Auth. For protected FastAPI calls, send `Authorization: Bearer <access_token>`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/v1/auth/test` | Confirm a Supabase token is valid and return its user UUID |
| GET | `/api/v1/auth/me` | Read the signed-in user's profile |
| PATCH | `/api/v1/auth/me` | Update only the signed-in user's `display_name` |

Example PATCH body: `{"display_name":"Asha"}`. Use `null` to clear the name. Auth and profile service outages return 503; missing profiles return 404.
