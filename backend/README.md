# ShwaasAI backend

Run `sql/profiles.sql`, `sql/sessions.sql`, `sql/session_metadata.sql`, `sql/embeddings.sql`, `sql/session_stats.sql`, and `sql/analysis_completion.sql` in that order through the Supabase SQL Editor or a server-side PostgreSQL session pooler connection before using these routes. The scripts added after profiles are safe to rerun when the tables already have the documented schema. Keep `backend/.env` with `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, and a server-only `SUPABASE_SECRET_KEY` set. `SESSION_POOLER_URL` is for database administration/migrations. Normal FastAPI reads and client-allowed writes use the caller's JWT and RLS. Only the analysis result writer uses the secret key after verifying the JWT and session owner.

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
| GET | `/api/v1/sessions/stats` | Aggregate only the signed-in user's screening history |
| GET | `/api/v1/sessions/{session_id}` | Read one owned session |
| DELETE | `/api/v1/sessions/{session_id}` | Delete one owned session |
| GET | `/api/v1/sessions/{session_id}/metadata` | Read metadata for an owned session |
| PUT | `/api/v1/sessions/{session_id}/metadata` | Create or replace metadata for an owned session |
| GET | `/api/v1/sessions/{session_id}/embedding` | Read a stored embedding for an owned session |
| POST | `/api/v1/analyze` | Analyze a pending cough session from multipart WAV audio and save the result |
| POST | `/api/v1/analyze/audio` | Same analysis from JSON base64 WAV audio |
| GET | `/api/v1/analyze/status` | Report loaded model components and audio format |
| GET | `/api/v1/health` | Check Data API connectivity and model readiness; 503 when unavailable |
| GET | `/api/v1/info` | Report the current mobile integration contract and model input format |
| GET | `/docs`, `/redoc`, `/openapi.json` | Swagger, ReDoc, and OpenAPI documentation |
| GET | `/api/v1/docs`, `/api/v1/redoc` | Redirect to the canonical documentation pages |

Example PATCH body: `{"display_name":"Asha"}`. Use `null` to clear the name. Auth and profile service outages return 503; missing profiles return 404.

Example session creation body:

```json
{"input_type":"cough","cough_type":"passive","recorded_at":"2026-10-02T12:00:00Z"}
```

`cough_type` and `recorded_at` are optional; `cough_type` must be omitted for breathing sessions. The backend derives `user_id` from the verified token, while Supabase creates the session ID, timestamps, and `pending` status. The mobile app must not send `user_id`, status, risk scores, or model output. All session reads and deletes require ownership; an unknown or another user's session returns 404. Delete returns 204 on success.

History is sorted by server `created_at` and then `session_id` for stable pagination. The default sort is `newest`. For example: `/api/v1/sessions?input_type=cough&sort=oldest&limit=20&offset=0`.

Metadata is a separate one-to-one record linked to a session. After creating a session, send a `PUT` body such as:

```json
{"age":28,"sex":"Male","fever":false,"smoker":false,"cough_duration":"< 1 week","night_sweats":false,"weight_loss":false}
```

Every metadata field is optional and nullable. `PUT` replaces the whole metadata record: omitted fields become `null`, so clients should include values they want to keep. A missing metadata record returns 404 on GET. Deleting a session also deletes its metadata.

Statistics use the JWT user, with no `user_id` parameter. `total_screenings`, `last_screened_at`, and the cough/breathing breakdown include pending sessions. `average_risk_score` averages scored sessions only; Low/Moderate/High counts include sessions with a level. `risk_trend` contains one chronologically ordered `{date, risk_score}` point per scored session, using the UTC recording date. A user with no sessions gets zero counts, an empty trend, and `null` for the average and last screening time.

`embeddings` holds one optional aggregated float vector per session as `real[]`, with a database check that `embedding_dim` matches its length. The analysis backend writes the 512-value vector after inference. Mobile clients can read only embeddings for their own sessions and cannot insert, change, or delete vectors. A missing embedding returns 404; deleting its session cascades to the embedding.

### Analyze a recording

Create a `cough` session first, then send its `session_id` and a WAV recording with the same Bearer token. `POST /api/v1/analyze` takes multipart fields `session_id`, `file`, and optional `symptoms` as a JSON string matching `ClinicalSymptoms`. For example:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/analyze -H "Authorization: Bearer <access_token>" -F "session_id=<session_uuid>" -F "file=@cough.wav;type=audio/wav" -F 'symptoms={"age":38,"has_fever":true}'
```

`POST /api/v1/analyze/audio` accepts JSON `{"session_id":"<session_uuid>","audio_base64":"<base64 WAV>","symptoms":{"age":38,"has_fever":true}}`. The mobile app sends recorded WAV bytes, not mel spectrograms or model embeddings. The server decodes, resamples to 16 kHz, creates 2-second overlapping waveform windows, runs the current extractor/classifiers, and stores the completed result plus a pooled 512-value embedding in one database transaction through a service-role-only function. Raw audio is not persisted. Files over 20 MiB, recordings over 60 seconds, silent clips, non-WAV files, and completed sessions are rejected. The current TB classifier has passive/forced cough heads only, so breathing sessions cannot be analyzed for TB risk yet. The response includes `session_id`, the acoustic and optional symptom-fused scores, per-window information, and a triage recommendation. `confidence` in the session row remains null because the model does not expose a calibrated overall TB confidence.

For the Expo session-creation and multipart upload sequence, see [mobile-upload.md](docs/mobile-upload.md). The mobile source is not in this checkout, so its recording/export code must be wired in that app.

`GET /api/v1/analyze/status` reports model availability. The bundled classifier heads use the 512-value `AcousticFeatureExtractor` representation, so that extractor is selected explicitly. `librosa==0.11.0` is a declared dependency for the feature layout used by the packaged scaler; a different 512-value SciPy fallback is not substituted at inference. Equal vector length does not make Google HeAR embeddings compatible with these heads. A future HeAR integration needs matching retrained heads and end-to-end validation before enabling it. Prediction failures now return HTTP 503 and leave the session pending; the backend does not replace failed predictions with heuristic scores.

### Health and API documentation

`GET /api/v1/health` is public. It checks the Supabase `sessions` Data API without fetching rows and reports database connectivity, feature-extractor readiness, loaded model components, API version, and process uptime. It returns `503` with `status: "unavailable"` if the database, `librosa` feature extractor, either cough classifier head, or the pathology model is unavailable. It returns `200` with `status: "degraded"` for the current acoustic-extractor configuration or when the optional fusion head is absent. The configured project currently reports `degraded` with the database and packaged model heads reachable. No credentials or database error text are included in the response.

The packaged heads have no documented real-patient training provenance or independent clinical calibration in this repository. The training script can generate synthetic coughs; its benchmarks must not be treated as clinical accuracy evidence. Scores and Low/Moderate/High labels are research outputs, not validated TB probabilities or medical diagnoses. The `is_cough_detected` and `windows_with_cough` response names are legacy: they reflect an RMS sound activity threshold, not a trained cough detector. The saved session metadata is not automatically supplied to multimodal inference: callers currently send a separate optional `symptoms` JSON object. The mobile recorder must also supply actual WAV bytes; its source is outside this checkout.

`GET /api/v1/info` is public and reports the actual accepted WAV format, cough-only analysis support, 16 kHz sample rate, 2-second windows with a 1-second hop, 512-value embeddings, upload limits, active extractor, and links to the generated API documentation. Swagger is at `/docs`, ReDoc at `/redoc`, and the schema at `/openapi.json`. The `/api/v1/docs` and `/api/v1/redoc` routes redirect to the canonical pages.

### Deployment controls

CORS middleware is temporarily removed. Native Expo requests can still call the API. A browser page on another origin will need CORS support restored or a same-origin proxy before it can call the API.

`RATE_LIMIT_PER_MINUTE` defaults to 120 requests per client IP; `ANALYZE_RATE_LIMIT_PER_MINUTE` defaults to 12 analysis requests per IP and Bearer-token fingerprint. Both use rolling 60-second windows. A limit returns HTTP 429 and `Retry-After`. These counters are in memory per API process; use a shared edge or gateway limit when deploying multiple workers. The app uses the connection's peer IP and does not trust arbitrary `X-Forwarded-For` values; configure trusted proxy forwarding at the server/gateway if one sits in front of FastAPI.

Every response includes `X-Request-ID`. Structured request logs include method, a UUID-redacted path, status, duration, and request ID; they exclude tokens, request bodies, and query strings. Unexpected server errors return a generic 500 body with the request ID. Existing expected 4xx and 503 errors keep their API meanings. Failed analysis leaves its session pending for a retry.
