# Expo recording upload contract

The mobile app must send **actual WAV file bytes** to the backend. A `.m4a`, `.aac`, or WebM recording renamed to `.wav` is still compressed audio and receives HTTP 415. Expo's common high-quality recording preset produces `.m4a`; use a recording path that emits PCM WAV or convert the recording to WAV before upload. The backend accepts WAV at any decodable sample rate and resamples it to 16 kHz mono. Prefer 16-bit PCM, mono, 16 kHz at capture/export time to keep uploads small. The recording must be non-silent, at most 60 seconds, and at most 20 MiB.

The flow uses the Supabase **user access token**, never the secret key:

1. Create `POST /api/v1/sessions` with `{ "input_type": "cough", "cough_type": "passive" }` or `"forced"` and `Authorization: Bearer <access_token>`.
2. Read the returned `session_id`. Do not send a `userId`.
3. Upload the WAV file to `POST /api/v1/analyze` as multipart fields `session_id`, `file`, and optional `symptoms` JSON. Keep the same Bearer token. Do not set multipart `Content-Type` manually; `fetch` must supply its boundary.
4. Use the returned screening result. The session and its embedding are saved by the backend. A failed inference leaves the session pending for retry.

Example React Native request when a valid WAV `file://` URI is available:

```ts
type AnalyzeOptions = {
  apiBaseUrl: string;
  accessToken: string;
  wavUri: string;
  coughType: 'passive' | 'forced';
  symptoms?: Record<string, unknown>;
};

export async function analyzeWavRecording(options: AnalyzeOptions) {
  const base = options.apiBaseUrl.replace(/\/$/, '');
  const headers = { Authorization: `Bearer ${options.accessToken}` };
  const created = await fetch(`${base}/api/v1/sessions`, {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({ input_type: 'cough', cough_type: options.coughType }),
  });
  if (!created.ok) throw new Error(`Session creation failed (${created.status})`);
  const session = await created.json();

  const form = new FormData();
  form.append('session_id', session.session_id);
  form.append('file', {
    uri: options.wavUri,
    name: 'cough.wav',
    type: 'audio/wav',
  } as unknown as Blob);
  if (options.symptoms) form.append('symptoms', JSON.stringify(options.symptoms));

  const response = await fetch(`${base}/api/v1/analyze`, {
    method: 'POST',
    headers,
    body: form,
  });
  if (!response.ok) {
    const retryAfter = response.headers.get('Retry-After');
    throw new Error(`Analysis failed (${response.status})${retryAfter ? `; retry after ${retryAfter}s` : ''}`);
  }
  return response.json();
}
```

Handle 401 by refreshing the Supabase session, 404 by creating a new screening session, 409 by reading the already completed session, 413/415/422 by correcting the recording, 429 by honoring `Retry-After`, and 503 by retrying later. The frontend should not create a new session for a transient 503 when the current session is still pending.

The Expo app source is not in this backend checkout. The recording code must be checked against its installed Expo SDK before selecting a WAV capture/export API. This example handles the authenticated upload once a real WAV URI exists.
