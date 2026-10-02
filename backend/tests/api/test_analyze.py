"""End-to-end analyze API tests with the real waveform pipeline and mocked Supabase."""

import asyncio
import base64
import io
import json
import httpx
import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from backend.app.core.config import get_settings
from backend.main import app


USER_A = "72ff8e22-31c2-4a06-a287-dcccb706bc29"
USER_B = "ed3cce37-1c91-4337-b2a7-9eb59a522532"
SESSION_A = "45172db7-39a9-46c0-b870-87e699994ad2"


class TestSettings:
    supabase_url = "https://example.supabase.co"
    publishable_key = "test-publishable-key"
    secret_key = "test-secret-key"


def wav_bytes(silent=False):
    sample_rate = 16000
    samples = np.zeros(sample_rate * 2, dtype=np.int16) if silent else (
        10000 * np.sin(2 * np.pi * 350 * np.arange(sample_rate * 2) / sample_rate)
    ).astype(np.int16)
    output = io.BytesIO()
    wavfile.write(output, sample_rate, samples)
    return output.getvalue()


@pytest.fixture
def api(monkeypatch):
    real_client = httpx.AsyncClient
    state = {"status": "pending", "input_type": "cough", "cough_type": "forced", "embedding": None, "writes": [], "race": False}

    def handler(request):
        path = request.url.path
        auth = request.headers.get("authorization")
        if path == "/auth/v1/user":
            user_id = {"Bearer token-a": USER_A, "Bearer token-b": USER_B}.get(auth)
            return httpx.Response(200 if user_id else 401, json={"id": user_id} if user_id else {})
        params = request.url.params
        if path == "/rest/v1/sessions" and request.method == "GET":
            assert request.headers["apikey"] == TestSettings.publishable_key
            assert params["user_id"] in (f"eq.{USER_A}", f"eq.{USER_B}")
            if auth != "Bearer token-a":
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[{
                "session_id": SESSION_A, "user_id": USER_A,
                "status": state["status"], "input_type": state["input_type"],
                "cough_type": state["cough_type"],
            }])
        assert request.headers["apikey"] == TestSettings.secret_key
        assert auth == "Bearer test-secret-key"
        state["writes"].append((request.method, path))
        if path == "/rest/v1/rpc/complete_screening_result" and request.method == "POST":
            body = json.loads(request.content)
            assert body["p_session_id"] == SESSION_A
            assert body["p_user_id"] == USER_A
            assert len(body["p_embedding_data"]) == 512
            assert body["p_model_used"] and 0 <= body["p_risk_score"] <= 100
            if state["status"] != "pending" or state["race"]:
                return httpx.Response(200, content=b"null", headers={"content-type": "application/json"})
            state["embedding"] = body
            state["status"] = "completed"
            return httpx.Response(200, json={
                "session_id": SESSION_A, "user_id": USER_A, "status": "completed",
                "risk_score": body["p_risk_score"], "level": body["p_level"],
            })
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: real_client(transport=httpx.MockTransport(handler)))
    app.dependency_overrides[get_settings] = lambda: TestSettings()

    def send(method, path, **kwargs):
        async def call():
            async with real_client(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(call())

    yield send, state
    app.dependency_overrides.clear()


def test_status(api):
    send, _ = api
    response = send("GET", "/api/v1/analyze/status")
    assert response.status_code == 200
    assert response.json()["target_sample_rate"] == 16000
    assert response.json()["accepted_audio_format"] == "WAV"


def test_upload_runs_waveform_model_and_saves_result(api):
    send, state = api
    response = send(
        "POST", "/api/v1/analyze",
        headers={"Authorization": "Bearer token-a"},
        data={"session_id": SESSION_A, "symptoms": json.dumps({"age": 38, "has_fever": True})},
        files={"file": ("cough.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == SESSION_A
    assert body["patient_id"] is None
    assert body["multimodal_risk_score"] is not None
    assert body["total_windows_analyzed"] >= 1
    assert state["status"] == "completed"
    assert state["embedding"] is not None
    assert state["writes"] == [("POST", "/rest/v1/rpc/complete_screening_result")]
    assert send("POST", "/api/v1/analyze", headers={"Authorization": "Bearer token-a"},
                data={"session_id": SESSION_A}, files={"file": ("cough.wav", wav_bytes(), "audio/wav")}).status_code == 409


def test_base64_uses_saved_cough_type_and_requires_owner(api, monkeypatch):
    send, state = api
    from backend.app.api.routers import analyze
    seen = []
    original = analyze.classifier_service.predict_tb_risk

    def spy(embedding, cough_type="both"):
        seen.append(cough_type)
        return original(embedding, cough_type)

    monkeypatch.setattr(analyze.classifier_service, "predict_tb_risk", spy)
    payload = {"session_id": SESSION_A, "audio_base64": base64.b64encode(wav_bytes()).decode()}
    assert send("POST", "/api/v1/analyze/audio", json=payload).status_code == 401
    assert send("POST", "/api/v1/analyze/audio", headers={"Authorization": "Bearer token-b"}, json=payload).status_code == 404
    assert state["writes"] == []
    assert send("POST", "/api/v1/analyze/audio", headers={"Authorization": "Bearer token-a"},
                json={**payload, "userId": USER_B}).status_code == 422
    response = send("POST", "/api/v1/analyze/audio", headers={"Authorization": "Bearer token-a"}, json=payload)
    assert response.status_code == 200, response.text
    assert seen and set(seen) == {"forced"}


def test_invalid_audio_and_breathing_do_not_write(api):
    send, state = api
    headers = {"Authorization": "Bearer token-a"}
    path = "/api/v1/analyze"
    assert send("POST", path, headers=headers, data={"session_id": SESSION_A},
                files={"file": ("x.mp3", b"not-a-wav", "audio/mpeg")}).status_code == 415
    assert send("POST", path, headers=headers, data={"session_id": SESSION_A},
                files={"file": ("silent.wav", wav_bytes(silent=True), "audio/wav")}).status_code == 422
    assert send("POST", "/api/v1/analyze/audio", headers=headers,
                json={"session_id": SESSION_A, "audio_base64": "%%%"}).status_code == 400
    state["input_type"] = "breathing"
    assert send("POST", path, headers=headers, data={"session_id": SESSION_A},
                files={"file": ("breathing.wav", wav_bytes(), "audio/wav")}).status_code == 422
    assert state["writes"] == []


def test_completion_race_returns_conflict_without_storing_result(api):
    send, state = api
    state["race"] = True
    response = send("POST", "/api/v1/analyze", headers={"Authorization": "Bearer token-a"},
                    data={"session_id": SESSION_A}, files={"file": ("cough.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 409
    assert state["status"] == "pending"
    assert state["embedding"] is None


def test_model_failure_returns_503_and_keeps_session_pending(api, monkeypatch):
    send, state = api
    from backend.app.api.routers import analyze
    from backend.app.services.classifier import ModelInferenceError

    def fail_prediction(*args, **kwargs):
        raise ModelInferenceError("simulated broken model")

    monkeypatch.setattr(analyze.classifier_service, "predict_tb_risk", fail_prediction)
    response = send("POST", "/api/v1/analyze", headers={"Authorization": "Bearer token-a"},
                    data={"session_id": SESSION_A}, files={"file": ("cough.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 503
    assert response.json()["detail"] == "Screening model unavailable"
    assert state["status"] == "pending"
    assert state["writes"] == []
