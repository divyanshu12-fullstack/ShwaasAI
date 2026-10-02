"""Public health, API capabilities, and generated documentation routes."""

import httpx
from fastapi.testclient import TestClient

from backend.app.api.routers import system
from backend.app.core.config import get_settings
from backend.main import app


class TestSettings:
    supabase_url = "https://example.supabase.co"
    secret_key = "server-only-test-key"


def test_health_reports_real_dependencies_and_no_rows(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.path == "/rest/v1/sessions"
        assert request.url.params["select"] == "session_id"
        assert request.url.params["limit"] == "0"
        assert request.headers["apikey"] == TestSettings.secret_key
        assert request.headers["authorization"] == f"Bearer {TestSettings.secret_key}"
        return httpx.Response(200, json=[])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(system.httpx, "AsyncClient", lambda *args, **kwargs: real_client(transport=httpx.MockTransport(handler)))
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    try:
        with TestClient(app) as client:
            result = client.get("/api/v1/health")
        assert result.status_code == 200
        assert result.headers["cache-control"] == "no-store"
        body = result.json()
        assert body["db_connected"] is True
        assert body["classifier_loaded"] is True
        assert body["feature_extractor_ready"] is True
        assert body["hear_model_loaded"] is False
        assert body["status"] == "degraded"
        assert body["uptime_seconds"] >= 0
        assert len(calls) == 1
    finally:
        app.dependency_overrides.clear()


def test_health_fails_closed_when_database_unavailable(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(system.httpx, "AsyncClient", lambda *args, **kwargs: real_client(transport=httpx.MockTransport(lambda _: httpx.Response(503))))
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    try:
        with TestClient(app) as client:
            result = client.get("/api/v1/health")
        assert result.status_code == 503
        assert result.json()["status"] == "unavailable"
        assert result.json()["db_connected"] is False
    finally:
        app.dependency_overrides.clear()


def test_health_is_unavailable_without_cough_heads(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(system.httpx, "AsyncClient", lambda *args, **kwargs: real_client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))))
    monkeypatch.setattr(system, "get_service_status", lambda: {
        "foundation_model_loaded": True,
        "models": {"tb_dual_head": False, "pathology_sound": True, "multimodal_fusion": True},
    })
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    try:
        with TestClient(app) as client:
            result = client.get("/api/v1/health")
        assert result.status_code == 503
        assert result.json()["classifier_loaded"] is False
        assert result.json()["db_connected"] is True
    finally:
        app.dependency_overrides.clear()


def test_health_is_unavailable_without_pathology_model(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(system.httpx, "AsyncClient", lambda *args, **kwargs: real_client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))))
    monkeypatch.setattr(system, "get_service_status", lambda: {
        "foundation_model_loaded": False,
        "models": {"tb_dual_head": True, "pathology_sound": False, "multimodal_fusion": True},
    })
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    try:
        with TestClient(app) as client:
            result = client.get("/api/v1/health")
        assert result.status_code == 503
        assert result.json()["status"] == "unavailable"
    finally:
        app.dependency_overrides.clear()


def test_health_is_unavailable_without_feature_extractor(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(system.httpx, "AsyncClient", lambda *args, **kwargs: real_client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))))
    monkeypatch.setattr(system, "get_service_status", lambda: {
        "feature_extractor_ready": False,
        "foundation_model_loaded": False,
        "models": {"tb_dual_head": True, "pathology_sound": True, "multimodal_fusion": True},
    })
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    try:
        with TestClient(app) as client:
            result = client.get("/api/v1/health")
        assert result.status_code == 503
        assert result.json()["feature_extractor_ready"] is False
    finally:
        app.dependency_overrides.clear()


def test_info_matches_supported_audio_contract_and_docs_work():
    with TestClient(app) as client:
        info = client.get("/api/v1/info")
        swagger = client.get("/docs")
        redoc = client.get("/redoc")
        schema = client.get("/openapi.json")
        docs_alias = client.get("/api/v1/docs", follow_redirects=False)
        redoc_alias = client.get("/api/v1/redoc", follow_redirects=False)
    assert info.status_code == 200
    body = info.json()
    assert body["analysis_input_types"] == ["cough"]
    assert body["session_input_types"] == ["cough", "breathing"]
    assert body["accepted_audio_formats"] == ["WAV"]
    assert body["sample_rate_hz"] == 16000
    assert body["window_duration_seconds"] == 2.0
    assert body["window_hop_seconds"] == 1.0
    assert body["embedding_dimension"] == 512
    assert body["active_extractor"] in {"HeAR", "acoustic_fallback"}
    assert swagger.status_code == redoc.status_code == schema.status_code == 200
    assert "/api/v1/health" in schema.json()["paths"]
    assert "/api/v1/info" in schema.json()["paths"]
    assert docs_alias.status_code == 307 and docs_alias.headers["location"] == "/docs"
    assert redoc_alias.status_code == 307 and redoc_alias.headers["location"] == "/redoc"
