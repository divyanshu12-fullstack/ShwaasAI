"""Deployment controls around the existing API routes."""

import pytest
from fastapi.testclient import TestClient

from backend.app.core import deployment
from backend.app.core.config import ConfigurationError
from backend.app.core.deployment import DeploymentSettings
from backend.app.core.security import AuthenticatedUser, get_current_user
from backend.main import create_app
from uuid import UUID


def test_rate_limit_settings_require_positive_values(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    with pytest.raises(ConfigurationError):
        DeploymentSettings.from_env()


def test_cors_middleware_is_not_installed():
    app = create_app(DeploymentSettings())
    headers = {
        "Origin": "https://app.example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }
    with TestClient(app) as client:
        response = client.options("/api/v1/sessions", headers=headers)
    assert response.status_code == 405
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers


def test_global_and_analysis_rate_limits_return_retry_after():
    app = create_app(DeploymentSettings(rate_limit_per_minute=3, analyze_rate_limit_per_minute=1))
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id=UUID("72ff8e22-31c2-4a06-a287-dcccb706bc29"), access_token="test-token"
    )
    with TestClient(app) as client:
        first = client.post("/api/v1/analyze/audio", headers={"Authorization": "Bearer token-a"}, json={})
        second = client.post("/api/v1/analyze/audio", headers={"Authorization": "Bearer token-a"}, json={})
        different_token = client.post("/api/v1/analyze/audio", headers={"Authorization": "Bearer token-b"}, json={})
        third = client.get("/api/v1/info")
        global_limit = client.get("/api/v1/info")
    assert first.status_code == 422
    assert second.status_code == 429
    assert 1 <= int(second.headers["retry-after"]) <= 60
    assert second.json()["request_id"] == second.headers["x-request-id"]
    assert different_token.status_code == 422
    assert third.status_code == 200
    assert global_limit.status_code == 429


def test_unhandled_failure_returns_safe_error_and_request_id(monkeypatch):
    events = []

    class Logger:
        def info(self, event):
            events.append(event)

    monkeypatch.setattr(deployment, "_request_logger", lambda: Logger())
    app = create_app(DeploymentSettings())

    @app.get("/boom")
    def boom():
        raise RuntimeError("private-detail")

    with TestClient(app) as client:
        result = client.get("/boom", headers={"Authorization": "Bearer hidden-token"})
    assert result.status_code == 500
    assert result.json()["detail"] == "Internal server error"
    assert result.json()["request_id"] == result.headers["x-request-id"]
    assert "private-detail" not in result.text
    logged = "\n".join(events)
    assert '"status":500' in logged
    assert '"error_type":"RuntimeError"' in logged
    assert "hidden-token" not in logged
    assert "private-detail" not in logged
