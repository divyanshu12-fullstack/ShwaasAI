import asyncio

import httpx
import pytest

from backend.app.core.config import get_settings
from backend.main import app


USER_A = "72ff8e22-31c2-4a06-a287-dcccb706bc29"
USER_B = "ed3cce37-1c91-4337-b2a7-9eb59a522532"
SESSION_A = "45172db7-39a9-46c0-b870-87e699994ad2"
TIME = "2026-10-02T00:00:00Z"


class TestSettings:
    supabase_url = "https://example.supabase.co"
    publishable_key = "test-publishable-key"


class SyncASGIClient:
    def __init__(self, async_client_class):
        self.async_client_class = async_client_class

    def request(self, method, path, **kwargs):
        async def send():
            async with self.async_client_class(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())


@pytest.fixture
def setup(monkeypatch):
    real_async_client = httpx.AsyncClient
    requests = []
    state = {"embedding": True, "bad_dimension": False, "db_status": 200}

    def handler(request):
        requests.append(request)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        user_id = {"token-a": USER_A, "token-b": USER_B}.get(token)
        if request.url.path == "/auth/v1/user":
            return httpx.Response(200 if user_id else 401, json={"id": user_id} if user_id else {})
        assert request.headers["apikey"] == "test-publishable-key"
        if state["db_status"] != 200:
            return httpx.Response(state["db_status"], json={"message": "database unavailable"})
        if request.url.path == "/rest/v1/rpc/get_my_session_stats":
            assert request.method == "GET" and not request.url.params
            payload = {
                "total_screenings": 1 if user_id == USER_A else 0,
                "last_screened_at": TIME if user_id == USER_A else None,
                "average_risk_score": 72.0 if user_id == USER_A else None,
                "risk_distribution": {"Low": 0, "Moderate": 0, "High": 1 if user_id == USER_A else 0},
                "input_type_breakdown": {"cough": 1 if user_id == USER_A else 0, "breathing": 0},
                "risk_trend": [{"date": "2026-10-02", "risk_score": 72}] if user_id == USER_A else [],
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/rest/v1/sessions":
            assert request.url.params["user_id"] == f"eq.{user_id}"
            rows = [{"session_id": SESSION_A, "user_id": USER_A}] if user_id == USER_A else []
            return httpx.Response(200, json=rows)
        assert request.url.path == "/rest/v1/embeddings"
        assert request.method == "GET" and request.url.params["session_id"] == f"eq.{SESSION_A}"
        embedding = {
            "session_id": SESSION_A,
            "embedding_data": [0.25, -0.5],
            "embedding_dim": 3 if state["bad_dimension"] else 2,
            "stored_at": TIME,
        }
        return httpx.Response(200, json=[embedding] if state["embedding"] else [])

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_async_client(transport=httpx.MockTransport(handler)))
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    yield SyncASGIClient(real_async_client), requests, state
    app.dependency_overrides.clear()


def auth(token="token-a"):
    return {"Authorization": f"Bearer {token}"}


def test_stats_static_route_uses_jwt_and_handles_empty_history(setup):
    client, requests, _ = setup
    own = client.request("GET", "/api/v1/sessions/stats", headers=auth())
    assert own.status_code == 200
    assert own.json()["total_screenings"] == 1
    assert own.json()["risk_distribution"] == {"Low": 0, "Moderate": 0, "High": 1}
    assert own.json()["risk_trend"] == [{"date": "2026-10-02", "risk_score": 72}]
    assert requests[-1].headers["authorization"] == "Bearer token-a"
    assert client.request("GET", "/api/v1/sessions/stats", headers=auth("token-b")).json()["total_screenings"] == 0
    assert client.request("GET", "/api/v1/sessions/stats").status_code == 401
    assert client.request("GET", "/api/v1/sessions/stats", headers=auth("bad-token")).status_code == 401


def test_embedding_read_requires_session_ownership(setup):
    client, requests, state = setup
    path = f"/api/v1/sessions/{SESSION_A}/embedding"
    own = client.request("GET", path, headers=auth())
    assert own.status_code == 200 and own.json()["embedding_data"] == [0.25, -0.5]
    assert client.request("GET", path, headers=auth("token-b")).status_code == 404
    assert requests[-1].url.path == "/rest/v1/sessions"
    state["embedding"] = False
    assert client.request("GET", path, headers=auth()).status_code == 404
    state["embedding"] = True
    state["bad_dimension"] = True
    assert client.request("GET", path, headers=auth()).status_code == 503


def test_stats_and_embedding_upstream_errors(setup):
    client, _, state = setup
    state["db_status"] = 503
    assert client.request("GET", "/api/v1/sessions/stats", headers=auth()).status_code == 503
    assert client.request("GET", f"/api/v1/sessions/{SESSION_A}/embedding", headers=auth()).status_code == 503
