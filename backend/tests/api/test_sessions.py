import asyncio
import json
from uuid import UUID

import httpx
import pytest

from app.core.config import get_settings
from main import app


USER_A = "72ff8e22-31c2-4a06-a287-dcccb706bc29"
USER_B = "ed3cce37-1c91-4337-b2a7-9eb59a522532"
SESSION_A = "45172db7-39a9-46c0-b870-87e699994ad2"
SESSION_B = "ef065569-718c-405c-9954-ef9b06afcfa6"
SESSION_A_BREATHING = "56b53746-f741-4f2e-98c5-bb1fa79286dc"
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


def row(session_id, user_id, input_type="cough"):
    return {
        "session_id": session_id,
        "user_id": user_id,
        "input_type": input_type,
        "cough_type": "passive" if input_type == "cough" else None,
        "status": "pending",
        "risk_score": None,
        "level": None,
        "confidence": None,
        "recommendation": None,
        "model_used": None,
        "recorded_at": TIME,
        "created_at": TIME,
        "updated_at": TIME,
    }


@pytest.fixture
def setup(monkeypatch):
    real_async_client = httpx.AsyncClient
    rows = {SESSION_A: row(SESSION_A, USER_A), SESSION_B: row(SESSION_B, USER_B)}
    requests = []
    db_status = 200

    def set_db_status(value):
        nonlocal db_status
        db_status = value

    def handler(request):
        requests.append(request)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        user_id = {"token-a": USER_A, "token-b": USER_B}.get(token)
        if request.url.path == "/auth/v1/user":
            return httpx.Response(200 if user_id else 401, json={"id": user_id} if user_id else {})
        assert request.url.path == "/rest/v1/sessions"
        if db_status != 200:
            return httpx.Response(db_status, json={"message": "database unavailable"})
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["user_id"] == user_id
            assert set(body) <= {"user_id", "input_type", "cough_type", "recorded_at"}
            created = row(SESSION_A, user_id, body["input_type"])
            created["cough_type"] = body.get("cough_type")
            rows[SESSION_A] = created
            return httpx.Response(201, json=[created])
        assert request.url.params["user_id"] == f"eq.{user_id}"
        found = [item for item in rows.values() if item["user_id"] == user_id]
        input_filter = request.url.params.get("input_type")
        if input_filter:
            found = [item for item in found if item["input_type"] == input_filter.removeprefix("eq.")]
        session_filter = request.url.params.get("session_id")
        if session_filter:
            found = [item for item in found if item["session_id"] == session_filter.removeprefix("eq.")]
        if request.method == "DELETE":
            for item in found:
                rows.pop(item["session_id"])
            return httpx.Response(200, json=[{"session_id": item["session_id"], "user_id": item["user_id"]} for item in found])
        if "order" in request.url.params:
            reverse = request.url.params["order"].startswith("created_at.desc")
            found.sort(key=lambda item: (item["created_at"], item["session_id"]), reverse=reverse)
        offset = int(request.url.params.get("offset", 0))
        limit = int(request.url.params.get("limit", len(found)))
        return httpx.Response(200, json=found[offset:offset + limit])

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_async_client(transport=httpx.MockTransport(handler)))
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    yield SyncASGIClient(real_async_client), rows, requests, set_db_status
    app.dependency_overrides.clear()


def auth(token="token-a"):
    return {"Authorization": f"Bearer {token}"}


def test_create_uses_verified_user_and_pending_status(setup):
    client, rows, requests, _ = setup
    response = client.request("POST", "/api/v1/sessions", headers=auth(), json={"input_type": "cough", "cough_type": "passive"})
    assert response.status_code == 201
    assert response.json()["user_id"] == USER_A
    assert response.json()["status"] == "pending"
    assert response.json()["risk_score"] is None
    assert rows[SESSION_A]["user_id"] == USER_A
    assert requests[-1].headers["authorization"] == "Bearer token-a"
    assert "missing=default" in requests[-1].headers["prefer"]


@pytest.mark.parametrize("body", [
    {"input_type": "cough", "user_id": USER_B},
    {"input_type": "cough", "status": "completed"},
    {"input_type": "cough", "risk_score": 99},
    {"input_type": "breathing", "cough_type": "forced"},
    {"input_type": "cough", "recorded_at": "2026-10-02T00:00:00"},
])
def test_create_rejects_untrusted_or_invalid_fields(setup, body):
    client, _, requests, _ = setup
    response = client.request("POST", "/api/v1/sessions", headers=auth(), json=body)
    assert response.status_code == 422
    assert all(request.url.path != "/rest/v1/sessions" for request in requests)


def test_list_is_scoped_and_paginated(setup):
    client, _, requests, _ = setup
    response = client.request("GET", "/api/v1/sessions?limit=1&offset=0", headers=auth())
    assert response.status_code == 200
    assert [item["session_id"] for item in response.json()["sessions"]] == [SESSION_A]
    assert response.json()["limit"] == 1
    assert requests[-1].url.params["order"] == "created_at.desc,session_id.desc"
    assert client.request("GET", "/api/v1/sessions?limit=101", headers=auth()).status_code == 422


def test_history_filters_and_sorting(setup):
    client, rows, requests, _ = setup
    rows[SESSION_A_BREATHING] = row(SESSION_A_BREATHING, USER_A, "breathing")
    rows[SESSION_A_BREATHING]["created_at"] = "2026-10-03T00:00:00Z"

    newest = client.request("GET", "/api/v1/sessions?sort=newest", headers=auth())
    assert [item["session_id"] for item in newest.json()["sessions"]] == [SESSION_A_BREATHING, SESSION_A]
    oldest = client.request("GET", "/api/v1/sessions?sort=oldest&offset=1", headers=auth())
    assert [item["session_id"] for item in oldest.json()["sessions"]] == [SESSION_A_BREATHING]
    assert requests[-1].url.params["order"] == "created_at.asc,session_id.asc"

    breathing = client.request("GET", "/api/v1/sessions?input_type=breathing", headers=auth())
    assert [item["session_id"] for item in breathing.json()["sessions"]] == [SESSION_A_BREATHING]
    assert requests[-1].url.params["input_type"] == "eq.breathing"
    assert client.request("GET", "/api/v1/sessions?input_type=other", headers=auth()).status_code == 422
    assert client.request("GET", "/api/v1/sessions?sort=random", headers=auth()).status_code == 422


def test_get_and_delete_are_owner_scoped(setup):
    client, rows, _, _ = setup
    own = client.request("GET", f"/api/v1/sessions/{SESSION_A}", headers=auth())
    assert own.status_code == 200 and UUID(own.json()["session_id"]) == UUID(SESSION_A)
    assert client.request("GET", f"/api/v1/sessions/{SESSION_B}", headers=auth()).status_code == 404
    assert client.request("DELETE", f"/api/v1/sessions/{SESSION_B}", headers=auth()).status_code == 404
    assert SESSION_B in rows
    assert client.request("DELETE", f"/api/v1/sessions/{SESSION_A}", headers=auth()).status_code == 204
    assert SESSION_A not in rows
    assert client.request("GET", f"/api/v1/sessions/{SESSION_A}", headers=auth()).status_code == 404


def test_auth_and_database_errors(setup):
    client, _, requests, set_db_status = setup
    assert client.request("GET", "/api/v1/sessions").status_code == 401
    assert client.request("GET", "/api/v1/sessions", headers=auth("bad-token")).status_code == 401
    assert all(request.url.path != "/rest/v1/sessions" for request in requests)
    set_db_status(503)
    assert client.request("GET", "/api/v1/sessions", headers=auth()).status_code == 503
