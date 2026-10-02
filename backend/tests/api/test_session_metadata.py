import asyncio
import json

import httpx
import pytest

from app.core.config import get_settings
from main import app


USER_A = "72ff8e22-31c2-4a06-a287-dcccb706bc29"
USER_B = "ed3cce37-1c91-4337-b2a7-9eb59a522532"
SESSION_A = "45172db7-39a9-46c0-b870-87e699994ad2"
SESSION_B = "ef065569-718c-405c-9954-ef9b06afcfa6"
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


def session_row(session_id, user_id):
    return {
        "session_id": session_id,
        "user_id": user_id,
        "input_type": "cough",
        "cough_type": "passive",
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
    sessions = {SESSION_A: session_row(SESSION_A, USER_A), SESSION_B: session_row(SESSION_B, USER_B)}
    metadata = {}
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
        if db_status != 200:
            return httpx.Response(db_status, json={"message": "database unavailable"})
        session_id = request.url.params.get("session_id", "").removeprefix("eq.")
        if request.url.path == "/rest/v1/sessions":
            assert request.method == "GET"
            assert request.url.params["user_id"] == f"eq.{user_id}"
            row = sessions.get(session_id)
            return httpx.Response(200, json=[row] if row and row["user_id"] == user_id else [])
        assert request.url.path == "/rest/v1/session_metadata"
        if request.method == "GET":
            row = metadata.get(session_id)
            return httpx.Response(200, json=[row] if row and sessions[session_id]["user_id"] == user_id else [])
        if request.method == "POST":
            body = json.loads(request.content)
            assert set(body) == {
                "session_id", "age", "sex", "fever", "smoker", "cough_duration", "night_sweats", "weight_loss"
            }
            assert sessions[body["session_id"]]["user_id"] == user_id
            assert "missing=default" in request.headers["prefer"]
            saved = {**body, "created_at": TIME, "updated_at": TIME}
            metadata[body["session_id"]] = saved
            return httpx.Response(201, json=[saved])
        assert request.method == "PATCH"
        assert sessions[session_id]["user_id"] == user_id
        assert "session_id" not in json.loads(request.content)
        metadata[session_id].update(json.loads(request.content))
        return httpx.Response(200, json=[metadata[session_id]])

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_async_client(transport=httpx.MockTransport(handler)))
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    yield SyncASGIClient(real_async_client), metadata, requests, set_db_status
    app.dependency_overrides.clear()


def auth(token="token-a"):
    return {"Authorization": f"Bearer {token}"}


def test_metadata_create_read_and_replace(setup):
    client, metadata, _, _ = setup
    path = f"/api/v1/sessions/{SESSION_A}/metadata"
    assert client.request("GET", path, headers=auth()).status_code == 404

    created = client.request("PUT", path, headers=auth(), json={"age": 28, "fever": False, "night_sweats": True})
    assert created.status_code == 200
    assert created.json()["session_id"] == SESSION_A
    assert created.json()["age"] == 28
    assert created.json()["fever"] is False
    assert client.request("GET", path, headers=auth()).json()["night_sweats"] is True

    replaced = client.request("PUT", path, headers=auth(), json={"age": 29, "weight_loss": False})
    assert replaced.status_code == 200
    assert replaced.json()["age"] == 29
    assert replaced.json()["night_sweats"] is None
    assert metadata[SESSION_A]["weight_loss"] is False


def test_metadata_rejects_cross_user_and_invalid_input(setup):
    client, metadata, requests, _ = setup
    other_path = f"/api/v1/sessions/{SESSION_B}/metadata"
    assert client.request("GET", other_path, headers=auth()).status_code == 404
    assert client.request("PUT", other_path, headers=auth(), json={"age": 30}).status_code == 404
    assert SESSION_B not in metadata
    assert all(request.url.path != "/rest/v1/session_metadata" for request in requests)

    own_path = f"/api/v1/sessions/{SESSION_A}/metadata"
    assert client.request("PUT", own_path, headers=auth(), json={"age": 121}).status_code == 422
    assert client.request("PUT", own_path, headers=auth(), json={"user_id": USER_B}).status_code == 422
    assert client.request("PUT", own_path, headers=auth(), json={"sex": "x" * 41}).status_code == 422


def test_metadata_requires_auth_and_reports_database_outage(setup):
    client, _, _, set_db_status = setup
    path = f"/api/v1/sessions/{SESSION_A}/metadata"
    assert client.request("GET", path).status_code == 401
    assert client.request("PUT", path, headers=auth("bad-token"), json={"age": 25}).status_code == 401
    set_db_status(503)
    assert client.request("GET", path, headers=auth()).status_code == 503
