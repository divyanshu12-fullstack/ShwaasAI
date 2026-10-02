import asyncio
from uuid import UUID

import httpx
import pytest

from backend.app.core.config import get_settings
from backend.main import app


USER_ID = "72ff8e22-31c2-4a06-a287-dcccb706bc29"
OTHER_ID = "ed3cce37-1c91-4337-b2a7-9eb59a522532"
PROFILE = {
    "user_id": USER_ID,
    "display_name": "Asha",
    "created_at": "2026-10-02T00:00:00Z",
    "updated_at": "2026-10-02T00:00:00Z",
}


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

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def patch(self, path, **kwargs):
        return self.request("PATCH", path, **kwargs)


@pytest.fixture
def client(monkeypatch):
    real_async_client = httpx.AsyncClient
    requests = []
    auth_status = 200
    profile_rows = [PROFILE]

    def handler(request):
        requests.append(request)
        if request.url.path == "/auth/v1/user":
            return httpx.Response(auth_status, json={"id": USER_ID} if auth_status == 200 else {})
        assert request.url.path == "/rest/v1/profiles"
        assert request.url.params["user_id"] == f"eq.{USER_ID}"
        assert request.headers["authorization"] == "Bearer valid-token"
        if profile_rows is None:
            return httpx.Response(503, json={"message": "unavailable"})
        if request.method == "PATCH":
            assert request.content == b'{"display_name":"New name"}'
            return httpx.Response(200, json=[{**PROFILE, "display_name": "New name"}])
        return httpx.Response(200, json=profile_rows)

    def set_auth_status(value):
        nonlocal auth_status
        auth_status = value

    def set_profile_rows(rows):
        nonlocal profile_rows
        profile_rows = rows

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_async_client(transport=transport))
    app.dependency_overrides[get_settings] = lambda: TestSettings()
    test_client = SyncASGIClient(real_async_client)
    yield test_client, requests, set_auth_status, set_profile_rows
    app.dependency_overrides.clear()


def auth_header():
    return {"Authorization": "Bearer valid-token"}


def test_missing_token_is_rejected(client):
    test_client, requests, _, _ = client
    assert test_client.get("/api/v1/auth/me").status_code == 401
    assert requests == []


def test_invalid_token_is_rejected(client):
    test_client, requests, set_auth_status, _ = client
    set_auth_status(401)
    assert test_client.get("/api/v1/auth/me", headers=auth_header()).status_code == 401
    assert all(request.url.path != "/rest/v1/profiles" for request in requests)


def test_auth_test_returns_verified_user(client):
    test_client, _, _, _ = client
    response = test_client.get("/api/v1/auth/test", headers=auth_header())
    assert response.status_code == 200
    assert response.json() == {"authenticated": True, "user_id": USER_ID}
    assert UUID(response.json()["user_id"])


def test_get_me_reads_only_verified_user(client):
    test_client, _, _, _ = client
    response = test_client.get("/api/v1/auth/me", headers=auth_header())
    assert response.status_code == 200
    assert response.json()["user_id"] == USER_ID


def test_patch_me_updates_only_verified_user(client):
    test_client, _, _, _ = client
    response = test_client.patch("/api/v1/auth/me", headers=auth_header(), json={"display_name": "New name"})
    assert response.status_code == 200
    assert response.json()["display_name"] == "New name"
    forbidden = test_client.patch("/api/v1/auth/me", headers=auth_header(), json={"user_id": OTHER_ID, "display_name": "New name"})
    assert forbidden.status_code == 422


def test_missing_profile_is_404(client):
    test_client, _, _, set_profile_rows = client
    set_profile_rows([])
    assert test_client.get("/api/v1/auth/me", headers=auth_header()).status_code == 404


def test_empty_patch_is_rejected(client):
    test_client, _, _, _ = client
    assert test_client.patch("/api/v1/auth/me", headers=auth_header(), json={}).status_code == 422


def test_auth_service_outage_is_503(client):
    test_client, _, set_auth_status, _ = client
    set_auth_status(503)
    assert test_client.get("/api/v1/auth/me", headers=auth_header()).status_code == 503


def test_profile_database_outage_is_503(client):
    test_client, _, _, set_profile_rows = client
    set_profile_rows(None)
    assert test_client.get("/api/v1/auth/me", headers=auth_header()).status_code == 503
