import json
from uuid import UUID

import httpx
from fastapi import HTTPException

from app.core.config import Settings


class ProfileService:
    def __init__(self, settings: Settings, access_token: str) -> None:
        self.base_url = f"{settings.supabase_url}/rest/v1/profiles"
        self.headers = {
            "apikey": settings.publishable_key,
            "Authorization": f"Bearer {access_token}",
        }

    async def get(self, user_id: UUID) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    self.base_url,
                    headers=self.headers,
                    params={"user_id": f"eq.{user_id}", "select": "user_id,display_name,created_at,updated_at"},
                )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="Profile database unavailable") from exc
        self._check_response(response)
        return self._own_row(response, user_id)

    async def update(self, user_id: UUID, display_name: str | None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.patch(
                    self.base_url,
                    headers={**self.headers, "Prefer": "return=representation"},
                    params={"user_id": f"eq.{user_id}", "select": "user_id,display_name,created_at,updated_at"},
                    json={"display_name": display_name},
                )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="Profile database unavailable") from exc
        self._check_response(response)
        return self._own_row(response, user_id)

    @staticmethod
    def _own_row(response: httpx.Response, user_id: UUID) -> dict:
        try:
            rows = response.json()
            if not isinstance(rows, list):
                raise ValueError("Expected a list")
            if not rows:
                raise HTTPException(status_code=404, detail="Profile not found")
            row = rows[0]
            if not isinstance(row, dict) or UUID(row["user_id"]) != user_id:
                raise ValueError("Profile owner mismatch")
            return row
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=503, detail="Invalid profile database response") from exc

    @staticmethod
    def _check_response(response: httpx.Response) -> None:
        if response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Profile access denied")
        if response.status_code >= 400:
            raise HTTPException(status_code=503, detail="Profile database unavailable")
