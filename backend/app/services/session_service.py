from __future__ import annotations

import json
from uuid import UUID

import httpx
from fastapi import HTTPException

from app.core.config import Settings
from app.models.session import InputType, SessionCreate


SESSION_COLUMNS = (
    "session_id,user_id,input_type,cough_type,status,risk_score,level,"
    "confidence,recommendation,model_used,recorded_at,created_at,updated_at"
)


class SessionService:
    def __init__(self, settings: Settings, access_token: str) -> None:
        self.url = f"{settings.supabase_url}/rest/v1/sessions"
        self.headers = {
            "apikey": settings.publishable_key,
            "Authorization": f"Bearer {access_token}",
        }

    async def create(self, user_id: UUID, details: SessionCreate) -> dict:
        payload = {
            "user_id": str(user_id),
            "input_type": details.input_type,
            "cough_type": details.cough_type,
        }
        if details.recorded_at is not None:
            payload["recorded_at"] = details.recorded_at.isoformat()
        response = await self._request(
            "POST",
            headers={"Prefer": "missing=default, return=representation"},
            params={"select": SESSION_COLUMNS},
            json=payload,
        )
        rows = self._owned_rows(response, user_id)
        if len(rows) != 1:
            raise HTTPException(status_code=503, detail="Unexpected session database response")
        return rows[0]

    async def list(self, user_id: UUID, limit: int, offset: int, input_type: InputType | None, sort: str) -> list[dict]:
        direction = "desc" if sort == "newest" else "asc"
        params = {
            "user_id": f"eq.{user_id}",
            "select": SESSION_COLUMNS,
            "order": f"created_at.{direction},session_id.{direction}",
            "limit": limit,
            "offset": offset,
        }
        if input_type is not None:
            params["input_type"] = f"eq.{input_type}"
        response = await self._request(
            "GET",
            params=params,
        )
        return self._owned_rows(response, user_id)

    async def get(self, user_id: UUID, session_id: UUID) -> dict:
        response = await self._request(
            "GET",
            params={"session_id": f"eq.{session_id}", "user_id": f"eq.{user_id}", "select": SESSION_COLUMNS, "limit": 1},
        )
        rows = self._owned_rows(response, user_id)
        if not rows:
            raise HTTPException(status_code=404, detail="Session not found")
        return rows[0]

    async def delete(self, user_id: UUID, session_id: UUID) -> None:
        response = await self._request(
            "DELETE",
            headers={"Prefer": "return=representation"},
            params={"session_id": f"eq.{session_id}", "user_id": f"eq.{user_id}", "select": "session_id,user_id"},
        )
        rows = self._owned_rows(response, user_id)
        if not rows:
            raise HTTPException(status_code=404, detail="Session not found")

    async def _request(self, method: str, **kwargs) -> httpx.Response:
        extra_headers = kwargs.pop("headers", {})
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.request(
                    method,
                    self.url,
                    headers={**self.headers, **extra_headers},
                    **kwargs,
                )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="Session database unavailable") from exc
        if response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Session access denied")
        if response.status_code >= 400:
            raise HTTPException(status_code=503, detail="Session database unavailable")
        return response

    @staticmethod
    def _owned_rows(response: httpx.Response, user_id: UUID) -> list[dict]:
        try:
            rows = response.json()
            if not isinstance(rows, list):
                raise ValueError("Expected a list")
            if any(not isinstance(row, dict) or UUID(row["user_id"]) != user_id for row in rows):
                raise ValueError("Session owner mismatch")
            return rows
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Invalid session database response") from exc
