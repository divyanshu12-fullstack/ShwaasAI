import json
from uuid import UUID

import httpx
from fastapi import HTTPException

from backend.app.core.config import Settings
from backend.app.models.session import SessionMetadataWrite
from backend.app.services.session_service import SessionService


METADATA_COLUMNS = (
    "session_id,age,sex,fever,smoker,cough_duration,night_sweats,"
    "weight_loss,created_at,updated_at"
)


class MetadataService:
    def __init__(self, settings: Settings, access_token: str) -> None:
        self.url = f"{settings.supabase_url}/rest/v1/session_metadata"
        self.headers = {
            "apikey": settings.publishable_key,
            "Authorization": f"Bearer {access_token}",
        }
        self.sessions = SessionService(settings, access_token)

    async def get(self, user_id: UUID, session_id: UUID) -> dict:
        await self.sessions.get(user_id, session_id)
        response = await self._request(
            "GET",
            params={"session_id": f"eq.{session_id}", "select": METADATA_COLUMNS, "limit": 1},
        )
        return self._one_row(response, session_id)

    async def replace(self, user_id: UUID, session_id: UUID, metadata: SessionMetadataWrite) -> dict:
        await self.sessions.get(user_id, session_id)
        current = await self._request(
            "GET",
            params={"session_id": f"eq.{session_id}", "select": "session_id", "limit": 1},
        )
        rows = self._rows(current)
        if rows:
            response = await self._update(session_id, metadata)
        else:
            response = await self._request(
                "POST",
                headers={"Prefer": "missing=default, return=representation"},
                params={"select": METADATA_COLUMNS},
                json={"session_id": str(session_id), **metadata.model_dump()},
                allow_conflict=True,
            )
            if response.status_code == 409:
                response = await self._update(session_id, metadata)
        return self._one_row(response, session_id)

    async def _update(self, session_id: UUID, metadata: SessionMetadataWrite) -> httpx.Response:
        return await self._request(
            "PATCH",
            headers={"Prefer": "return=representation"},
            params={"session_id": f"eq.{session_id}", "select": METADATA_COLUMNS},
            json=metadata.model_dump(),
        )

    async def _request(self, method: str, allow_conflict: bool = False, **kwargs) -> httpx.Response:
        extra_headers = kwargs.pop("headers", {})
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.request(method, self.url, headers={**self.headers, **extra_headers}, **kwargs)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="Metadata database unavailable") from exc
        if response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Metadata access denied")
        if response.status_code >= 400 and not (allow_conflict and response.status_code == 409):
            raise HTTPException(status_code=503, detail="Metadata database unavailable")
        return response

    @staticmethod
    def _rows(response: httpx.Response) -> list[dict]:
        try:
            rows = response.json()
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Expected a list")
            return rows
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Invalid metadata database response") from exc

    @classmethod
    def _one_row(cls, response: httpx.Response, session_id: UUID) -> dict:
        try:
            rows = cls._rows(response)
            if not rows:
                raise HTTPException(status_code=404, detail="Metadata not found")
            row = rows[0]
            if UUID(row["session_id"]) != session_id:
                raise ValueError("Metadata session mismatch")
            return row
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Invalid metadata database response") from exc
