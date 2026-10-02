import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.models.session import SessionStats


class SessionStatsService:
    def __init__(self, settings: Settings, access_token: str) -> None:
        self.url = f"{settings.supabase_url}/rest/v1/rpc/get_my_session_stats"
        self.headers = {
            "apikey": settings.publishable_key,
            "Authorization": f"Bearer {access_token}",
        }

    async def get(self) -> SessionStats:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(self.url, headers=self.headers)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="Statistics database unavailable") from exc
        if response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Statistics access denied")
        if response.status_code >= 400:
            raise HTTPException(status_code=503, detail="Statistics database unavailable")
        try:
            return SessionStats.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=503, detail="Invalid statistics database response") from exc
