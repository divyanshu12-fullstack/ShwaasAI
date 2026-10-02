from uuid import UUID

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.models.session import EmbeddingRead
from backend.app.services.session_service import SessionService


class EmbeddingService:
    def __init__(self, settings: Settings, access_token: str) -> None:
        self.url = f"{settings.supabase_url}/rest/v1/embeddings"
        self.headers = {
            "apikey": settings.publishable_key,
            "Authorization": f"Bearer {access_token}",
        }
        self.sessions = SessionService(settings, access_token)

    async def get(self, user_id: UUID, session_id: UUID) -> EmbeddingRead:
        await self.sessions.get(user_id, session_id)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    self.url,
                    headers=self.headers,
                    params={
                        "session_id": f"eq.{session_id}",
                        "select": "session_id,embedding_data,embedding_dim,stored_at",
                        "limit": 1,
                    },
                )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="Embedding database unavailable") from exc
        if response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Embedding access denied")
        if response.status_code >= 400:
            raise HTTPException(status_code=503, detail="Embedding database unavailable")
        try:
            rows = response.json()
            if not isinstance(rows, list) or len(rows) > 1:
                raise ValueError("Expected zero or one embedding")
            if not rows:
                raise HTTPException(status_code=404, detail="Embedding not found")
            embedding = EmbeddingRead.model_validate(rows[0])
            if embedding.session_id != session_id:
                raise ValueError("Embedding session mismatch")
            return embedding
        except (ValueError, ValidationError, TypeError) as exc:
            raise HTTPException(status_code=503, detail="Invalid embedding database response") from exc
