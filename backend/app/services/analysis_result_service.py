"""Server-only atomic persistence of a verified user's model result."""

from uuid import UUID

import httpx
from fastapi import HTTPException

from backend.app.core.config import Settings
from backend.app.models.response import PatientScreeningResponse


class AnalysisResultService:
    def __init__(self, settings: Settings) -> None:
        if not settings.secret_key:
            raise HTTPException(status_code=503, detail="Analysis result storage is not configured")
        self.url = f"{settings.supabase_url}/rest/v1/rpc/complete_screening_result"
        # Do not forward the caller's token: this restricted RPC runs only as
        # service_role, after the caller's session ownership was checked.
        self.headers = {
            "apikey": settings.secret_key,
            "Authorization": f"Bearer {settings.secret_key}",
        }

    async def complete(
        self,
        user_id: UUID,
        session_id: UUID,
        result: PatientScreeningResponse,
        embedding: list[float],
    ) -> dict:
        score = result.multimodal_risk_score if result.multimodal_risk_score is not None else result.acoustic_tb_risk_score
        category = result.multimodal_risk_category or result.acoustic_risk_category
        payload = {
            "p_user_id": str(user_id),
            "p_session_id": str(session_id),
            "p_embedding_data": embedding,
            "p_risk_score": round(score * 100),
            "p_level": category.removesuffix(" Risk"),
            "p_recommendation": result.triage_recommendation,
            "p_model_used": "ShwaasAI acoustic screening v1",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(self.url, headers=self.headers, json=payload)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="Analysis result database unavailable") from exc
        if response.status_code == 409:
            raise HTTPException(status_code=409, detail="Session analysis already exists")
        if response.status_code >= 400:
            raise HTTPException(status_code=503, detail="Could not complete analysis session")
        try:
            row = response.json()
            if row is None:
                raise HTTPException(status_code=409, detail="Session is no longer pending")
            if (not isinstance(row, dict) or row.get("user_id") != str(user_id)
                    or row.get("session_id") != str(session_id) or row.get("status") != "completed"):
                raise ValueError("Unexpected completion response")
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=503, detail="Invalid analysis database response") from exc
        return row
