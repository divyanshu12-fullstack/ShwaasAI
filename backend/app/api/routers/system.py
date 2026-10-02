"""Public service health, capability information, and documentation links."""

from time import monotonic
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.app.api.routers.analyze import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_SECONDS,
    get_service_status,
    preprocessor,
)
from backend.app.core.config import Settings, get_settings


router = APIRouter(prefix="/api/v1", tags=["system"])
STARTED_AT = monotonic()
API_VERSION = "1.0.0"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    feature_extractor_ready: bool
    hear_model_loaded: bool
    classifier_loaded: bool
    pathology_model_loaded: bool
    multimodal_model_loaded: bool
    db_connected: bool
    version: str
    uptime_seconds: int = Field(ge=0)


class InfoResponse(BaseModel):
    api_version: str
    documentation: str
    redoc: str
    openapi: str
    session_input_types: list[str]
    analysis_input_types: list[str]
    supported_cough_types: list[str]
    accepted_audio_formats: list[str]
    sample_rate_hz: int
    window_duration_seconds: float
    window_hop_seconds: float
    embedding_dimension: int
    max_audio_bytes: int
    max_audio_duration_seconds: int
    foundation_model_loaded: bool
    active_extractor: Literal["HeAR", "acoustic_fallback"]
    disclaimer: str


async def _database_connected(settings: Settings) -> bool:
    """Check the Data API path used by the screening backend without reading rows."""
    if not settings.secret_key:
        return False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(
                f"{settings.supabase_url}/rest/v1/sessions",
                headers={
                    "apikey": settings.secret_key,
                    "Authorization": f"Bearer {settings.secret_key}",
                },
                params={"select": "session_id", "limit": 0},
            )
    except httpx.RequestError:
        return False
    return response.status_code == 200


@router.get("/health", response_model=HealthResponse)
async def health(response: Response, settings: Settings = Depends(get_settings)) -> HealthResponse:
    models = get_service_status()
    loaded = models["models"]
    db_connected = await _database_connected(settings)
    classifier_loaded = bool(loaded["tb_dual_head"])
    feature_extractor_ready = bool(models.get("feature_extractor_ready", True))
    hear_loaded = bool(models["foundation_model_loaded"])
    if not db_connected or not feature_extractor_ready or not classifier_loaded or not loaded["pathology_sound"]:
        overall = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif not hear_loaded or not loaded["pathology_sound"] or not loaded["multimodal_fusion"]:
        overall = "degraded"
    else:
        overall = "ok"
    response.headers["Cache-Control"] = "no-store"
    return HealthResponse(
        status=overall,
        feature_extractor_ready=feature_extractor_ready,
        hear_model_loaded=hear_loaded,
        classifier_loaded=classifier_loaded,
        pathology_model_loaded=bool(loaded["pathology_sound"]),
        multimodal_model_loaded=bool(loaded["multimodal_fusion"]),
        db_connected=db_connected,
        version=API_VERSION,
        uptime_seconds=int(monotonic() - STARTED_AT),
    )


@router.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    model_status = get_service_status()
    foundation_loaded = bool(model_status["foundation_model_loaded"])
    return InfoResponse(
        api_version=API_VERSION,
        documentation="/docs",
        redoc="/redoc",
        openapi="/openapi.json",
        session_input_types=["cough", "breathing"],
        analysis_input_types=["cough"],
        supported_cough_types=["passive", "forced"],
        accepted_audio_formats=["WAV"],
        sample_rate_hz=preprocessor.target_sr,
        window_duration_seconds=preprocessor.window_duration,
        window_hop_seconds=preprocessor.hop_samples / preprocessor.target_sr,
        embedding_dimension=512,
        max_audio_bytes=MAX_AUDIO_BYTES,
        max_audio_duration_seconds=MAX_AUDIO_SECONDS,
        foundation_model_loaded=foundation_loaded,
        active_extractor="HeAR" if foundation_loaded else "acoustic_fallback",
        disclaimer="Research screening tool; not a medical diagnosis.",
    )


@router.get("/docs", include_in_schema=False)
def versioned_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/redoc", include_in_schema=False)
def versioned_redoc() -> RedirectResponse:
    return RedirectResponse(url="/redoc")
