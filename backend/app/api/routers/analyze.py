"""Authenticated screening of WAV audio using the existing waveform model pipeline."""

import base64
import binascii
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.core.config import Settings, get_settings
from backend.app.core.security import AuthenticatedUser, get_current_user
from backend.app.ml.aggregator import PatientAggregator
from backend.app.ml.feature_extractor import FeatureExtractionError, HAS_LIBROSA
from backend.app.ml.preprocessor import AudioPreprocessor
from backend.app.models.request import AudioScreeningRequest, ClinicalSymptoms
from backend.app.models.response import AudioWindowScore, PatientScreeningResponse
from backend.app.services.analysis_result_service import AnalysisResultService
from backend.app.services.classifier import ModelInferenceError, RespiratoryClassifierService
from backend.app.services.hear_service import HeARService
from backend.app.services.session_service import SessionService


router = APIRouter(prefix="/api/v1/analyze", tags=["Respiratory Screening"])

MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
preprocessor = AudioPreprocessor()
hear_service = HeARService()
classifier_service = RespiratoryClassifierService()
aggregator = PatientAggregator()


@router.get("/status")
def get_service_status():
    tb_ready = all((
        classifier_service.model_passive is not None,
        classifier_service.scaler_passive is not None,
        classifier_service.model_forced is not None,
        classifier_service.scaler_forced is not None,
    ))
    pathology_ready = classifier_service.model_pathology is not None and classifier_service.scaler_pathology is not None
    fusion_ready = classifier_service.model_multimodal is not None
    return {
        "status": "unavailable" if not (HAS_LIBROSA and tb_ready and pathology_ready) else "degraded",
        "service": "ShwaasAI Respiratory Screening Engine",
        "feature_extractor_ready": HAS_LIBROSA,
        "foundation_model_loaded": hear_service.is_foundation_model_loaded,
        "models": {
            "tb_dual_head": tb_ready,
            "pathology_sound": pathology_ready,
            "multimodal_fusion": fusion_ready,
        },
        "accepted_audio_format": "WAV",
        "target_sample_rate": preprocessor.target_sr,
        "window_duration_sec": preprocessor.window_duration,
        "version": "1.0.0",
    }


def _run_screening_pipeline(
    raw_audio_bytes: bytes,
    cough_type: str,
    symptoms: ClinicalSymptoms | None,
) -> tuple[PatientScreeningResponse, list[float]]:
    """Decode WAV, create 16 kHz 2-second windows, infer, and pool embeddings."""
    try:
        full_audio, raw_windows = preprocessor.process_raw_audio(raw_audio_bytes)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported WAV audio") from exc

    duration = len(full_audio) / preprocessor.target_sr
    if duration <= 0 or duration > MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=422, detail="Audio duration must be between 0 and 60 seconds")
    if not np.all(np.isfinite(full_audio)) or not any(window["is_active"] for window in raw_windows):
        raise HTTPException(status_code=422, detail="Audio contains no usable respiratory sound")

    embedded_windows = hear_service.extract_embeddings_for_windows(raw_windows)
    active_windows = [window for window in embedded_windows if window["is_active"]]
    pooled = np.mean([window["embedding"] for window in active_windows], axis=0)
    if pooled.shape != (512,) or not np.all(np.isfinite(pooled)):
        raise HTTPException(status_code=503, detail="Invalid model embedding")
    norm = float(np.linalg.norm(pooled))
    if norm == 0:
        raise HTTPException(status_code=503, detail="Invalid model embedding")
    embedding = (pooled / norm).astype(np.float32).tolist()

    window_scores: list[AudioWindowScore] = []
    for window in embedded_windows:
        vector = window["embedding"]
        risk = classifier_service.predict_tb_risk(vector, cough_type=cough_type)
        pathology, confidence = classifier_service.predict_pathology(vector)
        window_scores.append(AudioWindowScore(
            window_index=window["window_index"],
            start_time_sec=window["start_time_sec"],
            end_time_sec=window["end_time_sec"],
            tb_risk_score=risk,
            pathology=pathology,
            pathology_confidence=confidence,
            is_cough_detected=window["is_active"],
            rms_energy=round(window["rms_energy"], 4),
        ))

    multimodal_risk = None
    multimodal_category = None
    if symptoms is not None:
        acoustic_score = aggregator.aggregate_windows(window_scores)[0]
        multimodal_risk, multimodal_category = classifier_service.predict_multimodal_fusion(acoustic_score, symptoms)

    result = aggregator.build_screening_response(
        patient_id=None,
        total_duration_sec=duration,
        window_scores=window_scores,
        multimodal_risk=multimodal_risk,
        multimodal_category=multimodal_category,
    )
    return result, embedding


async def _analyze(
    session_id: UUID,
    audio_bytes: bytes,
    symptoms: ClinicalSymptoms | None,
    user: AuthenticatedUser,
    settings: Settings,
) -> PatientScreeningResponse:
    session = await SessionService(settings, user.access_token).get(user.id, session_id)
    if session["status"] != "pending":
        raise HTTPException(status_code=409, detail="Session is no longer pending")
    # Current TB heads were trained for passive/forced cough. Do not label a
    # breathing clip with a cough-based TB risk until a breathing model exists.
    if session["input_type"] != "cough":
        raise HTTPException(status_code=422, detail="The current TB model supports cough sessions only")
    if not settings.secret_key:
        raise HTTPException(status_code=503, detail="Analysis result storage is not configured")
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 20 MiB")
    if not (audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE"):
        raise HTTPException(status_code=415, detail="Upload a WAV file")

    try:
        result, embedding = await run_in_threadpool(
            _run_screening_pipeline, audio_bytes, session["cough_type"] or "both", symptoms
        )
    except (FeatureExtractionError, ModelInferenceError) as exc:
        raise HTTPException(status_code=503, detail="Screening model unavailable") from exc
    await AnalysisResultService(settings).complete(user.id, session_id, result, embedding)
    result.session_id = session_id
    return result


@router.post("", response_model=PatientScreeningResponse)
@router.post("/upload", response_model=PatientScreeningResponse, include_in_schema=False)
async def analyze_upload(
    session_id: UUID = Form(...),
    file: UploadFile = File(..., description="WAV audio; mono/stereo and sample rate are normalized server-side"),
    symptoms: str | None = Form(default=None, description="Optional JSON object matching ClinicalSymptoms"),
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> PatientScreeningResponse:
    clinical = None
    if symptoms is not None:
        try:
            clinical = ClinicalSymptoms.model_validate_json(symptoms)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid symptoms JSON") from exc
    audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)
    return await _analyze(session_id, audio_bytes, clinical, user, settings)


@router.post("/audio", response_model=PatientScreeningResponse)
async def analyze_base64_audio(
    payload: AudioScreeningRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> PatientScreeningResponse:
    encoded = payload.audio_base64
    if encoded.startswith("data:"):
        if "," not in encoded:
            raise HTTPException(status_code=400, detail="Invalid audio data URI")
        encoded = encoded.split(",", 1)[1]
    if len(encoded) > (MAX_AUDIO_BYTES * 4 // 3 + 8):
        raise HTTPException(status_code=413, detail="Audio file exceeds 20 MiB")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio") from exc
    return await _analyze(payload.session_id, audio_bytes, payload.symptoms, user, settings)
