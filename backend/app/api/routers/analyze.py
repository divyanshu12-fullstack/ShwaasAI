"""
FastAPI router for ShwaasAI respiratory audio screening.

Exposes:
- POST /api/v1/analyze/audio (JSON payload with base64 audio and clinical symptoms)
- POST /api/v1/analyze/upload (Multipart audio file upload with form fields)
- GET /api/v1/analyze/status (Model and pipeline status)
"""

import base64
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
import numpy as np

from backend.app.models.request import AudioScreeningRequest, ClinicalSymptoms
from backend.app.models.response import PatientScreeningResponse, AudioWindowScore
from backend.app.ml.preprocessor import AudioPreprocessor
from backend.app.services.hear_service import HeARService
from backend.app.services.classifier import RespiratoryClassifierService
from backend.app.ml.aggregator import PatientAggregator

router = APIRouter(prefix="/api/v1/analyze", tags=["Respiratory Screening"])

# Initialize singletons
preprocessor = AudioPreprocessor()
hear_service = HeARService()
classifier_service = RespiratoryClassifierService()
aggregator = PatientAggregator()


@router.get("/status")
def get_service_status():
    """
    Returns the current operational status of the ML pipeline.
    """
    return {
        "status": "online",
        "service": "ShwaasAI Respiratory Screening Engine",
        "foundation_model_loaded": hear_service.is_foundation_model_loaded,
        "models": {
            "tb_dual_head": classifier_service.model_passive is not None,
            "pathology_sound": classifier_service.model_pathology is not None,
            "multimodal_fusion": classifier_service.model_multimodal is not None,
        },
        "target_sample_rate": preprocessor.target_sr,
        "window_duration_sec": preprocessor.window_duration,
        "version": "1.0.0"
    }


def _run_screening_pipeline(
    raw_audio_bytes: bytes,
    patient_id: Optional[str] = None,
    cough_type: str = "both",
    symptoms: Optional[ClinicalSymptoms] = None
) -> PatientScreeningResponse:
    """
    Internal shared execution pipeline for audio bytes.
    """
    # 1. Preprocess audio & segment into 2s windows
    try:
        full_audio, raw_windows = preprocessor.process_raw_audio(raw_audio_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio decoding or preprocessing failed: {str(e)}"
        )

    if not raw_windows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio clip contains insufficient audio data."
        )

    total_duration_sec = len(full_audio) / preprocessor.target_sr

    # 2. Extract 512-dim health acoustic embeddings
    embedded_windows = hear_service.extract_embeddings_for_windows(raw_windows)

    # 3. Classify each window
    window_scores: list[AudioWindowScore] = []
    for w in embedded_windows:
        emb = w["embedding"]
        tb_risk = classifier_service.predict_tb_risk(emb, cough_type=cough_type)
        pathology_name, path_conf = classifier_service.predict_pathology(emb)

        window_scores.append(
            AudioWindowScore(
                window_index=w["window_index"],
                start_time_sec=w["start_time_sec"],
                end_time_sec=w["end_time_sec"],
                tb_risk_score=tb_risk,
                pathology=pathology_name,
                pathology_confidence=path_conf,
                is_cough_detected=w["is_active"],
                rms_energy=round(w["rms_energy"], 4),
            )
        )

    # 4. Multimodal fusion (if symptoms provided)
    multimodal_risk = None
    multimodal_category = None
    if symptoms is not None:
        (
            acoustic_score,
            _,
            _,
            _,
            _
        ) = aggregator.aggregate_windows(window_scores)
        multimodal_risk, multimodal_category = classifier_service.predict_multimodal_fusion(
            acoustic_score, symptoms
        )

    # 5. Patient-level aggregation & explainability
    response = aggregator.build_screening_response(
        patient_id=patient_id,
        total_duration_sec=total_duration_sec,
        window_scores=window_scores,
        multimodal_risk=multimodal_risk,
        multimodal_category=multimodal_category
    )

    return response


@router.post("/audio", response_model=PatientScreeningResponse)
def analyze_base64_audio(payload: AudioScreeningRequest):
    """
    Screens audio uploaded as base64 string with optional clinical questionnaire symptoms.
    """
    if not payload.audio_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="audio_base64 field is required."
        )

    b64 = payload.audio_base64
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 encoding: {e}"
        )

    return _run_screening_pipeline(
        raw_audio_bytes=raw_bytes,
        patient_id=payload.patient_id,
        cough_type=payload.cough_type or "both",
        symptoms=payload.symptoms
    )


@router.post("/upload", response_model=PatientScreeningResponse)
async def analyze_file_upload(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, WebM)"),
    patient_id: Optional[str] = Form(None),
    cough_type: str = Form("both"),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form("male"),
    cough_duration_days: Optional[int] = Form(0),
    has_fever: Optional[bool] = Form(False),
    has_night_sweats: Optional[bool] = Form(False),
    has_unexplained_weight_loss: Optional[bool] = Form(False),
    has_hemoptysis: Optional[bool] = Form(False),
    is_smoker: Optional[bool] = Form(False),
    has_chest_pain: Optional[bool] = Form(False),
):
    """
    Screens audio uploaded directly as a multipart form file from frontend recorder or mobile app.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty."
        )

    symptoms = None
    if any([age, cough_duration_days, has_fever, has_night_sweats, has_unexplained_weight_loss, has_hemoptysis, is_smoker]):
        symptoms = ClinicalSymptoms(
            age=age or 35,
            gender=gender or "male",
            cough_duration_days=cough_duration_days or 0,
            has_fever=bool(has_fever),
            has_night_sweats=bool(has_night_sweats),
            has_unexplained_weight_loss=bool(has_unexplained_weight_loss),
            has_hemoptysis=bool(has_hemoptysis),
            is_smoker=bool(is_smoker),
            has_chest_pain=bool(has_chest_pain),
        )

    return _run_screening_pipeline(
        raw_audio_bytes=audio_bytes,
        patient_id=patient_id,
        cough_type=cough_type,
        symptoms=symptoms
    )
