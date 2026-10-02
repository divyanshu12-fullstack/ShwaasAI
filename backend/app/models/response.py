"""
Pydantic schemas for ShwaasAI audio screening responses.
"""

from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class AudioWindowScore(BaseModel):
    """
    Risk and pathology assessment for an individual 2-second audio segment.
    """
    window_index: int = Field(..., description="0-indexed sequence of the 2-second slice")
    start_time_sec: float = Field(..., description="Start timestamp of the window in seconds")
    end_time_sec: float = Field(..., description="End timestamp of the window in seconds")
    tb_risk_score: float = Field(..., ge=0.0, le=1.0, description="Research model score; clinical calibration has not been established")
    pathology: str = Field(..., description="Detected acoustic pattern (normal, crackles, wheezes, abnormal)")
    pathology_confidence: float = Field(..., ge=0.0, le=1.0, description="Pathology prediction confidence")
    is_cough_detected: bool = Field(..., description="Legacy name: RMS activity above the silence threshold; this is not cough detection")
    rms_energy: float = Field(..., description="Acoustic energy of this segment")


class PatientScreeningResponse(BaseModel):
    """
    Comprehensive screening report aggregated across all audio windows and clinical metadata.
    """
    session_id: Optional[UUID] = Field(default=None, description="Saved screening session")
    patient_id: Optional[str] = Field(default=None, description="Legacy field; always null in authenticated analysis")
    status: str = Field(default="success", description="Status of the screening request")
    
    # Acoustic Cough Risk
    acoustic_tb_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Patient-level research acoustic score; not a validated TB probability"
    )
    acoustic_risk_category: str = Field(
        ..., description="Risk tier: 'Low Risk', 'Moderate Risk', or 'High Risk'"
    )
    
    # Respiratory Sound Pathology
    primary_pathology: str = Field(
        ..., description="Dominant respiratory sound pattern: 'Normal Respiratory Sound', 'Crackles Detected', 'Wheezes Detected', or 'Abnormal Acoustic Pattern'"
    )
    pathology_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of primary pathology detection"
    )

    # Multimodal Score (Audio + Clinical Metadata)
    multimodal_risk_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Combined score fusing acoustics with clinical symptoms"
    )
    multimodal_risk_category: Optional[str] = Field(
        default=None, description="Multimodal risk tier"
    )
    
    # Window Metadata & Explainability
    total_audio_duration_sec: float = Field(..., description="Total length of input audio in seconds")
    total_windows_analyzed: int = Field(..., description="Total number of 2-second windows processed")
    windows_with_cough: int = Field(..., description="Legacy name: number of windows above the RMS activity threshold")
    most_suspicious_window: Optional[AudioWindowScore] = Field(
        default=None, description="The window that contributed most to the elevated risk score"
    )
    window_breakdown: List[AudioWindowScore] = Field(
        default_factory=list, description="Per-window acoustic assessment"
    )

    # Clinical Triage & Disclaimer
    triage_recommendation: str = Field(
        ..., description="Research screening follow-up text; not validated medical advice"
    )
    disclaimer: str = Field(
        default=(
            "ShwaasAI is a research-oriented screening and risk-prioritization tool, "
            "not a definitive medical diagnostic device. A high risk score indicates need "
            "for clinical confirmation via microbiological tests (e.g. sputum smear / GeneXpert). "
            "Always consult a qualified healthcare professional."
        ),
        description="Mandatory medical and regulatory disclaimer"
    )
