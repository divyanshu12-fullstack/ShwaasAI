"""
Pydantic schemas for ShwaasAI audio screening requests.
"""

from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class ClinicalSymptoms(BaseModel):
    """
    Clinical symptom metadata for multimodal screening.
    Clinical indicators correspond to standard WHO presumptive-TB criteria.
    """
    age: Optional[int] = Field(
        default=35,
        ge=0,
        le=120,
        description="Patient age in years"
    )
    gender: Optional[str] = Field(
        default="male",
        description="Gender: male, female, or other"
    )
    cough_duration_days: Optional[int] = Field(
        default=0,
        ge=0,
        description="Duration of cough in days"
    )
    has_fever: Optional[bool] = Field(
        default=False,
        description="Persistent or recurrent fever"
    )
    has_night_sweats: Optional[bool] = Field(
        default=False,
        description="Drenching night sweats"
    )
    has_unexplained_weight_loss: Optional[bool] = Field(
        default=False,
        description="Unexplained significant weight loss"
    )
    has_hemoptysis: Optional[bool] = Field(
        default=False,
        description="Coughing blood (hemoptysis)"
    )
    is_smoker: Optional[bool] = Field(
        default=False,
        description="Current or former smoker"
    )
    has_chest_pain: Optional[bool] = Field(
        default=False,
        description="Pleuritic chest pain"
    )


class AudioScreeningRequest(BaseModel):
    """
    Request payload containing base64 audio and optional clinical metadata.
    """
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    audio_base64: str = Field(description="Base64-encoded WAV file; preprocessing creates 16 kHz waveform windows")
    symptoms: Optional[ClinicalSymptoms] = Field(
        default=None,
        description="Optional clinical symptoms for multimodal risk boost"
    )
