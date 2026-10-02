from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


InputType = Literal["cough", "breathing"]
CoughType = Literal["passive", "forced"]
SessionStatus = Literal["pending", "completed", "failed"]
RiskLevel = Literal["Low", "Moderate", "High"]


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_type: InputType
    cough_type: CoughType | None = None
    recorded_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def breathing_has_no_cough_type(self) -> "SessionCreate":
        if self.input_type == "breathing" and self.cough_type is not None:
            raise ValueError("cough_type must be omitted for breathing sessions")
        return self


class SessionRead(BaseModel):
    session_id: UUID
    user_id: UUID
    input_type: InputType
    cough_type: CoughType | None
    status: SessionStatus
    risk_score: int | None
    level: RiskLevel | None
    confidence: int | None
    recommendation: str | None
    model_used: str | None
    recorded_at: datetime
    created_at: datetime
    updated_at: datetime


class SessionList(BaseModel):
    sessions: list[SessionRead]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class SessionMetadataWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = Field(default=None, max_length=40)
    fever: bool | None = None
    smoker: bool | None = None
    cough_duration: str | None = Field(default=None, max_length=80)
    night_sweats: bool | None = None
    weight_loss: bool | None = None


class SessionMetadataRead(SessionMetadataWrite):
    session_id: UUID
    created_at: datetime
    updated_at: datetime


class EmbeddingRead(BaseModel):
    session_id: UUID
    embedding_data: list[float]
    embedding_dim: int = Field(gt=0)
    stored_at: datetime

    @model_validator(mode="after")
    def dimension_matches_data(self) -> "EmbeddingRead":
        if len(self.embedding_data) != self.embedding_dim:
            raise ValueError("embedding_dim must match embedding_data length")
        return self


class RiskDistribution(BaseModel):
    Low: int = Field(ge=0)
    Moderate: int = Field(ge=0)
    High: int = Field(ge=0)


class InputTypeBreakdown(BaseModel):
    cough: int = Field(ge=0)
    breathing: int = Field(ge=0)


class RiskTrendPoint(BaseModel):
    date: date
    risk_score: int = Field(ge=0, le=100)


class SessionStats(BaseModel):
    total_screenings: int = Field(ge=0)
    last_screened_at: datetime | None
    average_risk_score: float | None = Field(ge=0, le=100)
    risk_distribution: RiskDistribution
    input_type_breakdown: InputTypeBreakdown
    risk_trend: list[RiskTrendPoint]
