from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Profile(BaseModel):
    user_id: UUID
    display_name: str | None
    created_at: datetime
    updated_at: datetime


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=80)


class AuthTestResponse(BaseModel):
    authenticated: bool
    user_id: UUID
