from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response, status

from app.core.config import Settings, get_settings
from app.core.security import AuthenticatedUser, get_current_user
from app.models.session import InputType, SessionCreate, SessionList, SessionMetadataRead, SessionMetadataWrite, SessionRead
from app.services.metadata_service import MetadataService
from app.services.session_service import SessionService


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    details: SessionCreate,
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> SessionRead:
    row = await SessionService(settings, user.access_token).create(user.id, details)
    return SessionRead.model_validate(row)


@router.get("", response_model=SessionList)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    input_type: InputType | None = Query(default=None),
    sort: Literal["newest", "oldest"] = Query(default="newest"),
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> SessionList:
    rows = await SessionService(settings, user.access_token).list(user.id, limit, offset, input_type, sort)
    return SessionList(sessions=[SessionRead.model_validate(row) for row in rows], limit=limit, offset=offset)


@router.get("/{session_id}", response_model=SessionRead)
async def get_session(
    session_id: UUID,
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> SessionRead:
    row = await SessionService(settings, user.access_token).get(user.id, session_id)
    return SessionRead.model_validate(row)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> Response:
    await SessionService(settings, user.access_token).delete(user.id, session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{session_id}/metadata", response_model=SessionMetadataRead)
async def get_session_metadata(
    session_id: UUID,
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> SessionMetadataRead:
    row = await MetadataService(settings, user.access_token).get(user.id, session_id)
    return SessionMetadataRead.model_validate(row)


@router.put("/{session_id}/metadata", response_model=SessionMetadataRead)
async def replace_session_metadata(
    session_id: UUID,
    metadata: SessionMetadataWrite,
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> SessionMetadataRead:
    row = await MetadataService(settings, user.access_token).replace(user.id, session_id, metadata)
    return SessionMetadataRead.model_validate(row)
