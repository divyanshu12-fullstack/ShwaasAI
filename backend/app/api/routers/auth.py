from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import Settings, get_settings
from backend.app.core.security import AuthenticatedUser, get_current_user
from backend.app.models.profile import AuthTestResponse, Profile, ProfileUpdate
from backend.app.services.profile_service import ProfileService


router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/test", response_model=AuthTestResponse)
async def test_auth(user: AuthenticatedUser = Depends(get_current_user)) -> AuthTestResponse:
    return AuthTestResponse(authenticated=True, user_id=user.id)


@router.get("/me", response_model=Profile)
async def get_me(
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> Profile:
    return Profile.model_validate(await ProfileService(settings, user.access_token).get(user.id))


@router.patch("/me", response_model=Profile)
async def update_me(
    changes: ProfileUpdate,
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> Profile:
    if "display_name" not in changes.model_fields_set:
        raise HTTPException(status_code=422, detail="display_name is required")
    return Profile.model_validate(await ProfileService(settings, user.access_token).update(user.id, changes.display_name))
