from dataclasses import dataclass
from uuid import UUID

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings


bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    id: UUID
    access_token: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    token = credentials.credentials
    # Supabase Auth verifies both asymmetric and legacy HS256 access tokens.
    # Its /user endpoint is the documented verification path for legacy keys.
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{settings.supabase_url}/auth/v1/user",
                headers={"apikey": settings.publishable_key, "Authorization": f"Bearer {token}"},
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc

    if response.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Invalid or expired access token", headers={"WWW-Authenticate": "Bearer"})
    if response.status_code != 200:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    try:
        user_id = UUID(response.json()["id"])
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Invalid authentication service response") from exc
    return AuthenticatedUser(id=user_id, access_token=token)
