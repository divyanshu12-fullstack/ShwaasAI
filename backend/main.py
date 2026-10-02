from fastapi import FastAPI

from app.api.routers.auth import router as auth_router
from app.api.routers.sessions import router as sessions_router


app = FastAPI(title="ShwaasAI API", version="0.1.0")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")
