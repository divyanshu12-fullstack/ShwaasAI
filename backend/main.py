from fastapi import FastAPI

from app.api.routers.auth import router as auth_router


app = FastAPI(title="ShwaasAI API", version="0.1.0")
app.include_router(auth_router, prefix="/api/v1")
