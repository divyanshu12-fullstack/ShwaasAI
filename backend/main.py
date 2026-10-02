"""
ShwaasAI Backend Application Entry Point.
"""

from fastapi import FastAPI

from backend.app.core.deployment import DeploymentMiddleware, DeploymentSettings
from backend.app.api.routers.auth import router as auth_router
from backend.app.api.routers.analyze import router as analyze_router
from backend.app.api.routers.sessions import router as sessions_router
from backend.app.api.routers.system import router as system_router

def create_app(deployment: DeploymentSettings | None = None) -> FastAPI:
    deployment = deployment or DeploymentSettings.from_env()
    application = FastAPI(
        title="ShwaasAI Respiratory Health Screening API",
        description=(
            "Research respiratory sound screening API. The bundled classifier heads use a "
            "512-value acoustic feature extractor and have not been clinically validated. "
            "HeAR inference requires separately trained compatible classifier heads."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(DeploymentMiddleware, settings=deployment)

    application.include_router(auth_router, prefix="/api/v1")
    application.include_router(sessions_router, prefix="/api/v1")
    application.include_router(analyze_router)
    application.include_router(system_router)

    @application.get("/")
    def root():
        return {
            "project": "ShwaasAI",
            "description": "AI-Assisted Respiratory Health Screening System",
            "documentation": "/docs",
            "status": "online",
        }

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
