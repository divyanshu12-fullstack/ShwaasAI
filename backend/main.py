"""
ShwaasAI Backend Application Entry Point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routers.auth import router as auth_router
from backend.app.api.routers.analyze import router as analyze_router
from backend.app.api.routers.sessions import router as sessions_router

app = FastAPI(
    title="ShwaasAI Respiratory Health Screening API",
    description=(
        "AI-assisted respiratory health acoustic screening service using Google HeAR "
        "embeddings, dual-head TB screening, respiratory sound pathology classification, "
        "and multimodal clinical symptom fusion."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for React/Vite/PWA frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(analyze_router)


@app.get("/")
def root():
    return {
        "project": "ShwaasAI",
        "description": "AI-Assisted Respiratory Health Screening System",
        "documentation": "/docs",
        "status": "online"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
