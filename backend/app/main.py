"""
E-Dermatologist Backend Server (FastAPI)
AI-Assisted Skin Disease Screening and Clinical Triage
"""

import sys
from pathlib import Path

# Add backend/app and ml/src to sys.path
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent.parent
ML_SRC = BASE_DIR / "ml" / "src"

for p in [str(APP_DIR), str(ML_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import config
from routers.predict import router as predict_router

app = FastAPI(
    title="E-Dermatologist Screening API",
    description="Clinical decision-support API for smartphone + 3D spacer skin screening",
    version=config.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for mobile browsers, local dev server, and tunneling domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include primary endpoints (/api/v1/health, /api/v1/classes, /api/v1/predict)
app.include_router(predict_router)

# Mount frontend build if it exists
DIST_DIR = BASE_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "name": "E-Dermatologist Screening API",
            "version": config.version,
            "modality": config.modality,
            "docs": "/docs",
            "health": "/api/v1/health",
            "classes": "/api/v1/classes"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
