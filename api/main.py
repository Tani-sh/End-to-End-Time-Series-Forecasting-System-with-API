"""
main.py
=======
FastAPI application entry-point.

Startup
-------
On startup the app:
  1. Reads the model selection report (artifacts/best_models.json)
  2. Loads the full dataset into memory (shared across requests)
  3. Registers all API routes

Run
---
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import router, set_shared_state
from src.data_loader import load_excel
from src.model_selector import load_model_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJ_ROOT.parent / "Copy of Forecasting Case- Study.xlsx"
ARTIFACTS = PROJ_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Lifespan: load data once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load dataset and model report into shared memory at startup."""
    logger.info("Starting up — loading dataset and model report ...")

    try:
        df = load_excel(DATA_PATH)
        logger.info("Dataset loaded: %d rows, %d states", len(df), df["state"].nunique())
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        df = None

    try:
        report = load_model_report(ARTIFACTS)
        logger.info("Model report loaded: %d states trained", len(report))
    except FileNotFoundError:
        logger.warning(
            "No model report found at %s/best_models.json. "
            "Run 'python -m src.trainer' to train models first.",
            ARTIFACTS,
        )
        report = {}

    set_shared_state(df, report)
    logger.info("API ready.")

    yield   # ← application runs here

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Production-grade REST API for weekly beverage sales forecasting across US states.\n\n"
        "Models: **SARIMA · Prophet · XGBoost · LSTM**\n\n"
        "Auto-selects the best model per state based on validation RMSE."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes under /api/v1
app.include_router(router, prefix="/api/v1")

# Root redirect to docs
@app.get("/", include_in_schema=False)
def root():
    return JSONResponse(
        {"message": "Sales Forecasting API v1.0", "docs": "/docs", "api": "/api/v1"}
    )
