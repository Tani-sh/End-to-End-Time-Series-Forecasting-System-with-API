"""
routes.py
=========
FastAPI router with all forecasting endpoints.

Endpoints
---------
GET  /health                 – Liveness check
GET  /states                 – List all trained states
GET  /models                 – Model selection report for all states
GET  /forecast/{state}       – 8-week forecast for a state (path param)
POST /forecast               – Forecast with body (state + custom weeks)
GET  /history/{state}        – Historical sales data for a state
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelReport,
    StateInfo,
    WeeklyPrediction,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Shared state (loaded once at startup via dependency / lifespan)
# ---------------------------------------------------------------------------

_df: pd.DataFrame | None = None
_report: dict | None = None


def set_shared_state(df: pd.DataFrame, report: dict) -> None:
    global _df, _report
    _df = df
    _report = report


def _require_state(state: str) -> dict:
    """Return report entry for a state, raising 404 if missing."""
    if _report is None:
        raise HTTPException(503, "Model report not loaded. Run training first.")
    entry = _report.get(state)
    if entry is None:
        available = sorted(_report.keys())
        raise HTTPException(
            404,
            f"State '{state}' not found. Available states: {available[:10]}..."
            if len(available) > 10
            else f"State '{state}' not found. Available: {available}",
        )
    return entry


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns API liveness status and number of trained states."""
    trained = len(_report) if _report else 0
    return HealthResponse(
        status="ok",
        total_states_trained=trained,
        message=f"Forecasting API is live. {trained} states ready.",
    )


@router.get("/states", response_model=List[str], tags=["Data"])
def list_states():
    """Return the list of all states with trained models."""
    if not _report:
        raise HTTPException(503, "Model report not loaded.")
    return sorted(_report.keys())


@router.get("/models", response_model=ModelReport, tags=["Models"])
def get_model_report():
    """
    Return the best-model selection report for every trained state,
    including validation RMSE, MAPE, and observation counts.
    """
    if not _report:
        raise HTTPException(503, "Model report not loaded.")

    state_infos = []
    for state, rep in sorted(_report.items()):
        best = rep["best_model"]
        m    = rep["metrics"].get(best, {})
        state_infos.append(
            StateInfo(
                state=state,
                best_model=best,
                val_rmse=m.get("rmse"),
                val_mape=m.get("mape"),
                train_observations=rep.get("train_observations"),
                val_observations=rep.get("val_observations"),
                train_start=rep.get("train_start"),
                train_end=rep.get("train_end"),
            )
        )

    return ModelReport(states=state_infos, total_states=len(state_infos))


@router.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def forecast_get(
    state: str,
    weeks: int = Query(8, ge=1, le=52, description="Number of weeks to forecast"),
):
    """
    Generate a sales forecast for the given state (path parameter).

    Returns next `weeks` weeks of predicted weekly sales.
    """
    # Normalise state name
    state = state.strip().title()
    _require_state(state)

    return _run_forecast(state, weeks)


@router.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast_post(request: ForecastRequest):
    """
    Generate a sales forecast using a JSON request body.

    ```json
    { "state": "California", "weeks": 8 }
    ```
    """
    _require_state(request.state)
    return _run_forecast(request.state, request.weeks)


@router.get("/history/{state}", tags=["Data"])
def get_history(
    state: str,
    limit: int = Query(52, ge=1, le=500, description="Max number of historical weeks to return"),
):
    """
    Return the most recent `limit` weeks of historical sales for the state.
    Useful for plotting alongside forecasts.
    """
    state = state.strip().title()
    if _df is None:
        raise HTTPException(503, "Dataset not loaded.")
    if state not in _df["state"].unique():
        raise HTTPException(404, f"State '{state}' not in dataset.")

    from src.data_loader import get_state_series
    from src.preprocessor import fill_missing_dates

    series = get_state_series(_df, state)
    series = fill_missing_dates(series)
    series = series.tail(limit)

    return {
        "state": state,
        "observations": [
            {"date": str(d.date()), "sales": round(float(v), 2)}
            for d, v in zip(series.index, series.values)
        ],
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_forecast(state: str, weeks: int) -> ForecastResponse:
    """Shared forecast logic for GET and POST endpoints."""
    from src.forecaster import generate_forecast

    try:
        result = generate_forecast(state=state, weeks=weeks, df=_df, report=_report)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception("Forecast failed for state=%s", state)
        raise HTTPException(500, f"Forecast error: {exc}")

    return ForecastResponse(
        state=result["state"],
        model_used=result["model_used"],
        val_rmse=result.get("val_rmse"),
        val_mape=result.get("val_mape"),
        train_end=result.get("train_end"),
        forecast_start=result.get("forecast_start"),
        forecast_end=result.get("forecast_end"),
        predictions=[
            WeeklyPrediction(date=p["date"], sales=p["sales"])
            for p in result["predictions"]
        ],
    )
