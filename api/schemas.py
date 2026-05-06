"""
schemas.py
==========
Pydantic v2 request / response models for the Forecasting API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    state: str = Field(..., description="US state name (e.g. 'California')")
    weeks: int = Field(8, ge=1, le=52, description="Number of weeks to forecast (1–52)")

    @field_validator("state")
    @classmethod
    def normalise_state(cls, v: str) -> str:
        return v.strip().title()


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class WeeklyPrediction(BaseModel):
    date: str   = Field(..., description="ISO date string (YYYY-MM-DD) — Saturday of that week")
    sales: float = Field(..., description="Predicted sales in dollars")


class ForecastResponse(BaseModel):
    state:        str   = Field(..., description="US state name")
    model_used:   str   = Field(..., description="Model selected as best for this state")
    val_rmse:     Optional[float] = Field(None, description="RMSE on validation window")
    val_mape:     Optional[float] = Field(None, description="MAPE (%) on validation window")
    train_end:    Optional[str]   = Field(None, description="Last date in training data")
    forecast_start: Optional[str] = Field(None, description="First forecast date")
    forecast_end:   Optional[str] = Field(None, description="Last forecast date")
    predictions:  List[WeeklyPrediction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Supporting
# ---------------------------------------------------------------------------

class StateInfo(BaseModel):
    state: str
    best_model: str
    val_rmse: Optional[float] = None
    val_mape: Optional[float] = None
    train_observations: Optional[int] = None
    val_observations: Optional[int] = None
    train_start: Optional[str] = None
    train_end: Optional[str] = None


class ModelReport(BaseModel):
    states: List[StateInfo]
    total_states: int


class HealthResponse(BaseModel):
    status: str
    total_states_trained: int
    message: str
