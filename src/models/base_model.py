"""
base_model.py
=============
Abstract base class that all forecasting models must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (returns %)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


class BaseModel(ABC):
    """
    Every forecasting model must:
      - implement `fit`, `predict`, `save`, `load`
      - expose `name` (class-level str)
    """

    name: str = "base"

    def __init__(self, state: str):
        self.state = state
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, train_series: pd.Series, train_df: pd.DataFrame) -> None:
        """
        Fit the model.

        Parameters
        ----------
        train_series : pd.Series
            Raw weekly sales (DatetimeIndex) — used by stat models.
        train_df     : pd.DataFrame
            Feature DataFrame with columns from preprocessor.FEATURE_COLS
            plus 'sales' target — used by ML/DL models.
        """

    @abstractmethod
    def predict(self, steps: int, last_known: pd.Series) -> pd.Series:
        """
        Generate future predictions.

        Parameters
        ----------
        steps      : number of future weeks to forecast
        last_known : full gap-filled historical series (DatetimeIndex)

        Returns
        -------
        pd.Series with DatetimeIndex covering the forecast horizon
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artifacts to `path` (a directory)."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model artifacts from `path` (a directory)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        val_series: pd.Series,
        last_known: pd.Series,
    ) -> dict[str, float]:
        """
        Run inference on the validation window and compute metrics.

        Parameters
        ----------
        val_series  : ground-truth sales for the validation period
        last_known  : full series EXCLUDING the validation period
                      (to simulate real forecasting)

        Returns
        -------
        dict with keys: rmse, mape
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name}: model must be fitted before evaluate()")

        steps = len(val_series)
        preds = self.predict(steps=steps, last_known=last_known)

        # Align by index in case of minor offset
        y_true = val_series.values
        y_pred = preds.values[:steps]

        return {
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state={self.state!r}, fitted={self._is_fitted})"
