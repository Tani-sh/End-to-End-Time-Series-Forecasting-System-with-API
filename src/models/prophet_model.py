"""
prophet_model.py
================
Facebook / Meta Prophet forecasting model.

Configuration
-------------
- Weekly seasonality     : enabled (period=52.18)
- Yearly seasonality     : enabled
- US holidays            : added via built-in holiday support
- changepoint_prior_scale: tuned via a lightweight cross-validation search
- Growth model           : 'linear'  (sales show no saturating ceiling)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    name = "Prophet"

    def __init__(self, state: str, tune_hyperparams: bool = True):
        super().__init__(state)
        self.tune_hyperparams = tune_hyperparams
        self._model = None          # fitted Prophet object
        self._changepoint_prior: float = 0.05

    # ------------------------------------------------------------------
    # Lightweight hyperparameter search
    # ------------------------------------------------------------------

    def _tune(self, df_prophet: pd.DataFrame) -> float:
        """
        Try a small set of changepoint_prior_scale values on an initial
        train/test split (80/20 chronological).
        Returns the value with the lowest RMSE.
        """
        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics
        import numpy as np

        candidates = [0.001, 0.01, 0.1, 0.5]
        split_idx = int(len(df_prophet) * 0.8)
        train_part = df_prophet.iloc[:split_idx]
        test_part  = df_prophet.iloc[split_idx:]
        n_test = len(test_part)

        best_rmse = float("inf")
        best_cp = 0.05

        for cp in candidates:
            try:
                m = Prophet(
                    changepoint_prior_scale=cp,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                )
                m.add_country_holidays(country_name="US")
                m.fit(train_part)
                future = m.make_future_dataframe(periods=n_test, freq="W")
                forecast = m.predict(future)
                preds = forecast.tail(n_test)["yhat"].values
                actual = test_part["y"].values
                rms = float(
                    ((actual - preds) ** 2).mean() ** 0.5
                )
                if rms < best_rmse:
                    best_rmse = rms
                    best_cp = cp
            except Exception as exc:
                logger.debug("Prophet tune cp=%.3f failed: %s", cp, exc)

        logger.info("[%s] Prophet best changepoint_prior=%.4f", self.state, best_cp)
        return best_cp

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    @staticmethod
    def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
        """Convert pd.Series → Prophet-style df with columns [ds, y]."""
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        return df

    def fit(self, train_series: pd.Series, train_df: pd.DataFrame) -> None:
        from prophet import Prophet

        df_prophet = self._to_prophet_df(train_series)

        if self.tune_hyperparams and len(df_prophet) > 60:
            self._changepoint_prior = self._tune(df_prophet)

        self._model = Prophet(
            changepoint_prior_scale=self._changepoint_prior,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        self._model.add_country_holidays(country_name="US")
        self._model.fit(df_prophet)
        self._is_fitted = True
        logger.info("[%s] Prophet fitted on %d observations.", self.state, len(df_prophet))

    def predict(self, steps: int, last_known: pd.Series) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        # Re-fit on the full known series for a fresh forecast
        from prophet import Prophet

        df_prophet = self._to_prophet_df(last_known)
        m = Prophet(
            changepoint_prior_scale=self._changepoint_prior,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        m.add_country_holidays(country_name="US")
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=steps, freq="W")
        forecast = m.predict(future)
        result = forecast.tail(steps)[["ds", "yhat"]].copy()
        result["ds"] = pd.to_datetime(result["ds"])
        series = pd.Series(
            result["yhat"].values,
            index=pd.DatetimeIndex(result["ds"]),
            name="sales",
        )
        return series

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "prophet.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "changepoint_prior": self._changepoint_prior,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with open(path / "prophet.pkl", "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._changepoint_prior = data["changepoint_prior"]
        self._is_fitted = True
