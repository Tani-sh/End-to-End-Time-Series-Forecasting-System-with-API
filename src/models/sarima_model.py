"""
sarima_model.py
===============
SARIMA / ARIMA forecasting model using statsmodels SARIMAX.

Strategy
--------
1. Run a coarse AIC grid search over (p,d,q)(P,D,Q) with s=52 (weekly seasonality).
2. Fit the best SARIMAX specification on the training series.
3. For forecasting, re-fit with `apply()` on the full (train+val) series then extend.

The search space is intentionally small to keep training fast:
    p,q  ∈ {0,1,2}
    d    ∈ {0,1}
    P,Q  ∈ {0,1}
    D    ∈ {0,1}
    s    = 52  (weekly)
"""

from __future__ import annotations

import logging
import pickle
import warnings
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# Suppress convergence/numerical warnings from statsmodels
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SarimaModel(BaseModel):
    name = "SARIMA"

    # Default fallback order if grid search fails
    _DEFAULT_ORDER = (1, 1, 1)
    _DEFAULT_SEASONAL = (1, 0, 1, 52)

    def __init__(self, state: str, auto_search: bool = True):
        super().__init__(state)
        self.auto_search = auto_search
        self._order: Tuple[int, int, int] = self._DEFAULT_ORDER
        self._seasonal_order: Tuple[int, int, int, int] = self._DEFAULT_SEASONAL
        self._fitted_model = None          # SARIMAXResults object

    # ------------------------------------------------------------------
    # AIC grid search
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_orders():
        """Yield (p,d,q, P,D,Q,s) tuples."""
        for p, d, q, P, D, Q in product(
            [0, 1, 2],  # p
            [0, 1],     # d
            [0, 1, 2],  # q
            [0, 1],     # P
            [0, 1],     # D
            [0, 1],     # Q
        ):
            if p + q == 0 and P + Q == 0:
                continue          # trivial model
            yield (p, d, q), (P, D, Q, 52)

    def _grid_search(self, series: pd.Series) -> Tuple[Tuple, Tuple]:
        """Return (order, seasonal_order) with lowest AIC."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        best_aic = np.inf
        best_order = self._DEFAULT_ORDER
        best_seasonal = self._DEFAULT_SEASONAL

        for order, seasonal_order in self._candidate_orders():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=50)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = order
                    best_seasonal = seasonal_order
            except Exception:
                pass

        logger.info(
            "[%s] SARIMA best order=%s seasonal=%s AIC=%.1f",
            self.state, best_order, best_seasonal, best_aic,
        )
        return best_order, best_seasonal

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, train_series: pd.Series, train_df: pd.DataFrame) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Use only the raw series (SARIMA ignores exogenous features here)
        series = train_series.astype(float)

        if self.auto_search:
            try:
                self._order, self._seasonal_order = self._grid_search(series)
            except Exception as exc:
                logger.warning(
                    "[%s] SARIMA grid search failed (%s). Using defaults.", self.state, exc
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = SARIMAX(
                series,
                order=self._order,
                seasonal_order=self._seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)

        self._fitted_model = result
        self._is_fitted = True
        logger.info("[%s] SARIMA fitted. AIC=%.1f", self.state, result.aic)

    def predict(self, steps: int, last_known: pd.Series) -> pd.Series:
        """Forecast `steps` weeks beyond the last date in `last_known`."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        # Re-apply to full known series to get a fresh state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self._fitted_model.apply(last_known.astype(float))
            forecast = applied.forecast(steps=steps)

        # Build DatetimeIndex for the forecast
        last_date = last_known.index[-1]
        future_idx = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset("W-SAT"),
            periods=steps,
            freq="W-SAT",
        )
        return pd.Series(forecast.values, index=future_idx, name="sales")

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "sarima.pkl", "wb") as f:
            pickle.dump(
                {
                    "order": self._order,
                    "seasonal_order": self._seasonal_order,
                    "fitted_model": self._fitted_model,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with open(path / "sarima.pkl", "rb") as f:
            data = pickle.load(f)
        self._order = data["order"]
        self._seasonal_order = data["seasonal_order"]
        self._fitted_model = data["fitted_model"]
        self._is_fitted = True
