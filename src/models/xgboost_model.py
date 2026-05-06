"""
xgboost_model.py
================
XGBoost regression model for weekly sales forecasting.

Feature set (from preprocessor.FEATURE_COLS)
--------------------------------------------
  lag_1, lag_4, lag_13, lag_52
  rolling_mean_4, rolling_std_4, rolling_mean_8, rolling_mean_13
  week_of_year, month, quarter, year, day_of_week, is_month_end, is_holiday

Forecasting strategy (recursive)
---------------------------------
XGBoost is a point-in-time model — it cannot natively extrapolate.
For multi-step forecasting we use a **recursive** approach:
  1. Predict week t+1 using lag/rolling features derived from known history.
  2. Append the prediction to the known series.
  3. Re-derive lag/rolling features and predict week t+2.
  4. Repeat for `steps` weeks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.preprocessor import FEATURE_COLS, TARGET_COL, build_features

logger = logging.getLogger(__name__)


def _make_one_step_features(series: pd.Series) -> pd.DataFrame:
    """
    Build features for the NEXT step given the current history.
    Returns a single-row DataFrame using the last row of build_features().
    """
    feat_df = build_features(series)
    if feat_df.empty:
        raise ValueError("Not enough history to build features for next step.")
    return feat_df.iloc[[-1]][FEATURE_COLS]


class XGBoostModel(BaseModel):
    name = "XGBoost"

    def __init__(self, state: str):
        super().__init__(state)
        self._model = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, train_series: pd.Series, train_df: pd.DataFrame) -> None:
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit

        X = train_df[FEATURE_COLS].values
        y = train_df[TARGET_COL].values

        # Time-series cross-validation to find good early-stopping round
        tscv = TimeSeriesSplit(n_splits=3)
        val_rounds: list[int] = []

        for fold_train_idx, fold_val_idx in tscv.split(X):
            X_tr, X_val = X[fold_train_idx], X[fold_val_idx]
            y_tr, y_val = y[fold_train_idx], y[fold_val_idx]

            m = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                early_stopping_rounds=30,
            )
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_rounds.append(m.best_iteration)

        best_n = int(np.median(val_rounds)) if val_rounds else 200
        logger.info("[%s] XGBoost best n_estimators=%d", self.state, best_n)

        # Final model on full training set
        self._model = xgb.XGBRegressor(
            n_estimators=best_n,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self._model.fit(X, y)
        self._is_fitted = True
        logger.info("[%s] XGBoost fitted on %d samples.", self.state, len(y))

    def predict(self, steps: int, last_known: pd.Series) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        history = last_known.copy()
        predictions = []
        freq = pd.tseries.frequencies.to_offset("W-SAT")

        for _ in range(steps):
            try:
                X_next = _make_one_step_features(history)
                pred = float(self._model.predict(X_next.values)[0])
            except Exception as exc:
                logger.warning("[%s] XGBoost step failed (%s), using last value.", self.state, exc)
                pred = float(history.iloc[-1])

            pred = max(pred, 0.0)   # sales can't be negative
            next_date = history.index[-1] + freq
            history[next_date] = pred
            predictions.append((next_date, pred))

        dates, values = zip(*predictions)
        return pd.Series(list(values), index=pd.DatetimeIndex(list(dates)), name="sales")

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / "xgboost.json"))

    def load(self, path: Path) -> None:
        import xgboost as xgb
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(path / "xgboost.json"))
        self._is_fitted = True
