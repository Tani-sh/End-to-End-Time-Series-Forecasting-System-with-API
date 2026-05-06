"""
lstm_model.py
=============
LSTM deep learning model for weekly sales forecasting.

Architecture
------------
Input  : sequence of LOOKBACK weeks of sales (scaled)
Layers : LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
Output : next-week sales (inverse-scaled)

Forecasting strategy
--------------------
Recursive: at each step, the predicted value is appended to the input
window and the oldest value is dropped (sliding window).

Scaling
-------
Each state's series is independently MinMax-scaled to [0, 1].
Predictions are inverse-transformed before returning.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

LOOKBACK = 12       # number of past weeks used as model input
EPOCHS   = 50       # max training epochs
BATCH    = 16


class LSTMModel(BaseModel):
    name = "LSTM"

    def __init__(self, state: str):
        super().__init__(state)
        self._keras_model = None
        self._scaler = None          # sklearn MinMaxScaler

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sequences(values: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create (X, y) sliding-window sequences from a 1-D array."""
        X, y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i - lookback: i])
            y.append(values[i])
        return np.array(X)[..., np.newaxis], np.array(y)  # X: (N, lookback, 1)

    def _build_model(self, lookback: int):
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(shape=(lookback, 1)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, train_series: pd.Series, train_df: pd.DataFrame) -> None:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow import keras
        import tensorflow as tf

        # Suppress TF verbose output
        tf.get_logger().setLevel("ERROR")

        values = train_series.values.reshape(-1, 1).astype(float)

        self._scaler = MinMaxScaler()
        scaled = self._scaler.fit_transform(values).flatten()

        if len(scaled) <= LOOKBACK:
            raise ValueError(
                f"[{self.state}] LSTM needs > {LOOKBACK} samples, got {len(scaled)}"
            )

        X, y = self._make_sequences(scaled, LOOKBACK)

        # Split 80/20 for early stopping
        split = int(len(X) * 0.85)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        self._keras_model = self._build_model(LOOKBACK)
        self._keras_model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=0),
            ],
            verbose=0,
        )
        self._is_fitted = True
        logger.info("[%s] LSTM fitted. Params: %d", self.state, self._keras_model.count_params())

    def predict(self, steps: int, last_known: pd.Series) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        values = last_known.values.reshape(-1, 1).astype(float)
        scaled = self._scaler.transform(values).flatten()

        # Seed the window with the last LOOKBACK observations
        window = list(scaled[-LOOKBACK:])
        predictions_scaled = []

        for _ in range(steps):
            X_in = np.array(window[-LOOKBACK:])[np.newaxis, :, np.newaxis]
            pred_scaled = float(self._keras_model.predict(X_in, verbose=0)[0, 0])
            predictions_scaled.append(pred_scaled)
            window.append(pred_scaled)

        # Inverse transform
        preds_raw = self._scaler.inverse_transform(
            np.array(predictions_scaled).reshape(-1, 1)
        ).flatten()
        preds_raw = np.maximum(preds_raw, 0.0)  # no negative sales

        # Build future DatetimeIndex
        freq = pd.tseries.frequencies.to_offset("W-SAT")
        future_idx = pd.date_range(
            start=last_known.index[-1] + freq,
            periods=steps,
            freq="W-SAT",
        )
        return pd.Series(preds_raw, index=future_idx, name="sales")

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._keras_model.save(str(path / "lstm.keras"))
        with open(path / "lstm_scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

    def load(self, path: Path) -> None:
        from tensorflow import keras
        self._keras_model = keras.models.load_model(str(path / "lstm.keras"))
        with open(path / "lstm_scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)
        self._is_fitted = True
