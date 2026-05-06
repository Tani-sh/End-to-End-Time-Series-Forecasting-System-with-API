"""
forecaster.py
=============
Inference module — loads the best saved model for a state and generates
next-N-week predictions.

This is used both by the REST API and as a standalone CLI tool.

Usage (CLI)
-----------
    python -m src.forecaster --state Alabama --weeks 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_excel, get_state_series
from src.preprocessor import fill_missing_dates
from src.model_selector import load_model_report

logger = logging.getLogger(__name__)

PROJ_ROOT  = Path(__file__).resolve().parent.parent
DATA_PATH  = PROJ_ROOT.parent / "Copy of Forecasting Case- Study.xlsx"
ARTIFACTS  = PROJ_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"


def _load_best_model(state: str, report: dict) -> object:
    """Instantiate and load the best saved model for this state."""
    state_report = report.get(state)
    if state_report is None:
        raise KeyError(f"No trained model found for state '{state}'.")

    best_name = state_report["best_model"]
    state_dir = MODELS_DIR / state.replace(" ", "_") / best_name.lower()

    if best_name == "SARIMA":
        from src.models.sarima_model import SarimaModel
        m = SarimaModel(state)
    elif best_name == "Prophet":
        from src.models.prophet_model import ProphetModel
        m = ProphetModel(state)
    elif best_name == "XGBoost":
        from src.models.xgboost_model import XGBoostModel
        m = XGBoostModel(state)
    elif best_name == "LSTM":
        from src.models.lstm_model import LSTMModel
        m = LSTMModel(state)
    else:
        raise ValueError(f"Unknown model type: {best_name}")

    m.load(state_dir)
    return m, best_name, state_report


def generate_forecast(
    state: str,
    weeks: int = 8,
    df: Optional[pd.DataFrame] = None,
    report: Optional[dict] = None,
) -> dict:
    """
    Generate a forecast for `state` for the next `weeks` weeks.

    Parameters
    ----------
    state  : US state name (must match dataset)
    weeks  : number of weeks to forecast (default: 8)
    df     : pre-loaded full DataFrame (avoids re-reading Excel each call)
    report : pre-loaded best_models.json (avoids re-reading each call)

    Returns
    -------
    {
        "state": str,
        "model_used": str,
        "val_rmse": float,
        "val_mape": float,
        "forecast_start": str,
        "forecast_end": str,
        "predictions": [{"date": "YYYY-MM-DD", "sales": float}, ...]
    }
    """
    if report is None:
        report = load_model_report(ARTIFACTS)

    model, best_name, state_report = _load_best_model(state, report)

    if df is None:
        df = load_excel(DATA_PATH)

    raw_series  = get_state_series(df, state)
    full_series = fill_missing_dates(raw_series)

    preds = model.predict(steps=weeks, last_known=full_series)

    predictions = [
        {"date": str(d.date()), "sales": round(float(v), 2)}
        for d, v in zip(preds.index, preds.values)
    ]

    best_metrics = state_report["metrics"].get(best_name, {})

    return {
        "state": state,
        "model_used": best_name,
        "val_rmse": best_metrics.get("rmse"),
        "val_mape": best_metrics.get("mape"),
        "train_end": state_report.get("train_end"),
        "forecast_start": predictions[0]["date"] if predictions else None,
        "forecast_end":   predictions[-1]["date"] if predictions else None,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate sales forecast for a state.")
    parser.add_argument("--state", required=True, help="US state name")
    parser.add_argument("--weeks", type=int, default=8, help="Weeks to forecast")
    parser.add_argument("--data", type=str, default=None, help="Path to Excel dataset")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else DATA_PATH
    df = load_excel(data_path)

    result = generate_forecast(state=args.state, weeks=args.weeks, df=df)

    print(f"\nForecast for {result['state']} using {result['model_used']}")
    print(f"Validation RMSE: {result['val_rmse']:,.0f}  MAPE: {result['val_mape']:.2f}%")
    print(f"\n{'Date':<14} {'Predicted Sales':>20}")
    print("-" * 35)
    for p in result["predictions"]:
        print(f"{p['date']:<14} {p['sales']:>20,.0f}")


if __name__ == "__main__":
    main()
