"""
model_selector.py
=================
Evaluates all trained models on the validation window and selects the best
one per state by lowest RMSE.

Outputs
-------
artifacts/best_models.json — mapping of state → {model, rmse, mape}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.base_model import BaseModel, rmse

logger = logging.getLogger(__name__)


def evaluate_all(
    models: dict[str, BaseModel],
    val_series: pd.Series,
    train_series: pd.Series,
) -> dict[str, dict[str, float]]:
    """
    Evaluate every model in `models` on the validation window.

    Parameters
    ----------
    models       : {model_name: fitted BaseModel instance}
    val_series   : ground-truth sales for the validation period (pd.Series)
    train_series : the portion of the series EXCLUDING val (used as `last_known`)

    Returns
    -------
    {model_name: {'rmse': float, 'mape': float}}
    """
    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        try:
            metrics = model.evaluate(val_series, train_series)
            results[name] = metrics
            logger.info(
                "  [%s] %s → RMSE=%.0f  MAPE=%.2f%%",
                model.state, name, metrics["rmse"], metrics["mape"],
            )
        except Exception as exc:
            logger.error("  [%s] %s evaluation failed: %s", model.state, name, exc)
            results[name] = {"rmse": float("inf"), "mape": float("inf")}

    return results


def select_best(metrics: dict[str, dict[str, float]]) -> str:
    """Return the model name with the lowest RMSE."""
    return min(metrics, key=lambda k: metrics[k]["rmse"])


def save_model_report(
    report: dict[str, Any],
    artifacts_dir: Path,
) -> Path:
    """Persist the best-model report to JSON."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "best_models.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Model report saved to %s", out_path)
    return out_path


def load_model_report(artifacts_dir: Path) -> dict[str, Any]:
    """Load the best-model report from JSON."""
    path = artifacts_dir / "best_models.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No model report found at {path}. Run 'python setup.py' first."
        )
    with open(path) as f:
        return json.load(f)
