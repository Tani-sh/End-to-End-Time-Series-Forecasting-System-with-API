"""
trainer.py
==========
Orchestrates training of all 4 models for every state (or a single state).

Usage
-----
    # Train all states
    python -m src.trainer

    # Train a single state (fast for debugging)
    python -m src.trainer --state Alabama

    # Skip LSTM (saves ~20 min on CPU)
    python -m src.trainer --skip-lstm

    # Skip SARIMA auto search (faster)
    python -m src.trainer --fast
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# Ensure project root is in path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_excel, get_state_series, list_states
from src.preprocessor import preprocess_state, VAL_WEEKS
from src.models.sarima_model import SarimaModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.model_selector import evaluate_all, select_best, save_model_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJ_ROOT    = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJ_ROOT.parent / "Copy of Forecasting Case- Study.xlsx"
ARTIFACTS    = PROJ_ROOT / "artifacts"
MODELS_DIR   = ARTIFACTS / "models"


# ---------------------------------------------------------------------------
# Single-state training
# ---------------------------------------------------------------------------

def train_state(
    state: str,
    df: pd.DataFrame,
    skip_lstm: bool = False,
    fast_mode: bool = False,
) -> dict:
    """
    Train all models for one state and return the selection report dict.
    Saves model artifacts to MODELS_DIR/{state}/.
    """
    logger.info("=" * 60)
    logger.info("Training state: %s", state)
    t0 = time.time()

    # ── Data preparation ──────────────────────────────────────────────
    raw_series = get_state_series(df, state)

    try:
        train_df, val_df, filled_series = preprocess_state(raw_series)
    except ValueError as exc:
        logger.error("[%s] Preprocessing failed: %s — skipping.", state, exc)
        return {}

    train_series = filled_series.iloc[: -VAL_WEEKS]
    val_series   = filled_series.iloc[-VAL_WEEKS :]

    state_dir = MODELS_DIR / state.replace(" ", "_")
    state_dir.mkdir(parents=True, exist_ok=True)

    # ── Train each model ───────────────────────────────────────────────
    fitted_models: dict = {}
    metrics: dict = {}

    model_classes = [
        ("SARIMA",  lambda: SarimaModel(state, auto_search=not fast_mode)),
        ("Prophet", lambda: ProphetModel(state, tune_hyperparams=not fast_mode)),
        ("XGBoost", lambda: XGBoostModel(state)),
    ]
    if not skip_lstm:
        model_classes.append(("LSTM", lambda: LSTMModel(state)))

    for model_name, model_factory in model_classes:
        logger.info("[%s] Training %s ...", state, model_name)
        t1 = time.time()
        model = model_factory()
        try:
            model.fit(train_series, train_df)
            model.save(state_dir / model_name.lower())
            fitted_models[model_name] = model
            logger.info("[%s] %s done in %.1fs", state, model_name, time.time() - t1)
        except Exception as exc:
            logger.error("[%s] %s training FAILED: %s", state, model_name, exc)
            logger.debug(traceback.format_exc())

    if not fitted_models:
        logger.error("[%s] All models failed — skipping.", state)
        return {}

    # ── Evaluate & select ─────────────────────────────────────────────
    logger.info("[%s] Evaluating models on validation window ...", state)
    metrics = evaluate_all(fitted_models, val_series, train_series)
    best_name = select_best(metrics)

    state_report = {
        "state": state,
        "best_model": best_name,
        "metrics": {k: {mk: round(mv, 2) for mk, mv in v.items()} for k, v in metrics.items()},
        "training_seconds": round(time.time() - t0, 1),
        "train_observations": len(train_series),
        "val_observations": len(val_series),
        "train_start": str(train_series.index[0].date()),
        "train_end": str(train_series.index[-1].date()),
        "val_start": str(val_series.index[0].date()),
        "val_end": str(val_series.index[-1].date()),
    }

    logger.info(
        "[%s] BEST MODEL: %s  (RMSE=%.0f, MAPE=%.2f%%)",
        state,
        best_name,
        metrics[best_name]["rmse"],
        metrics[best_name]["mape"],
    )
    return state_report


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train forecasting models for all states.")
    parser.add_argument("--state", type=str, default=None,
                        help="Train only this state (e.g. 'Alabama')")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training (saves ~20 min on CPU)")
    parser.add_argument("--fast", action="store_true",
                        help="Disable auto-search for SARIMA/Prophet (faster)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the Excel dataset (overrides default)")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else DATA_PATH
    if not data_path.exists():
        logger.error("Dataset not found at %s", data_path)
        sys.exit(1)

    logger.info("Loading dataset from %s", data_path)
    df = load_excel(data_path)
    states = list_states(df)

    if args.state:
        if args.state not in states:
            logger.error("State '%s' not found. Available: %s", args.state, states)
            sys.exit(1)
        states = [args.state]

    logger.info("Will train %d state(s). skip_lstm=%s, fast=%s",
                len(states), args.skip_lstm, args.fast)

    full_report: dict = {}

    try:
        from tqdm import tqdm
        state_iter = tqdm(states, desc="States")
    except ImportError:
        state_iter = states

    for state in state_iter:
        report = train_state(state, df, skip_lstm=args.skip_lstm, fast_mode=args.fast)
        if report:
            full_report[state] = report

    # Save consolidated report
    report_path = save_model_report(full_report, ARTIFACTS)
    logger.info("All done! Report: %s", report_path)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'State':<22} {'Best Model':<12} {'RMSE':>14} {'MAPE':>8}")
    print("-" * 70)
    for state, rep in sorted(full_report.items()):
        best = rep["best_model"]
        rms  = rep["metrics"][best]["rmse"]
        mpe  = rep["metrics"][best]["mape"]
        print(f"{state:<22} {best:<12} {rms:>14,.0f} {mpe:>7.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
