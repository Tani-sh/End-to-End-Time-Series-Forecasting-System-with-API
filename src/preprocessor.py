"""
preprocessor.py
===============
Handles all feature engineering and train/validation splitting.

Features created
----------------
Lag features   : lag_1, lag_7, lag_30, lag_52
                 (t-1 week, t-7 weeks, t-30 weeks, t-52 weeks/yearly)
Rolling stats  : rolling_mean_4, rolling_std_4, rolling_mean_8, rolling_mean_13
Calendar       : week_of_year, month, quarter, year, day_of_week, is_month_end
Holiday flag   : is_holiday  (US federal holidays via 'holidays' lib)

Train / Val split
-----------------
Last VAL_WEEKS weeks are held out for validation (no data leakage).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VAL_WEEKS = 16          # held-out validation window
MIN_TRAIN_WEEKS = 52    # minimum required training observations


# ---------------------------------------------------------------------------
# Holiday helper (graceful fallback if 'holidays' not installed)
# ---------------------------------------------------------------------------

def _get_us_holidays(years: list[int]) -> set:
    try:
        import holidays as hol
        us = hol.US(years=years)
        return set(us.keys())
    except ImportError:
        logger.warning(
            "'holidays' package not installed – holiday_flag will be all zeros"
        )
        return set()


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def fill_missing_dates(series: pd.Series) -> pd.Series:
    """
    Reindex to a continuous weekly frequency, forward-fill short gaps (≤ 2 weeks),
    then interpolate any remaining NaNs linearly.
    """
    if series.empty:
        return series

    # Ensure weekly frequency anchored to Saturday
    idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="W-SAT")
    series = series.reindex(idx)

    # Forward fill gaps of 1-2 weeks (likely data-collection lag)
    series = series.ffill(limit=2)

    # Linear interpolation for any remaining NaNs
    series = series.interpolate(method="linear")

    # Back-fill leading NaNs (edge case: first values missing)
    series = series.bfill()

    return series


def build_features(series: pd.Series) -> pd.DataFrame:
    """
    Create a feature DataFrame from a weekly sales time series.

    Parameters
    ----------
    series : pd.Series
        Weekly sales with DatetimeIndex.

    Returns
    -------
    pd.DataFrame with all features + target column 'sales'
    """
    df = pd.DataFrame({"sales": series})

    # ----- Lag features (matching assignment: t-1, t-7, t-30, plus t-52 yearly) -----
    df["lag_1"]  = df["sales"].shift(1)   # 1 week ago
    df["lag_7"]  = df["sales"].shift(7)   # 7 weeks ago
    df["lag_30"] = df["sales"].shift(30)  # 30 weeks ago (~7 months)
    df["lag_52"] = df["sales"].shift(52)  # 52 weeks ago (1 year)

    # ----- Rolling statistics (computed on the original series, no leakage) -----
    df["rolling_mean_4"]  = df["sales"].shift(1).rolling(4).mean()
    df["rolling_std_4"]   = df["sales"].shift(1).rolling(4).std()
    df["rolling_mean_8"]  = df["sales"].shift(1).rolling(8).mean()
    df["rolling_mean_13"] = df["sales"].shift(1).rolling(13).mean()

    # ----- Calendar features -----
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["year"]         = df.index.year
    df["day_of_week"]  = df.index.dayofweek
    df["is_month_end"] = df.index.is_month_end.astype(int)

    # ----- Holiday flag -----
    years = list(df.index.year.unique())
    holidays = _get_us_holidays(years)
    df["is_holiday"] = df.index.normalize().map(
        lambda d: 1 if d.date() in holidays else 0
    )

    # Drop rows where lag_52 (the longest lag) is NaN
    df = df.dropna(subset=["lag_52"])  # lag_52 is most restrictive

    return df


def train_val_split(
    df: pd.DataFrame,
    val_weeks: int = VAL_WEEKS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation split (no shuffling).

    Parameters
    ----------
    df        : full feature DataFrame
    val_weeks : number of most-recent weeks to hold out

    Returns
    -------
    (train_df, val_df)
    """
    if len(df) <= val_weeks:
        raise ValueError(
            f"Not enough data to create a validation set "
            f"({len(df)} rows, need > {val_weeks})"
        )
    train = df.iloc[:-val_weeks]
    val   = df.iloc[-val_weeks:]
    logger.debug(
        "Split: train=%d rows (%s->%s), val=%d rows (%s->%s)",
        len(train), train.index[0].date(), train.index[-1].date(),
        len(val),   val.index[0].date(),   val.index[-1].date(),
    )
    return train, val


def preprocess_state(
    series: pd.Series,
    val_weeks: int = VAL_WEEKS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline for a single state.

    Returns
    -------
    train_df  : feature DataFrame for training
    val_df    : feature DataFrame for validation
    full_series : the gap-filled raw series (used by SARIMA/Prophet)
    """
    filled = fill_missing_dates(series)
    features = build_features(filled)
    train_df, val_df = train_val_split(features, val_weeks)
    return train_df, val_df, filled


# ---------------------------------------------------------------------------
# Feature column list (shared across models)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Lag features — t-1, t-7, t-30 (assignment requirement) + t-52 (yearly)
    "lag_1", "lag_7", "lag_30", "lag_52",
    # Rolling statistics
    "rolling_mean_4", "rolling_std_4", "rolling_mean_8", "rolling_mean_13",
    # Calendar features
    "week_of_year", "month", "quarter", "year", "day_of_week",
    "is_month_end", "is_holiday",
]

TARGET_COL = "sales"
