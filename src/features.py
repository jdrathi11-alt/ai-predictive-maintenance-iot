"""Leakage-safe preprocessing and time-series feature engineering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .simulator import SENSOR_COLUMNS


REQUIRED_COLUMNS: tuple[str, ...] = (
    "machine_id",
    "timestamp",
    *SENSOR_COLUMNS,
    "failure_next_24h",
)


def engineer_features(
    data: pd.DataFrame,
    rolling_windows: Sequence[int] = (3, 12),
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare telemetry and create features using current or past readings.

    The input is sorted by machine and timestamp before any group operation.
    Lag and rolling features use ``shift(1)`` so even rolling statistics stop
    before the current observation. Initial history is filled with zero rather
    than backfilled, which prevents a future row from leaking into the past.

    Args:
        data: Raw sensor data containing :data:`REQUIRED_COLUMNS`.
        rolling_windows: Positive lookback windows, measured in observations.

    Returns:
        ``(prepared_frame, feature_columns)``. The frame includes identifiers,
        raw sensors, engineered features, and the binary target.

    Raises:
        ValueError: If required columns, timestamps, values, target labels, or
            rolling windows are invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    windows = tuple(int(window) for window in rolling_windows)
    if not windows or any(window < 1 for window in windows):
        raise ValueError("rolling_windows must contain positive integers")

    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values")
    if frame["machine_id"].isna().any():
        raise ValueError("machine_id cannot contain missing values")

    for column in SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{column} contains non-numeric or missing values")

    target = pd.to_numeric(frame["failure_next_24h"], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("failure_next_24h must contain only binary 0/1 values")
    frame["failure_next_24h"] = target.astype(int)
    frame = frame.sort_values(["machine_id", "timestamp"], kind="stable").reset_index(drop=True)

    feature_columns = list(SENSOR_COLUMNS)
    grouped = frame.groupby("machine_id", sort=False)

    for sensor in SENSOR_COLUMNS:
        previous = grouped[sensor].shift(1)
        lag_one_name = f"{sensor}_lag_1"
        lag_three_name = f"{sensor}_lag_3"
        delta_name = f"{sensor}_delta_1"

        frame[lag_one_name] = previous.fillna(0.0)
        frame[lag_three_name] = grouped[sensor].shift(3).fillna(0.0)
        frame[delta_name] = (frame[sensor] - previous).fillna(0.0)
        feature_columns.extend([lag_one_name, lag_three_name, delta_name])

        for window in windows:
            history = grouped[sensor].shift(1)
            rolling_group = history.groupby(frame["machine_id"], sort=False)
            mean_name = f"{sensor}_rolling_mean_{window}"
            std_name = f"{sensor}_rolling_std_{window}"
            frame[mean_name] = rolling_group.transform(
                lambda values: values.rolling(window=window, min_periods=1).mean()
            ).fillna(0.0)
            frame[std_name] = rolling_group.transform(
                lambda values: values.rolling(window=window, min_periods=2).std()
            ).fillna(0.0)
            feature_columns.extend([mean_name, std_name])

    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame, feature_columns


# A descriptive alias for callers that prefer “prepare” terminology.
prepare_features = engineer_features


__all__ = ["REQUIRED_COLUMNS", "engineer_features", "prepare_features"]
