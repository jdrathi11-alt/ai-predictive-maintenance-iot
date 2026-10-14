"""Deterministic synthetic sensor data for the predictive-maintenance demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


SENSOR_COLUMNS: tuple[str, ...] = (
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "current_a",
    "rpm",
    "humidity_pct",
)


def generate_sensor_data(
    n_machines: int = 6,
    n_cycles: int = 240,
    seed: int = 42,
    start_time: str = "2024-01-01T00:00:00",
    frequency: str = "h",
    warning_horizon: int = 24,
) -> pd.DataFrame:
    """Generate learnable, machine-specific IoT time-series data.

    Every machine receives one deterministic failure event. Sensor degradation
    starts before the event and grows during a 48-hour warning period. The
    target is one when a failure is due in the next ``warning_horizon`` cycles.
    No external services or random state are used, so the same arguments always
    produce the same frame.

    Args:
        n_machines: Number of independent machines to simulate.
        n_cycles: Number of hourly (or requested-frequency) observations per
            machine.
        seed: Seed for all generated noise.
        start_time: Timestamp accepted by :func:`pandas.date_range`.
        frequency: Pandas date frequency, hourly by default.
        warning_horizon: Number of future cycles represented by the target.

    Returns:
        A sorted DataFrame with raw sensor readings and
        ``failure_next_24h`` as a binary target.

    Raises:
        ValueError: If the requested simulation is too small or invalid.
    """
    if n_machines < 1:
        raise ValueError("n_machines must be at least 1")
    if n_cycles <= warning_horizon:
        raise ValueError("n_cycles must be greater than warning_horizon")
    if warning_horizon < 1:
        raise ValueError("warning_horizon must be positive")

    timestamps = pd.date_range(start=start_time, periods=n_cycles, freq=frequency)
    time_index = np.arange(n_cycles, dtype=float)
    rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    warning_window = max(48, warning_horizon * 2)

    for machine_number in range(n_machines):
        # Stagger events so a grouped machine split still sees positive and
        # negative examples for machines with different operating phases.
        base_failure = int(n_cycles * 0.72)
        stagger = (machine_number - (n_machines - 1) / 2.0) * 3.0
        failure_index = int(np.clip(base_failure + stagger, warning_horizon + 1, n_cycles - 1))

        remaining = failure_index - np.arange(n_cycles)
        in_warning_window = remaining > 0
        severity = np.where(
            in_warning_window,
            np.clip((warning_window - remaining) / warning_window, 0.0, 1.0),
            0.0,
        )
        target = ((remaining > 0) & (remaining <= warning_horizon)).astype(int)

        phase = machine_number * 0.63
        machine_offset = machine_number * 0.45

        # Shared time-of-day patterns make the data feel like telemetry while
        # the degradation terms make the upcoming failure learnable.
        daily_sine = np.sin(2 * np.pi * time_index / 24.0 + phase)
        daily_cosine = np.cos(2 * np.pi * time_index / 24.0 + phase)
        noise = rng.normal(0.0, 1.0, size=(n_cycles, len(SENSOR_COLUMNS)))

        machine_frame = pd.DataFrame(
            {
                "machine_id": f"machine_{machine_number + 1:03d}",
                "timestamp": timestamps,
                "temperature_c": (
                    68.0
                    + machine_offset
                    + 2.0 * daily_sine
                    + 0.35 * time_index / n_cycles
                    + 10.0 * severity
                    + 0.45 * noise[:, 0]
                ),
                "vibration_mm_s": (
                    1.15
                    + 0.10 * daily_cosine
                    + 3.2 * severity
                    + 0.08 * noise[:, 1]
                ),
                "pressure_bar": (
                    5.1
                    + 0.12 * daily_cosine
                    - 0.85 * severity
                    + 0.06 * noise[:, 2]
                ),
                "current_a": (
                    8.0
                    + machine_offset * 0.1
                    + 0.30 * daily_sine
                    + 1.8 * severity
                    + 0.10 * noise[:, 3]
                ),
                "rpm": (
                    1_800.0
                    + 32.0 * daily_sine
                    - 190.0 * severity
                    + 7.0 * noise[:, 4]
                ),
                "humidity_pct": (
                    45.0
                    + 4.5 * daily_cosine
                    + 8.0 * severity
                    + 0.35 * noise[:, 5]
                ),
                "failure_next_24h": target,
            }
        )
        rows.append(machine_frame)

    data = pd.concat(rows, ignore_index=True)
    return data.sort_values(["machine_id", "timestamp"], kind="stable").reset_index(drop=True)
