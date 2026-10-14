"""Small, file-based plots for inspecting predictive-maintenance results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_sensor_trends(
    frame: pd.DataFrame,
    output_dir: str | Path,
    machine_id: str | None = None,
) -> Path:
    """Save sensor trend charts and mark rows where failure is imminent."""
    destination = _prepare_output_dir(output_dir) / "sensor_trends.png"
    plot_frame = frame if machine_id is None else frame[frame["machine_id"] == machine_id]
    if plot_frame.empty:
        raise ValueError("no rows available for sensor trend plot")

    sensors = [
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "current_a",
        "rpm",
        "humidity_pct",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for axis, sensor in zip(axes.flat, sensors):
        for current_machine, group in plot_frame.groupby("machine_id", sort=True):
            axis.plot(group["timestamp"], group[sensor], linewidth=1.1, label=current_machine)
            warning = group[group["failure_next_24h"] == 1]
            if not warning.empty:
                axis.scatter(
                    warning["timestamp"],
                    warning[sensor],
                    s=9,
                    color="crimson",
                    alpha=0.75,
                    label="warning" if current_machine == plot_frame["machine_id"].iloc[0] else None,
                )
        axis.set_title(sensor.replace("_", " ").title())
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Timestamp")
    axes[-1, 1].set_xlabel("Timestamp")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle("Synthetic IoT Sensor Trends (red = failure within 24h)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return destination


def save_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Save a heatmap of the selected model's confusion matrix."""
    destination = _prepare_output_dir(output_dir) / "confusion_matrix.png"
    values = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        values,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No failure", "Failure"],
        yticklabels=["No failure", "Failure"],
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Failure Alert Confusion Matrix")
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return destination


def save_feature_importance(
    model: Any,
    feature_columns: list[str],
    output_dir: str | Path,
    top_n: int = 20,
) -> Path | None:
    """Save feature importance when the chosen estimator exposes it."""
    importance = getattr(model, "feature_importances_", None)
    if importance is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        importance = getattr(classifier, "coef_", None)
        if importance is not None:
            importance = np.abs(np.asarray(importance)).ravel()
    if importance is None:
        return None

    importance = np.asarray(importance).ravel()
    if len(importance) != len(feature_columns):
        return None
    count = min(top_n, len(feature_columns))
    order = np.argsort(importance)[-count:]
    destination = _prepare_output_dir(output_dir) / "feature_importance.png"
    fig, axis = plt.subplots(figsize=(9, max(4, count * 0.3)))
    axis.barh(np.asarray(feature_columns)[order], importance[order], color="#277da1")
    axis.set_title("Most Important Failure-Prediction Features")
    axis.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return destination


def save_probability_timeline(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    output_dir: str | Path,
    threshold: float = 0.5,
) -> Path:
    """Save holdout failure probabilities over time, grouped by machine."""
    destination = _prepare_output_dir(output_dir) / "probability_timeline.png"
    plot_frame = frame[["machine_id", "timestamp", "failure_next_24h"]].copy()
    plot_frame["failure_probability"] = np.asarray(probabilities)
    fig, axis = plt.subplots(figsize=(13, 6))
    for machine_id, group in plot_frame.groupby("machine_id", sort=True):
        axis.plot(
            group["timestamp"],
            group["failure_probability"],
            marker=".",
            linewidth=1,
            label=machine_id,
        )
    axis.axhline(threshold, color="crimson", linestyle="--", label=f"threshold ({threshold:g})")
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Failure probability")
    axis.set_xlabel("Timestamp")
    axis.set_title("Holdout Failure Alert Probabilities")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return destination


__all__ = [
    "save_confusion_matrix",
    "save_feature_importance",
    "save_probability_timeline",
    "save_sensor_trends",
]
