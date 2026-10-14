"""End-to-end offline predictive-maintenance pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .features import engineer_features
from .model import save_model_bundle, train_and_select_model
from .simulator import generate_sensor_data
from .visualization import (
    save_confusion_matrix,
    save_feature_importance,
    save_probability_timeline,
    save_sensor_trends,
)


def run_pipeline(
    machines: int = 6,
    cycles: int = 240,
    seed: int = 42,
    output_dir: str | Path = "outputs",
    threshold: float = 0.5,
    skip_plots: bool = False,
) -> dict[str, Any]:
    """Generate, train, evaluate, and persist a complete demo run.

    All outputs are local files. The returned summary is JSON-compatible and
    includes paths, selected model information, and holdout metrics.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if machines < 2:
        raise ValueError("machines must be at least 2 for a grouped holdout")
    if cycles < 30:
        raise ValueError("cycles must be at least 30 for warning-window features")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_frame = generate_sensor_data(n_machines=machines, n_cycles=cycles, seed=seed)
    feature_frame, feature_columns = engineer_features(raw_frame)
    result = train_and_select_model(
        feature_frame,
        feature_columns,
        random_state=seed,
        threshold=threshold,
    )

    raw_csv = output_path / "sensor_data.csv"
    feature_csv = output_path / "engineered_features.csv"
    predictions_csv = output_path / "failure_alerts.csv"
    metrics_json = output_path / "metrics.json"
    model_path = output_path / "predictive_maintenance_model.joblib"
    raw_frame.to_csv(raw_csv, index=False)
    feature_frame.to_csv(feature_csv, index=False)

    predictions = result.test_frame[
        ["machine_id", "timestamp", "failure_next_24h"]
    ].copy()
    predictions = predictions.rename(columns={"failure_next_24h": "actual_failure_next_24h"})
    predictions["failure_probability"] = result.test_probabilities
    predictions["failure_alert"] = (
        predictions["failure_probability"] >= threshold
    ).astype(int)
    predictions.to_csv(predictions_csv, index=False)

    metrics = dict(result.metrics)
    metrics["model_name"] = result.model_name
    metrics["train_rows"] = int(len(result.train_frame))
    metrics["test_rows"] = int(len(result.test_frame))
    metrics["train_machines"] = sorted(result.train_frame["machine_id"].unique().tolist())
    metrics["test_machines"] = sorted(result.test_frame["machine_id"].unique().tolist())
    metrics["feature_count"] = len(feature_columns)
    metrics["machines"] = machines
    metrics["cycles_per_machine"] = cycles
    metrics["seed"] = seed
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_model_bundle(
        model_path,
        result.model,
        feature_columns,
        metadata={"model_name": result.model_name, "threshold": threshold, "seed": seed},
    )

    plot_paths: dict[str, str] = {}
    if not skip_plots:
        plot_functions = {
            "sensor_trends": lambda: save_sensor_trends(raw_frame, output_path),
            "confusion_matrix": lambda: save_confusion_matrix(
                result.metrics["confusion_matrix"], output_path
            ),
            "probability_timeline": lambda: save_probability_timeline(
                result.test_frame, result.test_probabilities, output_path, threshold
            ),
        }
        for name, create_plot in plot_functions.items():
            plot_paths[name] = str(create_plot())
        importance_path = save_feature_importance(result.model, feature_columns, output_path)
        if importance_path is not None:
            plot_paths["feature_importance"] = str(importance_path)

    return {
        "output_dir": str(output_path),
        "raw_data": str(raw_csv),
        "engineered_features": str(feature_csv),
        "predictions": str(predictions_csv),
        "metrics": str(metrics_json),
        "model": str(model_path),
        "plots": plot_paths,
        "model_name": result.model_name,
        "metrics_summary": metrics,
    }


__all__ = ["run_pipeline"]
