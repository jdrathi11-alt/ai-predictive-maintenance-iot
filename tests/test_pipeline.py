import json
from pathlib import Path

import pandas as pd

from src.model import load_model_bundle, predict_failure_alerts
from src.pipeline import run_pipeline


def test_pipeline_writes_outputs_and_reports_grouped_metrics(tmp_path):
    summary = run_pipeline(
        machines=4,
        cycles=72,
        seed=21,
        output_dir=tmp_path,
        skip_plots=True,
    )

    for key in ("raw_data", "engineered_features", "predictions", "metrics", "model"):
        assert Path(summary[key]).exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_name"] in {"logistic_regression", "random_forest"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert set(metrics["train_machines"]).isdisjoint(metrics["test_machines"])

    predictions = pd.read_csv(tmp_path / "failure_alerts.csv")
    assert list(predictions.columns) == [
        "machine_id",
        "timestamp",
        "actual_failure_next_24h",
        "failure_probability",
        "failure_alert",
    ]
    assert predictions["failure_alert"].isin([0, 1]).all()

    bundle = load_model_bundle(tmp_path / "predictive_maintenance_model.joblib")
    features = pd.read_csv(tmp_path / "engineered_features.csv")
    alerts = predict_failure_alerts(bundle, features)
    assert len(alerts) == len(features)
    assert alerts["failure_alert"].isin([0, 1]).all()
