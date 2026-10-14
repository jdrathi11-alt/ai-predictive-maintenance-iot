"""Model training, selection, evaluation, and persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainingResult:
    """Fitted winner plus the grouped holdout used to evaluate it."""

    model: Any
    model_name: str
    feature_columns: list[str]
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    metrics: dict[str, Any]
    test_probabilities: np.ndarray


def split_by_machine(
    frame: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by machine so one machine never appears in both partitions."""
    if "machine_id" not in frame.columns:
        raise ValueError("frame must contain machine_id")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    groups = frame["machine_id"]
    if groups.nunique() < 2:
        raise ValueError("at least two machines are required for a grouped split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(frame, groups=groups))
    train_frame = frame.iloc[train_indices].copy()
    test_frame = frame.iloc[test_indices].copy()
    return train_frame, test_frame


def _candidate_models(random_state: int) -> dict[str, Any]:
    """Build the small, intentionally readable model comparison set."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _probabilities(model: Any, features: Any) -> np.ndarray:
    """Return positive-class probabilities for common sklearn classifiers."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))
        if probabilities.ndim == 1:
            return probabilities.astype(float)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        if probabilities.shape[1] == 1:
            return np.ones(len(probabilities)) if classes[0] == 1 else np.zeros(len(probabilities))
        positive_index = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else 1
        return probabilities[:, positive_index].astype(float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
    raise TypeError("model must expose predict_proba or decision_function")


def evaluate_model(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], np.ndarray]:
    """Calculate classification metrics and positive-class probabilities."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    probabilities = _probabilities(model, features)
    predictions = (probabilities >= threshold).astype(int)
    actual = np.asarray(target).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }
    if np.unique(actual).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))
    else:
        metrics["roc_auc"] = None
    return metrics, probabilities


def train_and_select_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.25,
    random_state: int = 42,
    threshold: float = 0.5,
) -> TrainingResult:
    """Train two classifiers and select by F1, then recall.

    The holdout split is grouped by machine, and the selected model is kept
    fitted on the training machines so its reported metrics remain honest.
    """
    required = {"machine_id", "failure_next_24h", *feature_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frame is missing training columns: {', '.join(missing)}")
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")

    train_frame, test_frame = split_by_machine(frame, test_size, random_state)
    if train_frame["failure_next_24h"].nunique() < 2:
        raise ValueError("grouped training split must contain both target classes")
    if test_frame["failure_next_24h"].nunique() < 2:
        # Metrics remain well-defined, but ROC-AUC is not; this is useful for
        # very small custom datasets and is intentionally not a hard failure.
        pass

    x_train = train_frame[feature_columns]
    y_train = train_frame["failure_next_24h"]
    x_test = test_frame[feature_columns]
    y_test = test_frame["failure_next_24h"]

    fitted: list[tuple[str, Any, dict[str, Any], np.ndarray]] = []
    for name, candidate in _candidate_models(random_state).items():
        candidate.fit(x_train, y_train)
        metrics, probabilities = evaluate_model(candidate, x_test, y_test, threshold)
        fitted.append((name, candidate, metrics, probabilities))

    best_name, best_model, best_metrics, best_probabilities = max(
        fitted,
        key=lambda item: (item[2]["f1"], item[2]["recall"]),
    )
    return TrainingResult(
        model=best_model,
        model_name=best_name,
        feature_columns=list(feature_columns),
        train_frame=train_frame,
        test_frame=test_frame,
        metrics=best_metrics,
        test_probabilities=best_probabilities,
    )


def predict_failure_alerts(
    model_or_bundle: Any,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return probabilities and binary alerts using a configurable threshold.

    ``model_or_bundle`` can be a fitted sklearn model or the dictionary
    returned by :func:`load_model_bundle`. A bundle automatically selects its
    saved feature columns from a DataFrame containing extra columns.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    feature_columns: list[str] | None = None
    model = model_or_bundle
    if isinstance(model_or_bundle, Mapping) and "model" in model_or_bundle:
        model = model_or_bundle["model"]
        saved_columns = model_or_bundle.get("feature_columns")
        feature_columns = list(saved_columns) if saved_columns is not None else None

    model_features: Any = features
    if feature_columns is not None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("a model bundle requires features as a pandas DataFrame")
        model_features = features[feature_columns]

    probabilities = _probabilities(model, model_features)
    return pd.DataFrame(
        {
            "failure_probability": probabilities,
            "failure_alert": (probabilities >= threshold).astype(int),
        },
        index=features.index,
    )


def save_model_bundle(
    path: str | Path,
    model: Any,
    feature_columns: list[str],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a fitted model and the columns required to use it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": list(feature_columns),
        "metadata": dict(metadata or {}),
    }
    joblib.dump(bundle, destination)
    return destination


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load and validate a bundle saved by :func:`save_model_bundle`."""
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("invalid model bundle")
    return bundle


__all__ = [
    "TrainingResult",
    "evaluate_model",
    "load_model_bundle",
    "predict_failure_alerts",
    "save_model_bundle",
    "split_by_machine",
    "train_and_select_model",
]
