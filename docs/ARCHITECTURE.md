# Architecture and Data Flow

This document describes the deliberately small architecture of the AI-Powered Predictive Maintenance System for IoT Devices. It is an offline reference design for synthetic data and local artifacts, not a production deployment blueprint.

## Architectural intent

The design keeps responsibilities separate enough to teach an industry-shaped workflow:

- the simulator supplies a documented input contract;
- feature engineering makes availability and ordering explicit;
- the model layer owns validation, training, evaluation, and persistence;
- the pipeline coordinates one reproducible run;
- visualizations create reviewable evidence rather than a live dashboard.

There are no network calls, credentials, message brokers, databases, APIs, or external sensor dependencies.

## Block diagram

```text
┌──────────────────────────┐
│ User / terminal           │
│ main.py CLI arguments     │
└────────────┬─────────────┘
             │ machines, cycles, seed, threshold, output directory
             v
┌──────────────────────────┐
│ pipeline.run_pipeline     │
│ run orchestration         │
└───────┬──────────┬───────┘
        │          │
        v          v
┌──────────────┐  ┌──────────────────────────┐
│ simulator.py │  │ features.py              │
│ synthetic    │  │ validate, sort, engineer │
│ telemetry    │  │ past-only features       │
└──────┬───────┘  └────────────┬─────────────┘
       │ raw frame             │ feature frame + columns
       └──────────────┬────────┘
                      v
             ┌──────────────────────┐
             │ model.py             │
             │ GroupShuffleSplit    │
             │ train candidates     │
             │ select by F1/recall  │
             │ evaluate threshold   │
             └───────┬───────┬──────┘
                     │       │
          metrics    │       │ model + feature contract
                     v       v
            ┌────────────┐  ┌────────────────────┐
            │ metrics    │  │ joblib model bundle │
            │ JSON       │  │ + metadata         │
            └────────────┘  └────────────────────┘
                     │
                     v
            ┌────────────────────────┐
            │ visualization.py       │
            │ sensor trends, matrix,  │
            │ probabilities, import. │
            └────────────┬───────────┘
                         v
                 ┌─────────────────┐
                 │ local outputs/  │
                 │ CSV / JSON / PNG│
                 └─────────────────┘
```

## Data flow and boundaries

1. **Configuration boundary:** `main.py` parses typed values and passes them to the pipeline. The threshold must be between 0 and 1; at least two machines are required for grouped holdout.
2. **Simulation boundary:** `generate_sensor_data` produces one row per machine and timestamp, with six numeric sensors and the binary future-warning label.
3. **Feature boundary:** `engineer_features` checks columns and values, converts timestamps, sorts by machine/time, and derives lags, deltas, and shifted rolling summaries.
4. **Validation boundary:** `split_by_machine` assigns complete machines to train or holdout. The output records the IDs so overlap can be checked.
5. **Training boundary:** candidate models consume only `feature_columns`. The logistic regression's scaler is fitted inside its pipeline on the training partition.
6. **Decision boundary:** probabilities become `failure_alert` values using the configured threshold. This is a demonstration decision rule, not a maintenance policy.
7. **Evidence boundary:** CSVs, JSON, model bundle, and plots are written below the requested local output directory. The terminal summary references those paths.

## Raw data contract

Every raw row contains:

| Field | Type | Availability/meaning |
| --- | --- | --- |
| `machine_id` | string | Asset identifier; used as a grouping key, not a model feature |
| `timestamp` | datetime | Observation time; rows are sorted within machine |
| `temperature_c` | numeric | Temperature in degrees Celsius |
| `vibration_mm_s` | numeric | Vibration magnitude in mm/s |
| `pressure_bar` | numeric | Pressure in bar |
| `current_a` | numeric | Electrical current in amperes |
| `rpm` | numeric | Rotational speed |
| `humidity_pct` | numeric | Relative humidity percentage |
| `failure_next_24h` | integer 0/1 | Label: simulated failure due within the next 24 cycles |

The simulator uses hourly observations by default, so “24 cycles” is “24 hours” in the default run. A future real-data contract must document sampling irregularity, sensor quality, units, time zones, maintenance events, and label timing.

## Feature contract

The model feature list starts with the six raw sensors. For each sensor, the feature layer adds:

- one-step lag;
- three-step lag;
- one-step delta from the prior observation;
- rolling mean and standard deviation for each configured lookback (default 3 and 12 observations).

The rolling input is shifted before rolling, so the current row is not included in its own historical summary. The target, identifiers, and timestamp are retained in the frame for auditability but are not included in `feature_columns`.

## Model and selection contract

Two readable candidates are trained:

1. **Logistic regression:** `StandardScaler` followed by balanced `LogisticRegression`.
2. **Random forest:** balanced `RandomForestClassifier` with fixed estimator count/depth and seed.

Both candidates are fitted on the machine-grouped training frame. Metrics are computed on the holdout at the requested threshold. The selected model maximizes `(F1, recall)`. The bundle stores:

```text
{
  "model": fitted scikit-learn estimator,
  "feature_columns": ordered list of model inputs,
  "metadata": model name, threshold, and seed
}
```

Saving the ordered feature list prevents a later inference call from silently using a different column order. `predict_failure_alerts` can select those columns from a DataFrame containing extra audit columns.

## Validation and leakage controls

- Grouped splitting prevents a machine from appearing in both partitions.
- Past-only lags and rolling calculations prevent future sensor readings from changing earlier features.
- The scaler is part of the fitted model pipeline, preventing it from learning holdout distribution statistics.
- The output records train/test machine IDs and row counts.
- The tests deliberately change a future sensor reading and assert that earlier feature rows do not change.

These controls address common demonstration-level leakage paths. They do not replace a chronological production validation, a final untouched time period, label review, or data-availability audit.

## Output ownership

| Output | Producer | Review question |
| --- | --- | --- |
| `sensor_data.csv` | pipeline + simulator | What was the generated telemetry and label? |
| `engineered_features.csv` | pipeline + features | Which inputs did the model receive? |
| `failure_alerts.csv` | pipeline + selected model | What probability and binary alert were produced for each holdout row? |
| `metrics.json` | pipeline + model result | Which model, split, threshold, and metrics were reported? |
| `predictive_maintenance_model.joblib` | model persistence | Can the same feature contract be loaded for inference? |
| `sensor_trends.png` | visualization | Does the designed degradation pattern make sense? |
| `confusion_matrix.png` | visualization | What alert errors occurred at the chosen threshold? |
| `probability_timeline.png` | visualization | How did probabilities compare with the threshold? |
| `feature_importance.png` | visualization | Which inputs ranked highly, with the caveat that this is not causal explanation? |

## Operational evolution path

If this reference design were extended, the boundaries should remain explicit while adding:

```text
Sensor gateway -> validated event stream -> durable time-series store
              -> versioned feature computation -> time-forward training
              -> approved model registry -> monitored inference service
              -> human-reviewed alert/work-order feedback
```

That path requires security, identity and access management, data retention, schema evolution, monitoring, incident response, rollback, domain validation, and appropriate regulatory controls. Those concerns are intentionally outside this offline repository.
