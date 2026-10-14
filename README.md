# AI-Powered Predictive Maintenance for IoT Devices

> An offline, reproducible machine-learning project that simulates industrial IoT telemetry and predicts whether a machine failure is likely within the next 24 observations.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-modeling-F7931E?logo=scikit-learn&logoColor=white)
![IoT](https://img.shields.io/badge/IoT-synthetic%20telemetry-0B7285)
![Offline](https://img.shields.io/badge/runtime-offline-success)

This repository is designed as student proof-of-work for internships,
placements, and entry-level data science or machine-learning roles. It does
not require physical IoT hardware, a cloud account, an API key, or private
industrial data.

> **Important scope:** This is a portfolio and learning prototype using
> synthetic data. Its metrics are not industrial validation, a reliability
> guarantee, or a safety-critical maintenance recommendation.

## Contents

- [Why predictive maintenance?](#why-predictive-maintenance)
- [What this project demonstrates](#what-this-project-demonstrates)
- [Architecture](#architecture)
- [Dataset and simulation](#dataset-and-simulation)
- [Feature engineering and leakage controls](#feature-engineering-and-leakage-controls)
- [Models and evaluation](#models-and-evaluation)
- [Quick start](#quick-start)
- [Output artifacts](#output-artifacts)
- [Example proof run](#example-proof-run)
- [Project structure](#project-structure)
- [GitHub publishing](#github-publishing)
- [Limitations and next steps](#limitations-and-next-steps)
- [Documentation](#documentation)

## Why predictive maintenance?

Traditional maintenance often happens on a fixed schedule or after a failure.
Predictive maintenance uses sensor history to estimate equipment health before
an unplanned breakdown.

In a real industrial workflow, abnormal temperature, vibration, pressure,
current, speed, or other signals can help a maintenance team decide which
asset should be inspected first. The potential benefits include:

- Reducing unplanned downtime
- Prioritizing limited maintenance capacity
- Improving equipment availability
- Reducing unnecessary preventive work
- Creating an auditable basis for maintenance decisions

This project demonstrates the data and modeling workflow without pretending to
have access to a real factory, power plant, vehicle fleet, or aircraft system.

## What this project demonstrates

The repository contains a complete local pipeline:

1. Generate deterministic synthetic sensor data for multiple machines.
2. Validate timestamps, sensor values, and binary failure labels.
3. Create lag, delta, and past-only rolling features.
4. Split complete machines into training and holdout groups.
5. Train Logistic Regression and Random Forest classifiers.
6. Select a model using F1 score, then recall as a tie-breaker.
7. Generate failure probabilities and thresholded maintenance alerts.
8. Save metrics, predictions, a model bundle, and visualizations.
9. Run automated tests for determinism, leakage behavior, and outputs.

## Architecture

```text
CLI arguments
     |
     v
Synthetic IoT simulator
     |  sensor_data.csv
     v
Validation + past-only feature engineering
     |  engineered_features.csv
     v
Machine-grouped train/holdout split
     |
     +--> Scaled Logistic Regression
     +--> Balanced Random Forest
                    |
                    v
           Select by F1, then recall
                    |
        +-----------+----------------+
        |                            |
        v                            v
 metrics.json              failure_alerts.csv
 model bundle              probability predictions
        |
        v
 PNG charts + terminal summary
```

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `src/simulator.py` | Deterministic machine telemetry and failure-horizon labels |
| `src/features.py` | Input validation, sorting, lags, deltas, and rolling statistics |
| `src/model.py` | Grouped splitting, training, selection, metrics, persistence, inference |
| `src/pipeline.py` | End-to-end orchestration and output files |
| `src/visualization.py` | Sensor, probability, confusion-matrix, and importance plots |
| `main.py` | Beginner-friendly command-line entry point |

## Dataset and simulation

The default run generates **6 machines × 240 hourly readings = 1,440 rows**.
Each machine has one simulated failure event. Sensor degradation begins before
the event, and the label becomes positive when the failure is within the next
24 observations.

### Raw sensor columns

| Column | Meaning | Nominal unit |
| --- | --- | --- |
| `temperature_c` | Equipment temperature | °C |
| `vibration_mm_s` | Vibration severity | mm/s |
| `pressure_bar` | Operating pressure | bar |
| `current_a` | Electrical current | A |
| `rpm` | Rotational speed | revolutions/minute |
| `humidity_pct` | Relative humidity | % |
| `failure_next_24h` | Failure-horizon target | 0 or 1 |

The simulator intentionally contains learnable signal so the full workflow can
run offline. It is not a model of a specific real machine or failure
distribution.

## Feature engineering and leakage controls

For every sensor, the pipeline creates:

- One-step lag
- Three-step lag
- One-step change/delta
- Three-step rolling mean and standard deviation
- Twelve-step rolling mean and standard deviation

The rolling features use the previous observation via `shift(1)`. This means a
row cannot use future sensor values when its features are calculated. The
project also:

- Sorts by `machine_id` and timestamp before feature creation
- Avoids target-derived features
- Avoids future-aware backfilling
- Keeps complete machines separated between train and holdout
- Fits the Logistic Regression scaler inside the model pipeline

The leakage test deliberately changes a future sensor value and confirms that
earlier feature rows do not change.

## Models and evaluation

Two beginner-friendly classifiers are trained:

### Logistic Regression

- Standardized numeric features
- Balanced class weights
- Interpretable baseline
- Useful for understanding linear relationships

### Random Forest

- Multiple decision trees
- Balanced class weights
- Captures non-linear interactions
- Provides feature-importance values

The selected model is evaluated with:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC when both classes are present
- 2×2 confusion matrix
- Configurable probability threshold

Because predictive maintenance is often class-imbalanced, do not judge the
system using accuracy alone. Recall represents how many simulated upcoming
failures were detected; precision represents how many alerts were correct.

## Quick start

### Prerequisites

- Python **3.10 or newer**
- Git, if you are cloning the repository
- No database, cloud service, API key, or IoT hardware

### Windows PowerShell

Run these commands from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --machines 6 --cycles 240 --seed 42 --threshold 0.50 --output-dir outputs\proof_run
```

If PowerShell does not allow activation, use the environment interpreter
directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --output-dir outputs\proof_run
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --machines 6 --cycles 240 --seed 42 --threshold 0.50 --output-dir outputs/proof_run
```

### Useful commands

```bash
# Show all CLI options
python main.py --help

# Train on a larger simulated fleet
python main.py --machines 10 --cycles 300 --seed 42 --output-dir outputs/large_run

# Use a stricter alert threshold
python main.py --threshold 0.70 --output-dir outputs/high_threshold

# Generate tabular/model outputs without PNG charts
python main.py --skip-plots --output-dir outputs/tabular_only
```

The command creates the output directory automatically.

## Output artifacts

A normal run creates:

```text
outputs/proof_run/
├── sensor_data.csv
├── engineered_features.csv
├── failure_alerts.csv
├── metrics.json
├── predictive_maintenance_model.joblib
├── sensor_trends.png
├── confusion_matrix.png
├── probability_timeline.png
└── feature_importance.png
```

| Artifact | What it contains |
| --- | --- |
| `sensor_data.csv` | Simulated raw sensor readings and target labels |
| `engineered_features.csv` | Raw values plus model-ready time-series features |
| `failure_alerts.csv` | Holdout machine IDs, actual labels, probabilities, and binary alerts |
| `metrics.json` | Selected model, scores, threshold, split IDs, seed, and row counts |
| `predictive_maintenance_model.joblib` | Fitted model and saved feature-column contract |
| `sensor_trends.png` | Six-panel sensor trends with warning rows highlighted |
| `confusion_matrix.png` | Actual versus predicted alert counts |
| `probability_timeline.png` | Failure probabilities and alert threshold over time |
| `feature_importance.png` | Top feature importance or coefficient magnitudes |

Generated data and model files are intentionally ignored by Git. Keep source
code and documentation tracked; copy only reviewed, non-sensitive screenshots
to `images/` if you want to display them on GitHub.

## Example proof run

The following run was executed locally with the repository pipeline:

```powershell
python main.py --machines 6 --cycles 240 --seed 42 --threshold 0.50 --output-dir outputs\proof_run
```

Observed result on the synthetic holdout:

| Metric | Value |
| --- | ---: |
| Selected model | `logistic_regression` |
| Accuracy | `0.9979` |
| Precision | `1.0000` |
| Recall | `0.9792` |
| F1 | `0.9895` |
| ROC-AUC | `1.0000` |
| Threshold | `0.50` |
| Train rows | `960` |
| Holdout rows | `480` |
| Features | `48` |

The holdout used complete machines that were not used for training:

```text
Training machines: machine_003, machine_004, machine_005, machine_006
Holdout machines:  machine_001, machine_002
```

These numbers demonstrate that the code ran end-to-end on the designed
synthetic signal. They should not be presented as production performance.




## Project structure

```text
.
├── main.py                         # CLI entry point
├── requirements.txt                # runtime and test dependencies
├── src/
│   ├── simulator.py                # synthetic IoT telemetry
│   ├── features.py                 # validation and leakage-safe features
│   ├── model.py                    # training, evaluation, and inference
│   ├── pipeline.py                 # end-to-end orchestration
│   └── visualization.py            # PNG artifact generation
├── tests/                          # pytest tests
├── data/                           # optional local datasets
├── notebooks/                      # optional exploration notebooks
├── models/                         # optional intentionally versioned models
├── images/                         # curated GitHub screenshots
├── docs/
│   ├── PROJECT_GUIDE.md            # beginner-to-industry walkthrough
│   ├── ARCHITECTURE.md             # data flow and module contracts
│   └── PROOF_CHECKLIST.md          # reproducibility and screenshot checklist
└── outputs/                        # local generated outputs; ignored by Git
```



## Limitations and next steps

### Current limitations

- The telemetry is synthetic and deliberately learnable.
- Each machine has one simulated failure event.
- The dataset does not represent real sensor calibration, failure prevalence, or operating regimes.
- The current holdout is grouped by machine but is not a chronological production validation.
- There is no live MQTT/Kafka ingestion, dashboard, alert-routing service, model registry, or monitoring system.
- The threshold is configurable but is not optimized against real maintenance costs.
- Feature importance is not causal explanation.

### Realistic extensions

1. Compare the simulator with a public dataset such as AI4I 2020.
2. Add a documented NASA C-MAPSS remaining-useful-life experiment as a separate task.
3. Add chronological or rolling-origin evaluation.
4. Tune the alert threshold using false-negative and false-positive costs.
5. Add missing-data handling, sensor drift checks, and probability calibration.
6. Build a small Streamlit dashboard around the generated artifacts.
7. Add an inference API and a human-reviewed maintenance workflow.
8. Track model versions, data versions, and experiment metadata.

## Documentation

- [Project guide](docs/PROJECT_GUIDE.md) — concepts, setup, phases, commands, datasets, and modeling notes.
- [Architecture](docs/ARCHITECTURE.md) — components, contracts, data flow, and operational evolution.
- [Proof checklist](docs/PROOF_CHECKLIST.md) — tests, outputs, screenshots, and the seven-day GitHub plan.

## License and responsible use

This repository is intended for education, experimentation, and portfolio
demonstration. Add a formal open-source license before publishing if you want
others to reuse the code. Do not use this prototype to make unsupervised
industrial, safety, medical, aviation, or infrastructure decisions.

## Author
**Jayesh Rathi** — 3rd Year IT Student, Government College of Engineering, Amravati  
[LinkedIn](https://www.linkedin.com/in/jayesh-rathi-5ab8973b6) | [Email](mailto:jayeshrathinew@gmail.com)