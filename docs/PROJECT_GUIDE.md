# Project Guide: AI-Powered Predictive Maintenance for IoT Devices

This guide explains the project from first principles and then connects each step to an industry-oriented predictive-maintenance workflow. It is written so a beginner can run the demo while a reviewer can see the modeling boundaries and evidence behind the result.

## 1. Purpose and scope

The project answers a focused question:

> Given recent sensor readings from a machine, can a small supervised-learning workflow flag that a simulated failure is likely within the next 24 hours?

It demonstrates data generation, validation, time-aware feature construction, machine-level holdout validation, model comparison, persistence, and visual evidence. It does **not** connect to physical equipment, ingest real plant data, issue work orders, or provide a production reliability or safety guarantee.

### Simple explanation

Sensors provide clues before a machine fails. Temperature and vibration may rise, pressure may fall, and current or speed may change. The simulator creates those patterns. The model learns relationships between the current/past readings and a label that says whether a failure is due soon. The output is a probability and a binary alert that a maintenance team could investigate in a hypothetical workflow.

### Technical explanation

`main.py` passes CLI arguments to `src.pipeline.run_pipeline`. The pipeline generates a deterministic panel of machine/time observations, calls `engineer_features`, and trains two scikit-learn candidates. `GroupShuffleSplit` keeps each `machine_id` in only one partition. Candidates are evaluated at the requested threshold; the model with the highest F1, breaking ties with recall, is retained. CSV, JSON, `joblib`, and PNG files are written to the requested output directory.

## 2. Industry context and example use cases

The same pattern—telemetry, health indicators, a defined prediction horizon, a validated model, and an actionable alert—appears in several sectors:

| Domain | Example asset | Possible decision supported by a future production system |
| --- | --- | --- |
| Manufacturing | Motors, pumps, compressors, conveyors | Inspect or schedule service before an unplanned line stop |
| Factories/process plants | Rotating equipment and production cells | Prioritize a limited maintenance crew across many assets |
| Power and utilities | Generators, transformers, auxiliary systems | Investigate abnormal operating behavior before an outage |
| Automotive | Vehicle subsystems or production-line tooling | Route a component for inspection or controlled service |
| Aviation | Aircraft subsystem health indicators | Support engineering review under certification and human-approval controls |

These examples describe potential adaptations only. This repository's data is synthetic and cannot establish operational, safety, or regulatory suitability.

## 3. Three stack options and selection

There is no single correct stack. Choose based on the evidence and operational need required.

### Option A — Easiest: notebook-first learning demo

**Stack:** Python, pandas, one scikit-learn estimator, matplotlib, and a Jupyter notebook.

**Dataset type:** a small CSV such as the public AI4I 2020 Predictive
Maintenance dataset, or a synthetic sensor table generated in the notebook.

**Good for:** first experiments, teaching classification, and quickly plotting a small CSV.

**Trade-offs:** easy to explore but less repeatable; feature logic, model state, and output naming can become hidden inside notebook cells. It is not the selected repository approach because the project needs a copy-paste command and durable artifacts.

### Option B — Selected: intermediate offline reference stack

**Stack:** Python CLI, pandas/NumPy, scikit-learn, matplotlib/seaborn, joblib, and pytest.

**Dataset type:** deterministic simulated hourly telemetry for this repository;
the same interface can later be adapted to AI4I 2020 or a cleaned NASA
C-MAPSS-style run-to-failure table.

**Why selected:** it keeps setup small and local while adding useful engineering discipline: deterministic seeds, module boundaries, grouped validation, leakage-safe features, a model bundle with feature metadata, JSON metrics, plots, and automated tests. It is appropriate for a portfolio prototype or classroom reference, not a production deployment.

**Repository implementation:** this project uses Option B.

### Option C — Advanced: production-like streaming platform

**Stack example:** MQTT/Kafka or an industrial gateway, schema registry, time-series storage, feature pipelines, a model registry, scheduled/stream inference, monitoring, alert integration, access controls, and a cloud or on-prem orchestration layer.

**Dataset type:** high-volume streaming sensor events from a historian, gateway,
or a replay of run-to-failure data such as NASA C-MAPSS; the labels and event
definitions must be owned by the industrial domain rather than guessed from a
demo file.

**Good for:** real fleets with high-volume telemetry, ownership boundaries, retraining, auditability, and service-level requirements.

**Trade-offs:** higher cost and operational complexity. It requires real failure/maintenance labels, security review, data contracts, incident response, and domain validation; adding these technologies to this small synthetic demo would obscure the learning goal.

### Dataset guidance for a student extension

Start with the built-in simulator so every reviewer can reproduce the project
without a download or account. If you want a second dataset for comparison,
prefer a public source such as **AI4I 2020 Predictive Maintenance Dataset** for
a compact tabular baseline or **NASA C-MAPSS** for a more advanced remaining-
useful-life exercise. Do not mix their label definitions into the current
`failure_next_24h` target without documenting the mapping; “failure within 24
hours” and “remaining useful life” are different supervised-learning tasks.

### At-a-glance comparison

| Option | Tools/libraries | Models | Difficulty | GPU | Expected output |
| --- | --- | --- | --- | --- | --- |
| A — easiest | Python, pandas, scikit-learn, matplotlib, Jupyter | Logistic Regression or Random Forest | Beginner | Not required | One notebook, metrics, and a few plots |
| B — selected | Python CLI, pandas, NumPy, scikit-learn, seaborn, joblib, pytest | Logistic Regression + Random Forest with leakage-safe features | Beginner/intermediate | Not required | Reproducible CSV/JSON/model/PNG evidence and tests |
| C — advanced | MQTT/Kafka, time-series store, orchestration, model registry, monitoring, PyTorch/TensorFlow | LSTM/GRU, gradient boosting, anomaly detection, or survival/RUL models | Advanced | Helpful for deep learning; not always required | Streaming inference, monitored alerts, model versions, and operational dashboards |

## 4. End-to-end workflow

```text
1. CLI configuration
   machines, cycles, seed, output directory, threshold, plot switch
        |
2. Synthetic telemetry
   one deterministic failure event and six sensor channels per machine
        |
3. Input validation and feature engineering
   sort by machine/time; create lags, deltas, and shifted rolling statistics
        |
4. Machine-grouped holdout
   complete machines are assigned to train or test; no machine overlap
        |
5. Candidate training
   scaled logistic regression + balanced random forest
        |
6. Model selection and evaluation
   choose by F1, then recall; report thresholded alerts and probabilities
        |
7. Local evidence
   raw CSV, feature CSV, JSON metrics, joblib bundle, optional PNG charts
```

### What each module owns

- `src/simulator.py` owns the synthetic sensor contract and target construction.
- `src/features.py` owns input checks, sorting, and past-only feature engineering.
- `src/model.py` owns grouped splitting, candidate models, metrics, prediction, and bundle persistence.
- `src/pipeline.py` owns orchestration and output names.
- `src/visualization.py` owns file-based charts using a non-interactive matplotlib backend.
- `main.py` owns the beginner-friendly command-line interface and prints the run summary.

The component-level diagram and data contract are in [ARCHITECTURE.md](ARCHITECTURE.md).

## 5. Ten phases from setup to final output

Use these phases as a guided run and as a review checklist. Each phase has a reason, an expected result, common mistakes, and a verification step.

The workflow below is the detailed version of the requested student plan:

| Requested phase | What it covers in this repository | Main proof artifact |
| --- | --- | --- |
| Phase 1 — Setup | Python environment, dependencies, and project question | `requirements.txt`, passing imports |
| Phase 2 — Dataset loading | Deterministic synthetic telemetry generation | `sensor_data.csv` |
| Phase 3 — Data cleaning | Schema, timestamp, numeric, and binary-label validation | validation in `src/features.py` |
| Phase 4 — Feature engineering | Lags, deltas, and past-only rolling statistics | `engineered_features.csv` |
| Phase 5 — Model building | Grouped split, logistic regression, and random forest | saved `.joblib` bundle |
| Phase 6 — Evaluation | Accuracy, precision, recall, F1, ROC-AUC, and confusion matrix | `metrics.json` and heatmap |
| Phase 7 — Failure prediction | Probability threshold and binary maintenance alert | `failure_probability` / `failure_alert` columns |
| Phase 8 — Visualization | Trends, alert probabilities, confusion matrix, and feature importance | PNG files under `outputs/` |
| Phase 9 — GitHub publishing | README, focused commits, topics, and curated screenshots | repository history and `images/` |
| Phase 10 — Final output | Reproducible command, limitations, and proof checklist | `docs/PROOF_CHECKLIST.md` |

The numbered walkthrough that follows starts with scope and environment details
so a beginner understands *why* the later implementation choices are made.

### Phase 1 — Define the question and boundary

**Why:** A prediction horizon and decision context must be explicit before choosing features or metrics.

**Expected:** The question is “failure within the next 24 observations,” with synthetic data and local outputs clearly stated.

**Common mistakes:** Calling this anomaly detection when the label is supervised; implying that a demo score is an industrial reliability figure; failing to define what “failure” means.

**Verify:** Read the scope in this guide and confirm that a future adaptation will document the event definition, horizon, asset population, and decision owner.

### Phase 2 — Create and enter an isolated environment

**Why:** A virtual environment keeps project packages separate from other Python work.

**Expected:** `.venv/` exists and the shell prompt indicates the environment is active.

**Common mistakes:** Running commands from the parent folder; using a Python version older than 3.10; committing `.venv/`.

**Verify (Windows):**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

**Verify (macOS/Linux):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

### Phase 3 — Install and verify dependencies

**Why:** Reproducible dependencies make the same run easier to repeat and review.

**Expected:** pandas, NumPy, scikit-learn, plotting libraries, joblib, and pytest import successfully.

**Common mistakes:** Installing into a different interpreter; omitting `-r requirements.txt`; mixing incompatible global packages.

**Verify:**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import joblib, matplotlib, numpy, pandas, seaborn, sklearn; print('dependencies OK')"
```

### Phase 4 — Run the automated checks

**Why:** Tests establish a baseline before a run is used as evidence.

**Expected:** pytest completes with all tests passing.

**Common mistakes:** Running `pytest` from the wrong directory; assuming a passing test proves production validity; ignoring a failing test because the generated plot looks correct.

**Verify:**

```bash
python -m pytest -q
```

Review that tests cover deterministic simulation, schema, leakage behavior of past-only features, grouped machine separation, output files, and bundle inference.

### Phase 5 — Generate the synthetic telemetry

**Why:** A deterministic simulator lets the workflow run without credentials, hardware, or private data.

**Expected:** A frame sorted by `machine_id` and `timestamp`, with six sensors and `failure_next_24h`.

**Common mistakes:** Treating the generated values as measured plant data; changing the seed while comparing runs; requesting fewer cycles than the warning horizon.

**Verify:** Run the default command and inspect the first rows:

```bash
python main.py --output-dir outputs/default_run
python -c "import pandas as pd; d=pd.read_csv('outputs/default_run/sensor_data.csv'); print(d.shape); print(d.columns.tolist()); print(d['machine_id'].nunique())"
```

The default shape is 1,440 rows (6 machines × 240 cycles). The exact warning count and metrics depend on the arguments and split.

### Phase 6 — Validate and engineer history-based features

**Why:** Models need useful numeric inputs, but features must represent information available at prediction time.

**Expected:** Raw sensor columns remain available and additional lag, delta, rolling mean, and rolling standard-deviation columns are added.

**Common mistakes:** Sorting after calculating a lag; using `rolling()` on the current/future row without a shift; backfilling initial history from later observations; including the target in `feature_columns`.

**Verify:** Confirm the feature test passes and review `src/features.py`. It uses `groupby("machine_id")`, `shift(1)` for history, and zero fill for unavailable initial history. Do not replace this with a future-aware backfill.

### Phase 7 — Split by machine and train candidates

**Why:** A row-random split can put nearly identical readings from one machine in both sets and overstate generalization to new assets.

**Expected:** Each machine appears in exactly one partition; both candidates fit on training machines only.

**Common mistakes:** Splitting rows randomly; fitting a scaler on all rows before the split; selecting a model on the holdout repeatedly; allowing a machine ID to become a numeric feature.

**Verify:** Inspect `metrics.json` and confirm `train_machines` and `test_machines` are disjoint. The logistic-regression scaler is inside a scikit-learn pipeline, so it is fitted with the training data.

### Phase 8 — Evaluate with an explicit threshold

**Why:** Maintenance alerts are decisions, not just scores. A threshold controls how many positive alerts are issued.

**Expected:** `metrics.json` contains accuracy, precision, recall, F1, confusion matrix, threshold, and ROC-AUC when defined. The selected model has the highest F1, then recall, among the two candidates.

**Common mistakes:** Reporting only accuracy; treating 0.5 as universally correct; comparing metrics across different thresholds without recording the threshold; interpreting ROC-AUC when a class is absent.

**Verify:**

```bash
python -c "import json; m=json.load(open('outputs/default_run/metrics.json', encoding='utf-8')); print(json.dumps({k:m[k] for k in ['model_name','accuracy','precision','recall','f1','roc_auc','threshold']}, indent=2))"
```

### Phase 9 — Inspect and preserve output artifacts

**Why:** A reproducible result needs inspectable files, not only a terminal claim.

**Expected:** CSVs, JSON, a serialized model bundle, and (unless `--skip-plots` is used) PNGs are present.

**Common mistakes:** Committing generated outputs; opening a model file as text; overwriting a run that is being used as evidence; forgetting which command produced a chart.

**Verify (Windows PowerShell):**

```powershell
Get-ChildItem .\outputs\default_run
```

**Verify (macOS/Linux):**

```bash
find outputs/default_run -maxdepth 1 -type f -print
```

Check that `metrics.json` names the model and that the CSVs can be opened with pandas.

### Phase 10 — Record final proof and communicate limitations

**Why:** Reviewers need to reproduce the run, understand what is shown, and see the boundaries of the claim.

**Expected:** The test command, run command, environment, output paths, screenshots, and commits are recorded.

**Common mistakes:** Showing only a best-looking plot; omitting the seed or threshold; claiming the simulator is a live industrial system; mixing generated files into the source commit.

**Verify:** Complete [PROOF_CHECKLIST.md](PROOF_CHECKLIST.md), capture the requested screenshots, and write down the exact command and date/time for the final run.

## 6. Setup and full run commands

### Windows PowerShell

```powershell
# From the repository root
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --machines 6 --cycles 240 --seed 42 --threshold 0.50 --output-dir outputs\default_run
Get-ChildItem .\outputs\default_run
```

If activation is restricted, use the environment interpreter without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --output-dir outputs\default_run
```

### macOS/Linux

```bash
# From the repository root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --machines 6 --cycles 240 --seed 42 --threshold 0.50 --output-dir outputs/default_run
find outputs/default_run -maxdepth 1 -type f -print
```

### Command-line options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--machines` | `6` | Number of simulated machines; at least 2 for grouped holdout |
| `--cycles` | `240` | Observations per machine; the pipeline requires at least 30 |
| `--seed` | `42` | Reproducible random-noise seed |
| `--output-dir` | `outputs` | Destination for all generated files |
| `--threshold` | `0.5` | Probability at or above this value becomes an alert |
| `--skip-plots` | off | Avoid PNG generation while retaining tabular/model artifacts |

## 7. Simulation assumptions and data contract

The simulator is deliberately learnable and deterministic:

1. It creates a common timestamp range per machine, hourly by default, starting at `2024-01-01T00:00:00`.
2. Each machine receives one failure index near 72% of its history, slightly staggered by machine.
3. Degradation begins up to 48 cycles before failure and changes sensors in a designed direction.
4. The label is 1 only when failure is ahead and no more than 24 cycles away; rows after the event are not positive.
5. Noise is generated from a seeded NumPy generator, so the same inputs reproduce the same frame.
6. Machine identity, timestamp, six sensors, and the binary label form the raw contract.

Sensor columns and nominal units:

| Column | Meaning | Nominal unit |
| --- | --- | --- |
| `temperature_c` | equipment temperature | °C |
| `vibration_mm_s` | vibration severity | mm/s |
| `pressure_bar` | pressure | bar |
| `current_a` | electrical current | A |
| `rpm` | rotational speed | revolutions/minute |
| `humidity_pct` | relative humidity | % |

The values, failure timing, noise, and relationships are modeling assumptions for a demonstration. They are not calibration data, a reliability distribution, or a substitute for a domain-approved data-generation process.

## 8. Output artifacts

| Artifact | Contents | How to use it |
| --- | --- | --- |
| `sensor_data.csv` | Raw simulated rows and target | Inspect the source telemetry and labels |
| `engineered_features.csv` | Raw columns plus lag/delta/rolling features | Reproduce or audit model inputs |
| `failure_alerts.csv` | Holdout machine IDs, timestamps, actual labels, probabilities, and alerts | Inspect the user-facing failure-prediction output |
| `metrics.json` | Selected model, metrics, threshold, split IDs, seed, and counts | Review numeric evidence and validation boundaries |
| `predictive_maintenance_model.joblib` | Model, feature-column list, and metadata bundle | Load for local inference with compatible dependencies |
| `sensor_trends.png` | Six sensor trend panels; red points mark warning rows | Explain the designed degradation pattern |
| `confusion_matrix.png` | Actual vs predicted alert counts | Discuss false positives and missed warnings |
| `probability_timeline.png` | Holdout probabilities and chosen threshold | Show how alerts change over time |
| `feature_importance.png` | Top model importance/coefficient magnitudes when available | Support a cautious feature discussion |

Generated files belong under an ignored output directory. Keep source documentation and intentionally curated images outside `outputs/` if they need to be versioned.

## 9. Metric caveats and leakage-safe modeling

### Metric caveats

- **Accuracy:** overall fraction correct; can be deceptively high for rare failures.
- **Precision:** of the alerts raised, how many corresponded to the simulated positive label; relevant to inspection workload.
- **Recall:** of the simulated upcoming failures, how many were alerted; relevant to missed-failure risk.
- **F1:** harmonic mean of precision and recall; useful for this small comparison, but not a maintenance cost function.
- **ROC-AUC:** ranking quality across thresholds when both classes exist; it does not choose an operating threshold and is `null` for a one-class holdout.
- **Confusion matrix:** counts depend on the sample size and threshold; read the actual counts, not just a heatmap color.

The winner is selected on the grouped holdout for simplicity in this educational pipeline. A serious model study should use a validation set or cross-validation for selection and reserve a final untouched temporal test set. A threshold should be tuned using explicit false-negative/false-positive costs, available maintenance capacity, and alert response time.

### Leakage-safe rules

1. Define the label using a future window, but never expose that future window as a feature.
2. Sort by machine and timestamp before creating temporal features.
3. Use only current or prior observations. This implementation uses `shift(1)` before rolling statistics, so a row's rolling values stop before the current reading.
4. Do not backfill missing history with future rows. Initial history is filled with zero by this demo; production work should choose and document a domain-appropriate policy.
5. Keep complete machines separated between train and holdout when testing unseen-asset generalization.
6. Fit learned transforms only on training data. The logistic scaler is inside the model pipeline and is fit during candidate training.
7. Check feature availability at the actual alert timestamp; remove any maintenance outcome, post-failure repair, or future aggregate that would not yet exist.
8. For deployment-like claims, add chronological/rolling-origin splits and a final time-based test period. Grouping by machine alone does not prevent time leakage across periods.

## 10. Folder structure

```text
.
├── main.py
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── simulator.py
│   ├── features.py
│   ├── model.py
│   ├── pipeline.py
│   └── visualization.py
├── tests/
│   ├── test_simulator.py
│   ├── test_features.py
│   └── test_pipeline.py
├── data/                      # optional local datasets; ignored except README
├── notebooks/                 # optional exploration notebooks
├── models/                    # optional intentionally versioned models
├── images/                    # curated screenshots and plots
├── docs/
│   ├── PROJECT_GUIDE.md
│   ├── ARCHITECTURE.md
│   └── PROOF_CHECKLIST.md
└── outputs/                  # created by a run and ignored by Git
```

## 11. GitHub repository naming and presentation

**Suggested name:** `ai-predictive-maintenance-iot`

**Suggested description:**

> Offline, reproducible predictive-maintenance demo for synthetic IoT sensor data with leakage-safe features, machine-grouped validation, model artifacts, metrics, and plots.

**Suggested topics:**

```text
predictive-maintenance, internet-of-things, iot, machine-learning,
scikit-learn, time-series, anomaly-detection, industrial-ai,
condition-monitoring, python, manufacturing
```

Keep the repository description aligned with the scope: “offline,” “synthetic,” “demo,” or “reference” are important qualifiers.

### Copy-paste publishing sequence

After creating an empty repository on GitHub, run this from the project root:

```bash
git init
git branch -M main
git add README.md requirements.txt .gitignore src tests docs images data notebooks models outputs/README.md
git commit -m "feat: add reproducible predictive maintenance pipeline"
git remote add origin https://github.com/<YOUR-USERNAME>/ai-predictive-maintenance-iot.git
git push -u origin main
```

Use one focused commit for each milestone rather than one giant “final project”
commit. Keep generated `outputs/` files and serialized model binaries local;
commit only `outputs/README.md` and reviewed screenshots under `images/`.

## 12. Limitations and future improvements

### Limitations now

- Synthetic values encode a failure signal and one failure event per machine.
- The sample is small and does not represent real asset diversity, sensor faults, maintenance censoring, or class prevalence.
- A grouped random holdout checks unseen-machine separation but not forward-in-time performance.
- There is no live ingestion, feature store, model registry, API, dashboard, alert routing, access control, monitoring, or rollback process.
- The threshold is configurable but not cost-optimized or calibrated.
- Feature importance is not causal explanation and can be unstable under correlated sensors.

### Practical next steps

1. Establish an approved event/label definition with maintenance subject-matter experts.
2. Ingest real historian or gateway data with timestamps, quality flags, maintenance records, and asset metadata.
3. Add time-forward and asset-aware evaluation with an untouched final period.
4. Tune and calibrate alerts against downtime and inspection costs.
5. Add missing-data, outlier, sensor-drift, and operating-regime handling.
6. Track data/model versions, training runs, approvals, and rollback artifacts.
7. Integrate alerts into a human-reviewed maintenance workflow and measure realized outcomes.

## 13. Evidence and proof

Use [PROOF_CHECKLIST.md](PROOF_CHECKLIST.md) to record a clean run, expected outputs, screenshots, and a short commit/day plan. Evidence should show the command, seed, threshold, tests, and artifact names without exposing secrets or pretending that synthetic results are industrial validation.
