# Proof Checklist

Use this checklist to produce a clean, reviewable record of the project. It is designed for a portfolio review, class submission, or internal walkthrough. Record actual commands and observations; do not invent metrics or screenshots.

## A. Repository and scope

- [ ] Repository name is descriptive, for example `ai-predictive-maintenance-iot`.
- [ ] Repository description says that the project is offline, synthetic, reproducible, and a demo/reference.
- [ ] Topics include relevant terms such as `predictive-maintenance`, `iot`, `machine-learning`, `scikit-learn`, `time-series`, and `industrial-ai`.
- [ ] `README.md` links to `docs/PROJECT_GUIDE.md`, `docs/ARCHITECTURE.md`, and this checklist.
- [ ] No claim says that the project is connected to live equipment or certified for industrial/safety use.

## B. Environment setup

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- [ ] Python reports version 3.10 or newer.
- [ ] Dependency installation finishes without an unresolved error.
- [ ] `.venv/` is ignored by Git.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- [ ] Python reports version 3.10 or newer.
- [ ] Dependency installation finishes without an unresolved error.
- [ ] `.venv/` is ignored by Git.

## C. Automated verification

Run from the repository root:

```bash
python -m pytest -q
```

- [ ] Test command used: `________________________________________`.
- [ ] Test date/time and environment: `________________________________`.
- [ ] Result, including the number of passing tests: `__________________`.
- [ ] A failing test was investigated rather than hidden, skipped, or deleted.

The tests should cover:

- [ ] Deterministic simulator output and expected schema.
- [ ] Binary `failure_next_24h` labels.
- [ ] Past-only feature behavior when a future sensor value is changed.
- [ ] Machine-grouped train/test separation.
- [ ] Metrics and output files.
- [ ] Loading the model bundle and producing alert predictions.

## D. Reproducible run evidence

### Recommended default run

```bash
python main.py --machines 6 --cycles 240 --seed 42 --threshold 0.50 --output-dir outputs/proof_run
```

Windows PowerShell may use:

```powershell
python main.py --machines 6 --cycles 240 --seed 42 --threshold 0.50 --output-dir outputs\proof_run
```

- [ ] Exact run command recorded: `_____________________________________`.
- [ ] Seed recorded: `____________`.
- [ ] Threshold recorded: `____________`.
- [ ] Machines/cycles recorded: `____________ / ____________`.
- [ ] Run date/time and operating system recorded: `______________________`.
- [ ] Terminal JSON summary was saved or captured.

### Expected artifacts

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
└── feature_importance.png  # expected when the selected model exposes importance
```

- [ ] `sensor_data.csv` exists and opens as a table.
- [ ] `engineered_features.csv` exists and has more columns than the raw CSV.
- [ ] `failure_alerts.csv` exists and contains probabilities plus binary alerts.
- [ ] `metrics.json` exists and is valid JSON.
- [ ] `predictive_maintenance_model.joblib` exists and is treated as a binary file.
- [ ] `sensor_trends.png` exists unless `--skip-plots` was used.
- [ ] `confusion_matrix.png` exists unless `--skip-plots` was used.
- [ ] `probability_timeline.png` exists unless `--skip-plots` was used.
- [ ] `feature_importance.png` exists when the selected estimator exposes importances/coefficient magnitudes.
- [ ] Output files remain uncommitted because generated output is ignored.

### Metrics review

Open `metrics.json` and record:

| Field | Observed value |
| --- | --- |
| `model_name` | |
| `accuracy` | |
| `precision` | |
| `recall` | |
| `f1` | |
| `roc_auc` | |
| `threshold` | |
| `train_rows` | |
| `test_rows` | |
| `train_machines` | |
| `test_machines` | |
| `feature_count` | |

- [ ] Train and test machine lists are disjoint.
- [ ] Threshold is visible alongside the metrics.
- [ ] ROC-AUC is treated as undefined (`null`) when only one class is present.
- [ ] Accuracy is not presented as the only success criterion.
- [ ] Results are described as synthetic demonstration evidence.

## E. Screenshot list

Capture screenshots only after the run is complete. Include the application/window title or terminal context where useful, and avoid personal paths, tokens, or unrelated private information.

Recommended file names and contents:

1. `01-repository-overview.png` — repository root showing README and the main project folders.
2. `02-install-and-test.png` — environment setup or dependency installation followed by the passing `python -m pytest -q` result.
3. `03-run-command.png` — the exact `python main.py ...` command and terminal JSON summary.
4. `04-output-folder.png` — `outputs/proof_run` file listing with CSV, JSON, model, and PNG artifacts.
5. `05-sensor-trends.png` — the generated six-panel sensor trend plot with its title visible.
6. `06-confusion-matrix.png` — the confusion matrix with actual/predicted labels visible.
7. `07-probability-timeline.png` — holdout probabilities and threshold line.
8. `08-metrics-and-split.png` — `metrics.json` showing model name, threshold, metrics, and train/test machine IDs.
9. `09-git-history.png` — concise commit history showing the planned documentation/config milestones.

- [ ] Screenshots use the actual run and are named consistently.
- [ ] Screenshots do not expose secrets or imply real industrial data.
- [ ] The README or submission notes point to the screenshot set.

## F. Commit/day proof plan

Keep commits small and truthful. Use the actual date and a short message; do not rewrite history to create evidence after the fact.

| Day/stage | Suggested commit focus | Evidence to attach |
| --- | --- | --- |
| Day 1 — setup | Add `requirements.txt`, `.gitignore`, environment instructions, and repository skeleton | Python version, install command, and clean folder screenshot |
| Day 2 — dataset | Add the deterministic simulator and inspect the raw sensor schema | `sensor_data.csv` preview and simulator test |
| Day 3 — preprocessing | Add validation, sorting, lag/delta/rolling features, and leakage test | `engineered_features.csv` preview and future-leakage test |
| Day 4 — model | Add grouped split, Logistic Regression, Random Forest, selection, and persistence | Training terminal output and model bundle path |
| Day 5 — evaluation | Add metrics, confusion matrix, threshold review, and model comparison notes | `metrics.json`, confusion matrix, and metric interpretation |
| Day 6 — visualization | Add sensor trends, probability timeline, feature importance, and alert CSV | PNG files and `failure_alerts.csv` preview |
| Day 7 — upload | Polish README, copy reviewed images, create focused commits, and publish | GitHub repository view, README, topics, and commit history |

Suggested commands to inspect history:

```bash
git status
git log --oneline --decorate -10
```

If Git is not yet initialized, record the same milestones in a dated changelog or project journal and initialize the repository before publishing. Never commit `.venv/`, `outputs/`, local data, model binaries, credentials, or private screenshots.

- [ ] Each commit has a focused message.
- [ ] Documentation changes are separated from generated output.
- [ ] Final `git status` shows only intended tracked changes.
- [ ] The final proof run can be reproduced from the documented command.

## G. Final reviewer questions

- [ ] Can a new user set up Windows and macOS/Linux without guessing a path or command?
- [ ] Is the difference between synthetic telemetry and real industrial data obvious?
- [ ] Is the failure horizon defined?
- [ ] Is the machine-grouped validation boundary visible?
- [ ] Are past-only features and leakage risks documented?
- [ ] Are metrics interpreted with class imbalance and threshold caveats?
- [ ] Are limitations and realistic future improvements listed?
- [ ] Are all expected artifacts present and attributable to a specific command?
