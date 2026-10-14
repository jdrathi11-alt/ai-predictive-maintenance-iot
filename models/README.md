# Models directory

Use this directory for intentionally versioned model experiments only. Normal
run artifacts are written under `outputs/` and ignored by Git so that large or
machine-specific binaries do not accidentally enter the repository.

The pipeline saves a local `predictive_maintenance_model.joblib` bundle with
the feature-column contract and metadata. Keep such binaries local unless you
have a deliberate artifact-storage policy.
