# Data directory

Use this directory for optional local datasets that are not committed to the
repository. The default project run does **not** require a downloaded dataset:
`src/simulator.py` creates deterministic synthetic telemetry locally.

If you later add an external dataset, document its public source, license,
schema, preprocessing decisions, and the exact command used to reproduce the
derived files. Do not commit private industrial data or credentials.
