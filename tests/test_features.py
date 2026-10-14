import pandas as pd

from src.features import engineer_features
from src.simulator import generate_sensor_data


def test_feature_engineering_does_not_use_future_sensor_values():
    original = generate_sensor_data(n_machines=2, n_cycles=96, seed=8)
    changed = original.copy()
    machine_rows = changed.index[changed["machine_id"] == "machine_001"]
    changed.loc[machine_rows[60], "temperature_c"] += 500.0

    original_features, feature_columns = engineer_features(original)
    changed_features, changed_columns = engineer_features(changed)

    assert feature_columns == changed_columns
    earlier_rows = original_features["machine_id"].eq("machine_001") & (
        original_features["timestamp"] < original_features.loc[machine_rows[60], "timestamp"]
    )
    pd.testing.assert_frame_equal(
        original_features.loc[earlier_rows, feature_columns],
        changed_features.loc[earlier_rows, feature_columns],
    )
