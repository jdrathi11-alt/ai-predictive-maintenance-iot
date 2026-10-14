import pandas as pd

from src.simulator import SENSOR_COLUMNS, generate_sensor_data


def test_simulator_is_deterministic_and_has_expected_schema():
    first = generate_sensor_data(n_machines=3, n_cycles=72, seed=17)
    second = generate_sensor_data(n_machines=3, n_cycles=72, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [
        "machine_id",
        "timestamp",
        *SENSOR_COLUMNS,
        "failure_next_24h",
    ]
    assert first["machine_id"].nunique() == 3
    assert first["failure_next_24h"].isin([0, 1]).all()
