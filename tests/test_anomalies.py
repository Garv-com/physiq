import pandas as pd
import pytest

from src.anomalies import detect_weight_anomalies
from src.synthetic_generator import generate_weight_data


def test_a_large_spike_is_flagged_with_the_right_direction():
    data = generate_weight_data(days=90, seed=3)
    data.loc[70, "weight"] += 2.5
    flagged = {item["date"]: item for item in detect_weight_anomalies(data)}

    spike_date = data.loc[70, "date"].strftime("%Y-%m-%d")
    assert spike_date in flagged
    assert flagged[spike_date]["direction"] == "high"
    assert flagged[spike_date]["difference_kg"] > 1.5


def test_a_large_drop_is_flagged_as_low():
    data = generate_weight_data(days=90, seed=3)
    data.loc[60, "weight"] -= 2.5
    flagged = {item["date"]: item for item in detect_weight_anomalies(data)}
    assert flagged[data.loc[60, "date"].strftime("%Y-%m-%d")]["direction"] == "low"


def test_clean_data_raises_few_alarms():
    data = generate_weight_data(days=90, seed=3)
    assert len(detect_weight_anomalies(data)) <= 1


def test_higher_threshold_flags_no_more_than_lower_threshold():
    data = generate_weight_data(days=90, seed=3)
    data.loc[70, "weight"] += 1.5
    assert len(detect_weight_anomalies(data, threshold=4.0)) <= len(detect_weight_anomalies(data, threshold=3.0))


def test_empty_data_raises():
    with pytest.raises(ValueError):
        detect_weight_anomalies(pd.DataFrame({"date": [], "weight": []}))