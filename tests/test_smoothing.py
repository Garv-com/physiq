import numpy as np
import pandas as pd
import pytest

from src.smoothing import compute_ema_trend, compute_time_aware_ema


def make_frame(dates, weights):
    return pd.DataFrame({"date": dates, "weight": weights})


def test_first_value_is_first_observation():
    df = make_frame(pd.date_range("2026-01-01", periods=5), [75.0, 75.5, 74.5, 75.2, 75.1])
    result = compute_time_aware_ema(df, alpha=0.1)
    assert result["trend_weight"].iloc[0] == 75.0


def test_matches_plain_ema_when_there_are_no_gaps():
    weights = np.linspace(80, 78, 30) + np.sin(np.arange(30))
    df = make_frame(pd.date_range("2026-01-01", periods=30), weights)
    plain = compute_ema_trend(df, alpha=0.1)
    aware = compute_time_aware_ema(df, alpha=0.1)
    assert np.array_equal(plain["trend_weight"].to_numpy(), aware["trend_weight"].to_numpy())


def test_a_gap_gives_the_new_reading_more_weight():
    df = make_frame(pd.to_datetime(["2026-01-01", "2026-01-04"]), [70.0, 80.0])
    result = compute_time_aware_ema(df, alpha=0.1)
    expected_alpha = 1 - 0.9 ** 3
    assert result["trend_weight"].iloc[1] == pytest.approx(70.0 + expected_alpha * 10.0, abs=1e-3)


def test_unsorted_input_is_returned_in_date_order():
    df = make_frame(pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]), [75.0, 76.0, 75.5])
    result = compute_time_aware_ema(df, alpha=0.1)
    assert result["date"].is_monotonic_increasing
    assert result["trend_weight"].iloc[0] == 76.0


def test_string_dates_are_accepted():
    df = make_frame(["2026-01-01", "2026-01-02", "2026-01-03"], [75.0, 75.2, 75.1])
    result = compute_time_aware_ema(df, alpha=0.1)
    assert len(result) == 3